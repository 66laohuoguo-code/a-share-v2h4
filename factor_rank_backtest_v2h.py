"""
V2: Causal weekly A-share factor backtest.

Designed as a non-destructive companion to factor_rank_backtest.py.
Save this file next to the existing scripts. It imports the original file for
point-in-time data loading and feature construction, then replaces four layers:
1) continuous risk budget with a normal-market equity floor;
2) score + inverse-volatility constrained portfolio weights;
3) optional causal rolling-IC factor weights;
4) execution with slippage, turnover bands and participation limits.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import signal
import sqlite3
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import factor_rank_backtest as base
from opening_auction import (
    CausalOpeningGapEstimator,
    expected_open_price,
    opening_auction_limit_price,
    opening_auction_order_is_marketable,
)
from risk_aware_portfolio import (
    CausalRiskCalibrationStore,
    WeeklyAlphaFeatureStore,
    WeeklyRiskModelStore,
    apply_store_overlay,
)
from small_account_v3 import (
    commission_efficient_trade_floor,
    optimize_discrete_target_shares,
    select_cost_aware_codes,
)
from v31_strategy import (
    MonthlyFactorStateStore,
    V31AlphaFeatureStore,
    apply_score as apply_v31_score,
    industry_budget_caps,
    normalize_industry_series,
    normalized_component_weights as normalized_v31_component_weights,
)
from v22_strategy import (
    IndustrySatelliteController,
    apply_structural_components as apply_v22_structural_components,
    blend_continuous_industry_satellite,
)
from ashare_utils import (
    apply_risk_alignment_trade_floor,
    buy_order_size_rules,
    mandatory_trade_cost,
    round_portfolio_target_shares,
    round_target_shares_for_code,
    should_rebalance_on_date,
    trade_value_floor,
    trading_cost_snapshot,
    write_excel_workbook,
)


DEFAULT_DATABASE = Path("data/processed/stock_daily.sqlite")
DEFAULT_OUTPUT_DIR = Path("outputs/backtest_v2")
CHECKPOINT_VERSION = 1
FEATURE_CACHE_VERSION = 1

# The original low-beta/industry mix, re-normalized so the weights sum to one.
STATIC_COMPONENT_WEIGHTS: Dict[str, float] = {
    "low_beta_score": 0.25,
    "low_volatility_score": 0.21,
    "low_turnover_score": 0.17,
    "reversal_score": 0.15,
    "lower_drawdown_score": 0.10,
    "industry_trend_score": 0.12,
}
SMALL_ACCOUNT_V3_COMPONENT_WEIGHTS: Dict[str, float] = {
    "low_beta_score": 0.07,
    "low_volatility_score": 0.18,
    "low_turnover_score": 0.13,
    "reversal_score": 0.12,
    "lower_drawdown_score": 0.05,
    "industry_trend_score": 0.10,
    "earnings_yield_score": 0.25,
    "residual_momentum_score": 0.10,
}
EVENT_COMPONENT = "industry_event_score_ranked"


def configured_component_weights(args) -> Dict[str, float]:
    profile = str(getattr(args, "score_profile", "v2h4_legacy")).strip().lower()
    if profile == "v31":
        return normalized_v31_component_weights(
            getattr(args, "v31_component_weights", None)
        )
    if profile == "china_small_v3":
        return dict(SMALL_ACCOUNT_V3_COMPONENT_WEIGHTS)
    return dict(STATIC_COMPONENT_WEIGHTS)


def resolved_v31_industry_budget_mode(args) -> str:
    mode = str(
        getattr(args, "v31_industry_budget_mode", "auto")
    ).strip().lower()
    if mode == "auto":
        profile = str(
            getattr(args, "score_profile", "v2h4_legacy")
        ).strip().lower()
        return "soft" if profile == "v31" else "fixed"
    return mode


def uses_v31_alpha_features(args) -> bool:
    profile = str(
        getattr(args, "score_profile", "v2h4_legacy")
    ).strip().lower()
    return (
        profile == "v31"
        or float(getattr(args, "v31_alpha_tilt_weight", 0.0)) > 0.0
        or resolved_v31_industry_budget_mode(args) == "soft"
    )


def v22_market_risk_on_strength(regime: Mapping[str, object], args) -> float:
    mode = str(
        getattr(args, "v22_industry_satellite_risk_throttle", "none")
    ).strip().lower()
    if mode == "none":
        return 1.0
    if mode != "continuous_equity":
        raise ValueError(f"Unsupported V2.2 satellite risk throttle: {mode}")
    floor = clip(float(getattr(args, "min_equity_weight", 0.0)), 0.0, 1.0)
    ceiling = clip(1.0 - float(getattr(args, "cash_weight", 0.0)), floor, 1.0)
    target = clip(float(regime.get("target_equity_weight", floor)), 0.0, 1.0)
    if ceiling <= floor + 1e-12:
        return 1.0 if target >= ceiling else 0.0
    return clip((target - floor) / (ceiling - floor), 0.0, 1.0)


def resolved_economic_replacement_policy(args) -> str:
    policy = str(
        getattr(args, "economic_replacement_policy", "auto")
    ).strip().lower()
    if policy == "auto":
        return (
            "cost_aware"
            if bool(getattr(args, "enable_economic_replacement_hurdle", False))
            else "none"
        )
    return policy


class BacktestPaused(RuntimeError):
    """Raised after a requested pause has been saved successfully."""


class PriceDateStore:
    """Build daily price dictionaries on demand instead of duplicating the full database."""

    def __init__(self, prices: pd.DataFrame, max_cached_dates: int = 32):
        self.prices = prices.reset_index(drop=True)
        self.max_cached_dates = max(1, int(max_cached_dates))
        self.positions_by_date = {
            str(date): np.asarray(positions, dtype=np.int64)
            for date, positions in self.prices.groupby("trade_date", sort=False).indices.items()
        }
        self.cache: OrderedDict[str, Dict[str, Dict[str, object]]] = OrderedDict()

    def get(self, date: str, default=None):
        key = str(date)
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached
        positions = self.positions_by_date.get(key)
        if positions is None:
            return default
        daily = self.prices.iloc[positions].set_index("code").to_dict("index")
        self.cache[key] = daily
        while len(self.cache) > self.max_cached_dates:
            self.cache.popitem(last=False)
        return daily


def feature_cache_fingerprint(args, database_override: Optional[Path] = None) -> str:
    database = Path(database_override or args.database).resolve()
    stat = database.stat()
    feature_arguments = {}
    for key in [
        "feature_history_days",
        "min_history_days",
        "min_avg_amount",
        "min_market_cap_quantile",
        "market_cap_proxy_window",
        "disable_industry_neutral_factors",
        "industry_event_scores",
    ]:
        value = getattr(args, key, None)
        if isinstance(value, Path):
            path = value.resolve()
            value = str(path)
            if path.exists():
                path_stat = path.stat()
                value = {
                    "path": str(path),
                    "size": int(path_stat.st_size),
                    "modified_ns": int(path_stat.st_mtime_ns),
                }
        feature_arguments[key] = value
    payload = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "database": {
            "path": str(database),
            "size": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        },
        "feature_code_sha256": hashlib.sha256(Path(base.__file__).resolve().read_bytes()).hexdigest(),
        "arguments": feature_arguments,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FeatureSnapshotCache:
    """Persistent compressed cache shared by capital and portfolio-count experiments."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                fingerprint TEXT NOT NULL,
                decision_date TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                payload BLOB NOT NULL,
                saved_at TEXT NOT NULL,
                PRIMARY KEY (fingerprint, decision_date)
            )
            """
        )
        self.conn.commit()

    def get(self, fingerprint: str, decision_date: str) -> Optional[pd.DataFrame]:
        row = self.conn.execute(
            """
            SELECT payload
            FROM feature_snapshots
            WHERE fingerprint = ? AND decision_date = ?
            """,
            (str(fingerprint), str(decision_date)),
        ).fetchone()
        if row is None:
            return None
        return pickle.loads(zlib.decompress(row[0]))

    def put(self, fingerprint: str, decision_date: str, frame: pd.DataFrame) -> None:
        encoded = zlib.compress(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL), level=3)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO feature_snapshots
                (fingerprint, decision_date, row_count, payload, saved_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(fingerprint),
                str(decision_date),
                int(len(frame)),
                sqlite3.Binary(encoded),
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def count(self, fingerprint: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM feature_snapshots WHERE fingerprint = ?",
            (str(fingerprint),),
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.conn.close()


class SplitFeatureSnapshotCache:
    """Route snapshots to existing pre/post-cutover caches without copying them."""

    def __init__(
        self,
        before_path: Path,
        after_path: Path,
        cutover_date: str,
        before_fingerprint: str,
        after_fingerprint: str,
    ):
        self.before = FeatureSnapshotCache(before_path)
        self.after = FeatureSnapshotCache(after_path)
        self.path = self.after.path
        self.cutover_date = str(cutover_date)
        self.before_fingerprint = str(before_fingerprint)
        self.after_fingerprint = str(after_fingerprint)
        encoded = json.dumps(
            {
                "cutover_date": self.cutover_date,
                "before": self.before_fingerprint,
                "after": self.after_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.combined_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _route(self, decision_date: str) -> Tuple[FeatureSnapshotCache, str]:
        if str(decision_date) < self.cutover_date:
            return self.before, self.before_fingerprint
        return self.after, self.after_fingerprint

    def get(self, _fingerprint: str, decision_date: str) -> Optional[pd.DataFrame]:
        cache, fingerprint = self._route(decision_date)
        return cache.get(fingerprint, decision_date)

    def put(self, _fingerprint: str, decision_date: str, frame: pd.DataFrame) -> None:
        cache, fingerprint = self._route(decision_date)
        cache.put(fingerprint, decision_date, frame)

    def count(self, _fingerprint: str) -> int:
        return self.before.count(self.before_fingerprint) + self.after.count(
            self.after_fingerprint
        )

    def close(self) -> None:
        self.before.close()
        self.after.close()


def cached_feature_snapshot(
    cache: Optional[FeatureSnapshotCache],
    cache_fingerprint: Optional[str],
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    decision_date: str,
    args,
    industry_events: pd.DataFrame,
) -> pd.DataFrame:
    if cache is not None and cache_fingerprint is not None:
        cached = cache.get(cache_fingerprint, decision_date)
        if cached is not None:
            return cached
    features = base.feature_snapshot(prices, financial, decision_date, args, industry_events)
    if cache is not None and cache_fingerprint is not None:
        cache.put(cache_fingerprint, decision_date, features)
    return features


def checkpoint_path_from_args(args) -> Optional[Path]:
    configured = getattr(args, "checkpoint_file", None)
    if configured:
        return Path(configured).resolve()
    if bool(getattr(args, "resume", False)):
        return Path(args.output_dir).resolve() / "v2h_checkpoint.json.gz"
    return None


def checkpoint_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def checkpoint_recovery_paths(path: Path) -> List[Path]:
    candidates = [path.with_name(path.name + ".tmp")]
    candidates.extend(path.parent.glob(path.name + ".pending.*"))
    return [candidate for candidate in candidates if candidate.exists()]


def checkpoint_exists(path: Path) -> bool:
    return path.exists() or bool(checkpoint_recovery_paths(path))


def replace_with_retry(
    source: Path,
    destination: Path,
    attempts: int = 12,
) -> bool:
    delay = 0.05
    last_error: Optional[PermissionError] = None
    for attempt in range(max(1, attempts)):
        try:
            os.replace(source, destination)
            return True
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 1.0)
    print(
        f"Warning: Windows kept {destination} locked after {attempts} attempts; "
        f"the recoverable checkpoint remains at {source}. Error: {last_error}",
        flush=True,
    )
    return False


def cleanup_checkpoint_recovery_files(path: Path, keep: Optional[Path] = None) -> None:
    for candidate in checkpoint_recovery_paths(path):
        if keep is not None and candidate == keep:
            continue
        try:
            candidate.unlink()
        except (FileNotFoundError, PermissionError):
            pass


def write_checkpoint_file(path: Path, payload: bytes) -> bool:
    temporary = path.with_name(
        f"{path.name}.pending.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    replaced = replace_with_retry(temporary, path)
    if replaced:
        cleanup_checkpoint_recovery_files(path)
    return replaced


def checkpoint_fingerprint(args) -> str:
    excluded = {"checkpoint_file", "checkpoint_every_n_days", "resume"}
    arguments = {}
    for key, value in sorted(vars(args).items()):
        if key in excluded:
            continue
        if isinstance(value, Path):
            value = str(value.resolve())
        arguments[key] = value
    database = Path(args.database).resolve()
    stat = database.stat()
    before_database = None
    configured_before_database = getattr(args, "database_before_cutover", None)
    if configured_before_database:
        before_path = Path(configured_before_database).resolve()
        before_stat = before_path.stat()
        before_database = {
            "path": str(before_path),
            "size": int(before_stat.st_size),
            "modified_ns": int(before_stat.st_mtime_ns),
        }
    code_files = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(__file__).with_name("opening_auction.py").resolve(),
        Path(__file__).with_name("risk_aware_portfolio.py").resolve(),
        Path(__file__).with_name("risk_model_reporting.py").resolve(),
        Path(__file__).with_name("small_account_v3.py").resolve(),
        Path(__file__).with_name("v31_strategy.py").resolve(),
        Path(__file__).with_name("v22_strategy.py").resolve(),
    ]
    risk_database = None
    configured_risk_database = getattr(args, "risk_model_database", None)
    if configured_risk_database:
        risk_path = Path(configured_risk_database).resolve()
        if risk_path.exists():
            risk_stat = risk_path.stat()
            risk_database = {
                "path": str(risk_path),
                "size": int(risk_stat.st_size),
                "modified_ns": int(risk_stat.st_mtime_ns),
            }
    risk_database_before_cutover = None
    configured_risk_before = getattr(
        args, "risk_model_database_before_cutover", None
    )
    if configured_risk_before:
        risk_before_path = Path(configured_risk_before).resolve()
        risk_before_stat = risk_before_path.stat()
        risk_database_before_cutover = {
            "path": str(risk_before_path),
            "size": int(risk_before_stat.st_size),
            "modified_ns": int(risk_before_stat.st_mtime_ns),
        }
    risk_calibration_schedule = None
    configured_schedule = getattr(args, "risk_calibration_schedule", None)
    if configured_schedule:
        schedule_path = Path(configured_schedule).resolve()
        if schedule_path.exists():
            schedule_stat = schedule_path.stat()
            risk_calibration_schedule = {
                "path": str(schedule_path),
                "size": int(schedule_stat.st_size),
                "modified_ns": int(schedule_stat.st_mtime_ns),
            }
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "database": {
            "path": str(database),
            "size": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        },
        "database_before_cutover": before_database,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_files
        },
        "risk_database": risk_database,
        "risk_database_before_cutover": risk_database_before_cutover,
        "risk_calibration_schedule": risk_calibration_schedule,
        "arguments": arguments,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def serialize_weighter(weighter: "RollingICWeighter") -> Dict[str, object]:
    return {
        "pending": [
            {
                "decision_date": item.decision_date,
                "entry_date": item.entry_date,
                "exit_date": item.exit_date,
                "frame": item.frame.to_dict(orient="split"),
            }
            for item in weighter.pending
        ],
        "ic_rows": weighter.ic_rows,
        "weight_rows": weighter.weight_rows,
    }


def restore_weighter(weighter: "RollingICWeighter", payload: Mapping[str, object]) -> None:
    def restore_frame(item):
        frame = item["frame"]
        return pd.DataFrame(
            frame["data"],
            columns=frame["columns"],
            index=frame.get("index"),
        )

    weighter.pending = [
        PendingSnapshot(
            str(item["decision_date"]),
            str(item["entry_date"]),
            str(item["exit_date"]),
            restore_frame(item),
        )
        for item in payload.get("pending", [])
    ]
    weighter.ic_rows = list(payload.get("ic_rows", []))
    weighter.weight_rows = list(payload.get("weight_rows", []))


def make_checkpoint_state(
    fingerprint: str,
    next_offset: int,
    test_dates: Sequence[str],
    holdings: Mapping[str, int],
    cash: float,
    last_close: Mapping[str, float],
    previous_total: float,
    peak_total: float,
    equity_rows: Sequence[Mapping[str, object]],
    trade_rows: Sequence[Mapping[str, object]],
    weighter: "RollingICWeighter",
    satellite_controller: Optional[IndustrySatelliteController] = None,
) -> Dict[str, object]:
    return {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "next_offset": int(next_offset),
        "total_dates": int(len(test_dates)),
        "last_completed_date": test_dates[next_offset - 1] if next_offset > 0 else None,
        "holdings": dict(holdings),
        "cash": float(cash),
        "last_close": dict(last_close),
        "previous_total": float(previous_total),
        "peak_total": float(peak_total),
        "equity_rows": list(equity_rows),
        "trade_rows": list(trade_rows),
        "weighter": serialize_weighter(weighter),
        "v22_satellite_controller": (
            satellite_controller.serialize()
            if satellite_controller is not None
            else None
        ),
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def checkpoint_json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def save_checkpoint(path: Path, state: Mapping[str, object], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(state),
        ensure_ascii=False,
        separators=(",", ":"),
        default=checkpoint_json_default,
    ).encode("utf-8")
    write_checkpoint_file(path, gzip.compress(encoded, compresslevel=3))

    next_offset = int(state["next_offset"])
    total_dates = int(state["total_dates"])
    metadata = {
        "status": "paused_or_running",
        "strategy_name": str(getattr(args, "strategy_name", "V2H")),
        "checkpoint_file": str(path),
        "saved_at": state["saved_at"],
        "last_completed_date": state.get("last_completed_date"),
        "completed_dates": next_offset,
        "total_dates": total_dates,
        "progress_percent": round(100.0 * next_offset / max(total_dates, 1), 2),
        "holding_count": len(state.get("holdings", {})),
        "cash": float(state.get("cash", 0.0)),
        "portfolio_value": float(state.get("previous_total", 0.0)),
        "fingerprint": state["fingerprint"],
    }
    metadata_path = checkpoint_metadata_path(path)
    metadata_payload = json.dumps(
        metadata, ensure_ascii=False, indent=2
    ).encode("utf-8")
    write_checkpoint_file(metadata_path, metadata_payload)


def validate_checkpoint_state(
    state: Mapping[str, object],
    path: Path,
    fingerprint: str,
    total_dates: int,
) -> None:
    if int(state.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch in {path}; start a new checkpoint file."
        )
    if state.get("fingerprint") != fingerprint:
        raise ValueError(
            "Checkpoint does not match the current database, date range or strategy parameters: "
            f"{path}"
        )
    if int(state.get("total_dates", -1)) != int(total_dates):
        raise ValueError(f"Checkpoint trading-date count does not match this run: {path}")
    next_offset = int(state.get("next_offset", -1))
    if not 0 <= next_offset <= total_dates:
        raise ValueError(f"Checkpoint contains an invalid next offset: {next_offset}")


def read_checkpoint_state(path: Path) -> Dict[str, object]:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def load_checkpoint(path: Path, fingerprint: str, total_dates: int) -> Dict[str, object]:
    candidates = [candidate for candidate in [path, *checkpoint_recovery_paths(path)] if candidate.exists()]
    valid: List[Tuple[int, int, Path, Dict[str, object]]] = []
    errors: List[Tuple[Path, Exception]] = []
    for candidate in candidates:
        try:
            state = read_checkpoint_state(candidate)
            validate_checkpoint_state(state, candidate, fingerprint, total_dates)
            valid.append(
                (
                    int(state["next_offset"]),
                    int(candidate.stat().st_mtime_ns),
                    candidate,
                    state,
                )
            )
        except (OSError, EOFError, gzip.BadGzipFile, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append((candidate, exc))
    if not valid:
        if errors:
            raise errors[0][1]
        raise FileNotFoundError(f"No checkpoint or recovery candidate exists: {path}")

    _, _, selected, state = max(valid, key=lambda item: (item[0], item[1]))
    if selected != path:
        print(
            f"Recovering newer checkpoint candidate {selected} "
            f"through {state.get('last_completed_date')}.",
            flush=True,
        )
        if replace_with_retry(selected, path):
            cleanup_checkpoint_recovery_files(path)
    return state


def clear_checkpoint(path: Optional[Path]) -> None:
    if path is None:
        return
    metadata_path = checkpoint_metadata_path(path)
    targets = [
        path,
        metadata_path,
        *checkpoint_recovery_paths(path),
        *checkpoint_recovery_paths(metadata_path),
    ]
    for target in targets:
        if target.exists():
            try:
                target.unlink()
            except FileNotFoundError:
                pass


def clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def safe_float(value, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_series(values, index=None) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if index is not None:
        result = pd.Series(result, index=index)
    return result.replace([np.inf, -np.inf], np.nan)


def current_weights_from_open(
    holdings: Mapping[str, int],
    prices: Mapping[str, Mapping[str, object]],
    open_total: float,
    last_close: Mapping[str, float],
) -> Dict[str, float]:
    if open_total <= 0:
        return {}
    weights: Dict[str, float] = {}
    for code, shares in holdings.items():
        row = prices.get(code, {})
        price = safe_float(row.get("open"), np.nan)
        if not math.isfinite(price) or price <= 0:
            price = safe_float(last_close.get(code), np.nan)
        if math.isfinite(price) and price > 0 and int(shares) > 0:
            weights[code] = int(shares) * price / open_total
    return weights


def continuous_target_equity(
    market_state: pd.DataFrame,
    decision_date: str,
    previous_total: float,
    peak_total: float,
    args,
    event_regime_signals: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Causal continuous risk budget.

    The old strategy can spend long periods at 20%/35% equity because it switches
    between discrete labels. V2 remains defensive but keeps normal conditions in
    a configurable 65%-95% equity corridor. Only a hard crash or severe portfolio
    drawdown is allowed below the floor.
    """
    base_equity = clip(1.0 - float(args.cash_weight), 0.0, 1.0)
    min_equity = clip(float(args.min_equity_weight), 0.0, base_equity)

    row = market_state.loc[decision_date].to_dict() if decision_date in market_state.index else {}
    index = safe_float(row.get("market_index"), np.nan)
    ma_short = safe_float(row.get("market_ma_short"), np.nan)
    ma_long = safe_float(row.get("market_ma_long"), np.nan)
    breadth = safe_float(row.get("market_breadth"), np.nan)
    ret20 = safe_float(row.get("market_return_20"), np.nan)
    vol = safe_float(row.get("market_volatility_20"), np.nan)

    if args.risk_mode == "disabled":
        target = base_equity
        state = "disabled"
        trend_signal = 0.0
        vol_scale = 1.0
    elif args.risk_mode == "legacy":
        # Keep the original engine available for attribution experiments.
        legacy = base.target_equity_from_market(
            market_state, decision_date, previous_total, peak_total, args, event_regime_signals
        )
        return legacy
    else:
        short_gap = (index / ma_short - 1.0) if math.isfinite(index) and math.isfinite(ma_short) and ma_short > 0 else 0.0
        long_gap = (index / ma_long - 1.0) if math.isfinite(index) and math.isfinite(ma_long) and ma_long > 0 else 0.0
        breadth_signal = (breadth - 0.50) / float(args.breadth_band) if math.isfinite(breadth) else 0.0
        trend_band = max(float(args.trend_band), 1e-6)
        trend_signal = clip(
            0.35 * clip(short_gap / trend_band, -1.0, 1.0)
            + 0.45 * clip(long_gap / trend_band, -1.0, 1.0)
            + 0.20 * clip(breadth_signal, -1.0, 1.0),
            -1.0,
            1.0,
        )
        vol_scale = 1.0
        if math.isfinite(vol) and vol > 0 and float(args.target_market_volatility) > 0:
            vol_scale = clip(
                float(args.target_market_volatility) / vol,
                float(args.min_vol_scale),
                float(args.max_vol_scale),
            )
        trend_scale = 1.0 + float(args.trend_tilt) * trend_signal
        target = base_equity * vol_scale * trend_scale
        target = clip(target, min_equity, base_equity)
        state = "continuous_risk_on" if trend_signal >= 0 else "continuous_risk_off"
        if state == "continuous_risk_off":
            if math.isfinite(ret20) and ret20 <= float(args.soft_crash_return_20):
                target = min(target, float(args.soft_crash_equity_cap))
                state = "continuous_soft_crash_cap"
            elif float(args.risk_off_equity_cap) < base_equity:
                target = min(target, float(args.risk_off_equity_cap))
                state = "continuous_risk_off_cap"

    drawdown = previous_total / peak_total - 1.0 if peak_total > 0 else 0.0
    hard_crash = (
        math.isfinite(ret20)
        and ret20 <= float(args.hard_crash_return_20)
        and math.isfinite(breadth)
        and breadth <= float(args.hard_crash_breadth)
    )
    if hard_crash:
        target = min(target, float(args.hard_crash_equity_weight))
        state = "hard_crash"
    elif drawdown <= float(args.severe_drawdown_threshold):
        target = min(target, float(args.severe_drawdown_equity_weight))
        state = "portfolio_severe_drawdown"
    elif drawdown <= float(args.drawdown_reduce_threshold):
        target = max(min_equity, target * float(args.drawdown_reduce_multiplier))
        state = "portfolio_drawdown_guard"

    event = base.latest_event_regime_signal(event_regime_signals, decision_date)
    event_multiplier_used = 1.0
    if bool(args.enable_event_regime) and float(event["event_confidence"]) >= float(args.event_regime_min_confidence):
        # Blend, rather than fully obey, a model-generated event multiplier.
        event_multiplier_used = 1.0 + float(args.event_regime_blend) * (
            float(event["equity_multiplier"]) - 1.0
        )
        target *= event_multiplier_used

    # Keep the normal floor except in explicitly identified crash/portfolio-stress states.
    # V2E also allows explicit risk-off caps to go below the normal floor.
    if state not in {"hard_crash", "portfolio_severe_drawdown", "continuous_risk_off_cap", "continuous_soft_crash_cap"}:
        target = max(min_equity, target)
    target = clip(target, 0.0, base_equity)

    return {
        "target_equity_weight": float(target),
        "market_state": state,
        "market_index": index,
        "market_breadth": breadth,
        "market_return_20": ret20,
        "market_volatility_20": vol,
        "trend_signal": trend_signal,
        "vol_scale": vol_scale,
        "portfolio_drawdown": float(drawdown),
        "event_equity_multiplier": float(event["equity_multiplier"]),
        "event_equity_multiplier_used": float(event_multiplier_used),
        "event_risk_score": float(event["event_risk_score"]),
        "event_regime_confidence": float(event["event_confidence"]),
        "event_regime_reason": str(event["event_regime_reason"]),
    }


@dataclass
class PendingSnapshot:
    decision_date: str
    entry_date: str
    exit_date: str
    frame: pd.DataFrame


@dataclass
class RollingICWeighter:
    args: object
    component_names: Sequence[str]
    pending: List[PendingSnapshot] = field(default_factory=list)
    ic_rows: List[Dict[str, object]] = field(default_factory=list)
    weight_rows: List[Dict[str, object]] = field(default_factory=list)

    def add_snapshot(
        self,
        decision_date: str,
        entry_date: str,
        exit_date: str,
        features: pd.DataFrame,
    ) -> None:
        columns = ["code"] + [name for name in self.component_names if name in features.columns]
        if "score_v2" in features.columns and "score_v2" not in columns:
            columns.append("score_v2")
        frame = features[columns].copy()
        self.pending.append(PendingSnapshot(decision_date, entry_date, exit_date, frame))

    def resolve(self, decision_date: str, prices_by_date: Mapping[str, Mapping[str, Mapping[str, object]]]) -> None:
        """Resolve only labels whose exit date has already occurred by this decision date."""
        survivors: List[PendingSnapshot] = []
        for item in self.pending:
            if item.exit_date > decision_date:
                survivors.append(item)
                continue
            entries = prices_by_date.get(item.entry_date, {})
            exits = prices_by_date.get(item.exit_date, {})
            sample = item.frame.copy()
            sample["entry_open"] = sample["code"].map(lambda code: safe_float(entries.get(code, {}).get("open"), np.nan))
            sample["exit_close"] = sample["code"].map(lambda code: safe_float(exits.get(code, {}).get("close"), np.nan))
            sample["forward_return"] = sample["exit_close"] / sample["entry_open"] - 1.0
            sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=["forward_return"])
            row: Dict[str, object] = {
                "decision_date": item.decision_date,
                "entry_date": item.entry_date,
                "exit_date": item.exit_date,
                "sample_size": int(len(sample)),
            }
            for name in self.component_names:
                valid = sample[[name, "forward_return"]].dropna() if name in sample.columns else pd.DataFrame()
                ic = (
                    valid[name].rank(method="average").corr(
                        valid["forward_return"].rank(method="average"), method="pearson"
                    )
                    if len(valid) >= int(self.args.dynamic_min_cross_section)
                    else np.nan
                )
                row[name] = float(ic) if pd.notna(ic) else np.nan
            score_sample = (
                sample[["score_v2", "forward_return"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                if "score_v2" in sample.columns
                else pd.DataFrame()
            )
            if len(score_sample) >= int(self.args.dynamic_min_cross_section):
                lower = score_sample["forward_return"].quantile(0.01)
                upper = score_sample["forward_return"].quantile(0.99)
                score_sample["forward_return"] = score_sample["forward_return"].clip(lower, upper)
                centered = score_sample["score_v2"] - score_sample["score_v2"].mean()
                denominator = float(np.dot(centered, centered))
                slope = (
                    float(np.dot(centered, score_sample["forward_return"] - score_sample["forward_return"].mean()))
                    / denominator
                    if denominator > 1e-12
                    else np.nan
                )
                quantiles = score_sample["score_v2"].rank(pct=True)
                top = score_sample.loc[quantiles >= 0.80, "forward_return"].mean()
                bottom = score_sample.loc[quantiles <= 0.20, "forward_return"].mean()
                spread = float(top - bottom) if pd.notna(top) and pd.notna(bottom) else np.nan
            else:
                slope = np.nan
                spread = np.nan
            row["score_v2_forward_slope"] = float(slope) if pd.notna(slope) else np.nan
            row["score_v2_top_bottom_spread"] = float(spread) if pd.notna(spread) else np.nan
            self.ic_rows.append(row)
        self.pending = survivors

    def expected_score_return(self, decision_date: str) -> Tuple[float, Dict[str, object]]:
        history = pd.DataFrame(self.ic_rows)
        fallback = max(
            0.0,
            float(getattr(self.args, "economic_hurdle_fallback_return_per_score", 0.005)),
        )
        if history.empty or "score_v2_forward_slope" not in history.columns:
            return fallback, {
                "economic_score_slope_source": "fallback_warmup",
                "economic_score_slope_observations": 0,
            }
        history = history.loc[
            history["exit_date"].astype(str) <= str(decision_date)
        ].tail(int(getattr(self.args, "economic_hurdle_lookback_weeks", 52)))
        slopes = safe_series(history["score_v2_forward_slope"]).dropna()
        minimum = int(getattr(self.args, "economic_hurdle_min_observations", 16))
        if len(slopes) < minimum:
            return fallback, {
                "economic_score_slope_source": "fallback_warmup",
                "economic_score_slope_observations": int(len(slopes)),
            }
        cap = max(0.0, float(getattr(self.args, "economic_hurdle_max_return_per_score", 0.03)))
        mean_slope = float(slopes.clip(-cap, cap).mean())
        positive_ratio = float((slopes > 0).mean())
        # Shrink unstable historical slopes instead of selecting only positive weeks.
        estimated = max(0.0, mean_slope) * positive_ratio
        estimated = min(estimated, cap)
        return estimated, {
            "economic_score_slope_source": "causal_realized_labels",
            "economic_score_slope_observations": int(len(slopes)),
            "economic_score_slope_raw_mean": mean_slope,
            "economic_score_slope_positive_ratio": positive_ratio,
        }

    def weights(self, decision_date: str) -> Tuple[Dict[str, float], Dict[str, object]]:
        configured = configured_component_weights(self.args)
        static = pd.Series({name: configured.get(name, 0.0) for name in self.component_names}, dtype=float)
        if EVENT_COMPONENT in static.index:
            static.loc[EVENT_COMPONENT] = 0.0
        static = static / static.sum() if static.sum() > 0 else pd.Series(1.0 / len(static), index=static.index)

        if not bool(self.args.dynamic_factor_weights):
            output = static.to_dict()
            self._record(decision_date, output, "static")
            return output, {"weight_mode": "static", "ic_observations": 0}

        history = pd.DataFrame(self.ic_rows)
        if history.empty:
            output = static.to_dict()
            self._record(decision_date, output, "static_warmup")
            return output, {"weight_mode": "static_warmup", "ic_observations": 0}
        history = history.loc[history["exit_date"].astype(str) <= str(decision_date)].tail(int(self.args.dynamic_ic_lookback_weeks))
        if len(history) < int(self.args.dynamic_min_observations):
            output = static.to_dict()
            self._record(decision_date, output, "static_warmup")
            return output, {"weight_mode": "static_warmup", "ic_observations": int(len(history))}

        strengths: Dict[str, float] = {}
        diagnostics: Dict[str, object] = {}
        for name in self.component_names:
            series = safe_series(history.get(name, pd.Series(dtype=float))).dropna()
            mean_ic = series.mean() if len(series) else np.nan
            ic_std = series.std(ddof=1) if len(series) > 1 else np.nan
            positive_ratio = float((series > 0).mean()) if len(series) else 0.0
            # Positive, stable factor IC earns weight. Negative/unstable IC is set to zero.
            strength = max(0.0, float(mean_ic) if pd.notna(mean_ic) else 0.0)
            strength *= positive_ratio
            strength /= max(float(ic_std) if pd.notna(ic_std) else 0.0, float(self.args.dynamic_ic_std_floor))
            strengths[name] = strength
            diagnostics[f"ic_mean_{name}"] = float(mean_ic) if pd.notna(mean_ic) else np.nan
            diagnostics[f"ic_positive_ratio_{name}"] = positive_ratio

        raw = pd.Series(strengths, dtype=float)
        learned = raw / raw.sum() if raw.sum() > 0 else static.copy()
        blend = clip(float(self.args.dynamic_weight_strength), 0.0, 1.0)
        blended = (1.0 - blend) * static + blend * learned
        # No single factor gets to dominate merely because of a short lucky run.
        blended = blended.clip(lower=0.0, upper=float(self.args.dynamic_max_component_weight))
        output = (blended / blended.sum()).to_dict() if blended.sum() > 0 else static.to_dict()
        self._record(decision_date, output, "dynamic")
        diagnostics.update({"weight_mode": "dynamic", "ic_observations": int(len(history))})
        return output, diagnostics

    def _record(self, decision_date: str, weights: Mapping[str, float], mode: str) -> None:
        row: Dict[str, object] = {"decision_date": decision_date, "weight_mode": mode}
        for name in self.component_names:
            row[name] = float(weights.get(name, 0.0))
        self.weight_rows.append(row)


def apply_v2_score(
    features: pd.DataFrame,
    weights: Mapping[str, float],
    args,
    *,
    decision_date: Optional[str] = None,
    satellite_controller: Optional[IndustrySatelliteController] = None,
    market_risk_on_strength: float = 1.0,
) -> pd.DataFrame:
    result = features.copy()
    profile = str(getattr(args, "score_profile", "v2h4_legacy")).strip().lower()
    if profile == "v31":
        return apply_v31_score(result, weights)
    if profile == "china_small_v3":
        result["earnings_yield_score"] = base.factor_zscore(
            result,
            "risk_earnings_yield_raw",
            industry_neutral=not bool(args.disable_industry_neutral_factors),
        )
        result["residual_momentum_score"] = base.factor_zscore(
            result,
            "residual_momentum_raw",
            industry_neutral=False,
        )
    result = apply_v22_structural_components(
        result,
        defensive_industry_neutral=bool(
            getattr(args, "v22_defensive_industry_neutral", False)
        ),
        pure_industry_trend=bool(
            getattr(args, "v22_pure_industry_trend", False)
        ),
    )
    score = pd.Series(0.0, index=result.index)
    for name, weight in weights.items():
        if name in result.columns:
            score = score + float(weight) * safe_series(result[name], result.index).fillna(0.0)
    alpha_tilt_weight = clip(
        float(getattr(args, "v31_alpha_tilt_weight", 0.0)), 0.0, 1.0
    )
    if alpha_tilt_weight > 0.0:
        result["earnings_yield_score"] = base.factor_zscore(
            result,
            "v31_earnings_yield_raw",
            industry_neutral=True,
        )
        result["quality_score_v31"] = base.factor_zscore(
            result,
            "v31_quality_raw",
            industry_neutral=True,
        )
        alpha_score = (
            0.5 * result["earnings_yield_score"]
            + 0.5 * result["quality_score_v31"]
        )
        score = (
            (1.0 - alpha_tilt_weight) * base.zscore(score)
            + alpha_tilt_weight * base.zscore(alpha_score)
        )
    result["score_v2_core"] = score
    satellite_application = str(
        getattr(args, "v22_industry_satellite_application", "score_and_weight")
    ).strip().lower()
    satellite_arguments = {
        "maximum_weight": float(
            getattr(args, "v22_industry_satellite_max_weight", 0.0)
        ),
        "top_industries": int(
            getattr(args, "v22_industry_satellite_top_industries", 2)
        ),
        "excluded_industries": list(
            getattr(args, "v22_industry_satellite_excluded_industries", [])
            or []
        ),
    }
    if satellite_controller is not None and decision_date is not None:
        score, result, _ = satellite_controller.blend(
            result,
            score,
            decision_date=decision_date,
            market_risk_on_strength=market_risk_on_strength,
            **satellite_arguments,
        )
    else:
        score, result, _ = blend_continuous_industry_satellite(
            result,
            score,
            **satellite_arguments,
        )
    if satellite_application == "allocation_only":
        score = safe_series(result["score_v2_core"], result.index).fillna(0.0)
    if bool(args.enable_event_score) and EVENT_COMPONENT in result.columns:
        event = safe_series(result[EVENT_COMPONENT], result.index).fillna(0.0)
        # Event signal is deliberately capped and opt-in because the user's first
        # A/B test did not find a stable gain from direct event-score injection.
        score = score + clip(float(args.event_score_weight), 0.0, float(args.event_score_weight_cap)) * event
    result["score_v2"] = score
    return result.sort_values("score_v2", ascending=False).reset_index(drop=True)


def select_codes(
    features: pd.DataFrame,
    holdings: Mapping[str, int],
    args,
    target_equity_weight: float,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> List[str]:
    ranked = features.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    rank_by_code = ranked.set_index("code")["rank"].to_dict()
    retention_rank_by_code = rank_by_code
    if (
        str(getattr(args, "v22_industry_satellite_application", "score_and_weight"))
        .strip()
        .lower()
        in {"entry_only", "entry_and_weight"}
        and "score_v2_core" in ranked
    ):
        core_ranked = ranked.sort_values(
            ["score_v2_core", "code"], ascending=[False, True]
        ).copy()
        core_ranked["retention_rank"] = np.arange(1, len(core_ranked) + 1)
        retention_rank_by_code = core_ranked.set_index("code")[
            "retention_rank"
        ].to_dict()
    ranked["industry_1"] = normalize_industry_series(ranked["industry_1"])
    industry_by_code = ranked.set_index("code")["industry_1"].to_dict()

    needed_for_cap = int(math.ceil(target_equity_weight / max(float(args.max_stock_weight), 1e-6)))
    desired_count = max(int(args.target_count), needed_for_cap, int(args.min_target_count))
    desired_count = min(desired_count, len(ranked))
    industry_counts: Dict[str, int] = {}
    selected: List[str] = []

    def maximum_count(industry: str) -> int:
        fraction = (
            float(industry_caps.get(industry, args.max_industry_weight))
            if industry_caps is not None
            else float(args.max_industry_weight)
        )
        if fraction <= 0:
            return 0
        return max(
            1,
            int(
                math.ceil(
                    desired_count
                    * fraction
                    / max(float(target_equity_weight), 1e-8)
                    - 1e-9
                )
            ),
        )

    def add(code: str) -> bool:
        if code in selected:
            return False
        industry = industry_by_code.get(code, "UNKNOWN")
        if industry_counts.get(industry, 0) >= maximum_count(industry):
            return False
        selected.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return True

    kept = [
        code
        for code, shares in holdings.items()
        if int(shares) > 0
        and retention_rank_by_code.get(code, math.inf) <= int(args.sell_rank)
    ]
    for code in sorted(
        kept, key=lambda item: retention_rank_by_code.get(item, math.inf)
    ):
        if len(selected) >= desired_count:
            break
        add(code)

    # First add the intended buy universe. Then expand farther down the ranking only
    # when industry caps would otherwise leave too much capital unallocated.
    for code in ranked.loc[ranked["rank"] <= int(args.buy_rank), "code"].astype(str):
        if len(selected) >= desired_count:
            break
        add(code)
    if len(selected) < desired_count:
        for code in ranked["code"].astype(str):
            if len(selected) >= desired_count:
                break
            add(code)
    return selected


def select_lot_aware_codes(
    features: pd.DataFrame,
    holdings: Mapping[str, int],
    args,
    target_equity_weight: float,
    portfolio_value: float,
    decision_date: Optional[str] = None,
    expected_return_per_score: float = 0.0,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], pd.Series, Dict[str, object]]:
    if str(getattr(args, "portfolio_constructor", "legacy")).strip().lower() == "integer_cost_aware":
        return select_cost_aware_codes(
            features,
            holdings,
            target_equity_weight,
            portfolio_value,
            target_count=int(args.target_count),
            minimum_holdings=int(getattr(args, "lot_aware_min_holdings", args.target_count)),
            buy_rank=int(args.buy_rank),
            sell_rank=int(args.sell_rank),
            maximum_industry_weight=float(args.max_industry_weight),
            slippage_bps=float(args.slippage_bps),
            decision_date=str(decision_date or "9999-12-31"),
            expected_return_per_score=float(expected_return_per_score),
            hurdle_buffer_bps=float(getattr(args, "economic_hurdle_buffer_bps", 10.0)),
            broker_commission_rate=float(args.broker_commission_rate),
            broker_minimum_commission=float(args.broker_minimum_commission),
            replacement_policy=resolved_economic_replacement_policy(args),
            industry_caps=industry_caps,
        )
    ranked = features.copy().reset_index(drop=True)
    ranked["code"] = ranked["code"].astype(str).str.zfill(6)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    rank_by_code = ranked.set_index("code")["rank"].to_dict()
    retention_rank_by_code = rank_by_code
    if (
        str(getattr(args, "v22_industry_satellite_application", "score_and_weight"))
        .strip()
        .lower()
        in {"entry_only", "entry_and_weight"}
        and "score_v2_core" in ranked
    ):
        core_ranked = ranked.sort_values(
            ["score_v2_core", "code"], ascending=[False, True]
        ).copy()
        core_ranked["retention_rank"] = np.arange(1, len(core_ranked) + 1)
        retention_rank_by_code = core_ranked.set_index("code")[
            "retention_rank"
        ].to_dict()
    row_by_code = ranked.set_index("code").to_dict("index")
    equity_budget = max(0.0, float(portfolio_value) * float(target_equity_weight))
    target_count = min(max(1, int(args.target_count)), len(ranked))
    min_holdings = min(target_count, max(1, int(getattr(args, "lot_aware_min_holdings", 5))))
    max_lot_budget = equity_budget / min_holdings if min_holdings > 0 else equity_budget
    slippage = max(0.0, float(args.slippage_bps)) / 10000.0

    kept = [
        code
        for code, shares in holdings.items()
        if int(shares) > 0
        and retention_rank_by_code.get(str(code).zfill(6), math.inf)
        <= int(args.sell_rank)
    ]
    candidate_order: List[str] = []
    for code in sorted(
        kept,
        key=lambda item: retention_rank_by_code.get(str(item).zfill(6), math.inf),
    ):
        code = str(code).zfill(6)
        if code not in candidate_order:
            candidate_order.append(code)
    for code in ranked.loc[ranked["rank"] <= int(args.buy_rank), "code"]:
        if code not in candidate_order:
            candidate_order.append(code)
    for code in ranked["code"]:
        if code not in candidate_order:
            candidate_order.append(code)

    lot_values: Dict[str, float] = {}
    industries: Dict[str, str] = {}
    eligible_order: List[str] = []
    skipped_lot_too_expensive = 0
    for code in candidate_order:
        row = row_by_code.get(code, {})
        close_field = "execution_close" if "execution_close" in row else "close"
        close = safe_float(row.get(close_field), np.nan)
        if not math.isfinite(close) or close <= 0:
            continue
        minimum, _ = buy_order_size_rules(code)
        lot_value = float(minimum) * close * (1.0 + slippage)
        if lot_value > max_lot_budget + 1e-8:
            skipped_lot_too_expensive += 1
            continue
        lot_values[code] = lot_value
        raw_industry = row.get("industry_1")
        industries[code] = (
            "UNKNOWN"
            if pd.isna(raw_industry)
            or str(raw_industry).strip().upper() in {"", "NAN", "NONE", "NULL", "--", "UNKNOWN"}
            else str(raw_industry).strip()
        )
        eligible_order.append(code)

    provisional: List[str] = []
    reserved = 0.0
    for code in eligible_order:
        lot_value = lot_values[code]
        if reserved + lot_value > equity_budget + 1e-8:
            continue
        provisional.append(code)
        reserved += lot_value
        if len(provisional) >= target_count:
            break

    desired_count = len(provisional)
    def industry_cap(industry: str) -> float:
        if industry_caps is None:
            return float(args.max_industry_weight)
        return max(
            0.0,
            float(industry_caps.get(industry, args.max_industry_weight)),
        )

    def maximum_count(industry: str) -> int:
        cap = industry_cap(industry)
        if cap <= 0:
            return 0
        return max(
            1,
            int(
                math.ceil(
                    desired_count
                    * cap
                    / max(float(target_equity_weight), 1e-8)
                    - 1e-9
                )
            ),
        )
    selected: List[str] = []
    industry_counts: Dict[str, int] = {}
    reserved = 0.0
    for code in eligible_order:
        industry = industries[code]
        lot_value = lot_values[code]
        if industry_counts.get(industry, 0) >= maximum_count(industry):
            continue
        if reserved + lot_value > equity_budget + 1e-8:
            continue
        industry_minimum = sum(
            lot_values[item]
            for item in selected
            if industries[item] == industry
        )
        if (
            industry_minimum + lot_value
            > float(portfolio_value) * industry_cap(industry) + 1e-8
        ):
            continue
        selected.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        reserved += lot_value
        if len(selected) >= desired_count:
            break

    minimum_weights = pd.Series(
        {code: lot_values[code] / float(portfolio_value) for code in selected},
        dtype=float,
    )
    return selected, minimum_weights, {
        "lot_aware": True,
        "affordable_count": int(len(selected)),
        "minimum_lot_budget": float(reserved),
        "max_single_lot_budget": float(max_lot_budget),
        "skipped_lot_too_expensive": int(skipped_lot_too_expensive),
    }


def cap_and_redistribute(
    raw: pd.Series,
    industries: pd.Series,
    target_equity_weight: float,
    max_stock_weight: float,
    max_industry_weight: float,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """Allocate all feasible equity budget while satisfying stock/industry caps."""
    index = raw.index
    if len(index) == 0 or target_equity_weight <= 0:
        return pd.Series(0.0, index=index)
    raw = safe_series(raw, index).fillna(0.0).clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=index)
    weights = raw / raw.sum() * target_equity_weight
    industries = normalize_industry_series(industries.reindex(index))

    def group_cap(industry: str) -> float:
        if industry_caps is None:
            return float(max_industry_weight)
        return max(
            0.0,
            float(industry_caps.get(str(industry), max_industry_weight)),
        )

    for _ in range(100):
        previous = weights.copy()
        weights = weights.clip(upper=max_stock_weight)
        for industry, members in industries.groupby(industries).groups.items():
            member_index = list(members)
            total = float(weights.loc[member_index].sum())
            cap = group_cap(industry)
            if total > cap + 1e-12:
                weights.loc[member_index] *= cap / total

        deficit = target_equity_weight - float(weights.sum())
        if deficit <= 1e-8:
            break
        industry_total = weights.groupby(industries).sum()
        capacity = (max_stock_weight - weights).clip(lower=0.0)
        industry_capacity = industries.map(
            lambda industry: group_cap(industry)
            - float(industry_total.get(industry, 0.0))
        ).clip(lower=0.0)
        eligible = (capacity > 1e-10) & (industry_capacity > 1e-10)
        if not eligible.any():
            break
        extra = raw.where(eligible, 0.0)
        if extra.sum() <= 0:
            extra = capacity.where(eligible, 0.0)
        extra = extra / extra.sum() * deficit
        extra = np.minimum(extra, capacity)
        # A first industry cap pass; repeated outer iterations finish redistribution.
        for industry, members in industries.groupby(industries).groups.items():
            member_index = list(members)
            allowed = max(
                0.0,
                group_cap(industry) - float(weights.loc[member_index].sum()),
            )
            amount = float(extra.loc[member_index].sum())
            if amount > allowed + 1e-12 and amount > 0:
                extra.loc[member_index] *= allowed / amount
        weights += extra
        if float((weights - previous).abs().sum()) < 1e-9:
            break
    return weights.clip(lower=0.0)


def cap_and_redistribute_with_minimums(
    raw: pd.Series,
    industries: pd.Series,
    minimum_weights: pd.Series,
    target_equity_weight: float,
    max_stock_weight: float,
    max_industry_weight: float,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    index = raw.index
    raw = safe_series(raw, index).fillna(0.0).clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=index)
    industries = normalize_industry_series(industries.reindex(index))
    weights = safe_series(minimum_weights, index).fillna(0.0).clip(lower=0.0)

    def group_cap(industry: str) -> float:
        if industry_caps is None:
            return float(max_industry_weight)
        return max(
            0.0,
            float(industry_caps.get(str(industry), max_industry_weight)),
        )

    if float(weights.sum()) > target_equity_weight + 1e-10:
        return weights

    for _ in range(100):
        deficit = target_equity_weight - float(weights.sum())
        if deficit <= 1e-8:
            break
        industry_total = weights.groupby(industries).sum()
        stock_capacity = (max_stock_weight - weights).clip(lower=0.0)
        industry_capacity = industries.map(
            lambda industry: group_cap(industry)
            - float(industry_total.get(industry, 0.0))
        ).clip(lower=0.0)
        eligible = (stock_capacity > 1e-10) & (industry_capacity > 1e-10)
        if not eligible.any():
            break
        extra = raw.where(eligible, 0.0)
        if extra.sum() <= 0:
            extra = stock_capacity.where(eligible, 0.0)
        extra = extra / extra.sum() * deficit
        extra = np.minimum(extra, stock_capacity)
        for industry, members in industries.groupby(industries).groups.items():
            member_index = list(members)
            allowed = max(
                0.0,
                group_cap(industry) - float(weights.loc[member_index].sum()),
            )
            amount = float(extra.loc[member_index].sum())
            if amount > allowed + 1e-12 and amount > 0:
                extra.loc[member_index] *= allowed / amount
        added = float(extra.sum())
        weights += extra
        if added < 1e-10:
            break
    return weights.clip(lower=0.0)


def apply_v22_satellite_allocation(
    desired: pd.Series,
    raw: pd.Series,
    frame: pd.DataFrame,
    minimum_weights: pd.Series,
    target_equity_weight: float,
    max_stock_weight: float,
    max_industry_weight: float,
    lot_aware: bool,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Reserve the signaled fraction for stocks in the leading industries."""

    default_meta: Dict[str, object] = {
        "v22_satellite_weight": 0.0,
        "v22_leadership_strength": 0.0,
        "v22_market_risk_on_strength": 1.0,
        "v22_leading_industries": "",
        "v22_satellite_schedule": "weekly",
        "v22_satellite_signal_period": None,
    }
    if (
        desired.empty
        or "v22_industry_satellite_weight" not in frame
        or "v22_industry_satellite_leader" not in frame
    ):
        return desired, default_meta

    strength_values = pd.to_numeric(
        frame.get(
            "v22_industry_leadership_strength",
            pd.Series(0.0, index=frame.index),
        ),
        errors="coerce",
    ).dropna()
    risk_values = pd.to_numeric(
        frame.get(
            "v22_market_risk_on_strength",
            pd.Series(1.0, index=frame.index),
        ),
        errors="coerce",
    ).dropna()
    schedule_values = frame.get(
        "v22_satellite_schedule",
        pd.Series("weekly", index=frame.index),
    ).dropna()
    period_values = frame.get(
        "v22_satellite_signal_period",
        pd.Series(dtype=object),
    ).dropna()
    default_meta.update(
        {
            "v22_leadership_strength": (
                float(strength_values.max()) if not strength_values.empty else 0.0
            ),
            "v22_market_risk_on_strength": (
                float(risk_values.max()) if not risk_values.empty else 1.0
            ),
            "v22_satellite_schedule": (
                str(schedule_values.iloc[0]) if not schedule_values.empty else "weekly"
            ),
            "v22_satellite_signal_period": (
                str(period_values.iloc[0]) if not period_values.empty else None
            ),
        }
    )

    fraction_values = pd.to_numeric(
        frame["v22_industry_satellite_weight"], errors="coerce"
    ).dropna()
    fraction = float(fraction_values.max()) if not fraction_values.empty else 0.0
    fraction = float(np.clip(fraction, 0.0, 0.50))
    leader_mask = frame["v22_industry_satellite_leader"].fillna(False).astype(bool)
    if fraction <= 0.0 or not leader_mask.any():
        return desired, default_meta

    satellite_raw = safe_series(raw, frame.index).fillna(0.0).clip(lower=0.0)
    satellite_raw = satellite_raw.where(leader_mask, 0.0)
    if satellite_raw.sum() <= 0.0:
        return desired, default_meta

    if lot_aware:
        def allocator(values):
            return cap_and_redistribute_with_minimums(
                values,
                frame["industry_1"],
                minimum_weights,
                target_equity_weight,
                max_stock_weight,
                max_industry_weight,
                industry_caps=industry_caps,
            )
    else:
        def allocator(values):
            return cap_and_redistribute(
                values,
                frame["industry_1"],
                target_equity_weight,
                max_stock_weight,
                max_industry_weight,
                industry_caps=industry_caps,
            )
    satellite = allocator(satellite_raw)
    blended_raw = (1.0 - fraction) * desired + fraction * satellite
    blended = allocator(blended_raw)
    industries = normalize_industry_series(frame["industry_1"])
    leaders = sorted(set(industries.loc[leader_mask].astype(str)))
    return blended, {
        "v22_satellite_weight": fraction,
        "v22_leadership_strength": default_meta["v22_leadership_strength"],
        "v22_market_risk_on_strength": default_meta["v22_market_risk_on_strength"],
        "v22_leading_industries": "|".join(leaders),
        "v22_satellite_schedule": default_meta["v22_satellite_schedule"],
        "v22_satellite_signal_period": default_meta["v22_satellite_signal_period"],
    }


def build_targets_v2(
    features: pd.DataFrame,
    holdings: Mapping[str, int],
    current_weights: Mapping[str, float],
    args,
    target_equity_weight: float,
    portfolio_value: Optional[float] = None,
    force_risk_alignment: bool = False,
    decision_date: Optional[str] = None,
    expected_return_per_score: float = 0.0,
    target_transform: Optional[
        Callable[
            [pd.Series, pd.DataFrame, float, float],
            Tuple[pd.Series, Mapping[str, object]],
        ]
    ] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    if features.empty or target_equity_weight <= 0:
        return {}, {"selected_count": 0, "target_weight_sum": 0.0}
    profile = str(getattr(args, "score_profile", "v2h4_legacy")).strip().lower()
    industry_caps: Optional[Dict[str, float]] = None
    industry_market_weights: Dict[str, float] = {}
    industry_budget_mode = resolved_v31_industry_budget_mode(args)
    if industry_budget_mode == "soft":
        industry_caps, industry_market_weights = industry_budget_caps(
            features,
            args,
            target_equity_weight,
        )
    elif bool(getattr(args, "v31_enforce_unknown_industry_cap", False)):
        industry_caps = {
            "UNKNOWN": max(0.0, float(args.v31_unknown_industry_cap)),
        }
    lot_meta: Dict[str, object] = {"lot_aware": False}
    minimum_weights = pd.Series(dtype=float)
    if bool(getattr(args, "enable_lot_aware_selection", False)) and portfolio_value and portfolio_value > 0:
        selected, minimum_weights, lot_meta = select_lot_aware_codes(
            features,
            holdings,
            args,
            target_equity_weight,
            float(portfolio_value),
            decision_date=decision_date,
            expected_return_per_score=expected_return_per_score,
            industry_caps=industry_caps,
        )
    else:
        selected = select_codes(
            features,
            holdings,
            args,
            target_equity_weight,
            industry_caps=industry_caps,
        )
    if not selected:
        return {}, {"selected_count": 0, "target_weight_sum": 0.0}

    frame = features.set_index("code").loc[selected].copy()
    score_column = "score_v2"
    if (
        str(getattr(args, "v22_industry_satellite_application", "score_and_weight"))
        .strip()
        .lower()
        == "entry_only"
        and "score_v2_core" in frame
    ):
        score_column = "score_v2_core"
    score = safe_series(frame[score_column], frame.index).fillna(0.0)
    centered = (score - score.max()) / max(float(args.score_temperature), 1e-6)
    score_weight = np.exp(centered.clip(-30, 30))
    volatility = safe_series(frame.get("volatility_120", pd.Series(np.nan, index=frame.index)), frame.index)
    fallback_vol = volatility[volatility > 0].median()
    fallback_vol = fallback_vol if pd.notna(fallback_vol) and fallback_vol > 0 else 0.30
    risk_scale = volatility.fillna(fallback_vol).clip(lower=float(args.min_stock_volatility))
    raw = score_weight / (risk_scale ** float(args.inverse_vol_power))
    effective_max_stock_weight = float(args.max_stock_weight)
    effective_max_industry_weight = float(args.max_industry_weight)
    if bool(lot_meta.get("lot_aware")):
        count = max(1, len(selected))
        max_minimum = float(minimum_weights.max()) if not minimum_weights.empty else 0.0
        effective_max_stock_weight = min(
            float(getattr(args, "lot_aware_max_stock_weight", 0.25)),
            max(
                effective_max_stock_weight,
                target_equity_weight / count * float(getattr(args, "lot_aware_stock_cap_multiplier", 1.25)),
                max_minimum,
            ),
        )
        minimum_by_industry = minimum_weights.groupby(
            normalize_industry_series(frame["industry_1"])
        ).sum()
        effective_max_industry_weight = min(
            float(getattr(args, "lot_aware_max_industry_weight", 0.50)),
            max(
                effective_max_industry_weight,
                effective_max_stock_weight * 2.0,
                float(minimum_by_industry.max()) if not minimum_by_industry.empty else 0.0,
            ),
        )
        desired = cap_and_redistribute_with_minimums(
            raw,
            frame["industry_1"],
            minimum_weights,
            target_equity_weight,
            effective_max_stock_weight,
            effective_max_industry_weight,
            industry_caps=industry_caps,
        )
    else:
        desired = cap_and_redistribute(
            raw,
            frame["industry_1"],
            target_equity_weight,
            effective_max_stock_weight,
            effective_max_industry_weight,
            industry_caps=industry_caps,
        )
    satellite_desired, v22_meta = apply_v22_satellite_allocation(
        desired,
        raw,
        frame,
        minimum_weights,
        target_equity_weight,
        effective_max_stock_weight,
        effective_max_industry_weight,
        bool(lot_meta.get("lot_aware")),
        industry_caps=industry_caps,
    )
    if (
        str(getattr(args, "v22_industry_satellite_application", "score_and_weight"))
        .strip()
        .lower()
        != "entry_only"
    ):
        desired = satellite_desired
    transform_meta: Dict[str, object] = {}
    if target_transform is not None:
        transformed, returned_meta = target_transform(
            desired.copy(),
            frame.copy(),
            effective_max_stock_weight,
            effective_max_industry_weight,
        )
        desired = safe_series(transformed, frame.index).fillna(0.0).clip(lower=0.0)
        transform_meta = dict(returned_meta or {})
        if industry_caps is not None and float(desired.sum()) > 0:
            desired = cap_and_redistribute(
                desired,
                frame["industry_1"],
                float(desired.sum()),
                effective_max_stock_weight,
                effective_max_industry_weight,
                industry_caps=industry_caps,
            )

    # Avoid spending money on trivial changes; keep target allocations otherwise.
    target = desired.to_dict()
    if not force_risk_alignment:
        band = max(0.0, float(args.rebalance_band_weight))
        for code in list(set(target).union(current_weights)):
            desired_weight = float(target.get(code, 0.0))
            current_weight = float(current_weights.get(code, 0.0))
            if abs(desired_weight - current_weight) < band:
                target[code] = current_weight
    # Preserve the desired target when the band leaves a small residual; the execution
    # layer still controls cash, lots and liquidity.
    target = {code: float(weight) for code, weight in target.items() if float(weight) > 0}
    all_industries = (
        features.drop_duplicates("code", keep="last")
        .set_index("code")["industry_1"]
    )
    if industry_caps is not None and target:
        bounded = cap_and_redistribute(
            pd.Series(target, dtype=float),
            all_industries,
            min(float(sum(target.values())), float(target_equity_weight)),
            effective_max_stock_weight,
            effective_max_industry_weight,
            industry_caps=industry_caps,
        )
        target = {
            code: float(weight)
            for code, weight in bounded.items()
            if float(weight) > 0
        }
    normalized_industries = normalize_industry_series(
        all_industries.reindex(pd.Index(target, dtype=object))
    )
    unknown_target_weight = float(
        sum(
            weight
            for code, weight in target.items()
            if normalized_industries.get(code, "UNKNOWN") == "UNKNOWN"
        )
    )
    return target, {
        "selected_count": int(len(selected)),
        "target_weight_sum": float(sum(target.values())),
        "desired_equity_weight": float(target_equity_weight),
        "effective_max_stock_weight": float(effective_max_stock_weight),
        "effective_max_industry_weight": float(effective_max_industry_weight),
        "industry_budget_caps": industry_caps,
        "industry_market_weights": industry_market_weights,
        "v31_industry_budget_mode": industry_budget_mode,
        "v31_unknown_industry_cap_enforced": bool(
            getattr(args, "v31_enforce_unknown_industry_cap", False)
            or industry_budget_mode == "soft"
        ),
        "v31_alpha_tilt_weight": float(
            getattr(args, "v31_alpha_tilt_weight", 0.0)
        ),
        "unknown_target_weight": unknown_target_weight,
        "force_risk_alignment": bool(force_risk_alignment),
        **v22_meta,
        **transform_meta,
        **lot_meta,
    }


def should_force_risk_alignment(args, current_equity_weight: float, target_equity_weight: float) -> bool:
    mode = str(getattr(args, "risk_target_alignment", "banded")).strip().lower()
    if mode != "strict":
        return False
    return (
        abs(float(current_equity_weight) - float(target_equity_weight))
        > float(args.risk_rebalance_band)
    )


def should_run_unscheduled_risk_rebalance(
    args,
    current_equity_weight: float,
    target_equity_weight: float,
) -> bool:
    schedule = str(getattr(args, "risk_rebalance_schedule", "daily")).strip().lower()
    if schedule != "daily":
        return False
    return (
        float(current_equity_weight)
        > float(target_equity_weight) + float(args.risk_rebalance_band)
    )


def select_sparse_risk_alignment_orders(
    planned: Sequence[Mapping[str, object]],
    current_equity_weight: float,
    target_equity_weight: float,
    portfolio_value: float,
    max_orders: int,
    initial_max_orders: int = 0,
) -> List[Dict[str, object]]:
    items = [dict(item) for item in planned]
    cap = max(0, int(max_orders))
    if cap <= 0 or not items:
        return items

    risk_items = [
        item
        for item in items
        if str(item.get("risk_alignment_mode", "")).upper() in {"REDUCE", "INCREASE"}
    ]
    if not risk_items:
        return items

    initial_cap = max(0, int(initial_max_orders))
    if (
        initial_cap > 0
        and float(current_equity_weight) <= 0.01
        and float(target_equity_weight) > 0.01
    ):
        cap = initial_cap

    active_mode = (
        "REDUCE"
        if float(current_equity_weight) > float(target_equity_weight)
        else "INCREASE"
    )
    active = [
        item
        for item in risk_items
        if str(item.get("risk_alignment_mode", "")).upper() == active_mode
    ]
    if not active:
        return items

    active.sort(key=lambda item: float(item.get("gross_amount", 0.0)), reverse=True)
    required_gross = (
        abs(float(current_equity_weight) - float(target_equity_weight))
        * max(0.0, float(portfolio_value))
    )
    chosen: List[Dict[str, object]] = []
    cumulative_gross = 0.0
    for item in active:
        if len(chosen) >= cap:
            break
        chosen.append(item)
        cumulative_gross += max(0.0, float(item.get("gross_amount", 0.0)))
        if cumulative_gross + 1e-8 >= required_gross:
            break

    dropped = max(0, len(items) - len(chosen))
    for item in chosen:
        item["sparse_risk_execution"] = True
        item["sparse_risk_orders_dropped"] = dropped
        item["risk_alignment_order_cap"] = cap
    return chosen


def balance_executable_replacement_orders(
    planned: Sequence[Mapping[str, object]],
    current_equity_weight: float,
    target_equity_weight: float,
    portfolio_value: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Prevent filtered or unmarketable replacement legs from changing net equity."""
    items = [dict(item) for item in planned]
    value = max(0.0, float(portfolio_value))
    current = max(0.0, float(current_equity_weight))
    target = max(0.0, float(target_equity_weight))

    def is_actionable(item: Mapping[str, object]) -> bool:
        return bool(item.get("auction_marketable", True))

    def gross(item: Mapping[str, object]) -> float:
        return max(0.0, safe_float(item.get("gross_amount"), 0.0))

    actionable_buys = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("side", "")).upper() == "BUY" and is_actionable(item)
    ]
    actionable_sells = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("side", "")).upper() == "SELL" and is_actionable(item)
    ]

    def priority(pair, side: str):
        _, item = pair
        mode = str(item.get("risk_alignment_mode", "")).upper()
        transition = str(item.get("transition_type", "")).upper()
        rank = safe_float(item.get("rank"), np.nan)
        target_weight = max(0.0, safe_float(item.get("target_weight"), 0.0))
        if side == "SELL":
            return (
                0 if mode == "REDUCE" else 1,
                0 if transition == "EXIT" else 1,
                -rank if math.isfinite(rank) else float("-inf"),
                -gross(item),
            )
        return (
            0 if mode == "INCREASE" else 1,
            rank if math.isfinite(rank) else float("inf"),
            -target_weight,
            gross(item),
        )

    def select_nearest_budget(candidates, budget: float, side: str):
        selected = set()
        used = 0.0
        target_budget = max(0.0, float(budget))
        for index, item in sorted(candidates, key=lambda pair: priority(pair, side)):
            amount = gross(item)
            if amount <= 0:
                continue
            current_error = abs(target_budget - used)
            candidate_error = abs(target_budget - (used + amount))
            if candidate_error + 1e-8 < current_error:
                selected.add(index)
                used += amount
        return selected, used

    total_buy = sum(gross(item) for _, item in actionable_buys)
    total_sell = sum(gross(item) for _, item in actionable_sells)
    desired_net_buy = (target - current) * value

    if desired_net_buy > 0:
        sell_budget = max(0.0, total_buy - desired_net_buy)
        kept_sells, kept_sell_gross = select_nearest_budget(
            actionable_sells, sell_budget, "SELL"
        )
        buy_budget = desired_net_buy + kept_sell_gross
        if current <= 0.01:
            # Initial deployment already went through portfolio-level lot rounding.
            kept_buys = {index for index, _ in actionable_buys}
            kept_buy_gross = total_buy
        else:
            kept_buys, kept_buy_gross = select_nearest_budget(
                actionable_buys, buy_budget, "BUY"
            )
    elif desired_net_buy < 0:
        required_net_sell = -desired_net_buy
        buy_budget = max(0.0, total_sell - required_net_sell)
        kept_buys, kept_buy_gross = select_nearest_budget(
            actionable_buys, buy_budget, "BUY"
        )
        sell_budget = required_net_sell + kept_buy_gross
        kept_sells, kept_sell_gross = select_nearest_budget(
            actionable_sells, sell_budget, "SELL"
        )
    elif total_buy <= total_sell:
        kept_buys = {index for index, _ in actionable_buys}
        kept_buy_gross = total_buy
        kept_sells, kept_sell_gross = select_nearest_budget(
            actionable_sells, kept_buy_gross, "SELL"
        )
    else:
        kept_sells = {index for index, _ in actionable_sells}
        kept_sell_gross = total_sell
        kept_buys, kept_buy_gross = select_nearest_budget(
            actionable_buys, kept_sell_gross, "BUY"
        )

    actionable_indexes = {
        index for index, _ in actionable_buys + actionable_sells
    }
    kept_actionable_indexes = kept_buys | kept_sells
    kept = [
        item
        for index, item in enumerate(items)
        if index not in actionable_indexes or index in kept_actionable_indexes
    ]
    deferred_buy_count = len(actionable_buys) - len(kept_buys)
    deferred_sell_count = len(actionable_sells) - len(kept_sells)
    diagnostics = {
        "replacement_guard_applied": bool(
            deferred_buy_count > 0 or deferred_sell_count > 0
        ),
        "desired_net_buy_gross": float(desired_net_buy),
        "executable_buy_gross_before_guard": float(total_buy),
        "executable_sell_gross_before_guard": float(total_sell),
        "executable_buy_gross_after_guard": float(kept_buy_gross),
        "executable_sell_gross_after_guard": float(kept_sell_gross),
        "deferred_replacement_buy_count": int(deferred_buy_count),
        "deferred_replacement_sell_count": int(deferred_sell_count),
        "deferred_replacement_buy_gross": float(total_buy - kept_buy_gross),
        "deferred_replacement_sell_gross": float(total_sell - kept_sell_gross),
    }
    return kept, diagnostics


def execution_price(open_price: float, side: str, slippage_bps: float) -> float:
    slip = max(0.0, float(slippage_bps)) / 10000.0
    return open_price * (1.0 + slip) if side == "BUY" else open_price * (1.0 - slip)


def configured_trade_cost(
    gross: float,
    side: str,
    trade_date: str,
    code: str,
    args,
    shares: Optional[int] = None,
) -> float:
    return mandatory_trade_cost(
        gross,
        side,
        trade_date,
        code,
        broker_commission_rate=float(getattr(args, "broker_commission_rate", 0.0)),
        broker_minimum_commission=float(getattr(args, "broker_minimum_commission", 0.0)),
        shares=shares,
    )


def execute_trades_v2(
    trade_date: str,
    decision_date: str,
    holdings: Dict[str, int],
    cash: float,
    targets: Mapping[str, float],
    prices: Mapping[str, Mapping[str, object]],
    portfolio_open_value: float,
    liquidity_by_code: Mapping[str, float],
    args,
    decision_closes: Optional[Mapping[str, float]] = None,
    decision_portfolio_value: Optional[float] = None,
    opening_gap_estimator: Optional[CausalOpeningGapEstimator] = None,
    industries: Optional[Mapping[str, str]] = None,
    industry_caps: Optional[Mapping[str, float]] = None,
) -> Tuple[float, List[Dict[str, object]], Dict[str, object]]:
    execution_model = str(
        getattr(args, "execution_model", "next_open_fixed_bps_legacy")
    ).strip().lower()
    auction_mode = execution_model == "opening_auction_limit"
    if auction_mode and opening_gap_estimator is None:
        raise ValueError(
            "opening_auction_limit execution requires a causal opening-gap estimator."
        )
    decision_closes = decision_closes or {}
    planning_portfolio_value = safe_float(
        decision_portfolio_value, portfolio_open_value
    )
    if not math.isfinite(planning_portfolio_value) or planning_portfolio_value <= 0:
        planning_portfolio_value = float(portfolio_open_value)
    auction_estimates = {}

    def decision_reference_price(code, row):
        reference = safe_float(decision_closes.get(code), np.nan)
        if not math.isfinite(reference) or reference <= 0:
            reference = safe_float(row.get("prev_close"), np.nan)
        return reference

    def auction_estimate(code):
        if code not in auction_estimates:
            auction_estimates[code] = opening_gap_estimator.estimate(code)
        return auction_estimates[code]

    planned: List[Dict[str, object]] = []
    target_prices: Dict[str, float] = {}
    for code, target_weight in targets.items():
        row = prices.get(code)
        if row is None:
            continue
        raw_open = safe_float(row.get("open"), np.nan)
        if not math.isfinite(raw_open) or raw_open <= 0:
            continue
        current_shares = int(holdings.get(code, 0))
        if auction_mode:
            reference = decision_reference_price(code, row)
            if not math.isfinite(reference) or reference <= 0:
                continue
            side = (
                "BUY"
                if float(target_weight) * planning_portfolio_value
                > current_shares * reference
                else "SELL"
            )
            target_prices[code] = opening_auction_limit_price(
                reference,
                side,
                auction_estimate(code),
                float(getattr(args, "auction_limit_buffer_bps", 2.0)),
            )
        else:
            side = "BUY" if float(target_weight) * portfolio_open_value > current_shares * raw_open else "SELL"
            target_prices[code] = execution_price(raw_open, side, float(args.slippage_bps))
    for code in set(holdings) - set(targets):
        row = prices.get(code)
        if row is None:
            continue
        raw_open = safe_float(row.get("open"), np.nan)
        if not math.isfinite(raw_open) or raw_open <= 0:
            continue
        if auction_mode:
            reference = decision_reference_price(code, row)
            if not math.isfinite(reference) or reference <= 0:
                continue
            target_prices[code] = opening_auction_limit_price(
                reference,
                "SELL",
                auction_estimate(code),
                float(getattr(args, "auction_limit_buffer_bps", 2.0)),
            )
        else:
            target_prices[code] = execution_price(raw_open, "SELL", float(args.slippage_bps))
    integer_meta: Dict[str, object] = {"integer_optimizer_status": "legacy"}
    if str(getattr(args, "portfolio_constructor", "legacy")).strip().lower() == "integer_cost_aware":
        target_shares_by_code, integer_meta = optimize_discrete_target_shares(
            targets,
            planning_portfolio_value if auction_mode else portfolio_open_value,
            target_prices,
            holdings,
            cash,
            trade_date,
            broker_commission_rate=float(args.broker_commission_rate),
            broker_minimum_commission=float(args.broker_minimum_commission),
            minimum_final_holdings=int(
                getattr(args, "integer_optimizer_min_holdings", args.target_count)
            ),
            maximum_stock_weight=float(
                getattr(args, "lot_aware_max_stock_weight", args.max_stock_weight)
            ),
            tracking_penalty=float(getattr(args, "integer_tracking_penalty", 1.0)),
            cash_penalty=float(getattr(args, "integer_cash_penalty", 0.75)),
            transaction_cost_penalty=float(
                getattr(args, "integer_transaction_cost_penalty", 2.0)
            ),
            iterations=int(getattr(args, "integer_optimizer_iterations", 12)),
            industries=industries,
            industry_caps=industry_caps,
        )
    else:
        target_shares_by_code = round_portfolio_target_shares(
            targets,
            planning_portfolio_value if auction_mode else portfolio_open_value,
            target_prices,
        )
    weight_value = (
        planning_portfolio_value if auction_mode else float(portfolio_open_value)
    )
    current_equity_weight = (
        max(0.0, float(weight_value) - float(cash)) / float(weight_value)
        if float(weight_value) > 0
        else 0.0
    )
    target_equity_weight = float(sum(max(0.0, float(weight)) for weight in targets.values()))
    auction_blocked_by_price_limit = 0
    for code in sorted(set(holdings).union(targets)):
        row = prices.get(code)
        if row is None:
            continue
        raw_open = safe_float(row.get("open"), np.nan)
        if not math.isfinite(raw_open) or raw_open <= 0:
            continue
        current_shares = int(holdings.get(code, 0))
        target_weight = max(0.0, float(targets.get(code, 0.0)))
        target_shares = int(target_shares_by_code.get(code, 0)) if target_weight > 0 else 0
        trade_shares = int(target_shares - current_shares)
        if trade_shares == 0:
            continue
        side = "BUY" if trade_shares > 0 else "SELL"
        if base.blocked_by_price_limit(code, row, side, args):
            if auction_mode:
                auction_blocked_by_price_limit += 1
            continue
        reference = decision_reference_price(code, row)
        if auction_mode:
            if not math.isfinite(reference) or reference <= 0:
                continue
            estimate = auction_estimate(code)
            limit_price = opening_auction_limit_price(
                reference,
                side,
                estimate,
                float(getattr(args, "auction_limit_buffer_bps", 2.0)),
            )
            price = execution_price(
                raw_open,
                side,
                float(getattr(args, "auction_impact_bps", 0.0)),
            )
            auction_marketable = opening_auction_order_is_marketable(
                raw_open, limit_price, side
            ) and opening_auction_order_is_marketable(price, limit_price, side)
            sizing_price = limit_price
        else:
            estimate = None
            limit_price = np.nan
            auction_marketable = True
            price = execution_price(raw_open, side, float(args.slippage_bps))
            sizing_price = price
        order_floor, transition_type = trade_value_floor(
            planning_portfolio_value if auction_mode else portfolio_open_value,
            current_shares,
            target_shares,
            args.min_trade_value,
            getattr(args, "min_trade_weight", 0.0),
            getattr(args, "entry_exit_min_trade_value", None),
            getattr(args, "entry_exit_min_trade_weight", 0.0),
        )
        order_floor, risk_alignment_mode = apply_risk_alignment_trade_floor(
            order_floor,
            planning_portfolio_value if auction_mode else portfolio_open_value,
            side,
            current_equity_weight,
            target_equity_weight,
            args.risk_rebalance_band,
            getattr(args, "risk_reduction_min_trade_weight", 0.01),
            getattr(args, "risk_increase_min_trade_weight", 0.01),
        )
        if bool(getattr(args, "enforce_commission_efficient_floor", False)) and transition_type != "EXIT":
            order_floor = max(
                order_floor,
                commission_efficient_trade_floor(
                    float(args.broker_minimum_commission),
                    float(getattr(args, "max_broker_commission_fraction", 0.002)),
                ),
            )
        planning_gross = abs(trade_shares) * sizing_price
        if planning_gross < order_floor:
            continue
        # Do not pretend a backtest can trade a large fraction of a stock's daily volume.
        avg_amount = safe_float(liquidity_by_code.get(code), np.nan)
        if math.isfinite(avg_amount) and avg_amount > 0 and float(args.max_participation_rate) > 0:
            max_gross = avg_amount * float(args.max_participation_rate)
            minimum, increment = buy_order_size_rules(code)
            raw_max_shares = int(math.floor(max_gross / sizing_price))
            if raw_max_shares < minimum:
                continue
            max_shares = int(minimum + math.floor((raw_max_shares - minimum) / increment) * increment)
            trade_shares = int(math.copysign(min(abs(trade_shares), max_shares), trade_shares))
            planning_gross = abs(trade_shares) * sizing_price
            if planning_gross < order_floor:
                continue
        gross = abs(trade_shares) * price
        planned.append(
            {
                "trade_date": trade_date,
                "decision_date": decision_date,
                "code": code,
                "name": str(row.get("name", "")),
                "side": side,
                "transition_type": transition_type,
                "trade_value_floor": order_floor,
                "risk_alignment_mode": risk_alignment_mode,
                "risk_reduction_trade": risk_alignment_mode == "REDUCE",
                "risk_increase_trade": risk_alignment_mode == "INCREASE",
                "shares": abs(int(trade_shares)),
                "open_price": raw_open,
                "price": price,
                "slippage_bps": (
                    float(getattr(args, "auction_impact_bps", 0.0))
                    if auction_mode
                    else float(args.slippage_bps)
                ),
                "execution_model": execution_model,
                "decision_reference_close": reference,
                "auction_limit_price": limit_price,
                "auction_marketable": bool(auction_marketable),
                "auction_gap_observations": (
                    int(estimate.observations) if estimate is not None else 0
                ),
                "auction_expected_gap": (
                    float(estimate.expected_gap) if estimate is not None else np.nan
                ),
                "auction_lower_gap": (
                    float(estimate.lower_gap) if estimate is not None else np.nan
                ),
                "auction_upper_gap": (
                    float(estimate.upper_gap) if estimate is not None else np.nan
                ),
                "auction_expected_open_price": (
                    expected_open_price(reference, estimate)
                    if estimate is not None
                    else np.nan
                ),
                "gross_amount": gross,
                "target_weight": target_weight,
                "avg_amount_for_cap": avg_amount,
            }
        )

    planned = select_sparse_risk_alignment_orders(
        planned,
        current_equity_weight,
        target_equity_weight,
        portfolio_open_value,
        getattr(args, "risk_alignment_max_orders", 0),
        getattr(args, "risk_alignment_initial_max_orders", 0),
    )
    planned, replacement_guard = balance_executable_replacement_orders(
        planned,
        current_equity_weight,
        target_equity_weight,
        weight_value,
    )

    auction_attempts = len(planned) if auction_mode else 0
    auction_marketable = (
        sum(bool(item.get("auction_marketable", False)) for item in planned)
        if auction_mode
        else 0
    )
    executed: List[Dict[str, object]] = []
    for order in [
        item
        for item in planned
        if item["side"] == "SELL" and bool(item.get("auction_marketable", True))
    ]:
        code = str(order["code"])
        current_shares = int(holdings.get(code, 0))
        shares = min(int(order["shares"]), current_shares)
        if shares < current_shares:
            minimum, increment = buy_order_size_rules(code)
            if shares < minimum:
                continue
            shares = int(minimum + math.floor((shares - minimum) / increment) * increment)
        if shares <= 0:
            continue
        gross = shares * float(order["price"])
        if gross < float(order["trade_value_floor"]):
            continue
        fee = configured_trade_cost(gross, "SELL", trade_date, code, args, shares)
        holdings[code] = int(holdings.get(code, 0)) - shares
        if holdings[code] <= 0:
            holdings.pop(code, None)
        cash += gross - fee
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)

    buys = sorted(
        (
            item
            for item in planned
            if item["side"] == "BUY" and bool(item.get("auction_marketable", True))
        ),
        key=lambda item: item["target_weight"],
        reverse=True,
    )
    for order in buys:
        code = str(order["code"])
        minimum, increment = buy_order_size_rules(code)
        shares = int(order["shares"])
        if shares < minimum:
            continue
        shares = int(minimum + math.floor((shares - minimum) / increment) * increment)
        while shares >= minimum:
            gross = shares * float(order["price"])
            fee = configured_trade_cost(gross, "BUY", trade_date, code, args, shares)
            if gross + fee <= cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            continue
        gross = shares * float(order["price"])
        if gross < float(order["trade_value_floor"]):
            continue
        fee = configured_trade_cost(gross, "BUY", trade_date, code, args, shares)
        cash -= gross + fee
        holdings[code] = int(holdings.get(code, 0)) + shares
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)
    diagnostics = {
        "auction_order_attempts": int(auction_attempts),
        "auction_marketable_orders": int(auction_marketable),
        "auction_executed_orders": int(len(executed)) if auction_mode else 0,
        "auction_unmarketable_orders": int(auction_attempts - auction_marketable),
        "auction_marketable_not_executed": (
            int(auction_marketable - len(executed)) if auction_mode else 0
        ),
        "auction_blocked_by_price_limit": int(auction_blocked_by_price_limit),
        **integer_meta,
        **replacement_guard,
    }
    return float(cash), executed, diagnostics


def make_summary(equity: pd.DataFrame, trades: pd.DataFrame, initial_cash: float, final_value: float, args) -> Dict[str, object]:
    daily = safe_series(equity["daily_return"]).dropna()
    annual_return = (final_value / initial_cash) ** (244 / max(len(equity), 1)) - 1.0
    annual_vol = daily.std(ddof=1) * math.sqrt(244) if len(daily) > 1 else np.nan
    drawdown = equity["total_value"] / equity["total_value"].cummax() - 1.0
    auction_attempts = int(
        pd.to_numeric(
            equity.get("auction_order_attempts", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    auction_executed = int(
        pd.to_numeric(
            equity.get("auction_executed_orders", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    auction_unmarketable = int(
        pd.to_numeric(
            equity.get("auction_unmarketable_orders", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    overlay_applied = (
        equity.get("risk_overlay_status", pd.Series(dtype=str))
        .astype(str)
        .eq("applied")
    )
    return {
        "strategy": "factor_rank_v2_continuous_risk",
        "strategy_name": str(getattr(args, "strategy_name", "V2H")),
        "database": str(Path(args.database).resolve()),
        "database_before_cutover": (
            str(Path(args.database_before_cutover).resolve())
            if getattr(args, "database_before_cutover", None)
            else None
        ),
        "continuous_cutover_date": getattr(
            args, "continuous_cutover_date", None
        ),
        "feature_cache": (
            str(Path(args.feature_cache).resolve())
            if getattr(args, "feature_cache", None)
            else None
        ),
        "feature_cache_before_cutover": (
            str(Path(args.feature_cache_before_cutover).resolve())
            if getattr(args, "feature_cache_before_cutover", None)
            else None
        ),
        "risk_model_database": (
            str(Path(args.risk_model_database).resolve())
            if getattr(args, "risk_model_database", None)
            else None
        ),
        "risk_model_database_before_cutover": (
            str(Path(args.risk_model_database_before_cutover).resolve())
            if getattr(args, "risk_model_database_before_cutover", None)
            else None
        ),
        "score_profile": str(getattr(args, "score_profile", "v2h4_legacy")),
        "v31_industry_budget_mode": resolved_v31_industry_budget_mode(args),
        "v31_alpha_tilt_weight": float(
            getattr(args, "v31_alpha_tilt_weight", 0.0)
        ),
        "v31_unknown_industry_cap_enforced": bool(
            getattr(args, "v31_enforce_unknown_industry_cap", False)
            or resolved_v31_industry_budget_mode(args) == "soft"
        ),
        "v22_defensive_industry_neutral": bool(
            getattr(args, "v22_defensive_industry_neutral", False)
        ),
        "v22_pure_industry_trend": bool(
            getattr(args, "v22_pure_industry_trend", False)
        ),
        "v22_industry_satellite_max_weight": float(
            getattr(args, "v22_industry_satellite_max_weight", 0.0)
        ),
        "v22_industry_satellite_top_industries": int(
            getattr(args, "v22_industry_satellite_top_industries", 2)
        ),
        "v22_industry_satellite_excluded_industries": list(
            getattr(args, "v22_industry_satellite_excluded_industries", [])
            or []
        ),
        "v22_industry_satellite_schedule": str(
            getattr(args, "v22_industry_satellite_schedule", "weekly")
        ),
        "v22_industry_satellite_risk_throttle": str(
            getattr(args, "v22_industry_satellite_risk_throttle", "none")
        ),
        "v22_industry_satellite_application": str(
            getattr(args, "v22_industry_satellite_application", "score_and_weight")
        ),
        "average_v22_satellite_weight": float(
            pd.to_numeric(
                equity.get("v22_satellite_weight", pd.Series(dtype=float)),
                errors="coerce",
            ).mean()
        ),
        "maximum_v22_satellite_weight": float(
            pd.to_numeric(
                equity.get("v22_satellite_weight", pd.Series(dtype=float)),
                errors="coerce",
            ).max()
        ),
        "portfolio_constructor": str(getattr(args, "portfolio_constructor", "legacy")),
        "start_date": str(equity["trade_date"].iloc[0]),
        "end_date": str(equity["trade_date"].iloc[-1]),
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(final_value / initial_cash - 1.0),
        "annualized_return": float(annual_return),
        "annualized_volatility": float(annual_vol),
        "sharpe_no_risk_free": float(annual_return / annual_vol) if pd.notna(annual_vol) and annual_vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "trading_days": int(len(equity)),
        "trade_count": int(len(trades)),
        "buy_count": int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
        "sell_count": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
        "total_gross_traded": float(trades["gross_amount"].sum()) if not trades.empty else 0.0,
        "total_fees": float(trades["fee"].sum()) if not trades.empty else 0.0,
        "total_corporate_action_cash": float(equity.get("corporate_action_cash", pd.Series(dtype=float)).sum()),
        "corporate_action_days": int((equity.get("corporate_action_count", pd.Series(dtype=float)) > 0).sum()),
        "corporate_action_share_change_count": int(
            equity.get("corporate_action_share_changes", pd.Series(dtype=float)).sum()
        ),
        "average_daily_turnover": float(equity["turnover"].mean()),
        "average_rebalance_turnover": float(equity.loc[equity["rebalanced"], "turnover"].mean()),
        "average_target_equity_weight": float(equity["target_equity_weight"].mean()),
        "average_actual_equity_weight": float(equity["actual_equity_weight"].mean()),
        "min_actual_equity_weight": float(equity["actual_equity_weight"].min()),
        "max_actual_equity_weight": float(equity["actual_equity_weight"].max()),
        "risk_mode": str(args.risk_mode),
        "risk_overlay_mode": str(
            getattr(args, "risk_overlay_mode", "disabled")
        ),
        "risk_calibration_multiplier": float(
            getattr(args, "risk_calibration_multiplier", 1.0)
        ),
        "risk_calibration_mode": (
            "causal_expanding_schedule"
            if getattr(args, "risk_calibration_schedule", None)
            else "fixed"
        ),
        "risk_calibration_schedule": (
            str(Path(args.risk_calibration_schedule).resolve())
            if getattr(args, "risk_calibration_schedule", None)
            else None
        ),
        "average_risk_calibration_multiplier_used": float(
            pd.to_numeric(
                equity.get(
                    "risk_calibration_multiplier_used",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        ),
        "minimum_risk_calibration_multiplier_used": float(
            pd.to_numeric(
                equity.get(
                    "risk_calibration_multiplier_used",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).min()
        ),
        "maximum_risk_calibration_multiplier_used": float(
            pd.to_numeric(
                equity.get(
                    "risk_calibration_multiplier_used",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).max()
        ),
        "risk_overlay_applied_days": int(overlay_applied.sum()),
        "average_risk_predicted_volatility_before": float(
            pd.to_numeric(
                equity.get(
                    "risk_predicted_volatility_before",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        ),
        "average_risk_predicted_volatility_after": float(
            pd.to_numeric(
                equity.get(
                    "risk_predicted_volatility_after",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        ),
        "min_equity_weight": float(args.min_equity_weight),
        "target_count": int(args.target_count),
        "max_stock_weight": float(args.max_stock_weight),
        "max_industry_weight": float(args.max_industry_weight),
        "dynamic_factor_weights": bool(args.dynamic_factor_weights),
        "economic_replacement_policy": resolved_economic_replacement_policy(args),
        "economic_replacements_blocked": int(
            pd.to_numeric(
                equity.get("economic_replacements_blocked", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "economic_replacements_approved": int(
            pd.to_numeric(
                equity.get("economic_replacements_approved", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "integer_optimizer_applied_days": int(
            equity.get("integer_optimizer_status", pd.Series(dtype=str))
            .astype(str)
            .eq("applied")
            .sum()
        ),
        "execution_model": str(
            getattr(args, "execution_model", "next_open_fixed_bps_legacy")
        ),
        "slippage_bps": float(args.slippage_bps),
        "auction_fill_probability": float(
            getattr(args, "auction_fill_probability", 0.90)
        ),
        "auction_gap_lookback_days": int(
            getattr(args, "auction_gap_lookback_days", 252)
        ),
        "auction_limit_buffer_bps": float(
            getattr(args, "auction_limit_buffer_bps", 2.0)
        ),
        "auction_impact_bps": float(
            getattr(args, "auction_impact_bps", 0.0)
        ),
        "auction_order_attempts": auction_attempts,
        "auction_executed_orders": auction_executed,
        "auction_unmarketable_orders": auction_unmarketable,
        "auction_fill_rate": (
            float(auction_executed / auction_attempts)
            if auction_attempts > 0
            else np.nan
        ),
        "replacement_guard_days": int(
            pd.to_numeric(
                equity.get("replacement_guard_applied", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        ),
        "deferred_replacement_buy_count": int(
            pd.to_numeric(
                equity.get("deferred_replacement_buy_count", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "deferred_replacement_sell_count": int(
            pd.to_numeric(
                equity.get("deferred_replacement_sell_count", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "deferred_replacement_buy_gross": float(
            pd.to_numeric(
                equity.get("deferred_replacement_buy_gross", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "deferred_replacement_sell_gross": float(
            pd.to_numeric(
                equity.get("deferred_replacement_sell_gross", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        ),
        "broker_commission_rate": float(getattr(args, "broker_commission_rate", 0.0)),
        "broker_minimum_commission": float(getattr(args, "broker_minimum_commission", 0.0)),
        "max_broker_commission_fraction": float(
            getattr(args, "max_broker_commission_fraction", 0.0)
        ),
        "commission_efficient_trade_floor": float(
            commission_efficient_trade_floor(
                getattr(args, "broker_minimum_commission", 0.0),
                getattr(args, "max_broker_commission_fraction", 0.0),
            )
        ),
        "max_participation_rate": float(args.max_participation_rate),
        "rebalance_band_weight": float(args.rebalance_band_weight),
        "min_trade_value": float(args.min_trade_value),
        "min_trade_weight": float(getattr(args, "min_trade_weight", 0.0)),
        "entry_exit_min_trade_value": (
            None
            if getattr(args, "entry_exit_min_trade_value", None) is None
            else float(args.entry_exit_min_trade_value)
        ),
        "entry_exit_min_trade_weight": float(getattr(args, "entry_exit_min_trade_weight", 0.0)),
        "risk_reduction_min_trade_weight": float(
            getattr(args, "risk_reduction_min_trade_weight", 0.01)
        ),
        "risk_increase_min_trade_weight": float(
            getattr(args, "risk_increase_min_trade_weight", 0.01)
        ),
        "risk_target_alignment": str(
            getattr(args, "risk_target_alignment", "banded")
        ),
        "risk_rebalance_schedule": str(
            getattr(args, "risk_rebalance_schedule", "daily")
        ),
        "risk_alignment_max_orders": int(
            getattr(args, "risk_alignment_max_orders", 0)
        ),
        "risk_alignment_initial_max_orders": int(
            getattr(args, "risk_alignment_initial_max_orders", 0)
        ),
        "enable_lot_aware_selection": bool(getattr(args, "enable_lot_aware_selection", False)),
        "lot_aware_min_holdings": int(getattr(args, "lot_aware_min_holdings", 5)),
        "lot_aware_max_stock_weight": float(getattr(args, "lot_aware_max_stock_weight", 0.25)),
        "lot_aware_max_industry_weight": float(getattr(args, "lot_aware_max_industry_weight", 0.50)),
        "cost_model": (
            "date-aware statutory A-share costs + per-order broker commission + "
            "causal opening-auction limit fills or configurable one-sided slippage + "
            "participation cap"
        ),
        "corporate_action_model": "RESSET total-return/capital-return inferred cash distributions and share factors",
    }


def write_outputs_v2(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    summary: Mapping[str, object],
    factor_weights: pd.DataFrame,
    factor_ic: pd.DataFrame,
    args,
) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"factor_rank_v2_{summary['start_date'].replace('-', '')}_{summary['end_date'].replace('-', '')}_{stamp}"
    paths = {
        "equity_curve": output_dir / f"{base_name}_equity_curve.csv",
        "trades": output_dir / f"{base_name}_trades.csv",
        "positions": output_dir / f"{base_name}_final_positions.csv",
        "factor_weights": output_dir / f"{base_name}_factor_weights.csv",
        "factor_ic": output_dir / f"{base_name}_factor_ic.csv",
        "summary": output_dir / f"{base_name}_summary.json",
        "workbook": output_dir / f"{base_name}.xlsx",
    }
    equity.to_csv(paths["equity_curve"], index=False, encoding="utf-8-sig")
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    positions.to_csv(paths["positions"], index=False, encoding="utf-8-sig")
    factor_weights.to_csv(paths["factor_weights"], index=False, encoding="utf-8-sig")
    factor_ic.to_csv(paths["factor_ic"], index=False, encoding="utf-8-sig")
    payload = dict(summary)
    payload["trading_costs"] = trading_cost_snapshot(
        summary.get("end_date"),
        broker_commission_rate=summary.get("broker_commission_rate", 0.0),
        broker_minimum_commission=summary.get("broker_minimum_commission", 0.0),
    )
    paths["summary"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_excel_workbook(
        paths["workbook"],
        [
            ("summary", pd.DataFrame([summary])),
            ("equity_curve", equity),
            ("trades", trades),
            ("final_positions", positions),
            ("factor_weights", factor_weights),
            ("factor_ic", factor_ic),
        ],
    )
    return paths


def sqlite_read_only_uri(path: Path) -> str:
    return f"file:{Path(path).resolve().as_posix()}?mode=ro"


def connect_market_database(args) -> sqlite3.Connection:
    before_database = getattr(args, "database_before_cutover", None)
    if not before_database:
        return sqlite3.connect(args.database)
    cutover_date = str(getattr(args, "continuous_cutover_date", "") or "")
    try:
        datetime.strptime(cutover_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "--continuous-cutover-date must use YYYY-MM-DD when a pre-cutover "
            "database is configured."
        ) from exc

    conn = sqlite3.connect(
        sqlite_read_only_uri(Path(args.database)),
        uri=True,
        timeout=60.0,
    )
    try:
        conn.execute(
            "ATTACH DATABASE ? AS before_cutover",
            (sqlite_read_only_uri(Path(before_database)),),
        )
        after_columns = [
            row[1] for row in conn.execute("PRAGMA main.table_info(stock_daily)")
        ]
        before_columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA before_cutover.table_info(stock_daily)"
            )
        ]
        if not after_columns or after_columns != before_columns:
            raise ValueError(
                "Pre/post-cutover stock_daily schemas do not match exactly."
            )
        columns_sql = ", ".join(
            f'"{column.replace(chr(34), chr(34) * 2)}"'
            for column in after_columns
        )
        conn.execute(
            f"""
            CREATE TEMP VIEW stock_daily AS
            SELECT {columns_sql}
            FROM before_cutover.stock_daily
            WHERE trade_date < '{cutover_date}'
            UNION ALL
            SELECT {columns_sql}
            FROM main.stock_daily
            WHERE trade_date >= '{cutover_date}'
            """
        )
        return conn
    except Exception:
        conn.close()
        raise


def run_backtest(args):
    conn = connect_market_database(args)
    previous_sigint = None
    feature_cache: Optional[FeatureSnapshotCache] = None
    risk_store: Optional[WeeklyRiskModelStore] = None
    alpha_feature_store: Optional[WeeklyAlphaFeatureStore] = None
    v31_alpha_store: Optional[V31AlphaFeatureStore] = None
    v31_factor_state_store: Optional[MonthlyFactorStateStore] = None
    v22_satellite_controller: Optional[IndustrySatelliteController] = None
    risk_calibration_store: Optional[CausalRiskCalibrationStore] = None
    opening_gap_estimator: Optional[CausalOpeningGapEstimator] = None
    try:
        dates = base.trading_dates(conn)
        date_to_index = {date: idx for idx, date in enumerate(dates)}
        test_dates = [date for date in dates if args.start_date <= date <= args.end_date and date_to_index[date] > 0]
        if not test_dates:
            raise ValueError("No test dates in requested range.")
        prehistory = max(int(args.feature_history_days), int(args.min_history_days) + 30, 320)
        history_start = dates[max(0, date_to_index[test_dates[0]] - prehistory)]
        prices = base.load_prices(conn, history_start, test_dates[-1])
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        financial = base.load_financial_factors(conn)
        industry_events = base.load_industry_event_scores(args.industry_event_scores)
        feature_cache_path = getattr(args, "feature_cache", None)
        before_feature_cache_path = getattr(
            args, "feature_cache_before_cutover", None
        )
        cache_fingerprint = None
        if before_feature_cache_path:
            if not feature_cache_path or not getattr(
                args, "database_before_cutover", None
            ):
                raise ValueError(
                    "Split feature caching requires --feature-cache, "
                    "--feature-cache-before-cutover and --database-before-cutover."
                )
            before_fingerprint = feature_cache_fingerprint(
                args,
                Path(args.database_before_cutover),
            )
            after_fingerprint = feature_cache_fingerprint(args)
            feature_cache = SplitFeatureSnapshotCache(
                Path(before_feature_cache_path),
                Path(feature_cache_path),
                str(args.continuous_cutover_date),
                before_fingerprint,
                after_fingerprint,
            )
            cache_fingerprint = feature_cache.combined_fingerprint
        elif feature_cache_path:
            feature_cache = FeatureSnapshotCache(Path(feature_cache_path))
            cache_fingerprint = feature_cache_fingerprint(args)

        if bool(getattr(args, "build_feature_cache_only", False)):
            if feature_cache is None or cache_fingerprint is None:
                raise ValueError("--build-feature-cache-only requires --feature-cache.")
            scheduled_dates = []
            for offset, trade_date in enumerate(test_dates):
                decision_date = dates[date_to_index[trade_date] - 1]
                if should_rebalance_on_date(
                    args.rebalance_schedule,
                    offset,
                    args.rebalance_every_n_days,
                    dates,
                    date_to_index,
                    decision_date,
                ):
                    scheduled_dates.append(decision_date)
            scheduled_dates = list(dict.fromkeys(scheduled_dates))
            print(
                f"Feature cache: {feature_cache.path} "
                f"({feature_cache.count(cache_fingerprint)} snapshots already present)",
                flush=True,
            )
            for index, decision_date in enumerate(scheduled_dates, start=1):
                cached_feature_snapshot(
                    feature_cache,
                    cache_fingerprint,
                    prices,
                    financial,
                    decision_date,
                    args,
                    industry_events,
                )
                if index % 5 == 0 or index == len(scheduled_dates):
                    print(
                        f"Feature-cache progress: {index}/{len(scheduled_dates)} {decision_date}",
                        flush=True,
                    )
            print(
                f"Feature cache complete: {len(scheduled_dates)} scheduled snapshots.",
                flush=True,
            )
            return {
                "feature_cache": str(feature_cache.path),
                "feature_cache_fingerprint": cache_fingerprint,
                "snapshot_count": len(scheduled_dates),
            }

        market_state = base.build_market_state(prices, args)
        all_prices_by_date = PriceDateStore(
            prices,
            max_cached_dates=int(getattr(args, "price_date_cache_days", 32)),
        )
        event_regime = base.load_event_regime_signals(args.event_regime_signals)
        if str(getattr(args, "risk_overlay_mode", "disabled")) != "disabled":
            if not getattr(args, "risk_model_database", None):
                raise ValueError(
                    "Risk overlay requires --risk-model-database."
                )
            risk_store = WeeklyRiskModelStore(args.risk_model_database)
            if getattr(args, "risk_calibration_schedule", None):
                risk_calibration_store = CausalRiskCalibrationStore(
                    args.risk_calibration_schedule
                )

        score_profile = str(
            getattr(args, "score_profile", "v2h4_legacy")
        ).strip().lower()
        v22_satellite_controller = IndustrySatelliteController(
            getattr(args, "v22_industry_satellite_schedule", "weekly")
        )
        if score_profile == "china_small_v3":
            if not getattr(args, "risk_model_database", None):
                raise ValueError(
                    "china_small_v3 requires --risk-model-database."
                )
            alpha_feature_store = WeeklyAlphaFeatureStore(
                args.risk_model_database,
                lookback_weeks=int(args.residual_momentum_lookback_weeks),
                skip_weeks=int(args.residual_momentum_skip_weeks),
            )
        elif uses_v31_alpha_features(args):
            if not getattr(args, "risk_model_database", None):
                raise ValueError(
                    "V3.1 alpha or soft industry budgets require --risk-model-database."
                )
            v31_alpha_store = V31AlphaFeatureStore(
                args.risk_model_database,
                lookback_weeks=int(args.residual_momentum_lookback_weeks),
                skip_weeks=int(args.residual_momentum_skip_weeks),
                before_cutover_database=getattr(
                    args, "risk_model_database_before_cutover", None
                ),
                cutover_date=getattr(args, "continuous_cutover_date", None),
            )
            if (
                score_profile == "v31"
                and str(args.v31_factor_state_mode).strip().lower() == "monthly"
            ):
                v31_factor_state_store = MonthlyFactorStateStore(
                    args.risk_model_database,
                    args,
                )

        components = list(configured_component_weights(args))
        if bool(args.dynamic_include_event_component):
            components.append(EVENT_COMPONENT)
        weighter = RollingICWeighter(args, components)

        checkpoint_path = checkpoint_path_from_args(args)
        fingerprint = checkpoint_fingerprint(args)
        start_offset = 0
        holdings: Dict[str, int] = {}
        cash = float(args.initial_cash)
        last_close: Dict[str, float] = {}
        previous_total = float(args.initial_cash)
        peak_total = float(args.initial_cash)
        equity_rows: List[Dict[str, object]] = []
        trade_rows: List[Dict[str, object]] = []

        if checkpoint_path and checkpoint_exists(checkpoint_path):
            if not bool(args.resume):
                raise FileExistsError(
                    f"Checkpoint already exists; rerun with --resume or choose another file: {checkpoint_path}"
                )
            state = load_checkpoint(checkpoint_path, fingerprint, len(test_dates))
            start_offset = int(state["next_offset"])
            holdings = {str(code): int(shares) for code, shares in state["holdings"].items()}
            cash = float(state["cash"])
            last_close = {
                str(code): float(close) for code, close in state["last_close"].items()
            }
            previous_total = float(state["previous_total"])
            peak_total = float(state["peak_total"])
            equity_rows = list(state["equity_rows"])
            trade_rows = list(state["trade_rows"])
            restore_weighter(weighter, state["weighter"])
            v22_satellite_controller.restore(
                state.get("v22_satellite_controller")
            )
            if len(equity_rows) != start_offset:
                raise ValueError(
                    "Checkpoint equity history length does not match its next offset: "
                    f"{len(equity_rows)} != {start_offset}"
                )
            print(
                f"Resuming checkpoint: {start_offset}/{len(test_dates)} "
                f"after {state.get('last_completed_date')} from {checkpoint_path}",
                flush=True,
            )
        elif checkpoint_path:
            print(f"No checkpoint found; starting a new resumable run: {checkpoint_path}", flush=True)

        if (
            str(getattr(args, "execution_model", "next_open_fixed_bps_legacy"))
            == "opening_auction_limit"
        ):
            opening_gap_estimator = CausalOpeningGapEstimator(
                lookback_days=int(
                    getattr(args, "auction_gap_lookback_days", 252)
                ),
                min_observations=int(
                    getattr(args, "auction_min_gap_observations", 60)
                ),
                fill_probability=float(
                    getattr(args, "auction_fill_probability", 0.90)
                ),
                shrinkage_observations=float(
                    getattr(args, "auction_shrinkage_observations", 40.0)
                ),
                market_lookback_days=int(
                    getattr(args, "auction_market_lookback_days", 60)
                ),
            )
            if start_offset < len(test_dates):
                first_trade_date = test_dates[start_offset]
                seed_date = dates[date_to_index[first_trade_date] - 1]
            else:
                seed_date = test_dates[-1]
            opening_gap_estimator.seed_from_frame(prices, seed_date)

        pause_requested = False

        def request_pause(_signum, _frame):
            nonlocal pause_requested
            if pause_requested:
                raise KeyboardInterrupt
            pause_requested = True
            print(
                "Pause requested. Finishing the current trading day and saving a checkpoint; "
                "press Ctrl+C again only to force an immediate stop.",
                flush=True,
            )

        if checkpoint_path is not None:
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, request_pause)
        checkpoint_interval = max(1, int(args.checkpoint_every_n_days))

        for offset, trade_date in enumerate(test_dates[start_offset:], start=start_offset):
            decision_date = dates[date_to_index[trade_date] - 1]
            today_prices = all_prices_by_date.get(trade_date, {})
            corporate_action_cash, corporate_actions = base.apply_corporate_actions_before_open(
                holdings, today_prices, last_close
            )
            open_total = base.value_portfolio(holdings, cash, today_prices, "open", last_close)
            current_weights = current_weights_from_open(holdings, today_prices, open_total, last_close)
            economic_hurdle_enabled = bool(
                getattr(args, "enable_economic_replacement_hurdle", False)
            )
            if bool(args.dynamic_factor_weights) or economic_hurdle_enabled:
                weighter.resolve(decision_date, all_prices_by_date)

            regime = continuous_target_equity(
                market_state, decision_date, previous_total, peak_total, args, event_regime
            )
            scheduled = should_rebalance_on_date(
                args.rebalance_schedule, offset, args.rebalance_every_n_days, dates, date_to_index, decision_date
            )
            current_equity = sum(current_weights.values())
            risk_rebalance = should_run_unscheduled_risk_rebalance(
                args,
                current_equity,
                float(regime["target_equity_weight"]),
            )
            rebalanced = bool(scheduled or risk_rebalance)
            executed: List[Dict[str, object]] = []
            execution_diagnostics = {
                "auction_order_attempts": 0,
                "auction_marketable_orders": 0,
                "auction_executed_orders": 0,
                "auction_unmarketable_orders": 0,
                "auction_marketable_not_executed": 0,
                "auction_blocked_by_price_limit": 0,
                "replacement_guard_applied": False,
                "desired_net_buy_gross": 0.0,
                "executable_buy_gross_before_guard": 0.0,
                "executable_sell_gross_before_guard": 0.0,
                "executable_buy_gross_after_guard": 0.0,
                "executable_sell_gross_after_guard": 0.0,
                "deferred_replacement_buy_count": 0,
                "deferred_replacement_sell_count": 0,
                "deferred_replacement_buy_gross": 0.0,
                "deferred_replacement_sell_gross": 0.0,
            }
            target_meta: Dict[str, object] = {"selected_count": 0, "target_weight_sum": 0.0}
            weight_info: Dict[str, object] = {"weight_mode": "not_rebalanced", "ic_observations": len(weighter.ic_rows)}

            if rebalanced:
                features = cached_feature_snapshot(
                    feature_cache,
                    cache_fingerprint,
                    prices,
                    financial,
                    decision_date,
                    args,
                    industry_events,
                )
                alpha_feature_meta: Dict[str, object] = {}
                if alpha_feature_store is not None:
                    features, alpha_feature_meta = alpha_feature_store.augment(
                        features,
                        decision_date,
                        maximum_staleness_days=int(
                            args.risk_model_max_staleness_days
                        ),
                    )
                if v31_alpha_store is not None:
                    features, v31_alpha_meta = v31_alpha_store.augment(
                        features,
                        decision_date,
                        maximum_staleness_days=int(
                            args.risk_model_max_staleness_days
                        ),
                    )
                    alpha_feature_meta.update(v31_alpha_meta)
                factor_weights, weight_info = weighter.weights(decision_date)
                if score_profile == "v31":
                    factor_weights = configured_component_weights(args)
                    if v31_factor_state_store is not None:
                        factor_weights, state_meta = v31_factor_state_store.weights(
                            decision_date,
                            factor_weights,
                        )
                    else:
                        state_meta = {
                            "factor_state_mode": "static",
                            "offensive_weight": float(
                                sum(
                                    factor_weights.get(name, 0.0)
                                    for name in (
                                        "industry_trend_score",
                                        "earnings_yield_score",
                                        "quality_score_v31",
                                        "growth_score_v31",
                                        "residual_momentum_score",
                                    )
                                )
                            ),
                        }
                    weight_info.update(state_meta)
                features = apply_v2_score(
                    features,
                    factor_weights,
                    args,
                    decision_date=decision_date,
                    satellite_controller=v22_satellite_controller,
                    market_risk_on_strength=v22_market_risk_on_strength(
                        regime, args
                    ),
                )
                expected_return_per_score = 0.0
                economic_meta: Dict[str, object] = {}
                if economic_hurdle_enabled:
                    expected_return_per_score, economic_meta = (
                        weighter.expected_score_return(decision_date)
                    )
                target_transform = None
                if risk_store is not None:
                    def target_transform(
                        desired,
                        _frame,
                        effective_stock_cap,
                        _effective_industry_cap,
                    ):
                        calibration_multiplier = float(
                            args.risk_calibration_multiplier
                        )
                        calibration_metadata = {
                            "risk_calibration_source": "fixed",
                            "risk_calibration_as_of_date": None,
                            "risk_calibration_forecast_weeks": 0,
                        }
                        if risk_calibration_store is not None:
                            (
                                calibration_multiplier,
                                calibration_metadata,
                            ) = risk_calibration_store.multiplier_for(
                                decision_date,
                                default=calibration_multiplier,
                            )
                        weights, metadata = apply_store_overlay(
                            risk_store,
                            decision_date,
                            desired,
                            stock_cap=effective_stock_cap,
                            maximum_industry_fraction=float(
                                args.risk_model_max_industry_weight
                            ),
                            strength=float(args.risk_overlay_strength),
                            target_volatility=float(
                                args.target_portfolio_volatility
                            ),
                            calibration_multiplier=calibration_multiplier,
                            minimum_equity_scale=float(
                                args.risk_overlay_min_equity_scale
                            ),
                            maximum_staleness_days=int(
                                args.risk_model_max_staleness_days
                            ),
                            iterations=int(args.risk_overlay_iterations),
                        )
                        metadata.update(calibration_metadata)
                        return weights, metadata
                targets, target_meta = build_targets_v2(
                    features,
                    holdings,
                    current_weights,
                    args,
                    float(regime["target_equity_weight"]),
                    previous_total,
                    force_risk_alignment=should_force_risk_alignment(
                        args,
                        current_equity,
                        float(regime["target_equity_weight"]),
                    ),
                    decision_date=decision_date,
                    expected_return_per_score=expected_return_per_score,
                    target_transform=target_transform,
                )
                target_meta.update(alpha_feature_meta)
                target_meta.update(
                    {
                        key: value
                        for key, value in weight_info.items()
                        if key.startswith("factor_state")
                        or key in {
                            "offensive_weight",
                            "defensive_weight",
                            "factor_return_signal",
                            "industry_trend_signal",
                        }
                    }
                )
                target_meta.update(economic_meta)
                liquidity_by_code = features.set_index("code")["avg_amount_60"].to_dict() if not features.empty else {}
                cash, executed, execution_diagnostics = execute_trades_v2(
                    trade_date,
                    decision_date,
                    holdings,
                    cash,
                    targets,
                    today_prices,
                    open_total,
                    liquidity_by_code,
                    args,
                    last_close,
                    previous_total,
                    opening_gap_estimator,
                    industries=(
                        features.set_index("code")["industry_1"].to_dict()
                        if not features.empty and "industry_1" in features
                        else None
                    ),
                    industry_caps=target_meta.get("industry_budget_caps"),
                )
                trade_rows.extend(executed)

                # The label becomes eligible only after its own exit date has passed.
                decision_index = date_to_index[decision_date]
                entry_index = decision_index + 1
                exit_index = decision_index + int(args.forward_label_days)
                if (
                    (bool(args.dynamic_factor_weights) or economic_hurdle_enabled)
                    and entry_index < len(dates)
                    and exit_index < len(dates)
                ):
                    weighter.add_snapshot(decision_date, dates[entry_index], dates[exit_index], features)

            cash += corporate_action_cash
            close_total = base.value_portfolio(holdings, cash, today_prices, "close", last_close)
            for code, row in today_prices.items():
                close = safe_float(row.get("close"), np.nan)
                if math.isfinite(close) and close > 0:
                    last_close[code] = close
            if opening_gap_estimator is not None:
                opening_gap_estimator.update(today_prices)
            gross_traded = float(sum(float(row["gross_amount"]) for row in executed))
            fees = float(sum(float(row["fee"]) for row in executed))
            actual_equity = (close_total - cash) / close_total if close_total > 0 else 0.0
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "decision_date": decision_date,
                    "total_value": close_total,
                    "cash": cash,
                    "stock_value": close_total - cash,
                    "daily_return": close_total / previous_total - 1.0 if previous_total > 0 else np.nan,
                    "turnover": gross_traded / open_total if open_total > 0 else 0.0,
                    "gross_traded": gross_traded,
                    "fees": fees,
                    "corporate_action_cash": corporate_action_cash,
                    "corporate_action_count": int(len(corporate_actions)),
                    "corporate_action_share_changes": int(
                        sum(1 for item in corporate_actions if item["new_shares"] != item["old_shares"])
                    ),
                    "holding_count": int(sum(1 for shares in holdings.values() if int(shares) > 0)),
                    "rebalanced": rebalanced,
                    "scheduled_rebalance": scheduled,
                    "risk_rebalance": risk_rebalance,
                    "target_equity_weight": float(regime["target_equity_weight"]),
                    "actual_equity_weight": actual_equity,
                    "market_state": regime["market_state"],
                    "market_index": regime["market_index"],
                    "market_breadth": regime["market_breadth"],
                    "market_return_20": regime["market_return_20"],
                    "market_volatility_20": regime["market_volatility_20"],
                    "trend_signal": regime.get("trend_signal", np.nan),
                    "vol_scale": regime.get("vol_scale", np.nan),
                    "portfolio_drawdown_signal": regime["portfolio_drawdown"],
                    "event_equity_multiplier": regime["event_equity_multiplier"],
                    "event_equity_multiplier_used": regime.get("event_equity_multiplier_used", 1.0),
                    "event_risk_score": regime["event_risk_score"],
                    "event_regime_confidence": regime["event_regime_confidence"],
                    "selected_count": int(target_meta.get("selected_count", 0)),
                    "target_weight_sum": float(target_meta.get("target_weight_sum", 0.0)),
                    "factor_weight_mode": str(weight_info.get("weight_mode", "")),
                    "factor_ic_observations": int(weight_info.get("ic_observations", 0)),
                    "economic_replacements_blocked": int(
                        target_meta.get("economic_replacements_blocked", 0)
                    ),
                    "economic_replacements_approved": int(
                        target_meta.get("economic_replacements_approved", 0)
                    ),
                    "economic_replacement_policy": str(
                        target_meta.get(
                            "economic_replacement_policy",
                            resolved_economic_replacement_policy(args),
                        )
                    ),
                    "economic_score_slope_source": target_meta.get(
                        "economic_score_slope_source"
                    ),
                    "economic_score_slope_observations": int(
                        target_meta.get("economic_score_slope_observations", 0)
                    ),
                    "alpha_feature_status": target_meta.get(
                        "alpha_feature_status"
                    ),
                    "alpha_feature_exposure_date": target_meta.get(
                        "alpha_feature_exposure_date"
                    ),
                    "alpha_feature_momentum_date": target_meta.get(
                        "alpha_feature_momentum_date"
                    ),
                    "alpha_feature_earnings_coverage": target_meta.get(
                        "alpha_feature_earnings_coverage", np.nan
                    ),
                    "alpha_feature_residual_momentum_coverage": target_meta.get(
                        "alpha_feature_residual_momentum_coverage", np.nan
                    ),
                    "v31_alpha_status": target_meta.get("v31_alpha_status"),
                    "v31_alpha_date": target_meta.get("v31_alpha_date"),
                    "v31_residual_momentum_date": target_meta.get(
                        "v31_residual_momentum_date"
                    ),
                    "v31_unknown_target_weight": target_meta.get(
                        "unknown_target_weight", np.nan
                    ),
                    "v31_unknown_industry_cap_enforced": target_meta.get(
                        "v31_unknown_industry_cap_enforced",
                        bool(
                            getattr(
                                args,
                                "v31_enforce_unknown_industry_cap",
                                False,
                            )
                        ),
                    ),
                    "v31_factor_state_mode": target_meta.get(
                        "factor_state_mode"
                    ),
                    "v31_offensive_weight": target_meta.get(
                        "offensive_weight", np.nan
                    ),
                    "v31_factor_state_signal": target_meta.get(
                        "factor_state_signal", np.nan
                    ),
                    "v31_industry_budget_mode": target_meta.get(
                        "v31_industry_budget_mode",
                        resolved_v31_industry_budget_mode(args),
                    ),
                    "v31_alpha_tilt_weight": target_meta.get(
                        "v31_alpha_tilt_weight",
                        float(getattr(args, "v31_alpha_tilt_weight", 0.0)),
                    ),
                    "v22_satellite_weight": target_meta.get(
                        "v22_satellite_weight", np.nan
                    ),
                    "v22_leadership_strength": target_meta.get(
                        "v22_leadership_strength", np.nan
                    ),
                    "v22_leading_industries": target_meta.get(
                        "v22_leading_industries", ""
                    ),
                    "v22_market_risk_on_strength": target_meta.get(
                        "v22_market_risk_on_strength", np.nan
                    ),
                    "v22_satellite_schedule": target_meta.get(
                        "v22_satellite_schedule",
                        str(
                            getattr(
                                args,
                                "v22_industry_satellite_schedule",
                                "weekly",
                            )
                        ),
                    ),
                    "v22_satellite_signal_period": target_meta.get(
                        "v22_satellite_signal_period"
                    ),
                    "risk_overlay_status": str(
                        target_meta.get("risk_overlay_status", "not_applied")
                    ),
                    "risk_model_date": target_meta.get("risk_model_date"),
                    "risk_model_exact_weight_coverage": target_meta.get(
                        "risk_model_exact_weight_coverage", np.nan
                    ),
                    "risk_calibration_multiplier_used": target_meta.get(
                        "risk_calibration_multiplier", np.nan
                    ),
                    "risk_calibration_source": target_meta.get(
                        "risk_calibration_source"
                    ),
                    "risk_calibration_as_of_date": target_meta.get(
                        "risk_calibration_as_of_date"
                    ),
                    "risk_calibration_forecast_weeks": target_meta.get(
                        "risk_calibration_forecast_weeks", 0
                    ),
                    "risk_predicted_volatility_before": target_meta.get(
                        "risk_predicted_volatility_before", np.nan
                    ),
                    "risk_predicted_volatility_optimized": target_meta.get(
                        "risk_predicted_volatility_optimized", np.nan
                    ),
                    "risk_predicted_volatility_after": target_meta.get(
                        "risk_predicted_volatility_after", np.nan
                    ),
                    "risk_target_portfolio_volatility": target_meta.get(
                        "risk_target_portfolio_volatility", np.nan
                    ),
                    "risk_volatility_scale": target_meta.get(
                        "risk_volatility_scale", np.nan
                    ),
                    "risk_cap_met": target_meta.get("risk_cap_met"),
                    **execution_diagnostics,
                }
            )
            previous_total = close_total
            peak_total = max(peak_total, close_total)
            should_checkpoint = checkpoint_path is not None and (
                (offset + 1) % checkpoint_interval == 0
                or pause_requested
                or offset == len(test_dates) - 1
            )
            if should_checkpoint:
                state = make_checkpoint_state(
                    fingerprint,
                    offset + 1,
                    test_dates,
                    holdings,
                    cash,
                    last_close,
                    previous_total,
                    peak_total,
                    equity_rows,
                    trade_rows,
                    weighter,
                    v22_satellite_controller,
                )
                save_checkpoint(checkpoint_path, state, args)
            if pause_requested:
                raise BacktestPaused(
                    f"Backtest paused safely after {trade_date}. Resume from {checkpoint_path}"
                )
            if (offset + 1) % 20 == 0 or offset == len(test_dates) - 1:
                print(
                    f"V2 progress: {offset + 1}/{len(test_dates)} {trade_date} "
                    f"value={close_total:.2f} actual_equity={actual_equity:.1%}",
                    flush=True,
                )

        equity = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(trade_rows)
        if trades.empty:
            trades = pd.DataFrame(columns=["trade_date", "decision_date", "code", "name", "side", "shares", "open_price", "price", "gross_amount", "fee", "cash_after"])
        final_prices = all_prices_by_date.get(test_dates[-1], {})
        final_total = base.value_portfolio(holdings, cash, final_prices, "close", last_close)
        positions = pd.DataFrame(
            [
                {
                    "code": code,
                    "shares": int(shares),
                    "close": safe_float(final_prices.get(code, {}).get("close"), last_close.get(code, np.nan)),
                    "market_value": int(shares) * safe_float(final_prices.get(code, {}).get("close"), last_close.get(code, np.nan)),
                }
                for code, shares in holdings.items()
                if int(shares) > 0
            ]
        )
        summary = make_summary(equity, trades, float(args.initial_cash), final_total, args)
        paths = write_outputs_v2(
            equity,
            trades,
            positions,
            summary,
            pd.DataFrame(weighter.weight_rows),
            pd.DataFrame(weighter.ic_rows),
            args,
        )
        clear_checkpoint(checkpoint_path)
        print(f"V2 workbook: {paths['workbook']}")
        print(f"Summary JSON: {paths['summary']}")
        print(f"Final value: {summary['final_value']:.2f}")
        print(f"Total return: {summary['total_return']:.2%}")
        print(f"Average actual equity: {summary['average_actual_equity_weight']:.2%}")
        return summary, paths
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if feature_cache is not None:
            feature_cache.close()
        if risk_store is not None:
            risk_store.close()
        if alpha_feature_store is not None:
            alpha_feature_store.close()
        if v31_alpha_store is not None:
            v31_alpha_store.close()
        conn.close()


def load_strategy_config(path: Path, seen=None) -> Dict[str, object]:
    path = Path(path).resolve()
    seen = set(seen or ())
    if path in seen:
        raise ValueError(f"Circular strategy-config inheritance: {path}")
    seen.add(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy config must contain a JSON object: {path}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    merged = load_strategy_config(parent_path, seen)
    merged.update(payload)
    return merged


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="V2 causal weekly A-share factor backtest.")
    parser.add_argument("--strategy-config", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--database-before-cutover", type=Path)
    parser.add_argument("--continuous-cutover-date")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-every-n-days", type=int, default=1)
    parser.add_argument("--rebalance-schedule", choices=["every_n_days", "week_end", "month_end"], default="week_end")

    # Universe / portfolio.
    parser.add_argument(
        "--score-profile",
        choices=["v2h4_legacy", "china_small_v3", "v31"],
        default="v2h4_legacy",
    )
    parser.add_argument(
        "--portfolio-constructor",
        choices=["legacy", "integer_cost_aware"],
        default="legacy",
    )
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--min-target-count", type=int, default=30)
    parser.add_argument("--buy-rank", type=int, default=90)
    parser.add_argument("--sell-rank", type=int, default=180)
    parser.add_argument("--cash-weight", type=float, default=0.05)
    parser.add_argument("--max-stock-weight", type=float, default=0.035)
    parser.add_argument("--max-industry-weight", type=float, default=0.20)
    parser.add_argument("--score-temperature", type=float, default=0.80)
    parser.add_argument("--inverse-vol-power", type=float, default=1.0)
    parser.add_argument("--min-stock-volatility", type=float, default=0.08)
    parser.add_argument("--rebalance-band-weight", type=float, default=0.0025)
    parser.add_argument("--min-trade-value", type=float, default=20_000.0)
    parser.add_argument("--min-trade-weight", type=float, default=0.0)
    parser.add_argument("--entry-exit-min-trade-value", type=float)
    parser.add_argument("--entry-exit-min-trade-weight", type=float, default=0.0)
    parser.add_argument("--risk-reduction-min-trade-weight", type=float, default=0.01)
    parser.add_argument("--risk-increase-min-trade-weight", type=float, default=0.01)
    parser.add_argument(
        "--risk-target-alignment",
        choices=["banded", "strict"],
        default="banded",
    )
    parser.add_argument(
        "--risk-rebalance-schedule",
        choices=["daily", "scheduled_only"],
        default="daily",
    )
    parser.add_argument("--risk-alignment-max-orders", type=int, default=0)
    parser.add_argument("--risk-alignment-initial-max-orders", type=int, default=0)
    parser.add_argument("--enable-lot-aware-selection", action="store_true")
    parser.add_argument("--lot-aware-min-holdings", type=int, default=5)
    parser.add_argument("--lot-aware-stock-cap-multiplier", type=float, default=1.25)
    parser.add_argument("--lot-aware-max-stock-weight", type=float, default=0.25)
    parser.add_argument("--lot-aware-max-industry-weight", type=float, default=0.50)
    parser.add_argument("--integer-optimizer-min-holdings", type=int, default=1)
    parser.add_argument("--integer-tracking-penalty", type=float, default=1.0)
    parser.add_argument("--integer-cash-penalty", type=float, default=0.75)
    parser.add_argument("--integer-transaction-cost-penalty", type=float, default=2.0)
    parser.add_argument("--integer-optimizer-iterations", type=int, default=12)
    parser.add_argument("--enforce-commission-efficient-floor", action="store_true")
    parser.add_argument("--max-broker-commission-fraction", type=float, default=0.002)
    parser.add_argument("--enable-economic-replacement-hurdle", action="store_true")
    parser.add_argument(
        "--economic-replacement-policy",
        choices=["auto", "none", "always", "cost_aware"],
        default="auto",
    )
    parser.add_argument("--economic-hurdle-buffer-bps", type=float, default=10.0)
    parser.add_argument("--economic-hurdle-lookback-weeks", type=int, default=52)
    parser.add_argument("--economic-hurdle-min-observations", type=int, default=16)
    parser.add_argument("--economic-hurdle-fallback-return-per-score", type=float, default=0.005)
    parser.add_argument("--economic-hurdle-max-return-per-score", type=float, default=0.03)

    # Data / base features inherited from V1.
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--feature-history-days", type=int, default=320)
    parser.add_argument("--min-avg-amount", type=float, default=50_000_000.0)
    parser.add_argument("--min-market-cap-quantile", type=float, default=0.30)
    parser.add_argument("--market-cap-proxy-window", type=int, default=20)
    parser.add_argument("--disable-industry-neutral-factors", action="store_true")
    parser.add_argument("--residual-momentum-lookback-weeks", type=int, default=52)
    parser.add_argument("--residual-momentum-skip-weeks", type=int, default=4)

    # V3.1 industry-balanced alpha and monthly factor state.
    parser.add_argument("--v31-component-weights", type=json.loads)
    parser.add_argument(
        "--v31-industry-budget-mode",
        choices=["auto", "fixed", "soft"],
        default="auto",
    )
    parser.add_argument("--v31-alpha-tilt-weight", type=float, default=0.0)
    parser.add_argument("--v31-industry-cap-deviation", type=float, default=0.04)
    parser.add_argument("--v31-absolute-industry-cap", type=float, default=0.25)
    parser.add_argument("--v31-unknown-industry-cap", type=float, default=0.05)
    parser.add_argument(
        "--v31-enforce-unknown-industry-cap",
        action="store_true",
        help=(
            "Apply the UNKNOWN industry cap even when the remaining industry "
            "budgets use the fixed legacy mode."
        ),
    )
    parser.add_argument(
        "--v31-factor-state-mode",
        choices=["static", "monthly"],
        default="static",
    )
    parser.add_argument("--v31-offensive-base-weight", type=float, default=0.60)
    parser.add_argument("--v31-offensive-min-weight", type=float, default=0.45)
    parser.add_argument("--v31-offensive-max-weight", type=float, default=0.75)
    parser.add_argument("--v31-factor-state-lookback-weeks", type=int, default=52)
    parser.add_argument("--v31-factor-state-min-weeks", type=int, default=26)
    parser.add_argument("--v31-factor-state-return-weight", type=float, default=0.65)
    parser.add_argument("--v31-factor-state-max-tilt", type=float, default=0.12)
    parser.add_argument(
        "--v31-factor-state-max-monthly-step", type=float, default=0.04
    )

    # V2.2 structural industry-rotation experiments. All are opt-in.
    parser.add_argument("--v22-defensive-industry-neutral", action="store_true")
    parser.add_argument("--v22-pure-industry-trend", action="store_true")
    parser.add_argument(
        "--v22-industry-satellite-max-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--v22-industry-satellite-top-industries", type=int, default=2
    )
    parser.add_argument(
        "--v22-industry-satellite-excluded-industries",
        type=json.loads,
        default=[],
        help="JSON list of CSRC top-level industry codes excluded from the satellite.",
    )
    parser.add_argument(
        "--v22-industry-satellite-schedule",
        choices=["weekly", "monthly"],
        default="weekly",
    )
    parser.add_argument(
        "--v22-industry-satellite-risk-throttle",
        choices=["none", "continuous_equity"],
        default="none",
    )
    parser.add_argument(
        "--v22-industry-satellite-application",
        choices=[
            "score_and_weight",
            "entry_only",
            "entry_and_weight",
            "allocation_only",
        ],
        default="score_and_weight",
        help=(
            "Control whether industry leadership changes rankings, target weights, "
            "or only new-entry rankings while incumbent retention follows the core score."
        ),
    )

    # Risk budget.
    parser.add_argument("--risk-mode", choices=["continuous", "legacy", "disabled"], default="continuous")
    parser.add_argument("--min-equity-weight", type=float, default=0.65)
    parser.add_argument("--hard-crash-equity-weight", type=float, default=0.35)
    parser.add_argument("--hard-crash-return-20", type=float, default=-0.12)
    parser.add_argument("--hard-crash-breadth", type=float, default=0.28)
    parser.add_argument("--target-market-volatility", type=float, default=0.22)
    parser.add_argument("--min-vol-scale", type=float, default=0.78)
    parser.add_argument("--max-vol-scale", type=float, default=1.03)
    parser.add_argument("--trend-tilt", type=float, default=0.12)
    parser.add_argument("--trend-band", type=float, default=0.04)
    parser.add_argument("--breadth-band", type=float, default=0.15)
    parser.add_argument("--drawdown-reduce-threshold", type=float, default=-0.15)
    parser.add_argument("--drawdown-reduce-multiplier", type=float, default=0.88)
    parser.add_argument("--severe-drawdown-threshold", type=float, default=-0.25)
    parser.add_argument("--severe-drawdown-equity-weight", type=float, default=0.40)
    parser.add_argument("--risk-rebalance-band", type=float, default=0.08)
    parser.add_argument("--risk-off-equity-cap", type=float, default=0.95)
    parser.add_argument("--soft-crash-return-20", type=float, default=-0.06)
    parser.add_argument("--soft-crash-equity-cap", type=float, default=0.60)

    # Optional point-in-time factor risk overlay.
    parser.add_argument(
        "--risk-overlay-mode",
        choices=["disabled", "variance_blend"],
        default="disabled",
    )
    parser.add_argument("--risk-model-database", type=Path)
    parser.add_argument("--risk-model-database-before-cutover", type=Path)
    parser.add_argument(
        "--risk-calibration-multiplier", type=float, default=1.0
    )
    parser.add_argument("--risk-calibration-schedule", type=Path)
    parser.add_argument("--risk-overlay-strength", type=float, default=0.25)
    parser.add_argument(
        "--target-portfolio-volatility", type=float, default=0.20
    )
    parser.add_argument(
        "--risk-overlay-min-equity-scale", type=float, default=0.80
    )
    parser.add_argument(
        "--risk-model-max-industry-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--risk-model-max-staleness-days", type=int, default=14
    )
    parser.add_argument("--risk-overlay-iterations", type=int, default=80)

    # Base market-state arguments retained for --risk-mode legacy only.
    parser.add_argument("--disable-market-regime", action="store_true")
    parser.add_argument("--market-short-window", type=int, default=60)
    parser.add_argument("--market-long-window", type=int, default=200)
    parser.add_argument("--market-breadth-window", type=int, default=120)
    parser.add_argument("--bull-breadth-min", type=float, default=0.52)
    parser.add_argument("--neutral-breadth-min", type=float, default=0.45)
    parser.add_argument("--bear-breadth-max", type=float, default=0.38)
    parser.add_argument("--neutral-equity-weight", type=float, default=0.60)
    parser.add_argument("--defensive-equity-weight", type=float, default=0.35)
    parser.add_argument("--bear-equity-weight", type=float, default=0.20)
    parser.add_argument("--crash-equity-weight", type=float, default=0.05)
    parser.add_argument("--crash-return-20", type=float, default=-0.08)

    # Optional, gated event signals.
    parser.add_argument("--industry-event-scores", type=Path, default=base.DEFAULT_INDUSTRY_EVENT_SCORES)
    parser.add_argument("--event-regime-signals", type=Path, default=base.DEFAULT_EVENT_REGIME_SIGNALS)
    parser.add_argument("--enable-event-score", action="store_true")
    parser.add_argument("--event-score-weight", type=float, default=0.02)
    parser.add_argument("--event-score-weight-cap", type=float, default=0.04)
    parser.add_argument("--dynamic-include-event-component", action="store_true")
    parser.add_argument("--enable-event-regime", action="store_true")
    parser.add_argument("--event-regime-min-confidence", type=float, default=0.70)
    parser.add_argument("--event-regime-blend", type=float, default=0.50)

    # Causal factor-weight learning.
    parser.add_argument("--dynamic-factor-weights", action="store_true")
    parser.add_argument("--forward-label-days", type=int, default=20)
    parser.add_argument("--dynamic-ic-lookback-weeks", type=int, default=78)
    parser.add_argument("--dynamic-min-observations", type=int, default=16)
    parser.add_argument("--dynamic-min-cross-section", type=int, default=40)
    parser.add_argument("--dynamic-ic-std-floor", type=float, default=0.02)
    parser.add_argument("--dynamic-weight-strength", type=float, default=0.70)
    parser.add_argument("--dynamic-max-component-weight", type=float, default=0.35)

    # Execution realism.
    parser.add_argument(
        "--execution-model",
        choices=["next_open_fixed_bps_legacy", "opening_auction_limit"],
        default="next_open_fixed_bps_legacy",
    )
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--auction-fill-probability", type=float, default=0.90)
    parser.add_argument("--auction-gap-lookback-days", type=int, default=252)
    parser.add_argument("--auction-min-gap-observations", type=int, default=60)
    parser.add_argument("--auction-shrinkage-observations", type=float, default=40.0)
    parser.add_argument("--auction-market-lookback-days", type=int, default=60)
    parser.add_argument("--auction-limit-buffer-bps", type=float, default=2.0)
    parser.add_argument("--auction-impact-bps", type=float, default=0.0)
    parser.add_argument("--broker-commission-rate", type=float, default=0.0003)
    parser.add_argument("--broker-minimum-commission", type=float, default=5.0)
    parser.add_argument("--max-participation-rate", type=float, default=0.05)
    parser.add_argument("--disable-limit-trade-filter", action="store_true")
    parser.add_argument("--limit-trade-buffer", type=float, default=0.005)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=5)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--feature-cache-before-cutover", type=Path)
    parser.add_argument("--build-feature-cache-only", action="store_true")
    parser.add_argument("--price-date-cache-days", type=int, default=32)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching checkpoint, or start a new resumable run when none exists.",
    )
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.strategy_config:
        config_path = Path(preliminary.strategy_config)
        payload = load_strategy_config(config_path)
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(payload) - valid_destinations - {"strategy_name"})
        if unknown:
            raise ValueError(f"Unknown V2H strategy config keys: {', '.join(unknown)}")
        parser.set_defaults(**payload)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run_backtest(parse_args())
    except BacktestPaused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
