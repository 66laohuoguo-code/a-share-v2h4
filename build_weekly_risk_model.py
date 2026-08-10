"""Build weekly A-share risk exposures and covariance estimates."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from ashare_risk_model import (
    STYLE_FACTORS,
    build_cross_section_exposures,
    compute_stock_weekly_features,
    estimate_specific_variances,
    ewma_newey_west_covariance,
    fit_factor_returns,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MARKET_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_stock_daily_live.sqlite"
)
DEFAULT_RISK_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"
)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "risk_model_v1.json"
RAW_ALGORITHM_VERSION = "raw-weekly-style-v1.0"
MODEL_ALGORITHM_VERSION = "pit-factor-risk-v1.0"

RAW_COLUMNS = (
    "model_date",
    "code",
    "observation_date",
    "weekly_return",
    "total_market_cap",
    "float_market_cap",
    "beta_raw",
    "momentum_raw",
    "residual_volatility_raw",
    "liquidity_raw",
    "avg_amount_60",
    "history_days",
    "listed_state",
)
EXPOSURE_COLUMNS = (
    "model_date",
    "code",
    "observation_date",
    "financial_report_period",
    "financial_available_date",
    "industry_implement_date",
    "industry_group",
    "industry_code",
    "total_market_cap",
    "float_market_cap",
    "avg_amount_60",
    *STYLE_FACTORS,
)

_WORKER_CONNECTION = None
_WORKER_WEEKLY_DATES = None
_WORKER_CONFIG = None


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sqlite_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_hash(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Risk-model config must be a JSON object")
    return config


def stage_config_hashes(config, include_model_end_date=False):
    """Return checkpoint hashes, excluding the rolling live horizon by default."""
    raw_payload = {
        "algorithm_version": RAW_ALGORITHM_VERSION,
        "history_start_date": config["history_start_date"],
        "model_start_date": config["model_start_date"],
        "beta_window_days": config.get("beta_window_days"),
        "beta_minimum_days": config.get("beta_minimum_days"),
        "beta_half_life_days": config.get("beta_half_life_days"),
        "momentum_lookback_days": config.get("momentum_lookback_days"),
        "momentum_skip_days": config.get("momentum_skip_days"),
        "liquidity_windows_days": config.get("liquidity_windows_days"),
        "market_return_types": config.get("market_return_types"),
        "risk_free_benchmark": config.get("risk_free_benchmark"),
    }
    if include_model_end_date:
        raw_payload["model_end_date"] = config["model_end_date"]
    raw_hash = canonical_hash(raw_payload)

    model_config = dict(config)
    if not include_model_end_date:
        model_config.pop("model_end_date", None)
    model_hash = canonical_hash(
        {
            "algorithm_version": MODEL_ALGORITHM_VERSION,
            "raw_hash": raw_hash,
            "config": model_config,
        }
    )
    return raw_hash, model_hash


def migrate_legacy_horizon_hashes(conn, config, raw_hash, model_hash):
    """Migrate old checkpoints when only the live model end date was extended."""
    row = conn.execute(
        "SELECT value FROM risk_model_metadata WHERE key='config'"
    ).fetchone()
    if row is None:
        return False
    try:
        previous = json.loads(row[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(previous, dict):
        return False

    previous_end = str(previous.get("model_end_date", ""))
    current_end = str(config.get("model_end_date", ""))
    previous_comparable = dict(previous)
    current_comparable = dict(config)
    previous_comparable.pop("model_end_date", None)
    current_comparable.pop("model_end_date", None)
    if previous_comparable != current_comparable or current_end < previous_end:
        return False

    legacy_raw_hash, legacy_model_hash = stage_config_hashes(
        previous, include_model_end_date=True
    )
    migration = {
        "raw": (legacy_raw_hash, raw_hash),
        "model": (legacy_model_hash, model_hash),
    }
    updates = []
    for stage, (legacy_hash, stable_hash) in migration.items():
        hashes = {
            str(item[0])
            for item in conn.execute(
                "SELECT DISTINCT config_hash FROM risk_model_build_state WHERE stage=?",
                (stage,),
            )
        }
        if not hashes or hashes == {stable_hash}:
            continue
        if hashes != {legacy_hash}:
            return False
        updates.append((stable_hash, stage, legacy_hash))

    for stable_hash, stage, legacy_hash in updates:
        conn.execute(
            """
            UPDATE risk_model_build_state
            SET config_hash=?
            WHERE stage=? AND config_hash=?
            """,
            (stable_hash, stage, legacy_hash),
        )
    if updates:
        conn.commit()
        print(
            "Migrated legacy checkpoint hashes for an incremental model-end extension.",
            flush=True,
        )
    return bool(updates)


def create_model_schema(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;

        CREATE TABLE IF NOT EXISTS risk_model_build_state (
            stage TEXT NOT NULL,
            item TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_written INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            message TEXT,
            PRIMARY KEY (stage, item)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS risk_model_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_raw_exposure (
            model_date TEXT NOT NULL,
            code TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            weekly_return REAL,
            total_market_cap REAL,
            float_market_cap REAL,
            beta_raw REAL,
            momentum_raw REAL,
            residual_volatility_raw REAL,
            liquidity_raw REAL,
            avg_amount_60 REAL,
            history_days INTEGER,
            listed_state TEXT,
            PRIMARY KEY (model_date, code)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_exposure (
            model_date TEXT NOT NULL,
            code TEXT NOT NULL,
            observation_date TEXT,
            financial_report_period TEXT,
            financial_available_date TEXT,
            industry_implement_date TEXT,
            industry_group TEXT NOT NULL,
            industry_code TEXT,
            total_market_cap REAL NOT NULL,
            float_market_cap REAL,
            avg_amount_60 REAL,
            SIZE REAL NOT NULL,
            NONLINEAR_SIZE REAL NOT NULL,
            BETA REAL NOT NULL,
            MOMENTUM REAL NOT NULL,
            RESIDUAL_VOLATILITY REAL NOT NULL,
            LIQUIDITY REAL NOT NULL,
            VALUE REAL NOT NULL,
            EARNINGS_YIELD REAL NOT NULL,
            GROWTH REAL NOT NULL,
            LEVERAGE REAL NOT NULL,
            PRIMARY KEY (model_date, code)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_factor_return (
            model_date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            factor_return REAL NOT NULL,
            PRIMARY KEY (model_date, factor_name)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_specific_return (
            model_date TEXT NOT NULL,
            exposure_date TEXT NOT NULL,
            code TEXT NOT NULL,
            weekly_return REAL NOT NULL,
            predicted_return REAL NOT NULL,
            specific_return REAL NOT NULL,
            industry_group TEXT NOT NULL,
            regression_weight REAL NOT NULL,
            PRIMARY KEY (model_date, code)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_factor_covariance (
            model_date TEXT NOT NULL,
            factor_1 TEXT NOT NULL,
            factor_2 TEXT NOT NULL,
            covariance REAL NOT NULL,
            PRIMARY KEY (model_date, factor_1, factor_2)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_specific_risk (
            model_date TEXT NOT NULL,
            code TEXT NOT NULL,
            industry_group TEXT NOT NULL,
            specific_variance REAL NOT NULL,
            specific_volatility REAL NOT NULL,
            specific_observations INTEGER NOT NULL,
            PRIMARY KEY (model_date, code)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weekly_risk_diagnostics (
            model_date TEXT PRIMARY KEY,
            universe_count INTEGER NOT NULL,
            financial_book_coverage REAL,
            financial_ttm_coverage REAL,
            industry_coverage REAL,
            regression_count INTEGER,
            factor_count INTEGER,
            weighted_r_squared REAL,
            covariance_factor_count INTEGER,
            covariance_min_eigenvalue REAL,
            specific_risk_count INTEGER,
            created_at TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_weekly_raw_code
        ON weekly_raw_exposure(code, model_date);
        CREATE INDEX IF NOT EXISTS idx_weekly_exposure_code
        ON weekly_exposure(code, model_date);
        CREATE INDEX IF NOT EXISTS idx_specific_return_code
        ON weekly_specific_return(code, model_date);
        """
    )


def trading_week_ends(market_database, start_date, end_date):
    conn = sqlite3.connect(market_database)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT trade_date
            FROM stock_daily
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    dates = pd.to_datetime([row[0] for row in rows], errors="coerce")
    dates = dates[~dates.isna()]
    if len(dates) == 0:
        return []
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return (
        frame.groupby("week")["date"]
        .max()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )


def trading_calendar(market_database, start_date, end_date):
    conn = sqlite3.connect(market_database)
    try:
        return [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT trade_date
                FROM stock_daily
                WHERE trade_date BETWEEN ? AND ?
                ORDER BY trade_date
                """,
                (start_date, end_date),
            )
        ]
    finally:
        conn.close()


def initialize_worker(risk_database, market_database, weekly_dates, config):
    global _WORKER_CONNECTION, _WORKER_WEEKLY_DATES, _WORKER_CONFIG
    _WORKER_CONNECTION = sqlite3.connect(risk_database, timeout=120)
    _WORKER_CONNECTION.execute("ATTACH DATABASE ? AS market", (market_database,))
    _WORKER_CONNECTION.execute("PRAGMA query_only=ON")
    _WORKER_WEEKLY_DATES = weekly_dates
    _WORKER_CONFIG = config


def load_stock_daily_for_worker(code):
    market_types = [
        int(value) for value in _WORKER_CONFIG.get("market_return_types", [117, 53])
    ]
    market_cases = ", ".join(
        (
            f"MAX(CASE WHEN market_type={market_type} "
            "THEN total_weight_return END)"
        )
        for market_type in market_types
    )
    benchmark = str(_WORKER_CONFIG.get("risk_free_benchmark", "NRI01"))
    history_start = str(_WORKER_CONFIG["history_start_date"])
    model_end = str(_WORKER_CONFIG["model_end_date"])
    query = f"""
        SELECT d.trade_date, d.daily_return,
               CASE
                   WHEN c.float_market_cap > 0
                    AND d.volume IS NOT NULL
                    AND d.raw_close > 0
                   THEN d.volume * d.raw_close * 100.0 / c.float_market_cap
                   ELSE d.turnover_float
               END AS turnover_float,
               d.amount,
               d.listed_state, c.total_market_cap, c.float_market_cap,
               mr.market_return,
               COALESCE(rf.daily_rate_pct / 100.0, 0.0) AS risk_free_return
        FROM market.stock_daily d
        JOIN stock_market_cap c
          ON c.code=d.code AND c.trade_date=d.trade_date
        LEFT JOIN (
            SELECT trade_date, COALESCE({market_cases}) AS market_return
            FROM market_return_daily
            GROUP BY trade_date
        ) mr ON mr.trade_date=d.trade_date
        LEFT JOIN risk_free_daily rf
          ON rf.trade_date=d.trade_date AND rf.benchmark_code=?
        WHERE d.code=? AND d.trade_date BETWEEN ? AND ?
        ORDER BY d.trade_date
    """
    return pd.read_sql_query(
        query,
        _WORKER_CONNECTION,
        params=(benchmark, code, history_start, model_end),
    )


def process_stock(code):
    try:
        daily = load_stock_daily_for_worker(code)
        rows = compute_stock_weekly_features(
            daily, _WORKER_WEEKLY_DATES, _WORKER_CONFIG
        )
        for row in rows:
            row["code"] = code
        return code, rows, None
    except Exception as exc:
        return code, [], f"{type(exc).__name__}: {exc}"


def mark_state(conn, stage, item, config_hash, status, rows=0, message=None):
    conn.execute(
        """
        INSERT INTO risk_model_build_state (
            stage, item, config_hash, status, rows_written, updated_at, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stage, item) DO UPDATE SET
            config_hash=excluded.config_hash,
            status=excluded.status,
            rows_written=excluded.rows_written,
            updated_at=excluded.updated_at,
            message=excluded.message
        """,
        (stage, item, config_hash, status, int(rows), now_iso(), message),
    )


def clear_stage(conn, stage):
    if stage == "raw":
        conn.execute("DELETE FROM weekly_raw_exposure")
        conn.execute("DELETE FROM risk_model_build_state WHERE stage='raw'")
    elif stage == "model":
        for table in (
            "weekly_exposure",
            "weekly_factor_return",
            "weekly_specific_return",
            "weekly_factor_covariance",
            "weekly_specific_risk",
            "weekly_risk_diagnostics",
        ):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM risk_model_build_state WHERE stage='model'")
    conn.commit()


def truncate_model_from(conn, model_date):
    for table in (
        "weekly_exposure",
        "weekly_factor_return",
        "weekly_specific_return",
        "weekly_factor_covariance",
        "weekly_specific_risk",
        "weekly_risk_diagnostics",
    ):
        conn.execute(
            f"DELETE FROM {table} WHERE model_date>=?", (model_date,)
        )
    conn.execute(
        """
        DELETE FROM risk_model_build_state
        WHERE stage='model' AND item>=?
        """,
        (model_date,),
    )
    conn.commit()


def assert_stage_hash(conn, stage, config_hash, overwrite):
    hashes = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT config_hash FROM risk_model_build_state WHERE stage=?",
            (stage,),
        )
    }
    if not hashes or hashes == {config_hash}:
        return
    if overwrite:
        clear_stage(conn, stage)
        return
    raise ValueError(
        f"Existing {stage} checkpoints were built with another configuration. "
        f"Run again with --overwrite-stage {stage}."
    )


def pending_codes(conn, config_hash, force_all=False):
    all_codes = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT code
            FROM stock_market_cap
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY code
            """,
            (
                json.loads(
                    conn.execute(
                        "SELECT value FROM risk_model_metadata WHERE key='raw_start'"
                    ).fetchone()[0]
                ),
                json.loads(
                    conn.execute(
                        "SELECT value FROM risk_model_metadata WHERE key='raw_end'"
                    ).fetchone()[0]
                ),
            ),
        )
    ]
    if force_all:
        return all_codes
    complete = {
        row[0]
        for row in conn.execute(
            """
            SELECT item FROM risk_model_build_state
            WHERE stage='raw' AND status='complete' AND config_hash=?
            """,
            (config_hash,),
        )
    }
    return [code for code in all_codes if code not in complete]


def save_raw_result(
    conn, code, rows, error, config_hash, refresh_from_date=None
):
    if refresh_from_date:
        conn.execute(
            """
            DELETE FROM weekly_raw_exposure
            WHERE code=? AND model_date>=?
            """,
            (code, refresh_from_date),
        )
    else:
        conn.execute("DELETE FROM weekly_raw_exposure WHERE code=?", (code,))
    if error:
        mark_state(conn, "raw", code, config_hash, "error", 0, error)
        return
    if refresh_from_date:
        rows = [
            row for row in rows
            if str(row.get("model_date")) >= refresh_from_date
        ]
    values = [
        tuple(sqlite_value(row.get(column)) for column in RAW_COLUMNS)
        for row in rows
    ]
    if values:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO weekly_raw_exposure
            ({','.join(RAW_COLUMNS)})
            VALUES ({','.join('?' for _ in RAW_COLUMNS)})
            """,
            values,
        )
    mark_state(conn, "raw", code, config_hash, "complete", len(values))


def build_raw_features(
    conn,
    risk_database,
    market_database,
    weekly_dates,
    config,
    workers,
    config_hash,
    refresh_from_date=None,
):
    metadata = {
        "raw_start": config["history_start_date"],
        "raw_end": config["model_end_date"],
    }
    conn.executemany(
        """
        INSERT OR REPLACE INTO risk_model_metadata(key, value)
        VALUES (?, ?)
        """,
        ((key, json.dumps(value)) for key, value in metadata.items()),
    )
    conn.commit()
    worker_dates = weekly_dates
    if refresh_from_date:
        matching = [
            index
            for index, model_date in enumerate(weekly_dates)
            if model_date >= refresh_from_date
        ]
        if not matching:
            raise ValueError(
                f"No model week exists on or after {refresh_from_date}"
            )
        first_index = matching[0]
        worker_dates = weekly_dates[max(0, first_index - 1) :]
    codes = pending_codes(
        conn, config_hash, force_all=bool(refresh_from_date)
    )
    total_codes = conn.execute(
        "SELECT COUNT(DISTINCT code) FROM stock_market_cap"
    ).fetchone()[0]
    completed_before = 0 if refresh_from_date else total_codes - len(codes)
    print(
        f"Raw feature stage: total={total_codes:,}, "
        f"complete={completed_before:,}, pending={len(codes):,}, workers={workers}",
        flush=True,
    )
    if refresh_from_date:
        print(
            f"  refreshing model dates from {refresh_from_date}; "
            f"worker dates start at {worker_dates[0]}",
            flush=True,
        )
    if not codes:
        return

    started = time.monotonic()
    processed = 0
    errors = 0
    commit_every = max(1, int(config.get("raw_commit_every_codes", 10)))

    if workers <= 1:
        initialize_worker(
            str(risk_database), str(market_database), worker_dates, config
        )
        for code in codes:
            result_code, rows, error = process_stock(code)
            save_raw_result(
                conn,
                result_code,
                rows,
                error,
                config_hash,
                refresh_from_date,
            )
            processed += 1
            errors += int(bool(error))
            if processed % commit_every == 0:
                conn.commit()
            if processed % 20 == 0 or processed == len(codes):
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"  raw {completed_before + processed:,}/{total_codes:,}; "
                    f"errors={errors}; {processed / elapsed:.2f} stocks/s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(
                str(risk_database),
                str(market_database),
                worker_dates,
                config,
            ),
        ) as executor:
            code_iterator = iter(codes)
            active = {}
            maximum_active = max(workers * 3, workers)
            for _ in range(min(maximum_active, len(codes))):
                code = next(code_iterator, None)
                if code is None:
                    break
                active[executor.submit(process_stock, code)] = code

            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result_code, rows, error = future.result()
                    save_raw_result(
                        conn,
                        result_code,
                        rows,
                        error,
                        config_hash,
                        refresh_from_date,
                    )
                    processed += 1
                    errors += int(bool(error))
                    if processed % commit_every == 0:
                        conn.commit()
                    next_code = next(code_iterator, None)
                    if next_code is not None:
                        active[executor.submit(process_stock, next_code)] = next_code
                if processed % 20 == 0 or processed == len(codes):
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"  raw {completed_before + processed:,}/{total_codes:,}; "
                        f"errors={errors}; {processed / elapsed:.2f} stocks/s",
                        flush=True,
                    )
    conn.commit()
    if errors:
        print(
            f"Raw stage completed with {errors} stock errors. "
            "Re-running will retry only error rows.",
            flush=True,
        )


class FinancialPointInTimeStore:
    def __init__(self, rows, calendar=None):
        self.calendar = list(calendar or [])
        self.book_events = []
        self.ttm_events = []
        self.growth_events = []
        for row in rows:
            record = dict(row)
            code = str(record["code"])
            period = str(record["report_period"])
            if record.get("available_date"):
                effective = self._next_trading_date(record["available_date"])
                if effective:
                    self.book_events.append((effective, code, period, record))
            if record.get("ttm_available_date"):
                effective = self._next_trading_date(record["ttm_available_date"])
                if effective:
                    self.ttm_events.append((effective, code, period, record))
            if record.get("growth_available_date"):
                effective = self._next_trading_date(record["growth_available_date"])
                if effective:
                    self.growth_events.append((effective, code, period, record))
        self.book_events.sort(key=lambda item: item[:3])
        self.ttm_events.sort(key=lambda item: item[:3])
        self.growth_events.sort(key=lambda item: item[:3])
        self.book_index = 0
        self.ttm_index = 0
        self.growth_index = 0
        self.book_current = {}
        self.ttm_current = {}
        self.growth_current = {}

    def _next_trading_date(self, source_date):
        if not self.calendar:
            return source_date
        index = bisect_right(self.calendar, str(source_date))
        return self.calendar[index] if index < len(self.calendar) else None

    @staticmethod
    def _advance(events, index, current, model_date):
        while index < len(events) and events[index][0] <= model_date:
            _, code, period, record = events[index]
            old = current.get(code)
            if old is None or period >= old["report_period"]:
                current[code] = record
            index += 1
        return index

    def advance(self, model_date):
        self.book_index = self._advance(
            self.book_events, self.book_index, self.book_current, model_date
        )
        self.ttm_index = self._advance(
            self.ttm_events, self.ttm_index, self.ttm_current, model_date
        )
        self.growth_index = self._advance(
            self.growth_events,
            self.growth_index,
            self.growth_current,
            model_date,
        )

    def get(self, code):
        book = self.book_current.get(code, {})
        ttm = self.ttm_current.get(code, {})
        growth = self.growth_current.get(code, {})
        return {
            "financial_report_period": book.get("report_period"),
            "financial_available_date": book.get("available_date"),
            "total_assets": book.get("total_assets"),
            "total_liabilities": book.get("total_liabilities"),
            "parent_equity": book.get("parent_equity"),
            "total_equity": book.get("total_equity"),
            "parent_net_profit_ttm": ttm.get("parent_net_profit_ttm"),
            "operating_cashflow_ttm": ttm.get("operating_cashflow_ttm"),
            "revenue_growth": growth.get("revenue_growth"),
            "earnings_growth": growth.get("earnings_growth"),
        }


class IndustryPointInTimeStore:
    def __init__(
        self,
        rows,
        group_digits,
        classification_schedule=None,
        fallback_classifications=None,
    ):
        self.events = []
        for row in rows:
            record = dict(row)
            self.events.append(
                (
                    record["implement_date"],
                    str(record["code"]),
                    str(record.get("classification_name") or ""),
                    record,
                )
            )
        self.events.sort(key=lambda item: item[:3])
        self.index = 0
        self.current = {}
        self.model_date = None
        self.group_digits = int(group_digits)
        self.classification_schedule = list(classification_schedule or [])
        self.fallback_classifications = list(fallback_classifications or [])

    def advance(self, model_date):
        while self.index < len(self.events) and self.events[self.index][0] <= model_date:
            _, code, classification_name, record = self.events[self.index]
            self.current[(code, classification_name)] = record
            self.index += 1
        self.model_date = str(model_date)

    def get(self, code):
        active_primary = None
        for item in self.classification_schedule:
            if str(item["effective_date"]) <= str(self.model_date or ""):
                active_primary = str(item["classification_name"])
            else:
                break
        search_order = [active_primary] if active_primary else []
        search_order.extend(self.fallback_classifications)
        record = None
        for classification_name in search_order:
            record = self.current.get((str(code), str(classification_name)))
            if record:
                break
        if not record:
            return {
                "industry_implement_date": None,
                "industry_code": None,
                "industry_group": "UNKNOWN",
            }
        industry_code = str(record.get("industry_code") or "")
        return {
            "industry_implement_date": record.get("implement_date"),
            "industry_code": industry_code or None,
            "industry_group": (
                industry_code[: self.group_digits]
                if industry_code
                else "UNKNOWN"
            ),
        }


def industry_classification_schedule(config):
    configured = config.get("industry_classification_schedule")
    if not configured:
        return [
            {
                "classification_name": str(config["industry_classification_name"]),
                "effective_date": "1900-01-01",
            }
        ]
    schedule = []
    for item in configured:
        name = str(item.get("classification_name") or "").strip()
        effective_date = str(item.get("effective_date") or "")[:10]
        if not name or len(effective_date) != 10:
            raise ValueError(f"Invalid industry classification schedule row: {item!r}")
        schedule.append(
            {"classification_name": name, "effective_date": effective_date}
        )
    schedule.sort(key=lambda item: item["effective_date"])
    return schedule


def load_point_in_time_stores(conn, config, calendar):
    conn.row_factory = sqlite3.Row
    financial_rows = conn.execute(
        """
        SELECT *
        FROM financial_pit
        WHERE available_date IS NOT NULL
        ORDER BY available_date, code, report_period
        """
    ).fetchall()
    classification_schedule = industry_classification_schedule(config)
    fallback_classifications = [
        str(name).strip()
        for name in config.get("industry_fallback_classification_names", [])
        if str(name).strip()
    ]
    effective_dates = {
        item["classification_name"]: item["effective_date"]
        for item in classification_schedule
    }
    names = list(dict.fromkeys([*effective_dates, *fallback_classifications]))
    industry_rows_raw = conn.execute(
        f"""
        SELECT code, classification_name, implement_date, industry_code, industry_name
        FROM industry_history
        WHERE classification_name IN ({','.join('?' for _ in names)})
        ORDER BY implement_date, code, classification_name
        """,
        names,
    ).fetchall()
    industry_rows = []
    for row in industry_rows_raw:
        record = dict(row)
        record["source_implement_date"] = record["implement_date"]
        if str(record["classification_name"]) in effective_dates:
            record["implement_date"] = max(
                str(record["implement_date"]),
                effective_dates[str(record["classification_name"])],
            )
        industry_rows.append(record)
    conn.row_factory = None
    if not industry_rows:
        raise ValueError(
            "No rows found for industry classification schedule="
            f"{classification_schedule!r}"
        )
    return (
        FinancialPointInTimeStore(financial_rows, calendar),
        IndustryPointInTimeStore(
            industry_rows,
            config.get("industry_group_digits", 2),
            classification_schedule=classification_schedule,
            fallback_classifications=fallback_classifications,
        ),
    )


def attach_point_in_time_data(raw, financial_store, industry_store):
    records = []
    for row in raw.to_dict("records"):
        code = str(row["code"])
        row.update(financial_store.get(code))
        row.update(industry_store.get(code))
        records.append(row)
    return pd.DataFrame(records)


def insert_exposures(conn, exposure):
    conn.execute(
        "DELETE FROM weekly_exposure WHERE model_date=?",
        (str(exposure["model_date"].iloc[0]),),
    )
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO weekly_exposure
        ({','.join(EXPOSURE_COLUMNS)})
        VALUES ({','.join('?' for _ in EXPOSURE_COLUMNS)})
        """,
        (
            tuple(sqlite_value(row.get(column)) for column in EXPOSURE_COLUMNS)
            for row in exposure.to_dict("records")
        ),
    )


def load_factor_history(conn):
    rows = conn.execute(
        """
        SELECT model_date, factor_name, factor_return
        FROM weekly_factor_return
        ORDER BY model_date, factor_name
        """
    ).fetchall()
    history = defaultdict(dict)
    for model_date, factor_name, factor_return in rows:
        history[model_date][factor_name] = float(factor_return)
    return history


def load_specific_history(conn):
    history = defaultdict(list)
    rows = conn.execute(
        """
        SELECT code, specific_return
        FROM weekly_specific_return
        ORDER BY model_date, code
        """
    )
    for code, specific_return in rows:
        history[str(code)].append(float(specific_return))
    return history


def load_exposure(conn, model_date):
    return pd.read_sql_query(
        "SELECT * FROM weekly_exposure WHERE model_date=? ORDER BY code",
        conn,
        params=(model_date,),
    )


def save_factor_returns(conn, model_date, factor_returns):
    conn.execute(
        "DELETE FROM weekly_factor_return WHERE model_date=?", (model_date,)
    )
    conn.executemany(
        """
        INSERT INTO weekly_factor_return(model_date, factor_name, factor_return)
        VALUES (?, ?, ?)
        """,
        (
            (model_date, factor, float(value))
            for factor, value in sorted(factor_returns.items())
        ),
    )


def save_specific_returns(conn, model_date, exposure_date, specific):
    conn.execute(
        "DELETE FROM weekly_specific_return WHERE model_date=?", (model_date,)
    )
    rows = []
    for row in specific.to_dict("records"):
        predicted = float(row["weekly_return"] - row["specific_return"])
        rows.append(
            (
                model_date,
                exposure_date,
                str(row["code"]),
                float(row["weekly_return"]),
                predicted,
                float(row["specific_return"]),
                str(row["industry_group"]),
                float(row["regression_weight"]),
            )
        )
    conn.executemany(
        """
        INSERT INTO weekly_specific_return
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def save_factor_covariance(conn, model_date, factor_names, covariance):
    conn.execute(
        "DELETE FROM weekly_factor_covariance WHERE model_date=?",
        (model_date,),
    )
    rows = []
    for row_index, factor_1 in enumerate(factor_names):
        for column_index, factor_2 in enumerate(factor_names):
            rows.append(
                (
                    model_date,
                    factor_1,
                    factor_2,
                    float(covariance[row_index, column_index]),
                )
            )
    conn.executemany(
        "INSERT INTO weekly_factor_covariance VALUES (?, ?, ?, ?)", rows
    )


def save_specific_risk(conn, model_date, frame):
    conn.execute(
        "DELETE FROM weekly_specific_risk WHERE model_date=?", (model_date,)
    )
    conn.executemany(
        """
        INSERT INTO weekly_specific_risk
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            (
                model_date,
                str(row["code"]),
                str(row["industry_group"]),
                float(row["specific_variance"]),
                float(row["specific_volatility"]),
                int(row["specific_observations"]),
            )
            for row in frame.to_dict("records")
        ),
    )


def save_diagnostics(conn, model_date, values):
    columns = (
        "model_date",
        "universe_count",
        "financial_book_coverage",
        "financial_ttm_coverage",
        "industry_coverage",
        "regression_count",
        "factor_count",
        "weighted_r_squared",
        "covariance_factor_count",
        "covariance_min_eigenvalue",
        "specific_risk_count",
        "created_at",
    )
    conn.execute(
        f"""
        INSERT OR REPLACE INTO weekly_risk_diagnostics
        ({','.join(columns)})
        VALUES ({','.join('?' for _ in columns)})
        """,
        tuple(values.get(column) for column in columns),
    )


def build_model(
    conn,
    weekly_dates,
    calendar,
    config,
    config_hash,
):
    financial_store, industry_store = load_point_in_time_stores(
        conn, config, calendar
    )
    completed = {
        row[0]
        for row in conn.execute(
            """
            SELECT item
            FROM risk_model_build_state
            WHERE stage='model' AND status='complete' AND config_hash=?
            """,
            (config_hash,),
        )
    }
    factor_history = load_factor_history(conn)
    specific_history = load_specific_history(conn)
    previous_exposure = None
    previous_date = None
    if completed:
        last_completed = max(completed)
        previous_exposure = load_exposure(conn, last_completed)
        previous_date = last_completed
    else:
        last_completed = None

    classification_names = [
        item["classification_name"]
        for item in industry_classification_schedule(config)
    ]
    classification_names.extend(
        str(name).strip()
        for name in config.get("industry_fallback_classification_names", [])
        if str(name).strip()
    )
    classification_names = list(dict.fromkeys(classification_names))
    group_digits = int(config.get("industry_group_digits", 2))
    industry_groups = sorted(
        {
            str(row[0])[:group_digits]
            for row in conn.execute(
                f"""
                SELECT DISTINCT industry_code
                FROM industry_history
                WHERE classification_name IN (
                    {','.join('?' for _ in classification_names)}
                ) AND industry_code IS NOT NULL
                """,
                classification_names,
            )
            if row[0]
        }
    )
    if "UNKNOWN" not in industry_groups:
        industry_groups.append("UNKNOWN")
    expected_factors = [
        "MARKET",
        *STYLE_FACTORS,
        *(f"INDUSTRY:{industry}" for industry in industry_groups),
    ]

    model_dates = [
        date
        for date in weekly_dates
        if str(config["model_start_date"]) <= date <= str(config["model_end_date"])
    ]
    print(
        f"Model stage: dates={len(model_dates)}, "
        f"complete={len(completed)}, resume_after={last_completed}",
        flush=True,
    )
    started = time.monotonic()
    completed_this_run = 0

    for date_index, model_date in enumerate(model_dates, start=1):
        financial_store.advance(model_date)
        industry_store.advance(model_date)
        if model_date in completed:
            continue
        raw = pd.read_sql_query(
            """
            SELECT *
            FROM weekly_raw_exposure
            WHERE model_date=?
            ORDER BY code
            """,
            conn,
            params=(model_date,),
        )
        if raw.empty:
            raise ValueError(f"No weekly_raw_exposure rows for {model_date}")
        attached = attach_point_in_time_data(
            raw, financial_store, industry_store
        )
        exposure = build_cross_section_exposures(attached, config)
        if exposure.empty:
            raise ValueError(f"No usable risk exposure rows for {model_date}")

        regression_diagnostics = {}
        covariance = None
        specific_risk = pd.DataFrame()
        insert_exposures(conn, exposure)

        if previous_exposure is not None and previous_date is not None:
            usable_return_count = int(
                pd.to_numeric(
                    raw["weekly_return"], errors="coerce"
                ).notna().sum()
            )
            if usable_return_count == 0:
                observation_max = raw["observation_date"].max()
                raise ValueError(
                    f"No weekly returns are available for {model_date}. "
                    f"The latest joined market-cap observation is "
                    f"{observation_max}. Import market caps through "
                    f"{model_date}, then refresh raw features from this date."
                )
            factor_returns, specific, regression_diagnostics = fit_factor_returns(
                previous_exposure,
                raw[["code", "weekly_return"]],
                industry_groups,
                config,
            )
            factor_history[model_date] = factor_returns
            save_factor_returns(conn, model_date, factor_returns)
            save_specific_returns(
                conn, model_date, previous_date, specific
            )
            for row in specific[["code", "specific_return"]].itertuples(
                index=False
            ):
                specific_history[str(row.code)].append(float(row.specific_return))

            factor_dates = sorted(factor_history)
            minimum_factor_history = int(
                config.get("minimum_factor_history_weeks", 26)
            )
            if len(factor_dates) >= minimum_factor_history:
                covariance_window = int(
                    config.get("factor_covariance_window_weeks", 156)
                )
                selected_dates = factor_dates[-covariance_window:]
                factor_frame = pd.DataFrame(
                    [
                        {
                            factor: factor_history[date].get(factor, 0.0)
                            for factor in expected_factors
                        }
                        for date in selected_dates
                    ],
                    index=selected_dates,
                )
                covariance = ewma_newey_west_covariance(
                    factor_frame.to_numpy(dtype=float),
                    half_life=float(
                        config.get("factor_covariance_half_life_weeks", 52)
                    ),
                    newey_west_lags=int(config.get("newey_west_lags", 2)),
                    shrinkage=float(
                        config.get("factor_covariance_shrinkage", 0.10)
                    ),
                    annualization=52,
                )
                save_factor_covariance(
                    conn, model_date, expected_factors, covariance
                )

            specific_risk = estimate_specific_variances(
                specific_history,
                exposure,
                half_life=float(
                    config.get("specific_risk_half_life_weeks", 26)
                ),
                minimum_observations=int(
                    config.get("minimum_specific_history_weeks", 20)
                ),
                shrinkage=float(
                    config.get("specific_risk_shrinkage", 0.20)
                ),
                annualization=52,
            )
            if not specific_risk.empty:
                save_specific_risk(conn, model_date, specific_risk)

        raw_count = len(attached)
        diagnostics = {
            "model_date": model_date,
            "universe_count": len(exposure),
            "financial_book_coverage": float(
                attached["financial_available_date"].notna().mean()
            ),
            "financial_ttm_coverage": float(
                attached["parent_net_profit_ttm"].notna().mean()
            ),
            "industry_coverage": float(
                (attached["industry_group"] != "UNKNOWN").mean()
            ),
            "regression_count": regression_diagnostics.get("regression_count"),
            "factor_count": regression_diagnostics.get("factor_count"),
            "weighted_r_squared": regression_diagnostics.get(
                "weighted_r_squared"
            ),
            "covariance_factor_count": (
                covariance.shape[0] if covariance is not None else None
            ),
            "covariance_min_eigenvalue": (
                float(np.linalg.eigvalsh(covariance).min())
                if covariance is not None
                else None
            ),
            "specific_risk_count": len(specific_risk),
            "created_at": now_iso(),
            "raw_count": raw_count,
        }
        save_diagnostics(conn, model_date, diagnostics)
        mark_state(
            conn,
            "model",
            model_date,
            config_hash,
            "complete",
            len(exposure),
        )
        conn.commit()
        previous_exposure = exposure
        previous_date = model_date
        completed_this_run += 1
        elapsed = max(time.monotonic() - started, 0.001)
        print(
            f"  model {date_index}/{len(model_dates)} {model_date}: "
            f"universe={len(exposure):,}, "
            f"regression={regression_diagnostics.get('regression_count', 0):,}, "
            f"specific={len(specific_risk):,}, "
            f"{completed_this_run / elapsed:.2f} dates/s",
            flush=True,
        )


def update_model_metadata(
    conn, config, raw_hash, model_hash, market_database, risk_database
):
    metadata = {
        "risk_model_version": "1",
        "updated_at": now_iso(),
        "config": config,
        "raw_config_hash": raw_hash,
        "model_config_hash": model_hash,
        "market_database": str(market_database),
        "risk_database": str(risk_database),
    }
    conn.executemany(
        """
        INSERT OR REPLACE INTO risk_model_metadata(key, value)
        VALUES (?, ?)
        """,
        (
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
            for key, value in metadata.items()
        ),
    )
    conn.commit()


def run(args):
    config = load_config(args.config)
    config["history_start_date"] = args.history_start_date or config.get(
        "history_start_date", "2019-01-01"
    )
    config["model_start_date"] = args.start_date or config.get(
        "model_start_date", "2021-01-01"
    )
    config["model_end_date"] = args.end_date or config.get(
        "model_end_date", "2026-07-24"
    )
    if config["history_start_date"] > config["model_start_date"]:
        raise ValueError("history_start_date must not be later than model_start_date")
    if config["model_start_date"] > config["model_end_date"]:
        raise ValueError("model_start_date must not be later than model_end_date")

    weekly_dates = trading_week_ends(
        args.market_database,
        config["model_start_date"],
        config["model_end_date"],
    )
    calendar = trading_calendar(
        args.market_database,
        config["history_start_date"],
        config["model_end_date"],
    )
    if not weekly_dates:
        raise ValueError("No trading week-end dates were found")
    raw_hash, model_hash = stage_config_hashes(config)

    conn = sqlite3.connect(args.risk_database, timeout=120)
    try:
        create_model_schema(conn)
        migrate_legacy_horizon_hashes(conn, config, raw_hash, model_hash)
        overwrite = set(args.overwrite_stage)
        assert_stage_hash(conn, "raw", raw_hash, "raw" in overwrite)
        assert_stage_hash(conn, "model", model_hash, "model" in overwrite)
        if "raw" in overwrite:
            clear_stage(conn, "raw")
        if "model" in overwrite:
            clear_stage(conn, "model")

        stages = set(args.stage)
        if "all" in stages:
            stages = {"raw", "model"}
        if "raw" in stages:
            if args.refresh_from_date:
                truncate_model_from(conn, args.refresh_from_date)
            build_raw_features(
                conn,
                args.risk_database,
                args.market_database,
                weekly_dates,
                config,
                args.workers,
                raw_hash,
                args.refresh_from_date,
            )
        if "model" in stages:
            error_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM risk_model_build_state
                WHERE stage='raw' AND status='error' AND config_hash=?
                """,
                (raw_hash,),
            ).fetchone()[0]
            if error_count:
                raise ValueError(
                    f"Raw feature stage still has {error_count} errors. "
                    "Re-run --stage raw before building the model."
                )
            build_model(conn, weekly_dates, calendar, config, model_hash)
        update_model_metadata(
            conn,
            config,
            raw_hash,
            model_hash,
            args.market_database,
            args.risk_database,
        )
        conn.execute("PRAGMA optimize")
        conn.commit()
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build weekly A-share risk exposures and covariance estimates."
    )
    parser.add_argument(
        "--market-database", type=Path, default=DEFAULT_MARKET_DATABASE
    )
    parser.add_argument(
        "--risk-database", type=Path, default=DEFAULT_RISK_DATABASE
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history-start-date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--stage",
        action="append",
        choices=("all", "raw", "model"),
        default=[],
    )
    parser.add_argument(
        "--overwrite-stage",
        action="append",
        choices=("raw", "model"),
        default=[],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument(
        "--refresh-from-date",
        help=(
            "Rebuild raw weekly rows on and after this model date while "
            "retaining earlier checkpoints."
        ),
    )
    args = parser.parse_args(argv)
    if not args.stage:
        args.stage = ["all"]
    args.market_database = args.market_database.resolve()
    args.risk_database = args.risk_database.resolve()
    args.config = args.config.resolve()
    if not args.market_database.exists():
        parser.error(f"Market database does not exist: {args.market_database}")
    if not args.risk_database.exists():
        parser.error(f"Risk database does not exist: {args.risk_database}")
    if not args.config.exists():
        parser.error(f"Config does not exist: {args.config}")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.refresh_from_date and not any(
        stage in {"all", "raw"} for stage in args.stage
    ):
        parser.error("--refresh-from-date requires --stage raw or --stage all")
    return args


if __name__ == "__main__":
    run(parse_args())
