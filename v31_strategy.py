"""Shared causal alpha and monthly factor-state logic for V3.1."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import factor_rank_backtest as base


DEFAULT_COMPONENT_WEIGHTS = {
    "low_beta_score": 0.10,
    "low_volatility_score": 0.12,
    "low_turnover_score": 0.08,
    "lower_drawdown_score": 0.08,
    "industry_trend_score": 0.12,
    "earnings_yield_score": 0.16,
    "quality_score_v31": 0.14,
    "growth_score_v31": 0.10,
    "residual_momentum_score": 0.10,
}

DEFENSIVE_COMPONENTS = (
    "low_beta_score",
    "low_volatility_score",
    "low_turnover_score",
    "lower_drawdown_score",
)
OFFENSIVE_COMPONENTS = (
    "industry_trend_score",
    "earnings_yield_score",
    "quality_score_v31",
    "growth_score_v31",
    "residual_momentum_score",
)

UNKNOWN_INDUSTRY_VALUES = {"", "NAN", "NONE", "NULL", "--", "UNKNOWN"}


def normalize_industry_series(values):
    result = values.fillna("UNKNOWN").astype(str).str.strip()
    return result.mask(result.str.upper().isin(UNKNOWN_INDUSTRY_VALUES), "UNKNOWN")


def _read_only_connection(path):
    path = Path(path).resolve()
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _latest_date(dates, decision_date):
    index = bisect_right(dates, str(decision_date)) - 1
    return dates[index] if index >= 0 else None


def _is_stale(model_date, decision_date, maximum_staleness_days):
    if model_date is None:
        return True
    return (
        date.fromisoformat(str(decision_date)) - date.fromisoformat(str(model_date))
    ).days > int(maximum_staleness_days)


class V31AlphaFeatureStore:
    """Attach PIT value, quality, growth, residual momentum and float cap."""

    def __init__(
        self,
        database,
        lookback_weeks=52,
        skip_weeks=4,
        before_cutover_database=None,
        cutover_date=None,
    ):
        self.database = Path(database).resolve()
        self.conn = _read_only_connection(self.database)
        self.lookback_weeks = int(lookback_weeks)
        self.skip_weeks = int(skip_weeks)
        required = {"weekly_v31_alpha", "weekly_residual_momentum"}
        if before_cutover_database is not None:
            if cutover_date is None:
                self.close()
                raise ValueError(
                    "A cutover date is required for a split V3.1 alpha database."
                )
            cutover = str(cutover_date)
            try:
                date.fromisoformat(cutover)
            except ValueError as exc:
                self.close()
                raise ValueError(
                    "The V3.1 alpha cutover date must use YYYY-MM-DD."
                ) from exc
            before_path = Path(before_cutover_database).resolve()
            self.conn.execute(
                "ATTACH DATABASE ? AS before_cutover",
                (before_path.as_uri() + "?mode=ro",),
            )
            for table in sorted(required):
                after_columns = [
                    row[1]
                    for row in self.conn.execute(
                        f"PRAGMA main.table_info([{table}])"
                    )
                ]
                before_columns = [
                    row[1]
                    for row in self.conn.execute(
                        f"PRAGMA before_cutover.table_info([{table}])"
                    )
                ]
                if not after_columns or after_columns != before_columns:
                    self.close()
                    raise ValueError(
                        f"Pre/post-cutover {table} schemas do not match exactly."
                    )
                columns_sql = ", ".join(
                    f'"{column.replace(chr(34), chr(34) * 2)}"'
                    for column in after_columns
                )
                self.conn.execute(
                    f"""
                    CREATE TEMP VIEW [{table}] AS
                    SELECT {columns_sql}
                    FROM before_cutover.[{table}]
                    WHERE model_date < '{cutover}'
                    UNION ALL
                    SELECT {columns_sql}
                    FROM main.[{table}]
                    WHERE model_date >= '{cutover}'
                    """
                )
        found = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = sorted(required - found)
        if missing:
            self.close()
            raise ValueError(
                "V3.1 alpha cache is incomplete; missing tables: " + ", ".join(missing)
            )
        self.alpha_dates = [
            str(row[0])
            for row in self.conn.execute(
                "SELECT DISTINCT model_date FROM weekly_v31_alpha ORDER BY model_date"
            )
        ]
        self.momentum_dates = [
            str(row[0])
            for row in self.conn.execute(
                """
                SELECT DISTINCT model_date FROM weekly_residual_momentum
                WHERE lookback_weeks=? AND skip_weeks=? ORDER BY model_date
                """,
                (self.lookback_weeks, self.skip_weeks),
            )
        ]
        self._alpha_date = None
        self._alpha = None
        self._momentum_date = None
        self._momentum = None

    def augment(self, features, decision_date, maximum_staleness_days=14):
        result = features.copy()
        alpha_date = _latest_date(self.alpha_dates, decision_date)
        momentum_date = _latest_date(self.momentum_dates, decision_date)
        alpha_valid = not _is_stale(
            alpha_date, decision_date, maximum_staleness_days
        )
        momentum_valid = not _is_stale(
            momentum_date, decision_date, maximum_staleness_days
        )
        if result.empty:
            for column in (
                "v31_float_market_cap",
                "v31_market_industry_weight",
                "v31_earnings_yield_raw",
                "v31_quality_raw",
                "v31_growth_raw",
                "residual_momentum_raw",
            ):
                result[column] = np.nan
            return result, {
                "v31_alpha_status": "empty_universe",
                "v31_alpha_date": alpha_date,
                "v31_residual_momentum_date": momentum_date,
            }
        if alpha_valid and alpha_date != self._alpha_date:
            self._alpha = pd.read_sql_query(
                """
                SELECT code,
                       float_market_cap AS v31_float_market_cap,
                       market_industry_weight AS v31_market_industry_weight,
                       earnings_yield_raw AS v31_earnings_yield_raw,
                       quality_raw AS v31_quality_raw,
                       growth_raw AS v31_growth_raw
                FROM weekly_v31_alpha WHERE model_date=?
                """,
                self.conn,
                params=(alpha_date,),
            )
            self._alpha["code"] = self._alpha["code"].astype(str).str.zfill(6)
            self._alpha_date = alpha_date
        if momentum_valid and momentum_date != self._momentum_date:
            self._momentum = pd.read_sql_query(
                """
                SELECT code, residual_momentum AS residual_momentum_raw,
                       observations AS residual_momentum_observations
                FROM weekly_residual_momentum
                WHERE model_date=? AND lookback_weeks=? AND skip_weeks=?
                """,
                self.conn,
                params=(momentum_date, self.lookback_weeks, self.skip_weeks),
            )
            self._momentum["code"] = self._momentum["code"].astype(str).str.zfill(6)
            self._momentum_date = momentum_date
        result["code"] = result["code"].astype(str).str.zfill(6)
        if alpha_valid:
            result = result.merge(self._alpha, on="code", how="left")
        else:
            for column in (
                "v31_float_market_cap",
                "v31_market_industry_weight",
                "v31_earnings_yield_raw",
                "v31_quality_raw",
                "v31_growth_raw",
            ):
                result[column] = np.nan
        if momentum_valid:
            result = result.merge(self._momentum, on="code", how="left")
        else:
            result["residual_momentum_raw"] = np.nan
            result["residual_momentum_observations"] = np.nan
        coverage = {}
        for column in (
            "v31_earnings_yield_raw",
            "v31_quality_raw",
            "v31_growth_raw",
            "residual_momentum_raw",
        ):
            coverage[f"{column}_coverage"] = float(
                pd.to_numeric(result[column], errors="coerce").notna().mean()
            )
        return result, {
            "v31_alpha_status": (
                "applied"
                if alpha_valid and momentum_valid
                else "partial"
                if alpha_valid or momentum_valid
                else "no_causal_snapshot"
            ),
            "v31_alpha_date": alpha_date,
            "v31_residual_momentum_date": momentum_date,
            **coverage,
        }

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None


def normalized_component_weights(configured=None):
    source = dict(DEFAULT_COMPONENT_WEIGHTS)
    if configured:
        unknown = sorted(set(configured) - set(DEFAULT_COMPONENT_WEIGHTS))
        if unknown:
            raise ValueError("Unknown V3.1 component weights: " + ", ".join(unknown))
        source.update({name: float(value) for name, value in configured.items()})
    series = pd.Series(source, dtype=float).clip(lower=0.0)
    if series.sum() <= 0:
        raise ValueError("V3.1 component weights must contain a positive value")
    return (series / series.sum()).to_dict()


def apply_score(features, weights):
    result = features.copy()
    result["industry_1"] = normalize_industry_series(result["industry_1"])
    # Industry selection is deliberately independent of defensive stock traits.
    # Every stock in one industry receives the same 20/60-day relative-trend input.
    result["industry_trend_score"] = (
        0.55
        * base.factor_zscore(
            result,
            "industry_relative_20",
            industry_neutral=False,
        )
        + 0.45
        * base.factor_zscore(
            result,
            "industry_relative_60",
            industry_neutral=False,
        )
    )
    result["low_beta_score"] = base.factor_zscore(
        result, "beta_120", direction=-1.0, industry_neutral=True
    )
    result["low_volatility_score"] = base.factor_zscore(
        result, "volatility_120", direction=-1.0, industry_neutral=True
    )
    result["low_turnover_score"] = base.factor_zscore(
        result, "turnover_20", direction=-1.0, industry_neutral=True
    )
    result["lower_drawdown_score"] = base.factor_zscore(
        result, "drawdown_120", direction=1.0, industry_neutral=True
    )
    result["earnings_yield_score"] = base.factor_zscore(
        result, "v31_earnings_yield_raw", industry_neutral=True
    )
    result["quality_score_v31"] = base.factor_zscore(
        result, "v31_quality_raw", industry_neutral=True
    )
    result["growth_score_v31"] = base.factor_zscore(
        result, "v31_growth_raw", industry_neutral=True
    )
    result["residual_momentum_score"] = base.factor_zscore(
        result, "residual_momentum_raw", industry_neutral=True
    )
    score = pd.Series(0.0, index=result.index)
    for component, weight in weights.items():
        values = pd.to_numeric(result.get(component), errors="coerce")
        score += float(weight) * values.fillna(0.0)
    result["score_v2"] = score
    return result.sort_values("score_v2", ascending=False).reset_index(drop=True)


def industry_budget_caps(features, args, target_equity_weight):
    industries = normalize_industry_series(features["industry_1"])
    supplied_weights = pd.to_numeric(
        features.get(
            "v31_market_industry_weight",
            pd.Series(np.nan, index=features.index, dtype=float),
        ),
        errors="coerce",
    )
    if isinstance(supplied_weights, pd.Series) and supplied_weights.notna().any():
        market_weights = (
            supplied_weights.groupby(industries).max().dropna().clip(lower=0.0)
        )
    else:
        cap_source = pd.to_numeric(
            features.get(
                "v31_float_market_cap",
                pd.Series(np.nan, index=features.index, dtype=float),
            ),
            errors="coerce",
        )
        fallback = pd.to_numeric(
            features.get(
                "market_cap_proxy_20",
                pd.Series(np.nan, index=features.index, dtype=float),
            ),
            errors="coerce",
        )
        cap_source = (
            cap_source.where(cap_source > 0, fallback).fillna(0.0).clip(lower=0.0)
        )
        totals = cap_source.groupby(industries).sum()
        market_weights = (
            totals / totals.sum() if totals.sum() > 0 else pd.Series(dtype=float)
        )
    absolute = max(0.0, float(args.v31_absolute_industry_cap))
    deviation = max(0.0, float(args.v31_industry_cap_deviation))
    unknown_cap = max(0.0, float(args.v31_unknown_industry_cap))
    caps = {
        industry: min(
            absolute,
            (float(weight) + deviation) * float(target_equity_weight),
        )
        for industry, weight in market_weights.items()
        if industry != "UNKNOWN"
    }
    for industry in industries.unique():
        if industry != "UNKNOWN":
            caps.setdefault(
                industry,
                min(absolute, deviation * float(target_equity_weight)),
            )
    caps["UNKNOWN"] = min(unknown_cap, absolute if absolute > 0 else unknown_cap)
    return caps, market_weights.to_dict()


class MonthlyFactorStateStore:
    """Precompute causal monthly defensive/offensive allocations."""

    DEFENSIVE_RETURNS = {
        "BETA": -0.40,
        "RESIDUAL_VOLATILITY": -0.40,
        "LIQUIDITY": -0.20,
    }
    OFFENSIVE_RETURNS = {
        "EARNINGS_YIELD": 0.35,
        "GROWTH": 0.35,
        "MOMENTUM": 0.30,
    }

    def __init__(self, database, args):
        conn = _read_only_connection(database)
        try:
            frame = pd.read_sql_query(
                """
                SELECT model_date, factor_name, factor_return
                FROM weekly_factor_return ORDER BY model_date, factor_name
                """,
                conn,
            )
        finally:
            conn.close()
        frame["model_date"] = frame["model_date"].astype(str)
        frame["factor_return"] = pd.to_numeric(frame["factor_return"], errors="coerce")
        self.pivot = frame.pivot_table(
            index="model_date", columns="factor_name", values="factor_return", aggfunc="last"
        ).sort_index()
        self.args = args
        self.schedule = self._build_schedule()

    @staticmethod
    def _basket(frame, mapping):
        available = [(name, weight) for name, weight in mapping.items() if name in frame]
        if not available:
            return pd.Series(np.nan, index=frame.index)
        denominator = sum(abs(weight) for _, weight in available)
        return sum(frame[name].fillna(0.0) * weight for name, weight in available) / denominator

    def _signal(self, history):
        lookback = max(8, int(self.args.v31_factor_state_lookback_weeks))
        minimum = max(8, int(self.args.v31_factor_state_min_weeks))
        window = history.tail(lookback)
        if len(window) < minimum:
            return 0.0, 0.0, 0.0, len(window)
        defensive = self._basket(window, self.DEFENSIVE_RETURNS)
        offensive = self._basket(window, self.OFFENSIVE_RETURNS)
        difference = offensive - defensive
        scale = float(difference.std(ddof=1))
        factor_signal = 0.0
        if math.isfinite(scale) and scale > 1e-8:
            factor_signal = math.tanh(float(difference.mean()) / scale * math.sqrt(len(difference)))
        industry_columns = [column for column in window if str(column).startswith("INDUSTRY:")]
        industry_signal = 0.0
        if industry_columns:
            short = window[industry_columns].tail(4).sum()
            medium = window[industry_columns].tail(min(12, len(window))).sum()
            breadth = 0.5 * float((short > 0).mean()) + 0.5 * float((medium > 0).mean())
            industry_signal = max(-1.0, min(1.0, (breadth - 0.5) * 2.0))
        blend = max(0.0, min(1.0, float(self.args.v31_factor_state_return_weight)))
        combined = blend * factor_signal + (1.0 - blend) * industry_signal
        return combined, factor_signal, industry_signal, len(window)

    def _build_schedule(self):
        if self.pivot.empty:
            return {}
        first = pd.Period(str(self.pivot.index.min())[:7], freq="M")
        # Include the month after the latest factor return so a live decision made
        # before the next risk-model refresh still uses the latest causal state.
        last = pd.Period(str(self.pivot.index.max())[:7], freq="M") + 1
        months = [str(month) for month in pd.period_range(first, last, freq="M")]
        base_weight = float(self.args.v31_offensive_base_weight)
        previous = base_weight
        schedule = {}
        for month in months:
            month_start = month + "-01"
            history = self.pivot.loc[self.pivot.index < month_start]
            combined, factor_signal, industry_signal, observations = self._signal(history)
            desired = base_weight + float(self.args.v31_factor_state_max_tilt) * combined
            desired = max(float(self.args.v31_offensive_min_weight), desired)
            desired = min(float(self.args.v31_offensive_max_weight), desired)
            step = max(0.0, float(self.args.v31_factor_state_max_monthly_step))
            offensive = max(previous - step, min(previous + step, desired))
            schedule[month] = {
                "offensive_weight": offensive,
                "defensive_weight": 1.0 - offensive,
                "factor_state_signal": combined,
                "factor_return_signal": factor_signal,
                "industry_trend_signal": industry_signal,
                "factor_state_observations": observations,
            }
            previous = offensive
        return schedule

    def weights(self, decision_date, baseline):
        month = str(decision_date)[:7]
        metadata = self.schedule.get(month)
        if metadata is None:
            metadata = {
                "offensive_weight": float(self.args.v31_offensive_base_weight),
                "defensive_weight": 1.0 - float(self.args.v31_offensive_base_weight),
                "factor_state_signal": 0.0,
                "factor_return_signal": 0.0,
                "industry_trend_signal": 0.0,
                "factor_state_observations": 0,
            }
        baseline = pd.Series(baseline, dtype=float)
        defensive = baseline.reindex(DEFENSIVE_COMPONENTS).fillna(0.0)
        offensive = baseline.reindex(OFFENSIVE_COMPONENTS).fillna(0.0)
        if defensive.sum() <= 0 or offensive.sum() <= 0:
            return baseline.to_dict(), {**metadata, "factor_state_mode": "fallback_static"}
        defensive = defensive / defensive.sum() * float(metadata["defensive_weight"])
        offensive = offensive / offensive.sum() * float(metadata["offensive_weight"])
        output = pd.concat([defensive, offensive]).groupby(level=0).sum()
        return output.to_dict(), {**metadata, "factor_state_mode": "monthly"}
