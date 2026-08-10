"""Pure-numpy risk overlay for causal V2H portfolio construction."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from risk_model_reporting import (
    build_exposure_matrix,
    complete_exposures_for_codes,
    load_exposure_and_specific,
    load_factor_covariance,
)


@dataclass
class RiskSnapshot:
    model_date: str
    covariance: pd.DataFrame
    exposure: pd.DataFrame


def _read_only_connection(path):
    path = Path(path).resolve()
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


class WeeklyRiskModelStore:
    """Read the latest risk snapshot that existed by a decision date."""

    def __init__(self, database):
        self.database = Path(database).resolve()
        self.conn = _read_only_connection(self.database)
        self.dates = [
            str(row[0])
            for row in self.conn.execute(
                """
                SELECT c.model_date
                FROM (
                    SELECT DISTINCT model_date
                    FROM weekly_factor_covariance
                ) c
                WHERE EXISTS (
                    SELECT 1
                    FROM weekly_specific_risk s
                    WHERE s.model_date=c.model_date
                )
                  AND EXISTS (
                    SELECT 1
                    FROM weekly_exposure e
                    WHERE e.model_date=c.model_date
                )
                ORDER BY c.model_date
                """
            )
        ]
        if not self.dates:
            self.close()
            raise ValueError(
                f"No complete weekly risk snapshots in {self.database}"
            )
        self._cached_date = None
        self._cached_covariance = None
        self._cached_exposure = None

    def model_date_for(self, decision_date):
        index = bisect_right(self.dates, str(decision_date)) - 1
        return self.dates[index] if index >= 0 else None

    def snapshot(self, decision_date, codes, maximum_staleness_days=14):
        model_date = self.model_date_for(decision_date)
        if model_date is None:
            return None
        staleness = (
            date.fromisoformat(str(decision_date))
            - date.fromisoformat(model_date)
        ).days
        if staleness > int(maximum_staleness_days):
            return None
        if model_date != self._cached_date:
            self._cached_covariance = load_factor_covariance(
                self.conn, model_date
            )
            self._cached_exposure = load_exposure_and_specific(
                self.conn, model_date
            )
            self._cached_date = model_date
        completed = complete_exposures_for_codes(
            codes, self._cached_exposure
        )
        return RiskSnapshot(
            model_date=model_date,
            covariance=self._cached_covariance,
            exposure=completed,
        )

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None


class WeeklyAlphaFeatureStore:
    """Read causal earnings-yield and cached residual-momentum cross sections."""

    def __init__(self, database, lookback_weeks=52, skip_weeks=4):
        self.database = Path(database).resolve()
        self.conn = _read_only_connection(self.database)
        self.lookback_weeks = int(lookback_weeks)
        self.skip_weeks = int(skip_weeks)
        table = self.conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='weekly_residual_momentum'
            """
        ).fetchone()
        if table is None:
            self.close()
            raise ValueError(
                "weekly_residual_momentum is missing. Run "
                "build_residual_momentum_cache.py first."
            )
        self.exposure_dates = [
            str(row[0])
            for row in self.conn.execute(
                "SELECT DISTINCT model_date FROM weekly_exposure ORDER BY model_date"
            )
        ]
        self.momentum_dates = [
            str(row[0])
            for row in self.conn.execute(
                """
                SELECT DISTINCT model_date
                FROM weekly_residual_momentum
                WHERE lookback_weeks=? AND skip_weeks=?
                ORDER BY model_date
                """,
                (self.lookback_weeks, self.skip_weeks),
            )
        ]
        if not self.exposure_dates or not self.momentum_dates:
            self.close()
            raise ValueError(
                "The risk database has no usable alpha-factor snapshots for "
                f"lookback={self.lookback_weeks}, skip={self.skip_weeks}."
            )
        self._exposure_cache_date = None
        self._exposure_cache = None
        self._momentum_cache_date = None
        self._momentum_cache = None

    @staticmethod
    def _latest_date(dates, decision_date):
        index = bisect_right(dates, str(decision_date)) - 1
        return dates[index] if index >= 0 else None

    @staticmethod
    def _is_stale(model_date, decision_date, maximum_staleness_days):
        if model_date is None:
            return True
        return (
            date.fromisoformat(str(decision_date))
            - date.fromisoformat(str(model_date))
        ).days > int(maximum_staleness_days)

    def augment(self, features, decision_date, maximum_staleness_days=14):
        result = features.copy()
        if result.empty:
            return result, {
                "alpha_feature_status": "empty_features",
                "alpha_feature_exposure_date": None,
                "alpha_feature_momentum_date": None,
            }
        exposure_date = self._latest_date(self.exposure_dates, decision_date)
        momentum_date = self._latest_date(self.momentum_dates, decision_date)
        if self._is_stale(
            exposure_date, decision_date, maximum_staleness_days
        ) or self._is_stale(
            momentum_date, decision_date, maximum_staleness_days
        ):
            result["risk_earnings_yield_raw"] = np.nan
            result["residual_momentum_raw"] = np.nan
            return result, {
                "alpha_feature_status": "no_causal_snapshot",
                "alpha_feature_exposure_date": exposure_date,
                "alpha_feature_momentum_date": momentum_date,
            }
        if exposure_date != self._exposure_cache_date:
            self._exposure_cache = pd.read_sql_query(
                """
                SELECT code, EARNINGS_YIELD, industry_group
                FROM weekly_exposure
                WHERE model_date=?
                """,
                self.conn,
                params=(exposure_date,),
            )
            self._exposure_cache["code"] = (
                self._exposure_cache["code"].astype(str).str.zfill(6)
            )
            self._exposure_cache_date = exposure_date
        if momentum_date != self._momentum_cache_date:
            self._momentum_cache = pd.read_sql_query(
                """
                SELECT code, residual_momentum, observations
                FROM weekly_residual_momentum
                WHERE model_date=? AND lookback_weeks=? AND skip_weeks=?
                """,
                self.conn,
                params=(
                    momentum_date,
                    self.lookback_weeks,
                    self.skip_weeks,
                ),
            )
            self._momentum_cache["code"] = (
                self._momentum_cache["code"].astype(str).str.zfill(6)
            )
            self._momentum_cache_date = momentum_date

        exposure = self._exposure_cache.rename(
            columns={
                "EARNINGS_YIELD": "risk_earnings_yield_raw",
                "industry_group": "risk_industry_group",
            }
        )
        momentum = self._momentum_cache.rename(
            columns={
                "residual_momentum": "residual_momentum_raw",
                "observations": "residual_momentum_observations",
            }
        )
        result["code"] = result["code"].astype(str).str.zfill(6)
        result = result.merge(exposure, on="code", how="left")
        result = result.merge(momentum, on="code", how="left")
        earnings_coverage = float(
            pd.to_numeric(
                result["risk_earnings_yield_raw"], errors="coerce"
            ).notna().mean()
        )
        momentum_coverage = float(
            pd.to_numeric(
                result["residual_momentum_raw"], errors="coerce"
            ).notna().mean()
        )
        return result, {
            "alpha_feature_status": "applied",
            "alpha_feature_exposure_date": exposure_date,
            "alpha_feature_momentum_date": momentum_date,
            "alpha_feature_earnings_coverage": earnings_coverage,
            "alpha_feature_residual_momentum_coverage": momentum_coverage,
        }

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None


class CausalRiskCalibrationStore:
    """Read an expanding calibration schedule without crossing the decision date."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        frame = pd.read_csv(self.path, dtype={"as_of_date": str})
        required = {"as_of_date", "multiplier"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Risk calibration schedule is missing columns: "
                + ", ".join(missing)
            )
        frame["as_of_date"] = frame["as_of_date"].astype(str)
        frame["multiplier"] = pd.to_numeric(
            frame["multiplier"], errors="coerce"
        )
        frame = frame.loc[
            frame["as_of_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
            & frame["multiplier"].gt(0)
        ].copy()
        if frame.empty:
            raise ValueError(
                f"No usable rows in risk calibration schedule: {self.path}"
            )
        frame = (
            frame.sort_values("as_of_date")
            .drop_duplicates("as_of_date", keep="last")
            .reset_index(drop=True)
        )
        self.dates = frame["as_of_date"].tolist()
        self.rows = frame.set_index("as_of_date").to_dict(orient="index")

    def multiplier_for(self, decision_date, default=1.0):
        index = bisect_right(self.dates, str(decision_date)) - 1
        if index < 0:
            return float(default), {
                "risk_calibration_source": "default_before_schedule",
                "risk_calibration_as_of_date": None,
                "risk_calibration_forecast_weeks": 0,
            }
        as_of_date = self.dates[index]
        row = self.rows[as_of_date]
        return float(row["multiplier"]), {
            "risk_calibration_source": "causal_expanding_schedule",
            "risk_calibration_as_of_date": as_of_date,
            "risk_calibration_forecast_weeks": int(
                float(row.get("forecast_weeks", 0) or 0)
            ),
        }


def stock_covariance(snapshot, calibration_multiplier=1.0):
    factor_names = list(snapshot.covariance.index.astype(str))
    matrix = build_exposure_matrix(snapshot.exposure, factor_names)
    factor_covariance = snapshot.covariance.to_numpy(dtype=float)
    specific = pd.to_numeric(
        snapshot.exposure["specific_variance"], errors="coerce"
    ).to_numpy(dtype=float)
    covariance = matrix @ factor_covariance @ matrix.T
    covariance += np.diag(specific)
    multiplier = max(float(calibration_multiplier), 0.0)
    covariance *= multiplier * multiplier
    covariance = (covariance + covariance.T) / 2.0
    return covariance


def _allocate_with_caps(raw, groups, target_sum, stock_cap, group_cap):
    raw = pd.Series(raw, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    raw = raw.clip(lower=0.0)
    groups = pd.Series(groups, index=raw.index).fillna("UNKNOWN").astype(str)
    target_sum = max(float(target_sum), 0.0)
    stock_cap = max(float(stock_cap), 0.0)
    group_cap = max(float(group_cap), 0.0)
    if raw.empty or target_sum <= 0 or stock_cap <= 0 or group_cap <= 0:
        return pd.Series(0.0, index=raw.index)
    if raw.sum() <= 0:
        raw[:] = 1.0

    weights = raw.clip(upper=stock_cap)
    for _, members in groups.groupby(groups).groups.items():
        members = list(members)
        total = float(weights.loc[members].sum())
        if total > group_cap and total > 0:
            weights.loc[members] *= group_cap / total
    if weights.sum() > target_sum:
        weights *= target_sum / float(weights.sum())

    for _ in range(100):
        deficit = target_sum - float(weights.sum())
        if deficit <= 1e-10:
            break
        group_totals = weights.groupby(groups).sum()
        stock_capacity = (stock_cap - weights).clip(lower=0.0)
        group_capacity = groups.map(group_cap - group_totals).clip(lower=0.0)
        eligible = (stock_capacity > 1e-12) & (group_capacity > 1e-12)
        if not eligible.any():
            break
        preference = raw.where(eligible, 0.0)
        if preference.sum() <= 0:
            preference = stock_capacity.where(eligible, 0.0)
        extra = preference / float(preference.sum()) * deficit
        extra = np.minimum(extra, stock_capacity)
        for _, members in groups.groupby(groups).groups.items():
            members = list(members)
            allowed = max(
                0.0, group_cap - float(weights.loc[members].sum())
            )
            amount = float(extra.loc[members].sum())
            if amount > allowed and amount > 0:
                extra.loc[members] *= allowed / amount
        added = float(extra.sum())
        weights += extra
        if added <= 1e-12:
            break
    return weights.clip(lower=0.0)


def _objective(weights, baseline, covariance, strength, target_variance):
    difference = weights - baseline
    tracking_scale = max(float(np.dot(baseline, baseline)), 1e-12)
    tracking = float(np.dot(difference, difference)) / tracking_scale
    variance = float(weights @ covariance @ weights)
    risk = variance / max(float(target_variance), 1e-12)
    return (1.0 - strength) * tracking + strength * risk


def optimize_risk_aware_weights(
    baseline_weights,
    snapshot,
    stock_cap,
    maximum_industry_fraction=0.25,
    strength=0.25,
    target_volatility=0.20,
    calibration_multiplier=1.0,
    minimum_equity_scale=0.80,
    iterations=80,
):
    baseline = pd.Series(baseline_weights, dtype=float)
    baseline.index = baseline.index.astype(str).str.zfill(6)
    baseline = baseline.groupby(level=0).sum().clip(lower=0.0)
    if baseline.empty or baseline.sum() <= 0:
        return baseline, {
            "risk_overlay_status": "empty_target",
            "risk_model_date": snapshot.model_date,
        }
    exposure = snapshot.exposure.set_index("code").reindex(baseline.index)
    groups = exposure["industry_group"].fillna("UNKNOWN").astype(str)
    covariance = stock_covariance(
        RiskSnapshot(
            model_date=snapshot.model_date,
            covariance=snapshot.covariance,
            exposure=exposure.reset_index(),
        ),
        calibration_multiplier=calibration_multiplier,
    )
    baseline_values = baseline.to_numpy(dtype=float).copy()
    equity_target = float(baseline.sum())
    group_cap = equity_target * float(maximum_industry_fraction)
    strength = min(max(float(strength), 0.0), 1.0)
    target_variance = max(float(target_volatility), 1e-6) ** 2
    weights = _allocate_with_caps(
        baseline,
        groups,
        equity_target,
        stock_cap,
        group_cap,
    )
    values = weights.to_numpy(dtype=float).copy()

    if strength > 0:
        for _ in range(max(1, int(iterations))):
            difference = values - baseline_values
            tracking_scale = max(
                float(np.dot(baseline_values, baseline_values)), 1e-12
            )
            gradient = (
                2.0
                * (1.0 - strength)
                * difference
                / tracking_scale
                + 2.0
                * strength
                * (covariance @ values)
                / target_variance
            )
            current = _objective(
                values,
                baseline_values,
                covariance,
                strength,
                target_variance,
            )
            accepted = False
            step = 0.10
            for _ in range(12):
                proposal = pd.Series(
                    values - step * gradient, index=baseline.index
                )
                candidate = _allocate_with_caps(
                    proposal,
                    groups,
                    equity_target,
                    stock_cap,
                    group_cap,
                ).to_numpy(dtype=float).copy()
                candidate_objective = _objective(
                    candidate,
                    baseline_values,
                    covariance,
                    strength,
                    target_variance,
                )
                if candidate_objective <= current + 1e-12:
                    accepted = True
                    break
                step *= 0.5
            if not accepted or np.abs(candidate - values).sum() < 1e-8:
                break
            values = candidate

    pre_variance = max(
        float(baseline_values @ covariance @ baseline_values), 0.0
    )
    optimized_variance = max(float(values @ covariance @ values), 0.0)
    optimized_volatility = math.sqrt(optimized_variance)
    volatility_scale = 1.0
    if (
        target_volatility > 0
        and optimized_volatility > float(target_volatility)
    ):
        volatility_scale = max(
            min(
                float(target_volatility) / optimized_volatility,
                1.0,
            ),
            min(max(float(minimum_equity_scale), 0.0), 1.0),
        )
        values *= volatility_scale
    post_variance = max(float(values @ covariance @ values), 0.0)
    output = pd.Series(values, index=baseline.index).clip(lower=0.0)
    exact_coverage = float(
        baseline.loc[
            exposure["exposure_source"].eq("model").to_numpy()
        ].sum()
        / equity_target
    )
    return output, {
        "risk_overlay_status": "applied",
        "risk_model_date": snapshot.model_date,
        "risk_model_exact_weight_coverage": exact_coverage,
        "risk_calibration_multiplier": float(calibration_multiplier),
        "risk_overlay_strength": strength,
        "risk_predicted_volatility_before": math.sqrt(pre_variance),
        "risk_predicted_volatility_optimized": optimized_volatility,
        "risk_predicted_volatility_after": math.sqrt(post_variance),
        "risk_target_portfolio_volatility": float(target_volatility),
        "risk_volatility_scale": volatility_scale,
        "risk_cap_met": bool(
            math.sqrt(post_variance) <= float(target_volatility) + 1e-10
        ),
        "risk_target_weight_sum_before": equity_target,
        "risk_target_weight_sum_after": float(output.sum()),
        "risk_model_industry_cap": group_cap,
    }


def apply_store_overlay(
    store,
    decision_date,
    baseline_weights,
    stock_cap,
    maximum_industry_fraction,
    strength,
    target_volatility,
    calibration_multiplier,
    minimum_equity_scale,
    maximum_staleness_days,
    iterations,
):
    snapshot = store.snapshot(
        decision_date,
        baseline_weights.index,
        maximum_staleness_days=maximum_staleness_days,
    )
    if snapshot is None:
        return baseline_weights, {
            "risk_overlay_status": "no_causal_snapshot",
            "risk_model_date": None,
        }
    return optimize_risk_aware_weights(
        baseline_weights,
        snapshot,
        stock_cap=stock_cap,
        maximum_industry_fraction=maximum_industry_fraction,
        strength=strength,
        target_volatility=target_volatility,
        calibration_multiplier=calibration_multiplier,
        minimum_equity_scale=minimum_equity_scale,
        iterations=iterations,
    )
