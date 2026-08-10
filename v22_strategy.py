"""Structural V2.2 score adjustments for causal industry rotation tests."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import factor_rank_backtest as base


UNKNOWN_INDUSTRY_VALUES = {"", "NAN", "NONE", "NULL", "--", "UNKNOWN"}


def normalize_industries(values: pd.Series) -> pd.Series:
    result = values.fillna("UNKNOWN").astype(str).str.strip()
    return result.mask(result.str.upper().isin(UNKNOWN_INDUSTRY_VALUES), "UNKNOWN")


def apply_structural_components(
    features: pd.DataFrame,
    *,
    defensive_industry_neutral: bool,
    pure_industry_trend: bool,
) -> pd.DataFrame:
    """Rebuild only the explicitly enabled legacy score components."""

    result = features.copy()
    if defensive_industry_neutral:
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

    if pure_industry_trend:
        result["industry_trend_score"] = (
            0.50
            * base.factor_zscore(
                result, "industry_relative_20", industry_neutral=False
            )
            + 0.50
            * base.factor_zscore(
                result, "industry_relative_60", industry_neutral=False
            )
        )
    return result


def _industry_leadership_table(features: pd.DataFrame) -> pd.DataFrame:
    industries = normalize_industries(features["industry_1"])
    frame = pd.DataFrame(
        {
            "industry": industries,
            "relative_20": pd.to_numeric(
                features.get("industry_relative_20"), errors="coerce"
            ),
            "relative_60": pd.to_numeric(
                features.get("industry_relative_60"), errors="coerce"
            ),
        },
        index=features.index,
    )
    frame = frame.loc[frame["industry"] != "UNKNOWN"]
    if frame.empty:
        return pd.DataFrame(
            columns=("relative_20", "relative_60", "leadership_score", "strength")
        )

    grouped = frame.groupby("industry", sort=True)[["relative_20", "relative_60"]].median()
    grouped["z20"] = base.zscore(grouped["relative_20"])
    grouped["z60"] = base.zscore(grouped["relative_60"])
    grouped["leadership_score"] = 0.50 * grouped["z20"] + 0.50 * grouped["z60"]
    grouped["strength"] = (
        0.50 * grouped["z20"].clip(lower=0.0, upper=1.0)
        + 0.50 * grouped["z60"].clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    return grouped


def build_industry_satellite_snapshot(
    features: pd.DataFrame,
    *,
    top_industries: int,
    excluded_industries: Iterable[str] = (),
) -> Dict[str, object]:
    """Freeze the industries and cross-industry scores visible on one decision date."""

    if features.empty:
        return {"leaders": [], "leadership_strength": 0.0, "scores": {}}
    leadership = _industry_leadership_table(features)
    if leadership.empty:
        return {"leaders": [], "leadership_strength": 0.0, "scores": {}}

    excluded = {
        str(industry).strip().upper()
        for industry in excluded_industries
        if str(industry).strip()
    }
    eligible = leadership.loc[
        (leadership["relative_20"] > 0.0)
        & (leadership["relative_60"] > 0.0)
        & (leadership["leadership_score"] > 0.0)
        & ~leadership.index.astype(str).str.upper().isin(excluded)
    ].nlargest(max(1, int(top_industries)), "leadership_score")
    if eligible.empty:
        return {"leaders": [], "leadership_strength": 0.0, "scores": {}}
    return {
        "leaders": eligible.index.astype(str).tolist(),
        "leadership_strength": float(
            np.clip(eligible["strength"].max(), 0.0, 1.0)
        ),
        "scores": {
            str(industry): float(score)
            for industry, score in leadership["leadership_score"].items()
        },
    }


def apply_industry_satellite_snapshot(
    features: pd.DataFrame,
    core_score: pd.Series,
    snapshot: Mapping[str, object],
    *,
    maximum_weight: float,
    market_risk_on_strength: float,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    """Apply a frozen industry snapshot with a current continuous risk throttle."""

    result = features.copy()
    result["v22_industry_satellite_leader"] = False
    result["v22_industry_leadership_score"] = 0.0
    result["v22_industry_satellite_weight"] = 0.0
    result["v22_industry_leadership_strength"] = 0.0
    risk_on = float(np.clip(float(market_risk_on_strength), 0.0, 1.0))
    result["v22_market_risk_on_strength"] = risk_on

    cap = float(np.clip(float(maximum_weight), 0.0, 0.50))
    leader_names = [str(value) for value in snapshot.get("leaders", [])]
    leadership_strength = float(
        np.clip(float(snapshot.get("leadership_strength", 0.0)), 0.0, 1.0)
    )
    score_by_industry = {
        str(industry): float(score)
        for industry, score in dict(snapshot.get("scores", {})).items()
    }
    satellite_weight = cap * leadership_strength * risk_on
    if cap <= 0.0 or result.empty or not leader_names or satellite_weight <= 0.0:
        return core_score, result, {
            "v22_satellite_weight": 0.0,
            "v22_leadership_strength": leadership_strength,
            "v22_market_risk_on_strength": risk_on,
            "v22_leading_industries": "|".join(leader_names),
        }

    industry_by_row = normalize_industries(result["industry_1"])
    leader_mask = industry_by_row.isin(leader_names)
    stock_leadership = industry_by_row.map(score_by_industry).fillna(0.0)
    satellite_signal = stock_leadership.where(leader_mask, 0.0)
    result["v22_industry_satellite_leader"] = leader_mask
    result["v22_industry_leadership_score"] = satellite_signal
    result["v22_industry_satellite_weight"] = satellite_weight
    result["v22_industry_leadership_strength"] = leadership_strength

    if not leader_mask.any() or satellite_signal.std(ddof=1) <= 0.0:
        blended = core_score
    else:
        blended = (
            (1.0 - satellite_weight) * base.zscore(core_score)
            + satellite_weight * base.zscore(satellite_signal)
        )
    return blended, result, {
        "v22_satellite_weight": float(satellite_weight),
        "v22_leadership_strength": leadership_strength,
        "v22_market_risk_on_strength": risk_on,
        "v22_leading_industries": "|".join(leader_names),
    }


class IndustrySatelliteController:
    """Hold a causal industry snapshot for a week or a calendar month."""

    def __init__(self, schedule: str = "weekly"):
        schedule = str(schedule).strip().lower()
        if schedule not in {"weekly", "monthly"}:
            raise ValueError(f"Unsupported industry satellite schedule: {schedule}")
        self.schedule = schedule
        self.signal_period: Optional[str] = None
        self.snapshot: Dict[str, object] = {}
        self.initialized = False

    def blend(
        self,
        features: pd.DataFrame,
        core_score: pd.Series,
        *,
        decision_date: str,
        maximum_weight: float,
        top_industries: int,
        market_risk_on_strength: float,
        excluded_industries: Iterable[str] = (),
    ) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
        current_period = str(decision_date)[:7] if self.schedule == "monthly" else str(decision_date)
        if not self.initialized or current_period != self.signal_period:
            self.snapshot = build_industry_satellite_snapshot(
                features,
                top_industries=top_industries,
                excluded_industries=excluded_industries,
            )
            self.signal_period = current_period
            self.initialized = True
        score, result, metadata = apply_industry_satellite_snapshot(
            features,
            core_score,
            self.snapshot,
            maximum_weight=maximum_weight,
            market_risk_on_strength=market_risk_on_strength,
        )
        metadata.update(
            {
                "v22_satellite_schedule": self.schedule,
                "v22_satellite_signal_period": self.signal_period,
            }
        )
        result["v22_satellite_schedule"] = self.schedule
        result["v22_satellite_signal_period"] = self.signal_period
        return score, result, metadata

    def serialize(self) -> Dict[str, object]:
        return {
            "schedule": self.schedule,
            "signal_period": self.signal_period,
            "snapshot": self.snapshot,
            "initialized": bool(self.initialized),
        }

    def restore(self, payload: Optional[Mapping[str, object]]) -> None:
        if not payload:
            return
        saved_schedule = str(payload.get("schedule", self.schedule)).strip().lower()
        if saved_schedule != self.schedule:
            raise ValueError(
                "Industry satellite checkpoint schedule does not match current settings"
            )
        self.signal_period = payload.get("signal_period")
        self.snapshot = dict(payload.get("snapshot", {}))
        self.initialized = bool(payload.get("initialized", False))


def blend_continuous_industry_satellite(
    features: pd.DataFrame,
    core_score: pd.Series,
    *,
    maximum_weight: float,
    top_industries: int,
    excluded_industries: Iterable[str] = (),
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    """Blend a weekly, unthrottled industry-leadership sleeve into the core score.

    An industry is eligible only when both its 20-day and 60-day relative returns
    are positive. Signal strength is continuous and uses cross-industry z-scores,
    so no return threshold is fitted to the backtest sample.
    """
    snapshot = build_industry_satellite_snapshot(
        features,
        top_industries=top_industries,
        excluded_industries=excluded_industries,
    )
    return apply_industry_satellite_snapshot(
        features,
        core_score,
        snapshot,
        maximum_weight=maximum_weight,
        market_risk_on_strength=1.0,
    )
