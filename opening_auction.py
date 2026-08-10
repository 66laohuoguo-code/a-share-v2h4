"""Causal opening-auction price ranges and limit-order recommendations."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ashare_utils import round_price_to_tick


@dataclass(frozen=True)
class OpeningGapEstimate:
    observations: int
    expected_gap: float
    lower_gap: float
    upper_gap: float
    fill_probability: float
    source: str


class CausalOpeningGapEstimator:
    """Estimate a stock's next-session opening-gap interval from prior gaps only."""

    def __init__(
        self,
        lookback_days: int = 252,
        min_observations: int = 60,
        fill_probability: float = 0.90,
        shrinkage_observations: float = 40.0,
        market_lookback_days: int = 60,
        max_absolute_gap: float = 0.35,
    ) -> None:
        if lookback_days < 20:
            raise ValueError("Opening-gap lookback must be at least 20 trading days.")
        if min_observations < 10:
            raise ValueError("Opening-gap minimum observations must be at least 10.")
        if not 0.50 < fill_probability < 1.0:
            raise ValueError("Opening-auction fill probability must be between 0.50 and 1.0.")
        self.lookback_days = int(lookback_days)
        self.min_observations = int(min_observations)
        self.fill_probability = float(fill_probability)
        self.shrinkage_observations = max(0.0, float(shrinkage_observations))
        self.max_absolute_gap = max(0.05, float(max_absolute_gap))
        self.stock_gaps: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.lookback_days)
        )
        self.market_lower: Deque[float] = deque(maxlen=max(20, int(market_lookback_days)))
        self.market_median: Deque[float] = deque(maxlen=max(20, int(market_lookback_days)))
        self.market_upper: Deque[float] = deque(maxlen=max(20, int(market_lookback_days)))

    @property
    def lower_quantile(self) -> float:
        return 1.0 - self.fill_probability

    def update(self, rows: Mapping[str, Mapping[str, object]]) -> None:
        daily_gaps = []
        for code, row in rows.items():
            try:
                open_price = float(row.get("open"))
                previous_close = float(row.get("prev_close"))
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(open_price)
                or not math.isfinite(previous_close)
                or open_price <= 0
                or previous_close <= 0
            ):
                continue
            gap = open_price / previous_close - 1.0
            if not math.isfinite(gap) or abs(gap) > self.max_absolute_gap:
                continue
            self.stock_gaps[str(code).zfill(6)].append(float(gap))
            daily_gaps.append(float(gap))
        if daily_gaps:
            values = np.asarray(daily_gaps, dtype=float)
            self.market_lower.append(float(np.quantile(values, self.lower_quantile)))
            self.market_median.append(float(np.quantile(values, 0.50)))
            self.market_upper.append(float(np.quantile(values, self.fill_probability)))

    def seed_from_frame(self, frame: pd.DataFrame, through_date: Optional[str] = None) -> None:
        if frame.empty:
            return
        required = {"code", "trade_date", "prev_close", "open"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Opening-gap history is missing columns: " + ", ".join(missing)
            )
        history = frame.loc[:, ["code", "trade_date", "prev_close", "open"]].copy()
        if through_date is not None:
            history = history.loc[history["trade_date"].astype(str) <= str(through_date)]
        if history.empty:
            return
        history["code"] = history["code"].astype(str).str.zfill(6)
        history["trade_date"] = history["trade_date"].astype(str)
        previous_close = pd.to_numeric(history["prev_close"], errors="coerce")
        open_price = pd.to_numeric(history["open"], errors="coerce")
        history["gap"] = open_price / previous_close - 1.0
        history = history.loc[
            previous_close.gt(0)
            & open_price.gt(0)
            & history["gap"].replace([np.inf, -np.inf], np.nan).notna()
            & history["gap"].abs().le(self.max_absolute_gap)
        ]
        if history.empty:
            return

        stock_tail = (
            history.sort_values(["code", "trade_date"])
            .groupby("code", sort=False)
            .tail(self.lookback_days)
        )
        for code, values in stock_tail.groupby("code", sort=False)["gap"]:
            self.stock_gaps[str(code)].extend(
                float(value) for value in values.to_numpy(dtype=float)
            )

        market_dates = sorted(history["trade_date"].unique())[
            -int(self.market_median.maxlen or 60) :
        ]
        market_history = history.loc[history["trade_date"].isin(market_dates)]
        for _, values in market_history.groupby("trade_date", sort=True)["gap"]:
            array = values.to_numpy(dtype=float)
            self.market_lower.append(
                float(np.quantile(array, self.lower_quantile))
            )
            self.market_median.append(float(np.quantile(array, 0.50)))
            self.market_upper.append(
                float(np.quantile(array, self.fill_probability))
            )

    def estimate(self, code: str) -> OpeningGapEstimate:
        code = str(code).zfill(6)
        stock_values = np.asarray(self.stock_gaps.get(code, ()), dtype=float)
        observations = int(stock_values.size)

        market_lower = self._market_value(self.market_lower, -0.015)
        market_median = self._market_value(self.market_median, 0.0)
        market_upper = self._market_value(self.market_upper, 0.015)
        if observations == 0:
            return OpeningGapEstimate(
                observations=0,
                expected_gap=market_median,
                lower_gap=market_lower,
                upper_gap=market_upper,
                fill_probability=self.fill_probability,
                source="market_fallback",
            )

        stock_lower = float(np.quantile(stock_values, self.lower_quantile))
        stock_median = float(np.quantile(stock_values, 0.50))
        stock_upper = float(np.quantile(stock_values, self.fill_probability))
        reliability = observations / (
            observations + self.shrinkage_observations
        )
        if observations < self.min_observations:
            reliability *= observations / self.min_observations
        reliability = min(1.0, max(0.0, reliability))

        lower = reliability * stock_lower + (1.0 - reliability) * market_lower
        median = reliability * stock_median + (1.0 - reliability) * market_median
        upper = reliability * stock_upper + (1.0 - reliability) * market_upper
        lower, median, upper = sorted((float(lower), float(median), float(upper)))
        source = "stock_market_shrunk" if reliability < 0.999 else "stock_history"
        return OpeningGapEstimate(
            observations=observations,
            expected_gap=median,
            lower_gap=lower,
            upper_gap=upper,
            fill_probability=self.fill_probability,
            source=source,
        )

    @staticmethod
    def _market_value(values: Deque[float], fallback: float) -> float:
        if not values:
            return float(fallback)
        return float(np.median(np.asarray(values, dtype=float)))


def expected_open_price(reference_close: float, estimate: OpeningGapEstimate) -> float:
    return max(0.01, float(reference_close) * (1.0 + float(estimate.expected_gap)))


def opening_auction_limit_price(
    reference_close: float,
    side: str,
    estimate: OpeningGapEstimate,
    buffer_bps: float = 2.0,
) -> float:
    side = str(side).upper()
    buffer_rate = max(0.0, float(buffer_bps)) / 10000.0
    if side == "BUY":
        gap = float(estimate.upper_gap) + buffer_rate
    elif side == "SELL":
        gap = float(estimate.lower_gap) - buffer_rate
    else:
        raise ValueError(f"Unsupported auction order side: {side}")
    raw_price = max(0.01, float(reference_close) * (1.0 + gap))
    return round_price_to_tick(raw_price, side)


def opening_auction_order_is_marketable(
    open_price: float,
    limit_price: float,
    side: str,
) -> bool:
    side = str(side).upper()
    tolerance = 1e-9
    if side == "BUY":
        return float(open_price) <= float(limit_price) + tolerance
    if side == "SELL":
        return float(open_price) + tolerance >= float(limit_price)
    raise ValueError(f"Unsupported auction order side: {side}")
