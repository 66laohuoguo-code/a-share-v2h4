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
import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import factor_rank_backtest as base
from ashare_utils import (
    buy_order_size_rules,
    mandatory_trade_cost,
    round_target_shares_for_code,
    should_rebalance_on_date,
    trading_cost_snapshot,
    write_excel_workbook,
)


DEFAULT_DATABASE = Path("data/processed/stock_daily.sqlite")
DEFAULT_OUTPUT_DIR = Path("outputs/backtest_v2")

# The original low-beta/industry mix, re-normalized so the weights sum to one.
STATIC_COMPONENT_WEIGHTS: Dict[str, float] = {
    "low_beta_score": 0.25,
    "low_volatility_score": 0.21,
    "low_turnover_score": 0.17,
    "reversal_score": 0.15,
    "lower_drawdown_score": 0.10,
    "industry_trend_score": 0.12,
}
EVENT_COMPONENT = "industry_event_score_ranked"


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
            self.ic_rows.append(row)
        self.pending = survivors

    def weights(self, decision_date: str) -> Tuple[Dict[str, float], Dict[str, object]]:
        static = pd.Series({name: STATIC_COMPONENT_WEIGHTS.get(name, 0.0) for name in self.component_names}, dtype=float)
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


def apply_v2_score(features: pd.DataFrame, weights: Mapping[str, float], args) -> pd.DataFrame:
    result = features.copy()
    score = pd.Series(0.0, index=result.index)
    for name, weight in weights.items():
        if name in result.columns:
            score = score + float(weight) * safe_series(result[name], result.index).fillna(0.0)
    if bool(args.enable_event_score) and EVENT_COMPONENT in result.columns:
        event = safe_series(result[EVENT_COMPONENT], result.index).fillna(0.0)
        # Event signal is deliberately capped and opt-in because the user's first
        # A/B test did not find a stable gain from direct event-score injection.
        score = score + clip(float(args.event_score_weight), 0.0, float(args.event_score_weight_cap)) * event
    result["score_v2"] = score
    return result.sort_values("score_v2", ascending=False).reset_index(drop=True)


def select_codes(features: pd.DataFrame, holdings: Mapping[str, int], args, target_equity_weight: float) -> List[str]:
    ranked = features.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    rank_by_code = ranked.set_index("code")["rank"].to_dict()
    industry_by_code = ranked.set_index("code")["industry_1"].fillna("UNKNOWN").astype(str).to_dict()

    needed_for_cap = int(math.ceil(target_equity_weight / max(float(args.max_stock_weight), 1e-6)))
    desired_count = max(int(args.target_count), needed_for_cap, int(args.min_target_count))
    desired_count = min(desired_count, len(ranked))
    max_industry_count = max(1, int(math.floor(desired_count * float(args.max_industry_weight) + 1e-9)))
    industry_counts: Dict[str, int] = {}
    selected: List[str] = []

    def add(code: str) -> bool:
        if code in selected:
            return False
        industry = industry_by_code.get(code, "UNKNOWN")
        if industry_counts.get(industry, 0) >= max_industry_count:
            return False
        selected.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return True

    kept = [
        code
        for code, shares in holdings.items()
        if int(shares) > 0 and rank_by_code.get(code, math.inf) <= int(args.sell_rank)
    ]
    for code in sorted(kept, key=lambda item: rank_by_code.get(item, math.inf)):
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


def cap_and_redistribute(
    raw: pd.Series,
    industries: pd.Series,
    target_equity_weight: float,
    max_stock_weight: float,
    max_industry_weight: float,
) -> pd.Series:
    """Allocate all feasible equity budget while satisfying stock/industry caps."""
    index = raw.index
    if len(index) == 0 or target_equity_weight <= 0:
        return pd.Series(0.0, index=index)
    raw = safe_series(raw, index).fillna(0.0).clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=index)
    weights = raw / raw.sum() * target_equity_weight
    industries = industries.reindex(index).fillna("UNKNOWN").astype(str)

    for _ in range(100):
        previous = weights.copy()
        weights = weights.clip(upper=max_stock_weight)
        for industry, members in industries.groupby(industries).groups.items():
            member_index = list(members)
            total = float(weights.loc[member_index].sum())
            if total > max_industry_weight + 1e-12:
                weights.loc[member_index] *= max_industry_weight / total

        deficit = target_equity_weight - float(weights.sum())
        if deficit <= 1e-8:
            break
        industry_total = weights.groupby(industries).sum()
        capacity = (max_stock_weight - weights).clip(lower=0.0)
        industry_capacity = industries.map(max_industry_weight - industry_total).clip(lower=0.0)
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
            allowed = max(0.0, max_industry_weight - float(weights.loc[member_index].sum()))
            amount = float(extra.loc[member_index].sum())
            if amount > allowed + 1e-12 and amount > 0:
                extra.loc[member_index] *= allowed / amount
        weights += extra
        if float((weights - previous).abs().sum()) < 1e-9:
            break
    return weights.clip(lower=0.0)


def build_targets_v2(
    features: pd.DataFrame,
    holdings: Mapping[str, int],
    current_weights: Mapping[str, float],
    args,
    target_equity_weight: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    if features.empty or target_equity_weight <= 0:
        return {}, {"selected_count": 0, "target_weight_sum": 0.0}
    selected = select_codes(features, holdings, args, target_equity_weight)
    if not selected:
        return {}, {"selected_count": 0, "target_weight_sum": 0.0}

    frame = features.set_index("code").loc[selected].copy()
    score = safe_series(frame["score_v2"], frame.index).fillna(0.0)
    centered = (score - score.max()) / max(float(args.score_temperature), 1e-6)
    score_weight = np.exp(centered.clip(-30, 30))
    volatility = safe_series(frame.get("volatility_120", pd.Series(np.nan, index=frame.index)), frame.index)
    fallback_vol = volatility[volatility > 0].median()
    fallback_vol = fallback_vol if pd.notna(fallback_vol) and fallback_vol > 0 else 0.30
    risk_scale = volatility.fillna(fallback_vol).clip(lower=float(args.min_stock_volatility))
    raw = score_weight / (risk_scale ** float(args.inverse_vol_power))
    desired = cap_and_redistribute(
        raw,
        frame["industry_1"],
        target_equity_weight,
        float(args.max_stock_weight),
        float(args.max_industry_weight),
    )

    # Avoid spending money on trivial changes; keep target allocations otherwise.
    target = desired.to_dict()
    band = max(0.0, float(args.rebalance_band_weight))
    for code in list(set(target).union(current_weights)):
        desired_weight = float(target.get(code, 0.0))
        current_weight = float(current_weights.get(code, 0.0))
        if abs(desired_weight - current_weight) < band:
            target[code] = current_weight
    # Preserve the desired target when the band leaves a small residual; the execution
    # layer still controls cash, lots and liquidity.
    target = {code: float(weight) for code, weight in target.items() if float(weight) > 0}
    return target, {
        "selected_count": int(len(selected)),
        "target_weight_sum": float(sum(target.values())),
        "desired_equity_weight": float(target_equity_weight),
    }


def execution_price(open_price: float, side: str, slippage_bps: float) -> float:
    slip = max(0.0, float(slippage_bps)) / 10000.0
    return open_price * (1.0 + slip) if side == "BUY" else open_price * (1.0 - slip)


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
) -> Tuple[float, List[Dict[str, object]]]:
    planned: List[Dict[str, object]] = []
    min_trade_threshold = max(
        float(args.min_trade_value),
        float(portfolio_open_value) * max(0.0, float(getattr(args, "min_trade_weight", 0.0))),
    )

    for code in sorted(set(holdings).union(targets)):
        row = prices.get(code)
        if row is None:
            continue
        raw_open = safe_float(row.get("open"), np.nan)
        if not math.isfinite(raw_open) or raw_open <= 0:
            continue
        current_shares = int(holdings.get(code, 0))
        target_weight = max(0.0, float(targets.get(code, 0.0)))
        side = "BUY" if target_weight * portfolio_open_value > current_shares * raw_open else "SELL"
        if base.blocked_by_price_limit(code, row, side, args):
            continue
        price = execution_price(raw_open, side, float(args.slippage_bps))
        target_value = target_weight * portfolio_open_value
        target_shares = round_target_shares_for_code(target_value, price, code) if target_weight > 0 else 0
        trade_shares = int(target_shares - current_shares)
        if trade_shares == 0:
            continue
        side = "BUY" if trade_shares > 0 else "SELL"
        gross = abs(trade_shares) * price
        if gross < min_trade_threshold:
            continue
        # Do not pretend a backtest can trade a large fraction of a stock's daily volume.
        avg_amount = safe_float(liquidity_by_code.get(code), np.nan)
        if math.isfinite(avg_amount) and avg_amount > 0 and float(args.max_participation_rate) > 0:
            max_gross = avg_amount * float(args.max_participation_rate)
            minimum, increment = buy_order_size_rules(code)
            raw_max_shares = int(math.floor(max_gross / price))
            if raw_max_shares < minimum:
                continue
            max_shares = int(minimum + math.floor((raw_max_shares - minimum) / increment) * increment)
            trade_shares = int(math.copysign(min(abs(trade_shares), max_shares), trade_shares))
            gross = abs(trade_shares) * price
        planned.append(
            {
                "trade_date": trade_date,
                "decision_date": decision_date,
                "code": code,
                "name": str(row.get("name", "")),
                "side": side,
                "shares": abs(int(trade_shares)),
                "open_price": raw_open,
                "price": price,
                "slippage_bps": float(args.slippage_bps),
                "gross_amount": gross,
                "target_weight": target_weight,
                "avg_amount_for_cap": avg_amount,
            }
        )

    executed: List[Dict[str, object]] = []
    for order in [item for item in planned if item["side"] == "SELL"]:
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
        if gross < min_trade_threshold:
            continue
        fee = mandatory_trade_cost(gross, "SELL", trade_date, code)
        holdings[code] = int(holdings.get(code, 0)) - shares
        if holdings[code] <= 0:
            holdings.pop(code, None)
        cash += gross - fee
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)

    buys = sorted((item for item in planned if item["side"] == "BUY"), key=lambda item: item["target_weight"], reverse=True)
    for order in buys:
        code = str(order["code"])
        minimum, increment = buy_order_size_rules(code)
        shares = int(order["shares"])
        if shares < minimum:
            continue
        shares = int(minimum + math.floor((shares - minimum) / increment) * increment)
        while shares >= minimum:
            gross = shares * float(order["price"])
            fee = mandatory_trade_cost(gross, "BUY", trade_date, code)
            if gross + fee <= cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            continue
        gross = shares * float(order["price"])
        if gross < min_trade_threshold:
            continue
        fee = mandatory_trade_cost(gross, "BUY", trade_date, code)
        cash -= gross + fee
        holdings[code] = int(holdings.get(code, 0)) + shares
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)
    return float(cash), executed


def make_summary(equity: pd.DataFrame, trades: pd.DataFrame, initial_cash: float, final_value: float, args) -> Dict[str, object]:
    daily = safe_series(equity["daily_return"]).dropna()
    annual_return = (final_value / initial_cash) ** (244 / max(len(equity), 1)) - 1.0
    annual_vol = daily.std(ddof=1) * math.sqrt(244) if len(daily) > 1 else np.nan
    drawdown = equity["total_value"] / equity["total_value"].cummax() - 1.0
    return {
        "strategy": "factor_rank_v2_continuous_risk",
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
        "min_equity_weight": float(args.min_equity_weight),
        "target_count": int(args.target_count),
        "max_stock_weight": float(args.max_stock_weight),
        "max_industry_weight": float(args.max_industry_weight),
        "dynamic_factor_weights": bool(args.dynamic_factor_weights),
        "slippage_bps": float(args.slippage_bps),
        "max_participation_rate": float(args.max_participation_rate),
        "rebalance_band_weight": float(args.rebalance_band_weight),
        "min_trade_value": float(args.min_trade_value),
        "min_trade_weight": float(getattr(args, "min_trade_weight", 0.0)),
        "cost_model": "date-aware statutory A-share costs + configurable one-sided slippage + participation cap",
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
    payload["trading_costs"] = trading_cost_snapshot(summary.get("end_date"))
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


def run_backtest(args):
    conn = sqlite3.connect(args.database)
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
        market_state = base.build_market_state(prices, args)
        all_prices_by_date = {
            date: group.set_index("code").to_dict("index")
            for date, group in prices.groupby("trade_date")
        }
        financial = base.load_financial_factors(conn)
        industry_events = base.load_industry_event_scores(args.industry_event_scores)
        event_regime = base.load_event_regime_signals(args.event_regime_signals)

        components = list(STATIC_COMPONENT_WEIGHTS)
        if bool(args.dynamic_include_event_component):
            components.append(EVENT_COMPONENT)
        weighter = RollingICWeighter(args, components)

        holdings: Dict[str, int] = {}
        cash = float(args.initial_cash)
        last_close: Dict[str, float] = {}
        previous_total = float(args.initial_cash)
        peak_total = float(args.initial_cash)
        equity_rows: List[Dict[str, object]] = []
        trade_rows: List[Dict[str, object]] = []

        for offset, trade_date in enumerate(test_dates):
            decision_date = dates[date_to_index[trade_date] - 1]
            today_prices = all_prices_by_date.get(trade_date, {})
            corporate_action_cash, corporate_actions = base.apply_corporate_actions_before_open(
                holdings, today_prices, last_close
            )
            open_total = base.value_portfolio(holdings, cash, today_prices, "open", last_close)
            current_weights = current_weights_from_open(holdings, today_prices, open_total, last_close)
            if bool(args.dynamic_factor_weights):
                weighter.resolve(decision_date, all_prices_by_date)

            regime = continuous_target_equity(
                market_state, decision_date, previous_total, peak_total, args, event_regime
            )
            scheduled = should_rebalance_on_date(
                args.rebalance_schedule, offset, args.rebalance_every_n_days, dates, date_to_index, decision_date
            )
            current_equity = sum(current_weights.values())
            risk_rebalance = current_equity > float(regime["target_equity_weight"]) + float(args.risk_rebalance_band)
            rebalanced = bool(scheduled or risk_rebalance)
            executed: List[Dict[str, object]] = []
            target_meta: Dict[str, object] = {"selected_count": 0, "target_weight_sum": 0.0}
            weight_info: Dict[str, object] = {"weight_mode": "not_rebalanced", "ic_observations": len(weighter.ic_rows)}

            if rebalanced:
                features = base.feature_snapshot(prices, financial, decision_date, args, industry_events)
                factor_weights, weight_info = weighter.weights(decision_date)
                features = apply_v2_score(features, factor_weights, args)
                targets, target_meta = build_targets_v2(
                    features,
                    holdings,
                    current_weights,
                    args,
                    float(regime["target_equity_weight"]),
                )
                liquidity_by_code = features.set_index("code")["avg_amount_60"].to_dict() if not features.empty else {}
                cash, executed = execute_trades_v2(
                    trade_date,
                    decision_date,
                    holdings,
                    cash,
                    targets,
                    today_prices,
                    open_total,
                    liquidity_by_code,
                    args,
                )
                trade_rows.extend(executed)

                # The label becomes eligible only after its own exit date has passed.
                decision_index = date_to_index[decision_date]
                entry_index = decision_index + 1
                exit_index = decision_index + int(args.forward_label_days)
                if bool(args.dynamic_factor_weights) and entry_index < len(dates) and exit_index < len(dates):
                    weighter.add_snapshot(decision_date, dates[entry_index], dates[exit_index], features)

            cash += corporate_action_cash
            close_total = base.value_portfolio(holdings, cash, today_prices, "close", last_close)
            for code, row in today_prices.items():
                close = safe_float(row.get("close"), np.nan)
                if math.isfinite(close) and close > 0:
                    last_close[code] = close
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
                }
            )
            previous_total = close_total
            peak_total = max(peak_total, close_total)
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
        print(f"V2 workbook: {paths['workbook']}")
        print(f"Summary JSON: {paths['summary']}")
        print(f"Final value: {summary['final_value']:.2f}")
        print(f"Total return: {summary['total_return']:.2%}")
        print(f"Average actual equity: {summary['average_actual_equity_weight']:.2%}")
        return summary, paths
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="V2 causal weekly A-share factor backtest.")
    parser.add_argument("--strategy-config", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-every-n-days", type=int, default=1)
    parser.add_argument("--rebalance-schedule", choices=["every_n_days", "week_end", "month_end"], default="week_end")

    # Universe / portfolio.
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

    # Data / base features inherited from V1.
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--feature-history-days", type=int, default=320)
    parser.add_argument("--min-avg-amount", type=float, default=50_000_000.0)
    parser.add_argument("--min-market-cap-quantile", type=float, default=0.30)
    parser.add_argument("--market-cap-proxy-window", type=int, default=20)
    parser.add_argument("--disable-industry-neutral-factors", action="store_true")

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
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-participation-rate", type=float, default=0.05)
    parser.add_argument("--disable-limit-trade-filter", action="store_true")
    parser.add_argument("--limit-trade-buffer", type=float, default=0.005)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.strategy_config:
        config_path = Path(preliminary.strategy_config)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(payload) - valid_destinations - {"strategy_name"})
        if unknown:
            raise ValueError(f"Unknown V2H strategy config keys: {', '.join(unknown)}")
        parser.set_defaults(**payload)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_backtest(parse_args())
