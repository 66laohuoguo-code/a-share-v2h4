"""Cost-aware discrete portfolio helpers for small A-share accounts."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ashare_utils import (
    buy_order_size_rules,
    mandatory_trade_cost,
    round_portfolio_target_shares,
)


def safe_float(value, default=np.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def commission_efficient_trade_floor(
    minimum_commission: float,
    maximum_commission_fraction: float,
) -> float:
    """Minimum order value needed to keep the fixed commission under a limit."""
    minimum = max(0.0, float(minimum_commission))
    burden = max(0.0, float(maximum_commission_fraction))
    if minimum <= 0 or burden <= 0:
        return 0.0
    return minimum / burden


def round_trip_cost_fraction(
    sell_code: str,
    buy_code: str,
    position_value: float,
    trade_date: str,
    broker_commission_rate: float,
    broker_minimum_commission: float,
) -> float:
    value = max(0.0, float(position_value))
    if value <= 0:
        return math.inf
    sell_cost = mandatory_trade_cost(
        value,
        "SELL",
        trade_date,
        sell_code,
        broker_commission_rate=broker_commission_rate,
        broker_minimum_commission=broker_minimum_commission,
    )
    buy_cost = mandatory_trade_cost(
        value,
        "BUY",
        trade_date,
        buy_code,
        broker_commission_rate=broker_commission_rate,
        broker_minimum_commission=broker_minimum_commission,
    )
    return float((sell_cost + buy_cost) / value)


def select_cost_aware_codes(
    features: pd.DataFrame,
    holdings: Mapping[str, int],
    target_equity_weight: float,
    portfolio_value: float,
    target_count: int,
    minimum_holdings: int,
    buy_rank: int,
    sell_rank: int,
    maximum_industry_weight: float,
    slippage_bps: float,
    decision_date: str,
    expected_return_per_score: float,
    hurdle_buffer_bps: float,
    broker_commission_rate: float,
    broker_minimum_commission: float,
    replacement_policy: str = "cost_aware",
    industry_caps: Mapping[str, float] | None = None,
) -> Tuple[Sequence[str], pd.Series, Dict[str, object]]:
    """Keep incumbents unless a challenger clears an all-in economic hurdle."""
    ranked = features.copy().reset_index(drop=True)
    if ranked.empty:
        return [], pd.Series(dtype=float), {"cost_aware": True, "affordable_count": 0}
    ranked["code"] = ranked["code"].astype(str).str.zfill(6)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["score_v2"] = pd.to_numeric(ranked["score_v2"], errors="coerce").fillna(0.0)
    row_by_code = ranked.set_index("code").to_dict("index")
    rank_by_code = ranked.set_index("code")["rank"].to_dict()
    score_by_code = ranked.set_index("code")["score_v2"].to_dict()

    value = max(0.0, float(portfolio_value))
    equity_budget = value * max(0.0, float(target_equity_weight))
    requested_count = min(max(1, int(target_count)), len(ranked))
    required_count = min(requested_count, max(1, int(minimum_holdings)))
    maximum_lot_budget = equity_budget / required_count if required_count else equity_budget
    slippage = max(0.0, float(slippage_bps)) / 10000.0

    lot_values: Dict[str, float] = {}
    industries: Dict[str, str] = {}
    eligible = []
    skipped_expensive = 0
    for row in ranked.to_dict("records"):
        code = str(row["code"])
        close = safe_float(row.get("execution_close", row.get("close")))
        if not math.isfinite(close) or close <= 0:
            continue
        minimum, _ = buy_order_size_rules(code)
        lot_value = minimum * close * (1.0 + slippage)
        currently_held = int(holdings.get(code, 0)) > 0
        if lot_value > maximum_lot_budget + 1e-8 and not currently_held:
            skipped_expensive += 1
            continue
        lot_values[code] = float(lot_value)
        raw_industry = row.get("industry_1")
        industries[code] = (
            "UNKNOWN"
            if raw_industry is None
            or str(raw_industry).strip().upper()
            in {"", "NAN", "NONE", "NULL", "--", "UNKNOWN"}
            else str(raw_industry).strip()
        )
        eligible.append(code)

    def industry_cap(industry):
        if industry_caps is None:
            return float(maximum_industry_weight)
        return max(
            0.0,
            float(industry_caps.get(str(industry), maximum_industry_weight)),
        )

    def maximum_industry_count(industry):
        cap = industry_cap(industry)
        if cap <= 0:
            return 0
        return max(
            1,
            int(
                math.ceil(
                    requested_count
                    * cap
                    / max(float(target_equity_weight), 1e-8)
                    - 1e-9
                )
            ),
        )

    def industry_counts(codes):
        result: Dict[str, int] = {}
        for code in codes:
            industry = industries.get(code, "UNKNOWN")
            result[industry] = result.get(industry, 0) + 1
        return result

    def industry_lot_value(codes, industry):
        return sum(
            lot_values[code]
            for code in codes
            if industries.get(code, "UNKNOWN") == industry
        )

    def can_add(codes, code):
        industry = industries[code]
        if sum(lot_values[item] for item in codes) + lot_values[code] > equity_budget + 1e-8:
            return False
        counts = industry_counts(codes)
        if counts.get(industry, 0) >= maximum_industry_count(industry):
            return False
        return (
            industry_lot_value(codes, industry) + lot_values[code]
            <= value * industry_cap(industry) + 1e-8
        )

    retained = [
        code
        for code, shares in holdings.items()
        if int(shares) > 0
        and str(code).zfill(6) in lot_values
        and rank_by_code.get(str(code).zfill(6), math.inf) <= int(sell_rank)
    ]
    retained_candidates = sorted(
        (str(code).zfill(6) for code in retained),
        key=lambda code: rank_by_code.get(code, math.inf),
    )
    selected = []
    for code in retained_candidates:
        if len(selected) >= requested_count:
            break
        if can_add(selected, code):
            selected.append(code)

    for code in eligible:
        if len(selected) >= requested_count:
            break
        if rank_by_code.get(code, math.inf) > int(buy_rank):
            break
        if not can_add(selected, code):
            continue
        if code not in selected:
            selected.append(code)

    blocked_replacements = 0
    approved_replacements = 0
    policy = str(replacement_policy).strip().lower()
    if policy not in {"none", "always", "cost_aware"}:
        raise ValueError(f"Unsupported economic replacement policy: {replacement_policy}")
    expected_per_score = max(0.0, float(expected_return_per_score))
    buffer = max(0.0, float(hurdle_buffer_bps)) / 10000.0
    replacement_enabled = policy == "always" or (
        policy == "cost_aware" and expected_per_score > 0
    )
    if len(selected) >= requested_count and replacement_enabled:
        challengers = [
            code
            for code in eligible
            if code not in selected and rank_by_code.get(code, math.inf) <= int(buy_rank)
        ]
        for challenger in challengers:
            incumbents = [code for code in selected if int(holdings.get(code, 0)) > 0]
            if not incumbents:
                break
            incumbent = min(incumbents, key=lambda code: score_by_code.get(code, -math.inf))
            trial = [code for code in selected if code != incumbent] + [challenger]
            trial_counts = industry_counts(trial)
            if any(
                count > maximum_industry_count(industry)
                for industry, count in trial_counts.items()
            ):
                continue
            if any(
                industry_lot_value(trial, industry)
                > value * industry_cap(industry) + 1e-8
                for industry in trial_counts
            ):
                continue
            position_value = max(
                equity_budget / requested_count,
                lot_values.get(incumbent, 0.0),
                lot_values.get(challenger, 0.0),
            )
            cost = round_trip_cost_fraction(
                incumbent,
                challenger,
                position_value,
                decision_date,
                broker_commission_rate,
                broker_minimum_commission,
            )
            score_advantage = score_by_code.get(challenger, 0.0) - score_by_code.get(incumbent, 0.0)
            if score_advantage <= 0:
                continue
            if policy == "always":
                selected = trial
                approved_replacements += 1
                continue
            expected_advantage = max(0.0, score_advantage) * expected_per_score
            if expected_advantage + 1e-12 < cost + buffer:
                blocked_replacements += 1
                continue
            selected = trial
            approved_replacements += 1

    if len(selected) < requested_count:
        for code in eligible:
            if code in selected:
                continue
            if not can_add(selected, code):
                continue
            if sum(lot_values[item] for item in selected) + lot_values[code] > equity_budget + 1e-8:
                continue
            selected.append(code)
            if len(selected) >= requested_count:
                break

    selected = sorted(selected, key=lambda code: rank_by_code.get(code, math.inf))
    minimum_weights = pd.Series(
        {code: lot_values[code] / value for code in selected if value > 0},
        dtype=float,
    )
    return selected, minimum_weights, {
        "cost_aware": True,
        "affordable_count": int(len(selected)),
        "minimum_lot_budget": float(sum(lot_values[code] for code in selected)),
        "max_single_lot_budget": float(maximum_lot_budget),
        "skipped_lot_too_expensive": int(skipped_expensive),
        "economic_replacements_blocked": int(blocked_replacements),
        "economic_replacements_approved": int(approved_replacements),
        "economic_replacement_policy": policy,
        "economic_expected_return_per_score": float(expected_per_score),
    }


def _allocation_state(
    shares_by_code: Mapping[str, int],
    holdings: Mapping[str, int],
    cash: float,
    prices: Mapping[str, float],
    trade_date: str,
    broker_commission_rate: float,
    broker_minimum_commission: float,
):
    final_cash = float(cash)
    total_fees = 0.0
    order_count = 0
    for code in sorted(set(holdings).union(shares_by_code)):
        current = max(0, int(holdings.get(code, 0)))
        target = max(0, int(shares_by_code.get(code, 0)))
        difference = target - current
        if difference == 0:
            continue
        price = safe_float(prices.get(code))
        if not math.isfinite(price) or price <= 0:
            # A suspended or otherwise unpriced holding cannot be traded today.
            # Keep it frozen instead of assuming that its sale can fund new buys.
            continue
        side = "BUY" if difference > 0 else "SELL"
        gross = abs(difference) * price
        fee = mandatory_trade_cost(
            gross,
            side,
            trade_date,
            code,
            broker_commission_rate=broker_commission_rate,
            broker_minimum_commission=broker_minimum_commission,
            shares=abs(difference),
        )
        final_cash += gross - fee if side == "SELL" else -gross - fee
        total_fees += fee
        order_count += 1
    return float(final_cash), float(total_fees), int(order_count)


def optimize_discrete_target_shares(
    target_weights: Mapping[str, float],
    portfolio_value: float,
    prices: Mapping[str, float],
    holdings: Mapping[str, int],
    cash: float,
    trade_date: str,
    broker_commission_rate: float,
    broker_minimum_commission: float,
    minimum_final_holdings: int,
    maximum_stock_weight: float,
    tracking_penalty: float = 1.0,
    cash_penalty: float = 0.75,
    transaction_cost_penalty: float = 2.0,
    iterations: int = 12,
    industries: Mapping[str, str] | None = None,
    industry_caps: Mapping[str, float] | None = None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Use coordinate search over legal lot choices and exact order costs."""
    value = max(0.0, float(portfolio_value))
    targets = {
        str(code).zfill(6): max(0.0, float(weight))
        for code, weight in target_weights.items()
        if float(weight) > 0
    }
    normalized_prices = {
        str(code).zfill(6): safe_float(price)
        for code, price in prices.items()
        if math.isfinite(safe_float(price)) and safe_float(price) > 0
    }
    normalized_industries = {
        str(code).zfill(6): (
            "UNKNOWN"
            if value is None
            or str(value).strip().upper() in {"", "NAN", "NONE", "NULL", "--", "UNKNOWN"}
            else str(value).strip()
        )
        for code, value in (industries or {}).items()
    }
    normalized_industry_caps = {
        str(industry): max(0.0, float(cap))
        for industry, cap in (industry_caps or {}).items()
    }
    if value <= 0 or not targets:
        return {}, {"integer_optimizer_status": "empty_target"}

    unavailable_targets = {
        code: weight for code, weight in targets.items() if code not in normalized_prices
    }
    unavailable_unheld_weight = sum(
        weight
        for code, weight in unavailable_targets.items()
        if int(holdings.get(code, 0)) <= 0
    )
    targets = {
        code: weight for code, weight in targets.items() if code in normalized_prices
    }
    unavailable_meta = {
        "integer_optimizer_unavailable_target_count": int(len(unavailable_targets)),
        "integer_optimizer_unavailable_targets": sorted(unavailable_targets),
    }
    if not targets:
        return {}, {
            "integer_optimizer_status": "no_priced_targets",
            **unavailable_meta,
        }

    legacy = round_portfolio_target_shares(targets, value, normalized_prices)
    options: Dict[str, Sequence[int]] = {}
    for code, weight in targets.items():
        price = normalized_prices.get(code)
        if price is None:
            continue
        minimum, increment = buy_order_size_rules(code)
        desired = weight * value / price
        cap_weight = max(float(maximum_stock_weight), weight * 1.35)
        cap_shares = int(math.floor(cap_weight * value / price))
        if cap_shares < minimum:
            cap_shares = minimum
        cap_shares = minimum + max(0, (cap_shares - minimum) // increment) * increment
        floor_shares = int(legacy.get(code, 0))
        candidates = {minimum, floor_shares, cap_shares}
        current = int(holdings.get(code, 0))
        if current >= minimum:
            candidates.add(minimum + max(0, (current - minimum) // increment) * increment)
        center = minimum + max(0, (int(math.floor(desired)) - minimum) // increment) * increment
        for step in range(-4, 5):
            candidate = center + step * increment
            if minimum <= candidate <= cap_shares:
                candidates.add(candidate)
        options[code] = sorted(candidate for candidate in candidates if candidate >= minimum)

    required = min(max(1, int(minimum_final_holdings)), len(options))
    if len(options) < required:
        return legacy, {
            "integer_optimizer_status": "insufficient_affordable_targets",
            "integer_optimizer_target_count": int(len(options)),
            **unavailable_meta,
        }

    allocation = {code: min(values) for code, values in options.items()}
    target_cash = value * max(
        0.0,
        1.0 - sum(targets.values()) - sum(
            weight
            for code, weight in unavailable_targets.items()
            if int(holdings.get(code, 0)) > 0
        ),
    )
    target_norm = max(sum(weight * weight for weight in targets.values()), 1e-8)

    def objective(candidate):
        final_cash, fees, orders = _allocation_state(
            candidate,
            holdings,
            cash,
            normalized_prices,
            trade_date,
            broker_commission_rate,
            broker_minimum_commission,
        )
        if not math.isfinite(final_cash) or final_cash < -1e-7:
            return math.inf, final_cash, fees, orders
        holding_count = sum(int(shares) > 0 for shares in candidate.values())
        if holding_count < required:
            return math.inf, final_cash, fees, orders
        if normalized_industry_caps:
            actual_by_industry: Dict[str, float] = {}
            for code, shares in candidate.items():
                industry = normalized_industries.get(code, "UNKNOWN")
                actual_by_industry[industry] = actual_by_industry.get(industry, 0.0) + (
                    int(shares) * normalized_prices[code] / value
                )
            if any(
                weight > normalized_industry_caps.get(industry, 0.0) + 1e-8
                for industry, weight in actual_by_industry.items()
            ):
                return math.inf, final_cash, fees, orders
        tracking = 0.0
        for code, target_weight in targets.items():
            actual_weight = candidate.get(code, 0) * normalized_prices[code] / value
            tracking += (actual_weight - target_weight) ** 2
        cash_error = ((final_cash - target_cash) / value) ** 2
        score = (
            max(0.0, float(tracking_penalty)) * tracking / target_norm
            + max(0.0, float(cash_penalty)) * cash_error
            + max(0.0, float(transaction_cost_penalty)) * fees / value
        )
        return float(score), final_cash, fees, orders

    initial_score, initial_cash, _, _ = objective(allocation)
    if not math.isfinite(initial_score):
        return legacy, {
            "integer_optimizer_status": "minimum_lots_infeasible",
            "integer_optimizer_target_count": int(len(options)),
            **unavailable_meta,
        }

    current_score = initial_score
    for _ in range(max(1, int(iterations))):
        changed = False
        for code in sorted(options, key=lambda item: targets.get(item, 0.0), reverse=True):
            best_shares = allocation[code]
            best_score = current_score
            for candidate_shares in options[code]:
                if candidate_shares == allocation[code]:
                    continue
                trial = dict(allocation)
                trial[code] = int(candidate_shares)
                candidate_score, _, _, _ = objective(trial)
                if candidate_score + 1e-12 < best_score:
                    best_score = candidate_score
                    best_shares = int(candidate_shares)
            if best_shares != allocation[code]:
                allocation[code] = best_shares
                current_score = best_score
                changed = True
        if not changed:
            break

    final_score, final_cash, fees, orders = objective(allocation)
    return {code: int(shares) for code, shares in allocation.items() if int(shares) > 0}, {
        "integer_optimizer_status": "applied",
        "integer_optimizer_objective": float(final_score),
        "integer_optimizer_initial_objective": float(initial_score),
        "integer_optimizer_projected_cash": float(final_cash),
        "integer_optimizer_projected_fees": float(fees),
        "integer_optimizer_projected_orders": int(orders),
        "integer_optimizer_target_count": int(len(allocation)),
        "integer_optimizer_reserved_unavailable_weight": float(
            unavailable_unheld_weight
        ),
        **unavailable_meta,
    }
