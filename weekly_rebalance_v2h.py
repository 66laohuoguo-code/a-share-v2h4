"""Create an indicative next-session rebalance plan from the V2H strategy.

The script never submits orders. It uses only data available through the chosen
as-of close and writes a reviewable CSV/XLSX/JSON plan for the next session.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import factor_rank_backtest as base
import factor_rank_backtest_v2h as v2h
from opening_auction import (
    CausalOpeningGapEstimator,
    expected_open_price,
    opening_auction_limit_price,
)
from risk_aware_portfolio import (
    CausalRiskCalibrationStore,
    WeeklyRiskModelStore,
    apply_store_overlay,
)
from v31_strategy import MonthlyFactorStateStore, V31AlphaFeatureStore
from small_account_v3 import optimize_discrete_target_shares
from ashare_utils import (
    apply_risk_alignment_trade_floor,
    buy_order_size_rules,
    load_positions,
    mandatory_trade_cost,
    round_price_to_tick,
    round_portfolio_target_shares,
    round_target_shares_for_code,
    trade_value_floor,
    trading_cost_snapshot,
    write_excel_workbook,
)


DEFAULT_DATABASE = Path("data/processed/stock_daily.sqlite")
DEFAULT_ACCOUNTS_DIR = Path("data/input/accounts")
DEFAULT_STRATEGY_CONFIG = Path("config/v2h4_strategy.json")
DEFAULT_CAPITAL_STRATEGY_MAP = Path("config/weekly_capital_strategy_map.json")
DEFAULT_OUTPUT_DIR = Path("outputs/weekly_rebalance_v2h4")
ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def safe_float(value, default=np.nan):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def validate_account_id(account_id: str) -> str:
    account_id = str(account_id).strip()
    if not ACCOUNT_ID_PATTERN.fullmatch(account_id):
        raise ValueError(
            "Account ID must be 1-64 ASCII letters, digits, dots, underscores or hyphens "
            "and must start with a letter or digit."
        )
    return account_id


def default_account_state_path(account_id: str) -> Path:
    return DEFAULT_ACCOUNTS_DIR / validate_account_id(account_id) / "account_state.json"


def load_account_state(path: Path, expected_account_id: str) -> Dict[str, object]:
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Account state must be a JSON object: {path}")
    stored_account_id = str(state.get("account_id", "")).strip()
    if not stored_account_id:
        raise ValueError(
            f"Account state has no account_id and may belong to another account: {path}. "
            "Use a new account-specific state path or remove this legacy state after checking it."
        )
    if stored_account_id != expected_account_id:
        raise ValueError(
            f"Account state mismatch: requested {expected_account_id!r}, but {path} belongs to "
            f"{stored_account_id!r}."
        )
    return state


def save_account_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(state)), ensure_ascii=False, indent=2), encoding="utf-8")


def load_capital_strategy_map(path: Path) -> Dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Capital strategy map must be a JSON object: {path}")
    raw_tiers = payload.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError(f"Capital strategy map must contain a non-empty tiers list: {path}")

    tiers = []
    previous_limit = 0.0
    for index, raw_tier in enumerate(raw_tiers):
        if not isinstance(raw_tier, dict):
            raise ValueError(f"Capital strategy tier {index + 1} must be a JSON object: {path}")
        tier_name = str(raw_tier.get("tier", "")).strip()
        config_value = str(raw_tier.get("strategy_config", "")).strip()
        if not tier_name or not config_value:
            raise ValueError(
                f"Capital strategy tier {index + 1} needs tier and strategy_config: {path}"
            )
        maximum = raw_tier.get("max_value_exclusive")
        if maximum is None:
            if index != len(raw_tiers) - 1:
                raise ValueError("Only the final capital strategy tier may have no upper limit.")
            maximum_value = None
        else:
            maximum_value = safe_float(maximum, np.nan)
            if not math.isfinite(maximum_value) or maximum_value <= previous_limit:
                raise ValueError("Capital strategy tier limits must be finite and strictly increasing.")
            previous_limit = float(maximum_value)

        strategy_config = Path(config_value)
        if not strategy_config.is_absolute():
            strategy_config = path.parent / strategy_config
        if not strategy_config.is_file():
            raise FileNotFoundError(
                f"Strategy config for capital tier {tier_name!r} was not found: {strategy_config}"
            )
        tiers.append(
            {
                "tier": tier_name,
                "label": str(raw_tier.get("label", tier_name)),
                "max_value_exclusive": maximum_value,
                "strategy_config": strategy_config,
            }
        )
    if tiers[-1]["max_value_exclusive"] is not None:
        raise ValueError("The final capital strategy tier must have no upper limit.")

    validated_minimum = safe_float(payload.get("validated_min_value"), np.nan)
    validated_maximum = safe_float(payload.get("validated_max_value"), np.nan)
    return {
        "path": path,
        "version": int(payload.get("version", 1)),
        "selection_basis": str(
            payload.get("selection_basis", "current_total_value_at_latest_close")
        ),
        "validated_min_value": (
            float(validated_minimum) if math.isfinite(validated_minimum) else None
        ),
        "validated_max_value": (
            float(validated_maximum) if math.isfinite(validated_maximum) else None
        ),
        "tiers": tiers,
    }


def select_strategy_for_capital(
    total_value: float, capital_strategy_map: Mapping[str, object]
) -> Dict[str, object]:
    value = safe_float(total_value, np.nan)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Current portfolio value must be positive before selecting a strategy.")
    for tier in capital_strategy_map["tiers"]:
        maximum = tier["max_value_exclusive"]
        if maximum is None or value < float(maximum):
            return dict(tier)
    raise ValueError("Capital strategy map has no tier covering the current portfolio value.")


def strategy_args_from_config(path: Path):
    return v2h.parse_args(["--strategy-config", str(path)])


def required_v31_live_columns(strategy_args):
    profile = str(
        getattr(strategy_args, "score_profile", "v2h4_legacy")
    ).strip().lower()
    required = []
    if float(getattr(strategy_args, "v31_alpha_tilt_weight", 0.0)) > 0.0:
        required.extend(("v31_earnings_yield_raw", "v31_quality_raw"))
    if profile == "v31":
        required.extend(
            (
                "v31_earnings_yield_raw",
                "v31_quality_raw",
                "v31_growth_raw",
                "residual_momentum_raw",
            )
        )
    if v2h.resolved_v31_industry_budget_mode(strategy_args) == "soft":
        required.append("v31_market_industry_weight")
    return tuple(dict.fromkeys(required))


def latest_as_of_date(conn, requested=None):
    if requested:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM stock_daily WHERE trade_date <= ?",
            (str(requested),),
        ).fetchone()
    else:
        row = conn.execute("SELECT MAX(trade_date) FROM stock_daily").fetchone()
    if not row or not row[0]:
        raise ValueError("The database contains no trading date at or before the requested as-of date.")
    return str(row[0])


def live_monthly_satellite_signal_date(
    dates, date_to_index: Mapping[str, int], as_of_date: str
) -> str:
    """Return the first weekly decision date in the current calendar month.

    The backtest freezes a monthly industry-leadership snapshot on that week's
    decision date. Reconstructing it here makes a stateless weekly invocation
    reproduce the same causal signal without storing a second account state.
    """

    current_month = str(as_of_date)[:7]
    for decision_date in dates:
        if decision_date > as_of_date:
            break
        if str(decision_date)[:7] != current_month:
            continue
        if v2h.should_rebalance_on_date(
            "week_end", 0, 1, dates, date_to_index, decision_date
        ):
            return str(decision_date)
    return str(as_of_date)


def prepare_live_industry_satellite_controller(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    industry_events: pd.DataFrame,
    dates,
    date_to_index: Mapping[str, int],
    as_of_date: str,
    strategy_args,
) -> Tuple[Optional[object], Optional[str]]:
    """Prime the causal industry snapshot used by a live weekly run."""

    maximum_weight = float(
        getattr(strategy_args, "v22_industry_satellite_max_weight", 0.0)
    )
    if maximum_weight <= 0.0:
        return None, None
    schedule = str(
        getattr(strategy_args, "v22_industry_satellite_schedule", "weekly")
    ).strip().lower()
    if schedule not in {"weekly", "monthly"}:
        raise ValueError(f"Unsupported industry satellite schedule: {schedule}")
    signal_date = (
        live_monthly_satellite_signal_date(dates, date_to_index, as_of_date)
        if schedule == "monthly"
        else str(as_of_date)
    )
    snapshot_features = base.feature_snapshot(
        prices, financial, signal_date, strategy_args, industry_events
    )
    controller = v2h.IndustrySatelliteController(schedule)
    controller.blend(
        snapshot_features,
        pd.Series(0.0, index=snapshot_features.index, dtype=float),
        decision_date=signal_date,
        maximum_weight=maximum_weight,
        top_industries=int(
            getattr(strategy_args, "v22_industry_satellite_top_industries", 2)
        ),
        excluded_industries=list(
            getattr(strategy_args, "v22_industry_satellite_excluded_industries", [])
            or []
        ),
        market_risk_on_strength=1.0,
    )
    return controller, signal_date


def load_execution_prices(conn, start_date, end_date):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_daily)")}
    required = {"raw_prev_close", "raw_open", "raw_high", "raw_low", "raw_close"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            "The database does not yet contain real market-price columns. Run the weekly "
            "forward importer with --apply once before generating orders. Missing: "
            + ", ".join(missing)
        )
    return pd.read_sql_query(
        """
        SELECT code, name, trade_date,
               raw_prev_close AS prev_close,
               raw_open AS open,
               raw_high AS high,
               raw_low AS low,
               raw_close AS close,
               industry_1, industry_2
        FROM stock_daily
        WHERE trade_date BETWEEN ? AND ?
        """,
        conn,
        params=(start_date, end_date),
        parse_dates=[],
    )


def close_weights(holdings, prices, total_value, last_known_close=None):
    if total_value <= 0:
        return {}
    last_known_close = last_known_close or {}
    result = {}
    for code, shares in holdings.items():
        price = safe_float(prices.get(code, {}).get("close"), np.nan)
        if not math.isfinite(price) or price <= 0:
            price = safe_float(last_known_close.get(code), np.nan)
        if math.isfinite(price) and price > 0 and int(shares) > 0:
            result[code] = int(shares) * price / total_value
    return result


def limited_trade_shares(code, requested, side, liquidating=False):
    requested = max(0, int(requested))
    minimum, increment = buy_order_size_rules(code)
    if requested <= 0:
        return 0
    if side == "SELL" and liquidating:
        return requested
    if requested < minimum:
        return 0
    return int(minimum + math.floor((requested - minimum) / increment) * increment)


def configured_trade_cost(gross, side, trade_date, code, strategy_args, shares=None):
    return mandatory_trade_cost(
        gross,
        side,
        trade_date,
        code,
        broker_commission_rate=float(
            getattr(strategy_args, "broker_commission_rate", 0.0)
        ),
        broker_minimum_commission=float(
            getattr(strategy_args, "broker_minimum_commission", 0.0)
        ),
        shares=shares,
    )


def build_order_plan(
    holdings,
    cash,
    targets,
    feature_frame,
    prices,
    total_value,
    as_of_date,
    strategy_args,
    last_known_close=None,
    opening_gap_estimator=None,
    industry_caps=None,
):
    feature_by_code = feature_frame.set_index("code").to_dict("index") if not feature_frame.empty else {}
    planned = []
    warnings = []
    last_known_close = last_known_close or {}
    slippage = max(0.0, float(strategy_args.slippage_bps)) / 10000.0
    execution_model = str(
        getattr(strategy_args, "execution_model", "next_open_fixed_bps_legacy")
    ).strip().lower()
    auction_mode = execution_model == "opening_auction_limit"
    rejected_by_floor = []
    target_prices = {}
    for code, target_weight in targets.items():
        row = prices.get(code)
        if row is None:
            continue
        close = safe_float(row.get("close"), np.nan)
        if not math.isfinite(close) or close <= 0:
            continue
        current_shares = int(holdings.get(code, 0))
        side = "BUY" if float(target_weight) * total_value > current_shares * close else "SELL"
        if auction_mode and opening_gap_estimator is not None:
            target_prices[code] = opening_auction_limit_price(
                close,
                side,
                opening_gap_estimator.estimate(code),
                float(getattr(strategy_args, "auction_limit_buffer_bps", 2.0)),
            )
        else:
            target_prices[code] = round_price_to_tick(
                close * (1.0 + slippage if side == "BUY" else 1.0 - slippage),
                side,
            )
    for code in set(holdings) - set(targets):
        row = prices.get(code)
        close = safe_float(row.get("close"), np.nan) if row else np.nan
        if not math.isfinite(close) or close <= 0:
            continue
        if auction_mode and opening_gap_estimator is not None:
            target_prices[code] = opening_auction_limit_price(
                close,
                "SELL",
                opening_gap_estimator.estimate(code),
                float(getattr(strategy_args, "auction_limit_buffer_bps", 2.0)),
            )
        else:
            target_prices[code] = round_price_to_tick(
                close * (1.0 - slippage),
                "SELL",
            )
    integer_meta = {"integer_optimizer_status": "legacy"}
    if str(getattr(strategy_args, "portfolio_constructor", "legacy")).strip().lower() == "integer_cost_aware":
        target_shares_by_code, integer_meta = optimize_discrete_target_shares(
            targets,
            total_value,
            target_prices,
            holdings,
            cash,
            as_of_date,
            broker_commission_rate=float(strategy_args.broker_commission_rate),
            broker_minimum_commission=float(strategy_args.broker_minimum_commission),
            minimum_final_holdings=int(
                getattr(strategy_args, "integer_optimizer_min_holdings", strategy_args.target_count)
            ),
            maximum_stock_weight=float(
                getattr(strategy_args, "lot_aware_max_stock_weight", strategy_args.max_stock_weight)
            ),
            tracking_penalty=float(getattr(strategy_args, "integer_tracking_penalty", 1.0)),
            cash_penalty=float(getattr(strategy_args, "integer_cash_penalty", 0.75)),
            transaction_cost_penalty=float(
                getattr(strategy_args, "integer_transaction_cost_penalty", 2.0)
            ),
            iterations=int(getattr(strategy_args, "integer_optimizer_iterations", 12)),
            industries={
                code: feature.get("industry_1", "UNKNOWN")
                for code, feature in feature_by_code.items()
            },
            industry_caps=industry_caps,
        )
    else:
        target_shares_by_code = round_portfolio_target_shares(
            targets, total_value, target_prices
        )
    current_equity_weight = (
        max(0.0, float(total_value) - float(cash)) / float(total_value)
        if float(total_value) > 0
        else 0.0
    )
    target_equity_weight = float(sum(max(0.0, float(weight)) for weight in targets.values()))
    if integer_meta.get("integer_optimizer_status") not in {"legacy", "applied"}:
        warnings.append(
            "Integer target optimizer fallback: "
            + str(integer_meta.get("integer_optimizer_status"))
        )

    for code in sorted(set(holdings).union(targets)):
        row = prices.get(code)
        if row is None:
            warnings.append(f"{code}: latest close is missing; no order was generated.")
            continue
        close = safe_float(row.get("close"), np.nan)
        if not math.isfinite(close) or close <= 0:
            warnings.append(f"{code}: latest close is invalid; no order was generated.")
            continue
        current_shares = int(holdings.get(code, 0))
        target_weight = max(0.0, float(targets.get(code, 0.0)))
        target_shares = int(target_shares_by_code.get(code, 0)) if target_weight > 0 else 0
        difference = int(target_shares - current_shares)
        if difference == 0:
            continue
        side = "BUY" if difference > 0 else "SELL"
        order_floor, transition_type = trade_value_floor(
            total_value,
            current_shares,
            target_shares,
            strategy_args.min_trade_value,
            getattr(strategy_args, "min_trade_weight", 0.0),
            getattr(strategy_args, "entry_exit_min_trade_value", None),
            getattr(strategy_args, "entry_exit_min_trade_weight", 0.0),
        )
        order_floor, risk_alignment_mode = apply_risk_alignment_trade_floor(
            order_floor,
            total_value,
            side,
            current_equity_weight,
            target_equity_weight,
            strategy_args.risk_rebalance_band,
            getattr(strategy_args, "risk_reduction_min_trade_weight", 0.01),
            getattr(strategy_args, "risk_increase_min_trade_weight", 0.01),
        )
        trade_shares = limited_trade_shares(
            code,
            abs(difference),
            side,
            liquidating=(side == "SELL" and target_shares == 0),
        )
        if trade_shares <= 0:
            continue
        gap_estimate = (
            opening_gap_estimator.estimate(code)
            if opening_gap_estimator is not None
            else None
        )
        auction_limit = (
            opening_auction_limit_price(
                close,
                side,
                gap_estimate,
                float(getattr(strategy_args, "auction_limit_buffer_bps", 2.0)),
            )
            if gap_estimate is not None
            else np.nan
        )
        cash_reservation_price = (
            auction_limit
            if auction_mode and math.isfinite(auction_limit)
            else round_price_to_tick(
                close * (1.0 + slippage if side == "BUY" else 1.0 - slippage),
                side,
            )
        )
        expected_execution_price = (
            round_price_to_tick(
                expected_open_price(close, gap_estimate),
                side,
            )
            if auction_mode and gap_estimate is not None
            else cash_reservation_price
        )
        avg_amount = safe_float(feature_by_code.get(code, {}).get("avg_amount_60"), np.nan)
        if math.isfinite(avg_amount) and avg_amount > 0 and float(strategy_args.max_participation_rate) > 0:
            max_requested = int(
                math.floor(
                    avg_amount
                    * float(strategy_args.max_participation_rate)
                    / cash_reservation_price
                )
            )
            trade_shares = min(
                trade_shares,
                limited_trade_shares(code, max_requested, side, liquidating=False),
            )
        if trade_shares <= 0:
            continue
        gross = trade_shares * cash_reservation_price
        if gross < order_floor:
            rejected_by_floor.append(
                {
                    "code": code,
                    "name": str(row.get("name", "")),
                    "side": side,
                    "transition_type": transition_type,
                    "current_shares": current_shares,
                    "target_shares": target_shares,
                    "planned_shares": int(trade_shares),
                    "reference_close": close,
                    "cash_reservation_price": cash_reservation_price,
                    "gross_amount": gross,
                    "trade_value_floor": order_floor,
                    "current_weight": (
                        current_shares * close / total_value
                        if total_value > 0
                        else 0.0
                    ),
                    "target_weight": target_weight,
                    "rejection_reason": "BELOW_TRADE_VALUE_FLOOR",
                }
            )
            continue
        feature = feature_by_code.get(code, {})
        planned.append(
            {
                "as_of_date": as_of_date,
                "execution_session": (
                    "NEXT_OPENING_AUCTION"
                    if auction_mode
                    else "NEXT_TRADING_SESSION"
                ),
                "order_type": "LIMIT",
                "code": code,
                "name": str(row.get("name", "")),
                "side": side,
                "transition_type": transition_type,
                "trade_value_floor": order_floor,
                "risk_alignment_mode": risk_alignment_mode,
                "risk_reduction_trade": risk_alignment_mode == "REDUCE",
                "risk_increase_trade": risk_alignment_mode == "INCREASE",
                "shares": int(trade_shares),
                "reference_close": close,
                "indicative_price": expected_execution_price,
                "estimated_execution_price": expected_execution_price,
                "broker_order_limit_price": cash_reservation_price,
                "cash_reservation_price": cash_reservation_price,
                "limit_price_role": (
                    "MAXIMUM_BUY_PRICE"
                    if side == "BUY"
                    else "MINIMUM_SELL_PRICE"
                ),
                "limit_price_is_expected_fill": False,
                "recommended_submission_window": (
                    "OPENING_AUCTION_09:15_TO_09:20_REVIEWABLE"
                    if auction_mode
                    else "NEXT_TRADING_SESSION"
                ),
                "cancel_if_unfilled_after_opening_auction": bool(auction_mode),
                "auction_limit_price": auction_limit,
                "auction_expected_open_price": (
                    expected_open_price(close, gap_estimate)
                    if gap_estimate is not None
                    else np.nan
                ),
                "auction_gap_observations": (
                    int(gap_estimate.observations)
                    if gap_estimate is not None
                    else 0
                ),
                "auction_expected_gap": (
                    float(gap_estimate.expected_gap)
                    if gap_estimate is not None
                    else np.nan
                ),
                "auction_lower_gap": (
                    float(gap_estimate.lower_gap)
                    if gap_estimate is not None
                    else np.nan
                ),
                "auction_upper_gap": (
                    float(gap_estimate.upper_gap)
                    if gap_estimate is not None
                    else np.nan
                ),
                "auction_fill_probability": (
                    float(gap_estimate.fill_probability)
                    if gap_estimate is not None
                    else np.nan
                ),
                "gross_amount": gross,
                "expected_gross_amount": (
                    trade_shares * expected_execution_price
                ),
                "expected_fee": configured_trade_cost(
                    trade_shares * expected_execution_price,
                    side,
                    as_of_date,
                    code,
                    strategy_args,
                    trade_shares,
                ),
                "estimated_fee": configured_trade_cost(
                    gross, side, as_of_date, code, strategy_args, trade_shares
                ),
                "current_shares": current_shares,
                "target_shares_before_cash_check": target_shares,
                "current_weight": current_shares * close / total_value if total_value > 0 else 0.0,
                "target_weight": target_weight,
                "industry_1": str(feature.get("industry_1", row.get("industry_1", ""))),
                "rank": feature.get("rank", np.nan),
                "score_v2": feature.get("score_v2", np.nan),
                "avg_amount_60": avg_amount,
            }
        )

    original_planned_count = len(planned)
    planned = v2h.select_sparse_risk_alignment_orders(
        planned,
        current_equity_weight,
        target_equity_weight,
        total_value,
        getattr(strategy_args, "risk_alignment_max_orders", 0),
        getattr(strategy_args, "risk_alignment_initial_max_orders", 0),
    )
    if len(planned) < original_planned_count:
        warnings.append(
            "Sparse risk execution kept "
            f"{len(planned)} of {original_planned_count} executable orders; "
            "remaining risk alignment will be reconsidered next week."
        )
    planned, replacement_guard = v2h.balance_executable_replacement_orders(
        planned,
        current_equity_weight,
        target_equity_weight,
        total_value,
    )
    if int(replacement_guard["deferred_replacement_sell_count"]) > 0:
        warnings.append(
            "Equity-preservation guard deferred "
            f"{replacement_guard['deferred_replacement_sell_count']} SELL orders "
            "because executable replacement BUY orders were insufficient; the old "
            "positions remain until a balanced switch can be submitted."
        )
    if int(replacement_guard["deferred_replacement_buy_count"]) > 0:
        warnings.append(
            "Equity-preservation guard deferred "
            f"{replacement_guard['deferred_replacement_buy_count']} BUY orders "
            "because executable funding SELL orders were insufficient."
        )

    executed = []
    projected_cash = float(cash)
    projected_holdings = {str(code): int(shares) for code, shares in holdings.items() if int(shares) > 0}
    for order in [item for item in planned if item["side"] == "SELL"]:
        code = order["code"]
        shares = min(int(order["shares"]), int(projected_holdings.get(code, 0)))
        if shares <= 0:
            continue
        gross = shares * float(order["cash_reservation_price"])
        fee = configured_trade_cost(
            gross, "SELL", as_of_date, code, strategy_args, shares
        )
        projected_holdings[code] = int(projected_holdings.get(code, 0)) - shares
        if projected_holdings[code] <= 0:
            projected_holdings.pop(code, None)
        projected_cash += gross - fee
        item = dict(order)
        expected_gross = shares * float(order["estimated_execution_price"])
        item.update(
            {
                "shares": shares,
                "gross_amount": gross,
                "expected_gross_amount": expected_gross,
                "expected_fee": configured_trade_cost(
                    expected_gross, "SELL", as_of_date, code, strategy_args, shares
                ),
                "estimated_fee": fee,
                "projected_cash_after": projected_cash,
            }
        )
        executed.append(item)

    buys = sorted(
        (item for item in planned if item["side"] == "BUY"),
        key=lambda item: float(item["target_weight"]),
        reverse=True,
    )
    for order in buys:
        code = order["code"]
        minimum, increment = buy_order_size_rules(code)
        shares = int(order["shares"])
        while shares >= minimum:
            gross = shares * float(order["cash_reservation_price"])
            fee = configured_trade_cost(
                gross, "BUY", as_of_date, code, strategy_args, shares
            )
            if gross + fee <= projected_cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            warnings.append(f"{code}: projected cash was insufficient, so the buy order was omitted.")
            continue
        gross = shares * float(order["cash_reservation_price"])
        if gross < float(order["trade_value_floor"]):
            continue
        fee = configured_trade_cost(
            gross, "BUY", as_of_date, code, strategy_args, shares
        )
        projected_cash -= gross + fee
        projected_holdings[code] = int(projected_holdings.get(code, 0)) + shares
        item = dict(order)
        expected_gross = shares * float(order["estimated_execution_price"])
        item.update(
            {
                "shares": shares,
                "gross_amount": gross,
                "expected_gross_amount": expected_gross,
                "expected_fee": configured_trade_cost(
                    expected_gross, "BUY", as_of_date, code, strategy_args, shares
                ),
                "estimated_fee": fee,
                "projected_cash_after": projected_cash,
            }
        )
        executed.append(item)

    if rejected_by_floor:
        buy_rejections = [item for item in rejected_by_floor if item["side"] == "BUY"]
        sell_rejections = [item for item in rejected_by_floor if item["side"] == "SELL"]
        warnings.append(
            "Trade-value floors filtered "
            f"{len(rejected_by_floor)} planned orders "
            f"({len(buy_rejections)} BUY, {len(sell_rejections)} SELL)."
        )

    orders = pd.DataFrame(executed)
    if orders.empty:
        orders = pd.DataFrame(
            columns=[
                "as_of_date",
                "execution_session",
                "order_type",
                "code",
                "name",
                "side",
                "transition_type",
                "trade_value_floor",
                "shares",
                "reference_close",
                "indicative_price",
                "estimated_execution_price",
                "broker_order_limit_price",
                "cash_reservation_price",
                "limit_price_role",
                "limit_price_is_expected_fill",
                "recommended_submission_window",
                "cancel_if_unfilled_after_opening_auction",
                "auction_limit_price",
                "auction_expected_open_price",
                "auction_gap_observations",
                "auction_expected_gap",
                "auction_lower_gap",
                "auction_upper_gap",
                "auction_fill_probability",
                "gross_amount",
                "expected_gross_amount",
                "expected_fee",
                "estimated_fee",
                "current_shares",
                "target_shares_before_cash_check",
                "current_weight",
                "target_weight",
                "industry_1",
                "rank",
                "score_v2",
                "avg_amount_60",
                "projected_cash_after",
            ]
        )
    orders.attrs["replacement_guard"] = replacement_guard
    orders.attrs["filtered_orders"] = pd.DataFrame(rejected_by_floor)
    projected_rows = []
    for code, shares in sorted(projected_holdings.items()):
        close = safe_float(prices.get(code, {}).get("close"), np.nan)
        if not math.isfinite(close) or close <= 0:
            close = safe_float(last_known_close.get(code), np.nan)
        feature = feature_by_code.get(code, {})
        projected_rows.append(
            {
                "code": code,
                "name": str(prices.get(code, {}).get("name", "")),
                "projected_shares": shares,
                "reference_close": close,
                "projected_value_at_close": shares * close if math.isfinite(close) else np.nan,
                "target_weight": float(targets.get(code, 0.0)),
                "industry_1": str(feature.get("industry_1", prices.get(code, {}).get("industry_1", ""))),
                "rank": feature.get("rank", np.nan),
                "score_v2": feature.get("score_v2", np.nan),
            }
        )
    return orders, pd.DataFrame(projected_rows), float(projected_cash), warnings


def run(args):
    account_id = validate_account_id(args.account_id)
    positions_path = Path(args.positions)
    account_state_path = (
        Path(args.account_state)
        if args.account_state is not None
        else default_account_state_path(account_id)
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR / account_id
    )
    manual_strategy_config = (
        Path(args.strategy_config) if args.strategy_config is not None else None
    )
    capital_strategy_map = None
    if manual_strategy_config is not None:
        candidate_strategy_configs = [manual_strategy_config]
    else:
        capital_strategy_map = load_capital_strategy_map(Path(args.capital_strategy_map))
        candidate_strategy_configs = [
            Path(tier["strategy_config"]) for tier in capital_strategy_map["tiers"]
        ]
    candidate_strategy_args = [
        strategy_args_from_config(path) for path in candidate_strategy_configs
    ]
    conn = sqlite3.connect(args.database)
    risk_store = None
    risk_calibration_store = None
    v31_alpha_store = None
    v31_factor_state_store = None
    warnings = []
    try:
        as_of_date = latest_as_of_date(conn, args.as_of_date)
        dates = base.trading_dates(conn)
        date_to_index = {date: index for index, date in enumerate(dates)}
        as_of_index = date_to_index[as_of_date]
        prehistory = max(
            320,
            *[
                max(
                    int(candidate.feature_history_days),
                    int(candidate.min_history_days) + 30,
                )
                for candidate in candidate_strategy_args
            ],
        )
        history_start = dates[max(0, as_of_index - prehistory)]
        prices = base.load_prices(conn, history_start, as_of_date)
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        execution_prices = load_execution_prices(conn, history_start, as_of_date)
        execution_prices["code"] = execution_prices["code"].astype(str).str.zfill(6)
        execution_prices["close"] = pd.to_numeric(execution_prices["close"], errors="coerce")
        today = execution_prices.loc[
            execution_prices["trade_date"].astype(str).eq(as_of_date)
            & execution_prices["close"].gt(0)
        ].copy()
        if today.empty:
            raise ValueError(
                f"No real ClosePrice values are available for {as_of_date}. Re-run "
                "import_csmar_forward_quotation.py with --force-reimport --apply."
            )
        price_map = today.set_index("code").to_dict("index")
        latest_rows = (
            execution_prices.loc[execution_prices["close"].gt(0)]
            .sort_values(["code", "trade_date"])
            .groupby("code", as_index=False)
            .tail(1)
        )
        last_known_close = latest_rows.set_index("code")["close"].apply(safe_float).to_dict()

        positions, positions_cash = load_positions(positions_path)
        cash = float(args.cash) if args.cash is not None else float(positions_cash)
        holdings = {
            str(row.code).zfill(6): int(round(float(row.shares)))
            for row in positions.itertuples(index=False)
            if float(row.shares) > 0
        }
        total_value = base.value_portfolio(holdings, cash, price_map, "close", last_known_close)
        if total_value <= 0:
            raise ValueError(
                f"Account {account_id!r} has zero portfolio value because its positions file "
                f"contains no positive stock shares or CASH amount: {positions_path}. "
                "Fill this account-specific file with the broker's actual positions and available cash."
            )

        account_state = load_account_state(account_state_path, account_id)
        stored_last_value = safe_float(
            account_state.get("last_portfolio_value"), np.nan
        )
        value_change_since_last_run = (
            total_value / stored_last_value - 1.0
            if math.isfinite(stored_last_value) and stored_last_value > 0
            else np.nan
        )
        if (
            math.isfinite(value_change_since_last_run)
            and abs(value_change_since_last_run) >= 0.08
            and not bool(args.reset_peak_to_current)
            and args.peak_value is None
        ):
            warnings.append(
                f"Account value changed {value_change_since_last_run:.1%} since the last run. "
                "Verify available cash, fills and external deposits/withdrawals. If this is "
                "a cash correction or capital flow rather than investment loss, rerun with "
                "--reset-peak-to-current so the drawdown guard is not distorted."
            )
        if manual_strategy_config is not None:
            strategy_selection_mode = "manual_override"
            capital_strategy_tier = "manual"
            capital_strategy_label = "Manual strategy override"
            strategy_config_path = manual_strategy_config
        else:
            selected_tier = select_strategy_for_capital(total_value, capital_strategy_map)
            strategy_selection_mode = "automatic_by_current_total_value"
            capital_strategy_tier = str(selected_tier["tier"])
            capital_strategy_label = str(selected_tier["label"])
            strategy_config_path = Path(selected_tier["strategy_config"])
            validated_minimum = capital_strategy_map.get("validated_min_value")
            validated_maximum = capital_strategy_map.get("validated_max_value")
            if validated_minimum is not None and total_value < float(validated_minimum):
                warnings.append(
                    f"Current value CNY {total_value:,.2f} is below the validated capital range "
                    f"starting at CNY {float(validated_minimum):,.2f}; the smallest-account "
                    "strategy is used as an extrapolation."
                )
            if validated_maximum is not None and total_value > float(validated_maximum):
                warnings.append(
                    f"Current value CNY {total_value:,.2f} is above the validated capital range "
                    f"ending at CNY {float(validated_maximum):,.2f}; review liquidity and "
                    "participation constraints before trading."
                )
            previous_tier = str(account_state.get("last_capital_strategy_tier", "")).strip()
            if previous_tier and previous_tier != capital_strategy_tier:
                warnings.append(
                    f"Automatic strategy tier changed from {previous_tier!r} to "
                    f"{capital_strategy_tier!r} because the current account value crossed "
                    "a configured capital boundary."
                )

        strategy_args = strategy_args_from_config(strategy_config_path)
        score_profile = str(
            getattr(strategy_args, "score_profile", "v2h4_legacy")
        ).strip().lower()
        risk_overlay_mode = str(
            getattr(strategy_args, "risk_overlay_mode", "disabled")
        )
        if risk_overlay_mode != "disabled":
            if args.risk_model_database is None:
                raise ValueError(
                    f"Strategy {strategy_config_path.name!r} requires a risk model. "
                    "Pass --risk-model-database (or -RiskDatabase through run_weekly.ps1)."
                )
            risk_store = WeeklyRiskModelStore(args.risk_model_database)
            if args.risk_calibration_schedule is not None:
                risk_calibration_store = CausalRiskCalibrationStore(
                    args.risk_calibration_schedule
                )
        requires_v31_alpha = v2h.uses_v31_alpha_features(strategy_args)
        if requires_v31_alpha:
            if args.risk_model_database is None:
                raise ValueError(
                    f"Strategy {strategy_config_path.name!r} uses V3.1 alpha "
                    "features and requires --risk-model-database."
                )
            v31_alpha_store = V31AlphaFeatureStore(
                args.risk_model_database,
                lookback_weeks=int(strategy_args.residual_momentum_lookback_weeks),
                skip_weeks=int(strategy_args.residual_momentum_skip_weeks),
            )
            if (
                score_profile == "v31"
                and str(strategy_args.v31_factor_state_mode).strip().lower()
                == "monthly"
            ):
                v31_factor_state_store = MonthlyFactorStateStore(
                    args.risk_model_database,
                    strategy_args,
                )
        market_state = base.build_market_state(prices, strategy_args)
        financial = base.load_financial_factors(conn)
        industry_events = base.load_industry_event_scores(strategy_args.industry_event_scores)
        event_regime = base.load_event_regime_signals(strategy_args.event_regime_signals)
        stored_peak = safe_float(account_state.get("peak_portfolio_value"), np.nan)
        requested_peak = safe_float(args.peak_value, np.nan)
        if bool(args.reset_peak_to_current):
            peak_value = total_value
            warnings.append(
                "The portfolio peak was reset to the current account value because "
                "--reset-peak-to-current was supplied."
            )
        else:
            peak_value = requested_peak if math.isfinite(requested_peak) and requested_peak > 0 else stored_peak
        if not math.isfinite(peak_value) or peak_value <= 0:
            peak_value = total_value
            warnings.append("No portfolio peak was available; this run initializes the drawdown guard at the current value.")
        peak_value = max(float(peak_value), float(total_value))

        regime = v2h.continuous_target_equity(
            market_state,
            as_of_date,
            total_value,
            peak_value,
            strategy_args,
            event_regime,
        )
        (
            v22_satellite_controller,
            v22_satellite_signal_date,
        ) = prepare_live_industry_satellite_controller(
            prices,
            financial,
            industry_events,
            dates,
            date_to_index,
            as_of_date,
            strategy_args,
        )
        features = base.feature_snapshot(prices, financial, as_of_date, strategy_args, industry_events)
        v31_meta = {}
        if v31_alpha_store is not None:
            features, v31_meta = v31_alpha_store.augment(
                features,
                as_of_date,
                maximum_staleness_days=int(
                    strategy_args.risk_model_max_staleness_days
                ),
            )
            required_columns = required_v31_live_columns(strategy_args)
            insufficient = {}
            for column in required_columns:
                coverage = float(
                    pd.to_numeric(
                        features.get(
                            column,
                            pd.Series(np.nan, index=features.index, dtype=float),
                        ),
                        errors="coerce",
                    ).notna().mean()
                )
                if coverage < 0.50:
                    insufficient[column] = coverage
            if insufficient:
                details = ", ".join(
                    f"{column}={coverage:.1%}"
                    for column, coverage in insufficient.items()
                )
                raise ValueError(
                    f"Account {account_id!r} (current value CNY {total_value:,.2f}) "
                    f"selected {strategy_config_path.name!r} via "
                    f"{strategy_selection_mode} / tier {capital_strategy_tier!r}. "
                    "This strategy requires current causal V3.1 alpha data, "
                    f"but coverage is insufficient for {as_of_date}: "
                    f"{details}. Latest alpha date: "
                    f"{v31_meta.get('v31_alpha_date')}; latest residual-momentum "
                    f"date: {v31_meta.get('v31_residual_momentum_date')}. Update "
                    "the risk-model database before generating live orders."
                )
        component_weights = v2h.configured_component_weights(strategy_args)
        if v31_factor_state_store is not None:
            component_weights, factor_state_meta = v31_factor_state_store.weights(
                as_of_date,
                component_weights,
            )
            v31_meta.update(factor_state_meta)
        features = v2h.apply_v2_score(
            features,
            component_weights,
            strategy_args,
            decision_date=as_of_date,
            satellite_controller=v22_satellite_controller,
            market_risk_on_strength=v2h.v22_market_risk_on_strength(
                regime, strategy_args
            ),
        )
        if v22_satellite_signal_date is not None:
            features["v22_satellite_signal_date"] = v22_satellite_signal_date
        features["rank"] = np.arange(1, len(features) + 1)
        features["execution_close"] = features["code"].map(
            today.set_index("code")["close"].to_dict()
        )
        current_weights = close_weights(holdings, price_map, total_value, last_known_close)
        target_transform = None
        if risk_store is not None:
            def target_transform(
                desired,
                _frame,
                effective_stock_cap,
                _effective_industry_cap,
            ):
                calibration_multiplier = float(
                    strategy_args.risk_calibration_multiplier
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
                        as_of_date,
                        default=calibration_multiplier,
                    )
                weights, metadata = apply_store_overlay(
                    risk_store,
                    as_of_date,
                    desired,
                    stock_cap=effective_stock_cap,
                    maximum_industry_fraction=float(
                        strategy_args.risk_model_max_industry_weight
                    ),
                    strength=float(strategy_args.risk_overlay_strength),
                    target_volatility=float(
                        strategy_args.target_portfolio_volatility
                    ),
                    calibration_multiplier=calibration_multiplier,
                    minimum_equity_scale=float(
                        strategy_args.risk_overlay_min_equity_scale
                    ),
                    maximum_staleness_days=int(
                        strategy_args.risk_model_max_staleness_days
                    ),
                    iterations=int(strategy_args.risk_overlay_iterations),
                )
                metadata.update(calibration_metadata)
                return weights, metadata

        targets, target_meta = v2h.build_targets_v2(
            features,
            holdings,
            current_weights,
            strategy_args,
            float(regime["target_equity_weight"]),
            total_value,
            force_risk_alignment=v2h.should_force_risk_alignment(
                strategy_args,
                sum(current_weights.values()),
                float(regime["target_equity_weight"]),
            ),
            decision_date=as_of_date,
            target_transform=target_transform,
        )
        target_meta["v22_satellite_signal_date"] = v22_satellite_signal_date
        target_meta.update(v31_meta)
        if (
            risk_store is not None
            and str(target_meta.get("risk_overlay_status")) != "applied"
        ):
            latest_causal_risk_date = risk_store.model_date_for(as_of_date)
            latest_available_risk_date = (
                risk_store.dates[-1] if risk_store.dates else None
            )
            risk_staleness_days = (
                int(
                    (
                        pd.Timestamp(as_of_date)
                        - pd.Timestamp(latest_causal_risk_date)
                    ).days
                )
                if latest_causal_risk_date is not None
                else None
            )
            raise ValueError(
                f"Account {account_id!r} (current value CNY {total_value:,.2f}) "
                f"selected {strategy_config_path.name!r} via "
                f"{strategy_selection_mode} / tier {capital_strategy_tier!r}. "
                "That strategy requires the weekly risk overlay, but no usable "
                f"causal snapshot was available for {as_of_date}. Risk database: "
                f"{risk_store.database}; latest complete snapshot: "
                f"{latest_available_risk_date}; latest causal snapshot: "
                f"{latest_causal_risk_date}; staleness: {risk_staleness_days} days; "
                f"allowed: {int(strategy_args.risk_model_max_staleness_days)} days; "
                f"status: {target_meta.get('risk_overlay_status', 'missing')}. "
                "Update the risk-model database before generating live orders."
            )
        opening_gap_estimator = CausalOpeningGapEstimator(
            lookback_days=int(
                getattr(strategy_args, "auction_gap_lookback_days", 252)
            ),
            min_observations=int(
                getattr(strategy_args, "auction_min_gap_observations", 60)
            ),
            fill_probability=float(
                getattr(strategy_args, "auction_fill_probability", 0.90)
            ),
            shrinkage_observations=float(
                getattr(strategy_args, "auction_shrinkage_observations", 40.0)
            ),
            market_lookback_days=int(
                getattr(strategy_args, "auction_market_lookback_days", 60)
            ),
        )
        opening_gap_estimator.seed_from_frame(execution_prices, as_of_date)
        orders, projected_positions, projected_cash, order_warnings = build_order_plan(
            holdings,
            cash,
            targets,
            features,
            price_map,
            total_value,
            as_of_date,
            strategy_args,
            last_known_close,
            opening_gap_estimator,
            industry_caps=target_meta.get("industry_budget_caps"),
        )
        warnings.extend(order_warnings)
        replacement_guard = dict(orders.attrs.get("replacement_guard", {}))
        filtered_orders = orders.attrs.get("filtered_orders", pd.DataFrame()).copy()
        projected_stock_value = float(projected_positions["projected_value_at_close"].sum()) if not projected_positions.empty else 0.0
        projected_total_value = projected_stock_value + projected_cash
        projected_equity_weight = projected_stock_value / projected_total_value if projected_total_value > 0 else 0.0
        effective_target_equity_weight = float(
            target_meta.get("target_weight_sum", regime["target_equity_weight"])
        )
        target_shortfall = effective_target_equity_weight - projected_equity_weight
        if target_shortfall > 0.05:
            warnings.append(
                f"Projected equity is {projected_equity_weight:.1%}, {target_shortfall:.1%} below the model target; "
                "the minimum-trade threshold and board lots are the main likely causes."
            )
        if pd.Timestamp(as_of_date).weekday() < 3:
            warnings.append("The latest database date is early in the week; confirm that this is the intended weekly decision date.")
        if str(
            getattr(
                strategy_args,
                "execution_model",
                "next_open_fixed_bps_legacy",
            )
        ) == "opening_auction_limit":
            warnings.append(
                "indicative_price and estimated_execution_price are the model's central "
                "opening-price estimate, not the broker order price. broker_order_limit_price "
                "and auction_limit_price are the maximum BUY or minimum SELL boundary used "
                "for conservative share sizing and cash reservation."
            )
            warnings.append(
                "Submit auction orders while they remain reviewable, preferably during "
                "09:15-09:20. The protective limit is not an expected fill price. Cancel any "
                "unfilled remainder after the opening auction instead of letting an aggressive "
                "limit continue into continuous trading."
            )
        else:
            warnings.append(
                "reference_close is the latest unadjusted market close reconstructed from market "
                "value / shares. indicative_price is only a tick-rounded budget estimate using "
                "configured slippage, not a guaranteed or required order price."
            )
        warnings.append(
            "auction_limit_price is a causal historical-gap limit recommendation for the next "
            "opening auction. It protects the maximum BUY or minimum SELL price; it does not "
            "guarantee a fill. Check the live indicative auction price and unmatched volume "
            "before placing the final order."
        )

        summary = {
            "account_id": account_id,
            "strategy": str(getattr(strategy_args, "strategy_name", "V2H4")),
            "strategy_config": str(strategy_config_path),
            "strategy_selection_mode": strategy_selection_mode,
            "capital_strategy_tier": capital_strategy_tier,
            "capital_strategy_label": capital_strategy_label,
            "capital_strategy_map": (
                None
                if capital_strategy_map is None
                else str(capital_strategy_map["path"])
            ),
            "as_of_date": as_of_date,
            "database": str(args.database),
            "positions_file": str(positions_path),
            "account_state_file": str(account_state_path),
            "current_cash": cash,
            "current_stock_value": total_value - cash,
            "current_total_value": total_value,
            "execution_price_source": "unadjusted close reconstructed from market value / shares",
            "execution_model": str(
                getattr(
                    strategy_args,
                    "execution_model",
                    "next_open_fixed_bps_legacy",
                )
            ),
            "indicative_price_slippage_bps": float(strategy_args.slippage_bps),
            "auction_fill_probability": float(
                getattr(strategy_args, "auction_fill_probability", 0.90)
            ),
            "auction_gap_lookback_days": int(
                getattr(strategy_args, "auction_gap_lookback_days", 252)
            ),
            "auction_limit_buffer_bps": float(
                getattr(strategy_args, "auction_limit_buffer_bps", 2.0)
            ),
            "indicative_price_definition": (
                "central historical-gap estimate of the next opening price"
            ),
            "broker_order_limit_price_definition": (
                "protective maximum BUY or minimum SELL boundary; not expected fill price"
            ),
            "broker_commission_rate": float(
                getattr(strategy_args, "broker_commission_rate", 0.0)
            ),
            "broker_minimum_commission": float(
                getattr(strategy_args, "broker_minimum_commission", 0.0)
            ),
            "peak_portfolio_value": peak_value,
            "peak_reset_to_current": bool(args.reset_peak_to_current),
            "last_recorded_portfolio_value": (
                float(stored_last_value)
                if math.isfinite(stored_last_value)
                else None
            ),
            "portfolio_value_change_since_last_run": (
                float(value_change_since_last_run)
                if math.isfinite(value_change_since_last_run)
                else None
            ),
            "portfolio_drawdown": float(regime["portfolio_drawdown"]),
            "market_state": str(regime["market_state"]),
            "market_return_20": safe_float(regime["market_return_20"], np.nan),
            "market_breadth": safe_float(regime["market_breadth"], np.nan),
            "market_volatility_20": safe_float(regime["market_volatility_20"], np.nan),
            "target_equity_weight": float(regime["target_equity_weight"]),
            "effective_target_equity_weight": effective_target_equity_weight,
            "current_equity_weight": float((total_value - cash) / total_value),
            "selected_count": int(target_meta.get("selected_count", 0)),
            "target_weight_sum": float(target_meta.get("target_weight_sum", 0.0)),
            "risk_overlay_mode": risk_overlay_mode,
            "risk_overlay_status": str(
                target_meta.get("risk_overlay_status", "disabled")
            ),
            "risk_model_database": (
                None
                if args.risk_model_database is None
                else str(args.risk_model_database)
            ),
            "risk_model_date": target_meta.get("risk_model_date"),
            "risk_calibration_schedule": (
                None
                if args.risk_calibration_schedule is None
                else str(args.risk_calibration_schedule)
            ),
            "risk_calibration_source": target_meta.get(
                "risk_calibration_source"
            ),
            "risk_calibration_as_of_date": target_meta.get(
                "risk_calibration_as_of_date"
            ),
            "risk_calibration_multiplier": target_meta.get(
                "risk_calibration_multiplier"
            ),
            "risk_predicted_volatility_before": target_meta.get(
                "risk_predicted_volatility_before"
            ),
            "risk_predicted_volatility_after": target_meta.get(
                "risk_predicted_volatility_after"
            ),
            "v31_alpha_status": target_meta.get("v31_alpha_status"),
            "v31_alpha_date": target_meta.get("v31_alpha_date"),
            "v31_residual_momentum_date": target_meta.get(
                "v31_residual_momentum_date"
            ),
            "v31_unknown_target_weight": target_meta.get(
                "unknown_target_weight"
            ),
            "v31_unknown_industry_cap_enforced": target_meta.get(
                "v31_unknown_industry_cap_enforced",
                bool(
                    getattr(
                        strategy_args,
                        "v31_enforce_unknown_industry_cap",
                        False,
                    )
                ),
            ),
            "v31_factor_state_mode": target_meta.get("factor_state_mode"),
            "v31_offensive_weight": target_meta.get("offensive_weight"),
            "v31_factor_state_signal": target_meta.get(
                "factor_state_signal"
            ),
            "v31_industry_budget_caps": target_meta.get(
                "industry_budget_caps"
            ),
            "v22_industry_satellite_application": str(
                getattr(
                    strategy_args,
                    "v22_industry_satellite_application",
                    "score_and_weight",
                )
            ),
            "v22_industry_satellite_schedule": str(
                getattr(strategy_args, "v22_industry_satellite_schedule", "weekly")
            ),
            "v22_industry_satellite_signal_date": target_meta.get(
                "v22_satellite_signal_date"
            ),
            "v22_industry_satellite_weight": target_meta.get(
                "v22_satellite_weight", 0.0
            ),
            "v22_leading_industries": target_meta.get(
                "v22_leading_industries", ""
            ),
            "min_trade_value": float(strategy_args.min_trade_value),
            "min_trade_weight": float(getattr(strategy_args, "min_trade_weight", 0.0)),
            "entry_exit_min_trade_value": (
                None
                if getattr(strategy_args, "entry_exit_min_trade_value", None) is None
                else float(strategy_args.entry_exit_min_trade_value)
            ),
            "entry_exit_min_trade_weight": float(
                getattr(strategy_args, "entry_exit_min_trade_weight", 0.0)
            ),
            "risk_reduction_min_trade_weight": float(
                getattr(strategy_args, "risk_reduction_min_trade_weight", 0.01)
            ),
            "risk_increase_min_trade_weight": float(
                getattr(strategy_args, "risk_increase_min_trade_weight", 0.01)
            ),
            "lot_aware_selection": bool(target_meta.get("lot_aware", False)),
            "force_risk_alignment": bool(target_meta.get("force_risk_alignment", False)),
            "risk_target_alignment": str(
                getattr(strategy_args, "risk_target_alignment", "banded")
            ),
            "risk_rebalance_schedule": str(
                getattr(strategy_args, "risk_rebalance_schedule", "daily")
            ),
            "risk_alignment_max_orders": int(
                getattr(strategy_args, "risk_alignment_max_orders", 0)
            ),
            "risk_alignment_initial_max_orders": int(
                getattr(strategy_args, "risk_alignment_initial_max_orders", 0)
            ),
            "affordable_target_count": int(
                target_meta.get("affordable_count", target_meta.get("selected_count", 0))
            ),
            "effective_max_stock_weight": float(
                target_meta.get("effective_max_stock_weight", strategy_args.max_stock_weight)
            ),
            "effective_max_industry_weight": float(
                target_meta.get("effective_max_industry_weight", strategy_args.max_industry_weight)
            ),
            "order_count": int(len(orders)),
            "buy_count": int((orders["side"] == "BUY").sum()) if not orders.empty else 0,
            "sell_count": int((orders["side"] == "SELL").sum()) if not orders.empty else 0,
            "filtered_order_count": int(len(filtered_orders)),
            "filtered_buy_count": (
                int(filtered_orders["side"].eq("BUY").sum())
                if not filtered_orders.empty
                else 0
            ),
            "filtered_sell_count": (
                int(filtered_orders["side"].eq("SELL").sum())
                if not filtered_orders.empty
                else 0
            ),
            "estimated_gross_traded": float(orders["gross_amount"].sum()) if not orders.empty else 0.0,
            "expected_gross_traded": (
                float(orders["expected_gross_amount"].sum())
                if not orders.empty
                else 0.0
            ),
            "cash_reservation_gross": (
                float(orders["gross_amount"].sum()) if not orders.empty else 0.0
            ),
            "estimated_fees": float(orders["estimated_fee"].sum()) if not orders.empty else 0.0,
            "expected_fees": (
                float(orders["expected_fee"].sum()) if not orders.empty else 0.0
            ),
            "projected_cash": projected_cash,
            "projected_stock_value_at_close": projected_stock_value,
            "projected_total_value_at_close": projected_total_value,
            "projected_equity_weight": projected_equity_weight,
            **replacement_guard,
            "factor_weights": v2h.STATIC_COMPONENT_WEIGHTS,
            "trading_costs": trading_cost_snapshot(
                as_of_date,
                broker_commission_rate=float(
                    getattr(strategy_args, "broker_commission_rate", 0.0)
                ),
                broker_minimum_commission=float(
                    getattr(strategy_args, "broker_minimum_commission", 0.0)
                ),
            ),
            "warnings": warnings,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"v2h4_{account_id}_rebalance_{as_of_date.replace('-', '')}_{stamp}"
        paths = {
            "orders": output_dir / f"{stem}_orders.csv",
            "filtered_orders": output_dir / f"{stem}_filtered_orders.csv",
            "projected_positions": output_dir / f"{stem}_projected_positions.csv",
            "factor_ranking": output_dir / f"{stem}_factor_ranking.csv",
            "summary": output_dir / f"{stem}_summary.json",
            "workbook": output_dir / f"{stem}.xlsx",
        }
        orders.to_csv(paths["orders"], index=False, encoding="utf-8-sig")
        filtered_orders.to_csv(
            paths["filtered_orders"], index=False, encoding="utf-8-sig"
        )
        projected_positions.to_csv(paths["projected_positions"], index=False, encoding="utf-8-sig")
        ranking_columns = [
            "rank",
            "code",
            "name",
            "industry_1",
            "score_v2",
            "low_beta_score",
            "low_volatility_score",
            "low_turnover_score",
            "reversal_score",
            "lower_drawdown_score",
            "industry_trend_score",
            "earnings_yield_score",
            "quality_score_v31",
            "growth_score_v31",
            "residual_momentum_score",
            "v31_float_market_cap",
            "v31_market_industry_weight",
            "execution_close",
            "volatility_120",
            "avg_amount_60",
        ]
        ranking = features[[column for column in ranking_columns if column in features.columns]].copy()
        ranking.to_csv(paths["factor_ranking"], index=False, encoding="utf-8-sig")
        paths["summary"].write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        write_excel_workbook(
            paths["workbook"],
            [
                ("summary", pd.DataFrame([{key: json.dumps(json_ready(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in summary.items()}])),
                ("orders", orders),
                ("filtered_orders", filtered_orders),
                ("projected_positions", projected_positions),
                ("factor_ranking", ranking),
                ("warnings", pd.DataFrame({"warning": warnings})),
            ],
        )

        if not args.no_update_account_state:
            save_account_state(
                account_state_path,
                {
                    "account_id": account_id,
                    "positions_file": str(positions_path),
                    "peak_portfolio_value": peak_value,
                    "last_portfolio_value": total_value,
                    "last_as_of_date": as_of_date,
                    "last_strategy": str(getattr(strategy_args, "strategy_name", "V2H4")),
                    "last_strategy_config": str(strategy_config_path),
                    "last_strategy_selection_mode": strategy_selection_mode,
                    "last_capital_strategy_tier": capital_strategy_tier,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )

        print(f"Account: {account_id}")
        print(f"As-of date: {as_of_date}")
        print(f"Current value: {total_value:.2f}")
        print(f"Strategy selection: {strategy_selection_mode}")
        print(f"Capital tier: {capital_strategy_tier}")
        print(f"Strategy config: {strategy_config_path}")
        print(f"Regime target equity: {float(regime['target_equity_weight']):.2%}")
        print(f"Effective target equity: {effective_target_equity_weight:.2%}")
        print(f"Orders: {len(orders)}")
        print(f"Workbook: {paths['workbook']}")
        return summary, paths
    finally:
        if risk_store is not None:
            risk_store.close()
        if v31_alpha_store is not None:
            v31_alpha_store.close()
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate a V2H4 next-session rebalance plan from current positions.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--account-id", required=True, help="Stable ID used to isolate one brokerage account.")
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--account-state", type=Path)
    parser.add_argument(
        "--strategy-config",
        type=Path,
        help="Manual strategy override. Omit to select from the current total account value.",
    )
    parser.add_argument(
        "--capital-strategy-map",
        type=Path,
        default=DEFAULT_CAPITAL_STRATEGY_MAP,
        help="Capital-tier strategy map used when --strategy-config is omitted.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--as-of-date")
    parser.add_argument("--cash", type=float, help="Override CASH from the positions file for this run.")
    parser.add_argument("--peak-value", type=float, help="Override the stored historical peak portfolio value.")
    parser.add_argument(
        "--reset-peak-to-current",
        action="store_true",
        help="Use the current account value as a new drawdown peak after a capital flow or cash correction.",
    )
    parser.add_argument(
        "--risk-model-database",
        type=Path,
        help="Weekly point-in-time risk-model database required by risk-overlay strategies.",
    )
    parser.add_argument(
        "--risk-calibration-schedule",
        type=Path,
        help="Optional causal risk-calibration schedule used by risk-overlay strategies.",
    )
    parser.add_argument("--no-update-account-state", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
