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
from typing import Dict, Mapping

import numpy as np
import pandas as pd

import factor_rank_backtest as base
import factor_rank_backtest_v2h as v2h
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


def configured_trade_cost(gross, side, trade_date, code, strategy_args):
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
):
    feature_by_code = feature_frame.set_index("code").to_dict("index") if not feature_frame.empty else {}
    planned = []
    warnings = []
    last_known_close = last_known_close or {}
    slippage = max(0.0, float(strategy_args.slippage_bps)) / 10000.0
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
        target_prices[code] = round_price_to_tick(
            close * (1.0 + slippage if side == "BUY" else 1.0 - slippage),
            side,
        )
    target_shares_by_code = round_portfolio_target_shares(targets, total_value, target_prices)
    current_equity_weight = (
        max(0.0, float(total_value) - float(cash)) / float(total_value)
        if float(total_value) > 0
        else 0.0
    )
    target_equity_weight = float(sum(max(0.0, float(weight)) for weight in targets.values()))

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
        price = round_price_to_tick(
            close * (1.0 + slippage if side == "BUY" else 1.0 - slippage),
            side,
        )
        avg_amount = safe_float(feature_by_code.get(code, {}).get("avg_amount_60"), np.nan)
        if math.isfinite(avg_amount) and avg_amount > 0 and float(strategy_args.max_participation_rate) > 0:
            max_requested = int(math.floor(avg_amount * float(strategy_args.max_participation_rate) / price))
            trade_shares = min(
                trade_shares,
                limited_trade_shares(code, max_requested, side, liquidating=False),
            )
        if trade_shares <= 0:
            continue
        gross = trade_shares * price
        if gross < order_floor:
            rejected_by_floor.append((code, side, transition_type, gross, order_floor))
            continue
        feature = feature_by_code.get(code, {})
        planned.append(
            {
                "as_of_date": as_of_date,
                "execution_session": "NEXT_TRADING_SESSION",
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
                "indicative_price": price,
                "gross_amount": gross,
                "estimated_fee": configured_trade_cost(
                    gross, side, as_of_date, code, strategy_args
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

    executed = []
    projected_cash = float(cash)
    projected_holdings = {str(code): int(shares) for code, shares in holdings.items() if int(shares) > 0}
    for order in [item for item in planned if item["side"] == "SELL"]:
        code = order["code"]
        shares = min(int(order["shares"]), int(projected_holdings.get(code, 0)))
        if shares <= 0:
            continue
        gross = shares * float(order["indicative_price"])
        fee = configured_trade_cost(gross, "SELL", as_of_date, code, strategy_args)
        projected_holdings[code] = int(projected_holdings.get(code, 0)) - shares
        if projected_holdings[code] <= 0:
            projected_holdings.pop(code, None)
        projected_cash += gross - fee
        item = dict(order)
        item.update({"shares": shares, "gross_amount": gross, "estimated_fee": fee, "projected_cash_after": projected_cash})
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
            gross = shares * float(order["indicative_price"])
            fee = configured_trade_cost(gross, "BUY", as_of_date, code, strategy_args)
            if gross + fee <= projected_cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            warnings.append(f"{code}: projected cash was insufficient, so the buy order was omitted.")
            continue
        gross = shares * float(order["indicative_price"])
        if gross < float(order["trade_value_floor"]):
            continue
        fee = configured_trade_cost(gross, "BUY", as_of_date, code, strategy_args)
        projected_cash -= gross + fee
        projected_holdings[code] = int(projected_holdings.get(code, 0)) + shares
        item = dict(order)
        item.update({"shares": shares, "gross_amount": gross, "estimated_fee": fee, "projected_cash_after": projected_cash})
        executed.append(item)

    if rejected_by_floor:
        buy_rejections = [item for item in rejected_by_floor if item[1] == "BUY"]
        sell_rejections = [item for item in rejected_by_floor if item[1] == "SELL"]
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
                "code",
                "name",
                "side",
                "transition_type",
                "trade_value_floor",
                "shares",
                "reference_close",
                "indicative_price",
                "gross_amount",
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
        market_state = base.build_market_state(prices, strategy_args)
        financial = base.load_financial_factors(conn)
        industry_events = base.load_industry_event_scores(strategy_args.industry_event_scores)
        event_regime = base.load_event_regime_signals(strategy_args.event_regime_signals)
        stored_peak = safe_float(account_state.get("peak_portfolio_value"), np.nan)
        requested_peak = safe_float(args.peak_value, np.nan)
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
        features = base.feature_snapshot(prices, financial, as_of_date, strategy_args, industry_events)
        features = v2h.apply_v2_score(features, v2h.STATIC_COMPONENT_WEIGHTS, strategy_args)
        features["rank"] = np.arange(1, len(features) + 1)
        features["execution_close"] = features["code"].map(
            today.set_index("code")["close"].to_dict()
        )
        current_weights = close_weights(holdings, price_map, total_value, last_known_close)
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
        )
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
        )
        warnings.extend(order_warnings)
        projected_stock_value = float(projected_positions["projected_value_at_close"].sum()) if not projected_positions.empty else 0.0
        projected_total_value = projected_stock_value + projected_cash
        projected_equity_weight = projected_stock_value / projected_total_value if projected_total_value > 0 else 0.0
        target_shortfall = float(regime["target_equity_weight"]) - projected_equity_weight
        if target_shortfall > 0.05:
            warnings.append(
                f"Projected equity is {projected_equity_weight:.1%}, {target_shortfall:.1%} below the model target; "
                "the minimum-trade threshold and board lots are the main likely causes."
            )
        if pd.Timestamp(as_of_date).weekday() < 3:
            warnings.append("The latest database date is early in the week; confirm that this is the intended weekly decision date.")
        warnings.append(
            "reference_close is the latest unadjusted market close reconstructed from market "
            "value / shares. indicative_price is only a "
            "tick-rounded budget estimate using configured slippage, not a guaranteed or required order price."
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
            "indicative_price_slippage_bps": float(strategy_args.slippage_bps),
            "broker_commission_rate": float(
                getattr(strategy_args, "broker_commission_rate", 0.0)
            ),
            "broker_minimum_commission": float(
                getattr(strategy_args, "broker_minimum_commission", 0.0)
            ),
            "peak_portfolio_value": peak_value,
            "portfolio_drawdown": float(regime["portfolio_drawdown"]),
            "market_state": str(regime["market_state"]),
            "market_return_20": safe_float(regime["market_return_20"], np.nan),
            "market_breadth": safe_float(regime["market_breadth"], np.nan),
            "market_volatility_20": safe_float(regime["market_volatility_20"], np.nan),
            "target_equity_weight": float(regime["target_equity_weight"]),
            "current_equity_weight": float((total_value - cash) / total_value),
            "selected_count": int(target_meta.get("selected_count", 0)),
            "target_weight_sum": float(target_meta.get("target_weight_sum", 0.0)),
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
            "estimated_gross_traded": float(orders["gross_amount"].sum()) if not orders.empty else 0.0,
            "estimated_fees": float(orders["estimated_fee"].sum()) if not orders.empty else 0.0,
            "projected_cash": projected_cash,
            "projected_stock_value_at_close": projected_stock_value,
            "projected_total_value_at_close": projected_total_value,
            "projected_equity_weight": projected_equity_weight,
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
            "projected_positions": output_dir / f"{stem}_projected_positions.csv",
            "factor_ranking": output_dir / f"{stem}_factor_ranking.csv",
            "summary": output_dir / f"{stem}_summary.json",
            "workbook": output_dir / f"{stem}.xlsx",
        }
        orders.to_csv(paths["orders"], index=False, encoding="utf-8-sig")
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
        print(f"Target equity: {float(regime['target_equity_weight']):.2%}")
        print(f"Orders: {len(orders)}")
        print(f"Workbook: {paths['workbook']}")
        return summary, paths
    finally:
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
    parser.add_argument("--no-update-account-state", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
