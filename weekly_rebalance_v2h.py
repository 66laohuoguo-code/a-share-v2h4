"""Create an indicative next-session rebalance plan from the V2H strategy.

The script never submits orders. It uses only data available through the chosen
as-of close and writes a reviewable CSV/XLSX/JSON plan for the next session.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

import factor_rank_backtest as base
import factor_rank_backtest_v2h as v2h
from ashare_utils import (
    buy_order_size_rules,
    load_positions,
    mandatory_trade_cost,
    round_target_shares_for_code,
    trading_cost_snapshot,
    write_excel_workbook,
)


DEFAULT_DATABASE = Path("data/processed/stock_daily.sqlite")
DEFAULT_POSITIONS = Path("data/input/positions.csv")
DEFAULT_ACCOUNT_STATE = Path("data/input/account_state.json")
DEFAULT_STRATEGY_CONFIG = Path("config/v2h4_strategy.json")
DEFAULT_OUTPUT_DIR = Path("outputs/weekly_rebalance_v2h4")


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


def load_account_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_account_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(state)), ensure_ascii=False, indent=2), encoding="utf-8")


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
    min_trade_threshold = max(
        float(strategy_args.min_trade_value),
        float(total_value) * max(0.0, float(getattr(strategy_args, "min_trade_weight", 0.0))),
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
        indicative_side = "BUY" if target_weight * total_value > current_shares * close else "SELL"
        indicative_price = close * (1.0 + slippage if indicative_side == "BUY" else 1.0 - slippage)
        target_shares = (
            round_target_shares_for_code(target_weight * total_value, indicative_price, code)
            if target_weight > 0
            else 0
        )
        difference = int(target_shares - current_shares)
        if difference == 0:
            continue
        side = "BUY" if difference > 0 else "SELL"
        trade_shares = limited_trade_shares(
            code,
            abs(difference),
            side,
            liquidating=(side == "SELL" and target_shares == 0),
        )
        if trade_shares <= 0:
            continue
        price = close * (1.0 + slippage if side == "BUY" else 1.0 - slippage)
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
        if gross < min_trade_threshold:
            continue
        feature = feature_by_code.get(code, {})
        planned.append(
            {
                "as_of_date": as_of_date,
                "execution_session": "NEXT_TRADING_SESSION",
                "code": code,
                "name": str(row.get("name", "")),
                "side": side,
                "shares": int(trade_shares),
                "reference_close": close,
                "indicative_price": price,
                "gross_amount": gross,
                "estimated_fee": mandatory_trade_cost(gross, side, as_of_date, code),
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

    executed = []
    projected_cash = float(cash)
    projected_holdings = {str(code): int(shares) for code, shares in holdings.items() if int(shares) > 0}
    for order in [item for item in planned if item["side"] == "SELL"]:
        code = order["code"]
        shares = min(int(order["shares"]), int(projected_holdings.get(code, 0)))
        if shares <= 0:
            continue
        gross = shares * float(order["indicative_price"])
        fee = mandatory_trade_cost(gross, "SELL", as_of_date, code)
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
            fee = mandatory_trade_cost(gross, "BUY", as_of_date, code)
            if gross + fee <= projected_cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            warnings.append(f"{code}: projected cash was insufficient, so the buy order was omitted.")
            continue
        gross = shares * float(order["indicative_price"])
        if gross < min_trade_threshold:
            continue
        fee = mandatory_trade_cost(gross, "BUY", as_of_date, code)
        projected_cash -= gross + fee
        projected_holdings[code] = int(projected_holdings.get(code, 0)) + shares
        item = dict(order)
        item.update({"shares": shares, "gross_amount": gross, "estimated_fee": fee, "projected_cash_after": projected_cash})
        executed.append(item)

    orders = pd.DataFrame(executed)
    if orders.empty:
        orders = pd.DataFrame(
            columns=[
                "as_of_date",
                "execution_session",
                "code",
                "name",
                "side",
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
    strategy_argv = ["--strategy-config", str(args.strategy_config)]
    strategy_args = v2h.parse_args(strategy_argv)
    conn = sqlite3.connect(args.database)
    warnings = []
    try:
        as_of_date = latest_as_of_date(conn, args.as_of_date)
        dates = base.trading_dates(conn)
        date_to_index = {date: index for index, date in enumerate(dates)}
        as_of_index = date_to_index[as_of_date]
        prehistory = max(int(strategy_args.feature_history_days), int(strategy_args.min_history_days) + 30, 320)
        history_start = dates[max(0, as_of_index - prehistory)]
        prices = base.load_prices(conn, history_start, as_of_date)
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        today = prices.loc[prices["trade_date"].astype(str).eq(as_of_date)].copy()
        price_map = today.set_index("code").to_dict("index")
        latest_rows = prices.sort_values(["code", "trade_date"]).groupby("code", as_index=False).tail(1)
        last_known_close = latest_rows.set_index("code")["close"].apply(safe_float).to_dict()
        market_state = base.build_market_state(prices, strategy_args)
        financial = base.load_financial_factors(conn)
        industry_events = base.load_industry_event_scores(strategy_args.industry_event_scores)
        event_regime = base.load_event_regime_signals(strategy_args.event_regime_signals)

        positions, positions_cash = load_positions(Path(args.positions))
        cash = float(args.cash) if args.cash is not None else float(positions_cash)
        holdings = {
            str(row.code).zfill(6): int(round(float(row.shares)))
            for row in positions.itertuples(index=False)
            if float(row.shares) > 0
        }
        total_value = base.value_portfolio(holdings, cash, price_map, "close", last_known_close)
        if total_value <= 0:
            raise ValueError("Current portfolio value is zero. Add CASH and/or stock positions before running.")

        account_state = load_account_state(Path(args.account_state))
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
        current_weights = close_weights(holdings, price_map, total_value, last_known_close)
        targets, target_meta = v2h.build_targets_v2(
            features,
            holdings,
            current_weights,
            strategy_args,
            float(regime["target_equity_weight"]),
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
        warnings.append("Indicative prices use the latest close plus/minus configured slippage; re-check limits, suspension and cash at the next open.")

        summary = {
            "strategy": str(getattr(strategy_args, "strategy_name", "V2H4")),
            "as_of_date": as_of_date,
            "database": str(args.database),
            "positions_file": str(args.positions),
            "current_cash": cash,
            "current_stock_value": total_value - cash,
            "current_total_value": total_value,
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
            "trading_costs": trading_cost_snapshot(as_of_date),
            "warnings": warnings,
        }

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"v2h4_rebalance_{as_of_date.replace('-', '')}_{stamp}"
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
                Path(args.account_state),
                {
                    "peak_portfolio_value": peak_value,
                    "last_portfolio_value": total_value,
                    "last_as_of_date": as_of_date,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )

        print(f"As-of date: {as_of_date}")
        print(f"Current value: {total_value:.2f}")
        print(f"Target equity: {float(regime['target_equity_weight']):.2%}")
        print(f"Orders: {len(orders)}")
        print(f"Workbook: {paths['workbook']}")
        return summary, paths
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate a V2H4 next-session rebalance plan from current positions.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--account-state", type=Path, default=DEFAULT_ACCOUNT_STATE)
    parser.add_argument("--strategy-config", type=Path, default=DEFAULT_STRATEGY_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of-date")
    parser.add_argument("--cash", type=float, help="Override CASH from the positions file for this run.")
    parser.add_argument("--peak-value", type=float, help="Override the stored historical peak portfolio value.")
    parser.add_argument("--no-update-account-state", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
