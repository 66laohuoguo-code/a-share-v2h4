"""Shared A-share execution, cost, position, and reporting utilities."""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


POSITION_COLUMNS = ["code", "name", "shares", "cost_price"]

A_SHARE_TRADING_RULES = {
    "main_chinext_buy_minimum": 100,
    "main_chinext_buy_increment": 100,
    "star_buy_minimum": 200,
    "star_buy_increment": 1,
    "bse_buy_minimum": 100,
    "bse_buy_increment": 1,
    "price_tick": 0.01,
    "allow_odd_lot_sell": True,
    "enforce_t_plus_one": True,
    "main_board_price_limit_pct": 0.10,
    "st_price_limit_pct": 0.05,
    "star_chinext_price_limit_pct": 0.20,
    "bse_price_limit_pct": 0.30,
    "new_listing_no_price_limit_days": 5,
}

MANDATORY_A_SHARE_TRADING_COSTS = {
    "stamp_tax_sell_rate": 0.0005,
    "exchange_handling_fee_rate": 0.0000341,
    "securities_regulatory_fee_rate": 0.00002,
    "transfer_fee_rate": 0.00001,
    "broker_commission_rate": 0.0,
    "sources": [
        "MOF 2008 sell-side stamp-tax notice, https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm",
        "SSE 2012 A-share handling-fee reductions, https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988543.shtml",
        "SSE 2012 regulatory-fee adjustment, https://www.sse.com.cn/lawandrules/guide/other/c/c_20230116_5312163.shtml",
        "SSE/SZSE/CSDC 2015 handling and transfer-fee adjustment, https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988866.shtml",
        "MOF/SAT Announcement No.39 of 2023: securities transaction stamp tax halved from 2023-08-28, https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm",
        "SZSE fee table: A-share handling fee 0.0341 per mille both sides, https://www.szse.cn/marketServices/deal/payFees/",
        "CSRC 2023 handling-fee adjustment, https://www.csrc.gov.cn/csrc/c100028/c7426794/content.shtml",
        "ChinaClear A-share transfer fee table, https://www.chinaclear.cn/zdjs/editor_file/20220701154723234.pdf",
    ],
}

MANDATORY_A_SHARE_TRADING_COST_SCHEDULE = {
    "stamp_tax_sell_rate": [
        {"start_date": "1900-01-01", "end_date": "2023-08-27", "rate": 0.001},
        {"start_date": "2023-08-28", "end_date": None, "rate": 0.0005},
    ],
    "sh_exchange_handling_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2012-05-31", "rate": 0.0001100},
        {"start_date": "2012-06-01", "end_date": "2012-08-31", "rate": 0.0000870},
        {"start_date": "2012-09-01", "end_date": "2015-07-31", "rate": 0.0000696},
        {"start_date": "2015-08-01", "end_date": "2023-08-27", "rate": 0.0000487},
        {"start_date": "2023-08-28", "end_date": None, "rate": 0.0000341},
    ],
    "sz_exchange_handling_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2012-05-31", "rate": 0.0001220},
        {"start_date": "2012-06-01", "end_date": "2012-08-31", "rate": 0.0000870},
        {"start_date": "2012-09-01", "end_date": "2015-07-31", "rate": 0.0000696},
        {"start_date": "2015-08-01", "end_date": "2023-08-27", "rate": 0.0000487},
        {"start_date": "2023-08-28", "end_date": None, "rate": 0.0000341},
    ],
    "bse_exchange_handling_fee_rate": [
        {"start_date": "2021-11-15", "end_date": "2022-11-30", "rate": 0.0005},
        {"start_date": "2022-12-01", "end_date": "2023-08-27", "rate": 0.00025},
        {"start_date": "2023-08-28", "end_date": None, "rate": 0.000125},
    ],
    "sh_transfer_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2015-07-31", "rate": 0.0},
        {"start_date": "2015-08-01", "end_date": "2022-04-28", "rate": 0.00002},
        {"start_date": "2022-04-29", "end_date": None, "rate": 0.00001},
    ],
    "sz_transfer_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2015-07-31", "rate": 0.0000255},
        {"start_date": "2015-08-01", "end_date": "2022-04-28", "rate": 0.00002},
        {"start_date": "2022-04-29", "end_date": None, "rate": 0.00001},
    ],
    "bse_transfer_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2022-04-28", "rate": 0.00002},
        {"start_date": "2022-04-29", "end_date": None, "rate": 0.00001},
    ],
    "sh_par_value_transfer_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2012-05-31", "rate": 0.0005},
        {"start_date": "2012-06-01", "end_date": "2012-08-31", "rate": 0.000375},
        {"start_date": "2012-09-01", "end_date": "2015-07-31", "rate": 0.0003},
        {"start_date": "2015-08-01", "end_date": None, "rate": 0.0},
    ],
    "securities_regulatory_fee_rate": [
        {"start_date": "1900-01-01", "end_date": "2011-12-31", "rate": 0.00004},
        {"start_date": "2012-01-01", "end_date": None, "rate": 0.00002},
    ],
    "broker_commission_rate": 0.0,
}


def normalize_code(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "cash" or text == "现金":
        return "CASH"
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = str(int(float(text)))
    digits = "".join(char for char in text if char.isdigit())
    return digits.zfill(6) if digits else None


def load_positions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Positions file does not exist: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    if frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS), 0.0

    frame.columns = [str(column).strip() for column in frame.columns]
    if "code" not in frame.columns or "shares" not in frame.columns:
        raise ValueError("Positions file must contain at least code and shares columns")
    frame["code"] = frame["code"].map(normalize_code)
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    if "name" not in frame.columns:
        frame["name"] = ""
    if "cost_price" not in frame.columns:
        frame["cost_price"] = np.nan
    frame["cost_price"] = pd.to_numeric(frame["cost_price"], errors="coerce")

    cash_rows = frame["code"].eq("CASH")
    cash = float(frame.loc[cash_rows, "shares"].sum()) if cash_rows.any() else 0.0
    frame = frame.loc[~cash_rows].dropna(subset=["code"])
    frame = frame.loc[frame["shares"] != 0].copy()
    if frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS), cash
    grouped = (
        frame.groupby("code", as_index=False)
        .agg({"name": "last", "shares": "sum", "cost_price": "last"})
        .sort_values("code")
    )
    return grouped[POSITION_COLUMNS], cash


def classify_a_share_board(code):
    code = str(code).zfill(6)
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301")):
        return "CHINEXT"
    if code.startswith(("8", "4", "920")):
        return "BSE"
    return "MAIN"


def buy_order_size_rules(code):
    board = classify_a_share_board(code)
    if board == "STAR":
        return A_SHARE_TRADING_RULES["star_buy_minimum"], A_SHARE_TRADING_RULES["star_buy_increment"]
    if board == "BSE":
        return A_SHARE_TRADING_RULES["bse_buy_minimum"], A_SHARE_TRADING_RULES["bse_buy_increment"]
    return (
        A_SHARE_TRADING_RULES["main_chinext_buy_minimum"],
        A_SHARE_TRADING_RULES["main_chinext_buy_increment"],
    )


def round_price_to_tick(price, side):
    """Round an indicative A-share price conservatively to the valid price tick."""
    price = float(price)
    tick = float(A_SHARE_TRADING_RULES["price_tick"])
    units = price / tick
    if str(side).upper() == "BUY":
        result = math.ceil(units - 1e-10) * tick
    else:
        result = math.floor(units + 1e-10) * tick
    return round(result, 2)


def round_target_shares_for_code(target_value, price, code):
    if price is None or pd.isna(price) or price <= 0:
        return 0
    minimum, increment = buy_order_size_rules(code)
    raw = int(math.floor(float(target_value) / float(price)))
    if raw < minimum:
        return 0
    return int(minimum + math.floor((raw - minimum) / increment) * increment)


def round_portfolio_target_shares(target_weights, portfolio_value, prices):
    """Round a target portfolio by lots while keeping total equity close to target."""
    portfolio_value = max(0.0, float(portfolio_value))
    shares_by_code = {}
    candidates = []
    rounded_value = 0.0
    target_value = 0.0

    for code, weight in target_weights.items():
        code = str(code).zfill(6)
        price = prices.get(code)
        if price is None or pd.isna(price) or float(price) <= 0:
            continue
        price = float(price)
        desired_value = max(0.0, float(weight)) * portfolio_value
        shares = round_target_shares_for_code(desired_value, price, code)
        shares_by_code[code] = shares
        rounded_value += shares * price
        target_value += desired_value
        minimum, increment = buy_order_size_rules(code)
        extra_shares = minimum if shares == 0 else increment
        lot_value = extra_shares * price
        remainder = desired_value - shares * price
        candidates.append((remainder / lot_value if lot_value > 0 else 0.0, code, extra_shares, lot_value))

    residual = max(0.0, target_value - rounded_value)
    for _, code, extra_shares, lot_value in sorted(candidates, reverse=True):
        if abs(residual - lot_value) + 1e-8 >= abs(residual):
            continue
        shares_by_code[code] = int(shares_by_code.get(code, 0)) + int(extra_shares)
        residual -= lot_value
    return shares_by_code


def trade_value_floor(
    portfolio_value,
    current_shares,
    target_shares,
    min_trade_value,
    min_trade_weight=0.0,
    entry_exit_min_trade_value=None,
    entry_exit_min_trade_weight=0.0,
):
    """Return the applicable order-value floor and transition type."""
    current_shares = max(0, int(current_shares))
    target_shares = max(0, int(target_shares))
    if current_shares == 0 and target_shares > 0:
        transition = "ENTRY"
    elif current_shares > 0 and target_shares == 0:
        transition = "EXIT"
    else:
        transition = "ADJUST"

    if transition == "EXIT":
        return 0.0, transition
    if transition == "ENTRY" and entry_exit_min_trade_value is not None:
        fixed_floor = max(0.0, float(entry_exit_min_trade_value))
        weight_floor = max(0.0, float(entry_exit_min_trade_weight))
    else:
        fixed_floor = max(0.0, float(min_trade_value))
        weight_floor = max(0.0, float(min_trade_weight))
    return max(fixed_floor, max(0.0, float(portfolio_value)) * weight_floor), transition


def apply_risk_alignment_trade_floor(
    order_floor,
    portfolio_value,
    side,
    current_equity_weight,
    target_equity_weight,
    risk_rebalance_band,
    risk_reduction_min_trade_weight,
    risk_increase_min_trade_weight,
):
    """Lower order floors only when aggregate equity is outside its risk band."""
    side = str(side or "").upper()
    current = float(current_equity_weight)
    target = float(target_equity_weight)
    band = max(0.0, float(risk_rebalance_band))
    if side == "SELL" and current > target + band:
        mode = "REDUCE"
        weight_floor = risk_reduction_min_trade_weight
    elif side == "BUY" and current < target - band:
        mode = "INCREASE"
        weight_floor = risk_increase_min_trade_weight
    else:
        return max(0.0, float(order_floor)), ""
    risk_floor = max(0.0, float(portfolio_value)) * max(0.0, float(weight_floor))
    return min(max(0.0, float(order_floor)), risk_floor), mode


def _schedule_rate(schedule, trade_date):
    date = str(trade_date or datetime.now().date().isoformat())[:10]
    for row in schedule:
        if date >= row["start_date"] and (row["end_date"] is None or date <= row["end_date"]):
            return float(row["rate"])
    return float(schedule[-1]["rate"])


def _environment_float(name, default=0.0):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return max(0.0, float(value))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric, got: {value}") from exc


def mandatory_trade_cost_components(
    side,
    trade_date=None,
    code=None,
    broker_commission_rate=None,
):
    side = str(side or "").upper()
    board = classify_a_share_board(code) if code else "SH_SZ"
    normalized_code = str(code or "").zfill(6)
    exchange = "SH" if normalized_code.startswith("6") else "SZ"
    if board == "BSE":
        handling_key = "bse_exchange_handling_fee_rate"
        transfer_key = "bse_transfer_fee_rate"
    elif exchange == "SH":
        handling_key = "sh_exchange_handling_fee_rate"
        transfer_key = "sh_transfer_fee_rate"
    else:
        handling_key = "sz_exchange_handling_fee_rate"
        transfer_key = "sz_transfer_fee_rate"
    schedule = MANDATORY_A_SHARE_TRADING_COST_SCHEDULE
    configured_commission_rate = (
        _environment_float(
            "A_SHARE_BROKER_COMMISSION_RATE", schedule["broker_commission_rate"]
        )
        if broker_commission_rate is None
        else max(0.0, float(broker_commission_rate))
    )
    return {
        "stamp_tax": _schedule_rate(schedule["stamp_tax_sell_rate"], trade_date) if side == "SELL" else 0.0,
        "exchange_handling_fee": _schedule_rate(schedule[handling_key], trade_date),
        "securities_regulatory_fee": _schedule_rate(
            schedule["securities_regulatory_fee_rate"], trade_date
        ),
        "transfer_fee": _schedule_rate(schedule[transfer_key], trade_date),
        "broker_commission": configured_commission_rate,
    }


def mandatory_trade_cost_rate(
    side,
    trade_date=None,
    code=None,
    broker_commission_rate=None,
):
    return float(
        sum(
            mandatory_trade_cost_components(
                side,
                trade_date,
                code,
                broker_commission_rate,
            ).values()
        )
    )


def mandatory_trade_cost(
    amount,
    side,
    trade_date=None,
    code=None,
    broker_commission_rate=None,
    broker_minimum_commission=None,
    shares=None,
):
    gross = abs(float(amount or 0.0))
    if gross <= 0:
        return 0.0
    components = mandatory_trade_cost_components(
        side,
        trade_date,
        code,
        broker_commission_rate,
    )
    commission_rate = float(components.pop("broker_commission", 0.0))
    statutory_cost = gross * float(sum(components.values()))
    normalized_code = str(code or "").zfill(6)
    if normalized_code.startswith("6") and str(trade_date or "9999-12-31")[:10] < "2015-08-01":
        if shares is not None:
            statutory_cost += abs(int(shares)) * _schedule_rate(
                MANDATORY_A_SHARE_TRADING_COST_SCHEDULE[
                    "sh_par_value_transfer_fee_rate"
                ],
                trade_date,
            )
    minimum_commission = (
        _environment_float("A_SHARE_BROKER_MIN_COMMISSION", 0.0)
        if broker_minimum_commission is None
        else max(0.0, float(broker_minimum_commission))
    )
    commission = max(gross * commission_rate, minimum_commission) if commission_rate > 0 else 0.0
    return statutory_cost + commission


def trading_cost_snapshot(
    trade_date=None,
    code=None,
    broker_commission_rate=None,
    broker_minimum_commission=None,
):
    effective_date = str(trade_date or datetime.now().date().isoformat())[:10]
    snapshot = dict(MANDATORY_A_SHARE_TRADING_COSTS)
    snapshot.update(
        {
            "effective_date": effective_date,
            "board": classify_a_share_board(code) if code else "SH_SZ",
            "broker_commission_rate": mandatory_trade_cost_components(
                "BUY", effective_date, code, broker_commission_rate
            )["broker_commission"],
            "buy_total_rate": mandatory_trade_cost_rate(
                "BUY", effective_date, code, broker_commission_rate
            ),
            "sell_total_rate": mandatory_trade_cost_rate(
                "SELL", effective_date, code, broker_commission_rate
            ),
            "broker_minimum_commission_per_order": (
                _environment_float("A_SHARE_BROKER_MIN_COMMISSION", 0.0)
                if broker_minimum_commission is None
                else max(0.0, float(broker_minimum_commission))
            ),
            "historical_schedule": MANDATORY_A_SHARE_TRADING_COST_SCHEDULE,
        }
    )
    snapshot["round_trip_rate"] = snapshot["buy_total_rate"] + snapshot["sell_total_rate"]
    return snapshot


def should_rebalance_on_date(schedule, offset, rebalance_every_n_days, all_dates, date_to_index, decision_date):
    schedule = str(schedule or "every_n_days").strip().lower()
    if schedule in {"every_n_days", "every_n", "n_days"}:
        return offset % max(1, int(rebalance_every_n_days)) == 0
    decision_index = date_to_index.get(decision_date)
    if decision_index is None:
        return False
    if decision_index + 1 >= len(all_dates):
        return True
    current_day = pd.Timestamp(decision_date)
    next_day = pd.Timestamp(all_dates[decision_index + 1])
    if schedule in {"week_end", "weekly"}:
        return current_day.isocalendar()[:2] != next_day.isocalendar()[:2]
    if schedule in {"month_end", "monthly"}:
        return str(decision_date)[:7] != str(all_dates[decision_index + 1])[:7]
    raise ValueError(f"Unsupported rebalance schedule: {schedule}")


def safe_sheet_name(name):
    invalid = set('[]:*?/\\')
    cleaned = "".join("_" if char in invalid else char for char in str(name))
    return cleaned[:31] or "Sheet1"


def write_excel_workbook(path: Path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            frame.to_excel(writer, index=False, sheet_name=safe_sheet_name(sheet_name))
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row >= 1 and worksheet.max_column >= 1:
                worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) for cell in column_cells[:200] if cell.value is not None]
                width = min(max([len(value) for value in values] + [8]) + 2, 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
