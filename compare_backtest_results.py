"""Summarize V2H validation runs, including a chosen holdout period."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_utils import write_excel_workbook


SUMMARY_KEYS = [
    "strategy",
    "strategy_name",
    "start_date",
    "end_date",
    "initial_cash",
    "final_value",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_no_risk_free",
    "max_drawdown",
    "trade_count",
    "total_gross_traded",
    "total_fees",
    "total_corporate_action_cash",
    "corporate_action_share_change_count",
    "average_daily_turnover",
    "average_rebalance_turnover",
    "average_target_equity_weight",
    "average_actual_equity_weight",
    "min_trade_value",
    "min_trade_weight",
    "entry_exit_min_trade_value",
    "entry_exit_min_trade_weight",
    "enable_lot_aware_selection",
    "lot_aware_min_holdings",
    "lot_aware_max_stock_weight",
    "lot_aware_max_industry_weight",
    "slippage_bps",
]


def holdout_metrics(equity, start_date, end_date):
    rows = equity.loc[
        equity["trade_date"].astype(str).between(str(start_date), str(end_date), inclusive="both")
    ].copy()
    if rows.empty:
        return {}
    daily = pd.to_numeric(rows["daily_return"], errors="coerce").dropna()
    total_return = float((1.0 + daily).prod() - 1.0) if len(daily) else np.nan
    annual_return = (1.0 + total_return) ** (244 / len(daily)) - 1.0 if len(daily) and total_return > -1 else np.nan
    annual_vol = float(daily.std(ddof=1) * math.sqrt(244)) if len(daily) > 1 else np.nan
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "holdout_start": str(rows["trade_date"].iloc[0]),
        "holdout_end": str(rows["trade_date"].iloc[-1]),
        "holdout_days": int(len(rows)),
        "holdout_return": total_return,
        "holdout_annualized_return": annual_return,
        "holdout_annualized_volatility": annual_vol,
        "holdout_sharpe_no_risk_free": float(annual_return / annual_vol)
        if pd.notna(annual_return) and pd.notna(annual_vol) and annual_vol > 0
        else np.nan,
        "holdout_max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "holdout_average_equity_weight": float(
            pd.to_numeric(rows.get("actual_equity_weight"), errors="coerce").mean()
        )
        if "actual_equity_weight" in rows
        else np.nan,
        "holdout_gross_traded": float(pd.to_numeric(rows.get("gross_traded"), errors="coerce").sum())
        if "gross_traded" in rows
        else np.nan,
        "holdout_fees": float(pd.to_numeric(rows.get("fees"), errors="coerce").sum())
        if "fees" in rows
        else np.nan,
    }


def find_equity_path(summary_path):
    prefix = summary_path.name.removesuffix("_summary.json")
    candidate = summary_path.with_name(f"{prefix}_equity_curve.csv")
    return candidate if candidate.exists() else None


def run(args):
    rows = []
    for summary_path in sorted(Path(args.root).rglob("*_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {key: payload.get(key) for key in SUMMARY_KEYS}
        row["run_directory"] = str(summary_path.parent)
        row["summary_file"] = str(summary_path)
        equity_path = find_equity_path(summary_path)
        if equity_path:
            equity = pd.read_csv(equity_path, encoding="utf-8-sig")
            row.update(holdout_metrics(equity, args.holdout_start, args.holdout_end))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary JSON files found below {args.root}")
    frame = pd.DataFrame(rows).sort_values(
        ["holdout_return", "annualized_return"], ascending=False, na_position="last"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    write_excel_workbook(output.with_suffix(".xlsx"), [("comparison", frame)])
    print(frame.to_string(index=False))
    print(f"Comparison workbook: {output.with_suffix('.xlsx')}")
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare V2H backtest summary files.")
    parser.add_argument("--root", type=Path, default=Path("outputs/validation_full"))
    parser.add_argument("--holdout-start", default="2026-04-01")
    parser.add_argument("--holdout-end", default="2026-07-07")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation_full/v2h4_comparison.xlsx"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
