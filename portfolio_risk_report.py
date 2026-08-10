"""Create an account-level report from the weekly A-share risk model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from ashare_risk_model import STYLE_FACTORS
from ashare_utils import load_positions, write_excel_workbook
from risk_model_reporting import (
    calculate_portfolio_risk,
    complete_exposures_for_codes,
    json_ready,
    latest_model_date,
    load_exposure_and_specific,
    load_factor_covariance,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MARKET_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_stock_daily_live.sqlite"
)
DEFAULT_RISK_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "portfolio_risk"


def load_holding_prices(conn, codes, as_of_date):
    codes = sorted({str(code).zfill(6) for code in codes})
    if not codes:
        return pd.DataFrame(
            columns=("code", "market_name", "price_date", "price")
        )
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT d.code, d.name AS market_name, d.trade_date AS price_date,
               COALESCE(d.raw_close, d.close) AS price
        FROM stock_daily d
        JOIN (
            SELECT code, MAX(trade_date) AS price_date
            FROM stock_daily
            WHERE code IN ({placeholders}) AND trade_date<=?
            GROUP BY code
        ) x ON x.code=d.code AND x.price_date=d.trade_date
        ORDER BY d.code
    """
    return pd.read_sql_query(
        query,
        conn,
        params=(*codes, str(as_of_date)),
    )


def complete_holding_exposures(holdings, exposure, covariance):
    del covariance
    return complete_exposures_for_codes(holdings["code"], exposure)


def prepare_holdings(positions, cash, market_conn, model_date):
    prices = load_holding_prices(market_conn, positions["code"], model_date)
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    holdings = positions.copy()
    holdings["code"] = holdings["code"].astype(str).str.zfill(6)
    holdings = holdings.merge(prices, on="code", how="left")
    holdings["price"] = pd.to_numeric(holdings["price"], errors="coerce")
    holdings["cost_price"] = pd.to_numeric(
        holdings["cost_price"], errors="coerce"
    )
    holdings["price_source"] = np.where(
        holdings["price"].notna(), "market_raw_close", "cost_price_fallback"
    )
    holdings["price"] = holdings["price"].fillna(holdings["cost_price"])
    missing_price = holdings.loc[
        holdings["price"].isna() | (holdings["price"] <= 0), "code"
    ].tolist()
    if missing_price:
        raise ValueError(
            "No market or cost price is available for: "
            + ", ".join(missing_price)
        )
    holdings["market_value"] = holdings["shares"] * holdings["price"]
    total_value = float(holdings["market_value"].sum() + cash)
    if total_value <= 0:
        raise ValueError("Current portfolio value is zero")
    holdings["portfolio_weight"] = holdings["market_value"] / total_value
    holdings["equity_weight"] = (
        holdings["market_value"] / float(holdings["market_value"].sum())
        if holdings["market_value"].sum() != 0
        else 0.0
    )
    position_names = holdings["name"].fillna("").astype(str)
    holdings["display_name"] = position_names.where(
        position_names.str.strip().ne(""),
        holdings["market_name"],
    )
    return holdings, total_value


def build_report(args):
    risk_conn = sqlite3.connect(args.risk_database)
    market_conn = sqlite3.connect(args.market_database)
    try:
        model_date = latest_model_date(risk_conn, args.as_of_date)
        covariance = load_factor_covariance(risk_conn, model_date)
        model_exposure = load_exposure_and_specific(risk_conn, model_date)
        positions, positions_cash = load_positions(args.positions)
        cash = float(positions_cash if args.cash is None else args.cash)
        holdings, total_value = prepare_holdings(
            positions, cash, market_conn, model_date
        )
        completed_exposure = complete_holding_exposures(
            holdings, model_exposure, covariance
        )
        weights = holdings.set_index("code")["portfolio_weight"]
        specific = completed_exposure.set_index("code")["specific_variance"]
        risk = calculate_portfolio_risk(
            weights,
            completed_exposure,
            covariance,
            specific,
        )
    finally:
        risk_conn.close()
        market_conn.close()

    holding_exposure = completed_exposure[
        [
            "code",
            "industry_group",
            "exposure_source",
            "specific_variance",
            "specific_volatility",
            *STYLE_FACTORS,
        ]
    ]
    holdings = holdings.merge(holding_exposure, on="code", how="left")
    holdings = holdings.merge(
        risk["stock"].drop(columns="portfolio_weight"),
        on="code",
        how="left",
    )
    holdings = holdings.sort_values(
        "total_variance_contribution", ascending=False
    ).reset_index(drop=True)

    factor = risk["factor"].copy()
    factor["annual_volatility_contribution"] = np.where(
        risk["annual_volatility"] > 0,
        factor["annual_variance_contribution"] / risk["annual_volatility"],
        np.nan,
    )
    factor = factor.sort_values(
        "annual_variance_contribution", ascending=False
    ).reset_index(drop=True)

    industry = (
        holdings.groupby("industry_group", dropna=False)
        .agg(
            stock_count=("code", "count"),
            market_value=("market_value", "sum"),
            portfolio_weight=("portfolio_weight", "sum"),
        )
        .reset_index()
        .sort_values("portfolio_weight", ascending=False)
    )
    equity_value = float(holdings["market_value"].sum())
    exact_exposure_value = float(
        holdings.loc[
            holdings["exposure_source"].eq("model"), "market_value"
        ].sum()
    )
    exact_price_value = float(
        holdings.loc[
            holdings["price_source"].eq("market_raw_close"), "market_value"
        ].sum()
    )
    weekly_volatility = risk["weekly_volatility"]
    summary = {
        "account_id": args.account_id or args.positions.parent.name,
        "positions_file": str(args.positions),
        "risk_database": str(args.risk_database),
        "market_database": str(args.market_database),
        "model_date": model_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_portfolio_value": total_value,
        "equity_value": equity_value,
        "cash": cash,
        "equity_weight": equity_value / total_value,
        "stock_count": len(holdings),
        "exact_risk_exposure_coverage_of_equity": (
            exact_exposure_value / equity_value if equity_value else 0.0
        ),
        "market_price_coverage_of_equity": (
            exact_price_value / equity_value if equity_value else 0.0
        ),
        "predicted_annual_volatility": risk["annual_volatility"],
        "predicted_weekly_volatility": weekly_volatility,
        "common_risk_share": (
            risk["common_variance"] / risk["annual_variance"]
            if risk["annual_variance"] > 0
            else None
        ),
        "specific_risk_share": (
            risk["specific_variance"] / risk["annual_variance"]
            if risk["annual_variance"] > 0
            else None
        ),
        "normal_approximation_one_week_var_95": (
            1.6448536269514722 * weekly_volatility * total_value
        ),
        "normal_approximation_one_week_var_99": (
            2.3263478740408408 * weekly_volatility * total_value
        ),
        "risk_note": (
            "VaR is a model estimate under a normal approximation, not a "
            "maximum-loss guarantee."
        ),
    }
    summary_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    return summary, summary_frame, holdings, factor, industry


def run(args):
    summary, summary_frame, holdings, factor, industry = build_report(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    account_id = str(summary["account_id"])
    output_dir = args.output_dir / account_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_risk_{account_id}_{summary['model_date'].replace('-', '')}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    xlsx_path = output_dir / f"{stem}.xlsx"
    json_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_excel_workbook(
        xlsx_path,
        [
            ("summary", summary_frame),
            ("holdings", holdings),
            ("factor_risk", factor),
            ("industry_exposure", industry),
        ],
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"Summary JSON: {json_path}")
    print(f"Portfolio risk workbook: {xlsx_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create an account-level weekly risk-model report."
    )
    parser.add_argument(
        "--risk-database", type=Path, default=DEFAULT_RISK_DATABASE
    )
    parser.add_argument(
        "--market-database", type=Path, default=DEFAULT_MARKET_DATABASE
    )
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--account-id")
    parser.add_argument("--as-of-date")
    parser.add_argument(
        "--cash",
        type=float,
        help="Override the CASH row in the positions file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    for name in ("risk_database", "market_database", "positions"):
        value = getattr(args, name).resolve()
        setattr(args, name, value)
        if not value.exists():
            parser.error(f"{name.replace('_', ' ')} does not exist: {value}")
    args.output_dir = args.output_dir.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
