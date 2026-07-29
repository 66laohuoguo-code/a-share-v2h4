"""Compare V2H4 strategies across account sizes and execution stress scenarios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_utils import write_excel_workbook


SUMMARY_FIELDS = [
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
    "average_daily_turnover",
    "average_rebalance_turnover",
    "average_target_equity_weight",
    "average_actual_equity_weight",
    "target_count",
    "min_trade_weight",
    "entry_exit_min_trade_weight",
    "slippage_bps",
    "broker_commission_rate",
    "broker_minimum_commission",
    "max_participation_rate",
]


def companion_path(summary_path: Path, suffix: str) -> Path | None:
    prefix = summary_path.name.removesuffix("_summary.json")
    candidate = summary_path.with_name(f"{prefix}_{suffix}")
    return candidate if candidate.exists() else None


def infer_run_identity(summary_path: Path, root: Path) -> tuple[str, str]:
    try:
        parts = summary_path.parent.relative_to(root).parts
    except ValueError:
        return "unknown", summary_path.parent.name
    if len(parts) >= 3 and parts[0].startswith("capital_"):
        return parts[1], parts[2]
    return "unknown", summary_path.parent.name


def holdout_metrics(equity_path: Path | None, start_date: str, end_date: str) -> dict[str, Any]:
    if equity_path is None:
        return {}
    equity = pd.read_csv(equity_path, encoding="utf-8-sig")
    rows = equity.loc[
        equity["trade_date"].astype(str).between(str(start_date), str(end_date), inclusive="both")
    ].copy()
    if rows.empty:
        return {}
    daily = pd.to_numeric(rows["daily_return"], errors="coerce").dropna()
    if daily.empty:
        return {}
    total_return = float((1.0 + daily).prod() - 1.0)
    annualized_return = (
        (1.0 + total_return) ** (244.0 / len(daily)) - 1.0 if total_return > -1.0 else np.nan
    )
    annualized_volatility = float(daily.std(ddof=1) * math.sqrt(244.0)) if len(daily) > 1 else np.nan
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "holdout_start": str(rows["trade_date"].iloc[0]),
        "holdout_end": str(rows["trade_date"].iloc[-1]),
        "holdout_days": int(len(rows)),
        "holdout_return": total_return,
        "holdout_annualized_return": annualized_return,
        "holdout_annualized_volatility": annualized_volatility,
        "holdout_sharpe_no_risk_free": (
            float(annualized_return / annualized_volatility)
            if pd.notna(annualized_return)
            and pd.notna(annualized_volatility)
            and annualized_volatility > 0
            else np.nan
        ),
        "holdout_max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "holdout_fees": float(pd.to_numeric(rows.get("fees"), errors="coerce").sum())
        if "fees" in rows
        else np.nan,
    }


def trade_metrics(trades_path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if trades_path is None:
        return {}
    trades = pd.read_csv(trades_path, encoding="utf-8-sig")
    if trades.empty:
        return {
            "average_order_value": 0.0,
            "median_order_value": 0.0,
            "minimum_commission_order_count": 0,
            "minimum_commission_order_share": 0.0,
            "estimated_broker_commission": 0.0,
            "participation_p95": 0.0,
            "participation_max": 0.0,
        }

    gross = pd.to_numeric(trades["gross_amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
    commission_rate = float(payload.get("broker_commission_rate") or 0.0)
    minimum_commission = float(payload.get("broker_minimum_commission") or 0.0)
    proportional = gross * commission_rate
    minimum_hits = proportional < minimum_commission - 1e-10
    estimated_commission = np.maximum(proportional, minimum_commission)

    participation = pd.Series(dtype=float)
    if "avg_amount_for_cap" in trades:
        average_amount = pd.to_numeric(trades["avg_amount_for_cap"], errors="coerce")
        valid = average_amount > 0
        participation = gross.loc[valid] / average_amount.loc[valid]

    return {
        "average_order_value": float(gross.mean()),
        "median_order_value": float(gross.median()),
        "minimum_commission_order_count": int(minimum_hits.sum()),
        "minimum_commission_order_share": float(minimum_hits.mean()),
        "estimated_broker_commission": float(estimated_commission.sum()),
        "participation_p95": float(participation.quantile(0.95))
        if not participation.empty
        else np.nan,
        "participation_max": float(participation.max()) if not participation.empty else np.nan,
    }


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    cash = pd.to_numeric(result["initial_cash"], errors="coerce")
    result["fees_pct_initial_cash"] = pd.to_numeric(result["total_fees"], errors="coerce") / cash
    result["gross_traded_multiple"] = (
        pd.to_numeric(result["total_gross_traded"], errors="coerce") / cash
    )
    result["commission_pct_total_fees"] = (
        pd.to_numeric(result["estimated_broker_commission"], errors="coerce")
        / pd.to_numeric(result["total_fees"], errors="coerce").replace(0.0, np.nan)
    )
    return result


def rank_base_results(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base.empty:
        return base.copy(), pd.DataFrame()
    ranked_groups = []
    for _, group in base.groupby("initial_cash", sort=True):
        ranked = group.copy()
        ranking_rules = {
            "rank_annualized_return": ("annualized_return", False),
            "rank_sharpe": ("sharpe_no_risk_free", False),
            "rank_max_drawdown": ("max_drawdown", False),
            "rank_holdout_return": ("holdout_return", False),
            "rank_holdout_sharpe": ("holdout_sharpe_no_risk_free", False),
        }
        for output, (source, ascending) in ranking_rules.items():
            ranked[output] = pd.to_numeric(ranked[source], errors="coerce").rank(
                ascending=ascending,
                method="min",
                na_option="bottom",
            )
        rank_columns = list(ranking_rules)
        ranked["robust_rank_score"] = ranked[rank_columns].mean(axis=1)
        ranked["robust_rank"] = ranked["robust_rank_score"].rank(
            ascending=True,
            method="min",
        )
        ranked_groups.append(ranked)
    ranked_base = pd.concat(ranked_groups, ignore_index=True).sort_values(
        ["initial_cash", "robust_rank", "annualized_return"],
        ascending=[True, True, False],
    )
    winners = (
        ranked_base.loc[ranked_base["robust_rank"] == 1]
        .sort_values(["initial_cash", "annualized_return"], ascending=[True, False])
        .drop_duplicates("initial_cash")
        .reset_index(drop=True)
    )
    return ranked_base, winners


def build_methodology() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "robust_rank_score",
                "definition": (
                    "Mean rank of full-period annualized return, full-period Sharpe, "
                    "maximum drawdown, holdout return and holdout Sharpe. Lower is better."
                ),
            },
            {
                "item": "minimum_commission_order_share",
                "definition": (
                    "Share of orders where gross amount times broker commission rate is "
                    "below the configured per-order minimum commission."
                ),
            },
            {
                "item": "participation_p95 / participation_max",
                "definition": (
                    "Order gross amount divided by the stock's trailing average trading amount. "
                    "Stress scenarios lower the permitted participation rate."
                ),
            },
            {
                "item": "selection rule",
                "definition": (
                    "Do not select a method from annualized return alone. Require acceptable "
                    "drawdown, holdout behavior, fee drag and neighboring-capital robustness."
                ),
            },
        ]
    )


def run(args) -> pd.DataFrame:
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("*_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        scenario, variant = infer_run_identity(summary_path, root)
        row = {key: payload.get(key) for key in SUMMARY_FIELDS}
        row.update(
            {
                "scenario": scenario,
                "variant": variant,
                "run_directory": str(summary_path.parent),
                "summary_file": str(summary_path),
            }
        )
        row.update(
            holdout_metrics(
                companion_path(summary_path, "equity_curve.csv"),
                args.holdout_start,
                args.holdout_end,
            )
        )
        row.update(trade_metrics(companion_path(summary_path, "trades.csv"), payload))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary JSON files found below {root}")

    frame = add_derived_metrics(pd.DataFrame(rows))
    base = frame.loc[frame["scenario"] == "base"].copy()
    stress = frame.loc[frame["scenario"] != "base"].copy()
    ranked_base, winners = rank_base_results(base)
    annualized_matrix = ranked_base.pivot_table(
        index="initial_cash",
        columns="variant",
        values="annualized_return",
        aggfunc="first",
    ).reset_index()
    sharpe_matrix = ranked_base.pivot_table(
        index="initial_cash",
        columns="variant",
        values="sharpe_no_risk_free",
        aggfunc="first",
    ).reset_index()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked_base.to_csv(
        output.with_name(f"{output.stem}_base_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    stress.to_csv(
        output.with_name(f"{output.stem}_stress_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    winners.to_csv(
        output.with_name(f"{output.stem}_winners.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    write_excel_workbook(
        output.with_suffix(".xlsx"),
        [
            ("base_results", ranked_base),
            ("base_winners", winners),
            ("stress_results", stress),
            ("annualized_matrix", annualized_matrix),
            ("sharpe_matrix", sharpe_matrix),
            ("methodology", build_methodology()),
        ],
    )

    display_columns = [
        "initial_cash",
        "variant",
        "annualized_return",
        "sharpe_no_risk_free",
        "max_drawdown",
        "holdout_return",
        "total_fees",
        "minimum_commission_order_share",
        "participation_p95",
        "robust_rank",
    ]
    print(ranked_base[display_columns].to_string(index=False))
    print(f"Capital-scale comparison: {output.with_suffix('.xlsx')}")
    return ranked_base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze V2H4 capital-scale validation results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/validation_csmar_capital_scale"),
    )
    parser.add_argument("--holdout-start", default="2026-04-01")
    parser.add_argument("--holdout-end", default="2026-07-24")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/validation_csmar_capital_scale/v2h4_capital_scale_comparison.xlsx"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
