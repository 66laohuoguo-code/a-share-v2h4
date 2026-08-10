"""Calibrate one-week-ahead forecasts from the weekly A-share risk model."""

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
from ashare_utils import write_excel_workbook
from risk_model_reporting import (
    calculate_portfolio_risk,
    json_ready,
    load_exposure_and_specific,
    load_factor_covariance,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "risk_forecast_validation"


def equal_weights(codes):
    codes = pd.Index(codes).astype(str).str.zfill(6)
    if len(codes) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(codes), index=codes, dtype=float)


def capitalization_weights(frame):
    caps = pd.to_numeric(frame["total_market_cap"], errors="coerce")
    caps = caps.where(caps > 0)
    valid = caps.notna()
    if not valid.any():
        return pd.Series(dtype=float)
    caps = caps.loc[valid]
    return pd.Series(
        caps.to_numpy(dtype=float) / float(caps.sum()),
        index=frame.loc[valid, "code"].astype(str).str.zfill(6),
        dtype=float,
    )


def build_test_portfolios(exposure, quantile, minimum_industry_stocks):
    portfolios = [
        ("market", "MARKET_EW", equal_weights(exposure["code"])),
        ("market", "MARKET_CAP", capitalization_weights(exposure)),
    ]
    lower_cut = float(quantile)
    upper_cut = 1.0 - lower_cut
    for factor in STYLE_FACTORS:
        values = pd.to_numeric(exposure[factor], errors="coerce")
        valid = values.notna()
        if valid.sum() < 20 or values.loc[valid].nunique() < 5:
            continue
        ranks = values.loc[valid].rank(method="average", pct=True)
        low_codes = exposure.loc[ranks.index[ranks <= lower_cut], "code"]
        high_codes = exposure.loc[ranks.index[ranks >= upper_cut], "code"]
        if len(low_codes) >= 5:
            portfolios.append(
                ("style", f"{factor}_LOW", equal_weights(low_codes))
            )
        if len(high_codes) >= 5:
            portfolios.append(
                ("style", f"{factor}_HIGH", equal_weights(high_codes))
            )

    for industry, group in exposure.groupby("industry_group", sort=True):
        industry = str(industry)
        if industry == "UNKNOWN" or len(group) < int(minimum_industry_stocks):
            continue
        portfolios.append(
            (
                "industry",
                f"INDUSTRY_{industry}",
                equal_weights(group["code"]),
            )
        )
    return portfolios


def available_forecast_pairs(conn, start_date=None, end_date=None):
    rows = conn.execute(
        """
        SELECT DISTINCT r.exposure_date, r.model_date
        FROM weekly_specific_return r
        JOIN (
            SELECT DISTINCT model_date
            FROM weekly_factor_covariance
        ) c ON c.model_date=r.exposure_date
        JOIN (
            SELECT DISTINCT model_date
            FROM weekly_specific_risk
        ) s ON s.model_date=r.exposure_date
        ORDER BY r.exposure_date
        """
    ).fetchall()
    pairs = [(str(row[0]), str(row[1])) for row in rows]
    if start_date:
        pairs = [pair for pair in pairs if pair[0] >= str(start_date)]
    if end_date:
        pairs = [pair for pair in pairs if pair[0] <= str(end_date)]
    return pairs


def forecast_observations(
    conn,
    start_date=None,
    end_date=None,
    quantile=0.20,
    minimum_industry_stocks=30,
    minimum_realized_coverage=0.95,
):
    pairs = available_forecast_pairs(conn, start_date, end_date)
    if not pairs:
        raise ValueError("No forecast dates with next-week realized returns")

    rows = []
    for pair_index, (forecast_date, realized_date) in enumerate(pairs, start=1):
        exposure = load_exposure_and_specific(conn, forecast_date)
        exposure = exposure.loc[
            pd.to_numeric(
                exposure["specific_variance"], errors="coerce"
            ).notna()
        ].copy()
        if exposure.empty:
            continue
        covariance = load_factor_covariance(conn, forecast_date)
        realized = pd.read_sql_query(
            """
            SELECT code, weekly_return
            FROM weekly_specific_return
            WHERE model_date=? AND exposure_date=?
            """,
            conn,
            params=(realized_date, forecast_date),
        )
        realized["code"] = realized["code"].astype(str).str.zfill(6)
        return_map = realized.set_index("code")["weekly_return"].astype(float)
        exposure_by_code = exposure.set_index("code")
        specific = exposure_by_code["specific_variance"].astype(float)

        for portfolio_class, portfolio_name, weights in build_test_portfolios(
            exposure, quantile, minimum_industry_stocks
        ):
            if weights.empty:
                continue
            present = weights.index.isin(return_map.index)
            realized_coverage = float(weights.loc[present].sum())
            if realized_coverage < float(minimum_realized_coverage):
                continue
            aligned_returns = return_map.reindex(weights.index).fillna(0.0)
            realized_return = float(
                weights.to_numpy(dtype=float)
                @ aligned_returns.to_numpy(dtype=float)
            )
            risk = calculate_portfolio_risk(
                weights,
                exposure,
                covariance,
                specific,
            )
            rows.append(
                {
                    "forecast_date": forecast_date,
                    "realized_date": realized_date,
                    "portfolio_class": portfolio_class,
                    "portfolio_name": portfolio_name,
                    "stock_count": len(weights),
                    "realized_coverage": realized_coverage,
                    "predicted_annual_variance": risk["annual_variance"],
                    "predicted_annual_volatility": risk["annual_volatility"],
                    "predicted_weekly_variance": (
                        risk["annual_variance"] / 52.0
                    ),
                    "predicted_weekly_volatility": risk["weekly_volatility"],
                    "realized_weekly_return": realized_return,
                }
            )
        if pair_index % 26 == 0 or pair_index == len(pairs):
            print(
                f"Validated {pair_index}/{len(pairs)} forecast weeks",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No portfolio forecast observations passed coverage")
    return frame


def summarize_group(frame, label):
    residual = (
        frame["realized_residual"]
        if "realized_residual" in frame
        else frame["realized_weekly_return"]
        - float(frame["realized_weekly_return"].mean())
    )
    predicted_variance = frame["predicted_weekly_variance"].clip(lower=1e-16)
    standardized = residual / np.sqrt(predicted_variance)
    realized_variance = (
        float(residual.var(ddof=1)) if len(frame) > 1 else float("nan")
    )
    mean_predicted_variance = float(predicted_variance.mean())
    calibration_ratio = (
        realized_variance / mean_predicted_variance
        if mean_predicted_variance > 0
        else float("nan")
    )
    predicted_volatility = np.sqrt(predicted_variance)
    absolute_residual = residual.abs()
    correlation = (
        float(predicted_volatility.corr(absolute_residual))
        if len(frame) > 2
        and predicted_volatility.nunique() > 1
        and absolute_residual.nunique() > 1
        else float("nan")
    )
    return {
        "group": label,
        "observations": len(frame),
        "forecast_weeks": frame["forecast_date"].nunique(),
        "mean_realized_coverage": float(frame["realized_coverage"].mean()),
        "mean_predicted_annual_volatility": float(
            frame["predicted_annual_volatility"].mean()
        ),
        "realized_annual_volatility": (
            math.sqrt(max(realized_variance, 0.0) * 52.0)
            if math.isfinite(realized_variance)
            else float("nan")
        ),
        "calibration_ratio_realized_to_predicted_variance": calibration_ratio,
        "recommended_volatility_multiplier": (
            math.sqrt(max(calibration_ratio, 0.0))
            if math.isfinite(calibration_ratio)
            else float("nan")
        ),
        "predicted_volatility_vs_absolute_return_correlation": correlation,
        "standardized_return_rms": float(
            math.sqrt(np.mean(np.square(standardized)))
        ),
        "standardized_return_std": float(standardized.std(ddof=1)),
        "breach_rate_90_interval": float(
            (standardized.abs() > 1.6448536269514722).mean()
        ),
        "breach_rate_95_interval": float(
            (standardized.abs() > 1.959963984540054).mean()
        ),
    }


def safe_correlation(left, right):
    left = pd.Series(left, dtype=float)
    right = pd.Series(right, dtype=float)
    valid = left.notna() & right.notna()
    if (
        valid.sum() < 3
        or left.loc[valid].nunique() <= 1
        or right.loc[valid].nunique() <= 1
    ):
        return float("nan")
    return float(left.loc[valid].corr(right.loc[valid]))


def summarize_forecasts(observations):
    centered = observations.copy()
    centered["realized_residual"] = centered["realized_weekly_return"] - (
        centered.groupby("portfolio_name")["realized_weekly_return"]
        .transform("mean")
    )
    centered["standardized_return"] = centered["realized_residual"] / np.sqrt(
        centered["predicted_weekly_variance"].clip(lower=1e-16)
    )

    summaries = [summarize_group(group, name) for name, group in centered.groupby(
        "portfolio_name", sort=True
    )]
    class_summaries = [
        summarize_group(group, f"CLASS:{name}")
        for name, group in centered.groupby("portfolio_class", sort=True)
    ]

    predicted = centered["predicted_weekly_variance"].clip(lower=1e-16)
    residual = centered["realized_residual"]
    overall_ratio = float(np.mean(np.square(residual)) / predicted.mean())
    overall_multiplier = math.sqrt(max(overall_ratio, 0.0))
    overall = {
        "group": "OVERALL",
        "observations": len(centered),
        "forecast_weeks": centered["forecast_date"].nunique(),
        "portfolio_tests": centered["portfolio_name"].nunique(),
        "mean_realized_coverage": float(
            centered["realized_coverage"].mean()
        ),
        "mean_predicted_annual_volatility": float(
            centered["predicted_annual_volatility"].mean()
        ),
        "realized_annual_volatility": float(
            residual.std(ddof=1) * math.sqrt(52.0)
        ),
        "calibration_ratio_realized_to_predicted_variance": overall_ratio,
        "recommended_volatility_multiplier": overall_multiplier,
        "predicted_volatility_vs_absolute_return_correlation": safe_correlation(
            centered["predicted_weekly_volatility"], residual.abs()
        ),
        "standardized_return_rms": float(
            math.sqrt(np.mean(np.square(centered["standardized_return"])))
        ),
        "standardized_return_std": float(
            centered["standardized_return"].std(ddof=1)
        ),
        "breach_rate_90_interval": float(
            (centered["standardized_return"].abs() > 1.6448536269514722).mean()
        ),
        "breach_rate_95_interval": float(
            (centered["standardized_return"].abs() > 1.959963984540054).mean()
        ),
    }
    ratio_ok = 0.50 <= overall_ratio <= 2.00
    coverage_ok = overall["mean_realized_coverage"] >= 0.95
    history_ok = overall["forecast_weeks"] >= 100
    overall["status"] = (
        "calibrated"
        if ratio_ok and coverage_ok and history_ok
        else "needs_rescaling_or_review"
    )
    overall["checks"] = {
        "variance_ratio_between_0_5_and_2_0": ratio_ok,
        "mean_realized_coverage_at_least_95pct": coverage_ok,
        "at_least_100_forecast_weeks": history_ok,
    }
    excel_overall = dict(overall)
    excel_overall["checks"] = json.dumps(
        excel_overall["checks"], ensure_ascii=False, sort_keys=True
    )
    summary_frame = pd.DataFrame(
        [excel_overall, *class_summaries, *summaries]
    )
    return centered, overall, summary_frame


def run(args):
    conn = sqlite3.connect(args.database)
    try:
        observations = forecast_observations(
            conn,
            start_date=args.start_date,
            end_date=args.end_date,
            quantile=args.quantile,
            minimum_industry_stocks=args.minimum_industry_stocks,
            minimum_realized_coverage=args.minimum_realized_coverage,
        )
    finally:
        conn.close()

    observations, overall, summary_frame = summarize_forecasts(observations)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"risk_forecast_validation_{timestamp}.json"
    xlsx_path = args.output_dir / f"risk_forecast_validation_{timestamp}.xlsx"
    payload = json_ready(
        {
            "database": str(args.database),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": observations["forecast_date"].min(),
            "end_date": observations["forecast_date"].max(),
            "overall": overall,
        }
    )
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_excel_workbook(
        xlsx_path,
        [
            ("summary", summary_frame),
            ("observations", observations),
        ],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Summary JSON: {json_path}")
    print(f"Validation workbook: {xlsx_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one-week-ahead forecasts from the risk model."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--minimum-industry-stocks", type=int, default=30)
    parser.add_argument("--minimum-realized-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)
    args.database = args.database.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.database.exists():
        parser.error(f"Database does not exist: {args.database}")
    if not 0 < args.quantile < 0.5:
        parser.error("--quantile must be between 0 and 0.5")
    if not 0 < args.minimum_realized_coverage <= 1:
        parser.error("--minimum-realized-coverage must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
