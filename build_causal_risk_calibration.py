"""Build a weekly risk-calibration schedule using only outcomes known by each date."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from validate_risk_forecasts import forecast_observations


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"
)


def build_expanding_schedule(
    observations,
    minimum_forecast_weeks=52,
    lookback_weeks=0,
    default_multiplier=1.0,
):
    frame = observations.copy()
    required = {
        "forecast_date",
        "realized_date",
        "portfolio_name",
        "predicted_weekly_variance",
        "realized_weekly_return",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Forecast observations are missing columns: " + ", ".join(missing)
        )
    frame["forecast_date"] = frame["forecast_date"].astype(str)
    frame["realized_date"] = frame["realized_date"].astype(str)
    frame["predicted_weekly_variance"] = pd.to_numeric(
        frame["predicted_weekly_variance"], errors="coerce"
    )
    frame["realized_weekly_return"] = pd.to_numeric(
        frame["realized_weekly_return"], errors="coerce"
    )
    frame = frame.dropna(
        subset=["predicted_weekly_variance", "realized_weekly_return"]
    )
    frame = frame.loc[frame["predicted_weekly_variance"].gt(0)].copy()
    if frame.empty:
        raise ValueError("No usable forecast observations for calibration")

    rows = []
    realized_dates = sorted(frame["realized_date"].unique())
    minimum_forecast_weeks = max(1, int(minimum_forecast_weeks))
    lookback_weeks = max(0, int(lookback_weeks))
    for as_of_date in realized_dates:
        history = frame.loc[frame["realized_date"].le(as_of_date)].copy()
        available_dates = sorted(history["forecast_date"].unique())
        if lookback_weeks and len(available_dates) > lookback_weeks:
            selected_dates = set(available_dates[-lookback_weeks:])
            history = history.loc[
                history["forecast_date"].isin(selected_dates)
            ].copy()
            available_dates = sorted(selected_dates)

        forecast_weeks = len(available_dates)
        raw_multiplier = float("nan")
        multiplier = float(default_multiplier)
        status = "default_insufficient_history"
        if forecast_weeks >= minimum_forecast_weeks:
            known_means = history.groupby("portfolio_name")[
                "realized_weekly_return"
            ].transform("mean")
            residual = history["realized_weekly_return"] - known_means
            predicted = history["predicted_weekly_variance"].clip(lower=1e-16)
            ratio = float(np.mean(np.square(residual)) / predicted.mean())
            if math.isfinite(ratio) and ratio > 0:
                raw_multiplier = math.sqrt(ratio)
                multiplier = raw_multiplier
                status = "calibrated"

        rows.append(
            {
                "as_of_date": as_of_date,
                "latest_forecast_date": max(available_dates),
                "forecast_weeks": forecast_weeks,
                "observations": len(history),
                "raw_multiplier": raw_multiplier,
                "multiplier": multiplier,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


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

    schedule = build_expanding_schedule(
        observations,
        minimum_forecast_weeks=args.minimum_forecast_weeks,
        lookback_weeks=args.lookback_weeks,
        default_multiplier=args.default_multiplier,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(args.output, index=False, encoding="utf-8-sig")
    metadata = {
        "database": str(args.database),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": str(schedule["as_of_date"].min()),
        "end_date": str(schedule["as_of_date"].max()),
        "rows": len(schedule),
        "minimum_forecast_weeks": int(args.minimum_forecast_weeks),
        "lookback_weeks": int(args.lookback_weeks),
        "default_multiplier": float(args.default_multiplier),
        "first_calibrated_date": (
            str(
                schedule.loc[
                    schedule["status"].eq("calibrated"), "as_of_date"
                ].min()
            )
            if schedule["status"].eq("calibrated").any()
            else None
        ),
        "minimum_multiplier": float(schedule["multiplier"].min()),
        "maximum_multiplier": float(schedule["multiplier"].max()),
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Calibration schedule: {args.output}")
    print(f"Schedule metadata: {metadata_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a causal expanding weekly risk-calibration schedule."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--minimum-forecast-weeks", type=int, default=52)
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=0,
        help="Zero uses all previously realized forecast weeks.",
    )
    parser.add_argument("--default-multiplier", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--minimum-industry-stocks", type=int, default=30)
    parser.add_argument("--minimum-realized-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)
    args.database = args.database.resolve()
    args.output = args.output.resolve()
    if not args.database.exists():
        parser.error(f"Database does not exist: {args.database}")
    if args.minimum_forecast_weeks < 1:
        parser.error("--minimum-forecast-weeks must be positive")
    if args.lookback_weeks < 0:
        parser.error("--lookback-weeks cannot be negative")
    if args.default_multiplier <= 0:
        parser.error("--default-multiplier must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
