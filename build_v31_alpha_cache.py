"""Build causal weekly alpha inputs used by the V3.1 strategy."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from build_weekly_risk_model import FinancialPointInTimeStore, IndustryPointInTimeStore


ALGORITHM_VERSION = "v31-alpha-pit-v2"
CSRC_2001 = "证监会行业分类2001年版"
CSRC_2012 = "证监会行业分类2012年版"
CSRC_2012_EFFECTIVE_DATE = "2012-10-26"
ASSOCIATION = "中国上市公司协会上市公司行业分类"


def create_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS weekly_v31_alpha (
            model_date TEXT NOT NULL,
            code TEXT NOT NULL,
            financial_report_period TEXT,
            financial_available_date TEXT,
            float_market_cap REAL,
            market_industry_weight REAL,
            earnings_yield_raw REAL,
            quality_raw REAL,
            growth_raw REAL,
            PRIMARY KEY (model_date, code)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_weekly_v31_alpha_code
        ON weekly_v31_alpha(code, model_date);

        CREATE TABLE IF NOT EXISTS v31_alpha_cache_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(weekly_v31_alpha)")
    }
    if "market_industry_weight" not in columns:
        conn.execute(
            "ALTER TABLE weekly_v31_alpha ADD COLUMN market_industry_weight REAL"
        )


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def safe_ratio(numerator, denominator, require_positive_denominator=False):
    numerator = finite(numerator)
    denominator = finite(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return np.nan
    if require_positive_denominator and denominator <= 0:
        return np.nan
    if abs(denominator) <= 1e-12:
        return np.nan
    return numerator / denominator


def robust_zscore(series):
    values = pd.to_numeric(series, errors="coerce")
    finite_values = values.dropna()
    if len(finite_values) < 20:
        return pd.Series(np.nan, index=series.index, dtype=float)
    median = float(finite_values.median())
    mad = float((finite_values - median).abs().median())
    if not math.isfinite(mad) or mad <= 1e-12:
        standard = float(finite_values.std(ddof=1))
        if not math.isfinite(standard) or standard <= 1e-12:
            return pd.Series(0.0, index=series.index, dtype=float).where(values.notna())
        result = (values - float(finite_values.mean())) / standard
    else:
        scale = 1.4826 * mad
        result = (values - median) / scale
    return result.clip(-5.0, 5.0)


def weighted_available(frame, weights):
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        available = values.notna().astype(float)
        numerator += values.fillna(0.0) * float(weight)
        denominator += available * float(weight)
    return numerator / denominator.replace(0.0, np.nan)


def source_signature(conn):
    exposure = conn.execute(
        "SELECT COUNT(*), MIN(model_date), MAX(model_date) FROM weekly_exposure"
    ).fetchone()
    financial = conn.execute(
        "SELECT COUNT(*), MIN(available_date), MAX(available_date) FROM financial_pit"
    ).fetchone()
    raw = conn.execute(
        "SELECT COUNT(*), MIN(model_date), MAX(model_date) FROM weekly_raw_exposure"
    ).fetchone()
    industry = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT code), MIN(implement_date), MAX(implement_date)
        FROM industry_history
        WHERE classification_name IN (?, ?, ?)
        """,
        (CSRC_2001, CSRC_2012, ASSOCIATION),
    ).fetchone()
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "exposure_rows": int(exposure[0] or 0),
        "exposure_min_date": exposure[1],
        "exposure_max_date": exposure[2],
        "financial_rows": int(financial[0] or 0),
        "financial_min_available_date": financial[1],
        "financial_max_available_date": financial[2],
        "raw_exposure_rows": int(raw[0] or 0),
        "raw_exposure_min_date": raw[1],
        "raw_exposure_max_date": raw[2],
        "industry_rows": int(industry[0] or 0),
        "industry_stocks": int(industry[1] or 0),
        "industry_min_implement_date": industry[2],
        "industry_max_implement_date": industry[3],
    }


def can_append_cache(current, signature, existing_rows):
    if not isinstance(current, dict) or existing_rows <= 0:
        return False
    if existing_rows != int(current.get("output_rows", -1)):
        return False
    unchanged_keys = (
        "algorithm_version",
        "exposure_min_date",
        "financial_rows",
        "financial_min_available_date",
        "financial_max_available_date",
        "raw_exposure_min_date",
        "industry_rows",
        "industry_stocks",
        "industry_min_implement_date",
        "industry_max_implement_date",
    )
    if any(current.get(key) != signature.get(key) for key in unchanged_keys):
        return False
    old_exposure_max = str(current.get("exposure_max_date") or "")
    new_exposure_max = str(signature.get("exposure_max_date") or "")
    old_raw_max = str(current.get("raw_exposure_max_date") or "")
    new_raw_max = str(signature.get("raw_exposure_max_date") or "")
    return (
        old_exposure_max < new_exposure_max
        and old_raw_max <= new_raw_max
        and int(signature.get("exposure_rows", 0)) >= int(current.get("exposure_rows", 0))
        and int(signature.get("raw_exposure_rows", 0))
        >= int(current.get("raw_exposure_rows", 0))
    )


def build(args):
    database = Path(args.database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    conn = sqlite3.connect(database, timeout=120.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        create_schema(conn)
        signature = source_signature(conn)
        current_row = conn.execute(
            "SELECT value FROM v31_alpha_cache_metadata WHERE key='current'"
        ).fetchone()
        current = json.loads(current_row[0]) if current_row else None
        existing_rows = int(
            conn.execute("SELECT COUNT(*) FROM weekly_v31_alpha").fetchone()[0]
        )
        if (
            not args.overwrite
            and current is not None
            and all(current.get(key) == value for key, value in signature.items())
            and existing_rows == int(current.get("output_rows", -1))
        ):
            print(
                f"V3.1 alpha cache is current: {existing_rows:,} rows through "
                f"{signature['exposure_max_date']}",
                flush=True,
            )
            return

        dates = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT model_date FROM weekly_exposure ORDER BY model_date"
            )
        ]
        if not dates:
            raise ValueError("weekly_exposure contains no model dates")
        append_mode = can_append_cache(current, signature, existing_rows)
        build_dates = dates
        if append_mode:
            previous_max_date = str(current["exposure_max_date"])
            build_dates = [date for date in dates if date > previous_max_date]
            if not build_dates:
                append_mode = False
                build_dates = dates
        financial_rows = conn.execute(
            """
            SELECT * FROM financial_pit
            WHERE available_date IS NOT NULL
            ORDER BY available_date, code, report_period
            """
        ).fetchall()
        financial_store = FinancialPointInTimeStore(financial_rows, dates)
        industry_rows = []
        for row in conn.execute(
            """
            SELECT code, classification_name, implement_date, industry_code, industry_name
            FROM industry_history
            WHERE classification_name IN (?, ?, ?)
            ORDER BY implement_date, code, classification_name
            """,
            (CSRC_2001, CSRC_2012, ASSOCIATION),
        ):
            record = dict(row)
            if record["classification_name"] == CSRC_2012:
                record["implement_date"] = max(
                    str(record["implement_date"]), CSRC_2012_EFFECTIVE_DATE
                )
            industry_rows.append(record)
        industry_store = IndustryPointInTimeStore(
            industry_rows,
            1,
            classification_schedule=[
                {"classification_name": CSRC_2001, "effective_date": "1900-01-01"},
                {
                    "classification_name": CSRC_2012,
                    "effective_date": CSRC_2012_EFFECTIVE_DATE,
                },
            ],
            fallback_classifications=[ASSOCIATION],
        )
        if append_mode:
            print(
                f"Appending V3.1 alpha cache after {current['exposure_max_date']}: "
                f"{len(build_dates)} new model dates",
                flush=True,
            )
        else:
            conn.execute("DELETE FROM weekly_v31_alpha")
            conn.commit()

        insert_sql = """
            INSERT INTO weekly_v31_alpha(
                model_date, code, financial_report_period,
                financial_available_date, float_market_cap,
                market_industry_weight, earnings_yield_raw,
                quality_raw, growth_raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        output_rows = existing_rows if append_mode else 0
        for date_index, model_date in enumerate(build_dates, start=1):
            financial_store.advance(model_date)
            industry_store.advance(model_date)
            raw_market = pd.read_sql_query(
                """
                SELECT code, float_market_cap, total_market_cap
                FROM weekly_raw_exposure WHERE model_date=?
                """,
                conn,
                params=(model_date,),
            )
            raw_market["market_cap"] = pd.to_numeric(
                raw_market["float_market_cap"], errors="coerce"
            )
            raw_market["market_cap"] = raw_market["market_cap"].where(
                raw_market["market_cap"] > 0,
                pd.to_numeric(raw_market["total_market_cap"], errors="coerce"),
            )
            raw_market["market_cap"] = (
                raw_market["market_cap"].fillna(0.0).clip(lower=0.0)
            )
            raw_market["industry_group"] = raw_market["code"].map(
                lambda code: industry_store.get(str(code).zfill(6))["industry_group"]
            )
            industry_totals = raw_market.groupby("industry_group")["market_cap"].sum()
            market_industry_weights = (
                (industry_totals / float(industry_totals.sum())).to_dict()
                if float(industry_totals.sum()) > 0
                else {}
            )
            exposure = pd.read_sql_query(
                """
                SELECT code, float_market_cap, total_market_cap, industry_group
                FROM weekly_exposure WHERE model_date=?
                """,
                conn,
                params=(model_date,),
            )
            records = []
            for row in exposure.to_dict("records"):
                code = str(row["code"]).zfill(6)
                financial = financial_store.get(code)
                cap = finite(row.get("total_market_cap"))
                profit = finite(financial.get("parent_net_profit_ttm"))
                operating_cashflow = finite(financial.get("operating_cashflow_ttm"))
                assets = finite(financial.get("total_assets"))
                liabilities = finite(financial.get("total_liabilities"))
                equity = finite(financial.get("parent_equity"))
                earnings_to_price = safe_ratio(profit, cap, True)
                cashflow_to_price = safe_ratio(operating_cashflow, cap, True)
                if math.isfinite(earnings_to_price) and math.isfinite(cashflow_to_price):
                    earnings_yield = 0.67 * earnings_to_price + 0.33 * cashflow_to_price
                elif math.isfinite(earnings_to_price):
                    earnings_yield = earnings_to_price
                else:
                    earnings_yield = cashflow_to_price
                records.append(
                    {
                        "model_date": model_date,
                        "code": code,
                        "financial_report_period": financial.get("financial_report_period"),
                        "financial_available_date": financial.get("financial_available_date"),
                        "float_market_cap": finite(row.get("float_market_cap")),
                        "market_industry_weight": finite(
                            market_industry_weights.get(
                                str(row.get("industry_group") or "UNKNOWN"), np.nan
                            )
                        ),
                        "earnings_yield_raw": earnings_yield,
                        "roe": safe_ratio(profit, equity, True),
                        "cash_conversion": safe_ratio(operating_cashflow, abs(profit)),
                        "accruals": safe_ratio(profit - operating_cashflow, assets, True),
                        "leverage": safe_ratio(liabilities, assets, True),
                        "revenue_growth": finite(financial.get("revenue_growth")),
                        "earnings_growth": finite(financial.get("earnings_growth")),
                    }
                )
            frame = pd.DataFrame(records)
            if frame.empty:
                continue
            frame["cash_conversion"] = pd.to_numeric(
                frame["cash_conversion"], errors="coerce"
            ).clip(-3.0, 3.0)
            frame["quality_raw"] = weighted_available(
                pd.DataFrame(
                    {
                        "roe": robust_zscore(frame["roe"]),
                        "cash_conversion": robust_zscore(frame["cash_conversion"]),
                        "accruals": -robust_zscore(frame["accruals"]),
                        "leverage": -robust_zscore(frame["leverage"]),
                    }
                ),
                {
                    "roe": 0.35,
                    "cash_conversion": 0.30,
                    "accruals": 0.20,
                    "leverage": 0.15,
                },
            )
            frame["growth_raw"] = weighted_available(
                frame,
                {"revenue_growth": 0.45, "earnings_growth": 0.55},
            )
            payload = [
                (
                    row.model_date,
                    row.code,
                    row.financial_report_period,
                    row.financial_available_date,
                    finite(row.float_market_cap),
                    finite(row.market_industry_weight),
                    finite(row.earnings_yield_raw),
                    finite(row.quality_raw),
                    finite(row.growth_raw),
                )
                for row in frame.itertuples(index=False)
            ]
            conn.executemany(insert_sql, payload)
            output_rows += len(payload)
            if (
                date_index % int(args.commit_every_dates) == 0
                or date_index == len(build_dates)
            ):
                conn.commit()
                print(
                    f"V3.1 alpha progress: {date_index}/{len(build_dates)} "
                    f"{model_date}, rows={output_rows:,}",
                    flush=True,
                )

        metadata = {
            **signature,
            "output_rows": int(output_rows),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        conn.execute(
            "INSERT OR REPLACE INTO v31_alpha_cache_metadata VALUES ('current', ?)",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True),),
        )
        conn.commit()
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--commit-every-dates", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
