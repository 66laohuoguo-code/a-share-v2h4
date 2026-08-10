"""Shared readers and portfolio math for the weekly A-share risk model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ashare_risk_model import STYLE_FACTORS


def latest_model_date(conn, requested_date=None):
    if requested_date:
        row = conn.execute(
            """
            SELECT MAX(model_date)
            FROM weekly_factor_covariance
            WHERE model_date<=?
            """,
            (str(requested_date),),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(model_date) FROM weekly_factor_covariance"
        ).fetchone()
    model_date = row[0] if row else None
    if not model_date:
        suffix = f" on or before {requested_date}" if requested_date else ""
        raise ValueError(f"No factor covariance is available{suffix}")
    return str(model_date)


def load_factor_covariance(conn, model_date):
    rows = pd.read_sql_query(
        """
        SELECT factor_1, factor_2, covariance
        FROM weekly_factor_covariance
        WHERE model_date=?
        """,
        conn,
        params=(str(model_date),),
    )
    if rows.empty:
        raise ValueError(f"No factor covariance is available for {model_date}")
    factor_order = list(dict.fromkeys(rows["factor_1"].astype(str)))
    covariance = rows.pivot(
        index="factor_1", columns="factor_2", values="covariance"
    )
    covariance = covariance.reindex(index=factor_order, columns=factor_order)
    if covariance.isna().any().any():
        raise ValueError(f"Incomplete factor covariance for {model_date}")
    return covariance.astype(float)


def load_exposure_and_specific(conn, model_date):
    style_sql = ", ".join(f"e.{factor}" for factor in STYLE_FACTORS)
    frame = pd.read_sql_query(
        f"""
        SELECT e.code, e.industry_group, e.total_market_cap,
               e.float_market_cap, e.avg_amount_60, {style_sql},
               s.specific_variance, s.specific_volatility,
               s.specific_observations
        FROM weekly_exposure e
        LEFT JOIN weekly_specific_risk s
          ON s.model_date=e.model_date AND s.code=e.code
        WHERE e.model_date=?
        ORDER BY e.code
        """,
        conn,
        params=(str(model_date),),
    )
    if frame.empty:
        raise ValueError(f"No stock exposures are available for {model_date}")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def complete_exposures_for_codes(codes, exposure, specific_quantile=0.90):
    codes = list(dict.fromkeys(str(code).zfill(6) for code in codes))
    output_columns = [
        "code",
        "industry_group",
        "total_market_cap",
        "float_market_cap",
        "avg_amount_60",
        "specific_variance",
        "specific_volatility",
        "specific_observations",
        "exposure_source",
        *STYLE_FACTORS,
    ]
    if not codes:
        return pd.DataFrame(columns=output_columns)

    exposure_by_code = exposure.copy()
    exposure_by_code["code"] = (
        exposure_by_code["code"].astype(str).str.zfill(6)
    )
    exposure_by_code = exposure_by_code.drop_duplicates("code").set_index("code")
    specific_values = pd.to_numeric(
        exposure_by_code["specific_variance"], errors="coerce"
    )
    fallback_specific = float(specific_values.quantile(specific_quantile))
    if not math.isfinite(fallback_specific) or fallback_specific <= 0:
        fallback_specific = float(specific_values.median())
    if not math.isfinite(fallback_specific) or fallback_specific <= 0:
        raise ValueError("The model has no usable specific risk")

    rows = []
    for code in codes:
        if code in exposure_by_code.index:
            row = exposure_by_code.loc[code].to_dict()
            row["code"] = code
            row["exposure_source"] = "model"
            specific_variance = pd.to_numeric(
                pd.Series([row.get("specific_variance")]), errors="coerce"
            ).iloc[0]
            if not math.isfinite(float(specific_variance)):
                row["specific_variance"] = fallback_specific
                row["specific_volatility"] = math.sqrt(fallback_specific)
                row["exposure_source"] = "model_with_specific_p90"
            rows.append(row)
            continue
        synthetic = {
            "code": code,
            "industry_group": "UNKNOWN",
            "total_market_cap": np.nan,
            "float_market_cap": np.nan,
            "avg_amount_60": np.nan,
            "specific_variance": fallback_specific,
            "specific_volatility": math.sqrt(fallback_specific),
            "specific_observations": 0,
            "exposure_source": "synthetic_market_plus_specific_p90",
        }
        for factor in STYLE_FACTORS:
            synthetic[factor] = 0.0
        rows.append(synthetic)
    return pd.DataFrame(rows).reindex(columns=output_columns)


def build_exposure_matrix(exposures, factor_names):
    frame = exposures.copy()
    matrix = np.zeros((len(frame), len(factor_names)), dtype=float)
    industries = frame["industry_group"].fillna("UNKNOWN").astype(str)
    for column_index, factor in enumerate(factor_names):
        if factor == "MARKET":
            matrix[:, column_index] = 1.0
        elif factor in STYLE_FACTORS:
            matrix[:, column_index] = pd.to_numeric(
                frame[factor], errors="coerce"
            ).fillna(0.0)
        elif factor.startswith("INDUSTRY:"):
            industry = factor.split(":", 1)[1]
            matrix[:, column_index] = (industries == industry).astype(float)
    return matrix


def calculate_portfolio_risk(
    weights,
    exposures,
    factor_covariance,
    specific_variance,
):
    weights = pd.Series(weights, dtype=float)
    weights.index = weights.index.astype(str).str.zfill(6)
    weights = weights.groupby(level=0).sum()
    exposure_frame = exposures.copy()
    exposure_frame["code"] = exposure_frame["code"].astype(str).str.zfill(6)
    exposure_frame = exposure_frame.drop_duplicates("code").set_index("code")
    missing = weights.index.difference(exposure_frame.index)
    if len(missing):
        raise ValueError(
            "Missing risk exposures for: " + ", ".join(missing.tolist())
        )

    aligned = exposure_frame.reindex(weights.index).reset_index()
    factor_names = list(factor_covariance.index.astype(str))
    matrix = build_exposure_matrix(aligned, factor_names)
    covariance = factor_covariance.to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)
    portfolio_exposure = weight_values @ matrix
    factor_marginal = covariance @ portfolio_exposure
    factor_variance_contribution = portfolio_exposure * factor_marginal
    common_variance = float(factor_variance_contribution.sum())

    specific = pd.Series(specific_variance, dtype=float)
    specific.index = specific.index.astype(str).str.zfill(6)
    specific = specific.groupby(level=0).last().reindex(weights.index)
    if specific.isna().any():
        missing_specific = specific.index[specific.isna()].tolist()
        raise ValueError(
            "Missing specific variances for: " + ", ".join(missing_specific)
        )
    specific_values = specific.to_numpy(dtype=float)
    stock_specific_contribution = np.square(weight_values) * specific_values
    specific_component = float(stock_specific_contribution.sum())

    stock_common_marginal = matrix @ factor_marginal
    stock_common_contribution = weight_values * stock_common_marginal
    stock_total_contribution = (
        stock_common_contribution + stock_specific_contribution
    )
    total_variance = max(common_variance + specific_component, 0.0)
    annual_volatility = math.sqrt(total_variance)

    factor_frame = pd.DataFrame(
        {
            "factor_name": factor_names,
            "portfolio_exposure": portfolio_exposure,
            "annual_variance_contribution": factor_variance_contribution,
        }
    )
    if total_variance > 0:
        factor_frame["share_of_total_variance"] = (
            factor_frame["annual_variance_contribution"] / total_variance
        )
    else:
        factor_frame["share_of_total_variance"] = np.nan

    stock_frame = pd.DataFrame(
        {
            "code": weights.index,
            "portfolio_weight": weight_values,
            "common_variance_contribution": stock_common_contribution,
            "specific_variance_contribution": stock_specific_contribution,
            "total_variance_contribution": stock_total_contribution,
        }
    )
    if total_variance > 0:
        stock_frame["share_of_total_variance"] = (
            stock_frame["total_variance_contribution"] / total_variance
        )
    else:
        stock_frame["share_of_total_variance"] = np.nan
    if annual_volatility > 0:
        stock_frame["annual_volatility_contribution"] = (
            stock_frame["total_variance_contribution"] / annual_volatility
        )
    else:
        stock_frame["annual_volatility_contribution"] = np.nan

    return {
        "annual_variance": total_variance,
        "annual_volatility": annual_volatility,
        "weekly_volatility": annual_volatility / math.sqrt(52.0),
        "common_variance": common_variance,
        "specific_variance": specific_component,
        "factor": factor_frame,
        "stock": stock_frame,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
