"""Core calculations for the weekly A-share multi-factor risk model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


STYLE_FACTORS = (
    "SIZE",
    "NONLINEAR_SIZE",
    "BETA",
    "MOMENTUM",
    "RESIDUAL_VOLATILITY",
    "LIQUIDITY",
    "VALUE",
    "EARNINGS_YIELD",
    "GROWTH",
    "LEVERAGE",
)


def ewma_weights(length, half_life):
    if length <= 0:
        return np.array([], dtype=float)
    half_life = max(float(half_life), 1.0)
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / half_life)
    total = weights.sum()
    return weights / total if total > 0 else np.full(length, 1.0 / length)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    normalized = weights[valid] / weights[valid].sum()
    return float(np.dot(values[valid], normalized))


def weighted_variance(values, weights):
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    normalized = weights[valid] / weights[valid].sum()
    return float(np.dot(np.square(values[valid] - mean), normalized))


def weighted_beta(stock_excess, market_excess, half_life):
    stock = np.asarray(stock_excess, dtype=float)
    market = np.asarray(market_excess, dtype=float)
    valid = np.isfinite(stock) & np.isfinite(market)
    stock = stock[valid]
    market = market[valid]
    if len(stock) < 2:
        return np.nan
    weights = ewma_weights(len(stock), half_life)
    stock_mean = weighted_mean(stock, weights)
    market_mean = weighted_mean(market, weights)
    market_variance = weighted_variance(market, weights)
    if not np.isfinite(market_variance) or market_variance <= 1e-12:
        return np.nan
    covariance = np.dot(
        weights, (stock - stock_mean) * (market - market_mean)
    )
    return float(covariance / market_variance)


def compound_return(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > -1.0)]
    if not len(values):
        return np.nan
    return float(np.expm1(np.log1p(values).sum()))


def _window(values, end_index, length):
    start = max(0, end_index - int(length) + 1)
    return np.asarray(values[start : end_index + 1], dtype=float)


def compute_stock_weekly_features(daily, weekly_dates, config):
    if daily.empty:
        return []
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date")
    if frame.empty:
        return []

    dates = frame["trade_date"].to_numpy(dtype="datetime64[D]")
    daily_return = pd.to_numeric(frame["daily_return"], errors="coerce").to_numpy()
    market_return = pd.to_numeric(
        frame["market_return"], errors="coerce"
    ).to_numpy()
    risk_free = (
        pd.to_numeric(frame["risk_free_return"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    turnover = pd.to_numeric(
        frame["turnover_float"], errors="coerce"
    ).to_numpy()
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy()
    total_cap = pd.to_numeric(
        frame["total_market_cap"], errors="coerce"
    ).to_numpy()
    float_cap = pd.to_numeric(
        frame["float_market_cap"], errors="coerce"
    ).to_numpy()
    listed_state = frame["listed_state"].fillna("").astype(str).to_numpy()

    beta_window = int(config.get("beta_window_days", 252))
    beta_minimum = int(config.get("beta_minimum_days", 126))
    beta_half_life = float(config.get("beta_half_life_days", 63))
    momentum_lookback = int(config.get("momentum_lookback_days", 252))
    momentum_skip = int(config.get("momentum_skip_days", 21))
    liquidity_windows = tuple(
        int(value) for value in config.get("liquidity_windows_days", [21, 63, 252])
    )
    stale_days = int(config.get("maximum_stale_calendar_days", 10))

    rows = []
    previous_week = None
    for model_date_text in weekly_dates:
        model_date = np.datetime64(model_date_text, "D")
        index = int(np.searchsorted(dates, model_date, side="right") - 1)
        if index < 0:
            previous_week = model_date
            continue
        observation_date = dates[index]
        if int((model_date - observation_date).astype(int)) > stale_days:
            previous_week = model_date
            continue

        history_start = max(0, index - beta_window + 1)
        stock_excess = (
            daily_return[history_start : index + 1]
            - risk_free[history_start : index + 1]
        )
        market_excess = (
            market_return[history_start : index + 1]
            - risk_free[history_start : index + 1]
        )
        valid = np.isfinite(stock_excess) & np.isfinite(market_excess)
        beta = np.nan
        residual_volatility = np.nan
        if int(valid.sum()) >= beta_minimum:
            stock_valid = stock_excess[valid]
            market_valid = market_excess[valid]
            beta = weighted_beta(stock_valid, market_valid, beta_half_life)
            if np.isfinite(beta):
                residual = stock_valid - beta * market_valid
                residual_weights = ewma_weights(len(residual), beta_half_life)
                residual_variance = weighted_variance(residual, residual_weights)
                if np.isfinite(residual_variance):
                    residual_volatility = math.sqrt(
                        max(residual_variance, 0.0) * 244.0
                    )

        momentum_end = index - momentum_skip
        momentum_start = index - momentum_lookback + 1
        momentum = np.nan
        if momentum_start >= 0 and momentum_end >= momentum_start:
            momentum = compound_return(
                daily_return[momentum_start : momentum_end + 1]
            )

        liquidity_parts = []
        for length in liquidity_windows:
            values = _window(turnover, index, length)
            values = values[np.isfinite(values) & (values >= 0)]
            if len(values) >= max(10, min(length, length // 2)):
                turnover_fraction = values / 100.0
                liquidity_parts.append(math.log(max(turnover_fraction.sum(), 1e-8)))
        liquidity = (
            float(np.mean(liquidity_parts)) if liquidity_parts else np.nan
        )

        weekly_return = np.nan
        if previous_week is not None:
            mask = (dates > previous_week) & (dates <= model_date)
            weekly_return = compound_return(daily_return[mask])

        amount_window = _window(amount, index, 60)
        avg_amount = (
            float(np.nanmean(amount_window))
            if np.isfinite(amount_window).any()
            else np.nan
        )
        rows.append(
            {
                "model_date": str(model_date),
                "observation_date": str(observation_date),
                "weekly_return": weekly_return,
                "total_market_cap": total_cap[index],
                "float_market_cap": float_cap[index],
                "beta_raw": beta,
                "momentum_raw": momentum,
                "residual_volatility_raw": residual_volatility,
                "liquidity_raw": liquidity,
                "avg_amount_60": avg_amount,
                "history_days": index + 1,
                "listed_state": listed_state[index],
            }
        )
        previous_week = model_date
    return rows


def winsorize_mad(values, width=5.0):
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.empty:
        return series
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    if mad <= 1e-12:
        lower = float(finite.quantile(0.01))
        upper = float(finite.quantile(0.99))
    else:
        robust_sigma = 1.4826 * mad
        lower = median - float(width) * robust_sigma
        upper = median + float(width) * robust_sigma
    return series.clip(lower, upper)


def weighted_standardize(values, weights):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce")
    mean = weighted_mean(values.to_numpy(), weights.to_numpy())
    variance = weighted_variance(values.to_numpy(), weights.to_numpy())
    if not np.isfinite(variance) or variance <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - mean) / math.sqrt(variance)


def weighted_residualize(target, explanatory, weights):
    y = np.asarray(target, dtype=float)
    x = np.asarray(explanatory, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    w = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(x).all(axis=1)
    )
    result = np.full(len(y), np.nan, dtype=float)
    if valid.sum() <= x.shape[1] + 1:
        return result
    design = np.column_stack([np.ones(valid.sum()), x[valid]])
    root_weight = np.sqrt(w[valid])
    coefficients = np.linalg.lstsq(
        design * root_weight[:, None], y[valid] * root_weight, rcond=None
    )[0]
    result[valid] = y[valid] - design @ coefficients
    return result


def _fill_by_industry(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce")
    industry_median = values.groupby(frame["industry_group"]).transform("median")
    values = values.fillna(industry_median)
    global_median = values.median()
    return values.fillna(global_median if pd.notna(global_median) else 0.0)


def _average_available(*series):
    frame = pd.concat([pd.to_numeric(value, errors="coerce") for value in series], axis=1)
    return frame.mean(axis=1, skipna=True)


def _weighted_available(left, left_weight, right, right_weight):
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    numerator = left.fillna(0.0) * left_weight + right.fillna(0.0) * right_weight
    denominator = (
        left.notna().astype(float) * left_weight
        + right.notna().astype(float) * right_weight
    )
    return numerator / denominator.replace(0.0, np.nan)


def build_cross_section_exposures(raw, config):
    if raw.empty:
        return raw.copy()
    frame = raw.copy()
    numeric_columns = (
        "total_market_cap",
        "float_market_cap",
        "avg_amount_60",
        "history_days",
        "beta_raw",
        "momentum_raw",
        "residual_volatility_raw",
        "liquidity_raw",
        "total_assets",
        "total_liabilities",
        "parent_equity",
        "parent_net_profit_ttm",
        "operating_cashflow_ttm",
        "revenue_growth",
        "earnings_growth",
    )
    for column in numeric_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    minimum_history = int(config.get("minimum_history_days", 126))
    minimum_amount = float(config.get("minimum_average_amount", 5_000_000.0))
    frame = frame.loc[
        (frame["total_market_cap"] > 0)
        & (frame["history_days"] >= minimum_history)
        & (frame["avg_amount_60"] >= minimum_amount)
    ].copy()
    if bool(config.get("exclude_special_treatment", True)):
        frame = frame.loc[
            ~frame["listed_state"].fillna("").str.upper().str.contains("ST")
        ].copy()
    if frame.empty:
        return frame

    frame["industry_group"] = (
        frame["industry_group"].fillna("UNKNOWN").astype(str)
    )
    cap = frame["total_market_cap"]
    frame["size_raw"] = np.log(cap)
    frame["momentum_transformed"] = np.log1p(
        frame["momentum_raw"].clip(lower=-0.999999)
    )
    frame["residual_volatility_transformed"] = np.log(
        frame["residual_volatility_raw"].clip(lower=1e-6)
    )
    frame["value_raw"] = frame["parent_equity"] / cap
    earnings_to_price = frame["parent_net_profit_ttm"] / cap
    cashflow_to_price = frame["operating_cashflow_ttm"] / cap
    frame["earnings_yield_raw"] = _weighted_available(
        earnings_to_price, 0.67, cashflow_to_price, 0.33
    )
    frame["growth_raw"] = _average_available(
        frame["revenue_growth"], frame["earnings_growth"]
    )
    debt_to_assets = frame["total_liabilities"] / frame["total_assets"].replace(
        0, np.nan
    )
    market_leverage = np.log1p(
        frame["total_liabilities"].clip(lower=0) / cap
    )
    frame["leverage_raw"] = _average_available(debt_to_assets, market_leverage)

    cap_weights = np.sqrt(cap.clip(lower=1.0))
    cap_weights = cap_weights.clip(upper=cap_weights.quantile(0.99))
    frame["_weight"] = cap_weights
    raw_map = {
        "SIZE": "size_raw",
        "BETA": "beta_raw",
        "MOMENTUM": "momentum_transformed",
        "RESIDUAL_VOLATILITY": "residual_volatility_transformed",
        "LIQUIDITY": "liquidity_raw",
        "VALUE": "value_raw",
        "EARNINGS_YIELD": "earnings_yield_raw",
        "GROWTH": "growth_raw",
        "LEVERAGE": "leverage_raw",
    }
    mad_width = float(config.get("winsor_mad_width", 5.0))
    for factor, source in raw_map.items():
        frame[source] = winsorize_mad(frame[source], mad_width).to_numpy()
        frame[source] = _fill_by_industry(frame, source)
        frame[factor] = weighted_standardize(
            frame[source], frame["_weight"]
        ).to_numpy()

    nonlinear_raw = np.power(frame["SIZE"].to_numpy(), 3)
    nonlinear = weighted_residualize(
        nonlinear_raw, frame[["SIZE"]].to_numpy(), frame["_weight"].to_numpy()
    )
    frame["NONLINEAR_SIZE"] = weighted_standardize(
        nonlinear, frame["_weight"]
    ).to_numpy()

    residualized_volatility = weighted_residualize(
        frame["RESIDUAL_VOLATILITY"].to_numpy(),
        frame[["SIZE", "BETA"]].to_numpy(),
        frame["_weight"].to_numpy(),
    )
    frame["RESIDUAL_VOLATILITY"] = weighted_standardize(
        residualized_volatility, frame["_weight"]
    ).to_numpy()
    residualized_liquidity = weighted_residualize(
        frame["LIQUIDITY"].to_numpy(),
        frame[["SIZE"]].to_numpy(),
        frame["_weight"].to_numpy(),
    )
    frame["LIQUIDITY"] = weighted_standardize(
        residualized_liquidity, frame["_weight"]
    ).to_numpy()

    keep = [
        "model_date",
        "code",
        "observation_date",
        "financial_report_period",
        "financial_available_date",
        "industry_implement_date",
        "industry_group",
        "industry_code",
        "total_market_cap",
        "float_market_cap",
        "avg_amount_60",
        *STYLE_FACTORS,
    ]
    for column in keep:
        if column not in frame:
            frame[column] = None
    return frame[keep].reset_index(drop=True)


def robust_weighted_lstsq(design, target, weights, iterations=2, huber_k=2.5):
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    base_weight = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(y)
        & np.isfinite(base_weight)
        & (base_weight > 0)
        & np.isfinite(x).all(axis=1)
    )
    x = x[valid]
    y = y[valid]
    base_weight = base_weight[valid]
    if len(y) <= x.shape[1]:
        raise ValueError("Not enough observations for the factor regression")
    regression_weight = base_weight.copy()
    coefficients = np.zeros(x.shape[1], dtype=float)
    for _ in range(max(1, int(iterations))):
        root_weight = np.sqrt(regression_weight)
        coefficients = np.linalg.lstsq(
            x * root_weight[:, None], y * root_weight, rcond=None
        )[0]
        residual = y - x @ coefficients
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        if scale <= 1e-12:
            break
        standardized = np.abs(residual - median) / scale
        huber = np.ones(len(residual), dtype=float)
        outlier = standardized > huber_k
        huber[outlier] = huber_k / standardized[outlier]
        regression_weight = base_weight * huber
    residual = y - x @ coefficients
    return coefficients, residual, valid, regression_weight


def fit_factor_returns(previous_exposure, realized_returns, industry_groups, config):
    returns = realized_returns[["code", "weekly_return"]].copy()
    frame = previous_exposure.merge(returns, on="code", how="inner")
    frame["weekly_return"] = pd.to_numeric(
        frame["weekly_return"], errors="coerce"
    )
    frame = frame.dropna(subset=["weekly_return", "total_market_cap"])
    for factor in STYLE_FACTORS:
        frame[factor] = pd.to_numeric(frame[factor], errors="coerce")
    frame = frame.dropna(subset=list(STYLE_FACTORS))
    if frame.empty:
        raise ValueError("No usable rows for the factor-return regression")

    industries = [str(value) for value in industry_groups]
    frame["industry_group"] = (
        frame["industry_group"].fillna("UNKNOWN").astype(str)
    )
    if "UNKNOWN" not in industries:
        industries.append("UNKNOWN")
    cap_by_industry = (
        frame.groupby("industry_group")["total_market_cap"].sum().to_dict()
    )
    requested_base = str(config.get("industry_base_group", "40"))
    if cap_by_industry.get(requested_base, 0.0) > 0:
        base_industry = requested_base
    else:
        base_industry = max(cap_by_industry, key=cap_by_industry.get)
    base_cap = max(float(cap_by_industry.get(base_industry, 0.0)), 1.0)

    design_parts = [np.ones((len(frame), 1), dtype=float)]
    design_names = ["MARKET"]
    design_parts.append(frame.loc[:, STYLE_FACTORS].to_numpy(dtype=float))
    design_names.extend(STYLE_FACTORS)
    constrained_industries = []
    industry_values = frame["industry_group"].to_numpy()
    for industry in industries:
        if industry == base_industry:
            continue
        industry_cap = float(cap_by_industry.get(industry, 0.0))
        column = (industry_values == industry).astype(float)
        column[industry_values == base_industry] = -industry_cap / base_cap
        design_parts.append(column[:, None])
        constrained_industries.append(industry)
        design_names.append(f"INDUSTRY:{industry}")
    design = np.column_stack(design_parts)
    regression_weight = np.sqrt(
        frame["total_market_cap"].clip(lower=1.0).to_numpy(dtype=float)
    )
    cap_limit = np.quantile(regression_weight, 0.99)
    regression_weight = np.minimum(regression_weight, cap_limit)
    coefficients, residual, valid, final_weight = robust_weighted_lstsq(
        design,
        frame["weekly_return"].to_numpy(dtype=float),
        regression_weight,
        iterations=int(config.get("robust_regression_iterations", 2)),
        huber_k=float(config.get("huber_k", 2.5)),
    )
    factor_returns = dict(zip(design_names, coefficients))
    base_return = 0.0
    for industry in constrained_industries:
        base_return -= (
            float(cap_by_industry.get(industry, 0.0))
            * factor_returns[f"INDUSTRY:{industry}"]
            / base_cap
        )
    factor_returns[f"INDUSTRY:{base_industry}"] = base_return
    for industry in industries:
        factor_returns.setdefault(f"INDUSTRY:{industry}", 0.0)

    usable = frame.loc[valid].copy()
    usable["specific_return"] = residual
    usable["regression_weight"] = final_weight
    y = usable["weekly_return"].to_numpy(dtype=float)
    weighted_y_mean = weighted_mean(y, final_weight)
    total_ss = np.dot(final_weight, np.square(y - weighted_y_mean))
    residual_ss = np.dot(final_weight, np.square(residual))
    weighted_r_squared = (
        1.0 - residual_ss / total_ss if total_ss > 1e-16 else np.nan
    )
    diagnostics = {
        "regression_count": int(len(usable)),
        "factor_count": int(len(factor_returns)),
        "base_industry": base_industry,
        "weighted_r_squared": float(weighted_r_squared),
    }
    return factor_returns, usable, diagnostics


def nearest_psd(matrix, minimum_eigenvalue=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(
        float(minimum_eigenvalue),
        float(np.nanmax(eigenvalues)) * 1e-10 if len(eigenvalues) else 1e-10,
    )
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def ewma_newey_west_covariance(
    factor_returns,
    half_life=52,
    newey_west_lags=2,
    shrinkage=0.10,
    annualization=52,
):
    values = np.asarray(factor_returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("At least two factor-return observations are required")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    weights = ewma_weights(values.shape[0], half_life)
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    lags = min(int(newey_west_lags), values.shape[0] - 1)
    for lag in range(1, lags + 1):
        lag_weights = weights[lag:]
        lag_weights = lag_weights / lag_weights.sum()
        gamma = (centered[lag:] * lag_weights[:, None]).T @ centered[:-lag]
        kernel = 1.0 - lag / (lags + 1.0)
        covariance += kernel * (gamma + gamma.T)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage) * diagonal
    )
    covariance *= float(annualization)
    return nearest_psd(covariance)


def estimate_specific_variances(
    specific_history,
    current_exposure,
    half_life=26,
    minimum_observations=20,
    shrinkage=0.20,
    annualization=52,
):
    rows = []
    industry_lookup = current_exposure.set_index("code")["industry_group"].to_dict()
    for code in current_exposure["code"].astype(str):
        values = np.asarray(specific_history.get(code, []), dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < int(minimum_observations):
            continue
        weights = ewma_weights(len(values), half_life)
        variance = weighted_variance(values, weights) * float(annualization)
        rows.append(
            {
                "code": code,
                "industry_group": str(industry_lookup.get(code, "UNKNOWN")),
                "raw_specific_variance": variance,
                "specific_observations": len(values),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=(
                "code",
                "industry_group",
                "specific_variance",
                "specific_volatility",
                "specific_observations",
            )
        )
    frame = pd.DataFrame(rows)
    industry_median = frame.groupby("industry_group")[
        "raw_specific_variance"
    ].transform("median")
    global_median = float(frame["raw_specific_variance"].median())
    target = industry_median.fillna(global_median)
    frame["specific_variance"] = (
        (1.0 - float(shrinkage)) * frame["raw_specific_variance"]
        + float(shrinkage) * target
    ).clip(lower=1e-10)
    frame["specific_volatility"] = np.sqrt(frame["specific_variance"])
    return frame[
        [
            "code",
            "industry_group",
            "specific_variance",
            "specific_volatility",
            "specific_observations",
        ]
    ]


def portfolio_variance(weights, exposures, factor_covariance, specific_variance):
    weights = pd.Series(weights, dtype=float)
    common = exposures.set_index("code").reindex(weights.index)
    factor_names = list(factor_covariance.index)
    matrix = np.zeros((len(common), len(factor_names)), dtype=float)
    for column_index, factor in enumerate(factor_names):
        if factor == "MARKET":
            matrix[:, column_index] = 1.0
        elif factor in STYLE_FACTORS:
            matrix[:, column_index] = pd.to_numeric(
                common[factor], errors="coerce"
            ).fillna(0.0)
        elif factor.startswith("INDUSTRY:"):
            industry = factor.split(":", 1)[1]
            matrix[:, column_index] = (
                common["industry_group"].astype(str) == industry
            ).astype(float)
    portfolio_exposure = weights.to_numpy() @ matrix
    common_variance = float(
        portfolio_exposure
        @ factor_covariance.to_numpy(dtype=float)
        @ portfolio_exposure
    )
    specific = pd.Series(specific_variance, dtype=float).reindex(weights.index)
    specific_component = float(
        np.nansum(np.square(weights.to_numpy()) * specific.to_numpy())
    )
    return common_variance + specific_component
