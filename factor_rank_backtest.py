import argparse
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_utils import (
    buy_order_size_rules,
    mandatory_trade_cost,
    round_target_shares_for_code,
    should_rebalance_on_date,
    trading_cost_snapshot,
    write_excel_workbook,
)


DEFAULT_DATABASE = Path("data/processed/stock_daily.sqlite")
DEFAULT_OUTPUT_DIR = Path("outputs/backtest")
DEFAULT_INDUSTRY_EVENT_SCORES = Path("data/processed/industry_event_scores.csv")
DEFAULT_EVENT_REGIME_SIGNALS = Path("data/processed/event_regime_signals.csv")


def safe_number(value, default=np.nan):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def zscore(series):
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - series.mean(skipna=True)) / std).clip(-3, 3)


def factor_zscore(frame, column, direction=1.0, industry_neutral=True):
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce") * float(direction)
    global_score = zscore(values)
    if not industry_neutral or "industry_1" not in frame.columns:
        return global_score.fillna(0.0)

    temp = pd.DataFrame({"industry_1": frame["industry_1"].fillna("UNKNOWN"), "value": values}, index=frame.index)

    def score_group(group):
        if group.notna().sum() < 5:
            return pd.Series(np.nan, index=group.index)
        return zscore(group)

    industry_score = temp.groupby("industry_1", group_keys=False)["value"].apply(score_group)
    return industry_score.fillna(global_score).fillna(0.0)


def calc_market_cap_proxy(amount, turnover_total):
    amount = pd.to_numeric(amount, errors="coerce")
    turnover_total = pd.to_numeric(turnover_total, errors="coerce")
    proxy = amount * 100.0 / turnover_total.where(turnover_total > 0)
    return proxy.replace([np.inf, -np.inf], np.nan)


def trailing_median_market_cap(amount, turnover_total, window=20):
    proxy = calc_market_cap_proxy(amount, turnover_total)
    return proxy.tail(int(window)).median()


def trailing_max_drawdown(close, window=120):
    values = pd.to_numeric(close, errors="coerce").dropna().tail(int(window))
    if len(values) < 20:
        return np.nan
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def trailing_beta(group, market_returns, window=120):
    sample = group[["trade_date", "daily_return"]].tail(int(window)).copy()
    sample["daily_return"] = pd.to_numeric(sample["daily_return"], errors="coerce")
    sample["market_return"] = sample["trade_date"].map(market_returns)
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return", "market_return"])
    if len(sample) < max(40, int(window) // 2):
        return np.nan
    market_var = sample["market_return"].var(ddof=1)
    if pd.isna(market_var) or market_var <= 0:
        return np.nan
    return float(sample["daily_return"].cov(sample["market_return"]) / market_var)


def compound_return(values):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().clip(-0.12, 0.12)
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def build_industry_metrics(window, market_returns):
    frame = window[["trade_date", "industry_1", "daily_return"]].copy()
    frame["industry_1"] = frame["industry_1"].fillna("UNKNOWN").astype(str)
    frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce").clip(-0.12, 0.12)
    industry_daily = (
        frame.dropna(subset=["daily_return"])
        .groupby(["industry_1", "trade_date"], as_index=False)["daily_return"]
        .mean()
        .sort_values(["industry_1", "trade_date"])
    )
    market_series = pd.Series(market_returns).sort_index()
    market_return_20 = compound_return(market_series.tail(20))
    market_return_60 = compound_return(market_series.tail(60))
    rows = []
    for industry, group in industry_daily.groupby("industry_1"):
        returns = pd.to_numeric(group["daily_return"], errors="coerce").dropna()
        if len(returns) < 20:
            continue
        industry_return_20 = compound_return(returns.tail(20))
        industry_return_60 = compound_return(returns.tail(60))
        industry_volatility_60 = returns.tail(60).std(ddof=1) * math.sqrt(244) if len(returns) >= 20 else np.nan
        rows.append(
            {
                "industry_1": industry,
                "industry_return_20": industry_return_20,
                "industry_return_60": industry_return_60,
                "industry_relative_20": industry_return_20 - market_return_20 if pd.notna(market_return_20) else np.nan,
                "industry_relative_60": industry_return_60 - market_return_60 if pd.notna(market_return_60) else np.nan,
                "industry_volatility_60": industry_volatility_60,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["industry_1"])
    return pd.DataFrame(rows).set_index("industry_1")


def price_limit_rate(code):
    code = str(code).zfill(6)
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4", "920")):
        return 0.30
    return 0.10


def trading_dates(conn):
    rows = conn.execute("SELECT DISTINCT trade_date FROM stock_daily ORDER BY trade_date").fetchall()
    return [row[0] for row in rows]


def load_prices(conn, start_date, end_date):
    return pd.read_sql_query(
        """
        SELECT code, name, trade_date, prev_close, open, high, low, close,
               amount, daily_return, capital_return, adj_factor, turnover_total,
               listed_state, industry_1, industry_2
        FROM stock_daily
        WHERE trade_date BETWEEN ? AND ?
        """,
        conn,
        params=(start_date, end_date),
        parse_dates=[],
    )


def load_financial_factors(conn):
    try:
        frame = pd.read_sql_query("SELECT * FROM financial_factors", conn)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return frame
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def load_industry_event_scores(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["decision_date", "industry_1", "event_score", "event_confidence"])
    frame = pd.read_csv(path, dtype={"industry_1": str}, encoding="utf-8-sig")
    if frame.empty:
        return pd.DataFrame(columns=["decision_date", "industry_1", "event_score", "event_confidence"])
    rename = {}
    if "date" in frame.columns and "decision_date" not in frame.columns:
        rename["date"] = "decision_date"
    if "score" in frame.columns and "event_score" not in frame.columns:
        rename["score"] = "event_score"
    frame = frame.rename(columns=rename)
    for column in ["decision_date", "industry_1", "event_score"]:
        if column not in frame.columns:
            frame[column] = np.nan
    if "event_confidence" not in frame.columns:
        frame["event_confidence"] = 1.0
    frame["decision_date"] = frame["decision_date"].astype(str).str.slice(0, 10)
    frame["industry_1"] = frame["industry_1"].fillna("UNKNOWN").astype(str)
    frame["event_score"] = pd.to_numeric(frame["event_score"], errors="coerce").fillna(0.0).clip(-3, 3)
    frame["event_confidence"] = pd.to_numeric(frame["event_confidence"], errors="coerce").fillna(1.0).clip(0, 1)
    return frame[["decision_date", "industry_1", "event_score", "event_confidence"]].sort_values(
        ["industry_1", "decision_date"]
    )


def latest_industry_event_scores(event_scores, decision_date):
    if event_scores is None or event_scores.empty:
        return {}
    rows = event_scores.loc[event_scores["decision_date"].astype(str) <= str(decision_date)].copy()
    if rows.empty:
        return {}
    rows = rows.sort_values(["industry_1", "decision_date"]).groupby("industry_1", as_index=False).tail(1)
    return rows.set_index("industry_1")[["event_score", "event_confidence"]].to_dict("index")


def load_event_regime_signals(path):
    path = Path(path)
    columns = ["decision_date", "equity_multiplier", "event_risk_score", "event_confidence", "reason"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rename = {}
    if "date" in frame.columns and "decision_date" not in frame.columns:
        rename["date"] = "decision_date"
    if "confidence" in frame.columns and "event_confidence" not in frame.columns:
        rename["confidence"] = "event_confidence"
    frame = frame.rename(columns=rename)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["decision_date"] = frame["decision_date"].astype(str).str.slice(0, 10)
    frame["equity_multiplier"] = pd.to_numeric(frame["equity_multiplier"], errors="coerce").fillna(1.0).clip(0.60, 1.20)
    frame["event_risk_score"] = pd.to_numeric(frame["event_risk_score"], errors="coerce").fillna(0.0).clip(-3, 3)
    frame["event_confidence"] = pd.to_numeric(frame["event_confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    frame["reason"] = frame["reason"].fillna("").astype(str)
    return frame[columns].sort_values("decision_date")


def latest_event_regime_signal(event_regime_signals, decision_date):
    default = {
        "equity_multiplier": 1.0,
        "event_risk_score": 0.0,
        "event_confidence": 0.0,
        "event_regime_reason": "",
    }
    if event_regime_signals is None or event_regime_signals.empty:
        return default
    rows = event_regime_signals.loc[event_regime_signals["decision_date"].astype(str) <= str(decision_date)].copy()
    if rows.empty:
        return default
    row = rows.sort_values("decision_date").tail(1).iloc[0]
    return {
        "equity_multiplier": float(row.get("equity_multiplier", 1.0)),
        "event_risk_score": float(row.get("event_risk_score", 0.0)),
        "event_confidence": float(row.get("event_confidence", 0.0)),
        "event_regime_reason": str(row.get("reason", "")),
    }


def build_market_state(prices, args):
    frame = prices[["code", "trade_date", "close", "daily_return"]].copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce").clip(-0.12, 0.12)
    frame = frame.sort_values(["code", "trade_date"])
    breadth_window = int(args.market_breadth_window)
    frame["stock_ma"] = frame.groupby("code")["close"].transform(
        lambda values: values.rolling(breadth_window, min_periods=max(20, breadth_window // 2)).mean()
    )
    frame["above_stock_ma"] = frame["close"] > frame["stock_ma"]

    daily = frame.groupby("trade_date", as_index=False).agg(
        market_return=("daily_return", "mean"),
        market_breadth=("above_stock_ma", "mean"),
        stock_count=("code", "nunique"),
    )
    daily["market_return"] = pd.to_numeric(daily["market_return"], errors="coerce").fillna(0.0)
    daily["market_index"] = (1.0 + daily["market_return"]).cumprod()
    short_window = int(args.market_short_window)
    long_window = int(args.market_long_window)
    daily["market_ma_short"] = daily["market_index"].rolling(short_window, min_periods=max(20, short_window // 2)).mean()
    daily["market_ma_long"] = daily["market_index"].rolling(long_window, min_periods=max(60, long_window // 2)).mean()
    daily["market_return_20"] = daily["market_index"] / daily["market_index"].shift(20) - 1.0
    daily["market_return_60"] = daily["market_index"] / daily["market_index"].shift(60) - 1.0
    daily["market_volatility_20"] = daily["market_return"].rolling(20, min_periods=10).std(ddof=1) * math.sqrt(244)
    daily = daily.set_index("trade_date")
    return daily


def target_equity_from_market(market_state, decision_date, previous_total, peak_total, args, event_regime_signals=None):
    base_equity = max(0.0, min(1.0, 1.0 - float(args.cash_weight)))
    if bool(args.disable_market_regime):
        row = {}
        state = "disabled"
        target = base_equity
    else:
        if decision_date in market_state.index:
            row = market_state.loc[decision_date].to_dict()
        else:
            row = {}
        index = row.get("market_index", np.nan)
        ma_short = row.get("market_ma_short", np.nan)
        ma_long = row.get("market_ma_long", np.nan)
        breadth = row.get("market_breadth", np.nan)
        return_20 = row.get("market_return_20", np.nan)

        if pd.notna(return_20) and return_20 <= float(args.crash_return_20):
            state = "crash"
            target = float(args.crash_equity_weight)
        elif (
            pd.notna(index)
            and pd.notna(ma_long)
            and pd.notna(breadth)
            and index < ma_long
            and breadth < float(args.bear_breadth_max)
        ):
            state = "bear"
            target = float(args.bear_equity_weight)
        elif (
            pd.notna(index)
            and pd.notna(ma_short)
            and pd.notna(ma_long)
            and pd.notna(breadth)
            and pd.notna(return_20)
            and index > ma_short
            and index > ma_long
            and breadth >= float(args.bull_breadth_min)
            and return_20 > 0
        ):
            state = "bull"
            target = base_equity
        elif (
            (pd.notna(index) and pd.notna(ma_short) and index > ma_short)
            or (pd.notna(breadth) and breadth >= float(args.neutral_breadth_min))
        ):
            state = "neutral"
            target = float(args.neutral_equity_weight)
        else:
            state = "defensive"
            target = float(args.defensive_equity_weight)

    target = min(base_equity, max(0.0, target))
    vol = row.get("market_volatility_20", np.nan) if isinstance(row, dict) else np.nan
    if pd.notna(vol) and vol > 0 and float(args.target_market_volatility) > 0:
        target *= min(1.0, float(args.target_market_volatility) / float(vol))

    drawdown = previous_total / peak_total - 1.0 if peak_total and peak_total > 0 else 0.0
    if drawdown <= float(args.severe_drawdown_threshold):
        target = min(target, float(args.severe_drawdown_equity_weight))
    elif drawdown <= float(args.drawdown_reduce_threshold):
        target *= float(args.drawdown_reduce_multiplier)

    event_regime = latest_event_regime_signal(event_regime_signals, decision_date)
    if bool(getattr(args, "enable_event_regime", False)):
        target *= float(event_regime["equity_multiplier"])

    return {
        "target_equity_weight": float(max(0.0, min(base_equity, target))),
        "market_state": state,
        "market_index": row.get("market_index", np.nan) if isinstance(row, dict) else np.nan,
        "market_breadth": row.get("market_breadth", np.nan) if isinstance(row, dict) else np.nan,
        "market_return_20": row.get("market_return_20", np.nan) if isinstance(row, dict) else np.nan,
        "market_volatility_20": vol,
        "portfolio_drawdown": float(drawdown),
        "event_equity_multiplier": float(event_regime["equity_multiplier"]),
        "event_risk_score": float(event_regime["event_risk_score"]),
        "event_regime_confidence": float(event_regime["event_confidence"]),
        "event_regime_reason": event_regime["event_regime_reason"],
    }


def latest_financial_by_code(financial, decision_date):
    if financial.empty or "announcement_date" not in financial.columns:
        return pd.DataFrame(columns=["code", "fundamental_alpha"])
    rows = financial.loc[financial["announcement_date"].astype(str) <= str(decision_date)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["code", "fundamental_alpha"])
    rows = rows.sort_values(["code", "announcement_date", "report_period"])
    return rows.groupby("code", as_index=False).tail(1)


def feature_snapshot(history, financial, decision_date, args, industry_event_scores=None):
    usable_dates = history.loc[history["trade_date"] <= decision_date, "trade_date"].drop_duplicates().sort_values()
    if usable_dates.empty:
        return pd.DataFrame()
    history_window = max(
        int(args.feature_history_days),
        int(args.min_history_days) + 30,
        252 + 25,
    )
    cutoff_date = usable_dates.iloc[-history_window] if len(usable_dates) > history_window else usable_dates.iloc[0]
    window = history.loc[(history["trade_date"] >= cutoff_date) & (history["trade_date"] <= decision_date)].copy()
    if window.empty:
        return pd.DataFrame()
    window = window.sort_values(["code", "trade_date"])
    market_returns = (
        window.groupby("trade_date")["daily_return"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce").clip(-0.12, 0.12).mean())
        .to_dict()
    )
    industry_metrics = build_industry_metrics(window, market_returns)
    event_by_industry = latest_industry_event_scores(industry_event_scores, decision_date)
    latest = window.groupby("code", as_index=False).tail(1).set_index("code")
    grouped = window.groupby("code", sort=False)

    stats = pd.DataFrame(index=latest.index)
    stats["code"] = stats.index
    stats["name"] = latest["name"]
    stats["close"] = pd.to_numeric(latest["close"], errors="coerce")
    stats["latest_trade_date"] = latest["trade_date"].astype(str)
    stats["industry_1"] = latest.get("industry_1", "")
    stats["listed_state"] = latest.get("listed_state", "")
    stats["days"] = grouped["trade_date"].count()
    stats["avg_amount_60"] = grouped["amount"].apply(lambda x: pd.to_numeric(x, errors="coerce").tail(60).mean())
    stats["market_cap_proxy_20"] = grouped.apply(
        lambda x: trailing_median_market_cap(x["amount"], x["turnover_total"], int(args.market_cap_proxy_window))
    )

    def pct_change_from_tail(close, lag):
        values = pd.to_numeric(close, errors="coerce").dropna()
        if len(values) <= lag:
            return np.nan
        return values.iloc[-1] / values.iloc[-lag - 1] - 1.0

    def mom_skip(close, lookback, skip):
        values = pd.to_numeric(close, errors="coerce").dropna()
        if len(values) <= lookback:
            return np.nan
        end = -skip - 1
        start = -lookback - 1
        return values.iloc[end] / values.iloc[start] - 1.0

    stats["mom_252_skip_20"] = grouped["close"].apply(lambda x: mom_skip(x, 252, 20))
    stats["mom_120"] = grouped["close"].apply(lambda x: pct_change_from_tail(x, 120))
    stats["mom_60"] = grouped["close"].apply(lambda x: pct_change_from_tail(x, 60))
    stats["reversal_20"] = -grouped["close"].apply(lambda x: pct_change_from_tail(x, 20))
    stats["volatility_120"] = grouped["daily_return"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").tail(120).std(ddof=1) * math.sqrt(244)
    )
    stats["beta_120"] = grouped.apply(lambda x: trailing_beta(x, market_returns, 120))
    stats["turnover_20"] = grouped["turnover_total"].apply(lambda x: pd.to_numeric(x, errors="coerce").tail(20).mean())
    stats["drawdown_120"] = grouped["close"].apply(lambda x: trailing_max_drawdown(x, 120))
    for column in [
        "industry_return_20",
        "industry_return_60",
        "industry_relative_20",
        "industry_relative_60",
        "industry_volatility_60",
    ]:
        if not industry_metrics.empty and column in industry_metrics.columns:
            stats[column] = stats["industry_1"].fillna("UNKNOWN").map(industry_metrics[column])
        else:
            stats[column] = np.nan
    stats["industry_event_score"] = stats["industry_1"].fillna("UNKNOWN").map(
        lambda industry: event_by_industry.get(str(industry), {}).get("event_score", 0.0)
    )
    stats["industry_event_confidence"] = stats["industry_1"].fillna("UNKNOWN").map(
        lambda industry: event_by_industry.get(str(industry), {}).get("event_confidence", 0.0)
    )

    fin = latest_financial_by_code(financial, decision_date)
    if not fin.empty:
        fin_columns = [
            "code",
            "fundamental_alpha",
            "weighted_roe",
            "gross_margin",
            "debt_to_assets",
            "ocf_to_profit",
            "rd_to_revenue",
            "net_profit_yoy",
            "revenue_yoy",
            "net_profit_attributable",
            "profitability_score",
            "growth_score",
            "cashflow_quality_score",
            "leverage_score",
            "rd_intensity_score",
            "investment_score",
            "shareholder_yield_score",
            "accrual_quality_score",
        ]
        stats = stats.reset_index(drop=True).merge(fin[[col for col in fin_columns if col in fin.columns]], on="code", how="left")
    else:
        stats = stats.reset_index(drop=True)
        stats["fundamental_alpha"] = np.nan

    name = stats["name"].fillna("").astype(str)
    eligible = (
        (stats["days"] >= int(args.min_history_days))
        & (stats["avg_amount_60"] >= float(args.min_avg_amount))
        & (stats["listed_state"].fillna("").astype(str).eq("Norm"))
        & stats["latest_trade_date"].eq(str(decision_date))
        & stats["close"].notna()
        & ~name.str.contains("ST", case=False, na=False)
        & stats["market_cap_proxy_20"].notna()
    )
    stats = stats.loc[eligible].copy()
    if stats.empty:
        return stats
    if float(args.min_market_cap_quantile) > 0:
        market_cap_floor = stats["market_cap_proxy_20"].quantile(float(args.min_market_cap_quantile))
        stats = stats.loc[stats["market_cap_proxy_20"] >= market_cap_floor].copy()
        if stats.empty:
            return stats

    for column in [
        "fundamental_alpha",
        "weighted_roe",
        "gross_margin",
        "debt_to_assets",
        "ocf_to_profit",
        "rd_to_revenue",
        "net_profit_yoy",
        "revenue_yoy",
        "net_profit_attributable",
        "profitability_score",
        "growth_score",
        "cashflow_quality_score",
        "leverage_score",
        "rd_intensity_score",
        "investment_score",
        "shareholder_yield_score",
        "accrual_quality_score",
    ]:
        if column not in stats.columns:
            stats[column] = np.nan
    stats["earnings_yield_proxy"] = pd.to_numeric(stats["net_profit_attributable"], errors="coerce") / stats[
        "market_cap_proxy_20"
    ]

    industry_neutral = not bool(args.disable_industry_neutral_factors)
    quality_score = (
        0.22 * factor_zscore(stats, "profitability_score", industry_neutral=industry_neutral)
        + 0.15 * factor_zscore(stats, "weighted_roe", industry_neutral=industry_neutral)
        + 0.12 * factor_zscore(stats, "gross_margin", industry_neutral=industry_neutral)
        + 0.16 * factor_zscore(stats, "cashflow_quality_score", industry_neutral=industry_neutral)
        + 0.10 * factor_zscore(stats, "ocf_to_profit", industry_neutral=industry_neutral)
        + 0.10 * factor_zscore(stats, "investment_score", industry_neutral=industry_neutral)
        + 0.08 * factor_zscore(stats, "accrual_quality_score", industry_neutral=industry_neutral)
        + 0.07 * factor_zscore(stats, "leverage_score", industry_neutral=industry_neutral)
    )
    growth_score = (
        0.45 * factor_zscore(stats, "growth_score", industry_neutral=industry_neutral)
        + 0.30 * factor_zscore(stats, "net_profit_yoy", industry_neutral=industry_neutral)
        + 0.25 * factor_zscore(stats, "revenue_yoy", industry_neutral=industry_neutral)
    )
    value_score = (
        0.65 * factor_zscore(stats, "earnings_yield_proxy", industry_neutral=industry_neutral)
        + 0.35 * factor_zscore(stats, "shareholder_yield_score", industry_neutral=industry_neutral)
    )
    momentum_score = (
        0.60 * factor_zscore(stats, "mom_252_skip_20", industry_neutral=False)
        + 0.25 * factor_zscore(stats, "mom_120", industry_neutral=False)
        + 0.15 * factor_zscore(stats, "mom_60", industry_neutral=False)
    )
    low_volatility_score = factor_zscore(stats, "volatility_120", direction=-1.0, industry_neutral=False)
    low_beta_score = factor_zscore(stats, "beta_120", direction=-1.0, industry_neutral=False)
    low_turnover_score = factor_zscore(stats, "turnover_20", direction=-1.0, industry_neutral=False)
    lower_drawdown_score = factor_zscore(stats, "drawdown_120", industry_neutral=False)
    reversal_score = factor_zscore(stats, "reversal_20", industry_neutral=False)
    medium_reversal_score = (
        0.60 * factor_zscore(stats, "mom_60", direction=-1.0, industry_neutral=False)
        + 0.40 * factor_zscore(stats, "mom_120", direction=-1.0, industry_neutral=False)
    )
    industry_trend_score = (
        0.45 * factor_zscore(stats, "industry_relative_20", industry_neutral=False)
        + 0.35 * factor_zscore(stats, "industry_relative_60", industry_neutral=False)
        + 0.20 * factor_zscore(stats, "industry_volatility_60", direction=-1.0, industry_neutral=False)
    )
    industry_event_score = factor_zscore(stats, "industry_event_score", industry_neutral=False).fillna(0.0)
    industry_event_score = industry_event_score * pd.to_numeric(stats["industry_event_confidence"], errors="coerce").fillna(0.0)
    defensive_score = (
        0.45 * low_volatility_score
        + 0.30 * low_turnover_score
        + 0.25 * lower_drawdown_score
    )
    stats["quality_score"] = quality_score
    stats["growth_score_ranked"] = growth_score
    stats["value_score"] = value_score
    stats["momentum_score"] = momentum_score
    stats["defensive_score"] = defensive_score
    stats["low_volatility_score"] = low_volatility_score
    stats["low_beta_score"] = low_beta_score
    stats["low_turnover_score"] = low_turnover_score
    stats["lower_drawdown_score"] = lower_drawdown_score
    stats["reversal_score"] = reversal_score
    stats["medium_reversal_score"] = medium_reversal_score
    stats["industry_trend_score"] = industry_trend_score
    stats["industry_event_score_ranked"] = industry_event_score

    score_profile = str(getattr(args, "score_profile", "diagnostic_defensive")).strip().lower()
    if score_profile == "diagnostic_defensive":
        stats["score"] = (
            0.32 * low_volatility_score
            + 0.24 * low_turnover_score
            + 0.20 * reversal_score
            + 0.14 * lower_drawdown_score
            + 0.05 * factor_zscore(stats, "investment_score", industry_neutral=industry_neutral)
            + 0.05 * quality_score
        )
    elif score_profile == "low_risk_only":
        stats["score"] = (
            0.42 * low_volatility_score
            + 0.28 * low_turnover_score
            + 0.20 * lower_drawdown_score
            + 0.10 * quality_score
        )
    elif score_profile == "low_beta_defensive":
        stats["score"] = (
            0.30 * low_beta_score
            + 0.24 * low_volatility_score
            + 0.20 * low_turnover_score
            + 0.16 * reversal_score
            + 0.10 * lower_drawdown_score
        )
    elif score_profile == "low_beta_industry":
        stats["score"] = (
            0.24 * low_beta_score
            + 0.20 * low_volatility_score
            + 0.16 * low_turnover_score
            + 0.14 * reversal_score
            + 0.10 * lower_drawdown_score
            + 0.12 * industry_trend_score
            + float(getattr(args, "industry_event_weight", 0.04)) * industry_event_score
        )
    elif score_profile == "contrarian_low_risk":
        stats["score"] = (
            0.28 * low_volatility_score
            + 0.22 * low_turnover_score
            + 0.22 * reversal_score
            + 0.18 * medium_reversal_score
            + 0.10 * lower_drawdown_score
        )
    else:
        stats["score"] = (
            0.28 * quality_score
            + 0.12 * growth_score
            + 0.12 * value_score
            + 0.20 * momentum_score
            + 0.08 * reversal_score
            + 0.15 * defensive_score
            + 0.05 * factor_zscore(stats, "fundamental_alpha", industry_neutral=industry_neutral)
        )
    return stats.sort_values("score", ascending=False).reset_index(drop=True)


def build_targets(features, holdings, args, target_equity_weight, portfolio_value):
    target_equity_weight = max(0.0, min(1.0, float(target_equity_weight)))
    if target_equity_weight <= 0 or features.empty:
        return {}
    ranked = features.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    rank_by_code = ranked.set_index("code")["rank"].to_dict()
    industry_by_code = ranked.set_index("code")["industry_1"].fillna("UNKNOWN").to_dict()
    min_trade_count = int(math.floor(float(portfolio_value) * target_equity_weight / max(float(args.min_trade_value), 1.0)))
    dynamic_target_count = min(int(args.target_count), max(1, min_trade_count))
    max_industry_count = max(1, int(math.ceil(dynamic_target_count * float(args.max_industry_weight))))
    industry_counts = {}

    def can_add(code):
        industry = industry_by_code.get(code, "UNKNOWN")
        return industry_counts.get(industry, 0) < max_industry_count

    def add_code(code, selected):
        if code in selected or not can_add(code):
            return False
        selected.append(code)
        industry = industry_by_code.get(code, "UNKNOWN")
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return True

    kept = [
        code
        for code, shares in holdings.items()
        if int(shares) > 0 and rank_by_code.get(code, 10**9) <= int(args.sell_rank)
    ]
    selected = []
    for code in sorted(kept, key=lambda item: rank_by_code.get(item, 10**9)):
        if len(selected) >= dynamic_target_count:
            break
        add_code(code, selected)
    for code in ranked["code"].tolist():
        if len(selected) >= dynamic_target_count:
            break
        if code not in selected and rank_by_code.get(code, 10**9) <= int(args.buy_rank):
            add_code(code, selected)
    if not selected:
        return {}
    equity_weight = target_equity_weight
    per_stock = min(float(args.max_stock_weight), equity_weight / len(selected))
    return {code: per_stock for code in selected}


def blocked_by_price_limit(code, row, side, args):
    if bool(args.disable_limit_trade_filter):
        return False
    prev_close = row.get("prev_close")
    open_price = row.get("open")
    if prev_close is None or open_price is None or pd.isna(prev_close) or pd.isna(open_price):
        return False
    if prev_close <= 0 or open_price <= 0:
        return False
    open_return = float(open_price) / float(prev_close) - 1.0
    limit = price_limit_rate(code)
    buffer = float(args.limit_trade_buffer)
    if side == "BUY" and open_return >= limit - buffer:
        return True
    if side == "SELL" and open_return <= -limit + buffer:
        return True
    return False


def execute_trades(trade_date, decision_date, holdings, cash, targets, prices, portfolio_open_value, args):
    all_codes = sorted(set(holdings).union(targets))
    planned = []
    for code in all_codes:
        row = prices.get(code)
        if row is None:
            continue
        price = row.get("open")
        if price is None or pd.isna(price) or price <= 0:
            continue
        current_shares = int(holdings.get(code, 0))
        target_weight = max(0.0, float(targets.get(code, 0.0)))
        target_value = target_weight * portfolio_open_value
        target_shares = round_target_shares_for_code(target_value, price, code) if target_weight > 0 else 0
        trade_shares = int(target_shares - current_shares)
        side = "BUY" if trade_shares > 0 else "SELL"
        gross = abs(trade_shares) * float(price)
        below_min_buy = side == "BUY" and gross < float(args.min_trade_value)
        if trade_shares == 0 or below_min_buy or blocked_by_price_limit(code, row, side, args):
            continue
        planned.append(
            {
                "trade_date": trade_date,
                "decision_date": decision_date,
                "code": code,
                "name": row.get("name", ""),
                "side": side,
                "shares": abs(trade_shares),
                "price": float(price),
                "gross_amount": gross,
                "target_weight": target_weight,
            }
        )

    executed = []
    for order in [row for row in planned if row["side"] == "SELL"]:
        code = order["code"]
        current_shares = int(holdings.get(code, 0))
        shares = min(int(order["shares"]), current_shares)
        if shares < current_shares:
            minimum, increment = buy_order_size_rules(code)
            if shares < minimum:
                continue
            shares = int(minimum + math.floor((shares - minimum) / increment) * increment)
        if shares <= 0:
            continue
        gross = shares * float(order["price"])
        fee = mandatory_trade_cost(gross, "SELL", trade_date, code)
        holdings[code] = int(holdings.get(code, 0)) - shares
        if holdings[code] == 0:
            holdings.pop(code, None)
        cash += gross - fee
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)

    buys = sorted([row for row in planned if row["side"] == "BUY"], key=lambda x: x["target_weight"], reverse=True)
    for order in buys:
        code = order["code"]
        minimum, increment = buy_order_size_rules(code)
        shares = int(order["shares"])
        if shares < minimum:
            continue
        shares = int(minimum + math.floor((shares - minimum) / increment) * increment)
        while shares >= minimum:
            gross = shares * float(order["price"])
            fee = mandatory_trade_cost(gross, "BUY", trade_date, code)
            if gross + fee <= cash + 1e-8:
                break
            shares -= increment
        if shares < minimum:
            continue
        gross = shares * float(order["price"])
        if gross < float(args.min_trade_value):
            continue
        fee = mandatory_trade_cost(gross, "BUY", trade_date, code)
        cash -= gross + fee
        holdings[code] = int(holdings.get(code, 0)) + shares
        order.update({"shares": shares, "gross_amount": gross, "fee": fee, "cash_after": cash})
        executed.append(order)
    return float(cash), executed


def value_portfolio(holdings, cash, price_rows, column, last_close):
    total = float(cash)
    for code, shares in holdings.items():
        row = price_rows.get(code)
        price = None if row is None else row.get(column)
        if price is None or pd.isna(price) or price <= 0:
            price = last_close.get(code)
        if price is not None and not pd.isna(price) and price > 0:
            total += int(shares) * float(price)
    return float(total)


def apply_corporate_actions_before_open(holdings, price_rows, last_close):
    """Adjust overnight holdings for share distributions and accrue cash distributions.

    RESSET's daily return includes distributions while capital return excludes them.
    Their difference recovers cash-equivalent distributions. A large implied share
    factor recovers splits/bonus shares so raw-price portfolio valuation remains valid.
    """
    cash_distribution = 0.0
    actions = []
    for code, old_shares in list(holdings.items()):
        row = price_rows.get(code)
        prior_close = safe_number(last_close.get(code))
        current_close = safe_number(None if row is None else row.get("close"))
        daily_return = safe_number(None if row is None else row.get("daily_return"))
        capital_return = safe_number(None if row is None else row.get("capital_return"))
        if (
            row is None
            or old_shares <= 0
            or not math.isfinite(prior_close)
            or prior_close <= 0
            or not math.isfinite(current_close)
            or current_close <= 0
            or not math.isfinite(daily_return)
            or not math.isfinite(capital_return)
        ):
            continue

        share_factor = (1.0 + capital_return) * prior_close / current_close
        new_shares = int(old_shares)
        if 0.1 <= share_factor <= 10.0 and abs(share_factor - 1.0) >= 0.02:
            inferred_shares = int(round(int(old_shares) * share_factor))
            if inferred_shares > 0:
                new_shares = inferred_shares
                holdings[code] = new_shares

        distribution_per_old_share = prior_close * (daily_return - capital_return)
        distribution = max(0.0, int(old_shares) * distribution_per_old_share)
        cash_distribution += distribution
        if new_shares != int(old_shares) or distribution > 0.01:
            actions.append(
                {
                    "code": code,
                    "old_shares": int(old_shares),
                    "new_shares": int(new_shares),
                    "share_factor": float(share_factor),
                    "cash_distribution": float(distribution),
                }
            )
    return float(cash_distribution), actions


def summarize(equity, trades, initial_cash, final_value, args):
    daily_returns = equity["daily_return"].dropna()
    total_return = final_value / float(initial_cash) - 1.0
    annual_return = (final_value / float(initial_cash)) ** (244 / len(equity)) - 1.0
    annual_vol = daily_returns.std(ddof=1) * math.sqrt(244) if len(daily_returns) > 1 else np.nan
    drawdown = equity["total_value"] / equity["total_value"].cummax() - 1.0
    rebalance_turnover = equity.loc[equity["rebalanced"].astype(bool), "turnover"]
    market_state_counts = (
        {str(key): int(value) for key, value in equity["market_state"].value_counts(dropna=False).to_dict().items()}
        if "market_state" in equity.columns
        else {}
    )
    return {
        "strategy": "factor_rank_buffer",
        "start_date": str(equity["trade_date"].iloc[0]),
        "end_date": str(equity["trade_date"].iloc[-1]),
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "annualized_return": float(annual_return),
        "annualized_volatility": float(annual_vol),
        "sharpe_no_risk_free": float(annual_return / annual_vol) if annual_vol and annual_vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "trading_days": int(len(equity)),
        "trade_count": int(len(trades)),
        "buy_count": int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
        "sell_count": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
        "total_gross_traded": float(trades["gross_amount"].sum()) if not trades.empty else 0.0,
        "total_fees": float(trades["fee"].sum()) if not trades.empty else 0.0,
        "total_corporate_action_cash": float(equity.get("corporate_action_cash", pd.Series(dtype=float)).sum()),
        "corporate_action_days": int((equity.get("corporate_action_count", pd.Series(dtype=float)) > 0).sum()),
        "corporate_action_share_change_count": int(
            equity.get("corporate_action_share_changes", pd.Series(dtype=float)).sum()
        ),
        "average_daily_turnover": float(equity["turnover"].mean()),
        "average_rebalance_turnover": float(rebalance_turnover.mean()) if len(rebalance_turnover) else 0.0,
        "average_target_equity_weight": float(equity["target_equity_weight"].mean()) if "target_equity_weight" in equity.columns else np.nan,
        "average_actual_equity_weight": float(equity["actual_equity_weight"].mean()) if "actual_equity_weight" in equity.columns else np.nan,
        "scheduled_rebalance_count": int(equity["scheduled_rebalance"].sum()) if "scheduled_rebalance" in equity.columns else 0,
        "risk_rebalance_count": int(equity["risk_rebalance"].sum()) if "risk_rebalance" in equity.columns else 0,
        "market_state_counts": market_state_counts,
        "rebalance_schedule": args.rebalance_schedule,
        "score_profile": str(getattr(args, "score_profile", "")),
        "industry_event_scores": str(getattr(args, "industry_event_scores", "")),
        "industry_event_weight": float(getattr(args, "industry_event_weight", 0.0)),
        "event_regime_signals": str(getattr(args, "event_regime_signals", "")),
        "event_regime_enabled": bool(getattr(args, "enable_event_regime", False)),
        "target_count": int(args.target_count),
        "buy_rank": int(args.buy_rank),
        "sell_rank": int(args.sell_rank),
        "cash_weight": float(args.cash_weight),
        "max_stock_weight": float(args.max_stock_weight),
        "max_industry_weight": float(args.max_industry_weight),
        "min_history_days": int(args.min_history_days),
        "min_avg_amount": float(args.min_avg_amount),
        "min_market_cap_quantile": float(args.min_market_cap_quantile),
        "min_trade_value": float(args.min_trade_value),
        "market_regime_enabled": not bool(args.disable_market_regime),
        "target_market_volatility": float(args.target_market_volatility),
        "drawdown_reduce_threshold": float(args.drawdown_reduce_threshold),
        "severe_drawdown_threshold": float(args.severe_drawdown_threshold),
        "cost_model": "date-aware statutory A-share costs; no subjective turnover penalty",
        "corporate_action_model": "RESSET total-return/capital-return inferred cash distributions and share factors",
    }


def write_outputs(equity, trades, positions, summary, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"factor_rank_{summary['start_date'].replace('-', '')}_{summary['end_date'].replace('-', '')}_{stamp}"
    equity_path = output_dir / f"{base}_equity_curve.csv"
    trades_path = output_dir / f"{base}_trades.csv"
    positions_path = output_dir / f"{base}_final_positions.csv"
    summary_path = output_dir / f"{base}_summary.json"
    workbook_path = output_dir / f"{base}.xlsx"
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    payload = dict(summary)
    payload["trading_costs"] = trading_cost_snapshot(summary.get("end_date"))
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_excel_workbook(
        workbook_path,
        [
            ("summary", pd.DataFrame([summary])),
            ("equity_curve", equity),
            ("trades", trades),
            ("final_positions", positions),
        ],
    )
    return {
        "workbook": workbook_path,
        "summary": summary_path,
        "equity_curve": equity_path,
        "trades": trades_path,
        "final_positions": positions_path,
    }


def run_backtest(args):
    conn = sqlite3.connect(args.database)
    try:
        dates = trading_dates(conn)
        date_to_index = {date: idx for idx, date in enumerate(dates)}
        test_dates = [date for date in dates if args.start_date <= date <= args.end_date and date_to_index[date] > 0]
        if not test_dates:
            raise ValueError("No test dates in requested range.")
        history_start = dates[max(0, date_to_index[test_dates[0]] - 320)]
        prices = load_prices(conn, history_start, test_dates[-1])
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        market_state = build_market_state(prices, args)
        prices_by_date = {
            date: group.set_index("code").to_dict("index")
            for date, group in prices.loc[prices["trade_date"].isin(test_dates)].groupby("trade_date")
        }
        financial = load_financial_factors(conn)
        industry_event_scores = load_industry_event_scores(args.industry_event_scores)
        event_regime_signals = load_event_regime_signals(args.event_regime_signals)

        holdings = {}
        cash = float(args.initial_cash)
        last_close = {}
        equity_rows = []
        trade_rows = []
        previous_total = float(args.initial_cash)
        peak_total = float(args.initial_cash)

        for offset, trade_date in enumerate(test_dates):
            decision_date = dates[date_to_index[trade_date] - 1]
            today_prices = prices_by_date.get(trade_date, {})
            corporate_action_cash, corporate_actions = apply_corporate_actions_before_open(
                holdings, today_prices, last_close
            )
            open_total = value_portfolio(holdings, cash, today_prices, "open", last_close)
            stock_value_at_open = max(0.0, open_total - cash)
            current_equity_weight = stock_value_at_open / open_total if open_total > 0 else 0.0
            regime = target_equity_from_market(
                market_state,
                decision_date,
                previous_total,
                peak_total,
                args,
                event_regime_signals,
            )
            scheduled_rebalance = should_rebalance_on_date(
                args.rebalance_schedule,
                offset,
                args.rebalance_every_n_days,
                dates,
                date_to_index,
                decision_date,
            )
            risk_rebalance = current_equity_weight > regime["target_equity_weight"] + float(args.risk_rebalance_band)
            rebalanced = scheduled_rebalance or risk_rebalance
            executed = []
            if rebalanced:
                features = feature_snapshot(prices, financial, decision_date, args, industry_event_scores)
                targets = build_targets(features, holdings, args, regime["target_equity_weight"], open_total)
                cash, executed = execute_trades(
                    trade_date, decision_date, holdings, cash, targets, today_prices, open_total, args
                )
                trade_rows.extend(executed)
            cash += corporate_action_cash
            close_total = value_portfolio(holdings, cash, today_prices, "close", last_close)
            for code, row in today_prices.items():
                close = row.get("close")
                if close is not None and not pd.isna(close) and close > 0:
                    last_close[code] = float(close)
            gross_traded = float(sum(row["gross_amount"] for row in executed))
            fees = float(sum(row["fee"] for row in executed))
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "decision_date": decision_date,
                    "total_value": close_total,
                    "cash": cash,
                    "stock_value": close_total - cash,
                    "daily_return": close_total / previous_total - 1.0 if previous_total > 0 else np.nan,
                    "turnover": gross_traded / open_total if open_total > 0 else 0.0,
                    "gross_traded": gross_traded,
                    "fees": fees,
                    "corporate_action_cash": corporate_action_cash,
                    "corporate_action_count": int(len(corporate_actions)),
                    "corporate_action_share_changes": int(
                        sum(1 for item in corporate_actions if item["new_shares"] != item["old_shares"])
                    ),
                    "holding_count": int(sum(1 for shares in holdings.values() if int(shares) > 0)),
                    "rebalanced": rebalanced,
                    "scheduled_rebalance": scheduled_rebalance,
                    "risk_rebalance": risk_rebalance,
                    "target_equity_weight": regime["target_equity_weight"],
                    "actual_equity_weight": (close_total - cash) / close_total if close_total > 0 else 0.0,
                    "market_state": regime["market_state"],
                    "market_index": regime["market_index"],
                    "market_breadth": regime["market_breadth"],
                    "market_return_20": regime["market_return_20"],
                    "market_volatility_20": regime["market_volatility_20"],
                    "portfolio_drawdown_signal": regime["portfolio_drawdown"],
                    "event_equity_multiplier": regime["event_equity_multiplier"],
                    "event_risk_score": regime["event_risk_score"],
                    "event_regime_confidence": regime["event_regime_confidence"],
                    "event_regime_reason": regime["event_regime_reason"],
                }
            )
            previous_total = close_total
            peak_total = max(peak_total, close_total)
            if (offset + 1) % 20 == 0 or offset == len(test_dates) - 1:
                print(f"Factor-rank progress: {offset + 1}/{len(test_dates)} {trade_date} value={close_total:.2f}", flush=True)

        equity = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(trade_rows)
        if trades.empty:
            trades = pd.DataFrame(columns=["trade_date", "decision_date", "code", "name", "side", "shares", "price", "gross_amount", "fee", "cash_after"])
        final_prices = prices_by_date.get(test_dates[-1], {})
        final_total = value_portfolio(holdings, cash, final_prices, "close", last_close)
        positions = pd.DataFrame(
            [
                {
                    "code": code,
                    "shares": int(shares),
                    "close": final_prices.get(code, {}).get("close", last_close.get(code, np.nan)),
                    "market_value": int(shares) * float(final_prices.get(code, {}).get("close", last_close.get(code, np.nan))),
                }
                for code, shares in holdings.items()
                if int(shares) > 0
            ]
        )
        summary = summarize(equity, trades, args.initial_cash, final_total, args)
        paths = write_outputs(equity, trades, positions, summary, args)
        print(f"Factor-rank workbook: {paths['workbook']}")
        print(f"Summary JSON: {paths['summary']}")
        print(f"Final value: {summary['final_value']:.2f}")
        print(f"Total return: {summary['total_return']:.2%}")
        print(f"Max drawdown: {summary['max_drawdown']:.2%}")
        return summary, paths
    finally:
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Rank-based A-share multifactor backtest.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--initial-cash", type=float, default=1000000.0)
    parser.add_argument("--rebalance-every-n-days", type=int, default=1)
    parser.add_argument("--rebalance-schedule", choices=["every_n_days", "week_end", "month_end"], default="week_end")
    parser.add_argument(
        "--score-profile",
        choices=[
            "diagnostic_defensive",
            "low_risk_only",
            "low_beta_defensive",
            "low_beta_industry",
            "contrarian_low_risk",
            "balanced",
        ],
        default="low_beta_industry",
    )
    parser.add_argument("--industry-event-scores", type=Path, default=DEFAULT_INDUSTRY_EVENT_SCORES)
    parser.add_argument("--industry-event-weight", type=float, default=0.04)
    parser.add_argument("--event-regime-signals", type=Path, default=DEFAULT_EVENT_REGIME_SIGNALS)
    parser.add_argument("--enable-event-regime", action="store_true")
    parser.add_argument("--target-count", type=int, default=30)
    parser.add_argument("--buy-rank", type=int, default=60)
    parser.add_argument("--sell-rank", type=int, default=180)
    parser.add_argument("--cash-weight", type=float, default=0.05)
    parser.add_argument("--max-stock-weight", type=float, default=0.04)
    parser.add_argument("--max-industry-weight", type=float, default=0.25)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--feature-history-days", type=int, default=320)
    parser.add_argument("--min-avg-amount", type=float, default=50000000.0)
    parser.add_argument("--min-market-cap-quantile", type=float, default=0.30)
    parser.add_argument("--market-cap-proxy-window", type=int, default=20)
    parser.add_argument("--min-trade-value", type=float, default=20000.0)
    parser.add_argument("--disable-industry-neutral-factors", action="store_true")
    parser.add_argument("--disable-market-regime", action="store_true")
    parser.add_argument("--market-short-window", type=int, default=60)
    parser.add_argument("--market-long-window", type=int, default=200)
    parser.add_argument("--market-breadth-window", type=int, default=120)
    parser.add_argument("--bull-breadth-min", type=float, default=0.52)
    parser.add_argument("--neutral-breadth-min", type=float, default=0.45)
    parser.add_argument("--bear-breadth-max", type=float, default=0.38)
    parser.add_argument("--neutral-equity-weight", type=float, default=0.60)
    parser.add_argument("--defensive-equity-weight", type=float, default=0.35)
    parser.add_argument("--bear-equity-weight", type=float, default=0.20)
    parser.add_argument("--crash-equity-weight", type=float, default=0.05)
    parser.add_argument("--crash-return-20", type=float, default=-0.08)
    parser.add_argument("--target-market-volatility", type=float, default=0.18)
    parser.add_argument("--drawdown-reduce-threshold", type=float, default=-0.12)
    parser.add_argument("--drawdown-reduce-multiplier", type=float, default=0.50)
    parser.add_argument("--severe-drawdown-threshold", type=float, default=-0.22)
    parser.add_argument("--severe-drawdown-equity-weight", type=float, default=0.10)
    parser.add_argument("--risk-rebalance-band", type=float, default=0.15)
    parser.add_argument("--disable-limit-trade-filter", action="store_true")
    parser.add_argument("--limit-trade-buffer", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    run_backtest(parse_args())
