"""横截面统计原语：Newey-West t、周度 rank IC、偏斜率、截面正交化。

全部只用 numpy / pandas，不依赖 statsmodels 或 scipy。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_LAGS = 4
DEFAULT_MIN_NAMES = 80


def newey_west_t(values, lags: int = DEFAULT_LAGS) -> float:
    """重叠持有期下的 HAC(Newey-West) t 值。

    重叠的 forward return 会让相邻周的 IC 自相关，普通 t 会被高估；
    这里用 Bartlett 核做 HAC 修正。

    注意：这是个**双边**统计量，|t| 才有意义 —— 若符号是看过数据之后才选的，
    必须按双边门槛判（见 protocol.reversal_retest 的试验数校正）。
    """
    series = pd.Series(values, dtype="float64").dropna()
    n = len(series)
    if n < 4:
        return math.nan
    centred = series.to_numpy() - series.mean()
    gamma0 = float(centred @ centred) / n
    variance = gamma0
    for lag in range(1, min(int(lags), n - 1) + 1):
        weight = 1.0 - lag / (int(lags) + 1.0)
        covariance = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * weight * covariance
    if not np.isfinite(variance) or variance <= 0:
        return math.nan
    return float(series.mean() / math.sqrt(variance / n))


def _as_1d(obj):
    """把可能是 DataFrame（列名重复时会这样）的输入压成一维 Series。"""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    return pd.Series(obj)


def spearman(left, right) -> float:
    """秩相关；并列取平均秩，样本不足或方差为零时返回 nan。"""
    frame = pd.DataFrame({"l": _as_1d(left).to_numpy(),
                          "r": _as_1d(right).to_numpy()}).dropna()
    if len(frame) < 3 or frame["l"].nunique() < 2 or frame["r"].nunique() < 2:
        return math.nan
    return float(frame["l"].rank().corr(frame["r"].rank()))


def rank_ic_series(panel: pd.DataFrame, factor: str, date_col: str = "date",
                   ret_col: str = "forward_return",
                   min_names: int = DEFAULT_MIN_NAMES) -> pd.Series:
    """逐期截面 rank IC 序列（只在当期有效名字数 ≥ min_names 时计算）。"""
    out: dict = {}
    for date, group in panel.groupby(date_col, sort=True):
        sub = group[[factor, ret_col]].dropna()
        if len(sub) < min_names:
            continue
        out[date] = spearman(sub[factor], sub[ret_col])
    return pd.Series(out, dtype="float64")


def ic_summary(panel: pd.DataFrame, factor: str, lags: int = DEFAULT_LAGS,
               **kwargs) -> dict:
    series = rank_ic_series(panel, factor, **kwargs)
    if series.empty:
        return {"factor": factor, "weeks": 0, "mean_ic": math.nan, "t": math.nan}
    return {"factor": factor, "weeks": int(series.notna().sum()),
            "coverage": float(panel[factor].notna().mean()),
            "mean_ic": float(series.mean()),
            "median_ic": float(series.median()),
            "ic_sd": float(series.std()),
            "t": newey_west_t(series, lags),
            "positive_rate": float((series > 0).mean())}


def orthogonalise(panel: pd.DataFrame, factor: str, on: list[str],
                  date_col: str = "date", min_names: int = DEFAULT_MIN_NAMES,
                  new_name: str | None = None) -> pd.Series:
    """逐期在截面内把 factor 对 on 做线性投影，返回残差。

    残差 **保留原始索引**，可以直接 `panel[new] = orthogonalise(...)`。
    `on` 为常量列时该期跳过（否则 lstsq 会给出无意义的残差）。
    """
    result = pd.Series(np.nan, index=panel.index, dtype="float64")
    for _, group in panel.groupby(date_col, sort=True):
        columns = [factor, *on]
        sub = group[columns].dropna()
        if len(sub) < min_names:
            continue
        matrix = sub[on].to_numpy(float)
        if any(matrix[:, j].std() == 0 for j in range(matrix.shape[1])):
            continue
        design = np.column_stack([np.ones(len(sub)), matrix])
        target = sub[factor].to_numpy(float)
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        result.loc[sub.index] = target - design @ beta
    return result


def partial_slope_t(panel: pd.DataFrame, factor: str, controls: list[str],
                    date_col: str = "date", ret_col: str = "forward_return",
                    min_names: int = DEFAULT_MIN_NAMES,
                    lags: int = DEFAULT_LAGS) -> dict:
    """逐期把收益对 [factor, *controls] 回归，对斜率序列做 HAC t。

    这是**多变量**统计量：因子单独看可能很弱，但控制住风格后斜率可能显著
    （抑制效应 suppression），反之亦然。两个统计量必须一起看。
    """
    slopes: list[float] = []
    for _, group in panel.groupby(date_col, sort=True):
        sub = group[[factor, ret_col, *controls]].dropna()
        if len(sub) < min_names:
            continue
        design = np.column_stack([np.ones(len(sub)),
                                  sub[[factor, *controls]].to_numpy(float)])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        y = sub[ret_col].to_numpy(float)
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        slopes.append(float(beta[1]))
    series = pd.Series(slopes, dtype="float64")
    return {"factor": factor, "weeks": len(series),
            "mean_slope": float(series.mean()) if len(series) else math.nan,
            "t": newey_west_t(series, lags),
            "positive_rate": float((series > 0).mean()) if len(series) else math.nan}


def bonferroni_t_bar(n_trials: int, alpha: float = 0.05) -> float:
    """n 次检验、family-wise alpha 下的**双侧** |t| 门槛。

    用来防止"扫了几十列、挑出最好的那个"这种多重检验；因为翻符号本质上
    是把试验数翻倍，所以门槛必须按双边算。

    参考值：n=56 → 3.32；n=112 → 3.51；n=196 → 3.66。
    """
    if n_trials < 1:
        return math.nan
    tail = alpha / (2.0 * n_trials)              # 单尾概率
    if not 0.0 < tail < 1.0:
        return math.nan
    t = math.sqrt(-2.0 * math.log(tail))         # 尾部渐近初值
    for _ in range(80):                          # 牛顿迭代求 1 - Phi(t) = tail
        residual = (1.0 - _normal_cdf(t)) - tail
        t += residual / max(_normal_pdf(t), 1e-300)
    return float(t)


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
