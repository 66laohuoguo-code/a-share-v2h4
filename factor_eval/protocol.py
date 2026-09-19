"""三道核心协议：消融阶梯 / 反转向复检 / 分位与组合层脱钩诊断。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import stats


def ablation_ladder(panel: pd.DataFrame, factor: str, layers: dict[str, list[str]],
                    date_col: str = "date", ret_col: str = "forward_return",
                    min_names: int = 80, lags: int = 4) -> pd.DataFrame:
    """分层控制消融：L0（最弱）→ L3（最全），看每一层把斜率打到多少。

    `layers` 是有序字典，例如
        {"L0_base_styles": [...8 个风格...],
         "L1_plus_size":   [...风格 + log_size...],
         "L2_plus_baseline": [... + 基线因子...],
         "L3_full":        [...全控制集...]}
    只看 L0 会把结论夸大成"跨时代稳健"；L3 才是"相对现有体系有没有增量"的答案。
    """
    rows = []
    for name, controls in layers.items():
        result = stats.partial_slope_t(panel, factor, controls, date_col=date_col,
                                       ret_col=ret_col, min_names=min_names, lags=lags)
        rows.append({"layer": name, "n_controls": len(controls),
                     "mean_slope": result["mean_slope"], "t": result["t"],
                     "weeks": result["weeks"]})
    return pd.DataFrame(rows)


def subperiod_decay(panel: pd.DataFrame, factor: str, layers: dict[str, list[str]],
                    periods: dict[str, tuple[str, str]], date_col: str = "date",
                    ret_col: str = "forward_return", min_names: int = 80,
                    lags: int = 4) -> pd.DataFrame:
    """把同一份消融**按子区间拆开**跑。

    全期聚合常被早期段抬高（辛普森式）。若 L2 的 t 逐段单调衰减，
    那"全期显著"是一个平均值假象，必须如实报告。
    """
    rows = []
    for label, (start, end) in periods.items():
        subset = panel.loc[(panel[date_col] >= pd.Timestamp(start))
                           & (panel[date_col] <= pd.Timestamp(end))]
        if subset.empty:
            continue
        for name, controls in layers.items():
            result = stats.partial_slope_t(subset, factor, controls,
                                           date_col=date_col, ret_col=ret_col,
                                           min_names=min_names, lags=lags)
            rows.append({"period": label, "layer": name, "weeks": result["weeks"],
                         "mean_slope": result["mean_slope"], "t": result["t"]})
    return pd.DataFrame(rows)


def reversal_retest(panel: pd.DataFrame, candidates: list[str],
                    reversal_column: str, styles: list[str],
                    era_col: str = "era", train: str = "train", test: str = "test",
                    extra_controls: list[str] | None = None,
                    alpha: float = 0.05, date_col: str = "date",
                    ret_col: str = "forward_return",
                    min_names: int = 80, lags: int = 4) -> pd.DataFrame:
    """反转向复检协议。

    解决的问题
    ----------
    "我很多因子的符号和预测相反" —— 但**翻符号之后还剩多少信息，是不是只是把
    因子变成了短期反转的马甲？**

    协议（每一步都不可省）
    ----------------------
    1. 对象有两套：`raw`（因子原值）与 `res`（对 `reversal_column` 做逐期截面
       正交后的残差）。**两套都要做**，因为有些因子的原始与残差符号相反
       （抑制效应），只做一套会把真正有信息的那一侧当成"错的"丢掉。
    2. **符号只在训练段选定**，冻结后搬到测试段 —— 测试段是纯样本外。
    3. 门槛按**试验数**做 Bonferroni 校正：`n_trials ≈ 候选数 × 2 符号 × 2 对象`。
       因为符号是双侧选的，必须用**双边**门槛。
    4. 幸存者必须再通过：`styles + extra_controls` 的偏斜率检验。
       **这一条最关键** —— 只正交掉反转是不够的，必须把先前剔除的风格加回来。

    输出每行含：两套对象的训练/测试 t、与反转的秩相关、加回控制后的 t、判定。

    两条用血换来的教训（见 docs/FACTOR_EVAL_PROTOCOL.md）
    ---------------------------------------------------
    * **"与反转的相关系数低"完全不足以证明它不是反转。**
      一个单调变换过的反转信号可以把线性相关压到 0.2，正交后符号照样翻转。
      必须做多元投影，不能只看相关系数。
    * **正交掉反转还不够，必须把被剔除的风格加回来。**
      一个波动率类信号正交掉反转后可能显著，加回波动率风格立刻归零 ——
      它不是新因子，它就是你先前剔掉的那个风格。
    """
    extra_controls = extra_controls or []
    n_trials = max(1, len(candidates) * 2 * 2)
    bar = stats.bonferroni_t_bar(n_trials, alpha)
    train_panel = panel.loc[panel[era_col] == train].copy()
    test_panel = panel.loc[panel[era_col] == test].copy()

    rows = []
    for name in candidates:
        for obj in ("raw", "res"):
            entry = {"factor": name, "object": obj, "n_trials": n_trials,
                     "bar": bar}
            orient = 1.0
            for label, subset in (("train", train_panel), ("test", test_panel)):
                work = subset.copy()
                if obj == "res":
                    work["_x"] = stats.orthogonalise(work, name, [reversal_column],
                                                     date_col=date_col,
                                                     min_names=min_names)
                else:
                    work["_x"] = pd.to_numeric(work[name], errors="coerce")
                oriented = stats.rank_ic_series(work, "_x", date_col=date_col,
                                                ret_col=ret_col,
                                                min_names=min_names) * orient
                entry["t_%s" % label] = stats.newey_west_t(oriented, lags)
                entry["ic_%s" % label] = float(oriented.mean()) if len(oriented) else math.nan
                if label == "train":
                    # 符号只在训练段决定，之后冻结
                    orient = 1.0 if (len(oriented) and oriented.mean() >= 0) else -1.0
                    entry["orientation"] = orient
            if obj == "raw":
                if name == reversal_column:
                    entry["rho_with_reversal"] = 1.0     # 它本身就是基准
                else:
                    pair = train_panel[[name, reversal_column]].dropna()
                    entry["rho_with_reversal"] = (
                        stats.spearman(pair[name], pair[reversal_column])
                        if len(pair) > 100 else math.nan)
            # 加回风格（+可选的现有体系分数）
            controls = [*styles, *extra_controls]
            for label, subset in (("train", train_panel), ("test", test_panel)):
                work = subset.copy()
                if obj == "res":
                    work["_x"] = stats.orthogonalise(work, name, [reversal_column],
                                                     date_col=date_col,
                                                     min_names=min_names)
                else:
                    work["_x"] = pd.to_numeric(work[name], errors="coerce")
                work["_s"] = entry["orientation"] * work["_x"]
                result = stats.partial_slope_t(work, "_s", controls, date_col=date_col,
                                               ret_col=ret_col, min_names=min_names,
                                               lags=lags)
                entry["style_t_%s" % label] = result["t"]
            # 判定
            passed_screen = (abs(entry["t_train"]) >= bar
                             and entry["t_test"] >= bar / 2.0)
            survived = passed_screen and entry["style_t_test"] >= 3.0
            entry["pass_screen"] = passed_screen
            entry["survived"] = survived
            entry["verdict"] = (
                "★ 幸存" if survived else
                "① 未过样本外门槛" if not passed_screen else
                "② 加回风格后归零/反号")
            rows.append(entry)
    table = pd.DataFrame(rows)
    return table.sort_values(["survived", "t_test"], ascending=[False, False]
                             ).reset_index(drop=True)


def quantile_profile(panel: pd.DataFrame, factor: str, quantiles: int = 5,
                     date_col: str = "date", ret_col: str = "forward_return",
                     min_names: int = 100) -> dict:
    """分位画像：每一组的池化均收益，以及**和全池等权基准的对比**。

    关键一问：高分位组比"整个池子的等权"高多少？
    如果只高一点点（例如 20 日 +0.13%），那 alpha 在**规避低分位端**，
    只做多的组合赚不到信号的钱 —— 这正是 IC 不衰减、组合却亏钱的最常见原因。
    """
    bucket_means: dict[int, list[float]] = {}
    pool_means: list[float] = []
    for _, group in panel.groupby(date_col, sort=True):
        sub = group[[factor, ret_col]].dropna()
        if len(sub) < min_names:
            continue
        pool_means.append(float(sub[ret_col].mean()))
        try:
            labels = pd.qcut(sub[factor], quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        for label, part in sub.groupby(labels):
            bucket_means.setdefault(int(label), []).append(float(part[ret_col].mean()))
    if not pool_means:
        return {"factor": factor, "weeks": 0}
    profile = pd.DataFrame({
        "bucket": sorted(bucket_means),
        "mean_return": [float(np.mean(bucket_means[k])) for k in sorted(bucket_means)],
        "weeks": [len(bucket_means[k]) for k in sorted(bucket_means)],
    })
    pool = float(np.mean(pool_means))
    top = float(profile["mean_return"].iloc[-1])
    bottom = float(profile["mean_return"].iloc[0])
    # 分位次序的 Spearman：K 个点时用 1 - 6Σd²/(K(K²-1)) 的秩相关
    order_rho = stats.spearman(profile["bucket"], profile["mean_return"])
    return {"factor": factor, "weeks": len(pool_means), "pool_mean": pool,
            "top_mean": top, "bottom_mean": bottom,
            "top_minus_pool": top - pool, "pool_minus_bottom": pool - bottom,
            "top_minus_bottom": top - bottom, "monotonicity": order_rho,
            "profile": profile,
            "note": ("alpha 主要在规避端，只做多的组合赚不到"
                     if (top - pool) < 0.25 * (pool - bottom) else
                     "顶部组本身也贡献了可观收益")}
