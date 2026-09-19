"""两道"先验门"：可分组性、控制可用性。

这两道门都必须在**看业绩之前**跑。它们挡掉的不是坏因子，而是**坏结论**。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import stats

# 可分组门的预注册门槛（故意设成让"离散计数型"信号不通过）
GATES = {"min_unique_values": 50, "max_largest_tie": 0.20,
         "min_groupable_week_fraction": 0.80}


def groupability(panel: pd.DataFrame, factor: str, date_col: str = "date",
                 gates: dict | None = None, min_names: int = 80) -> dict:
    """分位验收的**前置门**。

    如果不先跑这一道，一个 70% 取值并列的信号会被报成
    "五分组没有组间价差" —— 那是**测量失败**，不是因子失败。
    正确结论应当是 `inapplicable: signal too tied for quantiles`。
    """
    gates = {**GATES, **(gates or {})}
    unique_counts, tie_fractions, usable = [], [], 0
    for _, group in panel.groupby(date_col, sort=True):
        values = group[factor].dropna()
        if len(values) < min_names:
            continue
        unique = int(values.nunique())
        largest = float(values.value_counts(normalize=True).iloc[0])
        unique_counts.append(unique)
        tie_fractions.append(largest)
        if (unique >= gates["min_unique_values"]
                and largest <= gates["max_largest_tie"]):
            usable += 1
    weeks = len(unique_counts)
    if weeks == 0:
        return {"factor": factor, "weeks": 0, "applicable": False,
                "reason": "no_usable_week"}
    fraction = usable / weeks
    applicable = (fraction >= gates["min_groupable_week_fraction"]
                  and float(np.mean(unique_counts)) >= gates["min_unique_values"]
                  and float(np.mean(tie_fractions)) <= gates["max_largest_tie"])
    reason = "" if applicable else "inapplicable_signal_too_tied_for_quantiles"
    return {"factor": factor, "weeks": weeks,
            "mean_unique_values": float(np.mean(unique_counts)),
            "min_unique_values": int(np.min(unique_counts)),
            "mean_largest_tie": float(np.mean(tie_fractions)),
            "max_largest_tie": float(np.max(tie_fractions)),
            "groupable_week_fraction": fraction,
            "gates": gates, "applicable": applicable, "reason": reason}


def control_audit(panel: pd.DataFrame, controls: list[str],
                  era_col: str | None = None, date_col: str = "date",
                  collinearity_threshold: float = 0.98) -> dict:
    """控制可用性审计。

    两件会在报告里**静默伪造结论**的事：

    1. **控制列在某段数据里恒为常数** —— 回归里它无法发挥任何控制作用，
       于是"已控制该风格"这句话是假的。列存在 ≠ 有效控制。
    2. **控制列两两高度共线** —— 设计矩阵接近奇异，单个系数不可解释。

    返回逐列逐时代的唯一值数 / 标准差，以及共线对。
    """
    rows = []
    eras = (sorted(panel[era_col].dropna().unique()) if era_col
            and era_col in panel.columns else [None])
    for era in eras:
        subset = panel if era is None else panel.loc[panel[era_col] == era]
        for column in controls:
            if column not in subset.columns:
                rows.append({"era": era, "control": column, "present": False})
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            rows.append({
                "era": era, "control": column, "present": True,
                "rows": int(len(values)),
                "n_unique": int(values.nunique()),
                "sd": float(values.std()) if len(values) else math.nan,
                "is_constant": bool(len(values) > 0 and values.nunique() <= 1),
            })
    table = pd.DataFrame(rows)
    constant = table.loc[table.get("is_constant", pd.Series(dtype=bool)) == True]  # noqa: E712

    pairs = []
    for era in eras:
        subset = panel if era is None else panel.loc[panel[era_col] == era]
        available = [c for c in controls if c in subset.columns]
        for i, left in enumerate(available):
            for right in available[i + 1:]:
                frame = subset[[left, right]].dropna()
                if len(frame) < 50:
                    continue
                if frame[left].std() == 0 or frame[right].std() == 0:
                    continue
                rho = stats.spearman(frame[left], frame[right])
                if np.isfinite(rho) and abs(rho) >= collinearity_threshold:
                    pairs.append({"era": era, "left": left, "right": right,
                                  "rho": rho})
    return {"table": table, "constant_controls": constant,
            "collinear_pairs": pd.DataFrame(pairs),
            "verdict": ("有控制列恒为常数，该段等于**没有**控制这些风格"
                        if len(constant) else "控制列均有效变化")}


def report_gates(panel: pd.DataFrame, factors: list[str], controls: list[str],
                 era_col: str | None = None) -> dict:
    """把两道门打包成一次调用，便于写进 CI 或报告。"""
    return {
        "groupability": pd.DataFrame([groupability(panel, f) for f in factors]),
        "controls": control_audit(panel, controls, era_col=era_col),
    }
