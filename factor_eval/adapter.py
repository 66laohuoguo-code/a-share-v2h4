"""数据适配层：只认一个长表，不认任何数据库 schema。

要求的列
--------
* `date`            —— 决策日（每期的截面）
* `code`            —— 标的代码
* `forward_return`  —— 决策日之后的前瞻收益（可以重叠）

可选列
------
* `era`             —— 时代标签（如 `train` / `test`）。做跨时代协议时需要。
* 任意因子列、任意控制列 —— 由调用方通过 `--factors` / `--controls` 指定。

支持 parquet / csv / feather。**本模块不包含任何绝对路径**，路径一律由命令行传入。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

REQUIRED = ("date", "code", "forward_return")


def load_panel(path: str, date_col: str = "date", code_col: str = "code",
               ret_col: str = "forward_return") -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif suffix in (".feather", ".ft"):
        frame = pd.read_feather(path)
    elif suffix in (".csv", ".txt"):
        frame = pd.read_csv(path)
    else:
        raise ValueError("不支持的扩展名 %r（支持 .parquet/.feather/.csv）" % suffix)

    missing = [c for c in (date_col, code_col, ret_col) if c not in frame.columns]
    if missing:
        raise ValueError("输入表缺少必需列：%s" % missing)

    frame = frame.rename(columns={date_col: "date", code_col: "code",
                                  ret_col: "forward_return"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    for column in frame.columns:
        if column not in ("date", "code", "era"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def make_synthetic_panel(n_dates: int = 240, n_codes: int = 160,
                         seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合成一份演示面板，用来展示这套检查**如何杀掉看起来不错的因子**。

    刻意埋了四种典型陷阱：

    * `honest_factor`      —— 有真实（较弱）预测力，应当通过
    * `reversal_proxy`     —— 直接就是 −过去收益，是反转的马甲
    * `disguised_reversal` —— 与过去收益的**线性相关很低**（0.2 上下），
                              但正交掉过去收益后符号整个翻转
    * `value_style`        —— 本质是某个控制变量的暴露，加上那个控制后归零

    另外把 `beta` 在 era=`train` 段设成**常数**，用来展示控制可用性审计。
    """
    rng = np.random.default_rng(seed)
    factor_names = ["honest_factor", "reversal_proxy", "disguised_reversal",
                    "value_style"]
    controls = ["size", "momentum", "liquidity", "value", "earnings_yield",
                "growth", "beta", "residual_vol"]
    rows = []
    for index in range(n_dates):
        date = pd.Timestamp("2015-01-02") + pd.Timedelta(days=7 * index)
        era = "train" if index < n_dates // 2 else "test"
        codes = ["S%04d" % c for c in range(n_codes)]
        size = rng.normal(size=n_codes)
        liquidity = rng.normal(size=n_codes)
        value = rng.normal(size=n_codes)
        growth = rng.normal(size=n_codes)
        momentum = rng.normal(size=n_codes)
        earnings_yield = rng.normal(size=n_codes)
        residual_vol = np.abs(rng.normal(size=n_codes)) + 0.5
        # 训练段的 beta 恒为常数：这正是"控制列存在 ≠ 有效控制"的陷阱
        beta = np.zeros(n_codes) if era == "train" else rng.normal(size=n_codes)

        past_return = (0.3 * momentum - 0.2 * size + rng.normal(scale=1.4,
                                                               size=n_codes))
        honest = rng.normal(size=n_codes)
        # 真实收益只由 honest 与风格暴露驱动；反转是负的过去收益
        forward = (0.030 * honest
                   - 0.020 * value
                   + 0.015 * size
                   - 0.025 * past_return
                   + rng.normal(scale=0.09, size=n_codes))

        # 只在**极端**过去收益上携带信号：因此与过去收益的秩相关很低，
        # 但它本质仍然是过去收益的函数 —— 正交化一上就会现形。
        tail_only = -np.sign(past_return) * (np.abs(past_return) > 1.0).astype(float)
        rows.append(pd.DataFrame({
            "date": date, "code": codes, "era": era,
            "forward_return": forward,
            "honest_factor": honest,
            "reversal_proxy": -past_return,
            "disguised_reversal": tail_only + rng.normal(scale=0.9, size=n_codes),
            "value_style": -value + rng.normal(scale=0.25, size=n_codes),
            "past_return": past_return,
            "size": size, "momentum": momentum, "liquidity": liquidity,
            "value": value, "earnings_yield": earnings_yield, "growth": growth,
            "beta": beta, "residual_vol": residual_vol,
        }))
    panel = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame({
        "factor": factor_names + ["past_return"],
        "expectation": ["应当通过（弱但真实）", "就是反转马甲，应被判死",
                        "低相关但实质是反转，应被判死",
                        "只是 value 暴露，加回控制后应归零", "（对照列）"],
    })
    return panel, truth


def split_eras(panel: pd.DataFrame, era_col: str = "era",
               train: str = "train", test: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    if era_col not in panel.columns:
        raise ValueError("面板里没有 %r 列；跨时代协议需要它" % era_col)
    return (panel.loc[panel[era_col] == train].copy(),
            panel.loc[panel[era_col] == test].copy())
