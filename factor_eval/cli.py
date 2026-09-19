"""命令行入口：把五道检查串成一条可复现的流水线。

用法
----
    # 用合成演示面板跑全部检查（不需要任何数据库）
    python -m factor_eval.cli demo

    # 用自己的长表跑
    python -m factor_eval.cli all --panel my_panel.parquet \
        --factors f1 f2 f3 --controls size,momentum,liquidity,value \
        --reversal past_return --era-col era --train train --test test

本模块**不包含任何绝对路径**，也不假设任何数据库 schema。
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import adapter, gates, protocol, stats

DEFAULT_CONTROLS = ["size", "momentum", "liquidity", "value", "earnings_yield",
                    "growth", "beta", "residual_vol"]
TRAIN, TEST = "train", "test"


def _fmt(value, width: int = 10, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return " " * (width - 3) + "nan"
    if isinstance(value, float):
        return ("%+*.*f" % (width, digits, value))
    return ("%*s" % (width, value))


def run_gate_report(panel: pd.DataFrame, factors, controls, era_col) -> None:
    print("=" * 100)
    print("门 1：可分组性（必须在算分位之前跑）")
    print("=" * 100)
    print("%-22s %6s %10s %10s %10s %s" % ("factor", "weeks", "uniq", "max_tie",
                                           "ok_frac", "verdict"))
    for name in factors:
        result = gates.groupability(panel, name)
        print("%-22s %6d %10.1f %10.4f %10.3f %s"
              % (name, result.get("weeks", 0), result.get("mean_unique_values", 0.0),
                 result.get("mean_largest_tie", 1.0),
                 result.get("groupable_week_fraction", 0.0),
                 "可用" if result["applicable"] else "不可用：" + result["reason"]))

    print()
    print("=" * 100)
    print("门 2：控制可用性（列存在 ≠ 有效控制）")
    print("=" * 100)
    audit = gates.control_audit(panel, controls, era_col=era_col)
    table = audit["table"]
    if "is_constant" in table.columns:
        print(table.to_string(index=False))
    if len(audit["constant_controls"]):
        print("\n⚠ 恒为常数的控制列（在该段等于没有控制）：")
        print(audit["constant_controls"][["era", "control", "n_unique"]]
              .to_string(index=False))
    if len(audit["collinear_pairs"]):
        print("\n⚠ 高度共线的控制对：")
        print(audit["collinear_pairs"].to_string(index=False))
    print("\n判定：%s" % audit["verdict"])


def run_ladder(panel: pd.DataFrame, factor: str, controls) -> None:
    layers = {
        "L0_style_only": controls[:4],
        "L1_style_wide": controls,
        "L2_style_plus_halves": controls,
    }
    print("=" * 100)
    print("协议 1：消融阶梯（L0 最弱 → L2 最全）")
    print("=" * 100)
    result = protocol.ablation_ladder(panel, factor, layers)
    print(result.to_string(index=False))


def run_reversal(panel: pd.DataFrame, factors, reversal: str, controls, era_col) -> None:
    print("=" * 100)
    print("协议 2：反转向复检（符号冻结在 %s，%s 段为纯样本外）" % (TRAIN, TEST))
    print("=" * 100)
    table = protocol.reversal_retest(panel, factors, reversal, controls,
                                     era_col=era_col, train=TRAIN, test=TEST)
    columns = ["factor", "object", "orientation", "t_train", "t_test",
               "rho_with_reversal", "style_t_test", "verdict"]
    print(table[columns].to_string(index=False))
    if not table["survived"].any():
        print("\n→ 没有任何候选通过。注意：这是**正常结果**，"
              "多数因子在加上风格控制后都会归零。")
    return table


def run_decoupling(panel: pd.DataFrame, factors) -> None:
    print("=" * 100)
    print("协议 3：分位画像 —— IC 与组合层是否脱钩")
    print("=" * 100)
    print("%-22s %10s %12s %12s %12s %10s" % ("factor", "pool", "top", "bottom",
                                              "top-pool", "mono"))
    for name in factors:
        result = protocol.quantile_profile(panel, name)
        if not result.get("weeks"):
            print("%-22s （无有效周）" % name)
            continue
        print("%-22s %+10.5f %+12.5f %+12.5f %+12.5f %10.3f"
              % (name, result["pool_mean"], result["top_mean"],
                 result["bottom_mean"], result["top_minus_pool"],
                 result["monotonicity"]))
        print("      └ %s" % result["note"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="factor_eval",
        description="因子评估纪律工具包（不含任何真实因子）")
    parser.add_argument("command", choices=["demo", "gates", "reversal", "decoupling",
                                            "all"])
    parser.add_argument("--panel", default=None,
                        help="长表路径（.parquet/.feather/.csv）；省略则用合成演示面板")
    parser.add_argument("--factors", nargs="*", default=None)
    parser.add_argument("--controls", default=None,
                        help="逗号分隔；默认 %s" % ",".join(DEFAULT_CONTROLS))
    parser.add_argument("--reversal", default="past_return",
                        help="作为反转基准的列（通常是过去 N 日收益）")
    parser.add_argument("--era-col", default="era")
    parser.add_argument("--out", default=None, help="把结果写进这个 csv 前缀")
    args = parser.parse_args(argv)

    if args.panel:
        panel = adapter.load_panel(args.panel)
    else:
        panel, truth = adapter.make_synthetic_panel()
        print("使用合成演示面板：%d 行 / %d 期 / %d 只"
              % (len(panel), panel["date"].nunique(), panel["code"].nunique()))
        print(truth.to_string(index=False))
        print()

    controls = ([c.strip() for c in args.controls.split(",")] if args.controls
                else DEFAULT_CONTROLS)
    controls = [c for c in controls if c in panel.columns]
    reserved = {"date", "code", "era", "forward_return", args.reversal, *controls}
    factors = args.factors or [c for c in panel.columns if c not in reserved]
    era_col = args.era_col if args.era_col in panel.columns else None

    if args.command in ("demo", "gates", "all"):
        run_gate_report(panel, factors, controls, era_col)
    if args.command in ("demo", "reversal", "all"):
        if era_col is None:
            print("（面板无 era 列，跳过跨时代协议）")
        else:
            run_reversal(panel, factors, args.reversal, controls, era_col)
    if args.command in ("demo", "decoupling", "all"):
        run_decoupling(panel, factors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
