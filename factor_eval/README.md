# factor_eval — 因子评估纪律工具包

> 这个包**不包含任何真实因子**。它只做一件事：在你看一个因子的业绩之前，
> 先用五道可自动化的检查，把明显不该相信的结论挡掉。

依赖：Python ≥ 3.10、`numpy`、`pandas`。**不需要** scipy / sklearn / statsmodels。
不读任何数据库，不假设任何 schema，不含任何绝对路径。

## 快速开始

```bash
# 1) 用内置的合成演示面板跑全部检查（不需要任何数据）
python -m factor_eval.cli demo

# 2) 跑自己的面板（长表：date, code, forward_return + 你的因子/控制列）
python -m factor_eval.cli all \
    --panel my_panel.parquet \
    --factors factor_a factor_b \
    --controls size,momentum,liquidity,value,earnings_yield,growth,beta,residual_vol \
    --reversal past_return_20 \
    --era-col era
```

Windows：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\run_factor_eval.ps1 -Task demo
```

## 五道检查，以及各自防的是什么错

| # | 检查 | 它防的错误 |
|---|---|---|
| 1 | `gates.groupability` | 一个 70% 取值并列的信号被报成"五分组没有组间价差"——那是**测量失败**，不是因子失败 |
| 2 | `gates.control_audit` | 控制列在某段数据里**恒为常数**，于是"已控制该风格"是假的；或控制列两两高度共线 |
| 3 | `protocol.ablation_ladder` + `subperiod_decay` | 只看最弱的控制集（L0）会把结论夸大成"跨时代稳健"；全期聚合被早期段抬高（辛普森式） |
| 4 | `protocol.reversal_retest` | 符号是看过数据之后才翻的；"与反转的相关系数低"被误当成"不是反转"；正交掉反转就以为够了 |
| 5 | `protocol.quantile_profile` | IC 显著但组合不赚钱——因为 alpha 在**规避端**而不是头部 |

## 演示面板埋的四个陷阱

`cli demo` 会生成一份合成面板，刻意埋了四种典型情况，用来验证这套检查真的抓得到：

| 因子 | 真相 | 检查应当给出的结论 |
|---|---|---|
| `honest_factor` | 弱但真实的预测力 | ★ 幸存 |
| `reversal_proxy` | 就是 −过去收益 | 原始版本通过，**正交后归零** |
| `disguised_reversal` | 只在极端过去收益上携带信号，与过去收益的**秩相关只有 −0.57** | 原始版本看起来很强，**正交后符号翻转、样本外归零** |
| `value_style` | 只是 `value` 的暴露 | **加回风格后归零** |

另外把 `beta` 在 `train` 段的取值设成**常数**，用来验证控制可用性审计。

## 两条最重要的教训（都有单元测试守着）

**1. "与反转的相关系数低"完全不足以证明它不是反转。**

演示：`disguised_reversal` 与反转的秩相关只有 **−0.566**，原始版本两段 t 都超过 23；
可是把它对反转做逐期截面正交后，符号翻转、样本外 t 只剩 **+0.54**。
原因是一个**单调/非线性变换**可以任意压低线性相关，却完全保留单调信息 ——
必须做**多元投影**，不能只看相关系数。

**2. 正交掉反转还不够，必须把先前剔除的风格加回来。**

演示：`value_style` 的原始版本 t 超过 25，正交掉反转后依然显著；
但把 `value` 加回控制集，偏斜率 t 立刻变成 **−0.46**。
它不是新因子，它就是那个被剔掉的风格。

这两条在真实项目里各自都会静默地把一个"新因子"送进报告。

## 工程约束（为什么这样写）

* **不依赖 scipy**：Newey-West t 与正态分位数都手写（牛顿迭代），避免环境里没有 scipy 就崩。
* **不依赖 tempfile**：某些受限执行环境会拒绝 `tempfile` 的 `0700` chmod，测试改用相对目录。
* **不做任何 I/O 副作用**：所有函数只接收/返回 DataFrame，路径一律由调用方传入。
* **每个检查都返回结构化结果**，可以直接写进 CI 或报告，而不是只打印。

## 目录

```
factor_eval/
  __init__.py      包说明与五道检查的索引
  adapter.py       长表输入 + 合成演示面板（含四个陷阱）
  stats.py         Newey-West t / rank IC / 偏斜率 / 截面正交化 / Bonferroni 门槛
  gates.py         门 1 可分组性、门 2 控制可用性
  protocol.py      消融阶梯、子区间衰减、反转向复检、分位画像
  cli.py           命令行入口
tests/test_factor_eval.py   15 个测试（合成数据，无需数据库）
docs/FACTOR_EVAL_PROTOCOL.md  方法论文档
tools/run_factor_eval.ps1     Windows 运行入口
```

## 它**不**做什么

* 不提供任何因子。示例全是合成数据。
* 不保证任何因子有效。这套东西的用途是**更快地杀掉想法**，不是找到想法。
* 不做组合回测。评估协议只回答"这个信号在横截面上有没有独立信息"，
  至于能不能赚钱，还要另外考虑成交可行性、换手成本、容量与路径相依性
  （见 `docs/FACTOR_EVAL_PROTOCOL.md` 第 6 节）。
