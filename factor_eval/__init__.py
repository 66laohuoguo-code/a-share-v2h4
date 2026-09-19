"""factor_eval — 因子评估纪律工具包（可公开部分）。

这个包**不包含任何真实因子**。它只做一件事：在你看一个因子的业绩之前，
先用几道可自动化的检查把明显不该相信的结论挡掉。

设计约束
--------
* 只依赖 numpy / pandas，不依赖 scipy、sklearn。
* **不读任何本机绝对路径**：所有输入由调用方通过 `--panel` 传入长表
  （parquet / csv / feather），或使用内置的合成演示面板。
* 不假设任何数据库 schema：只要求列名 `date`、`code`、`forward_return`。
* 每个检查都返回结构化结果（dict / DataFrame），便于写进报告或 CI。

五道检查
--------
1. `gates.groupability`      —— 算分位统计**之前**先问"这个信号能不能分组"
2. `gates.control_audit`     —— 控制变量在某段数据里是否恒为常数 / 高度共线
3. `protocol.ablation_ladder`—— 分层控制消融，并按子区间拆开看衰减
4. `protocol.reversal_retest`—— 反转向复检（符号冻结 + 试验数校正 + 残差对象）
5. `protocol.quantile_profile`—— IC 与组合层是否脱钩（alpha 在头部还是在规避端）
"""

from . import adapter, gates, protocol, stats  # noqa: F401

__all__ = ["adapter", "gates", "protocol", "stats"]
__version__ = "1.0.0"
