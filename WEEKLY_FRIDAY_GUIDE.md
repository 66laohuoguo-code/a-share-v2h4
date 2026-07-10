# 每周五收盘后的操作

这份文档只讲每周实际要做的事情。正常情况下，不需要单独运行清洗、因子和调仓三个程序，`run_weekly.ps1` 会按顺序全部完成。

## 一、每周只需要准备三样东西

1. 本周新增的日线行情 Excel。
2. 券商账户中周五收盘后的真实持仓。
3. 你一直使用的 SQLite 数据库路径。

## 二、下载并放好本周数据

周五收盘并等待数据源更新后，下载本周新增的日线数据，放到：

```text
data/raw/market_data/
```

文件名统一写成：

```text
年份_下载日期_序号.xlsx
```

例如本次被拆成两份：

```text
2026_20260710_1.xlsx
2026_20260710_2.xlsx
```

Excel 内必须继续包含代码、名称、交易日期、昨收、开高低收、成交额、换手率、总回报、资本回报、上市状态、币种和行业等与历史文件相同的字段。

## 三、更新真实持仓

打开：

```text
data/input/positions.csv
```

填写周五收盘后券商显示的实际数据：

```csv
code,name,shares,cost_price
000001,股票A,1200,10.85
600000,股票B,800,8.42
CASH,现金,235000,
```

- `shares` 填实际股数。
- `cost_price` 填券商显示的成本价，只用于核对。
- `CASH` 行的 `shares` 填可用于买股的现金金额。
- 已清仓股票直接删除该行。
- 不要填写尚未成交的委托。

## 四、运行每周程序

打开 PowerShell，进入项目目录并激活环境：

```powershell
Set-Location "<项目目录>"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

这里的 `Process` 只对当前 PowerShell 窗口生效，关闭窗口后自动恢复，不会永久修改系统策略。

设置你实际使用的数据库。这里只需要把占位符换成自己的文件路径：

```powershell
$Database = "<数据库文件路径>"
```

执行完整周流程：

```powershell
.\run_weekly.ps1 `
  -Year (Get-Date).Year `
  -Database $Database
```

程序会自动完成：

1. 导入尚未处理的新 Excel。
2. 更新数据库中的重复日期记录。
3. 检查数据库质量并打印最大交易日。
4. 计算本周 V2H4 因子排名。
5. 读取你的已有持股和现金。
6. 计算目标股票仓位。
7. 生成下一交易日调仓建议。

看到下面这行时，日期必须等于本周最后一个交易日：

```text
Database maximum trading date: YYYY-MM-DD
```

如果日期不对，立即停止，不要使用这次订单。先检查下载文件是否包含周五数据，再重新导入。

## 五、查看调仓结果

打开下面目录中时间最新的 `.xlsx`：

```text
outputs/weekly_rebalance_v2h4/
```

只需要重点看两个工作表：

- `summary`：当前仓位、模型目标仓位、市场状态和警告。
- `orders`：下一个交易日建议买卖的代码和股数。

`factor_ranking` 是完整排名，平时不需要逐只查看。`projected_positions` 只是按参考价假设全部成交后的结果，不是真实持仓。

出现以下情况时不要直接执行：

- 数据库最大日期不正确。
- `summary` 提示目标仓位与预计仓位相差很大。
- 某股票停牌、涨跌停或你没有对应交易权限。
- 实际可用现金与 `positions.csv` 不一致。

## 六、下一个交易日怎么操作

1. 开盘前重新检查停牌和风险警示状态。
2. 优先处理 `SELL` 卖单。
3. 等卖单成交并确认可用现金。
4. 再处理 `BUY` 买单。
5. 开盘价与报告参考价差异过大时，不要机械追价。

程序只生成辅助计划，不会自动连接券商或自动下单。

## 七、成交后做什么

以券商的真实成交结果为准，更新 `data/input/positions.csv`：

- 改成实际成交后的股数。
- 更新实际可用现金。
- 更新成本价。
- 删除已经清仓的股票。

下周五再次重复本文件的步骤即可。

## 八、运行修正后的完整回测

修正版会处理历史税费、送转股、现金分红、不同板块申报数量和交易门槛。数据库最大日期会自动作为回测结束日期：

```powershell
$Database = "<数据库文件路径>"

.\run_v2h4_validation.ps1 `
  -Database $Database `
  -StartDate 2021-01-05 `
  -IncludeStress
```

它会顺序运行原 V2H4、缩放交易门槛候选版和 10 bps 滑点压力测试，最后生成：

```text
outputs/validation_full/v2h4_comparison.xlsx
```

在完整对比结果出来前，每周默认仍使用 `config/v2h4_strategy.json`，不要直接把候选版设为实盘默认。

## 九、检查数据库到底截止哪一天

检查一个数据库：

```powershell
python database_status.py --database "<数据库文件路径>"
```

检查 `data/processed` 中所有 SQLite 数据库：

```powershell
Get-ChildItem data/processed -Filter *.sqlite | ForEach-Object {
  python database_status.py --database $_.FullName
}
```
