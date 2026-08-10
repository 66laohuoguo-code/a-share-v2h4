# 每周五收盘后的操作

这份文档只描述每周新增前推行情后的标准流程。历史行情数据库已经建立后，不要再用旧历史清洗器处理周文件。

## 1. 下载本周前推行情

每周文件必须是日行情表，并至少包含：

```text
TradingDate, Symbol, OpenPrice, ClosePrice, HighPrice, LowPrice,
Volume, Amount, StateCode, ChangeRatio, TurnoverRate1,
AValue, ACirculatedShare
```

`AValue + ACirculatedShare` 是 A 股的优先口径。若下载页面不提供它们，请把 `MarketValue, TotalShare, CirculatedMarketValue, CirculatedShare` 四列一起下载。下载前复权行情没有问题，但其中的 OHLC 不能当作券商价格，也不能把不同周文件的复权价格直接拼接。程序优先用“流通 A 股市值（元）/流通 A 股股本（股）”恢复券商口径价格；备用公司口径发生冲突时，会用上一真实收盘价和当日涨跌幅续接并在日志中显示 `execution_price_return_chain_fallback`。

`TRD_Dalyr（日个股回报率文件）` 的未复权 OHLC 是更理想的执行价来源，但当前周流程尚未适配它。现阶段继续下载上述前复权字段，不要把 `TRD_Dalyr` 文件混入 `data/raw/market_data/`。

把文件放入：

```text
data/raw/market_data/
```

建议命名：

```text
YYYY_YYYYMMDD_N.xlsx
```

例如：

```text
2026_20260712_1.xlsx
2026_20260712_2.xlsx
```

文件名用于整理和年份筛选，实际交易日期以 `TradingDate` 为准。同一周文件可以与数据库已有日期重叠，程序会更新重复日期并新增后续日期。

## 2. 为每个账户保存独立持仓

每个券商账户必须有一个固定且互不重复的 `AccountId`。例如：

```text
account_a
account_b
```

账户文件分别放在：

```text
data/input/accounts/account_a/positions.csv
data/input/accounts/account_b/positions.csv
```

首次创建目录和模板：

```powershell
New-Item -ItemType Directory -Force "data/input/accounts/account_a"
New-Item -ItemType Directory -Force "data/input/accounts/account_b"
Copy-Item "data/input/positions.example.csv" "data/input/accounts/account_a/positions.csv"
Copy-Item "data/input/positions.example.csv" "data/input/accounts/account_b/positions.csv"
```

然后根据对应券商账户修改各自的 `positions.csv`：

```csv
code,name,shares,cost_price
000001,股票A,1200,10.85
600000,股票B,800,8.42
CASH,现金,235000,
```

- 股票行填写实际股数和成本价。
- `CASH` 行填写可用于买股的现金。
- 删除已经清仓的股票。
- 不填写尚未成交的委托。

不要让两个账户轮流覆盖同一个 `positions.csv`。程序会为每个 `AccountId` 自动维护独立的 `account_state.json`，其中的历史峰值只属于该账户。第一次运行时不需要手工创建状态文件。

## 3. 运行每周流程

已经配置本机 `live_trading_official/settings.local.json` 时，优先使用
`live_trading_official/run_account_*.ps1` 或 `run_all_accounts.ps1`。下面的
长命令保留给通用部署和故障排查。

```powershell
Set-Location "<项目目录>"

powershell.exe `
  -NoProfile `
  -ExecutionPolicy Bypass `
  -File ".\run_weekly.ps1" `
  -Python ".\.venv\Scripts\python.exe" `
  -Database ".\data\processed\stock_daily.sqlite" `
  -SourceDir ".\data\raw\market_data" `
  -RiskDatabase ".\data\processed\weekly_risk_model.sqlite" `
  -RiskCalibrationSchedule ".\data\processed\weekly_risk_calibration_schedule.csv" `
  -AccountId "account_a" `
  -Year (Get-Date).Year
```

第二个账户单独再运行一次，只改变账户 ID：

```powershell
powershell.exe `
  -NoProfile `
  -ExecutionPolicy Bypass `
  -File ".\run_weekly.ps1" `
  -Python ".\.venv\Scripts\python.exe" `
  -Database ".\data\processed\stock_daily.sqlite" `
  -SourceDir ".\data\raw\market_data" `
  -RiskDatabase ".\data\processed\weekly_risk_model.sqlite" `
  -RiskCalibrationSchedule ".\data\processed\weekly_risk_calibration_schedule.csv" `
  -AccountId "account_b" `
  -Year (Get-Date).Year
```

默认情况下，`run_weekly.ps1` 会根据 `AccountId` 自动选择账户文件：

```text
持仓：data/input/accounts/<AccountId>/positions.csv
状态：data/input/accounts/<AccountId>/account_state.json
结果：outputs/weekly_rebalance_v2h4/<AccountId>/
```

`account_state.json` 会保存该账户自己的历史峰值和最近净值。状态文件中的账户 ID 与本次参数不一致时，程序会直接报错，防止账户串用。

程序还会按照最新收盘价计算当前账户总资产并自动选策略：

| 当前账户总资产 | 自动策略 |
|---:|---|
| 低于 5 万元 | `v22s_20k_entry_weight_monthly_06_official.json`（12只、风险叠加、月度行业入场与权重卫星、Q90） |
| 5 万元至低于 30 万元 | `v22r3_weekly_100k_official.json`（20只、价值质量倾斜、UNKNOWN行业5%上限、Q97.5） |
| 30 万元至低于 75 万元 | `v22r3_weekly_560k_official.json`（20只、价值质量倾斜、UNKNOWN行业5%上限、Q97.5） |
| 75 万元及以上 | `v22r3_weekly_1m_official.json`（20只、价值质量倾斜、UNKNOWN行业5%上限、Q97.5） |

这里使用的是实际资产，不是根据 `account_a`、`account_b` 等名称猜测。资金分档配置保存在 `config/weekly_capital_strategy_map.json`。只有需要故意固定某个策略时才传入 `-StrategyConfig`；正常每周流程不要传。

所有自动档都要读取风险数据库。2万元档使用周频风险叠加，并在每月第一个周末决策日冻结行业 20/60 日相对趋势信号；V2.2 R3资金档读取时点化盈利收益率和质量Alpha。`run_weekly.ps1` 默认会先把风险模型和Alpha缓存增量更新到行情库的最大日期，再生成订单。只有显式传入 `-SkipRiskModelUpdate` 才会跳过这一步；一般实盘周流程不要使用该开关。

四个资金档的验证资金、自动选择范围和使用边界见 [ACCOUNT_STRATEGY_TIERS.md](ACCOUNT_STRATEGY_TIERS.md)。每次运行后，在 `summary` 中确认 `capital_strategy_tier` 与 `strategy_config` 是否符合该表。

程序会自动：

1. 用 `import_csmar_forward_quotation.py` 读取前推行情。
2. 跳过已经成功导入且没有变化的 Excel。
3. 以前一有效收盘价为锚点续接周度价格链。
4. 校验数据库并显示最大交易日。
5. 将新行情中的总市值和流通市值写入风险侧库。
6. 只计算尚未存在的新风险模型周，并增量追加V3.1 Alpha缓存。
7. 读取真实持仓和现金，按当前总资产选择策略。
8. 用本周最新数据重新计算选股因子排名。
9. 读取截至决策日已经存在的风险快照、时点财务Alpha和可用的校准倍率。
10. 生成目标组合、集合竞价限价和订单。

同一周运行多个账户时，第一个账户会完成风险模型更新，耗时会明显更长；后续账户检查到风险库已经是最新日期后会直接复用。不要并行运行两个写入同一风险数据库的周流程。

正式策略会跳过最小申报数量超出账户预算的候选，继续寻找下一只可买股票；小账户会减少持股数量，并在报告中显示实际持股数、动态单股上限和动态行业上限。

## 4. 检查最大日期

必须看到：

```text
Database maximum trading date: YYYY-MM-DD
```

它必须等于新 Excel 中最后一个实际交易日。如果不一致，不要使用本次订单。

## 5. 查看和执行订单

打开对应账户目录中最新的 `.xlsx`，例如：

```text
outputs/weekly_rebalance_v2h4/account_a/
outputs/weekly_rebalance_v2h4/account_b/
```

- `summary`：当前仓位、目标仓位、市场状态和警告。
- `orders`：建议买卖方向与股数。
- `filtered_orders`：被最小交易金额、账户净值比例等规则过滤的候选订单。
- `projected_positions`：假设全部成交后的预计持仓。

在 `summary` 中，`target_equity_weight` 是行情风控的基础目标，`effective_target_equity_weight` 是风险模型处理后的最终有效目标。实际执行应以有效目标为准，并同时检查 `current_equity_weight` 与 `projected_equity_weight`。如果警告显示换仓仓位保护延期了卖单，表示替代买单当前不可执行；保留旧仓是为了避免只卖不买导致意外低仓位。

`orders` 中的价格必须分开理解：

- `reference_close`：最近交易日未复权收盘价。
- `indicative_price` / `estimated_execution_price`：模型预计的开盘中心价格，只用于判断，不是券商委托价。
- `broker_order_limit_price` / `auction_limit_price`：券商集合竞价限价；买单是最高接受价，卖单是最低接受价。
- `cash_reservation_price`：程序计算整手数量和预留现金时采用的保守价格，通常等于保护限价。

建议在 `09:15-09:20` 仍可撤单的阶段查看虚拟开盘参考价，再提交集合竞价限价单。集合竞价的实际成交使用交易所形成的单一开盘价，并不直接使用保护限价；开盘集合竞价后仍未成交的剩余委托应撤销，不要继续留在连续竞价。

正式 `config/v2h4_strategy.json` 当前按券商万分之三、每笔最低 5 元估算佣金。`estimated_fee` 已包含券商佣金和法定费用，并参与买入现金检查。更换券商或费率后必须同步修改配置中的 `broker_commission_rate` 和 `broker_minimum_commission`。

开盘前重新检查停牌、ST、涨跌停、权限和现金。优先处理卖单，确认卖出资金后再处理买单。程序不会自动连接券商或自动下单。

## 6. 文件格式出错时先干跑

```powershell
python import_csmar_forward_quotation.py `
  --source-xlsx "data/raw/market_data/YYYY_YYYYMMDD_1.xlsx" `
  --database "data/processed/stock_daily.sqlite"
```

不添加 `--apply` 时不会修改数据库。正常输出应列出已识别字段、源文件日期范围、锚点日期和可导入行数。

常见错误：

- 缺少 `Symbol`：下载时没有选择证券代码。
- 缺少 `ChangeRatio`：无法连续计算收益率和合成复权价格。
- 缺少市值与股本组合：无法从前复权行情还原真实交易价格。
- 数据库日期没有变化：文件被识别为未改变并跳过，或文件不含新交易日。
- `missing required columns`：误把旧历史格式、财报或其他 Excel 放入了周行情目录。

成交后，以券商实际结果更新对应账户目录里的 `positions.csv`，下周继续使用同一个 `AccountId`。不要把一个账户的 `account_state.json` 复制给另一个账户。
