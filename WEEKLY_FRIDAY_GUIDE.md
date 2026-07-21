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
account_56w
account_1w
```

账户文件分别放在：

```text
data/input/accounts/account_56w/positions.csv
data/input/accounts/account_1w/positions.csv
```

首次创建目录和模板：

```powershell
New-Item -ItemType Directory -Force "data/input/accounts/account_56w"
New-Item -ItemType Directory -Force "data/input/accounts/account_1w"
Copy-Item "data/input/positions.example.csv" "data/input/accounts/account_56w/positions.csv"
Copy-Item "data/input/positions.example.csv" "data/input/accounts/account_1w/positions.csv"
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

```powershell
Set-Location "<项目目录>"

powershell.exe `
  -NoProfile `
  -ExecutionPolicy Bypass `
  -File ".\run_weekly.ps1" `
  -Python ".\.venv\Scripts\python.exe" `
  -Database ".\data\processed\stock_daily.sqlite" `
  -SourceDir ".\data\raw\market_data" `
  -AccountId "account_56w" `
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
  -AccountId "account_1w" `
  -Year (Get-Date).Year
```

默认情况下，`run_weekly.ps1` 会根据 `AccountId` 自动选择：

```text
持仓：data/input/accounts/<AccountId>/positions.csv
状态：data/input/accounts/<AccountId>/account_state.json
结果：outputs/weekly_rebalance_v2h4/<AccountId>/
```

`account_state.json` 会保存该账户自己的历史峰值和最近净值。状态文件中的账户 ID 与本次参数不一致时，程序会直接报错，防止账户串用。

程序会自动：

1. 用 `import_csmar_forward_quotation.py` 读取前推行情。
2. 跳过已经成功导入且没有变化的 Excel。
3. 以前一有效收盘价为锚点续接周度价格链。
4. 校验数据库并显示最大交易日。
5. 读取真实持仓，计算目标组合和订单。

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
outputs/weekly_rebalance_v2h4/account_56w/
outputs/weekly_rebalance_v2h4/account_1w/
```

- `summary`：当前仓位、目标仓位、市场状态和警告。
- `orders`：建议买卖方向与股数。
- `projected_positions`：假设全部成交后的预计持仓。

`orders` 中 `reference_close` 是最近交易日的未复权市场收盘价。`indicative_price` 只是在收盘价上加入配置滑点并按 `0.01` 元取整的预算价格，不是下个交易日的保证成交价，也不是必须照抄的委托价格。下个交易日会因为集合竞价、盘口变化和价格优先/时间优先规则产生不同的实际成交价。

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
