# 每周五收盘后的操作

这份文档只描述每周新增前推行情后的标准流程。历史行情数据库已经建立后，不要再用旧历史清洗器处理周文件。

## 1. 下载本周前推行情

每周文件必须是日行情表，并至少包含：

```text
TradingDate, Symbol, OpenPrice, ClosePrice, HighPrice, LowPrice,
Volume, Amount, StateCode, ChangeRatio, TurnoverRate1
```

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

## 2. 更新真实持仓

编辑：

```text
data/input/positions.csv
```

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
  -Positions ".\data\input\positions.csv" `
  -Year (Get-Date).Year
```

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

打开 `outputs/weekly_rebalance_v2h4/` 中最新的 `.xlsx`：

- `summary`：当前仓位、目标仓位、市场状态和警告。
- `orders`：建议买卖方向与股数。
- `projected_positions`：假设全部成交后的预计持仓。

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
- 数据库日期没有变化：文件被识别为未改变并跳过，或文件不含新交易日。
- `missing required columns`：误把旧历史格式、财报或其他 Excel 放入了周行情目录。

成交后，以券商实际结果更新 `data/input/positions.csv`，下周继续相同步骤。
