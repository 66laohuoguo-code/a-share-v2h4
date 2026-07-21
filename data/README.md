# Local data directory

Everything below `data/` is local and ignored by Git, except the two example account files.

- `raw/market_data/`: local weekly forward-adjusted quotation `.xlsx` exports. Recommended name: `YYYY_YYYYMMDD_N.xlsx`. Required columns are `TradingDate`, `Symbol`, OHLC, `Volume`, `Amount`, `StateCode`, `ChangeRatio`, and `TurnoverRate1`. Prefer `AValue + ACirculatedShare` for A-share execution prices; otherwise include both `MarketValue + TotalShare` and `CirculatedMarketValue + CirculatedShare`. Adjusted OHLC values may change scale between export batches and must not be used directly for orders. Do not mix raw `TRD_Dalyr` workbooks into this directory until a dedicated adapter is added.
- `processed/`: SQLite databases and generated factor/event tables.
- `input/accounts/`: real holdings, cash and account-specific risk state. Never commit it.
- `reports/`: downloaded financial reports. Market and financial data may have redistribution restrictions.

Create missing directories as needed. The Python programs create output and account-state files automatically.

## Account inputs

Keep every brokerage account in its own directory:

```text
data/input/accounts/<account_id>/positions.csv
data/input/accounts/<account_id>/account_state.json
```

Use the same stable `account_id` on every weekly run. The state file is created automatically and must never be shared between accounts.
