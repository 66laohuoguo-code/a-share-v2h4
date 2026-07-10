# Local data directory

Everything below `data/` is local and ignored by Git, except the two example account files.

- `raw/market_data/`: local market-data `.xlsx` exports. Recommended name: `YYYY_YYYYMMDD_N.xlsx`.
- `processed/`: SQLite databases and generated factor/event tables.
- `input/positions.csv`: real holdings and cash. Never commit it.
- `input/account_state.json`: local portfolio peak used by the drawdown guard.
- `reports/`: downloaded financial reports. RESSET/CNINFO files may have redistribution restrictions.

Create missing directories as needed. The Python programs create output directories automatically.
