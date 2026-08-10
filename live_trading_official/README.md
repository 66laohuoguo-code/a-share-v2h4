# Live Trading Operations

This directory provides the stable entry points for weekly portfolio maintenance.
The implementation remains at the repository root so there is only one copy of
each strategy, data-pipeline, and risk-model module.

## Setup

1. Install the dependencies listed in the root `requirements.txt`.
2. Copy `settings.example.json` to `settings.local.json`.
3. Edit `settings.local.json` with local Python, database, input, and output paths.
4. Create each account position file at
   `data/input/accounts/<account_id>/positions.csv` using
   `data/input/positions.example.csv` as the template.
5. Keep licensed data, databases, account files, API keys, and generated reports
   outside Git.

`settings.local.json` is ignored by Git. Do not place secrets in the example file.

## Commands

Run commands from this directory in Windows PowerShell:

```powershell
Copy-Item ".\settings.example.json" ".\settings.local.json"
notepad ".\settings.local.json"

& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -NoProfile -ExecutionPolicy Bypass -File ".\check_status.ps1"
```

Update market data, the weekly risk model, and the V3.1 alpha cache without
creating orders:

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -NoProfile -ExecutionPolicy Bypass -File ".\update_market_and_risk.ps1"
```

Run one account:

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -NoProfile -ExecutionPolicy Bypass `
  -File ".\run_account.ps1" -AccountId "account_example"
```

The amount-named wrappers are optional convenience commands for the default
sample account IDs. Strategy selection itself uses current portfolio value and
`config/weekly_capital_strategy_map.json`; it is not fixed by the wrapper name.

## Weekly Workflow

1. Place the latest market workbook in the configured `SourceDir`.
2. Reconcile `positions.csv` with actual broker holdings and available cash.
3. Run `check_status.ps1`.
4. Run the required account command.
5. Review `summary`, `orders`, `filtered_orders`, and `projected_positions` in
   the newest output workbook.
6. Check suspensions, price limits, auction prices, available funds, and broker
   order rules before submitting any order.
7. After execution, update `positions.csv` from actual fills.

The first account run may update the shared market and risk databases. Do not run
multiple accounts concurrently while they target the same database files.

`official_files.json` is the machine-readable list of source modules and formal
strategy configurations required by the live workflow.
