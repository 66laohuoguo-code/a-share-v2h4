param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$StartDate = "2021-01-05",
    [string]$EndDate = "",
    [switch]$IncludeStress,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EndDate)) {
    $EndDate = (& $Python database_status.py --database $Database --field max_date).Trim()
}

Write-Host "Backtest database: $Database"
Write-Host "Backtest date range: $StartDate to $EndDate"

if ($PlanOnly) {
    Write-Host "Plan-only check passed; no backtest was started."
    exit 0
}

& $Python factor_rank_backtest_v2h.py `
    --strategy-config config/v2h4_strategy.json `
    --database $Database `
    --start-date $StartDate `
    --end-date $EndDate `
    --output-dir outputs/validation_full/v2h4_lot_aware_official

& $Python factor_rank_backtest_v2h.py `
    --strategy-config config/v2h4_fixed_floor_legacy.json `
    --database $Database `
    --start-date $StartDate `
    --end-date $EndDate `
    --output-dir outputs/validation_full/v2h4_fixed_floor_legacy

if ($IncludeStress) {
    & $Python factor_rank_backtest_v2h.py `
        --strategy-config config/v2h4_strategy.json `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --slippage-bps 10 `
        --output-dir outputs/validation_full/v2h4_lot_aware_official_slippage10
}

& $Python compare_backtest_results.py `
    --root outputs/validation_full `
    --holdout-start 2026-04-01 `
    --holdout-end $EndDate `
    --output outputs/validation_full/v2h4_comparison.xlsx
