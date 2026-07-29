param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "2026-07-24",
    [double]$InitialCash = 100000.0,
    [string]$OutputRoot = "outputs/validation_csmar_100k_candidates",
    [int]$CheckpointEveryNDays = 5,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

$Variants = @(
    @{
        Name = "v2h4_10w_25stock_balanced"
        Config = "config/v2h4_strategy_10w_25stock_balanced.json"
    },
    @{
        Name = "v2h4_10w_20stock_concentrated"
        Config = "config/v2h4_strategy_10w_20stock_concentrated.json"
    }
)

function Test-StageComplete {
    param([string]$Directory)
    $Summary = Get-ChildItem -LiteralPath $Directory -Filter "*_summary.json" -File -ErrorAction SilentlyContinue
    $Workbook = Get-ChildItem -LiteralPath $Directory -Filter "*.xlsx" -File -ErrorAction SilentlyContinue
    return ($null -ne $Summary -and $null -ne $Workbook)
}

Write-Host "Database: $Database"
Write-Host "Date range: $StartDate to $EndDate"
Write-Host "Initial cash: $InitialCash"
Write-Host "Causality: prior week decision, next-week execution, no midweek risk rebalance"
Write-Host "Broker commission: 0.03%, minimum CNY 5 per order"
Write-Host "Output root: $OutputRoot"

foreach ($Variant in $Variants) {
    Write-Host ("Variant: {0} <- {1}" -f $Variant.Name, $Variant.Config)
}

if ($PlanOnly) {
    Write-Host "Plan-only check passed; no backtest was started."
    exit 0
}

foreach ($Variant in $Variants) {
    $VariantDir = Join-Path $OutputRoot $Variant.Name
    if ($Resume -and (Test-StageComplete $VariantDir)) {
        Write-Host ("{0} is complete; skipping it." -f $Variant.Name)
        continue
    }

    New-Item -ItemType Directory -Force -Path $VariantDir | Out-Null
    $Checkpoint = Join-Path $VariantDir "v2h_checkpoint.json.gz"

    & $Python factor_rank_backtest_v2h.py `
        --strategy-config $Variant.Config `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --initial-cash $InitialCash `
        --output-dir $VariantDir `
        --checkpoint-file $Checkpoint `
        --checkpoint-every-n-days $CheckpointEveryNDays `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw ("Backtest {0} failed with exit code {1}." -f $Variant.Name, $LASTEXITCODE)
    }
}

& $Python compare_backtest_results.py `
    --root $OutputRoot `
    --holdout-start "2026-04-01" `
    --holdout-end $EndDate `
    --output (Join-Path $OutputRoot "v2h4_100k_candidate_comparison.xlsx")
if ($LASTEXITCODE -ne 0) {
    throw "Backtest comparison failed with exit code $LASTEXITCODE."
}
