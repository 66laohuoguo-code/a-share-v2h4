param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$StartDate = "2021-01-05",
    [string]$EndDate = "",
    [string]$OutputRoot = "outputs/validation_full",
    [switch]$IncludeStress,
    [switch]$EnableCheckpoints,
    [int]$CheckpointEveryNDays = 5,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EndDate)) {
    $EndDate = (& $Python database_status.py --database $Database --field max_date).Trim()
}

Write-Host "Backtest database: $Database"
Write-Host "Backtest date range: $StartDate to $EndDate"
Write-Host "Backtest output root: $OutputRoot"
Write-Host "Checkpoint resume: $Resume"

if ($PlanOnly) {
    Write-Host "Plan-only check passed; no backtest was started."
    exit 0
}

if ($Resume) {
    $EnableCheckpoints = $true
}

function Test-StageComplete {
    param([string]$Directory)
    $Summary = Get-ChildItem -LiteralPath $Directory -Filter "*_summary.json" -File -ErrorAction SilentlyContinue
    $Workbook = Get-ChildItem -LiteralPath $Directory -Filter "*.xlsx" -File -ErrorAction SilentlyContinue
    return ($null -ne $Summary -and $null -ne $Workbook)
}

$OfficialDir = Join-Path $OutputRoot "v2h4_lot_aware_official"
$LegacyDir = Join-Path $OutputRoot "v2h4_fixed_floor_legacy"
$StressDir = Join-Path $OutputRoot "v2h4_lot_aware_official_slippage10"

$OfficialCheckpointArgs = @()
$LegacyCheckpointArgs = @()
$StressCheckpointArgs = @()
if ($EnableCheckpoints) {
    $OfficialCheckpointArgs = @(
        "--checkpoint-file", (Join-Path $OfficialDir "v2h_checkpoint.json.gz"),
        "--checkpoint-every-n-days", "$CheckpointEveryNDays",
        "--resume"
    )
    $LegacyCheckpointArgs = @(
        "--checkpoint-file", (Join-Path $LegacyDir "v2h_checkpoint.json.gz"),
        "--checkpoint-every-n-days", "$CheckpointEveryNDays",
        "--resume"
    )
    $StressCheckpointArgs = @(
        "--checkpoint-file", (Join-Path $StressDir "v2h_checkpoint.json.gz"),
        "--checkpoint-every-n-days", "$CheckpointEveryNDays",
        "--resume"
    )
}

if ($Resume -and (Test-StageComplete $OfficialDir)) {
    Write-Host "Official V2H4 is complete; skipping it."
}
else {
    & $Python factor_rank_backtest_v2h.py `
        --strategy-config config/v2h4_strategy.json `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --output-dir $OfficialDir `
        @OfficialCheckpointArgs
    if ($LASTEXITCODE -ne 0) { throw "Official V2H4 backtest failed with exit code $LASTEXITCODE." }
}

if ($Resume -and (Test-StageComplete $LegacyDir)) {
    Write-Host "Legacy comparison is complete; skipping it."
}
else {
    & $Python factor_rank_backtest_v2h.py `
        --strategy-config config/v2h4_fixed_floor_legacy.json `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --output-dir $LegacyDir `
        @LegacyCheckpointArgs
    if ($LASTEXITCODE -ne 0) { throw "Legacy comparison backtest failed with exit code $LASTEXITCODE." }
}

if ($IncludeStress) {
    if ($Resume -and (Test-StageComplete $StressDir)) {
        Write-Host "Slippage stress test is complete; skipping it."
    }
    else {
        & $Python factor_rank_backtest_v2h.py `
            --strategy-config config/v2h4_strategy.json `
            --database $Database `
            --start-date $StartDate `
            --end-date $EndDate `
            --slippage-bps 10 `
            --output-dir $StressDir `
            @StressCheckpointArgs
        if ($LASTEXITCODE -ne 0) { throw "Slippage stress backtest failed with exit code $LASTEXITCODE." }
    }
}

& $Python compare_backtest_results.py `
    --root $OutputRoot `
    --holdout-start 2026-04-01 `
    --holdout-end $EndDate `
    --output (Join-Path $OutputRoot "v2h4_comparison.xlsx")
if ($LASTEXITCODE -ne 0) { throw "Backtest comparison failed with exit code $LASTEXITCODE." }
