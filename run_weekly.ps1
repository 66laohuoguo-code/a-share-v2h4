param(
    [int]$Year = (Get-Date).Year,
    [string]$Python = "python",
    [string]$SourceDir = "data/raw/market_data",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$AccountId,
    [string]$ForwardCutoffDate = "2026-03-31",
    [switch]$ForceReimport,
    [string]$Positions = "",
    [string]$AccountState = "",
    [string]$StrategyConfig = "",
    [string]$CapitalStrategyMap = "config/weekly_capital_strategy_map.json",
    [string]$RiskDatabase = $env:ASHARE_RISK_DATABASE,
    [string]$RiskCalibrationSchedule = $env:ASHARE_RISK_CALIBRATION_SCHEDULE,
    [string]$RiskModelConfig = "config/risk_model_v31_full_2019_20260717.json",
    [string]$RiskDataRawRoot = "data/raw/CSMAR raw data",
    [ValidateRange(1, 32)]
    [int]$RiskModelWorkers = 4,
    [switch]$SkipRiskModelUpdate,
    [switch]$ResetPeakToCurrent,
    [string]$OutputDir = "outputs/weekly_rebalance_v2h4"
)

$ErrorActionPreference = "Stop"

$AccountDir = Join-Path "data/input/accounts" $AccountId
if ([string]::IsNullOrWhiteSpace($Positions)) {
    $Positions = Join-Path $AccountDir "positions.csv"
}
if ([string]::IsNullOrWhiteSpace($AccountState)) {
    $AccountState = Join-Path $AccountDir "account_state.json"
}
$AccountOutputDir = Join-Path $OutputDir $AccountId

if (-not (Test-Path -LiteralPath $Positions -PathType Leaf)) {
    throw "Positions file not found for account '$AccountId': $Positions"
}

$ImportArgs = @(
    "import_csmar_forward_quotation.py",
    "--source-dir", $SourceDir,
    "--database", $Database,
    "--cutoff-date", $ForwardCutoffDate,
    "--years", $Year,
    "--apply"
)
if ($ForceReimport) {
    $ImportArgs += "--force-reimport"
}

& $Python @ImportArgs
if ($LASTEXITCODE -ne 0) {
    throw "Weekly market-data import failed with exit code $LASTEXITCODE."
}

& $Python validate_clean_data.py $Database
if ($LASTEXITCODE -ne 0) {
    throw "Database validation failed with exit code $LASTEXITCODE."
}

$MaxDate = (& $Python database_status.py --database $Database --field max_date).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Database status check failed with exit code $LASTEXITCODE."
}
Write-Host "Database maximum trading date: $MaxDate"

if (
    -not [string]::IsNullOrWhiteSpace($RiskDatabase) -and
    -not $SkipRiskModelUpdate
) {
    & (Join-Path $PSScriptRoot "update_live_risk_model.ps1") `
        -Python $Python `
        -MarketDatabase $Database `
        -RiskDatabase $RiskDatabase `
        -SourceDir $SourceDir `
        -RiskModelConfig $RiskModelConfig `
        -RiskDataRawRoot $RiskDataRawRoot `
        -TargetDate $MaxDate `
        -Workers $RiskModelWorkers
}

$RebalanceArgs = @(
    "weekly_rebalance_v2h.py",
    "--database", $Database,
    "--account-id", $AccountId,
    "--positions", $Positions,
    "--account-state", $AccountState,
    "--capital-strategy-map", $CapitalStrategyMap,
    "--output-dir", $AccountOutputDir
)
if (-not [string]::IsNullOrWhiteSpace($StrategyConfig)) {
    $RebalanceArgs += @("--strategy-config", $StrategyConfig)
}
if (-not [string]::IsNullOrWhiteSpace($RiskDatabase)) {
    $RebalanceArgs += @("--risk-model-database", $RiskDatabase)
}
if (-not [string]::IsNullOrWhiteSpace($RiskCalibrationSchedule)) {
    $RebalanceArgs += @("--risk-calibration-schedule", $RiskCalibrationSchedule)
}
if ($ResetPeakToCurrent) {
    $RebalanceArgs += "--reset-peak-to-current"
}

& $Python @RebalanceArgs
if ($LASTEXITCODE -ne 0) {
    throw "Weekly rebalance failed with exit code $LASTEXITCODE."
}
