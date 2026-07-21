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
    [string]$StrategyConfig = "config/v2h4_strategy.json",
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

& $Python weekly_rebalance_v2h.py `
    --database $Database `
    --account-id $AccountId `
    --positions $Positions `
    --account-state $AccountState `
    --strategy-config $StrategyConfig `
    --output-dir $AccountOutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Weekly rebalance failed with exit code $LASTEXITCODE."
}
