param(
    [int]$Year = (Get-Date).Year,
    [string]$Python = "python",
    [string]$SourceDir = "data/raw/market_data",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$ForwardCutoffDate = "2026-03-31",
    [switch]$ForceReimport,
    [string]$Positions = "data/input/positions.csv",
    [string]$StrategyConfig = "config/v2h4_strategy.json",
    [string]$OutputDir = "outputs/weekly_rebalance_v2h4"
)

$ErrorActionPreference = "Stop"

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
    --positions $Positions `
    --strategy-config $StrategyConfig `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Weekly rebalance failed with exit code $LASTEXITCODE."
}
