param(
    [int]$Year = (Get-Date).Year,
    [string]$Python = "python",
    [string]$SourceDir = "data/raw/market_data",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$Positions = "data/input/positions.csv",
    [string]$StrategyConfig = "config/v2h4_strategy.json",
    [string]$OutputDir = "outputs/weekly_rebalance_v2h4"
)

$ErrorActionPreference = "Stop"

& $Python clean_resset_data.py `
    --source-dir $SourceDir `
    --database $Database `
    --years $Year

& $Python validate_clean_data.py $Database

$MaxDate = (& $Python database_status.py --database $Database --field max_date).Trim()
Write-Host "Database maximum trading date: $MaxDate"

& $Python weekly_rebalance_v2h.py `
    --database $Database `
    --positions $Positions `
    --strategy-config $StrategyConfig `
    --output-dir $OutputDir
