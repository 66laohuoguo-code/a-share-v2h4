param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$AccountId,
    [switch]$ForceReimport,
    [switch]$ResetPeakToCurrent
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Settings = Get-OfficialLiveSettings

$ProjectRoot = [string]$Settings.ProjectRoot
$Positions = Join-Path $ProjectRoot "data\input\accounts\$AccountId\positions.csv"
Assert-OfficialLivePath -Path $Positions -Label "Positions file for $AccountId"

$RunParameters = @{
    Year = [int]$Settings.Year
    Python = [string]$Settings.Python
    SourceDir = [string]$Settings.SourceDir
    Database = [string]$Settings.MarketDatabase
    AccountId = $AccountId
    ForwardCutoffDate = [string]$Settings.ForwardCutoffDate
    CapitalStrategyMap = [string]$Settings.CapitalStrategyMap
    RiskDatabase = [string]$Settings.RiskDatabase
    RiskModelConfig = [string]$Settings.RiskModelConfig
    RiskDataRawRoot = [string]$Settings.RiskDataRawRoot
    RiskModelWorkers = [int]$Settings.RiskModelWorkers
    OutputDir = [string]$Settings.OutputDir
}
if (
    $Settings.PSObject.Properties.Name -contains "RiskCalibrationSchedule" -and
    -not [string]::IsNullOrWhiteSpace([string]$Settings.RiskCalibrationSchedule)
) {
    $RunParameters["RiskCalibrationSchedule"] = [string]$Settings.RiskCalibrationSchedule
}
if ($ForceReimport) {
    $RunParameters["ForceReimport"] = $true
}
if ($ResetPeakToCurrent) {
    $RunParameters["ResetPeakToCurrent"] = $true
}

Push-Location $ProjectRoot
try {
    & (Join-Path $ProjectRoot "run_weekly.ps1") @RunParameters
}
finally {
    Pop-Location
}
