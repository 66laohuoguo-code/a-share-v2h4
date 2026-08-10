param(
    [string]$TargetDate = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Settings = Get-OfficialLiveSettings

$Parameters = @{
    Python = [string]$Settings.Python
    MarketDatabase = [string]$Settings.MarketDatabase
    RiskDatabase = [string]$Settings.RiskDatabase
    SourceDir = [string]$Settings.SourceDir
    RiskModelConfig = [string]$Settings.RiskModelConfig
    RiskDataRawRoot = [string]$Settings.RiskDataRawRoot
    Workers = [int]$Settings.RiskModelWorkers
}
if (-not [string]::IsNullOrWhiteSpace($TargetDate)) {
    $Parameters["TargetDate"] = $TargetDate
}

& (Join-Path ([string]$Settings.ProjectRoot) "update_live_risk_model.ps1") @Parameters
