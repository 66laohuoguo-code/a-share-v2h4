param(
    [switch]$ForceReimport,
    [switch]$ResetPeakToCurrent
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Settings = Get-OfficialLiveSettings

foreach ($AccountId in @($Settings.Accounts)) {
    Write-Host ""
    Write-Host "===== Running weekly rebalance for $AccountId ====="
    $Parameters = @{AccountId = [string]$AccountId}
    if ($ForceReimport) { $Parameters["ForceReimport"] = $true }
    if ($ResetPeakToCurrent) { $Parameters["ResetPeakToCurrent"] = $true }
    & (Join-Path $PSScriptRoot "run_account.ps1") @Parameters
}
