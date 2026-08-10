param(
    [switch]$ForceReimport,
    [switch]$ResetPeakToCurrent
)

$Parameters = @{AccountId = "account_1w"}
if ($ForceReimport) { $Parameters["ForceReimport"] = $true }
if ($ResetPeakToCurrent) { $Parameters["ResetPeakToCurrent"] = $true }
& (Join-Path $PSScriptRoot "run_account.ps1") @Parameters
