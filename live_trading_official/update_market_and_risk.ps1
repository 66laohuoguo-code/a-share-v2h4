param(
    [switch]$ForceReimport
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Settings = Get-OfficialLiveSettings
$ProjectRoot = [string]$Settings.ProjectRoot

Push-Location $ProjectRoot
try {
    $ImportArgs = @(
        "import_csmar_forward_quotation.py",
        "--source-dir", [string]$Settings.SourceDir,
        "--database", [string]$Settings.MarketDatabase,
        "--cutoff-date", [string]$Settings.ForwardCutoffDate,
        "--years", [string]$Settings.Year,
        "--apply"
    )
    if ($ForceReimport) {
        $ImportArgs += "--force-reimport"
    }
    & ([string]$Settings.Python) @ImportArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Weekly market-data import failed with exit code $LASTEXITCODE."
    }

    & ([string]$Settings.Python) validate_clean_data.py ([string]$Settings.MarketDatabase)
    if ($LASTEXITCODE -ne 0) {
        throw "Market database validation failed with exit code $LASTEXITCODE."
    }

    & (Join-Path $PSScriptRoot "update_risk_model.ps1")
}
finally {
    Pop-Location
}
