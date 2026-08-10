$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Settings = Get-OfficialLiveSettings
$ProjectRoot = [string]$Settings.ProjectRoot

Write-Host "Checking official live-trading files..."
$ManifestPath = Join-Path $PSScriptRoot "official_files.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Missing = @()
foreach ($Group in $Manifest.PSObject.Properties) {
    foreach ($RelativePath in @($Group.Value)) {
        $FullPath = Join-Path $ProjectRoot ([string]$RelativePath)
        if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
            $Missing += "$($Group.Name): $FullPath"
        }
    }
}
if ($Missing.Count -gt 0) {
    $Missing | ForEach-Object { Write-Host "MISSING $_" -ForegroundColor Red }
    throw "Official live-trading file check failed with $($Missing.Count) missing files."
}
Write-Host "Official code/config files: OK" -ForegroundColor Green

foreach ($PathCheck in @(
    @{Label = "Python"; Path = [string]$Settings.Python; Directory = $false},
    @{Label = "Market database"; Path = [string]$Settings.MarketDatabase; Directory = $false},
    @{Label = "Risk database"; Path = [string]$Settings.RiskDatabase; Directory = $false},
    @{Label = "Weekly market folder"; Path = [string]$Settings.SourceDir; Directory = $true},
    @{Label = "Risk raw-data folder"; Path = [string]$Settings.RiskDataRawRoot; Directory = $true}
)) {
    Assert-OfficialLivePath `
        -Path $PathCheck.Path `
        -Label $PathCheck.Label `
        -Directory:$PathCheck.Directory
}

foreach ($AccountId in @($Settings.Accounts)) {
    $Positions = Join-Path $ProjectRoot "data\input\accounts\$AccountId\positions.csv"
    Assert-OfficialLivePath -Path $Positions -Label "Positions file for $AccountId"
}
Write-Host "Local databases, raw folders and account files: OK" -ForegroundColor Green

Push-Location $ProjectRoot
try {
    $MarketStatusText = (
        & ([string]$Settings.Python) database_status.py `
            --database ([string]$Settings.MarketDatabase) | Out-String
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Market database status failed with exit code $LASTEXITCODE."
    }
    $RiskStatusText = (
        & ([string]$Settings.Python) risk_model_status.py `
            --database ([string]$Settings.RiskDatabase) | Out-String
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Risk database status failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$MarketStatus = $MarketStatusText | ConvertFrom-Json
$RiskStatus = $RiskStatusText | ConvertFrom-Json
$RiskDate = [string]$RiskStatus.model_progress.last_model_date
$AlphaDate = [string]$RiskStatus.v31_alpha.max_date
$IsCurrent = (
    $RiskDate -eq [string]$MarketStatus.max_date -and
    $AlphaDate -eq [string]$MarketStatus.max_date
)

[pscustomobject]@{
    MarketDate = [string]$MarketStatus.max_date
    RiskModelDate = $RiskDate
    V31AlphaDate = $AlphaDate
    RiskBuildErrors = [int]$RiskStatus.model_progress.errors
    AllCurrent = $IsCurrent
} | Format-List

if (-not $IsCurrent) {
    Write-Warning "Risk model or V3.1 Alpha is behind the market database. Run update_market_and_risk.ps1 or an account workflow."
}
