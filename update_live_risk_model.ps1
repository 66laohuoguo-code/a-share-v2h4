param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$MarketDatabase,
    [Parameter(Mandatory = $true)]
    [string]$RiskDatabase,
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,
    [string]$RiskModelConfig = "config/risk_model_v31_full_2019_20260717.json",
    [string]$RiskDataRawRoot = "data/raw/CSMAR raw data",
    [string]$TargetDate = "",
    [ValidateRange(1, 32)]
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Push-Location $ProjectRoot
try {
    foreach ($RequiredPath in @($MarketDatabase, $RiskDatabase, $SourceDir, $RiskModelConfig)) {
        if (-not (Test-Path -LiteralPath $RequiredPath)) {
            throw "Required live risk-model path does not exist: $RequiredPath"
        }
    }

    if ([string]::IsNullOrWhiteSpace($TargetDate)) {
        $TargetDate = (
            & $Python database_status.py --database $MarketDatabase --field max_date
        ).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Market database status check failed with exit code $LASTEXITCODE."
        }
    }
    if ($TargetDate -notmatch '^\d{4}-\d{2}-\d{2}$') {
        throw "TargetDate must use YYYY-MM-DD format: $TargetDate"
    }

    $RiskStatusText = (
        & $Python risk_model_status.py --database $RiskDatabase | Out-String
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Risk-model status check failed with exit code $LASTEXITCODE."
    }
    try {
        $RiskStatus = $RiskStatusText | ConvertFrom-Json
    }
    catch {
        throw "Risk-model status output was not valid JSON: $RiskStatusText"
    }
    $LastRiskDate = [string]$RiskStatus.model_progress.last_model_date

    if (
        [string]::IsNullOrWhiteSpace($LastRiskDate) -or
        [string]::CompareOrdinal($LastRiskDate, $TargetDate) -lt 0
    ) {
        Write-Host "Updating risk-model market-cap data through $TargetDate..."
        & $Python build_risk_model_data.py `
            --raw-root $RiskDataRawRoot `
            --risk-raw-root (Join-Path $RiskDataRawRoot "risk model") `
            --forward-market-root $SourceDir `
            --output $RiskDatabase `
            --start-date "2019-01-01" `
            --financial-start-date "2017-01-01" `
            --end-date $TargetDate `
            --stage market-cap `
            --overwrite-forward
        if ($LASTEXITCODE -ne 0) {
            throw "Risk-model market-cap update failed with exit code $LASTEXITCODE."
        }

        $RiskModelArgs = @(
            "build_weekly_risk_model.py",
            "--market-database", $MarketDatabase,
            "--risk-database", $RiskDatabase,
            "--config", $RiskModelConfig,
            "--end-date", $TargetDate,
            "--stage", "all",
            "--workers", [string]$Workers
        )
        if (-not [string]::IsNullOrWhiteSpace($LastRiskDate)) {
            $LastRiskDateValue = [DateTime]::ParseExact(
                $LastRiskDate,
                "yyyy-MM-dd",
                [Globalization.CultureInfo]::InvariantCulture
            )
            $RefreshFromDate = $LastRiskDateValue.AddDays(1).ToString("yyyy-MM-dd")
            $RiskModelArgs += @("--refresh-from-date", $RefreshFromDate)
        }
        Write-Host "Extending weekly risk model from $LastRiskDate through $TargetDate..."
        & $Python @RiskModelArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Weekly risk-model update failed with exit code $LASTEXITCODE."
        }
    }
    else {
        Write-Host "Weekly risk model is already current through $LastRiskDate."
    }

    & $Python build_v31_alpha_cache.py --database $RiskDatabase
    if ($LASTEXITCODE -ne 0) {
        throw "V3.1 alpha-cache update failed with exit code $LASTEXITCODE."
    }

    $UpdatedRiskStatusText = (
        & $Python risk_model_status.py --database $RiskDatabase | Out-String
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Updated risk-model status check failed with exit code $LASTEXITCODE."
    }
    $UpdatedRiskStatus = $UpdatedRiskStatusText | ConvertFrom-Json
    $UpdatedRiskDate = [string]$UpdatedRiskStatus.model_progress.last_model_date
    if (
        [string]::IsNullOrWhiteSpace($UpdatedRiskDate) -or
        [string]::CompareOrdinal($UpdatedRiskDate, $TargetDate) -lt 0
    ) {
        throw (
            "Risk model is still stale after the update. " +
            "Market date: $TargetDate; risk-model date: $UpdatedRiskDate."
        )
    }
    Write-Host "Risk model and V3.1 alpha cache are current through $UpdatedRiskDate."
}
finally {
    Pop-Location
}
