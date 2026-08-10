function Get-OfficialLiveSettings {
    param(
        [string]$Path = (Join-Path $PSScriptRoot "settings.local.json")
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw (
            "Local settings file not found: $Path. " +
            "Create it from settings.example.json before running live workflows."
        )
    }
    $Settings = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    $Required = @(
        "ProjectRoot",
        "Python",
        "MarketDatabase",
        "RiskDatabase",
        "SourceDir",
        "RiskDataRawRoot",
        "RiskModelConfig",
        "CapitalStrategyMap",
        "OutputDir",
        "ForwardCutoffDate",
        "RiskModelWorkers",
        "Accounts"
    )
    foreach ($Key in $Required) {
        if ($Settings.PSObject.Properties.Name -notcontains $Key) {
            throw "Missing setting '$Key' in $Path."
        }
    }
    return $Settings
}


function Assert-OfficialLivePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [switch]$Directory
    )

    $PathType = if ($Directory) { "Container" } else { "Leaf" }
    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) {
        throw "$Label does not exist: $Path"
    }
}
