param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$StartDate = "2021-01-05",
    [string]$EndDate = "",
    [double]$InitialCash = 20000.0,
    [string]$OutputRoot = "outputs/validation_csmar_20k_commission_riskalign",
    [switch]$EnableCheckpoints,
    [int]$CheckpointEveryNDays = 5,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EndDate)) {
    $EndDate = (& $Python database_status.py --database $Database --field max_date).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Database status check failed with exit code $LASTEXITCODE."
    }
}

$Variants = @(
    @{
        Name = "official_baseline_20k_min5"
        Config = "config/v2h4_strategy.json"
    },
    @{
        Name = "small_20k_balanced_min5"
        Config = "config/v2h4_small_account_20k_balanced.json"
    },
    @{
        Name = "small_20k_concentrated_min5"
        Config = "config/v2h4_small_account_20k_concentrated.json"
    }
)

Write-Host "Database: $Database"
Write-Host "Date range: $StartDate to $EndDate"
Write-Host "Initial cash: $InitialCash"
Write-Host "Broker commission: 0.03%, minimum CNY 5 per order (from each strategy config)"
Write-Host "Output root: $OutputRoot"

foreach ($Variant in $Variants) {
    Write-Host ("Variant: {0} <- {1}" -f $Variant.Name, $Variant.Config)
}

if ($PlanOnly) {
    Write-Host "Plan-only check passed; no backtest was started."
    exit 0
}

if ($Resume) {
    $EnableCheckpoints = $true
}

function Test-StageComplete {
    param([string]$Directory)
    $Summary = Get-ChildItem -LiteralPath $Directory -Filter "*_summary.json" -File -ErrorAction SilentlyContinue
    $Workbook = Get-ChildItem -LiteralPath $Directory -Filter "*.xlsx" -File -ErrorAction SilentlyContinue
    return ($null -ne $Summary -and $null -ne $Workbook)
}

foreach ($Variant in $Variants) {
    $VariantDir = Join-Path $OutputRoot $Variant.Name
    if ($Resume -and (Test-StageComplete $VariantDir)) {
        Write-Host ("{0} is complete; skipping it." -f $Variant.Name)
        continue
    }

    $CheckpointArgs = @()
    if ($EnableCheckpoints) {
        $CheckpointArgs = @(
            "--checkpoint-file", (Join-Path $VariantDir "v2h_checkpoint.json.gz"),
            "--checkpoint-every-n-days", "$CheckpointEveryNDays",
            "--resume"
        )
    }

    & $Python factor_rank_backtest_v2h.py `
        --strategy-config $Variant.Config `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --initial-cash $InitialCash `
        --output-dir $VariantDir `
        @CheckpointArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Backtest {0} failed with exit code {1}." -f $Variant.Name, $LASTEXITCODE)
    }
}

& $Python compare_backtest_results.py `
    --root $OutputRoot `
    --holdout-start "2026-04-01" `
    --holdout-end $EndDate `
    --output (Join-Path $OutputRoot "v2h4_20k_commission_comparison.xlsx")
if ($LASTEXITCODE -ne 0) {
    throw "Backtest comparison failed with exit code $LASTEXITCODE."
}
