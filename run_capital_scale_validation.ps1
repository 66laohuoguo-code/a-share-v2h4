param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "",
    [double[]]$Capitals = @(
        20000,
        50000,
        100000,
        200000,
        500000,
        1000000,
        5000000,
        10000000
    ),
    [string]$OutputRoot = "outputs/validation_csmar_capital_scale",
    [string]$FeatureCache = "",
    [string]$HoldoutStart = "2026-04-01",
    [int]$CheckpointEveryNDays = 5,
    [ValidateRange(1, 8)]
    [int]$ShardCount = 1,
    [ValidateRange(0, 7)]
    [int]$ShardIndex = 0,
    [switch]$IncludeCapacityStress,
    [switch]$Resume,
    [switch]$PlanOnly,
    [switch]$AnalyzeOnly,
    [switch]$BuildFeatureCacheOnly,
    [switch]$SkipFinalAnalysis
)

$ErrorActionPreference = "Stop"

$Variants = @{
    stock12_sparse = "config/v2h4_small_account_20k_best_20260724_sparse_weekly.json"
    stock20 = "config/v2h4_strategy_10w_20stock_concentrated.json"
    stock25 = "config/v2h4_strategy_10w_25stock_balanced.json"
    stock40 = "config/v2h4_strategy_10w_weekly_causal.json"
    stock60 = "config/v2h4_strategy_60stock_capacity.json"
    stock80 = "config/v2h4_strategy_80stock_capacity.json"
}

function Get-VariantsForCapital {
    param([double]$Capital)
    if ($Capital -le 50000) {
        return @("stock12_sparse", "stock20", "stock25", "stock40")
    }
    if ($Capital -le 200000) {
        return @("stock12_sparse", "stock20", "stock25", "stock40", "stock60")
    }
    return @("stock20", "stock25", "stock40", "stock60", "stock80")
}

function Test-StageComplete {
    param([string]$Directory)
    $Summary = @(
        Get-ChildItem -LiteralPath $Directory -Filter "*_summary.json" -File -ErrorAction SilentlyContinue
    )
    $Workbook = @(
        Get-ChildItem -LiteralPath $Directory -Filter "factor_rank*.xlsx" -File -ErrorAction SilentlyContinue
    )
    return ($Summary.Count -gt 0 -and $Workbook.Count -gt 0)
}

function Invoke-Analysis {
    & $Python capital_scale_analysis.py `
        --root $OutputRoot `
        --holdout-start $HoldoutStart `
        --holdout-end $EndDate `
        --output (Join-Path $OutputRoot "v2h4_capital_scale_comparison.xlsx")
    if ($LASTEXITCODE -ne 0) {
        throw "Capital-scale comparison failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Database -PathType Leaf)) {
    throw "Database not found: $Database"
}

if ([string]::IsNullOrWhiteSpace($EndDate)) {
    $EndDate = (& $Python database_status.py --database $Database --field max_date).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Database status check failed with exit code $LASTEXITCODE."
    }
}

$ReferenceFeatureConfig = $null
foreach ($VariantName in $Variants.Keys) {
    $ConfigPath = $Variants[$VariantName]
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "Strategy configuration not found: $ConfigPath"
    }
    $Config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([double]$Config.broker_commission_rate -ne 0.0003) {
        throw "$ConfigPath must use broker_commission_rate=0.0003."
    }
    if ([double]$Config.broker_minimum_commission -ne 5.0) {
        throw "$ConfigPath must use broker_minimum_commission=5.0."
    }
    if ([string]$Config.rebalance_schedule -ne "week_end") {
        throw "$ConfigPath must use rebalance_schedule=week_end."
    }
    if ([string]$Config.risk_rebalance_schedule -ne "scheduled_only") {
        throw "$ConfigPath must use risk_rebalance_schedule=scheduled_only."
    }
    $FeatureSignature = @(
        $Config.feature_history_days,
        $Config.min_history_days,
        $Config.min_avg_amount,
        $Config.min_market_cap_quantile,
        $Config.market_cap_proxy_window,
        $Config.disable_industry_neutral_factors
    ) -join "|"
    if ($null -eq $ReferenceFeatureConfig) {
        $ReferenceFeatureConfig = $FeatureSignature
    } elseif ($FeatureSignature -ne $ReferenceFeatureConfig) {
        throw "$ConfigPath does not share the common feature-cache parameters."
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if ([string]::IsNullOrWhiteSpace($FeatureCache)) {
    $FeatureCache = Join-Path $OutputRoot "shared_feature_snapshots.sqlite"
}
$FeatureCacheMarker = "$FeatureCache.complete.json"

$FullPlan = @()
foreach ($Capital in ($Capitals | Sort-Object -Unique)) {
    if ($Capital -le 0) {
        throw "Every capital value must be positive."
    }
    $CapitalLabel = "capital_{0:D8}" -f [int64]$Capital
    foreach ($VariantName in (Get-VariantsForCapital $Capital)) {
        $FullPlan += [pscustomobject]@{
            capital = [double]$Capital
            capital_label = $CapitalLabel
            scenario = "base"
            variant = $VariantName
            config = $Variants[$VariantName]
            slippage_bps = 5.0
            max_participation_rate = 0.05
        }
    }
    if ($IncludeCapacityStress -and $Capital -ge 5000000) {
        foreach ($VariantName in (Get-VariantsForCapital $Capital)) {
            $FullPlan += [pscustomobject]@{
                capital = [double]$Capital
                capital_label = $CapitalLabel
                scenario = "stress_slip10_part2"
                variant = $VariantName
                config = $Variants[$VariantName]
                slippage_bps = 10.0
                max_participation_rate = 0.02
            }
            $FullPlan += [pscustomobject]@{
                capital = [double]$Capital
                capital_label = $CapitalLabel
                scenario = "stress_slip20_part1"
                variant = $VariantName
                config = $Variants[$VariantName]
                slippage_bps = 20.0
                max_participation_rate = 0.01
            }
        }
    }
}

if ($ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be smaller than ShardCount."
}
$Plan = @()
for ($Index = 0; $Index -lt $FullPlan.Count; $Index += 1) {
    if (($Index % $ShardCount) -eq $ShardIndex) {
        $Plan += $FullPlan[$Index]
    }
}

$FullPlanPath = Join-Path $OutputRoot "capital_scale_plan.csv"
$FullPlan | Export-Csv -LiteralPath $FullPlanPath -NoTypeInformation -Encoding UTF8
$PlanPath = Join-Path $OutputRoot (
    "capital_scale_plan_shard_{0}_of_{1}.csv" -f $ShardIndex, $ShardCount
)
$Plan | Export-Csv -LiteralPath $PlanPath -NoTypeInformation -Encoding UTF8

Write-Host "Database: $Database"
Write-Host "Date range: $StartDate to $EndDate"
Write-Host "Holdout: $HoldoutStart to $EndDate"
Write-Host "Broker commission: 0.03%, minimum CNY 5 per order"
Write-Host "Causality: prior available trading day decision, next trading session execution"
Write-Host "Shared feature cache: $FeatureCache"
Write-Host "Full planned runs: $($FullPlan.Count)"
Write-Host "This shard: $ShardIndex / $ShardCount ($($Plan.Count) runs)"
Write-Host "Full plan: $FullPlanPath"
Write-Host "Shard plan: $PlanPath"
Write-Host "Output root: $OutputRoot"

if ($PlanOnly) {
    $Plan | Format-Table capital, scenario, variant, slippage_bps, max_participation_rate -AutoSize
    Write-Host "Plan-only validation passed; no backtest was started."
    exit 0
}

if ($AnalyzeOnly) {
    Invoke-Analysis
    exit 0
}

if ($BuildFeatureCacheOnly) {
    & $Python factor_rank_backtest_v2h.py `
        --strategy-config $Variants["stock40"] `
        --database $Database `
        --start-date $StartDate `
        --end-date $EndDate `
        --initial-cash 100000 `
        --output-dir (Join-Path $OutputRoot "feature_cache_builder") `
        --feature-cache $FeatureCache `
        --build-feature-cache-only
    if ($LASTEXITCODE -ne 0) {
        throw "Feature-cache build failed with exit code $LASTEXITCODE."
    }
    @{
        database = $Database
        start_date = $StartDate
        end_date = $EndDate
        feature_cache = $FeatureCache
        completed_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $FeatureCacheMarker -Encoding UTF8
    Write-Host "Feature cache is complete: $FeatureCache"
    exit 0
}

if (-not (Test-Path -LiteralPath $FeatureCacheMarker -PathType Leaf)) {
    throw (
        "Shared feature cache is not complete. Run this script once with " +
        "-BuildFeatureCacheOnly before starting one or more backtest shards."
    )
}

$RunNumber = 0
foreach ($Run in $Plan) {
    $RunNumber += 1
    $RunDirectory = Join-Path $OutputRoot $Run.capital_label
    $RunDirectory = Join-Path $RunDirectory $Run.scenario
    $RunDirectory = Join-Path $RunDirectory $Run.variant

    if (Test-StageComplete $RunDirectory) {
        if ($Resume) {
            Write-Host "[$RunNumber/$($Plan.Count)] Complete; skipping $RunDirectory"
            continue
        }
        throw "Completed output already exists at $RunDirectory. Use -Resume to skip it."
    }

    New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null
    $Checkpoint = Join-Path $RunDirectory "v2h_checkpoint.json.gz"
    Write-Host (
        "[$RunNumber/$($Plan.Count)] Capital={0:N0}, scenario={1}, variant={2}" -f
        $Run.capital,
        $Run.scenario,
        $Run.variant
    )

    $BacktestArgs = @(
        "factor_rank_backtest_v2h.py",
        "--strategy-config", $Run.config,
        "--database", $Database,
        "--start-date", $StartDate,
        "--end-date", $EndDate,
        "--initial-cash", ([string]$Run.capital),
        "--slippage-bps", ([string]$Run.slippage_bps),
        "--max-participation-rate", ([string]$Run.max_participation_rate),
        "--output-dir", $RunDirectory,
        "--feature-cache", $FeatureCache,
        "--checkpoint-file", $Checkpoint,
        "--checkpoint-every-n-days", ([string]$CheckpointEveryNDays),
        "--resume"
    )
    & $Python @BacktestArgs
    $BacktestExitCode = $LASTEXITCODE
    if ($BacktestExitCode -eq 75) {
        Write-Host "Backtest paused after saving its checkpoint. Run the same command again with -Resume."
        exit 75
    }
    if ($BacktestExitCode -ne 0) {
        throw (
            "Backtest capital={0}, scenario={1}, variant={2} failed with exit code {3}." -f
            $Run.capital,
            $Run.scenario,
            $Run.variant,
            $BacktestExitCode
        )
    }
}

if (-not $SkipFinalAnalysis -and $ShardCount -eq 1) {
    Invoke-Analysis
}
if ($ShardCount -gt 1) {
    Write-Host "Shard $ShardIndex of $ShardCount is complete. Run -AnalyzeOnly after all shards finish."
} else {
    Write-Host "All planned capital-scale tests are complete."
}
