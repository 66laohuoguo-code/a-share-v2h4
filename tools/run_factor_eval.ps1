# factor_eval runner (Windows)
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\run_factor_eval.ps1 -Task demo
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\run_factor_eval.ps1 -Task tests
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\run_factor_eval.ps1 -Task panel -Panel .\my_panel.parquet
#
# No absolute paths. The interpreter is probed as .venv -> venv -> python.
# ASCII-only on purpose: Windows PowerShell reads BOM-less files as ANSI.

param(
    [ValidateSet('demo', 'tests', 'panel', 'gates', 'reversal', 'decoupling')]
    [string]$Task = 'demo',
    [string]$Panel = '',
    [string]$Factors = '',
    [string]$Controls = 'size,momentum,liquidity,value,earnings_yield,growth,beta,residual_vol',
    [string]$Reversal = 'past_return_20',
    [string]$EraCol = 'era'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$python = 'python'
foreach ($candidate in @('..\.venv\Scripts\python.exe', '..\venv\Scripts\python.exe')) {
    $full = Join-Path $PSScriptRoot $candidate
    if (Test-Path $full) { $python = (Resolve-Path $full).Path; break }
}

Write-Host "python: $python"
Write-Host "task  : $Task"

switch ($Task) {
    'tests' {
        & $python -m unittest tests.test_factor_eval -v
    }
    'demo' {
        & $python -m factor_eval.cli demo
    }
    default {
        if (-not $Panel) { throw "-Task $Task requires -Panel <path>" }
        $arguments = @('-m', 'factor_eval.cli', $Task,
                       '--panel', $Panel,
                       '--controls', $Controls,
                       '--reversal', $Reversal,
                       '--era-col', $EraCol)
        if ($Factors) { $arguments += @('--factors') + ($Factors -split ',') }
        & $python @arguments
    }
}

exit $LASTEXITCODE
