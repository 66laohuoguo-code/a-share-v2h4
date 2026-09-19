# Safe public-release push wrapper: audit first, then push.
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\push_public_release.ps1 `
#       -Message "Add factor evaluation toolkit" -Files a.py,b.md -DryRun
#
#   -Files a.py,b.md        stage ONLY these files (never `git add .`)
#   -DryRun                 stage + audit only; do not commit or push
#   -Forbid "D:\my\data","myuser"   extra substrings that must not appear in the payload
#   -GitPath <path>         force a specific git.exe
#
# Why not `git add .`: an enumerated .gitignore always lags behind new files.
# The moment you add a research script and forget one line, `git add .` ships it.
# An explicit file list plus a pre-push audit is the only reliable pattern.
#
# ASCII-only on purpose: Windows PowerShell reads BOM-less files as ANSI, which
# silently corrupts non-ASCII comments and can break parsing.

param(
    [Parameter(Mandatory = $true)][string]$Message,
    [string[]]$Files = @(),
    [switch]$DryRun,
    [string[]]$Forbid = @(),
    [string]$GitPath = ''
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
# `powershell -File` passes a comma list as ONE string, so split it back into an array.
$Files = @($Files | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' } | ForEach-Object { $_.Trim() })

function Resolve-Git {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "git.exe not found: $Explicit" }
        return (Resolve-Path $Explicit).Path
    }
    $onPath = Get-Command git -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    # Editors and agent runtimes often ship a private git that is not on PATH.
    $candidates = @(
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'),
        (Join-Path $env:APPDATA 'kimi-desktop\daimon-bundle\runtime\git\cmd\git.exe'),
        (Join-Path $env:LOCALAPPDATA 'GitHubDesktop'),
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Git\cmd\git.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe'),
        'C:\msys64\usr\bin\git.exe'
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate -PathType Leaf)) {
            return (Resolve-Path $candidate).Path
        }
        if ($candidate -and (Test-Path $candidate -PathType Container)) {
            $nested = Get-ChildItem $candidate -Filter git.exe -Recurse -Depth 4 -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($nested) { return $nested.FullName }
        }
    }
    throw ("git.exe not found. Install it, or pass -GitPath.`n" +
           "  winget install --id Git.Git -e --source winget")
}

function Resolve-Python {
    foreach ($candidate in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
        $full = Join-Path $repo $candidate
        if (Test-Path $full) { return $full }
    }
    return 'python'
}

$git = Resolve-Git -Explicit $GitPath
$python = Resolve-Python
Push-Location $repo
try {
    if (-not (Test-Path (Join-Path $repo '.git'))) { throw "Not a git repository: $repo" }

    Write-Host "== repo  : $repo" -ForegroundColor Cyan
    Write-Host "== git   : $git" -ForegroundColor Cyan
    Write-Host "== python: $python" -ForegroundColor Cyan
    & $git --version
    & $git remote -v

    Write-Host "`n== 1/5 working tree status" -ForegroundColor Cyan
    & $git status --short

    Write-Host "`n== 2/5 staging (explicit list, never 'git add .')" -ForegroundColor Cyan
    if ($Files.Count -gt 0) {
        foreach ($file in $Files) {
            if (-not (Test-Path (Join-Path $repo $file))) { throw "Missing file: $file" }
            & $git add -- $file
            Write-Host "   + $file"
        }
    } else {
        Write-Host "   no -Files given; auditing whatever is already staged" -ForegroundColor Yellow
    }

    Write-Host "`n== 3/5 staged file list (review this by eye)" -ForegroundColor Cyan
    $staged = & $git diff --cached --name-only
    if (-not $staged) { Write-Host "   (empty)" -ForegroundColor Yellow }
    $staged | ForEach-Object { Write-Host "   $_" }

    Write-Host "`n== 4/5 leak audit" -ForegroundColor Cyan
    $listFile = Join-Path $repo '.git\public_release_staged.txt'
    $staged | Set-Content -Path $listFile -Encoding utf8
    $auditArgs = @('tools\audit_public_release.py', '--root', '.', '--from-file', $listFile)
    foreach ($fragment in $Forbid) { $auditArgs += @('--forbid', $fragment) }
    & $python @auditArgs
    $auditExit = $LASTEXITCODE
    Remove-Item $listFile -ErrorAction SilentlyContinue
    if ($auditExit -ne 0) {
        throw "Audit failed; aborting. Fix .gitignore or run 'git restore --staged <file>' and retry."
    }

    if ($DryRun) {
        Write-Host "`n== DryRun: nothing committed or pushed." -ForegroundColor Yellow
        Write-Host "   To unstage everything: & '$git' restore --staged ." -ForegroundColor Yellow
        return
    }

    Write-Host "`n== 5/5 commit and push" -ForegroundColor Cyan
    & $git commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw "commit failed" }
    & $git push origin main
    if ($LASTEXITCODE -ne 0) { throw "push failed" }

    Write-Host "`nDone. Verify on the GitHub page that there is:" -ForegroundColor Green
    Write-Host "  * no *.sqlite / outputs/ / research notes containing formulas"
    Write-Host "  * no local absolute paths, usernames, or sk- style secrets"
}
finally {
    Pop-Location
}

exit 0
