$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Python launcher not found. Installing Python 3.11 via winget..."
        & winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --silent
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python 3.11 and rerun installer."
}

Write-Host "[1/5] Creating virtual environment (.venv)..."
if (-not (Test-Path $venvDir)) {
    & py -3.11 -m venv $venvDir
}

Write-Host "[2/5] Upgrading pip/setuptools/wheel..."
& $venvPython -m pip install -U pip setuptools wheel

Write-Host "[3/5] Installing VibeMouse..."
& $venvPython -m pip install -e $repoRoot

Write-Host "[4/6] Installing selftest dependency (edge-tts)..."
& $venvPython -m pip install edge-tts

Write-Host "[5/6] Running doctor check..."
& $venvPython -m vibemouse.main doctor

Write-Host "[6/6] Running selftest (mic + edge-tts + ASR)..."
& $venvPython -m vibemouse.main selftest

Write-Host "Done."
Write-Host "Start with: .\scripts\run-win.ps1"
