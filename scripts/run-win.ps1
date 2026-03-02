$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvVibemouse = Join-Path $repoRoot ".venv\Scripts\vibemouse.exe"

if (-not (Test-Path $venvPython)) {
    throw "Missing .venv. Please initialize environment first."
}

$env:VIBEMOUSE_BACKEND = if ($env:VIBEMOUSE_BACKEND) { $env:VIBEMOUSE_BACKEND } else { "funasr_onnx" }
$env:VIBEMOUSE_DEVICE = if ($env:VIBEMOUSE_DEVICE) { $env:VIBEMOUSE_DEVICE } else { "cpu" }
$env:VIBEMOUSE_PREWARM_ON_START = if ($env:VIBEMOUSE_PREWARM_ON_START) { $env:VIBEMOUSE_PREWARM_ON_START } else { "true" }

& $venvVibemouse run
