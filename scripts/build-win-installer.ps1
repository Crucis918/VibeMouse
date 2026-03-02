$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvPip = Join-Path $repoRoot ".venv\Scripts\pip.exe"
$isccPathCandidates = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)

$isccPath = $null
foreach ($candidate in $isccPathCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $isccPath = $candidate
        break
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Missing .venv. Run .\\scripts\\install-win.ps1 first."
}

Write-Host "[1/5] Installing build tools..."
& $venvPip install -U pyinstaller

Write-Host "[2/5] Building one-file EXE..."
& $venvPython -m PyInstaller --noconfirm --clean --noconsole --name VibeMouse --onefile --collect-all funasr_onnx --hidden-import pynput.keyboard._win32 --hidden-import pynput.mouse._win32 --hidden-import sounddevice --hidden-import soundfile --hidden-import pystray._win32 --hidden-import PIL.Image --hidden-import PIL.ImageDraw scripts/launch.py

Write-Host "[3/5] Preparing installer payload..."
$payloadDir = Join-Path $repoRoot "dist\VibeMouse"
if (Test-Path $payloadDir) {
    Remove-Item -Path $payloadDir -Recurse -Force
}
New-Item -ItemType Directory -Path $payloadDir | Out-Null
Copy-Item -Path (Join-Path $repoRoot "dist\VibeMouse.exe") -Destination (Join-Path $payloadDir "VibeMouse.exe") -Force

Write-Host "[4/5] Checking Inno Setup..."
if (-not (Test-Path $isccPath)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Inno Setup not found. Installing via winget..."
        & winget install -e --id JRSoftware.InnoSetup --source winget --accept-package-agreements --accept-source-agreements --silent
    }
}

if (-not $isccPath -or -not (Test-Path $isccPath)) {
    foreach ($candidate in $isccPathCandidates) {
        if ($candidate -and (Test-Path $candidate)) {
            $isccPath = $candidate
            break
        }
    }
}

if (-not (Test-Path $isccPath)) {
    throw "Inno Setup missing. Install Inno Setup 6 and rerun build script."
}

Write-Host "[5/5] Building installer (.exe)..."
& $isccPath (Join-Path $repoRoot "packaging\windows\VibeMouse.iss")

Write-Host "Done: dist\\VibeMouse-Setup.exe"
