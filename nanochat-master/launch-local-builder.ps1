param(
    [int]$Port = 8000,
    [int]$RuntimePort = 8091,
    [string]$RuntimeModel = "",
    [ValidateSet("auto", "gpu", "cpu")]
    [string]$RuntimeDeviceStrategy = "auto",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating .venv with system site packages..."
    python -m venv .venv --system-site-packages
}

$missing = & $pythonExe -c "import importlib.util; mods=['fastapi','uvicorn']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('|'.join(missing))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect Python dependencies in .venv."
}

if ($missing) {
    throw "Missing Python packages in .venv: $missing. Install them locally first; this launcher does not fetch from the network."
}

$env:NANOCHAT_LOCAL_ONLY = "1"
$env:WANDB_MODE = "disabled"
$env:WANDB_DISABLED = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"

$serverArgs = @(
    "-m", "scripts.chat_web",
    "--port", $Port,
    "--runtime-autostart", "1",
    "--runtime-port", $RuntimePort,
    "--runtime-device-strategy", $RuntimeDeviceStrategy
)

if ($RuntimeModel) {
    $serverArgs += @("--runtime-model", $RuntimeModel)
}

Write-Host "Starting local builder at http://localhost:$Port"
Write-Host "Local runtime will use port $RuntimePort"

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($targetUrl)
        Start-Sleep -Seconds 3
        Start-Process $targetUrl
    } -ArgumentList "http://localhost:$Port" | Out-Null
}

& $pythonExe @serverArgs
