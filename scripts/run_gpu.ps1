# Run the desktop app with YOLO explicitly on the first CUDA GPU.
# The environment override is scoped to this PowerShell process and restored on exit.
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gpuPython = Join-Path $projectRoot '.venv-gpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $gpuPython)) {
    throw 'Chưa có môi trường .venv-gpu. Xem docs/ai-stability-performance-plan.md.'
}
$previousDevice = [Environment]::GetEnvironmentVariable('PROCTORING_AI_DEVICE', 'Process')
try {
    [Environment]::SetEnvironmentVariable('PROCTORING_AI_DEVICE', '0', 'Process')
    Push-Location $projectRoot
    & $gpuPython apps/desktop/main.py
    if ($LASTEXITCODE -ne 0) {
        throw "Ứng dụng GPU dừng với mã $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    [Environment]::SetEnvironmentVariable('PROCTORING_AI_DEVICE', $previousDevice, 'Process')
}
