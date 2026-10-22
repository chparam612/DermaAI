# ==============================================================================
# SkinCancerTracker Phase 6-lite Demo Backend Runner
# Launches the FastAPI inference service on 127.0.0.1:8000
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  SkinCancerTracker Inference Backend Launcher" -ForegroundColor Cyan
Write-Host "  Decision-Support / Research Prototype (Phase 6-lite)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Check Python availability
try {
    $pythonCmd = (Get-Command py -ErrorAction SilentlyContinue) ? "py" : "python"
    & $pythonCmd --version
} catch {
    Write-Error "Python interpreter not found. Please install Python 3.10+."
    exit 1
}

# 2. Check slim checkpoint existence
$ckptPath = Join-Path $PSScriptRoot "checkpoints\final\demo_model.pt"
if (-not (Test-Path $ckptPath)) {
    Write-Warning "Demo checkpoint not found at: $ckptPath"
    Write-Host "Generating slim demo checkpoint from Exp-2c seed 42..." -ForegroundColor Yellow
    & $pythonCmd scripts\gate1_export_and_parity.py
    if (-not (Test-Path $ckptPath)) {
        Write-Error "Failed to generate demo checkpoint. Please verify checkpoints/exp2c_panderm_uniform_seed42/best_model.pt exists."
        exit 1
    }
}

Write-Host "`nCheckpoint verified: $ckptPath" -ForegroundColor Green
Write-Host "Starting Uvicorn server on http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "CORS origins allowed: http://localhost:5173, http://127.0.0.1:5173" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop the server.`n" -ForegroundColor Yellow

# 3. Launch Uvicorn
& $pythonCmd -m uvicorn api.app:app --host 127.0.0.1 --port 8000
