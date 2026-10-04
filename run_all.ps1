# End-to-end execution of the Elderly Bed-Activity Monitor pipeline (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Elderly Bed-Activity Monitor — End-to-End Execution" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[1/4] Running Unit Test Suite (pytest)..." -ForegroundColor Yellow
pytest -v

Write-Host "[2/4] Generating Architecture Diagram..." -ForegroundColor Yellow
python scripts/render_architecture.py

Write-Host "[3/4] Running Full Evaluation Suite across all videos..." -ForegroundColor Yellow
python scripts/run_eval_suite.py

Write-Host "[4/4] Verification of Key Outputs..." -ForegroundColor Yellow
Get-ChildItem outputs\gmdcsa_01\viterbi
Get-ChildItem outputs\gmdcsa_01\eval

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Elderly Bed-Activity Monitor — PoC Execution & Evaluation Suite Complete!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

