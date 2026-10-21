# PARALLAX External Services & Tool Virtual Environments Setup (PowerShell)
# This script configures all external dependencies, venvs, and Ollama models
# without modifying the frozen project contracts or main virtual environment.

[CmdletBinding()]
param(
    [switch]$SkipOllamaPull = $false,
    [switch]$SkipVenvs = $false,
    [switch]$StartDaemons = $false
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PARALLAX: External Services & Environment Setup (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check prerequisites
Write-Host "`n[1/5] Checking prerequisites (uv, git, ollama)..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed or not on PATH. Please install uv: https://docs.astral.sh/uv/"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is not installed or not on PATH."
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "ollama is not on PATH. Please ensure Ollama is installed from https://ollama.ai/"
} else {
    Write-Host "  [+] Prerequisites found." -ForegroundColor Green
}

# 2. Main workspace sync
Write-Host "`n[2/5] Synchronizing main PARALLAX workspace via uv..." -ForegroundColor Yellow
uv sync
Write-Host "  [+] Main workspace synchronized." -ForegroundColor Green

# 3. Pull Ollama models
if (-not $SkipOllamaPull) {
    Write-Host "`n[3/5] Pulling Ollama judge models..." -ForegroundColor Yellow
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 3
        Write-Host "  [+] Ollama service is reachable on 127.0.0.1:11434" -ForegroundColor Green
        
        Write-Host "  Pulling qwen3:8b..."
        ollama pull qwen3:8b
        Write-Host "  Pulling gemma3:4b..."
        ollama pull gemma3:4b
        Write-Host "  [+] Judge models ready." -ForegroundColor Green
    } catch {
        Write-Warning "Ollama is not running on 127.0.0.1:11434. Start Ollama and run: 'ollama pull qwen3:8b; ollama pull gemma3:4b'"
    }
} else {
    Write-Host "`n[3/5] Skipping Ollama model pulls (-SkipOllamaPull specified)." -ForegroundColor Gray
}

# 4. Create isolated external tool venvs
if (-not $SkipVenvs) {
    Write-Host "`n[4/5] Setting up isolated external tool venvs..." -ForegroundColor Yellow

    # PyRIT venv
    if (-not (Test-Path ".venv_pyrit")) {
        Write-Host "  Creating .venv_pyrit..."
        uv venv .venv_pyrit
        & .venv_pyrit\Scripts\python -m pip install --upgrade pip pyrit
        Write-Host "  [+] .venv_pyrit configured." -ForegroundColor Green
    } else {
        Write-Host "  [+] .venv_pyrit already exists." -ForegroundColor Gray
    }

    # LlamaFirewall venv
    if (-not (Test-Path ".venv_llamafirewall")) {
        Write-Host "  Creating .venv_llamafirewall..."
        uv venv .venv_llamafirewall
        & .venv_llamafirewall\Scripts\python -m pip install --upgrade pip "llamafirewall==1.0.3"
        Write-Host "  [+] .venv_llamafirewall configured." -ForegroundColor Green
    } else {
        Write-Host "  [+] .venv_llamafirewall already exists." -ForegroundColor Gray
    }

    # Garak venv
    if (-not (Test-Path ".venv_garak")) {
        Write-Host "  Creating .venv_garak..."
        uv venv .venv_garak
        & .venv_garak\Scripts\python -m pip install --upgrade pip garak
        Write-Host "  [+] .venv_garak configured." -ForegroundColor Green
    } else {
        Write-Host "  [+] .venv_garak already exists." -ForegroundColor Gray
    }
} else {
    Write-Host "`n[4/5] Skipping external tool venvs (-SkipVenvs specified)." -ForegroundColor Gray
}

# 5. Summary & Daemons Guide
Write-Host "`n[5/5] Background Detector Daemons" -ForegroundColor Yellow
Write-Host "To run the full PromptForest 3-detector baseline live:"
Write-Host "  1. PromptForest 2-detector server (Port 8765):"
Write-Host "     promptforest serve --port 8765 --config configs/pf_two_detectors.yaml" -ForegroundColor Cyan
Write-Host "  2. Deepset DeBERTa third detector service (Port 8766):"
Write-Host "     uv run python scripts/detector_services/third_detector_service.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "To verify the complete environment:"
Write-Host "  uv run python scripts/preflight.py" -ForegroundColor Cyan
Write-Host "  uv run python scripts/gate.py" -ForegroundColor Cyan

if ($StartDaemons) {
    Write-Host "`nStarting third detector daemon on 127.0.0.1:8766 in background..." -ForegroundColor Yellow
    Start-Process -FilePath "uv" -ArgumentList "run", "python", "scripts/detector_services/third_detector_service.py" -NoNewWindow
    Write-Host "  [+] Service launched on 127.0.0.1:8766." -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "PARALLAX setup complete! All external venvs and tools ready." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
