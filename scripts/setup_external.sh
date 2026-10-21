#!/usr/bin/env bash
# PARALLAX External Services & Tool Virtual Environments Setup (Bash)
# This script configures all external dependencies, venvs, and Ollama models
# without modifying the frozen project contracts or main virtual environment.

set -euo pipefail

echo "============================================================"
echo "PARALLAX: External Services & Environment Setup (Linux/macOS)"
echo "============================================================"

# 1. Check prerequisites
echo ""
echo "[1/5] Checking prerequisites (uv, git, ollama)..."
command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but not installed. Install uv: https://docs.astral.sh/uv/"; exit 1; }
command -v git >/dev/null 2>&1 || { echo >&2 "git is required but not installed."; exit 1; }
if ! command -v ollama >/dev/null 2>&1; then
    echo "Warning: ollama is not on PATH. Please ensure Ollama is installed from https://ollama.ai/"
else
    echo "  [+] Prerequisites found."
fi

# 2. Main workspace sync
echo ""
echo "[2/5] Synchronizing main PARALLAX workspace via uv..."
uv sync
echo "  [+] Main workspace synchronized."

# 3. Pull Ollama models
echo ""
echo "[3/5] Pulling Ollama judge models..."
if curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "  [+] Ollama service is reachable on 127.0.0.1:11434"
    echo "  Pulling qwen3:8b..."
    ollama pull qwen3:8b || true
    echo "  Pulling gemma3:4b..."
    ollama pull gemma3:4b || true
    echo "  [+] Judge models ready."
else
    echo "Warning: Ollama is not running on 127.0.0.1:11434. Start Ollama and run: 'ollama pull qwen3:8b; ollama pull gemma3:4b'"
fi

# 4. Create isolated external tool venvs
echo ""
echo "[4/5] Setting up isolated external tool venvs..."

# PyRIT venv
if [ ! -d ".venv_pyrit" ]; then
    echo "  Creating .venv_pyrit..."
    uv venv .venv_pyrit
    .venv_pyrit/bin/python -m pip install --upgrade pip pyrit
    echo "  [+] .venv_pyrit configured."
else
    echo "  [+] .venv_pyrit already exists."
fi

# LlamaFirewall venv
if [ ! -d ".venv_llamafirewall" ]; then
    echo "  Creating .venv_llamafirewall..."
    uv venv .venv_llamafirewall
    .venv_llamafirewall/bin/python -m pip install --upgrade pip "llamafirewall==1.0.3"
    echo "  [+] .venv_llamafirewall configured."
else
    echo "  [+] .venv_llamafirewall already exists."
fi

# Garak venv
if [ ! -d ".venv_garak" ]; then
    echo "  Creating .venv_garak..."
    uv venv .venv_garak
    .venv_garak/bin/python -m pip install --upgrade pip garak
    echo "  [+] .venv_garak configured."
else
    echo "  [+] .venv_garak already exists."
fi

# 5. Summary & Daemons Guide
echo ""
echo "[5/5] Background Detector Daemons"
echo "To run the full PromptForest 3-detector baseline live:"
echo "  1. PromptForest 2-detector server (Port 8765):"
echo "     promptforest serve --port 8765 --config configs/pf_two_detectors.yaml"
echo "  2. Deepset DeBERTa third detector service (Port 8766):"
echo "     uv run python scripts/detector_services/third_detector_service.py"
echo ""
echo "To verify the complete environment:"
echo "  uv run python scripts/preflight.py"
echo "  uv run python scripts/gate.py"

echo ""
echo "============================================================"
echo "PARALLAX setup complete! All external venvs and tools ready."
echo "============================================================"
