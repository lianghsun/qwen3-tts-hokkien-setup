#!/bin/bash
set -e

echo "=== Qwen3-TTS Hokkien TTS Setup ==="

# ── 0. uv ──────────────────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[0] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 1. Python 3.12 venv ────────────────────────────────────────────────────────
echo "[1/3] Creating Python 3.12 venv..."
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

# ── 2. 安裝依賴 ────────────────────────────────────────────────────────────────
echo "[2/3] Installing dependencies..."
uv pip install --index-strategy unsafe-best-match \
    "qwen-tts" \
    "soundfile" \
    "huggingface_hub" \
    "pandas" \
    "pyarrow" \
    "datasets" \
    "rich"

# flash-attn 可減少 VRAM 用量（選裝，編譯較久）
# uv pip install flash-attn --no-build-isolation

echo ""
echo "=== Setup complete! ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Run:            python synthesize_audio.py --src-dir ./tw-hokkien-seed-text"
