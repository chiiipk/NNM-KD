#!/usr/bin/env bash
# ============================================================
# setup_env.sh — one-time environment setup
#
# Creates .venv in the project root and installs all deps.
# Run once before first training:
#   bash scripts/setup_env.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "[setup] Project root : ${SCRIPT_DIR}"
echo "[setup] Venv target  : ${VENV_DIR}"

# ── Python version check ─────────────────────────────────────
PYTHON="${PYTHON:-python3}"
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[setup] Python version: ${PY_VER}"

# ── Create venv ──────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] Creating virtual environment..."
    $PYTHON -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "[setup] Activated: $(which python)"

# ── Upgrade pip ──────────────────────────────────────────────
pip install --upgrade pip wheel setuptools -q

# ── Install PyTorch (CUDA 12.1 — adjust index-url if needed) ─
echo "[setup] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 -q

# ── Install project requirements ─────────────────────────────
echo "[setup] Installing project requirements..."
pip install -r "${SCRIPT_DIR}/requirements.txt" -q

echo ""
echo "[setup] Done. Activate with:"
echo "        source ${VENV_DIR}/bin/activate"
