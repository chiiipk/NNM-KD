#!/usr/bin/env bash
# ============================================================
# run.sh — NNM-KD v3 server launch script
#
# Usage:
#   bash run.sh                        # full train + eval
#   bash run.sh --eval-only            # eval checkpoints only
#   bash run.sh --epochs 3             # override epoch count
#   bash run.sh --max-batches 2000     # quick smoke-test run
#   bash run.sh --save-dir /path/out   # custom output dir
#
# All flags after 'run.sh' are forwarded directly to train.py.
# ============================================================

set -euo pipefail

# ── 0. Paths ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# ── 1. HuggingFace token ─────────────────────────────────────
: "${HF_TOKEN:=}"
export HF_TOKEN

# ── 2. CUDA / env settings ───────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Uncomment to pin specific GPUs:
# export CUDA_VISIBLE_DEVICES=0,1

echo "======================================================"
echo "  NNM-KD v3 — launch"
echo "  Timestamp : ${TIMESTAMP}"
echo "  Log file  : ${LOG_FILE}"
echo "  Working   : ${SCRIPT_DIR}"
echo "  CUDA devs : ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================================"

# ── 3. Python / venv detection ───────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    echo "[setup] Activating virtual environment: ${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif command -v conda &>/dev/null && conda env list | grep -q "nnm_kd"; then
    echo "[setup] Activating conda env: nnm_kd"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate nnm_kd
else
    echo "[setup] No venv found — using system Python: $(which python3)"
fi

PYTHON="${PYTHON:-python3}"
echo "[setup] Python: $($PYTHON --version)"

# ── 4. Dependency install (idempotent) ───────────────────────
echo "[setup] Installing / verifying requirements..."
# $PYTHON -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"

# Verify GPU availability
$PYTHON - <<'EOF'
import torch, sys
n = torch.cuda.device_count()
if n < 2:
    print(f"[WARNING] Expected ≥2 GPUs, found {n}. Teacher/student split may fail.")
else:
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_memory//1024**3} GB)")
EOF

# ── 5. Launch training ───────────────────────────────────────
echo ""
echo "[train] Starting NNM-KD v3 — output to ${LOG_FILE}"
echo "        Args forwarded to train.py: $*"
echo ""

cd "${SCRIPT_DIR}"

$PYTHON train.py "$@" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "======================================================"
    echo "  Training complete. Logs → ${LOG_FILE}"
    echo "======================================================"
else
    echo "======================================================"
    echo "  [ERROR] Training exited with code ${EXIT_CODE}."
    echo "  Check logs: ${LOG_FILE}"
    echo "======================================================"
    exit ${EXIT_CODE}
fi
