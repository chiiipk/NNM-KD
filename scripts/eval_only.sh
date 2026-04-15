#!/usr/bin/env bash
# ============================================================
# eval_only.sh — evaluate saved NNM-KD v3 checkpoints
#
# Usage:
#   bash scripts/eval_only.sh                         # default save_dir
#   bash scripts/eval_only.sh --save-dir /path/ckpts  # custom dir
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ -d "${VENV_DIR}" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:=}"
export HF_TOKEN

cd "${SCRIPT_DIR}"
python3 train.py --eval-only "$@"
