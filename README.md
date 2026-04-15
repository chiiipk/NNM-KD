# NNM-KD

Distills **Qwen2.5-Math-1.5B-Instruct** (teacher) → **Qwen2.5-0.5B** (student) using a combination of:

- **Nuclear Norm Matching (NNM)** — aligns hidden-state geometry via polar-factor approximation
- **Token-level KL filtering** — S2T spurious-token mask (STAPO) + high-entropy selection
- **Top-K logit KD** — KL only over teacher's top-K tokens per position
- **Cosine temperature annealing** — T_max → T_min over training
- **On-policy distillation** — student self-generation → teacher scoring (Phase 2)
- **Difficulty-aware weighting** — JSD(early layer, final layer) per token
- **Teacher data filtering** — only supervise on samples the teacher solves confidently

---

## Project layout

```
nnm_kd_v3/
├── configs/
│   ├── __init__.py
│   └── default.py          ← all hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py          ← MetaMathQA builder + teacher filter
│   ├── models.py           ← teacher/student loaders + HiddenProjector
│   ├── nnm.py              ← Newton-Schulz, centroids, NNM loss
│   ├── losses.py           ← CE, all KL variants, difficulty weights
│   ├── utils.py            ← forward_with_hiddens, on-policy step
│   └── evaluate.py         ← GSM8K + MATH-500 evaluation
├── scripts/
│   ├── run.sh              ← main server launch script
│   ├── setup_env.sh        ← one-time venv + dependency install
│   └── eval_only.sh        ← evaluate existing checkpoints only
├── train.py                ← entry point
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. One-time setup

```bash
bash scripts/setup_env.sh
```

This creates `.venv/` and installs all dependencies (PyTorch CUDA 12.1 wheel + project deps).  
Adjust the `--index-url` in `setup_env.sh` if you're on a different CUDA version.

### 2. Train + eval

```bash
bash scripts/run.sh
```

### 3. Common CLI overrides

```bash
# Quick smoke test (2000 batches, 1 epoch)
bash scripts/run.sh --max-batches 2000

# Multi-epoch run with custom output dir
bash scripts/run.sh --epochs 3 --save-dir /data/ckpts/nnm_v3

# Eval only (no training)
bash scripts/eval_only.sh --save-dir /data/ckpts/nnm_v3
```

### 4. Direct Python invocation

```bash
source .venv/bin/activate
python train.py --epochs 2 --max-batches 5000 --save-dir ./outputs
python train.py --eval-only --save-dir ./outputs
```

---

## Hardware requirements

| Role    | Device  | Minimum VRAM |
|---------|---------|--------------|
| Teacher | cuda:1  | 8 GB (4-bit NF4 quantised) |
| Student | cuda:0  | 6 GB (float32 params + AMP fp16 fwd) |

Two GPUs required. Adjust `DEVICE_S` / `DEVICE_T` in `src/models.py` and `src/utils.py` for single-GPU setups (will require gradient checkpointing and sequential forward passes).

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_temp_max / min` | 3.0 / 1.0 | Cosine temperature schedule |
| `kl_weight` | 0.6 | KL loss weight (dominant) |
| `ce_weight` | 0.3 | CE loss weight |
| `lambda_nnm` | 0.10 | NNM loss weight |
| `top_k_logits` | 50 | Top-K logit KD |
| `high_ent_rho` | 0.3 | Fraction of high-entropy tokens for KL |
| `s2t_tau_p / h` | 0.01 / 0.5 | Spurious token thresholds |
| `K_centroids` | 128 | Number of running centroids per layer |
| `d_prime` | 256 | Random projection dimension |
| `grad_accum` | 16 | Gradient accumulation steps |
| `on_policy_interval` | 200 | Steps between on-policy KD rounds |

All parameters are in `configs/default.py`.

---

## Output structure

```
nnm_kd_v3_outputs/
├── epoch_1/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files ...
│   ├── projector.pt
│   └── detail_NNM-KD_v3_epoch_1.json   ← per-sample eval records
├── eval_results.json                    ← summary table
└── logs/
    └── train_YYYYMMDD_HHMMSS.log
```
