"""
train.py — NNM-KD v3 main training loop.

Usage:
    python train.py                      # train + eval (default config)
    python train.py --eval-only          # eval latest checkpoints only
    python train.py --config configs/custom.py   # override config file

CHANGELOG v3 vs v2:
  1. REMOVED Frobenius loss (too rigid for 3× capacity gap)
  2. TOKEN-LEVEL FILTERING  — S2T (STAPO) + high-entropy (Beyond 80/20)
  3. TOP-K LOGIT KD         — KL only on top-K teacher tokens
  4. TEMPERATURE ANNEALING  — cosine decay T_max → T_min
  5. ON-POLICY DISTILLATION — student generates → teacher scores (Phase 2)
  6. DIFFICULTY-AWARE WEIGHTING — JSD(early, final layer) per token
  7. TEACHER DATA FILTERING — only train on samples teacher solves
  8. 2-LAYER MLP PROJECTOR  — with GELU
  9. SHIFTED layer selection — 40%–85% (later layers for reasoning)
 10. LARGER centroids/projection (K=128, d'=384)
 11. INCREASED max_length (768), grad_accum (16), warmup_ratio (0.08)
"""

import os
import sys
import argparse
import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────
from configs import CFG, SYSTEM_PROMPT, QWEN_CHAT_TEMPLATE, SEED
from src.dataset  import build_metamath, filter_dataset_by_teacher
from src.models   import load_teacher, load_student, HiddenProjector
from src.nnm      import (
    RunningCentroids, make_R, layer_weight, select_mid_layers,
    nnm_loss_one_layer, measure_nuclear_norms,
    correct_teacher_hiddens, build_teacher_centroids,
)
from src.losses   import (
    ce_loss, compute_kl_loss, compute_difficulty_weights, get_temperature,
)
from src.utils    import forward_with_hiddens, on_policy_kl_step
from src.evaluate import compare_all

# ── determinism ──────────────────────────────────────────────
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE_S = torch.device("cuda:0")
DEVICE_T = torch.device("cuda:1")

os.makedirs(CFG["save_dir"], exist_ok=True)


# ════════════════════════════════════════════════════════════
#  NNM weight scheduler
# ════════════════════════════════════════════════════════════

def nnm_weight(step: int) -> float:
    return CFG["lambda_nnm"] * min(1.0, step / max(1, CFG["nnm_warmup"]))


# ════════════════════════════════════════════════════════════
#  Main training function
# ════════════════════════════════════════════════════════════

def train():
    print(f"\n{'='*70}")
    print("  NNM-KD v3: Qwen2.5-Math-1.5B-Instruct → Qwen2.5-0.5B")
    print(f"  Teacher → {DEVICE_T}  |  Student → {DEVICE_S}")
    print(f"  Loss: CE(w={CFG['ce_weight']}) + KL({CFG['kl_mode']}, w={CFG['kl_weight']}) + NNM")
    print(f"  v3: S2T mask · high-ent filter · top-K KD · temp annealing")
    print(f"      on-policy phase · difficulty weighting · 2-layer MLP projector")
    print(f"  Max batches/epoch: {CFG['max_train_batches']}")
    print(f"{'='*70}\n")

    # ── Tokeniser ────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(
        CFG["student_id"], trust_remote_code=True, padding_side="right",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = QWEN_CHAT_TEMPLATE

    # ── Dataset ──────────────────────────────────────────────
    print("Loading dataset...")
    train_ds = build_metamath(tok, max_len=CFG["max_length"], max_prompt_len=CFG["max_prompt_length"])

    # ── Models ───────────────────────────────────────────────
    print("Loading models...")
    teacher = load_teacher(CFG["teacher_id"])
    student = load_student(CFG["student_id"])

    # ── Teacher quality filter ───────────────────────────────
    if CFG["filter_by_teacher"]:
        train_ds = filter_dataset_by_teacher(
            teacher, tok, train_ds,
            max_samples=CFG["teacher_filter_max_samples"],
            batch_size=CFG["teacher_filter_batch_size"],
        )

    loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=2, persistent_workers=True, prefetch_factor=1,
        pin_memory=True, drop_last=True,
    )
    actual_batches = min(len(loader), CFG["max_train_batches"])
    print(f"  Train: {len(train_ds)} samples | will use {actual_batches} batches\n")

    # ── Architecture dims ────────────────────────────────────
    d_t  = teacher.config.hidden_size
    d_s  = student.config.hidden_size
    L_t  = teacher.config.num_hidden_layers
    L_s  = student.config.num_hidden_layers
    t_mid = select_mid_layers(L_t, CFG["n_mid_layers"])
    s_mid = select_mid_layers(L_s, CFG["n_mid_layers"])
    lw    = {s_lid: layer_weight(s_lid, L_s, CFG["sigma_layer"]) for s_lid in s_mid}

    print(f"  Teacher hidden={d_t}, layers={L_t}, mid={t_mid}")
    print(f"  Student hidden={d_s}, layers={L_s}, mid={s_mid}")
    print(f"  Layer weights: { {k: round(v, 4) for k, v in lw.items()} }\n")

    # ── NNM components ───────────────────────────────────────
    projector = HiddenProjector(d_t, d_s).to(DEVICE_S)
    R = make_R(d_s, CFG["d_prime"], DEVICE_S)

    print("Building teacher centroids...")
    t_cents = build_teacher_centroids(
        teacher, projector, loader, t_mid, s_mid,
        CFG["K_centroids"], d_s, CFG["eta_centroid"], CFG["T_dead"],
    )
    s_cents = {
        s_lid: RunningCentroids(CFG["K_centroids"], d_s,
                                CFG["eta_centroid"], CFG["T_dead"], DEVICE_S)
        for s_lid in s_mid
    }

    # ── Optimiser / scheduler ────────────────────────────────
    trainable    = list(student.parameters()) + list(projector.parameters())
    total_steps  = actual_batches * CFG["epochs"] // CFG["grad_accum"]
    warmup_steps = max(1, int(total_steps * CFG["warmup_ratio"]))
    optimizer    = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nTraining: {CFG['epochs']} epochs | {total_steps} opt-steps | {warmup_steps} warmup")
    print(f"  Temp annealing: {CFG['kl_temp_max']} → {CFG['kl_temp_min']}")
    print(f"  Token filtering: S2T(τp={CFG['s2t_tau_p']}, τh={CFG['s2t_tau_h']}), "
          f"HighEnt(ρ={CFG['high_ent_rho']}), TopK={CFG['top_k_logits']}")
    if CFG["do_on_policy"]:
        print(f"  On-policy KD: every {CFG['on_policy_interval']} steps, "
              f"weight={CFG['on_policy_weight']}")
    print()

    global_step = 0
    log_buf = {"total": 0., "ce": 0., "kl": 0., "nnm": 0., "on_pol": 0., "n": 0}

    # ════════════════════════════════════════════════════════
    #  Training loop
    # ════════════════════════════════════════════════════════
    for epoch in range(1, CFG["epochs"] + 1):
        student.train()
        projector.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CFG['epochs']}", total=actual_batches)

        for step, batch in enumerate(pbar, 1):
            if step > CFG["max_train_batches"]:
                print(f"\n  Reached max_train_batches={CFG['max_train_batches']}, stopping.")
                break

            ids_t  = batch["input_ids"].to(DEVICE_T)
            mask_t = batch["attention_mask"].to(DEVICE_T)
            ids_s  = batch["input_ids"].to(DEVICE_S)
            mask_s = batch["attention_mask"].to(DEVICE_S)
            labels = batch["labels"].to(DEVICE_S)

            # ── cosine temperature ────────────────────────
            T_cur = get_temperature(global_step, total_steps,
                                    CFG["kl_temp_max"], CFG["kl_temp_min"])

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=CFG["fp16"]):

                # Teacher forward (no grad)
                t_act, t_full, t_logits = forward_with_hiddens(
                    teacher, ids_t, mask_t, t_mid, DEVICE_T, no_grad=True,
                )
                t_logits_s = t_logits.to(DEVICE_S)

                # Difficulty weights (on student, no grad)
                diff_weights = None
                if CFG["use_difficulty_weight"]:
                    diff_weights = compute_difficulty_weights(
                        student, ids_s, mask_s,
                        early_layer_idx=CFG["difficulty_early_layer"],
                    )

                # Project + optionally correct teacher hiddens
                t_full_corrected: dict[int, torch.Tensor] = {}
                for t_lid, s_lid in zip(t_mid, s_mid):
                    h_t  = t_full[t_lid].to(DEVICE_S)
                    B, T_len, _ = h_t.shape
                    h_proj = projector(h_t.reshape(-1, d_t)).detach()

                    if CFG["do_teacher_correction"]:
                        flat_mask = mask_s.reshape(-1).bool()
                        h_active  = h_proj[flat_mask]
                        h_corr    = correct_teacher_hiddens(
                            h_active, t_cents[s_lid].C, R,
                            CFG["tc_lambda"], CFG["ns_iters"], CFG["tc_steps"],
                        )
                        h_corr_full = h_proj.clone()
                        h_corr_full[flat_mask] = h_corr.to(h_proj.dtype)
                        t_full_corrected[s_lid] = h_corr_full.reshape(B, T_len, d_s)
                    else:
                        t_full_corrected[s_lid] = h_proj.reshape(B, T_len, d_s)

                # Student forward (with grad)
                s_act, s_full, s_logits = forward_with_hiddens(
                    student, ids_s, mask_s, s_mid, DEVICE_S, no_grad=False,
                    label_mask=(labels != -100),
                )

                # CE
                loss_ce = ce_loss(s_logits, labels)

                # KL (all v3 filters)
                loss_kl = compute_kl_loss(
                    s_logits, t_logits_s, labels,
                    mode=CFG["kl_mode"], T=T_cur, lam=CFG["kl_skew_lam"],
                    top_k=CFG["top_k_logits"],
                    s2t_tau_p=CFG["s2t_tau_p"],
                    s2t_tau_h=CFG["s2t_tau_h"],
                    high_ent_rho=CFG["high_ent_rho"],
                    difficulty_weights=diff_weights,
                )

                # NNM (no Frobenius)
                lam_nnm  = nnm_weight(global_step)
                loss_nnm = torch.tensor(0., device=DEVICE_S)
                label_mask_flat = (labels != -100).float()
                for t_lid, s_lid in zip(t_mid, s_mid):
                    label_flat      = label_mask_flat.reshape(-1).bool()
                    h_t_proj_active = t_full_corrected[s_lid].reshape(-1, d_s)[label_flat]
                    h_s_active      = s_act[s_lid]
                    loss_nnm = loss_nnm + nnm_loss_one_layer(
                        h_s_active, h_t_proj_active,
                        s_cents[s_lid].C, t_cents[s_lid].C, R,
                        lw[s_lid], CFG["ns_iters"],
                    )
                loss_nnm = loss_nnm / len(s_mid)

                # Combined loss
                loss = (
                    CFG["ce_weight"] * loss_ce
                    + CFG["kl_weight"] * loss_kl
                    + lam_nnm * loss_nnm
                ) / CFG["grad_accum"]

            loss.backward()

            # ── On-policy KD (periodic) ───────────────────
            loss_on_policy = torch.tensor(0., device=DEVICE_S)
            if (CFG["do_on_policy"]
                    and global_step > 0
                    and step % CFG["on_policy_interval"] == 0):
                try:
                    loss_on_policy = on_policy_kl_step(
                        student, teacher, tok, train_ds, CFG, T_cur,
                    )
                    (CFG["on_policy_weight"] * loss_on_policy / CFG["grad_accum"]).backward()
                except Exception as e:
                    print(f"  [on-policy warning] {e}")

            # ── Optimiser step ────────────────────────────
            if step % CFG["grad_accum"] == 0:
                nn.utils.clip_grad_norm_(trainable, CFG["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # ── Update student centroids ──────────────────
            with torch.no_grad():
                for s_lid in s_mid:
                    s_cents[s_lid].update(s_act[s_lid].float().detach())

            # ── Logging ───────────────────────────────────
            log_buf["total"]  += loss.item() * CFG["grad_accum"]
            log_buf["ce"]     += loss_ce.item()
            log_buf["kl"]     += loss_kl.item()
            log_buf["nnm"]    += loss_nnm.item()
            log_buf["on_pol"] += loss_on_policy.item()
            log_buf["n"]      += 1

            if step % CFG["log_every"] == 0:
                n = log_buf["n"]
                pbar.set_postfix({
                    "loss": f"{log_buf['total']/n:.3f}",
                    "kl":   f"{log_buf['kl']/n:.3f}",
                    "nnm":  f"{log_buf['nnm']/n:.3f}",
                    "T":    f"{T_cur:.2f}",
                    "lam":  f"{lam_nnm:.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
                })
                log_buf = {"total": 0., "ce": 0., "kl": 0., "nnm": 0., "on_pol": 0., "n": 0}

        # ── Checkpoint ───────────────────────────────────────
        ckpt = os.path.join(CFG["save_dir"], f"epoch_{epoch}")
        os.makedirs(ckpt, exist_ok=True)
        student.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        torch.save(projector.state_dict(), os.path.join(ckpt, "projector.pt"))
        print(f"\n  ✓ Checkpoint → {ckpt}\n")

    print("Training complete!\n")
    return student, tok


# ════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NNM-KD v3 trainer")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only evaluate existing checkpoints.")
    p.add_argument("--save-dir",  type=str, default=None,
                   help="Override CFG['save_dir'].")
    p.add_argument("--epochs",    type=int, default=None,
                   help="Override CFG['epochs'].")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Override CFG['max_train_batches'].")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save_dir:
        CFG["save_dir"] = args.save_dir
    if args.epochs:
        CFG["epochs"] = args.epochs
    if args.max_batches:
        CFG["max_train_batches"] = args.max_batches

    os.makedirs(CFG["save_dir"], exist_ok=True)

    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(
            CFG["student_id"], trust_remote_code=True, padding_side="right",
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.chat_template = QWEN_CHAT_TEMPLATE
    else:
        _, tok = train()

    compare_all(
        tok,
        distilled_ckpt_dir=CFG["save_dir"],
        n_eval_gsm=CFG["n_eval_gsm8k"],
        n_eval_math=CFG["n_eval_math"],
    )
