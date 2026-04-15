"""
utils.py — Shared forward helpers + on-policy distillation step.
"""

import contextlib
import random

import torch
import torch.nn.functional as F

from configs import CFG

DEVICE_S = torch.device("cuda:0")
DEVICE_T = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════
#  Forward with hidden states
# ═══════════════════════════════════════════════════════════════

def forward_with_hiddens(
    model,
    input_ids:       torch.Tensor,
    attention_mask:  torch.Tensor,
    layer_ids:       list[int],
    target_device:   torch.device,
    no_grad:         bool = False,
    label_mask:      torch.Tensor | None = None,
):
    """
    Run a forward pass and collect hidden states at specified layer indices.

    Returns:
        hiddens_active  — dict[lid] → (N_active, d)   flat, active-token only
        hiddens_full    — dict[lid] → (B, T, d)        full sequence
        logits          — (B, T, vocab) on target_device
    """
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    logits       = out.logits.to(target_device)
    hiddens_active: dict[int, torch.Tensor] = {}
    hiddens_full:   dict[int, torch.Tensor] = {}

    for lid in layer_ids:
        h = out.hidden_states[lid].to(target_device).float()
        hiddens_full[lid] = h

        flat_h = h.reshape(-1, h.shape[-1])
        if label_mask is not None:
            flat_mask = label_mask.to(target_device).reshape(-1).bool()
        else:
            flat_mask = attention_mask.to(target_device).reshape(-1).bool()

        ha = flat_h[flat_mask]
        hiddens_active[lid] = ha.detach() if no_grad else ha

    return hiddens_active, hiddens_full, logits


# ═══════════════════════════════════════════════════════════════
#  On-policy distillation step (v3)
# ═══════════════════════════════════════════════════════════════

def on_policy_kl_step(student, teacher, tokenizer, train_ds, cfg: dict, T: float) -> torch.Tensor:
    """
    SDFT-inspired on-policy KD:
      1. Sample random prompts from training data.
      2. Student generates responses (on-policy, with gradients off).
      3. Teacher scores the student's own tokens.
      4. Reverse KL: D_KL(student || teacher).
    """
    from src.losses import compute_kl_loss  # avoid circular at module level

    student.eval()
    indices = random.sample(range(len(train_ds)), min(cfg["on_policy_batch"], len(train_ds)))

    _orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    prompts = []
    for idx in indices:
        item   = train_ds[idx]
        ids    = item["input_ids"]
        labels = item["labels"]
        prompt_len = 0
        for j, l in enumerate(labels.tolist() if hasattr(labels, "tolist") else labels):
            if l != -100:
                prompt_len = j
                break
        if prompt_len > 0:
            prompt_ids = ids[:prompt_len]
            prompts.append(tokenizer.decode(
                prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else prompt_ids,
                skip_special_tokens=False,
            ))

    if not prompts:
        tokenizer.padding_side = _orig_padding
        student.train()
        return torch.tensor(0., device=DEVICE_S)

    with torch.no_grad():
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg["max_prompt_length"],
        ).to(DEVICE_S)
        gen_out = student.generate(
            **enc,
            max_new_tokens=cfg["on_policy_max_new"],
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    student.train()
    gen_ids  = gen_out.to(DEVICE_S)
    gen_mask = (gen_ids != tokenizer.pad_token_id).long()
    prompt_len_tok = enc["input_ids"].shape[1]

    gen_labels = gen_ids.clone()
    gen_labels[:, :prompt_len_tok] = -100
    gen_labels[gen_mask == 0] = -100

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg["fp16"]):
        s_out    = student(input_ids=gen_ids, attention_mask=gen_mask, return_dict=True)
        s_logits = s_out.logits

        gen_ids_t  = gen_ids.to(DEVICE_T)
        gen_mask_t = gen_mask.to(DEVICE_T)
        with torch.no_grad():
            t_out    = teacher(input_ids=gen_ids_t, attention_mask=gen_mask_t, return_dict=True)
            t_logits = t_out.logits.to(DEVICE_S)

        on_policy_loss = compute_kl_loss(
            s_logits, t_logits, gen_labels,
            mode="reverse_kl", T=T, lam=cfg["kl_skew_lam"],
            top_k=cfg["top_k_logits"],
        )

    tokenizer.padding_side = _orig_padding
    return on_policy_loss
