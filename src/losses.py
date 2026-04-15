"""
losses.py — All KL divergence variants + CE loss + difficulty-aware weighting.

v3 enhancements:
  • Top-K logit filtering      (reduce long-tail waste)
  • S2T spurious token mask    (STAPO)
  • High-entropy token selection (Beyond 80/20)
  • Difficulty-aware per-token weighting via JSD (early vs final layer)
  • Temperature annealing helper
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  CE loss
# ═══════════════════════════════════════════════════════════════

def ce_loss(s_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = s_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ═══════════════════════════════════════════════════════════════
#  KL helpers
# ═══════════════════════════════════════════════════════════════

def _amask(labels: torch.Tensor) -> torch.Tensor:
    return (labels != -100).float()


def distillm_skewed_fwd(s, t, labels, lam=0.1, **kw):
    t_probs  = F.softmax(t, dim=-1)
    s_probs  = F.softmax(s, dim=-1)
    mixed    = lam * t_probs + (1 - lam) * s_probs
    m_lp     = torch.log(mixed.clamp(min=1e-10))
    inf_mask = torch.isinf(s) | torch.isinf(t)
    prod     = torch.masked_fill(t_probs * m_lp, inf_mask, 0.).sum(-1)
    mask     = _amask(labels)
    return -(prod * mask).sum() / mask.sum().clamp(min=1)


def distillm_forward_kl(s, t, labels, **kw):
    t_probs  = F.softmax(t, dim=-1)
    s_lp     = F.log_softmax(s, dim=-1)
    inf_mask = torch.isinf(s) | torch.isinf(t)
    prod     = torch.masked_fill(t_probs * s_lp, inf_mask, 0.).sum(-1)
    mask     = _amask(labels)
    return -(prod * mask).sum() / mask.sum().clamp(min=1)


def distillm_reverse_kl(s, t, labels, **kw):
    s_probs  = F.softmax(s, dim=-1)
    s_lp     = F.log_softmax(s, dim=-1)
    t_lp     = F.log_softmax(t, dim=-1)
    inf_mask = torch.isinf(s) | torch.isinf(t)
    prod     = torch.masked_fill(s_probs * (t_lp - s_lp), inf_mask, 0.).sum(-1)
    mask     = _amask(labels)
    return -(prod * mask).sum() / mask.sum().clamp(min=1)


def distillm_symmetric_kl(s, t, labels, lam=0.9, **kw):
    return (1 - lam) * distillm_forward_kl(s, t, labels) + lam * distillm_reverse_kl(s, t, labels)


def distillm_js(s, t, labels, lam=0.9, **kw):
    t_probs  = F.softmax(t, dim=-1)
    s_probs  = F.softmax(s, dim=-1)
    mixed    = (1 - lam) * t_probs + lam * s_probs
    t_lp     = F.log_softmax(t, dim=-1)
    s_lp     = F.log_softmax(s, dim=-1)
    m_lp     = torch.log(mixed.clamp(min=1e-10))
    inf_mask = torch.isinf(s) | torch.isinf(t)
    mask     = _amask(labels)
    d        = mask.sum().clamp(min=1)
    x1 = torch.masked_fill(s_probs * (m_lp - s_lp), inf_mask, 0.).sum(-1)
    x2 = torch.masked_fill(t_probs * (m_lp - t_lp), inf_mask, 0.).sum(-1)
    return lam * (-(x1 * mask).sum() / d) + (1 - lam) * (-(x2 * mask).sum() / d)


def distillm_tv(s, t, labels, **kw):
    t_probs  = F.softmax(t, dim=-1)
    s_probs  = F.softmax(s, dim=-1)
    inf_mask = torch.isinf(s) | torch.isinf(t)
    prod     = 0.5 * torch.masked_fill((t_probs - s_probs).abs(), inf_mask, 0.).sum(-1)
    mask     = _amask(labels)
    return (prod * mask).sum() / mask.sum().clamp(min=1)


def distillm_skewed_rev(s, t, labels, lam=0.1, **kw):
    t_probs  = F.softmax(t, dim=-1)
    s_probs  = F.softmax(s, dim=-1)
    mixed    = (1 - lam) * t_probs + lam * s_probs
    s_lp     = F.log_softmax(s, dim=-1)
    m_lp     = torch.log(mixed.clamp(min=1e-10))
    inf_mask = torch.isinf(s) | torch.isinf(t)
    prod     = torch.masked_fill(s_probs * (m_lp - s_lp), inf_mask, 0.).sum(-1)
    mask     = _amask(labels)
    return -(prod * mask).sum() / mask.sum().clamp(min=1)


_KL_FNS = {
    "forward_kl":    distillm_forward_kl,
    "reverse_kl":    distillm_reverse_kl,
    "symmetric_kl":  distillm_symmetric_kl,
    "js":            distillm_js,
    "tv":            distillm_tv,
    "skewed_forward": distillm_skewed_fwd,
    "skewed_reverse": distillm_skewed_rev,
}


# ═══════════════════════════════════════════════════════════════
#  Main KL loss (v3 — all enhancements)
# ═══════════════════════════════════════════════════════════════

def compute_kl_loss(
    s_logits:          torch.Tensor,
    t_logits:          torch.Tensor,
    labels:            torch.Tensor,
    mode:              str   = "skewed_forward",
    T:                 float = 2.0,
    lam:               float = 0.1,
    top_k:             int   | None = None,
    s2t_tau_p:         float | None = None,
    s2t_tau_h:         float | None = None,
    high_ent_rho:      float | None = None,
    difficulty_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL loss with v3 token-level filters:
      1. S2T spurious-token mask  (STAPO)
      2. High-entropy selection   (Beyond 80/20)
      3. Top-K logit KD
    """
    fn = _KL_FNS.get(mode)
    if fn is None:
        raise ValueError(f"Unknown kl_mode '{mode}'")

    s_logits_s = s_logits[..., :-1, :]
    t_logits_s = t_logits[..., :-1, :]
    labels_s   = labels[..., 1:]

    vocab  = min(s_logits_s.size(-1), t_logits_s.size(-1))
    active = (labels_s != -100)

    s = s_logits_s[active][..., :vocab].float()
    t = t_logits_s[active][..., :vocab].float()

    if s.shape[0] == 0:
        return torch.tensor(0., device=s_logits.device)

    # ── 1. S2T spurious token mask (STAPO) ──
    if s2t_tau_p is not None and s2t_tau_h is not None:
        with torch.no_grad():
            s_probs_check = F.softmax(s, dim=-1)
            s_entropy = -(s_probs_check * s_probs_check.log().clamp(min=-100)).sum(-1)
            t_top = t.argmax(dim=-1)
            s_prob_for_t_top = s_probs_check.gather(1, t_top.unsqueeze(1)).squeeze(1)
            spurious = (s_prob_for_t_top < s2t_tau_p) & (s_entropy < s2t_tau_h)
            valid = ~spurious
        if valid.any() and valid.sum() < s.shape[0]:
            s = s[valid]
            t = t[valid]
            difficulty_weights = None   # reset — alignment broken after masking

    # ── 2. High-entropy token selection (Beyond 80/20) ──
    if high_ent_rho is not None and high_ent_rho < 1.0:
        with torch.no_grad():
            s_probs_ent = F.softmax(s, dim=-1)
            s_ent = -(s_probs_ent * s_probs_ent.log().clamp(min=-100)).sum(-1)
            k = max(1, int(high_ent_rho * s_ent.shape[0]))
            _, topk_idx = s_ent.topk(k)
        s = s[topk_idx]
        t = t[topk_idx]

    # ── 3. Top-K logit filtering ──
    if top_k is not None and top_k < vocab:
        with torch.no_grad():
            t_topk_vals, t_topk_idx = t.topk(top_k, dim=-1)
        s_topk = s.gather(-1, t_topk_idx)
        s = s_topk / T
        t = t_topk_vals / T
    else:
        s = s / T
        t = t / T

    fake_labels = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
    loss = fn(s, t, fake_labels, lam=lam) * (T ** 2)
    return loss


# ═══════════════════════════════════════════════════════════════
#  Difficulty-aware token weighting (v3)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_difficulty_weights(
    model,
    input_ids:          torch.Tensor,
    attention_mask:     torch.Tensor,
    early_layer_idx:    int = 2,
) -> torch.Tensor | None:
    """
    JSD between early-layer and final-layer token predictions.
    High JSD ↔ hard reasoning token → weight up.
    Returns weight tensor [B, T] normalised to [0.5, 3.0].
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)

    lm_head = getattr(model, 'lm_head', None) or model.get_output_embeddings()
    if lm_head is None:
        return None

    h_early      = out.hidden_states[early_layer_idx]
    logits_early = lm_head(h_early)
    logits_final = out.logits

    p_early = F.softmax(logits_early.float(), dim=-1)
    p_final = F.softmax(logits_final.float(), dim=-1)
    m       = 0.5 * (p_early + p_final)

    kl_early = (p_early * (p_early.log() - m.log()).clamp(min=-100)).sum(-1)
    kl_final = (p_final * (p_final.log() - m.log()).clamp(min=-100)).sum(-1)
    jsd      = 0.5 * (kl_early + kl_final)  # [B, T]

    jsd_norm = jsd / (jsd.mean() + 1e-8)
    return jsd_norm.clamp(0.5, 3.0)


# ═══════════════════════════════════════════════════════════════
#  Temperature annealing helper (v3)
# ═══════════════════════════════════════════════════════════════

def get_temperature(step: int, total_steps: int, T_max: float, T_min: float) -> float:
    """Cosine decay from T_max → T_min over total_steps."""
    progress = min(1.0, step / max(1, total_steps))
    return T_min + 0.5 * (T_max - T_min) * (1 + math.cos(math.pi * progress))
