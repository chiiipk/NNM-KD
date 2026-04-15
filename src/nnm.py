"""
nnm.py — Nuclear Norm Matching core:
  • Newton-Schulz polar factor (fast, gradient-friendly)
  • RunningCentroids (EMA with dead-centroid revival)
  • NNM loss (per-layer, no Frobenius term — v3)
  • Teacher-centroid pre-pass
  • Teacher hidden correction
"""

import math
import random

import torch
import torch.nn as nn

from configs import CFG, SEED

DEVICE_S = torch.device("cuda:0")
DEVICE_T = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════
#  Newton-Schulz polar factor
# ═══════════════════════════════════════════════════════════════

_NS_COEFFS = (15 / 8, -10 / 8, 3 / 8)


def newton_schulz_polar(M: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    assert M.ndim == 2
    dtype = M.dtype
    transposed = False
    if M.shape[0] < M.shape[1]:
        M = M.T
        transposed = True
    X = M / (M.norm() + 1e-7)
    a, b, c = _NS_COEFFS
    for _ in range(n_iters):
        A = X.T @ X
        X = a * X + b * (X @ A) + c * (X @ (A @ A))
    if transposed:
        X = X.T
    return X.to(dtype)


class _NuclearNormNS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, n_iters):
        with torch.no_grad():
            P = newton_schulz_polar(M.detach(), n_iters)
        ctx.save_for_backward(P)
        return (P * M).sum()

    @staticmethod
    def backward(ctx, grad_output):
        (P,) = ctx.saved_tensors
        return grad_output * P, None


def nuclear_norm_ns(M: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    return _NuclearNormNS.apply(M, n_iters)


# ═══════════════════════════════════════════════════════════════
#  Running Centroids
# ═══════════════════════════════════════════════════════════════

class RunningCentroids:
    """
    EMA centroid tracker with dead-centroid revival and decaying eta.
    """

    def __init__(self, K: int, d: int, eta: float, T_dead: int, device: torch.device):
        self.K      = K
        self.d      = d
        self.eta    = eta
        self.T_dead = T_dead
        self.device = device
        self.C    = torch.randn(K, d, device=device, dtype=torch.float32) * 0.01
        self.dead = torch.zeros(K, device=device, dtype=torch.int32)
        self._step = 0

    @torch.no_grad()
    def update(self, H: torch.Tensor) -> None:
        H = H.to(self.device).float()
        if H.shape[0] == 0:
            return
        self._step += 1
        eta = self.eta / (1 + 0.001 * self._step)   # decaying eta

        dists  = torch.cdist(H, self.C)
        assign = dists.argmin(dim=1)
        for k in range(self.K):
            mask = (assign == k)
            if mask.any():
                self.C[k] = (1 - eta) * self.C[k] + eta * H[mask].mean(0)
                self.dead[k] = 0
            else:
                self.dead[k] += 1
                if self.dead[k] >= self.T_dead:
                    self.C[k] = H[random.randint(0, len(H) - 1)].clone()
                    self.dead[k] = 0


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def make_R(d: int, d_prime: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(SEED)
    return (torch.randn(d, d_prime, device=device) / math.sqrt(d_prime)).float()


def layer_weight(l: int, L: int, sigma: float = 0.15) -> float:
    return math.exp(-((l / L - 0.5) ** 2) / (2 * sigma ** 2))


def select_mid_layers(n_layers: int, n_mid: int) -> list[int]:
    """v3: shifted to 40%–85% (later layers encode reasoning logic)."""
    import numpy as np
    lo = max(1, int(0.40 * n_layers))
    hi = min(n_layers, int(0.85 * n_layers))
    if lo >= hi:
        lo = max(0, hi - n_mid)
    return sorted(set(int(i) for i in np.linspace(lo, hi, n_mid, dtype=int).tolist()))


# ═══════════════════════════════════════════════════════════════
#  NNM loss (per layer, v3 — no Frobenius)
# ═══════════════════════════════════════════════════════════════

def nnm_loss_one_layer(
    H_s:        torch.Tensor,
    H_t_proj:   torch.Tensor,
    C_s:        torch.Tensor,
    C_t:        torch.Tensor,
    R:          torch.Tensor,
    lw:         float,
    ns_iters:   int,
) -> torch.Tensor:
    H_s      = H_s.float()
    H_t_proj = H_t_proj.float().detach()
    C_s      = C_s.float().detach()
    C_t      = C_t.float().detach()
    R        = R.float()

    M_s = torch.cat([C_s, H_s],       dim=0) @ R
    M_t = torch.cat([C_t, H_t_proj],  dim=0) @ R
    m, n = M_s.shape
    scale = math.sqrt(m * n)

    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale
    nn_t = (nuclear_norm_ns(M_t, ns_iters) / scale).detach()
    return lw * (nn_s - nn_t) ** 2


@torch.no_grad()
def measure_nuclear_norms(
    s_act:     dict[int, torch.Tensor],
    s_cents:   dict[int, RunningCentroids],
    R:         torch.Tensor,
    ns_iters:  int,
) -> dict[int, float]:
    result = {}
    for s_lid, H_s in s_act.items():
        C_s = s_cents[s_lid].C.float()
        H   = H_s.float().detach()
        M   = torch.cat([C_s, H], dim=0) @ R.float()
        m, n = M.shape
        result[s_lid] = (nuclear_norm_ns(M, ns_iters) / math.sqrt(m * n)).item()
    return result


# ═══════════════════════════════════════════════════════════════
#  Teacher correction
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def correct_teacher_hiddens(
    H_T:       torch.Tensor,
    C_T:       torch.Tensor,
    R:         torch.Tensor,
    lam:       float,
    ns_iters:  int,
    tc_steps:  int = 1,
) -> torch.Tensor:
    H = H_T.float().clone()
    K = C_T.shape[0]
    R = R.float()
    for _ in range(tc_steps):
        M0  = torch.cat([C_T.float(), H], dim=0) @ R
        P   = newton_schulz_polar(M0, ns_iters)
        G_X = P[K:] @ R.T
        H   = H + lam * G_X
    return H.to(H_T.dtype)


# ═══════════════════════════════════════════════════════════════
#  Teacher centroid pre-pass
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def build_teacher_centroids(
    teacher,
    projector,
    dataloader,
    t_mid:  list[int],
    s_mid:  list[int],
    K:      int,
    d_s:    int,
    eta:    float,
    T_dead: int,
) -> dict[int, RunningCentroids]:
    from src.utils import forward_with_hiddens  # avoid circular at module level

    centroids = {
        s_lid: RunningCentroids(K, d_s, eta, T_dead, DEVICE_S)
        for s_lid in s_mid
    }
    teacher.eval()
    projector.eval()
    max_batches = 3000

    from tqdm import tqdm
    for i, batch in enumerate(tqdm(dataloader, desc="  Teacher centroid pre-pass", total=max_batches)):
        if i >= max_batches:
            break
        ids  = batch["input_ids"].to(DEVICE_T)
        mask = batch["attention_mask"].to(DEVICE_T)
        t_act, _, _ = forward_with_hiddens(teacher, ids, mask, t_mid, DEVICE_T, no_grad=True)
        for t_lid, s_lid in zip(t_mid, s_mid):
            h_proj = projector(t_act[t_lid].to(DEVICE_S)).float()
            centroids[s_lid].update(h_proj)

    projector.train()
    return centroids
