"""
models.py — Teacher / Student loader + HiddenProjector (2-layer MLP, v3).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

DEVICE_S = torch.device("cuda:0")
DEVICE_T = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════
#  Loaders
# ═══════════════════════════════════════════════════════════════

def load_teacher(model_id: str) -> nn.Module:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": DEVICE_T},
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Teacher {n:.2f}B params, 4-bit NF4 → {DEVICE_T}")
    return model


def load_student(model_id: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(DEVICE_S)
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Student {n:.2f}B params, float32 (AMP fp16 fwd) → {DEVICE_S}")
    return model


# ═══════════════════════════════════════════════════════════════
#  HiddenProjector — v3: 2-layer MLP with GELU
# ═══════════════════════════════════════════════════════════════

class HiddenProjector(nn.Module):
    """
    Maps teacher hidden states (d_teacher) → student space (d_student).
    v3 upgrade: 2-layer MLP with GELU for better cross-space mapping.
    Last layer initialised orthogonal for training stability.
    """

    def __init__(self, d_teacher: int, d_student: int):
        super().__init__()
        d_mid = d_teacher // 2
        self.net = nn.Sequential(
            nn.Linear(d_teacher, d_mid, bias=False, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(d_mid, d_student, bias=False, dtype=torch.float32),
        )
        nn.init.orthogonal_(self.net[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())
