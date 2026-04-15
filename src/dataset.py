"""
dataset.py — MetaMathQA dataset builder + teacher-quality filter.
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

from configs import CFG, SYSTEM_PROMPT, SEED

DEVICE_T = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════
#  MathDistillDataset
# ═══════════════════════════════════════════════════════════════

class MathDistillDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return {
            "input_ids":      torch.tensor(row["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(row["labels"],         dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════
#  Builder
# ═══════════════════════════════════════════════════════════════

def build_metamath(tokenizer, max_len: int = 768, max_prompt_len: int = 256) -> MathDistillDataset:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    raw = load_dataset(CFG["train_dataset"], split="train").shuffle(seed=SEED)
    print(f"  MetaMathQA columns: {raw.column_names}  |  rows: {len(raw)}")

    Q_CANDS = ["query", "instruction", "problem", "question"]
    A_CANDS = ["response", "answer", "output", "solution"]
    col0 = raw.column_names[0]

    def make_texts(batch):
        prompts, fulls = [], []
        for i in range(len(batch[col0])):
            q = next((str(batch[c][i]) for c in Q_CANDS if c in batch and batch[c][i]), "")
            a = next((str(batch[c][i]) for c in A_CANDS if c in batch and batch[c][i]), "")
            if not q or not a:
                prompts.append(""); fulls.append(""); continue
            p = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            f = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": q},
                 {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False,
            )
            prompts.append(p); fulls.append(f)
        return {"prompt": prompts, "full": fulls}

    ds = raw.map(make_texts, batched=True, batch_size=2000, num_proc=1,
                 desc="  Building texts", remove_columns=raw.column_names)
    ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(x["full"]) > 0, num_proc=1)

    def tokenize(batch):
        full_enc   = tokenizer(batch["full"],   max_length=max_len,        truncation=True, padding="max_length")
        prompt_enc = tokenizer(batch["prompt"], max_length=max_prompt_len, truncation=True, add_special_tokens=True)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, ids in enumerate(full_enc["input_ids"]):
            p_len  = len(prompt_enc["input_ids"][i])
            attn   = full_enc["attention_mask"][i]
            labels = [
                -100 if (j < p_len or attn[j] == 0) else ids[j]
                for j in range(len(ids))
            ]
            if sum(1 for l in labels if l != -100) >= 10:
                out["input_ids"].append(ids)
                out["attention_mask"].append(attn)
                out["labels"].append(labels)
        return out

    ds = ds.map(tokenize, batched=True, batch_size=1000, num_proc=1,
                desc="  Tokenising", remove_columns=["prompt", "full"])
    ds.set_format(type=None)
    print(f"  Final dataset: {len(ds)} samples")
    return MathDistillDataset(ds)


# ═══════════════════════════════════════════════════════════════
#  Teacher-quality filter
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def filter_dataset_by_teacher(
    teacher,
    tokenizer,
    dataset: MathDistillDataset,
    max_samples: int = 5000,
    batch_size:  int = 16,
) -> MathDistillDataset:
    """
    v3: Chạy teacher trên subset, chỉ giữ samples teacher giải đúng.
    Proxy: teacher per-sample CE loss < 3.0.
    """
    print(f"\n  Filtering dataset by teacher correctness (up to {max_samples} samples)...")
    teacher.eval()
    _orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    n_check   = min(max_samples, len(dataset))
    good_indices: set[int] = set()
    bad_count = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, batch in enumerate(tqdm(loader, desc="  Teacher filter", total=n_check // batch_size)):
        if i * batch_size >= n_check:
            break
        ids    = batch["input_ids"].to(DEVICE_T)
        mask   = batch["attention_mask"].to(DEVICE_T)
        labels = batch["labels"].to(DEVICE_T)

        out    = teacher(input_ids=ids, attention_mask=mask, return_dict=True)
        logits = out.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        for j in range(ids.shape[0]):
            sample_idx = i * batch_size + j
            active = (shift_labels[j] != -100)
            if active.sum() == 0:
                continue
            sample_loss = F.cross_entropy(
                shift_logits[j][active],
                shift_labels[j][active],
                reduction="mean",
            )
            if sample_loss.item() < 3.0:
                good_indices.add(sample_idx)
            else:
                bad_count += 1

    tokenizer.padding_side = _orig_padding
    print(f"  Teacher filter: kept {len(good_indices)}/{n_check} checked, "
          f"removed {bad_count} high-loss samples")

    all_good = sorted(list(good_indices) + list(range(n_check, len(dataset))))
    filtered_ds = dataset.ds.select(all_good)
    print(f"  Filtered dataset: {len(filtered_ds)} samples (from {len(dataset)})")
    return MathDistillDataset(filtered_ds)
