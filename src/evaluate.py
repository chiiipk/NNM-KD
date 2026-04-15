"""
evaluate.py — Evaluation on GSM8K and MATH-500.
Answer extraction logic kept identical to v2.
"""

import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from configs import CFG, SYSTEM_PROMPT

DEVICE_S = torch.device("cuda:0")


# ═══════════════════════════════════════════════════════════════
#  Answer extraction helpers
# ═══════════════════════════════════════════════════════════════

def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        return s


def extract_answer_gsm(text: str) -> str:
    m = re.search(r"####\s*([\-\$]?[\d,\.]+)", text)
    if m:
        return normalize_number(m.group(1))
    tail = text[-600:]
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?([\-\d,\.]+)",
        r"answer[:\s]+\$?([\-\d,\.]+)",
        r"=\s*\$?([\-\d,\.]+)\s*$",
        r"total[:\s]+\$?([\-\d,\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return normalize_number(m.group(1))
    cleaned = re.sub(r"Step\s+\d+[:\.].*?(?=Step|\Z)", "", text,
                     flags=re.IGNORECASE | re.DOTALL)
    nums = re.findall(r"(?<![a-zA-Z\d])([-]?\d[\d,]*\.?\d*)(?![a-zA-Z\d])", cleaned)
    if nums:
        return normalize_number(nums[-1])
    return ""


def normalise_math_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.strip()


def extract_answer_math(text: str) -> str:
    def find_all_boxed(s: str):
        results = []
        for marker in [r"\boxed{", r"\boxed {"]:
            pos = 0
            while True:
                idx = s.find(marker, pos)
                if idx == -1:
                    break
                brace_start = s.find("{", idx)
                if brace_start == -1:
                    break
                depth = 0
                end = -1
                for i in range(brace_start, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end != -1:
                    results.append((brace_start + 1, end))
                pos = idx + 1
        return results

    spans = find_all_boxed(text)
    if spans:
        start, end = spans[-1]
        return normalise_math_answer(text[start:end])
    tail = text[-800:]
    for pat in [
        r"(?:answer|therefore|thus|so)[:\s=]+([^\n\.]{1,60})",
        r"=\s*([^\n=]{1,40})\s*$",
    ]:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".,;")
            if len(candidate) < 60:
                return normalise_math_answer(candidate)
    return ""


def check_gsm_correct(pred: str, gold: str) -> bool:
    p = normalize_number(pred)
    g = normalize_number(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
#  GSM8K evaluator
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, n_eval: int = 99999, batch_size: int = 32):
    model.eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=0).select(range(min(n_eval, len(ds))))
    total   = len(ds)
    correct = 0
    records = []

    _orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, total, batch_size), desc="  Eval GSM8K"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for question, answer in zip(batch["question"], batch["answer"]):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            golds.append(extract_answer_gsm(answer))

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=CFG["max_prompt_length"],
        ).to(DEVICE_S)
        out = model.generate(
            **enc, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        for j, (gen, gold, question, answer) in enumerate(
            zip(decoded, golds, batch["question"], batch["answer"])
        ):
            pred       = normalise_math_answer(extract_answer_math(gen))
            is_correct = (pred == gold) or check_gsm_correct(pred, gold)
            if is_correct:
                correct += 1
            records.append({
                "id": i + j, "question": question,
                "gold_raw": answer, "gold_extracted": gold,
                "pred_extracted": pred, "generated": gen,
                "correct": is_correct,
            })

    tokenizer.padding_side = _orig_padding
    acc = correct / total * 100
    print(f"\n  GSM8K Pass@1: {correct}/{total} = {acc:.2f}%")
    return acc, records


# ═══════════════════════════════════════════════════════════════
#  MATH-500 evaluator
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_math500(model, tokenizer, n_eval: int = 99999, batch_size: int = 32):
    model.eval()
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
    print(f"  MATH: HuggingFaceH4/MATH-500 ({len(ds)} samples)")
    pcol, acol = "problem", "solution"
    ds = ds.shuffle(seed=0).select(range(min(n_eval, len(ds))))
    total   = len(ds)
    correct = 0
    records = []

    _orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, total, batch_size), desc="  Eval MATH-500"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for problem, solution in zip(batch[pcol], batch[acol]):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": problem},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            golds.append(normalise_math_answer(extract_answer_math(solution)))

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=CFG["max_prompt_length"],
        ).to(DEVICE_S)
        out = model.generate(
            **enc, max_new_tokens=786, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        for j, (gen, gold, problem, solution) in enumerate(
            zip(decoded, golds, batch[pcol], batch[acol])
        ):
            pred       = normalise_math_answer(extract_answer_math(gen))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            records.append({
                "id": i + j, "problem": problem,
                "gold_raw": solution, "gold_extracted": gold,
                "pred_extracted": pred, "generated": gen,
                "correct": is_correct,
            })

    tokenizer.padding_side = _orig_padding
    acc = correct / total * 100
    print(f"\n  MATH-500 Pass@1: {correct}/{total} = {acc:.2f}%")
    return acc, records


# ═══════════════════════════════════════════════════════════════
#  Result persistence
# ═══════════════════════════════════════════════════════════════

def save_detail(label, gsm_acc, gsm_records, math_acc, math_records, out_dir):
    safe    = label.replace(" ", "_").replace("/", "-")
    path    = os.path.join(out_dir, f"detail_{safe}.json")
    payload = {
        "label":   label,
        "summary": {
            "gsm8k":   {"correct": sum(r["correct"] for r in gsm_records),
                        "total": len(gsm_records), "acc": gsm_acc},
            "math500": {"correct": sum(r["correct"] for r in math_records),
                        "total": len(math_records), "acc": math_acc},
            "avg": (gsm_acc + math_acc) / 2,
        },
        "gsm8k_records":  gsm_records,
        "math500_records": math_records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Detail saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  Comparison runner
# ═══════════════════════════════════════════════════════════════

def compare_all(tokenizer, distilled_ckpt_dir: str, n_eval_gsm: int = 99999, n_eval_math: int = 99999):
    from transformers import AutoModelForCausalLM

    summary = {}

    def run_both(model, label, detail_dir):
        gsm_acc,  gsm_rec  = evaluate_gsm8k(model, tokenizer, n_eval=n_eval_gsm)
        math_acc, math_rec = evaluate_math500(model, tokenizer, n_eval=n_eval_math)
        avg = (gsm_acc + math_acc) / 2
        summary[label] = {"gsm8k": gsm_acc, "math500": math_acc, "avg": avg}
        save_detail(label, gsm_acc, gsm_rec, math_acc, math_rec, detail_dir)
        print(f"  [{label}] GSM8K={gsm_acc:.2f}%  MATH-500={math_acc:.2f}%  Avg={avg:.2f}%")

    for ep in range(1, CFG["epochs"] + 1):
        ckpt = os.path.join(distilled_ckpt_dir, f"epoch_{ep}")
        if not os.path.isdir(ckpt):
            print(f"  [skip] {ckpt} not found")
            continue
        print(f"\n{'='*62}\n  NNM-KD v3 student — epoch {ep}\n{'='*62}")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(DEVICE_S)
        model.eval()
        run_both(model, f"NNM-KD_v3_epoch_{ep}", ckpt)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  {'Model':<30}  {'MATH-500':>10}  {'GSM8K':>8}  {'Avg':>6}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*6}")
    for name, r in summary.items():
        print(f"  {name:<30}  {r['math500']:>10.2f}  {r['gsm8k']:>8.2f}  {r['avg']:>6.2f}")
    print(f"{'='*70}\n")

    summary_path = os.path.join(distilled_ckpt_dir, "eval_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary → {summary_path}")
    return summary
