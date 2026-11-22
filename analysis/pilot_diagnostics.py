"""Reproduce the pilot-study numbers reported in paper_usc_dpo.tex (Sections 6.2-6.5).

Run from the repository root:  python3 analysis/pilot_diagnostics.py
Writes analysis/regrade_labels.csv (150 rows: original, blind rule v2, manual labels).
Only the standard library and numpy are required. The relative weight change
(||dW|| / ||W||) needs the base model and is reported in the paper; here only
the adapter-side quantities are recomputed.
"""
import csv
import json
import math
import random
import re
import struct
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "USC_DPO_final_evaluation (2)" / "final_evaluation_predictions.json"
ADAPTERS = {
    "SFT-CoT": ROOT / "USC_DPO_final_evaluation (2)" / "sft_lora_adapter",
    "Sol-DPO": ROOT / "USC_DPO_training_artifacts" / "dpo_lora_adapter",
}
NAME = {"Base": "Base", "SFT-CoT": "SFT-CoT", "USC-DPO": "Sol-DPO"}

# Manual (non-blind) re-grading by the authors: problems marked incorrect by the
# original grader but judged correct after reading the final sentence.
MANUAL_FIXES = {
    "Base": {5, 15, 18, 23, 27, 30, 36},
    "SFT-CoT": {5, 15, 18, 23, 27, 36},
    "USC-DPO": {5, 18, 19, 23, 27, 30, 36},
}

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def norm(s):
    s = s.replace("$", "").replace(",", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_v2(text):
    """Blind rule v2: marker > boxed > 'the answer is' > last '=' or first number in final sentence."""
    for pattern in (r"####\s*(" + NUM + ")", r"\\boxed\{\s*\\?\$?\s*(" + NUM + ")"):
        m = re.findall(pattern, text)
        if m:
            return norm(m[-1])
    m = re.findall(r"answer is[:\s]*\**\s*(" + NUM + ")", text, re.I)
    if m:
        return norm(m[-1])
    body = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", body) if re.search(r"\d", x)]
    if not sents:
        return None
    last = sents[-1]
    if "=" in last:
        m = re.findall(NUM, last.split("=")[-1])
        if m:
            return norm(m[0])
    m = re.findall(NUM, last)
    return norm(m[0]) if m else None


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_safetensors(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        assert meta["dtype"] == "F32", meta["dtype"]
        out[key] = np.frombuffer(raw[8 + n + s:8 + n + e], np.float32).astype(np.float64).reshape(meta["shape"])
    return out


def main():
    preds = json.loads(PRED.read_text())

    # Grade in shuffled order without looking at the model name (blind rule).
    order = list(range(len(preds)))
    random.Random(0).shuffle(order)
    v2 = {}
    for i in order:
        e = preds[i]
        p = extract_v2(e["generated_text"])
        v2[i] = p is not None and p == norm(e["gold"])

    rows = []
    for i, e in enumerate(preds):
        manual = bool(e["correct"]) or e["test_index"] in MANUAL_FIXES[e["model"]]
        rows.append({
            "model": NAME[e["model"]],
            "test_index": e["test_index"],
            "gold": e["gold"],
            "original_correct": int(bool(e["correct"])),
            "v2_correct": int(v2[i]),
            "manual_correct": int(manual),
            "generated_tokens": e["generated_tokens"],
            "latency_seconds": e["latency_seconds"],
        })
    out = ROOT / "analysis" / "regrade_labels.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}")

    print("\nAccuracy (Wilson 95%)")
    for model in ("Base", "SFT-CoT", "Sol-DPO"):
        r = [x for x in rows if x["model"] == model]
        parts = []
        for col in ("original_correct", "v2_correct", "manual_correct"):
            k = sum(x[col] for x in r)
            lo, hi = wilson(k, len(r))
            parts.append(f"{col.split('_')[0]} {k}/{len(r)} ({100*lo:.0f}-{100*hi:.0f})")
        print(f"  {model:8s} " + " | ".join(parts))

    agree = sum(x["v2_correct"] == x["manual_correct"] for x in rows)
    pa = sum(x["v2_correct"] for x in rows) / len(rows)
    pm = sum(x["manual_correct"] for x in rows) / len(rows)
    po = agree / len(rows)
    pe = pa * pm + (1 - pa) * (1 - pm)
    print(f"\nv2 vs manual agreement {agree}/{len(rows)}, Cohen's kappa {(po - pe) / (1 - pe):.2f}")

    lab = {(x["model"], x["test_index"]): x["v2_correct"] for x in rows}
    print("\nPaired comparisons on v2 labels (A only, B only, exact McNemar p)")
    for a, b in (("SFT-CoT", "Base"), ("Sol-DPO", "SFT-CoT"), ("Sol-DPO", "Base")):
        only_a = sum(lab[(a, i)] and not lab[(b, i)] for i in range(50))
        only_b = sum(lab[(b, i)] and not lab[(a, i)] for i in range(50))
        print(f"  {a} vs {b}: {only_a} / {only_b}, p = {mcnemar_exact(only_a, only_b):.2f}")

    text = {(NAME[e["model"]], e["test_index"]): e["generated_text"] for e in preds}
    print("\nOutputs character-identical to Base")
    for m in ("SFT-CoT", "Sol-DPO"):
        same = sum(text[(m, i)] == text[("Base", i)] for i in range(50))
        print(f"  {m}: {same}/50")

    print("\nLoRA update size (dW = alpha/r * B A)")
    for name, d in ADAPTERS.items():
        cfg = json.loads((d / "adapter_config.json").read_text())
        t = load_safetensors(d / "adapter_model.safetensors")
        scale = cfg["lora_alpha"] / cfg["r"]
        norms, bmax = [], []
        for ka in (k for k in t if "lora_A" in k):
            B = t[ka.replace("lora_A", "lora_B")]
            with np.errstate(all="ignore"):
                norms.append(np.linalg.norm(scale * (B @ t[ka])))
            bmax.append(np.abs(B).max())
        print(f"  {name}: {len(norms)} matrices, mean ||dW||_F {np.mean(norms):.2e}, max|B| {max(bmax):.2e}")

    print("\nPaired-test sample size (alpha 0.05, power 0.8)")
    for pd in (0.06, 0.10):
        cells = []
        for d in (0.02, 0.03, 0.05):
            n = (1.959964 * math.sqrt(pd) + 0.841621 * math.sqrt(pd - d * d)) ** 2 / d ** 2
            cells.append(f"d={d:.2f}: {math.ceil(n)}")
        print(f"  discordance {pd:.2f}: " + ", ".join(cells))


if __name__ == "__main__":
    main()
