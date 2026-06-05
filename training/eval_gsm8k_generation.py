#!/usr/bin/env python3
"""
GSM8K generation-based accuracy at varying recurrent-depth K.

Tests whether extra K lets the model construct better chains-of-thought,
producing more accurate final answers. Existing reasoning_eval.py scores
length-normalized log-probabilities of pre-written choices; this script
generates a CoT response and regex-extracts the final numeric answer.

For each ckpt and each K in {4, 8, 16, 32}, we:
    1. prompt with 4 GSM8K-style demos
    2. greedy-generate up to 256 tokens at temperature 0
    3. regex-extract the final number after "####" or in the last sentence
    4. exact-match against the ground-truth answer

Inputs (env):
    CKPT      path to full-state-dict checkpoint
    OUT       output JSON path
    DEPTHS    comma-separated K values (default: 4,8,16,32)
    LIMIT     number of GSM8K test problems (default: 100)
    MAX_NEW   max generated tokens (default: 256)
    SEED      RNG seed for problem subsetting (default: 1234)
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


FEWSHOT = """Solve each math word problem step by step. Show your work, then give the final numeric answer after "####".

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: She has 16 - 3 - 4 = 9 eggs left to sell. She makes 9 * 2 = 18 dollars.
#### 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: White fiber = 2 / 2 = 1 bolt. Total = 2 + 1 = 3 bolts.
#### 3

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: New value = 80000 * (1 + 1.5) = 200000. Cost = 80000 + 50000 = 130000. Profit = 200000 - 130000 = 70000.
#### 70000

Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
Answer: Sprints per week = 3 * 3 = 9. Total meters = 9 * 60 = 540.
#### 540
"""


HASH_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def normalize_num(s):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return None


def extract_answer(generated_text):
    """Pick #### N if present, otherwise last number in continuation."""
    m = HASH_RE.search(generated_text)
    if m:
        return normalize_num(m.group(1))
    nums = NUM_RE.findall(generated_text)
    if nums:
        return normalize_num(nums[-1])
    return None


def extract_gold(answer_field):
    """GSM8K gold answer is in the form 'reasoning... #### final'."""
    m = HASH_RE.search(answer_field)
    if m:
        return normalize_num(m.group(1))
    return None


def build_prompt(question):
    return f"{FEWSHOT}\nQuestion: {question}\nAnswer:"


def main():
    ckpt_path = os.environ.get("CKPT", "")
    out_json = os.environ.get("OUT", "/home/alex/OpenMythos/docs/eval_gsm8k_generation.json")
    depths = [int(d) for d in os.environ.get("DEPTHS", "4,8,16,32").split(",")]
    limit = int(os.environ.get("LIMIT", "100"))
    max_new = int(os.environ.get("MAX_NEW", "256"))
    seed = int(os.environ.get("SEED", "1234"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))

    if not ckpt_path or not Path(ckpt_path).exists():
        raise SystemExit(f"CKPT not set or missing: {ckpt_path!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")
    logger.info(f"depths={depths}  limit={limit}  max_new={max_new}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_step = ckpt.get("step", "?")
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)

    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max  # keep at trained; LoRA clamps for K > max

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device).eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success(f"loaded step {ckpt_step}  saved_T_max={saved_T_max}")

    logger.info("loading GSM8K test split")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:limit]
    problems = [(ds[i]["question"], extract_gold(ds[i]["answer"])) for i in indices]
    problems = [p for p in problems if p[1] is not None]
    logger.info(f"using {len(problems)} GSM8K problems")

    raw = []
    by_K = {K: [0, 0] for K in depths}

    # Incremental save: a single K loop can take 30-60 min and the gpufarm
    # timeout has killed prior runs mid-loop, losing all completed K passes.
    # Write a partial payload after each K so the next-tick output_valid
    # check sees real data even on timeout.
    def write_payload(complete):
        results = {
            f"K{K}": {"acc": by_K[K][0] / max(1, by_K[K][1]), "n": by_K[K][1]}
            for K in depths
        }
        payload = {
            "ckpt_path": ckpt_path,
            "step": ckpt_step,
            "depths": depths,
            "limit": limit,
            "max_new": max_new,
            "results": results,
            "complete": complete,
            "raw": raw,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    for K in depths:
        logger.info(f"=== K={K} ===")
        t0 = time.perf_counter()
        skipped = 0
        for question, gold in problems:
            prompt = build_prompt(question)
            ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            if ids.size(1) + max_new > seq_len:
                skipped += 1
                continue
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.generate(
                    ids,
                    max_new_tokens=max_new,
                    n_loops=K,
                    temperature=1.0,
                    top_k=1,
                )
            full = tokenizer.decode(out[0].tolist())
            cont = full[len(prompt):]
            pred = extract_answer(cont)
            correct = pred is not None and pred == gold
            by_K[K][1] += 1
            if correct:
                by_K[K][0] += 1
            raw.append({
                "K": K,
                "question": question,
                "gold": gold,
                "predicted": pred,
                "correct": bool(correct),
                "continuation": cont[:400],
            })
        elapsed = time.perf_counter() - t0
        logger.info(f"K={K}: {by_K[K][0]}/{by_K[K][1]} acc={by_K[K][0]/max(1,by_K[K][1]):.3f}  ({elapsed:.0f}s, skipped={skipped})")
        write_payload(complete=False)
        logger.info(f"wrote partial {out_json}  K_done={K}")

    payload = write_payload(complete=True)
    logger.success(f"wrote {out_json}")
    results = payload["results"]

    print()
    print("GSM8K generation accuracy by K:")
    for K in depths:
        print(f"  K={K:>3d}  acc={results[f'K{K}']['acc']:.3f}  n={results[f'K{K}']['n']}")


if __name__ == "__main__":
    main()
