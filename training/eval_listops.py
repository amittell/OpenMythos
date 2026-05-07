#!/usr/bin/env python3
"""
ListOps depth-graded reasoning probe at varying recurrent-depth K.

Synthetic, depth-graded by tree-nesting (Nangia & Bowman 2018 / Long Range
Arena). Bracketed integer ops [MIN 1 [MAX 4 7] 2] with single-digit (0-9)
output. Tree depth = nesting depth = computation depth required to reduce.

For each ckpt, we measure accuracy at K in {4, 8, 16, 32} on problems of
tree-depths {3, 5, 7, 10}, n_per_depth=100, with 6 few-shot demonstrations
in context. The model generates greedily; we score the first digit it
emits against the ground-truth reduction.

Inputs (env):
    CKPT      path to full-state-dict checkpoint
    OUT       output JSON path
    DEPTHS    comma-separated K values (default: 4,8,16,32)
    TREE_DS   comma-separated tree depths (default: 3,5,7,10)
    N         problems per (K, tree depth) (default: 100)
    SEED      RNG seed (default: 1234)
    MAX_NEW   max generated tokens to scan for the answer digit (default: 8)
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
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


OPS = ("MIN", "MAX", "MED", "SUM_MOD")


def reduce_op(op, args):
    if op == "MIN":
        return min(args)
    if op == "MAX":
        return max(args)
    if op == "MED":
        s = sorted(args)
        return s[len(s) // 2]
    if op == "SUM_MOD":
        return sum(args) % 10
    raise ValueError(op)


def gen_expr(rng, depth, n_args_range=(2, 4)):
    """Generate a ListOps expression of exactly `depth` nesting depth."""
    if depth == 1:
        op = rng.choice(OPS)
        n_args = rng.randint(*n_args_range)
        args = [rng.randint(0, 9) for _ in range(n_args)]
        target = reduce_op(op, args)
        body = " ".join(str(a) for a in args)
        return f"[ {op} {body} ]", target

    op = rng.choice(OPS)
    n_args = rng.randint(*n_args_range)
    deeper_idx = rng.randint(0, n_args - 1)

    arg_strs = []
    arg_vals = []
    for i in range(n_args):
        if i == deeper_idx:
            sub_str, sub_val = gen_expr(rng, depth - 1, n_args_range)
        elif rng.random() < 0.4 and depth > 2:
            sub_str, sub_val = gen_expr(rng, rng.randint(1, depth - 1), n_args_range)
        else:
            sub_val = rng.randint(0, 9)
            sub_str = str(sub_val)
        arg_strs.append(sub_str)
        arg_vals.append(sub_val)

    target = reduce_op(op, arg_vals)
    body = " ".join(arg_strs)
    return f"[ {op} {body} ]", target


FEWSHOT = """Solve the bracketed expression. Operators: MIN, MAX, MED (median),
SUM_MOD (sum mod 10). Output only the final single digit (0-9).

Input: [ MAX 3 [ MIN 1 4 ] ]
Output: 4

Input: [ SUM_MOD 5 [ MAX 1 8 ] [ MIN 9 2 ] ]
Output: 5

Input: [ MED 2 7 4 8 1 ]
Output: 4

Input: [ MIN [ MAX 6 2 ] [ SUM_MOD 7 8 ] ]
Output: 5

Input: [ MAX 1 [ MED 9 3 5 7 ] [ MIN 8 4 ] ]
Output: 5

Input: [ SUM_MOD [ MIN 6 9 ] [ MAX 2 5 ] [ MED 1 7 3 ] ]
Output: 4
"""


def build_prompt(expr):
    return f"{FEWSHOT}\nInput: {expr}\nOutput:"


DIGIT_RE = re.compile(r"\b(\d)\b")


def parse_answer(generated_text):
    m = DIGIT_RE.search(generated_text)
    return m.group(1) if m else None


def main():
    ckpt_path = os.environ.get("CKPT", "")
    out_json = os.environ.get("OUT", "/home/alex/OpenMythos/docs/eval_listops.json")
    depths = [int(d) for d in os.environ.get("DEPTHS", "4,8,16,32").split(",")]
    tree_depths = [int(d) for d in os.environ.get("TREE_DS", "3,5,7,10").split(",")]
    n_per_depth = int(os.environ.get("N", "100"))
    seed = int(os.environ.get("SEED", "1234"))
    max_new = int(os.environ.get("MAX_NEW", "8"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))

    if not ckpt_path or not Path(ckpt_path).exists():
        raise SystemExit(f"CKPT not set or missing: {ckpt_path!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")
    logger.info(f"depths={depths}  tree_depths={tree_depths}  N={n_per_depth}")

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

    rng = random.Random(seed)
    problems = []
    for d in tree_depths:
        for _ in range(n_per_depth):
            expr, target = gen_expr(rng, d)
            problems.append((d, expr, target))
    logger.info(f"generated {len(problems)} problems")

    raw = []
    by_K = {K: {d: [0, 0] for d in tree_depths} for K in depths}

    for K in depths:
        logger.info(f"=== K={K} ===")
        t0 = time.perf_counter()
        for tree_d, expr, target in problems:
            prompt = build_prompt(expr)
            ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            if ids.size(1) + max_new > seq_len:
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
            pred = parse_answer(cont)
            correct = pred is not None and int(pred) == target
            by_K[K][tree_d][1] += 1
            if correct:
                by_K[K][tree_d][0] += 1
            raw.append({
                "K": K,
                "tree_depth": tree_d,
                "expr": expr,
                "target": int(target),
                "predicted": pred,
                "correct": bool(correct),
            })
        elapsed = time.perf_counter() - t0
        logger.info(f"K={K} done in {elapsed:.1f}s")

    results = {}
    for K in depths:
        per_d = {}
        ok_total, n_total = 0, 0
        for d in tree_depths:
            ok, n = by_K[K][d]
            per_d[f"d{d}"] = ok / n if n else float("nan")
            ok_total += ok
            n_total += n
        per_d["overall"] = ok_total / n_total if n_total else float("nan")
        results[f"K{K}"] = per_d

    payload = {
        "ckpt_path": ckpt_path,
        "step": ckpt_step,
        "depths": depths,
        "tree_depths": tree_depths,
        "n_per_depth": n_per_depth,
        "seed": seed,
        "results": results,
        "raw": raw,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.success(f"wrote {out_json}")

    print()
    print("ListOps accuracy by K and tree depth:")
    print(f"  {'':6s}", "  ".join(f"d{d:>2d}" for d in tree_depths), "  overall")
    for K in depths:
        cells = "  ".join(f"{results[f'K{K}'][f'd{d}']:.3f}" for d in tree_depths)
        print(f"  K={K:>3d}  {cells}  {results[f'K{K}']['overall']:.3f}")


if __name__ == "__main__":
    main()
