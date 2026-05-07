#!/usr/bin/env python3
"""
Multiple-choice reasoning evaluation at varying recurrent-depth K.

Loads a single full-state-dict checkpoint and computes accuracy on three
multiple-choice probes at K in {4, 8, 16, 32, 64}:

    ARC-Easy        general-knowledge science multiple choice
    ARC-Challenge   harder science multiple choice
    HellaSwag       commonsense sentence completion

For each question, we tokenize `context + choice` for every choice,
sum log-probabilities of the choice tokens conditional on the context,
length-normalize by the number of choice tokens, and predict the argmax.
Length normalization matches the lm-evaluation-harness `acc_norm` metric.

Why this evaluation: the bypass-trained backbone is depth-flat on
language modelling cross-entropy (round 2.1 §7.2), but Geiping et al.
(2025) report that math and code tasks continue to improve past
training depth even when language tasks saturate. We test whether
reasoning probes show the same task-dependence at our scale.

Inputs (env):
    CKPT       path to the full-state-dict checkpoint
    OUT        output JSON path
    DATASETS   comma-separated subset of {arc-easy, arc-challenge, hellaswag}
               (default: all three)
    DEPTHS     comma-separated K values (default: 4,8,16,32,64)
    LIMIT      max questions per task (default: 500)

Outputs JSON with shape:
    {
      "ckpt_path": ...,
      "step": ...,
      "depths": [...],
      "results": {
        "arc-easy":      {"4": {"acc": 0.31, "n": 500}, "8": ..., ...},
        "arc-challenge": ...,
        "hellaswag":     ...
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


def arc_iter(split_name: str, limit: int):
    """Yield (context, choices, gold_idx) for ARC-Easy / ARC-Challenge."""
    ds = load_dataset("allenai/ai2_arc", split_name, split="validation")
    n = 0
    for ex in ds:
        if n >= limit:
            break
        question = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        gold_label = ex["answerKey"]
        if gold_label not in labels:
            continue
        gold_idx = labels.index(gold_label)
        context = f"Question: {question}\nAnswer:"
        choices = [f" {t}" for t in texts]
        yield context, choices, gold_idx
        n += 1


def hellaswag_iter(limit: int):
    """Yield (context, choices, gold_idx) for HellaSwag."""
    ds = load_dataset("hellaswag", split="validation")
    n = 0
    for ex in ds:
        if n >= limit:
            break
        ctx = (ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        endings = [" " + e for e in ex["endings"]]
        gold = int(ex["label"]) if ex["label"] != "" else None
        if gold is None:
            continue
        yield ctx, endings, gold
        n += 1


TASKS = {
    "arc-easy": ("ARC-Easy", lambda lim: arc_iter("ARC-Easy", lim)),
    "arc-challenge": ("ARC-Challenge", lambda lim: arc_iter("ARC-Challenge", lim)),
    "hellaswag": ("HellaSwag", lambda lim: hellaswag_iter(lim)),
}


def score_one(
    model,
    tokenizer,
    context: str,
    choices: list[str],
    n_loops: int,
    device: str,
    max_seq_len: int,
) -> int:
    """
    Return the index of the highest-scoring choice under length-normalized
    conditional log-likelihood.

    Each choice c is scored as:
        sum_{i in choice_tokens} log P(t_i | context, t_<i)  /  len(choice_tokens)

    All choices share the same context, so we tokenize the full sequence
    `context + choice` per choice and slice out the choice-token positions
    after a single forward pass per choice.
    """
    ctx_ids = tokenizer.encode(context)
    best_idx = -1
    best_score = float("-inf")
    for ci, choice in enumerate(choices):
        full_ids = tokenizer.encode(context + choice)
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[-max_seq_len:]
        # Choice tokens are everything after the context boundary; we use
        # token-count in the choice to be robust against re-tokenization
        # at the join boundary (context + " word" vs context + "word").
        ch_ids = tokenizer.encode(choice)
        n_choice_tokens = max(1, min(len(ch_ids), len(full_ids) - 1))

        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, n_loops=n_loops)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # Choice tokens are the last n_choice_tokens positions of y.
        gathered = log_probs[0, -n_choice_tokens:].gather(
            -1, y[0, -n_choice_tokens:].unsqueeze(-1)
        ).squeeze(-1)
        score = float(gathered.sum().item()) / n_choice_tokens

        if score > best_score:
            best_score = score
            best_idx = ci
    return best_idx


def evaluate_task(
    model,
    tokenizer,
    task_key: str,
    depths: list[int],
    limit: int,
    device: str,
    max_seq_len: int,
) -> dict:
    """Evaluate a single task across all depths; return per-K accuracy."""
    label, iter_fn = TASKS[task_key]
    out = {}
    # Cache examples once so we evaluate the same set at every K.
    examples = list(iter_fn(limit))
    logger.info(f"{label}: {len(examples)} examples cached")
    for K in depths:
        t0 = time.perf_counter()
        correct = 0
        total = 0
        for ctx, choices, gold in examples:
            pred = score_one(
                model, tokenizer, ctx, choices, K, device, max_seq_len
            )
            correct += int(pred == gold)
            total += 1
        elapsed = time.perf_counter() - t0
        acc = correct / total if total else 0.0
        out[str(K)] = {
            "acc": acc,
            "n": total,
            "correct": correct,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info(
            f"  K={K:>2}  acc={acc:.4f} ({correct}/{total})  {elapsed:.1f}s"
        )
    return out


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_varT_act_v2/step_0003051_full.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/reasoning_eval_round21.json",
    )
    datasets_arg = os.environ.get("DATASETS", "arc-easy,arc-challenge,hellaswag")
    depths_arg = os.environ.get("DEPTHS", "4,8,16,32,64")
    limit = int(os.environ.get("LIMIT", "500"))

    task_keys = [t.strip() for t in datasets_arg.split(",") if t.strip()]
    depths = [int(d) for d in depths_arg.split(",") if d.strip()]
    for tk in task_keys:
        if tk not in TASKS:
            logger.error(f"unknown task: {tk}; valid: {list(TASKS)}")
            sys.exit(2)

    seq_len = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device} ckpt={ckpt_path}")
    logger.info(f"tasks={task_keys} depths={depths} limit/task={limit}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)
    vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    step = ckpt.get("step", "unknown")

    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max
    logger.info(f"max_loop_iters={saved_T_max} (from saved cfg)")

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success("model loaded")

    # Warmup so the first measured forward isn't penalized by lazy init.
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        warmup_x = torch.randint(0, vocab_size, (1, 32), device=device)
        _ = model(warmup_x, n_loops=4)

    results = {}
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for tk in task_keys:
            label = TASKS[tk][0]
            logger.info(f"=== {label} ===")
            results[tk] = evaluate_task(
                model, tokenizer, tk, depths, limit, device, seq_len
            )

    payload = {
        "ckpt_path": ckpt_path,
        "step": step,
        "vocab_size": vocab_size,
        "trained_max_loop_iters": saved_T_max,
        "depths": depths,
        "limit_per_task": limit,
        "results": results,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.success(f"wrote {out_path}")

    # Stdout summary table
    print()
    print(f"Reasoning eval at step {step}, K-vs-accuracy:")
    print()
    header = f"{'task':<20}" + "".join(f"  K={K:<4}" for K in depths)
    print(header)
    print("-" * len(header))
    for tk in task_keys:
        row = f"{TASKS[tk][0]:<20}"
        for K in depths:
            row += f"  {results[tk][str(K)]['acc']:.4f}"
        print(row)


if __name__ == "__main__":
    main()
