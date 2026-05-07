#!/usr/bin/env python3
"""
Synthetic depth-dependent task evaluator.

The held-out cross-entropy on FineWeb-Edu, GSM8K, and TinyStories shows
small (≤ 0.1 nat) test-time compute scaling under the anti-collapse head.
That is real but easy to dismiss as noise. This script tries to find a
task where the gap between K=4 and K=32 is large in *capability* terms
rather than nats, by constructing synthetic items whose answer is
recoverable in principle from the input alone, with the depth budget
acting as the bottleneck.

Tasks (each is a multiple-choice probe):

    arithmetic       "{a} + {b} =" with answer choices {true, ±1, ±2, ±10}
    last_word        "The last word of: A B C D is" -> "D"
    nth_word         "The third word of: A B C D is" -> "C"
    copy             "Copy: A B C D ->" -> "A B C D"
    multi_hop        "X has Y. Y is the Z of W. W is the ___" -> "Y"
    successor        "After 7 comes" -> "8"

For each task and each K, we score every (prompt, answer) candidate by
length-normalised log-likelihood of the answer given the prompt, pick
the argmax, and report accuracy. We want to see at least one task where
accuracy meaningfully grows with K.

These are deliberately easy in absolute terms; if a 3.1B model trained
on 250M tokens cannot solve a "third word of: ..." task with infinite
compute, none of the synthetic depth-extrapolation arguments survive.
We are testing whether the gap K=4-vs-K=32 is bridgeable, not whether
the model is good.

Output:
    docs/synthetic_depth_tasks.json
        per (task, K) accuracy and per-item example traces
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


WORDS = [
    "apple", "banana", "carrot", "diamond", "elephant",
    "feather", "garden", "harbor", "island", "jacket",
    "kettle", "lemon", "marble", "needle", "ocean",
    "pencil", "quartz", "river", "saddle", "tunnel",
]


def task_arithmetic(rng: random.Random, n: int) -> list[tuple[str, list[str], int]]:
    """`a + b =` with distractors. Numbers are 0..9 to keep it within reach."""
    items = []
    for _ in range(n):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        true = a + b
        # Distractors: ±1, ±2, ±10. Pick 3 wrong unique values around true.
        candidates = {true + d for d in (-2, -1, 1, 2, 10, -10) if true + d >= 0}
        candidates.discard(true)
        choices = rng.sample(sorted(candidates), 3) + [true]
        rng.shuffle(choices)
        gold = choices.index(true)
        prompt = f"{a} + {b} ="
        choice_strs = [f" {c}" for c in choices]
        items.append((prompt, choice_strs, gold))
    return items


def task_successor(rng: random.Random, n: int) -> list[tuple[str, list[str], int]]:
    """`After {n} comes` task. Tests if the model knows numerical successor."""
    items = []
    for _ in range(n):
        a = rng.randint(0, 8)
        true = a + 1
        candidates = list(range(0, 10))
        candidates.remove(true)
        choices = rng.sample(candidates, 3) + [true]
        rng.shuffle(choices)
        gold = choices.index(true)
        prompt = f"After {a} comes"
        choice_strs = [f" {c}" for c in choices]
        items.append((prompt, choice_strs, gold))
    return items


def task_last_word(rng: random.Random, n: int) -> list[tuple[str, list[str], int]]:
    """`The last word of A B C D is` -> D."""
    items = []
    for _ in range(n):
        words = rng.sample(WORDS, 4)
        true = words[-1]
        distractors = rng.sample([w for w in WORDS if w not in words], 3)
        choices = distractors + [true]
        rng.shuffle(choices)
        gold = choices.index(true)
        prompt = f"The last word of: {' '.join(words)} is"
        choice_strs = [f" {c}" for c in choices]
        items.append((prompt, choice_strs, gold))
    return items


def task_nth_word(rng: random.Random, n: int) -> list[tuple[str, list[str], int]]:
    """`The third word of A B C D is` -> C, etc."""
    nth_names = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    items = []
    for _ in range(n):
        words = rng.sample(WORDS, 4)
        idx = rng.randint(0, 3)
        true = words[idx]
        distractors = [w for w in words if w != true]
        choices = distractors + [true]
        rng.shuffle(choices)
        gold = choices.index(true)
        prompt = f"The {nth_names[idx + 1]} word of: {' '.join(words)} is"
        choice_strs = [f" {c}" for c in choices]
        items.append((prompt, choice_strs, gold))
    return items


def task_multi_hop(rng: random.Random, n: int) -> list[tuple[str, list[str], int]]:
    """`X is Y's parent. Y is Z's parent. Z's grandparent is` -> X."""
    items = []
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
    for _ in range(n):
        chosen = rng.sample(names, 3)
        x, y, z = chosen
        prompt = f"{x} is {y}'s parent. {y} is {z}'s parent. {z}'s grandparent is"
        true = x
        distractors = rng.sample([w for w in names if w not in chosen], 3)
        choices = distractors + [true]
        rng.shuffle(choices)
        gold = choices.index(true)
        choice_strs = [f" {c}" for c in choices]
        items.append((prompt, choice_strs, gold))
    return items


TASKS = {
    "arithmetic": task_arithmetic,
    "successor": task_successor,
    "last_word": task_last_word,
    "nth_word": task_nth_word,
    "multi_hop": task_multi_hop,
}


def score(model, tokenizer, context: str, choices: list[str], n_loops: int, device: str) -> int:
    """Length-normalised log-likelihood scoring; returns argmax choice index."""
    best_idx = -1
    best_score = float("-inf")
    for ci, choice in enumerate(choices):
        full_ids = tokenizer.encode(context + choice)
        ch_ids = tokenizer.encode(choice)
        n_ch = max(1, min(len(ch_ids), len(full_ids) - 1))
        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, n_loops=n_loops)
        logp = F.log_softmax(logits.float(), dim=-1)
        gathered = logp[0, -n_ch:].gather(-1, y[0, -n_ch:].unsqueeze(-1)).squeeze(-1)
        s = float(gathered.sum().item()) / n_ch
        if s > best_score:
            best_score = s
            best_idx = ci
    return best_idx


def evaluate_task(model, tokenizer, items, depths, device) -> dict:
    """Run all items at every depth K; report accuracy."""
    out = {}
    for K in depths:
        t0 = time.perf_counter()
        correct = 0
        sample_traces = []
        for prompt, choices, gold in items:
            pred = score(model, tokenizer, prompt, choices, K, device)
            ok = pred == gold
            correct += int(ok)
            if len(sample_traces) < 4:
                sample_traces.append({
                    "prompt": prompt,
                    "choices": choices,
                    "gold": gold,
                    "pred": pred,
                    "correct": ok,
                })
        elapsed = time.perf_counter() - t0
        out[str(K)] = {
            "acc": correct / len(items),
            "n": len(items),
            "correct": correct,
            "elapsed_s": round(elapsed, 1),
            "samples": sample_traces,
        }
        logger.info(f"  K={K:>2}  acc={correct/len(items):.4f}  ({correct}/{len(items)})  {elapsed:.1f}s")
    return out


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_act_finetune_anti10/step_0001220_full.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/synthetic_depth_tasks_anti10.json",
    )
    depths = [int(d) for d in os.environ.get("DEPTHS", "4,8,16,32,64").split(",")]
    n_per_task = int(os.environ.get("N_PER_TASK", "200"))
    seed = int(os.environ.get("SEED", "1234"))
    seq_len = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}  depths={depths}  n_per_task={n_per_task}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)
    vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    step = ckpt.get("step", "?")

    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success(f"model loaded from step {step}")

    # Build all task items up-front with a fixed seed so the same items run
    # at every K and can be reproduced from the JSON.
    rng = random.Random(seed)
    items_by_task = {name: fn(rng, n_per_task) for name, fn in TASKS.items()}

    results = {}
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for tname, items in items_by_task.items():
            logger.info(f"=== {tname} (n={len(items)}) ===")
            results[tname] = evaluate_task(model, tokenizer, items, depths, device)

    payload = {
        "ckpt_path": ckpt_path,
        "step": step,
        "depths": depths,
        "n_per_task": n_per_task,
        "seed": seed,
        "results": results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    logger.success(f"wrote {out_json}")

    # Stdout summary
    print()
    header = f"{'task':<14}" + "".join(f"  K={K:<3}" for K in depths)
    print(header); print("-" * len(header))
    for tname, by_K in results.items():
        row = f"{tname:<14}"
        for K in depths:
            row += f"  {by_K[str(K)]['acc']:.4f}"
        print(row)


if __name__ == "__main__":
    main()
