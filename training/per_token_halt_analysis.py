#!/usr/bin/env python3
"""
Per-token ACT halt-step analysis on a fine-tuned (or any) ACT model.

The act_halt_histogram.py script reports the *aggregate* halt-step distribution
across all tokens. It cannot tell us whether the model halts adaptively per
token — easy tokens halting early, hard tokens halting late, which is the
intended behaviour of ACT.

This script is the per-token version. For every (batch, position) pair on a
held-out FineWeb-Edu sample, we record:

    halt_step    iteration index at which cumulative_p first crossed the
                 ACT threshold (1-indexed; n_loops = "never halted")
    p_first      the halting probability emitted at iteration 1 (the head's
                 raw prediction, before threshold/remainder bookkeeping)
    token_id     the input token id at this position
    target_id    the next-token target id
    is_correct   whether the model's argmax prediction matches the target

We then bucket halt steps by:

    - Token category: punctuation / function / content / rare, derived from
      a precomputed tokenizer-vocab partition.
    - Position-in-sequence (early / middle / late).
    - Top-1 prediction correctness (correct / incorrect at greedy decoding).

Output JSON:

    docs/per_token_halt_round21_anti10.json
        {
          "ckpt_path": ...,
          "K": 32,
          "n_tokens": ...,
          "summary": { "by_category": {...}, "by_correctness": {...} },
          "examples": {
              "early_halters": [{"token": "the", "halt_step": 1, ...}, ...],
              "late_halters":  [{"token": "RaR3wOrd", "halt_step": 32, ...}, ...]
          }
        }

We only run at one inference K (default 32). Per-token analysis at multiple K
can be reproduced by setting K_LIST.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos.main import (  # noqa: E402
    OpenMythos,
    RecurrentBlock,
    loop_index_embedding,
)
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


# Token-category heuristics. Categories are computed once over the tokenizer
# vocab. We rely only on string properties of the decoded form to keep this
# tokenizer-agnostic; specialised vocabs (BPE merges, byte-level) all decode
# to readable Unicode and we bucket on that.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "this", "that", "these", "those", "it", "its",
    "he", "she", "him", "her", "his", "they", "them", "their", "we", "us",
    "our", "you", "your", "i", "my", "me", "if", "then", "than", "so", "not",
    "no", "yes",
}


def categorise_token(decoded: str) -> str:
    """Map a decoded-token string to one of {punct, function, content, rare}."""
    s = decoded.strip()
    if not s:
        return "punct"
    if all(not ch.isalnum() for ch in s):
        return "punct"
    lowered = s.lower()
    if lowered in FUNCTION_WORDS:
        return "function"
    if any(ch.isdigit() for ch in s):
        return "rare"
    if len(s) <= 2:
        return "rare"
    return "content"


# Module-level capture for the patched recurrent forward.
_collected: dict[str, list[torch.Tensor]] = {
    "halt_step": [],
    "p_first": [],
}


def recurrent_forward_capture(
    self: RecurrentBlock,
    h: torch.Tensor,
    e: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask=None,
    n_loops: int | None = None,
    kv_cache=None,
) -> torch.Tensor:
    """ACT recurrent forward that records halt_step and p_first per (B, T)."""
    n_loops = n_loops or self.cfg.max_loop_iters
    B, T, D = h.shape
    halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
    cumulative_p = torch.zeros(B, T, device=h.device)
    h_out = torch.zeros_like(h)
    halt_step = torch.zeros(B, T, device=h.device, dtype=torch.int64)
    p_first = torch.zeros(B, T, device=h.device, dtype=torch.float32)

    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h = self.injection(h, e, trans_out)

        p = self.act(h)
        if t == 0:
            p_first.copy_(p.detach().float())
        halted_before = halted
        still_running = ~halted
        remainder = (1.0 - cumulative_p).clamp(min=0)
        weight = torch.where(
            cumulative_p + p >= self.cfg.act_threshold, remainder, p
        )
        weight = weight * still_running.float()
        h_out = h_out + weight.unsqueeze(-1) * h
        cumulative_p = cumulative_p + p * still_running.float()
        halted = halted | (cumulative_p >= self.cfg.act_threshold)

        newly_halted = halted & ~halted_before
        halt_step = torch.where(
            newly_halted & (halt_step == 0),
            torch.full_like(halt_step, t + 1),
            halt_step,
        )

    halt_step = torch.where(
        halt_step == 0, torch.full_like(halt_step, n_loops), halt_step
    )
    _collected["halt_step"].append(halt_step.detach().flatten().cpu())
    _collected["p_first"].append(p_first.detach().flatten().cpu())
    return h_out


def build_batches(tokenizer, seq_len, n_batches, batch_size):
    """Same held-out FineWeb-Edu window used by depth_extrap and halt_histogram."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    ).skip(300_000)
    needed = n_batches * batch_size * (seq_len + 1)
    buf = []
    for sample in ds:
        buf.extend(tokenizer.encode(sample["text"]))
        if len(buf) >= needed:
            break
    out_x, out_y = [], []
    stride = seq_len + 1
    for i in range(n_batches):
        rows_x, rows_y = [], []
        for j in range(batch_size):
            start = (i * batch_size + j) * stride
            chunk = buf[start : start + seq_len + 1]
            rows_x.append(chunk[:-1])
            rows_y.append(chunk[1:])
        out_x.append(torch.tensor(rows_x, dtype=torch.long))
        out_y.append(torch.tensor(rows_y, dtype=torch.long))
    return out_x, out_y


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_act_finetune_anti10/step_0001220_full.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/per_token_halt_anti10.json",
    )
    K = int(os.environ.get("K", "32"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    batch_size = int(os.environ.get("BATCH", "4"))
    n_batches = int(os.environ.get("N_BATCHES", "8"))
    n_examples = int(os.environ.get("N_EXAMPLES", "20"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}  K={K}")

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
    logger.info(f"max_loop_iters={saved_T_max}; act_threshold={cfg.act_threshold}")

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success(f"model loaded from step {step}")

    logger.info("building held-out batches")
    xs, ys = build_batches(tokenizer, seq_len, n_batches, batch_size)
    logger.success(f"built {len(xs)} batches of shape {xs[0].shape}")

    logger.info("monkey-patching RecurrentBlock.forward to capture per-token halt info")
    original_forward = RecurrentBlock.forward
    RecurrentBlock.forward = recurrent_forward_capture

    halt_steps_all = []
    p_first_all = []
    token_ids_all = []
    target_ids_all = []
    pred_ids_all = []
    try:
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for x, y in zip(xs, ys):
                _collected["halt_step"].clear()
                _collected["p_first"].clear()
                x_dev = x.to(device)
                logits = model(x_dev, n_loops=K)
                pred = logits.argmax(dim=-1).detach().cpu().flatten()

                # In a normal forward there is one RecurrentBlock invocation
                # per call; concat in case the architecture invokes it more
                # than once.
                halt_step = torch.cat(_collected["halt_step"])
                p_first = torch.cat(_collected["p_first"])
                halt_steps_all.append(halt_step)
                p_first_all.append(p_first)
                token_ids_all.append(x.flatten())
                target_ids_all.append(y.flatten())
                pred_ids_all.append(pred)
    finally:
        RecurrentBlock.forward = original_forward

    halt = torch.cat(halt_steps_all).numpy()
    pf = torch.cat(p_first_all).numpy()
    tok = torch.cat(token_ids_all).numpy()
    tgt = torch.cat(target_ids_all).numpy()
    prd = torch.cat(pred_ids_all).numpy()
    n = len(halt)
    logger.success(f"collected {n:,} (halt_step, p_first, token, target, pred) tuples")

    decoded = [tokenizer.decode([int(t)]) for t in tok]
    categories = [categorise_token(d) for d in decoded]
    correct = (prd == tgt)

    # --- summary by category ---
    cat_summary = {}
    for cat in ("punct", "function", "content", "rare"):
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        h = halt[idx]
        p1 = pf[idx]
        cat_summary[cat] = {
            "n_tokens": len(idx),
            "halt_step_mean": float(h.mean()),
            "halt_step_median": float(sorted(h)[len(h) // 2]),
            "p_first_mean": float(p1.mean()),
            "frac_halt_at_1": float((h == 1).mean()),
            "frac_halt_at_K": float((h == K).mean()),
            "accuracy": float(correct[idx].mean()),
        }

    # --- summary by correctness ---
    correctness_summary = {}
    for label, mask in (("correct", correct), ("incorrect", ~correct)):
        if not mask.any():
            continue
        h = halt[mask]
        p1 = pf[mask]
        correctness_summary[label] = {
            "n_tokens": int(mask.sum()),
            "halt_step_mean": float(h.mean()),
            "halt_step_median": float(sorted(h)[len(h) // 2]),
            "p_first_mean": float(p1.mean()),
            "frac_halt_at_1": float((h == 1).mean()),
            "frac_halt_at_K": float((h == K).mean()),
        }

    # --- example tokens ---
    order = halt.argsort()
    early_idx = order[:n_examples]
    late_idx = order[-n_examples:][::-1]

    def example_row(i):
        return {
            "token": decoded[i],
            "category": categories[i],
            "halt_step": int(halt[i]),
            "p_first": float(pf[i]),
            "target": tokenizer.decode([int(tgt[i])]),
            "pred": tokenizer.decode([int(prd[i])]),
            "correct": bool(correct[i]),
        }

    examples = {
        "early_halters": [example_row(int(i)) for i in early_idx],
        "late_halters": [example_row(int(i)) for i in late_idx],
    }

    # --- raw histogram (all tokens) ---
    halt_counts = Counter(int(h) for h in halt)
    histogram = [halt_counts.get(s, 0) for s in range(0, K + 1)]

    payload = {
        "ckpt_path": ckpt_path,
        "step": step,
        "K": K,
        "act_threshold": cfg.act_threshold,
        "n_tokens": int(n),
        "halt_step_histogram": histogram,
        "halt_step_mean": float(halt.mean()),
        "halt_step_median": float(sorted(halt)[n // 2]),
        "p_first_mean": float(pf.mean()),
        "frac_halt_at_1": float((halt == 1).mean()),
        "frac_halt_at_K": float((halt == K).mean()),
        "by_category": cat_summary,
        "by_correctness": correctness_summary,
        "examples": examples,
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.success(f"wrote {out_path}")

    # Stdout summary table
    print()
    print(f"=== Per-token halt analysis at K={K}, n={n:,} tokens ===")
    print(f"overall mean halt step:   {payload['halt_step_mean']:.2f}")
    print(f"overall median halt step: {payload['halt_step_median']:.0f}")
    print(f"frac halt at iter 1:      {payload['frac_halt_at_1']:.3f}")
    print(f"frac halt at iter K:      {payload['frac_halt_at_K']:.3f}")
    print(f"mean p_1:                 {payload['p_first_mean']:.4f}")
    print()
    print(f"{'category':<10}  {'n':>8}  {'mean':>6}  {'median':>6}  {'p_1':>6}  {'%h=1':>6}  {'%h=K':>6}  {'acc':>6}")
    for cat, s in cat_summary.items():
        print(f"{cat:<10}  {s['n_tokens']:>8,}  {s['halt_step_mean']:>6.2f}  {s['halt_step_median']:>6.0f}  {s['p_first_mean']:>6.4f}  {s['frac_halt_at_1']:>6.3f}  {s['frac_halt_at_K']:>6.3f}  {s['accuracy']:>6.3f}")
    print()
    for label, s in correctness_summary.items():
        print(f"{label:<10}  n={s['n_tokens']:>8,}  mean halt={s['halt_step_mean']:6.2f}  p_1={s['p_first_mean']:.4f}")


if __name__ == "__main__":
    main()
