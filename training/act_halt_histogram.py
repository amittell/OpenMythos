#!/usr/bin/env python3
"""
Measure the per-token ACT halting-step distribution at several depth budgets.

Quantifies whether the model actually USES the loop budget at inference. If
at K=8 every token halts at step 3, ACT is dominating and depth is wasted.
If halt steps spread across [1..K], the model is using depth productively.

Approach:
  - Monkey-patch RecurrentBlock.forward to record, per (B, T) position, the
    iteration index at which `halted` first became True. Tokens that never
    cross the threshold get halt_step = n_loops.
  - Run inference on a fixed held-out batch (same shard as depth_extrap.py)
    at K in [4, 8, 16, 32] with ACT enabled.
  - Aggregate halt_step counts across all tokens and dump per-K histograms.

Output:
    docs/act_halt_histogram_round2.json   raw histograms
    stdout                                 small markdown summary
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
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


DEPTHS = [4, 8, 16, 32]


# Buckets are populated in the patched forward below; the script consumes
# whichever was last filled at the end of each measurement pass.
_collected_halt_steps: list[torch.Tensor] = []


def recurrent_forward_with_halt_log(
    self: RecurrentBlock,
    h: torch.Tensor,
    e: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask=None,
    n_loops: int | None = None,
    kv_cache=None,
) -> torch.Tensor:
    """
    Drop-in replacement for RecurrentBlock.forward that records, for every
    (batch, token) position, the loop iteration at which it first halted.

    Tokens that never cross the threshold get halt_step = n_loops (i.e. used
    the full budget). Halt steps are 1-indexed: halt_step=1 means the token
    halted after the FIRST iteration, halt_step=n_loops means it never did.

    Records via append to the module-level _collected_halt_steps list, with
    one (B*T,) int64 tensor per RecurrentBlock invocation. The caller is
    responsible for consuming and clearing that list between measurements.
    """
    n_loops = n_loops or self.cfg.max_loop_iters
    B, T, D = h.shape
    halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
    cumulative_p = torch.zeros(B, T, device=h.device)
    h_out = torch.zeros_like(h)
    halt_step = torch.zeros(B, T, device=h.device, dtype=torch.int64)  # 0 = unused

    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h = self.injection(h, e, trans_out)

        p = self.act(h)
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
        # 1-indexed halt step: t+1
        halt_step = torch.where(
            newly_halted & (halt_step == 0),
            torch.full_like(halt_step, t + 1),
            halt_step,
        )

    # Tokens that never halted: assign n_loops (used the full budget)
    halt_step = torch.where(
        halt_step == 0, torch.full_like(halt_step, n_loops), halt_step
    )
    _collected_halt_steps.append(halt_step.detach().flatten().cpu())
    return h_out


def build_batches(tokenizer, seq_len, n_batches, batch_size):
    """Reuse the same FineWeb-Edu held-out window as depth_extrap.py."""
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
    out = []
    stride = seq_len + 1
    for i in range(n_batches):
        rows = []
        for j in range(batch_size):
            start = (i * batch_size + j) * stride
            rows.append(buf[start : start + seq_len])
        out.append(torch.tensor(rows, dtype=torch.long))
    return out


def main():
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/act_halt_histogram_round2.json",
    )
    seq_len = 1024
    batch_size = 4
    n_batches = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_step = ckpt.get("step", "?")
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)

    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success(f"Model loaded from step {ckpt_step}  (max_loop_iters={saved_T_max})")

    logger.info("loading FineWeb-Edu held-out batches...")
    batches = build_batches(tokenizer, seq_len, n_batches, batch_size)
    logger.success(f"built {len(batches)} batches")

    logger.info("monkey-patching RecurrentBlock.forward to record halt steps")
    original = RecurrentBlock.forward
    RecurrentBlock.forward = recurrent_forward_with_halt_log

    results = {}
    try:
        for K in DEPTHS:
            _collected_halt_steps.clear()
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for x in batches:
                    x = x.to(device)
                    _ = model(x, n_loops=K)
            elapsed = time.time() - t0
            steps = torch.cat(_collected_halt_steps)
            counts = torch.bincount(steps, minlength=K + 1).tolist()  # 0..K
            mean = float(steps.float().mean().item())
            median = int(steps.median().item())
            total = int(steps.numel())
            results[K] = {
                "histogram": counts,  # index = halt_step (0..K), but 0 should be empty
                "total_tokens": total,
                "mean_halt_step": mean,
                "median_halt_step": median,
                "elapsed_s": round(elapsed, 2),
                "max_budget": K,
                "frac_using_full_budget": counts[K] / total if total else 0.0,
            }
            logger.success(
                f"K={K}  mean={mean:.2f}  median={median}  "
                f"frac_full_budget={results[K]['frac_using_full_budget']:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )
    finally:
        RecurrentBlock.forward = original

    payload = {
        "ckpt_path": ckpt_path,
        "step": ckpt_step,
        "vocab_size": cfg.vocab_size,
        "trained_max_loop_iters": saved_T_max,
        "depths": DEPTHS,
        "results": {str(k): v for k, v in results.items()},
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    logger.success(f"wrote {out_json}")

    # Pretty stdout summary so it lands in /tmp/auto_eval.log
    print()
    print("ACT halt-step distribution at each requested K (FineWeb-Edu held-out):")
    print(f"  trained_max_loop_iters={saved_T_max}")
    print()
    print(f"  {'K':>3}  {'mean':>6}  {'median':>6}  {'frac_full':>9}  histogram (1..K)")
    for K in DEPTHS:
        r = results[K]
        hist = r["histogram"][1:]  # drop the empty halt_step=0 bucket
        # collapse if very long
        if len(hist) <= 16:
            hist_str = " ".join(f"{c:>5}" for c in hist)
        else:
            buckets = []
            chunk = max(1, len(hist) // 8)
            for i in range(0, len(hist), chunk):
                buckets.append(sum(hist[i : i + chunk]))
            hist_str = " ".join(f"{c:>5}" for c in buckets)
        print(
            f"  {K:>3}  {r['mean_halt_step']:>6.2f}  {r['median_halt_step']:>6}  "
            f"{r['frac_using_full_budget']:>9.3f}  {hist_str}"
        )


if __name__ == "__main__":
    main()
