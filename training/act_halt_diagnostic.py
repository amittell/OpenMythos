#!/usr/bin/env python3
"""
Diagnostic: log per-iteration ACT halting probabilities to find out *why* the
round-2 model halts at iteration 1 for every token.

Two hypotheses:

    (a) Trivial halt — model has learned p_1 ≈ 1.0 at iter 1 regardless of
        input. Variable-T training rewarded immediate halting because the
        ACT-weighted output h_out converges in one step under the trained
        distribution. Architecturally uninteresting; needs training-regime
        change (raised threshold, harder data) to fix.

    (b) Threshold too low — model emits modest p (e.g. 0.6–0.9) at iter 1
        but threshold is 0.5, so it crosses immediately even though it would
        benefit from more iterations. Architecturally fine; just need to
        bump the threshold.

This script patches RecurrentBlock.forward to record p_t (the per-iteration
halting probability emitted by ACTHalting) for every (batch, position, t)
triple, runs forward on a small held-out batch, and reports the distribution
of p_t at each iteration.

Output:
    docs/act_halt_diagnostic_round2.json   raw per-iteration stats
    stdout                                 summary table

If `mean(p_1) >= act_threshold`, the model has trivially learned to halt at
iteration 1 → hypothesis (a). If `mean(p_1) << act_threshold` and the model
still halts at iter 1, dig further (something else is hitting threshold).
"""

from __future__ import annotations

import json
import os
import sys
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


# Each forward pushes one (T_max, B*Tseq) tensor of p values plus one
# matching cumulative_p tensor. Cleared between depth measurements.
_collected_p: list[torch.Tensor] = []
_collected_cumulative: list[torch.Tensor] = []


def recurrent_forward_logging_p(
    self: RecurrentBlock,
    h: torch.Tensor,
    e: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask=None,
    n_loops: int | None = None,
    kv_cache=None,
) -> torch.Tensor:
    """
    Drop-in replacement that records p_t and cumulative_p_t for every
    iteration. Otherwise mirrors the original forward (including the ACT
    halting and weighted h_out accumulation).
    """
    n_loops = n_loops or self.cfg.max_loop_iters
    B, T, D = h.shape
    halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
    cumulative_p = torch.zeros(B, T, device=h.device)
    h_out = torch.zeros_like(h)
    p_log = torch.zeros(n_loops, B, T, device=h.device)
    cum_log = torch.zeros(n_loops, B, T, device=h.device)

    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h = self.injection(h, e, trans_out)

        p = self.act(h)
        p_log[t] = p.detach()
        still_running = ~halted
        remainder = (1.0 - cumulative_p).clamp(min=0)
        weight = torch.where(
            cumulative_p + p >= self.cfg.act_threshold, remainder, p
        )
        weight = weight * still_running.float()
        h_out = h_out + weight.unsqueeze(-1) * h
        cumulative_p = cumulative_p + p * still_running.float()
        cum_log[t] = cumulative_p.detach()
        halted = halted | (cumulative_p >= self.cfg.act_threshold)

    _collected_p.append(p_log.float().cpu())
    _collected_cumulative.append(cum_log.float().cpu())
    return h_out


def build_batches(tokenizer, seq_len, n_batches, batch_size):
    """Same held-out FineWeb-Edu window as depth_extrap.py."""
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


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/act_halt_diagnostic_round2.json",
    )
    seq_len = 1024
    batch_size = 4
    n_batches = 4  # smaller — per-iteration tensors are heavy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)
    saved_act_threshold = getattr(saved_cfg, "act_threshold", 0.99)

    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max
    cfg.act_threshold = saved_act_threshold

    logger.info(
        f"trained_max_loop_iters={saved_T_max}  act_threshold={saved_act_threshold}"
    )

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success("Model loaded")

    batches = build_batches(tokenizer, seq_len, n_batches, batch_size)
    logger.success(f"built {len(batches)} batches")

    logger.info("monkey-patching RecurrentBlock.forward to record p_t")
    original = RecurrentBlock.forward
    RecurrentBlock.forward = recurrent_forward_logging_p

    results = {}
    try:
        for K in DEPTHS:
            _collected_p.clear()
            _collected_cumulative.clear()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for x in batches:
                    x = x.to(device)
                    _ = model(x, n_loops=K)
            # _collected_p[i] has shape (K, B, T) for each forward call
            # stack along last dim, average over per-iter populations
            stacked = torch.cat(
                [pt.reshape(K, -1) for pt in _collected_p], dim=1
            )  # (K, total_tokens)
            cum_stacked = torch.cat(
                [pt.reshape(K, -1) for pt in _collected_cumulative], dim=1
            )
            per_iter = []
            for t in range(K):
                p_t = stacked[t]
                cum_t = cum_stacked[t]
                per_iter.append(
                    {
                        "iter": t + 1,
                        "p_mean": float(p_t.mean()),
                        "p_median": float(p_t.median()),
                        "p_min": float(p_t.min()),
                        "p_max": float(p_t.max()),
                        "p_std": float(p_t.std()),
                        "cum_p_mean": float(cum_t.mean()),
                        "cum_p_median": float(cum_t.median()),
                        "frac_above_threshold": float(
                            (cum_t >= saved_act_threshold).float().mean()
                        ),
                    }
                )
            results[K] = per_iter
            logger.success(
                f"K={K}  iter1: p_mean={per_iter[0]['p_mean']:.4f}  "
                f"p_median={per_iter[0]['p_median']:.4f}  "
                f"p_min={per_iter[0]['p_min']:.4f}  "
                f"p_max={per_iter[0]['p_max']:.4f}"
            )
    finally:
        RecurrentBlock.forward = original

    payload = {
        "ckpt_path": ckpt_path,
        "act_threshold": saved_act_threshold,
        "trained_max_loop_iters": saved_T_max,
        "depths": DEPTHS,
        "results": {str(k): v for k, v in results.items()},
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    logger.success(f"wrote {out_json}")

    # Stdout summary
    print()
    print(f"act_threshold = {saved_act_threshold}")
    print(f"trained max_loop_iters = {saved_T_max}")
    print()
    print("Per-iteration halting probability stats at each depth K:")
    for K in DEPTHS:
        print()
        print(f"  K={K}")
        print(f"    {'iter':>4}  {'p_mean':>8}  {'p_median':>8}  "
              f"{'p_min':>8}  {'p_max':>8}  {'cum_p':>8}  {'%>=thr':>7}")
        for r in results[K]:
            print(
                f"    {r['iter']:>4}  {r['p_mean']:>8.4f}  {r['p_median']:>8.4f}  "
                f"{r['p_min']:>8.4f}  {r['p_max']:>8.4f}  {r['cum_p_mean']:>8.4f}  "
                f"{r['frac_above_threshold']:>7.3f}"
            )

    print()
    p1_mean = results[DEPTHS[0]][0]["p_mean"]
    if p1_mean >= saved_act_threshold * 0.95:
        print(f"DIAGNOSIS: trivial halt (hypothesis a)")
        print(f"  p_mean at iter 1 = {p1_mean:.4f} >= 0.95 * act_threshold")
        print(f"  Model has learned to emit near-1.0 halting probability immediately.")
        print(f"  Variable-T training rewarded immediate convergence under the trained")
        print(f"  distribution. Need: raised threshold + harder data for round 2.1.")
    elif p1_mean < 0.5:
        print(f"DIAGNOSIS: threshold too low (hypothesis b)")
        print(f"  p_mean at iter 1 = {p1_mean:.4f} but cumulative crosses {saved_act_threshold}")
        print(f"  somehow. Inspect cumulative_p evolution.")
    else:
        print(f"DIAGNOSIS: intermediate")
        print(f"  p_mean at iter 1 = {p1_mean:.4f}, threshold = {saved_act_threshold}")
        print(f"  Model emits modest halting probability that crosses threshold")
        print(f"  combined across iterations. Threshold tuning may help.")


if __name__ == "__main__":
    main()
