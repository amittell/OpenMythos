#!/usr/bin/env python3
"""
Generate the paper's §7 (empirical results) fragment for round 2.1.

Reads four artifacts produced by `auto_eval_round21.sh`:

    DEPTH_JSON   docs/depth_extrap_round21.json      from depth_extrap.py
    DIAG_JSON    docs/act_halt_diagnostic_round21.json from act_halt_diagnostic.py
    HIST_JSON    docs/act_halt_histogram_round21.json  from act_halt_histogram.py
    SAMPLES_TXT  docs/gen_samples_round21_multidepth.txt from gen_samples_multidepth.py

Writes a markdown fragment to OUT (default: docs/paper/round21_results.md)
that drops directly into docs/paper/main.md as §7.1 through §7.4.

Round-2 reference numbers (read from docs/round1_vs_round2.md and
docs/depth_extrap_round2.json) are inlined for the comparison table.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


ROUND2_FINEWEB_ACT_OFF = {4: 4.2455, 8: 4.2522, 16: 4.2525, 32: 4.2524}
ROUND2_FINEWEB_ACT_ON = {4: 4.1130, 8: 4.1130, 16: 4.1130, 32: 4.1130}
ROUND2_GSM8K = {4: 4.1348, 8: 4.1348, 16: 4.1348, 32: 4.1348}
ROUND2_TINYSTORIES = {4: 3.8242, 8: 3.8242, 16: 3.8242, 32: 3.8242}


def fmt_loss(x: float) -> str:
    return f"{x:.4f}"


def fmt_ppl(x: float) -> str:
    return f"{x:.2f}"


def ppl(loss: float) -> float:
    return math.exp(loss)


def by_act(rows: list[dict]) -> dict[str, dict[int, dict]]:
    """Bucket depth_extrap rows by act flag, keyed by n_loops."""
    out = {"on": {}, "off": {}}
    for r in rows:
        out[r["act"]][r["n_loops"]] = r
    return out


def fineweb_table(r21_results: list[dict]) -> str:
    by = by_act(r21_results)
    lines = [
        "**FineWeb-Edu held-out cross-entropy at K iterations**, ACT off vs ACT on, "
        "round 2.1 vs round 2:",
        "",
        "| K  | r2.1 ACT-off | r2 ACT-off | Δ (off) | r2.1 ACT-on | r2 ACT-on | Δ (on) |",
        "|----|--------------|------------|---------|-------------|-----------|--------|",
    ]
    for k in (4, 8, 16, 32):
        off = by["off"].get(k, {}).get("loss", float("nan"))
        on = by["on"].get(k, {}).get("loss", float("nan"))
        r2_off = ROUND2_FINEWEB_ACT_OFF[k]
        r2_on = ROUND2_FINEWEB_ACT_ON[k]
        d_off = off - r2_off
        d_on = on - r2_on
        lines.append(
            f"| {k:>2} | {fmt_loss(off)}       | {fmt_loss(r2_off)}     "
            f"| {d_off:+.4f} | {fmt_loss(on)}      | {fmt_loss(r2_on)}    "
            f"| {d_on:+.4f} |"
        )
    return "\n".join(lines)


def gsm8k_table(r21_results: list[dict]) -> str:
    by_K = {r["n_loops"]: r for r in r21_results}
    lines = [
        "**GSM8K answer-only CE** at K iterations, round 2.1 vs round 2:",
        "",
        "| K  | r2.1 loss | r2.1 PPL  | r2 loss | Δ |",
        "|----|-----------|-----------|---------|---|",
    ]
    for k in (4, 8, 16, 32):
        r = by_K.get(k, {})
        loss = r.get("loss", float("nan"))
        p = r.get("ppl", ppl(loss) if loss == loss else float("nan"))
        r2 = ROUND2_GSM8K[k]
        lines.append(
            f"| {k:>2} | {fmt_loss(loss)}    | {fmt_ppl(p)}     "
            f"| {fmt_loss(r2)}  | {loss - r2:+.4f} |"
        )
    return "\n".join(lines)


def tinystories_table(r21_results: list[dict]) -> str:
    by_K = {r["n_loops"]: r for r in r21_results}
    lines = [
        "**TinyStories CE** (general-distribution sanity probe) at K iterations, "
        "round 2.1 vs round 2:",
        "",
        "| K  | r2.1 loss | r2.1 PPL  | r2 loss | Δ |",
        "|----|-----------|-----------|---------|---|",
    ]
    for k in (4, 8, 16, 32):
        r = by_K.get(k, {})
        loss = r.get("loss", float("nan"))
        p = r.get("ppl", ppl(loss) if loss == loss else float("nan"))
        r2 = ROUND2_TINYSTORIES[k]
        lines.append(
            f"| {k:>2} | {fmt_loss(loss)}    | {fmt_ppl(p)}     "
            f"| {fmt_loss(r2)}  | {loss - r2:+.4f} |"
        )
    return "\n".join(lines)


def diag_table(diag: dict) -> str:
    """Per-iteration p_t at K=4, comparing round 2 (saturated) to round 2.1."""
    results = diag["results"]
    threshold = diag.get("act_threshold", 0.99)
    lines = [
        f"**Per-iteration halting probability at K=4** (threshold = {threshold}):",
        "",
        "| iter | p_mean | p_median | p_min  | p_max  | cum_p | %≥thr |",
        "|------|--------|----------|--------|--------|-------|-------|",
    ]
    rows_k4 = results.get("4", [])
    for r in rows_k4:
        lines.append(
            f"| {r['iter']:>4} "
            f"| {r['p_mean']:.4f} | {r['p_median']:.4f} "
            f"| {r['p_min']:.4f} | {r['p_max']:.4f} "
            f"| {r['cum_p_mean']:.4f} | {r['frac_above_threshold']:.3f} |"
        )
    return "\n".join(lines)


def histogram_summary(hist: dict) -> str:
    """One-line summary of the halt-step histogram per K."""
    results = hist.get("results", {})
    lines = [
        "**Halt-step histogram per K**:",
        "",
        "| K  | mean halt step | median halt step | tokens at iter 1 | full budget used |",
        "|----|----------------|------------------|------------------|------------------|",
    ]
    for k in (4, 8, 16, 32):
        r = results.get(str(k), {})
        if not r:
            continue
        bins = r.get("histogram", [])
        at_iter_1 = bins[1] if len(bins) > 1 else 0
        total = r.get("total_tokens", sum(bins) if bins else 0)
        frac_full = r.get("frac_using_full_budget", 0.0)
        lines.append(
            f"| {k:>2} | {r.get('mean_halt_step', float('nan')):.2f} "
            f"          | {r.get('median_halt_step', 0):>4}             "
            f"| {at_iter_1:,} / {total:,} "
            f"   | {frac_full:.3f}            |"
        )
    return "\n".join(lines)


def collapse_verdict(diag: dict) -> str:
    """Has ACT collapse cleared, persisted, or shifted?"""
    p1 = diag["results"]["4"][0]["p_mean"]
    threshold = diag.get("act_threshold", 0.99)
    if p1 >= threshold * 0.95:
        return (
            f"`p_1 = {p1:.4f}` at K=4, still saturated above 0.95 × threshold. "
            "The ACT head was frozen during ACT-bypass training (it received no "
            "gradient), so we did not expect it to recover from the round-2 "
            "saturation. The evaluation with ACT enabled at inference therefore "
            "still produces trivial halting; the meaningful signal is the "
            "ACT-off column of the depth-extrapolation table."
        )
    if p1 < 0.5:
        return (
            f"`p_1 = {p1:.4f}` at K=4, well below threshold. The ACT head has "
            "drifted off the saturated point during round 2.1 despite no direct "
            "gradient, possibly due to weight decay or downstream parameter "
            "shifts that propagate into the head. Inference-time ACT may now "
            "produce useful halting; the ACT-on column is now interpretable."
        )
    return (
        f"`p_1 = {p1:.4f}` at K=4, intermediate. The head has partially "
        "de-saturated. ACT-on inference may produce a non-trivial halting "
        "distribution; treat the ACT-on column with caution and inspect the "
        "histogram for the actual shape of the halt distribution."
    )


def parse_samples(samples_path: Path, max_chars: int = 800) -> str:
    """Take the first generation sample (multi-depth) and clip to max_chars."""
    if not samples_path.exists():
        return "(samples file not found)"
    text = samples_path.read_text()
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    return head + "\n\n[truncated; full output in " + str(samples_path) + "]"


def main() -> None:
    depth_path = Path(os.environ.get("DEPTH_JSON", "docs/depth_extrap_round21.json"))
    diag_path = Path(os.environ.get("DIAG_JSON", "docs/act_halt_diagnostic_round21.json"))
    hist_path = Path(os.environ.get("HIST_JSON", "docs/act_halt_histogram_round21.json"))
    samples_path = Path(os.environ.get("SAMPLES_TXT", "docs/gen_samples_round21_multidepth.txt"))
    step = os.environ.get("STEP", "?")
    out_path = Path(os.environ.get("OUT", "docs/paper/round21_results.md"))

    for p in (depth_path, diag_path, hist_path):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            sys.exit(1)

    depth = json.loads(depth_path.read_text())
    diag = json.loads(diag_path.read_text())
    hist = json.loads(hist_path.read_text())

    fineweb = depth.get("results_fineweb", depth.get("results", []))
    gsm8k = depth.get("results_gsm8k", [])
    tinystories = depth.get("results_tinystories", [])

    fragment = f"""<!--
auto-generated by training/build_paper_section7.py
ckpt: {depth.get('ckpt_path', '?')}, step {step}
inputs: {depth_path.name}, {diag_path.name}, {hist_path.name}, {samples_path.name}
paste this into docs/paper/main.md, replacing the existing §7 stub
-->

## 7. Empirical results (round 2.1, ACT-bypass training)

Round 2.1 continues from the round-2 final checkpoint, with the
recurrent block's forward monkey-patched to run all `T` iterations and
return `h_T` directly (no ACT weighting). All other settings are held
fixed: variable-T sampling `T_step ∼ U{{2, 12}}`, FSDP SHARD_GRAD_OP,
bf16 mixed precision, 50M-token target.

### 7.1 Training curves

Loss vs step and per-T loss bands for round 2.1 are in
`figures/round21_training_curve.png` and `figures/round21_loss_by_T.png`.
The training curve continues from where round 2 left off; the per-T plot
shows whether the model now exploits depth (lower loss at higher T) or
remains depth-flat (the round-2 pattern under collapsed ACT).

### 7.2 Depth extrapolation

{fineweb_table(fineweb)}

The ACT-off column is the primary read on whether the bypass-trained
backbone learned to leverage depth. Round 2 was already depth-flat at
~4.25 nats with ACT off; the question for round 2.1 is whether the
model now improves with K (loss decreasing as iterations grow) or
remains flat. The ACT-on column reflects whatever the (untrained)
halting head does at inference and is interpretable only in light of
§7.3 below.

{gsm8k_table(gsm8k) if gsm8k else "_GSM8K table unavailable_"}

{tinystories_table(tinystories) if tinystories else "_TinyStories table unavailable_"}

### 7.3 ACT halting state after bypass training

Round 2 left the ACT head saturated at `p_t = 1.0` for every iteration
and every token (§5). Round 2.1 bypassed the head during training, so
the head received no direct gradient. The diagnostic output below shows
whether the head stayed saturated or drifted off the fixed point through
indirect parameter shifts.

{diag_table(diag)}

{histogram_summary(hist)}

**Verdict.** {collapse_verdict(diag)}

### 7.4 Generation samples

First multi-depth generation block (full set in
`docs/gen_samples_round21_multidepth.txt`):

```
{parse_samples(samples_path)}
```
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fragment)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
