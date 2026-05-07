#!/usr/bin/env python3
"""
Post-process round-2 results: produce a side-by-side comparison markdown
against round 1, and append a results section to the cluster journal.

Inputs:
    docs/depth_extrap_results.json   round 1 measurements
    docs/depth_extrap_round2.json    round 2 measurements
    /tmp/train_r0.log                round 2 training log (for duration + final loss)

Outputs:
    docs/round1_vs_round2.md         side-by-side comparison
    docs/first_cluster_training_run.md   appended round-2 results section

Run with no args; relies on default paths.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/alexm/OpenMythos")
ROUND1_JSON = ROOT / "docs" / "depth_extrap_results.json"
ROUND2_JSON = ROOT / "docs" / "depth_extrap_round2.json"
COMPARE_MD = ROOT / "docs" / "round1_vs_round2.md"
JOURNAL_MD = ROOT / "docs" / "first_cluster_training_run.md"
TRAIN_LOG = Path("/tmp/train_r0.log")


def fmt_loss(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def fmt_ppl(x):
    if not isinstance(x, (int, float)):
        return str(x)
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


def fineweb_table(r1_results, r2_results):
    """
    Round 1 JSON had a flat `results` key (mix of ACT-on then ACT-off).
    Round 2 JSON has `results_fineweb` with the same structure.
    """
    by_act_K = {}
    for r in r1_results:
        by_act_K[(r["act"], r["n_loops"])] = ("r1", r["loss"], r["ppl"])
    for r in r2_results:
        prev = by_act_K.get((r["act"], r["n_loops"]))
        by_act_K[(r["act"], r["n_loops"])] = (
            (prev or ("r1", None, None))[:1] + (prev or ("", None, None))[1:],
            r["loss"],
            r["ppl"],
        )

    lines = ["| ACT | K | round 1 loss | round 1 ppl | round 2 loss | round 2 ppl | delta loss |",
             "|-----|---|--------------|-------------|--------------|-------------|------------|"]
    r1_map = {(r["act"], r["n_loops"]): r for r in r1_results}
    r2_map = {(r["act"], r["n_loops"]): r for r in r2_results}
    keys = sorted(set(r1_map) | set(r2_map), key=lambda k: (k[0], k[1]))
    for act, K in keys:
        r1 = r1_map.get((act, K))
        r2 = r2_map.get((act, K))
        l1 = r1["loss"] if r1 else None
        p1 = r1["ppl"] if r1 else None
        l2 = r2["loss"] if r2 else None
        p2 = r2["ppl"] if r2 else None
        delta = (l2 - l1) if (l1 is not None and l2 is not None) else None
        delta_s = f"{delta:+.4f}" if delta is not None else "-"
        lines.append(
            f"| {act} | {K} | {fmt_loss(l1) if l1 is not None else '-'} | "
            f"{fmt_ppl(p1) if p1 is not None else '-'} | "
            f"{fmt_loss(l2) if l2 is not None else '-'} | "
            f"{fmt_ppl(p2) if p2 is not None else '-'} | {delta_s} |"
        )
    return "\n".join(lines)


def gsm8k_table(r2_gsm8k):
    if not r2_gsm8k:
        return "(round 1 had no GSM8K probe; round 2 also missing this section.)"
    lines = ["| K | answer-only loss | ppl | tokens measured |",
             "|---|------------------|-----|-----------------|"]
    for r in r2_gsm8k:
        lines.append(
            f"| {r['n_loops']} | {fmt_loss(r['loss'])} | {fmt_ppl(r['ppl'])} | "
            f"{r.get('answer_tokens_measured', '-')} |"
        )
    return "\n".join(lines)


def parse_training_stats(log_path: Path):
    """
    Scan the training log for first/last step timestamp, total steps, and
    minimum logged loss. Used to fill in the journal section.
    """
    if not log_path.exists():
        return {}
    first_ts = last_ts = None
    last_step = 0
    min_loss = float("inf")
    min_loss_step = 0
    min_loss_T = 0
    step_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\..*step\s+(\d+)/\d+\s+\| loss\s+([\d.]+).*\| T\s*(\d+)"
    )
    with open(log_path) as f:
        for line in f:
            m = step_re.search(line)
            if m:
                if first_ts is None:
                    first_ts = m.group(1)
                last_ts = m.group(1)
                last_step = int(m.group(2))
                loss = float(m.group(3))
                T = int(m.group(4))
                if loss < min_loss:
                    min_loss = loss
                    min_loss_step = last_step
                    min_loss_T = T
    duration = None
    if first_ts and last_ts:
        a = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
        duration = b - a
    return {
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration": duration,
        "last_step": last_step,
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
        "min_loss_T": min_loss_T,
    }


def tinystories_table(r2_ts):
    if not r2_ts:
        return "(round 2 missing TinyStories results.)"
    lines = ["| K | loss | ppl | tokens measured |",
             "|---|------|-----|-----------------|"]
    for r in r2_ts:
        lines.append(
            f"| {r['n_loops']} | {fmt_loss(r['loss'])} | {fmt_ppl(r['ppl'])} | "
            f"{r.get('tokens_measured', '-')} |"
        )
    return "\n".join(lines)


def write_comparison(r1, r2):
    fb_table = fineweb_table(r1.get("results", []), r2.get("results_fineweb", []))
    gsm_table = gsm8k_table(r2.get("results_gsm8k", []))
    ts_table = tinystories_table(r2.get("results_tinystories", []))
    md = [
        "# Round 1 vs Round 2: depth-extrapolation comparison",
        "",
        f"- Round 1 ckpt: `{r1.get('ckpt_path', '?')}` step {r1.get('step', '?')}; "
        f"trained at fixed T = {r1.get('trained_max_loop_iters', 4)}",
        f"- Round 2 ckpt: `{r2.get('ckpt_path', '?')}` step {r2.get('step', '?')}; "
        f"trained at variable T = {r2.get('trained_max_loop_iters', 12)} max",
        "",
        "## FineWeb-Edu held-out CE",
        "",
        fb_table,
        "",
        "## Round 2 GSM8K answer-only CE",
        "",
        gsm_table,
        "",
        "## Round 2 TinyStories CE (general-distribution sanity probe)",
        "",
        ts_table,
        "",
        "## Round 2 generation samples (vs round 1 samples in `gen_samples_round1.txt`)",
        "",
        f"Prompt: `{r2.get('prompt', '?')}`",
        "",
        "```",
        *(f"K={k}: {v}" for k, v in r2.get("generations", {}).items()),
        "```",
        "",
        "Round 1 samples for the same prompt:",
        "",
        "```",
        *(f"K={k}: {v}" for k, v in r1.get("generations", {}).items()),
        "```",
    ]
    COMPARE_MD.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {COMPARE_MD}")


def append_journal(r2, stats):
    if not JOURNAL_MD.exists():
        print(f"WARNING: {JOURNAL_MD} not found; skipping journal update")
        return
    ckpt_path = r2.get("ckpt_path", "?")
    section = [
        "",
        "## Round 2: variable-T training (200M tokens)",
        "",
        f"Started: {stats.get('first_ts', '?')}  ",
        f"Completed: {stats.get('last_ts', '?')}  ",
        f"Wall-clock training: {stats.get('duration', '?')}  ",
        f"Steps: {stats.get('last_step', '?')} / 12207  ",
        f"Final ckpt: `{ckpt_path}` (step {r2.get('step', '?')})  ",
        f"Lowest training loss observed: "
        f"{stats.get('min_loss', float('nan')):.4f} at step {stats.get('min_loss_step', '?')} "
        f"(T={stats.get('min_loss_T', '?')})  ",
        "",
        "Configuration changes vs round 1:",
        "",
        "- Variable T: sampled `T ~ Uniform(2, 12)` per optimizer step (round 1 was fixed T=4)",
        "- LoRA per-loop scale embedding sized to T_MAX=12 so every sampled depth has its own slot",
        "- Sharded checkpoints (`FSDP.SHARDED_STATE_DICT`): each rank writes its own ~11 GB shard",
        "  to local disk. No 45 GB cross-node rsync. Save time dropped from ~7 min to ~25 sec.",
        "- Worker prune fix: `_distribute_checkpoint` now applies `keep_last=3` on each peer.",
        "- Target tokens: 200M (2x round 1)",
        "",
        "Artifacts:",
        "",
        "- `docs/depth_extrap_round2.json` — raw eval numbers (FineWeb-Edu, GSM8K, TinyStories)",
        "- `docs/round1_vs_round2.md` — side-by-side comparison",
        "- `docs/gen_samples_round2.txt` — 8 prompts at mid-depth",
        "- `docs/gen_samples_round2_multidepth.txt` — same prompts at K=2/6/12/24",
        "- `docs/act_halt_histogram_round2.json` — per-token halt-step distribution",
        "- `docs/training_curve.png` — loss vs step (color = T)",
        "- `docs/loss_by_T.png` — loss binned by sampled depth",
        "",
        "To back up the consolidated checkpoint to the qnap NAS (run from local Mac):",
        "",
        "```bash",
        f"scp -3 kebab-spark.lan:{ckpt_path} \\",
        "    alexm@kebabstore.lan:/share/CACHEDEV1_DATA/openmythos_backups/",
        "```",
        "",
    ]
    with open(JOURNAL_MD, "a") as f:
        f.write("\n".join(section))
    print(f"appended round-2 section to {JOURNAL_MD}")


def main():
    if not ROUND2_JSON.exists():
        print(f"ERROR: {ROUND2_JSON} not found", file=sys.stderr)
        sys.exit(1)
    r2 = json.loads(ROUND2_JSON.read_text())
    r1 = (
        json.loads(ROUND1_JSON.read_text())
        if ROUND1_JSON.exists()
        else {"results": [], "generations": {}}
    )
    write_comparison(r1, r2)
    stats = parse_training_stats(TRAIN_LOG)
    append_journal(r2, stats)


if __name__ == "__main__":
    main()
