#!/usr/bin/env python3
"""
Pull round 2.6/2.7/2.8 result JSONs from spark/mini-beast and insert
§7.11/§7.12/§7.13 into docs/paper/main.md.

Idempotent: if the section heading already exists in main.md, skips.
Each insertion writes a backup main.md.bak.{round} alongside.

Usage:
    python3 training/auto_paper_integrate.py --round 26
    python3 training/auto_paper_integrate.py --round 27
    python3 training/auto_paper_integrate.py --round 28

Or scan-all:
    python3 training/auto_paper_integrate.py --scan
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/alex/git/OpenMythos")
MAIN_MD = REPO / "docs/paper/main.md"
DOCS = REPO / "docs"
INSERT_BEFORE_HEADER = "## 8. Discussion"

REMOTE_HOSTS = {
    26: "alexm@kebab-spark.lan",
    27: "alexm@kebab-spark.lan",
    28: "alexm@kebab-rtx6000.lan",
    29: "alexm@kebab-spark.lan",
    210: "alexm@kebab-spark.lan",
    211: "alexm@kebab-spark.lan",
    212: "alexm@kebab-spark.lan",
    213: "alexm@kebab-spark.lan",
    214: "alexm@kebab-spark.lan",
}
REMOTE_DOCS = {
    26: "/home/alexm/OpenMythos/docs",
    27: "/home/alexm/OpenMythos/docs",
    28: "/home/alexm/OpenMythos/docs",
    29: "/home/alexm/OpenMythos/docs",
    210: "/home/alexm/OpenMythos/docs",
    211: "/home/alexm/OpenMythos/docs",
    212: "/home/alexm/OpenMythos/docs",
    213: "/home/alexm/OpenMythos/docs",
    214: "/home/alexm/OpenMythos/docs",
}

ROUND_META = {
    26: {
        "section": "### 7.11 Joint training stability past 50M tokens (round 2.6)",
        "compare_round": 23,
        "compare_label": "round 2.3",
        "blurb": (
            "Round 2.6 continues round 2.3 for an additional 50M tokens with identical "
            "hyperparameters (`λ_KL = 1.0`, `λ_p = 0.2`, `REINIT_HEAD = 0` to preserve the trained "
            "head). The question is whether the joint equilibrium continues to gain CE past the "
            "50M-token mark or saturates."
        ),
    },
    27: {
        "section": "### 7.12 Halt-prior sensitivity (round 2.7, λ_p = 0.1)",
        "compare_round": 24,
        "compare_label": "round 2.4",
        "blurb": (
            "Round 2.7 retrains PonderNet-KL joint training from the round-2 collapsed "
            "checkpoint with `λ_p = 0.1` (target mean halt step 10) instead of round 2.4's "
            "`λ_p = 0.2` (target mean step 5). All other hyperparameters and the 50M-token "
            "budget match round 2.4. The question is whether the model's halt distribution "
            "adapts to the longer prior, or whether the head locks at one specific mean "
            "regardless of the targeted prior."
        ),
    },
    28: {
        "section": "### 7.13 Anti-collapse halt-recovery on a PonderNet-rescued backbone (round 2.8)",
        "compare_round": 24,
        "compare_label": "round 2.4 (PonderNet head)",
        "blurb": (
            "Round 2.8 takes the round-2.4 backbone (rescued via PonderNet-KL joint training) "
            "and re-initialises the halting head, then trains the head only with the §7.6 "
            "anti-collapse penalty `λ · mean(p_1²)` at `λ = 10` for 5M tokens (rest of the model "
            "frozen). This combines path A (anti-collapse head) with path B (PonderNet-rescued "
            "backbone), allowing a head-vs-head comparison on the same backbone."
        ),
    },
    29: {
        "section": "### 7.14 No-recurrence baseline at matched compute (round 2.9)",
        "compare_round": 25,
        "compare_label": "round 2.5 (fixed T = 8)",
        "blurb": (
            "Round 2.9 isolates the contribution of the recurrent loop itself. We restart "
            "training from the round-2 collapsed checkpoint with `T_FIXED = 1`: the recurrent "
            "block is applied exactly once per token, no iteration. All other settings match "
            "rounds 2.4 and 2.5 (50M-token continuation, same data, same optimiser). Effective "
            "depth is `prelude (2) + recurrent (1) + coda (2) = 5` transformer blocks. Note: "
            "this is not a pure vanilla transformer baseline because the architecture retains "
            "the LTI injection and prelude/coda split; the comparison is specifically against "
            "test-time-compute scaling at matched parameter count and matched training tokens."
        ),
    },
    210: {
        "section": "### 7.16 Extended joint training to 150M tokens (round 2.10)",
        "compare_round": 26,
        "compare_label": "round 2.6 (100M joint tokens)",
        "blurb": (
            "Round 2.10 continues round 2.6 for an additional 50M tokens of joint "
            "PonderNet-KL training (`λ_KL = 1.0`, `λ_p = 0.2`, `REINIT_HEAD = 0` to preserve "
            "the trained halting head). Total joint training tokens reach 150M. The question is "
            "whether CE loss and halt-head quality continue to improve past 100M joint tokens, "
            "or whether the joint equilibrium has saturated."
        ),
    },
    211: {
        "section": "### 7.17 Compute-scaling: T_FIXED = 2 (round 2.11)",
        "compare_round": 29,
        "compare_label": "round 2.9 (T = 1)",
        "blurb": (
            "Round 2.11 trains with `T_FIXED = 2` (one extra recurrent iteration) from the "
            "round-2 collapsed checkpoint, on a 50M-token budget identical to round 2.9 "
            "(`T = 1`). Together with rounds 2.5 (T = 8), 2.9 (T = 1), 2.12 (T = 4), and 2.14 "
            "(T = 16), this fills the compute scaling curve to test whether deeper recurrence "
            "during training continues to help or saturates."
        ),
    },
    212: {
        "section": "### 7.18 Compute-scaling: T_FIXED = 4 (round 2.12)",
        "compare_round": 211,
        "compare_label": "round 2.11 (T = 2)",
        "blurb": (
            "Round 2.12 trains with `T_FIXED = 4` from the round-2 collapsed checkpoint, "
            "on a 50M-token budget. The third point on the compute-scaling curve "
            "(T = 1, 2, 4, 8, 16). Tests whether the marginal benefit of additional recurrent "
            "iterations during training continues past T = 2 or shows diminishing returns."
        ),
    },
    213: {
        "section": "### 7.19 Joint training to 200M tokens (round 2.13)",
        "compare_round": 210,
        "compare_label": "round 2.10 (150M joint tokens)",
        "blurb": (
            "Round 2.13 continues round 2.10 for an additional 50M tokens of joint "
            "PonderNet-KL training, bringing total joint-training tokens to 200M. This is "
            "the fourth and final point on the joint-token scaling curve "
            "(50M r2.3, 100M r2.6, 150M r2.10, 200M r2.13). Tests whether joint training "
            "continues to improve past 150M tokens or fully saturates."
        ),
    },
    214: {
        "section": "### 7.20 Compute-scaling: T_FIXED = 16 (round 2.14)",
        "compare_round": 25,
        "compare_label": "round 2.5 (T = 8)",
        "blurb": (
            "Round 2.14 extends the compute-scaling curve to `T_FIXED = 16`, twice the depth "
            "of round 2.5's `T = 8`. Tests whether deeper recurrence during training "
            "continues to help past T = 8 or shows diminishing returns at the high end of "
            "the curve."
        ),
    },
}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def fetch_remote(round_n: int) -> bool:
    """rsync the JSON inputs and the auto-generated fragment for this round."""
    host = REMOTE_HOSTS[round_n]
    remote_docs = REMOTE_DOCS[round_n]
    files = [
        f"depth_extrap_round{round_n}.json",
        f"act_halt_diagnostic_round{round_n}.json",
        f"act_halt_histogram_round{round_n}.json",
        f"gen_samples_round{round_n}_multidepth.txt",
        f"paper/round{round_n}_results.md",
    ]
    any_ok = False
    for f in files:
        remote_path = f"{host}:{remote_docs}/{f}"
        local_path = DOCS / f
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["rsync", "-a", remote_path, str(local_path)],
                check=True,
                stderr=subprocess.PIPE,
            )
            if local_path.exists():
                any_ok = True
        except subprocess.CalledProcessError:
            pass
    return any_ok


def have_inputs(round_n: int) -> bool:
    needed = [
        DOCS / f"depth_extrap_round{round_n}.json",
        DOCS / f"act_halt_diagnostic_round{round_n}.json",
        DOCS / f"act_halt_histogram_round{round_n}.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def load_depth_extrap(path: Path) -> tuple[list[dict], list[dict]]:
    with open(path) as f:
        d = json.load(f)
    rows_off = [r for r in d.get("results_fineweb", []) if r.get("act") == "off"]
    rows_on = [r for r in d.get("results_fineweb", []) if r.get("act") == "on"]
    rows_off.sort(key=lambda r: r["n_loops"])
    rows_on.sort(key=lambda r: r["n_loops"])
    return rows_off, rows_on


def load_diagnostic(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_histogram(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def round_label(round_n: int) -> str:
    """Map 26 -> 'r2.6', 27 -> 'r2.7', etc."""
    return f"r2.{round_n - 20}"


def fmt_depth_table(round_n: int, this_off: list, this_on: list, comp_off: list) -> str:
    out = ["**FineWeb-Edu held-out cross-entropy at K iterations:**", ""]
    cmp_label = ROUND_META[round_n]["compare_label"]
    out.append(f"| K  | {round_label(round_n)} ACT-off | {cmp_label} ACT-off | Δ      | r2 ACT-off |")
    out.append("|----|--------------|---------------------|--------|------------|")
    comp_by_K = {r["n_loops"]: r["loss"] for r in comp_off}
    R2_BASELINE_OFF = {4: 4.2455, 8: 4.2522, 16: 4.2525, 32: 4.2524}
    for r in this_off:
        K = r["n_loops"]
        ce = r["loss"]
        cmp_ce = comp_by_K.get(K, float("nan"))
        delta = ce - cmp_ce
        baseline = R2_BASELINE_OFF.get(K, float("nan"))
        out.append(
            f"| {K:>2} | {ce:.4f}       | {cmp_ce:.4f}              | {delta:+.4f} | {baseline:.4f}     |"
        )
    return "\n".join(out)


def fmt_diag_table(diag: dict, k_value: int = 4) -> str:
    """Per-iteration halting probability table at the requested K."""
    out = [
        f"**Per-iteration halting probability at K = {k_value}** (threshold = 0.99):",
        "",
        "| iter | p_mean | p_median | p_min  | p_max  | cum_p  | %≥thr |",
        "|------|--------|----------|--------|--------|--------|-------|",
    ]
    per_iter = diag.get("results", {}).get(str(k_value), [])
    for entry in per_iter:
        out.append(
            f"| {entry['iter']:>4} | {entry['p_mean']:.4f} | {entry['p_median']:.4f} | "
            f"{entry['p_min']:.4f} | {entry['p_max']:.4f} | "
            f"{entry.get('cum_p_mean', entry.get('cum_p', 0.0)):.4f} | "
            f"{entry.get('frac_above_threshold', entry.get('frac_ge_threshold', 0.0)):.3f} |"
        )
    return "\n".join(out)


def fmt_hist_table(hist: dict) -> str:
    out = [
        "**Halt-step histogram per K:**",
        "",
        "| K  | mean halt step | median | tokens at iter 1 | full budget used |",
        "|----|----------------|--------|------------------|------------------|",
    ]
    for K_str, r in sorted(hist.get("results", {}).items(), key=lambda kv: int(kv[0])):
        K = int(K_str)
        bins = r.get("histogram", [])
        at_iter1 = bins[1] if len(bins) > 1 else 0
        total = r.get("total_tokens", sum(bins))
        out.append(
            f"| {K:>2} | {r.get('mean_halt_step', float('nan')):.2f}           | "
            f"{r.get('median_halt_step', 0):>6} | {at_iter1:,} / {total:,}    | "
            f"{r.get('frac_using_full_budget', 0.0):.3f}            |"
        )
    return "\n".join(out)


def render_section(round_n: int) -> str:
    meta = ROUND_META[round_n]
    comp_n = meta["compare_round"]
    this_off, this_on = load_depth_extrap(DOCS / f"depth_extrap_round{round_n}.json")
    comp_off, _ = load_depth_extrap(DOCS / f"depth_extrap_round{comp_n}.json")
    diag = load_diagnostic(DOCS / f"act_halt_diagnostic_round{round_n}.json")
    hist = load_histogram(DOCS / f"act_halt_histogram_round{round_n}.json")

    parts = [
        meta["section"],
        "",
        meta["blurb"],
        "",
        fmt_depth_table(round_n, this_off, this_on, comp_off),
        "",
        fmt_diag_table(diag, k_value=4),
        "",
        fmt_hist_table(hist),
        "",
        "**[DRAFT verdict — polish before submission.]** Numbers above were auto-extracted "
        f"from `docs/depth_extrap_round{round_n}.json`, "
        f"`docs/act_halt_diagnostic_round{round_n}.json`, and "
        f"`docs/act_halt_histogram_round{round_n}.json`. The auto-generated paper fragment is "
        f"at `docs/paper/round{round_n}_results.md` for reference.",
    ]
    return "\n".join(parts) + "\n\n"


def insert_into_main(round_n: int, section_text: str) -> bool:
    """Insert section just before §8 if not already present. Returns True if inserted."""
    if not MAIN_MD.exists():
        print(f"main.md missing at {MAIN_MD}", file=sys.stderr)
        return False
    content = MAIN_MD.read_text()
    section_header = ROUND_META[round_n]["section"]
    if section_header in content:
        print(f"already present: {section_header}")
        return False
    if INSERT_BEFORE_HEADER not in content:
        print(f"insertion anchor missing: {INSERT_BEFORE_HEADER}", file=sys.stderr)
        return False
    # Backup
    backup = MAIN_MD.with_suffix(f".md.bak.round{round_n}")
    shutil.copy2(MAIN_MD, backup)
    new_content = content.replace(
        INSERT_BEFORE_HEADER, section_text + INSERT_BEFORE_HEADER, 1
    )
    MAIN_MD.write_text(new_content)
    print(f"inserted {section_header}; backup at {backup}")
    return True


REMOTE_HOSTS_FOR_FALLBACK = ("alexm@kebab-spark.lan", "alex@mini-beast.lan")
REMOTE_DOCS_FOR_FALLBACK = ("/home/alexm/OpenMythos/docs", "/home/alex/OpenMythos/docs")


def fetch_comparison_depth(round_n: int) -> bool:
    """Pull just depth_extrap_round{N}.json for the comparison round if missing locally."""
    f = f"depth_extrap_round{round_n}.json"
    local_path = DOCS / f
    if local_path.exists() and local_path.stat().st_size > 0:
        return True
    for host, remote_docs in zip(REMOTE_HOSTS_FOR_FALLBACK, REMOTE_DOCS_FOR_FALLBACK):
        try:
            subprocess.run(
                ["rsync", "-a", f"{host}:{remote_docs}/{f}", str(local_path)],
                check=True, stderr=subprocess.PIPE,
            )
            if local_path.exists() and local_path.stat().st_size > 0:
                return True
        except subprocess.CalledProcessError:
            continue
    return False


def integrate_round(round_n: int) -> bool:
    print(f"--- round {round_n} ---")
    if not have_inputs(round_n):
        print("inputs missing locally; attempting rsync from remote")
        fetch_remote(round_n)
        if not have_inputs(round_n):
            print("inputs still missing; skipping")
            return False
    # Ensure the comparison round's depth_extrap is also pulled.
    comp_n = ROUND_META[round_n]["compare_round"]
    if not fetch_comparison_depth(comp_n):
        print(f"comparison round {comp_n} depth_extrap unavailable; skipping")
        return False
    section = render_section(round_n)
    return insert_into_main(round_n, section)


def main() -> None:
    all_rounds = sorted(ROUND_META.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=all_rounds)
    parser.add_argument("--scan", action="store_true",
                        help=f"try all known rounds ({', '.join(map(str, all_rounds))}); insert any that are ready")
    args = parser.parse_args()

    if args.scan or args.round is None:
        for r in all_rounds:
            print(f"--- round {r} ---")
            integrate_round(r)
    else:
        integrate_round(args.round)


if __name__ == "__main__":
    main()
