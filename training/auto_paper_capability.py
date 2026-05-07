#!/usr/bin/env python3
"""
Pull eval_listops_round{23,24,25}.json and eval_gsm8k_generation_round{23,24,25}.json
from mini-beast, render a §7.15 paper section with the K-vs-accuracy
tables, and insert into docs/paper/main.md before §8.

Idempotent: skips if section heading already present. Backs up main.md.

Usage:
    python3 training/auto_paper_capability.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/alex/git/OpenMythos")
MAIN_MD = REPO / "docs/paper/main.md"
DOCS = REPO / "docs"
INSERT_BEFORE_HEADER = "## 8. Discussion"
SECTION_HEADER = "### 7.15 Depth-graded capability probes (ListOps and GSM8K full generation)"

ROUNDS = (23, 24, 25)

REMOTE_SOURCES = [
    ("alexm@kebab-rtx6000.lan", "/home/alexm/OpenMythos/docs"),
    ("alex@mini-beast.lan", "/home/alex/OpenMythos/docs"),
]


def fetch_remote(filename: str) -> Path | None:
    local = DOCS / filename
    local.parent.mkdir(parents=True, exist_ok=True)
    for host, remote_docs in REMOTE_SOURCES:
        try:
            subprocess.run(
                ["rsync", "-a", f"{host}:{remote_docs}/{filename}", str(local)],
                check=True,
                stderr=subprocess.PIPE,
            )
            if local.exists() and local.stat().st_size > 0:
                return local
        except subprocess.CalledProcessError:
            continue
    return local if local.exists() and local.stat().st_size > 0 else None


def have_all_inputs() -> bool:
    needed: list[Path] = []
    for r in ROUNDS:
        needed.append(DOCS / f"eval_listops_round{r}.json")
        needed.append(DOCS / f"eval_gsm8k_generation_round{r}.json")
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def load_listops(round_n: int) -> dict | None:
    p = DOCS / f"eval_listops_round{round_n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_gsm8k(round_n: int) -> dict | None:
    p = DOCS / f"eval_gsm8k_generation_round{round_n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt_listops_table() -> str:
    out = [
        "**ListOps accuracy by K and tree depth (synthetic, 100 problems per (K, tree-depth)):**",
        "",
    ]
    sample = load_listops(ROUNDS[0])
    if sample is None:
        return "_listops data unavailable_"
    tree_depths = sample.get("tree_depths", [3, 5, 7, 10])
    Ks = sample.get("depths", [4, 8, 16, 32])

    for r in ROUNDS:
        d = load_listops(r)
        if d is None:
            continue
        out.append(f"*Round 2.{r % 10 + 2 if r // 10 == 2 else r}:*")
        header = "| K  | " + " | ".join(f"d{td}" for td in tree_depths) + " | overall |"
        sep = "|----|" + "|".join("------" for _ in tree_depths) + "|---------|"
        out.append(header)
        out.append(sep)
        for K in Ks:
            row = d["results"].get(f"K{K}", {})
            cells = " | ".join(f"{row.get(f'd{td}', float('nan')):.3f}" for td in tree_depths)
            overall = row.get("overall", float("nan"))
            out.append(f"| {K:>2} | {cells} | {overall:.3f}   |")
        out.append("")
    return "\n".join(out)


def fmt_gsm8k_table() -> str:
    out = [
        "**GSM8K full-CoT generation accuracy by K (4-shot prompting, exact-match on extracted final answer):**",
        "",
        "| K  | r2.3 | r2.4 | r2.5 |",
        "|----|------|------|------|",
    ]
    Ks = [4, 8, 16, 32]
    rows = {r: load_gsm8k(r) for r in ROUNDS}
    for K in Ks:
        cells = []
        for r in ROUNDS:
            d = rows.get(r)
            if d is None:
                cells.append("--")
            else:
                acc = d["results"].get(f"K{K}", {}).get("acc", float("nan"))
                cells.append(f"{acc:.3f}")
        out.append(f"| {K:>2} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_section() -> str:
    blurb = (
        "Every accuracy probe we have run on this model so far has been depth-flat: "
        "synthetic_depth's five-task suite (§8.4 footnote), and the multiple-choice "
        "log-likelihood evals on ARC-Easy, ARC-Challenge, and HellaSwag, all yield "
        "K-invariant accuracy at every round. Two of these probes are arguably ill-suited "
        "to recurrent-depth scaling: synthetic_depth's tasks are mostly single-step, and "
        "multiple-choice log-likelihood scoring is dominated by argmax stability rather "
        "than per-token logit shifts. We add two further probes designed to be more "
        "directly responsive to additional iteration: ListOps (Nangia & Bowman 2018) "
        "where the tree-nesting depth maps onto the number of reduction steps required, "
        "and GSM8K with full chain-of-thought generation, where extra K could in "
        "principle let the model construct more accurate intermediate steps."
    )
    return "\n".join([
        SECTION_HEADER,
        "",
        blurb,
        "",
        fmt_listops_table(),
        "",
        fmt_gsm8k_table(),
        "",
        "**[DRAFT verdict — polish before submission.]** Numbers above were auto-extracted "
        "from `docs/eval_listops_round{23,24,25}.json` and "
        "`docs/eval_gsm8k_generation_round{23,24,25}.json`. Whether either probe shows "
        "monotone K-improvement or remains depth-flat determines the §8.4 framing.",
        "",
    ])


def insert_into_main(section_text: str) -> bool:
    if not MAIN_MD.exists():
        print(f"main.md missing at {MAIN_MD}", file=sys.stderr)
        return False
    content = MAIN_MD.read_text()
    if SECTION_HEADER in content:
        print(f"already present: {SECTION_HEADER}")
        return False
    if INSERT_BEFORE_HEADER not in content:
        print(f"insertion anchor missing: {INSERT_BEFORE_HEADER}", file=sys.stderr)
        return False
    backup = MAIN_MD.with_suffix(".md.bak.capability")
    shutil.copy2(MAIN_MD, backup)
    new_content = content.replace(
        INSERT_BEFORE_HEADER, section_text + INSERT_BEFORE_HEADER, 1
    )
    MAIN_MD.write_text(new_content)
    print(f"inserted {SECTION_HEADER}; backup at {backup}")
    return True


def main() -> None:
    if not have_all_inputs():
        print("inputs missing locally; attempting rsync from mini-beast")
        for r in ROUNDS:
            fetch_remote(f"eval_listops_round{r}.json")
            fetch_remote(f"eval_gsm8k_generation_round{r}.json")
        if not have_all_inputs():
            print("inputs still missing; skipping")
            return
    section = render_section()
    insert_into_main(section)


if __name__ == "__main__":
    main()
