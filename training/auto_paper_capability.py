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

# All rounds we have or expect to have capability eval data for.
# Rounds without data are skipped gracefully (rendered as "--" cells).
ROUNDS = (23, 24, 25, 26, 27, 211, 212, 213)

REMOTE_SOURCES = [
    ("alexm@kebab-rtx6000.lan", "/home/alexm/OpenMythos/docs"),
    ("alexm@kebab-spark.lan", "/home/alexm/OpenMythos/docs"),
    ("alex@mini-beast.lan", "/home/alex/OpenMythos/docs"),
]


def round_label(r: int) -> str:
    """Render round number as a paper label: 23 -> 'r2.3', 210 -> 'r2.10'."""
    if r >= 100:
        # 210 -> 2.10, 211 -> 2.11
        return f"r2.{r - 200}"
    # 23 -> 2.3
    return f"r2.{r - 20}"


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


def have_any_inputs() -> bool:
    """Returns True if at least one round has at least one of the two eval JSONs.
    The renderer handles missing data gracefully (writes -- cells), so we don't
    need ALL rounds present to render a useful section."""
    for r in ROUNDS:
        for stem in (f"eval_listops_round{r}.json", f"eval_gsm8k_generation_round{r}.json"):
            p = DOCS / stem
            if p.exists() and p.stat().st_size > 0:
                return True
    return False


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
    # Find first available round to get schema
    sample = None
    for r in ROUNDS:
        sample = load_listops(r)
        if sample is not None:
            break
    if sample is None:
        return "_listops data unavailable_"
    tree_depths = sample.get("tree_depths", [3, 5, 7, 10])
    Ks = sample.get("depths", [4, 8, 16, 32])

    for r in ROUNDS:
        d = load_listops(r)
        if d is None:
            continue
        out.append(f"*Round {round_label(r)}:*")
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
    """Render a GSM8K table with one column per round that has data."""
    rows = {r: load_gsm8k(r) for r in ROUNDS}
    have_data = [r for r in ROUNDS if rows.get(r) is not None]
    if not have_data:
        return "_gsm8k data unavailable_"

    Ks = [4, 8, 16, 32]
    header_cells = " | ".join(round_label(r) for r in have_data)
    sep_cells = "|".join("------" for _ in have_data)
    out = [
        "**GSM8K full-CoT generation accuracy by K (4-shot prompting, exact-match on extracted final answer):**",
        "",
        f"| K  | {header_cells} |",
        f"|----|{sep_cells}|",
    ]
    for K in Ks:
        cells = []
        for r in have_data:
            d = rows[r]
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


def remove_section_if_present(content: str, header: str, next_header_pattern: str) -> str:
    """Remove a markdown section starting at `header` and ending before the next
    section heading (matching `next_header_pattern`). Returns content with the
    section removed (or unchanged if header not found)."""
    import re
    if header not in content:
        return content
    start = content.index(header)
    # Find the next ### header after start, or fall back to next_header_pattern
    rest = content[start + len(header):]
    next_match = re.search(r"\n##+ ", rest)
    anchor_match = content.find(next_header_pattern, start)
    if next_match:
        end = start + len(header) + next_match.start() + 1  # +1 to keep the leading \n
    elif anchor_match != -1:
        end = anchor_match
    else:
        end = len(content)
    return content[:start] + content[end:]


def insert_into_main(section_text: str, update: bool = False) -> bool:
    if not MAIN_MD.exists():
        print(f"main.md missing at {MAIN_MD}", file=sys.stderr)
        return False
    content = MAIN_MD.read_text()
    if SECTION_HEADER in content and not update:
        print(f"already present: {SECTION_HEADER} (use --update to replace)")
        return False
    if INSERT_BEFORE_HEADER not in content:
        print(f"insertion anchor missing: {INSERT_BEFORE_HEADER}", file=sys.stderr)
        return False
    backup = MAIN_MD.with_suffix(".md.bak.capability")
    shutil.copy2(MAIN_MD, backup)
    if SECTION_HEADER in content and update:
        content = remove_section_if_present(content, SECTION_HEADER, INSERT_BEFORE_HEADER)
        print(f"removed existing {SECTION_HEADER}")
    new_content = content.replace(
        INSERT_BEFORE_HEADER, section_text + INSERT_BEFORE_HEADER, 1
    )
    MAIN_MD.write_text(new_content)
    print(f"inserted {SECTION_HEADER}; backup at {backup}")
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="replace existing section even if already present (idempotent skip otherwise)")
    args = parser.parse_args()

    print("fetching capability eval JSONs from remote sources...")
    for r in ROUNDS:
        fetch_remote(f"eval_listops_round{r}.json")
        fetch_remote(f"eval_gsm8k_generation_round{r}.json")
    if not have_any_inputs():
        print("no capability eval data available; skipping")
        return

    # Auto-update if the existing section has fewer rounds than what we now have
    auto_update = False
    if MAIN_MD.exists() and SECTION_HEADER in MAIN_MD.read_text():
        existing = MAIN_MD.read_text()
        # crude heuristic: count round labels in the existing section
        section_start = existing.index(SECTION_HEADER)
        next_h = existing.find("\n### ", section_start + 1)
        section_text_existing = existing[section_start:next_h] if next_h != -1 else existing[section_start:]
        n_round_refs_existing = sum(1 for r in ROUNDS if round_label(r) in section_text_existing or f"Round {round_label(r)}:" in section_text_existing)
        n_round_refs_now = sum(1 for r in ROUNDS if (DOCS / f"eval_listops_round{r}.json").exists() or (DOCS / f"eval_gsm8k_generation_round{r}.json").exists())
        if n_round_refs_now > n_round_refs_existing:
            print(f"existing section has {n_round_refs_existing} rounds, we now have {n_round_refs_now}; auto-updating")
            auto_update = True

    section = render_section()
    insert_into_main(section, update=args.update or auto_update)


if __name__ == "__main__":
    main()
