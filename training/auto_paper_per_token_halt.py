#!/usr/bin/env python3
"""
Pull per_token_halt_round*.json from remote hosts and insert §7.23
"Per-token halt distribution across rounds" into docs/paper/main.md.

per_token_halt_analysis.py reports, for a given checkpoint, the
distribution of which iteration each token halts at -- both globally
and broken down by token type / surprise level. This script collates
those measurements across rounds into one cross-round table.

Idempotent + auto-update.

Usage:
    python3 training/auto_paper_per_token_halt.py
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
SECTION_HEADER = "### 7.23 Per-token halt distribution across rounds"

# Rounds to look for. Skipped silently if missing.
ROUNDS = (21, 22, 23, 24, 25, 26, 27, 29, 210, 211, 212, 213, 214)
# Special variants
EXTRAS = ("anti10",)  # round-2.6 + anti-collapse head experiment, etc.

REMOTE_SOURCES = [
    ("alexm@kebab-spark.lan", "/home/alexm/OpenMythos/docs"),
    ("alexm@kebab-rtx6000.lan", "/home/alexm/OpenMythos/docs"),
    ("alex@mini-beast.lan", "/home/alex/OpenMythos/docs"),
]


def round_label(r: int) -> str:
    if r >= 100:
        return f"r2.{r - 200}"
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


def load_per_token_halt(name: str) -> dict | None:
    p = DOCS / f"per_token_halt_{name}.json"
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size > 0 else None


def fmt_summary_table(available_keys: list[tuple[str, str]]) -> str:
    """Cross-round summary: rows = round, columns = global stats."""
    rows = []
    for label, key in available_keys:
        d = load_per_token_halt(key)
        if d is None:
            continue
        # Schema-tolerant: try common keys
        global_stats = d.get("global", d.get("overall", d))
        mean = global_stats.get("mean_halt_step", global_stats.get("mean", float("nan")))
        median = global_stats.get("median_halt_step", global_stats.get("median", 0))
        p1 = global_stats.get("p_iter1", global_stats.get("p_1", float("nan")))
        full = global_stats.get("frac_using_full_budget", float("nan"))
        rows.append((label, mean, median, p1, full))

    if not rows:
        return "_no per_token_halt data available_"

    out = [
        "**Cross-round summary (global statistics):**",
        "",
        "| Round | mean halt step | median halt | p(iter=1) | frac using full budget |",
        "|-------|---------------|-------------|-----------|------------------------|",
    ]
    for label, mean, median, p1, full in rows:
        mean_s = f"{mean:.2f}" if mean == mean else "--"
        median_s = f"{median}" if isinstance(median, (int, float)) and median == median else "--"
        p1_s = f"{p1:.4f}" if p1 == p1 else "--"
        full_s = f"{full:.3f}" if full == full else "--"
        out.append(f"| {label:>5} | {mean_s:>13} | {median_s:>11} | {p1_s:>9} | {full_s:>22} |")
    return "\n".join(out)


def fmt_by_token_type(available_keys: list[tuple[str, str]]) -> str:
    """If JSONs include per-token-type breakdowns (function words, content words,
    rare tokens, etc.), surface that. Schema-tolerant."""
    parts = ["**Halt step by token category:**", ""]
    any_data = False
    for label, key in available_keys:
        d = load_per_token_halt(key)
        if d is None:
            continue
        # Look for "by_category" or "by_token_type" or similar
        cats = d.get("by_category") or d.get("by_token_type") or d.get("categories")
        if not cats or not isinstance(cats, dict):
            continue
        any_data = True
        parts.append(f"*Round {label}:*")
        # Each category typically maps to a dict like {"n": ..., "mean_halt_step": ..., "p_iter1": ...}
        parts.append("| category | n | mean halt | p(iter=1) |")
        parts.append("|----------|---|-----------|-----------|")
        for cat_name, cat_data in cats.items():
            if not isinstance(cat_data, dict):
                continue
            n = cat_data.get("n", "")
            mean = cat_data.get("mean_halt_step", cat_data.get("mean", float("nan")))
            p1 = cat_data.get("p_iter1", cat_data.get("p_1", float("nan")))
            mean_s = f"{mean:.2f}" if isinstance(mean, (int, float)) and mean == mean else "--"
            p1_s = f"{p1:.4f}" if isinstance(p1, (int, float)) and p1 == p1 else "--"
            n_s = f"{int(n):,}" if isinstance(n, (int, float)) else str(n)
            parts.append(f"| {cat_name} | {n_s} | {mean_s} | {p1_s} |")
        parts.append("")

    if not any_data:
        return ""
    return "\n".join(parts)


def render_section(available_keys: list[tuple[str, str]]) -> str:
    blurb = (
        "Where the round-level §7.13--§7.20 sections show the *aggregate* halt-step "
        "histogram (one number per K), this section breaks halt step out across "
        "individual tokens and rounds. The question is whether the trained head learns "
        "an interesting per-token allocation policy (different halt steps for content "
        "vs. function tokens, for instance) or whether it collapses onto one global mean."
    )
    parts = [SECTION_HEADER, "", blurb, ""]
    parts.append(fmt_summary_table(available_keys))
    parts.append("")
    by_cat = fmt_by_token_type(available_keys)
    if by_cat:
        parts.append(by_cat)
        parts.append("")
    parts.append(
        "**[DRAFT verdict — polish before submission.]** Auto-extracted from "
        "`docs/per_token_halt_round*.json` and special-variant JSONs. The mean halt step "
        "and `p(iter=1)` columns are the most diagnostic: stable mean across rounds + "
        "non-trivial spread by token category = head is doing something useful; "
        "saturated `p(iter=1) ≈ 1` = collapsed."
    )
    parts.append("")
    return "\n".join(parts)


def remove_section_if_present(content: str, header: str, next_header_pattern: str) -> str:
    import re
    if header not in content:
        return content
    start = content.index(header)
    rest = content[start + len(header):]
    next_match = re.search(r"\n### ", rest)
    anchor_match = content.find(next_header_pattern, start)
    if next_match:
        end = start + len(header) + next_match.start() + 1
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
    backup = MAIN_MD.with_suffix(".md.bak.per_token_halt")
    shutil.copy2(MAIN_MD, backup)
    if SECTION_HEADER in content and update:
        content = remove_section_if_present(content, SECTION_HEADER, INSERT_BEFORE_HEADER)
    new_content = content.replace(
        INSERT_BEFORE_HEADER, section_text + INSERT_BEFORE_HEADER, 1
    )
    MAIN_MD.write_text(new_content)
    print(f"inserted {SECTION_HEADER}; backup at {backup}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    print("fetching per_token_halt JSONs...")
    for r in ROUNDS:
        fetch_remote(f"per_token_halt_round{r}.json")
    for variant in EXTRAS:
        fetch_remote(f"per_token_halt_{variant}.json")

    available_keys: list[tuple[str, str]] = []
    for r in ROUNDS:
        if (DOCS / f"per_token_halt_round{r}.json").exists():
            available_keys.append((round_label(r), f"round{r}"))
    for variant in EXTRAS:
        if (DOCS / f"per_token_halt_{variant}.json").exists():
            available_keys.append((variant, variant))

    if not available_keys:
        print("no per_token_halt data; skipping")
        return
    print(f"rendering with: {[k[0] for k in available_keys]}")

    auto_update = False
    if MAIN_MD.exists() and SECTION_HEADER in MAIN_MD.read_text():
        existing = MAIN_MD.read_text()
        s = existing.index(SECTION_HEADER)
        next_h = existing.find("\n### ", s + 1)
        section = existing[s:next_h] if next_h != -1 else existing[s:]
        n_existing = sum(1 for label, _ in available_keys if label in section)
        if len(available_keys) > n_existing:
            auto_update = True
            print(f"auto-updating: existing has {n_existing}, now have {len(available_keys)}")

    section_text = render_section(available_keys)
    insert_into_main(section_text, update=args.update or auto_update)


if __name__ == "__main__":
    main()
