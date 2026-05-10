#!/usr/bin/env python3
"""
Pull depth_extrap_round*_k64.json from spark and insert §7.22
"Extended depth extrapolation up to K=64" into docs/paper/main.md.

These are extended runs of depth_extrap.py with DEPTHS=4,8,16,32,64
(default goes only to K=32). Tests whether the model can usefully exploit
inference depth ~5x larger than its training T_max.

Idempotent + auto-update.

Usage:
    python3 training/auto_paper_depth_k64.py
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
SECTION_HEADER = "### 7.22 Extended depth extrapolation up to K=64"

# Which rounds we have or expect to have K=64 data for.
ROUNDS = (23, 24, 25, 26, 27, 29, 210, 211, 212, 213, 214)

REMOTE_SOURCES = [
    ("alexm@kebab-spark.lan", "/home/alexm/OpenMythos/docs"),
    ("alexm@kebab-rtx6000.lan", "/home/alexm/OpenMythos/docs"),
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


def load_depth_k64(r: int) -> list[dict] | None:
    p = DOCS / f"depth_extrap_round{r}_k64.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    data = json.loads(p.read_text())
    # Schema: {"results": [{"K": int, "fineweb_ce_off": ..., "fineweb_ce_on": ..., ...}, ...]}
    # OR {"K": [...], "fineweb_ce_off": [...], ...}  -- depends on script
    return data.get("results", data.get("by_K"))


def fmt_k64_table(available_rounds: list[int]) -> str:
    """Cross-round table: rows = K (4,8,16,32,64), columns = round.
    Shows FineWeb-Edu CE with ACT off (i.e., uses full K iterations)."""
    rows: dict[int, dict] = {}  # round -> K -> ce_off
    Ks_set = set()
    for r in available_rounds:
        results = load_depth_k64(r)
        if not results:
            continue
        rows[r] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            K = entry.get("K") or entry.get("k") or entry.get("depth")
            if K is None:
                continue
            ce = entry.get("fineweb_ce_off") or entry.get("ce_off") or entry.get("fineweb_ce")
            if ce is None:
                continue
            rows[r][int(K)] = ce
            Ks_set.add(int(K))

    rounds_with_data = [r for r in available_rounds if r in rows and rows[r]]
    if not rounds_with_data:
        return "_no K=64 data available_"

    Ks = sorted(Ks_set)
    header = "| K  | " + " | ".join(round_label(r) for r in rounds_with_data) + " |"
    sep = "|----|" + "|".join("------" for _ in rounds_with_data) + "|"
    out = [
        "**FineWeb-Edu CE (ACT off, full K iterations) by K and round:**",
        "",
        header,
        sep,
    ]
    for K in Ks:
        cells = []
        for r in rounds_with_data:
            v = rows[r].get(K)
            cells.append(f"{v:.4f}" if v is not None else "--")
        out.append(f"| {K:>2} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_section(available_rounds: list[int]) -> str:
    blurb = (
        "Standard `depth_extrap.py` runs evaluate at K ∈ {4, 8, 16, 32}. We extend the "
        "range to K = 64 (~5x the training `T_max = 12`) on selected rounds to test "
        "whether models continue to benefit from inference depth far beyond the training "
        "regime. The expectation: a saturating curve, with the marginal CE improvement "
        "per extra K shrinking at high K. A non-monotone curve (CE re-increasing past "
        "some K) would indicate the recurrent block has learned a diverging fixed-point."
    )
    return "\n".join([
        SECTION_HEADER,
        "",
        blurb,
        "",
        fmt_k64_table(available_rounds),
        "",
        "**[DRAFT verdict — polish before submission.]** Auto-extracted from "
        "`docs/depth_extrap_round*_k64.json`. Numbers tell us whether the iteration "
        "fixed-point converges or diverges past T_max.",
        "",
    ])


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
    backup = MAIN_MD.with_suffix(".md.bak.depth_k64")
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

    print("fetching K=64 depth_extrap JSONs...")
    for r in ROUNDS:
        fetch_remote(f"depth_extrap_round{r}_k64.json")

    available = [r for r in ROUNDS if (DOCS / f"depth_extrap_round{r}_k64.json").exists()]
    if not available:
        print("no K=64 data available; skipping")
        return
    print(f"rendering with rounds {available}")

    auto_update = False
    if MAIN_MD.exists() and SECTION_HEADER in MAIN_MD.read_text():
        existing = MAIN_MD.read_text()
        s = existing.index(SECTION_HEADER)
        next_h = existing.find("\n### ", s + 1)
        section = existing[s:next_h] if next_h != -1 else existing[s:]
        n_existing = sum(1 for r in ROUNDS if round_label(r) in section)
        if len(available) > n_existing:
            auto_update = True
            print(f"upgrading section: existing has {n_existing}, now have {len(available)}")

    section_text = render_section(available)
    insert_into_main(section_text, update=args.update or auto_update)


if __name__ == "__main__":
    main()
