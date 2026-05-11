#!/usr/bin/env python3
"""
gpufarm/gaps.py -- Layer 1 gap detector.

Reads gpufarm/resources.yaml, gpufarm/jobs.yaml, gpufarm/rounds.yaml. For each
(round, eval_in_bundle) pair, tests whether the corresponding output JSON
already exists locally. Prints a table of gaps plus per-round/per-job stats.

Usage:
    python3 gpufarm/gaps.py                       # full report, table format
    python3 gpufarm/gaps.py --round r210          # one round only
    python3 gpufarm/gaps.py --job depth_extrap_k64 # one job type only
    python3 gpufarm/gaps.py --plan                # only show actionable gaps
                                                    (excludes ckpt_status: cleaned_up)
    python3 gpufarm/gaps.py --json                # machine-readable output

Treats a JSON file as "done" iff it exists and is >= 100 bytes. (Defends
against the 0-byte truncated files left by failed rsyncs.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent  # repo root
GPUFARM = ROOT / "gpufarm"
DOCS = ROOT / "docs"

MIN_VALID_JSON_BYTES = 100  # below this is almost certainly a failed write


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def render_output_path(pattern: str, round_id: str) -> Path:
    """
    Render a job's output_pattern for a given round_id (e.g., 'r210' -> 210).
    The patterns mostly use {round_num} or {label}.
    """
    # round_id like "r210" -> round_num "210"
    round_num = round_id.lstrip("r")
    rendered = pattern.format(
        round_num=round_num,
        label=round_id,
    )
    return ROOT / rendered


def is_done(path: Path) -> bool:
    """A job is done iff its output file exists and is non-trivially sized."""
    if not path.exists():
        return False
    if not path.is_file():
        return False
    return path.stat().st_size >= MIN_VALID_JSON_BYTES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", dest="round_filter", help="Only this round")
    parser.add_argument("--job", dest="job_filter", help="Only this job type")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Only show actionable gaps (skip rounds with ckpt_status: cleaned_up)",
    )
    parser.add_argument("--json", dest="emit_json", action="store_true")
    args = parser.parse_args()

    rounds = load_yaml(GPUFARM / "rounds.yaml")["rounds"]
    jobs_doc = load_yaml(GPUFARM / "jobs.yaml")
    jobs_by_id = {j["id"]: j for j in jobs_doc["jobs"]}

    gaps: list[dict] = []
    done: list[dict] = []
    skipped: list[dict] = []

    for round_id, rd in rounds.items():
        if args.round_filter and round_id != args.round_filter:
            continue
        ckpt_status = rd.get("ckpt_status", "available")
        for job_id in rd.get("eval_bundle", []):
            if args.job_filter and job_id != args.job_filter:
                continue
            if job_id not in jobs_by_id:
                gaps.append({
                    "round": round_id, "job": job_id, "status": "unknown_job",
                    "expected_path": None,
                })
                continue
            job = jobs_by_id[job_id]
            pattern = job["output_pattern"]
            path = render_output_path(pattern, round_id)
            rec = {
                "round": round_id,
                "job": job_id,
                "expected_path": str(path.relative_to(ROOT)),
                "resource_class": job.get("resource_class", []),
                "estimated_minutes": job.get("estimated_minutes"),
            }
            if is_done(path):
                done.append(rec)
            elif ckpt_status == "cleaned_up":
                rec["reason"] = "ckpt cleaned up"
                skipped.append(rec)
            else:
                gaps.append(rec)

    if args.emit_json:
        print(json.dumps({"gaps": gaps, "done": done, "skipped": skipped}, indent=2))
        return 0

    # Human-readable summary
    print(f"=== gpufarm gap report (root: {ROOT}) ===\n")

    print(f"DONE: {len(done):3d} artifacts")
    print(f"GAPS: {len(gaps):3d} artifacts to produce")
    if skipped:
        print(f"SKIP: {len(skipped):3d} (ckpt unavailable)")
    print()

    if gaps and not args.plan:
        print("--- Pending artifacts ---")
        # Group by job_id
        by_job: dict[str, list[dict]] = {}
        for g in gaps:
            by_job.setdefault(g["job"], []).append(g)
        for job_id in sorted(by_job):
            rounds_missing = [g["round"] for g in by_job[job_id]]
            est = by_job[job_id][0].get("estimated_minutes") or "?"
            classes = ",".join(by_job[job_id][0].get("resource_class") or ["?"])
            print(f"  {job_id:24s} [{classes:18s}] {est:>4} min  missing on: {' '.join(rounds_missing)}")
        print()

    if args.plan:
        print("--- Actionable plan (sorted by quickest first) ---")
        # Group gaps by job_id, sort by estimated_minutes
        by_job_plan: dict[str, list[dict]] = {}
        for g in gaps:
            by_job_plan.setdefault(g["job"], []).append(g)
        plan = []
        for job_id, items in by_job_plan.items():
            job = jobs_by_id[job_id]
            est = job.get("estimated_minutes", 999)
            plan.append((est, job_id, items, job.get("resource_class", [])))
        plan.sort(key=lambda x: x[0])
        total_minutes = 0
        for est, job_id, items, classes in plan:
            rounds_missing = " ".join(item["round"] for item in items)
            jobs_total = est * len(items)
            total_minutes += jobs_total
            print(
                f"  {job_id:24s} x{len(items):2d}  "
                f"~{est}min each = ~{jobs_total}min  "
                f"[{','.join(classes)}]  rounds: {rounds_missing}"
            )
        print(f"\n  Total compute budget: ~{total_minutes} min ({total_minutes/60:.1f}h)")
        print(
            "  (Parallelize across GPU classes -- see resources.yaml. "
            "blackwell can run ~2 in parallel via GPU 0 + GPU 1.)"
        )

    if skipped and not args.plan:
        print("--- Skipped (ckpt no longer available) ---")
        for s in skipped:
            print(f"  {s['round']}/{s['job']:24s} ({s.get('reason')})")

    return 0 if not gaps else 1


if __name__ == "__main__":
    sys.exit(main())
