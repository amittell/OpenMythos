#!/usr/bin/env python3
"""
gpufarm_bootstrap_from_disk.py

One-shot helper that pre-populates the gpufarm coordinator's ``state.sqlite``
with synthetic ``runs`` rows for every (round, job) pair whose output JSON
already exists on disk.

Background
----------
The cutover from the legacy ``training/queue_*.sh`` shell loop to the gpufarm
coordinator daemon happens mid-paper -- rounds r2.3 / r2.4 / ... / r2.14 have
all produced eval artifacts already. A freshly-initialised state.sqlite has
no row for any of these completed runs, so the dashboard's history view would
look empty (and any future ``gpufarm history`` query is blind to them).

The gpufarm gap detector compensates because it reads the filesystem, not the
DB: ``gpufarm gaps --plan`` will not re-queue completed work even with an
empty DB. So this script is OPTIONAL -- the daemon still does the right thing
without it. Run this only if you want the COMPLETED runs to show up in
``gpufarm history`` after cutover.

The cutover plan at gpufarm/docs/cutover_plan.md (PR-8) calls this out
explicitly: see the "Bootstrap state.sqlite from existing on-disk artifacts"
section.

Behaviour
---------
* Loads the manifests via ``gpufarm.manifests.Manifests.load``.
* Walks every (round, job) in each round's ``eval_bundle``.
* For each pair whose expected ``output_pattern`` resolves to a file on disk
  with size >= ``MIN_VALID_JSON_BYTES`` (the same 100-byte threshold the gap
  detector uses), inserts a COMPLETED run with ``output_path`` and
  ``started_at`` / ``ended_at`` backfilled from the file's mtime.
* Idempotent: re-running on a populated DB never inserts duplicates. The
  schema has no unique constraint on (job_id, round_id, output_path), so we
  check-then-insert under a transaction (the writer is the operator running
  this script -- the daemon is not started yet in shadow mode).
* Missing artifacts get NO row. The daemon surfaces them through
  ``gpufarm gaps --plan`` / ``gpufarm submit --all-gaps``.

Usage
-----
    python3 tools/gpufarm_bootstrap_from_disk.py \\
        --manifest-dir gpufarm \\
        --state-db ~/.local/state/gpufarm/state.sqlite

Defaults pick up ``$GPUFARM_MANIFEST_DIR`` and ``$GPUFARM_STATE_DB`` from the
environment if those flags are omitted; the same env vars the daemon honours.

Exit codes:
    0  -- success
    1  -- bad CLI args
    2  -- manifest load failed
    3  -- state.sqlite open / migration failed
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


# Match gpufarm/_gaps.py. Below this size we treat the file as a failed write,
# not a completed run -- same heuristic the gap detector uses so the two
# tools agree on what "done" means.
MIN_VALID_JSON_BYTES = 100


def _utc_iso(epoch_secs: float) -> str:
    """Convert a POSIX timestamp into the ISO-8601 UTC string the store uses."""
    return datetime.fromtimestamp(epoch_secs, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_output_path(pattern: str, round_id: str, root: Path) -> Path:
    """Resolve an output_pattern template against (round, repo_root).

    Mirrors gpufarm._gaps._render_output_path so the bootstrap stays in sync
    with the gap detector.
    """
    round_num = round_id.lstrip("r")
    rendered = pattern.format(round_num=round_num, label=round_id)
    return root / rendered


def _is_done(path: Path) -> bool:
    """A file counts as a completed artifact iff it exists and is non-trivial."""
    if not path.exists() or not path.is_file():
        return False
    return path.stat().st_size >= MIN_VALID_JSON_BYTES


def _ensure_store(state_db: Path):
    """Open / create the SQLite store with the gpufarm schema applied.

    We import gpufarm.store.Store here so the schema migration logic stays in
    one place. If gpufarm is not importable, fall back to a clear error -- the
    bootstrap can't proceed without the schema.
    """
    try:
        from gpufarm.store import Store
    except ImportError as exc:
        print(
            f"error: gpufarm package not importable ({exc}). "
            "Install with `pip install --user -e /path/to/gpufarm` first.",
            file=sys.stderr,
        )
        sys.exit(3)

    state_db.parent.mkdir(parents=True, exist_ok=True)
    return Store(state_db)


def _load_manifests(manifest_dir: Path):
    try:
        from gpufarm.manifests import Manifests
    except ImportError as exc:
        print(
            f"error: gpufarm.manifests not importable ({exc})",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        return Manifests.load(manifest_dir)
    except Exception as exc:
        print(f"error: failed to load manifests at {manifest_dir}: {exc}", file=sys.stderr)
        sys.exit(2)


def _row_exists(
    conn: sqlite3.Connection,
    job_id: str,
    round_id: str,
    output_path: str,
) -> bool:
    """The schema has no unique index on (job_id, round_id, output_path).

    We emulate INSERT OR IGNORE by checking before insert. Safe under our
    single-writer assumption (the daemon is NOT running while bootstrap runs;
    shadow-mode cutover guarantees this).
    """
    row = conn.execute(
        """
        SELECT 1 FROM runs
        WHERE job_id = ?
          AND round_id = ?
          AND output_path = ?
        LIMIT 1
        """,
        (job_id, round_id, output_path),
    ).fetchone()
    return row is not None


def _insert_completed_run(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    round_id: str,
    priority: int,
    output_path: str,
    ts_iso: str,
) -> int:
    """Insert one COMPLETED row, using the file mtime for started/ended_at.

    submitted_at is set to the same mtime so the row's "age" matches reality;
    that is what ``gpufarm history --since`` filters on.

    Returns the new row id.
    """
    cur = conn.execute(
        """
        INSERT INTO runs (
            job_id, round_id, status, priority,
            submitted_at, started_at, ended_at,
            output_path, output_valid,
            env_overrides, job_type, exit_code
        ) VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, 1, '{}', 'compute', 0)
        """,
        (
            job_id,
            round_id,
            priority,
            ts_iso,
            ts_iso,
            ts_iso,
            output_path,
        ),
    )
    return int(cur.lastrowid)


def bootstrap(
    manifest_dir: Path,
    state_db: Path,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Drive the bootstrap. Returns a summary dict for logging."""
    repo_root = (repo_root or manifest_dir.parent).resolve()
    manifests = _load_manifests(manifest_dir)

    # Build the gap report ourselves rather than calling compute_gaps so we
    # have direct access to per-record file stats (mtime). The two paths
    # agree on what counts as "done" because they share MIN_VALID_JSON_BYTES.
    summary = {
        "manifest_dir": str(manifest_dir),
        "state_db": str(state_db),
        "repo_root": str(repo_root),
        "completed_artifacts_found": 0,
        "rows_inserted": 0,
        "rows_skipped_already_present": 0,
        "rows_skipped_missing_pattern": 0,
        "gaps_left_for_daemon": 0,
        "skipped_ckpt_cleaned_up": 0,
        "dry_run": bool(dry_run),
    }

    store = _ensure_store(state_db)
    conn = store.conn

    try:
        for round_id, rd in manifests.rounds.items():
            # Manifests._RoundY exposes ckpt_status via the pydantic model's
            # attribute access; default "available" matches the gap detector.
            ckpt_status = getattr(rd, "ckpt_status", "available") or "available"
            for job_id in rd.eval_bundle:
                if job_id not in manifests.jobs:
                    # Daemon will surface as "unknown_job" gap.
                    summary["gaps_left_for_daemon"] += 1
                    continue
                job = manifests.jobs[job_id]
                if not job.output_pattern:
                    summary["rows_skipped_missing_pattern"] += 1
                    continue
                expected = _render_output_path(job.output_pattern, round_id, repo_root)
                if not _is_done(expected):
                    # Match gpufarm._gaps.compute_gaps: a missing artifact for
                    # a round whose ckpt has been cleaned up is "skipped",
                    # not a gap (we can't refill it without retraining).
                    if ckpt_status == "cleaned_up":
                        summary["skipped_ckpt_cleaned_up"] += 1
                    else:
                        summary["gaps_left_for_daemon"] += 1
                    continue

                summary["completed_artifacts_found"] += 1
                stored_path = str(expected)
                if _row_exists(conn, job_id, round_id, stored_path):
                    summary["rows_skipped_already_present"] += 1
                    continue

                if dry_run:
                    continue

                mtime_iso = _utc_iso(expected.stat().st_mtime)
                _insert_completed_run(
                    conn,
                    job_id=job_id,
                    round_id=round_id,
                    priority=job.priority,
                    output_path=stored_path,
                    ts_iso=mtime_iso,
                )
                summary["rows_inserted"] += 1
    finally:
        store.close()

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap gpufarm state.sqlite with synthetic COMPLETED runs for "
            "every (round, job) whose output artifact already exists on disk. "
            "Idempotent; safe to re-run."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        default=os.environ.get("GPUFARM_MANIFEST_DIR"),
        help=(
            "Path to the gpufarm/ manifest directory (resources/jobs/rounds/"
            "models YAMLs). Defaults to $GPUFARM_MANIFEST_DIR."
        ),
    )
    parser.add_argument(
        "--state-db",
        default=os.environ.get("GPUFARM_STATE_DB"),
        help=(
            "Path to the SQLite state DB. Defaults to $GPUFARM_STATE_DB; "
            "the systemd-managed location is typically "
            "~/.local/state/gpufarm/state.sqlite."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Root used to resolve relative output_pattern values. Defaults to "
            "the parent of --manifest-dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what WOULD be inserted without writing any rows.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON (for piping into jq / logging).",
    )
    args = parser.parse_args(argv)

    if not args.manifest_dir:
        parser.error("--manifest-dir (or $GPUFARM_MANIFEST_DIR) is required")
    if not args.state_db:
        parser.error("--state-db (or $GPUFARM_STATE_DB) is required")

    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    state_db = Path(args.state_db).expanduser()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else None

    if not manifest_dir.is_dir():
        print(f"error: manifest dir does not exist: {manifest_dir}", file=sys.stderr)
        return 2

    summary = bootstrap(
        manifest_dir,
        state_db,
        repo_root=repo_root,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        verb = "would insert" if args.dry_run else "inserted"
        print(
            f"gpufarm bootstrap @ {_now_iso()}\n"
            f"  manifest_dir: {summary['manifest_dir']}\n"
            f"  state_db:     {summary['state_db']}\n"
            f"  repo_root:    {summary['repo_root']}\n"
            f"  completed artifacts on disk: {summary['completed_artifacts_found']}\n"
            f"  {verb}: {summary['rows_inserted']} new COMPLETED rows\n"
            f"  skipped (already present):   {summary['rows_skipped_already_present']}\n"
            f"  skipped (no output_pattern): {summary['rows_skipped_missing_pattern']}\n"
            f"  skipped (ckpt cleaned_up):   {summary['skipped_ckpt_cleaned_up']}\n"
            f"  gaps left for the daemon:    {summary['gaps_left_for_daemon']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
