"""
Unit tests for ``training/bootstrap_dispatch.py``.

The dispatch logic decides at trainer startup whether the script should
resume from sharded ckpts in ``CKPT_DIR``, auto-shard-from-full via the
``BOOTSTRAP_CKPT`` env var, or fall through to fresh init / legacy
auto-discovery. The decision is the load-bearing part of OpenMythos #156
and is the only part of the trainer that can be exercised on a workstation
without torch + 4-node FSDP.

Each test isolates one branch of ``resolve_bootstrap_mode`` so a regression
will name the broken branch directly in the failure message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The dispatch helper lives under ``training/`` (not under the
# ``open_mythos`` package) because the trainer scripts are flat modules.
# Add it to sys.path so this test runs cleanly under ``pytest`` from the
# repo root or from inside ``tests/``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from bootstrap_dispatch import (  # noqa: E402
    BootstrapDecision,
    resolve_bootstrap_mode,
)


def _touch(path: Path) -> None:
    """Create an empty file, including any missing parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# ---------------------------------------------------------------------------
# resume_shards: CKPT_DIR has this rank's shards
# ---------------------------------------------------------------------------


def test_resume_shards_picks_latest_for_this_rank(tmp_path: Path) -> None:
    """Two shard files for rank 0 -> mode=resume_shards, picks the newer."""
    ckpt_dir = tmp_path / "ckpts"
    _touch(ckpt_dir / "step_0000100_rank0.pt")
    _touch(ckpt_dir / "step_0000400_rank0.pt")
    # Other ranks' shards live on other nodes' disks; simulate by creating
    # a stray rank1 file -- the dispatcher must ignore it.
    _touch(ckpt_dir / "step_0000400_rank1.pt")

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt="/some/other/path_full.pt",  # ignored when shards present
    )

    assert decision.mode == "resume_shards"
    assert decision.shard_path == str(ckpt_dir / "step_0000400_rank0.pt")
    assert decision.bootstrap_path is None


def test_resume_shards_per_rank_scoping(tmp_path: Path) -> None:
    """rank=2 only sees rank2 shards even if rank0 shards are also present.

    Each node's CKPT_DIR is local disk; in production rank2 wouldn't see
    rank0 shards at all, but we still want the dispatcher to scope by
    rank suffix so a misconfigured shared mount doesn't cause rank2 to
    try to resume from rank0's file.
    """
    ckpt_dir = tmp_path / "ckpts"
    _touch(ckpt_dir / "step_0000200_rank0.pt")
    _touch(ckpt_dir / "step_0000200_rank2.pt")
    _touch(ckpt_dir / "step_0000400_rank0.pt")

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=2,
        bootstrap_ckpt=None,
    )

    assert decision.mode == "resume_shards"
    assert decision.shard_path == str(ckpt_dir / "step_0000200_rank2.pt")


# ---------------------------------------------------------------------------
# bootstrap_full: CKPT_DIR empty, BOOTSTRAP_CKPT points at existing file
# ---------------------------------------------------------------------------


def test_bootstrap_full_when_ckpt_dir_empty_and_full_pt_exists(
    tmp_path: Path,
) -> None:
    """The OpenMythos #156 happy path: fresh CKPT_DIR + existing full.pt."""
    ckpt_dir = tmp_path / "ckpts_round221"
    ckpt_dir.mkdir()
    full_pt = tmp_path / "round218" / "step_0024414_full.pt"
    _touch(full_pt)

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt=str(full_pt),
    )

    assert decision.mode == "bootstrap_full"
    assert decision.bootstrap_path == str(full_pt)
    assert decision.shard_path is None


def test_bootstrap_full_when_ckpt_dir_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    """First launch: supervisor sets CKPT_DIR to a path that doesn't exist."""
    ckpt_dir = tmp_path / "never_created"
    full_pt = tmp_path / "round218" / "step_0024414_full.pt"
    _touch(full_pt)

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=3,
        bootstrap_ckpt=str(full_pt),
    )

    assert decision.mode == "bootstrap_full"
    assert decision.bootstrap_path == str(full_pt)


def test_bootstrap_full_per_rank_decision_is_identical(tmp_path: Path) -> None:
    """All four ranks must agree on the mode; otherwise FSDP collectives hang.

    The dispatcher doesn't broadcast -- the agreement is structural (all
    ranks see the same empty CKPT_DIR and the same BOOTSTRAP_CKPT env var)
    -- but the test pins the invariant so a future change can't make the
    decision rank-dependent in the bootstrap_full case.
    """
    ckpt_dir = tmp_path / "fresh"
    full_pt = tmp_path / "src_full.pt"
    _touch(full_pt)

    decisions = [
        resolve_bootstrap_mode(
            ckpt_dir=str(ckpt_dir),
            rank=r,
            bootstrap_ckpt=str(full_pt),
        )
        for r in range(4)
    ]
    modes = {d.mode for d in decisions}
    assert modes == {"bootstrap_full"}, (
        f"All ranks must agree in bootstrap_full case, got modes={modes}"
    )


# ---------------------------------------------------------------------------
# fresh_start: BOOTSTRAP_CKPT unset, empty, or pointing at missing file
# ---------------------------------------------------------------------------


def test_fresh_start_when_bootstrap_ckpt_is_none(tmp_path: Path) -> None:
    """No shards + no env var -> caller falls through to legacy auto-discovery."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt=None,
    )

    assert decision.mode == "fresh_start"
    assert decision.bootstrap_path is None
    assert decision.shard_path is None


def test_fresh_start_when_bootstrap_ckpt_is_empty_string(tmp_path: Path) -> None:
    """``BOOTSTRAP_CKPT=`` in EXTRA_ENV must not trip the bootstrap branch."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt="",
    )

    assert decision.mode == "fresh_start"
    assert decision.bootstrap_path is None


def test_fresh_start_when_bootstrap_ckpt_is_whitespace(tmp_path: Path) -> None:
    """Shell plumbing can leave stray whitespace; treat as unset, not as a path."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt="   \t  ",
    )

    assert decision.mode == "fresh_start"
    assert decision.bootstrap_path is None


def test_fresh_start_surfaces_missing_file_path(tmp_path: Path) -> None:
    """When BOOTSTRAP_CKPT points to a missing file, surface the path so the
    caller can log a clear warning before falling through to auto-discovery."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    missing = tmp_path / "does_not_exist_full.pt"

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt=str(missing),
    )

    assert decision.mode == "fresh_start"
    assert decision.bootstrap_path == str(missing)


# ---------------------------------------------------------------------------
# Precedence: shards beat BOOTSTRAP_CKPT (crash-restart must NOT re-bootstrap)
# ---------------------------------------------------------------------------


def test_resume_shards_takes_precedence_over_bootstrap_ckpt(
    tmp_path: Path,
) -> None:
    """Crash-restart path: even if BOOTSTRAP_CKPT is still set in EXTRA_ENV,
    a freshly-written shard for this rank must resume from disk, not re-read
    the 6 GB full.pt and re-broadcast it across the cluster."""
    ckpt_dir = tmp_path / "ckpts"
    _touch(ckpt_dir / "step_0000200_rank0.pt")
    full_pt = tmp_path / "src_full.pt"
    _touch(full_pt)

    decision = resolve_bootstrap_mode(
        ckpt_dir=str(ckpt_dir),
        rank=0,
        bootstrap_ckpt=str(full_pt),
    )

    assert decision.mode == "resume_shards"
    assert decision.shard_path == str(ckpt_dir / "step_0000200_rank0.pt")
    # bootstrap_path is intentionally not populated in resume mode; the
    # crash-restart path must not even consider re-reading the full.pt.
    assert decision.bootstrap_path is None


# ---------------------------------------------------------------------------
# Decision object is hashable/equatable (callers may stash in metadata)
# ---------------------------------------------------------------------------


def test_decision_is_frozen_dataclass() -> None:
    d = BootstrapDecision(mode="fresh_start")
    with pytest.raises(Exception):
        # Frozen dataclass mutation raises; the exact exception type is
        # ``dataclasses.FrozenInstanceError`` which subclasses AttributeError.
        d.mode = "bootstrap_full"  # type: ignore[misc]
