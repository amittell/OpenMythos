"""
Checkpoint bootstrap dispatch logic for the 3b varT pondernet joint trainer.

Factored out of ``3b_varT_pondernet_joint.py`` so the decision tree
(resume-from-shards vs. auto-shard-from-full vs. fresh-start) is testable
without torch/FSDP/CUDA -- the function depends only on the stdlib.

OpenMythos #156: when the cluster supervisor starts a new round with a
fresh ``CKPT_DIR`` and points ``BOOTSTRAP_CKPT`` at a consolidated
``*_full.pt``, the trainer should auto-shard-from-full rather than
require the operator to pre-place per-rank shards on every node.

The three modes returned here map directly onto the three branches in
``main()``:

* ``"resume_shards"``  -- existing sharded ckpts in ``CKPT_DIR``; load
  via ``load_checkpoint`` (model + optimizer + step).
* ``"bootstrap_full"`` -- ``CKPT_DIR`` empty AND ``BOOTSTRAP_CKPT`` set;
  load weights from the full ckpt via ``bootstrap_model_weights`` and let
  FSDP re-shard in memory. Optimizer is fresh, step counter is 0.
* ``"fresh_start"``    -- no shards, no bootstrap source; random init.

Notes:

* The previous trainer behaviour also auto-discovered round-2.2 / round-2.1
  consolidated ckpts when ``BOOTSTRAP_CKPT`` was unset. That auto-discovery
  is preserved by the caller (``main()``) -- the dispatch helper only
  reasons about the explicit ``BOOTSTRAP_CKPT`` override, which is the
  case OpenMythos #156 cares about. When the helper returns
  ``"fresh_start"``, the caller may still run its legacy auto-discovery
  before falling through to random init.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BootstrapDecision:
    """Result of inspecting CKPT_DIR + BOOTSTRAP_CKPT for the current rank.

    Attributes:
        mode -- one of ``"resume_shards"``, ``"bootstrap_full"``,
                ``"fresh_start"``.
        shard_path -- when ``mode == "resume_shards"``, the absolute path
                to THIS rank's latest shard file (suitable for passing
                straight to ``load_checkpoint``). ``None`` otherwise.
        bootstrap_path -- when ``mode == "bootstrap_full"``, the absolute
                or relative path to the consolidated ``*_full.pt`` file
                that should be loaded by ``bootstrap_model_weights``.
                ``None`` otherwise.
    """

    mode: str
    shard_path: Optional[str] = None
    bootstrap_path: Optional[str] = None


def _list_rank_shards(ckpt_dir: str, rank: int) -> list[str]:
    """List ``step_*_rank{rank}.pt`` files in ``ckpt_dir``, oldest first.

    Mirrors the sharded-flavour branch of the trainer's ``_list_ckpts``
    helper but is intentionally narrower: this dispatcher does NOT
    consider legacy full-state-dict shards (``step_N.pt`` without a
    ``_rank`` suffix) as resumable, because the only callers that ever
    wrote that flavour are pre-r2.1 runs that have all since been
    consolidated. Restricting to the rank-suffixed form keeps the
    dispatch decision unambiguous.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    suffix = f"_rank{rank}.pt"
    out = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_") and name.endswith(suffix):
            out.append(os.path.join(ckpt_dir, name))
    return sorted(out)


def resolve_bootstrap_mode(
    ckpt_dir: str,
    rank: int,
    bootstrap_ckpt: Optional[str],
) -> BootstrapDecision:
    """Decide how the trainer should initialise model/optimizer state.

    Args:
        ckpt_dir -- the per-rank-local directory where this round's
            sharded ckpts live (``step_*_rank{rank}.pt``). May not yet
            exist on a first launch.
        rank -- this process's distributed rank. Used to scope the
            sharded-file search to THIS rank's shard so the decision
            is correct even when other ranks' shard files have not
            yet been rsync'd into ``ckpt_dir`` (they won't be -- each
            rank's shard lives on its own node's local disk).
        bootstrap_ckpt -- value of the ``BOOTSTRAP_CKPT`` env var. May
            be ``None`` or an empty/whitespace string when unset.

    Returns:
        A ``BootstrapDecision`` whose ``mode`` field tells the caller
        which branch to take.

    Precedence (matches OpenMythos #156's scope rules):
        1. Sharded ckpts present in ``ckpt_dir`` for this rank
           -> ``"resume_shards"``. This is the crash-restart path.
        2. ``ckpt_dir`` empty AND ``bootstrap_ckpt`` is a non-empty
           string that points at an existing path -> ``"bootstrap_full"``.
           The trainer will rank-0-broadcast-load the full ckpt and let
           FSDP re-shard in memory.
        3. Otherwise -> ``"fresh_start"``. The caller may still run its
           legacy round-2.2/round-2.1 auto-discovery before deciding
           to random-init.

    Implementation notes:
        * ``bootstrap_ckpt`` is ``.strip()``-checked because env-var
          plumbing through shell -> ssh -> torchrun -> Python often leaves
          stray whitespace, and we don't want a whitespace-only string
          to trigger the bootstrap path with a missing file.
        * Path existence of ``bootstrap_ckpt`` is checked here rather
          than letting ``torch.load`` raise on every rank, so the trainer
          can emit a clear single-rank error message before the broadcast.
        * If ``bootstrap_ckpt`` is set but the file does not exist, we
          return ``"fresh_start"`` with ``bootstrap_path = bootstrap_ckpt``
          so the caller can log the attempted path and decide whether to
          hard-fail or fall back to auto-discovery.
    """
    shards = _list_rank_shards(ckpt_dir, rank)
    if shards:
        return BootstrapDecision(
            mode="resume_shards",
            shard_path=shards[-1],
        )

    if bootstrap_ckpt is None:
        return BootstrapDecision(mode="fresh_start")

    bp = bootstrap_ckpt.strip()
    if not bp:
        return BootstrapDecision(mode="fresh_start")

    if not os.path.exists(bp):
        # File missing: caller decides how loud to be. Surface the
        # attempted path so the log line is actionable.
        return BootstrapDecision(
            mode="fresh_start",
            bootstrap_path=bp,
        )

    return BootstrapDecision(
        mode="bootstrap_full",
        bootstrap_path=bp,
    )
