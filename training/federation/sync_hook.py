"""
Federation sync hook.

Imported by trainer scripts. Provides `maybe_sync()` which the trainer
calls after each step. When wall-clock interval has elapsed, performs:

  1. Save full state dict to <FED_DIR>/round_NNNN/<role>_state.pt
  2. Save metadata (tokens_since_last_sync, step, loss_ema)
  3. Touch <role>_ready marker
  4. Block until <FED_DIR>/round_NNNN/avg_state.pt exists
  5. Load avg_state.pt and broadcast across FSDP ranks
  6. Touch <role>_loaded marker
  7. Returns updated counters

For RTX6000 the local fed dir is mirrored to spark via rsync calls
embedded in this hook. For cluster the fed dir is on spark's local
filesystem so rank 0 writes/reads directly.

Environment variables:
    FED_SYNC_DIR              local fed working dir (must be set to enable federation)
    FED_ROLE                  "cluster" or "rtx"
    FED_REMOTE_HOST           SSH host for spark (only set on RTX6000 side)
    FED_REMOTE_DIR            fed dir on spark
    FED_SYNC_INTERVAL_SEC     wall-clock seconds between syncs (default: 1800)
    FED_POLL_INTERVAL_SEC     seconds between filesystem polls (default: 10)
    FED_AVG_WAIT_TIMEOUT      max seconds to wait for avg before declaring stuck (default: 7200)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

logger = logging.getLogger("fed.sync_hook")


class _FederationContext:
    """Per-process federation state. One instance per trainer process."""

    def __init__(self):
        self.fed_dir: Optional[Path] = None
        self.role: Optional[str] = None
        self.remote_host: Optional[str] = None
        self.remote_dir: Optional[Path] = None
        self.sync_interval_sec: float = 1800.0
        self.poll_interval_sec: float = 10.0
        self.avg_wait_timeout: float = 7200.0
        self.last_sync_time: float = 0.0
        self.tokens_since_last_sync: int = 0
        self.loss_ema: float = float("nan")
        self.local_round: int = 0
        self.enabled: bool = False

    def configure_from_env(self) -> bool:
        d = os.environ.get("FED_SYNC_DIR", "").strip()
        role = os.environ.get("FED_ROLE", "").strip().lower()
        if not d or role not in ("cluster", "rtx"):
            self.enabled = False
            return False
        self.fed_dir = Path(d)
        self.role = role
        self.remote_host = os.environ.get("FED_REMOTE_HOST", "").strip() or None
        rd = os.environ.get("FED_REMOTE_DIR", "").strip()
        self.remote_dir = Path(rd) if rd else None
        self.sync_interval_sec = float(os.environ.get("FED_SYNC_INTERVAL_SEC", "1800"))
        self.poll_interval_sec = float(os.environ.get("FED_POLL_INTERVAL_SEC", "10"))
        self.avg_wait_timeout = float(os.environ.get("FED_AVG_WAIT_TIMEOUT", "7200"))
        self.fed_dir.mkdir(parents=True, exist_ok=True)
        self.last_sync_time = time.time()  # don't fire immediately on first step
        self.enabled = True
        return True


_CTX = _FederationContext()


def is_enabled() -> bool:
    return _CTX.enabled


def configure() -> bool:
    """Read env vars and enable federation if FED_SYNC_DIR is set. Idempotent."""
    return _CTX.configure_from_env()


def update_loss_ema(loss_value: float, alpha: float = 0.05) -> None:
    """Trainer should call this each step with current loss. Used for telemetry only."""
    if not _CTX.enabled:
        return
    if not (loss_value == loss_value):  # NaN check
        return
    if _CTX.loss_ema != _CTX.loss_ema:
        _CTX.loss_ema = loss_value
    else:
        _CTX.loss_ema = (1 - alpha) * _CTX.loss_ema + alpha * loss_value


def add_tokens(n: int) -> None:
    """Trainer should call this each step with tokens processed (after grad accum)."""
    if not _CTX.enabled:
        return
    _CTX.tokens_since_last_sync += int(n)


# ---------------------------------------------------------------------------
# Coord state read (round number)
# ---------------------------------------------------------------------------

def _read_coord_round() -> int:
    """Get current round number from coord.state.json. Falls back to local counter."""
    if _CTX.role == "rtx" and _CTX.remote_host:
        # RTX side: pull coord.state.json from remote first
        local_copy = _CTX.fed_dir / "coord.state.json"
        try:
            subprocess.run(
                ["rsync", "-az", "--timeout=60",
                 f"{_CTX.remote_host}:{_CTX.remote_dir}/coord.state.json",
                 str(local_copy)],
                check=False,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
    coord_path = _CTX.fed_dir / "coord.state.json"
    if coord_path.exists():
        try:
            with open(coord_path) as f:
                data = json.load(f)
            return int(data.get("round", _CTX.local_round))
        except Exception as e:
            logger.warning("Failed to read coord state: %s", e)
    return _CTX.local_round


# ---------------------------------------------------------------------------
# Filesystem helpers (local + RTX-remote variants)
# ---------------------------------------------------------------------------

def _round_dir(round_num: int) -> Path:
    return _CTX.fed_dir / f"round_{round_num:04d}"


def _push_to_remote(local_path: Path, remote_subpath: str) -> bool:
    """Push a local file to spark via rsync. Only used on RTX side."""
    if not _CTX.remote_host or not _CTX.remote_dir:
        return True  # cluster side, no-op
    remote_full = f"{_CTX.remote_host}:{_CTX.remote_dir}/{remote_subpath}"
    # Make sure remote dir exists
    remote_dir = os.path.dirname(remote_full.split(":", 1)[1])
    try:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", _CTX.remote_host, f"mkdir -p {remote_dir}"],
            check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["rsync", "-az", "--timeout=300", str(local_path), remote_full],
            check=True, capture_output=True, timeout=900,
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning("rsync push failed for %s: %s", local_path, e)
        return False


def _pull_from_remote(remote_subpath: str, local_path: Path) -> bool:
    """Pull a file from spark via rsync. Only used on RTX side."""
    if not _CTX.remote_host or not _CTX.remote_dir:
        return True  # cluster side, no-op
    remote_full = f"{_CTX.remote_host}:{_CTX.remote_dir}/{remote_subpath}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["rsync", "-az", "--timeout=300", remote_full, str(local_path)],
            check=False,  # don't fail if file doesn't exist yet
            capture_output=True,
            timeout=900,
        )
        return local_path.exists()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


# ---------------------------------------------------------------------------
# Full state dict gather/scatter for FSDP
# ---------------------------------------------------------------------------

def _full_state_dict_to_disk(model, dst_path: Path) -> None:
    """Gather full state dict on rank 0 and save atomically. All ranks call."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if rank == 0:
        tmp = dst_path.with_suffix(".tmp")
        torch.save(sd, tmp)
        os.replace(tmp, dst_path)


def _load_full_state_dict_from_disk(model, src_path: Path) -> None:
    """Load full state dict on rank 0, scatter via FSDP. All ranks call."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        sd = torch.load(src_path, map_location="cpu")
    else:
        sd = None
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        # Non-rank-0 just need a stub dict
        if sd is None:
            sd = {}
        model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# The actual sync orchestration
# ---------------------------------------------------------------------------

def maybe_sync(model, step: int) -> bool:
    """Call after each training step. Returns True if a sync was performed."""
    if not _CTX.enabled:
        return False
    if (time.time() - _CTX.last_sync_time) < _CTX.sync_interval_sec:
        return False

    # Determine round number from coordinator's state. If coord state is
    # unavailable, use local counter (which still produces consistent round
    # dirs as long as coordinator catches up).
    round_num = _read_coord_round()
    rd = _round_dir(round_num)
    rd.mkdir(parents=True, exist_ok=True)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        logger.info("[fed] starting sync round %d (role=%s, tokens=%d, loss_ema=%.4f)",
                    round_num, _CTX.role, _CTX.tokens_since_last_sync, _CTX.loss_ema)

    # 1) Save full state dict (all ranks participate, rank 0 writes)
    state_path = rd / f"{_CTX.role}_state.pt"
    _full_state_dict_to_disk(model, state_path)

    if rank == 0:
        # 2) Write metadata
        meta = {
            "tokens_since_last_sync": int(_CTX.tokens_since_last_sync),
            "step": int(step),
            "loss_ema": float(_CTX.loss_ema) if _CTX.loss_ema == _CTX.loss_ema else None,
            "role": _CTX.role,
            "wall_unix": time.time(),
        }
        meta_path = rd / f"{_CTX.role}_meta.json"
        with open(meta_path.with_suffix(".tmp"), "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(meta_path.with_suffix(".tmp"), meta_path)

        # 3) On RTX side, push state + meta to spark
        if _CTX.role == "rtx":
            _push_to_remote(state_path, f"round_{round_num:04d}/rtx_state.pt")
            _push_to_remote(meta_path, f"round_{round_num:04d}/rtx_meta.json")

        # 4) Touch ready marker (and push to remote if RTX)
        ready_path = rd / f"{_CTX.role}_ready"
        ready_path.touch()
        if _CTX.role == "rtx":
            _push_to_remote(ready_path, f"round_{round_num:04d}/rtx_ready")

    # All ranks barrier here so non-rank-0 ranks don't proceed past save
    if dist.is_initialized():
        dist.barrier()

    # 5) Wait for averaged state (rank 0 polls, then all ranks load together)
    avg_path = rd / "avg_state.pt"
    deadline = time.time() + _CTX.avg_wait_timeout
    if rank == 0:
        while not avg_path.exists():
            if (_CTX.fed_dir / "halt").exists():
                raise RuntimeError("Federation halted by coordinator")
            if time.time() > deadline:
                raise TimeoutError(
                    f"avg_state.pt did not appear within {_CTX.avg_wait_timeout}s for round {round_num}"
                )
            # On RTX side, poll spark for avg
            if _CTX.role == "rtx":
                _pull_from_remote(f"round_{round_num:04d}/avg_state.pt", avg_path)
            time.sleep(_CTX.poll_interval_sec)
        logger.info("[fed] avg arrived for round %d (%d bytes)", round_num, avg_path.stat().st_size)

    if dist.is_initialized():
        dist.barrier()

    # 6) Load averaged state
    _load_full_state_dict_from_disk(model, avg_path)

    if dist.is_initialized():
        dist.barrier()

    # 7) Touch loaded marker
    if rank == 0:
        loaded_path = rd / f"{_CTX.role}_loaded"
        loaded_path.touch()
        if _CTX.role == "rtx":
            _push_to_remote(loaded_path, f"round_{round_num:04d}/rtx_loaded")
        logger.info("[fed] sync round %d complete; resetting tokens counter", round_num)

    # 8) Reset counters
    _CTX.tokens_since_last_sync = 0
    _CTX.last_sync_time = time.time()
    _CTX.local_round = round_num + 1
    return True
