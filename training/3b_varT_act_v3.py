#!/usr/bin/env python3
"""
mythos_3b round 2.2 — continuation of the ACT-bypassed regime, with throughput
optimizations queued up for the post-round-2.1 run.

Round 2.1 (3b_varT_act_v2.py) is the ACT-bypass regime test at 50M tokens.
Round 2.2 either extends 2.1 with more tokens (if 2.1 shows depth-using
behavior) or re-runs the regime under different hyperparameters. Same
recurrent_forward_no_act monkey-patch — gradient must flow through every
iteration of the recurrent loop.

Optimizations bolted on (vs v2):
  - ckpt_every=200 (was 100): halves checkpoint I/O cost (~3% wall-clock)
  - Async sharded checkpoint writes via ThreadPoolExecutor: overlaps disk
    write with subsequent training steps (~3-5% wall-clock)
  - Activation checkpointing on TransformerBlock as opt-in via USE_ACT_CKPT=1:
    re-runs forward during backward, saves activation memory, allows trying
    micro_batch=2 (uncertain net win, must validate per-run)
  - All hyperparameters env-overridable for cheap A/B testing:
      TARGET_TOKENS, MICRO_BATCH, GRAD_ACCUM, USE_ACT_CKPT,
      CKPT_EVERY, LR, WARMUP_STEPS

Optimizations rejected (with reasons):
  - micro_batch=2 alone: OOM-killed rank 0 in prior tests (MoE expert
    unshards push past 128 GB unified memory)
  - NO_SHARD: adds ~33 GB optimizer state per rank, won't fit
  - torch.compile: 10-34% slower in smoke tests on this architecture
    (dynamic n_loops + MoE + ACT branches are compile-hostile)

Bootstrap path:
  - Auto-discovers the latest checkpoints_3b_varT_act_v2/step_*_full.pt
    (consolidated round-2.1 final) and loads model-only weights
  - Optimizer is fresh (round 2.2 has its own LR schedule)
  - If no consolidated v2 ckpt exists, bootstraps from round 2 (the
    checkpoints_3b_varT_fast/*_full.pt path) as a fallback

Inherits from round 2.1:
  - recurrent_forward_no_act monkey-patch: bypass ACT, run all T iters
  - Variable T sampled uniformly from [T_MIN, T_MAX] per optimizer step
  - FSDP SHARD_GRAD_OP + bf16 mixed precision
  - Sharded per-rank checkpoints
  - target_tokens, lr, warmup default to round-2.1 values
"""

import glob
import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, Future
from loguru import logger
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock, loop_index_embedding
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer

# Federated training hook (no-op unless FED_SYNC_DIR is set)
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "federation"))
try:
    from federation import sync_hook as _fed_sync_hook  # type: ignore
except ImportError:
    import sync_hook as _fed_sync_hook  # type: ignore


def recurrent_forward_no_act(
    self: RecurrentBlock,
    h: torch.Tensor,
    e: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask=None,
    n_loops=None,
    kv_cache=None,
) -> torch.Tensor:
    """
    Drop-in replacement for RecurrentBlock.forward that bypasses ACT halting
    entirely. Runs n_loops iterations unconditionally and returns the final
    hidden state h_T (not the ACT-weighted sum h_out).

    Used for round 2.1 training only: forces gradient to flow through every
    iteration so the model cannot shortcut depth via ACT saturation.
    """
    n_loops = n_loops or self.cfg.max_loop_iters
    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h = self.injection(h, e, trans_out)
    return h


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → half-cosine decay to `min_lr`.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to `min_lr` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)

    Args:
        step    -- current global optimizer step (0-indexed)
        warmup  -- number of warmup steps before cosine decay begins
        total   -- step at which the cosine reaches `min_lr`
        max_lr  -- peak learning rate reached at the end of warmup
        min_lr  -- floor learning rate at and after `total` steps

    Returns:
        Scalar learning rate for this step.
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str, rank: int = 0) -> list[str]:
    """
    Return checkpoint paths visible to this rank's local disk, oldest → newest.

    Handles both checkpoint flavours:
      * Legacy full-state-dict: `step_0000000.pt` (one file, all ranks).
      * Sharded per-rank: `step_0000000_rank{R}.pt` (one file per rank on
        its own local disk).

    For the sharded flavour we return only files whose rank suffix matches
    `rank` — other ranks' shards are on other nodes' disks anyway, but this
    keeps the list clean and makes resume-latest do the right thing.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []
        rank     -- current rank index; used to pick the caller's shard file
                    in the sharded flavour

    Returns:
        Sorted list of absolute paths. Sort order is by step number via the
        zero-padded filename, so lexicographic sort = chronological.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    my_suffix = f"_rank{rank}.pt"
    for f in os.listdir(ckpt_dir):
        if not f.startswith("step_"):
            continue
        if f.endswith(my_suffix) or (f.endswith(".pt") and "_rank" not in f):
            out.append(os.path.join(ckpt_dir, f))
    return sorted(out)


def _step_of(path: str) -> int:
    """Extract the integer step number from either flavour of checkpoint path."""
    name = os.path.basename(path)
    # "step_0000400_rank0.pt" or "step_0000400.pt"
    core = name[len("step_"):-len(".pt")]
    if "_rank" in core:
        core = core.split("_rank", 1)[0]
    return int(core)


# Single-worker executor for async checkpoint I/O. The FSDP state_dict
# collective MUST stay synchronous (all ranks gather together), but the
# subsequent torch.save + atomic rename + prune is local-disk-only and
# can be overlapped with the next training step. One worker (not a pool)
# because sequential I/O on the same disk is the bottleneck — extra
# threads just contend.
_ckpt_executor: ThreadPoolExecutor | None = None
_ckpt_pending: Future | None = None


def _get_ckpt_executor() -> ThreadPoolExecutor:
    """Lazily construct the per-process checkpoint I/O executor."""
    global _ckpt_executor
    if _ckpt_executor is None:
        _ckpt_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-io"
        )
    return _ckpt_executor


def wait_for_pending_ckpts() -> None:
    """
    Block until any in-flight async checkpoint write has completed and the
    executor is shut down. Call at the end of training so the run doesn't
    exit while a final write is still draining to disk.
    """
    global _ckpt_pending, _ckpt_executor
    if _ckpt_pending is not None:
        _ckpt_pending.result()
        _ckpt_pending = None
    if _ckpt_executor is not None:
        _ckpt_executor.shutdown(wait=True)
        _ckpt_executor = None


def _save_payload_to_disk(
    payload: dict,
    final_path: str,
    keep_last: int,
    ckpt_dir: str,
    rank: int,
) -> None:
    """
    I/O-only half of save_checkpoint. Runs in the executor thread:
    writes to a temp file, atomically renames, then prunes old shards
    from the same rank's local disk. Anything collective lives in the
    caller — by the time we get here, the state_dict is already gathered
    and detached from NCCL.
    """
    tmp_path = final_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    for old in _list_ckpts(ckpt_dir, rank=rank)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Sharded save: each rank writes its own slice of the FSDP state to its own
    local disk. No NCCL all-gather to rank 0, no 45 GB file on one node, no
    cross-node rsync. Total save time drops from ~7 min (round 1) to ~90 sec.

    Filename convention: `step_{N:07d}_rank{R}.pt`. Each rank pulls its shard
    via `FSDP.state_dict_type(SHARDED_STATE_DICT)`, saves atomically via
    tempfile + os.replace, and prunes its own older files. Because all ranks
    save in lockstep (this function is collective), they all agree on which
    steps exist and which to prune.

    Args:
        model       -- FSDP-wrapped model (ddp=True path is the main use;
                       single-GPU falls back to plain torch.save)
        optimizer   -- the optimizer whose state round-trips with the model
        step        -- global step number; zero-padded into the filename
        cfg         -- model config object, saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- rank-0 flag (used only for the end-of-save log line)
        keep_last   -- number of most-recent checkpoints to retain per rank

    Returns:
        None. Writes to local disk as a side effect on every rank.
    """
    global _ckpt_pending
    os.makedirs(ckpt_dir, exist_ok=True)

    # Drain the previous async write before issuing a new one. This caps
    # in-flight writes at one and surfaces any I/O exception in the
    # main-thread context (otherwise it would hide in the future result).
    if _ckpt_pending is not None:
        _ckpt_pending.result()
        _ckpt_pending = None

    if not ddp:
        # Single-GPU path stays synchronous: state_dict() returns live tensor
        # refs, and adding an async clone would cost more than the save itself
        # at this scale. Async I/O is only worth the complexity for the FSDP
        # path where offload_to_cpu already gives us a safe snapshot.
        if master:
            final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            payload = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "vocab_size": vocab_size,
            }
            _save_payload_to_disk(payload, final_path, keep_last, ckpt_dir, 0)
            logger.success(f"Checkpoint saved -> {final_path}")
        return

    rank = dist.get_rank()
    # offload_to_cpu=True is critical for the async path: the executor
    # thread will torch.save these tensors after this function returns,
    # while the main thread continues training. If the snapshot stayed on
    # GPU, the optimizer step on the next iteration could mutate the same
    # storage the writer is reading. Offload makes it a CPU-side copy that
    # is decoupled from the live model.
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    # Barrier here (not after the I/O) ensures every rank has detached its
    # state_dict from NCCL before any rank starts touching disk. After this
    # barrier the per-rank writes are independent — no cross-rank ordering
    # needed for prune, since each rank only sees its own local files.
    dist.barrier()

    final_path = os.path.join(ckpt_dir, f"step_{step:07d}_rank{rank}.pt")
    payload = {
        "step": step,
        "model": model_state,
        "optimizer": optim_state,
        "cfg": cfg,
        "vocab_size": vocab_size,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "sharded": True,
    }
    _ckpt_pending = _get_ckpt_executor().submit(
        _save_payload_to_disk,
        payload,
        final_path,
        keep_last,
        ckpt_dir,
        rank,
    )

    if master:
        logger.success(
            f"Sharded checkpoint write queued -> step {step} across "
            f"{dist.get_world_size()} ranks (each rank writes its own shard "
            f"asynchronously to local disk)"
        )


def _distribute_checkpoint(final_path: str, keep_last: int = 3) -> None:
    """
    rsync `final_path` from rank 0 to every worker's local disk over the 200G
    fabric, then prune each worker's checkpoint dir to match rank 0's
    keep_last retention policy. Cluster-specific: hardcodes the 4-node Spark
    cluster's 200G IPs.

    Runs rsyncs in parallel, blocks until they all complete. Typical transfer
    time for a 42 GB checkpoint: ~1 min 20 sec per worker at ~540 MB/s on
    200G RoCE (all three in parallel, so ~90 sec total per save). Remote
    pruning uses `ls | sort | head -n -keep_last | xargs rm` (GNU head),
    deleting all but the most recent `keep_last` checkpoints on the peer.
    Without the prune, workers accumulated every rsynced checkpoint in round
    1 and hit disk-full (~45 GB per .pt file, 900 GB NVMe filled in ~20 saves).

    Fail-soft: any rsync or ssh error is logged but not raised.
    """
    import subprocess
    worker_ips = ["192.168.100.11", "192.168.100.12", "192.168.100.13"]
    ssh_key = os.path.expanduser("~/.ssh/id_ed25519_shared")
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-i", ssh_key]
    ckpt_dir = os.path.dirname(os.path.abspath(final_path))
    procs = []
    for ip in worker_ips:
        cmd = [
            "rsync", "-az",
            "-e", f"ssh {' '.join(ssh_opts)}",
            final_path, f"alexm@{ip}:{ckpt_dir}/",
        ]
        procs.append((ip, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    for ip, p in procs:
        try:
            rc = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            logger.warning(f"Checkpoint rsync to {ip} timed out after 10 min")
            continue
        if rc != 0:
            err = (p.stderr.read() or b"").decode("utf-8", errors="replace")[:200]
            logger.warning(f"Checkpoint rsync to {ip} failed (rc={rc}): {err}")
            continue

        prune_shell = (
            f"cd {ckpt_dir} && ls -1 step_*.pt 2>/dev/null | sort "
            f"| head -n -{keep_last} | xargs -r rm -f"
        )
        prune_cmd = ["ssh", *ssh_opts, f"alexm@{ip}", prune_shell]
        try:
            pr = subprocess.run(
                prune_cmd, timeout=60, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            if pr.returncode != 0:
                err = (pr.stderr or b"").decode("utf-8", errors="replace")[:200]
                logger.warning(f"Checkpoint prune on {ip} failed (rc={pr.returncode}): {err}")
            else:
                logger.info(f"Checkpoint distributed and pruned on {ip} (keep_last={keep_last})")
        except subprocess.TimeoutExpired:
            logger.warning(f"Checkpoint prune on {ip} timed out after 60s")


def bootstrap_model_weights(model, path: str, ddp: bool) -> None:
    """
    Load ONLY the model weights from a full-state-dict ckpt; leave the
    optimizer untouched. Used to seed round 2.1 from round-2's consolidated
    ckpt, which was written by `consolidate_ckpt.py` and contains no optimizer
    state (model + cfg + step + vocab_size only).

    Mirrors load_checkpoint's FSDP FULL_STATE_DICT path but skips the
    `optim_state_dict_to_load` call that would crash on missing optimizer key.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not ddp:
        model.load_state_dict(ckpt["model"])
        return
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(ckpt["model"])


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    """
    Restore model + optimizer from disk, returning the step to resume at.

    Dispatches by filename so both checkpoint flavours work:

      * Legacy full-state-dict (`step_N.pt` with no `_rank` suffix): every
        rank reads the same 45 GB file under
        `FSDP.state_dict_type(FULL_STATE_DICT, rank0_only=False)`. Used when
        resuming from round-1 checkpoints.

      * Sharded per-rank (`step_N_rankR.pt`): each rank reads ITS OWN local
        shard file under `FSDP.state_dict_type(SHARDED_STATE_DICT)`. No
        cross-rank file sharing, no rsync dependency.

    `weights_only=False` is required because the checkpoint contains the cfg
    dataclass serialized via torch.save.

    Args:
        model     -- same FSDP-wrapped or raw model used during save
        optimizer -- freshly constructed optimizer to be filled in-place
        path      -- absolute path to a checkpoint file. For the sharded
                     flavour, pass the current rank's `_rankR.pt` file; the
                     loader uses the suffix to detect the flavour.
        ddp       -- whether the model is FSDP-wrapped; must match the save run

    Returns:
        The step number the checkpoint was taken at.
    """
    name = os.path.basename(path)
    is_sharded = "_rank" in name

    if not ddp:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt["step"])

    if is_sharded:
        # Each rank reads its OWN rank's file from its local disk. The caller
        # has already localised the path to this rank (see main()'s resume
        # logic); we still swap in the correct rank suffix as a safety net in
        # case a full-state-dict pathname was passed by mistake.
        rank = dist.get_rank()
        if not name.endswith(f"_rank{rank}.pt"):
            base_dir = os.path.dirname(path)
            step_prefix = name.split("_rank", 1)[0]
            path = os.path.join(base_dir, f"{step_prefix}_rank{rank}.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(),
            optim_state_dict_config=ShardedOptimStateDictConfig(),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
        return int(ckpt["step"])

    # Legacy full-state-dict path.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(ckpt["model"])
        optim_state = FSDP.optim_state_dict_to_load(
            model=model,
            optim=optimizer,
            optim_state_dict=ckpt["optimizer"],
        )
        optimizer.load_state_dict(optim_state)
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    End-to-end pretraining entry point.

    Order matters: distributed init must run before any CUDA allocation, the
    tokenizer must exist before the model is built (vocab_size flows into
    cfg), and FSDP must wrap the model before the optimizer is constructed
    (FSDP re-flattens parameters, so an optimizer built on the unwrapped
    model would track stale param objects). Resume then loads state into the
    already-constructed optimizer in-place.

    Lifecycle:
        1. Initialize torch.distributed (NCCL) if launched under torchrun.
        2. Build tokenizer → derive vocab_size.
        3. Construct OpenMythos with the 3B variant config.
        4. Wrap in FSDP with FULL_SHARD + bf16/fp16 mixed precision (multi-GPU)
           or move to device + autocast (single-GPU).
        5. Build fused AdamW on (possibly sharded) parameters.
        6. Resume from the latest checkpoint in `ckpt_dir` if one exists.
        7. Stream FineWeb-Edu through grad-accumulation microbatches with
           cosine LR schedule, per-step logging, and periodic checkpoints.
        8. Write a final checkpoint if the last save wasn't aligned to
           `ckpt_every`, then barrier + tear down the process group.

    All hyperparameters are literal constants in this function by design —
    pretraining runs are long-lived and each run pins exact settings; a
    CLI/config layer is deliberately avoided to keep the file self-auditable.
    """
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1

    # Configure federated sync if FED_SYNC_DIR is set in env. No-op otherwise.
    if _fed_sync_hook.configure():
        logger.info(f"[fed] federated training enabled (role={os.environ.get('FED_ROLE')}, "
                    f"interval={os.environ.get('FED_SYNC_INTERVAL_SEC', '1800')}s)")
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters (env-overridable for cheap A/B testing)
    # ------------------------------------------------------------------
    seq_len = 1024
    # micro_batch defaults to 1 — known stable. micro_batch=2 has historically
    # OOM-killed rank 0 (MoE expert unshards push past 128 GB unified). Try
    # MICRO_BATCH=2 only with USE_ACT_CKPT=1 (which trades 30% extra compute
    # for activation memory headroom). A 5-min OOM-test launch is recommended
    # before committing to a multi-hour run with non-default settings.
    micro_batch = int(os.environ.get("MICRO_BATCH", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    target_tokens = int(os.environ.get("TARGET_TOKENS", "50000000"))
    use_act_ckpt = os.environ.get("USE_ACT_CKPT", "0") == "1"
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    # LR schedule: brief warmup to peak LR, cosine decay to 10% of peak.
    # Defaults match round 2.1 (post-cosine continuation, no re-spike).
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "200"))
    lr = float(os.environ.get("LR", "1e-4"))
    wd = 0.1
    log_every = 5
    # ckpt_every=200 (was 100 in v2): halves checkpoint I/O cost. Combined
    # with async writes (see save_checkpoint_async), the ckpt overhead is
    # mostly hidden behind subsequent training steps.
    ckpt_every = int(os.environ.get("CKPT_EVERY", "200"))
    ckpt_dir = os.environ.get("CKPT_DIR", "checkpoints_3b_varT_act_v3")
    dataset_subset = os.environ.get("DATASET_SUBSET", "sample-10BT")

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )
        logger.info(
            f"lr={lr} | warmup_steps={warmup_steps} | ckpt_every={ckpt_every} | "
            f"use_act_ckpt={use_act_ckpt} | ckpt_dir={ckpt_dir}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    # Variable-T: sample T ~ Uniform(T_MIN, T_MAX) per optimizer step.
    # cfg.max_loop_iters drives the LoRAAdapter embedding table size, so it
    # must be at least T_MAX for every sampled depth to index a learned slot.
    T_MIN = 2
    T_MAX = 12
    # Optional: fix T to a single value (e.g., T_FIXED=8) for fixed-depth
    # training as a baseline against variable-T. Must satisfy T_MIN <= T <= T_MAX.
    T_FIXED_ENV = os.environ.get("T_FIXED", "").strip()
    T_FIXED = int(T_FIXED_ENV) if T_FIXED_ENV else 0
    if T_FIXED and not (1 <= T_FIXED <= T_MAX):
        raise ValueError(f"T_FIXED={T_FIXED} not in [1,{T_MAX}]")
    cfg.max_loop_iters = T_MAX

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    # ACT bypass (inherited from round 2.1): the recurrent loop runs all T
    # iterations and returns h_T; gradient flows through every iteration.
    # Applied before FSDP wrap, so the patched method is what FSDP sees.
    RecurrentBlock.forward = recurrent_forward_no_act
    if master:
        logger.info("ACT bypass active: RecurrentBlock returns h_T (all iters run)")

    # Activation checkpointing on TransformerBlock (opt-in via USE_ACT_CKPT=1).
    # During backward, the checkpointed module's forward is re-executed to
    # regenerate activations rather than holding them in memory. With T~7
    # recurrent iters and ~32 layers, this frees substantial activation
    # memory at the cost of ~1 extra forward per backward (~30% step time).
    # Enables trying micro_batch=2 without the prior MoE-unshard OOM.
    # NON_REENTRANT impl is the modern choice — interacts cleanly with FSDP,
    # avoids the autograd-recursion edge cases of the legacy REENTRANT impl.
    if use_act_ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )
        if master:
            logger.info(
                "Activation checkpointing enabled on TransformerBlock"
            )

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
            cast_forward_inputs=True,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        # SHARD_GRAD_OP: params stay replicated per rank (6 GB bf16 each),
        # but grads + optim state are sharded. Skips the forward all-gather.
        # We have ~70 GB headroom/rank on 128 GB unified memory, so the full
        # param replication fits easily.
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    # FSDP handles its own mixed precision; only need autocast for single-GPU
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    # Streaming datasets are not resumable by position, so re-iterating from
    # the beginning is accepted — at pretraining scale the loss of dataset
    # position is negligible vs. the cost of discarded training steps.
    #
    # Bootstrap precedence:
    #   1. Existing sharded ckpts in `ckpt_dir` (continuing a round-2.2 run)
    #   2. Latest consolidated full-state-dict in checkpoints_3b_varT_act_v2/
    #      (round 2.1 final — primary bootstrap path for round 2.2)
    #   3. Round 2 final (checkpoints_3b_varT_fast/step_0012207_full.pt) as
    #      a fallback if v2 hasn't been consolidated yet
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir, rank=rank if ddp else 0)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming round 2.2 checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")
    else:
        # First launch: bootstrap model weights from the latest available
        # consolidated full-state-dict. Optimizer is fresh.
        # Precedence:
        #   1. BOOTSTRAP_CKPT env var (explicit override; round 2.5 uses
        #      this to pin the round-2 collapsed ckpt for the fixed-T baseline).
        #   2. round-2.1 final (`checkpoints_3b_varT_act_v2/`).
        #   3. round-2 final (`checkpoints_3b_varT_fast/step_0012207_full.pt`).
        bootstrap_path = None
        bootstrap_label = ""
        bootstrap_override = os.environ.get("BOOTSTRAP_CKPT", "").strip()
        if bootstrap_override:
            bootstrap_path = bootstrap_override
            bootstrap_label = f"BOOTSTRAP_CKPT override ({bootstrap_override})"
        else:
            v2_candidates = sorted(
                glob.glob("checkpoints_3b_varT_act_v2/step_*_full.pt")
            )
            round2_full = "checkpoints_3b_varT_fast/step_0012207_full.pt"
            if v2_candidates:
                bootstrap_path = v2_candidates[-1]
                bootstrap_label = "round-2.1 final"
            elif os.path.exists(round2_full):
                bootstrap_path = round2_full
                bootstrap_label = "round-2 final (fallback; v2 not consolidated)"

        if bootstrap_path is not None:
            if master:
                logger.info(
                    f"Bootstrapping from {bootstrap_label}: {bootstrap_path}"
                )
            bootstrap_model_weights(model, bootstrap_path, ddp)
            start_step = 0
            if master:
                logger.success(
                    f"Loaded weights from {bootstrap_label}; "
                    "starting round-2.2 step counter at 0"
                )
        else:
            if master:
                logger.warning(
                    "No prior ckpt and no bootstrap source available; "
                    "starting from random init"
                )

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    # num_workers=1 (was 4): each worker holds a copy of the 200k-vocab tokenizer
    # and buffered text; at 3B FSDP + 8 loops we hit OOM with 4 workers/rank.
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=1, pin_memory=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        # Variable-T: sample once per optimizer step so all micro-steps use
        # the same depth, and seed the RNG on `step` so every rank in the
        # FSDP group picks the same T without a NCCL broadcast. Different
        # T across ranks would desync the recurrent loop's collectives.
        # T_FIXED env var pins T to a constant for fixed-depth ablations.
        T_step = T_FIXED if T_FIXED else random.Random(step).randint(T_MIN, T_MAX)

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x, n_loops=T_step)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
        # against each rank's local norm and miss the cross-shard gather.
        # FSDP.clip_grad_norm_ computes the true global norm and returns it.
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| T {T_step:2d} | {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen"
            )
            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

        # Federated sync hook (no-op unless FED_SYNC_DIR is set)
        _fed_sync_hook.add_tokens(global_batch_tok)
        _fed_sync_hook.update_loss_ema(loss_accum)
        _fed_sync_hook.maybe_sync(model, step)

    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    # Drain any in-flight async ckpt write before the process group is
    # destroyed; otherwise the executor thread can race with interpreter
    # shutdown and a half-written .tmp file may be left on disk.
    wait_for_pending_ckpts()

    if ddp:
        # Barrier so no rank exits while another is still finishing its
        # checkpoint gather — avoids NCCL "process group destroyed" noise.
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("Training complete.")


if __name__ == "__main__":
    main()
