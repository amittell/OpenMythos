#!/usr/bin/env python3
"""
mythos_3b variable-T training variant (round 2).

Identical infrastructure to 3b_loops4_fast.py but samples the recurrent loop
depth T uniformly from [T_MIN, T_MAX] on every optimizer step. All micro-steps
within one optimizer step share the same T so gradients see a consistent
recurrence count. The LoRA per-loop-index embedding is sized to T_MAX so every
slot receives gradient signal across the run.

Motivation: round 1 trained at a fixed T=4, and the post-hoc depth-extrapolation
probe showed the model does not gain from K>4 iterations at inference (ACT
halts early; forcing deeper iterations degrades loss monotonically and
saturates). The Saunshi/Geiping results on "trained-shallow, infer-deep"
require variable-T training; this script is that.

Inherits from round 1:
  - FSDP SHARD_GRAD_OP + bf16 mixed precision
  - grad_accum=4, micro_batch=1 (known-stable on 128 GB GB10)
  - Checkpoint auto-distribute with worker prune (keep_last=3)

Separate checkpoint dir (checkpoints_3b_varT_fast) so we do not collide with
the round-1 run. Same cluster-launch recipe.
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
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
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


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
    os.makedirs(ckpt_dir, exist_ok=True)

    if not ddp:
        if master:
            final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            tmp_path = final_path + ".tmp"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "vocab_size": vocab_size,
                },
                tmp_path,
            )
            os.replace(tmp_path, final_path)
            for old in _list_ckpts(ckpt_dir, rank=0)[:-keep_last]:
                try:
                    os.remove(old)
                except OSError as exc:
                    logger.warning(f"Failed to prune old checkpoint {old}: {exc}")
            logger.success(f"Checkpoint saved -> {final_path}")
        return

    rank = dist.get_rank()
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
        optim_state_dict_config=ShardedOptimStateDictConfig(),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    final_path = os.path.join(ckpt_dir, f"step_{step:07d}_rank{rank}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
            "rank": rank,
            "world_size": dist.get_world_size(),
            "sharded": True,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    # Barrier so no rank's prune starts while another rank is still writing
    # its own file of the current step. Prune is per-rank (each rank only
    # sees its own local disk) but all ranks saved the same set of steps,
    # so they agree on which to drop.
    dist.barrier()
    for old in _list_ckpts(ckpt_dir, rank=rank)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    if master:
        logger.success(
            f"Sharded checkpoint saved -> step {step} across "
            f"{dist.get_world_size()} ranks (each rank wrote its own shard "
            f"to local disk)"
        )
    # No rsync: the sharded path keeps each rank's shard on its own local disk.
    # Resume reads per-rank files locally, so there's nothing to distribute.


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
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 1024
    # micro_batch=1, grad_accum=4 — reverted from micro_batch=2/grad_accum=2
    # which OOM-killed rank 0 at restart (2x activation mem + FSDP buffers +
    # MoE expert unshards pushed past 128 GB unified). The micro_batch=1
    # config is known stable from 1150 prior steps.
    micro_batch = 1
    # 100M-token target (was 30B) → total_steps=6103, realistic session-scale run.
    # Resume from existing checkpoint_3b_loops4_fast/step_00*.pt is automatic.
    # Keeping warmup_steps=2000 unchanged so the LR trajectory is continuous
    # through the restart — no LR shock on resume.
    target_tokens = 200_000_000  # round 2 initial budget; extend mid-run if warranted
    grad_accum = 4
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    wd = 0.1
    log_every = 5
    # ckpt_every=100 (was 50): each save takes ~5 min on rank 0 and briefly
    # swaps that node — cutting save frequency in half saves ~2.5 min per
    # 100 steps (~10% of wallclock) and halves the rank-0 swap thrashing.
    # Cost: up to 100 steps (~20 min) of re-work on crash. Fine at this scale.
    ckpt_every = 100
    ckpt_dir = "checkpoints_3b_varT_fast"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
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
    cfg.max_loop_iters = T_MAX

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

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
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir, rank=rank if ddp else 0)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

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
        T_step = random.Random(step).randint(T_MIN, T_MAX)

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

    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if ddp:
        # Barrier so no rank exits while another is still finishing its
        # checkpoint gather — avoids NCCL "process group destroyed" noise.
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("Training complete.")


if __name__ == "__main__":
    main()
