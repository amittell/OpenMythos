#!/usr/bin/env python3
"""
Single-host variant of consolidate_ckpt.py.

The cluster-trained FSDP SHARDED_STATE_DICT layout saves one .pt per
rank. ``consolidate_ckpt.py`` consolidates by running one process per
cluster node (4-rank torchrun spanning 4 hosts) and FSDP-gathering to
rank 0. That requires the cluster, which is unavailable while training
is in flight on it.

This variant collapses the 4 ranks onto a single multi-GPU host. The
operator pre-rsyncs all 4 rank shards into one directory; this script
spawns 4 local processes that each load one shard, then FSDP-gathers
to rank 0 and writes the consolidated full state dict.

LOCAL_RANK is mapped to a GPU via modulo: on a 2-GPU host the 4 ranks
share two cards (ranks 0+2 on cuda:0, ranks 1+3 on cuda:1). Each shard
is ~11 GB so 2 shards per 96 GB GPU is well within budget.

==============================================================
KNOWN HARDWARE CEILING: needs ~256 GB system RAM, not just VRAM
==============================================================

Empirically tested on kebab-rtx6000 (2x RTX PRO 6000 Blackwell @ 96 GB,
124 GB system RAM) on 2026-05-17. Three configurations failed:

1. NCCL backend, 4 procs on GPU 1 only -> CUDA OOM during FSDP wrap.
   GPU 1 capped at 96 GB; 4 procs * ~25 GB peak per rank exceeds it.

2. NCCL backend, 4 procs across both GPUs (modulo mapping) -> NCCL
   aborts with "Duplicate GPU detected: rank 0 and rank 2 both on
   CUDA device". NCCL strictly enforces 1-proc-per-GPU.

3. Gloo backend, 4 procs across both GPUs, vllm-120b + embedding
   services stopped to clear GPU 0 -> got past load_state_dict (GPU
   peak 91/90 GB, fit) but the FULL_STATE_DICT gather to rank 0 with
   offload_to_cpu=True triggered a global system OOM-kill.
   Rank 0's total_vm hit 295 GB; physical RAM is only 124 GB; OOM
   killer took out the consolidator (and dbus along with it).

The 124 GB system RAM is the binding ceiling, not the 192 GB combined
VRAM. FSDP's FULL_STATE_DICT gather over gloo materialises a working
set roughly equal to ``world_size * full_model_bytes`` in CPU memory
during the all-gather, before the rank0_only=True offload kicks in.
For a 3B-param bf16 model that's ~4 * 6 GB = 24 GB of resident copies
plus several multiples in virtual address space for the staging
buffers, so a host with 256+ GB RAM (or a smaller model) is the
realistic target.

In production we fell back to wait-for-completion: r2.15 will
self-consolidate on the cluster when training finishes (its training
script's final consolidate step runs as part of the 4-rank cluster
job, which has 4*128 GB = 512 GB unified gb10 memory across the four
nodes -- plenty of headroom for offload_to_cpu).

If you DO have a 256+ GB host, the script below should work as-is.
The driver script training/intermediate_eval_r215.sh is the
end-to-end pipeline.

Usage (run on a single multi-GPU host with 256+ GB RAM):

    torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 \\
             --master_port=29555 \\
             training/consolidate_ckpt_single_host.py \\
             checkpoints_3b_varT_pondernet_round215/step_0001400 \\
             checkpoints_3b_varT_pondernet_round215/step_0001400_full.pt

Args:
    src_pattern  Path prefix without _rankR.pt suffix; ALL rank shards
                 (0..3) must exist in this dir on the local machine
    dst_path     Where rank 0 writes the consolidated full state dict
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
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
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.main import TransformerBlock, RecurrentBlock  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: consolidate_ckpt_single_host.py SRC_PATTERN DST_PATH",
            file=sys.stderr,
        )
        sys.exit(2)

    src_pattern = sys.argv[1]
    dst_path = sys.argv[2]

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    # Map local_rank to a real GPU via modulo so 4 procs can colocate
    # on a 2-GPU host (or N procs on M GPUs where M < N).
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("no CUDA devices visible; this consolidator needs at least one GPU")
    gpu_idx = local_rank % n_gpus
    torch.cuda.set_device(gpu_idx)
    device = f"cuda:{gpu_idx}"

    # gloo, not NCCL: NCCL strictly enforces 1-proc-per-GPU and aborts
    # with "Duplicate GPU detected" when we colocate 4 ranks on 2 cards
    # (the only way to match the saved world_size=4 on a 2-GPU host).
    # gloo performs collectives over CPU memory, transparently moving
    # GPU tensors to/from CPU as needed -- slower than NCCL but works
    # with shared-GPU procs. Consolidation is one-shot anyway, not a
    # training inner loop.
    dist.init_process_group("gloo")

    shard_path = f"{src_pattern}_rank{rank}.pt"
    if rank == 0:
        logger.info(
            f"world_size={world} n_gpus={n_gpus} "
            f"(local_rank {local_rank} -> {device})"
        )
        logger.info(f"loading shard 0 from {shard_path}")
    # mmap=True is load-bearing on a single-host 4-rank consolidate.
    # Without it, each of the 4 ranks does a full ``torch.load(...,
    # map_location='cpu')`` into anonymous CPU memory in parallel: for
    # the r2.18 case (~11 GB per shard) peak virtual memory on the host
    # hit ~125 GB with 124 GB of system RAM and the kernel OOM-killer
    # took out the consolidator (and dbus along with it -- 2026-05-31).
    # mmap=True maps the .pt file's tensor storage on demand instead,
    # letting the kernel page in only what FSDP actually touches during
    # ``load_state_dict``. Peak resident dropped to ~25 GB in testing.
    ckpt = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    saved_cfg = ckpt["cfg"]

    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", saved_cfg.vocab_size))
    cfg.max_seq_len = saved_cfg.max_seq_len
    cfg.max_loop_iters = saved_cfg.max_loop_iters

    if rank == 0:
        logger.info(
            f"cfg: vocab={cfg.vocab_size} seq={cfg.max_seq_len} "
            f"max_loop_iters={cfg.max_loop_iters}"
        )

    model = OpenMythos(cfg)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=gpu_idx,
    )

    if rank == 0:
        logger.info("loading sharded state dict on all ranks...")
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
        optim_state_dict_config=ShardedOptimStateDictConfig(),
    ):
        model.load_state_dict(ckpt["model"])

    if rank == 0:
        logger.info("gathering full state dict to rank 0...")
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = model.state_dict()

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        tmp_path = dst_path + ".tmp"
        torch.save(
            {
                "step": ckpt["step"],
                "model": full_state,
                "cfg": ckpt["cfg"],
                "vocab_size": ckpt["vocab_size"],
            },
            tmp_path,
        )
        os.replace(tmp_path, dst_path)
        logger.success(f"wrote consolidated full-state-dict to {dst_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
