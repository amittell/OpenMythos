#!/usr/bin/env python3
"""
Consolidate a sharded round-2 checkpoint into a full state-dict file.

Round 2 saves under FSDP.SHARDED_STATE_DICT — one .pt per rank on each node's
local disk. Downstream eval (depth_extrap.py) consumes a single full state
dict, so we re-load the sharded ckpt under FSDP, gather to FULL_STATE_DICT
on rank 0, and torch.save a unified file.

Usage (run via 4-rank torchrun, mirroring the training launch):

    torchrun --nnodes=4 --nproc_per_node=1 --node_rank=$R \\
             --master_addr=192.168.100.10 --master_port=29555 \\
             training/consolidate_ckpt.py \\
             checkpoints_3b_varT_fast/step_0012207 \\
             checkpoints_3b_varT_fast/step_0012207_full.pt

Args:
    src_pattern  Path prefix without _rankR.pt suffix
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
    # Accept either positional argv (training/cluster_consolidate.sh path) or
    # env vars (gpufarm's executor declares env_required=[SRC_PATTERN, DST_PATH]
    # and renders them into the process environment, not argv). Either works;
    # argv wins when both are present so the manual invocation pattern in the
    # docstring continues to override.
    if len(sys.argv) == 3:
        src_pattern = sys.argv[1]
        dst_path = sys.argv[2]
    elif len(sys.argv) == 1 and os.environ.get("SRC_PATTERN") and os.environ.get("DST_PATH"):
        src_pattern = os.environ["SRC_PATTERN"]
        dst_path = os.environ["DST_PATH"]
    else:
        print(
            "usage: consolidate_ckpt.py SRC_PATTERN DST_PATH "
            "(or set SRC_PATTERN= and DST_PATH= in the environment)",
            file=sys.stderr,
        )
        sys.exit(2)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ["WORLD_SIZE"])

    # Per-collective NCCL watchdog timeout. Default 600s is too tight
    # for a 4-node bf16 ALLGATHER of ~1.15 GB through cold-start RoCE
    # handshake; bump to 30 min via env (CLUSTER_NCCL_TIMEOUT_SEC) when
    # called from training/cluster_consolidate.sh.
    from datetime import timedelta
    _nccl_timeout_sec = int(os.environ.get("CLUSTER_NCCL_TIMEOUT_SEC", "1800"))
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=_nccl_timeout_sec)
    )
    if rank == 0:
        logger.info(f"NCCL collective timeout = {_nccl_timeout_sec}s")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    shard_path = f"{src_pattern}_rank{rank}.pt"
    if rank == 0:
        logger.info(f"world_size={world}  loading shard 0 from {shard_path}")
    ckpt = torch.load(shard_path, map_location="cpu", weights_only=False)
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
        device_id=local_rank,
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
