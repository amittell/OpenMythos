#!/usr/bin/env python3
"""
Multi-node FSDP mythos_1b shakeout on FineWeb-Edu sample-10BT.

Mirrors 3b_fine_web_edu.py for distributed training validation, but:
  - mythos_1b variant (smaller for quicker iteration)
  - MAX_STEPS env override (default 500)
  - No checkpointing (shakeout, not a real run)

Launch example (3-node):
  # on head (rank 0)
  torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 \
    --master_addr=192.168.100.10 --master_port=29500 training/shakeout_1b.py
  # on worker N
  torchrun --nnodes=3 --nproc_per_node=1 --node_rank=N \
    --master_addr=192.168.100.10 --master_port=29500 training/shakeout_1b.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer


class FineWebEduDataset(IterableDataset):
    def __init__(self, encoding, seq_len, subset, rank, world_size):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
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


def main():
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
        device = "cuda"

    master = rank == 0

    seq_len = 1024
    micro_batch = 2
    max_steps = int(os.environ.get("MAX_STEPS", 500))
    grad_accum = max(1, 64 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    warmup_steps = 50
    lr = 3e-4
    wd = 0.1
    log_every = 10
    dataset_subset = "sample-10BT"

    if master:
        logger.info(
            f"world_size={world_size} seq_len={seq_len} micro_batch={micro_batch} "
            f"grad_accum={grad_accum} global_batch_tokens={global_batch_tok:,} "
            f"max_steps={max_steps}"
        )

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    if master:
        logger.info(f"Tokenizer vocab: {vocab_size:,}")

    cfg = mythos_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    amp_dtype = torch.bfloat16

    model = OpenMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
            cast_forward_inputs=True,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
        amp_ctx = nullcontext()
    else:
        model = model.to(device)
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Params (per rank after shard): {n_params:,}  |  dtype: {amp_dtype}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=2, pin_memory=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = 0

    while step < max_steps:
        if step < warmup_steps:
            cur_lr = lr * step / max(1, warmup_steps)
        else:
            decay = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cur_lr = lr * 0.1 + 0.5 * (lr - lr * 0.1) * (1.0 + math.cos(math.pi * decay))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            loss.backward()
            loss_accum += loss.item()

        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"step {step:4d}/{max_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec/1e3:.0f}K tok/s | rank0 mem {mem_gb:.1f}GB"
            )
            t0 = time.perf_counter()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success(f"Shakeout complete: {max_steps} steps on {world_size} nodes.")


if __name__ == "__main__":
    main()
