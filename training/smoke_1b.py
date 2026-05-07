#!/usr/bin/env python3
"""
Single-GPU mythos_1b smoke run on FineWeb-Edu sample-10BT.

Mirrors 3b_fine_web_edu.py but:
  - mythos_1b variant (fits comfortably on one GB10)
  - shorter seq_len, smaller micro_batch for faster iteration
  - hard stop at MAX_STEPS (default 100)
  - no checkpointing (cost not worth it for 100-step run)
  - no FSDP wrap (single GPU only)

Used to validate the training stack end-to-end before multi-node launch.
"""

import os
import math
import time
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer


class FineWebEduDataset(torch.utils.data.IterableDataset):
    def __init__(self, encoding, seq_len, subset):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset

    def __iter__(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        )
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
    device = "cuda"
    torch.cuda.set_device(0)

    seq_len = 1024
    micro_batch = 2
    grad_accum = 4
    max_steps = int(os.environ.get("MAX_STEPS", 100))
    warmup_steps = 20
    lr = 3e-4
    wd = 0.1
    log_every = 5
    dataset_subset = "sample-10BT"

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    logger.info(f"Tokenizer vocab: {vocab_size:,}")

    cfg = mythos_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    logger.info(
        f"seq_len={seq_len} micro_batch={micro_batch} grad_accum={grad_accum} "
        f"max_steps={max_steps} tokens/step={micro_batch*grad_accum*seq_len:,}"
    )

    model = OpenMythos(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built mythos_1b: {n_params:,} params  |  GPU: {torch.cuda.get_device_name(0)}")

    amp_dtype = torch.bfloat16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=2, pin_memory=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = 0

    while step < max_steps:
        if step < warmup_steps:
            cur_lr = lr * step / warmup_steps
        else:
            decay = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cur_lr = lr * 0.1 + 0.5 * (lr - lr * 0.1) * (1.0 + math.cos(math.pi * decay))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            loss.backward()
            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = micro_batch * grad_accum * seq_len * log_every / dt
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"step {step:4d}/{max_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec/1e3:.1f}K tok/s | mem {mem_gb:.1f}GB"
            )
            t0 = time.perf_counter()

    logger.success(f"Smoke run complete: {max_steps} steps")


if __name__ == "__main__":
    main()
