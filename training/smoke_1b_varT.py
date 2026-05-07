#!/usr/bin/env python3
"""
Single-GPU mythos_1b smoke run with variable-T schedule.

Mirrors smoke_1b.py but samples T ~ Uniform(T_MIN, T_MAX) per optimizer step
and passes it to model(x, n_loops=T). cfg.max_loop_iters is set to T_MAX so
the LoRA per-loop embedding table has slots for every sampled depth.

Used to confirm variable-T doesn't destabilize loss before committing to a
multi-day 4-node round-2 run.
"""

import os
import math
import random
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
    max_steps = int(os.environ.get("MAX_STEPS", 200))
    warmup_steps = 20
    lr = 3e-4
    wd = 0.1
    log_every = 5
    dataset_subset = "sample-10BT"

    T_MIN = 2
    T_MAX = 12

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    logger.info(f"Tokenizer vocab: {vocab_size:,}")

    cfg = mythos_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = T_MAX

    logger.info(
        f"seq_len={seq_len} micro_batch={micro_batch} grad_accum={grad_accum} "
        f"max_steps={max_steps} T in [{T_MIN},{T_MAX}] "
        f"tokens/step={micro_batch*grad_accum*seq_len:,}"
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

    # Track how many times each T was sampled and the per-T running loss sum,
    # so the smoke report can show variable-T is exercising the full range.
    t_hist = {T: [0, 0.0] for T in range(T_MIN, T_MAX + 1)}

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

        T_step = random.Random(step).randint(T_MIN, T_MAX)

        for _ in range(grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_ctx:
                logits = model(x, n_loops=T_step)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            loss.backward()
            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        t_hist[T_step][0] += 1
        t_hist[T_step][1] += loss_accum

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = micro_batch * grad_accum * seq_len * log_every / dt
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"step {step:4d}/{max_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| T {T_step:2d} | {tok_per_sec/1e3:.1f}K tok/s | mem {mem_gb:.1f}GB"
            )
            t0 = time.perf_counter()

    logger.success(f"Smoke run complete: {max_steps} steps")

    logger.info("Per-T sample count and mean loss over run:")
    logger.info("  T | count | mean_loss")
    for T in range(T_MIN, T_MAX + 1):
        count, loss_sum = t_hist[T]
        mean = loss_sum / count if count else float("nan")
        logger.info(f"  {T:2d} | {count:5d} | {mean:.4f}")

    # LoRA slot gradient sanity: confirm every slot 0..T_MAX-1 has non-zero
    # grad accumulated somewhere in the last step. (Cheap proxy: check that
    # the embedding weight rows have non-trivial magnitude after training.)
    lora_scale = model.recurrent.lora.scale.weight.detach()
    logger.info("LoRA per-loop scale row norms after training:")
    for t in range(T_MAX):
        logger.info(f"  slot {t:2d} norm={lora_scale[t].norm().item():.4f}")


if __name__ == "__main__":
    main()
