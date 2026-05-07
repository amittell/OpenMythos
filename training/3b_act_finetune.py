#!/usr/bin/env python3
"""
3b_act_finetune.py: ACT-head fine-tune phase on top of the round-2.1
bypass-trained backbone.

Round 2.1 (3b_varT_act_v2.py) trained the recurrent backbone with the ACT
head bypassed; the head therefore received no gradient and stayed at the
round-2 saturated state (p_t = 1.0 at every iteration). §8.3 of the paper
proposes a follow-up experiment: re-initialize the head, freeze everything
else, and train the head with the standard ACT-weighted output objective
to see if it can learn a useful halting distribution against a backbone
that genuinely uses depth.

Possible outcomes:
  (a) head escapes saturation, settles into a non-trivial halting profile
      that exploits the backbone's depth-using behavior. Paper §7.6 gets
      a positive result.
  (b) head re-collapses to p = 1.0 at iteration 1. Confirms §8.1's
      fixed-point analysis: saturation is intrinsic to the ACT-weighted
      output objective, not a property of an under-trained backbone.
  (c) head collapses to a different trivial mode (e.g., halt at the last
      iteration only). Also informative.

This script is single-GPU on spark; no FSDP. The 3.1B model in bf16 fits
in the 128 GB unified memory of one GB10 with grad_accum=4 micro_batch=1
seq_len=1024 T_max=12.

Inputs (env-overridable):
    CKPT          path to bootstrap full-state-dict (round 2.1 final)
    OUT_DIR       directory for checkpoints (default: checkpoints_3b_act_finetune)
    TARGET_TOKENS default 5_000_000
    LR            default 1e-4 on the head params; backbone is frozen
    REINIT_HEAD   "1" to re-init ACTHalting.halt (default), "0" to keep
                  the saturated head and try to nudge it off
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


class FineWebEduDataset(IterableDataset):
    """Streaming FineWeb-Edu loader; same shape as the round 2.1 trainer's."""

    def __init__(self, encoding, seq_len: int, subset: str):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset

    def __iter__(self):
        worker = get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=nw, index=wid)
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


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def main() -> None:
    ckpt_in = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_varT_act_v2/step_0003051_full.pt",
    )
    out_dir = os.environ.get(
        "OUT_DIR", "/home/alexm/OpenMythos/checkpoints_3b_act_finetune"
    )
    target_tokens = int(os.environ.get("TARGET_TOKENS", "5000000"))
    lr_max = float(os.environ.get("LR", "1e-4"))
    reinit = os.environ.get("REINIT_HEAD", "1") == "1"
    # Anti-collapse penalty: L_total = CE + lambda * mean(p_1^2). When > 0,
    # the head pays for emitting high halting probability at iteration 1,
    # which directly opposes the trivial-halt fixed point.
    lambda_anti = float(os.environ.get("LAMBDA_ANTI_COLLAPSE", "0.0"))
    # PonderNet-style geometric prior. When USE_PONDERNET=1, the regulariser
    # is the per-iteration Bernoulli KL: for every iteration t, push p_t
    # toward LAMBDA_P (the geometric halting prior). Mean halt step under
    # this regime is ~ 1/LAMBDA_P. With LAMBDA_P=0.2, expected halt is
    # iteration 5 — a useful intermediate that the simple anti-collapse
    # penalty pushes past (it drives p_t toward 0).
    use_pondernet = os.environ.get("USE_PONDERNET", "0") == "1"
    lambda_kl = float(os.environ.get("LAMBDA_KL", "1.0"))
    lambda_p = float(os.environ.get("LAMBDA_P", "0.2"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    micro_batch = int(os.environ.get("MICRO_BATCH", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    log_every = int(os.environ.get("LOG_EVERY", "5"))
    ckpt_every = int(os.environ.get("CKPT_EVERY", "200"))
    subset = os.environ.get("DATASET_SUBSET", "sample-10BT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    logger.info(f"device={device} amp_dtype={amp_dtype}")
    logger.info(f"bootstrap from {ckpt_in}")
    logger.info(
        f"target_tokens={target_tokens:,} lr_max={lr_max} reinit_head={reinit} "
        f"seq_len={seq_len} micro_batch={micro_batch} grad_accum={grad_accum}"
    )

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)
    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", vocab_size))
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max
    T_MIN = int(os.environ.get("T_MIN", "2"))
    T_MAX = int(os.environ.get("T_MAX", str(saved_T_max)))
    if T_MIN > T_MAX:
        raise ValueError(f"T_MIN={T_MIN} > T_MAX={T_MAX}")
    if T_MAX > saved_T_max:
        logger.warning(
            f"T_MAX={T_MAX} > saved max_loop_iters={saved_T_max}; "
            "LoRA index will clamp at training_max - 1"
        )
    if T_MIN == T_MAX:
        logger.info(f"max_loop_iters={saved_T_max}; FIXED T={T_MIN} per step")
    else:
        logger.info(
            f"max_loop_iters={saved_T_max}; sampling T from U[{T_MIN},{T_MAX}]"
        )

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt

    # Re-initialize the ACT head if requested. The head is a single Linear:
    # ACTHalting.halt: dim -> 1. The round-2.1 weights drive sigmoid output
    # to ~1.0 for every input; default Kaiming-uniform init produces small
    # weights that put p_t near 0.5, breaking the saturated fixed point.
    head = model.recurrent.act.halt
    head_w_norm_before = head.weight.detach().norm().item()
    head_b_before = head.bias.detach().item()
    if reinit:
        head.reset_parameters()
        logger.info(
            f"ACT head re-initialized; weight norm "
            f"{head_w_norm_before:.4f} -> {head.weight.detach().norm().item():.4f}, "
            f"bias {head_b_before:+.4f} -> {head.bias.detach().item():+.4f}"
        )
    else:
        logger.info(
            f"ACT head left as-loaded (weight norm {head_w_norm_before:.4f}, "
            f"bias {head_b_before:+.4f})"
        )

    # Freeze every parameter that is NOT in the ACT head.
    head_params = list(model.recurrent.act.parameters())
    head_param_ids = {id(p) for p in head_params}
    n_train = 0
    n_freeze = 0
    for p in model.parameters():
        if id(p) in head_param_ids:
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
            n_freeze += p.numel()
    logger.info(
        f"trainable params: {n_train:,} (ACT head only); "
        f"frozen: {n_freeze:,} ({n_freeze / (n_train + n_freeze) * 100:.2f} %)"
    )

    model = model.to(device)
    model.train(False)            # backbone in eval mode (BN/dropout off)
    model.recurrent.act.train(True)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success("model on device")

    # Optimizer over the trainable subset only.
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_max,
        weight_decay=0.0,        # tiny head, no weight decay
        betas=(0.9, 0.95),
    )

    global_batch_tok = micro_batch * grad_accum * seq_len
    total_steps = max(1, target_tokens // global_batch_tok)
    warmup_steps = max(20, min(100, total_steps // 10))
    logger.info(
        f"global_batch_tok={global_batch_tok:,} total_steps={total_steps:,} "
        f"warmup_steps={warmup_steps}"
    )

    dataset = FineWebEduDataset(encoding, seq_len, subset)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=1, pin_memory=True)
    data_iter = iter(loader)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if "cuda" in device
        else nullcontext()
    )

    step = 0
    t0 = time.perf_counter()
    while step < total_steps:
        cur_lr = cosine_lr(step, warmup_steps, total_steps, lr_max, lr_max * 0.1)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        T_step = random.Random(step + 31337).randint(T_MIN, T_MAX)
        loss_acc = 0.0
        ce_acc = 0.0
        reg_acc = 0.0
        p_ones = []

        # Hook on ACTHalting to capture p emitted per call (one call per
        # iteration). We retain the live tensor (no detach) so the
        # regulariser's gradient flows back to the head.
        captured: list[torch.Tensor] = []
        handle = model.recurrent.act.register_forward_hook(
            lambda m, i, o: captured.append(o)
        )

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            captured.clear()
            with amp_ctx:
                logits = model(x, n_loops=T_step)
                ce = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                if use_pondernet and captured:
                    # PonderNet-style: per-iteration Bernoulli KL between
                    # the head's emitted p_t and the geometric prior
                    # Bernoulli(lambda_p). Summed over iterations, mean
                    # over (B, T_seq). Drives p_t toward lambda_p at every
                    # step, producing a halt distribution close to
                    # Geom(lambda_p) with expected halt step 1/lambda_p.
                    p_stack = torch.stack(captured, dim=0).float().clamp(1e-6, 1.0 - 1e-6)
                    log_lp = math.log(lambda_p)
                    log_1mlp = math.log(1.0 - lambda_p)
                    kl_per_iter = (
                        p_stack * (torch.log(p_stack) - log_lp)
                        + (1.0 - p_stack) * (torch.log(1.0 - p_stack) - log_1mlp)
                    )
                    reg = kl_per_iter.sum(dim=0).mean()
                    weight = lambda_kl
                elif lambda_anti > 0 and captured:
                    # captured[0] is iteration-1 p; shape (B, T_seq).
                    # Penalize squared probability so saturation costs
                    # progressively more than near-0.5.
                    reg = (captured[0].float() ** 2).mean()
                    weight = lambda_anti
                else:
                    reg = torch.zeros((), device=ce.device, dtype=ce.dtype)
                    weight = 0.0
                loss = (ce + weight * reg) / grad_accum
            loss.backward()
            loss_acc += float(loss.item())
            ce_acc += float(ce.item()) / grad_accum
            reg_acc += float(reg.item()) / grad_accum
            if captured:
                p_ones.append(float(captured[0].detach().mean().item()))
        handle.remove()

        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            mean_p1 = sum(p_ones) / max(1, len(p_ones))
            if use_pondernet:
                extra = f" | ce {ce_acc:.4f} kl {reg_acc:.4f}"
            elif lambda_anti > 0:
                extra = f" | ce {ce_acc:.4f} anti {reg_acc:.4f}"
            else:
                extra = ""
            logger.info(
                f"step {step:5d}/{total_steps} | loss {loss_acc:.4f} "
                f"| lr {cur_lr:.2e} | T {T_step:2d} | mean(p_1)={mean_p1:.4f}"
                f"{extra} | {tok_per_sec / 1e3:.1f}k tok/s "
                f"| {tokens_seen / 1e6:.1f}M seen"
            )
            t0 = time.perf_counter()

        if step % ckpt_every == 0 or step == total_steps:
            head_state = {
                "step": step,
                "act_head_state_dict": model.recurrent.act.state_dict(),
                "cfg": cfg,
                "vocab_size": vocab_size,
                "bootstrap_ckpt": ckpt_in,
            }
            head_path = Path(out_dir) / f"head_step_{step:06d}.pt"
            torch.save(head_state, head_path)
            logger.success(f"saved head ckpt: {head_path}")

    # Save final full-state-dict (model with the new head merged in) for
    # downstream eval; depth_extrap.py expects a full ckpt.
    final_full_path = Path(out_dir) / f"step_{step:07d}_full.pt"
    full_state = {
        "step": step,
        "model": model.state_dict(),
        "cfg": cfg,
        "vocab_size": vocab_size,
        "bootstrap_ckpt": ckpt_in,
    }
    torch.save(full_state, final_full_path)
    logger.success(f"saved final full-state-dict: {final_full_path}")
    logger.success("ACT fine-tune complete.")


if __name__ == "__main__":
    main()
