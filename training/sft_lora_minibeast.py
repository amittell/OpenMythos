#!/usr/bin/env python3
"""
sft_lora_minibeast.py

LoRA-based supervised fine-tuning of the recurrent-depth 3B model on a chat
dataset (default: HuggingFaceH4/ultrachat_200k). Designed for single-GPU
training on the RTX 5090 (32 GB VRAM).

Memory budget:
  - frozen 3B bf16 weights: ~7 GB
  - LoRA params (rank 32, ~12M trainable): ~25 MB
  - LoRA grads + Adam state: ~150 MB
  - activations at seq=2048 micro_batch=1: ~5 GB
  - total: ~12 GB, fits comfortably in 32 GB

Inputs (env-overridable):
    CKPT          full state-dict checkpoint (required, base model)
    OUT_DIR       output directory for adapter ckpts (required)
    DATASET       HF dataset id (default: HuggingFaceH4/ultrachat_200k)
    SPLIT         dataset split (default: train_sft)
    MAX_SAMPLES   cap on number of examples (default: 50000)
    SEQ_LEN       max sequence length (default: 2048)
    LORA_RANK     LoRA rank (default: 32)
    LORA_ALPHA    LoRA alpha (default: 64)
    LORA_DROPOUT  LoRA dropout (default: 0.05)
    LR            learning rate for LoRA params (default: 2e-4)
    EPOCHS        epochs over capped dataset (default: 1)
    MICRO_BATCH   micro batch size (default: 1)
    GRAD_ACCUM    gradient accumulation (default: 8)
    WARMUP_STEPS  LR warmup (default: 100)
    SAVE_EVERY    save adapter every N steps (default: 500)
    LOG_EVERY     log loss every N steps (default: 10)
    SAVE_MERGED   if "1", also save merged full ckpt at end (default: 0)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import IterableDataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank update.

    Output: y = base(x) + (x @ A^T @ B^T) * (alpha / rank)
    The base linear's weights are frozen; only A and B are trainable.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        in_dim = base.in_features
        out_dim = base.out_features
        self.A = nn.Parameter(torch.empty(rank, in_dim, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(out_dim, rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.A.T @ self.B.T
        return out + delta * self.scale


def inject_lora(model: nn.Module, target_substrings: list[str], rank: int,
                alpha: float, dropout: float) -> int:
    n_replaced = 0
    for name, child in list(model.named_modules()):
        for child_name, sub in list(child.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if not isinstance(sub, nn.Linear):
                continue
            if not any(s in full_name for s in target_substrings):
                continue
            wrapped = LoRALinear(sub, rank, alpha, dropout)
            setattr(child, child_name, wrapped)
            n_replaced += 1
    return n_replaced


def lora_state_dict(model: nn.Module) -> dict:
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad and (".A" in name or ".B" in name):
            out[name] = param.detach().cpu()
    return out


def merge_lora(model: nn.Module) -> None:
    """In-place: fold LoRA delta into base weights, replace LoRALinear with Linear."""
    for name, child in list(model.named_modules()):
        for child_name, sub in list(child.named_children()):
            if not isinstance(sub, LoRALinear):
                continue
            with torch.no_grad():
                delta = sub.scale * (sub.B @ sub.A)
                sub.base.weight.data.add_(delta.to(sub.base.weight.dtype))
            setattr(child, child_name, sub.base)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChatSFTDataset(IterableDataset):
    """Streams (input_ids, labels). Labels are -100 on prompt tokens, equal
    to input_ids on assistant-response tokens (causal-LM SFT loss masking).
    """

    def __init__(self, hf_dataset, tokenizer, seq_len: int, max_samples: int):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

    def __iter__(self):
        n = 0
        for ex in self.ds:
            if n >= self.max_samples:
                return
            msgs = ex.get("messages") or []
            if not msgs:
                continue
            text_parts = []
            assistant_spans = []
            cursor = 0
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "") or ""
                if role == "user":
                    chunk = f"<|user|>\n{content}\n"
                elif role == "assistant":
                    chunk = f"<|assistant|>\n{content}\n"
                else:
                    chunk = content + "\n"
                start = cursor
                text_parts.append(chunk)
                cursor += len(chunk)
                if role == "assistant":
                    assistant_spans.append((start, cursor))
            text = "".join(text_parts)
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) < 16:
                continue
            ids = ids[: self.seq_len]
            labels = [-100] * len(ids)
            char_to_token = []
            for token_idx in range(len(ids)):
                cum_prefix = self.tok.decode(ids[: token_idx + 1])
                char_to_token.append(len(cum_prefix))
            for start_c, end_c in assistant_spans:
                for ti, end_pos in enumerate(char_to_token):
                    if start_c < end_pos <= end_c:
                        labels[ti] = ids[ti]
            x = torch.tensor(ids, dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.long)
            yield x, y
            n += 1


def collate_fn(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    out_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    out_y = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        out_x[i, : x.size(0)] = x
        out_y[i, : y.size(0)] = y
    return out_x, out_y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ckpt_path = os.environ.get("CKPT")
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"CKPT must point to existing checkpoint (got: {ckpt_path})")
        sys.exit(1)
    out_dir = Path(os.environ.get("OUT_DIR", "checkpoints_3b_sft_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = os.environ.get("DATASET", "HuggingFaceH4/ultrachat_200k")
    split = os.environ.get("SPLIT", "train_sft")
    max_samples = int(os.environ.get("MAX_SAMPLES", "50000"))
    seq_len = int(os.environ.get("SEQ_LEN", "2048"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    lora_alpha = float(os.environ.get("LORA_ALPHA", "64"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    lr = float(os.environ.get("LR", "2e-4"))
    epochs = int(os.environ.get("EPOCHS", "1"))
    micro_batch = int(os.environ.get("MICRO_BATCH", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "100"))
    save_every = int(os.environ.get("SAVE_EVERY", "500"))
    log_every = int(os.environ.get("LOG_EVERY", "10"))
    save_merged = os.environ.get("SAVE_MERGED", "0") == "1"
    device = "cuda:0"

    logger.info("=== SFT LoRA config ===")
    logger.info(f"  ckpt:      {ckpt_path}")
    logger.info(f"  out_dir:   {out_dir}")
    logger.info(f"  dataset:   {dataset_id} [{split}] cap={max_samples}")
    logger.info(f"  seq_len:   {seq_len}")
    logger.info(f"  LoRA:      rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout}")
    logger.info(f"  optimizer: lr={lr} epochs={epochs} micro={micro_batch} grad_accum={grad_accum}")

    logger.info("loading base model")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = mythos_3b()
    ckpt_vocab = ckpt.get("vocab_size") if isinstance(ckpt, dict) else None
    if ckpt_vocab is not None:
        cfg.vocab_size = int(ckpt_vocab)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if ckpt_vocab is None and isinstance(state, dict) and "head.weight" in state:
        cfg.vocab_size = int(state["head.weight"].shape[0])
    saved_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    saved_t_max = getattr(saved_cfg, "max_loop_iters", None) if saved_cfg is not None else None
    if saved_t_max is None and isinstance(ckpt, dict):
        saved_t_max = ckpt.get("__saved_t_max__")
    if saved_t_max is None and isinstance(state, dict):
        saved_t_max = state.get("__saved_t_max__", getattr(cfg, "max_loop_iters", 12))
    cfg.max_loop_iters = int(saved_t_max)
    cfg.max_seq_len = int(seq_len)
    model = OpenMythos(cfg)
    sd = {k: v for k, v in state.items() if not k.startswith("__") and k not in ("freqs_cis", "freqs_cis_mla")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"missing keys: {missing[:3]}")
    if unexpected:
        logger.warning(f"unexpected keys: {unexpected[:3]}")
    model.to(torch.bfloat16)

    targets = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    n_replaced = inject_lora(model, targets, lora_rank, lora_alpha, lora_dropout)
    logger.info(f"injected LoRA into {n_replaced} linear modules")

    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        is_lora = ".A" in name or ".B" in name
        p.requires_grad = bool(is_lora)
        if is_lora:
            n_train += p.numel()
    logger.info(f"trainable: {n_train:,} ({100*n_train/n_total:.3f}% of {n_total:,})")

    model.to(device)
    model.train(True)

    tok = MythosTokenizer()
    pad_id = tok.pad_id if hasattr(tok, "pad_id") else 0
    from datasets import load_dataset
    logger.info(f"loading dataset {dataset_id}[{split}]")
    hf_ds = load_dataset(dataset_id, split=split, streaming=True)
    sft_ds = ChatSFTDataset(hf_ds, tok, seq_len, max_samples * epochs)
    loader = DataLoader(
        sft_ds,
        batch_size=micro_batch,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    target_steps = max_samples // (micro_batch * grad_accum) * epochs

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, target_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    history = []
    step = 0
    micro = 0
    loss_accum = 0.0
    t0 = time.perf_counter()
    logger.success(f"starting SFT (target_steps={target_steps})")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            (loss / grad_accum).backward()
        loss_accum += float(loss.item())
        micro += 1
        if micro < grad_accum:
            continue
        cur_lr = lr_at(step)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        avg_loss = loss_accum / grad_accum
        loss_accum = 0.0
        micro = 0

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_s = micro_batch * grad_accum * seq_len * log_every / max(1e-9, dt)
            logger.info(f"step {step}/{target_steps} | loss {avg_loss:.4f} | lr {cur_lr:.2e} | {tok_s/1e3:.1f}k tok/s")
            history.append({"step": step, "loss": avg_loss, "lr": cur_lr})
            t0 = time.perf_counter()

        if step % save_every == 0:
            sd_lora = lora_state_dict(model)
            adapter_path = out_dir / f"lora_adapter_step_{step:06d}.pt"
            torch.save(sd_lora, adapter_path)
            logger.success(f"saved {adapter_path} ({len(sd_lora)} tensors)")

        if step >= target_steps:
            break

    sd_lora = lora_state_dict(model)
    final_path = out_dir / "lora_adapter_final.pt"
    torch.save(sd_lora, final_path)
    logger.success(f"saved final LoRA adapter: {final_path}")
    with open(out_dir / "training_curve.json", "w") as f:
        json.dump({
            "history": history,
            "config": {
                "ckpt": str(ckpt_path),
                "dataset": dataset_id,
                "split": split,
                "max_samples": max_samples,
                "seq_len": seq_len,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lr": lr,
                "epochs": epochs,
            },
        }, f, indent=2)

    if save_merged:
        logger.info("merging LoRA into base weights for export")
        model.train(False)
        with torch.no_grad():
            merge_lora(model)
        merged_path = out_dir / "merged_full.pt"
        torch.save(model.state_dict(), merged_path)
        logger.success(f"saved merged full ckpt: {merged_path}")


if __name__ == "__main__":
    main()
