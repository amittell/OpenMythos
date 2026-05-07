#!/usr/bin/env python3
"""
Smoke-test torch.compile on mythos_1b under variable-T training to find out
whether compile actually accelerates this architecture, given that the
RecurrentBlock has features that tend to break the compiler:

  - Dynamic `n_loops` (sampled per optimizer step in varT)
  - ACT halting with a conditional `break` on `halted.all()`
  - MoE expert routing (dynamic dispatch)
  - Loop-index embedding indexed by Python int `t`

Runs four passes, each with 50 measured optimizer steps after a 5-step warmup,
and reports tokens/sec for each:

  PASS 1 (eager)        no torch.compile
  PASS 2 (model)        torch.compile(model, dynamic=True)
  PASS 3 (transformer)  compile only each TransformerBlock (leaves the
                        RecurrentBlock loop in eager Python; the per-iteration
                        attention+FFN compute is compiled)
  PASS 4 (expert)       compile only the Expert FFN inside each block (the
                        finest-grained compile target, isolates whether the
                        per-token MLP is the bottleneck and whether compile
                        helps at that granularity)

Reports tokens/sec for each pass, ratio vs eager baseline, and any recompile
warnings emitted by torch._dynamo. Reset between passes is done by detaching
the optimizer and rebuilding.

Hardware: any single GPU with >= 12 GB. Runs on RTX 5090, RTX 6000 Pro, etc.

Usage:
    python3 training/smoke_1b_compile_test.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch._dynamo as dynamo
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos.main import (  # noqa: E402
    OpenMythos,
    MythosConfig,
    TransformerBlock,
    Expert,
)
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402


SEQ_LEN = 1024
MICRO_BATCH = 2
GRAD_ACCUM = 4
WARMUP = 5
MEASURE = 50
T_MIN, T_MAX = 2, 12  # full varT range — small model fits fine
LR = 3e-4


def small_test_config(vocab_size: int) -> MythosConfig:
    """
    ~80M-param config with the same architectural features as mythos_3b
    (MoE, MLA, recurrent block + ACT, LoRA per-loop adapter). Sized to fit
    comfortably on a 32 GB GPU at varT=12, so torch.compile compatibility
    decisions transfer to the larger model.
    """
    return MythosConfig(
        vocab_size=vocab_size,
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=SEQ_LEN,
        max_loop_iters=T_MAX,
        prelude_layers=2,
        coda_layers=2,
        n_experts=16,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1024,
        lora_rank=16,
        attn_type="mla",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=16,
        qk_nope_head_dim=48,
        v_head_dim=48,
        rope_theta=500_000.0,
    )


def synth_batch(vocab_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Random tokens are sufficient for a throughput smoke test."""
    x = torch.randint(0, vocab_size, (MICRO_BATCH, SEQ_LEN), device=device)
    y = torch.randint(0, vocab_size, (MICRO_BATCH, SEQ_LEN), device=device)
    return x, y


def build_model(vocab_size: int, device: str) -> OpenMythos:
    cfg = small_test_config(vocab_size)
    model = OpenMythos(cfg).to(device)
    return model


def reset_dynamo_counters() -> None:
    """Wipe dynamo's recompile/cache counters so each pass starts clean."""
    dynamo.reset()


def run_pass(
    label: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str,
) -> dict:
    """
    Drive WARMUP + MEASURE optimizer steps through `model`, with random T per
    step (variable-T regime). Reports avg tokens/sec across the measured
    portion only.
    """
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()

    # warmup so compile-driven first-call overhead is amortized out
    logger.info(f"[{label}] warmup ({WARMUP} steps)...")
    for step in range(WARMUP):
        T = random.Random(step).randint(T_MIN, T_MAX)
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM):
            x, y = synth_batch(vocab_size, device)
            with amp_ctx:
                logits = model(x, n_loops=T)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / GRAD_ACCUM
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seen_tokens = 0
    for step in range(MEASURE):
        T = random.Random(WARMUP + step).randint(T_MIN, T_MAX)
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM):
            x, y = synth_batch(vocab_size, device)
            with amp_ctx:
                logits = model(x, n_loops=T)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / GRAD_ACCUM
            loss.backward()
            seen_tokens += MICRO_BATCH * SEQ_LEN
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = seen_tokens / elapsed
    sec_per_step = elapsed / MEASURE
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return {
        "label": label,
        "tok_per_sec": tok_per_sec,
        "sec_per_step": sec_per_step,
        "elapsed": elapsed,
        "max_mem_gb": mem_gb,
    }


def fresh_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=True,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)
    device = "cuda"
    torch.cuda.set_device(0)

    # Allow more compile-cache entries in case of input-shape variation
    dynamo.config.cache_size_limit = 64

    tokenizer = MythosTokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                f"vocab_size={vocab_size}  seq_len={SEQ_LEN}  "
                f"micro_batch={MICRO_BATCH}  grad_accum={GRAD_ACCUM}")

    results: list[dict] = []

    # PASS 1: eager baseline ---------------------------------------------------
    logger.info("=" * 60)
    logger.info("PASS 1: eager baseline (no torch.compile)")
    logger.info("=" * 60)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size, device)
    opt = fresh_optimizer(model)
    r1 = run_pass("eager", model, opt, vocab_size, device)
    results.append(r1)
    logger.success(f"eager: {r1['tok_per_sec']:.0f} tok/s  "
                   f"{r1['sec_per_step']:.2f} s/step  mem {r1['max_mem_gb']:.1f} GB")
    del model, opt
    torch.cuda.empty_cache()
    reset_dynamo_counters()

    # PASS 2: compile whole model ----------------------------------------------
    logger.info("=" * 60)
    logger.info("PASS 2: torch.compile(model, dynamic=True)")
    logger.info("=" * 60)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size, device)
    try:
        model_c = torch.compile(model, dynamic=True)
        opt = fresh_optimizer(model)  # optimizer over the un-compiled module's params
        r2 = run_pass("compile(model)", model_c, opt, vocab_size, device)
        results.append(r2)
        logger.success(f"compile(model): {r2['tok_per_sec']:.0f} tok/s  "
                       f"{r2['sec_per_step']:.2f} s/step  mem {r2['max_mem_gb']:.1f} GB")
    except Exception as exc:
        logger.error(f"compile(model) failed: {exc}")
        results.append({"label": "compile(model)", "tok_per_sec": 0,
                        "sec_per_step": float("inf"), "elapsed": 0,
                        "max_mem_gb": 0, "error": str(exc)})
    del model, model_c
    if "opt" in dir():
        del opt
    torch.cuda.empty_cache()
    reset_dynamo_counters()

    # PASS 3: compile each TransformerBlock ------------------------------------
    logger.info("=" * 60)
    logger.info("PASS 3: per-block compile (TransformerBlock only)")
    logger.info("=" * 60)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size, device)
    try:
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                module.forward = torch.compile(module.forward, dynamic=True)
        opt = fresh_optimizer(model)
        r3 = run_pass("compile(TransformerBlock)", model, opt, vocab_size, device)
        results.append(r3)
        logger.success(
            f"compile(TransformerBlock): {r3['tok_per_sec']:.0f} tok/s  "
            f"{r3['sec_per_step']:.2f} s/step  mem {r3['max_mem_gb']:.1f} GB"
        )
    except Exception as exc:
        logger.error(f"compile(TransformerBlock) failed: {exc}")
        results.append({"label": "compile(TransformerBlock)", "tok_per_sec": 0,
                        "sec_per_step": float("inf"), "elapsed": 0,
                        "max_mem_gb": 0, "error": str(exc)})
    del model, opt
    torch.cuda.empty_cache()
    reset_dynamo_counters()

    # PASS 4: compile only Expert FFNs -----------------------------------------
    logger.info("=" * 60)
    logger.info("PASS 4: per-Expert compile (dense FFN only)")
    logger.info("=" * 60)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size, device)
    try:
        compiled = 0
        for module in model.modules():
            if isinstance(module, Expert):
                module.forward = torch.compile(module.forward, dynamic=True)
                compiled += 1
        logger.info(f"compiled {compiled} Expert modules")
        opt = fresh_optimizer(model)
        r4 = run_pass("compile(Expert)", model, opt, vocab_size, device)
        results.append(r4)
        logger.success(
            f"compile(Expert): {r4['tok_per_sec']:.0f} tok/s  "
            f"{r4['sec_per_step']:.2f} s/step  mem {r4['max_mem_gb']:.1f} GB"
        )
    except Exception as exc:
        logger.error(f"compile(Expert) failed: {exc}")
        results.append({"label": "compile(Expert)", "tok_per_sec": 0,
                        "sec_per_step": float("inf"), "elapsed": 0,
                        "max_mem_gb": 0, "error": str(exc)})

    # Summary ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    baseline = results[0]["tok_per_sec"]
    print()
    print(f"  {'pass':<28} {'tok/s':>10} {'s/step':>8} {'speedup':>8}")
    for r in results:
        speedup = (r["tok_per_sec"] / baseline) if baseline else 0
        err = r.get("error", "")
        line = f"  {r['label']:<28} {r['tok_per_sec']:>10.0f} {r['sec_per_step']:>8.2f} {speedup:>7.2f}x"
        if err:
            line += f"  ERROR: {err[:60]}"
        print(line)


if __name__ == "__main__":
    main()
