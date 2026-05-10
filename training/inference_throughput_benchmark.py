#!/usr/bin/env python3
"""
inference_throughput_benchmark.py

Single-GPU inference throughput measurement for the recurrent-depth model.
Designed for mini-beast.lan (RTX 5090, 32 GB) but works on any single GPU
that fits the 3B model in bf16.

Reports tokens/sec for prefill and decode at T values {1, 2, 4, 8, 16}.
At each T, also generates a short sample to verify outputs.

Inputs (env-overridable):
    CKPT       full state-dict checkpoint (default: required)
    OUT        output JSON path (default: docs/inference_benchmark.json)
    T_VALUES   comma-separated T values to test (default: "1,2,4,8,16")
    PROMPT_LEN tokens of prefill (default: 512)
    GEN_LEN    tokens to generate (default: 128)
    DEVICE     cuda device (default: cuda:0)
    BATCH      batch size (default: 1)

The model uses recurrent depth -- at inference time, a user-controlled
n_loops controls compute spend. This script quantifies the compute/quality
tradeoff across T values for paper reporting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


def parse_t_values(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def load_model(ckpt_path: str, device: str):
    logger.info(f"loading checkpoint: {ckpt_path}")
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
        saved_t_max = state.get("__saved_t_max__")
    if saved_t_max is None:
        saved_t_max = getattr(cfg, "max_loop_iters", 12)
    cfg.max_loop_iters = int(saved_t_max)
    logger.info(f"using vocab_size={cfg.vocab_size} max_loop_iters={cfg.max_loop_iters} (T values >max are clamped via LoRAAdapter)")
    model = OpenMythos(cfg)
    sd = {k: v for k, v in state.items() if not k.startswith("__")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"unexpected keys: {unexpected[:5]}")
    model.to(device).to(torch.bfloat16).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.success(f"model loaded: {n_params/1e9:.2f}B params on {device}")
    return model, cfg


@torch.no_grad()
def benchmark_at_T(model, tokenizer, T, prompt_len, gen_len, batch, device):
    """Measure prefill + decode tokens/sec at fixed T."""
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator(device="cpu").manual_seed(42)
    prefill = torch.randint(low=0, high=min(vocab_size, 50000),
                             size=(batch, prompt_len), generator=rng).to(device)

    # Warmup pass
    _ = model(prefill, n_loops=T)
    torch.cuda.synchronize()

    # Prefill timing
    t0 = time.perf_counter()
    _ = model(prefill, n_loops=T)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    prefill_tps = batch * prompt_len / prefill_time

    # Decode timing: generate gen_len tokens autoregressively (no kv cache)
    cur = prefill
    decode_times = []
    for _ in range(gen_len):
        t0 = time.perf_counter()
        out = model(cur, n_loops=T)
        next_tok = out[:, -1:, :].argmax(dim=-1)
        cur = torch.cat([cur, next_tok], dim=1)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t0)
    decode_total = sum(decode_times)
    decode_tps = batch * gen_len / decode_total
    per_token_ms = decode_total * 1000 / gen_len

    # Sample text generation from a meaningful prompt
    sample_prompt = "The recurrent depth transformer iterates the same block multiple times to"
    prompt_ids = tokenizer.encode(sample_prompt, add_special_tokens=False)
    prompt_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    cur = prompt_ids
    for _ in range(64):
        out = model(cur, n_loops=T)
        next_tok = out[:, -1:, :].argmax(dim=-1)
        cur = torch.cat([cur, next_tok], dim=1)
    sample_text = tokenizer.decode(cur[0].tolist())

    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
    mem_reserved = torch.cuda.memory_reserved(device) / 1e9

    return {
        "T": T,
        "prefill_tokens_per_sec": prefill_tps,
        "decode_tokens_per_sec": decode_tps,
        "decode_per_token_ms": per_token_ms,
        "prefill_time_sec": prefill_time,
        "decode_total_sec": decode_total,
        "memory_allocated_gb": mem_alloc,
        "memory_reserved_gb": mem_reserved,
        "sample_text": sample_text[:500],
    }


def main():
    ckpt_path = os.environ.get("CKPT")
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"CKPT env var must point to existing checkpoint (got: {ckpt_path})")
        sys.exit(1)
    out_path = os.environ.get("OUT", "inference_benchmark.json")
    t_values = parse_t_values(os.environ.get("T_VALUES", "1,2,4,8,16"))
    prompt_len = int(os.environ.get("PROMPT_LEN", "512"))
    gen_len = int(os.environ.get("GEN_LEN", "128"))
    device = os.environ.get("DEVICE", "cuda:0")
    batch = int(os.environ.get("BATCH", "1"))

    logger.info(f"config: T={t_values} prompt={prompt_len} gen={gen_len} batch={batch} device={device}")

    model, cfg = load_model(ckpt_path, device)
    tokenizer = MythosTokenizer()

    results = {
        "ckpt": ckpt_path,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "n_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "batch": batch,
        "by_T": [],
    }

    for T in t_values:
        logger.info(f"=== T={T} ===")
        try:
            r = benchmark_at_T(model, tokenizer, T, prompt_len, gen_len, batch, device)
            logger.success(f"T={T} | prefill={r['prefill_tokens_per_sec']:.1f} tok/s | "
                           f"decode={r['decode_tokens_per_sec']:.1f} tok/s | "
                           f"per_token={r['decode_per_token_ms']:.1f} ms | "
                           f"mem={r['memory_allocated_gb']:.1f} GB")
            results["by_T"].append(r)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM at T={T}; skipping higher T values: {e}")
            results["by_T"].append({"T": T, "oom": True, "error": str(e)})
            break
        except Exception as e:
            logger.error(f"unexpected error at T={T}: {e}")
            results["by_T"].append({"T": T, "error": str(e)})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"wrote {out_path}")


if __name__ == "__main__":
    main()
