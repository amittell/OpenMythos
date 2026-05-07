#!/usr/bin/env python3
"""
Generate the same prompts at multiple recurrent loop depths to qualitatively
inspect whether varying inference T changes the model's behavior.

Reuses the prompt list from gen_samples.py. For each prompt, samples at
n_loops in [2, 6, 12, 24] with the same fixed seed, so any differences
across depths come from the architecture, not from sampling randomness.

Env vars:
    CKPT  path to consolidated full-state-dict checkpoint
    OUT   output text path (markdown-style blocks)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.gen_samples import PROMPTS  # noqa: E402
from open_mythos import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


DEPTHS = [2, 6, 12, 24]


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt",
    )
    out_path = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/gen_samples_round2_multidepth.txt",
    )
    seq_len = 1024
    max_new_tokens = 100
    temperature = 0.8
    top_k = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")

    tokenizer = MythosTokenizer()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_step = ckpt.get("step", "?")
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 12)

    cfg = mythos_3b()
    cfg.vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.success(f"Model loaded from step {ckpt_step}  (max_loop_iters={saved_T_max})")

    out_lines: list[str] = []
    out_lines.append(
        f"# Multi-depth generation samples (step {ckpt_step}, depths={DEPTHS})\n"
    )

    for i, prompt in enumerate(PROMPTS):
        out_lines.append(f"\n## Sample {i + 1}\n")
        out_lines.append(f"Prompt: `{prompt}`\n")
        for K in DEPTHS:
            torch.manual_seed(1000 + i)
            ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    n_loops=K,
                    temperature=temperature,
                    top_k=top_k,
                )
            txt = tokenizer.decode(out[0].tolist())
            out_lines.append(f"\n**K={K}:** {txt}\n")
            logger.info(f"sample {i+1} K={K}: {txt[:80]!r}...")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out_lines) + "\n")
    logger.success(f"wrote {out_path}")


if __name__ == "__main__":
    main()
