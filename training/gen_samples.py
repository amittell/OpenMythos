"""
gen_samples.py

Generate varied completions from a trained mythos_3b checkpoint to inspect
what the model actually produces. Non-FSDP, single GPU, ACT-on inference
at the trained depth.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos.main import OpenMythos  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


PROMPTS = [
    "Once upon a time in a small village nestled between two mountains,",
    "The key difference between mitosis and meiosis is that",
    "To compute the area of a triangle, you",
    "Q: What is the capital of France?\nA:",
    "In Python, a list comprehension is written as",
    "The Roman Empire fell primarily because",
    "Dear Claude,\n\nI wanted to write to you because",
    "Photosynthesis is the process by which plants",
]


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_loops4_fast/step_0006103.pt",
    )
    out_path = os.environ.get("OUT", "")  # if set, also write all samples to this path
    seq_len = 1024
    max_new_tokens = 120
    temperature = 0.8
    top_k = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer...")
    tokenizer = MythosTokenizer()

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    ckpt_step = ckpt.get("step", "?")
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 4)
    n_loops = max(saved_T_max // 2, 4)  # mid-range of trained T's; round-1 -> 4, round-2 -> 6

    cfg = mythos_3b()
    cfg.vocab_size = ckpt_vocab_size
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max

    logger.info("Building model...")
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.train(False)
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.success(f"Model loaded from step {ckpt_step}  (max_loop_iters={saved_T_max})")
    logger.info(f"Sampling: n_loops={n_loops}  temp={temperature}  top_k={top_k}")

    out_lines: list[str] = []
    out_lines.append(f"# Generation samples (step {ckpt_step}, n_loops={n_loops})\n")
    for i, prompt in enumerate(PROMPTS):
        torch.manual_seed(1000 + i)
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                n_loops=n_loops,
                temperature=temperature,
                top_k=top_k,
            )
        txt = tokenizer.decode(out[0].tolist())
        block = (
            f"\n===== Sample {i + 1} =====\n"
            f"Prompt: {prompt!r}\n"
            f"Output: {txt}\n"
            f"{'=' * 72}"
        )
        print(block)
        out_lines.append(block)

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
        logger.success(f"Samples saved to {out_path}")


if __name__ == "__main__":
    main()
