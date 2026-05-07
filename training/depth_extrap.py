"""
depth_extrap.py

Depth-extrapolation measurement for a trained mythos_3b checkpoint.

Loads a checkpoint that was trained at max_loop_iters=4 and measures it at
n_loops in [4, 8, 16, 32] on a fixed held-out batch from FineWeb-Edu. For
each depth, reports cross-entropy loss, perplexity, and a short generation
sample. Results are written to a JSON file.

Design notes:
  * The model is loaded as a non-FSDP module on a single GPU. save_checkpoint
    in training/3b_loops4_fast.py writes FULL_STATE_DICT, so a raw OpenMythos
    can consume ckpt["model"] directly via model.load_state_dict.
  * cfg.max_loop_iters is kept at 4 at construction time so the internal LoRA
    embedding table matches the trained weights exactly. RecurrentBlock's
    forward accepts n_loops as an argument and overrides the config value at
    call time, so inference at 8/16/32 is just a kwarg.
  * The LoRAAdapter clamps loop_t to num_embeddings-1 (main.py:588), so
    iterations beyond the trained range reuse the last trained per-loop scale.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_mythos.main import OpenMythos, RecurrentBlock, loop_index_embedding  # noqa: E402
from open_mythos.tokenizer import MythosTokenizer  # noqa: E402
from open_mythos.variants import mythos_3b  # noqa: E402


def recurrent_forward_no_act(
    self: RecurrentBlock,
    h: torch.Tensor,
    e: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask=None,
    n_loops: int | None = None,
    kv_cache=None,
) -> torch.Tensor:
    """
    Replacement forward for RecurrentBlock that bypasses ACT halting.

    Runs exactly n_loops iterations unconditionally and returns the final
    hidden state (rather than an ACT-weighted sum). Used to measure the
    pure effect of recurrent depth on loss, independent of the halting
    mechanism that otherwise caps effective depth at the trained range.
    """
    n_loops = n_loops or self.cfg.max_loop_iters
    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h = self.injection(h, e, trans_out)
    return h


def build_held_out_batches(
    tokenizer: MythosTokenizer,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    skip_samples: int = 300_000,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Pull a small fixed slice of FineWeb-Edu sample-10BT from past the training
    consumption window. Training saw roughly 200K samples total across 4 ranks
    (100M tokens / ~500 tokens per doc), so skipping 300K samples lands us in
    fresh territory for a generalization probe that compares depths on
    identical inputs.
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    ).skip(skip_samples)

    needed_tokens = n_batches * batch_size * (seq_len + 1)
    buf: list[int] = []
    for sample in ds:
        buf.extend(tokenizer.encode(sample["text"]))
        if len(buf) >= needed_tokens:
            break

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len + 1
    for i in range(n_batches):
        inputs_rows, targets_rows = [], []
        for j in range(batch_size):
            start = (i * batch_size + j) * stride
            chunk = buf[start : start + stride]
            inputs_rows.append(chunk[:-1])
            targets_rows.append(chunk[1:])
        x = torch.tensor(inputs_rows, dtype=torch.long)
        y = torch.tensor(targets_rows, dtype=torch.long)
        batches.append((x, y))
    return batches


@torch.no_grad()
def measure_at_depth(
    model: OpenMythos,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    n_loops: int,
    device: str,
) -> dict:
    """Run forward-only CE over all batches at fixed n_loops."""
    model.train(False)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(y.numel())
    elapsed = time.time() - t0
    avg_loss = total_loss / total_tokens
    return {
        "n_loops": n_loops,
        "loss": avg_loss,
        "ppl": math.exp(avg_loss) if avg_loss < 50 else float("inf"),
        "tokens_measured": total_tokens,
        "elapsed_s": round(elapsed, 2),
    }


def build_gsm8k_batches(
    tokenizer: MythosTokenizer,
    n_problems: int = 50,
    max_len: int = 512,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load the first n_problems from GSM8K test split and tokenize them as
    Q/A pairs. Returns a list of (input_ids, target_ids, answer_mask)
    tuples where answer_mask is 1 on answer tokens and 0 on prompt tokens,
    so CE loss can be restricted to the answer span only.

    Format: "Q: <question>\\nA: <answer_with_####_line>"
    The answer_mask lets us measure the model's loss on ONLY the answer
    tokens - we do not care how well the model reconstructs the question.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seen = 0
    for sample in ds:
        if seen >= n_problems:
            break
        q = sample["question"].strip()
        a = sample["answer"].strip()
        prompt = f"Q: {q}\nA:"
        full = f"{prompt} {a}"

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(full)
        if len(full_ids) < len(prompt_ids) + 2:
            continue

        if len(full_ids) > max_len + 1:
            full_ids = full_ids[: max_len + 1]

        inputs = full_ids[:-1]
        targets = full_ids[1:]
        # answer mask is 1 on positions whose TARGET token is an answer token
        prompt_len = len(prompt_ids)
        mask = [0] * len(targets)
        for i in range(len(targets)):
            # target[i] is full_ids[i+1]; it is an answer token iff i+1 >= prompt_len
            if (i + 1) >= prompt_len:
                mask[i] = 1

        batches.append(
            (
                torch.tensor([inputs], dtype=torch.long),
                torch.tensor([targets], dtype=torch.long),
                torch.tensor([mask], dtype=torch.float32),
            )
        )
        seen += 1
    return batches


def build_tinystories_batches(
    tokenizer: MythosTokenizer,
    seq_len: int,
    n_batches: int,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Pull a fixed slice of TinyStories validation set as a complementary
    perplexity probe. TinyStories is simple narrative text; if web-text
    perplexity improved from round 1 to round 2 but TinyStories stayed flat,
    the model overfit to FineWeb-Edu's style rather than learning general
    distributional behavior.
    """
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    needed = n_batches * batch_size * (seq_len + 1)
    buf: list[int] = []
    for sample in ds:
        buf.extend(tokenizer.encode(sample["text"]))
        if len(buf) >= needed:
            break

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len + 1
    for i in range(n_batches):
        inputs_rows, targets_rows = [], []
        for j in range(batch_size):
            start = (i * batch_size + j) * stride
            chunk = buf[start : start + stride]
            inputs_rows.append(chunk[:-1])
            targets_rows.append(chunk[1:])
        x = torch.tensor(inputs_rows, dtype=torch.long)
        y = torch.tensor(targets_rows, dtype=torch.long)
        batches.append((x, y))
    return batches


@torch.no_grad()
def measure_gsm8k_at_depth(
    model: OpenMythos,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_loops: int,
    device: str,
) -> dict:
    """
    Measure mean per-answer-token CE loss on GSM8K at the given depth.

    Each problem is forwarded individually because lengths vary. The answer
    mask selects only the answer-span positions for the loss mean.
    """
    model.train(False)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for x, y, mask in batches:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, n_loops=n_loops)
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)
        total_loss += float((per_tok * mask).sum().item())
        total_tokens += int(mask.sum().item())
    elapsed = time.time() - t0
    avg = total_loss / total_tokens if total_tokens else float("nan")
    return {
        "n_loops": n_loops,
        "loss": avg,
        "ppl": math.exp(avg) if avg < 50 else float("inf"),
        "answer_tokens_measured": total_tokens,
        "elapsed_s": round(elapsed, 2),
    }


@torch.no_grad()
def generate_sample(
    model: OpenMythos,
    tokenizer: MythosTokenizer,
    prompt: str,
    n_loops: int,
    device: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Sample a short completion at the given loop depth."""
    model.train(False)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=temperature,
            top_k=top_k,
        )
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    ckpt_path = os.environ.get(
        "CKPT",
        "/home/alexm/OpenMythos/checkpoints_3b_loops4_fast/step_0006103.pt",
    )
    out_json = os.environ.get(
        "OUT",
        "/home/alexm/OpenMythos/docs/depth_extrap_results.json",
    )
    seq_len = 1024
    batch_size = 4
    n_batches = 16  # 64 sequences x 1024 tokens ~ 65K tokens
    depths_env = os.environ.get("DEPTHS")
    depths = (
        [int(d) for d in depths_env.split(",") if d.strip()]
        if depths_env
        else [4, 8, 16, 32]
    )
    skip_gsm8k = os.environ.get("SKIP_GSM8K", "0") == "1"
    skip_tinystories = os.environ.get("SKIP_TINYSTORIES", "0") == "1"
    prompt = "The recurrent-depth transformer architecture"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  ckpt={ckpt_path}")

    logger.info("Loading tokenizer...")
    tokenizer = MythosTokenizer()
    logger.info(f"tokenizer vocab_size={tokenizer.vocab_size}")

    logger.info("Loading checkpoint to CPU first (45 GB pt file)...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_step = ckpt.get("step", "unknown")
    ckpt_vocab_size = int(ckpt.get("vocab_size", tokenizer.vocab_size))
    logger.success(f"Checkpoint loaded: step={ckpt_step}  vocab_size={ckpt_vocab_size}")

    # Read training-time max_loop_iters from the saved cfg if available, so the
    # LoRA per-loop embedding table sizes match the trained weights. Round 1
    # used max_loop_iters=4; round 2 uses 12. Falls back to 4 for older ckpts
    # that did not save cfg.
    saved_cfg = ckpt.get("cfg", None)
    saved_T_max = getattr(saved_cfg, "max_loop_iters", 4)
    cfg = mythos_3b()
    cfg.vocab_size = ckpt_vocab_size
    cfg.max_seq_len = seq_len
    cfg.max_loop_iters = saved_T_max
    logger.info(f"using max_loop_iters={saved_T_max} from saved cfg")

    logger.info("Building model...")
    model = OpenMythos(cfg)
    logger.info(f"params={sum(p.numel() for p in model.parameters()):,}")

    logger.info("Loading weights into model...")
    model.load_state_dict(ckpt["model"])
    del ckpt  # free the CPU memory before moving to device
    model = model.to(device)
    if device == "cuda":
        torch.cuda.empty_cache()

    # spectral radius sanity (must be < 1 for Parcae stability)
    A = model.recurrent.injection.get_A()
    logger.info(f"spectral radius rho(A) max={A.max().item():.4f} (must be < 1)")

    logger.info(
        f"Pulling {n_batches * batch_size} held-out sequences from FineWeb-Edu..."
    )
    t0 = time.time()
    batches = build_held_out_batches(tokenizer, seq_len, n_batches, batch_size)
    logger.success(
        f"Built {len(batches)} batches x {batch_size} x {seq_len} tokens "
        f"in {time.time() - t0:.1f}s"
    )

    # Warmup to absorb first-call compilation / allocation cost
    logger.info("Warmup forward at n_loops=4...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(batches[0][0].to(device), n_loops=4)
    if device == "cuda":
        torch.cuda.synchronize()

    # --- Pass 1: measure with ACT enabled (normal inference path) ---
    results_act_on = []
    for K in depths:
        logger.info(f"--- ACT-on measure at n_loops={K} ---")
        r = measure_at_depth(model, batches, K, device)
        r["act"] = "on"
        logger.success(
            f"ACT-on  n_loops={K}  loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  "
            f"elapsed={r['elapsed_s']}s"
        )
        results_act_on.append(r)

    # --- Pass 2: bypass ACT, force all K iterations, use final h ---
    logger.info("Patching RecurrentBlock to bypass ACT halting...")
    original_forward = RecurrentBlock.forward
    RecurrentBlock.forward = recurrent_forward_no_act
    try:
        results_act_off = []
        for K in depths:
            logger.info(f"--- ACT-off measure at n_loops={K} ---")
            r = measure_at_depth(model, batches, K, device)
            r["act"] = "off"
            logger.success(
                f"ACT-off n_loops={K}  loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  "
                f"elapsed={r['elapsed_s']}s"
            )
            results_act_off.append(r)
    finally:
        RecurrentBlock.forward = original_forward
    results = results_act_on + results_act_off

    # --- Pass 3: GSM8K probe (answer-only CE at each depth, ACT-on) ---
    results_gsm8k = []
    if not skip_gsm8k:
        logger.info("Loading GSM8K test split (first 50 problems)...")
        t0 = time.time()
        gsm8k_batches = build_gsm8k_batches(tokenizer, n_problems=50, max_len=512)
        logger.success(
            f"Built {len(gsm8k_batches)} GSM8K problems in {time.time() - t0:.1f}s"
        )
        for K in depths:
            logger.info(f"--- GSM8K ACT-on measure at n_loops={K} ---")
            r = measure_gsm8k_at_depth(model, gsm8k_batches, K, device)
            r["act"] = "on"
            logger.success(
                f"GSM8K  n_loops={K}  answer_loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  "
                f"elapsed={r['elapsed_s']}s  (tokens={r['answer_tokens_measured']})"
            )
            results_gsm8k.append(r)
    else:
        logger.info("SKIP_GSM8K=1; skipping GSM8K probe")

    # --- Pass 4: TinyStories probe (general-distribution PPL, ACT-on) ---
    results_tinystories = []
    if not skip_tinystories:
        logger.info("Pulling TinyStories validation batches...")
        t0 = time.time()
        tinystories_batches = build_tinystories_batches(
            tokenizer, seq_len, n_batches, batch_size
        )
        logger.success(
            f"Built {len(tinystories_batches)} TinyStories batches in {time.time() - t0:.1f}s"
        )
        for K in depths:
            logger.info(f"--- TinyStories ACT-on measure at n_loops={K} ---")
            r = measure_at_depth(model, tinystories_batches, K, device)
            r["act"] = "on"
            logger.success(
                f"TinyStories  n_loops={K}  loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  "
                f"elapsed={r['elapsed_s']}s"
            )
            results_tinystories.append(r)
    else:
        logger.info("SKIP_TINYSTORIES=1; skipping TinyStories probe")

    # Generation samples: same prompt, reseeded per depth for comparability
    gens: dict[int, str] = {}
    for K in depths:
        logger.info(f"--- Generation at n_loops={K} ---")
        torch.manual_seed(1234)
        txt = generate_sample(model, tokenizer, prompt, K, device)
        gens[K] = txt
        logger.info(f"n_loops={K}: {txt!r}")

    payload = {
        "ckpt_path": ckpt_path,
        "step": ckpt_step,
        "vocab_size": ckpt_vocab_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "n_sequences_measured": n_batches * batch_size,
        "trained_max_loop_iters": 4,
        "depths": depths,
        "results_fineweb": results,
        "results_gsm8k": results_gsm8k,
        "results_tinystories": results_tinystories,
        "prompt": prompt,
        "generations": gens,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.success(f"Wrote {out_json}")

    logger.info("--- FineWeb-Edu summary ---")
    logger.info("n_loops  ACT  loss     ppl       s")
    for r in results:
        logger.info(
            f"{r['n_loops']:>7}  {r['act']:>3}  {r['loss']:7.4f}  {r['ppl']:8.2f}  {r['elapsed_s']:6.1f}"
        )
    logger.info("--- GSM8K summary (answer-only CE) ---")
    logger.info("n_loops  ACT  loss     ppl       s")
    for r in results_gsm8k:
        logger.info(
            f"{r['n_loops']:>7}  {r['act']:>3}  {r['loss']:7.4f}  {r['ppl']:8.2f}  {r['elapsed_s']:6.1f}"
        )
    logger.info("--- TinyStories summary ---")
    logger.info("n_loops  ACT  loss     ppl       s")
    for r in results_tinystories:
        logger.info(
            f"{r['n_loops']:>7}  {r['act']:>3}  {r['loss']:7.4f}  {r['ppl']:8.2f}  {r['elapsed_s']:6.1f}"
        )


if __name__ == "__main__":
    main()
