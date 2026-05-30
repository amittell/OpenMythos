"""Toy synthetic depth-required task: nested-XOR expressions.

Tier 2.4 of the paper-revision plan. Trains a small (~50M param)
mythos-like recurrent-depth transformer with the same joint ACT-PonderNet-KL
loss as 3b_varT_pondernet_joint.py on synthetic nested-XOR data, where the
required reasoning depth equals the input expression's tree depth.

Three runs at LAMBDA_P in {0.1, 0.2, 0.5} discriminate the three hypotheses
in docs/experiments/toy_depth_required_task.md:
  - H1 (floor universal): mean halt = max(round(1/lambda_p), 3) for every d
  - H2 (depth-adaptive):  mean halt = min(d, round(1/lambda_p))
  - H3 (prior-only):      mean halt = round(1/lambda_p) for every d

Usage:
  CUDA_VISIBLE_DEVICES=0 LAMBDA_P=0.5 TARGET_TOKENS=5000000 \\
    OUT=/tmp/toy_lp05 python3 training/toy_depth_task.py

Environment variables:
  LAMBDA_P        halt-rate prior strength (default 0.5)
  LAMBDA_KL       KL term weight (default 1.0)
  T_MAX           max recurrence depth at training time (default 16)
  TARGET_TOKENS   total training tokens (default 5_000_000)
  OUT             output dir for ckpt + per-depth halt summary JSON
  D_MAX           max tree depth to generate (default 8)
  LR              learning rate (default 3e-4)
  D_MODEL         model dim (default 384)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- vocab + data generator -----

VOCAB = ["0", "1", "(", ")", "^", "=", "<PAD>", "<ANS>"]
PAD_ID = VOCAB.index("<PAD>")
ANS_ID = VOCAB.index("<ANS>")
ZERO_ID = VOCAB.index("0")
ONE_ID = VOCAB.index("1")
VS = len(VOCAB)


def gen_expr(d: int) -> tuple[list[int], int]:
    """Random balanced XOR expression of tree depth d. Returns (token ids, answer 0/1)."""
    if d == 0:
        b = random.randint(0, 1)
        return [ZERO_ID if b == 0 else ONE_ID], b
    L, lv = gen_expr(d - 1)
    R, rv = gen_expr(d - 1)
    expr = [VOCAB.index("(")] + L + [VOCAB.index("^")] + R + [VOCAB.index(")")]
    return expr, lv ^ rv


def gen_batch_random_depth(batch_size: int, d_min: int, d_max: int, seq_len: int, device: str):
    """Each example has a depth uniformly sampled in {d_min, ..., d_max}."""
    xs = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    depths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        d = random.randint(d_min, d_max)
        expr, ans = gen_expr(d)
        seq = expr + [VOCAB.index("="), ANS_ID]
        if len(seq) > seq_len:
            expr, ans = gen_expr(1)
            seq = expr + [VOCAB.index("="), ANS_ID]
            d = 1
        xs[i, : len(seq)] = torch.tensor(seq, device=device)
        targets[i] = ZERO_ID if ans == 0 else ONE_ID
        pos[i] = len(seq) - 1
        depths[i] = d
    return xs, targets, depths, pos


def gen_batch_fixed_depth(batch_size: int, d: int, seq_len: int, device: str):
    """Every example has exactly depth d."""
    xs = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        expr, ans = gen_expr(d)
        seq = expr + [VOCAB.index("="), ANS_ID]
        if len(seq) > seq_len:
            expr, ans = gen_expr(1)
            seq = expr + [VOCAB.index("="), ANS_ID]
        xs[i, : len(seq)] = torch.tensor(seq, device=device)
        targets[i] = ZERO_ID if ans == 0 else ONE_ID
        pos[i] = len(seq) - 1
    return xs, targets, pos


# ----- model -----


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.ln1(h)
        attn_out, _ = self.attn(a, a, a, need_weights=False)
        h = h + attn_out
        h = h + self.mlp(self.ln2(h))
        return h


class RecurrentBlock(nn.Module):
    """Single transformer block + loop-index embedding + ACT halting head.

    Mirrors the paper's §3.1 recurrent block: per-iteration loop-index embedding
    e_t is added before the transformer block, then a sigmoid halting head reads
    the post-block hidden state. ACT-weighted output accumulation happens in the
    outer ToyModel.forward.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, t_max: int):
        super().__init__()
        self.loop_emb = nn.Embedding(t_max + 1, d_model)
        self.block = TransformerBlock(d_model, n_heads, d_ff)
        self.act_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.act_head.bias)

    def forward(self, h: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        loop_e = self.loop_emb(torch.tensor(t, device=h.device, dtype=torch.long))
        h_in = h + loop_e.view(1, 1, -1)
        h_out = self.block(h_in)
        p = torch.sigmoid(self.act_head(h_out)).squeeze(-1)  # [B, L]
        return h_out, p


class ToyModel(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        d_ff: int = 1536,
        n_prelude: int = 4,
        n_coda: int = 4,
        t_max: int = 16,
        max_len: int = 512,
        vocab_size: int = VS,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.prelude = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_prelude)])
        self.recurrent = RecurrentBlock(d_model, n_heads, d_ff, t_max)
        self.coda = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_coda)])
        self.out_ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size)
        self.t_max = t_max
        self.max_len = max_len

    def forward(self, x: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run with K recurrence iterations under ACT weighting.

        Returns (logits [B, L, V], p_t [K, B, L]).
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).view(1, L)
        h = self.embed(x) + self.pos(pos)
        for blk in self.prelude:
            h = blk(h)
        # ACT loop.
        h_out = torch.zeros_like(h)
        cum_p = torch.zeros(B, L, device=h.device)
        p_list = []
        for t in range(1, K + 1):
            h, p = self.recurrent(h, t)
            p_list.append(p)
            remainder = (1.0 - cum_p).clamp(min=0.0)
            if t < K:
                w = p * remainder
                cum_p = cum_p + w
            else:
                # Last iteration takes the remainder (PonderNet-style).
                w = remainder
            h_out = h_out + w.unsqueeze(-1) * h
        h_out = self.out_ln(h_out)
        for blk in self.coda:
            h_out = blk(h_out)
        logits = self.out_head(h_out)
        p_stack = torch.stack(p_list, dim=0)  # [K, B, L]
        return logits, p_stack


# ----- loss -----


def joint_loss(
    logits: torch.Tensor,
    p_t: torch.Tensor,
    target: torch.Tensor,
    pos: torch.Tensor,
    lambda_p: float,
    lambda_kl: float,
) -> tuple[torch.Tensor, dict]:
    """Mirrors training/3b_varT_pondernet_joint.py's joint loss at the answer position."""
    B = logits.shape[0]
    # Gather logits at the <ANS> position.
    idx = pos.view(B, 1, 1).expand(B, 1, logits.shape[-1])
    logits_ans = logits.gather(1, idx).squeeze(1)  # [B, V]
    ce = F.cross_entropy(logits_ans, target)
    # Gather p_t at the same position.
    K = p_t.shape[0]
    pos_exp = pos.view(1, B, 1).expand(K, B, 1)
    p_ans = p_t.gather(2, pos_exp).squeeze(-1)  # [K, B]
    # Bernoulli KL: p log(p/lp) + (1-p) log((1-p)/(1-lp)).
    eps = 1e-6
    p_c = p_ans.clamp(eps, 1.0 - eps)
    lp_c = max(eps, min(1.0 - eps, lambda_p))
    log_lp = math.log(lp_c)
    log_1m_lp = math.log(1.0 - lp_c)
    kl = p_c * (torch.log(p_c) - log_lp) + (1.0 - p_c) * (torch.log(1.0 - p_c) - log_1m_lp)
    kl_term = kl.mean()
    loss = ce + lambda_kl * kl_term
    return loss, {"ce": ce.item(), "kl": kl_term.item(), "p_first": p_ans[0].mean().item()}


# ----- evaluation helpers -----


def measure_halt_distribution(p_ans: torch.Tensor) -> torch.Tensor:
    """Given per-iteration halting probabilities at the answer position p_ans of shape [T, B],
    return mean halt step per example as a [B] tensor.
    Computes halt-step weights w_t = p_t prod_{s<t}(1 - p_s), with the last iteration
    taking the remainder mass.
    """
    T, B = p_ans.shape
    one_minus = 1.0 - p_ans
    cum_one_minus = torch.cumprod(one_minus, dim=0)
    shifted = torch.cat([torch.ones(1, B, device=p_ans.device), cum_one_minus[:-1]], dim=0)
    w = p_ans * shifted
    w[-1] = w[-1] + cum_one_minus[-1]
    w_norm = w / w.sum(dim=0, keepdim=True).clamp(min=1e-6)
    t_idx = torch.arange(1, T + 1, device=p_ans.device, dtype=w.dtype).view(T, 1)
    halt_mean = (t_idx * w_norm).sum(dim=0)
    return halt_mean


# ----- training -----


def main():
    lambda_p = float(os.environ.get("LAMBDA_P", "0.5"))
    lambda_kl = float(os.environ.get("LAMBDA_KL", "1.0"))
    t_max = int(os.environ.get("T_MAX", "16"))
    target_tokens = int(os.environ.get("TARGET_TOKENS", "5000000"))
    d_max = int(os.environ.get("D_MAX", "8"))
    d_min = int(os.environ.get("D_MIN", "1"))
    lr = float(os.environ.get("LR", "3e-4"))
    d_model = int(os.environ.get("D_MODEL", "384"))
    n_heads = int(os.environ.get("N_HEADS", "6"))
    d_ff = int(os.environ.get("D_FF", str(4 * d_model)))
    batch = int(os.environ.get("BATCH", "32"))
    seq_len = int(os.environ.get("SEQ_LEN", "512"))
    log_every = int(os.environ.get("LOG_EVERY", "50"))
    out_dir = Path(os.environ.get("OUT", "/tmp/toy_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    random.seed(0)

    model = ToyModel(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        t_max=t_max,
        max_len=seq_len,
        vocab_size=VS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"toy model params: {n_params/1e6:.1f}M  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}")
    print(
        f"config: lambda_p={lambda_p} lambda_kl={lambda_kl} t_max={t_max} target={target_tokens/1e6:.1f}M tok  d_min={d_min} d_max={d_max}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = max(1, target_tokens // (batch * seq_len))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    tokens_seen = 0
    step = 0
    t0 = time.time()
    model.train(True)
    while tokens_seen < target_tokens:
        # Variable-T training: sample K each step uniformly in {2, ..., t_max}.
        K = random.randint(2, t_max)
        xs, targets, depths, pos = gen_batch_random_depth(batch, d_min, d_max, seq_len, device)
        logits, p_t = model(xs, K)
        loss, stats = joint_loss(logits, p_t, targets, pos, lambda_p, lambda_kl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens_seen += batch * seq_len
        step += 1
        if step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{total_steps}  K={K:2d}  loss={loss.item():.4f}  "
                f"ce={stats['ce']:.4f}  kl={stats['kl']:.4f}  p1={stats['p_first']:.3f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  {tokens_seen/1e6:.2f}M tok  {elapsed:.0f}s"
            )

    print("training done; running per-depth halt + accuracy evaluation")
    model.train(False)
    n_eval = 256
    K_eval_list = [1, 2, 3, 4, 6, 8, 12, 16]
    results: dict = {}
    with torch.no_grad():
        for d in range(1, d_max + 1):
            xs, targets, pos = gen_batch_fixed_depth(n_eval, d, seq_len, device)
            logits, p_t = model(xs, t_max)
            B = xs.shape[0]
            pos_exp = pos.view(1, B, 1).expand(t_max, B, 1)
            p_ans = p_t.gather(2, pos_exp).squeeze(-1)  # [T, B]
            halt_mean = measure_halt_distribution(p_ans)
            row: dict = {
                "halt_mean": halt_mean.mean().item(),
                "halt_std": halt_mean.std().item(),
                "halt_per_example": halt_mean.cpu().tolist(),
                "p_first_mean": p_ans[0].mean().item(),
                "acc_by_K": {},
            }
            for K in K_eval_list:
                logits_K, _ = model(xs, K)
                idx = pos.view(B, 1, 1).expand(B, 1, logits_K.shape[-1])
                logits_ans = logits_K.gather(1, idx).squeeze(1)
                pred = logits_ans.argmax(-1)
                acc = (pred == targets).float().mean().item()
                row["acc_by_K"][K] = acc
            results[d] = row
            a = row["acc_by_K"]
            print(
                f"d={d}: halt_mean={row['halt_mean']:.2f} (+/- {row['halt_std']:.2f})  "
                f"acc[K=1,2,4,8,16]={a[1]:.2f}/{a[2]:.2f}/{a[4]:.2f}/{a[8]:.2f}/{a[16]:.2f}"
            )

    out = {
        "config": {
            "lambda_p": lambda_p,
            "lambda_kl": lambda_kl,
            "t_max": t_max,
            "target_tokens": target_tokens,
            "d_min": d_min,
            "d_max": d_max,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "n_params": n_params,
        },
        "results_by_depth": results,
    }
    out_path = out_dir / f"toy_results_lp{lambda_p}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    torch.save(model.state_dict(), out_dir / f"toy_ckpt_lp{lambda_p}.pt")
    print(f"wrote ckpt to {out_dir / f'toy_ckpt_lp{lambda_p}.pt'}")


if __name__ == "__main__":
    main()
