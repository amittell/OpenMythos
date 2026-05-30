# Toy depth-required task (Tier 2.4 of the paper-revision plan)

## Question

Does the architectural floor at iteration ~3 (§8.1.1) reproduce on a task where the optimal halt depth is **provably greater than 3 for some inputs**, and where the optimal halt depth is **a strictly increasing function of an input parameter** the model can observe?

If joint ACT-PonderNet-KL training on such a task produces:

- **halt = 3 for every input** regardless of true required depth: the architectural floor is *universal* across tasks. Strongest reading: the floor is a property of the architecture and the joint loss, not the data. This makes the §8.1.1 sketch much more powerful.
- **halt scales with the input's required depth**: the floor reading is *LM-pretraining-specific*. Still publishable, sharper claim, refines §8.1.1 to "the floor is set by the per-iteration CE landscape, which is task-dependent."
- **halt = the prior target regardless of input**: the prior dominates and the model can't use depth even when it would help -- a third interpretation that further constrains the framing.

## Task design: nested-XOR expressions of variable depth

Each example is a balanced binary tree of XOR operations over bits, presented as a flat token sequence. The required reasoning depth equals the tree depth.

**Vocabulary** (8 tokens): `0`, `1`, `(`, `)`, `^`, `=`, `<PAD>`, `<ANS>`.

**Generation procedure for depth d:**

```
def gen(d):
    if d == 0:
        return random.choice(['0', '1'])
    left  = gen(d - 1)
    right = gen(d - 1)
    return f'({left} ^ {right})'
```

The full example shown to the model:

```
<expression> = <ANS>
```

The model's target is the single token after `<ANS>`: either `0` or `1`. Loss is cross-entropy on that one token; everything else is conditioning.

**Why this task discriminates the hypotheses:**

- Depth-1 example `(0 ^ 1) = 1` requires evaluating one XOR. A 1-layer transformer can do this in one iteration.
- Depth-2 example `((0 ^ 1) ^ (1 ^ 0)) = 0` requires evaluating two nested XORs. With a recurrent block applied K times, the natural minimum K to compute it is 2 (one iteration per nesting level).
- Depth-d example needs d iterations of "evaluate one XOR" in the worst case.

The prior expects halt at `round(1/lambda_p)` for every input; the task expects halt at iteration `d` for input of depth `d`. The model has access to the depth (indirectly through the parenthesis nesting visible in the input tokens), so a sufficiently expressive halting head *could* learn `halt = d`.

## Training recipe (matches paper)

- **Model:** scaled-down mythos_3b architecture: same prelude/recurrent/coda structure, same MLA + MoE feed-forward, but ~50M parameters (4-head MLA, 64-dim heads, 2-expert MoE, 4 prelude + 4 recurrent + 4 coda layers, 512-dim residual stream). The point is to keep architectural inductive biases similar to the 3.1B paper model so the floor argument carries.
- **Trainer:** `training/toy_depth_task.py` (new file; see implementation skeleton below).
- **Loss:** identical to `training/3b_varT_pondernet_joint.py` -- ACT-weighted CE plus per-iteration Bernoulli-KL against geometric prior at `lambda_p`. `LAMBDA_KL=1.0`, `REINIT_HEAD=0`.
- **Sweep:** three values of `lambda_p`, matching the paper sweep: `0.1` (target halt 10), `0.2` (target 5), `0.5` (target 2). 
- **Data:** on-the-fly generation, balanced across depths `d in {1, 2, 3, 4, 5, 6, 7, 8}`. 50k examples per training "epoch", train for 5M tokens total (a few hours per `lambda_p`).
- **Eval:** held-out 1000 examples per depth, with halt-step histogram and accuracy reported per (input depth, K) cell.

## Evaluation table to produce

For each `lambda_p in {0.1, 0.2, 0.5}`:

| input depth d | mean halt step (model) | accuracy K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|
| 1 | ? | ? | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? | ? | ? |
| 3 | ? | ? | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? | ? | ? |
| 5 | ? | ? | ? | ? | ? | ? |
| 6 | ? | ? | ? | ? | ? | ? |
| 7 | ? | ? | ? | ? | ? | ? |
| 8 | ? | ? | ? | ? | ? | ? |

**Hypothesis 1 (floor universal):** mean halt is approximately constant across `d` at each `lambda_p`, and equal to `max(round(1/lambda_p), 3)`.

**Hypothesis 2 (depth-adaptive):** mean halt scales with `d` up to the prior's cap, e.g. `halt = min(d, round(1/lambda_p))`.

**Hypothesis 3 (prior-only):** mean halt equals `round(1/lambda_p)` for every `d`, no architectural floor at all.

The accuracy columns test whether the model *can* solve depth-d examples at a given K. Even if halt = 3 for all d, accuracy at K=8 should reach ceiling for d=8 if the recurrent block can actually use the depth when forced (ACT-off).

## Implementation skeleton

`training/toy_depth_task.py`:

```python
import os, math, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocab. 8 tokens.
VOCAB = ['0', '1', '(', ')', '^', '=', '<PAD>', '<ANS>']
PAD, ANS = VOCAB.index('<PAD>'), VOCAB.index('<ANS>')
VS = len(VOCAB)

def gen_expr(d):
    """Return (token_ids, answer_id)."""
    if d == 0:
        b = random.randint(0, 1)
        return [b], b
    L, lv = gen_expr(d - 1)
    R, rv = gen_expr(d - 1)
    expr = [VOCAB.index('(')] + L + [VOCAB.index('^')] + R + [VOCAB.index(')')]
    return expr, lv ^ rv

def gen_example(d, max_len):
    expr, ans = gen_expr(d)
    seq = expr + [VOCAB.index('='), ANS]
    seq = seq + [PAD] * (max_len - len(seq))
    return torch.tensor(seq[:max_len]), ans

# -----
# Tiny recurrent-depth model.
# Prelude (4 transformer blocks)
# Recurrent block (1 transformer block, applied T times, with ACT head)
# Coda (4 transformer blocks)
# Output: standard LM head.
# -----

class RecurrentBlock(nn.Module):
    """Single transformer block, plus loop-index embedding, plus ACT head."""
    def __init__(self, d_model=512, n_heads=4, T_max=16):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True, norm_first=True)
        self.loop_emb = nn.Embedding(T_max + 1, d_model)
        self.act_head = nn.Linear(d_model, 1)

    def forward(self, h, t):
        h_in = h + self.loop_emb(torch.tensor(t, device=h.device))
        h_out = self.transformer(h_in)
        p = torch.sigmoid(self.act_head(h_out)).squeeze(-1)  # [B, L]
        return h_out, p

class ToyModel(nn.Module):
    def __init__(self, d_model=512, n_heads=4, n_prelude=4, n_coda=4, T_max=16, vocab_size=VS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.prelude = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True, norm_first=True) for _ in range(n_prelude)])
        self.recurrent = RecurrentBlock(d_model, n_heads, T_max)
        self.coda = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True, norm_first=True) for _ in range(n_coda)])
        self.out_head = nn.Linear(d_model, vocab_size)
        self.T_max = T_max

    def forward(self, x, K):
        """Forward at depth K; returns ACT-weighted output and the per-iteration halting probabilities p_t."""
        h = self.embed(x)
        for blk in self.prelude:
            h = blk(h)
        # Recurrent block applied K times with ACT weighting.
        h_out = torch.zeros_like(h)
        cum_p = torch.zeros(h.shape[:2], device=h.device)
        p_list = []
        for t in range(1, K + 1):
            h, p = self.recurrent(h, t)
            p_list.append(p)
            if t < K:
                w = p * (1 - cum_p).clamp(min=0)
                cum_p = cum_p + w
            else:
                # remainder
                w = (1 - cum_p).clamp(min=0)
            h_out = h_out + w.unsqueeze(-1) * h
        for blk in self.coda:
            h_out = blk(h_out)
        logits = self.out_head(h_out)
        return logits, torch.stack(p_list, dim=0)  # [T, B, L]

# Training loop: variable-T per minibatch, joint ACT-PonderNet-KL loss.
# See 3b_varT_pondernet_joint.py for the reference loss expression to mirror.
```

## Launch plan (when GPU 1 is free)

```
ssh alexm@kebab-rtx6000 'nohup python3 /home/alexm/OpenMythos/training/toy_depth_task.py \
    --lambda_p 0.5 --target_tokens 5000000 --out_dir /tmp/toy_lp05 \
    > /tmp/toy_lp05.log 2>&1 & disown'
```

Run all three `lambda_p` values sequentially (~3-4h per value on the rtx6000 GPU 1 at ~50M params), or in parallel if memory allows. Compare halt-step distributions and accuracy tables against the three hypotheses above.

## Decision tree

- **If hypothesis 1 (floor universal) holds**: the next paper revision becomes much stronger. Floor argument upgraded from "we observed it" to "we observed it AND it reproduces on a task with very different structure". §8.1.1 gains a paragraph; abstract gains a sentence.
- **If hypothesis 2 (depth-adaptive) holds**: §8.1.1 is refined. The floor is set by the per-iteration CE landscape, which is data-dependent. The 3.1B LM result becomes a special case (per-iteration CE on language has a plateau at small N because language modeling is not strictly compositional in input depth).
- **If hypothesis 3 (prior-only) holds**: §8.1.1's floor argument is weakened; the r2.17 observed halt 3.00 vs target 2.0 needs a different explanation (e.g., training-time stochasticity, optimization basin).

In every outcome the paper gains a sharper claim about the mechanism.
