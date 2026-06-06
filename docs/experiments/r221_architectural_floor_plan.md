# Round 2.21 -- Architectural-floor discriminator (T_MAX ablation)

The paper §9 nominates "is the floor N architecture-dependent or training-distribution-dependent?" as the most leveraged open question. The four-point sweep (§7.26) established N = 2 for the canonical mythos_3b architecture trained with `T_MIN=2, T_MAX=12`. The two readings consistent with that:

1. **Architectural floor.** N = 2 because the model only produces a meaningful intermediate representation after at least one full recurrent pass. The floor is set by the recurrent block's inductive biases (depth, MLA rank, MoE capacity, LTI eigenstructure) and is independent of training-time depth sampling.

2. **Training-distribution floor.** N = 2 is a residual property of training with `T_MIN=2`. The model never saw `T = 1` during training, so it never learned that iteration 1's representation could be useful, so at inference the head can only commit to halt steps that the training distribution exposed.

These two readings make incompatible predictions when we change the training-time `T` distribution. r2.21 is the discriminator.

## Setup

- Bootstrap: r2.18 step 24414 full ckpt (`checkpoints_3b_varT_pondernet_round218/step_0024414_full.pt`). Continues from the λ_p=1.0 floor-pinned model.
- Training script: same `training/3b_varT_pondernet_joint.py` as r2.15-r2.18, with one one-line edit (now landed) that exposes `T_MIN` and `T_MAX` as environment variables.
- Continuation budget: 50M tokens (same as the r2.17/r2.18 deltas), `TARGET_TOKENS=50000000`.
- Halt prior: `LAMBDA_P=1.0` (same as r2.18). This pins the prior at "halt at iteration 1, the floor disagrees".
- KL weight: `LAMBDA_KL=1.0` (default).
- Architectural variable: `T_MIN=1, T_MAX=4`. The lower bound drops from 2 → 1 so the model is now exposed to iteration-1 representations during training. The upper bound drops from 12 → 4 so the per-step compute roughly matches r2.18's effective depth (the halt distribution sits near 2 with λ_p=1.0, so depths above 4 contribute little).
- Head reinit: `REINIT_HEAD=0`. We want to continue the r2.18 halt head's state rather than restart it; the question is whether *additional training* with the new T distribution moves the floor.

## Predictions

**If the architectural-floor reading is correct (N is structural):**
- Mean halt step stays near 2.0 (≥ 1.9), top step stays at 2.
- ACT-on FineWeb CE stays near r2.18's 2.42 nats (small drift from the additional 50M tokens is fine).
- Halt entropy stays ≤ 0.2 bits.

**If the training-distribution reading is correct (N is a function of `T_MIN`):**
- Mean halt step drops to ≈ 1.0 (the prior's target).
- A non-trivial fraction (>10%) of tokens halt at iteration 1.
- ACT-on FineWeb CE rises modestly (the model has to commit to a worse `h_1` representation).

A halt step that lands between 1.5 and 1.9 would be ambiguous and demands a tie-breaker (e.g., extend to `T_MIN=1, T_MAX=2` with a small-budget continuation).

## Implementation

Done in-tree on this branch:
- `training/3b_varT_pondernet_joint.py:754-755`: replaced the hard-coded `T_MIN=2, T_MAX=12` with `os.environ.get(...)` lookups defaulting to the original values. Backward-compat: omitting the env vars reproduces r2.15-r2.18 exactly.
- `gpufarm/supervisors.yaml`: new `train-r221` supervisor entry continuing from r2.18's step_0024414 with `EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_round221 TARGET_TOKENS=50000000 BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_round218/step_0024414_full.pt T_MIN=1 T_MAX=4 LAMBDA_P=1.0 LAMBDA_KL=1.0 REINIT_HEAD=0"`.
- `gpufarm/rounds.yaml`: new `r221` round entry.

## Expected walltime

The r2.17 → r2.18 +50M-token delta took ~9h cluster walltime (4-node DGX Spark, NCCL over RoCE bond0). r2.21 should match. Eval bundle (depth_extrap_k64, per_token_halt, reasoning_eval, etc.) is the same as r2.18's, ~30 minutes after consolidation.

## Paper integration

§7.27 (new): "Training-distribution ablation: does the floor move when T_MIN drops to 1?" Three outcomes possible:
- N stays at 2 → §7.26's "floor at 2 is architectural" reading is confirmed.
- N drops to 1 → §7.26 reframes: the floor is a function of training-time depth distribution, not the architecture. Paper's structural claim weakens but the cap-knob claim survives (it's still tunable).
- N moves to an intermediate value → reports as nuanced + opens follow-up.
