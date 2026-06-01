# Round 2.20: matched-compute non-ACT baseline (Tier 3.5)

## Purpose

Provide a single, quotable number for the absolute cost or benefit of the
ACT mechanism, in the form: "ACT-on FineWeb-Edu held-out CE under joint
PonderNet-KL training at `lambda_p = 0.5` is X nats; the same architecture
and matched compute with no ACT at all (T_FIXED=1) is Y nats; difference
= Y - X."

This is the missing control referenced in §10. With it in hand, the
paper's central positive claim becomes: under joint training, ACT delivers
a measurable improvement of (Y - X) nats over the no-ACT baseline at the
same architecture and same continuation budget.

## Setup

- **CKPT_DIR:** `checkpoints_3b_varT_pondernet_round220`
- **Seed shards:** copy `step_0021362_rank{0..3}.pt` from `round218` (or
  from `round217` if 218's are not yet complete by launch time). Both are
  legitimate starting points -- r2.20's purpose is to measure the non-ACT
  CE floor at this architecture and matched continuation compute, so any
  reasonable shared init works.
- **EXTRA_ENV:**
  `CKPT_DIR=checkpoints_3b_varT_pondernet_round220 TARGET_TOKENS=400000000 REINIT_HEAD=1 T_MAX=1 LAMBDA_KL=0.0 LAMBDA_P=0.5`
  - `T_MAX=1`: recurrent block runs exactly one iteration. No depth, no
    halt selection.
  - `LAMBDA_KL=0.0`: KL term zeroed. With only one iteration the term
    would be vacuous anyway, but explicit zeroing prevents any residual
    contribution.
  - `LAMBDA_P=0.5`: value is unused at T_MAX=1 but is set to the r2.17
    value for cosmetic parity in logs.
  - `REINIT_HEAD=1`: head is meaningless at T_MAX=1; reinit avoids carrying
    over the r2.17 lambda_p=0.5 head.

## Compute

- Cluster: 4 nodes, ~20 hours wall-clock, matched to r2.17's continuation
  budget (50M tokens, step 21362 -> 24414).
- Must run sequentially with r2.18 (not concurrent) on the cluster.
- After r2.20: r2.19 (lambda_p = 0.33) for the second floor data point.

## What we measure

After r2.20 completes at step 24414:

1. **FineWeb-Edu held-out CE** at K=1 (the only natural inference depth).
   This is the headline non-ACT baseline number.
2. **Cross-task evals**: ARC-Easy, ARC-Challenge, HellaSwag, ListOps,
   GSM8K-CoT at limit=200, matching the r2.15/r2.17 evaluation harness.
   These tell us whether non-ACT is dominated, neutral, or superior on
   downstream tasks at this scale.

## Paper integration

A single new subsection §7.26 "Round 2.20: matched-compute non-ACT
baseline." Table:

| Round | mechanism | ACT-on CE @ K=4 | mean halt | downstream summary |
|---|---|---|---|---|
| r2.15 | joint PonderNet-KL, lambda_p=0.2 | 2.8271 | 5.33 | (existing values) |
| r2.17 | joint PonderNet-KL, lambda_p=0.5 | 2.4469 | 3.00 | (existing values) |
| **r2.20** | **T_FIXED=1 (no recurrence)** | **?** | **1 by construction** | **?** |

The single sentence that goes in the §1 contribution list and §10
conclusion:

> "Joint ACT-PonderNet-KL training at `lambda_p = 0.5` delivers a
> (Y - X)-nat improvement over the matched-compute non-ACT baseline at
> the same architecture and same 50M-token continuation budget."

If (Y - X) is positive: ACT is justified. If negative: the paper still
publishes but the framing shifts to "the recurrent block does productive
work *at the architecture's natural ready iteration* but the optimisation
finds a worse local minimum than a simpler trainer would." Either is
useful.

## Open question

Verify before launch that `lambda_p=0.5` with `T_MAX=1` does not trigger
any edge case in the loss code (e.g., zero iterations of the inner
loop). A 5-minute dry-run on rtx6000 GPU 1 before the full cluster
launch is recommended.

The earlier `lambda_p=1.0` `math.log(0)` crash (task #106) has been
fixed: lambda_p is now clamped to `1 - 1e-6` (matching the p_stack
clamp) with a one-time warning. The clamp is unreachable for r2.20's
`lambda_p=0.5`, but the fix means future rounds can use `lambda_p=1.0`
without crashing if a degenerate "halt on step 1" prior is wanted.
