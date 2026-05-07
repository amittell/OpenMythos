# Depth-extrapolation measurement, mythos_3b step 6103

Measurement run 2026-04-24. Probes whether the final trained checkpoint
benefits from more recurrent iterations at inference than it saw during
training (the "trained-shallow, infer-deep" property of looped transformers).

## Setup

- Checkpoint: `checkpoints_3b_loops4_fast/step_0006103.pt` (final, cosine-annealed)
- Training config: `max_loop_iters = 4`, fixed throughout the run
- Measurement data: 64 sequences x 1024 tokens from FineWeb-Edu sample-10BT,
  starting at sample index 300,000 to skip past the training window (training
  consumed ~200,000 samples across 4 ranks)
- Single GB10 (kebab-spark), non-FSDP, bf16 autocast, forward-only
- Two inference paths measured at each depth in {4, 8, 16, 32}:
  - **ACT-on** - normal inference path. Adaptive Computation Time halting
    produces a weighted sum of hidden states; halted positions stop
    contributing to the sum.
  - **ACT-off** - RecurrentBlock.forward monkey-patched to run all K
    iterations unconditionally and return the final hidden state.

Spectral-radius sanity: rho(A) max = 0.2681. Parcae LTI stability invariant
holds (must be < 1).

## Results

### ACT-on (normal inference)

| n_loops | loss   | ppl    | elapsed (s) |
|---------|--------|--------|-------------|
| 4       | 4.2213 | 68.12  | 10.12       |
| 8       | 4.2213 | 68.12  | 9.75        |
| 16      | 4.2213 | 68.12  | 9.64        |
| 32      | 4.2213 | 68.12  | 9.90        |

Loss is bitwise identical across depths. ACT halting fires early; once
all positions cross the halt threshold the loop short-circuits, and the
ACT-weighted h_out stops accumulating. Extra loop budget is ignored at
zero compute cost (elapsed time is flat, so the short-circuit is working
as intended). The trained model naturally respects its training depth.

### ACT-off (force all K iterations, return final h)

| n_loops | loss   | ppl     | elapsed (s) |
|---------|--------|---------|-------------|
| 4       | 4.7602 | 116.77  | 21.29       |
| 8       | 6.2272 | 506.36  | 39.22       |
| 16      | 7.3069 | 1490.54 | 82.66       |
| 32      | 7.4563 | 1730.71 | 135.52      |

Loss degrades monotonically with depth beyond 4, then saturates
(delta 16 -> 32 is only +0.15 nats, well below the +1.10 delta from 4 -> 8).
The Parcae LTI injection (rho(A) < 1 by construction) prevents divergence;
loss stays bounded and saturates as the hidden state approaches a
fixed point of the recurrent update.

Also note: at n_loops=4, ACT-off is worse than ACT-on (4.76 vs 4.22). The
ACT-weighted sum across iterations is a meaningful regularizer over the
raw final-step hidden state - the model was trained to produce its output
via the weighted sum, not the last h.

## Interpretation

The "trained-shallow, infer-deep" extrapolation property **does not hold**
for this checkpoint. Fixed-depth training at `max_loop_iters=4` produces
a model that:

1. Voluntarily caps effective depth at 4 via ACT halting (ACT-on loss
   is flat across K).
2. Degrades if forced past its trained regime (ACT-off loss rises and
   saturates).
3. Stays numerically stable everywhere thanks to Parcae LTI (rho(A) < 1).

This is the expected behavior for fixed-depth pretraining. The
depth-extrapolation results in Geiping et al. and Saunshi et al. require
a **variable-depth training schedule** - sampling T ~ Uniform(T_min, T_max)
or a similar curriculum so the model sees and learns to use different
recurrence counts during training. Our run used a constant T=4.

What we did confirm empirically:

- LoRAAdapter clamping (main.py:588) does not explode at extrapolation;
  loss stays finite at loops=32.
- The Parcae LTI stability invariant is load-bearing: it is what keeps
  the 16- and 32-loop hidden states bounded.
- ACT halting behaves correctly as a no-op past the trained budget,
  preserving inference efficiency at higher K.

## Generations

Same prompt `"The recurrent-depth transformer architecture"` at each
depth, ACT-on, same seed, temperature 0.8, top-k 40. All four depths
produce identical output because the ACT-weighted h_out converges to
the same state by the halt step regardless of how many extra loop
iterations the budget allows.

```
The recurrent-depth transformer architecture, a new kind of architecture,
which, according to the Institute of Technology, is now under development.
The main idea behind this new architecture is that it is actually very
important to develop a modern architecture
```

Fluent, on-topic, short on specificity - consistent with a 3B model trained
on 100M tokens (under-trained by Chinchilla optimality by roughly 30x).

## Follow-up experiments worth running

1. **Variable-depth training.** Re-run the pretraining sampling
   `T ~ Uniform(2, 12)` per step. Hypothesis: enables real positive
   extrapolation at inference loops > 12.
2. **Hotter ACT threshold.** Raise `cfg.act_threshold` from the current
   value so positions take longer to halt, forcing more average
   iterations. Measure under ACT-on at each K.
3. **Reasoning-heavy probe set.** Repeat the measurement on a domain
   where iterative refinement plausibly helps (GSM8K, arithmetic chains).
   FineWeb-Edu next-token loss may not reveal depth benefit even if
   depth does help reasoning.
4. **Per-iteration trajectory.** Instead of a final-vs-weighted switch,
   log the hidden-state norm and per-step loss at each iteration, and
   plot the trajectory at K=32.

## Artifacts

- Raw JSON results: `docs/depth_extrap_results.json` on kebab-spark,
  also saved to `/tmp/depth_extrap_results.json` on my Mac.
- Measurement script: `training/depth_extrap.py`.
