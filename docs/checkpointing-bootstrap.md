# Auto-bootstrap from a consolidated `_full.pt`

OpenMythos #156 (this doc) extends `training/3b_varT_pondernet_joint.py` so
that when a supervisor starts a new round and the per-rank `CKPT_DIR` has
no sharded checkpoints, the trainer can auto-shard-from-full directly off
a single consolidated `*_full.pt`. The operator no longer has to manually
pre-place `step_*_rank{0..3}.pt` shards on every node before launch.

This complements gpufarm PR #157, which adds a VRAM precondition so the
supervisor refuses to start a round when cluster VRAM is too low to hold
the model. Together: #157 catches the "node not ready" case before any
work is done, and #156 catches the "node ready but ckpt not pre-staged"
case automatically.

## When the new path triggers

The dispatch is in `training/bootstrap_dispatch.py::resolve_bootstrap_mode`
and runs on every rank at startup. The three modes:

| Mode              | Condition                                                          | Trainer action                                                      |
| ----------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `resume_shards`   | `step_*_rank{rank}.pt` exists in `CKPT_DIR`                        | Load this rank's shard via `load_checkpoint` (model + optim + step) |
| `bootstrap_full`  | `CKPT_DIR` empty AND `BOOTSTRAP_CKPT` set AND file exists locally  | `bootstrap_model_weights` -> FSDP re-shards in memory, step=0       |
| `fresh_start`     | otherwise                                                          | Legacy round-2.2/round-2.1 auto-discovery, else random init         |

`resume_shards` always wins over `bootstrap_full`, so a crash-restart
during a bootstrapped round will resume from the freshly-written shards
on disk instead of re-broadcasting the 6 GB `_full.pt` across the cluster.

## Operator action: set up a round with a consolidated bootstrap

In the supervisor's `EXTRA_ENV`, just set `CKPT_DIR` to a fresh path and
point `BOOTSTRAP_CKPT` at the consolidated `*_full.pt`. The `_full.pt`
file must be present on each node's local filesystem at that path -- on
the kebab-spark cluster this is satisfied because consolidated ckpts
are rsync'd to all four nodes by `training/cluster_consolidate.sh`.

Example (round 2.21 architectural-floor experiment):

```yaml
EXTRA_ENV: "CKPT_DIR=checkpoints_3b_varT_pondernet_round221 \
            BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_round218/step_0024414_full.pt \
            TARGET_TOKENS=50000000 REINIT_HEAD=0 T_MIN=1 T_MAX=4 \
            LAMBDA_P=1.0 LAMBDA_KL=1.0"
```

No `cp step_0024414_rank{0..3}.pt $CKPT_DIR/` step is needed.

## What the trainer logs when the path triggers

On rank 0 you will see:

```
first-launch bootstrap: CKPT_DIR has no shards;
loading BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_round218/step_0024414_full.pt
at rank 0 and letting FSDP re-shard
```

followed by the existing `Bootstrapping from BOOTSTRAP_CKPT override (...)`
message and (if `REINIT_HEAD=1`) `ACT head re-initialised`.

If `BOOTSTRAP_CKPT` is set but the file does not exist on this rank's
local filesystem, you will instead see:

```
BOOTSTRAP_CKPT=<path> does not exist on this rank's filesystem;
falling back to auto-discovery of round-2.2/round-2.1 finals
```

Treat that warning as a launch-time bug -- it means either the rsync to
this node didn't run or the `BOOTSTRAP_CKPT` path has a typo. Catching
it early is the whole point; without the explicit existence check,
`torch.load` would block all four ranks on a collective that never
completes.

## What this does NOT do

* Sharded-prefix bootstrap (`BOOTSTRAP_CKPT` pointing to
  `step_NNNN_rank` rather than a `_full.pt`) is not supported. If you
  need to continue from another round's sharded ckpts without going
  through consolidation, manually rsync `step_*_rank{rank}.pt` from the
  source round's `CKPT_DIR` into the new round's `CKPT_DIR` on each
  node; the trainer will then take the `resume_shards` path. This is
  the same manual step that was documented in the r2.20 supervisor
  comment.

* The dispatch helper does NOT cross-rank-broadcast existence. Each rank
  inspects its own filesystem independently. If a `_full.pt` exists on
  rank 0 but not on rank 2, the existence checks disagree and the trainer
  will hang on the FSDP collective. The fix in that case is operator-side
  (rsync the file to rank 2), not in the dispatcher.

## Testing

Unit tests for the dispatch decision live at
`tests/test_bootstrap_dispatch.py`. They cover all three modes plus the
precedence rule (shards beat `BOOTSTRAP_CKPT` for crash-restarts) and
the env-var-hygiene cases (`None`, `""`, whitespace, missing file).
Run them with:

```
pytest tests/test_bootstrap_dispatch.py -v
```

The tests use only the stdlib, so they pass in any environment that can
import `pytest` -- torch / CUDA / FSDP are not required.
