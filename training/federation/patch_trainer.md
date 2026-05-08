# Trainer-script patches for federation

The federation hook is intentionally minimal: ~10 lines added to each
trainer script. Both `3b_varT_act_v3.py` and `3b_varT_pondernet_joint.py`
have already been patched in this repo. This file documents what was
added so the same pattern can be applied to other trainers if needed.

## Patch 1: import the hook

Add at the bottom of the import block:

```python
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "federation"))
try:
    from federation import sync_hook as _fed_sync_hook
except ImportError:
    import sync_hook as _fed_sync_hook
```

The dual-import handles being run from either `training/` or
`training/federation/`.

## Patch 2: configure after dist init

Inside `main()`, immediately after `dist.init_process_group(...)` and
the rank/world_size variables are set:

```python
if _fed_sync_hook.configure():
    if master:
        logger.info(f"[fed] federated training enabled (role={os.environ.get('FED_ROLE')}, "
                    f"interval={os.environ.get('FED_SYNC_INTERVAL_SEC', '1800')}s)")
```

`configure()` reads env vars and returns False (no-op) if `FED_SYNC_DIR`
isn't set. Trainers run identically to today's behavior unless federation
env vars are present.

## Patch 3: hook into the training loop

In the per-step loop, AFTER `optimizer.step()` and AFTER the regular
checkpoint-save block, add:

```python
# Federated sync hook (no-op unless FED_SYNC_DIR is set)
_fed_sync_hook.add_tokens(global_batch_tok)
_fed_sync_hook.update_loss_ema(loss_accum)
_fed_sync_hook.maybe_sync(model, step)
```

`maybe_sync()` returns immediately unless wall-clock since last sync
exceeds `FED_SYNC_INTERVAL_SEC`. When a sync fires it:
1. Gathers full state dict from FSDP, writes to disk
2. Touches a `<role>_ready` marker
3. Blocks all ranks until the coordinator writes `avg_state.pt`
4. Loads the averaged state back into FSDP
5. Touches `<role>_loaded` ack
6. Returns; trainer continues with new weights

The function is FSDP-aware and broadcasts state across ranks correctly.
On RTX6000 it also pushes/pulls files via rsync to/from spark.

## Why this design

- **Minimal trainer changes**: 3 short additions, all gated on env vars.
  Default trainer behavior is unchanged when `FED_SYNC_DIR` is unset.
- **Full state dict, not deltas**: averaging deltas would require
  per-tensor history; full state averaging is simpler and well-studied
  (FedAvg).
- **Token-weighted average**: the coordinator weights each side's
  contribution by `tokens_since_last_sync` so a faster trainer doesn't
  get pulled back by a slower one.
- **Filesystem-based protocol**: no NCCL or RPC across machines; works
  through normal SSH+rsync. Robust to transient network failures.
- **All ranks barrier on sync**: ensures FSDP shards are coherent before
  and after weight load. Without this, partial state could leak.
