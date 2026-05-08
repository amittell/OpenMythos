# Federated Training: Cluster + RTX6000

Train a single model across the GB10 Spark cluster (4 nodes, FSDP) AND the
RTX6000 Blackwell box (1-2 GPUs) simultaneously, using federated weight
averaging instead of cross-machine NCCL.

## Why federated averaging instead of cross-machine FSDP

Cross-machine FSDP would require NCCL collectives spanning the cluster's
200G IB/storage net AND the RTX6000's regular LAN. The slowest link
dominates collective time, which would erase the Blackwell's compute
advantage. Federated averaging avoids cross-machine collectives entirely:
each side runs an independent local training job, weights are exchanged
periodically through a coordinator that pulls full state dicts via rsync.

## Architecture

```
+--------------------+          +-----------------------+
| Spark cluster      |          | RTX6000               |
| 4-node FSDP        |          | single or 2-GPU FSDP  |
| trainer_cluster.py |          | trainer_rtx.py        |
+----------+---------+          +-----------+-----------+
           |                                |
           |   ready/state/loaded markers   |
           |          via filesystem        |
           |                                |
           v                                v
        +-----------------------------------+
        | Coordinator (runs on spark.lan)   |
        | coordinator.py                    |
        | - Pulls state dicts (rsync)       |
        | - Validates (no NaN/Inf)          |
        | - Token-weighted average          |
        | - Pushes avg back to both         |
        | - Crash-resumable via on-disk     |
        |   round state                     |
        +-----------------------------------+
```

## Sync protocol

Each "round" of sync:

1. **Trainer (each side, after `SYNC_TOKEN_INTERVAL` tokens processed):**
   - Save full model state dict to `<FED_DIR>/round_NNNN/<role>_state.pt.tmp`
   - Atomic rename to `<role>_state.pt`
   - Write metadata `<role>_meta.json` with `tokens_since_last_sync`, `step`, `loss_ema`
   - Touch `<role>_ready` marker
   - Block until `avg_state.pt` exists in the round directory
   - Load and broadcast averaged state, replacing local model weights
   - Touch `<role>_loaded` ack marker
   - Continue training (with state dict from avg)

2. **Coordinator (polls every 10s):**
   - Identify current round number from on-disk state
   - Wait until both `cluster_ready` and `rtx_ready` exist
   - Pull both state dicts to coordinator's local working dir (rsync if remote)
   - Validate: no NaN or Inf in any tensor
   - Token-weighted average: `avg = (w_c * S_c + w_r * S_r) / (w_c + w_r)`
   - Write `avg_state.pt.tmp` then atomic rename
   - rsync avg back to both sides into the round dir
   - Wait for both `loaded` markers (ack)
   - Mark round complete, increment round number

## Failure modes and recovery

| Failure | Detection | Recovery |
|--------|-----------|----------|
| One trainer crashes mid-step | Other trainer hits sync timeout | Coordinator declares missing side `dead`, proceeds with single-side update (degraded mode); restarted side rejoins on its next sync |
| Coordinator crashes pre-average | Both trainers waiting | On restart, coordinator scans round dir, resumes from current state (`AVERAGING` if both ready, `DISTRIBUTING` if avg exists) |
| Coordinator crashes mid-rsync | One side has avg, other doesn't | On restart, idempotent re-rsync to whichever side lacks the file |
| NaN/Inf in state dict | Validation in coordinator | Abort round; coordinator alerts via log; trainers resume from last good state on next sync attempt |
| Disk full | rsync error | Coordinator logs error, retries; if persistent, halts both sides via `<FED_DIR>/halt` flag |
| Sync timeout (one side never ready) | 60 min wait | Declare dead. Continue with present side as solo update. Allows hardware failures without aborting full run. |

## Resumability

Coordinator state is persisted in `<FED_DIR>/coord.state.json`:

```json
{
  "round": 47,
  "phase": "WAITING",
  "last_sync_unix": 1715201234,
  "cluster_dead": false,
  "rtx_dead": false,
  "history": [...],
}
```

On any process restart (trainer or coordinator), the system picks up
mid-round. Markers are filesystem-atomic so partial state is detectable.

## Token weighting

Both sides may process different numbers of tokens per sync interval if
their step rates differ. Equal-weight averaging would pull the faster
side back (wasting its progress). Use token-weighted average instead:

```
w_c = cluster_meta["tokens_since_last_sync"]
w_r = rtx_meta["tokens_since_last_sync"]
avg[k] = (w_c * cluster_state[k] + w_r * rtx_state[k]) / (w_c + w_r)
```

Equivalent to FedAvg with sample-count weighting.

## Data sharding

Both sides read the same dataset (FineWeb-Edu) but with different RNG
seeds (e.g., cluster seed=42, RTX seed=43) so they see disjoint shuffled
slices. Without disjoint data, federated averaging has no statistical
benefit over solo training.

## Sync interval choice

Trade-off:
- More frequent sync = closer to true SGD, more overhead, more network bandwidth
- Less frequent sync = each side drifts further apart, FedAvg accuracy degrades

Default: `SYNC_TOKEN_INTERVAL = 4_000_000` tokens (every ~4M tokens).
At cluster's 16k tokens/step that's 250 steps; at RTX6000's 4k tokens/step
that's 1000 steps. Sync wall time is ~1-3 min (rsync + average + rsync).
Overhead: ~2-5% of training time.

## Files

- `coordinator.py` - main orchestrator (Python)
- `sync_hook.py` - utility called from trainer scripts at sync points
- `launch_federation.sh` - master launch script (kicks off both trainers + coordinator)
- `patch_trainer.md` - notes on minimal trainer-script changes
- `validate.sh` - dry-run validator (no actual training)

## Throughput estimate

| Configuration | Per-sec tokens | 2B-token wall time |
|---------------|---------------|--------------------|
| Cluster alone | ~580 tok/s | ~40 days |
| RTX6000 alone | ~1300 tok/s | ~18 days |
| Federated (both, 4M-token sync) | ~1700 tok/s | ~14 days |

The combined throughput is sub-additive due to sync overhead and
straggler effects but still ~3x cluster alone.
