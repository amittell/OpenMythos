#!/usr/bin/env bash
# Continuous watcher: poll the r2.15 checkpoint dir on kebab-spark, run
# intermediate_eval_r215.sh for every new step that lands.
#
# Run from anywhere with SSH access to the fleet. Designed to be idempotent
# and crash-safe: re-runs at any time pick up wherever they left off because
# intermediate_eval_r215.sh skips steps whose output JSONs already exist.
#
# Usage:
#   bash training/intermediate_eval_r215_watcher.sh                # default 60s poll
#   POLL_SEC=120 bash training/intermediate_eval_r215_watcher.sh   # slower poll
#   START_STEP=5000 bash training/intermediate_eval_r215_watcher.sh # skip earlier steps
#   EVALS="per_token_halt_analysis depth_extrap" bash ./intermediate_eval_r215_watcher.sh
#
# Env vars:
#   POLL_SEC      seconds between polls (default 60)
#   START_STEP    skip steps below this (default 0 -- process everything)
#   EVALS         passed through to intermediate_eval_r215.sh (default per_token_halt only)
#   MIN_STEP_GAP  only run eval every N steps (default 200 -- matches save cadence)

set -euo pipefail

# ---- Singleton guard ----------------------------------------------------
# Only ONE watcher may ever run fleet-wide. flock holds an exclusive advisory
# lock for the life of this process and releases automatically on ANY exit
# (clean, crash, or SIGKILL), so a systemd restart always re-acquires cleanly
# while a stray second launch (manual run, or a Restart-vs-pkill respawn race)
# exits immediately instead of dogpiling the dispatch loop. This is the primary
# fix for the 2026-05-20 eval storm: multiple concurrent watchers each
# dispatching evals onto the same GPU. exit 0 (not non-zero) so systemd's
# Restart=on-failure does NOT treat losing the race as a crash to respawn.
WATCHER_LOCK=${WATCHER_LOCK:-/tmp/r215_eval_watcher.lock}
exec 200>"$WATCHER_LOCK"
if ! flock -n 200; then
    echo "[$(date '+%H:%M:%S')] [watcher] another watcher already holds $WATCHER_LOCK; exiting" >&2
    exit 0
fi

POLL_SEC=${POLL_SEC:-60}
START_STEP=${START_STEP:-0}
EVALS=${EVALS:-per_token_halt_analysis}
MIN_STEP_GAP=${MIN_STEP_GAP:-200}
# Hard cap per dispatched cycle. A hung consolidator / rsync / eval would
# otherwise block all subsequent ticks. Default 1500s = 25 min comfortably
# covers stream-consolidate (~2 min) + ship (~30s) + 3 evals (~5+5+1.5 min)
# + vision cycle (~10s) with margin.
CYCLE_TIMEOUT_SEC=${CYCLE_TIMEOUT_SEC:-1500}

REPO_SPARK=/home/alexm/OpenMythos
CKPT_DIR=$REPO_SPARK/checkpoints_3b_varT_pondernet_round215
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_SCRIPT=$SCRIPT_DIR/intermediate_eval_r215.sh

log() { echo "[$(date '+%H:%M:%S')] [watcher] $*"; }

log "starting; poll=${POLL_SEC}s start_step=$START_STEP min_step_gap=$MIN_STEP_GAP evals=[$EVALS]"
log "eval-script=$EVAL_SCRIPT"

# Track which steps we've already kicked off this run (in-memory; rely on the
# downstream script's idempotency for crash recovery across restarts).
declare -A SEEN

# gx10 hostnames -- training cleans up old shards aggressively (keeps ~2 saves),
# so the watcher must verify each step is on ALL 4 nodes before dispatching.
GX10_HOSTS=(kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g)

# Is a given step still present on all 3 gx10 nodes? (spark rank-0 already
# verified by the step enumeration.) Returns 0 if yes, 1 if any rank missing.
shards_complete() {
    local step="$1"
    local padded
    padded=$(printf "%07d" "$step")
    local idx=1
    for h in "${GX10_HOSTS[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes -q alexm@"$h" \
                "test -f $CKPT_DIR/step_${padded}_rank${idx}.pt" 2>/dev/null; then
            return 1
        fi
        idx=$((idx + 1))
    done
    return 0
}

# Health pre-flight: confirm rtx6000 can actually run an eval before we
# stage 16 GB of weights to it. A wedged driver (Xid / GSP fault / OOM
# pending reboot) will cause nvidia-smi to return non-zero or empty; the
# 2026-05-20 cluster outage hit exactly that state and the prior watcher
# kept dispatching cycles into the void. Cheap (one ssh + one nvidia-smi).
rtx_gpu_ok() {
    local out
    out=$(ssh -o ConnectTimeout=5 -o BatchMode=yes -q alexm@kebab-rtx6000.lan \
            'nvidia-smi -L 2>/dev/null' 2>/dev/null)
    # Expect at least 2 lines starting with "GPU " (both Blackwell GPUs).
    [[ $(echo "$out" | grep -c "^GPU ") -ge 2 ]]
}

while true; do
    if ! rtx_gpu_ok; then
        log "rtx6000 not healthy (nvidia-smi failing or short); skipping dispatch this tick"
        sleep "$POLL_SEC"
        continue
    fi

    # Enumerate available checkpoints on kebab-spark (just rank-0 is enough; the
    # eval script handles the gather + rsync). Sort REVERSE so we process the
    # newest step first -- gx10 rank shards age out fast, so the latest step
    # is the most likely to still be intact across all 4 nodes.
    steps=$(ssh -q alexm@kebab-spark.lan "
        ls $CKPT_DIR/step_*_rank0.pt 2>/dev/null \
            | sed -n 's|.*step_0*\([0-9]\+\)_rank0\.pt|\1|p' \
            | sort -rn
    " || true)

    for step in $steps; do
        if [[ "$step" -lt "$START_STEP" ]]; then
            continue
        fi
        # Step-gap filter (200 by default = save cadence)
        if (( step % MIN_STEP_GAP != 0 )); then
            continue
        fi
        if [[ -n "${SEEN[$step]:-}" ]]; then
            continue
        fi

        # Skip the step if the gx10 rank shards have already been cleaned up
        # by training (we can't consolidate without all 4).
        if ! shards_complete "$step"; then
            log "skip step=$step: rank 1/2/3 already cleaned from gx10 nodes"
            SEEN[$step]=1
            continue
        fi
        SEEN[$step]=1

        log "dispatch step=$step (timeout ${CYCLE_TIMEOUT_SEC}s)"
        # Sequential dispatch -- the underlying script already handles concurrency
        # (rsync in parallel, evals on rtx6000 GPU 1). Running two eval bundles at
        # once would fight for GPU 1 and break the vision lifecycle dance.
        # `timeout` kills the pipeline if a consolidator/rsync/eval hangs so the
        # next watcher tick can move on to a later step. --kill-after=300 (not 60):
        # on SIGTERM the eval script's EXIT trap reloads vision via the router
        # (curl --max-time 240); the grace window must exceed that so a timed-out
        # cycle still restores vision before SIGKILL (else router stays degraded).
        if timeout --signal=TERM --kill-after=300 "$CYCLE_TIMEOUT_SEC" \
            env EVALS="$EVALS" bash "$EVAL_SCRIPT" "$step"; then
            log "step=$step done"
        else
            rc=$?
            if [[ "$rc" -eq 124 ]] || [[ "$rc" -eq 137 ]]; then
                log "step=$step TIMED OUT after ${CYCLE_TIMEOUT_SEC}s (rc=$rc) -- continuing"
            elif [[ "$rc" -eq 3 ]]; then
                # rc=3 = eval script couldn't get the GPU-1 lock (another eval
                # owns it). Transient, not a failure: clear SEEN so we retry on
                # the next poll instead of skipping this step forever.
                log "step=$step skipped (GPU 1 busy) -- will retry next tick"
                unset 'SEEN[$step]'
            else
                log "step=$step FAILED (rc=$rc) -- continuing"
            fi
        fi
    done

    sleep "$POLL_SEC"
done
