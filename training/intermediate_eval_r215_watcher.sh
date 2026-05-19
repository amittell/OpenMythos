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

POLL_SEC=${POLL_SEC:-60}
START_STEP=${START_STEP:-0}
EVALS=${EVALS:-per_token_halt_analysis}
MIN_STEP_GAP=${MIN_STEP_GAP:-200}

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

while true; do
    # Enumerate available checkpoints on kebab-spark (just rank-0 is enough; the
    # eval script handles the gather + rsync).
    steps=$(ssh -q alexm@kebab-spark.lan "
        ls $CKPT_DIR/step_*_rank0.pt 2>/dev/null \
            | sed -n 's|.*step_0*\([0-9]\+\)_rank0\.pt|\1|p' \
            | sort -n
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
        SEEN[$step]=1

        log "dispatch step=$step"
        # Sequential dispatch -- the underlying script already handles concurrency
        # (rsync in parallel, evals on rtx6000 GPU 1). Running two eval bundles at
        # once would fight for GPU 1 and break the vision lifecycle dance.
        EVALS="$EVALS" bash "$EVAL_SCRIPT" "$step" || log "step=$step FAILED (continuing)"
    done

    sleep "$POLL_SEC"
done
