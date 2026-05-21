#!/usr/bin/env bash
# NOTE: This script is now subsumed by gpufarm supervisor train-r215
# (see /Users/alex/git/OpenMythos/gpufarm/supervisors.yaml). Kept as a
# manual fallback for ad-hoc reruns; do NOT run concurrently with the
# coordinator's supervisor or both will race to relaunch on hang.
#
# retry_cluster_training.sh
#
# Generic retry wrapper for 4-node FSDP cluster training. Watches the
# rank-0 train log; if no progress for >7 min, declares NCCL hang,
# kills all ranks, and relaunches (training auto-resumes from latest
# sharded ckpt).
#
# Required env vars:
#   ROUND_NAME    short name for logging (e.g. "r213", "r214")
#   SCRIPT        path under repo to training script (e.g. "training/3b_varT_pondernet_joint.py")
#   PORT          torchrun rendezvous port
#   EXTRA_ENV     env-vars for the training script (CKPT_DIR=... BOOTSTRAP_CKPT=... etc)
#   TRAIN_LOG     rank-0 log path (default: /tmp/train_r0.log)
#   STALL_SEC     seconds without progress before declaring hang (default: 420 = 7 min)
#
# Optional env vars:
#   COMPLETE_MARKER  string to look for in TRAIN_LOG to mark training done
#                    (default: "Training complete.")
#
# Exits cleanly when training completes (marker observed in log).

set -uo pipefail

: "${ROUND_NAME:?ROUND_NAME required}"
: "${SCRIPT:?SCRIPT required}"
: "${PORT:?PORT required}"
: "${EXTRA_ENV:?EXTRA_ENV required}"
: "${TRAIN_LOG:=/tmp/train_r0.log}"
: "${STALL_SEC:=420}"
: "${COMPLETE_MARKER:=Training complete.}"
: "${REPO:=/home/alexm/OpenMythos}"

NODES=(kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g)
LOG=/tmp/retry_${ROUND_NAME}.log

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

kill_all_ranks() {
    log "killing all training ranks for $ROUND_NAME"
    for h in "${NODES[@]}"; do
        ssh -q -o ConnectTimeout=5 alexm@"$h" \
            'pkill -9 -f "python3 .*training/3b_varT" 2>/dev/null; pkill -9 -f "python3 .*torch.distributed.run" 2>/dev/null; true' \
            || true
    done
    sleep 10
}

any_rank_running() {
    for h in "${NODES[@]}"; do
        # Match the actual python training process (cmdline starts with
        # /usr/bin/python3 then later contains training/3b_varT). This anchored
        # pattern avoids matching bash/ssh wrappers from other queues running
        # on the same host that include "python3 training/..." as a substring.
        ssh -q -o ConnectTimeout=5 alexm@"$h" \
            'pgrep -f "^/usr/bin/python3 .*training/3b_varT" >/dev/null 2>&1' && return 0
    done
    return 1
}

launch() {
    log "launching $ROUND_NAME training (auto-resume from latest sharded ckpt)"
    cd "$REPO"
    SCRIPT="$SCRIPT" PORT="$PORT" EXTRA_ENV="$EXTRA_ENV" \
        bash training/launch_3b.sh 2>&1 | tee -a "$LOG"
    log "launch_3b.sh returned"
}

train_log_age() {
    # Return seconds since last modification of the rank-0 log on its native host.
    # The log lives on kebab-spark-200g (rank 0). We're typically running this
    # script on spark.lan which is the same machine, so /tmp/train_r0.log is
    # directly accessible.
    local mod
    mod=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0)
    if [ "$mod" = "0" ]; then
        # Log file gone or unreadable; treat as fresh (don't kill on missing log)
        echo 0
        return
    fi
    echo $(( $(date +%s) - mod ))
}

log "retry_$ROUND_NAME started; watching $TRAIN_LOG for '$COMPLETE_MARKER'"

while true; do
    if grep -q "$COMPLETE_MARKER" "$TRAIN_LOG" 2>/dev/null; then
        log "$COMPLETE_MARKER detected; exiting retry loop"
        break
    fi

    if any_rank_running; then
        age=$(train_log_age)
        if [ "$age" -gt "$STALL_SEC" ]; then
            log "no log progress for ${age}s (>${STALL_SEC}s); assuming NCCL hang"
            kill_all_ranks
            launch
        else
            sleep 30
        fi
    else
        log "no training procs running; relaunching"
        launch
        sleep 60
    fi
done

log "retry_$ROUND_NAME done"
