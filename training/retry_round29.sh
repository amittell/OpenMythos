#!/usr/bin/env bash
# Retry wrapper for round 2.9 training.
# Loops until "Training complete." appears in /tmp/train_r0.log.
# If training is running but the log hasn't advanced in 7 minutes, kills all
# ranks and relaunches. Training auto-resumes from the latest sharded ckpt.
#
# Run once in background; the auto_eval_round29 watcher handles post-training.

set -uo pipefail

REPO=/home/alexm/OpenMythos
LOG=/tmp/retry_round29.log
TRAIN_LOG=/tmp/train_r0.log
NODES=(kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

kill_all_ranks() {
    log "killing all training ranks"
    for h in "${NODES[@]}"; do
        ssh -q -o ConnectTimeout=5 alexm@"$h" \
            'pkill -9 -f 3b_varT_act_v3 2>/dev/null; pkill -9 -f torchrun 2>/dev/null' \
            || true
    done
    sleep 10
}

any_rank_running() {
    for h in "${NODES[@]}"; do
        ssh -q -o ConnectTimeout=5 alexm@"$h" \
            'pgrep -f 3b_varT_act_v3 >/dev/null 2>&1' && return 0
    done
    return 1
}

launch() {
    log "launching training (will auto-resume from latest checkpoint)"
    cd "$REPO"
    SCRIPT=training/3b_varT_act_v3.py \
        EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_act_v3_round29_T1 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt T_FIXED=1" \
        bash training/launch_3b.sh 2>&1 | tee -a "$LOG"
    log "launch returned"
}

log "retry_round29 started"

while true; do
    if grep -q 'Training complete\.' "$TRAIN_LOG" 2>/dev/null; then
        log "Training complete detected; exiting retry loop"
        break
    fi

    if any_rank_running; then
        LAST_MOD=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0)
        AGE=$(( $(date +%s) - LAST_MOD ))
        if [ "$AGE" -gt 420 ]; then
            log "No log progress for ${AGE}s (>7min); assuming NCCL hang"
            kill_all_ranks
            launch
        else
            sleep 30
        fi
    else
        log "No training ranks running; relaunching"
        launch
        sleep 60
    fi
done

log "retry_round29 done"
