#!/usr/bin/env bash
# Fires after auto_eval_round29 pipeline completes.
# Single-GPU inference on kebab-spark.lan (no distributed needed):
#   1. reasoning_eval.py on r2.6 and r2.7  (K in {4,8,16,32,64} by default)
#   2. depth_extrap.py at K in {4,8,16,32,64} on r2.3/r2.4/r2.6
# Then queues r2.10 joint continuation.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

LOG=/tmp/post_r29_evals.log
R29_LOG=/home/alexm/OpenMythos/training/auto_eval_round29.log
DOCS=/home/alexm/OpenMythos/docs
REPO=/home/alexm/OpenMythos
PY=python3

CKPT_R23=$REPO/checkpoints_3b_varT_pondernet_joint/step_0003051_full.pt
CKPT_R24=$REPO/checkpoints_3b_varT_pondernet_from_r2/step_0003051_full.pt
CKPT_R26=$REPO/checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt
CKPT_R27=$REPO/checkpoints_3b_varT_pondernet_round27_lp01/step_0003051_full.pt

log "post_r29_evals started; waiting for auto_eval_round29 pipeline complete"
DEADLINE=$(($(date +%s) + 12 * 3600))
while true; do
    grep -q "auto_eval_round29 pipeline complete" "$R29_LOG" 2>/dev/null && { log "r29 pipeline done"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 12h deadline"; exit 1; }
    sleep 60
done

cd "$REPO"

run_one() {
    local label=$1 script=$2 out=$3 ckpt=$4
    shift 4
    [ ! -f "$ckpt" ] && { log "WARN: $label ckpt missing; skipping"; return; }
    log "--- $label -> $out"
    env CKPT="$ckpt" OUT="$out" "$@" "$PY" "training/$script" 2>&1 | tee -a "$LOG"
    log "--- $label done"
}

log "=== reasoning evals (r2.6 and r2.7) ==="
run_one r26_reasoning reasoning_eval.py \
    "$DOCS/reasoning_eval_round26.json" "$CKPT_R26"
run_one r27_reasoning reasoning_eval.py \
    "$DOCS/reasoning_eval_round27.json" "$CKPT_R27"

log "=== depth extrap at K up to 64 (r2.3, r2.4, r2.6) ==="
run_one r23_depthk64 depth_extrap.py \
    "$DOCS/depth_extrap_round23_k64.json" "$CKPT_R23" DEPTHS=4,8,16,32,64
run_one r24_depthk64 depth_extrap.py \
    "$DOCS/depth_extrap_round24_k64.json" "$CKPT_R24" DEPTHS=4,8,16,32,64
run_one r26_depthk64 depth_extrap.py \
    "$DOCS/depth_extrap_round26_k64.json" "$CKPT_R26" DEPTHS=4,8,16,32,64

log "=== post_r29_evals complete; launching queue_r210 ==="
nohup bash "$REPO/training/queue_r210.sh" \
    >>/tmp/queue_r210.log 2>&1 </dev/null &
disown $!
log "queue_r210 PID=$!"
