#!/usr/bin/env bash
# RTX6000 GPU 1: capability evals on r2.6 and r2.7 after current cap evals finish.
# Waits for the r2.3/r2.4/r2.5 capability eval chain to complete, then
# rsyncs r2.6 and r2.7 checkpoints from spark and runs ListOps + GSM8K.

set -uo pipefail
LOG=/tmp/queue_post_capevals_gpu1.log
PY=/home/alexm/venvs/vllm-turboquant/bin/python3
REPO=/home/alexm/OpenMythos
DOCS=$REPO/docs
SPARK=alexm@kebab-spark.lan
CAP_LOG=/tmp/capability_evals_rtx6000.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "post_capevals_gpu1 started; waiting for current capability evals to finish"
DEADLINE=$(($(date +%s) + 24 * 3600))
while true; do
    grep -q "capability evals DONE" "$CAP_LOG" 2>/dev/null && { log "cap evals done"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 24h deadline"; exit 1; }
    sleep 120
done

log "rsyncing r2.6 and r2.7 checkpoints from spark"
mkdir -p "$REPO/checkpoints"
CKPT_R26=$REPO/checkpoints/ckpt_r26_full.pt
CKPT_R27=$REPO/checkpoints/ckpt_r27_full.pt

[ ! -f "$CKPT_R26" ] && rsync -a \
    "$SPARK:$REPO/checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt" \
    "$CKPT_R26" && log "r2.6 ckpt synced"
[ ! -f "$CKPT_R27" ] && rsync -a \
    "$SPARK:$REPO/checkpoints_3b_varT_pondernet_round27_lp01/step_0003051_full.pt" \
    "$CKPT_R27" && log "r2.7 ckpt synced"

cd "$REPO"

run_one() {
    local label=$1 script=$2 out=$3 ckpt=$4
    shift 4
    [ ! -f "$ckpt" ] && { log "WARN: $label ckpt missing; skipping"; return; }
    log "--- $label -> $out"
    env CUDA_VISIBLE_DEVICES=1 CKPT="$ckpt" OUT="$out" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$@" "$PY" "training/$script" 2>&1 | tee -a "$LOG"
    log "--- $label done"
}

log "=== ListOps on r2.6 and r2.7 ==="
run_one r26_listops  eval_listops.py  "$DOCS/eval_listops_round26.json"  "$CKPT_R26" DEPTHS=4,8,16,32 N=100
run_one r27_listops  eval_listops.py  "$DOCS/eval_listops_round27.json"  "$CKPT_R27" DEPTHS=4,8,16,32 N=100

log "=== GSM8K on r2.6 and r2.7 ==="
run_one r26_gsm8k eval_gsm8k_generation.py "$DOCS/eval_gsm8k_generation_round26.json" "$CKPT_R26" DEPTHS=4,8,16,32 LIMIT=100
run_one r27_gsm8k eval_gsm8k_generation.py "$DOCS/eval_gsm8k_generation_round27.json" "$CKPT_R27" DEPTHS=4,8,16,32 LIMIT=100

log "=== GPU 1 post-capevals queue DONE ==="
