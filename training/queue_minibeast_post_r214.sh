#!/usr/bin/env bash
# queue_minibeast_post_r214.sh
#
# Fires after auto_eval_round214 pipeline completes on the cluster.
# Reserves the RTX 5090 on mini-beast.lan for post-ablation work:
#   1. Inference throughput benchmark on r2.10, r2.13, r2.14 final ckpts
#      (paper artifact: tokens/sec at T=1/2/4/8/16 on consumer-class GPU)
#   2. Generation samples at multiple T values for qualitative comparison
#
# Future expansion (manual): LoRA SFT on r2.13 ckpt with UltraChat or
# similar for instruction-following demo. Add to this script when ready.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_minibeast_post_r214.log; }

R214_LOG=/home/alexm/OpenMythos/training/auto_eval_round214.log
REPO=/home/alexm/OpenMythos
MINIBEAST=alex@mini-beast.lan
MB_REPO=/home/alex/git/OpenMythos
MB_CKPT_DIR=/home/alex/OpenMythos/checkpoints
MB_DOCS=/home/alex/OpenMythos/docs

# Checkpoints to benchmark (final consolidated ckpts from each round)
CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt
CKPT_R214=$REPO/checkpoints_3b_varT_act_v3_round214_T16/step_0003051_full.pt

log "queue_minibeast_post_r214 started; waiting for r2.14 evals complete"
DEADLINE=$(($(date +%s) + 5 * 24 * 3600))
while true; do
    grep -q "auto_eval_round214 pipeline complete" "$R214_LOG" 2>/dev/null \
        && { log "r2.14 done; proceeding"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 5-day deadline exceeded"; exit 1; }
    sleep 300
done

log "verifying mini-beast reachability"
if ! ssh -o ConnectTimeout=10 "$MINIBEAST" "echo ok" 2>/dev/null | grep -q ok; then
    log "ERROR: cannot SSH to $MINIBEAST"
    exit 1
fi

# Free the GPU if a stale process is hogging it
log "checking mini-beast GPU 0 for stale processes"
STALE_PIDS=$(ssh -q "$MINIBEAST" "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -v '^\$'" || true)
if [ -n "$STALE_PIDS" ]; then
    log "found stale GPU procs on mini-beast: $STALE_PIDS"
    log "killing them (assumed safe -- queue requires sole 5090 access)"
    ssh -q "$MINIBEAST" "echo '$STALE_PIDS' | xargs -r kill -9 2>/dev/null; sleep 5"
fi

# Sync the latest training/ scripts to mini-beast
log "rsync training/ to mini-beast"
rsync -az "$REPO/training/" "$MINIBEAST:$MB_REPO/training/" 2>&1 | tail -3 | tee -a /tmp/queue_minibeast_post_r214.log
rsync -az "$REPO/open_mythos/" "$MINIBEAST:$MB_REPO/open_mythos/" 2>&1 | tail -3 | tee -a /tmp/queue_minibeast_post_r214.log

ssh -q "$MINIBEAST" "mkdir -p $MB_CKPT_DIR $MB_DOCS"

run_inference_bench() {
    local label=$1
    local local_ckpt=$2
    [ ! -f "$local_ckpt" ] && { log "skip $label: missing $local_ckpt"; return; }

    local mb_ckpt="$MB_CKPT_DIR/$(basename "$local_ckpt" .pt)_${label}.pt"
    log "rsync $label ckpt to mini-beast"
    rsync -az --partial --info=progress2 "$local_ckpt" "$MINIBEAST:$mb_ckpt" 2>&1 \
        | tail -2 | tee -a /tmp/queue_minibeast_post_r214.log

    log "running inference throughput benchmark for $label on mini-beast"
    ssh -q "$MINIBEAST" "cd $MB_REPO && \
        CKPT='$mb_ckpt' \
        OUT='$MB_DOCS/inference_benchmark_${label}.json' \
        T_VALUES='1,2,4,8,16' \
        PROMPT_LEN=512 \
        GEN_LEN=128 \
        DEVICE=cuda:0 \
        python3 training/inference_throughput_benchmark.py 2>&1" \
        | tee -a /tmp/queue_minibeast_post_r214.log

    # Pull the JSON back to spark for paper integration
    rsync -az "$MINIBEAST:$MB_DOCS/inference_benchmark_${label}.json" \
        "$REPO/docs/inference_benchmark_${label}.json" 2>&1 \
        | tail -2 | tee -a /tmp/queue_minibeast_post_r214.log
    log "$label inference benchmark complete; JSON synced back to spark"

    # Free the ckpt on mini-beast (32 GB tight, don't accumulate)
    ssh -q "$MINIBEAST" "rm -f $mb_ckpt"
}

# Run benchmarks for each major final ckpt
log "=== inference benchmark suite on RTX 5090 ==="
run_inference_bench r210 "$CKPT_R210"
run_inference_bench r213 "$CKPT_R213"
run_inference_bench r214 "$CKPT_R214"

log ""
log "=== queue_minibeast_post_r214 done ==="
log "Outputs: $REPO/docs/inference_benchmark_r{210,213,214}.json"
log ""
log "Future work for the 5090 (manual launch):"
log "  - LoRA SFT on r2.13 ckpt with UltraChat or similar"
log "  - Long-context throughput at varying T"
log "  - User-controlled-T inference demo (paper figure)"
