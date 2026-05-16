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

# Free the GPU if anything is using it. Distinguish systemd-managed services
# (stop via systemctl) from interactive/orphan processes (kill directly).
log "checking mini-beast GPU 0 for processes to clear"
ssh -q "$MINIBEAST" 'bash -s' <<'REMOTE_EOF' 2>&1 | tee -a /tmp/queue_minibeast_post_r214.log
set -uo pipefail
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | grep -v '^$' || true)
if [ -z "$PIDS" ]; then
    echo "[mb] GPU 0 clear (no compute apps)"
    exit 0
fi
echo "[mb] GPU 0 has compute apps: $PIDS"
SERVICES_TO_STOP=""
PIDS_TO_KILL=""
for pid in $PIDS; do
    [ -d /proc/$pid ] || continue
    cgroup=$(cat /proc/$pid/cgroup 2>/dev/null | head -1)
    cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 120)
    echo "[mb] PID $pid: $cmd"
    echo "[mb]   cgroup: $cgroup"
    # Match systemd .service cgroup paths (system OR user services). Exclude session.scope.
    svc=$(echo "$cgroup" | grep -oE '[a-zA-Z0-9_.@-]+\.service' | tail -1 || true)
    if [ -n "$svc" ] && ! echo "$cgroup" | grep -q 'session.*\.scope'; then
        echo "[mb]   -> systemd service detected: $svc; will stop via systemctl"
        SERVICES_TO_STOP="$SERVICES_TO_STOP $svc"
    else
        echo "[mb]   -> not a service (session/scope/orphan); will kill directly"
        PIDS_TO_KILL="$PIDS_TO_KILL $pid"
    fi
done
# De-dup services
SERVICES_TO_STOP=$(echo "$SERVICES_TO_STOP" | tr ' ' '\n' | sort -u | tr '\n' ' ')
for svc in $SERVICES_TO_STOP; do
    echo "[mb] sudo systemctl stop $svc"
    sudo systemctl stop "$svc" 2>&1 | head -3 || true
done
if [ -n "$PIDS_TO_KILL" ]; then
    echo "[mb] killing orphan PIDs:$PIDS_TO_KILL"
    echo "$PIDS_TO_KILL" | xargs -r kill -TERM 2>/dev/null
    sleep 5
    # Force any survivors
    for pid in $PIDS_TO_KILL; do
        if [ -d /proc/$pid ]; then
            echo "[mb] PID $pid survived SIGTERM; SIGKILL"
            kill -9 $pid 2>/dev/null
        fi
    done
fi
# Final verification with timeout
for i in $(seq 1 12); do
    REMAIN=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -v '^$' || true)
    [ -z "$REMAIN" ] && { echo "[mb] GPU 0 fully clear"; exit 0; }
    sleep 5
done
echo "[mb] WARN: GPU 0 still has procs after clearing: $REMAIN"
exit 1
REMOTE_EOF
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: failed to clear mini-beast GPU 0; aborting"
    exit 1
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

# ----------------------------------------------------------------------------
# Phase 2: LoRA SFT on r2.13 ckpt with UltraChat
# ----------------------------------------------------------------------------

log ""
log "=== LoRA SFT on r2.13 with UltraChat 200K ==="

if [ ! -f "$CKPT_R213" ]; then
    log "WARN: r2.13 ckpt missing; skipping SFT"
else
    SFT_OUT_DIR_LOCAL=/home/alex/OpenMythos/checkpoints_3b_sft_lora_r213
    log "rsync r2.13 ckpt to mini-beast for SFT"
    MB_SFT_CKPT=/home/alex/OpenMythos/checkpoints/ckpt_r213_sft_base.pt
    rsync -az --partial "$CKPT_R213" "$MINIBEAST:$MB_SFT_CKPT" 2>&1 | tail -2 | tee -a /tmp/queue_minibeast_post_r214.log

    log "running SFT on mini-beast (~3-4h expected)"
    ssh -q "$MINIBEAST" "cd $MB_REPO && \
        CKPT='$MB_SFT_CKPT' \
        OUT_DIR='$SFT_OUT_DIR_LOCAL' \
        DATASET='HuggingFaceH4/ultrachat_200k' \
        SPLIT='train_sft' \
        MAX_SAMPLES=50000 \
        SEQ_LEN=2048 \
        LORA_RANK=32 \
        LORA_ALPHA=64 \
        LR=2e-4 \
        EPOCHS=1 \
        MICRO_BATCH=1 \
        GRAD_ACCUM=8 \
        SAVE_EVERY=500 \
        SAVE_MERGED=1 \
        python3 training/sft_lora_minibeast.py 2>&1" \
        | tee -a /tmp/queue_minibeast_post_r214.log

    # Pull adapter + curve back to spark for archival
    log "pulling SFT outputs back to spark"
    rsync -az "$MINIBEAST:$SFT_OUT_DIR_LOCAL/lora_adapter_final.pt" \
        "$REPO/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt" 2>&1 \
        | tail -2 | tee -a /tmp/queue_minibeast_post_r214.log
    rsync -az "$MINIBEAST:$SFT_OUT_DIR_LOCAL/training_curve.json" \
        "$REPO/docs/sft_lora_r213_training_curve.json" 2>&1 \
        | tail -2 | tee -a /tmp/queue_minibeast_post_r214.log
    log "SFT outputs synced back to spark"

    # Clean up the sft base ckpt on mini-beast (keep adapter and merged)
    ssh -q "$MINIBEAST" "rm -f $MB_SFT_CKPT"
fi

log ""
log "=== queue_minibeast_post_r214 done ==="
log "Inference benchmarks: $REPO/docs/inference_benchmark_r{210,213,214}.json"
log "SFT artifacts:        $REPO/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt"
log "                      $REPO/docs/sft_lora_r213_training_curve.json"
log ""
log "Future expansion (manual):"
log "  - DPO on the SFT'd model with UltraFeedback"
log "  - Long-context throughput benchmark at varying T"
log "  - User-controlled-T inference demo (paper figure)"
