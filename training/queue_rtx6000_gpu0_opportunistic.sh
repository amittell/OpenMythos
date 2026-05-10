#!/usr/bin/env bash
# queue_rtx6000_gpu0_opportunistic.sh
#
# RTX6000 GPU 0: LoRA SFT on each round's ckpt as it becomes available.
# Works through r2.10, r2.13 (when ready), and skips compute-scaling rounds
# (r2.11/r2.12/r2.14) which are ablation baselines, not SFT candidates.
#
# Stops vllm-20b/embedding-related services at start, restarts at end.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_rtx6000_gpu0_op.log; }

REPO=/home/alexm/OpenMythos
RTX=alexm@kebab-rtx6000.lan
RTX_REPO=/home/alexm/OpenMythos
RTX_CKPTS=/home/alexm/OpenMythos/checkpoints
PY=/home/alexm/venvs/vllm-turboquant/bin/python3

CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt

log "queue_rtx6000_gpu0_opportunistic started"

# Stop kebab-rtx-vllm + embedding services to free GPU 0
log "stopping kebab-rtx-vllm + kebab-rtx-embedding"
ssh -q "$RTX" "sudo systemctl stop kebab-rtx-vllm.service 2>&1; \
    sudo systemctl stop kebab-rtx-embedding.service 2>&1; sleep 5"

# rsync infrastructure to RTX6000
log "rsync training/ + open_mythos/"
rsync -az "$REPO/training/" "$RTX:$RTX_REPO/training/" 2>&1 | tail -2
rsync -az "$REPO/open_mythos/" "$RTX:$RTX_REPO/open_mythos/" 2>&1 | tail -2
ssh -q "$RTX" "mkdir -p $RTX_CKPTS"

run_sft() {
    local label=$1     # e.g., r210, r213
    local local_ckpt=$2
    local out_subdir="checkpoints_3b_sft_lora_${label}"

    [ ! -f "$local_ckpt" ] && { log "skip $label: missing $local_ckpt"; return; }

    local rtx_ckpt="$RTX_CKPTS/ckpt_${label}_full.pt"
    log "=== $label SFT ==="
    log "rsync ckpt to RTX6000"
    rsync -az --partial "$local_ckpt" "$RTX:$rtx_ckpt" 2>&1 | tail -2 | tee -a /tmp/queue_rtx6000_gpu0_op.log

    log "launching SFT on GPU 0 (~1.5-2h)"
    ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$rtx_ckpt' OUT_DIR='$RTX_REPO/$out_subdir' \
        DATASET='HuggingFaceH4/ultrachat_200k' SPLIT='train_sft' \
        MAX_SAMPLES=50000 SEQ_LEN=2048 \
        LORA_RANK=32 LORA_ALPHA=64 LR=2e-4 EPOCHS=1 \
        MICRO_BATCH=2 GRAD_ACCUM=4 SAVE_EVERY=500 SAVE_MERGED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY training/sft_lora_minibeast.py 2>&1" \
        | tee -a /tmp/queue_rtx6000_gpu0_op.log

    # Pull artifacts back
    mkdir -p "$REPO/$out_subdir"
    ssh -q "$RTX" "[ -f $RTX_REPO/$out_subdir/lora_adapter_final.pt ]" && \
        rsync -az "$RTX:$RTX_REPO/$out_subdir/lora_adapter_final.pt" "$REPO/$out_subdir/" 2>&1 | tail -1
    ssh -q "$RTX" "[ -f $RTX_REPO/$out_subdir/training_curve.json ]" && \
        rsync -az "$RTX:$RTX_REPO/$out_subdir/training_curve.json" \
            "$REPO/docs/sft_lora_${label}_training_curve.json" 2>&1 | tail -1

    # Free RTX6000 ckpt copy
    ssh -q "$RTX" "rm -f $rtx_ckpt"
    log "$label SFT done; artifacts synced to spark"
}

# Phase 1: SFT on r2.10 (ready now)
run_sft r210 "$CKPT_R210"

# Phase 2: wait for r2.13 ckpt, then SFT
log "waiting for r2.13 final ckpt..."
while [ ! -f "$CKPT_R213" ]; do
    sleep 120
done
log "r2.13 ckpt available"
run_sft r213 "$CKPT_R213"

# We deliberately skip SFT on r2.11/r2.12/r2.14 (compute-scaling baselines).
# The post-SFT artifacts of interest are r210 + r213.

# Restart RTX6000 services at end (only after GPU 1 queue done too — coordinate via marker)
log "marking GPU 0 SFT phase done; waiting on GPU 1 queue before restarting services"
touch /tmp/rtx6000_gpu0_done
while [ ! -f /tmp/rtx6000_gpu1_done ]; do
    sleep 60
done
log "both GPU 0 and GPU 1 queues done; restarting services"
ssh -q "$RTX" "sudo systemctl start kebab-rtx-embedding.service 2>&1; \
    sudo systemctl start kebab-rtx-vllm.service 2>&1"
rm -f /tmp/rtx6000_gpu0_done /tmp/rtx6000_gpu1_done

log "=== queue_rtx6000_gpu0_opportunistic done ==="
