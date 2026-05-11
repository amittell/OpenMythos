#!/usr/bin/env bash
# gpu0_sft_backfill.sh
#
# Standalone supplemental SFT backfill for GPU 0 on RTX6000. Runs while the
# main queue (queue_rtx6000_gpu0_opportunistic.sh) is in its "waiting on GPU 1
# done marker" sleep loop -- GPU 0 is otherwise idle for ~37h until r2.14
# finishes.
#
# Targets the rounds that ARE NOT YET in docs/sft_lora_*_training_curve.json:
#   r2.6  joint PonderNet-KL at 100M tokens
#   r2.5  fixed T=8 baseline
#   r2.11 T_FIXED=2 compute-scaling
#   r2.12 T_FIXED=4 compute-scaling
# Existing SFT artifacts (r2.10, r2.13) are NOT re-done.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/gpu0_sft_backfill.log; }

REPO=/home/alexm/OpenMythos
RTX=alexm@kebab-rtx6000.lan
RTX_REPO=/home/alexm/OpenMythos
RTX_CKPTS=/home/alexm/OpenMythos/checkpoints
PY=/home/alexm/venvs/vllm-turboquant/bin/python3

declare -A LOCAL_CKPTS=(
    [r26]=$REPO/checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt
    [r25]=$REPO/checkpoints_3b_varT_act_v3_round25_fixedT8/step_0003051_full.pt
    [r211]=$REPO/checkpoints_3b_varT_act_v3_round211_T2/step_0012207_full.pt
    [r212]=$REPO/checkpoints_3b_varT_act_v3_round212_T4/step_0012207_full.pt
)

log "gpu0_sft_backfill started (4 SFTs sequential on RTX6000 GPU 0)"
log "rsync training/ + open_mythos/ to RTX6000 (idempotent, incremental)"
rsync -az "$REPO/training/" "$RTX:$RTX_REPO/training/" 2>&1 | tail -2
rsync -az "$REPO/open_mythos/" "$RTX:$RTX_REPO/open_mythos/" 2>&1 | tail -2

run_sft() {
    local label=$1
    local local_ckpt=$2
    local out_subdir="checkpoints_3b_sft_lora_${label}"

    [ ! -f "$local_ckpt" ] && { log "skip $label: missing $local_ckpt"; return; }
    [ -f "$REPO/docs/sft_lora_${label}_training_curve.json" ] && { log "skip $label: training_curve already exists"; return; }

    local rtx_ckpt="$RTX_CKPTS/ckpt_${label}_full.pt"
    local expected_size
    expected_size=$(stat -c%s "$local_ckpt")

    log "=== $label SFT ==="
    log "rsync ckpt to RTX6000"
    local actual=0
    for attempt in 1 2 3; do
        rsync -az --partial "$local_ckpt" "$RTX:$rtx_ckpt" 2>&1 | tail -2 | tee -a /tmp/gpu0_sft_backfill.log
        actual=$(ssh -q "$RTX" "stat -c%s $rtx_ckpt 2>/dev/null || echo 0")
        [ "$actual" = "$expected_size" ] && { log "rsync ok ($actual bytes)"; break; }
        log "rsync attempt $attempt incomplete (got=$actual want=$expected_size); retrying"
        sleep 10
    done
    [ "$actual" != "$expected_size" ] && { log "ERROR $label rsync failed; skipping"; return; }

    log "launching SFT on GPU 0 (~2h)"
    ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$rtx_ckpt' OUT_DIR='$RTX_REPO/$out_subdir' \
        DATASET='HuggingFaceH4/ultrachat_200k' SPLIT='train_sft' \
        MAX_SAMPLES=50000 SEQ_LEN=2048 \
        LORA_RANK=32 LORA_ALPHA=64 LR=2e-4 EPOCHS=1 \
        MICRO_BATCH=2 GRAD_ACCUM=4 SAVE_EVERY=500 SAVE_MERGED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY training/sft_lora_minibeast.py 2>&1" \
        | tee -a /tmp/gpu0_sft_backfill.log

    mkdir -p "$REPO/$out_subdir"
    ssh -q "$RTX" "[ -f $RTX_REPO/$out_subdir/lora_adapter_final.pt ]" && \
        rsync -az "$RTX:$RTX_REPO/$out_subdir/lora_adapter_final.pt" "$REPO/$out_subdir/" 2>&1 | tail -1
    ssh -q "$RTX" "[ -f $RTX_REPO/$out_subdir/training_curve.json ]" && \
        rsync -az "$RTX:$RTX_REPO/$out_subdir/training_curve.json" \
            "$REPO/docs/sft_lora_${label}_training_curve.json" 2>&1 | tail -1

    ssh -q "$RTX" "rm -f $rtx_ckpt"
    log "$label SFT done; ckpt cleaned up"
}

run_sft r26  "${LOCAL_CKPTS[r26]}"
run_sft r25  "${LOCAL_CKPTS[r25]}"
run_sft r211 "${LOCAL_CKPTS[r211]}"
run_sft r212 "${LOCAL_CKPTS[r212]}"

log "=== gpu0_sft_backfill done ==="
