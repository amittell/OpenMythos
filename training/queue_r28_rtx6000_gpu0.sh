#!/usr/bin/env bash
# RTX6000 GPU 0: anti-collapse head fine-tune experiments.
# Assumes gpt-oss-120b (kebab-vllm) has already been stopped by caller.
#
# 1. R2.8 retry: head-only fine-tune from r2.4 backbone (5M tokens, lambda=10)
# 2. R2.8b:      head-only fine-tune from r2.6 backbone (5M tokens, lambda=10)
#
# r2.4 checkpoint already at /home/alexm/OpenMythos/checkpoints/ckpt_r24_full.pt
# r2.6 checkpoint rsynced from spark at the start of this script.

set -uo pipefail
LOG=/tmp/queue_r28_gpu0.log
PY=/home/alexm/venvs/vllm-turboquant/bin/python3
REPO=/home/alexm/OpenMythos
DOCS=$REPO/docs
SPARK=alexm@kebab-spark.lan

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== r2.8/r2.8b head fine-tune queue (GPU 0) starting ==="

# Free GPU 0: stop services via systemd so they don't auto-restart (docker stop
# alone is insufficient -- systemd unit RestartPolicy immediately respawns the container)
log "stopping kebab-rtx-vllm.service and kebab-rtx-embedding.service via systemd"
sudo systemctl stop kebab-rtx-vllm.service 2>/dev/null \
    && log "kebab-rtx-vllm.service stopped" \
    || log "kebab-rtx-vllm.service: already stopped or not found"
sudo systemctl stop kebab-rtx-embedding.service 2>/dev/null \
    && log "kebab-rtx-embedding.service stopped" \
    || log "kebab-rtx-embedding.service: already stopped or not found"

# Kill any lingering GPU 0 processes directly (CUDA driver may hold context after container exit)
GPU0_PIDS=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
if [ -n "$GPU0_PIDS" ]; then
    log "killing lingering GPU 0 PIDs: $GPU0_PIDS"
    echo "$GPU0_PIDS" | xargs -r kill -9 2>/dev/null || true
fi

# Poll until GPU 0 VRAM is actually clear (timeout 120s)
log "waiting for GPU 0 VRAM to clear..."
for i in $(seq 1 24); do
    GPU0_USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    log "GPU 0: ${GPU0_USED} MiB used (check $i/24)"
    [ "$GPU0_USED" -lt 5000 ] && { log "GPU 0 clear"; break; }
    [ "$i" -eq 24 ] && { log "ERROR: GPU 0 still at ${GPU0_USED} MiB after 120s; aborting"; exit 1; }
    sleep 5
done

# Rsync r2.6 checkpoint from spark
CKPT_R26_REMOTE=checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt
CKPT_R26_LOCAL=$REPO/checkpoints/ckpt_r26_full.pt
if [ ! -f "$CKPT_R26_LOCAL" ]; then
    log "rsyncing r2.6 ckpt from spark..."
    rsync -a "$SPARK:$REPO/$CKPT_R26_REMOTE" "$CKPT_R26_LOCAL"
    log "r2.6 ckpt synced: $(ls -lh $CKPT_R26_LOCAL | awk '{print $5}')"
fi

cd "$REPO"

run_finetune() {
    local label=$1 ckpt=$2 out_dir=$3
    log "--- $label: head fine-tune from $ckpt -> $out_dir"
    mkdir -p "$REPO/$out_dir"
    CUDA_VISIBLE_DEVICES=0 \
        CKPT="$ckpt" \
        OUT_DIR="$REPO/$out_dir" \
        TARGET_TOKENS=5000000 \
        LAMBDA_ANTI_COLLAPSE=10 \
        REINIT_HEAD=1 \
        LR=1e-4 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" training/3b_act_finetune.py 2>&1 | tee -a "$LOG"
    log "--- $label fine-tune done"

    local ckpt_out
    ckpt_out=$(ls "$REPO/$out_dir"/step_*_full.pt 2>/dev/null | sort | tail -1 || true)
    [ -z "$ckpt_out" ] && { log "WARN: no full ckpt in $out_dir; skipping evals"; return; }
    log "--- $label evals (depth_extrap, halt_diag, halt_hist)"
    CUDA_VISIBLE_DEVICES=0 CKPT="$ckpt_out" OUT="$DOCS/${label}_depth_extrap.json" \
        "$PY" training/depth_extrap.py 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 CKPT="$ckpt_out" OUT="$DOCS/${label}_act_halt_diagnostic.json" \
        "$PY" training/act_halt_diagnostic.py 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 CKPT="$ckpt_out" OUT="$DOCS/${label}_act_halt_histogram.json" \
        "$PY" training/act_halt_histogram.py 2>&1 | tee -a "$LOG"
    log "--- $label evals done"
}

run_finetune r28_retry \
    "$REPO/checkpoints/ckpt_r24_full.pt" \
    checkpoints_3b_act_finetune_round28_retry

run_finetune r28b_from_r26 \
    "$CKPT_R26_LOCAL" \
    checkpoints_3b_act_finetune_round28b_from_r26

log "=== GPU 0 queue done; restarting services via systemd ==="
sudo systemctl start kebab-rtx-embedding.service \
    && log "kebab-rtx-embedding.service restarted" \
    || log "WARN: kebab-rtx-embedding.service restart failed"
sudo systemctl start kebab-rtx-vllm.service \
    && log "kebab-rtx-vllm.service restarted" \
    || log "WARN: kebab-rtx-vllm.service restart failed"
