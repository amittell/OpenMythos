#!/usr/bin/env bash
# queue_rtx6000_post_r214.sh
#
# Replaces queue_minibeast_post_r214.sh with a faster RTX6000-based path.
# After auto_eval_round214 completes, on RTX6000 we stop vllm to free GPU 0,
# then run two jobs in PARALLEL:
#   - GPU 0: SFT on r2.13 with UltraChat 200K (full SFT or LoRA, configurable)
#   - GPU 1: 3 inference throughput benchmarks (r2.10, r2.13, r2.14) sequentially
#
# Mini-beast 5090 runs in parallel (queue_minibeast_supplementary.sh) doing
# reasoning_eval, per_token_halt, K=64 depth_extrap on r2.10/r2.13/r2.14 ckpts
# so nothing is idle.
#
# After both phases: restart kebab-rtx-vllm.service.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_rtx6000_post_r214.log; }

R214_LOG=/home/alexm/OpenMythos/training/auto_eval_round214.log
REPO=/home/alexm/OpenMythos
RTX=alexm@kebab-rtx6000.lan
RTX_REPO=/home/alexm/OpenMythos
RTX_CKPTS=/home/alexm/OpenMythos/checkpoints
RTX_DOCS=/home/alexm/OpenMythos/docs
PY=/home/alexm/venvs/vllm-turboquant/bin/python3

# Final consolidated ckpts on spark (sources to rsync to RTX6000)
CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt
CKPT_R214=$REPO/checkpoints_3b_varT_act_v3_round214_T16/step_0003051_full.pt

log "queue_rtx6000_post_r214 started; waiting for r2.14 evals complete"
DEADLINE=$(($(date +%s) + 5 * 24 * 3600))
while true; do
    grep -q "auto_eval_round214 pipeline complete" "$R214_LOG" 2>/dev/null \
        && { log "r2.14 done; proceeding"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 5-day deadline"; exit 1; }
    sleep 300
done

log "verifying RTX6000 reachability"
if ! ssh -o ConnectTimeout=10 "$RTX" "echo ok" 2>/dev/null | grep -q ok; then
    log "ERROR: cannot SSH to $RTX"
    exit 1
fi

log "stopping kebab-rtx-vllm.service to free GPU 0"
ssh -q "$RTX" "sudo systemctl stop kebab-rtx-vllm.service 2>&1; \
    sudo systemctl stop kebab-rtx-embedding.service 2>&1; \
    sleep 5"

# Wait for GPU memory to clear on both GPUs
log "waiting for GPU 0 + GPU 1 to be clear"
ssh -q "$RTX" 'for i in $(seq 1 24); do
    used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
    used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
    echo "  check $i/24: GPU0=${used0} MiB, GPU1=${used1} MiB"
    if [ "$used0" -lt 5000 ] && [ "$used1" -lt 5000 ]; then
        echo "  both GPUs clear"
        exit 0
    fi
    sleep 5
done
echo "WARN: GPUs not fully clear after 120s; proceeding anyway"' 2>&1 | tee -a /tmp/queue_rtx6000_post_r214.log

# Make sure scripts and ckpts are on RTX6000
log "rsync training/ to RTX6000"
rsync -az "$REPO/training/" "$RTX:$RTX_REPO/training/" 2>&1 | tail -3 | tee -a /tmp/queue_rtx6000_post_r214.log
rsync -az "$REPO/open_mythos/" "$RTX:$RTX_REPO/open_mythos/" 2>&1 | tail -3 | tee -a /tmp/queue_rtx6000_post_r214.log

ssh -q "$RTX" "mkdir -p $RTX_CKPTS $RTX_DOCS"

# Rsync the 3 ckpts to RTX6000 in parallel
log "rsync r2.10/r2.13/r2.14 ckpts to RTX6000 (parallel)"
for src in "$CKPT_R210" "$CKPT_R213" "$CKPT_R214"; do
    [ -f "$src" ] || { log "WARN: missing $src; skipping"; continue; }
    fname="$(basename "$src")"
    case "$src" in
        *round210*) dst="$RTX_CKPTS/ckpt_r210_full.pt" ;;
        *round213*) dst="$RTX_CKPTS/ckpt_r213_full.pt" ;;
        *round214*) dst="$RTX_CKPTS/ckpt_r214_full.pt" ;;
    esac
    rsync -az --partial "$src" "$RTX:$dst" 2>&1 | tail -2 &
done
wait
log "ckpt rsync complete"

# Verify all 3 ckpts present on RTX6000
ssh -q "$RTX" "ls -la $RTX_CKPTS/ckpt_r{210,213,214}_full.pt 2>/dev/null" | tee -a /tmp/queue_rtx6000_post_r214.log

# ----------------------------------------------------------------------------
# Phase: parallel GPU 0 (SFT) + GPU 1 (3 inference benchmarks)
# ----------------------------------------------------------------------------

log "=== launching parallel: GPU 0 = SFT, GPU 1 = 3 inference benchmarks ==="

# GPU 0: LoRA SFT on r2.13 with UltraChat 200K (50k samples, ~1.5-2h on Blackwell)
# Output: /home/alexm/OpenMythos/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt
ssh -q "$RTX" "cd $RTX_REPO && nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    CKPT='$RTX_CKPTS/ckpt_r213_full.pt' \
    OUT_DIR='$RTX_REPO/checkpoints_3b_sft_lora_r213' \
    DATASET='HuggingFaceH4/ultrachat_200k' \
    SPLIT='train_sft' \
    MAX_SAMPLES=50000 \
    SEQ_LEN=2048 \
    LORA_RANK=32 \
    LORA_ALPHA=64 \
    LR=2e-4 \
    EPOCHS=1 \
    MICRO_BATCH=2 \
    GRAD_ACCUM=4 \
    SAVE_EVERY=500 \
    SAVE_MERGED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY training/sft_lora_minibeast.py \
    >/tmp/sft_r213_rtx6000.log 2>&1 </dev/null & echo SFT_PID=\$!"

# GPU 1: inference throughput benchmarks for all 3 ckpts (sequential since same GPU)
ssh -q "$RTX" "cd $RTX_REPO && nohup bash -c '
    for label in r210 r213 r214; do
        echo \"[\$(date +%T)] === inference benchmark \$label ===\"
        CUDA_VISIBLE_DEVICES=1 \
        CKPT=$RTX_CKPTS/ckpt_\${label}_full.pt \
        OUT=$RTX_DOCS/inference_benchmark_\${label}.json \
        T_VALUES=1,2,4,8,16 \
        PROMPT_LEN=512 \
        GEN_LEN=128 \
        DEVICE=cuda:0 \
        $PY training/inference_throughput_benchmark.py 2>&1
    done' >/tmp/infer_bench_rtx6000.log 2>&1 </dev/null & echo INFER_PID=\$!"

log "both GPU 0 (SFT) and GPU 1 (benchmarks) launched in parallel"
log "  monitor SFT:        ssh $RTX 'tail -F /tmp/sft_r213_rtx6000.log'"
log "  monitor benchmarks: ssh $RTX 'tail -F /tmp/infer_bench_rtx6000.log'"

# ----------------------------------------------------------------------------
# Wait for both jobs to finish
# ----------------------------------------------------------------------------

log "waiting for both jobs to complete..."
DEADLINE=$(($(date +%s) + 6 * 3600))  # 6h cap
while true; do
    sft_running=$(ssh -q "$RTX" "pgrep -fc 'python3 .*sft_lora_minibeast.py' || echo 0")
    bench_running=$(ssh -q "$RTX" "pgrep -fc 'python3 .*inference_throughput_benchmark.py' || echo 0")
    [ "$sft_running" = "0" ] && [ "$bench_running" = "0" ] && break
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 6h deadline; jobs still running (sft=$sft_running bench=$bench_running)"; break; }
    sleep 60
done
log "both jobs finished"

# ----------------------------------------------------------------------------
# Pull artifacts back to spark
# ----------------------------------------------------------------------------

log "syncing inference benchmark JSONs back to spark"
for label in r210 r213 r214; do
    ssh -q "$RTX" "[ -f $RTX_DOCS/inference_benchmark_${label}.json ]" && \
        rsync -az "$RTX:$RTX_DOCS/inference_benchmark_${label}.json" "$REPO/docs/" 2>&1 | tail -1 || \
        log "WARN: missing inference_benchmark_${label}.json on RTX6000"
done

log "syncing SFT artifacts"
mkdir -p "$REPO/checkpoints_3b_sft_lora_r213"
ssh -q "$RTX" "[ -f $RTX_REPO/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt ]" && \
    rsync -az "$RTX:$RTX_REPO/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt" \
        "$REPO/checkpoints_3b_sft_lora_r213/" 2>&1 | tail -1
ssh -q "$RTX" "[ -f $RTX_REPO/checkpoints_3b_sft_lora_r213/training_curve.json ]" && \
    rsync -az "$RTX:$RTX_REPO/checkpoints_3b_sft_lora_r213/training_curve.json" \
        "$REPO/docs/sft_lora_r213_training_curve.json" 2>&1 | tail -1

# ----------------------------------------------------------------------------
# Restart services
# ----------------------------------------------------------------------------

log "restarting kebab-rtx services"
ssh -q "$RTX" "sudo systemctl start kebab-rtx-embedding.service 2>&1; \
    sudo systemctl start kebab-rtx-vllm.service 2>&1"

log ""
log "=== queue_rtx6000_post_r214 done ==="
log "Inference benchmarks: $REPO/docs/inference_benchmark_r{210,213,214}.json"
log "SFT artifacts:        $REPO/checkpoints_3b_sft_lora_r213/lora_adapter_final.pt"
log "                      $REPO/docs/sft_lora_r213_training_curve.json"
