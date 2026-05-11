#!/usr/bin/env bash
# queue_rtx6000_gpu1_opportunistic.sh
#
# RTX6000 GPU 1: inference throughput benchmarks (T=1/2/4/8/16) on each
# round's ckpt as it becomes available. Processes:
#   r2.10, r2.11, r2.12 immediately
#   r2.13 when ready
#   r2.14 when ready
# Each benchmark ~10-15 min on Blackwell.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_rtx6000_gpu1_op.log; }

REPO=/home/alexm/OpenMythos
RTX=alexm@kebab-rtx6000.lan
RTX_REPO=/home/alexm/OpenMythos
RTX_CKPTS=/home/alexm/OpenMythos/checkpoints
RTX_DOCS=/home/alexm/OpenMythos/docs
PY=/home/alexm/venvs/vllm-turboquant/bin/python3

CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R211=$REPO/checkpoints_3b_varT_act_v3_round211_T2/step_0012207_full.pt
CKPT_R212=$REPO/checkpoints_3b_varT_act_v3_round212_T4/step_0012207_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt
CKPT_R214=$REPO/checkpoints_3b_varT_act_v3_round214_T16/step_0003051_full.pt

# Note: r2.11/r2.12 ckpts live on RTX6000 (single-GPU trained there), not
# on spark. Need to fetch from RTX6000 itself or use them in place.
RTX_CKPT_R211=$RTX_CKPTS/../checkpoints_3b_varT_act_v3_round211_T2/step_0012207_full.pt
RTX_CKPT_R212=$RTX_CKPTS/../checkpoints_3b_varT_act_v3_round212_T4/step_0012207_full.pt

log "queue_rtx6000_gpu1_opportunistic started"

# Sync infra
log "rsync training/ + open_mythos/ to RTX6000"
rsync -az "$REPO/training/" "$RTX:$RTX_REPO/training/" 2>&1 | tail -2
rsync -az "$REPO/open_mythos/" "$RTX:$RTX_REPO/open_mythos/" 2>&1 | tail -2
ssh -q "$RTX" "mkdir -p $RTX_CKPTS"

run_benchmark() {
    local label=$1
    local local_or_rtx_ckpt=$2  # path on whichever side has it
    local on_rtx=${3:-0}        # 1 if ckpt is already on RTX6000

    local rtx_ckpt
    if [ "$on_rtx" = "1" ]; then
        rtx_ckpt="$local_or_rtx_ckpt"
        ssh -q "$RTX" "[ -f $rtx_ckpt ]" || { log "skip $label: missing $rtx_ckpt on RTX6000"; return; }
        log "=== $label inference benchmark (ckpt already on RTX6000) ==="
    else
        [ ! -f "$local_or_rtx_ckpt" ] && { log "skip $label: missing $local_or_rtx_ckpt on spark"; return; }
        rtx_ckpt="$RTX_CKPTS/ckpt_${label}_full.pt"
        log "=== $label inference benchmark ==="
        log "rsync ckpt to RTX6000"
        local expected_size
        expected_size=$(stat -c%s "$local_or_rtx_ckpt")
        for attempt in 1 2 3; do
            rsync -az --partial "$local_or_rtx_ckpt" "$RTX:$rtx_ckpt" 2>&1 | tail -2 | tee -a /tmp/queue_rtx6000_gpu1_op.log
            actual=$(ssh -q "$RTX" "stat -c%s $rtx_ckpt 2>/dev/null || echo 0")
            if [ "$actual" = "$expected_size" ]; then
                log "rsync ok ($actual bytes)"
                break
            fi
            log "rsync attempt $attempt incomplete (got=$actual want=$expected_size); retrying"
            sleep 10
        done
        [ "$actual" != "$expected_size" ] && { log "ERROR rsync failed after 3 attempts; skip $label"; return; }
    fi

    log "running benchmark T=1,2,4,8,16 (~10-15 min)"
    ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
        CKPT='$rtx_ckpt' OUT='$RTX_DOCS/inference_benchmark_${label}.json' \
        T_VALUES=1,2,4,8,16 PROMPT_LEN=512 GEN_LEN=128 DEVICE=cuda:0 \
        $PY training/inference_throughput_benchmark.py 2>&1" \
        | tee -a /tmp/queue_rtx6000_gpu1_op.log

    # Pull JSON back
    rsync -az "$RTX:$RTX_DOCS/inference_benchmark_${label}.json" "$REPO/docs/" 2>&1 | tail -1

    # Free copy if we made one
    if [ "$on_rtx" = "0" ]; then
        ssh -q "$RTX" "rm -f $rtx_ckpt"
    fi
    log "$label benchmark done"
}

# Phase 1: rounds with ckpts ready now
run_benchmark r210 "$CKPT_R210" 0

# r2.11/r2.12 ckpts are already on RTX6000 (trained there directly)
run_benchmark r211 "$RTX_CKPT_R211" 1
run_benchmark r212 "$RTX_CKPT_R212" 1

# Phase 2: wait for r2.13
log "waiting for r2.13 ckpt..."
while [ ! -f "$CKPT_R213" ]; do
    sleep 120
done
run_benchmark r213 "$CKPT_R213" 0

# Phase 3: r2.14. ABSORBS the mini-beast queue's r2.14 cross-round evals --
# 5090 proved unreliable on K=64 (hung 15h, OOMed at K=32, required host reboot)
# so per_token_halt + K=64 depth_extrap now run here on GPU 1 after benchmark.
log "waiting for r2.14 ckpt..."
while [ ! -f "$CKPT_R214" ]; do
    sleep 300
done

R214_RTX_CKPT="$RTX_CKPTS/ckpt_r214_full.pt"
log "=== r214 phase 3 (benchmark + per_token_halt + K=64 depth_extrap) ==="
log "rsync r214 ckpt to RTX6000"
R214_SIZE=$(stat -c%s "$CKPT_R214")
for attempt in 1 2 3; do
    rsync -az --partial "$CKPT_R214" "$RTX:$R214_RTX_CKPT" 2>&1 | tail -2 | tee -a /tmp/queue_rtx6000_gpu1_op.log
    ACTUAL=$(ssh -q "$RTX" "stat -c%s $R214_RTX_CKPT 2>/dev/null || echo 0")
    [ "$ACTUAL" = "$R214_SIZE" ] && { log "rsync ok ($ACTUAL bytes)"; break; }
    log "rsync attempt $attempt incomplete (got=$ACTUAL want=$R214_SIZE); retrying"
    sleep 10
done
[ "$ACTUAL" != "$R214_SIZE" ] && { log "ERROR r214 rsync failed; skipping phase 3"; touch /tmp/rtx6000_gpu1_done; exit 1; }

# 3a. inference benchmark (~10-15 min)
log "r214: inference benchmark T=1,2,4,8,16"
ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
    CKPT='$R214_RTX_CKPT' OUT='$RTX_DOCS/inference_benchmark_r214.json' \
    T_VALUES=1,2,4,8,16 PROMPT_LEN=512 GEN_LEN=128 DEVICE=cuda:0 \
    $PY training/inference_throughput_benchmark.py 2>&1" \
    | tee -a /tmp/queue_rtx6000_gpu1_op.log
rsync -az "$RTX:$RTX_DOCS/inference_benchmark_r214.json" "$REPO/docs/" 2>&1 | tail -1

# 3b. per_token_halt (~5-10 min on Blackwell)
log "r214: per_token_halt_analysis"
ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
    CKPT='$R214_RTX_CKPT' OUT='$RTX_DOCS/per_token_halt_round214.json' \
    $PY training/per_token_halt_analysis.py 2>&1" \
    | tee -a /tmp/queue_rtx6000_gpu1_op.log
rsync -az "$RTX:$RTX_DOCS/per_token_halt_round214.json" "$REPO/docs/" 2>&1 | tail -1

# 3c. K=64 depth_extrap (~6 min on Blackwell, matched r210/r213 timing)
log "r214: K=64 depth_extrap"
ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
    CKPT='$R214_RTX_CKPT' OUT='$RTX_DOCS/depth_extrap_round214_k64.json' \
    DEPTHS=4,8,16,32,64 \
    $PY training/depth_extrap.py 2>&1" \
    | tee -a /tmp/queue_rtx6000_gpu1_op.log
rsync -az "$RTX:$RTX_DOCS/depth_extrap_round214_k64.json" "$REPO/docs/" 2>&1 | tail -1

# Cleanup r214 ckpt
ssh -q "$RTX" "rm -f $R214_RTX_CKPT"
log "r214 phase 3 done (benchmark + per_token_halt + K=64 depth_extrap)"

log "marking GPU 1 phase done"
touch /tmp/rtx6000_gpu1_done
log "=== queue_rtx6000_gpu1_opportunistic done ==="
