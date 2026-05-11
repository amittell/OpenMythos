#!/usr/bin/env bash
# gpu1_eval_backfill.sh
#
# Standalone supplemental eval backfill for GPU 1 on RTX6000. Runs while the
# main GPU 1 queue (queue_rtx6000_gpu1_opportunistic.sh) is in its r2.14 ckpt
# sleep loop. Started AFTER the queue has finished its r2.10/r2.11/r2.12/r2.13
# inference benchmarks (which is fast, ~10 min on Blackwell).
#
# Coordination: this script does NOT touch /tmp/rtx6000_gpu1_done. The main
# queue still creates that marker after the r2.14 phase 3 work.
#
# Targets:
#   K=64 depth_extrap: r2.5, r2.7, r2.9 (cross-round figure completeness)
#   reasoning_eval (ARC + HellaSwag): r2.10, r2.5, r2.9

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/gpu1_eval_backfill.log >&2; }

REPO=/home/alexm/OpenMythos
RTX=alexm@kebab-rtx6000.lan
RTX_REPO=/home/alexm/OpenMythos
RTX_CKPTS=/home/alexm/OpenMythos/checkpoints
RTX_DOCS=/home/alexm/OpenMythos/docs
PY=/home/alexm/venvs/vllm-turboquant/bin/python3

declare -A LOCAL_CKPTS=(
    [r25]=$REPO/checkpoints_3b_varT_act_v3_round25_fixedT8/step_0003051_full.pt
    [r27]=$REPO/checkpoints_3b_varT_pondernet_round27_lp01/step_0003051_full.pt
    [r29]=$REPO/checkpoints_3b_varT_act_v3_round29_T1/step_0003051_full.pt
    [r210]=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
)
# r25/r27 ckpts are also already on RTX6000 (ckpt_r25_full.pt / ckpt_r27_full.pt
# under /home/alexm/OpenMythos/checkpoints/), the rsync below will be incremental.

log "gpu1_eval_backfill started"
log "rsync training/ + open_mythos/ to RTX6000 (idempotent)"
rsync -az "$REPO/training/" "$RTX:$RTX_REPO/training/" 2>&1 | tail -2
rsync -az "$REPO/open_mythos/" "$RTX:$RTX_REPO/open_mythos/" 2>&1 | tail -2

ensure_ckpt() {
    local label=$1
    local local_ckpt=$2
    local rtx_ckpt="$RTX_CKPTS/ckpt_${label}_full.pt"
    local expected_size
    expected_size=$(stat -c%s "$local_ckpt")
    local actual
    actual=$(ssh -q "$RTX" "stat -c%s $rtx_ckpt 2>/dev/null || echo 0")
    if [ "$actual" = "$expected_size" ]; then
        log "ckpt $label already on RTX6000 ($actual bytes)"
        echo "$rtx_ckpt"
        return 0
    fi
    log "rsync $label ckpt to RTX6000"
    for attempt in 1 2 3; do
        rsync -az --partial "$local_ckpt" "$RTX:$rtx_ckpt" 2>&1 | tail -1 | tee -a /tmp/gpu1_eval_backfill.log
        actual=$(ssh -q "$RTX" "stat -c%s $rtx_ckpt 2>/dev/null || echo 0")
        [ "$actual" = "$expected_size" ] && { log "rsync ok ($actual bytes)"; echo "$rtx_ckpt"; return 0; }
        log "rsync $label attempt $attempt incomplete; retrying"
        sleep 10
    done
    log "ERROR ckpt $label rsync failed after 3 attempts"
    return 1
}

# K=64 depth_extrap on r2.5, r2.7, r2.9 (sequential on GPU 1; ~6 min each)
for r in r25 r27 r29; do
    round_num=${r#r}
    out_json="$REPO/docs/depth_extrap_round${round_num}_k64.json"
    [ -f "$out_json" ] && { log "skip $r K=64: $out_json already exists"; continue; }
    rtx_ckpt=$(ensure_ckpt "$r" "${LOCAL_CKPTS[$r]}") || continue
    log "$r: K=64 depth_extrap"
    ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
        CKPT='$rtx_ckpt' OUT='$RTX_DOCS/depth_extrap_round${round_num}_k64.json' \
        DEPTHS=4,8,16,32,64 \
        $PY training/depth_extrap.py 2>&1" | tee -a /tmp/gpu1_eval_backfill.log
    rsync -az "$RTX:$RTX_DOCS/depth_extrap_round${round_num}_k64.json" "$REPO/docs/" 2>&1 | tail -1
    log "$r K=64 done"
done

# reasoning_eval (ARC + HellaSwag) on r2.5, r2.9, r2.10 (~30 min each)
for r in r25 r29 r210; do
    round_num=${r#r}
    out_json="$REPO/docs/reasoning_eval_round${round_num}.json"
    [ -f "$out_json" ] && { log "skip $r reasoning_eval: $out_json already exists"; continue; }
    rtx_ckpt=$(ensure_ckpt "$r" "${LOCAL_CKPTS[$r]}") || continue
    log "$r: reasoning_eval"
    ssh -q "$RTX" "cd $RTX_REPO && CUDA_VISIBLE_DEVICES=1 \
        CKPT='$rtx_ckpt' OUT='$RTX_DOCS/reasoning_eval_round${round_num}.json' \
        $PY training/reasoning_eval.py 2>&1" | tee -a /tmp/gpu1_eval_backfill.log
    rsync -az "$RTX:$RTX_DOCS/reasoning_eval_round${round_num}.json" "$REPO/docs/" 2>&1 | tail -1
    log "$r reasoning_eval done"
done

# Cleanup ckpts we added (keep r213 / r210 / r25 / r27 if main queue may still need)
# Actually leave them: main queue is done with these rounds; only r2.14 ckpt
# matters going forward. We free disk only if needed.

log "=== gpu1_eval_backfill done ==="
