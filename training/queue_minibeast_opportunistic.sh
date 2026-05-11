#!/usr/bin/env bash
# queue_minibeast_opportunistic.sh
#
# mini-beast 5090: cross-round supplementary evals (per_token_halt + K=64
# depth_extrap; reasoning_eval already done for most rounds).
# Processes r2.10/r2.11/r2.12 immediately, adds r2.13/r2.14 when ready.
# Stops vllm-20b (gpt-oss-20b + Eagle3) before, restarts at end.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_minibeast_op.log; }

REPO=/home/alexm/OpenMythos
MINIBEAST=alex@mini-beast.lan
MB_REPO=/home/alex/git/OpenMythos
MB_CKPTS=/home/alex/OpenMythos/checkpoints
MB_DOCS=/home/alex/OpenMythos/docs

CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt
CKPT_R214=$REPO/checkpoints_3b_varT_act_v3_round214_T16/step_0003051_full.pt

log "queue_minibeast_opportunistic started"

# Stop user services on mini-beast (capture list for later restart)
log "stopping vllm-20b (gpt-oss-20b + Eagle3) and other 5090 services"
ssh -q "$MINIBEAST" 'bash -s' <<'REMOTE_EOF' 2>&1 | tee -a /tmp/queue_minibeast_op.log
ACTIVE_GPU_SERVICES=""
for svc in vllm-20b vllm-20b-128k vllm-20b-104k vllm-20b-120k vllm-20b-150k vllm-20b-fast vllm-20b-vanilla trtllm-20b sdxl florence2 whisper-cpp outetts; do
    if systemctl --user is-active "$svc" >/dev/null 2>&1; then
        ACTIVE_GPU_SERVICES="$ACTIVE_GPU_SERVICES $svc"
    fi
done
echo "$ACTIVE_GPU_SERVICES" > /tmp/mb_paused_services.list
echo "[mb] active GPU services to stop+restart:$ACTIVE_GPU_SERVICES"
for svc in $ACTIVE_GPU_SERVICES; do
    systemctl --user stop "$svc" 2>&1 | head -2 || true
done
systemctl --user stop vllm-20b 2>/dev/null || true
sleep 5
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d " " | grep -v "^$" || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null
sleep 3
echo "[mb] gpu after clear:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
REMOTE_EOF

log "rsync training/ + open_mythos/ to mini-beast"
rsync -az "$REPO/training/" "$MINIBEAST:$MB_REPO/training/" 2>&1 | tail -2
rsync -az "$REPO/open_mythos/" "$MINIBEAST:$MB_REPO/open_mythos/" 2>&1 | tail -2
ssh -q "$MINIBEAST" "mkdir -p $MB_CKPTS $MB_DOCS"

run_supplementary_evals() {
    local label=$1     # r210, r211, r212, r213, r214
    local round_num=$2 # 210, 211, 212, ...
    local local_ckpt=$3

    [ ! -f "$local_ckpt" ] && { log "skip $label: missing $local_ckpt"; return; }

    local mb_ckpt="$MB_CKPTS/ckpt_${label}_full.pt"
    log "=== $label cross-round evals ==="
    log "rsync ckpt to mini-beast"
    rsync -az --partial "$local_ckpt" "$MINIBEAST:$mb_ckpt" 2>&1 | tail -2 | tee -a /tmp/queue_minibeast_op.log

    # 1) per_token_halt (~10 min)
    log "  $label: per_token_halt_analysis.py"
    ssh -q "$MINIBEAST" "cd $MB_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$mb_ckpt' OUT='$MB_DOCS/per_token_halt_round${round_num}.json' \
        python3 training/per_token_halt_analysis.py 2>&1" 2>&1 | tail -10 | tee -a /tmp/queue_minibeast_op.log

    # 2) K=64 depth_extrap (~30 min)
    log "  $label: depth_extrap K=64"
    ssh -q "$MINIBEAST" "cd $MB_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$mb_ckpt' OUT='$MB_DOCS/depth_extrap_round${round_num}_k64.json' \
        DEPTHS=4,8,16,32,64 \
        python3 training/depth_extrap.py 2>&1" 2>&1 | tail -10 | tee -a /tmp/queue_minibeast_op.log

    # Sync JSONs back
    for stem in "per_token_halt_round${round_num}" "depth_extrap_round${round_num}_k64"; do
        ssh -q "$MINIBEAST" "[ -f $MB_DOCS/${stem}.json ]" && \
            rsync -az "$MINIBEAST:$MB_DOCS/${stem}.json" "$REPO/docs/" 2>&1 | tail -1
    done

    # Free disk on mini-beast (32 GB tight)
    ssh -q "$MINIBEAST" "rm -f $mb_ckpt"
    log "$label evals done; ckpt cleaned up"
}

# Phase 1: r2.10 (joint, ckpt on spark)
run_supplementary_evals r210 210 "$CKPT_R210"

# Phase 1b: r2.11 + r2.12 ckpts live on RTX6000 (single-GPU trained), not spark.
# Bring them to spark first then to mini-beast. Or fetch directly from RTX6000.
# Direct fetch from RTX6000 to mini-beast:
RTX=alexm@kebab-rtx6000.lan
RTX_R211=/home/alexm/OpenMythos/checkpoints_3b_varT_act_v3_round211_T2/step_0012207_full.pt
RTX_R212=/home/alexm/OpenMythos/checkpoints_3b_varT_act_v3_round212_T4/step_0012207_full.pt

for round_meta in "r211 211 $RTX_R211" "r212 212 $RTX_R212"; do
    set -- $round_meta
    label=$1; round_num=$2; src_ckpt=$3
    if ssh -q "$RTX" "[ -f $src_ckpt ]"; then
        local_tmp="$REPO/checkpoints_3b_varT_act_v3_round${round_num}_T2/step_0012207_full.pt"
        [ "$round_num" = "212" ] && local_tmp="$REPO/checkpoints_3b_varT_act_v3_round${round_num}_T4/step_0012207_full.pt"
        # Fetch ckpt to spark first
        log "fetching r${round_num} ckpt from RTX6000 to spark cache"
        mkdir -p "$(dirname "$local_tmp")"
        if [ ! -f "$local_tmp" ]; then
            rsync -az "$RTX:$src_ckpt" "$local_tmp" 2>&1 | tail -2
        fi
        run_supplementary_evals "$label" "$round_num" "$local_tmp"
    else
        log "skip $label: ckpt not found on RTX6000"
    fi
done

# Phase 2: wait for r2.13
log "waiting for r2.13 final ckpt..."
while [ ! -f "$CKPT_R213" ]; do
    sleep 120
done
run_supplementary_evals r213 213 "$CKPT_R213"

# Phase 3 (r2.14) intentionally skipped: moved to RTX6000 GPU 1 queue because
# the 5090 proved unreliable on K=64 depth_extrap (hung 15h, OOMed K=32 once,
# required host reboot after the kill -9 left the GPU in error state).
log "skipping mini-beast r2.14 phase (moved to RTX6000 GPU 1)"

# Restart user services on mini-beast
log "restarting paused user services on mini-beast"
ssh -q "$MINIBEAST" 'bash -s' <<'REMOTE_EOF' 2>&1 | tee -a /tmp/queue_minibeast_op.log
PAUSED=$(cat /tmp/mb_paused_services.list 2>/dev/null || echo "")
echo "$PAUSED" | grep -qw vllm-20b || PAUSED="$PAUSED vllm-20b"
echo "[mb] starting:$PAUSED"
for svc in $PAUSED; do
    systemctl --user start "$svc" 2>&1 | head -2 || true
done
sleep 5
echo "[mb] vllm-20b status:"
systemctl --user status vllm-20b --no-pager 2>&1 | head -6
rm -f /tmp/mb_paused_services.list
REMOTE_EOF

log "=== queue_minibeast_opportunistic done ==="
