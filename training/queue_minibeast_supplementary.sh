#!/usr/bin/env bash
# queue_minibeast_supplementary.sh
#
# Runs in PARALLEL with queue_rtx6000_post_r214.sh. While RTX6000 is busy
# with SFT (GPU 0) and inference benchmarks (GPU 1), mini-beast 5090 picks
# up supplementary cross-round evals to fill out paper sections §7.21
# (reasoning), §7.22 (K=64 depth_extrap), §7.23 (per_token_halt) for the
# new rounds (r2.10, r2.13, r2.14) where data was missing.
#
# Sequential per ckpt (one GPU on mini-beast), but starts immediately when
# r2.14 evals complete, in parallel with RTX6000 work.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_minibeast_supplementary.log; }

R214_LOG=/home/alexm/OpenMythos/training/auto_eval_round214.log
REPO=/home/alexm/OpenMythos
MINIBEAST=alex@mini-beast.lan
MB_REPO=/home/alex/git/OpenMythos
MB_CKPTS=/home/alex/OpenMythos/checkpoints
MB_DOCS=/home/alex/OpenMythos/docs

CKPT_R210=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt
CKPT_R213=$REPO/checkpoints_3b_varT_pondernet_round213/step_0003051_full.pt
CKPT_R214=$REPO/checkpoints_3b_varT_act_v3_round214_T16/step_0003051_full.pt

log "queue_minibeast_supplementary started; waiting for r2.14 evals complete"
DEADLINE=$(($(date +%s) + 5 * 24 * 3600))
while true; do
    grep -q "auto_eval_round214 pipeline complete" "$R214_LOG" 2>/dev/null \
        && { log "r2.14 done; proceeding"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 5-day deadline"; exit 1; }
    sleep 300
done

log "verifying mini-beast reachability"
ssh -o ConnectTimeout=10 "$MINIBEAST" "echo ok" 2>/dev/null | grep -q ok \
    || { log "ERROR: cannot SSH to mini-beast"; exit 1; }

# Free GPU on mini-beast (graceful systemd stop where applicable, fallback to kill)
log "clearing mini-beast GPU"
ssh -q "$MINIBEAST" 'bash -s' <<'REMOTE_EOF' 2>&1 | tee -a /tmp/queue_minibeast_supplementary.log
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d " " | grep -v "^$" || true)
if [ -n "$PIDS" ]; then
    echo "[mb] killing GPU procs: $PIDS"
    for pid in $PIDS; do
        cgroup=$(cat /proc/$pid/cgroup 2>/dev/null | head -1 || echo "")
        svc=$(echo "$cgroup" | grep -oE '[a-zA-Z0-9_.@-]+\.service' | tail -1 || true)
        if [ -n "$svc" ] && ! echo "$cgroup" | grep -q 'session.*\.scope'; then
            sudo systemctl stop "$svc" 2>&1 | head -2
        else
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    sleep 5
    # Force kill any survivors
    PIDS2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d " " | grep -v "^$" || true)
    [ -n "$PIDS2" ] && echo "$PIDS2" | xargs -r kill -9 2>/dev/null
fi
echo "[mb] gpu state:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
REMOTE_EOF

# Sync scripts
log "rsync training/ + open_mythos/ to mini-beast"
rsync -az "$REPO/training/" "$MINIBEAST:$MB_REPO/training/" 2>&1 | tail -2 | tee -a /tmp/queue_minibeast_supplementary.log
rsync -az "$REPO/open_mythos/" "$MINIBEAST:$MB_REPO/open_mythos/" 2>&1 | tail -2 | tee -a /tmp/queue_minibeast_supplementary.log
ssh -q "$MINIBEAST" "mkdir -p $MB_CKPTS $MB_DOCS"

# Run supplementary evals on each round's ckpt sequentially (one GPU)
# Total work per ckpt: rsync (~30s, 16 GB) + reasoning (~30 min) + per_token_halt
# (~10 min) + K=64 depth_extrap (~30 min) = ~70 min per round, 3.5h for 3 rounds.

run_supplementary() {
    local label=$1
    local local_ckpt=$2
    [ ! -f "$local_ckpt" ] && { log "skip $label: missing $local_ckpt"; return; }

    local mb_ckpt="$MB_CKPTS/ckpt_${label}_full.pt"
    log "=== $label: rsync ckpt ($(stat -c %s "$local_ckpt" | numfmt --to=iec)) ==="
    rsync -az --partial "$local_ckpt" "$MINIBEAST:$mb_ckpt" 2>&1 | tail -2 | tee -a /tmp/queue_minibeast_supplementary.log

    local round_id="${label#r2}"  # r210 -> 210, r213 -> 213
    [ "$label" = "r210" ] && round_id=210
    [ "$label" = "r213" ] && round_id=213
    [ "$label" = "r214" ] && round_id=214

    # 1) reasoning_eval (ARC-Easy, ARC-Challenge, HellaSwag)
    log "  $label: reasoning_eval"
    ssh -q "$MINIBEAST" "cd $MB_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$mb_ckpt' OUT='$MB_DOCS/reasoning_eval_round${round_id}.json' \
        python3 training/reasoning_eval.py 2>&1" \
        | tee -a /tmp/queue_minibeast_supplementary.log

    # 2) per_token_halt
    log "  $label: per_token_halt_analysis"
    ssh -q "$MINIBEAST" "cd $MB_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$mb_ckpt' OUT='$MB_DOCS/per_token_halt_round${round_id}.json' \
        python3 training/per_token_halt_analysis.py 2>&1" \
        | tee -a /tmp/queue_minibeast_supplementary.log

    # 3) K=64 depth_extrap
    log "  $label: depth_extrap K=64"
    ssh -q "$MINIBEAST" "cd $MB_REPO && CUDA_VISIBLE_DEVICES=0 \
        CKPT='$mb_ckpt' OUT='$MB_DOCS/depth_extrap_round${round_id}_k64.json' \
        DEPTHS='4,8,16,32,64' \
        python3 training/depth_extrap.py 2>&1" \
        | tee -a /tmp/queue_minibeast_supplementary.log

    # Sync JSONs back to spark for paper integration
    log "  $label: rsync JSONs back to spark"
    for stem in "reasoning_eval_round${round_id}" "per_token_halt_round${round_id}" "depth_extrap_round${round_id}_k64"; do
        ssh -q "$MINIBEAST" "[ -f $MB_DOCS/${stem}.json ]" && \
            rsync -az "$MINIBEAST:$MB_DOCS/${stem}.json" "$REPO/docs/" 2>&1 | tail -1
    done

    # Free disk on mini-beast (32 GB tight)
    ssh -q "$MINIBEAST" "rm -f $mb_ckpt"
    log "  $label: done; ckpt cleaned up"
}

run_supplementary r210 "$CKPT_R210"
run_supplementary r213 "$CKPT_R213"
run_supplementary r214 "$CKPT_R214"

log ""
log "=== queue_minibeast_supplementary done ==="
log "Cross-round data added for r2.10/r2.13/r2.14:"
log "  reasoning_eval_round{210,213,214}.json    -> §7.21"
log "  per_token_halt_round{210,213,214}.json    -> §7.23"
log "  depth_extrap_round{210,213,214}_k64.json  -> §7.22"
log ""
log "Mac watcher (auto_paper_integrate_watcher.sh) will detect new JSONs and"
log "auto-update the paper sections within 10 min."
