#!/usr/bin/env bash
# Compute-scaling ablations: T_FIXED=2 then T_FIXED=4, both from r2 collapsed.
# Together with r2.9 (T=1), r2.5 (T=8), and variable-T training, this fills
# the compute scaling curve for the paper's §7.14 analysis.
# Fires after queue_r210 (r2.10 training) completes.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_r29_scaling.log; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|consolidate_ckpt|eval_listops|eval_gsm8k|act_halt|gen_samples|synthetic_depth)"
R210_LOG=/home/alexm/OpenMythos/training/auto_eval_round210.log
REPO=/home/alexm/OpenMythos
R2_BOOTSTRAP=$REPO/checkpoints_3b_varT_fast/step_0012207_full.pt
DOCS=$REPO/docs
PY=python3

log "queue_r29_scaling started; waiting for auto_eval_round210 pipeline complete"
DEADLINE=$(($(date +%s) + 24 * 3600))
while true; do
    grep -q "auto_eval_round210 pipeline complete" "$R210_LOG" 2>/dev/null && { log "r210 done"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 24h deadline"; exit 1; }
    sleep 60
done

fence() {
    log "fencing on cluster procs"
    while true; do
        any=0
        for h in $NODES_200G; do
            ssh -q -o ConnectTimeout=5 alexm@$h "pgrep -f '$PROC_PATTERN' >/dev/null 2>&1" && any=1 && break
        done
        [ $any -eq 0 ] && break
        sleep 30
    done
    log "fence cleared"
}

run_fixed_T() {
    local T=$1 round=$2 ckpt_dir=$3 eval_log=$4
    log "=== T_FIXED=$T (round $round) ==="
    fence
    log "  verifying bootstrap ckpt on workers"
    for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
        ssh -q "alexm@$h" "test -f $R2_BOOTSTRAP" || {
            ssh -q "alexm@$h" "mkdir -p $REPO/checkpoints_3b_varT_fast"
            rsync -a "$R2_BOOTSTRAP" "alexm@$h:$R2_BOOTSTRAP"
        }
    done
    log "  firing T_FIXED=$T training"
    cd "$REPO"
    SCRIPT=training/3b_varT_act_v3.py \
        EXTRA_ENV="CKPT_DIR=$ckpt_dir BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt T_FIXED=$T" \
        bash training/launch_3b.sh 2>&1 | tee -a /tmp/queue_r29_scaling.log
    log "  training returned; running evals"

    CKPT_PATH=$(ls "$REPO/$ckpt_dir"/step_*_full.pt 2>/dev/null | sort | tail -1 || true)
    if [ -z "$CKPT_PATH" ]; then
        # Consolidate first
        log "  no full ckpt found; running quick consolidate"
        LATEST=$(ls "$REPO/$ckpt_dir"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
        [ -z "$LATEST" ] && { log "ERROR: no shards in $ckpt_dir"; return; }
        STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
        SRC="$REPO/$ckpt_dir/step_${STEP}"
        DST="$REPO/$ckpt_dir/step_${STEP}_full.pt"
        RANK=0
        for HOST in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
            GID=$(ssh -q "alexm@$HOST" "show_gids rocep1s0f1 | awk '\$7==\"bond0\" && \$6==\"v2\" && \$5 ~ /^192\\.168\\.100\\./ {print \$3; exit}'")
            ssh -q "alexm@$HOST" \
                "cd ~/OpenMythos && nohup env NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:\$PATH NCCL_IB_GID_INDEX=${GID} \
                 torchrun --nnodes=4 --nproc_per_node=1 --node_rank=${RANK} --master_addr=192.168.100.10 --master_port=29612 \
                 training/consolidate_ckpt.py ${SRC} ${DST} >/tmp/consolidate_scaling_r${RANK}.log 2>&1 </dev/null & disown" &
            RANK=$((RANK+1))
        done
        wait
        DEADLINE_C=$(($(date +%s) + 1800))
        while [ ! -s "$DST" ]; do
            [ "$(date +%s)" -gt "$DEADLINE_C" ] && { log "ERROR: consolidation timed out"; return; }
            sleep 5
        done
        CKPT_PATH="$DST"
    fi
    log "  ckpt: $CKPT_PATH"

    CKPT="$CKPT_PATH" OUT="$DOCS/depth_extrap_${round}.json" \
        "$PY" training/depth_extrap.py 2>&1 | tee -a /tmp/queue_r29_scaling.log
    CKPT="$CKPT_PATH" OUT="$DOCS/act_halt_diagnostic_${round}.json" \
        "$PY" training/act_halt_diagnostic.py 2>&1 | tee -a /tmp/queue_r29_scaling.log
    CKPT="$CKPT_PATH" OUT="$DOCS/act_halt_histogram_${round}.json" \
        "$PY" training/act_halt_histogram.py 2>&1 | tee -a /tmp/queue_r29_scaling.log
    CKPT="$CKPT_PATH" OUT="$DOCS/reasoning_eval_${round}.json" \
        "$PY" training/reasoning_eval.py 2>&1 | tee -a /tmp/queue_r29_scaling.log
    log "  $round evals done"
}

run_fixed_T 2 round211_T2 checkpoints_3b_varT_act_v3_round211_T2 /tmp/queue_r29_scaling.log
run_fixed_T 4 round212_T4 checkpoints_3b_varT_act_v3_round212_T4 /tmp/queue_r29_scaling.log

log "=== queue_r29_scaling DONE (T=2 and T=4 ablations complete) ==="
