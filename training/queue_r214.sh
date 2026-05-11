#!/usr/bin/env bash
# Round 2.14 = T_FIXED=16 compute-scaling ablation from r2 collapsed.
# Extends the compute scaling curve T=1/2/4/8 with T=16, tests whether
# deeper recurrence helps or shows diminishing returns.
# Fires after auto_eval_round213 pipeline completes.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_r214.log; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
# Match actual cluster python processes (/usr/bin/python3 ... torchrun ... training/X.py).
# Not bash/ssh wrappers which also contain "python3 training/..." as a substring.
PROC_PATTERN="^/usr/bin/python3 .*training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|consolidate_ckpt|eval_listops|eval_gsm8k|act_halt|gen_samples|synthetic_depth)"
R213_LOG=/home/alexm/OpenMythos/training/auto_eval_round213.log
REPO=/home/alexm/OpenMythos
R2_BOOTSTRAP=$REPO/checkpoints_3b_varT_fast/step_0012207_full.pt

log "queue_r214 started; waiting for auto_eval_round213 pipeline complete (no deadline)"
while true; do
    grep -q "auto_eval_round213 pipeline complete" "$R213_LOG" 2>/dev/null && { log "r213 done"; break; }
    sleep 60
done

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

[ ! -f "$R2_BOOTSTRAP" ] && { log "ERROR: r2 collapsed ckpt missing at $R2_BOOTSTRAP"; exit 1; }
log "verifying r2 collapsed ckpt on workers"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q "alexm@$h" "test -f $R2_BOOTSTRAP" && log "  $h: present" || {
        log "  $h: rsyncing r2 collapsed ckpt"
        ssh -q "alexm@$h" "mkdir -p $REPO/checkpoints_3b_varT_fast"
        rsync -a "$R2_BOOTSTRAP" "alexm@$h:$R2_BOOTSTRAP"
    }
done

log "clearing rank logs so retry_cluster_training does not see r213's 'Training complete.'"
> /tmp/train_r0.log
for r_idx in 1 2 3; do
    case $r_idx in
        1) h=kebab-gx10-200g ;;
        2) h=kebab-gx10-2-200g ;;
        3) h=kebab-gx10-3-200g ;;
    esac
    ssh -q "alexm@$h" "rm -f /tmp/train_r${r_idx}.log; touch /tmp/train_r${r_idx}.log" || true
done

log "starting auto_eval_round214 watcher"
nohup bash "$REPO/training/auto_eval_round214.sh" \
    >"$REPO/training/auto_eval_round214.log" 2>&1 </dev/null &
disown $!
log "auto_eval_round214 watcher PID=$!"

log "firing r2.14 training under retry_cluster_training supervisor"
log "(supervisor handles NCCL hangs by killing+relaunching on >7min stall)"
cd "$REPO"
ROUND_NAME=r214 \
SCRIPT=training/3b_varT_act_v3.py \
PORT=29514 \
EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_act_v3_round214_T16 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt T_FIXED=16 T_MAX=16" \
    bash training/retry_cluster_training.sh
log "retry_cluster_training (r214) returned"

log "queue_r214 done"
