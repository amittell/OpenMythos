#!/usr/bin/env bash
# Round 2.10 = continue r2.6 joint PonderNet-KL training for +50M tokens.
# Answers whether the joint CE improvement (still gaining at r2.6/100M tokens)
# converges or continues at 150M joint tokens.
# Uses REINIT_HEAD=0 to preserve the trained halting head from r2.6.
# Fires after post_r29_evals.sh completes.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_r210.log; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|consolidate_ckpt|eval_listops|eval_gsm8k|act_halt|gen_samples|synthetic_depth)"
POST_R29_LOG=/tmp/post_r29_evals.log
REPO=/home/alexm/OpenMythos
R26_CKPT=$REPO/checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt

log "queue_r210 started; waiting for post_r29_evals complete"
DEADLINE=$(($(date +%s) + 12 * 3600))
while true; do
    grep -q "post_r29_evals complete" "$POST_R29_LOG" 2>/dev/null && { log "post_r29_evals done"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 12h deadline"; exit 1; }
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

[ ! -f "$R26_CKPT" ] && { log "ERROR: r2.6 ckpt missing"; exit 1; }
log "verifying r2.6 ckpt on workers"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q "alexm@$h" "test -f $R26_CKPT" && log "  $h: present" || {
        log "  $h: rsyncing r2.6 ckpt"
        ssh -q "alexm@$h" "mkdir -p $REPO/checkpoints_3b_varT_pondernet_round26"
        rsync -a "$R26_CKPT" "alexm@$h:$R26_CKPT"
    }
done

log "starting auto_eval_round210 watcher"
nohup bash "$REPO/training/auto_eval_round210.sh" \
    >"$REPO/training/auto_eval_round210.log" 2>&1 </dev/null &
disown $!
log "auto_eval_round210 watcher PID=$!"

log "firing 4-node round-2.10 joint training (r2.6 +50M tokens, REINIT_HEAD=0)"
cd "$REPO"
SCRIPT=training/3b_varT_pondernet_joint.py PORT=29511 \
    EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_round210 BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_round26/step_0003051_full.pt REINIT_HEAD=0 TARGET_TOKENS=50000000" \
    bash training/launch_3b.sh
log "launch_3b.sh returned"

log "queue_r210 done; launching queue_r29_scaling"
nohup bash "$REPO/training/queue_r29_scaling.sh" \
    >>/tmp/queue_r29_scaling.log 2>&1 </dev/null &
disown $!
