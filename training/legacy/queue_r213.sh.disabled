#!/usr/bin/env bash
# Round 2.13 = continue r2.10 joint PonderNet-KL training for +50M tokens.
# Tests whether joint training plateaus at 150M or continues improving.
# REINIT_HEAD=0 to preserve the trained halting head from r2.10.
# Fires after auto_eval_round210 pipeline completes.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a /tmp/queue_r213.log; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|consolidate_ckpt|eval_listops|eval_gsm8k|act_halt|gen_samples|synthetic_depth)"
R210_LOG=/home/alexm/OpenMythos/training/auto_eval_round210.log
REPO=/home/alexm/OpenMythos
R210_CKPT=$REPO/checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt

log "queue_r213 started; waiting for auto_eval_round210 pipeline complete"
DEADLINE=$(($(date +%s) + 24 * 3600))
while true; do
    grep -q "auto_eval_round210 pipeline complete" "$R210_LOG" 2>/dev/null && { log "r210 done"; break; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 24h deadline"; exit 1; }
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

[ ! -f "$R210_CKPT" ] && { log "ERROR: r2.10 ckpt missing at $R210_CKPT"; exit 1; }
log "verifying r2.10 ckpt on workers"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q "alexm@$h" "test -f $R210_CKPT" && log "  $h: present" || {
        log "  $h: rsyncing r2.10 ckpt"
        ssh -q "alexm@$h" "mkdir -p $REPO/checkpoints_3b_varT_pondernet_round210"
        rsync -a "$R210_CKPT" "alexm@$h:$R210_CKPT"
    }
done

log "starting auto_eval_round213 watcher"
nohup bash "$REPO/training/auto_eval_round213.sh" \
    >"$REPO/training/auto_eval_round213.log" 2>&1 </dev/null &
disown $!
log "auto_eval_round213 watcher PID=$!"

log "firing 4-node round-2.13 joint training (r2.10 +50M tokens, REINIT_HEAD=0)"
cd "$REPO"
SCRIPT=training/3b_varT_pondernet_joint.py PORT=29513 \
    EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_round213 BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_round210/step_0003051_full.pt REINIT_HEAD=0 TARGET_TOKENS=50000000" \
    bash training/launch_3b.sh
log "launch_3b.sh returned"

log "queue_r213 done; launching queue_r214"
nohup bash "$REPO/training/queue_r214.sh" \
    >>/tmp/queue_r214.log 2>&1 </dev/null &
disown $!
