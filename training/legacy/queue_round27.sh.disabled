#!/bin/bash
# Round 2.7 = PonderNet-KL joint training from r2 collapsed with
# lambda_p=0.1 (target mean halt step 10). Tests whether the model
# adapts to a longer halt budget or stays at the geometric mean of
# whatever lambda_p we set.
#
# Triggers on "auto_eval_round26 pipeline complete" line in
# /tmp/auto_eval_round26.log. 24h max wait.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
ROUND26_LOG=/tmp/auto_eval_round26.log
R2_BOOTSTRAP=/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt

log "queue_round27 started; waiting for auto_eval_round26 pipeline-complete"
DEADLINE=$(($(date +%s) + 86400))
while true; do
  if grep -q "auto_eval_round26 pipeline complete" "$ROUND26_LOG" 2>/dev/null; then
    log "auto_eval_round26 pipeline-complete seen"; break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then log "ERROR: 24h deadline"; exit 1; fi
  sleep 60
done

log "fencing on cluster procs"
while true; do
  any=0
  for h in $NODES_200G; do
    if ssh -q -o ConnectTimeout=5 alexm@$h "pgrep -f '$PROC_PATTERN' >/dev/null 2>&1"; then any=1; break; fi
  done
  [ $any -eq 0 ] && break
  sleep 30
done
log "fence cleared"

[ ! -f "$R2_BOOTSTRAP" ] && { log "ERROR: $R2_BOOTSTRAP missing"; exit 1; }
log "verifying r2 collapsed ckpt on each worker (already distributed in prior rounds)"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  ssh -q "alexm@$h" "test -f $R2_BOOTSTRAP" && log "  $h: present" || {
    log "  $h: rsyncing"
    ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_fast"
    rsync -a "$R2_BOOTSTRAP" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_fast/"
  }
done

log "starting auto_eval_round27 watcher"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round27.sh \
  >/tmp/auto_eval_round27.log 2>&1 </dev/null &
disown $!
log "auto_eval_round27 watcher PID=$!"

log "firing 4-node round-2.7 training (PonderNet-KL lambda_p=0.1 from r2 collapsed)"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_pondernet_joint.py \
  EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_round27_lp01 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt LAMBDA_P=0.1 LAMBDA_KL=1.0" \
  bash training/launch_3b.sh
log "launch_3b.sh returned"
