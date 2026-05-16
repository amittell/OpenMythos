#!/bin/bash
# Round 2.9 = no-recurrence baseline (T_FIXED=1) from r2 collapsed.
# Compute-matched compute-matched comparison vs r2.4 (PonderNet) and r2.5
# (fixed T=8): same starting point, same 50M tokens, T=1 means the
# recurrent block is applied just once.
#
# Caveat for paper: not a pure vanilla transformer because the
# architecture still has LTI injection and the prelude/coda structure.
# It is "no test-time-compute scaling at matched params and tokens".
#
# Trigger: blocks on auto_eval_round27 pipeline-complete.

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
ROUND27_LOG=/tmp/auto_eval_round27.log
R2_BOOTSTRAP=/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt

log "queue_round29 started; waiting for auto_eval_round27 pipeline-complete"
DEADLINE=$(($(date +%s) + 60 * 3600))
while true; do
  if grep -q "auto_eval_round27 pipeline complete" "$ROUND27_LOG" 2>/dev/null; then
    log "auto_eval_round27 pipeline-complete seen"; break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then log "ERROR: 60h deadline"; exit 1; fi
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
log "verifying r2 collapsed ckpt on each worker"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  ssh -q "alexm@$h" "test -f $R2_BOOTSTRAP" && log "  $h: present" || {
    log "  $h: rsyncing"
    ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_fast"
    rsync -a "$R2_BOOTSTRAP" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_fast/"
  }
done

log "starting auto_eval_round29 watcher"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round29.sh \
  >/tmp/auto_eval_round29.log 2>&1 </dev/null &
disown $!
log "auto_eval_round29 watcher PID=$!"

log "firing 4-node round-2.9 training (T_FIXED=1 from r2 collapsed)"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_act_v3.py \
  EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_act_v3_round29_T1 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt T_FIXED=1" \
  bash training/launch_3b.sh
log "launch_3b.sh returned"
