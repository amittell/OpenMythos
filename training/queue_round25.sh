#!/bin/bash
# Queue round 2.5 = FIXED T=8 from r2 collapsed ckpt. Symmetric ablation
# to round 2.4 (variable-T PonderNet from same starting point); tests
# whether PonderNet KL halting beats fixed-depth training at matched
# compute and matched bootstrap.
#
# Trigger: after "auto_eval_round24 pipeline complete" appears in
# /tmp/auto_eval_round24.log (NOT on the ckpt-exists fallback that
# burned us with round 2.3 -- this fixes that race).
#
# Round 2.5 settings:
#   script        training/3b_varT_act_v3.py (bypass-only forward)
#   bootstrap     checkpoints_3b_varT_fast/step_0012207_full.pt (r2 final)
#   T_FIXED       8
#   CKPT_DIR      checkpoints_3b_varT_act_v3_round25_fixedT8
#   target        50M tokens (same as r2.1/2.2/2.3/2.4)

set -uo pipefail

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
ROUND24_AUTOEVAL_LOG=/tmp/auto_eval_round24.log
R2_BOOTSTRAP=/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt

log "queue_round25 started; waiting for round-2.4 auto-eval pipeline-complete line"

# Step 1: ONLY trigger on the explicit pipeline-complete line. No
# ckpt-exists fallback (that's what caused the round23 race). 12h max.
DEADLINE=$(($(date +%s) + 43200))
while true; do
  if grep -q "auto_eval_round24 pipeline complete" "$ROUND24_AUTOEVAL_LOG" 2>/dev/null; then
    log "auto-eval-24 pipeline-complete line seen"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "ERROR: 12h deadline elapsed without round-2.4 pipeline-complete; aborting"
    exit 1
  fi
  sleep 60
done

# Step 2: fence on cluster training/eval procs
log "fence: waiting for all training/eval procs across all 4 nodes to exit"
while true; do
  any_busy=0
  for h in $NODES_200G; do
    if ssh -q -o ConnectTimeout=5 alexm@$h "pgrep -f '$PROC_PATTERN' >/dev/null 2>&1"; then
      any_busy=1
      break
    fi
  done
  [ $any_busy -eq 0 ] && break
  sleep 30
done
log "fence cleared"

# Step 3: bootstrap ckpt should already be on all 4 nodes (queue_round24
# distributed it). Verify; rsync any missing.
if [ ! -f "$R2_BOOTSTRAP" ]; then
  log "ERROR: $R2_BOOTSTRAP missing on spark; cannot bootstrap round 2.5"
  exit 1
fi
log "verifying r2 collapsed ckpt on each worker node"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  if ssh -q "alexm@$h" "test -f $R2_BOOTSTRAP"; then
    log "  $h: ckpt present"
  else
    log "  $h: rsyncing"
    ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_fast"
    rsync -a "$R2_BOOTSTRAP" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_fast/"
  fi
done

# Step 4: fire the round-2.5 auto-eval watcher
log "starting auto_eval_round25 watcher in background"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round25.sh \
  >/tmp/auto_eval_round25.log 2>&1 </dev/null &
disown $!
log "auto_eval_round25 watcher PID=$!"

# Step 5: launch 4-node round-2.5 training. v3 script in bypass-only
# mode, T fixed at 8, bootstrap from r2 collapsed, fresh ckpt dir.
log "firing 4-node round-2.5 training (fixed T=8 from r2 collapsed)"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_act_v3.py \
  EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_act_v3_round25_fixedT8 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt T_FIXED=8" \
  bash training/launch_3b.sh
log "launch_3b.sh returned"
