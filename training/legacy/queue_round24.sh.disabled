#!/bin/bash
# Wait for round 2.3 + its auto-eval pipeline to finish, then fire round 2.4
# (PonderNet joint training from the round-2 collapsed ckpt). Round 2.4
# tests whether the round-2.1/2.2 bypass interlude was load-bearing or
# whether a freshly-init'd PonderNet head + KL prior alone can rescue a
# collapsed backbone.
#
# Trigger criteria for round 2.4 launch:
#   1. The round-2.3 auto-eval log shows "auto_eval_round23 pipeline complete"
#      OR the round-2.3 consolidated full-state-dict exists.
#   2. No python3 / torchrun training procs are running on any of the 4 nodes.
#   3. The bootstrap source (round-2 collapsed ckpt
#      checkpoints_3b_varT_fast/step_0012207_full.pt) exists on every
#      cluster node.
set -uo pipefail

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
NODES_LAN="kebab-spark.lan kebab-gx10.lan kebab-gx10-2.lan kebab-gx10-3.lan"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
ROUND23_AUTOEVAL_LOG=/tmp/auto_eval_round23.log
ROUND23_DST_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint
ROUND2_BOOTSTRAP=/home/alexm/OpenMythos/checkpoints_3b_varT_fast/step_0012207_full.pt

log "queue_round24 started; waiting for round-2.3 auto-eval pipeline to complete"

# Step 1: wait for the auto_eval_round23 pipeline-complete line.
while true; do
  if grep -q "auto_eval_round23 pipeline complete" "$ROUND23_AUTOEVAL_LOG" 2>/dev/null; then
    log "auto-eval-23 pipeline-complete line seen"
    break
  fi
  if ls "$ROUND23_DST_DIR"/step_*_full.pt 2>/dev/null | grep -q .; then
    log "round-2.3 consolidated full-state-dict exists; treating as complete"
    break
  fi
  sleep 30
done

# Step 2: fence on cluster training procs across all 4 nodes
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

# Step 3: distribute the round-2 collapsed ckpt to all 4 nodes (only the
# original training rank wrote it; verify and rsync if missing).
if [ ! -f "$ROUND2_BOOTSTRAP" ]; then
  log "ERROR: $ROUND2_BOOTSTRAP missing on spark; cannot bootstrap round 2.4"
  exit 1
fi
log "distributing round-2 collapsed ckpt: $ROUND2_BOOTSTRAP"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_fast"
  if ssh -q "alexm@$h" "test -f $ROUND2_BOOTSTRAP"; then
    log "  $h: already has ckpt; skipping"
  else
    log "  $h: rsyncing"
    rsync -a "$ROUND2_BOOTSTRAP" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_fast/" &
  fi
done
wait
log "ckpt distributed to all worker nodes"

# Step 4: fire the round-2.4 auto-eval watcher in background.
log "starting auto_eval_round24 watcher in background"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round24.sh \
  >/tmp/auto_eval_round24.log 2>&1 </dev/null &
disown $!
log "auto_eval_round24 watcher PID=$!"

# Step 5: fire 4-node round-2.4 joint training. CKPT_DIR isolates round 2.4
# checkpoints from round 2.3's, and BOOTSTRAP_CKPT overrides the script's
# default v3 -> v2 discovery so we start from the round-2 collapsed ckpt.
log "firing 4-node round-2.4 joint training (PonderNet from r2 collapsed)"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_pondernet_joint.py \
  EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_from_r2 BOOTSTRAP_CKPT=checkpoints_3b_varT_fast/step_0012207_full.pt" \
  bash training/launch_3b.sh
log "launch_3b.sh returned"
