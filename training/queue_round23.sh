#!/bin/bash
# Wait for round 2.2 + its auto-eval pipeline to finish, then fire round 2.3
# (joint backbone+head training with PonderNet KL regulariser) plus its
# own auto-eval watcher.
#
# Trigger criteria for round 2.3 launch:
#   1. The round-2.2 auto-eval log shows "auto_eval_round22 pipeline complete"
#      OR the round-2.2 consolidated full-state-dict exists.
#   2. No python3 / torchrun training procs are running on any of the 4 nodes.
#   3. The bootstrap source (round-2.2 final consolidated ckpt) exists on
#      every cluster node.
set -uo pipefail

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
NODES_LAN="kebab-spark.lan kebab-gx10.lan kebab-gx10-2.lan kebab-gx10-3.lan"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
ROUND22_AUTOEVAL_LOG=/tmp/auto_eval_round22.log
ROUND22_DST_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_act_v3

log "queue_round23 started; waiting for round-2.2 auto-eval pipeline to complete"

# Step 1: wait for the auto_eval_round22 pipeline-complete line. (We know the
# auto_eval is running on spark in background.)
while true; do
  if grep -q "auto_eval_round22 pipeline complete" "$ROUND22_AUTOEVAL_LOG" 2>/dev/null; then
    log "auto-eval-22 pipeline-complete line seen"
    break
  fi
  if ls "$ROUND22_DST_DIR"/step_*_full.pt 2>/dev/null | grep -q .; then
    log "round-2.2 consolidated full-state-dict exists; treating as complete"
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

# Step 3: distribute the round-2.2 consolidated ckpt to all 4 nodes (only
# rank 0 wrote it; other ranks need a local copy for the bootstrap path).
LATEST=$(ls /home/alexm/OpenMythos/checkpoints_3b_varT_act_v3/step_*_full.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$LATEST" ]; then
  log "ERROR: no round-2.2 full ckpt found on spark; cannot bootstrap round 2.3"
  exit 1
fi
log "distributing round-2.2 final ckpt: $LATEST"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_act_v3"
  rsync -a "$LATEST" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_act_v3/" &
done
wait
log "ckpt distributed to all worker nodes"

# Step 4: fire the round-2.3 auto-eval watcher in background. It blocks on
# tail -F /tmp/train_r0.log waiting for "Training complete." and then runs
# the consolidate + eval pipeline.
log "starting auto_eval_round23 watcher in background"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round23.sh \
  >/tmp/auto_eval_round23.log 2>&1 </dev/null &
disown $!
log "auto_eval_round23 watcher PID=$!"

# Step 5: fire 4-node round-2.3 joint training
log "firing 4-node round-2.3 joint training"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_pondernet_joint.py bash training/launch_3b.sh
log "launch_3b.sh returned"
