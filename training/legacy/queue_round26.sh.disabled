#!/bin/bash
# Round 2.6 = continue r2.3 (joint PonderNet-KL bypass + head) for another
# 50M tokens. Tests whether the joint equilibrium keeps gaining CE past
# 50M tokens or saturates.
#
# Bootstrap: r2.3 final at checkpoints_3b_varT_pondernet_joint/step_0003051_full.pt
# Output:    checkpoints_3b_varT_pondernet_round26/
# Hyperparams: lambda_kl=1.0, lambda_p=0.2 (same as r2.3), REINIT_HEAD=0
# (preserve trained head).

set -uo pipefail
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
PROC_PATTERN="python3 training/(3b_varT|reasoning_eval|depth_extrap|per_token_halt|synthetic_depth|act_finetune|consolidate_ckpt)"
R23_BOOTSTRAP=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint/step_0003051_full.pt

log "queue_round26 started; fencing on cluster procs"
while true; do
  any=0
  for h in $NODES_200G; do
    if ssh -q -o ConnectTimeout=5 alexm@$h "pgrep -f '$PROC_PATTERN' >/dev/null 2>&1"; then any=1; break; fi
  done
  [ $any -eq 0 ] && break
  sleep 30
done
log "fence cleared"

[ ! -f "$R23_BOOTSTRAP" ] && { log "ERROR: $R23_BOOTSTRAP missing"; exit 1; }
log "distributing r2.3 final ckpt (16 GB) to worker nodes"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  ssh -q "alexm@$h" "mkdir -p /home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint"
  if ssh -q "alexm@$h" "test -f $R23_BOOTSTRAP"; then
    log "  $h: present"
  else
    log "  $h: rsyncing"
    rsync -a "$R23_BOOTSTRAP" "alexm@$h:/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint/" &
  fi
done
wait
log "r2.3 ckpt distributed"

log "starting auto_eval_round26 watcher"
nohup bash /home/alexm/OpenMythos/training/auto_eval_round26.sh \
  >/tmp/auto_eval_round26.log 2>&1 </dev/null &
disown $!
log "auto_eval_round26 watcher PID=$!"

log "firing 4-node round-2.6 training (joint PonderNet-KL continue from r2.3)"
cd /home/alexm/OpenMythos
SCRIPT=training/3b_varT_pondernet_joint.py \
  EXTRA_ENV="CKPT_DIR=checkpoints_3b_varT_pondernet_round26 BOOTSTRAP_CKPT=checkpoints_3b_varT_pondernet_joint/step_0003051_full.pt REINIT_HEAD=0 LAMBDA_P=0.2 LAMBDA_KL=1.0" \
  bash training/launch_3b.sh
log "launch_3b.sh returned"
