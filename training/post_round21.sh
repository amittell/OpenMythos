#!/usr/bin/env bash
# Post-round-2.1 pipeline. Runs from any host that can SSH the cluster.
#
# What it does:
#   1. Polls rank 0's training log on spark for "Training complete."
#   2. Discovers the latest sharded checkpoint step in
#      checkpoints_3b_varT_act_v2/ on spark.
#   3. Launches a 4-rank torchrun across the cluster to consolidate the
#      sharded shards into a single full-state-dict (`step_NNNNNN_full.pt`).
#   4. Verifies the consolidated file landed on spark.
#
# What it does NOT do:
#   - Auto-launch round 2.2. That choice (conservative vs aggressive
#     settings) is intentionally left to a human review. Use
#     `training/launch_round22.sh {conservative|aggressive}` once this
#     script reports success.
#
# Usage:
#   bash training/post_round21.sh                    # default poll cadence
#   POLL=120 bash training/post_round21.sh           # custom poll secs
#   nohup bash training/post_round21.sh > /tmp/post_r21.log 2>&1 &
#
# Exits non-zero on consolidation failure or unrecoverable infra error.

set -euo pipefail

POLL=${POLL:-60}
PORT=${PORT:-29555}                 # different from training's 29510
MASTER=192.168.100.10
SRC_DIR=${SRC_DIR:-checkpoints_3b_varT_act_v2}

# Map node_rank -> 200g hostname (mirrors launch_3b.sh's fixed map).
declare -A NODE_HOST=(
  [0]=kebab-spark-200g
  [1]=kebab-gx10-200g
  [2]=kebab-gx10-2-200g
  [3]=kebab-gx10-3-200g
)

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ----------------------------------------------------------------------
# 1. Wait for round 2.1 completion
# ----------------------------------------------------------------------
log "polling spark:/tmp/train_r0.log for 'Training complete.' (every ${POLL}s)"
while true; do
  if ssh -q alexm@kebab-spark.lan \
       'grep -q "Training complete." /tmp/train_r0.log 2>/dev/null'; then
    break
  fi
  sleep "$POLL"
done
log "round 2.1 finished"

# ----------------------------------------------------------------------
# 2. Find the latest sharded step on spark
# ----------------------------------------------------------------------
LATEST_STEP=$(ssh -q alexm@kebab-spark.lan \
  "cd OpenMythos && ls -1 ${SRC_DIR}/step_*_rank0.pt 2>/dev/null \
   | sed 's|.*/step_||;s|_rank0.pt||' | sort -n | tail -1")

if [[ -z "${LATEST_STEP}" ]]; then
  log "ERROR: no sharded checkpoints found in spark:OpenMythos/${SRC_DIR}/"
  exit 2
fi

SRC_PREFIX="${SRC_DIR}/step_${LATEST_STEP}"
DST_PATH="${SRC_DIR}/step_${LATEST_STEP}_full.pt"
log "consolidating step ${LATEST_STEP}: ${SRC_PREFIX}_rank{0..3}.pt -> ${DST_PATH}"

# ----------------------------------------------------------------------
# 3. Verify all 4 shards exist before launching the collective
# ----------------------------------------------------------------------
for r in 0 1 2 3; do
  host=${NODE_HOST[$r]}
  if ! ssh -q alexm@$host \
        "test -f OpenMythos/${SRC_PREFIX}_rank${r}.pt"; then
    log "ERROR: missing shard ${SRC_PREFIX}_rank${r}.pt on $host"
    exit 3
  fi
done
log "all 4 shards present"

# ----------------------------------------------------------------------
# 4. Launch the 4-rank consolidate torchrun
# ----------------------------------------------------------------------
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'

get_gid() {
  ssh -q alexm@$1 bash -s << 'EOF'
show_gids rocep1s0f1 | awk '$7=="bond0" && $6=="v2" && $5 ~ /^192\.168\.100\./ {print $3; exit}'
EOF
}

# Kill any straggler consolidate or torchrun procs left over from a
# previous attempt. (Training procs are already gone since 2.1 completed.)
for r in 0 1 2 3; do
  ssh -q alexm@${NODE_HOST[$r]} \
    'pkill -9 -f "consolidate_ckpt" 2>/dev/null || true' &
done
wait

# Launch each rank. Output goes to /tmp/consolidate_rN.log per node.
log "launching 4-rank consolidate (master ${MASTER}:${PORT})"
for r in 0 1 2 3; do
  host=${NODE_HOST[$r]}
  gid=$(get_gid "$host")
  ssh -q alexm@$host \
    "rm -f /tmp/consolidate_r${r}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${gid} \
     torchrun --nnodes=4 --nproc_per_node=1 --node_rank=$r \
              --master_addr=${MASTER} --master_port=${PORT} \
              training/consolidate_ckpt.py ${SRC_PREFIX} ${DST_PATH} \
       >/tmp/consolidate_r${r}.log 2>&1 </dev/null & disown" &
done
wait
log "consolidate ranks launched; waiting for completion"

# ----------------------------------------------------------------------
# 5. Wait for the consolidated file to appear on spark
# ----------------------------------------------------------------------
DEADLINE=$(( $(date +%s) + 30 * 60 ))   # 30 min cap
while true; do
  if ssh -q alexm@kebab-spark.lan \
       "test -f OpenMythos/${DST_PATH} && \
        ! test -f OpenMythos/${DST_PATH}.tmp"; then
    break
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    log "ERROR: consolidate timed out after 30 min"
    log "rank 0 log tail:"
    ssh -q alexm@kebab-spark.lan 'tail -40 /tmp/consolidate_r0.log' || true
    exit 4
  fi
  sleep 30
done

SIZE=$(ssh -q alexm@kebab-spark.lan \
  "stat -c %s OpenMythos/${DST_PATH} | numfmt --to=iec")
log "consolidated -> spark:OpenMythos/${DST_PATH} (${SIZE})"

# ----------------------------------------------------------------------
# 6. Done — summary for human review
# ----------------------------------------------------------------------
cat <<EOF

[$(ts)] Round 2.1 post-pipeline complete.

  Consolidated checkpoint: spark:OpenMythos/${DST_PATH}
  Size: ${SIZE}
  Step: ${LATEST_STEP}

Next step (human-driven):
  conservative defaults:
    SCRIPT=training/3b_varT_act_v3.py bash training/launch_round22.sh conservative
  aggressive (USE_ACT_CKPT=1 MICRO_BATCH=2 GRAD_ACCUM=2):
    SCRIPT=training/3b_varT_act_v3.py bash training/launch_round22.sh aggressive

EOF
