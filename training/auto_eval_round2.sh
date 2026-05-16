#!/usr/bin/env bash
# auto_eval_round2.sh
#
# Watcher that runs on kebab-spark.lan. Waits for round-2 training to print
# "Training complete." in its log, then triggers:
#   1. 4-rank torchrun consolidate_ckpt.py to merge sharded shards into a
#      single full-state-dict file on spark's local disk
#   2. depth_extrap.py on spark to run the FineWeb-Edu + GSM8K probes at
#      n_loops in {4, 8, 16, 32}, ACT-on and ACT-off
#
# Output:
#   /tmp/auto_eval.log         per-step progress
#   ~/OpenMythos/checkpoints_3b_varT_fast/step_NNNNNNN_full.pt   consolidated
#   ~/OpenMythos/docs/depth_extrap_round2.json   results
#
# Launch on spark with:
#   ssh kebab-spark.lan "nohup bash ~/OpenMythos/training/auto_eval_round2.sh \\
#                       >/tmp/auto_eval.log 2>&1 </dev/null & disown; echo \$!"

set -uo pipefail

LOG=/home/alexm/OpenMythos/training/auto_eval_round2.log
TRAIN_LOG=/tmp/train_r0.log
CKPT_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_fast
RESULTS_JSON=/home/alexm/OpenMythos/docs/depth_extrap_round2.json
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'
PORT=29555

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "auto_eval_round2 watcher started; tailing $TRAIN_LOG for 'Training complete.'"
tail -F -n 0 "$TRAIN_LOG" 2>/dev/null | grep -m 1 'Training complete\.'
log "training completion detected"

# Find the latest sharded ckpt's step number on spark (rank 0's local disk).
LATEST=$(ls "$CKPT_DIR"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$LATEST" ]; then
  log "ERROR: no sharded rank-0 checkpoint found in $CKPT_DIR; aborting"
  exit 1
fi
STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
SRC_PATTERN="$CKPT_DIR/step_${STEP}"
DST_PATH="$CKPT_DIR/step_${STEP}_full.pt"
log "final ckpt step=${STEP}; src=${SRC_PATTERN}_rankN.pt; dst=${DST_PATH}"

# Per-node GID detection (RoCE v2 IPv4 GID index can differ by node).
get_gid() {
  ssh -q "alexm@$1" "show_gids rocep1s0f1 | awk '\$7==\"bond0\" && \$6==\"v2\" && \$5 ~ /^192\\.168\\.100\\./ {print \$3; exit}'"
}

log "launching 4-rank torchrun consolidate_ckpt.py on cluster (port ${PORT})"
# NB: using *-200g hostnames (resolve via /etc/hosts to the storage-net IPs)
# rather than kebab-*.lan hostnames. Spark's /etc/hosts has the wrong entry for
# kebab-gx10-2.lan (resolves to NAS), so SSH from spark to gx10-2 by .lan name
# fails silently. The -200g aliases are correct on all 4 nodes.
RANK=0
for HOST in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  GID=$(get_gid "$HOST")
  log "  rank ${RANK} on ${HOST} (GID=${GID})"
  ssh -q -o StrictHostKeyChecking=accept-new "alexm@$HOST" "rm -f /tmp/consolidate_r${RANK}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${GID} torchrun --nnodes=4 --nproc_per_node=1 --node_rank=${RANK} --master_addr=192.168.100.10 --master_port=${PORT} training/consolidate_ckpt.py ${SRC_PATTERN} ${DST_PATH} >/tmp/consolidate_r${RANK}.log 2>&1 </dev/null & disown; echo r${RANK}=\$!" &
  RANK=$((RANK+1))
done
wait
log "all 4 consolidate ranks launched; waiting for ${DST_PATH} to appear"

# Block until rank 0 finishes writing the full-state-dict file. Cap at 30 min;
# typical wall-clock for the gather+save is 1-3 min on the 200G fabric.
DEADLINE=$(($(date +%s) + 1800))
while [ ! -s "$DST_PATH" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "ERROR: consolidation timed out after 30 min; aborting"
    log "rank 0 log tail:"
    tail -n 40 /tmp/consolidate_r0.log 2>&1 | tee -a "$LOG"
    exit 1
  fi
  sleep 5
done
log "consolidated checkpoint written: $(ls -lh $DST_PATH | awk '{print $5}') bytes"

# Wait for any cluster-wide ranks still running to exit cleanly.
sleep 10

log "running depth_extrap.py on spark"
cd /home/alexm/OpenMythos
CKPT="$DST_PATH" OUT="$RESULTS_JSON" python3 training/depth_extrap.py 2>&1 | tee -a "$LOG"

SAMPLES_TXT=/home/alexm/OpenMythos/docs/gen_samples_round2.txt
log "running gen_samples.py on spark"
CKPT="$DST_PATH" OUT="$SAMPLES_TXT" python3 training/gen_samples.py 2>&1 | tee -a "$LOG"

MULTI_DEPTH_TXT=/home/alexm/OpenMythos/docs/gen_samples_round2_multidepth.txt
log "running gen_samples_multidepth.py on spark"
CKPT="$DST_PATH" OUT="$MULTI_DEPTH_TXT" python3 training/gen_samples_multidepth.py 2>&1 | tee -a "$LOG"

ACT_HIST_JSON=/home/alexm/OpenMythos/docs/act_halt_histogram_round2.json
log "running act_halt_histogram.py on spark"
CKPT="$DST_PATH" OUT="$ACT_HIST_JSON" python3 training/act_halt_histogram.py 2>&1 | tee -a "$LOG"

log "running training_curves.py on spark"
python3 training/training_curves.py 2>&1 | tee -a "$LOG"

log "running postprocess_round2.py on spark"
python3 training/postprocess_round2.py 2>&1 | tee -a "$LOG"

log "auto_eval pipeline complete"
log "  results JSON:    ${RESULTS_JSON}"
log "  samples:         ${SAMPLES_TXT}"
log "  multi-depth:     ${MULTI_DEPTH_TXT}"
log "  act halt hist:   ${ACT_HIST_JSON}"
log "  loss curves:     /home/alexm/OpenMythos/docs/training_curve.png"
log "  loss by T:       /home/alexm/OpenMythos/docs/loss_by_T.png"
log "  comparison:      /home/alexm/OpenMythos/docs/round1_vs_round2.md"
log "  journal:         /home/alexm/OpenMythos/docs/first_cluster_training_run.md"
