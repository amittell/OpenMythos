#!/usr/bin/env bash
# auto_eval_round23.sh
#
# Watcher that runs on kebab-spark.lan. Waits for round 2.3 training to print
# "Training complete." in /tmp/train_r0.log, then drives the post-training
# pipeline:
#
#   1. 4-rank torchrun consolidate_ckpt.py to merge sharded shards into a
#      single full-state-dict file.
#   2. depth_extrap.py    FineWeb-Edu + GSM8K + TinyStories CE at
#                          K in {4, 8, 16, 32}, ACT-on and ACT-off
#   3. act_halt_diagnostic.py  per-iteration p_t and cumulative_p
#                              (key for paper §7: did ACT-bypass training
#                              leave the head still saturated, or did
#                              the optimizer drift it?)
#   4. act_halt_histogram.py   per-(batch,position) halt-step histogram
#   5. gen_samples_multidepth.py  samples at K in {4, 8, 16, 32}
#   6. training_curves.py    loss vs step, loss vs T (PNGs renamed to
#                            round23_*.png so round 2 figures survive)
#   7. build_paper_section7.py   fills docs/paper/round23_results.md
#                                with the empirical §7 numbers
#
# Output:
#   /home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint/step_NNNNNNN_full.pt
#   /home/alexm/OpenMythos/docs/depth_extrap_round23.json
#   /home/alexm/OpenMythos/docs/act_halt_diagnostic_round23.json
#   /home/alexm/OpenMythos/docs/act_halt_histogram_round23.json
#   /home/alexm/OpenMythos/docs/gen_samples_round23_multidepth.txt
#   /home/alexm/OpenMythos/docs/paper/figures/round23_training_curve.png
#   /home/alexm/OpenMythos/docs/paper/figures/round23_loss_by_T.png
#   /home/alexm/OpenMythos/docs/paper/round23_results.md
#
# Launch on spark with:
#   ssh kebab-spark.lan "nohup bash ~/OpenMythos/training/auto_eval_round23.sh \\
#                       >/tmp/auto_eval_round23.log 2>&1 </dev/null & disown; echo \$!"

set -uo pipefail

LOG=/home/alexm/OpenMythos/training/auto_eval_round23.log
TRAIN_LOG=/tmp/train_r0.log
CKPT_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint
DOCS_DIR=/home/alexm/OpenMythos/docs
PAPER_FIG_DIR=$DOCS_DIR/paper/figures
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'
PORT=29558

mkdir -p "$PAPER_FIG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ----------------------------------------------------------------------
# 1. Wait for round 2.3 completion
# ----------------------------------------------------------------------
log "auto_eval_round23 watcher started; tailing $TRAIN_LOG for 'Training complete.'"
tail -F -n 0 "$TRAIN_LOG" 2>/dev/null | grep -m 1 'Training complete\.'
log "round 2.3 completion detected"

# ----------------------------------------------------------------------
# 2. Find the latest sharded ckpt; build src/dst paths
# ----------------------------------------------------------------------
LATEST=$(ls "$CKPT_DIR"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$LATEST" ]; then
  log "ERROR: no sharded rank-0 checkpoint found in $CKPT_DIR; aborting"
  exit 1
fi
STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
SRC_PATTERN="$CKPT_DIR/step_${STEP}"
DST_PATH="$CKPT_DIR/step_${STEP}_full.pt"
log "round 2.3 final ckpt step=${STEP}"
log "  src: ${SRC_PATTERN}_rank{0..3}.pt"
log "  dst: ${DST_PATH}"

# Verify all 4 shards exist before launching the collective.
for r in 0 1 2 3; do
  HOST_KEY=( kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g )
  HOST=${HOST_KEY[$r]}
  if ! ssh -q "alexm@$HOST" "test -f $CKPT_DIR/step_${STEP}_rank${r}.pt"; then
    log "ERROR: missing shard step_${STEP}_rank${r}.pt on $HOST; aborting"
    exit 1
  fi
done
log "all 4 shards present"

# ----------------------------------------------------------------------
# 3. 4-rank consolidate
# ----------------------------------------------------------------------
get_gid() {
  ssh -q "alexm@$1" "show_gids rocep1s0f1 | awk '\$7==\"bond0\" && \$6==\"v2\" && \$5 ~ /^192\\.168\\.100\\./ {print \$3; exit}'"
}

log "launching 4-rank torchrun consolidate_ckpt.py (port ${PORT})"
RANK=0
for HOST in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
  GID=$(get_gid "$HOST")
  log "  rank ${RANK} on ${HOST} (GID=${GID})"
  ssh -q -o StrictHostKeyChecking=accept-new "alexm@$HOST" \
    "rm -f /tmp/consolidate_round23_r${RANK}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${GID} \
     torchrun --nnodes=4 --nproc_per_node=1 --node_rank=${RANK} \
              --master_addr=192.168.100.10 --master_port=${PORT} \
              training/consolidate_ckpt.py ${SRC_PATTERN} ${DST_PATH} \
       >/tmp/consolidate_round23_r${RANK}.log 2>&1 </dev/null & disown" &
  RANK=$((RANK+1))
done
wait
log "consolidate ranks launched; waiting for ${DST_PATH}"

DEADLINE=$(($(date +%s) + 1800))   # 30-min cap
while [ ! -s "$DST_PATH" ] || [ -f "${DST_PATH}.tmp" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "ERROR: consolidation timed out after 30 min"
    log "rank 0 log tail:"
    tail -n 40 /tmp/consolidate_round23_r0.log 2>&1 | tee -a "$LOG"
    exit 1
  fi
  sleep 5
done
SIZE=$(stat -c %s "$DST_PATH" | numfmt --to=iec)
log "consolidated ckpt: ${DST_PATH} (${SIZE})"

sleep 10  # let any straggler ranks exit cleanly

# ----------------------------------------------------------------------
# 4. Eval suite (single-GPU, on spark)
# ----------------------------------------------------------------------
cd /home/alexm/OpenMythos

DEPTH_JSON=$DOCS_DIR/depth_extrap_round23.json
log "depth_extrap.py -> ${DEPTH_JSON}"
CKPT="$DST_PATH" OUT="$DEPTH_JSON" python3 training/depth_extrap.py 2>&1 | tee -a "$LOG"

DIAG_JSON=$DOCS_DIR/act_halt_diagnostic_round23.json
log "act_halt_diagnostic.py -> ${DIAG_JSON}"
CKPT="$DST_PATH" OUT="$DIAG_JSON" python3 training/act_halt_diagnostic.py 2>&1 | tee -a "$LOG"

HIST_JSON=$DOCS_DIR/act_halt_histogram_round23.json
log "act_halt_histogram.py -> ${HIST_JSON}"
CKPT="$DST_PATH" OUT="$HIST_JSON" python3 training/act_halt_histogram.py 2>&1 | tee -a "$LOG"

SAMPLES_TXT=$DOCS_DIR/gen_samples_round23_multidepth.txt
log "gen_samples_multidepth.py -> ${SAMPLES_TXT}"
CKPT="$DST_PATH" OUT="$SAMPLES_TXT" python3 training/gen_samples_multidepth.py 2>&1 | tee -a "$LOG"

# training_curves.py uses hardcoded paths; run, then move to round23 names
# inside docs/paper/figures/. Round 2's figures at docs/training_curve.png and
# docs/loss_by_T.png are preserved.
log "training_curves.py (then rename PNGs into paper figures dir)"
python3 training/training_curves.py 2>&1 | tee -a "$LOG"
mv -v "$DOCS_DIR/training_curve.png" "$PAPER_FIG_DIR/round23_training_curve.png" 2>&1 | tee -a "$LOG"
mv -v "$DOCS_DIR/loss_by_T.png" "$PAPER_FIG_DIR/round23_loss_by_T.png" 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------
# 5. Generate the paper §7 fragment
# ----------------------------------------------------------------------
PAPER_FRAG=$DOCS_DIR/paper/round23_results.md
log "build_paper_section7.py -> ${PAPER_FRAG}"
DEPTH_JSON="$DEPTH_JSON" \
  DIAG_JSON="$DIAG_JSON" \
  HIST_JSON="$HIST_JSON" \
  SAMPLES_TXT="$SAMPLES_TXT" \
  STEP="$STEP" \
  OUT="$PAPER_FRAG" \
  python3 training/build_paper_section7.py 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------
# 6. Done
# ----------------------------------------------------------------------
cat <<EOF | tee -a "$LOG"

[$(date '+%F %T')] auto_eval_round23 pipeline complete.

  Consolidated checkpoint: ${DST_PATH} (${SIZE})
  Depth extrap JSON:       ${DEPTH_JSON}
  ACT diagnostic JSON:     ${DIAG_JSON}
  ACT histogram JSON:      ${HIST_JSON}
  Generation samples:      ${SAMPLES_TXT}
  Training curves:         ${PAPER_FIG_DIR}/round23_*.png
  Paper §7 fragment:       ${PAPER_FRAG}

Next step: open ${PAPER_FRAG} and integrate into docs/paper/main.md §7.
EOF
