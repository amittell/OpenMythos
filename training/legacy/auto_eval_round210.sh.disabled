#!/usr/bin/env bash
# auto_eval_round210.sh
# Watcher for round 2.10 (joint PonderNet-KL continuation of r2.6).
# Waits for "Training complete." in /tmp/train_r0.log, consolidates,
# then runs the standard eval pipeline.

set -uo pipefail

LOG=/home/alexm/OpenMythos/training/auto_eval_round210.log
TRAIN_LOG=/tmp/train_r0.log
CKPT_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_round210
DOCS_DIR=/home/alexm/OpenMythos/docs
PAPER_FIG_DIR=$DOCS_DIR/paper/figures
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'
PORT=29611

mkdir -p "$PAPER_FIG_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "auto_eval_round210 watcher started; tailing $TRAIN_LOG for 'Training complete.'"
tail -F -n 0 "$TRAIN_LOG" 2>/dev/null | grep -m 1 'Training complete\.'
log "round 2.10 completion detected"

LATEST=$(ls "$CKPT_DIR"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
[ -z "$LATEST" ] && { log "ERROR: no rank-0 shard in $CKPT_DIR; aborting"; exit 1; }
STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
SRC_PATTERN="$CKPT_DIR/step_${STEP}"
DST_PATH="$CKPT_DIR/step_${STEP}_full.pt"
log "round 2.10 final step=${STEP}"

get_gid() {
    ssh -q "alexm@$1" "show_gids rocep1s0f1 | awk '\$7==\"bond0\" && \$6==\"v2\" && \$5 ~ /^192\\.168\\.100\\./ {print \$3; exit}'"
}

log "launching 4-rank consolidate_ckpt.py (port ${PORT})"
RANK=0
for HOST in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    GID=$(get_gid "$HOST")
    ssh -q "alexm@$HOST" \
        "rm -f /tmp/consolidate_round210_r${RANK}.log; cd ~/OpenMythos && \
         nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${GID} \
         torchrun --nnodes=4 --nproc_per_node=1 --node_rank=${RANK} \
                  --master_addr=192.168.100.10 --master_port=${PORT} \
                  training/consolidate_ckpt.py ${SRC_PATTERN} ${DST_PATH} \
           >/tmp/consolidate_round210_r${RANK}.log 2>&1 </dev/null & disown" &
    RANK=$((RANK+1))
done
wait
log "consolidate ranks launched; waiting for ${DST_PATH}"

DEADLINE=$(($(date +%s) + 1800))
while [ ! -s "$DST_PATH" ] || [ -f "${DST_PATH}.tmp" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: consolidation timed out"; exit 1; }
    sleep 5
done
SIZE=$(stat -c %s "$DST_PATH" | numfmt --to=iec)
log "consolidated ckpt: ${DST_PATH} (${SIZE})"
sleep 10

cd /home/alexm/OpenMythos

DEPTH_JSON=$DOCS_DIR/depth_extrap_round210.json
log "depth_extrap.py -> ${DEPTH_JSON}"
CKPT="$DST_PATH" OUT="$DEPTH_JSON" python3 training/depth_extrap.py 2>&1 | tee -a "$LOG"

DIAG_JSON=$DOCS_DIR/act_halt_diagnostic_round210.json
log "act_halt_diagnostic.py -> ${DIAG_JSON}"
CKPT="$DST_PATH" OUT="$DIAG_JSON" python3 training/act_halt_diagnostic.py 2>&1 | tee -a "$LOG"

HIST_JSON=$DOCS_DIR/act_halt_histogram_round210.json
log "act_halt_histogram.py -> ${HIST_JSON}"
CKPT="$DST_PATH" OUT="$HIST_JSON" python3 training/act_halt_histogram.py 2>&1 | tee -a "$LOG"

SAMPLES_TXT=$DOCS_DIR/gen_samples_round210_multidepth.txt
log "gen_samples_multidepth.py -> ${SAMPLES_TXT}"
CKPT="$DST_PATH" OUT="$SAMPLES_TXT" python3 training/gen_samples_multidepth.py 2>&1 | tee -a "$LOG"

PAPER_FRAG=$DOCS_DIR/paper/round210_results.md
log "build_paper_section7.py -> ${PAPER_FRAG}"
DEPTH_JSON="$DEPTH_JSON" DIAG_JSON="$DIAG_JSON" HIST_JSON="$HIST_JSON" \
    SAMPLES_TXT="$SAMPLES_TXT" STEP="$STEP" OUT="$PAPER_FRAG" \
    python3 training/build_paper_section7.py 2>&1 | tee -a "$LOG"

log "auto_eval_round210 pipeline complete."
