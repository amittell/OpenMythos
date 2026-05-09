#!/usr/bin/env bash
# auto_eval_round213.sh
#
# Watcher that runs on kebab-spark.lan. Waits for round 2.13 training
# (joint PonderNet-KL +50M from r2.10, total 200M joint tokens) to print
# "Training complete." in /tmp/train_r0.log, then drives the post-training
# pipeline:
#
#   1. 4-rank torchrun consolidate_ckpt.py
#   2. depth_extrap.py
#   3. act_halt_diagnostic.py
#   4. act_halt_histogram.py
#   5. reasoning_eval.py     (head was trained, expect non-trivial results)
#   6. gen_samples_multidepth.py
#   7. build_paper_section7.py -> docs/paper/round213_results.md

set -uo pipefail

LOG=/home/alexm/OpenMythos/training/auto_eval_round213.log
TRAIN_LOG=/tmp/train_r0.log
CKPT_DIR=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_round213
DOCS_DIR=/home/alexm/OpenMythos/docs
PAPER_FIG_DIR=$DOCS_DIR/paper/figures
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=420 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'
PORT=29613

mkdir -p "$PAPER_FIG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "auto_eval_round213 watcher started; checking $TRAIN_LOG for 'Training complete.'"
if grep -q 'Training complete\.' "$TRAIN_LOG" 2>/dev/null; then
    log "Training already complete; skipping wait"
else
    log "tailing $TRAIN_LOG for 'Training complete.'"
    tail -F -n 0 "$TRAIN_LOG" 2>/dev/null | grep -m 1 'Training complete\.'
fi
log "round 2.13 completion detected"

LATEST=$(ls "$CKPT_DIR"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$LATEST" ]; then
    log "ERROR: no sharded rank-0 checkpoint found in $CKPT_DIR; aborting"
    exit 1
fi
STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
SRC_PATTERN="$CKPT_DIR/step_${STEP}"
DST_PATH="$CKPT_DIR/step_${STEP}_full.pt"
log "round 2.13 final ckpt step=${STEP}"

for r in 0 1 2 3; do
    HOST_KEY=( kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g )
    HOST=${HOST_KEY[$r]}
    if ! ssh -q "alexm@$HOST" "test -f $CKPT_DIR/step_${STEP}_rank${r}.pt"; then
        log "ERROR: missing shard step_${STEP}_rank${r}.pt on $HOST; aborting"
        exit 1
    fi
done
log "all 4 shards present"

get_gid() {
    ssh -q "alexm@$1" "show_gids rocep1s0f1 | awk '\$7==\"bond0\" && \$6==\"v2\" && \$5 ~ /^192\\.168\\.100\\./ {print \$3; exit}'"
}

log "launching 4-rank torchrun consolidate_ckpt.py (port ${PORT})"
RANK=0
for HOST in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    GID=$(get_gid "$HOST")
    log "  rank ${RANK} on ${HOST} (GID=${GID})"
    ssh -q -o StrictHostKeyChecking=accept-new "alexm@$HOST" \
        "rm -f /tmp/consolidate_round213_r${RANK}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${GID} \
         torchrun --nnodes=4 --nproc_per_node=1 --node_rank=${RANK} \
                  --master_addr=192.168.100.10 --master_port=${PORT} \
                  training/consolidate_ckpt.py ${SRC_PATTERN} ${DST_PATH} \
           >/tmp/consolidate_round213_r${RANK}.log 2>&1 </dev/null & disown" &
    RANK=$((RANK+1))
done
wait
log "consolidate ranks launched; waiting for ${DST_PATH}"

DEADLINE=$(($(date +%s) + 1800))
while [ ! -s "$DST_PATH" ] || [ -f "${DST_PATH}.tmp" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        log "ERROR: consolidation timed out after 30 min"
        tail -n 40 /tmp/consolidate_round213_r0.log 2>&1 | tee -a "$LOG"
        exit 1
    fi
    sleep 5
done
SIZE=$(stat -c %s "$DST_PATH" | numfmt --to=iec)
log "consolidated ckpt: ${DST_PATH} (${SIZE})"

sleep 10

cd /home/alexm/OpenMythos

# ---------------------------------------------------------------------------
# Parallel evals across all 4 cluster nodes:
#   spark (rank 0): depth_extrap + gen_samples (lightweight, can run two)
#   gx10-200g:      act_halt_diagnostic
#   gx10-2-200g:    act_halt_histogram
#   gx10-3-200g:    reasoning_eval (slowest)
# ---------------------------------------------------------------------------

DEPTH_JSON=$DOCS_DIR/depth_extrap_round213.json
DIAG_JSON=$DOCS_DIR/act_halt_diagnostic_round213.json
HIST_JSON=$DOCS_DIR/act_halt_histogram_round213.json
REASONING_JSON=$DOCS_DIR/reasoning_eval_round213.json
SAMPLES_TXT=$DOCS_DIR/gen_samples_round213_multidepth.txt

log "rsyncing consolidated ckpt to workers (parallel, over 200G storage net)"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q "alexm@$h" "mkdir -p $(dirname $DST_PATH)" &
done
wait
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    rsync -a "$DST_PATH" "alexm@$h:$DST_PATH" 2>&1 | tee -a "$LOG" &
done
wait
log "ckpt distributed to all workers"

log "launching parallel evals across 4 nodes"
PIDS=()

# rank 0 / spark: depth_extrap + gen_samples (sequential on spark, both small)
(
    log "[rank0] depth_extrap.py -> $DEPTH_JSON"
    CKPT="$DST_PATH" OUT="$DEPTH_JSON" python3 training/depth_extrap.py 2>&1 \
        | tee -a "$DOCS_DIR/.eval_depth_round213.log"
    log "[rank0] gen_samples_multidepth.py -> $SAMPLES_TXT"
    CKPT="$DST_PATH" OUT="$SAMPLES_TXT" python3 training/gen_samples_multidepth.py 2>&1 \
        | tee -a "$DOCS_DIR/.eval_samples_round213.log"
    log "[rank0] depth+samples done"
) &
PIDS+=($!)

# rank 1 / gx10-200g: act_halt_diagnostic
(
    log "[gx10-200g] act_halt_diagnostic.py -> $DIAG_JSON"
    ssh -q "alexm@kebab-gx10-200g" "cd /home/alexm/OpenMythos && \
        CKPT='$DST_PATH' OUT='$DIAG_JSON' python3 training/act_halt_diagnostic.py 2>&1" \
        > "$DOCS_DIR/.eval_diag_round213.log" 2>&1
    rsync -a "alexm@kebab-gx10-200g:$DIAG_JSON" "$DOCS_DIR/" 2>&1 \
        | tee -a "$LOG"
    log "[gx10-200g] act_halt_diagnostic done"
) &
PIDS+=($!)

# rank 2 / gx10-2-200g: act_halt_histogram
(
    log "[gx10-2-200g] act_halt_histogram.py -> $HIST_JSON"
    ssh -q "alexm@kebab-gx10-2-200g" "cd /home/alexm/OpenMythos && \
        CKPT='$DST_PATH' OUT='$HIST_JSON' python3 training/act_halt_histogram.py 2>&1" \
        > "$DOCS_DIR/.eval_hist_round213.log" 2>&1
    rsync -a "alexm@kebab-gx10-2-200g:$HIST_JSON" "$DOCS_DIR/" 2>&1 \
        | tee -a "$LOG"
    log "[gx10-2-200g] act_halt_histogram done"
) &
PIDS+=($!)

# rank 3 / gx10-3-200g: reasoning_eval (slowest, longest pole)
(
    log "[gx10-3-200g] reasoning_eval.py -> $REASONING_JSON (longest)"
    ssh -q "alexm@kebab-gx10-3-200g" "cd /home/alexm/OpenMythos && \
        CKPT='$DST_PATH' OUT='$REASONING_JSON' python3 training/reasoning_eval.py 2>&1" \
        > "$DOCS_DIR/.eval_reasoning_round213.log" 2>&1
    rsync -a "alexm@kebab-gx10-3-200g:$REASONING_JSON" "$DOCS_DIR/" 2>&1 \
        | tee -a "$LOG"
    log "[gx10-3-200g] reasoning_eval done"
) &
PIDS+=($!)

log "waiting on ${#PIDS[@]} parallel eval jobs (PIDs: ${PIDS[*]})"
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
log "all parallel evals complete"

# Clean up the distributed ckpt copies on workers (16 GB each)
log "cleaning up worker ckpt copies"
for h in kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q "alexm@$h" "rm -f $DST_PATH" &
done
wait

cat <<EOF | tee -a "$LOG"

[$(date '+%F %T')] auto_eval_round213 pipeline complete.

  Consolidated checkpoint: ${DST_PATH} (${SIZE})
  Depth extrap JSON:       ${DEPTH_JSON}
  ACT diagnostic JSON:     ${DIAG_JSON}
  ACT histogram JSON:      ${HIST_JSON}
  Reasoning eval JSON:     ${REASONING_JSON}
  Generation samples:      ${SAMPLES_TXT}

Round 2.13 = continuation of r2.10 for +50M tokens (200M joint training total).
Tests whether joint PonderNet-KL training plateaus at 150M or continues
improving. The 4th data point on the joint-token scaling curve.
EOF
