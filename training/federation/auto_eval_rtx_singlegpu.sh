#!/usr/bin/env bash
# auto_eval_rtx_singlegpu.sh
#
# Watches a single-GPU RTX6000 training for "Training complete." and runs
# the standard evals: depth_extrap, act_halt_diagnostic, act_halt_histogram,
# reasoning_eval. Single-GPU means rank 0's ckpt IS the full state, no
# consolidate_ckpt.py needed.
#
# Usage:
#   ROUND=r211_T2 CKPT_DIR=checkpoints_3b_varT_act_v3_round211_T2 \
#     bash auto_eval_rtx_singlegpu.sh
#
# Required env:
#   ROUND          short name (used in JSON filenames, e.g. "round211" or "r211")
#   CKPT_DIR       checkpoint dir (under /home/alexm/OpenMythos)
#   TRAIN_LOG      path to training stdout log (default: /tmp/${ROUND}_train.log)
#
# Optional:
#   GPU            CUDA_VISIBLE_DEVICES (default: 0)
#   PY             python (default: /home/alexm/venvs/vllm-turboquant/bin/python3)
#   DOCS           docs dir (default: /home/alexm/OpenMythos/docs)
#   POLL_INTERVAL  seconds between training-complete polls (default: 60)

set -uo pipefail
: "${ROUND:?ROUND env required (e.g. round211)}"
: "${CKPT_DIR:?CKPT_DIR env required}"
: "${TRAIN_LOG:=/tmp/${ROUND}_train.log}"
: "${GPU:=0}"
: "${PY:=/home/alexm/venvs/vllm-turboquant/bin/python3}"
: "${DOCS:=/home/alexm/OpenMythos/docs}"
: "${REPO:=/home/alexm/OpenMythos}"
: "${POLL_INTERVAL:=60}"

LOG_DIR=$REPO/training/federation/logs
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/auto_eval_${ROUND}.log
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "auto_eval watcher started for $ROUND (CKPT_DIR=$CKPT_DIR, GPU=$GPU)"
log "  watching: $TRAIN_LOG"

# Wait for "Training complete." in train log. If already there, proceed immediately.
DEADLINE=$(($(date +%s) + 5 * 24 * 3600))  # 5-day cap
while true; do
    if grep -q 'Training complete\.' "$TRAIN_LOG" 2>/dev/null; then
        log "training-complete detected"
        break
    fi
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ERROR: 5-day deadline; aborting"; exit 1; }
    sleep "$POLL_INTERVAL"
done

# Find latest ckpt (single-GPU rank0.pt = full state already)
LATEST=$(ls "$REPO/$CKPT_DIR"/step_*_rank0.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$LATEST" ]; then
    log "ERROR: no ckpt found in $REPO/$CKPT_DIR"
    exit 1
fi
log "latest ckpt: $LATEST"

# Single-GPU FSDP saves ShardedTensor objects in the rank0 shard, which
# downstream eval scripts cannot load via torch.load() without an active
# process group. Run the single-rank consolidator to materialize the
# tensors and emit a plain dict at step_${STEP}_full.pt.
STEP=$(basename "$LATEST" | sed -e 's/^step_//' -e 's/_rank0\.pt$//')
FULL_PATH=$REPO/$CKPT_DIR/step_${STEP}_full.pt
if [ ! -f "$FULL_PATH" ] || [ -L "$FULL_PATH" ]; then
    rm -f "$FULL_PATH"  # remove any stale symlink from earlier broken version
    log "consolidating sharded -> full ckpt for ${ROUND}"
    "$PY" "$REPO/training/consolidate_single_rank.py" "$LATEST" "$FULL_PATH" 2>&1 | tee -a "$LOG"
    if [ ! -s "$FULL_PATH" ]; then
        log "ERROR: consolidator failed to produce $FULL_PATH"
        exit 1
    fi
    log "consolidated: $FULL_PATH ($(stat -c %s "$FULL_PATH" | numfmt --to=iec))"
fi

cd "$REPO"

DEPTH_JSON=$DOCS/depth_extrap_${ROUND}.json
log "depth_extrap.py -> $DEPTH_JSON"
CUDA_VISIBLE_DEVICES=$GPU CKPT="$FULL_PATH" OUT="$DEPTH_JSON" \
    "$PY" training/depth_extrap.py 2>&1 | tee -a "$LOG"

DIAG_JSON=$DOCS/act_halt_diagnostic_${ROUND}.json
log "act_halt_diagnostic.py -> $DIAG_JSON"
CUDA_VISIBLE_DEVICES=$GPU CKPT="$FULL_PATH" OUT="$DIAG_JSON" \
    "$PY" training/act_halt_diagnostic.py 2>&1 | tee -a "$LOG"

HIST_JSON=$DOCS/act_halt_histogram_${ROUND}.json
log "act_halt_histogram.py -> $HIST_JSON"
CUDA_VISIBLE_DEVICES=$GPU CKPT="$FULL_PATH" OUT="$HIST_JSON" \
    "$PY" training/act_halt_histogram.py 2>&1 | tee -a "$LOG"

REASONING_JSON=$DOCS/reasoning_eval_${ROUND}.json
log "reasoning_eval.py -> $REASONING_JSON"
CUDA_VISIBLE_DEVICES=$GPU CKPT="$FULL_PATH" OUT="$REASONING_JSON" \
    "$PY" training/reasoning_eval.py 2>&1 | tee -a "$LOG"

log ""
log "=== auto_eval_${ROUND} complete ==="
log "  depth extrap:   $DEPTH_JSON"
log "  halt diag:      $DIAG_JSON"
log "  halt hist:      $HIST_JSON"
log "  reasoning eval: $REASONING_JSON"
log ""
log "consider rsync'ing JSONs to spark for paper integration:"
log "  rsync -av $DOCS/*_${ROUND}.json alexm@kebab-spark.lan:$DOCS/"
