#!/usr/bin/env bash
# After round 2.4's auto-eval pipeline completes (cluster idle), regenerate
# the round-2.3 paper artifacts that auto_eval_round23 failed to produce
# because round 2.4 launched too early and overran spark's GPU memory.
#
# Failures being remediated:
#   - act_halt_histogram_round23.json  (CUDA OOM at 09:14 May 2 during r2.4
#                                       bootstrap on spark)
#   - docs/paper/round23_results.md    (build_paper_section7 aborted because
#                                       histogram input was missing)
#
# Not remediable here (artifact lost, not regenerable):
#   - round23 training_curve.png and round23_loss_by_T.png
#     (loguru wrote training step lines only to /tmp/train_r0.log, which
#     launch_3b.sh truncated when round 2.4 fired)
#
# Trigger:
#   - "auto_eval_round24 pipeline complete" in /tmp/auto_eval_round24.log
#   - 6h max wait before giving up

set -uo pipefail

LOG=/home/alexm/OpenMythos/training/post_round24_fix_round23.log
WATCH=/tmp/auto_eval_round24.log
CKPT=/home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_joint/step_0003051_full.pt
DOCS=/home/alexm/OpenMythos/docs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "post_round24_fix_round23 watcher started"

# Wait for round 2.4's auto-eval to complete. 6h cap.
DEADLINE=$(($(date +%s) + 21600))
while true; do
  if grep -q "auto_eval_round24 pipeline complete" "$WATCH" 2>/dev/null; then
    log "auto_eval_round24 pipeline-complete line seen"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "ERROR: 6h deadline elapsed without pipeline-complete; aborting"
    exit 1
  fi
  sleep 60
done

# Defensive: wait until no pondernet/torchrun/consolidate procs remain on
# spark before allocating GPU memory.
PROC_PATTERN="pondernet|torchrun|consolidate_ckpt|depth_extrap|act_halt|gen_samples|training_curves"
log "fence: waiting for spark eval/training procs to exit"
while pgrep -f "$PROC_PATTERN" >/dev/null 2>&1; do
  sleep 30
done
log "fence cleared"

# Verify the round 2.3 ckpt still exists.
if [ ! -f "$CKPT" ]; then
  log "ERROR: $CKPT missing; cannot regenerate histogram"
  exit 1
fi

cd /home/alexm/OpenMythos

HIST_JSON="$DOCS/act_halt_histogram_round23.json"
log "act_halt_histogram.py -> $HIST_JSON"
CKPT="$CKPT" OUT="$HIST_JSON" python3 training/act_halt_histogram.py 2>&1 | tee -a "$LOG"

if [ ! -s "$HIST_JSON" ]; then
  log "ERROR: histogram JSON not written; aborting paper rebuild"
  exit 1
fi

PAPER_FRAG="$DOCS/paper/round23_results.md"
log "build_paper_section7.py -> $PAPER_FRAG"
DEPTH_JSON="$DOCS/depth_extrap_round23.json" \
  DIAG_JSON="$DOCS/act_halt_diagnostic_round23.json" \
  HIST_JSON="$HIST_JSON" \
  SAMPLES_TXT="$DOCS/gen_samples_round23_multidepth.txt" \
  STEP="0003051" \
  OUT="$PAPER_FRAG" \
  python3 training/build_paper_section7.py 2>&1 | tee -a "$LOG"

cat <<EOF | tee -a "$LOG"

[$(date '+%F %T')] post_round24_fix_round23 complete.

  Histogram JSON:         $HIST_JSON
  Paper section fragment: $PAPER_FRAG

NOTE: round23 training_curve.png and loss_by_T.png are not recoverable
(loguru output to /tmp/train_r0.log was truncated by round 2.4's launch).
The paper §7.1 reference will need a manual edit to remove the figure
links for round 2.3 or substitute the round 2.4 curves as illustrative.
EOF
