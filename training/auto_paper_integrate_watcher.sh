#!/usr/bin/env bash
# Local watcher that polls remote hosts for round 2.6/2.7/2.8 result
# fragments and invokes auto_paper_integrate.py to insert §7.11/§7.12/§7.13
# into docs/paper/main.md as each round lands.
#
# Idempotent: auto_paper_integrate.py skips rounds whose section header
# already exists in main.md. Stops once all 3 are integrated.

set -uo pipefail

LOG=/tmp/auto_paper_integrate.log
REPO=/Users/alex/git/OpenMythos
SCRIPT="$REPO/training/auto_paper_integrate.py"
INTERVAL_S=1800   # 30 min between scans
DEADLINE_HOURS=72

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "auto_paper_integrate watcher started; polling every ${INTERVAL_S}s"
DEADLINE=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

CAPABILITY_SCRIPT="$REPO/training/auto_paper_capability.py"

while true; do
  python3 "$SCRIPT" --scan 2>&1 | tee -a "$LOG"
  python3 "$CAPABILITY_SCRIPT" 2>&1 | tee -a "$LOG"

  # Stop once all 4 round sections + capability section are in main.md
  done_count=0
  for n in 26 27 28 29; do
    case $n in
      26) header="### 7.11 Joint training stability past 50M tokens" ;;
      27) header="### 7.12 Halt-prior sensitivity" ;;
      28) header="### 7.13 Anti-collapse halt-recovery on a PonderNet-rescued backbone" ;;
      29) header="### 7.14 No-recurrence baseline at matched compute" ;;
    esac
    grep -qF "$header" "$REPO/docs/paper/main.md" 2>/dev/null && done_count=$((done_count + 1))
  done
  cap_header="### 7.15 Depth-graded capability probes"
  grep -qF "$cap_header" "$REPO/docs/paper/main.md" 2>/dev/null && done_count=$((done_count + 1))

  if [ "$done_count" -eq 5 ]; then
    log "all 5 sections integrated; watcher exiting"
    exit 0
  fi

  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "deadline exceeded; exiting with $done_count/3 integrated"
    exit 1
  fi

  log "$done_count/3 integrated; sleeping ${INTERVAL_S}s"
  sleep "$INTERVAL_S"
done
