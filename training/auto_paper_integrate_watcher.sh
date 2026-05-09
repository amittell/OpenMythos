#!/usr/bin/env bash
# auto_paper_integrate_watcher.sh
#
# Polls remote hosts for round result JSONs and runs auto_paper_integrate.py
# (and auto_paper_capability.py) to insert sections into docs/paper/main.md
# as each round's data becomes available.
#
# Idempotent: auto_paper_integrate.py skips rounds whose section header is
# already in main.md. Watcher self-terminates once all known rounds are
# integrated, OR after DEADLINE_HOURS.
#
# Runs locally on the Mac because auto_paper_integrate.py uses absolute
# paths under /Users/alex/git/OpenMythos.
#
# Usage:
#   nohup bash training/auto_paper_integrate_watcher.sh \
#       >>/tmp/auto_paper_integrate.log 2>&1 </dev/null & disown

set -uo pipefail

LOG=/tmp/auto_paper_integrate.log
REPO=/Users/alex/git/OpenMythos
SCRIPT="$REPO/training/auto_paper_integrate.py"
CAPABILITY_SCRIPT="$REPO/training/auto_paper_capability.py"
INTERVAL_S=${INTERVAL_S:-600}      # poll every 10 min
DEADLINE_HOURS=${DEADLINE_HOURS:-96}

# Round number -> expected section header substring in main.md
declare -A HEADER_FOR
HEADER_FOR[26]="### 7.11 Joint training stability past 50M tokens"
HEADER_FOR[27]="### 7.12 Halt-prior sensitivity"
HEADER_FOR[28]="### 7.13 Anti-collapse halt-recovery on a PonderNet-rescued backbone"
HEADER_FOR[29]="### 7.14 No-recurrence baseline at matched compute"
HEADER_FOR[210]="### 7.16 Extended joint training to 150M tokens"
HEADER_FOR[211]="### 7.17 Compute-scaling: T_FIXED = 2"
HEADER_FOR[212]="### 7.18 Compute-scaling: T_FIXED = 4"
HEADER_FOR[213]="### 7.19 Joint training to 200M tokens"
HEADER_FOR[214]="### 7.20 Compute-scaling: T_FIXED = 16"
TOTAL_ROUNDS=${#HEADER_FOR[@]}
CAP_HEADER="### 7.15 Depth-graded capability probes"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "auto_paper_integrate watcher started; polling every ${INTERVAL_S}s"
log "tracking $TOTAL_ROUNDS rounds + 1 capability section"

DEADLINE=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

while true; do
    log "=== scan pass ==="

    # Run paper integration (fetches from remotes, integrates whatever it can)
    python3 "$SCRIPT" --scan 2>&1 | tail -30 | tee -a "$LOG"

    # Run capability section integration if the script exists
    if [ -f "$CAPABILITY_SCRIPT" ]; then
        python3 "$CAPABILITY_SCRIPT" 2>&1 | tail -10 | tee -a "$LOG"
    fi

    # Count integrated sections
    done_count=0
    missing_rounds=()
    for n in "${!HEADER_FOR[@]}"; do
        if grep -qF "${HEADER_FOR[$n]}" "$REPO/docs/paper/main.md" 2>/dev/null; then
            done_count=$((done_count + 1))
        else
            missing_rounds+=("$n")
        fi
    done

    cap_done=0
    if grep -qF "$CAP_HEADER" "$REPO/docs/paper/main.md" 2>/dev/null; then
        cap_done=1
    fi

    log "integrated: $done_count/$TOTAL_ROUNDS rounds + $cap_done/1 capability section"
    if [ ${#missing_rounds[@]} -gt 0 ]; then
        log "missing: ${missing_rounds[*]}"
    fi

    if [ "$done_count" -eq "$TOTAL_ROUNDS" ] && [ "$cap_done" -eq 1 ]; then
        log "all sections integrated; watcher exiting cleanly"
        exit 0
    fi

    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        log "deadline exceeded ($DEADLINE_HOURS hours); exiting with $done_count/$TOTAL_ROUNDS done"
        exit 1
    fi

    log "sleeping ${INTERVAL_S}s"
    sleep "$INTERVAL_S"
done
