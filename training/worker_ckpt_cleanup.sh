#!/usr/bin/env bash
# worker_ckpt_cleanup.sh
#
# Periodic watchdog for cluster worker nodes (gx10-200g, gx10-2-200g,
# gx10-3-200g). Keeps only the 2 most recent sharded ckpts per round
# directory under /home/alexm/OpenMythos/. Prevents disk-fill from
# accumulated training shards across long runs.
#
# Designed to be launched on each worker via:
#   ssh alexm@<worker> 'nohup bash worker_ckpt_cleanup.sh >/tmp/wckpt_cleanup.log 2>&1 </dev/null & disown'
#
# Safe to run continuously. The keep-2 policy means the latest training
# shard is preserved even if cleanup runs mid-save (the .tmp file is
# excluded from the glob).

set -uo pipefail

INTERVAL_SEC=${INTERVAL_SEC:-300}
KEEP=${KEEP:-2}
ROOT=${ROOT:-/home/alexm/OpenMythos}

log() { echo "[$(date '+%F %T')] $*"; }

log "worker_ckpt_cleanup started (keep=$KEEP, interval=${INTERVAL_SEC}s)"

while true; do
    for d in "$ROOT"/checkpoints_3b_*/; do
        [ -d "$d" ] || continue
        cd "$d" || continue
        # Match only completed shard files (not .tmp partials)
        # Sort by step number ascending; remove all but the last $KEEP
        before=$(ls step_*_rank*.pt 2>/dev/null | wc -l)
        ls step_*_rank*.pt 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -f
        after=$(ls step_*_rank*.pt 2>/dev/null | wc -l)
        if [ "$before" -ne "$after" ]; then
            log "$(basename $d): kept $after/$before shards"
        fi
        # Also clean stale .tmp files (older than 30 minutes)
        find . -maxdepth 1 -name "*.tmp" -mmin +30 -delete 2>/dev/null
    done
    sleep "$INTERVAL_SEC"
done
