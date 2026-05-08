#!/usr/bin/env bash
# Abort an in-flight federated training run. Kills coordinator and both
# trainers, leaves on-disk state intact for inspection.

set -uo pipefail
: "${FED_DIR:=/home/alexm/OpenMythos/fed}"
: "${FED_RTX_HOST:=alexm@kebab-rtx6000.lan}"

log() { echo "[$(date '+%F %T')] $*"; }

log "aborting federation"

# Set halt flag (signals coordinator + trainers to exit cleanly first)
touch "$FED_DIR/halt"
ssh -q "$FED_RTX_HOST" "touch /home/alexm/OpenMythos/fed/halt 2>/dev/null || true"

sleep 5

# Kill coordinator
pkill -f "training/federation/coordinator.py" 2>/dev/null && log "coord killed" || log "coord already gone"

# Kill cluster trainer (all 4 ranks)
for h in kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g; do
    ssh -q -o ConnectTimeout=5 alexm@"$h" \
        "pkill -9 -f 'python3 .*training/3b_varT' 2>/dev/null; pkill -9 -f 'python3 .*torch.distributed.run' 2>/dev/null; true" &
done
wait

# Kill RTX trainer
ssh -q "$FED_RTX_HOST" \
    "pkill -9 -f 'python3 .*training/3b_varT' 2>/dev/null; pkill -9 -f 'python3 .*torch.distributed.run' 2>/dev/null; true"

log "abort complete; on-disk state preserved at $FED_DIR (and on RTX6000)"
log "to fully reset: rm -rf $FED_DIR && ssh $FED_RTX_HOST 'rm -rf /home/alexm/OpenMythos/fed'"
