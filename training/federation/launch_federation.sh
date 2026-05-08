#!/usr/bin/env bash
# launch_federation.sh
#
# Master launcher for federated training across cluster + RTX6000.
#
# Required env vars:
#   FED_BOOTSTRAP_CKPT_CLUSTER  full path on cluster of starting ckpt (e.g. r2.13 final)
#   FED_BOOTSTRAP_CKPT_RTX      same, on RTX6000 (rsync first if needed)
#   FED_TARGET_TOKENS_TOTAL     total tokens to train (e.g. 2000000000)
#   FED_TRAINING_SCRIPT         training script name (e.g. training/3b_varT_pondernet_joint.py)
#
# Optional env vars:
#   FED_SYNC_INTERVAL_SEC       default 1800 (30 min)
#   FED_DIR                     default /home/alexm/OpenMythos/fed
#   FED_RTX_DIR                 default /home/alexm/OpenMythos/fed (on RTX6000)
#   FED_RTX_HOST                default alexm@kebab-rtx6000.lan
#   FED_EXTRA_ENV_CLUSTER       extra env vars for cluster trainer
#   FED_EXTRA_ENV_RTX           extra env vars for RTX trainer
#
# This script:
#   1. Verifies preconditions (ckpts exist on each side, no other training running)
#   2. Initializes FED_DIR on both sides (clears stale state if --fresh)
#   3. Launches coordinator on spark
#   4. Launches cluster trainer (4-node FSDP via launch_3b.sh)
#   5. Launches RTX6000 trainer (single GPU torchrun)
#   6. Logs PIDs and exit instructions

set -uo pipefail

# Defaults
: "${FED_DIR:=/home/alexm/OpenMythos/fed}"
: "${FED_RTX_DIR:=/home/alexm/OpenMythos/fed}"
: "${FED_RTX_HOST:=alexm@kebab-rtx6000.lan}"
: "${FED_SYNC_INTERVAL_SEC:=1800}"
: "${FED_EXTRA_ENV_CLUSTER:=}"
: "${FED_EXTRA_ENV_RTX:=}"
: "${FED_TRAINING_SCRIPT:=training/3b_varT_pondernet_joint.py}"

REPO=/home/alexm/OpenMythos
RTX_REPO=/home/alexm/OpenMythos
LOG_DIR=$REPO/training/federation/logs
mkdir -p "$LOG_DIR"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/launch.log"; }

require() {
    local name=$1
    local val
    val="$(eval echo "\${$name:-}")"
    [ -n "$val" ] || { log "ERROR: $name must be set"; exit 1; }
}

require FED_BOOTSTRAP_CKPT_CLUSTER
require FED_BOOTSTRAP_CKPT_RTX
require FED_TARGET_TOKENS_TOTAL

FRESH=0
SKIP_VALIDATE=0
for arg in "$@"; do
    [ "$arg" = "--fresh" ] && FRESH=1
    [ "$arg" = "--skip-validate" ] && SKIP_VALIDATE=1
done

# ----------------------------------------------------------------------------
# Mandatory: run validate.sh before doing anything else.
# Anyone bypassing this should pass --skip-validate AND understand what they
# are skipping (they will not be saved by this script if validate would have
# caught a problem).
# ----------------------------------------------------------------------------

if [ "$SKIP_VALIDATE" -eq 0 ]; then
    log "running validate.sh as precondition (use --skip-validate to bypass; not recommended)"
    if ! FED_DIR="$FED_DIR" \
         FED_RTX_HOST="$FED_RTX_HOST" \
         FED_RTX_DIR="$FED_RTX_DIR" \
         FED_BOOTSTRAP_CKPT_CLUSTER="$FED_BOOTSTRAP_CKPT_CLUSTER" \
         FED_BOOTSTRAP_CKPT_RTX="$FED_BOOTSTRAP_CKPT_RTX" \
         bash "$REPO/training/federation/validate.sh" 2>&1 | tee -a "$LOG_DIR/launch.log"; then
        log "ERROR: validate.sh failed; refusing to launch federation"
        log "       fix the failures above, or pass --skip-validate to override (NOT recommended)"
        exit 1
    fi
    log "validate.sh PASSED"
else
    log "WARN: --skip-validate passed; bypassing precondition checks"
fi

log "=== federated training launch ==="
log "fed_dir (cluster)    = $FED_DIR"
log "fed_rtx_host         = $FED_RTX_HOST"
log "fed_rtx_dir          = $FED_RTX_DIR"
log "sync_interval        = ${FED_SYNC_INTERVAL_SEC}s"
log "training_script      = $FED_TRAINING_SCRIPT"
log "bootstrap (cluster)  = $FED_BOOTSTRAP_CKPT_CLUSTER"
log "bootstrap (rtx)      = $FED_BOOTSTRAP_CKPT_RTX"
log "target_tokens_total  = $FED_TARGET_TOKENS_TOTAL"
log "fresh start          = $FRESH"

# ----------------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------------

[ -f "$FED_BOOTSTRAP_CKPT_CLUSTER" ] || { log "ERROR: cluster bootstrap ckpt missing: $FED_BOOTSTRAP_CKPT_CLUSTER"; exit 1; }

if ! ssh -o ConnectTimeout=10 "$FED_RTX_HOST" "test -f $FED_BOOTSTRAP_CKPT_RTX"; then
    log "ERROR: rtx bootstrap ckpt missing on $FED_RTX_HOST: $FED_BOOTSTRAP_CKPT_RTX"
    exit 1
fi

# Make sure no other 3b_varT training procs are running
log "checking for stale training procs cluster-wide..."
NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
for h in $NODES_200G; do
    n=$(ssh -q -o ConnectTimeout=5 alexm@"$h" "pgrep -fc 'python3 .*training/3b_varT' 2>/dev/null || echo 0")
    if [ "${n:-0}" -gt 0 ]; then
        log "WARN: $h has $n training procs running. Refusing to launch federation."
        log "      Stop them with: ssh alexm@$h 'pkill -9 -f \"python3 .*training/3b_varT\"'"
        exit 1
    fi
done
n_rtx=$(ssh -q -o ConnectTimeout=10 "$FED_RTX_HOST" "pgrep -fc 'python3 .*training/3b_varT' 2>/dev/null || echo 0")
if [ "${n_rtx:-0}" -gt 0 ]; then
    log "WARN: RTX6000 has $n_rtx training procs. Refusing to launch."
    exit 1
fi

# ----------------------------------------------------------------------------
# Initialize fed dirs
# ----------------------------------------------------------------------------

if [ $FRESH -eq 1 ]; then
    log "FRESH mode: clearing $FED_DIR and remote $FED_RTX_DIR"
    rm -rf "$FED_DIR"
    ssh -q "$FED_RTX_HOST" "rm -rf $FED_RTX_DIR"
fi

mkdir -p "$FED_DIR"
ssh -q "$FED_RTX_HOST" "mkdir -p $FED_RTX_DIR"

# Verify federation scripts are present on RTX6000
for f in coordinator.py sync_hook.py; do
    if ! ssh -q "$FED_RTX_HOST" "test -f $RTX_REPO/training/federation/$f"; then
        log "rsyncing federation/ to RTX6000..."
        rsync -az "$REPO/training/federation/" "$FED_RTX_HOST:$RTX_REPO/training/federation/"
        break
    fi
done

# ----------------------------------------------------------------------------
# Launch coordinator (on spark.lan)
# ----------------------------------------------------------------------------

log "launching coordinator..."
nohup python3 "$REPO/training/federation/coordinator.py" \
    --fed-dir "$FED_DIR" \
    --rtx-host "$FED_RTX_HOST" \
    --rtx-remote-dir "$FED_RTX_DIR" \
    --sync-timeout 7200 \
    --poll-interval 10 \
    >>"$LOG_DIR/coordinator.log" 2>&1 </dev/null &
COORD_PID=$!
disown $COORD_PID
log "coordinator PID=$COORD_PID, log=$LOG_DIR/coordinator.log"

sleep 5

# ----------------------------------------------------------------------------
# Launch cluster trainer
# ----------------------------------------------------------------------------

CLUSTER_CKPT_DIR="checkpoints_3b_fed_cluster"
mkdir -p "$REPO/$CLUSTER_CKPT_DIR"

log "launching cluster trainer (4-node FSDP)..."
cd "$REPO"
SCRIPT="$FED_TRAINING_SCRIPT" PORT=29520 \
    EXTRA_ENV="CKPT_DIR=$CLUSTER_CKPT_DIR \
               BOOTSTRAP_CKPT=$(realpath --relative-to=$REPO $FED_BOOTSTRAP_CKPT_CLUSTER) \
               REINIT_HEAD=0 \
               TARGET_TOKENS=$FED_TARGET_TOKENS_TOTAL \
               FED_SYNC_DIR=$FED_DIR \
               FED_ROLE=cluster \
               FED_SYNC_INTERVAL_SEC=$FED_SYNC_INTERVAL_SEC \
               $FED_EXTRA_ENV_CLUSTER" \
    nohup bash training/launch_3b.sh >>"$LOG_DIR/cluster_launch.log" 2>&1 </dev/null &
CLUSTER_LAUNCH_PID=$!
disown $CLUSTER_LAUNCH_PID
log "cluster launch PID=$CLUSTER_LAUNCH_PID, log=$LOG_DIR/cluster_launch.log"

sleep 5

# ----------------------------------------------------------------------------
# Launch RTX6000 trainer (single-GPU FSDP)
# ----------------------------------------------------------------------------

RTX_CKPT_DIR="checkpoints_3b_fed_rtx"

log "launching RTX6000 trainer (single GPU)..."
ssh -q "$FED_RTX_HOST" "mkdir -p $RTX_REPO/$RTX_CKPT_DIR"
ssh -q "$FED_RTX_HOST" "cd $RTX_REPO && nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    CKPT_DIR=$RTX_CKPT_DIR \
    BOOTSTRAP_CKPT=$(ssh -q $FED_RTX_HOST "realpath --relative-to=$RTX_REPO $FED_BOOTSTRAP_CKPT_RTX") \
    REINIT_HEAD=0 \
    TARGET_TOKENS=$FED_TARGET_TOKENS_TOTAL \
    FED_SYNC_DIR=$FED_RTX_DIR \
    FED_ROLE=rtx \
    FED_REMOTE_HOST=alexm@kebab-spark.lan \
    FED_REMOTE_DIR=$FED_DIR \
    FED_SYNC_INTERVAL_SEC=$FED_SYNC_INTERVAL_SEC \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $FED_EXTRA_ENV_RTX \
    /home/alexm/venvs/vllm-turboquant/bin/torchrun \
        --nnodes=1 --nproc_per_node=1 --master_port=29521 \
        $FED_TRAINING_SCRIPT \
    >/tmp/fed_rtx_train.log 2>&1 </dev/null & echo rtx_PID=\$!"
log "RTX trainer launched"

# ----------------------------------------------------------------------------
# Status summary
# ----------------------------------------------------------------------------

log ""
log "=== federation running ==="
log "  coordinator:      tail -F $LOG_DIR/coordinator.log"
log "  cluster trainer:  ssh alexm@kebab-spark-200g 'tail -F /tmp/train_r0.log'"
log "  RTX trainer:      ssh $FED_RTX_HOST 'tail -F /tmp/fed_rtx_train.log'"
log ""
log "  to halt federation gracefully: touch $FED_DIR/halt"
log "  to abort (kill all):           bash training/federation/abort_federation.sh"
