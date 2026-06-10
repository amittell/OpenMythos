#!/usr/bin/env bash
# Launcher for mythos_3b full training across the DGX Spark cluster.
# Usage:
#   ./launch_3b.sh             # all reachable nodes that are internet-ok
#   NODES="spark gx10 gx10-2"  ./launch_3b.sh
#
# Expects training/3b_fine_web_edu.py present at ~/OpenMythos on every node,
# plus open_mythos package installed (pip + pth file) and torch 2.9+.
#
# Writes logs to /tmp/train_rN.log on each node and checkpoints into
# ~/OpenMythos/checkpoints/ on each node (rank 0 writes; others only read).

set -euo pipefail

# Map of node_rank -> hostname.
# Uses *-200g aliases because spark's /etc/hosts has stale/broken entries
# for the .lan names (kebab-gx10-2.lan resolves to the NAS; the others
# resolve to IPv6 ::). The 200g aliases resolve cleanly to 192.168.100.x
# on every node.
declare -A NODE_HOST=(
  [0]=kebab-spark-200g
  [1]=kebab-gx10-200g
  [2]=kebab-gx10-2-200g
  [3]=kebab-gx10-3-200g
)

NODES=${NODES:-"spark gx10 gx10-2 gx10-3"}
MASTER=192.168.100.10
PORT=${PORT:-29510}
SCRIPT=${SCRIPT:-training/3b_fine_web_edu.py}

# Build node list
ACTIVE=()
for n in $NODES; do
  case $n in
    spark)   ACTIVE+=(0);;
    gx10)    ACTIVE+=(1);;
    gx10-2)  ACTIVE+=(2);;
    gx10-3)  ACTIVE+=(3);;
  esac
done
NNODES=${#ACTIVE[@]}
echo "Launching $SCRIPT on $NNODES nodes: ${ACTIVE[@]} (port $PORT, master $MASTER)"

# Kill any stale python training procs cluster-wide. Match python-process
# command lines specifically (not SSH wrappers that might contain the script
# name in their argv on the master node, which would self-kill the launcher).
for r in "${ACTIVE[@]}"; do
  ssh -q alexm@${NODE_HOST[$r]} '
    pkill -9 -f "[p]ython3 .*training/3b_fine_web_edu" 2>/dev/null
    pkill -9 -f "[p]ython3 .*training/3b_varT" 2>/dev/null
    pkill -9 -f "[p]ython3 .*training/3b_loops" 2>/dev/null
    pkill -9 -f "[p]ython3 .*training/shakeout_1b" 2>/dev/null
    pkill -9 -f "[p]ython3 .*torch.distributed.run" 2>/dev/null
    true
  ' &
done
wait

NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=420 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'

# Optional space-separated KEY=VALUE pairs appended to the launch env, e.g.
#   EXTRA_ENV="CKPT_DIR=foo BOOTSTRAP_CKPT=bar/step.pt"
EXTRA_ENV=${EXTRA_ENV:-}

# Extract BOOTSTRAP_CKPT from EXTRA_ENV if present so we can pre-distribute
# the consolidated full.pt to worker nodes. The trainer's bootstrap_full
# mode (training/bootstrap_dispatch.py, OpenMythos #156) loads the same
# file on every rank then lets FSDP re-shard in memory -- so the file
# must exist on each worker's local disk before torchrun starts. Without
# this step ranks 1..N-1 die instantly with FileNotFoundError, which
# burns the supervisor's restart budget on a non-recoverable cause.
# Distribution runs in parallel from rank 0 over the 200G storage fabric
# (~16 GB takes ~15-30 s per worker). Skips workers whose copy already
# matches in size (cheap idempotency for re-launches and supervisor
# restarts; full byte-compare would be too slow on 16 GB).
BOOTSTRAP_CKPT=""
for kv in $EXTRA_ENV; do
  case $kv in BOOTSTRAP_CKPT=*) BOOTSTRAP_CKPT=${kv#BOOTSTRAP_CKPT=};; esac
done

if [ -n "$BOOTSTRAP_CKPT" ] && [ ${#ACTIVE[@]} -gt 1 ]; then
  RANK0_HOST=${NODE_HOST[${ACTIVE[0]}]}
  src_size=$(ssh -q alexm@$RANK0_HOST "stat -c %s ~/OpenMythos/$BOOTSTRAP_CKPT 2>/dev/null || echo 0")
  if [ "$src_size" = "0" ]; then
    echo "ERROR: BOOTSTRAP_CKPT '$BOOTSTRAP_CKPT' not found on rank0 host $RANK0_HOST" >&2
    exit 1
  fi
  echo "BOOTSTRAP_CKPT=$BOOTSTRAP_CKPT (${src_size} bytes on $RANK0_HOST); distributing to workers"
  rel_dir=$(dirname "$BOOTSTRAP_CKPT")
  for r in "${ACTIVE[@]:1}"; do
    host=${NODE_HOST[$r]}
    (
      dst_size=$(ssh -q alexm@$host "stat -c %s ~/OpenMythos/$BOOTSTRAP_CKPT 2>/dev/null || echo 0")
      if [ "$dst_size" = "$src_size" ]; then
        echo "  [rank $r @ $host] already present (size match: $dst_size); skip"
      else
        echo "  [rank $r @ $host] rsync start (have=$dst_size, want=$src_size)"
        ssh -q alexm@$RANK0_HOST "mkdir -p ~/OpenMythos/$rel_dir && rsync -a --partial --inplace --bwlimit=0 ~/OpenMythos/$BOOTSTRAP_CKPT alexm@$host:OpenMythos/$BOOTSTRAP_CKPT"
        rc=$?
        echo "  [rank $r @ $host] rsync done rc=$rc"
        if [ $rc -ne 0 ]; then exit $rc; fi
      fi
    ) &
  done
  wait
  echo "BOOTSTRAP_CKPT distribution complete"
fi

# Fetch per-node RoCE v2 IPv4 GID index via a simple heredoc (outer quoting is tricky)
get_gid() {
  ssh -q alexm@$1 bash -s << 'EOF'
show_gids rocep1s0f1 | awk '$7=="bond0" && $6=="v2" && $5 ~ /^192\.168\.100\./ {print $3; exit}'
EOF
}

# Launch each rank
for r in "${ACTIVE[@]}"; do
  host=${NODE_HOST[$r]}
  gid=$(get_gid "$host")
  echo "-> rank $r on $host (GID=$gid)"
  ssh -q alexm@$host "rm -f /tmp/train_r${r}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${gid} $EXTRA_ENV torchrun --nnodes=$NNODES --nproc_per_node=1 --node_rank=$r --master_addr=$MASTER --master_port=$PORT $SCRIPT >/tmp/train_r${r}.log 2>&1 </dev/null & disown; echo 'r${r} pid='\$!" &
done
wait
echo
echo "All ranks launched. Monitor rank 0 with:"
echo "  ssh alexm@${NODE_HOST[${ACTIVE[0]}]} 'tail -F /tmp/train_r${ACTIVE[0]}.log'"
