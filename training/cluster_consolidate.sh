#!/usr/bin/env bash
# Hardened FSDP consolidate runner for the 4-node DGX Spark cluster.
#
# Why this script exists
# ----------------------
# r2.20 consolidate via ad-hoc torchrun timed out at 600s on every rank
# during the first _ALLGATHER_BASE (153.6M -> 614.4M bf16). All four
# ranks reached the collective, but NCCL never moved the bytes. Root
# cause: the runner did not set NCCL_SOCKET_IFNAME, so NCCL fell back
# to interface auto-discovery and could not pick a coherent path across
# the 200G fabric. Every working training launcher in this repo
# (training/launch_3b.sh and friends) sets NCCL_SOCKET_IFNAME=bond0,
# NCCL_IB_HCA=rocep1s0f1, GLOO_SOCKET_IFNAME=bond0, and a per-node
# NCCL_IB_GID_INDEX. This runner mirrors that pattern, adds pre-flight
# reachability checks, and bumps the collective timeout from 600s
# (default) to 1800s so a single ALLGATHER stall does not abort.
#
# Usage (rsync this file to ~/OpenMythos/training/ on every cluster
# node, then run once per node with the matching NODE_RANK):
#
#   NODE_RANK=0 bash ~/OpenMythos/training/cluster_consolidate.sh
#
# Optional env vars (defaults shown):
#
#   MASTER_ADDR=192.168.100.10     # rank-0 host on the 200G fabric
#   MASTER_PORT=29555              # rendezvous port (TCPStore)
#   CKPT_DIR=~/OpenMythos/checkpoints_3b_varT_pondernet_round220
#   CKPT_PREFIX=step_0024414       # so input shards are CKPT_DIR/${CKPT_PREFIX}_rank${R}.pt
#   OUT_PATH=$CKPT_DIR/${CKPT_PREFIX}_full.pt   # rank-0-only target
#   ROUND_TAG=r220                 # used in /tmp/consolidate_<tag>_r<rank>.log
#   COLLECTIVE_TIMEOUT_SEC=1800    # NCCL watchdog timeout
#   TORCHRUN=/home/alexm/.local/bin/torchrun
#   NCCL_IFACE=bond0               # 200G fabric interface
#   NCCL_IB_HCA=rocep1s0f1         # RoCE HCA matching bond0
#   NCCL_USE_IB=1                  # set 0 to force TCP-over-bond0 only
#
# Exit codes:
#   0  consolidated checkpoint written (rank 0 only verifies it)
#   2  pre-flight check failed (env, file, network, interface)
#   non-zero from torchrun otherwise.

set -uo pipefail

# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
NODE_RANK="${NODE_RANK:?usage: NODE_RANK=0..3 bash $0}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
MASTER_PORT="${MASTER_PORT:-29555}"
CKPT_DIR="${CKPT_DIR:-$HOME/OpenMythos/checkpoints_3b_varT_pondernet_round220}"
CKPT_PREFIX="${CKPT_PREFIX:-step_0024414}"
OUT_PATH="${OUT_PATH:-${CKPT_DIR}/${CKPT_PREFIX}_full.pt}"
ROUND_TAG="${ROUND_TAG:-r220}"
COLLECTIVE_TIMEOUT_SEC="${COLLECTIVE_TIMEOUT_SEC:-1800}"
TORCHRUN="${TORCHRUN:-/home/alexm/.local/bin/torchrun}"
NCCL_IFACE="${NCCL_IFACE:-bond0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_USE_IB="${NCCL_USE_IB:-1}"

LOG_PATH="/tmp/consolidate_${ROUND_TAG}_r${NODE_RANK}.log"

# Echo every meaningful line to stderr so the caller sees it even when
# they tail the log file separately.
note() { printf '[cluster_consolidate r%s] %s\n' "${NODE_RANK}" "$*"; }
die()  { note "FATAL: $*" >&2; exit 2; }

# ---------------------------------------------------------------------
# Pre-flight: rank
# ---------------------------------------------------------------------
case "${NODE_RANK}" in
  0|1|2|3) ;;
  *) die "NODE_RANK must be 0..3, got '${NODE_RANK}'";;
esac

# ---------------------------------------------------------------------
# Pre-flight: shard file
# ---------------------------------------------------------------------
SHARD_PATH="${CKPT_DIR}/${CKPT_PREFIX}_rank${NODE_RANK}.pt"
if [[ ! -f "${SHARD_PATH}" ]]; then
  die "shard file missing: ${SHARD_PATH}"
fi
shard_bytes=$(stat -c%s "${SHARD_PATH}")
if (( shard_bytes < 1073741824 )); then
  die "shard ${SHARD_PATH} is suspiciously small (${shard_bytes} bytes; expected >1 GB)"
fi
note "shard OK: ${SHARD_PATH} ($((shard_bytes / 1024 / 1024)) MB)"

# ---------------------------------------------------------------------
# Pre-flight: the 200G fabric interface exists and has an IP on the
# 192.168.100.0/24 net.
# ---------------------------------------------------------------------
if ! ip -br link show "${NCCL_IFACE}" >/dev/null 2>&1; then
  die "interface ${NCCL_IFACE} not present on $(hostname)"
fi

iface_state=$(ip -br link show "${NCCL_IFACE}" | awk '{print $2}')
if [[ "${iface_state}" != "UP" && "${iface_state}" != "UNKNOWN" ]]; then
  die "interface ${NCCL_IFACE} is in state ${iface_state} (need UP)"
fi

iface_ip=$(ip -4 -br addr show "${NCCL_IFACE}" | awk '{print $3}' | head -1 | cut -d/ -f1)
if [[ -z "${iface_ip}" ]]; then
  die "interface ${NCCL_IFACE} has no IPv4 address"
fi
if [[ "${iface_ip}" != 192.168.100.* ]]; then
  die "interface ${NCCL_IFACE} has IP ${iface_ip}, expected 192.168.100.x"
fi
note "iface ${NCCL_IFACE} OK: ${iface_ip}"

# ---------------------------------------------------------------------
# Pre-flight: master reachability. Rank 0 has nothing to dial; ranks
# 1..3 must be able to open a TCP socket to MASTER_ADDR:MASTER_PORT
# *after* rank 0 has started torchrun. We accept either success or
# connection-refused (the master listener races our check); we reject
# timeout / no-route / unknown-host because those mean we are on the
# wrong fabric.
# ---------------------------------------------------------------------
if [[ "${NODE_RANK}" != "0" ]]; then
  note "probing master ${MASTER_ADDR}:${MASTER_PORT} (up to 60s, master may still be starting)..."
  reach_ok=0
  for attempt in $(seq 1 60); do
    if nc -z -w 1 "${MASTER_ADDR}" "${MASTER_PORT}" >/dev/null 2>&1; then
      reach_ok=1; break
    fi
    # If we can't even reach the host (no route, host down), bail
    # fast. We don't care about the port; we care that the host is
    # reachable on the storage fabric.
    if ! ping -c 1 -W 1 "${MASTER_ADDR}" >/dev/null 2>&1; then
      if (( attempt >= 5 )); then
        die "master ${MASTER_ADDR} unreachable via ping after ${attempt} attempts; wrong network?"
      fi
    else
      # Host reachable, port not yet open -> master torchrun still
      # warming up. Accept this and let NCCL handshake retry.
      reach_ok=1; break
    fi
    sleep 1
  done
  if (( reach_ok == 0 )); then
    die "could not confirm reachability of ${MASTER_ADDR}:${MASTER_PORT} after 60s"
  fi
  note "master reachable"
fi

# ---------------------------------------------------------------------
# NCCL env. Mirrors training/launch_3b.sh (the only known-good cluster
# launcher), with two changes:
#   - NCCL_DEBUG bumped to INFO so the next failure is debuggable.
#   - TORCH_NCCL_* flags expanded (FlightRecorder, timeout dump,
#     async error handling) so we get a stack trace not just a timeout.
# ---------------------------------------------------------------------
export NCCL_SOCKET_IFNAME="${NCCL_IFACE}"
export GLOO_SOCKET_IFNAME="${NCCL_IFACE}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Bump the per-collective watchdog from 10 min default to COLLECTIVE_TIMEOUT_SEC.
# This is the env that torch.distributed reads on top of the default
# pg_options timeout: keep both in sync.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${COLLECTIVE_TIMEOUT_SEC}"
# The consolidate script reads this to pass timeout= into
# init_process_group("nccl", timeout=...), which is the only knob that
# actually moves the per-collective Watchdog timeout.
export CLUSTER_NCCL_TIMEOUT_SEC="${COLLECTIVE_TIMEOUT_SEC}"

if (( NCCL_USE_IB == 1 )); then
  export NCCL_IB_HCA
  # Auto-detect the RoCE v2 IPv4 GID index on bond0. Mirrors
  # training/launch_3b.sh::get_gid(). Falls back to GID 3 (typical
  # default on Mellanox CX-7) only if show_gids is missing.
  if command -v show_gids >/dev/null 2>&1; then
    NCCL_IB_GID_INDEX=$(show_gids "${NCCL_IB_HCA}" \
      | awk -v iface="${NCCL_IFACE}" '$7==iface && $6=="v2" && $5 ~ /^192\.168\.100\./ {print $3; exit}')
    if [[ -z "${NCCL_IB_GID_INDEX}" ]]; then
      note "WARN: show_gids returned no v2/100.x GID for ${NCCL_IB_HCA}; disabling IB"
      unset NCCL_IB_HCA
      export NCCL_IB_DISABLE=1
    else
      export NCCL_IB_GID_INDEX
      note "RoCE v2 setup: NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
    fi
  else
    note "WARN: show_gids not installed; falling back to TCP-over-${NCCL_IFACE}"
    unset NCCL_IB_HCA
    export NCCL_IB_DISABLE=1
  fi
else
  export NCCL_IB_DISABLE=1
  note "NCCL_USE_IB=0 -> running TCP-only over ${NCCL_IFACE}"
fi

# Match training-time perf hygiene.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PATH="/home/alexm/.local/bin:${PATH}"

# ---------------------------------------------------------------------
# Tee env summary into the log for postmortem.
# ---------------------------------------------------------------------
mkdir -p "$(dirname "${LOG_PATH}")"
{
  echo "==================== cluster_consolidate.sh ===================="
  date -u
  echo "host=$(hostname)"
  echo "NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  echo "SHARD=${SHARD_PATH} (${shard_bytes} bytes)"
  echo "OUT_PATH=${OUT_PATH}"
  echo "iface=${NCCL_IFACE} ip=${iface_ip}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-<unset>} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
  echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
  echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  echo "================================================================"
} | tee "${LOG_PATH}" >&2

# ---------------------------------------------------------------------
# Run torchrun. We tee so the parent ssh session sees output AND we
# keep a log on disk.
# ---------------------------------------------------------------------
cd "${HOME}/OpenMythos"
set +e
"${TORCHRUN}" \
  --nnodes=4 \
  --nproc_per_node=1 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  training/consolidate_ckpt.py \
  "${CKPT_DIR}/${CKPT_PREFIX}" \
  "${OUT_PATH}" 2>&1 | tee -a "${LOG_PATH}"
rc=${PIPESTATUS[0]}
set -e

if (( rc != 0 )); then
  note "torchrun exited with rc=${rc}; see ${LOG_PATH}"
  exit "${rc}"
fi

# ---------------------------------------------------------------------
# Rank-0 sanity check on the produced checkpoint.
# ---------------------------------------------------------------------
if [[ "${NODE_RANK}" == "0" ]]; then
  if [[ ! -f "${OUT_PATH}" ]]; then
    die "rank 0 finished but ${OUT_PATH} does not exist"
  fi
  out_bytes=$(stat -c%s "${OUT_PATH}")
  if (( out_bytes < 1073741824 )); then
    die "consolidated ckpt ${OUT_PATH} is ${out_bytes} bytes (<1 GB); something is wrong"
  fi
  out_mb=$((out_bytes / 1024 / 1024))
  note "OK: ${OUT_PATH} (${out_mb} MB)"
  # MD5 is slow on 30+ GB files; report size unconditionally and MD5
  # if the user explicitly asked for it via SANITY_MD5=1.
  if [[ "${SANITY_MD5:-0}" == "1" ]]; then
    md5sum "${OUT_PATH}" | tee -a "${LOG_PATH}"
  fi
fi

exit 0
