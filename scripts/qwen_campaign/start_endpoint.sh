#!/bin/bash
# Start a vLLM endpoint with multiple served names + Qwen3.6 reasoning parser.
# Args: MODEL_PATH SERVED_NAMES PORT GPU_DEVICE MAX_NUM_SEQS EXTRA_FLAGS
set -euo pipefail

MODEL_PATH=$1
SERVED_NAMES=$2     # space-separated list, e.g. "qwen3.6-27b qwen3.6-27b-fp8"
PORT=$3
GPU_DEVICE=$4
MAX_NUM_SEQS=${5:-4}
EXTRA_FLAGS=${6:-}
CONTAINER_NAME="kebab-vllm-eval-gpu${GPU_DEVICE}"
IMG=vllm/vllm-openai:v0.21.0

log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "stopping any existing $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

log "starting: model=$MODEL_PATH served_names='$SERVED_NAMES' port=$PORT gpu=$GPU_DEVICE max_num_seqs=$MAX_NUM_SEQS"

docker run -d --name "$CONTAINER_NAME" \
  --gpus device="$GPU_DEVICE" \
  -p "${PORT}:8000" \
  -v /models:/models:ro \
  --shm-size=32g \
  "$IMG" \
  --model "$MODEL_PATH" \
  --served-model-name $SERVED_NAMES \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  $EXTRA_FLAGS >/dev/null

log "waiting for ready on port $PORT (max 1200s)"
for i in $(seq 1 120); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null; then
    log "endpoint ready after ${i}0s"
    exit 0
  fi
  sleep 10
done
log "ERROR: endpoint never ready"
docker logs --tail 80 "$CONTAINER_NAME" >&2
exit 1
