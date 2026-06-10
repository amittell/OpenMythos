#!/bin/bash
# Qwen3.6 HE+/MBPP+ rerun with the max_new_tokens=32768 thinking fix.
#
# Wraps endpoint provisioning + evalplus run + teardown in one script so it
# can be invoked as a gpufarm job. Each invocation handles ONE model on ONE
# Blackwell GPU. evalplus default max_new_tokens=768 truncates Qwen3.6 mid-
# thinking (see project memory evalplus-max-tokens-thinking-trap); without
# the override only 1/165 HE+ samples reaches </think>.
#
# Required env:
#   MODEL_PATH     /models/<dir> path on the host
#   MODEL_TAG      e.g. Qwen3.6-27B-BF16
#   SERVED_NAME    vLLM served-model-name, e.g. qwen3.6-27b (must match
#                  what evalplus passes as the openai model name)
#   OUT_ROOT       output dir; results land under $OUT_ROOT/evalplus_{he,mbpp}
#   GPU_DEVICE     0 or 1 (which Blackwell to use). When dispatched via
#                  gpufarm with cuda_device set, CUDA_VISIBLE_DEVICES is
#                  injected and we read it here.
#
# Optional:
#   MAX_NEW_TOKENS  default 32768
#   TEMP            default 0.6
#   TOP_P           default 0.95   (informational; evalplus hardcodes 0.95)
#   PORT            default 8000+GPU_DEVICE
#   CONTAINER_TAG   default kebab-vllm-rerun-gpu${GPU_DEVICE}
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH required}"
: "${MODEL_TAG:?MODEL_TAG required}"
: "${SERVED_NAME:?SERVED_NAME required}"
: "${OUT_ROOT:?OUT_ROOT required}"
: "${GPU_DEVICE:=${CUDA_VISIBLE_DEVICES:-0}}"
GPU_DEVICE="${GPU_DEVICE%%,*}"   # in case CUDA_VISIBLE_DEVICES has multiple
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMP="${TEMP:-0.6}"
PORT="${PORT:-$((8000 + GPU_DEVICE))}"
CONTAINER_TAG="${CONTAINER_TAG:-kebab-vllm-rerun-gpu${GPU_DEVICE}}"
IMG=vllm/vllm-openai:v0.21.0
EP_PY=/home/alexm/git/benchmarks/evalplus-venv/bin/python

ROOT="$OUT_ROOT/$MODEL_TAG"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/rerun.log") 2>&1
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "MODEL_TAG=$MODEL_TAG MODEL_PATH=$MODEL_PATH SERVED_NAME=$SERVED_NAME"
log "GPU_DEVICE=$GPU_DEVICE PORT=$PORT MAX_NEW_TOKENS=$MAX_NEW_TOKENS TEMP=$TEMP"

# Step 1: provision endpoint
log "stopping any existing $CONTAINER_TAG"
docker rm -f "$CONTAINER_TAG" >/dev/null 2>&1 || true

log "starting endpoint"
docker run -d --name "$CONTAINER_TAG" \
  --gpus "device=${GPU_DEVICE}" \
  -p "${PORT}:8000" \
  -v /models:/models:ro \
  --shm-size=32g \
  "$IMG" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  >/dev/null

log "waiting for endpoint ready on port $PORT (up to 1200s)"
ready=false
for i in $(seq 1 120); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null; then
    log "endpoint ready after ${i}0s"
    ready=true
    break
  fi
  sleep 10
done
if [ "$ready" != "true" ]; then
  log "ERROR: endpoint never ready"
  docker logs --tail 60 "$CONTAINER_TAG" >&2 || true
  docker rm -f "$CONTAINER_TAG" >/dev/null 2>&1 || true
  exit 1
fi

trap 'log "tearing down $CONTAINER_TAG"; docker rm -f "$CONTAINER_TAG" >/dev/null 2>&1 || true' EXIT

BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="$BASE_URL"

run_evalplus(){
  local dataset=$1
  local out="$ROOT/evalplus_${dataset}"
  mkdir -p "$out"
  log "[evalplus $dataset] codegen start (max_new_tokens=$MAX_NEW_TOKENS)"
  cd /home/alexm/git/benchmarks/evalplus-venv
  $EP_PY -m evalplus.codegen \
    "$SERVED_NAME" "$dataset" \
    --backend openai \
    --base_url "$BASE_URL" \
    --temperature "$TEMP" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --n_samples 1 \
    --greedy False \
    --root "$out" \
    > "$out/codegen.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "[evalplus $dataset] codegen FAILED rc=$rc"; return $rc; fi

  local samples
  samples=$(find "$out" -name "*.jsonl" -not -name "*sanitized*" -size +0c | head -1)
  if [ -z "$samples" ]; then log "[evalplus $dataset] no samples produced"; return 1; fi

  log "[evalplus $dataset] sanitize + evaluate"
  $EP_PY -m evalplus.sanitize --samples "$samples" > "$out/sanitize.log" 2>&1
  local san="${samples%.jsonl}-sanitized.jsonl"
  $EP_PY -m evalplus.evaluate \
    --dataset "$dataset" --samples "$san" \
    > "$out/eval.log" 2>&1
  log "[evalplus $dataset] done"
}

# Health diagnostic: how many raw outputs reached </think>?
diag_thinking(){
  local out="$1" tag="$2"
  local raw
  raw=$(find "$out" -name "*.raw.jsonl" -size +0c | head -1)
  [ -z "$raw" ] && return 0
  $EP_PY - <<PYEOF
import json, sys
lines = open("$raw").readlines()
n_close = sum(1 for l in lines if "</think>" in (json.loads(l).get("solution") or ""))
print(f"[$tag] thinking-mode reached </think>: {n_close}/{len(lines)} samples", file=sys.stderr)
PYEOF
}

log "--- HumanEval+ ---"
run_evalplus humaneval
diag_thinking "$ROOT/evalplus_humaneval" "HE+"

log "--- MBPP+ ---"
run_evalplus mbpp
diag_thinking "$ROOT/evalplus_mbpp" "MBPP+"

# Marker for gap detector
date -u +%FT%TZ > "$ROOT/rerun.done"
log "done"
