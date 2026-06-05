#!/bin/bash
# Campaign orchestrator v2: handles all 3 phases, detects already-running endpoints.
# Run on cruelist; SSHes to rtx6000 to manage vLLM containers.
set -uo pipefail

CAMPAIGN_ROOT=/home/alexm/qwen_campaign
mkdir -p "$CAMPAIGN_ROOT"
MASTER_LOG="$CAMPAIGN_ROOT/campaign.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

RTX_HOST=kebab-rtx6000.lan
RTX_USER=alexm
BASE_GPU0=http://${RTX_HOST}:8000/v1
BASE_GPU1=http://${RTX_HOST}:8002/v1

endpoint_serves(){
  local port=$1 expected_name=$2
  local out
  out=$(curl -fsS --max-time 5 http://${RTX_HOST}:${port}/v1/models 2>/dev/null) || return 1
  echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); names=[m['id'] for m in d['data']]; sys.exit(0 if '$expected_name' in names else 1)" 2>/dev/null
}

start_endpoint(){
  local gpu=$1 port=$2 model_path=$3 served_names=$4 max_num_seqs=$5 primary_name=$6
  # If already serving the right model, skip
  if endpoint_serves $port "$primary_name"; then
    log "  GPU$gpu port $port already serving $primary_name -- skip start"
    return 0
  fi
  log "  starting GPU$gpu: $model_path -> $served_names port=$port mns=$max_num_seqs"
  ssh -o ConnectTimeout=10 -4 ${RTX_USER}@${RTX_HOST} \
    "/home/alexm/cascade/start_endpoint_v2.sh \"$model_path\" \"$served_names\" $port $gpu $max_num_seqs \"\"" \
    >> "$MASTER_LOG" 2>&1
}

stop_endpoint(){
  local gpu=$1
  log "  stopping GPU$gpu container"
  ssh -o ConnectTimeout=10 -4 ${RTX_USER}@${RTX_HOST} \
    "docker rm -f kebab-vllm-eval-gpu${gpu} 2>/dev/null || true; sleep 5" \
    >> "$MASTER_LOG" 2>&1
}

run_phase_dual(){
  local label=$1
  local gpu0_model_path=$2 gpu0_served=$3 gpu0_tag=$4 gpu0_lcb=$5 gpu0_family=$6 gpu0_mns=$7
  local gpu1_model_path=$8 gpu1_served=$9 gpu1_tag=${10} gpu1_lcb=${11} gpu1_family=${12} gpu1_mns=${13}

  log "==== PHASE $label START (dual: $gpu0_tag + $gpu1_tag) ===="

  # If new models needed, tear down + start; otherwise reuse
  local g0_primary="${gpu0_served%% *}"
  local g1_primary="${gpu1_served%% *}"

  if ! endpoint_serves 8000 "$g0_primary"; then
    stop_endpoint 0
    start_endpoint 0 8000 "$gpu0_model_path" "$gpu0_served" "$gpu0_mns" "$g0_primary"
  fi
  if ! endpoint_serves 8002 "$g1_primary"; then
    stop_endpoint 1
    start_endpoint 1 8002 "$gpu1_model_path" "$gpu1_served" "$gpu1_mns" "$g1_primary"
  fi

  # Drivers
  mkdir -p "$CAMPAIGN_ROOT/$gpu0_tag" "$CAMPAIGN_ROOT/$gpu1_tag"
  log "  spawning drivers"
  nohup /home/alexm/qwen_bench_driver.sh \
    "$gpu0_tag" "$gpu0_lcb" "$g0_primary" "$BASE_GPU0" "$gpu0_family" \
    >> "$CAMPAIGN_ROOT/${gpu0_tag}/driver.out" 2>&1 &
  D0=$!
  nohup /home/alexm/qwen_bench_driver.sh \
    "$gpu1_tag" "$gpu1_lcb" "$g1_primary" "$BASE_GPU1" "$gpu1_family" \
    >> "$CAMPAIGN_ROOT/${gpu1_tag}/driver.out" 2>&1 &
  D1=$!
  log "  driver pids: GPU0=$D0 GPU1=$D1"

  wait $D0; log "  $gpu0_tag driver join rc=$?"
  wait $D1; log "  $gpu1_tag driver join rc=$?"
  log "==== PHASE $label END ===="
}

run_phase_single(){
  local label=$1
  local model_path=$2 served=$3 tag=$4 lcb=$5 family=$6 mns=$7

  log "==== PHASE $label START (single: $tag on GPU0) ===="
  local primary="${served%% *}"

  stop_endpoint 0
  stop_endpoint 1
  start_endpoint 0 8000 "$model_path" "$served" "$mns" "$primary"

  mkdir -p "$CAMPAIGN_ROOT/$tag"
  nohup /home/alexm/qwen_bench_driver.sh \
    "$tag" "$lcb" "$primary" "$BASE_GPU0" "$family" \
    >> "$CAMPAIGN_ROOT/${tag}/driver.out" 2>&1
  log "==== PHASE $label END ===="
}

# ------ PHASES ------
# Phase 1: 27B-FP8 + 35B-A3B-FP8 in parallel
run_phase_dual 1 \
  /models/Qwen3.6-27B-FP8     "qwen3.6-27b qwen3.6-27b-fp8"         Qwen3.6-27B-FP8     qwen3.6-27b     qwen3.6 4 \
  /models/Qwen3.6-35B-A3B-FP8 "qwen3.6-35b-a3b qwen3.6-35b-a3b-fp8" Qwen3.6-35B-A3B-FP8 qwen3.6-35b-a3b qwen3.6 2

# Phase 2: Coder-Next-FP8 + 27B-BF16 in parallel
run_phase_dual 2 \
  /models/Qwen3-Coder-Next-FP8 "qwen3-coder-next-bench qwen3-coder-next" Qwen3-Coder-Next-FP8 qwen3-coder-next-bench qwen3-coder-next 2 \
  /models/Qwen3.6-27B          "qwen3.6-27b qwen3.6-27b-bf16"            Qwen3.6-27B-BF16     qwen3.6-27b            qwen3.6         4

# Phase 3: 35B-A3B-BF16 alone
run_phase_single 3 \
  /models/Qwen3.6-35B-A3B "qwen3.6-35b-a3b qwen3.6-35b-a3b-bf16" Qwen3.6-35B-A3B-BF16 qwen3.6-35b-a3b qwen3.6 2

# Cleanup + restart gpt-oss
log "==== CAMPAIGN COMPLETE -- tearing down eval endpoints ===="
stop_endpoint 0
stop_endpoint 1
log "==== Restarting gpt-oss-120b production service ===="
ssh -o ConnectTimeout=10 -4 ${RTX_USER}@${RTX_HOST} 'sudo systemctl start kebab-rtx-vllm.service' >> "$MASTER_LOG" 2>&1
log "==== gpt-oss-120b restart complete; campaign fully done ===="
