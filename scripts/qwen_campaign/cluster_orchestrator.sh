#!/bin/bash
# Cluster-augmented campaign orchestrator (v2 clean structure).
# 5 endpoints, 5 models in parallel:
#   rtx6000 GPU1  : Qwen3.6-35B-A3B-BF16
#   kebab-spark   : Qwen3-Coder-Next-FP8
#   kebab-gx10    : Qwen3.6-27B-FP8
#   kebab-gx10-2  : Qwen3.6-35B-A3B-FP8
#   kebab-gx10-3  : Qwen3.6-27B-BF16
set -uo pipefail

CAMPAIGN_ROOT=/home/alexm/qwen_campaign_v2
mkdir -p "$CAMPAIGN_ROOT"
MASTER_LOG="$CAMPAIGN_ROOT/campaign.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

IMG=vllm/vllm-openai:v0.21.0

start_endpoint(){
  local tag=$1 host=$2 port=$3 model_path=$4 served_names=$5 gpu=$6 mns=$7
  local cn="kebab-vllm-eval-gpu${gpu}"
  log "  [start] $tag on $host gpu=$gpu port=$port mns=$mns"
  ssh -o ConnectTimeout=10 alexm@$host \
    "docker rm -f $cn 2>/dev/null
     docker run -d --name $cn \
       --gpus device=$gpu \
       -p ${port}:8000 \
       -v \$(dirname $model_path):\$(dirname $model_path):ro \
       --shm-size=32g \
       $IMG \
       --model $model_path \
       --served-model-name $served_names \
       --max-model-len 262144 \
       --gpu-memory-utilization 0.92 \
       --max-num-seqs $mns \
       --tensor-parallel-size 1 \
       --trust-remote-code" >> "$MASTER_LOG" 2>&1
}

wait_ready(){
  local tag=$1 host=$2 port=$3
  log "  [wait]  $tag at $host:$port"
  for i in $(seq 1 180); do
    if curl -fsS --max-time 5 "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      log "  [READY] $tag after ${i}0s"
      return 0
    fi
    sleep 10
  done
  log "  [FAIL]  $tag never ready"
  ssh -o ConnectTimeout=5 alexm@$host "docker logs --tail 40 kebab-vllm-eval-gpu0 2>&1; docker logs --tail 40 kebab-vllm-eval-gpu1 2>&1" >> "$MASTER_LOG" 2>&1
  return 1
}

spawn_driver(){
  local tag=$1 lcb_name=$2 served_primary=$3 base_url=$4 family=$5
  mkdir -p "$CAMPAIGN_ROOT/$tag"
  nohup /home/alexm/qwen_bench_driver.sh \
    "$tag" "$lcb_name" "$served_primary" "$base_url" "$family" \
    >> "$CAMPAIGN_ROOT/${tag}/driver.out" 2>&1 &
  log "  [driver] $tag pid=$! url=$base_url"
  echo $!
}

# ---------- Endpoint definitions ----------
declare -a TAGS HOSTS PORTS MODEL_PATHS SERVED LCB_NAMES FAMILIES GPUS MNS

T0=Qwen3.6-35B-A3B-BF16; H0=kebab-rtx6000.lan; P0=8002
M0=/models/Qwen3.6-35B-A3B; S0="qwen3.6-35b-a3b qwen3.6-35b-a3b-bf16"
L0=qwen3.6-35b-a3b; F0=qwen3.6; G0=1; N0=2

T1=Qwen3-Coder-Next-FP8; H1=kebab-spark.lan; P1=8001
M1=/home/alexm/models/Qwen3-Coder-Next-FP8; S1="qwen3-coder-next-bench qwen3-coder-next"
L1=qwen3-coder-next-bench; F1=qwen3-coder-next; G1=0; N1=2

T2=Qwen3.6-27B-FP8; H2=kebab-gx10.lan; P2=8001
M2=/home/alexm/models/Qwen3.6-27B-FP8; S2="qwen3.6-27b qwen3.6-27b-fp8"
L2=qwen3.6-27b; F2=qwen3.6; G2=0; N2=4

T3=Qwen3.6-35B-A3B-FP8; H3=kebab-gx10-2.lan; P3=8001
M3=/home/alexm/models/Qwen3.6-35B-A3B-FP8; S3="qwen3.6-35b-a3b qwen3.6-35b-a3b-fp8"
L3=qwen3.6-35b-a3b; F3=qwen3.6; G3=0; N3=2

T4=Qwen3.6-27B-BF16; H4=kebab-gx10-3.lan; P4=8001
M4=/home/alexm/models/Qwen3.6-27B; S4="qwen3.6-27b qwen3.6-27b-bf16"
L4=qwen3.6-27b; F4=qwen3.6; G4=0; N4=4

# ---------- Start all 5 endpoints in parallel ----------
log "==== Starting 5 endpoints in parallel ===="
start_endpoint "$T0" "$H0" "$P0" "$M0" "$S0" "$G0" "$N0" &
start_endpoint "$T1" "$H1" "$P1" "$M1" "$S1" "$G1" "$N1" &
start_endpoint "$T2" "$H2" "$P2" "$M2" "$S2" "$G2" "$N2" &
start_endpoint "$T3" "$H3" "$P3" "$M3" "$S3" "$G3" "$N3" &
start_endpoint "$T4" "$H4" "$P4" "$M4" "$S4" "$G4" "$N4" &
wait

log "==== Waiting for all endpoints ready ===="
wait_ready "$T0" "$H0" "$P0" &
wait_ready "$T1" "$H1" "$P1" &
wait_ready "$T2" "$H2" "$P2" &
wait_ready "$T3" "$H3" "$P3" &
wait_ready "$T4" "$H4" "$P4" &
wait

log "==== All endpoints ready. Spawning 5 bench drivers ===="
PID0=$(spawn_driver "$T0" "$L0" "${S0%% *}" "http://${H0}:${P0}/v1" "$F0")
PID1=$(spawn_driver "$T1" "$L1" "${S1%% *}" "http://${H1}:${P1}/v1" "$F1")
PID2=$(spawn_driver "$T2" "$L2" "${S2%% *}" "http://${H2}:${P2}/v1" "$F2")
PID3=$(spawn_driver "$T3" "$L3" "${S3%% *}" "http://${H3}:${P3}/v1" "$F3")
PID4=$(spawn_driver "$T4" "$L4" "${S4%% *}" "http://${H4}:${P4}/v1" "$F4")

log "==== Drivers running. Waiting for all to complete (campaign in progress) ===="
wait $PID0; log "  $T0 driver join rc=$?"
wait $PID1; log "  $T1 driver join rc=$?"
wait $PID2; log "  $T2 driver join rc=$?"
wait $PID3; log "  $T3 driver join rc=$?"
wait $PID4; log "  $T4 driver join rc=$?"

log "==== ALL BENCHES COMPLETE. Tearing down endpoints ===="
for entry in "$H0:$G0" "$H1:$G1" "$H2:$G2" "$H3:$G3" "$H4:$G4"; do
  host="${entry%%:*}"; gpu="${entry##*:}"
  ssh -o ConnectTimeout=10 alexm@$host "docker rm -f kebab-vllm-eval-gpu${gpu} 2>/dev/null" >> "$MASTER_LOG" 2>&1
done

log "==== Restarting gpt-oss-120b on rtx6000 GPU0 ===="
ssh -o ConnectTimeout=10 alexm@kebab-rtx6000.lan 'sudo systemctl start kebab-rtx-vllm.service' >> "$MASTER_LOG" 2>&1

log "==== CAMPAIGN COMPLETE ===="
