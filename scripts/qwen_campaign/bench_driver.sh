#!/bin/bash
# Drive all benchmarks against one endpoint on cruelist.
# Output layout: /home/alexm/qwen_campaign/<MODEL_TAG>/{evalplus_*,lcb_*,bcb}
#
# Args:
#   MODEL_TAG        e.g. Qwen3.6-27B-FP8 (used in output paths + LCB save names)
#   LCB_MODEL_NAME   matches lcb_runner/lm_styles entry (qwen3.6-27b, qwen3.6-35b-a3b, qwen3-coder-next-bench)
#   VLLM_MODEL_NAME  served-model-name the endpoint accepts (often same as LCB_MODEL_NAME)
#   BASE_URL         http://kebab-rtx6000.lan:8000/v1
#   FAMILY           qwen3.6 | qwen3-coder-next   (sampling defaults)
set -uo pipefail

MODEL_TAG=$1
LCB_MODEL_NAME=$2
VLLM_MODEL_NAME=$3
BASE_URL=$4
FAMILY=$5

ROOT=/home/alexm/qwen_campaign/$MODEL_TAG
mkdir -p "$ROOT"
LOG="$ROOT/log.txt"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

case "$FAMILY" in
  qwen3.6)          TEMP=0.6; TOP_P=0.95 ;;
  qwen3-coder-next) TEMP=1.0; TOP_P=0.95 ;;
  *) log "ERROR unknown FAMILY=$FAMILY"; exit 1 ;;
esac
MAX_TOK=65536

log "=== START model=$MODEL_TAG endpoint=$BASE_URL family=$FAMILY temp=$TEMP top_p=$TOP_P max_tokens=$MAX_TOK ==="
if ! curl -fsS -o /dev/null --max-time 5 "${BASE_URL}/models" 2>/dev/null; then
  log "ERROR: endpoint $BASE_URL unreachable"
  exit 1
fi
log "endpoint reachable"

LCB_DIR=/home/alexm/git/benchmarks/LiveCodeBench
LCB_PY=$LCB_DIR/venv/bin/python
EP_PY=/home/alexm/git/benchmarks/evalplus-venv/bin/python
BCB_PY=/home/alexm/git/benchmarks/bigcodebench-venv/bin/python

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="$BASE_URL"

# ---------- 1. EvalPlus HumanEval+ ----------
run_evalplus(){
  local dataset=$1
  local out="$ROOT/evalplus_${dataset}"
  mkdir -p "$out"
  log "  [evalplus $dataset] codegen start"
  cd /home/alexm/git/benchmarks/evalplus-venv
  $EP_PY -m evalplus.codegen \
    "$VLLM_MODEL_NAME" "$dataset" \
    --backend openai \
    --base_url "$BASE_URL" \
    --temperature $TEMP \
    --n_samples 1 \
    --greedy False \
    --root "$out" \
    > "$out/codegen.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "  [evalplus $dataset] codegen FAILED rc=$rc"; return $rc; fi

  local samples
  samples=$(find "$out" -name "*.jsonl" -size +0c | head -1)
  if [ -z "$samples" ]; then log "  [evalplus $dataset] no samples produced"; return 1; fi

  log "  [evalplus $dataset] sanitize + evaluate"
  $EP_PY -m evalplus.sanitize --samples "$samples" > "$out/sanitize.log" 2>&1
  local san="${samples%.jsonl}-sanitized.jsonl"
  $EP_PY -m evalplus.evaluate \
    --dataset "$dataset" --samples "$san" \
    > "$out/eval.log" 2>&1
  log "  [evalplus $dataset] done"
}

# ---------- 2. LCB scenarios ----------
run_lcb(){
  local scen=$1
  local save="${MODEL_TAG}-${scen}-mt65k"
  log "  [LCB $scen] start (save_name=$save)"
  cd $LCB_DIR
  $LCB_PY -m lcb_runner.runner.main \
    --model "$LCB_MODEL_NAME" \
    --scenario "$scen" \
    --n 1 \
    --temperature $TEMP \
    --top_p $TOP_P \
    --max_tokens $MAX_TOK \
    --multiprocess 8 \
    --start_date 2024-08-01 \
    --end_date 2025-05-01 \
    --evaluate \
    --num_process_evaluate 12 \
    --openai_timeout 3600 \
    --custom_output_save_name "$save" \
    > "$ROOT/lcb_${scen}.log" 2>&1
  local rc=$?
  log "  [LCB $scen] done rc=$rc"
  return $rc
}

# ---------- 3. BigCodeBench ----------
run_bcb(){
  local out="$ROOT/bcb"
  mkdir -p "$out"
  log "  [BCB] generate start"
  cd "$out"
  $BCB_PY -m bigcodebench.generate \
    "$VLLM_MODEL_NAME" instruct full \
    --backend openai \
    --base_url "$BASE_URL" \
    --temperature $TEMP \
    --max_new_tokens $MAX_TOK \
    --bs 8 \
    --n_samples 1 \
    --resume True \
    > "$out/generate.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "  [BCB] generate FAILED rc=$rc"; return $rc; fi

  local gen
  gen=$(find "$out" -name "*.jsonl" -size +0c | head -1)
  if [ -z "$gen" ]; then log "  [BCB] no generations found"; return 1; fi

  log "  [BCB] sanitize + evaluate"
  $BCB_PY -m bigcodebench.sanitize --samples "$gen" > "$out/sanitize.log" 2>&1
  local san="${gen%.jsonl}-sanitized-calibrated.jsonl"
  [ -f "$san" ] || san="${gen%.jsonl}-sanitized.jsonl"
  $BCB_PY -m bigcodebench.evaluate \
    --split instruct --subset full --samples "$san" \
    > "$out/eval.log" 2>&1
  log "  [BCB] done"
}

# Execution order: short benches first, then long
log "--- Phase A: EvalPlus quick benches (parallel) ---"
run_evalplus humaneval &
PID_HE=$!
run_evalplus mbpp &
PID_MB=$!
wait $PID_HE; log "  HumanEval+ join rc=$?"
wait $PID_MB; log "  MBPP+ join rc=$?"

log "--- Phase B: LCB short scenarios (parallel) ---"
run_lcb testoutputprediction &
PID_TP=$!
run_lcb codeexecution &
PID_CE=$!
wait $PID_TP; log "  LCB testpred join rc=$?"
wait $PID_CE; log "  LCB codeexec join rc=$?"

log "--- Phase C: LCB codegeneration (the headline, long) ---"
run_lcb codegeneration

log "--- Phase D: BigCodeBench (long) ---"
run_bcb

log "=== COMPLETE for $MODEL_TAG ==="
