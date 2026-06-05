#!/bin/bash
# Print a snapshot of the qwen campaign status.
# Usage: qwen_campaign_monitor.sh [tail-lines]
set -uo pipefail
ROOT=/home/alexm/qwen_campaign
TAIL=${1:-5}

echo "=================================================================="
echo "Qwen eval campaign status as of $(date '+%F %T %Z')"
echo "=================================================================="

# 1. Endpoint health
echo
echo "[Endpoints]"
for entry in "GPU0:rtx6000.lan:8000" "GPU1:rtx6000.lan:8002"; do
  IFS=: read -r tag host port <<<"$entry"
  # remove leading whitespace
  if curl -fsS --max-time 3 "http://kebab-$host:$port/v1/models" 2>/dev/null >/tmp/_models_check; then
    names=$(python3 -c "import json; d=json.load(open('/tmp/_models_check')); print(','.join(m['id'] for m in d['data']))" 2>/dev/null)
    printf "  %-6s port %s  UP    served=[%s]\n" "$tag" "$port" "$names"
  else
    printf "  %-6s port %s  DOWN\n" "$tag" "$port"
  fi
done

# 2. Per-model progress
echo
echo "[Per-model progress]"
if [ ! -d "$ROOT" ]; then echo "  (no campaign root yet)"; exit; fi
for dir in $ROOT/*/; do
  [ -d "$dir" ] || continue
  tag=$(basename "$dir")
  log="$dir/log.txt"
  [ -f "$log" ] || continue
  last=$(tail -1 "$log" 2>/dev/null)
  printf "  %-30s  last: %s\n" "$tag" "${last:0:90}"
done

# 3. Bench output sizes (proxy for progress)
echo
echo "[Output volumes (size = work done)]"
for dir in $ROOT/*/; do
  [ -d "$dir" ] || continue
  tag=$(basename "$dir")
  for sub in evalplus_humaneval evalplus_mbpp lcb_codegen lcb_testpred lcb_codeexec bcb; do
    sz=$(du -sh "$dir$sub" 2>/dev/null | awk '{print $1}')
    [ -n "$sz" ] && printf "  %-30s %-22s %s\n" "$tag" "$sub" "$sz"
  done
done

# 4. Campaign master log
echo
echo "[Campaign master log tail]"
if [ -f $ROOT/campaign.log ]; then
  tail -$TAIL $ROOT/campaign.log | sed 's/^/  /'
else
  echo "  (no campaign.log)"
fi
