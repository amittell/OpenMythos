#!/usr/bin/env bash
# Round 2.2 launcher with regime presets.
#
# Wraps `training/launch_3b.sh` with one of two pre-baked env profiles:
#
#   conservative  (default, safe)
#     - micro_batch=1, grad_accum=4, USE_ACT_CKPT=0
#     - Same memory regime as round 2.1, just picks up the v3 script's
#       free wins: async ckpt I/O + ckpt_every=200.
#     - Expected wall-clock: ~5-8% faster than round 2.1.
#
#   aggressive    (faster, OOM risk)
#     - micro_batch=2, grad_accum=2, USE_ACT_CKPT=1
#     - Activation checkpointing on TransformerBlock to free activation
#       memory; trades ~30% extra compute per step for the 2x batch.
#     - Net throughput: maybe 15-25% faster, but unproven at this scale.
#     - Recommended: pre-flight with TARGET_TOKENS=2000000 (~5 min smoke
#       test) before committing to a multi-hour run.
#
# Usage:
#   bash training/launch_round22.sh                         # conservative
#   bash training/launch_round22.sh conservative
#   bash training/launch_round22.sh aggressive
#   bash training/launch_round22.sh smoke-aggressive        # 2M-token test
#
# Any extra env vars set by the caller win against the preset.

set -euo pipefail

REGIME=${1:-conservative}

case "${REGIME}" in
  conservative)
    PRESET_ENV=(
      "MICRO_BATCH=1"
      "GRAD_ACCUM=4"
      "USE_ACT_CKPT=0"
    )
    ;;
  aggressive)
    PRESET_ENV=(
      "MICRO_BATCH=2"
      "GRAD_ACCUM=2"
      "USE_ACT_CKPT=1"
    )
    ;;
  smoke-aggressive)
    PRESET_ENV=(
      "MICRO_BATCH=2"
      "GRAD_ACCUM=2"
      "USE_ACT_CKPT=1"
      "TARGET_TOKENS=2000000"
      "CKPT_EVERY=50"
      "WARMUP_STEPS=20"
    )
    ;;
  *)
    echo "usage: $0 [conservative|aggressive|smoke-aggressive]" >&2
    exit 2
    ;;
esac

echo "Round 2.2 launch profile: ${REGIME}"
for kv in "${PRESET_ENV[@]}"; do echo "  $kv"; done

# Hand off to the standard launcher with the preset env in scope. Caller
# env still wins because Bash gives caller-scope precedence over the
# explicit assignments below when SCRIPT/etc. are already exported.
exec env "${PRESET_ENV[@]}" \
  SCRIPT=${SCRIPT:-training/3b_varT_act_v3.py} \
  bash "$(dirname "$0")/launch_3b.sh"
