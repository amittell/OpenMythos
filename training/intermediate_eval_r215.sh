#!/usr/bin/env bash
# Run an intermediate-checkpoint eval of r2.15 on kebab-rtx6000 GPU 1.
#
# Pipeline (idempotent, skip-if-output-exists):
#   1. Stage rank 1/2/3 shards from gx10 nodes to kebab-spark
#   2. Run streaming consolidator on kebab-spark (single process, ~2 min, ~17 GB RAM)
#      -- replaces the OLD 4-rank torchrun consolidator that OOM'd at 295 GB
#   3. Ship the 16 GB consolidated full.pt from spark to kebab-rtx6000
#      (vs the OLD 44 GB-of-rank-shards transfer)
#   4. Use kebab-rtx-router's /admin/models/unload to evict vision off GPU 1
#   5. Run the eval bundle on kebab-rtx6000 GPU 1
#   6. Re-load vision via /admin/models/load
#   7. Pull eval JSON results back to kebab-spark
#
# Usage (run on kebab-spark.lan, your laptop, or anywhere with ssh access):
#
#   bash training/intermediate_eval_r215.sh 7400
#   EVALS="per_token_halt_analysis depth_extrap" bash training/intermediate_eval_r215.sh 7400
#   bash training/intermediate_eval_r215.sh 7400 --force        # ignore existing outputs
#
# Env vars:
#   EVALS  whitespace-separated list of eval script basenames (no .py).
#          Default: "per_token_halt_analysis" (the cheap one, ~80s).
#          Heavier: "per_token_halt_analysis depth_extrap reasoning_eval gen_samples_multidepth".

set -euo pipefail

STEP=${1:?STEP required (e.g. 7400)}
FORCE_FLAG=${2:-}
PADDED=$(printf "%07d" "$STEP")
CKPT_DIR=checkpoints_3b_varT_pondernet_round215
SHARD_PREFIX="$CKPT_DIR/step_${PADDED}"
FULL_NAME="step_${PADDED}_full.pt"
FULL_PATH="$CKPT_DIR/$FULL_NAME"

REPO_SPARK=/home/alexm/OpenMythos
REPO_RTX=/home/alexm/OpenMythos
ROUTER_URL=http://kebab-rtx6000.lan:8080
EVALS=${EVALS:-per_token_halt_analysis}

# Where the 3 rank shards live (one per gx10 node, via .200g storage net).
declare -A RANK_HOST=(
  [1]=kebab-gx10-200g
  [2]=kebab-gx10-2-200g
  [3]=kebab-gx10-3-200g
)

log() { echo "[$(date '+%H:%M:%S')] [r215-eval step=$STEP] $*"; }

# ---- Idempotency: if all requested eval outputs exist AND are valid JSON, exit early ----
# Bug we caught the hard way: `test -f` accepts 0-byte / corrupt JSON, so a
# crashed eval that left a stub file would be silently treated as "already
# done" forever. Now validates non-empty + json.load parseability.
json_valid_remote() {
    local host="$1" path="$2"
    ssh -o ConnectTimeout=5 -q "$host" \
        "test -s '$path' && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' '$path' 2>/dev/null" \
        2>/dev/null
}

if [[ "$FORCE_FLAG" != "--force" ]]; then
    all_valid=1
    for evalname in $EVALS; do
        out="$REPO_SPARK/docs/intermediate_r215_step${STEP}_${evalname}.json"
        if ! json_valid_remote alexm@kebab-spark.lan "$out"; then
            all_valid=0
            break
        fi
    done
    if [[ "$all_valid" -eq 1 ]]; then
        log "all eval outputs already present + valid on spark; pass --force to rerun"
        exit 0
    fi
fi

log "starting pipeline; evals=[$EVALS]"

# ---- 1. Verify rank-0 shard on kebab-spark ----
ssh -q alexm@kebab-spark.lan "test -f $REPO_SPARK/${SHARD_PREFIX}_rank0.pt" || {
    log "FATAL: rank-0 shard not on kebab-spark: $SHARD_PREFIX"
    exit 1
}

# ---- 2. Pre-check: all 4 ranks reachable on their origin nodes BEFORE we rsync ----
# Training cleans up old shards on gx10 nodes (keeps last ~2 saves). If we try
# to rsync a step that's already been deleted from gx10 we get a silent
# "broken pipe" and the consolidator runs with only rank-0, producing a
# corrupted full state dict. Verify presence first.
log "pre-check: rank 1/2/3 source files exist on origin nodes..."
for r in 1 2 3; do
    h=${RANK_HOST[$r]}
    src_path="$REPO_SPARK/${SHARD_PREFIX}_rank${r}.pt"
    if ! ssh -o ConnectTimeout=5 -q alexm@$h "test -f $src_path" 2>/dev/null; then
        log "FATAL: rank $r shard not on $h (training likely cleaned it; pick a newer step)"
        exit 2
    fi
done
log "  all 4 source shards present"

# ---- 3. Stage rank 1/2/3 to kebab-spark in parallel, checking each exit code ----
log "rsync rank 1/2/3 from gx10 nodes to spark..."
declare -A RSYNC_PIDS
for r in 1 2 3; do
    h=${RANK_HOST[$r]}
    src="alexm@$h:$REPO_SPARK/${SHARD_PREFIX}_rank${r}.pt"
    dst="$REPO_SPARK/${SHARD_PREFIX}_rank${r}.pt"
    ssh -o ConnectTimeout=10 -q alexm@kebab-spark.lan \
        "test -f $dst || rsync -az --partial '$src' '$dst'" &
    RSYNC_PIDS[$r]=$!
done
fail_count=0
for r in 1 2 3; do
    if ! wait "${RSYNC_PIDS[$r]}"; then
        log "FATAL: rsync rank $r failed (pid ${RSYNC_PIDS[$r]})"
        fail_count=$((fail_count + 1))
    fi
done
if [[ "$fail_count" -gt 0 ]]; then
    log "FATAL: $fail_count rsyncs failed; aborting (no consolidator dispatched)"
    exit 2
fi

# Belt-and-braces: verify all 4 rank files now exist on spark with non-zero size
for r in 0 1 2 3; do
    dst="$REPO_SPARK/${SHARD_PREFIX}_rank${r}.pt"
    if ! ssh -q alexm@kebab-spark.lan "test -s $dst" 2>/dev/null; then
        log "FATAL: rank $r missing or empty on spark after stage: $dst"
        exit 2
    fi
done
log "all 4 ranks staged on spark (verified)"

# ---- 3. Run streaming consolidator on kebab-spark (single process, ~2 min) ----
if ssh -q alexm@kebab-spark.lan "test -f $REPO_SPARK/$FULL_PATH" && [[ "$FORCE_FLAG" != "--force" ]]; then
    log "consolidated ckpt already exists on spark; skipping consolidator"
else
    log "running streaming consolidator on spark..."
    ssh -q alexm@kebab-spark.lan "
        cd $REPO_SPARK && \
        python3 training/consolidate_ckpt_streaming.py $SHARD_PREFIX $FULL_PATH
    "
fi
log "consolidated ckpt at spark:$REPO_SPARK/$FULL_PATH"

# ---- 4. Ship the 16 GB full.pt to kebab-rtx6000 ----
log "ship consolidated ckpt to rtx6000..."
ssh -q alexm@kebab-spark.lan "mkdir -p $REPO_SPARK/$CKPT_DIR" 2>/dev/null
ssh -q alexm@kebab-rtx6000.lan "mkdir -p $REPO_RTX/$CKPT_DIR"
ssh -q alexm@kebab-spark.lan "rsync -avz --partial $REPO_SPARK/$FULL_PATH alexm@kebab-rtx6000.lan:$REPO_RTX/$FULL_PATH"
log "consolidated ckpt at rtx6000:$REPO_RTX/$FULL_PATH"

# ---- 5. Evict vision via the router we built (frees GPU 1 for the eval) ----
log "unload vision-cuda via $ROUTER_URL/admin/models/unload..."
unload_resp=$(curl -s --max-time 10 -X POST "$ROUTER_URL/admin/models/unload" \
    -H 'Content-Type: application/json' \
    -d '{"model_id":"qwen3-vl-32b"}' || echo "{}")
echo "    -> $unload_resp"
# kebab-rtx-router's admin/models/unload uses systemctl; tolerate a stale
# "unknown model: qwen3-vl-32b" by falling back to systemctl on the host.
if ! echo "$unload_resp" | grep -q '"status"'; then
    log "router unload didn't respond cleanly; falling back to ssh+systemctl"
    ssh -q alexm@kebab-rtx6000.lan "sudo systemctl stop kebab-rtx-vision-cuda.service"
fi

# ---- 6. Run eval bundle on kebab-rtx6000 GPU 1 ----
PYTHON=/home/alexm/venvs/vllm-turboquant/bin/python3
log "run eval bundle on rtx6000 GPU 1: $EVALS"
eval_failures=0
for evalname in $EVALS; do
    out="$REPO_RTX/docs/intermediate_r215_step${STEP}_${evalname}.json"
    if [[ "$FORCE_FLAG" != "--force" ]] \
        && json_valid_remote alexm@kebab-rtx6000.lan "$out"; then
        log "  $evalname: valid output exists, skipping"
        continue
    fi
    log "  $evalname running..."
    if ! ssh -q alexm@kebab-rtx6000.lan "
        cd $REPO_RTX && \
        CUDA_VISIBLE_DEVICES=1 \
        CKPT=$REPO_RTX/$FULL_PATH \
        CKPT_DIR=$CKPT_DIR \
        OUT=$out \
        $PYTHON training/${evalname}.py
    "; then
        log "  $evalname FAILED (script exited non-zero)"
        eval_failures=$((eval_failures + 1))
        continue
    fi
    # Validate the output before declaring success -- a script that exits 0
    # but produced a 0-byte / unparseable JSON should be flagged so the next
    # watcher tick re-runs it instead of permanently treating it as "done".
    if ! json_valid_remote alexm@kebab-rtx6000.lan "$out"; then
        log "  $evalname FAILED (script exited 0 but output is missing or invalid JSON)"
        # Remove the bogus file so the per-eval idempotency check above
        # re-dispatches on the next watcher tick.
        ssh -q alexm@kebab-rtx6000.lan "rm -f $out" 2>/dev/null
        eval_failures=$((eval_failures + 1))
        continue
    fi
    log "  $evalname done -> $out"
done

# ---- 7. Restart vision-cuda via the router ----
log "reload vision-cuda via $ROUTER_URL/admin/models/load..."
load_resp=$(curl -s --max-time 240 -X POST "$ROUTER_URL/admin/models/load" \
    -H 'Content-Type: application/json' \
    -d '{"model_id":"qwen3-vl-32b"}' || echo "{}")
echo "    -> $(echo "$load_resp" | head -c 200)"
if ! echo "$load_resp" | grep -qE '"status":"(loaded|already_loaded)"'; then
    log "router load didn't confirm; falling back to ssh+systemctl"
    ssh -q alexm@kebab-rtx6000.lan "sudo systemctl start kebab-rtx-vision-cuda.service"
fi

# ---- 8. Pull eval JSON results back to spark for the paper pipeline ----
log "pull eval JSON outputs back to spark..."
for evalname in $EVALS; do
    out="docs/intermediate_r215_step${STEP}_${evalname}.json"
    ssh -q alexm@kebab-spark.lan "
        rsync -az alexm@kebab-rtx6000.lan:$REPO_RTX/$out $REPO_SPARK/$out 2>/dev/null
    " || true
done

# ---- 9. Cleanup: free disk on spark + rtx6000 only if every eval succeeded ----
# Bytes per evaluated step otherwise leak permanently:
#   spark:   3x ~11 GB rank shards (rsync'd from gx10) + 1x ~16 GB full.pt
#   rtx6000: 1x ~16 GB full.pt
# Rank 0 on spark is kept (training writes it; training also self-prunes).
# All deletes guarded by json_valid_remote() checks so we never throw away
# the consolidated ckpt when an eval might still need a retry.
if [[ "$eval_failures" -eq 0 ]]; then
    all_valid=1
    for evalname in $EVALS; do
        out_spark="$REPO_SPARK/docs/intermediate_r215_step${STEP}_${evalname}.json"
        if ! json_valid_remote alexm@kebab-spark.lan "$out_spark"; then
            all_valid=0
            break
        fi
    done
    if [[ "$all_valid" -eq 1 ]]; then
        log "cleanup: all evals valid, freeing disk..."
        # spark: ranks 1/2/3 (rsync'd from gx10; originals still on gx10) + full.pt
        ssh -q alexm@kebab-spark.lan "
            rm -f $REPO_SPARK/${SHARD_PREFIX}_rank1.pt \
                  $REPO_SPARK/${SHARD_PREFIX}_rank2.pt \
                  $REPO_SPARK/${SHARD_PREFIX}_rank3.pt \
                  $REPO_SPARK/$FULL_PATH
        " 2>/dev/null
        # rtx6000: full.pt (eval JSONs are tiny and pulled back to spark)
        ssh -q alexm@kebab-rtx6000.lan "rm -f $REPO_RTX/$FULL_PATH" 2>/dev/null
        log "cleanup: freed ~60 GB on spark + ~16 GB on rtx6000"
    else
        log "cleanup: skipped (some eval JSONs not yet valid on spark; retry on next tick)"
    fi
else
    log "cleanup: skipped ($eval_failures eval failure(s); preserving ckpts for retry)"
fi

if [[ "$eval_failures" -gt 0 ]]; then
    log "PIPELINE PARTIAL step=$STEP evals=[$EVALS] failures=$eval_failures"
    exit 1
fi

log "PIPELINE COMPLETE step=$STEP evals=[$EVALS]"
