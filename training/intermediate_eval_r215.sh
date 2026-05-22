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

RTX=alexm@kebab-rtx6000.lan

# ---- GPU-1 mutex + guaranteed vision restore ----------------------------
# The eval phase (unload vision -> run evals -> reload vision) has an exclusive
# claim on rtx6000 GPU 1. Two eval invocations overlapping here is exactly the
# dogpile we hit on 2026-05-20 (a stray cycle fighting the live one for the GPU,
# vision flapping). Two guards:
#   1. A cross-host advisory lock on rtx6000 (atomic `mkdir`, with stale
#      takeover) so only one eval owns GPU 1 at a time -- even a manual run
#      can't collide with the watcher.
#   2. An EXIT trap that ALWAYS reloads vision (if we unloaded it) and releases
#      the lock, so a timeout-killed or crashed cycle never leaves the GPU
#      stranded unloaded (degraded router) or the lock held.
GPU_LOCK_DIR=${GPU_LOCK_DIR:-/tmp/r215_gpu1.lock}   # on rtx6000
GPU_LOCK_TTL=${GPU_LOCK_TTL:-3000}                  # steal if older than this (> cycle timeout)
GPU_LOCK_HELD=0

# Names of router backends we evicted from GPU 1 so the cleanup trap can
# restore them after the eval. Populated by `free_gpu1` from the router's
# /admin/gpus/1/free response; consumed by `restore_gpu1_tenants`. Bash
# arrays survive function scope as long as we don't `declare` them inside
# the function.
GPU1_RESTORE_LIST=()

# Ask the router to free every backend currently loaded on cuda_device=1.
# Replaces the older "unload qwen3-vl-32b" call which knew only about
# vision. As of 2026-05-22 GPU 1 may also host the embedding service
# (relocated 0->1) and a parallel coder backend (qwen3-coder-next-gpu1);
# the GPU-level endpoint handles all of them atomically without the
# eval-script needing to know the inventory. See kebab_rtx_router_fastapi.py
# `admin_gpus_free` for response shape: {status, cuda_device, unloaded,
# skipped, errors}.
free_gpu1() {
    log "free GPU 1 via $ROUTER_URL/admin/gpus/1/free..."
    local resp
    # --max-time 240 covers the worst case of vLLM drain (docker stop --time=120)
    # plus router-side per-backend serialisation.
    resp=$(curl -s --max-time 300 -X POST "$ROUTER_URL/admin/gpus/1/free" \
        -H 'Content-Type: application/json' -d '{}' || echo "{}")
    echo "    -> $resp"
    if ! echo "$resp" | grep -qE '"status":"(free|partial)"'; then
        log "WARN: /admin/gpus/1/free returned non-{free,partial} status; eval may OOM if GPU 1 still occupied"
        return 1
    fi
    # Parse the `unloaded` list so the EXIT trap can restore them.
    local list_json
    list_json=$(echo "$resp" | python3 -c \
        "import json,sys; d=json.load(sys.stdin); print(' '.join(d.get('unloaded', [])))" 2>/dev/null)
    GPU1_RESTORE_LIST=($list_json)
    log "freed GPU 1, will restore: ${GPU1_RESTORE_LIST[*]:-(nothing to restore)}"
    return 0
}

# Restore everything we evicted from GPU 1 via /admin/models/load. Iterates
# GPU1_RESTORE_LIST. Idempotent: a model already loaded returns
# already_loaded and we move on. Ordering: vision-cuda first (largest, most
# user-visible) then embedding then anything else, so a partial recovery at
# least gets the highest-impact backends back.
restore_gpu1_tenants() {
    (( ${#GPU1_RESTORE_LIST[@]} )) || return 0
    local sorted=()
    local m
    # Stable priority ordering of known backends.
    for m in qwen3-vl-32b embedding; do
        for actual in "${GPU1_RESTORE_LIST[@]}"; do
            [[ "$actual" == "$m" ]] && sorted+=("$m")
        done
    done
    # Append any leftovers that weren't in the explicit priority list.
    for actual in "${GPU1_RESTORE_LIST[@]}"; do
        local found=0
        for s in "${sorted[@]}"; do [[ "$s" == "$actual" ]] && found=1 && break; done
        (( found )) || sorted+=("$actual")
    done

    log "restore GPU 1 tenants: ${sorted[*]}"
    local m resp
    local remaining=()
    for m in "${sorted[@]}"; do
        resp=$(curl -s --max-time 300 -X POST "$ROUTER_URL/admin/models/load" \
            -H 'Content-Type: application/json' -d "{\"model_id\":\"$m\"}" || echo "{}")
        if echo "$resp" | grep -qE '"status":"(loaded|already_loaded)"'; then
            log "  $m: loaded"
        else
            log "  $m: router load did not confirm; falling back to ssh+systemctl"
            # reset-failed defensively in case the backend's previous stop
            # left the unit in `failed` (e.g. docker SIGKILL on slow drain).
            local unit=""
            case "$m" in
                qwen3-vl-32b)         unit="kebab-rtx-vision-cuda.service" ;;
                embedding)            unit="kebab-rtx-embedding.service" ;;
                qwen3-coder-next-gpu1) unit="kebab-rtx-vllm-coder-gpu1.service" ;;
                gpt-oss-120b)         unit="kebab-rtx-vllm.service" ;;
                qwen3-coder-next)     unit="kebab-rtx-vllm-coder.service" ;;
            esac
            if [[ -n "$unit" ]]; then
                ssh -o ConnectTimeout=10 -q "$RTX" \
                    "sudo systemctl reset-failed $unit 2>/dev/null; sudo systemctl start $unit" \
                    || remaining+=("$m")
            else
                log "  $m: no known systemd unit, cannot ssh-fallback"
                remaining+=("$m")
            fi
        fi
    done
    GPU1_RESTORE_LIST=("${remaining[@]}")
}

acquire_gpu_lock() {
    local owner now lockts age
    owner="step=$STEP host=$(hostname -s) pid=$$ ts=$(date +%s)"
    for _ in 1 2; do
        if ssh -o ConnectTimeout=10 -q "$RTX" "mkdir '$GPU_LOCK_DIR' 2>/dev/null"; then
            ssh -o ConnectTimeout=10 -q "$RTX" "printf '%s\n' '$owner' > '$GPU_LOCK_DIR/owner'" 2>/dev/null || true
            GPU_LOCK_HELD=1
            return 0
        fi
        # Lock is held -- check whether it's stale (holder crashed without the trap).
        lockts=$(ssh -o ConnectTimeout=10 -q "$RTX" "stat -c %Y '$GPU_LOCK_DIR' 2>/dev/null" || echo 0)
        now=$(ssh -o ConnectTimeout=10 -q "$RTX" "date +%s" 2>/dev/null || echo 0)
        age=$(( now - lockts ))
        if (( lockts > 0 && now > 0 && age > GPU_LOCK_TTL )); then
            log "GPU-1 lock is stale (age ${age}s > ${GPU_LOCK_TTL}s); taking it over"
            ssh -o ConnectTimeout=10 -q "$RTX" "rm -rf '$GPU_LOCK_DIR'" 2>/dev/null || true
            continue
        fi
        log "GPU-1 busy (held by: $(ssh -o ConnectTimeout=10 -q "$RTX" "cat '$GPU_LOCK_DIR/owner' 2>/dev/null" || echo unknown), age ${age}s)"
        return 1
    done
    return 1
}

release_gpu_lock() {
    [[ "$GPU_LOCK_HELD" -eq 1 ]] || return 0
    ssh -o ConnectTimeout=10 -q "$RTX" "rm -rf '$GPU_LOCK_DIR'" 2>/dev/null || true
    GPU_LOCK_HELD=0
}

cleanup() {
    # Stop the GPU watchdog first so it doesn't try to kill a non-existent eval
    # after we've already moved past the eval phase.
    if [[ -n "${WATCHDOG_PID:-}" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null
        wait "$WATCHDOG_PID" 2>/dev/null
    fi
    restore_gpu1_tenants
    release_gpu_lock
}
trap cleanup EXIT
trap 'exit 143' INT TERM

# ---- Mid-eval GPU-fault watchdog ----------------------------------------
# Polls nvidia-smi on rtx6000 every $GPU_WATCHDOG_INTERVAL seconds while an
# eval is running. Two consecutive failures (~120s total) -> mark the GPU
# wedged and signal the parent eval-script to abort cleanly. The EXIT trap
# then reloads vision and releases the lock as usual. Without this, a mid-
# eval Xid / GSP fault leaves the script grinding into the same broken state
# until the per-eval $REMOTE_EVAL_TIMEOUT (1200s) elapses. See 2026-05-20
# postmortem in docs/first_cluster_training_run.md for the failure mode.
GPU_WATCHDOG_INTERVAL=${GPU_WATCHDOG_INTERVAL:-60}
GPU_FAULT_FILE=/tmp/r215_gpu_fault_${STEP}_$$
WATCHDOG_PID=""

start_gpu_watchdog() {
    rm -f "$GPU_FAULT_FILE"
    (
        local fail_count=0
        while sleep "$GPU_WATCHDOG_INTERVAL"; do
            if ssh -o ConnectTimeout=5 -o BatchMode=yes -q "$RTX" \
                    "nvidia-smi -L 2>/dev/null | grep -qc \"^GPU \" && exit 0 || exit 1" 2>/dev/null; then
                fail_count=0
            else
                fail_count=$((fail_count + 1))
                if (( fail_count >= 2 )); then
                    echo "$(date '+%H:%M:%S')" > "$GPU_FAULT_FILE"
                    # Kill any in-flight remote-eval SSH from THIS script's process
                    # tree. The SSH wrapper's death cascades the remote `timeout`,
                    # which TERMs the python eval (which may not respond if it's
                    # blocked on a hung CUDA call; the kernel SIGKILLs it eventually).
                    pkill -P $$ -f "ssh.*training/.*\.py" 2>/dev/null
                    return
                fi
            fi
        done
    ) &
    WATCHDOG_PID=$!
}

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

# ---- 5. Acquire GPU-1 lock, THEN free GPU 1 of all tenants for the eval ----
# Acquire before touching any tenants: if another eval owns GPU 1 we must
# not yank tenants out from under it. exit 3 is a distinct "GPU busy, retry
# later" signal the watcher recognises (it clears SEEN so the step is
# retried next tick).
if ! acquire_gpu_lock; then
    log "ABORT step=$STEP: another eval owns GPU 1; will retry on a later tick"
    exit 3
fi
# Use the GPU-level admin endpoint instead of unloading a single named
# backend. GPU 1's tenant inventory has changed over time (vision only ->
# vision + embedding after 2026-05-22 relocate -> possibly + parallel coder
# during eval sweeps), and the eval-script should not need to know the
# current list. The router walks `backends` for cuda_device==1 LOADED and
# unloads them all; we cache the unloaded list in GPU1_RESTORE_LIST so the
# EXIT trap can restore exactly what we evicted (and nothing we did not).
# See kebab_rtx_router_fastapi.py admin_gpus_free for the endpoint impl.
free_gpu1 || log "WARN: GPU 1 may not be fully free; proceeding anyway"

# ---- 6. Run eval bundle on kebab-rtx6000 GPU 1 (per-eval pull-back) ----
# Each eval's JSON is pulled back to spark IMMEDIATELY after it completes +
# validates, so if a later (slower) eval times out and the watcher kills the
# whole cycle, the already-finished evals are safe on spark. (Previously the
# pull-back was a single end-of-pipeline step, so a timeout stranded all
# completed evals on rtx6000 -- caught 2026-05-20 with reasoning_eval.)
#
# Per-eval scope overrides: reasoning_eval at full scope (3 tasks x 5 depths
# x 500 examples, K=64 dominant) runs ~25 min and blew the cycle budget.
# For intermediate trend-tracking we trim it to LIMIT=200 + DEPTHS=4,8,16,32
# (~3x faster, K=64 dropped) -- still enough signal to watch the curve.
# depth_extrap KEEPS K=64 (extrapolation past trained depth is its whole point).
PYTHON=/home/alexm/venvs/vllm-turboquant/bin/python3
log "run eval bundle on rtx6000 GPU 1: $EVALS"
start_gpu_watchdog
log "GPU watchdog started (probe ${GPU_WATCHDOG_INTERVAL}s, abort on 2 consecutive failures)"
eval_failures=0
for evalname in $EVALS; do
    out="$REPO_RTX/docs/intermediate_r215_step${STEP}_${evalname}.json"
    if [[ "$FORCE_FLAG" != "--force" ]] \
        && json_valid_remote alexm@kebab-rtx6000.lan "$out"; then
        log "  $evalname: valid output exists, skipping"
        # Still ensure spark has it (an earlier cycle may have produced it
        # on rtx6000 but timed out before pulling it back).
        ssh -q alexm@kebab-spark.lan \
            "rsync -az alexm@kebab-rtx6000.lan:$out $REPO_SPARK/docs/ 2>/dev/null" || true
        continue
    fi
    # Per-eval env overrides (trim the slow reasoning_eval; leave others default)
    extra_env=""
    if [[ "$evalname" == "reasoning_eval" ]]; then
        extra_env="LIMIT=${REASONING_LIMIT:-200} DEPTHS=${REASONING_DEPTHS:-4,8,16,32}"
    fi
    log "  $evalname running... ${extra_env:+($extra_env)}"
    # Remote-side `timeout` bounds each eval AND -- critically -- makes the
    # remote python self-terminate if the local watcher kills our ssh client
    # (e.g. the cycle-level timeout fires). Without it, SSH-launched evals
    # orphan onto GPU 1 and contend with the next cycle's eval (seen live
    # 2026-05-20: a step-10600 reasoning_eval survived a watcher restart and
    # fought the step-11000 eval for the GPU). --signal=TERM + a short
    # --kill-after gives the script a chance to flush before SIGKILL.
    if ! ssh -q alexm@kebab-rtx6000.lan "
        cd $REPO_RTX && \
        CUDA_VISIBLE_DEVICES=1 \
        CKPT=$REPO_RTX/$FULL_PATH \
        CKPT_DIR=$CKPT_DIR \
        OUT=$out \
        $extra_env \
        timeout --signal=TERM --kill-after=30 ${REMOTE_EVAL_TIMEOUT:-1200} $PYTHON training/${evalname}.py
    "; then
        log "  $evalname FAILED (script exited non-zero or hit remote ${REMOTE_EVAL_TIMEOUT:-1200}s timeout)"
        eval_failures=$((eval_failures + 1))
        continue
    fi
    if ! json_valid_remote alexm@kebab-rtx6000.lan "$out"; then
        log "  $evalname FAILED (exited 0 but output missing/invalid JSON)"
        ssh -q alexm@kebab-rtx6000.lan "rm -f $out" 2>/dev/null
        eval_failures=$((eval_failures + 1))
        continue
    fi
    # PER-EVAL pull-back: get this JSON onto spark right now.
    ssh -q alexm@kebab-spark.lan \
        "rsync -az alexm@kebab-rtx6000.lan:$out $REPO_SPARK/docs/ 2>/dev/null" || true
    log "  $evalname done -> pulled to spark"

    # If the GPU watchdog flagged a mid-eval fault, abort the loop cleanly.
    # The EXIT trap reloads vision + releases lock; the watcher recognises
    # rc=4 as "GPU fault detected" and skips re-dispatch until pre-flight
    # confirms the node is healthy again.
    if [[ -f "$GPU_FAULT_FILE" ]]; then
        log "ABORT: GPU watchdog signalled fault at $(cat "$GPU_FAULT_FILE"); skipping remaining evals"
        rm -f "$GPU_FAULT_FILE"
        exit 4
    fi
done

# ---- 7. Restore GPU 1 tenants + release the GPU-1 lock ----
# Done promptly on the happy path here; the EXIT trap repeats both
# idempotently so a kill anywhere above still restores the GPU and frees
# the lock. restore_gpu1_tenants consumes GPU1_RESTORE_LIST (populated by
# free_gpu1 in step 5) and reloads exactly what we evicted -- in priority
# order (vision first, then embedding, then anything else) so a partial
# recovery still gets the highest-impact backends back.
restore_gpu1_tenants
release_gpu_lock

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
