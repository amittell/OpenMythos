# Qwen3.6 LCB bench cascade -- relaunch plan (2026-05-31)

Operator runbook for relaunching the LiveCodeBench cascade across the
Qwen3.6 family + Qwen3-Coder-Next-FP8 on `kebab-rtx6000.lan` after two
consecutive failed attempts.

## Background

We run three legs sequentially through `lcb_runner`, with vLLM serving the
model on GPU 1 (`device=1`, the second RTX 6000 Pro Blackwell) on TCP
port 8002:

| Leg | Model path                          | Served name                | Save name                          |
| --- | ------------------------------------ | -------------------------- | ---------------------------------- |
| 1   | `/models/Qwen3.6-27B`                | `qwen3.6-27b`              | `Qwen3.6-27B-bench-run2`           |
| 2   | `/models/Qwen3.6-35B-A3B`            | `qwen3.6-35b-a3b`          | `Qwen3.6-35B-A3B-bench-run2`       |
| 3   | `/models/Qwen3-Coder-Next-FP8`       | `qwen3-coder-next-bench`   | `Qwen3-Coder-Next-FP8-bench-run2`  |

The driver script is `/tmp/qwen36_bench_cascade.sh` on
`kebab-rtx6000.lan`. The standing baseline to compare against is
`Qwen3-Coder-Next-FP8` at pass@1 = 0.7645 from the earlier May run.

## Failure post-mortem

### Run #1 (May 30, driver 595.71.05, `--multiprocess 8`)

Qwen3.6-27B produced 1 of 329 completions. The Coder-Next-FP8 leg then
triggered the GSP wedge documented in
`memory/rtx6000_gsp_wedge.md`: Blackwell PRO 6000 hard-locks under
sustained FP8 MoE inference and requires a power cycle.

A later post-mortem (Claude, May 31) showed the 328/329 empty-completion
outcome was **not** a chat-template mismatch as originally suspected
(task #109's original framing). The actual root cause was **concurrency
overload**: `--multiprocess 8` saturating a 27B vLLM whose KV cache was
already pressured by `--max-model-len 32768`. With 8 concurrent
generation requests in flight, vLLM silently returned empty completions
for the overflow rather than queueing them. The chat template was fine.

Mitigation captured in run #2: `--multiprocess 2` plus
`--max-num-seqs 4` on the vLLM side.

### Run #2 (May 31, driver 610.43.02, post-upgrade)

All three legs crashed instantly:

```
openai.OpenAIError: Missing credentials. Please pass an `api_key`,
`workload_identity`, `admin_api_key`, or set the `OPENAI_API_KEY` or
`OPENAI_ADMIN_KEY` environment variable.
```

`lcb_runner` uses the OpenAI Python SDK to talk to vLLM's
OpenAI-compatible endpoint. Recent versions of the SDK now require
`OPENAI_API_KEY` to be set even when `OPENAI_BASE_URL` points at a
purely local endpoint that does not authenticate. This was a silent
behavior change in the SDK; nothing on the vLLM side regressed.

## Fix applied 2026-05-31

The `run_lcb()` function in `/tmp/qwen36_bench_cascade.sh` on
`kebab-rtx6000.lan` was patched to export the two env vars before
invoking `lcb_runner`:

```bash
run_lcb(){
  local model_id=$1
  local save_name=$2
  log "=== START $model_id (save_name=$save_name) ==="
  cd $LCB_DIR
  # OpenAI SDK (used by lcb_runner) now requires OPENAI_API_KEY even when
  # OPENAI_BASE_URL points at a local OpenAI-compatible endpoint (vLLM).
  # Without these, all legs crash immediately with:
  #   openai.OpenAIError: Missing credentials.
  export OPENAI_API_KEY=dummy
  export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
  $LCB_PY -m lcb_runner.runner.main \
    ...
}
```

`PORT` is already set to `8002` at the top of the script, matching the
`docker run -p ${PORT}:8000` mapping for the vLLM container. The vLLM
endpoint does not validate the bearer token, so `dummy` is sufficient.

The pre-patch script is preserved on rtx6000 as
`/tmp/qwen36_bench_cascade.sh.bak.20260531_215831`.

## Relaunch invocation (when GPUs are free)

The rtx6000 GPUs are currently busy with the r217 eval queue (runs 145,
146, ...). Do **not** launch the cascade until that queue drains and
GPU 1 reports free in `nvidia-smi`.

When ready, launch from any host:

```bash
ssh alexm@kebab-rtx6000.lan "nohup /tmp/qwen36_bench_cascade.sh > /tmp/qwen36_bench_cascade.log 2>&1 & disown"
```

Expected runtime: 6-10 hours total (three legs at roughly 2-3 hours
each, at `--multiprocess 2`). The Coder-Next-FP8 leg is the longest
because of FP8 MoE startup overhead.

## Monitoring

Tail the run log on rtx6000:

```bash
ssh alexm@kebab-rtx6000.lan "tail -f /tmp/qwen36_bench.log"
```

The script logs `=== START ... ===` and `=== DONE ... rc=N ===` around
each leg. A leg exiting with `rc=0` and no completions still indicates
a silent vLLM problem; verify the corresponding output directory was
populated.

Concurrently check the vLLM container health:

```bash
ssh alexm@kebab-rtx6000.lan "docker logs --tail 50 kebab-vllm-bench"
```

GSP wedge warning: if `nvidia-smi` on rtx6000 starts reporting
"Unknown Error" against GPU 1 during the Coder-Next-FP8 leg startup,
that is the Blackwell GSP full-chip-reset failure mode (see
`memory/rtx6000_gsp_wedge.md`). Recovery requires power-cycling the
host; the cascade cannot recover in-place.

## Output locations

Per-leg results land under
`/home/alexm/git/benchmarks/LiveCodeBench/output/` on rtx6000:

- `Qwen3.6-27B-bench-run2/`
- `Qwen3.6-35B-A3B-bench-run2/`
- `Qwen3-Coder-Next-FP8-bench-run2/`

Each directory contains the per-problem completions (JSON) plus a
`_eval.json` summary with pass@1. Compare leg 3's pass@1 against the
0.7645 baseline; legs 1 and 2 are first-pass numbers for the Qwen3.6
family on LCB and have no prior baseline.

## Post-cascade GPU handoff

The cascade ends by handing GPU 1 back to gpufarm so the queued
`toy_depth_task` runs (#121, #122, #123) can dispatch:

```bash
ssh -4 alexm@kebab-spark.lan \
  "GPUFARM_STATE_DB=/home/alexm/.local/state/gpufarm/state.sqlite \
   /home/alexm/.local/bin/gpufarm resume blackwell-gpu1"
```

This step is already wired into the script (see the trailing block) and
runs even if individual legs return non-zero exit codes.

## Related task

Task #109 ("Patch LCB chat template for Qwen3.6 family then rerun bench
cascade") tracks the relaunch. The task description has been updated to
reflect the corrected root-cause analysis (concurrency overload in run
#1, OpenAI credentials in run #2) and now points at this document.
