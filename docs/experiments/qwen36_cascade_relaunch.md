# Qwen3.6 LCB bench cascade -- relaunch plan (2026-05-31)

Operator runbook for relaunching the LiveCodeBench cascade across the
Qwen3.6 family + Qwen3-Coder-Next-FP8 on `kebab-rtx6000.lan` after two
consecutive failed attempts.

## Background

We run three legs sequentially through `lcb_runner`. Legs 1 and 3 are
served by a vLLM instance on `kebab-rtx6000.lan` GPU 1
(`device=1`, the second RTX 6000 Pro Blackwell) on TCP port 8002.
Leg 2 was split onto `mini-beast.lan`'s AMD Radeon 8060S (Strix Halo)
on port 8003 -- see "Cascade leg 2 split to mini-beast Radeon 8060S"
below for the rationale and details.

| Leg | Model path                          | Served name                | Save name                          | Backend / endpoint                                   |
| --- | ------------------------------------ | -------------------------- | ---------------------------------- | ---------------------------------------------------- |
| 1   | `/models/Qwen3.6-27B`                | `qwen3.6-27b`              | `Qwen3.6-27B-bench-run2`           | vLLM on `kebab-rtx6000.lan:8002`                     |
| 2   | `/models/Qwen3.6-35B-A3B` (-> GGUF)  | `qwen3.6-35b-a3b`          | `Qwen3.6-35B-A3B-bench-run2`       | llama.cpp Vulkan on `mini-beast.lan:8003`            |
| 3   | `/models/Qwen3-Coder-Next-FP8`       | `qwen3-coder-next-bench`   | `Qwen3-Coder-Next-FP8-bench-run2`  | vLLM on `kebab-rtx6000.lan:8002`                     |

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

## Cascade leg 2 split to mini-beast Radeon 8060S (2026-05-31)

Leg 2 (Qwen3.6-35B-A3B) has been moved off `kebab-rtx6000.lan` and onto
`mini-beast.lan`'s AMD Radeon 8060S (gfx1151 Strix Halo APU, 128 GB
unified memory). The other two legs continue to run on rtx6000 GPU 1
via vLLM as before.

### Rationale

- The 35B-A3B MoE is the largest of the three legs (67 GB safetensors
  on rtx6000, 29.3 GB as UD-Q6_K GGUF). Even with vLLM-20b parked on
  GPU 0 of rtx6000, stacking a 35B MoE alongside the future Coder-Next
  leg on GPU 1 was the leg that wedged the Blackwell on run #1.
- Strix Halo has a 128 GB unified pool. A 29 GB Q6_K weights set plus
  KV cache for 32 K context sits comfortably inside that pool, and the
  APU sits idle while rtx6000 is otherwise busy.
- Running this leg on AMD also de-correlates the cascade from rtx6000
  GSP wedge risk on the largest single model.

### Model artifact

A pre-built GGUF exists upstream and was downloaded directly to
mini-beast rather than converted in-house:

- Source: `unsloth/Qwen3.6-35B-A3B-GGUF`, file
  `Qwen3.6-35B-A3B-UD-Q6_K.gguf` (29.3 GB).
- Destination: `/home/alex/models/Qwen3.6-35B-A3B-GGUF/` on
  mini-beast (mapped to `/models/Qwen3.6-35B-A3B-GGUF/` inside the
  llama.cpp container per the `qwen3-coder.service` volume convention).
- Quant choice rationale: Q6_K over Q4_K_M (29.3 vs 22.1 GB). Strix
  Halo's 128 GB pool makes the 7 GB delta cheap, and Q6_K is the
  closest llama.cpp quant to Qwen's intended deployment precision.

No bespoke conversion was needed; `rtx6000:/home/alexm/llama-cpp-build/`
remains the fallback path if a future Qwen3.6 release lands without a
prebuilt GGUF.

### Systemd unit

A user-scope unit at
`mini-beast:/home/alex/.config/systemd/user/qwen36-35b-a3b.service`
runs `llama-server` inside the `kyuz0/amd-strix-halo-toolboxes:vulkan-radv`
container, modeled on the existing `qwen3-coder.service`:

```
ExecStart=/usr/bin/podman run --name qwen36-35b-a3b-vulkan \
  --device /dev/dri --security-opt seccomp=unconfined \
  -p 8003:8003 -v /home/alex/models:/models:z \
  docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server \
    -m /models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
    -ngl 999 -c 32768 -b 1024 -ub 512 \
    --jinja --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --cont-batching --metrics \
    --host 0.0.0.0 --port 8003
```

Port `8003` is free on mini-beast (router lives at 8080, qwen-vision at
8081, qwen3-coder at 8093). The container ships llama.cpp build 6946
(commit `afd353246`), which has Qwen3.6 support landed upstream.

The unit is `enabled` but not started; the cascade script starts it
just-in-time for leg 2 and stops it on completion so it does not hold
APU memory between runs.

### Router registration

`mini-beast:/home/alex/mini-beast-repo/router.py` gained four entries
for the new model id `qwen36-35b-a3b`, mirroring how `qwen3-coder` is
registered:

- `BACKENDS["qwen36-35b-a3b"] = "http://mini-beast.lan:8003"`
- `BACKEND_HEALTH_PATHS["qwen36-35b-a3b"] = "/health"`
- `BACKEND_KINDS["qwen36-35b-a3b"] = "llama-server"`
- `SERVICE_UNITS["qwen36-35b-a3b"] = {unit, display_name, cuda_device=None}`

This makes the model visible in `/admin/models/status` and lets the
existing idle-unload machinery manage the unit. The router was **not**
bounced to pick up the change (it serves the live UI); the change takes
effect at the next routine router restart.

### Cascade script changes

`/tmp/qwen36_bench_cascade.sh` on rtx6000 now contains:

1. New constants `MINIBEAST_HOST`, `MINIBEAST_USER`, `MINIBEAST_PORT`,
   `MINIBEAST_UNIT`.
2. New helpers `start_minibeast_llama()` (SSH `systemctl --user start`
   plus a 10-minute readiness poll on `:8003/v1/models`) and
   `stop_minibeast_llama()`.
3. `run_lcb()` gained an optional third arg `base_url`, defaulting to
   the local vLLM endpoint as before.
4. Leg 2 now (a) tears down the local vLLM container so rtx6000 GPU 1
   is idle while mini-beast runs, (b) calls `start_minibeast_llama`,
   (c) runs `lcb_runner` with `OPENAI_BASE_URL=http://mini-beast.lan:8003/v1`,
   (d) calls `stop_minibeast_llama` on the way out.
5. The exit `trap cleanup` also calls `stop_minibeast_llama` so an
   interrupted run does not leave the APU loaded.

Legs 1 and 3 are unchanged; they still spin up the local vLLM container
on rtx6000 GPU 1.

### Expected throughput

Strix Halo prompt processing on Vulkan typically lands between 0.5x and
1.5x of a dGPU at comparable memory bandwidth on dense decoders.
Qwen3.6-35B-A3B is a sparse MoE with ~3 B active params per token, so
the effective compute is closer to a 3 B dense model and the bottleneck
is unified-memory bandwidth (~200-250 GB/s on Strix Halo) rather than
APU compute. Rough ballpark for decode-only throughput at batch 1 at
this quant: 12-25 tok/s. Prompt processing at 32 K context is the
slower side of the budget; expect leg 2 to be longer than legs 1 and 3
by a factor of ~2-3x even though the model is smaller in active params.

Wall-clock estimate for leg 2: 4-6 hours at `--multiprocess 2` against
the LCB 2024-08 to 2025-02 problem slice. This pushes the cascade total
from "6-10 hours" (above) to "8-14 hours" when leg 2 runs on mini-beast.

### Backend-mixed caveat

Legs 1 and 3 run on vLLM with the HF tokenizer and the safetensors
weights directly. Leg 2 runs on llama.cpp with the GGUF tokenizer (which
re-implements the BPE merges from the HF tokenizer.json into its own
representation) and a Q6_K quant of the same weights. Two non-trivial
deltas vs an apples-to-apples vLLM-only cascade:

- Tokenizer: small differences in edge-case Unicode normalization and
  added-token handling between llama.cpp's GGUF tokenizer and HF
  `transformers` have shown up historically as 1-3 token differences on
  long prompts. For LCB code-generation tasks this is usually
  immaterial but it is not zero.
- Sampler: `lcb_runner` sets `temperature=0.0` per request in the
  OpenAI-compatible payload (the llama-server systemd unit does not set
  a CLI temperature flag, and for the OpenAI endpoint sampling is
  controlled per-request anyway). Both backends therefore run greedy
  argmax. Tie-breaking on equal logits may differ between llama.cpp and
  vLLM, which can produce small token-level deltas on prompts that
  happen to hit ties.

When comparing leg 2's pass@1 against future vLLM-on-Blackwell runs of
the same model, treat the result as a backend-mixed baseline and footnote
accordingly.

### Manual smoke test (when ready, before launching the full cascade)

To validate the service end-to-end without burning the full LCB run. The
smoke test goes **directly** to `llama-server` on port 8003 rather than
through `mini-beast.lan:8080` (the model-router); llama-server does not
validate the `model` field and we want to exercise the exact path the
cascade uses (`lcb_runner` -> direct llama-server). The model id in the
payload below matches `qwen3.6-35b-a3b` because that's the id LCB
`lm_styles.py` registers and what `lcb_runner --model ...` will send
during the actual cascade run. **Note**: the model-router on mini-beast
registers this backend under a different id (`qwen36-35b-a3b`, no period)
to mirror how `qwen3-coder` is registered there; the cascade bypasses
the router so the id mismatch is intentional and does not cause a
mis-route.

```
ssh alex@mini-beast.lan "systemctl --user start qwen36-35b-a3b.service"
# wait ~30-60s for the model to memory-map and warmup
curl -fsS http://mini-beast.lan:8003/v1/models | jq .
curl -fsS http://mini-beast.lan:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
ssh alex@mini-beast.lan "systemctl --user stop qwen36-35b-a3b.service"
```

### Launch command (unchanged form)

```
ssh alexm@kebab-rtx6000.lan "nohup /tmp/qwen36_bench_cascade.sh > /tmp/qwen36_bench_cascade.log 2>&1 & disown"
```

The cascade script now drives mini-beast on its own; no separate
operator action is required for leg 2 once the rtx6000 GPUs drain.
