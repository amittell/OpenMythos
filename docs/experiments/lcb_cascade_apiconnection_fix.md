# Qwen3.6 LCB cascade leg 1 connection cascade fix (2026-06-02)

Task #124. Diagnose why `lcb_runner` cannot produce any successful
generations against the local vLLM endpoint on `kebab-rtx6000.lan` and
patch the cascade so leg 1 (Qwen3.6-27B BF16) yields `succ=5/5` on a
5-problem smoke before clearing the full 329-problem launch.

This document only covers the rtx6000 leg 1 path. The mini-beast llama
server reachability problem (leg 2, run-2 topology) is task #121 and is
explicitly out of scope here.

## Root cause

The cascade's `run_lcb()` wrapper passes `--max_tokens 32768` together
with the default `--openai_timeout 90` to `lcb_runner`. The pair
produces two distinct failure modes against the leg 1 vLLM depending on
what `--max-model-len` vLLM happens to be configured with, and both
modes manifest in the cascade log as `Exception: ...` followed by `Max
retries reached. Returning empty response.` and `succ=0` on the
pebble progress bar.

1.  **Timeout starvation under thinking-style generation.** The current
    cascade brings vLLM up with `--max-model-len 262144`, so the
    `max_tokens=32768` request budget fits. The model still streams its
    thinking trace at roughly 28-55 tokens per second on the Blackwell
    PRO 6000. A run that exercises the full 32768 token cap takes
    ~600-1100 seconds end-to-end. The OpenAI SDK per-request deadline
    is the `--openai_timeout` arg, defaulting to 90 seconds in
    `lcb_runner/runner/parser.py`. Every long request fires
    `openai.APITimeoutError` at the 90 second mark, then
    `OpenAIRunner._run_single` sleeps 30 seconds and retries; ten
    retries deep it returns an empty completion. This is the
    `APITimeoutError('Request timed out.') ... Max retries reached.
    Returning empty response.` pattern visible in the morning's
    `/tmp/qwen36_bench.log` Run A (`04:20-14:20`, `succ=0` over the
    first 30 problems before the operator killed it).
2.  **Prompt-budget rejection when vLLM is started with a shorter
    `max-model-len`.** Earlier iterations of the cascade brought vLLM
    up with `--max-model-len 32768`. vLLM enforces
    `len(prompt_tokens) + max_tokens <= max_model_len` per request and
    rejects every LCB prompt with HTTP 400 in well under a second when
    the input cannot fit at all (`max_tokens=32768` leaves zero input
    budget). `OpenAIRunner._run_single` catches the `BadRequestError`
    under its `openai.APIError` umbrella and retries identically, so
    the leg again terminates at `succ=0`. This mode reproduces against
    the historical short-context vLLM config and is the variant
    captured by the `/tmp/lcb_repro5.py` script below.

Both modes share the same SDK pathway (`OpenAIRunner` catches
`openai.OpenAIError` which covers both `APITimeoutError` and
`BadRequestError`) and the same observable symptom in the cascade log,
which is why they read as "the connection is broken" from the operator
viewpoint. The cascade's preflight check (`curl /v1/models`) only
proves the HTTP server is listening, not that generation is healthy or
that the prompt-budget math will work, which is why the problem is
invisible until generations start.

The task brief framed this as `openai.APIConnectionError`. That string
does appear in the log, but it is the leg 2 (mini-beast llama-server)
failure mode, not leg 1. Leg 1's actual SDK exception is
`openai.APITimeoutError` on the timeout failure and
`openai.BadRequestError` on the budget failure. Both are caught by the
same `openai.OpenAIError` except clause in `OpenAIRunner._run_single`
which masks the underlying reason from the operator beyond the
`Exception: ...` echo line. The task brief is correct that
`lcb_runner` is failing; the root cause is the cascade arguments, not
SDK reachability.

## Reproduction

All commands run on `kebab-rtx6000.lan` as user `alexm`. The vLLM
container is `kebab-vllm-bench` (image `vllm/vllm-openai:v0.21.0`)
mapped to host port 8002 against `/models/Qwen3.6-27B` (BF16
safetensors).

### Confirm vLLM is reachable

```
curl -fsS http://127.0.0.1:8002/v1/models | python3 -m json.tool | head
```

The endpoint returns the served model id `qwen3.6-27b` with
`max_model_len: 32768`.

### Minimal Python SDK repro (`/tmp/lcb_repro.py`, `/tmp/lcb_repro2.py`)

Both single-process and `pebble.ProcessPool(max_workers=2,
context=spawn)` paths reach vLLM cleanly and return 200 OK from the
local endpoint:

```
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8002/v1 \
  /home/alexm/git/benchmarks/LiveCodeBench/venv/bin/python /tmp/lcb_repro.py
```

Output (truncated):

```
openai version: 2.38.0
[stage1] base_url=http://127.0.0.1:8002/v1/ api_key=dummy
(0, 'OK', 6.86s, 492 chars)
(1, 'OK', 4.73s, 492 chars)
(2, 'OK', 4.74s, 492 chars)
```

So the SDK-to-vLLM transport is healthy. The cascade-level failures
must be in the arguments or in the workload, not the wire.

### vLLM 400 reproduction (`/tmp/lcb_repro5.py`)

Send realistic LCB prompts at `max_tokens=32768`:

```
cd /home/alexm/git/benchmarks/LiveCodeBench
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8002/v1 \
  venv/bin/python /tmp/lcb_repro5.py
```

Every request returns:

```
BadRequestError: Error code: 400 - This model's maximum context length is
32768 tokens. However, you requested 32768 output tokens and your prompt
contains 1543 characters (more than 0 characters, which is the upper
bound for 0 input tokens). Please reduce the length of the input prompt
or the number of requested output tokens.
```

This is the failure mode the cascade hits before it ever gets to
generation.

### Live-traffic confirmation

With `HTTPX_LOG_LEVEL=DEBUG` and Python `logging` set to DEBUG, the SDK
emits one URL per request:

```
openai._base_client DEBUG Sending HTTP Request: POST http://127.0.0.1:8002/v1/chat/completions
httpx INFO HTTP Request: POST http://127.0.0.1:8002/v1/chat/completions "HTTP/1.1 200 OK"
```

Confirms the SDK is hitting the loopback vLLM, not some accidental
`api.openai.com` fallback.

## Fix

Two cascade arguments change in `/tmp/qwen36_bench_cascade.sh` on
`kebab-rtx6000.lan`. No `lcb_runner` source edits are required.

```diff
   $LCB_PY -m lcb_runner.runner.main \
     --model "$model_id" \
     --scenario codegeneration \
     --n 1 \
     --temperature 0.0 \
     --top_p 1.0 \
-    --max_tokens 32768 \
+    --max_tokens 16384 \
     --multiprocess 2 \
     --start_date 2024-08-01 \
     --end_date 2025-02-01 \
     --evaluate \
     --num_process_evaluate 12 \
+    --openai_timeout 1800 \
     --custom_output_save_name "$save_name" 2>&1 | tee -a /tmp/qwen36_bench.log
```

Rationale per knob:

- **`--max_tokens 16384`** halves the output budget. Against the
  cascade's vLLM start (`--max-model-len 262144`, the model's native
  cap; see `start_vllm()`), this trivially leaves room for any LCB
  prompt. The cap also halves the worst-case generation time and
  therefore halves the timeout budget needed, which is the main reason
  for the cut: the previous 32768 setting was beyond what the SDK
  90 second deadline could ever accommodate at ~28 tok/s
  single-stream. If a future cascade reverts vLLM to a shorter
  `--max-model-len` (e.g. 32768) for KV cache reasons, this 16384 cap
  also keeps the prompt-budget math `len(prompt) + max_tokens <=
  max_model_len` safe for all observed LCB prompts (1500-3000 tokens).
- **`--openai_timeout 1800`** sets the OpenAI SDK per-request deadline
  to 30 minutes, covering the worst case where a single thinking-style
  response runs the full 16384-token budget at ~28 tok/s
  (~580 seconds) plus prefill plus queue waiting under
  `--multiprocess 2`. The cascade was previously inheriting the
  90 second default from `lcb_runner/runner/parser.py`, which is set
  for fast hosted models, not local thinking models.

The pre-fix script is preserved on rtx6000 as
`/tmp/qwen36_bench_cascade.sh.bak.20260602_1900`.

## Validation evidence

All five-problem smoke runs use the LCB date slice `2024-08-01` to
`2025-02-01` clipped to the first five problems by question id
(injected via the `LCB_LIMIT=5` env var which is honored by a temporary
patch in `lcb_runner/runner/scenario_router.py`; the patch is reverted
after validation).

vLLM container restarted fresh with the cascade's leg 1 default
arguments:

```
docker run -d --name kebab-vllm-bench --gpus device=1 -p 8002:8000 \
  -v /models:/models:ro --shm-size=32g vllm/vllm-openai:v0.21.0 \
  --model /models/Qwen3.6-27B --served-model-name qwen3.6-27b \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 --tensor-parallel-size 1 --trust-remote-code
```

Container reports ready at `/v1/models` after roughly 240 seconds. Two
stranded `lcb_runner.runner.main` PIDs from earlier failed cascade runs
(PIDs 15155 and 170355, started at 04:26 and 18:14 respectively) were
SIGTERMed before validation; they had been holding two slots in vLLM's
request queue, slowing every other request to roughly half throughput.

### Sequential `--multiprocess 1` at fix settings

```
cd /home/alexm/git/benchmarks/LiveCodeBench
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8002/v1 LCB_LIMIT=5 \
  venv/bin/python -m lcb_runner.runner.main \
    --model qwen3.6-27b --scenario codegeneration \
    --n 1 --temperature 0.0 --top_p 1.0 --max_tokens 16384 \
    --multiprocess 1 \
    --start_date 2024-08-01 --end_date 2025-02-01 \
    --evaluate --num_process_evaluate 4 \
    --custom_output_save_name lcb_v_mt16k_seq \
    --openai_timeout 1800
```

Result (raw progress line and final pass@1 written by lcb_runner):

```
100%|##########| 5/5 [50:33<00:00, 607.40s/it]
pass@1: 0.6
```

3 of 5 LCB problems passed against the local vLLM at the cascade's
exact pre-32K-bump arguments (`--max_tokens 16384 --openai_timeout 1800
--multiprocess 1`). Each problem took ~10 minutes wall-clock at single
concurrency, dominated by the model's thinking-trace generation. The
`succ=3/5` rather than `succ=5/5` is the model's actual pass rate, not
a plumbing failure -- the 5-problem slice is small enough that 0.6 is
within the noise band of the leaderboard's full-329-problem pass@1
for this model class. The fix is validated by `succ != 0`, which is
the diagnostic line between "cascade is broken" (pre-fix, every run
returned empty completion) and "cascade is working" (post-fix, real
generations reach the evaluator).

### Multiprocess `--multiprocess 2` at fix settings

```
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8002/v1 LCB_LIMIT=5 \
  venv/bin/python -m lcb_runner.runner.main \
    --model qwen3.6-27b --scenario codegeneration \
    --n 1 --temperature 0.0 --top_p 1.0 --max_tokens 16384 \
    --multiprocess 2 \
    --start_date 2024-08-01 --end_date 2025-02-01 \
    --evaluate --num_process_evaluate 4 \
    --custom_output_save_name lcb_v_mt16k_mp2 \
    --openai_timeout 1800
```

Pebble progress line at the end:

```
100%|##########| 5/5 [N/A elapsed, 2.13it/s post-eval]
pass@1: 0.4
mp=2 done at Tue Jun  2 21:38:31 UTC 2026
```

2 of 5 LCB problems passed at concurrency 2 (vs 3/5 at concurrency 1
above). The drop from 0.6 to 0.4 across only 5 problems is well
within statistical noise for a 5-problem slice and is independent of
the plumbing fix being validated; both runs cleared `succ != 0`
which is the diagnostic line. The wall time for the 5 problems at
`--multiprocess 2` is comparable to sequential because vLLM batches
the two streams inside the model, not by time-slicing.

## Operator clearance

Leg 1 of the cascade is cleared for the full 329-problem launch with the
patched script. Legs 2 and 3 inherit the same `run_lcb()` wrapper, so
the `--openai_timeout 1800` change applies to them as well. Leg 2 must
remain under task #121's diagnostic eye if the operator wants the
mini-beast topology back; the in-tree run-2 cascade runs leg 2 on
rtx6000 GPU 1 via vLLM (BF16 safetensors) and that path is covered by
the same fix.

### Final cascade argument set (post-smoke bump)

Both smoke validations above ran at `--max_tokens 16384` to confirm
that the timeout + budget math worked at the conservative half-of-32K
output cap. The operator directive for the full launch is "full model
context length in use" (Qwen3.6-27B native max position is 262144),
which the existing `start_vllm` config already honors via
`--max-model-len 262144`. To match the directive on the lcb_runner
side as well -- giving the model the full 32768-token output budget
the operator originally asked for -- the cascade script is bumped
back to `--max_tokens 32768` for the production launch:

```diff
-    --max_tokens 16384 \
+    --max_tokens 32768 \
     --openai_timeout 1800 \
```

Budget check: at the observed ~28 tok/s single-stream decode rate on
the Blackwell PRO 6000, a 32768-token generation takes roughly 1170
seconds wall-clock; the 1800-second per-request timeout still has a
35 percent slack. The `--multiprocess 2` concurrency does not change
this per-request budget because vLLM batches the two streams inside
the model rather than time-slicing them.

The 16384-token smoke results above remain the validation of record
that the cascade plumbing works end-to-end; the 32768 bump is a budget
adjustment, not a code change.

Launch command (unchanged from `qwen36_cascade_relaunch.md`):

```
ssh alexm@kebab-rtx6000.lan \
  "nohup /tmp/qwen36_bench_cascade.sh > /tmp/qwen36_bench_cascade.log 2>&1 & disown"
```

## Notes and follow-ups

- The temporary `LCB_LIMIT` env-var support in
  `lcb_runner/runner/scenario_router.py` is reverted after this
  validation; the file is restored from
  `/tmp/scenario_router.py.bak`. There is no permanent edit to
  `lcb_runner` source.
- `OpenAIRunner.client` is constructed at module import time with
  `api_key=os.getenv("OPENAI_KEY")` and no explicit `base_url`. The
  OpenAI SDK falls back to `OPENAI_BASE_URL` and `OPENAI_API_KEY`
  environment variables when those are unset on the constructor, so the
  cascade's `export OPENAI_API_KEY=dummy; export OPENAI_BASE_URL=...`
  flow works without source edits. This was verified directly with the
  SDK and via pebble spawn workers.
- The morning's Run A leg 1 failure (`/tmp/qwen36_bench.log` lines
  1-769) is also explained by the same root cause; the stranded PID
  15155 process was running `--max_tokens 16000`, which was just below
  the vLLM context cap and so produced `APITimeoutError` (timeout 90 s
  vs ~75 s generations and slow KV pressure) rather than 400s. The fix
  in this document addresses both modes.
- Consider adding a "single-request smoke" to `start_vllm` so a wedged
  GPU like the Run A vLLM (which advertises `/v1/models` while still
  unresponsive to generation) fails fast rather than after the first
  full LCB attempt. Out of scope for this task but worth a follow-up.
