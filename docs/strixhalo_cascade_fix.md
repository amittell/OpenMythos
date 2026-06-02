# Strix Halo qwen36-35b-a3b cascade leg 2 diagnosis (task #121)

Status: partial fix landed in-place on mini-beast; full inference speed
not viable on the current Vulkan path. Cascade leg 2 rerouted to rtx6000
vLLM safetensors (task #120).

## Symptom

LCB cascade leg 2 against `http://mini-beast.lan:8003/v1/chat/completions`
hit 44 client-side exceptions over 44 attempts, with `succ=0` and
`timeouts=0`. The "timeouts=0" was misleading because the failures
surfaced to `lcb_runner` as `APIConnectionError` (httpx stream RST), not
as `APITimeoutError`.

## Root cause of the original 44/44 pattern

Default `n_parallel=4` in `llama-server` plus four concurrent LCB
requests from `lcb_runner --multiprocess 2` (plus retries) split the
iGPU between the four slots. Per-slot generation rate dropped to roughly
5 t/s, every client request hit the 30s timeout, and `httpx` reset the
TCP connection. From the client side this looks like the server is
refusing connections; from the server side every request gets a
"cancelled after 30s" log line.

Companion failure: `--cache-type-k q8_0 --cache-type-v q8_0` combined
with `n_parallel=4` and a 32k context caused a `radv/amdgpu: Failed to
allocate a buffer` warning at the synthetic warmup step, which Agent B
mistook for a load-time crash and disabled via `--no-warmup`.

## What Agent B landed (mini-beast unit file)

`/home/alex/.config/systemd/user/qwen36-35b-a3b.service` was rewritten to:
- `-np 1`: single parallel slot, serialize incoming requests.
- Drop `--cache-type-k q8_0 --cache-type-v q8_0`: fp16 KV under np=1
  fits the GTT pool with no buffer-alloc warning.
- `-b 1024 -ub 512`: prompt-eval batch sizes tuned for the 8060S Vulkan
  path.
- `--no-warmup`: skip the synthetic warmup whose GTT spike was the
  source of the earlier buffer-alloc warning.
- `--jinja`: honour the bundled Qwen3.6 chat template
  (im_start/im_end). Server still logs `Chat format: peg-native`; that
  is the per-request parser, not a parse failure.
- `--cont-batching` retained: helps streaming generation inside the
  single slot.

The unit-file comment block documents each flag with the failure mode
it prevents. Future operators do not need to re-derive the np=1 fix.

## Why generation is still not usable

A single 13-token test prompt against the np=1 config produced:
- Prompt processing: 6.19s for 13 tokens = 2.1 t/s. Acceptable for
  Strix Halo Vulkan cold-compile.
- Generation: 5 tokens in roughly 78 seconds = approximately 16
  seconds per generated token.

That is 500x slower than the expected Strix Halo Vulkan throughput for
a 3B-active MoE. Plausible causes, in declining order of likelihood:
- qwen35moe MoE routing in llama.cpp build 9464 lands a CPU-fallback
  path for the routing kernel. Build 9464 is the first build that
  understands the `qwen35moe` architecture at all, and arch support
  can land before the Vulkan inference kernels are optimised.
- Vulkan shader cold-compile cascading per generated token (each
  step triggers fresh shader compilation). `--no-warmup` makes this
  worse because the warmup would have pre-compiled the shaders.
  Re-enabling warmup risks the original buffer-alloc warning.
- The 62.8 MiB context checkpoint snapshot logged by
  `slot create_check` may be triggering on every generated token in
  some path that should only fire on prompt-eval completion.

Diagnosing this further is multi-hour llama.cpp build-and-rebuild
territory and not on the LCB cascade critical path.

## Decision

Cascade leg 2 stays on rtx6000 vLLM safetensors per task #120's
rewrite. Strix Halo qwen36-35b-a3b serves test connections cleanly but
is not viable as an LCB benchmark backend at current llama.cpp build.

The mini-beast unit file is preserved in its corrected form so that:
- The qwen36-35b-a3b.service can be re-tested against future
  llama.cpp builds without re-deriving the np=1 fix.
- The mini-beast-8060s gpufarm resource correctly declares the
  service in its conflicts_with and external_lease blocks
  (resources.yaml).

## Files touched

- `mini-beast:/home/alex/.config/systemd/user/qwen36-35b-a3b.service`
  (Agent B-retry, in-place edit, not committed because mini-beast is
  not a git repo). The unit file is the canonical artifact; this doc
  references it but does not duplicate it.
- This file: `docs/strixhalo_cascade_fix.md`.

## Follow-up if Strix Halo path becomes useful

- Retest after llama.cpp >= build 9500 lands optimised qwen35moe
  Vulkan kernels.
- If retesting, restore `--warmup 1` and verify the
  buffer-alloc warning is gone (likely an artifact of the np=4 +
  Q8 KV combination, not warmup itself).
- Investigate the per-token `slot create_check` log line: if it
  fires on every generated token, llama.cpp's context-checkpoint
  cadence is the bottleneck and `--context-checkpoint-spacing`
  (or the per-build equivalent) should be raised.
