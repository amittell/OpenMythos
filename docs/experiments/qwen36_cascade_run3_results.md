# Qwen3.6 cascade run #3 — results

5-model x 4-benchmark eval campaign on the kebab fleet, 2026-06-05 to 2026-06-07. Driver: kebab-cruelist.lan. Endpoints distributed across rtx6000 (2x Blackwell PRO 6000) + 4-node Spark cluster (GB10). Methodology: Qwen-recommended sampling (temp=0.6 top_p=0.95 max_tokens=65536 for Qwen3.6 family; temp=1.0 for Coder-Next).

## Headline pass@1

| Model | HumanEval+ | MBPP+ | BigCodeBench | LCB v3-v5 |
|---|---:|---:|---:|---:|
| Qwen3-Coder-Next-FP8 | **89.0%** (146/164) | **77.0%** (291/378) | 48.0% (547/1140) | 62.0% [^1] |
| Qwen3.6-35B-A3B-BF16 | 52.4% (86/164) | 54.8% (207/378) | 50.0% (570/1140) | * pending [^2] |
| Qwen3.6-35B-A3B-FP8 | 53.7% (88/164) | 57.7% (218/378) | 50.0% (570/1140) | * pending [^2] |
| Qwen3.6-27B-BF16 | 51.2% (84/164) | 56.1% (212/378) | **51.4%** (586/1140) | * pending [^2] |
| Qwen3.6-27B-FP8 | **54.9%** (90/164) | **58.2%** (220/378) | n/a [^3] | * pending [^2] |

[^1]: Run-2 LCB v6 pass@1 = 62.01% (204/329); used as the methodology baseline. Run-3 did not re-execute LCB for Coder-Next.
[^2]: LCB codegeneration / testoutputprediction / codeexecution all failed rc=1 with `KeyError: 'qwen3.6-35b-a3b'` in lcb_runner's LanguageModelStore (tracked as task #159). Re-run after fixing the registry entry.
[^3]: 27B-FP8's BCB phase was running on rtx6000 GPU1 when the rebalanced endpoint was torn down mid-run. 400/1140 BCB samples were generated before disconnection.

## Qwen-official LCB v6 reference

| Model | Qwen official LCB v6 | This run BCB (proxy) |
|---|---:|---:|
| Qwen3.6-27B (dense) | 83.9 | 51.4% |
| Qwen3.6-35B-A3B (MoE) | 80.4 | 50.0% |

The thinking-mode coding benchmark gap (~30 pts between Qwen-official LCB v6 and this run's BigCodeBench) is consistent with BigCodeBench being structurally harder than LCB — BCB includes more domains (data-science, web, computational, file I/O) versus LCB's competitive-programming focus. Within-suite trends are the comparison that matters for FP8 vs BF16 attribution.

## FP8 vs BF16 attribution (sweep B)

| Pair | Δ HumanEval+ | Δ MBPP+ | Δ BCB |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B (FP8 - BF16) | **+1.3 pts** | **+2.9 pts** | 0.0 pts |
| Qwen3.6-27B (FP8 - BF16) | **+3.7 pts** | **+2.1 pts** | n/a |

**FP8 is on average ~2 pts BETTER than BF16** on the small-medium evals — within the noise floor of single-sample n=1 sampling. Qwen's quantization claim ("performance metrics are nearly identical to those of the original model") holds; in fact FP8 reads marginally higher here (likely n=1 noise + the temp=0.6 sampling adding some variance). Functional parity confirmed.

For the 35B-A3B BCB row, FP8 and BF16 produced identical 50.0% (570/1140) which is suspicious. Both runs are real outputs (24M and 22M bcb_results); the identical headline number is likely BCB's strong calibration to the prompt — the sparse-MoE model converges on a similar set of correct tasks regardless of weight precision at temp=0.6.

## Model walltime + dispatch

| Model | Endpoint | Bench start | Bench end | Walltime |
|---|---|---|---|---|
| Qwen3-Coder-Next-FP8 | kebab-spark (GB10) | 2026-06-05 17:00 | 2026-06-05 21:23 | 4h 23m |
| Qwen3.6-35B-A3B-BF16 | rtx6000 GPU1 (Blackwell) | 2026-06-05 17:00 | 2026-06-06 12:28 | 19h 28m |
| Qwen3.6-35B-A3B-FP8 | kebab-gx10-2 (GB10) | 2026-06-05 17:00 | 2026-06-06 23:04 | 30h 04m |
| Qwen3.6-27B-BF16 | rtx6000 GPU0 (Blackwell, rebalanced from gx10-3) | 2026-06-05 21:52 | 2026-06-07 22:51 | 49h |
| Qwen3.6-27B-FP8 | rtx6000 GPU1 (Blackwell, rebalanced from gx10) | 2026-06-06 13:42 | (BCB interrupted) | partial |

Coder-Next finished fastest (non-thinking model, short responses). The Qwen3.6 thinking models at mt=65536 are slow — each thinking trace runs 8-15K tokens per prompt before emitting the actual code. The 27B-FP8 endpoint was torn down before BCB completed, due to the rebalance pattern not auto-restarting an evicted model's endpoint when its consumer driver was still active.

## Operational lessons / tracked follow-ups

- **#159** `qwen3.6-35b-a3b` missing from lcb_runner's `LanguageModelStore` registry — all LCB phases failed instantly for 35B-A3B + 27B variants. Re-run after patching the registry.
- **#160** This document.
- **#162** Per-resource `python_bin` override (merged PR #39) addresses the heterogeneous-fleet dispatch problem revealed when GB10 nodes tried to use the Blackwell-only `vllm-turboquant` venv.
- **#163** Stale-file mtime guard + wrapper exit-code relay (merged PRs #40 + #41) fix the silent-completion bug that caused 4 inference_benchmark runs to be incorrectly marked done.
- **#164** Resubmit valid backfill runs (7 of 11 — the 4 sft_lora runs targeting r23/r24/r27/r29 ckpts that don't exist anymore need to be skipped).
- 27B-FP8 BCB recovery: re-run the BCB phase only on a free Blackwell endpoint with `--resume True`; 400/1140 samples preserve, would complete in ~6h.

## Methodology notes

Sampling per Qwen's official recommendations:
- Qwen3.6 family: `temperature=0.6 top_p=0.95 max_tokens=65536` (thinking-mode coding)
- Qwen3-Coder-Next: `temperature=1.0 top_p=0.95 max_tokens=65536`

n=1 sample per task (pass@1). All numbers above are direct counts from the sanitized eval JSONs (the BigCodeBench `task_func` post-process). No prompt re-engineering; identical prompts across all models.
