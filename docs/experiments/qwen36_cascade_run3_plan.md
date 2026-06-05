# Qwen3.6 LCB bench cascade -- run #3 plan (2026-06-05)

Operator runbook for cascade run #3. Run #2 completed all three legs cleanly
(rc=0 for legs 1, 2, 3 at 2026-06-05 10:20:33 UTC) but the absolute pass@1
numbers came in 30-40 percentage points below Qwen's official LiveCodeBench v6
scores. The methodology gap is the cause; this plan corrects it and adds the
FP8 vs BF16 attribution sweep the operator requested.

## Run #2 results (BF16, greedy, mt=16384)

| Model                  | run #2 pass@1 | Qwen official LCB v6 | gap        |
| ---------------------- | -------------:| --------------------:| ----------:|
| Qwen3.6-27B            | 0.4377        | 0.839                | -40.1 pts  |
| Qwen3.6-35B-A3B        | 0.3891        | 0.804                | -41.5 pts  |
| Qwen3-Coder-Next-FP8   | 0.6201        | n/a (image-only)     | n/a        |

Walltime: leg 1 25h 46m; leg 2 5h 10m; leg 3 1h 58m. Logs at
`kebab-rtx6000:/home/alexm/cascade/qwen36_bench.log`. The model registry
in `lcb_runner/lm_styles.py` lists Qwen3.6-27B and Qwen3.6-35B-A3B at
`LMStyle.OpenAIChat` (chat-template path), so vLLM applies the official
Qwen chat template by default. The Qwen3.6 chat template enables
thinking-trace generation; this matters for sampling parameters.

## Root cause of the gap

Cascade run #2 sampling parameters vs Qwen-recommended for thinking-mode
coding:

| Parameter         | Cascade run #2 | Qwen3.6 recommended           | Qwen3-Coder-Next recommended |
| ----------------- | -------------- | ------------------------------ | ---------------------------- |
| `temperature`     | 0.0 (greedy)   | 0.6 (precise coding)          | 1.0                          |
| `top_p`           | 1.0            | 0.95                           | 0.95                         |
| `top_k`           | unset          | 20                             | 40                           |
| `max_tokens`      | 16,384         | 32,768 standard / 81,920 math | 65,536                       |
| `presence_penalty`| unset (0.0)    | 0.0 (precise coding) / 1.5 (general) | unset                |
| thinking mode     | default (on)   | on (precise coding)            | on                           |

Three things compound to produce the 40-pt depression:

1. **Greedy decoding with a thinking model.** Thinking-trace generation
   produces non-deterministic exploration internally; greedy decoding
   locks the trace into a single narrow trajectory and the resulting
   distribution is much lower-coverage than the temperature sampling
   the model was tuned against.

2. **`max_tokens=16384` is too small.** Qwen3.6's thinking trace alone
   often runs 8-12K tokens on harder LiveCodeBench problems before the
   model emits the `</think>` tag and produces the actual code. With
   only 16K, the trace gets truncated mid-reasoning and the response
   never contains a valid solution. Qwen's own tech report uses 81,920
   for the math/programming-competition LCB v6 score.

3. **`top_p=1.0` with `top_k` unset** gives the sampler no diversity
   shaping. Combined with the truncated thinking trace, the model often
   produces an empty `code_list` even when `output_list` is non-empty.

The Qwen3-Coder-Next leg suffered the same methodology gap but is less
sensitive because the Coder-Next variant is the non-thinking model in
the family: its trace is short and 16K is sufficient. Its 62% pass@1
is likely a true number within ~5 pts of methodology-correct.

## Run #3 plan

Three legs as before but with three independent variables: chat-template
sampling parameters, weight precision (FP8 vs BF16), and one
dataset-window check. We split run #3 into three sweeps to attribute
each effect cleanly.

### Sweep A: methodology fix on BF16 (the primary)

Same three weights (BF16) cascade run #2 used, with correct sampling:

```bash
# Qwen3.6 dense + MoE legs
--temperature 0.6 \
--top_p 0.95 \
--max_tokens 65536 \
--multiprocess 2

# Qwen3-Coder-Next leg
--temperature 1.0 \
--top_p 0.95 \
--max_tokens 65536 \
--multiprocess 2
```

Apply `top_k=20` (Qwen3.6) or `top_k=40` (Coder-Next) via the lcb_runner
`--top_k` arg if present, otherwise via the vLLM serving-side
`--default-sampling-params` file.

Expected outcome: pass@1 within 5 pts of Qwen's official numbers for
the two Qwen3.6 weights (~80% and ~75% respectively); Coder-Next-FP8
should reach 75-80% from its current 62.

### Sweep B: FP8 attribution

Concurrent with sweep A: replace the BF16 weights with the official
Qwen FP8 variants (downloaded 2026-06-05; see "Models on disk" below):

* `/models/Qwen3.6-27B-FP8` (30.9 GB, vs 52 GB BF16)
* `/models/Qwen3.6-35B-A3B-FP8` (37.5 GB, vs 67 GB BF16)
* `/models/Qwen3-Coder-Next-FP8` (already FP8 in run #2)

Same sampling as sweep A. Expected outcome: Qwen's FP8 model cards
state "performance metrics are nearly identical to those of the
original model" -- run sweep B to verify (within ~1-2 pts of sweep A).
If the gap is larger than 2 pts, the cascade prefers BF16 going
forward; if smaller, FP8 becomes the standard because the smaller
footprint allows running two legs on different GPUs in parallel and
cuts cascade walltime from 33h to ~17h.

### Sweep C: date-window check

A trailing concern: cascade ran `--start_date 2024-08-01 --end_date
2025-02-01` which corresponds to the LiveCodeBench v3/v4 problem set,
not v6. Qwen's 83.9 pass@1 is reported against the v6 cutoff
(2024-08-01 to 2025-05-01). Re-run only the BF16 27B leg under the v6
window (`--end_date 2025-05-01`) for ~50 additional problems and
verify the score is consistent with the sweep A number; if not, our
problem-window choice is also contributing to the gap and must be
fixed for the canonical comparison.

## Models on disk

| Weight                                 | Path                                       | Size  | Pulled            |
| -------------------------------------- | ------------------------------------------ | -----:| ----------------- |
| `Qwen/Qwen3.6-27B` (BF16)              | `/models/Qwen3.6-27B`                      | 52 GB | pre-run-2         |
| `Qwen/Qwen3.6-35B-A3B` (BF16)          | `/models/Qwen3.6-35B-A3B`                  | 67 GB | pre-run-2         |
| `Qwen/Qwen3-Coder-Next-FP8`            | `/models/Qwen3-Coder-Next-FP8`             | 75 GB | pre-run-2         |
| `Qwen/Qwen3.6-27B-FP8`                 | `/models/Qwen3.6-27B-FP8`                  | 31 GB | 2026-06-05        |
| `Qwen/Qwen3.6-35B-A3B-FP8`             | `/models/Qwen3.6-35B-A3B-FP8`              | 38 GB | 2026-06-05        |

All FP8 variants use fine-grained per-block FP8 with block size 128
(same scheme across the family); this matters because vLLM v0.21
needs no special flags to serve them.

## Broader benchmark expansion

`kebab-cruelist.lan` (Ubuntu 24.04, 192.168.210.86, no local GPU)
already has six benchmark suites set up with their own venvs:

| Suite                       | Path                                          | Status     | What it adds              |
| --------------------------- | --------------------------------------------- | ---------- | ------------------------- |
| LiveCodeBench               | `/home/alexm/git/benchmarks/LiveCodeBench`    | active     | already-run baseline      |
| BigCodeBench v0.2.5         | `bigcodebench-venv`                           | venv-only  | 1140 hand-curated tasks, 7 domains, harder than LCB |
| EvalPlus v0.3.1             | `evalplus-venv`                               | venv-only  | HumanEval+/MBPP+, rigorous tests |
| bigcode-evaluation-harness  | `bigcode-evaluation-harness/`                 | repo+venv  | MultiPL-E, RepoBench, DS-1000 wrapper |
| DS-1000                     | `DS-1000/`                                    | repo-only  | 1000 data-science tasks  |
| Spider2                     | `Spider2/`                                    | repo-only  | text-to-SQL              |
| SWE-bench                   | `/home/alexm/git/SWE-bench`                   | repo-only  | repo-level patch tasks   |

`kebab-cruelist.lan` runs the benchmarks against remote vLLM endpoints
(no local GPU), so we point each suite at `http://kebab-rtx6000.lan:8002/v1`
the same way LCB run #2 did.

### Recommended next benchmarks (priority order)

1. **EvalPlus HumanEval+/MBPP+** (1-2h per model, very low walltime).
   Rigorous tests of saturated HumanEval/MBPP; published 2024 Qwen
   numbers are 92-95%, so these still discriminate the Qwen3.6 family
   from Coder-Next at the top of the curve. Good sanity check before
   the cascade rerun: if Qwen3.6-27B FP8 lands below 90% on
   HumanEval+, methodology is still wrong.

2. **BigCodeBench v0.2.5** (6-12h per model, the hardest published
   coding benchmark today). 1140 tasks across data science, web,
   computational, file I/O, etc. Most frontier models score 50-60%;
   Qwen has not published a BigCodeBench number for Qwen3.6.

3. **LCB CodeExecution and TestPrediction scenarios** (lcb_runner
   already has them, just `--scenario codeexecution` or
   `testoutputprediction`). 2-3h per scenario per model.
   Complements the codegeneration scenario already run and is a
   direct contamination-resistant alternative.

4. **SWE-bench Verified** (50+ GPU-hours per model, repo-level
   patches). The gold standard but expensive; Qwen3.6-27B official
   score 77.2%. Only worth running if we want a third-party
   reproduction of the headline number.

5. **DS-1000** (4-6h per model) and **Spider2** (6-10h per model):
   domain-specific (data-science and SQL respectively). Skip until
   the cascade primary is repaired.

## Operator runbook (sweep A)

```bash
# On kebab-rtx6000.lan
cd /home/alexm/cascade
cp qwen36_bench_cascade.sh qwen36_bench_cascade_run3.sh

# Edit run_lcb() in qwen36_bench_cascade_run3.sh:
#  --temperature 0.6 \                 # Qwen3.6 legs; use 1.0 for Coder-Next
#  --top_p 0.95 \
#  --max_tokens 65536 \                # was 16384
#  --custom_output_save_name "..-bench-run3-bf16-mfix"

# Run cascade. Expected walltime ~24h with the larger max_tokens.
nohup bash qwen36_bench_cascade_run3.sh > /tmp/run3.log 2>&1 &
```

For sweep B (FP8), additionally edit the `model_path` arguments in the
three `start_vllm` calls to point at the FP8 variants. Otherwise
identical to sweep A.

## Success criteria

* Sweep A pass@1 within 5 pts of Qwen-official for the BF16 weights.
* Sweep B FP8 pass@1 within 2 pts of sweep A BF16 pass@1.
* If sweep A still falls short by >10 pts, the issue is not sampling
  parameters; investigate (in order): chat-template thinking enable,
  vLLM tokenizer handling of `<think>` tokens, lcb_runner answer
  extraction with thinking traces present.

## References

* [Qwen3.6-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B)
* [Qwen3.6-35B-A3B on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
* [Qwen3.6-27B-FP8 on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
* [Qwen3.6-35B-A3B-FP8 on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
* [Qwen3-Coder-Next-FP8 on Hugging Face](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8)
* run #2 driver script: `kebab-rtx6000:/home/alexm/cascade/qwen36_bench_cascade.sh`
* run #2 log: `kebab-rtx6000:/home/alexm/cascade/qwen36_bench.log`
* run #2 LCB outputs: `kebab-rtx6000:/home/alexm/git/benchmarks/LiveCodeBench/output/Qwen3.6-*`
