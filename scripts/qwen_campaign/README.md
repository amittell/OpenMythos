# Qwen 5-model x 6-benchmark eval campaign (run3)

Autonomous orchestration of LCB codegen + LCB testpred + LCB codeexec +
EvalPlus HumanEval+ + EvalPlus MBPP+ + BigCodeBench across:

- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-27B` (BF16)
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.6-35B-A3B` (BF16)
- `Qwen/Qwen3-Coder-Next-FP8`

Runs on `kebab-cruelist.lan` (driver) against vLLM endpoints on
`kebab-rtx6000.lan` GPU 0 + GPU 1.

## Layout

| Script                  | Where it runs          | Purpose                                             |
| ----------------------- | ---------------------- | --------------------------------------------------- |
| `start_endpoint.sh`     | rtx6000                | Boot one vLLM container with multi-name aliases     |
| `bench_driver.sh`       | cruelist               | Run 6 benches against one endpoint                  |
| `orchestrator.sh`       | cruelist               | Cycle through 3 phases, manage endpoint swaps       |
| `monitor.sh`            | cruelist               | One-shot status snapshot                            |

## Phase layout (parallelised on 2 GPUs)

| Phase | GPU 0                  | GPU 1                  | Walltime budget |
| ----- | ---------------------- | ---------------------- | --------------- |
| 1     | Qwen3.6-27B-FP8        | Qwen3.6-35B-A3B-FP8    | ~30 h           |
| 2     | Qwen3-Coder-Next-FP8   | Qwen3.6-27B-BF16       | ~30 h           |
| 3     | Qwen3.6-35B-A3B-BF16   | (idle)                 | ~25 h           |

Total wall: ~85 hours = ~3.5 days.

## Sampling parameters (per Qwen recommendations)

| Family                  | temperature | top_p | top_k | max_tokens |
| ----------------------- | -----------:| -----:| -----:| ----------:|
| qwen3.6 (dense + MoE)   | 0.6         | 0.95  | 20    | 65,536     |
| qwen3-coder-next        | 1.0         | 0.95  | 40    | 65,536     |

Run #2 used `temp=0, top_p=1, max_tokens=16384` against thinking-mode models
and landed 40 pts below Qwen-official LCB v6 numbers; this campaign fixes that.

## Preconditions

* `kebab-rtx-vllm.service` (gpt-oss-120b on GPU 0) **stopped** before launch;
  orchestrator restarts it on Phase 3 completion.
* `/models/Qwen3.6-27B-FP8` and `/models/Qwen3.6-35B-A3B-FP8` downloaded
  (each is fine-grained block-128 FP8 from Qwen).
* `kebab-cruelist.lan` venvs ready: `evalplus-venv` (0.3.1),
  `bigcodebench-venv` (0.2.5), and `LiveCodeBench/venv`.

## Outputs

All under `kebab-cruelist:/home/alexm/qwen_campaign/<MODEL_TAG>/`:

```
Qwen3.6-27B-FP8/
  log.txt                    # driver log
  evalplus_humaneval/
  evalplus_mbpp/
  lcb_codegen.log + LCB output via $LCB_DIR/output/Qwen3.6-27B/
  lcb_testpred.log
  lcb_codeexec.log
  bcb/                       # BigCodeBench generations + eval
```

The master orchestrator log lives at
`kebab-cruelist:/home/alexm/qwen_campaign/campaign.log`.

## Monitor

```bash
ssh -4 alexm@kebab-cruelist.lan '/home/alexm/qwen_campaign_monitor.sh'
```

Shows endpoint health + per-model progress + output volumes + master log tail.

## After completion

The orchestrator's last action is `sudo systemctl start kebab-rtx-vllm.service`
to restore gpt-oss-120b on GPU 0. Verify in `campaign.log`.

Rsync results back to OpenMythos:
```bash
rsync -ah --progress alexm@kebab-cruelist.lan:/home/alexm/qwen_campaign/ \
  /Users/alex/git/OpenMythos/docs/benchmarks/qwen_run3/
```
