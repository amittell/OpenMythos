# OpenMythos mythos_3b — First 4-Node Training Run

Session journal for the first end-to-end training of `mythos_3b` on the DGX Spark cluster.
Date: 2026-04-21. Operator: amittell. Platform: 4x NVIDIA GB10 (DGX Spark), 200G RoCE fabric.

## Summary

Ran `mythos_3b` (3.1B params, MLA attention, 64-expert MoE, 4 recurrent loops) with FSDP
across 4 nodes. Validated the architecture trains cleanly in bf16 under FSDP + NCCL RoCE,
loss descends smoothly from init ~12.8 into the 5s, all LTI / MoE / ACT invariants hold.
Final run targets 100 M FineWeb-Edu tokens with a proper cosine-annealed finish at step 6103
(~16-18 hr wall clock).

This is a research-grade proof of viability for the Parcae-style "train shallow, infer deep"
looped-transformer thesis at 3B scale. Not a production model. No post-training, no SFT, no
alignment.

## Cluster topology

| node | mgmt hostname | 200G IP | role | HW |
|---|---|---|---|---|
| kebab-spark | 192.168.210.107 | 192.168.100.10 | rank 0, rendezvous, HF head | GB10, 128 GB unified LPDDR5X |
| kebab-gx10 | 192.168.210.119 | 192.168.100.11 | rank 1 | GB10, 128 GB unified LPDDR5X |
| kebab-gx10-2 | 192.168.210.67 | 192.168.100.12 | rank 2 | GB10, 128 GB unified LPDDR5X |
| kebab-gx10-3 | 192.168.210.184 | 192.168.100.13 | rank 3 | GB10, 128 GB unified LPDDR5X |

Fabric: MikroTik CRS812-8DS 200G switch, bond0 active-backup on each node (enp1s0f1np1
active slave, enP2p1s0f1np1 standby). MTU 9000. RoCE v2 over bond0, NCCL `NET/IB` path.
Management net (enP7s7) terminates at a Firewalla gateway at `192.168.210.1`.

## Software baseline (on every node)

- Ubuntu (kernel 6.14.0-1015-nvidia)
- NVIDIA driver 580.95.05, CUDA 13.0
- Python 3.12.3, `torch 2.9.0+cu130`, `transformers 4.56.0`, `datasets 4.8.4`, `loguru 0.7.3`
- `open_mythos` editable-installed via `pip install --user --break-system-packages --no-deps -e ~/OpenMythos`
  (on gx10-3, no poetry-core available, so installed via `.pth` file pointing at `~/OpenMythos`)

## Issues encountered and fixes

### Infrastructure

| # | issue | root cause | fix | where |
|---|---|---|---|---|
| 1 | gx10-3 docker daemon failing to start since Feb 10 2026 | stale swarm state pointing at unreachable IP after a reboot | `systemctl stop docker && rm -rf /var/lib/docker/swarm && systemctl start docker`; `docker node rm kebab-gx10-3` on head | gx10-3 |
| 2 | gx10-3 200G mesh isolation (pings fail both ways) | bond0 came up with `enP2p1s0f1np1` as active slave post-reboot; that PF's path doesn't reach the switch. Other 3 nodes correctly used `enp1s0f1np1` | pin active slave: `echo enp1s0f1np1 > /sys/class/net/bond0/bonding/active_slave` | all nodes |
| 3 | same as #2 recurs after every reboot | netplan couldn't persist `primary:` because slave is defined by UUID, not device name | installed `bond0-primary.service` systemd unit on all 4 nodes, runs at `network-online.target` | /etc/systemd/system/bond0-primary.service |
| 3a | bond0-primary.service silently failed on all 4 nodes after 2026-05-20 power event; #2 symptom returned | the unit's `ExecStartPre` used `\$(seq 1 60)` to escape the dollar from systemd. Modern systemd (256+) treats `\$` as an unknown escape, leaving it literal; bash then chokes on the malformed `$(`. Symptom in journal: `syntax error near unexpected token '('` plus a `Ignoring unknown escape sequences` warning | rewrite as `$$(seq 1 60)` (the correct systemd escape: `$$` → `$`). Tested 2026-05-21 on all 4 nodes, unit now goes `active` cleanly on boot | /etc/systemd/system/bond0-primary.service |
| 4 | bond0 came up with random MAC on spark/gx10 after reboot → switch ARP confusion | NetworkManager's default cloned-mac behavior for bonds | `nmcli con mod bond0 ethernet.cloned-mac-address <enp1_permanent_mac>` on each node | persistent in NM connection |
| 5 | 3 of 4 nodes SSH-unresponsive during high-load 3B training with num_workers=4 | CPU / RAM starvation from 12 tokenizer worker copies × 200k-vocab; OOM killer fired → left zombies | user reboot; reconfigured to `num_workers=1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` | training script + env |
| 6 | all 4 nodes `HTTP 000` to huggingface.co after reboot | Firewalla had a block rule on `huggingface.co` from `kebab-spark` (17 blocked flows visible in Flows); other nodes had DNS return only AAAA | user added "allow LAN to huggingface.co" rule in Firewalla app | external |
| 7 | after Firewalla unblock, still `HTTP 000` — DNS returns A records but app can't resolve | systemd-resolved had no upstream DNS configured for `enP7s7` post-reboot (NM DHCP didn't push) | `resolvectl dns enP7s7 192.168.210.1 && resolvectl domain enP7s7 ~.`; persisted with `nmcli con mod "Wired connection 2" ipv4.dns 192.168.210.1 ipv4.ignore-auto-dns no` | all nodes |
| 8 | rank 3 launch: `torchrun: command not found` | torch installed via `pip --no-deps`, skipped creating the torchrun entry-point script | scp'd torchrun wrapper (220 B) from gx10 to gx10-3's `~/.local/bin/` | gx10-3 |

### Code bugs in `open_mythos/main.py`

| # | bug | symptom | fix |
|---|---|---|---|
| 9 | `mask.dtype` (fp32) mixed into `attn.dtype` (bf16) under FSDP MixedPrecision | `RuntimeError: expected scalar type Float but found BFloat16` at `matmul(attn, v)` in MLA and GQA forward | `attn = attn + mask.to(attn.dtype)` and `attn = attn_drop(softmax(...)).to(v.dtype)` in both `MLAttention.forward` and `GQAttention.forward` |
| 10 | FSDP-wrapped child (TransformerBlock) receives fp32 input, tries matmul with bf16 params | `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` at `q_down(x)` | add `cast_forward_inputs=True` to `MixedPrecision(...)` in training scripts |
| 11 | `RecurrentBlock` early-exit on `halted.all()` diverges between ranks because halting is data-dependent | 10-minute NCCL collective timeout on `_ALLGATHER_BASE` at second forward | skip the short-circuit under distributed: guard with `not (dist.is_available() and dist.is_initialized())` |

### Training instability

| # | issue | fix |
|---|---|---|
| 12 | NCCL handshake failure (`ibv_modify_qp failed with 22 Invalid argument`) at FSDP first all-gather | per-node RoCE v2 IPv4 GID index differs (spark/gx10=5 pre-reboot, gx10-2=3, all=3 post-reboot) — auto-detect with `show_gids rocep1s0f1 \| awk '$7=="bond0" && $6=="v2" && $5~/^192\.168\.100\./ {print $3; exit}'` per rank at launch |
| 13 | first 3B-8loops launch OOM-killed a DataLoader worker, left rank 0 and rank 2 wedged | reduced `num_workers` 4→1, `micro_batch` 2→1, added `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` |
| 14 | restarted with `micro_batch=2` (speedup attempt after 1150 clean steps) → rank 0 SIGKILL'd by OOM killer within minutes; 3 workers sshd-wedged. `micro_batch=2` + FSDP `SHARD_GRAD_OP` (full param replicas) + MoE transient unshards pushed past 128 GB unified memory on the head. | reverted to `micro_batch=1, grad_accum=4`. Known-stable config at 91 GB used / 128 GB on workers. Takeaway: with SHARD_GRAD_OP, `micro_batch=1` is the ceiling for 3B + 4 loops + 16K vocab on 128 GB GB10. To get `micro_batch=2` later, either (a) switch back to FULL_SHARD (slower forward), (b) add activation checkpointing, or (c) drop to 2 loops. |

## Training runs (chronological)

| run | time | config | outcome |
|---|---|---|---|
| CPU smoke | — | `example.py`, vocab 1000, 4 loops | forward + generate OK, `ρ(A)=0.37` |
| 1B GPU smoke (single node) | — | `mythos_1b`, b=1, s=512, bf16 autocast, 4 loops | 4.34 GB peak, 0.59 s forward |
| **mythos_1b FSDP shakeout** | 3-node, 100 steps planned, hit issues | — | many fixes applied iteratively (see #9-11), ran clean for 30 steps, loss 12.80→7.83 |
| 2-node NCCL sanity | — | `torch.distributed` allreduce, 256 MB | 13.35 GB/s aggregate over RoCE — confirmed NCCL using NET/IB not sockets |
| **mythos_3b @ 8 loops v1** | ~18 min | `num_workers=4`, `micro_batch=2` | OOM killer fired, all 4 nodes wedged → user reboot |
| **mythos_3b @ 8 loops v2** | 37 min, step 0-65 | `num_workers=1`, `micro_batch=1`, `grad_accum=8`, FSDP FULL_SHARD | healthy, 78 s/step (too slow) |
| **mythos_3b @ 4 loops fast (initial)** | 7 hr, step 0-1160 | `max_loop_iters=4`, `grad_accum=4`, SHARD_GRAD_OP | 12-22 s/step effective, loss 12.78→5.34 |
| **mythos_3b @ 4 loops fast (restart attempt 1)** | ~3 min @ 16:28 | changed `micro_batch=2`, `grad_accum=2` | OOM-killed rank 0; workers sshd-wedged for ~1 hr |
| **mythos_3b @ 4 loops fast (final)** | resume from step_0001150 after worker recovery | reverted `micro_batch=1, grad_accum=4`, `target_tokens=100M`, `ckpt_every=100` | cosine anneal to step 6103, ~18 hr |

## Final training configuration

File: `training/3b_loops4_fast.py`

```
model:              mythos_3b variant
  dim:              3072
  n_heads:          24
  n_kv_heads:       6
  attn:             MLA (kv_lora_rank=384, q_lora_rank=768, qk_rope=32, qk_nope=96, v=96)
  MoE:              64 experts, 2 shared, 4 routed per token, expert_dim=4096
  max_loop_iters:   4  (architectural default is 16; trained shallow, can infer deep)
  RoPE theta:       500000.0
  params:           ~3.1B total, ~190M activated/token

distributed:
  FSDP:             SHARD_GRAD_OP (params replicated, grads+optim sharded)
  wrap policy:      TransformerBlock, RecurrentBlock
  MixedPrecision:   param/reduce/buffer bf16, cast_forward_inputs=True

optimizer:          AdamW fused, lr=3e-4, wd=0.1, betas=(0.9, 0.95)
schedule:           linear warmup 2000 steps → cosine decay to 3e-5 over (total_steps - warmup)
target:             100_000_000 tokens → total_steps=6103

batching:
  seq_len:          1024
  micro_batch:      2
  grad_accum:       2
  global batch:     4 × 2 × 2 × 1024 = 16,384 tokens/step

dataset:            HuggingFaceFW/fineweb-edu sample-10BT (streaming, sharded rank-modulo)
tokenizer:          openai/gpt-oss-20b (vocab 199,998)

logging:            every 5 steps (rank 0)
checkpoint:         every 100 steps to ~/OpenMythos/checkpoints_3b_loops4_fast/, keep_last=3
                    ~42 GB per checkpoint (3B bf16 model + AdamW fp32 state)
```

## Intermediate-eval pipeline (`training/intermediate_eval_r215*.sh`)

Per-checkpoint evaluator that runs on kebab-rtx6000 GPU 1 against the
4-node sharded checkpoint produced on the kebab-spark cluster:

1. Watcher (`intermediate_eval_r215_watcher.sh`, systemd user unit
   `r215-eval-watcher`) polls the rank-0 checkpoint dir on kebab-spark
   newest-first.
2. For each step where all 4 rank shards exist + the eval JSONs aren't yet
   on disk, dispatches `intermediate_eval_r215.sh STEP`. Sequential — one
   cycle at a time, gated by a flock at `/tmp/r215_eval_watcher.lock`.
3. The eval-script (a) stages rank 1/2/3 shards from gx10 nodes to spark,
   (b) runs the streaming consolidator (`consolidate_ckpt_streaming.py`),
   (c) ships the full.pt to rtx6000, (d) **calls
   `POST /admin/gpus/1/free`** on the kebab-rtx-router to evict whatever
   is currently on GPU 1 (vision-cuda + embedding + any parallel coder),
   (e) runs the eval bundle, (f) **restores exactly the backends it
   evicted** via `/admin/models/load` for each name in the router's
   `unloaded[]` response.

The GPU-1-aware admin endpoint matters because GPU 1's tenant inventory
has shifted (vision-only -> vision + embedding -> + parallel coder), and
hard-coding "unload qwen3-vl-32b" would leave embedding + coder occupying
GPU 1 during the eval (causing OOMs). See
`/Users/alex/git/kebab-rtx6000/README.md` "Admin endpoints" for the
router contract.

Hardening layers (introduced 2026-05-20 / 21):

- **flock singleton** at `/tmp/r215_eval_watcher.lock` — second instance
  of the watcher exits 0 instead of dogpiling.
- **GPU-1 cross-host mutex** at `/tmp/r215_gpu1.lock` (mkdir-based, with
  stale takeover after `GPU_LOCK_TTL=3000s`) — prevents two concurrent
  evals on GPU 1.
- **EXIT trap** restores GPU 1 tenants + releases the lock on any exit
  path, so a timeout-killed cycle does not leave the router degraded.
- **Mid-eval GPU watchdog** — backgrounded SSH probe of `nvidia-smi -L`
  every 60s; 2 consecutive failures triggers `pkill -P $$` of the in-
  flight remote-eval SSH wrappers + `exit 4`. The watcher recognises
  rc=4 as a GPU fault and stops dispatching until the node recovers.
- **systemd StartLimit** — 5 crashes / 300s -> unit parks in `failed`
  instead of respawning. The flock guard exits 0 (not non-zero) on a
  lost race so it does not count toward this limit.

## Running / monitoring / recovery

### Check status
```bash
# from my Mac
ssh alexm@kebab-spark.lan 'grep -E "step|Checkpoint" /tmp/train_r0.log | tail -5'
ssh alexm@kebab-spark.lan 'ls -la ~/OpenMythos/checkpoints_3b_loops4_fast/'
for h in kebab-spark.lan kebab-gx10.lan kebab-gx10-2.lan kebab-gx10-3.lan; do
  echo "=== $h ==="
  ssh alexm@$h 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; free -h | head -2'
done
```

### Kill cleanly
```bash
for h in kebab-spark.lan kebab-gx10.lan kebab-gx10-2.lan kebab-gx10-3.lan; do
  ssh alexm@$h 'pkill -9 -f "3b_loops4\|torchrun\|torch.distributed.run" 2>/dev/null' &
done
wait
```

### Relaunch (auto-resumes from latest checkpoint)

Script: from my Mac, equivalent to what the session used. Each rank's GID is auto-detected.

```bash
get_gid() {
  ssh alexm@$1 bash -s << 'EOF'
show_gids rocep1s0f1 | awk '$7=="bond0" && $6=="v2" && $5 ~ /^192\.168\.100\./ {print $3; exit}'
EOF
}
MASTER=192.168.100.10
PORT=29520       # pick a fresh one each run
NCCL_BASE='NCCL_DEBUG=WARN NCCL_IB_HCA=rocep1s0f1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/home/alexm/.local/bin:$PATH'

for i in 0 1 2 3; do
  case $i in
    0) h=kebab-spark.lan ;;
    1) h=kebab-gx10.lan ;;
    2) h=kebab-gx10-2.lan ;;
    3) h=kebab-gx10-3.lan ;;
  esac
  gid=$(get_gid "$h")
  ssh alexm@$h "rm -f /tmp/train_r${i}.log; cd ~/OpenMythos && nohup env $NCCL_BASE NCCL_IB_GID_INDEX=${gid} torchrun --nnodes=4 --nproc_per_node=1 --node_rank=$i --master_addr=$MASTER --master_port=$PORT training/3b_loops4_fast.py >/tmp/train_r${i}.log 2>&1 </dev/null & disown" &
done
wait
```

The script's `_list_ckpts(ckpt_dir)[-1]` loads the latest checkpoint automatically; `start_step` advances to where the checkpoint left off.

## Loss trajectory (pre-restart run)

| step | tokens | loss | grad norm | LR | notes |
|---|---|---|---|---|---|
| 5 | 80 K | 12.78 | 8.13 | 6.0e-07 | random init baseline |
| 50 | 820 K | 10.45 | 3.72 | 7.35e-06 | first checkpoint |
| 100 | 1.64 M | 8.33 | 1.63 | 1.48e-05 | cleared MoE settling |
| 200 | 3.28 M | 7.30 | 2.30 | 2.98e-05 | into steady descent |
| 500 | 8.19 M | 6.60 | 1.91 | 7.49e-05 | 25% through warmup |
| 700 | 11.47 M | 6.13 | 2.26 | 1.06e-04 | |
| 1000 | 16.38 M | 5.84 | 1.10 | 1.50e-04 | 50% through warmup |
| 1135 | 18.6 M | **5.34** | 1.30 | 1.70e-04 | all-time low in pre-restart run |
| 1150 | 18.84 M | 5.89 | 1.13 | 1.72e-04 | resume point for post-restart |

## Post-restart schedule (projected)

- step 1150 → 2000: remainder of linear warmup, LR 1.72e-04 → 3.00e-04 (~850 steps, ~3 hr)
- step 2000 → 6103: cosine decay LR 3e-4 → 3e-5 (~4100 steps, ~13 hr)
- **Expected finish**: step 6103, ~100 M tokens seen, fully annealed checkpoint, ~16-18 hr from restart at 16:28

## Architectural invariants being verified

- `ρ(A) < 1` at all times — LTI injection matrix spectral radius bounded by construction
  via `A = Diag(-exp(log_A))` and ZOH discretization with learned `Δt`. Parcae (Prairie
  et al., 2026). This is *not* a soft constraint — it's mathematically impossible for
  the parameterization to violate it, so training should remain stable under any LR.
- MoE router bias updates are non-gradient (no load-balance loss needed); routing
  distribution balanced across experts via external bias adjustment.
- ACT halting integrates per-position — halted positions contribute their `weight` exactly
  once on halting step, then zero.
- Depth extrapolation: evaluate the trained 4-loop checkpoint at `n_loops=8`, `16`, `32`
  after training to test whether loss/perplexity improves at deeper inference than training
  (the signature looped-transformer experiment).

## What this model will and won't do

**Will:**
- Continue educational web text coherently (it's just a base LM)
- Show measurable perplexity gains at inference when looped deeper than training
- Serve as a starting point for further pretraining or fine-tuning
- Demonstrate architectural viability: 3B looped MoE + LTI-stable injection trains cleanly

**Won't:**
- Follow instructions (no SFT)
- Chat (no post-training)
- Avoid harmful output (no safety training)
- Match open-source 3B baselines on benchmarks — 100 M tokens is ~1/600th of what
  SmolLM-1.7B or Pythia-2.8B saw; expect equivalence to a 300-500M model trained longer

## References

- README.md — full architectural thesis
- docs/open_mythos.md — API reference
- Parcae (Prairie et al., 2026) — LTI-stable looped transformer scaling laws
- Saunshi et al., 2025 — "Reasoning with Latent Thoughts: On the Power of Looped Transformers"
- DeepSeekMoE (Dai et al., 2024) — fine-grained MoE basis
- DeepSeek-V2 (2024) — MLA attention

## Round 2: variable-T training (200M tokens)

Started: 2026-04-24 17:35:02  
Completed: 2026-04-28 03:09:38  
Wall-clock training: 3 days, 9:34:36  
Steps: 12205 / 12207  
Final ckpt: `checkpoints_3b_varT_fast/step_0012207_full.pt` (step 12207)  
Lowest training loss observed: 2.9648 at step 11990 (T=3)  

Configuration changes vs round 1:

- Variable T: sampled `T ~ Uniform(2, 12)` per optimizer step (round 1 was fixed T=4)
- LoRA per-loop scale embedding sized to T_MAX=12 so every sampled depth has its own slot
- Sharded checkpoints (`FSDP.SHARDED_STATE_DICT`): each rank writes its own ~11 GB shard
  to local disk. No 45 GB cross-node rsync. Save time dropped from ~7 min to ~25 sec.
- Worker prune fix: `_distribute_checkpoint` now applies `keep_last=3` on each peer.
- Target tokens: 200M (2x round 1)

Artifacts:

- `docs/depth_extrap_round2.json` — raw eval numbers (FineWeb-Edu, GSM8K, TinyStories)
- `docs/round1_vs_round2.md` — side-by-side comparison
- `docs/gen_samples_round2.txt` — 8 prompts at mid-depth
- `docs/gen_samples_round2_multidepth.txt` — same prompts at K=2/6/12/24
- `docs/act_halt_histogram_round2.json` — per-token halt-step distribution
- `docs/training_curve.png` — loss vs step (color = T)
- `docs/loss_by_T.png` — loss binned by sampled depth

To back up the consolidated checkpoint to the qnap NAS (run from local Mac):

```bash
scp -3 kebab-spark.lan:checkpoints_3b_varT_fast/step_0012207_full.pt \
    alexm@kebabstore.lan:/share/CACHEDEV1_DATA/openmythos_backups/
```
