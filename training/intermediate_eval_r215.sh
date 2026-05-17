#!/usr/bin/env bash
# Run an intermediate-checkpoint eval of r2.15 on kebab-rtx6000 GPU 1.
#
# Pipeline:
#   1. Pick the most recent step_NNNNNNN_rank0.pt on kebab-spark
#   2. Rsync the matching rank1/2/3 shards from the gx10 nodes back to
#      kebab-rtx6000 (NOT kebab-spark -- the consolidator runs there
#      and we want all 4 shards local for it)
#   3. Run consolidate_ckpt_single_host.py via 4-rank torchrun on
#      kebab-rtx6000 (2 GPUs, 2 procs per GPU)
#   4. Dispatch the standard eval bundle (per_token_halt, depth_extrap,
#      reasoning_eval, gen_samples_multidepth) against the consolidated
#      ckpt -- pinned to blackwell-gpu1 via env_overrides so the
#      operator's vllm-120b on gpu0 stays untouched
#
# Usage (run on kebab-spark.lan):
#
#   STEP=$(ls /home/alexm/OpenMythos/checkpoints_3b_varT_pondernet_round215/step_*_rank0.pt \
#          | sort | tail -1 | sed 's/.*step_0*\([0-9]*\)_rank0.pt/\1/')
#   bash training/intermediate_eval_r215.sh $STEP
#
# Or pass a specific step:
#   bash training/intermediate_eval_r215.sh 1400

set -euo pipefail

STEP=${1:?STEP required (e.g. 1400)}
PADDED=$(printf "%07d" "$STEP")
CKPT_DIR=checkpoints_3b_varT_pondernet_round215
SHARD_PREFIX="$CKPT_DIR/step_${PADDED}"
FULL_PATH="$CKPT_DIR/step_${PADDED}_full.pt"

REPO_SPARK=/home/alexm/OpenMythos
REPO_RTX=/home/alexm/OpenMythos

# Where the 3 rank shards live (one per gx10 node).
declare -A RANK_HOST=(
  [1]=kebab-gx10-200g
  [2]=kebab-gx10-2-200g
  [3]=kebab-gx10-3-200g
)

echo "[intermediate-eval] step=$STEP src=$SHARD_PREFIX dst=$FULL_PATH"

# --------------- 1. Verify rank-0 shard on kebab-spark ---------------
ssh -q alexm@kebab-spark.lan "test -f $REPO_SPARK/${SHARD_PREFIX}_rank0.pt" || {
  echo "[intermediate-eval] FATAL: rank-0 shard not on kebab-spark: $SHARD_PREFIX"
  exit 1
}

# --------------- 2. Mkdir on kebab-rtx6000 + rsync all 4 shards there ---------------
ssh -q alexm@kebab-rtx6000.lan "mkdir -p $REPO_RTX/$CKPT_DIR"

# Rank 0: kebab-spark -> kebab-rtx6000
echo "[intermediate-eval] rsync rank0: kebab-spark -> kebab-rtx6000"
ssh -q alexm@kebab-spark.lan \
    "rsync -avz --partial $REPO_SPARK/${SHARD_PREFIX}_rank0.pt alexm@kebab-rtx6000.lan:$REPO_RTX/$CKPT_DIR/" &

# Ranks 1/2/3: gx10 nodes -> kebab-rtx6000 (parallel)
for r in 1 2 3; do
    host=${RANK_HOST[$r]}
    echo "[intermediate-eval] rsync rank$r: $host -> kebab-rtx6000"
    ssh -q alexm@$host \
        "rsync -avz --partial $REPO_SPARK/${SHARD_PREFIX}_rank${r}.pt alexm@kebab-rtx6000.lan:$REPO_RTX/$CKPT_DIR/" &
done
wait
echo "[intermediate-eval] all 4 shards on kebab-rtx6000"

# --------------- 3. Run single-host consolidator on kebab-rtx6000 ---------------
echo "[intermediate-eval] consolidating via 4-rank single-host torchrun..."
ssh -q alexm@kebab-rtx6000.lan "
    cd $REPO_RTX && \
    /home/alexm/venvs/vllm-turboquant/bin/python3 -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port=29555 \
        training/consolidate_ckpt_single_host.py \
        $SHARD_PREFIX \
        $FULL_PATH
"
echo "[intermediate-eval] consolidation done; full ckpt at $FULL_PATH"

# --------------- 4. Run eval bundle on blackwell-gpu1 ---------------
# Pin to GPU 1 via CUDA_VISIBLE_DEVICES so vllm-120b on GPU 0 stays warm.
ssh -q alexm@kebab-rtx6000.lan "
    cd $REPO_RTX && \
    rm -f /run/user/1000/r215_step${STEP}_evals.log
    nohup setsid -f bash -c '
        export CUDA_VISIBLE_DEVICES=1
        export CKPT=$FULL_PATH
        export CKPT_DIR=$CKPT_DIR
        for evalname in per_token_halt_analysis depth_extrap reasoning_eval; do
            export OUT=$REPO_RTX/docs/intermediate_r215_step${STEP}_\${evalname}.json
            echo \"[eval] running \$evalname -> \$OUT\"
            /home/alexm/venvs/vllm-turboquant/bin/python3 training/\${evalname}.py
        done
    ' > /run/user/1000/r215_step${STEP}_evals.log 2>&1 </dev/null
"
echo "[intermediate-eval] eval bundle dispatched on kebab-rtx6000 GPU 1"
echo "[intermediate-eval] tail /run/user/1000/r215_step${STEP}_evals.log on kebab-rtx6000 to watch progress"
echo "[intermediate-eval] outputs will land at $REPO_RTX/docs/intermediate_r215_step${STEP}_*.json"
