#!/bin/bash
#SBATCH -p kyutai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c 224
#SBATCH --mem=512G
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH -J e2e_3stacks
#SBATCH --exclude=par2dc5-ai-prd-cl02s03dgx01
#SBATCH -o /home/gabriel/ddp_work/logs/e2e_three_stacks_%j.log
# nanoGPT DDP, 16 ranks on 2 nodes, 30 steps, three stacks. Per batch size: one discarded warm-up
# run PER STACK (they double as the fit gate: a batch size whose warm-up fails on any stack is
# skipped), then five interleaved rounds (ABC CBA ABC CBA ABC). Every run carries the evidence the
# summariser checks: NCCL's INIT/NET debug lines (NCCL stacks), the loader's "collectives via ..."
# trace line (mojo stacks), and rank_bind's NUMA binding lines.
#
# Env: MOJO_TREE (the checkout to run; default: the submitting checkout), BATCHES (default
# "48 32 12"), MODEL_ARGS (nanoGPT config flags, default the 124M model; GPT-2 XL is
# "--n-layer 48 --n-head 25 --n-embd 1600 --bias", which fits batch 8 only), E2E_TAG (log-name
# prefix, default e2e_three_stacks; the summariser reads the same tag). A 1-rank mojo prewarm
# builds every kernel specialization into the shared cache before 16 ranks race for it.
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1); export MASTER_ADDR
scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '; echo
W=${MOJO_TREE:-$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)}; cd $W; echo "mojo tree: $(git log --oneline -1)"
NANOGPT=/home/gabriel/ddp_work/nanoGPT
STOCK_TR=/home/gabriel/ddp_work/torch_cu128/bin/torchrun
NCCLDBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
TAG=${E2E_TAG:-e2e_three_stacks}
LOGS=/home/gabriel/ddp_work/logs
DATA="--nanogpt-path $NANOGPT --data-dir $NANOGPT/data/shakespeare --log-interval 1 --seed 1337 --eval-interval 0 ${MODEL_ARGS:-}"

echo "=== prewarm mojo kernels (1 rank) ==="
srun -N1 -n1 bash -c "cd $W && env PYTHONPATH=$W TORCH_MOJO_BACKEND_CCL=mojo uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29853 --rdzv-id=\$SLURM_JOB_ID-prewarm demo_scripts/nanogpt_ddp.py --device mojo $DATA --max-iters 3 --batch-size 2" > $LOGS/${TAG}_${SLURM_JOB_ID}_prewarm.log 2>&1
echo "=== prewarm exit: $? ==="

run_one() {  # BS index cfg
  local BS=$1 i=$2 cfg=$3
  local LOG=$LOGS/${TAG}_${SLURM_JOB_ID}_bs${BS}_$(printf %02d $i)_$cfg.log
  case $cfg in
    warmup_stock|stock)          TR="$STOCK_TR"; DEV=cuda; ENV="PYTHONPATH= $NCCLDBG";;
    warmup_nccl|mojo_nccl)       TR="uv run --no-sync torchrun"; DEV=mojo; ENV="$NCCLDBG";;
    warmup_mojoccl|mojo_mojoccl) TR="uv run --no-sync torchrun"; DEV=mojo; ENV="TORCH_MOJO_BACKEND_CCL=mojo";;
  esac
  echo "=== bs=$BS run $i cfg=$cfg ==="
  srun bash -c "cd $W && env PYTHONPATH=$W $ENV $TR --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29851 --rdzv-id=\$SLURM_JOB_ID-$BS-$i $W/tests/multinode/rank_bind.py --device $DEV $DATA --max-iters 30 --batch-size $BS" > $LOG 2>&1
  local rc=$?
  echo "=== bs=$BS run $i exit: $rc ==="
  return $rc
}

for BS in ${BATCHES:-48 32 12}; do
  i=0; fit=1
  for cfg in warmup_stock warmup_nccl warmup_mojoccl; do
    i=$((i+1)); run_one $BS $i $cfg || fit=0
  done
  if [ $fit = 0 ]; then echo "=== bs=$BS: a warm-up failed (does not fit?), skipping this batch size ==="; continue; fi
  for cfg in stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl; do
    i=$((i+1)); run_one $BS $i $cfg
  done
done
echo "=== HONEST DONE ==="
