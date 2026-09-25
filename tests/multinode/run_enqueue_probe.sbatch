#!/bin/bash
#SBATCH -p kyutai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c 224
#SBATCH --mem=512G
#SBATCH -t 1:30:00
#SBATCH --exclusive
#SBATCH -J mojoccl_probe
#SBATCH --exclude=par2dc5-ai-prd-cl02s03dgx01
#SBATCH -o /home/gabriel/ddp_work/logs/mojoccl_probe_%j.log
#
# The host-cost picture of one allreduce, both CCLs, in one job: what the DDP
# reducer pays on the autograd thread per call (enqueue_bench.py), the bucket
# list DDP actually issues for GPT-2 XL (ddp_buckets.py), that list replayed
# back to back (bucket_loop.py), the work-ring pressure counters of a real
# DDP step (MOJOCCL_IB_TRACE=1) and ring_pressure.py's broadcast burst.
#
# Run it before and after a change to the collective's launch structure; the
# enqueue legs are ABBA (vendor, mojo, mojo, vendor) so a clock ramp cancels.
#
#   REPO=<worktree> sbatch tests/multinode/run_enqueue_probe.sbatch

set -x
: "${REPO:=${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO/torch_mojo_backend/distributed/mojoccl" ]; then
  echo "REPO=$REPO is not a torch-mojo-backend checkout" >&2
  exit 1
fi
LOGDIR=/home/gabriel/ddp_work/logs
NANOGPT=/home/gabriel/ddp_work/nanoGPT
mkdir -p "$LOGDIR"

export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4
export PYTHONPATH=$REPO
cd "$REPO"

echo "=== nodes ==="
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "=== branch/commit ==="
git rev-parse --abbrev-ref HEAD; git log --oneline -1
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR
XL="--n-layer 48 --n-head 25 --n-embd 1600 --bias"

set_ccl() {  # $1 = vendor|mojo
  if [ "$1" = "mojo" ]; then export TORCH_MOJO_BACKEND_CCL=mojo
  else unset TORCH_MOJO_BACKEND_CCL; fi
}

# --- the bucket list DDP issues for GPT-2 XL (one rank is enough) ----------
# `--gres=gpu:8` on the step, not just the job: a step that inherits no GPU
# cannot even autodetect an architecture to build the Mojo library for.
echo "=== ddp_buckets (GPT-2 XL) ==="
set_ccl mojo
BUCKLOG="$LOGDIR/mojoccl_probe_${SLURM_JOB_ID}_buckets.log"
srun --nodes=1 --ntasks=1 --gres=gpu:8 bash -c "uv run --no-sync torchrun \
    --standalone --nproc-per-node=1 \
    tests/multinode/ddp_buckets.py --nanogpt-path $NANOGPT $XL" \
  2>&1 | tee "$BUCKLOG" | grep -E "^RESULT|Error|Traceback"
BUCKETS=$(grep -o 'RESULT BUCKETS=.*' "$BUCKLOG" | head -1 | cut -d= -f2)
# GPT-2 XL as DDP issued it when this was last measured, so a failed probe
# still replays something of the right shape instead of failing the leg.
: "${BUCKETS:=1,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25}"
echo "=== BUCKETS=$BUCKETS ==="

# --- host enqueue cost per dist.all_reduce, ABBA ---------------------------
run_enqueue() {  # $1 = ccl, $2 = leg
  echo "=== enqueue_bench ccl=$1 leg=$2 ==="
  set_ccl "$1"
  local log="$LOGDIR/mojoccl_probe_${SLURM_JOB_ID}_enqueue_${1}_leg${2}.log"
  srun bash -c "uv run --no-sync torchrun \
      --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=$MASTER_ADDR:29611 --rdzv-id=\$SLURM_JOB_ID-eq-$1-$2 \
      tests/multinode/enqueue_bench.py" \
    2>&1 | tee "$log" | grep -E "^RESULT|Error|Traceback"
  echo "=== enqueue_bench ccl=$1 leg=$2 exit: ${PIPESTATUS[0]} ==="
}
run_enqueue vendor 1
run_enqueue mojo 1
run_enqueue mojo 2
run_enqueue vendor 2

# --- the same bucket list replayed back to back ----------------------------
run_buckets() {  # $1 = ccl
  echo "=== bucket_loop ccl=$1 ==="
  set_ccl "$1"
  local log="$LOGDIR/mojoccl_probe_${SLURM_JOB_ID}_bucketloop_${1}.log"
  srun bash -c "BUCKETS=$BUCKETS STEPS=20 MM=1 \
      uv run --no-sync torchrun \
      --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=$MASTER_ADDR:29612 --rdzv-id=\$SLURM_JOB_ID-bl-$1 \
      tests/multinode/bucket_loop.py" \
    2>&1 | tee "$log" | grep -E "^\[rank|Error|Traceback"
  echo "=== bucket_loop ccl=$1 exit: ${PIPESTATUS[0]} ==="
}
run_buckets vendor
run_buckets mojo

# --- work-ring pressure inside a real GPT-2 XL DDP step --------------------
echo "=== nanogpt_ddp GPT-2 XL, mojoccl, MOJOCCL_IB_TRACE=1 ==="
set_ccl mojo
TRLOG="$LOGDIR/mojoccl_probe_${SLURM_JOB_ID}_ibtrace.log"
srun bash -c "MOJOCCL_IB_TRACE=1 uv run --no-sync torchrun \
    --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29613 --rdzv-id=\$SLURM_JOB_ID-tr \
    tests/multinode/rank_bind.py --device mojo --nanogpt-path $NANOGPT \
    --data-dir $NANOGPT/data/shakespeare --log-interval 1 --seed 1337 \
    --max-iters 12 --eval-interval 0 --batch-size 8 $XL" \
  2>&1 | tee "$TRLOG" | grep -E "^step|mojoccl net:|collectives via|Error|Traceback"
echo "=== ibtrace exit: ${PIPESTATUS[0]} ==="

# Fused geometry and pipeline chunking use measured code defaults.
# Benchmark a source change through the normal stack protocol above.

# --- broadcast burst (work-ring back-pressure) -----------------------------
echo "=== ring_pressure (mojoccl) ==="
srun bash -c "N=1500 MIB=4 MOJOCCL_IB_TRACE=1 \
    uv run --no-sync torchrun \
    --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29614 --rdzv-id=\$SLURM_JOB_ID-rp \
    tests/multinode/ring_pressure.py" \
  2>&1 | tee "$LOGDIR/mojoccl_probe_${SLURM_JOB_ID}_ringpressure.log" \
  | grep -E "^\[rank|mojoccl net:|Error|Traceback"
echo "=== ring_pressure exit: ${PIPESTATUS[0]} ==="

echo "=== ENQUEUE PROBE DONE ==="
