#!/bin/bash
#SBATCH -p kyutai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 1:30:00
#SBATCH --exclusive
#SBATCH -J mojoccl_2node
#SBATCH --exclude=par2dc5-ai-prd-cl02s03dgx01
#SBATCH -o /home/gabriel/ddp_work/logs/mojoccl_2node_%j.log
#
# 2-node x 8-GPU (16-rank) validation + bench for mojoccl
# (torch_mojo_backend/distributed/mojoccl/), the Mojo re-implementation of
# NCCL's C ABI selected with TORCH_MOJO_BACKEND_CCL=mojo. "vendor" below
# means that variable left UNSET (real NCCL). See tests/multinode/README.md
# for the full leg-by-leg description and RUN_MOJO=0/1 semantics.
#
# Three phases, in order:
#   A. tests/ddp_worker.py collectives/ddp_parity/stress and
#      tests/fsdp_worker.py reduce_scatter/fsdp_collectives_stress at 16
#      ranks, once per ccl.
#   B. ar_bench_gpt2.py (allreduce device-time bench) in ABBA order
#      (vendor, mojo, mojo, vendor) so a thermal/clock ramp cancels to
#      first order (see AGENTS.md "Benchmark harness notes").
#   C. demo_scripts/nanogpt_ddp.py, 40 steps, ABBA order, under both ccls.
#
# RUN_MOJO=0 (default 1): skip every mojo-ccl leg. mojoccl's multi-node
# transport (TCP bootstrap carried in the unique id, GPUDirect RDMA over
# libibverbs -- see the "Multi-node" subsection of agents_docs/distributed.md) is
# exercised by default; RUN_MOJO=0 instead gets a vendor-only NCCL reference
# run without spending the job's time budget on the mojo legs.
#
# Every leg is one torchrun launch across both nodes (--nnodes=2
# --nproc-per-node=8), guarded per node by `flock /tmp/gpu_lock_0.lock`
# inside the srun'd command (srun runs one task per node; that task's
# flock is local to that node's /tmp). `--exclusive` because that lock is
# node-wide: sharing a node with another GPU job means every leg queues
# behind it, and a job that spent its whole time budget waiting on the
# lock rather than running was mistaken for a hang. Each leg's full output
# goes to its own log under /home/gabriel/ddp_work/logs/; a filtered view (result
# lines, per-rank OK/FAIL, step lines, errors) is echoed to this job's
# main -o log, which is what tests/multinode/summarize.py reads.

set -x
: "${RUN_MOJO:=1}"

# The checkout to test. Defaults to the directory sbatch was run from, so a
# worktree tests its own code; `REPO=<path> sbatch ...` overrides it. SLURM
# copies this script to its spool directory, so $0 says nothing useful.
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
echo "=== per-node SM clock ==="
srun --ntasks-per-node=1 bash -c 'hostname; nvidia-smi --query-gpu=clocks.sm --format=csv,noheader'
echo "=== branch/commit ==="
git rev-parse --abbrev-ref HEAD; git log --oneline -1
echo "=== RUN_MOJO=$RUN_MOJO ==="

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR

set_ccl() {  # $1 = vendor|mojo
  if [ "$1" = "mojo" ]; then
    export TORCH_MOJO_BACKEND_CCL=mojo
  else
    unset TORCH_MOJO_BACKEND_CCL
  fi
}

run_worker() {  # $1 = ccl, $2 = mode, $3 = worker (default ddp_worker)
  local ccl=$1 mode=$2 worker=${3:-ddp_worker}
  echo "=== $worker ccl=$ccl mode=$mode (16 ranks) ==="
  set_ccl "$ccl"
  local log="$LOGDIR/mojoccl_2node_${SLURM_JOB_ID}_${worker}_${ccl}_${mode}.log"
  srun bash -c "flock /tmp/gpu_lock_0.lock uv run --no-sync torchrun \
      --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=$MASTER_ADDR:29601 --rdzv-id=\$SLURM_JOB_ID \
      tests/$worker.py $mode" \
    2>&1 | tee "$log" | grep -iE "^\[rank|FAILURES|Error|Traceback"
  echo "=== $worker ccl=$ccl mode=$mode exit: ${PIPESTATUS[0]} ==="
}

run_bench() {  # $1 = ccl, $2 = leg number (label only)
  local ccl=$1 leg=$2
  echo "=== ar_bench_gpt2 ccl=$ccl leg=$leg ==="
  set_ccl "$ccl"
  local log="$LOGDIR/mojoccl_2node_${SLURM_JOB_ID}_arbench_${ccl}_leg${leg}.log"
  local dbg_prefix=""
  # Algo/protocol choices only need capturing on the real-NCCL legs.
  [ "$ccl" = "vendor" ] && dbg_prefix="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING"
  srun bash -c "$dbg_prefix flock /tmp/gpu_lock_0.lock uv run --no-sync torchrun \
      --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=$MASTER_ADDR:29602 --rdzv-id=\$SLURM_JOB_ID \
      /home/gabriel/ddp_work/mojo_collectives/ar_bench_gpt2.py" \
    2>&1 | tee "$log" | grep -E "^RESULT|Bytes -> Algo|Error|Traceback"
  echo "=== ar_bench_gpt2 ccl=$ccl leg=$leg exit: ${PIPESTATUS[0]} ==="
}

run_train() {  # $1 = ccl, $2 = leg number (label only)
  local ccl=$1 leg=$2
  echo "=== nanogpt_ddp ccl=$ccl leg=$leg ==="
  set_ccl "$ccl"
  local log="$LOGDIR/mojoccl_2node_${SLURM_JOB_ID}_nanogpt_${ccl}_leg${leg}.log"
  local common="--nanogpt-path $NANOGPT --data-dir $NANOGPT/data/shakespeare --log-interval 1 --eval-iters 10 --seed 1337 --max-iters 40 --eval-interval 40"
  srun bash -c "flock /tmp/gpu_lock_0.lock uv run --no-sync torchrun \
      --nnodes=\$SLURM_NNODES --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=$MASTER_ADDR:29603 --rdzv-id=\$SLURM_JOB_ID \
      demo_scripts/nanogpt_ddp.py --device mojo $common" \
    2>&1 | tee "$log" | grep -E "^step|val loss|^done:|Error|Traceback"
  echo "=== nanogpt_ddp ccl=$ccl leg=$leg exit: ${PIPESTATUS[0]} ==="
}

# --- Phase A: ddp_worker checks, 16 ranks, per ccl -------------------------
for ccl in vendor mojo; do
  if [ "$ccl" = "mojo" ] && [ "$RUN_MOJO" = "0" ]; then
    echo "=== SKIP ddp_worker ccl=mojo (RUN_MOJO=0) ==="
    continue
  fi
  for mode in collectives ddp_parity stress; do
    run_worker "$ccl" "$mode"
  done
  # The multi-node all-gather/reduce-scatter paths FSDP2 uses (split
  # pipelined gathers, the hierarchical reduce-scatter) at FSDP2 sizes. Not
  # `parity`: its DCP round trip wants a temp directory shared by the nodes.
  for mode in reduce_scatter fsdp_collectives_stress; do
    run_worker "$ccl" "$mode" fsdp_worker
  done
done

# --- Phase B: allreduce bench, ABBA order -----------------------------------
if [ "$RUN_MOJO" = "0" ]; then
  echo "=== SKIP ar_bench_gpt2 ccl=mojo (RUN_MOJO=0) ==="
  run_bench vendor 1
  run_bench vendor 2
else
  run_bench vendor 1
  run_bench mojo 1
  run_bench mojo 2
  run_bench vendor 2
fi

# --- Phase C: nanoGPT DDP training, 40 steps, under both ccls --------------
if [ "$RUN_MOJO" = "0" ]; then
  echo "=== SKIP nanogpt_ddp ccl=mojo (RUN_MOJO=0) ==="
  run_train vendor 1
else
  run_train vendor 1
  run_train mojo 1
  run_train mojo 2
  run_train vendor 2
fi

echo "=== 2-NODE CHECKS DONE ==="
