#!/bin/bash
# tests/multinode/e2e_three_stacks.sbatch, Adastra edition: 8 ranks on 2 x 4 MI300A, Slingshot (libfabric cxi).
# Usage: as a batch script (sbatch ... e2e_adastra.sh) or from a login node with J=<jobid> e2e_adastra.sh.
# Env: BATCHES (default "48 32 12"), ENDURANCE=1 adds one 1500-step run per stack at batch 32.
S=${E2E_ROOT:-/lus/scratch/CT10/cad17896/gdemarmiesse/mojo-rccl}; J=${SLURM_JOB_ID:-$J}; W=$S/repo; LOGDIR=$S/logs
until [ "$(squeue -h -j $J -o %T)" = "RUNNING" ]; do sleep 20; done
NODES=$(squeue -h -j $J -o %N); MASTER=$(scontrol show hostnames "$NODES" | head -1)
echo "=== job $J nodes $NODES master $MASTER $(date +%T) tree $(git -C $W log --oneline -1) ==="
module load aws-ofi-rccl/1.18.0_rocm6 2>/dev/null   # site RCCL plugin: NCCL_NET_PLUGIN, FI_MR_CACHE_MONITOR=kdreg2, FI_CXI_DISABLE_HOST_REGISTER=1, libfabric 1.23.1
NANOGPT=$S/nanoGPT
NCCLDBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
MOJOCCL_ENV="TORCH_MOJO_BACKEND_CCL=mojo TORCH_MOJO_BACKEND_TRACE=1 FI_CXI_DISABLE_EQ_HUGETLB=1 FI_CXI_DISABLE_CQ_HUGETLB=1"
RCCL_ENV="$NCCLDBG TORCH_MOJO_BACKEND_TRACE=1 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1"   # RCCL inside the mojo backend needs the VMM allocator on 2 MI300A nodes (see review/honest_table.md)
launch() { # launch CFG RDZV LOG ARGS...   -> one torchrun per node over the 2 nodes
  local cfg=$1 rdzv=$2 log=$3; shift 3
  case $cfg in
    warmup|stock) PY=$S/torch-rocm/bin/python; DEV=cuda; ENV="PYTHONPATH= $NCCLDBG";;
    mojo_nccl)    PY=$S/venv-cpu/bin/python; DEV=mojo; ENV="PYTHONPATH=$W $RCCL_ENV";;
    mojo_mojoccl) PY=$S/venv-cpu/bin/python; DEV=mojo; ENV="PYTHONPATH=$W $MOJOCCL_ENV";;
  esac
  srun --jobid=$J --overlap --nodes=2 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=192 --mem=0 --label bash -c "source $S/env.sh; cd $W; env $ENV OMP_NUM_THREADS=4 $PY -u -m torch.distributed.run --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29851 --rdzv-id=$rdzv tests/multinode/rank_bind.py --device $DEV $*" > $log 2>&1
}
COMMON0="--nanogpt-path $NANOGPT --data-dir $NANOGPT/data/shakespeare --log-interval 1 --seed 1337 --eval-interval 0"
# 0. binding check + library map of one mojoccl rank (the "what is on the run path" evidence)
launch mojo_mojoccl $J-check $LOGDIR/e2e_check_$J.log $COMMON0 --max-iters 3 --batch-size 12
echo "=== check exit $? : $(grep -c 'rank_bind\[numa\]' $LOGDIR/e2e_check_$J.log) numa-bound ranks, $(grep -c 'rank_bind\[compact\]' $LOGDIR/e2e_check_$J.log) compact; $(grep -m1 'collectives via' $LOGDIR/e2e_check_$J.log | sed 's/^[0-9]*: //')"
srun --jobid=$J --overlap --nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=192 --mem=0 bash -c "source $S/env.sh; cd $W; env PYTHONPATH=$W $MOJOCCL_ENV OMP_NUM_THREADS=4 $S/venv-cpu/bin/python -u -m torch.distributed.run --standalone --nproc-per-node=4 demo_scripts/nanogpt_ddp.py --device mojo $COMMON0 --max-iters 40 --batch-size 12 > /dev/null 2>&1 & sleep 45; for p in \$(pgrep -f 'demo_scripts/nanogpt_ddp.py' | head -1); do echo \"rank pid \$p:\"; grep -oE '/[^ ]+\.so[^ ]*' /proc/\$p/maps | sort -u; done; wait" > $LOGDIR/e2e_libmap_$J.log 2>&1
echo "=== libmap: $(grep -c '\.so' $LOGDIR/e2e_libmap_$J.log) shared objects mapped by a mojoccl rank; ROCm math libs: $(grep -cE 'rocblas|hipblas|miopen|rocfft|rocsolver|hipsparse|rccl' $LOGDIR/e2e_libmap_$J.log)"
# 1. the three-stack protocol: per batch, one discarded warm-up, then ABC CBA ABC CBA ABC
for BS in ${BATCHES:-48 32 12}; do
  COMMON="$COMMON0 --max-iters 30 --batch-size $BS"
  i=0
  for cfg in warmup stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl  mojo_mojoccl mojo_nccl stock  stock mojo_nccl mojo_mojoccl; do
    i=$((i+1)); LOG=$LOGDIR/e2e_three_stacks_${J}_bs${BS}_$(printf %02d $i)_$cfg.log
    launch $cfg $J-$BS-$i $LOG $COMMON
    echo "=== bs=$BS run $i cfg=$cfg exit $? $(grep -E ' step +30 ' $LOG | head -1 | sed 's/^[0-9]*: //') $(date +%T) ==="
  done
done
# 2. endurance: 1500 steps at batch 32, one run per stack
if [ "${ENDURANCE:-0}" = 1 ]; then for cfg in stock mojo_nccl mojo_mojoccl; do
  LOG=$LOGDIR/e2e_endurance_${J}_$cfg.log; launch $cfg $J-end-$cfg $LOG $COMMON0 --max-iters 1500 --batch-size 32
  echo "=== endurance cfg=$cfg exit $? $(grep -E ' step +1500 ' $LOG | head -1 | sed 's/^[0-9]*: //') $(date +%T) ==="
done; fi
echo "=== E2E DONE $(date +%T) ==="
