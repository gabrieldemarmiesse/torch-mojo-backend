#!/bin/bash
# mojoccl correctness suites on Adastra (CINES MI300A, Slingshot/libfabric cxi):
# the tests/ddp_worker.py and tests/fsdp_worker.py modes plus the multi-node
# probes, first across NNODES nodes x 4 APUs, then on one node x 4 APUs.
#
# The 2-node pass is the one that reaches the multi-node paths FSDP2 uses:
# the pipelined mapped all-gather (split and NIC-staged chunks, the in-place
# layout), the hierarchical reduce-scatter, the fabric flush endpoint.
# `tests/test_distributed.py` runs these workers on one node only.
#
# Usage, from a login node against an existing allocation of >= 2 nodes:
#   J=<jobid> ENV_SH=<env file> tests/multinode/run_worker_suites_adastra.sh
# or as a batch script (sbatch --nodes=2 --exclusive ...). Env:
#   ENV_SH   sourced on every node before anything runs: the ROCm/libstdc++
#            setup (ROCM_PATH, LD_LIBRARY_PATH with /opt/cray/pe/gcc-libs
#            and /opt/rocm/lib) and TORCH_MOJO_BACKEND_CACHE_DIR on scratch.
#   REPO     checkout to test (default: this script's checkout). Run it from
#            /lus/scratch, never /lus/work (agents_docs/distributed.md).
#   NNODES   nodes of the multi-node pass (default 2); NPROC ranks per node
#            (default 4). LOGDIR (default $REPO/logs/worker_suites).
#   PASSES   "multi single" (default), or one of them.
#
# One torchrun per node, every rank of a node under that node's per-GPU flocks
# (/tmp/gpu_lock_<g>.lock), each rank through vmm_exit_entry.py (MAX's VMM
# allocator, needed on the APU, segfaults at exit). The host libraries and libmojoccl build before the
# locks are taken; eager kernel specializations compile at first use, so warm
# TORCH_MOJO_BACKEND_CACHE_DIR first or they compile under the locks.
# Prints one PASS/FAIL line per suite; each suite's full output is in LOGDIR.
# Every requested suite runs; the exit status is nonzero if any of them failed
# (a worker assertion, a crash or a timeout), so sbatch and callers see it.
set -uo pipefail
J=${SLURM_JOB_ID:-${J:?set J=<jobid> or run under sbatch}}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
NNODES=${NNODES:-2}
NPROC=${NPROC:-4}
LOGDIR=${LOGDIR:-$REPO/logs/worker_suites}
PASSES=${PASSES:-multi single}
ENV_SH=${ENV_SH:-}
mkdir -p "$LOGDIR" "$LOGDIR/dcp"
NODES=$(scontrol show hostnames "$(squeue -h -j "$J" -o %N)")
MASTER=$(echo "$NODES" | head -1)
echo "=== job $J nodes $(echo $NODES) tree $(git -C "$REPO" log --oneline -1) ==="

# The per-node command: build outside the locks, then torchrun under them.
# SUITE_ENV: extra VAR=value words exported for this suite only.
node_cmd() {  # $1 = nnodes, $2 = rdzv id, rest = worker script + args
  local nn=$1 id=$2; shift 2
  local rdzv="--standalone"
  [ "$nn" -gt 1 ] && rdzv="--nnodes=$nn --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29871 --rdzv-id=$id"
  local locks=""
  for ((g = 0; g < NPROC; g++)); do locks+="flock /tmp/gpu_lock_$g.lock "; done
  cat <<EOF
${ENV_SH:+source $ENV_SH;} cd $REPO
export TORCH_MOJO_BACKEND_CCL=mojo MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1
export FI_CXI_DISABLE_EQ_HUGETLB=1 FI_CXI_DISABLE_CQ_HUGETLB=1 PYTHONUNBUFFERED=1
export ROCR_VISIBLE_DEVICES=\$(seq -s, 0 $((NPROC - 1))) ${SUITE_ENV:-}
ROCR_VISIBLE_DEVICES=0 uv run --no-sync python -c 'from torch_mojo_backend import native; native.build_shim(); native.build_backend(); from torch_mojo_backend.distributed.mojoccl_build import ensure_built; ensure_built()'
exec $locks uv run --no-sync torchrun $rdzv --nproc-per-node=$NPROC tests/multinode/vmm_exit_entry.py $*
EOF
}

suite() {  # $1 = pass, $2 = nnodes, $3 = name, rest = worker script + args
  local pass=$1 nn=$2 name=$3; shift 3
  local log="$LOGDIR/${pass}_${name}.log" id="ws_${J}_${pass}_${name}_$$"
  local where=(--nodes="$nn")
  [ "$nn" = 1 ] && where=(--nodes=1 --nodelist="$MASTER")
  if timeout --signal=TERM --kill-after=20s "${SUITE_TIMEOUT:-1500}" \
    srun --jobid="$J" --overlap "${where[@]}" --ntasks-per-node=1 \
      --gpus-per-task="$NPROC" --cpus-per-task=192 --mem=0 --label \
      bash -c "$(node_cmd "$nn" "$id" "$@")" > "$log" 2>&1; then
    echo "PASS $pass $name"
  else
    echo "FAIL $pass $name ($log)"
    FAILED+=("$pass/$name")
  fi
}
FAILED=()

for pass in $PASSES; do
  nn=$NNODES; [ "$pass" = single ] && nn=1
  suite "$pass" "$nn" collectives tests/ddp_worker.py collectives
  suite "$pass" "$nn" stress tests/ddp_worker.py stress
  suite "$pass" "$nn" ddp_parity tests/ddp_worker.py ddp_parity
  suite "$pass" "$nn" stream_ordering tests/ddp_worker.py stream_ordering
  suite "$pass" "$nn" abort tests/ddp_worker.py abort
  suite "$pass" "$nn" reduce_scatter tests/fsdp_worker.py reduce_scatter
  suite "$pass" "$nn" fsdp_collectives_stress tests/fsdp_worker.py fsdp_collectives_stress
  # The DCP round trip wants a temp directory every rank can see; /tmp is
  # node-local.
  SUITE_ENV="TMPDIR=$LOGDIR/dcp" suite "$pass" "$nn" fsdp_parity \
    tests/fsdp_worker.py parity
  if [ "$pass" = multi ]; then
    SUITE_ENV="N=1500" suite "$pass" "$nn" ring_pressure \
      tests/multinode/ring_pressure.py
    SUITE_ENV="MOJOCCL_REGION_MB=1" suite "$pass" "$nn" small_region \
      tests/multinode/small_region_probe.py
    SUITE_ENV="MOJOCCL_IB_TIMEOUT_S=3" suite "$pass" "$nn" deadline \
      tests/multinode/deadline_probe.py
    # Past the fused work ring's capacity: the split wait kernel's deadline.
    SUITE_ENV="MOJOCCL_REGION_MB=1 MOJOCCL_IB_TIMEOUT_S=3" suite "$pass" "$nn" \
      deadline_split tests/multinode/deadline_probe.py --size-mib 129
  fi
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== ${#FAILED[@]} suite(s) failed: ${FAILED[*]} ==="
  exit 1
fi
echo "=== all suites passed ==="
