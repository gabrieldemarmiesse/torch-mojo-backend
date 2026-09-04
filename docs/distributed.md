# Distributed training (DDP) on the mojo device

The mojo eager device supports `torch.nn.parallel.DistributedDataParallel`
through a c10d backend named `"mojo"`, registered automatically by
`register_mojo_devices()`. Collectives on mojo tensors run over the NCCL C
API — **NCCL** on NVIDIA, **RCCL** on AMD — driven with ctypes from
`torch_mojo_backend/distributed/nccl.py`, no CUDA/ROCm torch build and no
libcudart needed, in keeping with the project's "CPU-only torch install, we
bring the GPU stack" motto:

- NVIDIA: `libnccl.so.2` comes from the `nvidia-nccl-cu12` wheel (a
  dependency of this package).
- AMD: `librccl.so.1` comes from the ROCm install MAX itself already loads
  its HIP runtime from — the one at `$ROCM_PATH` or `/opt/rocm`, or a
  path in `TORCH_MOJO_BACKEND_RCCL_LIB`. Nothing extra to install: every
  ROCm ships RCCL, and taking it from the same install as the HIP runtime
  is what keeps one runtime (one device numbering) in the process. That
  assumes the CPU torch wheel: a ROCm torch wheel loads its own bundled HIP
  runtime next to MAX's, RCCL and the pointer-ownership query would bind to
  one while MAX's buffers belong to the other, and `register_mojo_devices()`
  warns about it (untested; use the CPU wheel).

Collectives on CPU tensors (object collectives, `barrier()`) are served by a
private gloo backend inside the same process group.

## Usage

One process per GPU, launched by torchrun:

```python
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()  # FIRST: pin this rank's GPU before CUDA/MAX initialize

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_mojo_backend import register_mojo_devices

register_mojo_devices()
dist.init_process_group(backend="mojo")

model = MyModel().to("mojo")
model = DDP(model, broadcast_buffers=False)
# training loop as usual; move batches to "mojo" yourself
```

```bash
# single node
uv run torchrun --standalone --nproc-per-node=8 train.py
# multi node (see demo_scripts/nanogpt_ddp.py for a SLURM recipe)
uv run torchrun --nnodes=$NNODES --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 train.py
```

`use_local_rank_gpu()` gives every rank exactly one visible GPU by slicing
the launcher's whole-allocation list by `LOCAL_RANK` — SLURM-style
`CUDA_VISIBLE_DEVICES=0,...,7` on NVIDIA, `ROCR_VISIBLE_DEVICES=0,...,3` (or
`HIP_VISIBLE_DEVICES`) on AMD — so `"mojo"` is always the right device and
each process binds one CUDA context / HIP device. A list already narrowed to
one entry (one srun task per GPU) is left alone. AMD has two levels, and only
one is narrowed: when `ROCR_VISIBLE_DEVICES` is present it takes the rank's
entry, and a `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` list next to it
(SLURM's gres plugin exports both by default; the HIP runtime reads either as
an index into the HSA-visible set) is rewritten to `0`. Call it before
anything touches the GPU runtime or enumerates MAX devices.

## What works, what to avoid

- `DDP(model)` with the defaults; keep `device_ids=None` (the default for a
  non-CUDA module) and move inputs to the device yourself.
- `broadcast_buffers=False` is recommended when buffers never change (e.g.
  causal masks) — it removes a per-step broadcast.
- **`find_unused_parameters=True` and `static_graph=True` are unsupported**:
  that path needs a pinned-memory allocator PyTorch does not let a
  Python-level PrivateUse1 backend register (`reducer.cpp`
  `all_reduce_local_used_map`).
- `dist.all_reduce/broadcast/all_gather(_into_tensor)/reduce_scatter_tensor/
  send/recv/barrier` and the object collectives all work; `ReduceOp`
  SUM/PROD/MIN/MAX/AVG map to NCCL/RCCL (PREMUL_SUM does not).
- Per-rank randomness: seed the device RNG per rank
  (`torch.mojo.manual_seed_all(seed + rank)`); weight init runs on the CPU
  RNG (`torch.manual_seed`) and DDP broadcasts rank 0's weights anyway.

## Design notes

- **A comm stream overlaps compute.** Collectives run on a dedicated side
  device stream per device (`mojo_device/device_streams.py`): it waits for
  the default stream so every producer kernel comes first, then the
  collective is enqueued. Every tensor a collective touches is fenced with
  `device_streams.record_use` (the backend's `record_stream` analog): its
  eventual stream-ordered free is ordered after the collective on the
  device, because MAX does not fence frees across streams by itself
  (measured; see the memory note in `mojo_device/device_streams.py`).
  `TORCH_MOJO_BACKEND_COMM_STREAM=0` pins collectives to the default stream
  instead (simplest ordering, zero overlap) — also the automatic path for
  collectives needing default-stream copies after the NCCL call. Both paths
  drain the host-side kernel-call queue first so producers are actually on
  a stream (`docs/kernel_call_queue.md`). One contract carried over from
  stock torch: `wait()` an async collective before reading its result —
  including before exporting it through DLPack.
- **Work objects** wrap already-completed `torch.futures.Future`s (no
  `devices=` — the PrivateUse1 device guard is a stub, and a device-typed
  future would do out-of-bounds bookkeeping for index ≥ 1), so `wait()` is
  a host no-op in both paths.
- **The default stream is ordered after the comm stream lazily**, at the
  first op that touches a buffer a collective read or wrote
  (`mojo_device/comm_fence.py`): the collective records those buffers as
  pending, and a hook in front of every eager op makes the default stream
  wait on the comm stream when it sees one. Host reads no op mediates —
  `torch.mojo.synchronize()`, a default-stream `synchronize()`/`query()`,
  DLPack export, the D2H copy — fence the same way. This is what lets the
  host run ahead: DDP's reducer never blocks on a bucket's future, so it
  keeps enqueuing while allreduces fly, and the fence lands in
  `finalize_backward` where it first reads a reduced bucket — after every
  backward kernel is already enqueued, so overlap is unchanged. Blocking
  the host on those futures instead cost ~2 ms of a 96 ms step, spent
  launching the bucket→grad copies, `clip_grad_norm_` and the optimizer
  against an idle GPU: nanoGPT 124M on 32 H100s (4 nodes, bf16, batch
  32×1024 per rank) went 10.89 → 11.10 M tok/s when it stopped doing so
  (paired A/B/B/A runs, medians of the 10-step windows), closing most of
  the gap to stock CUDA torch's 11.16. Stock `ProcessGroupNCCL` gets there with a
  device-typed future whose `wait()` makes the current stream wait; that
  needs a C++ DeviceGuardImpl for PrivateUse1 which torch does not provide
  and this backend cannot ship.
- **The Python PG replaces the whole process group** (torch ≥ 2.10 behavior),
  so torch cannot compose `cpu:gloo` alongside it; the internal gloo handles
  CPU tensors instead, and `_device_types` stays empty, which routes object
  collectives to CPU — exactly what the internal gloo serves.
- **Comm setup**: rank 0 calls `ncclGetUniqueId` and publishes the 128
  raw bytes through the c10d store (the same rendezvous torchrun already
  provides); every rank then calls `ncclCommInitRank`. Both libraries bind
  the communicator to per-thread runtime state — the current CUDA context
  on NVIDIA (a 4-call libcuda sequence, `nccl.set_current_cuda_device`),
  the current HIP device on AMD (`hipSetDevice`, `mojo_device/hip_peer.py`)
  — and DDP invokes the PG from the autograd thread, so this is re-asserted
  per thread. Which physical GPU a mojo tensor lives on is read off its
  pointer (`cuda_peer`/`hip_peer.device_ordinal`), never assumed from an
  ordinal. The library is picked once per process from the device api of
  the first mojo tensor a collective sees (`Device.api` is `"cuda"` or
  `"hip"`).
- Errors raised inside collectives print a full traceback to stderr before
  propagating (`_loud`): an exception escaping into the autograd engine on
  this backend can otherwise kill the process with no Python traceback.

## Cluster notes (SLURM, IB — NVIDIA)

- Export `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` on nodes
  whose driver is older than r580 (see the MAX GPU requirements) — in the
  sbatch script, so every rank gets it.
- `NCCL_DEBUG=WARN` (or `INFO` during bring-up) is the first knob for
  diagnosing init hangs; on multi-homed nodes set `NCCL_SOCKET_IFNAME` if
  NCCL's interface auto-detection picks a dead interface.
- First-run kernel builds: the JIT compile pool sizes itself per process
  from whole-node resources, so 8 cold ranks can oversubscribe a node.
  The kernel caches (`__mojocache__`, `~/.modular`) are shared over NFS, so
  a one-off single-process warmup run (or just letting step 1 be slow once)
  populates them for every node.

## Cluster notes (SLURM, Slingshot — AMD MI300A)

Verified on CINES's Adastra (4 × MI300A per node, ROCm 6.4.3, RCCL 2.22.3).

- **Install the CPU wheel of torch on AMD machines**
  (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`).
  The default PyPI wheel is the CUDA build, and it maps ~3 GB of NVIDIA
  libraries the process never uses; the HIP runtime walks every mapped
  shared object at each kernel load, so every first-use kernel load pays
  for them. Measured with nanoGPT 124M under DDP on 4 MI300A: the first
  training step took 14.7–18.5 s with torch 2.11+cu130 and 1.0 s with
  torch 2.11+cpu, and steady state was 6% faster too (680k vs 640k tok/s).
  `register_mojo_devices()` warns when it sees a CUDA torch build next to
  HIP devices.
- SLURM's GPU binding sets `ROCR_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`
  (`0,1,2,3` for one task with `--gpus-per-task=4`); `use_local_rank_gpu()`
  slices it. Launch one `torchrun` per node with `--nproc-per-node` equal to
  the node's GPU count, exactly as on NVIDIA.
- MAX dlopens `libamdhip64.so`/`libhsa-runtime64.so` from `$ROCM_PATH` or
  `/opt/rocm`; RCCL is taken from the same place. If the site's ROCm is not
  where MAX looks, set `ROCM_PATH` (or `module load rocm`) in the sbatch
  script so every rank gets it. A `GLIBCXX_3.4.30 not found` from
  `max._core` means the system libstdc++ is older than GCC 12: put a newer
  one on `LD_LIBRARY_PATH` (on Cray systems `/opt/cray/pe/gcc-libs`).
- Multi-node over Slingshot needs the site's libfabric RCCL plugin
  (`module load aws-ofi-rccl`, which puts `librccl-net.so` on the path);
  without it RCCL falls back to TCP sockets. `NCCL_DEBUG=INFO` shows
  `NET/OFI Selected Provider is cxi` when it took.
- RCCL's knob for MI300 is `NCCL_MIN_NCHANNELS`; the site recommends 42 for
  up to 4 APUs and 32 beyond. The single-node numbers above were taken
  with the defaults.
- Put the checkout, its `.venv` and `__mojocache__` on the fast parallel
  filesystem (scratch on Adastra, not work): a first-use kernel load makes
  the HIP runtime walk every mapped shared object, and with the venv on a
  slow filesystem each one costs tens of seconds per rank.
- **Memory on an APU.** MAX's default device allocator reserves ~115 GB
  (≈90% of the MI300A's 128 GB) per process at its first allocation, and on
  an APU that is the host's RAM: four ranks leave ~15 GB of a 512 GB node
  for everything else. Two consequences were measured: four ranks compiling
  kernels at once (and once a plain 4-rank run, on a node with less free
  memory) were OOM-killed, and the reservation evicts the page cache
  between import and the first training step, so the first step re-reads
  every kernel extension and every mapped library from Lustre — 12 to 18 s
  in about half the runs, ~1 s in the others. Modular's on-demand allocator
  fixes both: with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` a process
  holds 4 GB instead of 115, the first step is a steady 1.5 s and the
  steady state is unchanged. Its one defect with MAX 26.5 + ROCm 6.4.3 is a
  segfault at interpreter exit, inside ROCr's `Runtime::~Runtime` tearing
  down the VMM mappings from HIP's atexit handler (a pure-MAX script
  reproduces it; dropping every device reference first does not help), so
  a script that selects it must end with `os._exit(0)` once its own cleanup
  (`destroy_process_group`, checkpoint) is done — `demo_scripts/nanogpt_ddp.py`
  does. Without the knob, warm the kernel cache with a single process first
  (`--nproc-per-node=1`, a couple of steps) and expect the bimodal first
  step. Capping the HIP heap instead (`GPU_MAX_HEAP_SIZE=30`) is not an
  option: MAX's allocator becomes ~40x slower.

### Measured: nanoGPT 124M, bf16 autocast, batch 12×1024 per rank, 20 steps

Wall time of the training loop (after model, DDP and optimizer
construction), the first step included; three interleaved runs each, on
Adastra MI300A nodes, ours with the CPU torch wheel and
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`, stock torch 2.9.1+rocm6.4:

| | ours | stock ROCm torch |
|---|---|---|
| 1 node, 4 ranks: 20 steps | 3.1 / 3.2 / 3.3 s | 10.3 / 10.1 / 10.8 s |
| of which step 1 | ~1.5 s | ~8.7 s |
| steady state | 687–698k tok/s | 660k tok/s |
| whole process (imports to exit) | 12–13 s | 42–56 s |
| 2 nodes × 2 ranks over Slingshot: 30 steps | 4.1 / 4.6 s | 10.7 / 10.4 s |
| of which step 1 | 1.5 / 1.4 s | 8.1 / 7.6 s |
| steady state (steps 3–30) | 80 / 84 ms per step, an occasional 100–180 ms step | 79 ms per step, within 1 ms |

The two stacks print identical losses at every logged step. The 2-node
runs used the default GPU-Direct transport for both; per-step times come
from the tokens/s the demo prints for each step (its elapsed column has
0.1 s resolution).

- **Multi-node status.** On one node pair (a1003/a1004) our ranks failed in
  `ncclCommInitRank` with an RCCL internal error from the libfabric
  plugin's Connect step, after topology setup; the stock legs on the same
  pair worked. On the next pair (a1016/a1019) every configuration passed
  the full collective and DDP-parity checks across nodes — the default
  GPU-Direct path, `NCCL_NET_GDR_LEVEL=0` and `NCCL_NET=Socket` alike — so
  the failure was not reproduced and looks node-pair specific (that pair
  also held a job stuck in COMPLETING). If it recurs, `NCCL_NET_GDR_LEVEL=0`
  and `NCCL_NET=Socket` are the fallbacks, in that order. The 2-node
  numbers above are with 2 ranks per node because of the memory paragraph
  above (4 ranks per node work on the single node once the kernel cache is
  warm, and with the VMM allocator without caveat).

```bash
#!/bin/bash
#SBATCH --account=<account> --constraint=MI300 --nodes=2 --exclusive --time=1:00:00
module purge
module load aws-ofi-rccl   # multi-node only
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/cray/pe/gcc-libs:/opt/rocm/lib:${LD_LIBRARY_PATH}
export NCCL_DEBUG=WARN
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1   # APU: see the memory paragraph
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
srun --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 -- \
    uv run torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="$SLURM_JOB_ID" demo_scripts/nanogpt_ddp.py ...
```
