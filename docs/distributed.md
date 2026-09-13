# Distributed training (DDP) on the mojo device

The mojo device supports `torch.nn.parallel.DistributedDataParallel` through
a c10d backend named `"mojo"`, registered automatically by
`register_mojo_devices()`. Collectives on mojo tensors run over the NCCL C
API — **NCCL** on NVIDIA, **RCCL** on AMD — dlopened and called from Mojo
(`torch_mojo_backend/native/mojo/pg.mojo`); `torch_mojo_backend/distributed/
nccl.py` only resolves which library that is and the dtype/op constant maps —
no ctypes calls into NCCL/RCCL happen in Python any more. No CUDA/ROCm torch
build and no libcudart needed, in keeping with the project's "CPU-only torch
install, we bring the GPU stack" motto:

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
- **`find_unused_parameters=True` and `static_graph=True`: unverified, treat
  as unsupported.** That path wants a pinned-memory allocator for
  `reducer.cpp`'s `all_reduce_local_used_map`. The old reason this could not
  work — a Python-level PrivateUse1 backend cannot register one — no longer
  applies: `native/csrc/shim_runtime.cpp`'s `MojoHooks` is a real C++
  `PrivateUse1HooksInterface`, and it does register a
  `getPinnedMemoryAllocator()`. But that allocator is the ordinary
  (unpinned) CPU one — `isPinnedPtr()` always returns false — so the buffer
  `all_reduce_local_used_map` wants is still not actually pinned; this has
  not been re-tried against the native backend.
- `dist.all_reduce/broadcast/reduce/all_gather(_into_tensor, coalesced)/
  reduce_scatter(_tensor, coalesced)/all_to_all(_single)/gather/scatter/
  send/recv/barrier`, `batch_isend_irecv` and the object collectives all
  work; `ReduceOp` SUM/PROD/MIN/MAX/AVG map to NCCL/RCCL for float and int
  tensors (PREMUL_SUM does not); a bool tensor maps SUM/MAX to `ncclMax` and
  PRODUCT/MIN to `ncclMin` and rejects AVG, matching `ProcessGroupNCCL`. Mojo
  collectives (`TORCH_MOJO_BACKEND_CCL=mojo`) implement only allreduce,
  broadcast and all_gather — see "Mojo collectives" below.
- The MAX **CPU pseudo-device** (`mojo:{N-1}`, the last index —
  `torch.mojo.cpu()`) cannot take part in a collective at all: construction
  itself needs a real accelerator (communicators are created eagerly, see
  Design notes), and a collective call on a tensor living there raises
  `NotImplementedError`. Use a real GPU, or a plain `torch.device("cpu")`
  tensor (routed to the internal gloo group).
- Per-rank randomness: seed the device RNG per rank
  (`torch.mojo.manual_seed_all(seed + rank)`); weight init runs on the CPU
  RNG (`torch.manual_seed`) and DDP broadcasts rank 0's weights anyway.

## Design notes

The process group is split the way the rest of the native backend is
(`docs/native_backend.md`): a thin Python adapter
(`torch_mojo_backend/distributed/process_group.py`, `MojoProcessGroup`) over
a Mojo core (`torch_mojo_backend/native/mojo/pg.mojo`, `PG`) that owns the
communicators and does the actual library calls.

- **One communicator and one dedicated comm stream per device, in Mojo.**
  `PG` (`pg.mojo`) holds an `OwnedDLHandle` to whichever library `nccl.py`
  resolved (NCCL, RCCL, or mojoccl — all three share the NCCL C ABI) plus one
  `Comm` per device index: the communicator handle, the device's comm stream,
  and the raw vendor stream handle the library enqueues on
  (`tmb_pg_init_device`). Bring-up is the usual NCCL dance: rank 0 calls
  `ncclGetUniqueId` and publishes the 128 bytes through the c10d store
  (`MojoProcessGroup._ensure`, the same rendezvous torchrun already
  provides), every rank then calls `ncclCommInitRank` inside the device's
  context (`d[].ctx.push_context()` in `pg.mojo`) — that push, not a
  Python-side `hipSetDevice`/CUDA-context dance, is what binds the
  communicator to the right GPU now.
- **Communicators are created eagerly, once, at construction** —
  `MojoProcessGroup.__init__` calls `_ensure(device_module.current_device())`
  itself, collectively, while every rank is still inside
  `init_process_group` — rather than lazily on a rank's first collective.
  `pg.mojo` refuses to build one on the MAX CPU device
  (`tmb_pg_init_device`: "the mojo process group needs an accelerator
  device"), and the Python side refuses a collective on the MAX CPU
  *pseudo*-device the same way (`_is_cpu`: `NotImplementedError` for a
  tensor whose device equals `device_module.cpu()`) — use plain CPU tensors
  (routed to the internal gloo group) or a real GPU.
- **Every collective: sync-in, issue, fence.** `PG.sync_in` makes the comm
  stream wait for the caller's current stream before the library call is
  issued on it, so every producer kernel is ordered before the collective
  with no host blocking. The Python adapter resolves data pointers with
  plain `tensor.data_ptr()`; a non-contiguous input is copied dense on the
  comm stream (`_stage_in`), a non-contiguous output gets a fresh dense
  buffer there instead (`_stage_out` — never `.contiguous()` on the
  caller's output, so a partial write can't alias it); the call itself goes
  through a ctypes vtable wrapper resolved once per process group from
  `tmb_pg_vtable()` (`_Core`). Every NCCL group (`_group()`, a context
  manager) calls `ncclGroupEnd` on every exit path, including an exception,
  so a failed submission never leaves the communicator wedged inside an open
  group. Then — still with the comm stream current — a staged result is
  copied back and every touched buffer is `record_stream`d on that stream
  (`_finish`/`_record`), fencing its eventual release there: MAX does not
  fence frees across streams by itself.
- **Work objects are real device-typed futures, and `wait()` is what orders
  a stream** — the contract `ProcessGroupNCCL` implements, now reachable
  because the device guard (streams, events, current device) is real C++
  (`native/csrc/shim_runtime.cpp`), not a Python stub. `_work()` wraps a
  `torch.futures.Future(devices=[torch.device("mojo", index)])` whose result
  is set while the comm stream is current (`device_module.stream(...)`);
  calling `.wait()` on that future from any stream inserts a wait for that
  stream specifically. A synchronous collective (`async_op=False`) calls
  `wait()` internally, like any c10d backend; an async one hands the Work to
  the caller, who must `wait()` it before reading the result on their own
  stream — including a host read (`.cpu()`), a default-stream op, a side
  stream, or a DLPack export.
- **Memory safety does not depend on `wait()`.** The `record_stream` fence in
  `_finish` runs unconditionally, so a tensor dropped right after an async
  collective — before anyone calls `wait()` — is still not corrupted by a
  later allocation reusing its memory while the collective still writes it.
  That guarantee is independent of, and weaker than, the value-visibility one
  `wait()` gives: without `wait()` the memory is safe but the *value* is not
  guaranteed to be observed yet.
- **Known bug: `wait()` from a non-default stream does not reliably order
  that stream.** A consumer on the *same* stream the collective was issued
  from (the default stream, after `work.wait()`, including when
  `async_op=False` waits internally) sees the correct result every time; a
  consumer on a *different* `torch.Stream` that itself calls `work.wait()`
  intermittently reads a partially-applied buffer (the tail of a large
  tensor correct, the head still the pre-collective value) even after an
  additional `torch.accelerator.synchronize()`. Reproduced on both vendor
  NCCL and mojoccl, so the bug is in the Future/event plumbing shared by
  both, not in either collectives library. `tests/ddp_worker.py`'s
  `stream_ordering` mode (`stream_ordering.side_stream`) is the repro;
  it currently fails and is left failing on purpose rather than weakened,
  since passing it is the point.
- **The Python PG still replaces the whole process group** (torch ≥ 2.10
  behavior), so torch cannot compose `cpu:gloo` alongside it;
  `MojoProcessGroup` keeps its own private `ProcessGroupGloo` for CPU tensors
  (`_is_cpu`) and object collectives, and `_device_types` stays empty, which
  routes object collectives to CPU — exactly what the internal gloo serves.
- **The group is its own backend** (`supports_coalescing = True`,
  `_get_backend` returns `self`): torch's `batch_isend_irecv` and
  `_coalescing_manager` look a device's backend up with `_get_backend(device)`
  and check `supports_coalescing` on it, and the C++-side lookup would find
  nothing registered for `"mojo"` otherwise. This is what lets
  `batch_isend_irecv` group a rank's sends with its receives into one NCCL
  group instead of issuing them one at a time.
- **bool reduces the way `ProcessGroupNCCL` does**: SUM and MAX both become
  `ncclMax` (logical OR), PRODUCT and MIN both become `ncclMin` (logical
  AND), and AVG raises `TypeError` — bool has no meaningful average.
- Errors raised inside collectives print a full traceback to stderr before
  propagating (`_loud`): an exception escaping into the autograd engine on
  this backend can otherwise kill the process with no Python traceback.

## Cluster notes (SLURM, IB — NVIDIA)

- ptxas needs no configuration: the package defaults
  `MODULAR_NVPTX_COMPILER_PATH` to the CUDA 12.8 ptxas of the
  `nvidia-cuda-nvcc-cu12` wheel it depends on, whose cubins load on r570+
  drivers (`torch_mojo_backend/_ptxas.py`). Export the variable yourself
  only to use another ptxas.
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
- **But do NOT set that knob for a MULTI-NODE mojoccl run.** With
  `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` a two-node DDP step costs
  ~26x what it should. nanoGPT-124M, 2 nodes x 4 MI300A over cxi, batch 12,
  everything else identical (same region size, same build, same job):

  | | steady-state tok/s | ms/step |
  |---|---|---|
  | RCCL over the same NICs (reference), 20 steps | 1317.0k | 75 |
  | **knob unset, `MOJOCCL_REGION_MB=64`**, 20 steps x3 | **1150.7k / 1189.0k / 1186.3k** | **83 / 81 / 81** |
  | knob set, same region, 11 steps | 45.4k | 2160 |
  | knob set, default 256 MiB region, 20 steps | 35.2k | 2790 |

  Unset, mojoccl lands within 11-14% of RCCL end to end; set, it is 26-37x
  slower. The same A/B at 2 ranks per node, identical losses either way:
  20.3k against 537.8k tok/s at step 10, a 30x gap. Single-node runs are
  unaffected, which is why this hid for so long.

  **Why.** `py-spy dump --native` of a stalled 8-rank run caught it: the one
  rank that was not waiting had its autograd worker inside

      Engine::evaluate_function -> ~vector<at::Tensor> -> decref_pyobject
        -> TensorHolder tp_dealloc -> AsyncRT_DeviceBuffer_release
          -> M::Driver::DeviceBuffer::~DeviceBuffer -> libamdhip64 -> sched_yield

  i.e. backward blocked *freeing a device buffer*, spinning in the HIP
  runtime; the other seven were parked in the next step's blocking H2D
  (`_record_h2d_source`'s `event.synchronize()`), which cannot complete until
  their own device drains. The VMM allocator's release is a real unmap rather
  than a return to a cache, so it waits on the device -- and a multi-node
  collective keeps an item on that device for milliseconds while it waits for
  a remote peer. DDP frees intermediates continuously during backward, so
  every free lands on a busy device and backward serialises behind the
  network. Nothing in mojoccl fixes this; the release has to become
  stream-ordered in MAX.

  **What to do instead.** Leave the knob unset and shrink the communicator's
  region so four ranks still fit: `MOJOCCL_REGION_MB=64` gives a 192 MiB
  region per rank against 768 MiB at the default, and that is what the
  numbers above were taken with. At the default 256 MiB, four ranks per node
  without the knob leave too little to pin and every rank dies in
  `fi_mr_regattr` with `-FI_ENOMEM` (measured: node at 485 of 501 GB).
  Ruled out as explanations, each with its own run: the comm stream
  (`TORCH_MOJO_BACKEND_COMM_STREAM=0` is just as slow), the collective
  kernels' grid (capping them to 8 blocks changes nothing), the pipeline
  chunk count (forcing K=1 changes nothing), and the transport itself (its
  own blocking totals 42 ms of a 31 s run).

  The smaller region costs the collectives nothing, which is the thing to
  check before recommending it: the 8-rank allreduce at `MOJOCCL_REGION_MB=64`
  measures 793 us at 27 MiB (busbw 62.5 GB/s) and 10509 us at 512 MiB (89.4
  GB/s), against 10568 us at 512 MiB with the default 256 MiB region. The 27
  MiB figure is at parity with RCCL's 794 us; the 512 MiB one is still 1.49x
  RCCL's 7072 us, which is the separate transport-level gap analysed below and
  is unrelated to the allocator.

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
# NO MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM here: on two nodes that knob
# costs 26x (see the memory paragraph). A single-node job still wants it.
export MOJOCCL_REGION_MB=64   # what makes four ranks per node fit without it
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
srun --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 -- \
    uv run torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="$SLURM_JOB_ID" demo_scripts/nanogpt_ddp.py ...
```

## Mojo collectives (experimental): `TORCH_MOJO_BACKEND_CCL=mojo`

An in-repo replacement for NCCL/RCCL's intra-node collectives, written in Mojo
and exposed through **NCCL's own C ABI**: `torch_mojo_backend/distributed/mojoccl/`
builds `libmojoccl.so` on first use (into the eager kernels' `__mojocache__`,
same lock/atomic-rename machinery), and `nccl.py`'s `library_path()` resolves
to it instead of `libnccl.so.2`/`librccl.so.1` when `TORCH_MOJO_BACKEND_CCL=mojo`
— `pg.mojo` dlopens whichever path comes back, so neither it nor
`process_group.py` special-cases mojoccl; NCCL/RCCL stays the default. Design
and measurements: `docs/mojo_collectives_feasibility.md` (study) and
`docs/mojo_collectives_kernel_results.md` (kernels).

Scope, deliberately narrow — it is an experiment showing Mojo can write
NCCL-class collectives, not a general library:

- 2–8 ranks per node (`MAX_WORLD = 8`), up to 16 nodes, one process per GPU
  under torchrun; the inter-node hop is Mojo over libibverbs — see
  "Multi-node" below;
- `ncclAllReduce` (float32/float16/bfloat16/int32/int64, SUM and AVG),
  `ncclBroadcast` and `ncclAllGather` (every dtype, byte-granular);
  `ncclReduce`, `ncclReduceScatter`, `ncclSend`, `ncclRecv` return
  `ncclInvalidUsage`, so DDP works and anything needing them does not;
- the rendezvous is a TCP socket that `ncclGetUniqueId` opens on rank 0;
  the 128-byte `ncclUniqueId` carries its address, port and a random magic
  (NCCL's shape), and `ncclCommInitRank` runs three relayed all-gathers over
  it (host identity, IPC handle plus IB connection data, barrier);
- every rank owns one shared staging region (`MOJOCCL_REGION_MB`, default
  256 MiB, multiple of 4 KiB; larger requests are chunked). MAX's own
  allocations cannot be shared across processes (§5.6 of the study), which
  is why the kernels stage through this region; the push / local-reduce /
  pull design makes the staging free. The region is either a `cuMemAlloc`
  block shared with legacy IPC or, where NVSwitch multicast is available,
  VMM memory bound to a multicast object — see "NVLS" below;
- **a device deadline is loud.** A rank that stops answering makes its peers
  give up after `MOJOCCL_IB_TIMEOUT_S` (60 s) inside the kernel. The block
  that gave up records it twice: in its arena's error word, as it always
  did, and in the communicator's pinned *status page* — host memory the
  kernel stores into, so the host reads it with one load, no copy and no
  stream synchronize (`ncclCommGetAsyncError`'s read of the arena word needs
  a device-to-host copy, which blocks behind the very kernels it is asking
  about, so no collective could afford to call it — which is why a timed-out
  barrier used to leave no trace anybody read). That latch is what keeps a
  timed-out collective from turning into wrong data:
  - `ncclAllReduce` / `ncclBroadcast` / `ncclAllGather` return
    `ncclRemoteError` from the next call on; `pg.mojo`'s `check()` turns that
    into a Mojo `Error`, and the Python adapter surfaces it as a plain
    `RuntimeError` out of the collective — with a traceback, through `_loud` —
    so the rank fails instead of training on garbage;
  - the call that notices prints one line naming the rank, the collective,
    the arena, the generation, the phase, the block, and the peer whose flag
    never arrived with the value seen against the value wanted:
    `mojoccl: rank 1: DEVICE DEADLINE in the allreduce (arena 0, generation
    3, phase 0): block 0 waited 60.0 s for rank 0's flag, and saw 17 wanting
    24 -- ...`;
  - on the inter-node path `proxy_request` stops advancing the mailbox once
    the fault is latched, so the peer node sees no request rather than a
    shard nobody produced and reports its own deadline (`inter-node engine
    gave up after 60 s ...`); `proxy_wait` treats a latched fault like the
    abort word, so the waits already queued behind the failed one return at
    once instead of costing a deadline each.

  The FIRST failure is the one kept: the arena word is claimed with a
  compare-exchange and the status page is written only while it is clear, so
  what you read is the deadline that started the trouble, not the last of its
  consequences. The reporting call is one collective behind the kernel that
  failed — collectives are asynchronous, and catching it on the failing call
  itself would mean synchronizing the stream on every call.
- `ncclCommAbort` implements nccl.h's contract: it stops submissions (later
  collectives return `ncclInvalidUsage`), raises the pinned abort word (word 0
  of that status page) every device spin polls —
  the intra-node barriers, the NVLS barrier and the inter-node wait kernel
  all leave within a millisecond with their region's error word set, instead
  of running to that deadline — stops the progress thread, and then, once
  the local streams have gone idle (polled with `cuStreamQuery` under a 5 s
  bound, never a wait on a peer), releases the region, the peer mappings,
  the IB resources and the pinned memory. `ncclCommDestroy` on an aborted
  communicator is a no-op. If the device does not go idle inside that bound
  the memory is deliberately kept — freeing a region a kernel may still read
  faults the process — and abort says so on stderr.

Measured on 8×H100 SXM through the process group (wall over 20 launches,
median of 5, interleaved legs; NCCL 2.31.2 NVLS for comparison), **unicast
kernels only** — i.e. `MOJOCCL_NVLS=0`, which is what everything below the
48 MiB crossover runs anyway: 1 MiB 29 vs 31 µs, 9 MiB 70 vs 92, 27 MiB (the
DDP bucket) 164 vs 181, 168 MiB 988 vs 756, 512 MiB 2.99 vs 2.15 ms. The last
two rows are where the multicast path takes over — next subsection. AMD: the
same source cross-compiles for gfx942, and it now runs there — see the
subsection after that.

### AMD MI300A: every cross-link byte goes in the write direction

Measured on one Adastra node, 4 × MI300A (gfx942, 228 CUs), ROCm 6.4.3,
against RCCL 2.22.3 on the same node.

On an xGMI mesh the two directions are not equivalent. A micro-benchmark that
copies with the same 16-byte, four-in-flight loop these kernels use
(per-GPU GB/s, 168 MiB, all four GPUs active):

| | one link | three links at once |
|---|---|---|
| remote write | 91 | **233** |
| remote read | 90 | **93** |

A GPU-initiated remote load is limited per GPU, not per link, so three
concurrent inbound streams are worth barely more than one — and a ring of
simultaneous readers is worth *less* (56). Writes scale. RCCL knows this: its
P2P transport hard-wires `read = 0` on AMD (`rccl:src/graph/paths.cc:441`
only lets compCap 80 read), the sender stores into the receiver's buffer, and
RCCL reaches 236 GB/s of busbw at 512 MiB — the same ceiling.

So on AMD the allreduce's third phase is a second push instead of a pull:
each rank writes its reduced shard into every peer's gather slot, and then
copies those slots into the user's output *locally*, because a MAX-allocated
output cannot be IPC-mapped and a peer cannot write it directly. The
cross-link traffic is unchanged — `2(world-1)/world × bytes` per GPU, the
unicast minimum — only its direction is. That local copy is the price and it
is small: the region is `hipDeviceMallocUncached` yet reads out of it at full
HBM rate (1434 GB/s measured, against 1453 for a normal buffer). The
all-gather and the broadcast's gather half became pushes for the same reason.

Two other gfx942 details are copied from RCCL and matter as much as the
direction:

- **The barrier could not be made cheaper, and that is measured, not assumed.**
  A release store lowers to `buffer_wbl2 sc0 sc1` and an acquire load to
  `buffer_inv sc0 sc1`, both whole-cache operations, and the acquire sits
  inside the spin loop — so the barrier costs more the larger the grid (27 MiB
  measured 243 µs at 128 blocks and 465 at 1024). Three cheaper spellings were
  tried and all three broke *small* collectives only, which is the trap: a
  megabyte payload drains out of a cache on its own and a few hundred bytes do
  not, so "the big benchmark still passes" says nothing. RCCL's own cheap
  variant for gfx942 (`skip_fence`,
  `rccl:src/include/rccl_common.h:262-273`) is among them: it is sound for
  RCCL because its P2P buffers are uncached, and unsound here because the
  mapping a peer writes *through* comes from `hipIpcOpenMemHandle`, which does
  not carry the memory type. Details and the failure table are in
  `docs/mojo_collectives_kernel_results.md` §3 and §7.
- **The grid caps are not H100's**, and the response to the grid is not
  monotonic. See the sweeps in `docs/mojo_collectives_kernel_results.md`.

Everything above is behind `has_amd_gpu_accelerator()` at compile time, and
the sm_90a device code is byte-identical to the tree before this work (97
kernels, PTX compared with the mangling hash masked).

**One known flake, unresolved.** A *one-element* int64 allreduce at **2 ranks**
fails intermittently — 2 runs in 14 of `tests/ddp_worker.py collectives`;
every 4-rank run of every mode passed. It is the smallest collective in the
suite (one 8-byte store, one 8-byte load, one active thread) and the one-shot
kernel under it is unchanged by the MI300A work, so the suspect is the
barrier's cheapened acquire. Whether the pre-MI300A tree flakes the same way
was not established. See `docs/mojo_collectives_kernel_results.md` §7.

### NVLS: the large sizes go through the switch

The 168 and 512 MiB rows above are the **unicast** ceiling (~310 GB/s per
direction over NVSwitch): a push/reduce/pull allreduce moves
`2(world-1)/world × bytes` per GPU each way, and no schedule beats that
while every byte travels point to point. NCCL closes it with NVSwitch
multicast, and so does this library now.

A single-node communicator whose devices all report
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` builds its region as VMM memory
bound to a per-node multicast object, and routes allreduces of
`MOJOCCL_NVLS_MIN_MB` (48 MiB) or more through `multimem.ld_reduce` /
`multimem.st`: rank r pulls its 1/world slice through the switch, which sums
the `world` contributions and returns one value, and pushes the result back
into all `world` regions in one instruction. NVLink traffic falls to `bytes`
per GPU each way — 1.75× less at world 8 — paid for with ~1.5× the HBM
traffic, because the user's tensors are MAX-allocated and cannot be bound to
a multicast object, so every byte is staged in and out. That trade is why
the path has a size floor: 48 MiB is the measured crossover, sharp (4% the
wrong side at 40 MiB, 4% the right side at 48) and the same for fp32 and
bf16. int32/int64 stay unicast.

**Bring-up** (`vmm.mojo`, NCCL's `src/transport/nvls.cc` sequence). The
node's local rank 0 calls `cuMulticastCreate` and exports the object as a
POSIX file descriptor; the fd travels to its node-mates over an AF_UNIX
`SOCK_DGRAM` socket as an `SCM_RIGHTS` control message, which is what NCCL
does and, unlike `pidfd_getfd`, works whatever
`/proc/sys/kernel/yama/ptrace_scope` says. Every rank then
`cuMulticastAddDevice`s its own device, **barrier**, `cuMemCreate`s its
physical memory (GPUDirect-RDMA-capable, so an HCA can register it) and
`cuMulticastBindMem`s it at multicast offset 0 — so one
multicast address covers the node's eight distinct allocations — and maps it
**twice**: a multicast VA that only `multimem.*` may touch, and a plain VA
that the unicast kernels, the staging copies and the flag spin use. Peers are
imported with `cuMemImportFromShareableHandle` over the same sockets, because
legacy `cuIpcOpenMemHandle` cannot open VMM memory. Then **barrier**: no rank
issues a multimem instruction before every rank has mapped. The whole
sequence costs 150–230 ms once per communicator, dominated by
`cuMulticastBindMem` and the mappings.

The socket name is derived from the unique id's magic and the local rank
(`/tmp`, or `MOJOCCL_SOCKET_DIR`), so the rendezvous carries no extra round.
`tests/multinode/selftest/fd_exchange.mojo` runs that transport over ordinary
file descriptors with no GPU and no multicast hardware — the msghdr /
cmsghdr / sockaddr_un structs are laid out by hand over `UInt64` words
(`std.ffi` has no C-struct ABI) and a wrong offset does not fail loudly.

**Memory.** A multicast object's size must be a multiple of what
`cuMulticastGetGranularity` reports, and on H100 that is **512 MiB** for
`RECOMMENDED` against **2 MiB** for `MINIMUM`. NCCL uses RECOMMENDED; this
uses MINIMUM, so that `MOJOCCL_REGION_MB` keeps meaning what it says — at
RECOMMENDED the default region (`128 KiB + 2 × 256 MiB`) rounds up to a 1 GiB
allocation per rank and even a deliberately tiny test region costs 512 MiB,
while at MINIMUM the same region is 514 MiB. `MOJOCCL_NVLS_GRANULARITY=rec`
asks for NCCL's choice back.

The NVLS kernel stages **one** buffer, not two, so it uses the whole `2 × cap`
arena rather than a half: a 512 MiB allreduce is one launch on the default
256 MiB region. That is safe for the same reason a broadcast may use the whole
arena — the start barrier below.

**Fallback is a decision, not a recovery.** The capability travels in the
first bootstrap round, before anything is allocated: every rank contributes
"my device reports multicast and `MOJOCCL_NVLS` is not 0", rank 0 additionally
does a real `cuMulticastCreate` of a granularity-sized object and releases it
(88 µs, and it is where a broken fabric-manager setup shows up), and every
rank ANDs the whole column. One "no" — AMD, pre-Hopper, no NVSwitch, a rank
with `MOJOCCL_NVLS=0` — and the whole communicator builds the `cuMemAlloc`
region with legacy IPC exactly as before, with no half-built state to unwind.
A failure *after* that point is reported rather than papered over, and the
message names `MOJOCCL_NVLS=0`.

**The kernel** (`nvls_kernels.mojo`) is the prototype's split-grid schedule:
the low 25% of the blocks only drive the switch and the rest only drive HBM,
the message is cut into `clamp(bytes/4, 21 MiB, 86 MiB)` chunks, and chunk
c's reduction runs at the same time as chunk c+1's copy-in and chunk c-1's
copy-out. Two things about it are worth knowing before touching it:

* **The barrier is a full barrier**, not the block-index-matched one every
  other kernel in this library uses. There, block b only ever consumes bytes
  block b of a peer produced; here the reduce phase reads a contiguous slice
  that *every* block of every peer helped stage. The index-matched version
  passes every small case and fails from n = 65537 up. It is two levels: a
  device-scope arrival counter, then one
  `multimem.red.release.sys.global.add.u64` from the last block to arrive,
  which posts that GPU's arrival to all eight counters in one instruction; the
  wait is on the plain mapping, so it costs no fabric traffic.
* **A full barrier needs the whole grid resident** or it deadlocks. The grid
  is therefore `min(216, 2 × SM count)` blocks of 256 threads with
  `nvvm.minctasm=2`, and 216 was fitted on H100's 132 SMs.
* **It opens with a start barrier**, before a byte of staging is written, for
  the same reason every other kernel in `collectives_kernels.mojo` does: the
  arena is shared scratch, and a broadcast or an all-gather retiring just
  before this kernel has its peers still *reading* the bytes the copy-in is
  about to overwrite. One barrier per call, not per chunk.

Everything above is behind a compile-time sm_90+ gate (`multimem` exists
nowhere else and RCCL has no equivalent) and a runtime capability check; the
gfx942 and sm_90a cross-compiles both stay clean.

**Checking it.** `tests/ddp_worker.py stress` covers the path at 8 ranks and
256 MiB (every dtype × ragged size × SUM/AVG × in-place/out-of-place);
`tests/nvls_check.py` is the same shape of check at world 2 and 4, which
`ddp_worker` never runs, over sizes that straddle both the dispatch threshold
and the staging arena:

```bash
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=2 tests/nvls_check.py
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 tests/ddp_worker.py stress
# the numbers below: one leg per configuration, palindromic order
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 ar_bench_gpt2.py
MOJOCCL_NVLS=0 TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 ar_bench_gpt2.py
```

**Results**, 8×H100 SXM on one node (job 234314, `cl02s01dgx05`),
`ar_bench_gpt2.py` through the process group, medians in µs. Six legs in
palindromic order — NCCL, unicast, NVLS, NVLS, unicast, NCCL — so a clock or
thermal ramp cancels to first order; each column is the mean of its two legs:

| dtype | MiB | unicast | **NVLS** | NCCL | NVLS/unicast | NVLS/NCCL |
|---|---|---|---|---|---|---|
| fp32 | 9 | 70 | 70 | 94 | 1.00 | 0.74 |
| fp32 | 27 (DDP bucket) | 165 | 164 | 182 | **1.00** | 0.91 |
| fp32 | 168 (tail bucket) | 988 | **799** | 756 | **0.81** | 1.06 |
| fp32 | 512 | 2991 | **2218** | 2154 | **0.74** | 1.03 |
| bf16 | 9 | 70 | 70 | 91 | 1.00 | 0.77 |
| bf16 | 27 | 165 | 165 | 179 | **1.00** | 0.92 |
| bf16 | 168 | 994 | **798** | 745 | **0.80** | 1.07 |
| bf16 | 512 | 2988 | **2213** | 2126 | **0.74** | 1.04 |

At and below 27 MiB the two mojo columns run the same unicast kernel over the
same layout and read the same, which is the point of the crossover: the dispatch buys the tail bucket 19%
and the 512 MiB bucket 26% and costs the DDP bucket nothing. Against NCCL the
tail bucket goes from 1.31× to 1.06× and 512 MiB from 1.39× to 1.03×. 1 MiB
is below this bench's noise floor (per-leg medians 27–59 µs on every
configuration, NVLS or not) and is left out of the table.

A seventh leg measured `MOJOCCL_NVLS_GRANULARITY=rec` — NCCL's 512 MiB
multicast objects instead of the default 2 MiB ones — at 806/2213 fp32 and
798/2213 bf16 for the two large sizes: **the same within noise**, and the
2 MiB objects allocate 514 MiB per rank against 1 GiB. That granularity had
never been measured before (the prototype only ever used RECOMMENDED).

### Multi-node

**A stdlib clock bug on AMD, worked around here.** Mojo's
`global_perf_counter_ns()` on AMD computes `(s_memrealtime ticks * 1e9) //
1e8` in 64 bits: the product overflows 184 s after the GPU's counter started,
so on any node up for more than three minutes the "clock" is a saw-tooth of
period 184.47 s. Every device spin in this library bounds itself with
`now - t0 > timeout`, and a spin that straddles a wrap sees an enormous
unsigned difference and fires its 60 s deadline at once: the block records
the error word and returns, its node-mates wait a real 60 s for flags it
never publishes, and -- before the status-page latch above, which nothing
read the error word to notice -- the collective completed with garbage on
that node and the run carried on. Measured on 2x4 MI300A: about one 40 s `stress` run in
five corrupted a check, always a run 60-120 s longer than a clean one, and
the same event is what hung DDP runs. `device_now_ns` in
`collectives_kernels.mojo` reads the 100 MHz counter directly on AMD (with a
volatile intrinsic -- a side-effect-free read is hoisted out of the spin
loop and the deadline never fires); after it, 0 of 15 stress runs at 8 ranks
failed. NVIDIA's `globaltimer` is nanoseconds and unchanged (sm_90a device
code byte-identical). Report the stdlib bug upstream.


The inter-node hop is Mojo too: no vendor collective library anywhere.
An allreduce on a communicator spanning N nodes runs, per chunk: the
intra-node reduce-scatter (`reduce_scatter_stage`) leaves every rank its
shard of the node-reduced bucket in its own `stage_out`; each rank
RDMA-writes that shard to the counterpart rank (same `local_rank`) on every
other node and receives theirs into the `network` area of its region; a
small kernel sums the N−1 inbox shards into the shard; the intra-node
all-gather (`allgather_finish`) then pulls the globally reduced shards into
the user output. AVG's 1/world is applied by the reduce-scatter to each input
(NCCL's PreMulSum), so no node partial or inbox sum is ever an unscaled total
in a half dtype. Broadcast and all-gather use the same RDMA
path with a simpler schedule (root's node fans out to its counterparts, then
intra-node; node blocks exchanged, then placed by global rank) and stay
unpipelined — they run at DDP init, not in the step. Single-node
communicators keep the fused intra-node path and never touch IB.

**The three phases overlap.** The bucket is cut into K chunks and issued on
the one comm stream, with at most `PIPE_ARENAS` chunks alive:

```
RS(0) rel(0)  RS(1) rel(1)  RS(2) rel(2)  RS(3) rel(3)
              wait(0) add(0) AG(0)  RS(4) rel(4)
              wait(1) add(1) AG(1)  RS(5) rel(5)  …
```

so the proxy exchanges chunk k while the GPU reduce-scatters later chunks
and all-gathers earlier ones. `K = sqrt(bytes / (local_world × 640 KB))`,
capped at 16 by choice and raised from below by geometry when a chunk would
not fit — the square root is of the 16 µs an extra chunk costs (two more
launches and two more 8-way start barriers) against the 40–45 GB/s the RDMA
runs at. It gives K = 1 up to ~10 MiB, 2 at the 27 MiB DDP bucket, 5 at
168 MiB and 10 at 512 MiB, and keeps a chunk's shard above 1 MiB without a
second clause.

Two things make concurrent chunks safe, and neither is stream order across
ranks. **The staging arena is replicated.** A multi-node region is
`PIPE_ARENAS` complete arenas — each its own signal area and its own
`[stage_in | stage_out]` of `cap/PIPE_ARENAS` — followed by one cap-sized
network area; chunk k uses arena `k % PIPE_ARENAS`, so concurrent chunks
cannot collide in the push slots or in the shard, and that arena's own start
barrier is what orders chunk k+`PIPE_ARENAS` behind chunk k's pulls, the
invariant one arena already had. `collectives_kernels.mojo` is untouched:
the split kernels take a shifted base and a smaller cap, nothing more. The
staging total is `2 × cap` either way, so the region is the size it always
was. **The inbox is reused only against a credit** — see the transport
below. `PIPE_ARENAS` (4) and the derived `INBOX_SLOTS` (5) are source
constants in `mojoccl.mojo`, not environment variables: they are part of the
wire layout and every rank has to agree on them.

The chunk cap is now one arena, and the inbox slot group, rather than the
whole region: 64 MiB at 2 nodes and 36 MiB at 8 with the default 256 MiB
region, so the large sizes are chunked by geometry as well as by choice.
`tests/multinode/selftest/geometry_test.mojo` sweeps that arithmetic over
regions of 1 MiB–1 GiB, `local_world` 1–8 and 2–16 nodes.

**Two transports, one engine.** `internode.mojo` is the transport-neutral
progress engine (work ring, credits, arrival tally, flush, abort,
teardown); the six operations it needs -- post a payload, post an
immediate, post the flush read, poll completions, fill the bootstrap blob,
attach a peer -- are implemented twice, in `ibverbs.mojo` (InfiniBand) and
`libfabric.mojo` (HPE Slingshot through the `cxi` provider). `ib_setup`
picks one at run time: `MOJOCCL_NET=verbs|fabric` wins outright, otherwise
verbs if libibverbs opens and lists an ACTIVE InfiniBand port, else
libfabric if `libfabric.so.1` opens and `fi_getinfo` finds an FI_EP_RDM
provider with FI_RMA|FI_MSG|FI_HMEM, else the same clear error as before.
The engine's dispatch is one `st.net == NET_VERBS` branch per post and one
per poll batch -- loop-invariant and perfectly predicted -- so the verbs
path costs what it always did.

**Transport A, InfiniBand** (`torch_mojo_backend/distributed/mojoccl/{ibverbs,internode,
internode_kernels,bootstrap}.mojo`): libibverbs is dlopened; setup calls are
symbols, the data path (`ibv_post_send`/`post_recv`/`poll_cq`) is reached
through the `ibv_context_ops` table at the header's offsets, as NCCL's
`ibvwrap` does. One RC queue pair per remote node, attributes borrowed from
NCCL's `net_ib/connect.cc` (cited in the source); one `ibv_reg_mr` of the
whole region (`nvidia_peermem`; dmabuf is not implemented), with relaxed
ordering through `ibv_reg_mr_iova2`; each shard is one
`IBV_WR_RDMA_WRITE_WITH_IMM`, empty recv work requests exist only so the
immediate produces a completion, and a self-QP `IBV_WR_RDMA_READ` flush
orders the payload in GPU memory behind the completion that landed in host
memory. Every exchange is all-to-all — a rank whose shard is empty (7 of 8
ranks on DDP's 4-byte AVG allreduce) still posts a 16-byte placeholder — so
that an arrival tally of N−1 is what completes one.

**Transport B, Slingshot / libfabric** (`libfabric.mojo`): `libfabric.so.1`
is dlopened; `fi_getinfo`/`fi_freeinfo`/`fi_fabric`/`fi_version`/`fi_strerror`
are symbols and the whole data path (`fi_writemsg`, `fi_sendmsg`, `fi_recv`,
`fi_read`, `fi_cq_read`, `fi_mr_regattr`, `fi_ep_bind`, `fi_close`, …) is
`static inline` in the headers and is reached through the `fid_*` ops tables
(`ep->rma->writemsg`, `cq->ops->read`, `domain->mr->regattr`,
`fid->ops->bind`) exactly as the verbs path reaches `ibv_context_ops`. One
connectionless FI_EP_RDM endpoint per rank, one FI_CQ_FORMAT_DATA completion
queue bound for both directions, one FI_AV_TABLE address vector holding every
peer's `fi_getname` address plus this rank's own (the flush reads from
itself), and one `fi_mr_regattr` of the whole region with `iface =
FI_HMEM_ROCR`.

Four differences from the verbs semantics, forced by what the cxi provider
implements:

- **No write-with-immediate.** cxi's `fi_ops_rma.writedata` is
  `fi_no_rma_writedata` and `cxip_rma_writemsg` rejects `FI_REMOTE_CQ_DATA`
  (it is not in `CXIP_WRITEMSG_ALLOWED_FLAGS`); measured, `fi_writedata`
  returns `-FI_ENOSYS`. One shard is therefore an `fi_writemsg` of the
  payload followed by a **zero-length `fi_sendmsg`** whose 64-bit remote CQ
  data carries the immediate. cxi does support `FI_REMOTE_CQ_DATA` on the
  message path and reports `cq_data_size` 8.
- **Ordering between the two** is `FI_FENCE` on the first notification of an
  exchange, after all of that exchange's payload writes have been posted:
  fi_endpoint(3) defers a fenced operation until previous operations to that
  peer have completed, and cxi implements it as a hardware `C_CMD_CQ_FENCE`
  that drains the transmit command queue, so the exchange's remaining
  notifications are ordered by queue position alone — one fence per
  exchange, not one per peer. `FI_FENCE` must be named in `caps` or cxi
  returns `-FI_EINVAL`. A credit carries no fence: it announces nothing that
  was written.
- **No `FI_SOURCE`**, so a completion does not say who sent it: the CQ data
  is `(sender node index << 32) | immediate`, and the 32-bit immediate keeps
  exactly the meaning it has on the verbs path (bit 31 credit, bits 0..30
  the exchange counter), so the credit protocol and the pipeline are
  untouched.
- **`mr_mode` is FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT with no
  FI_MR_VIRT_ADDR**: RMA targets are OFFSETS into the peer's region, the key
  comes from `fi_mr_key` after the MR is `fi_mr_bind`-ed to the endpoint and
  `fi_mr_enable`-d, and the engine's region-relative offsets go on the wire
  unchanged. Both addressing modes are handled at run time
  (`FabricNet.virt_addr`), not assumed.

Receives are posted to the endpoint rather than per peer (an RDM endpoint has
one receive queue), `max(64, 32 × peers)` of them, and every one consumed is
reposted from the poll. The flush read stays: it is a short `fi_read` of this
rank's own region through its own address-vector entry, in the same role as
the verbs self-QP read. The fence argument probably already covers it and
MI300A's "device memory" is host-attached HBM anyway, but no cxi
documentation was found that promises the ordering, it costs ~3.6 µs, and
`MOJOCCL_FABRIC_FLUSH=0` turns it off for measurement.

Every struct offset, size and constant `libfabric.mojo` hard-codes is dumped
by `tests/multinode/selftest/fabric_abi.c` (gcc, against the installed
`<libfabric>/include`) and cross-checked by
`tests/multinode/selftest/fabric_abi.mojo`; 135 of them, and that self-test
is the only thing standing between a wrong offset and a garbage pointer
handed to the NIC.

**Flow control is explicit credits.** The second half of the network area is
carved once into `INBOX_SLOTS` fixed groups and exchange `e` lands in
`e % INBOX_SLOTS`; a peer may write that group again only once every
receiver has released it, published as a cumulative "consumed through e"
counter in a 4-byte `RDMA_WRITE_WITH_IMM` whose immediate carries a credit
bit — NCCL's head/tail pair in miniature
(`nccl:src/transport/net.cc`). This replaces a double-buffer-by-parity
argument that derived reuse safety from stream order ("peer B cannot post
e+2 before receiving my e+1 data, which I send only after my own add kernel
for e"), which holds for exactly one exchange in flight and breaks under the
pipelined schedule. The credit needs no kernel of its own: it rides on this
rank's next request as `credit_upto`, the number of consumer kernels already
enqueued ahead of that request, and when the proxy observes the request that
kernel has run, so every kernel enqueued before it has completed.
`INBOX_SLOTS = PIPE_ARENAS + 1` is what keeps the rank furthest behind from
ever waiting on a credit.

The GPU/network hand-off is a progress thread per rank driven through a
pinned, device-mapped mailbox: a one-thread kernel releases the exchange
into `MB_REQUEST`, the thread posts it, and a second one-thread kernel spins
on `MB_DONE` until the thread has seen the N−1 arrivals and flushed. Several
exchanges live between the two, so the thread is one non-blocking step
function (`ib_drive`, shared with the `MOJOCCL_IB_PROXY=0` callback and the
GPU-free self-tests) that retires exchanges in sequence order — RC ordering
is per queue pair, so arrivals are not ordered across peers — and pipelines
the flush read the same way. The thread spins only while something is
outstanding; idle, it yields and then sleeps in `MOJOCCL_IB_PROXY_IDLE_US`
steps, because a thread spinning between exchanges competed with the
host-bound Python dispatch thread and cost ~20% of end-to-end training
throughput at 16 ranks. A `cuLaunchHostFunc`/`hipLaunchHostFunc` stream
callback does the same job behind `MOJOCCL_IB_PROXY=0` and costs about
480 µs of fixed driver latency per exchange on this cluster (2.6× slower at
the DDP bucket), which is why the thread is the default. HCA choice: the
longest common `/sys/devices` prefix between the GPU's and the HCA's PCI
paths, ties by `local_rank`; only ACTIVE InfiniBand ports (the RoCE ports
are skipped). Addressing is LID-only, so one IB subnet.

Unpinned, the OS scheduler can still park the progress thread on the same
CPU as this rank's busy Python dispatch thread and leave it there — measured
end to end (job 234455, 16 ranks/2×8 H100, nanoGPT DDP) as a mode in one of
three ABBA runs: steps 4–11 at 45–46 ms (within 2% of NCCL's steady 44.6–44.9
ms), steps 12–35 at a sustained 51 ms (~13% slower), then back to 46 ms —
the scheduler leaving the progress thread on a shared core for a stretch of
the run, not throughout it. `MOJOCCL_IB_PROXY_CPU` unset therefore pins by
default:
`sched_getaffinity` reads this process's mask, and if it holds at least
`2 × local_world` CPUs (torchrun gives every rank of a node the same mask),
the thread goes on that mask's CPUs taken in descending order, indexed by
`local_rank` — the top `local_world` CPUs, distinct per rank, Python's
threads left wherever the scheduler already put them. Within that, a
candidate whose SMT sibling is itself another rank's pin is deprioritized in
favor of one that isn't (`topology/thread_siblings_list`, best-effort — read
failures just skip the check). A mask smaller than `2 × local_world` leaves
the thread unpinned rather than fight Python for a scarce core.
`MOJOCCL_IB_PROXY_CPU=<n>` overrides with an exact CPU;
`MOJOCCL_IB_PROXY_CPU=none` opts out of pinning entirely.
`MOJOCCL_IB_TRACE=1` prints the chosen CPU (or why none was chosen).

**Follow-up, unfixed: the sibling check does not catch the common case.**
Measured on Adastra (`taskset -pc` of every rank of a 2x4 job): each rank's
affinity mask is `0-47,96-143`, which is 48 physical cores with BOTH their
SMT threads -- `96+c` is the sibling of `c`. Taking the mask's CPUs in
descending order therefore pins the four progress threads to 143, 142, 141,
140, which are the siblings of cores 47, 46, 45, 44 -- cores the ranks' own
Python and autograd threads run on. `_smt_sibling_free` accepts them because
it only rejects a CPU whose sibling is ANOTHER RANK'S PIN, and 44-47 are not
pins. So on any node whose mask is a full-SMT range the policy reliably puts
every progress thread on a busy core's sibling, which is the placement it
exists to avoid; the ~20% this cost on H100 is the size of the effect.
A fix would prefer, among the mask's CPUs, ones whose sibling is not also in
the mask, and only then fall back to the descending rule. Not the cause of
any bug currently open.

| variable | default | controls |
|---|---|---|
| `MOJOCCL_SOCKET_IFNAME` | first UP non-loopback IPv4 interface with a default route (`bond0` here) | interface whose address rank 0 publishes in the unique id; one name, no lists |
| `MOJOCCL_BOOTSTRAP_TIMEOUT_S` | 120 | absolute deadline for the whole rendezvous. Every socket it opens is non-blocking and every wait is a `poll(2)` computed from the deadline (`connect` included, verified with `SO_ERROR`), so no syscall can outlive it; `SO_RCVTIMEO`/`SO_SNDTIMEO` stay on as a backstop |
| `MOJOCCL_IB_HCA` | affinity choice | exact HCA name to use instead (`mlx5_4`) |
| `MOJOCCL_IB_TIMEOUT_S` | 60 | how long a rank waits for a peer that stopped answering before latching an error — **every** wait: the inter-node exchange, its wait kernel, and the intra-node and NVLS barrier spins, which read it once per process |
| `MOJOCCL_IB_PROXY` | 1 | `0`: stream host callback instead of the progress thread |
| `MOJOCCL_IB_PROXY_IDLE_US` | 20 | sleep quantum of the idle progress thread (it spins only during an exchange) |
| `MOJOCCL_IB_PROXY_CPU` | unset: auto-pin (mask permitting), see above | an exact CPU to pin the progress thread to; `none` disables pinning |
| `MOJOCCL_IB_RELAXED_ORDERING` | 1 | `0`: plain `ibv_reg_mr` |
| `MOJOCCL_IB_TRACE` | 0 | `1`: one line per rank at destroy — backend and device, peers, slot groups, exchanges, credit stalls, mean µs posting / in flight / flushing; on libfabric a second line with the negotiated FI_HMEM interface, memory key, addressing mode, address length and receive depth |
| `MOJOCCL_NET` | unset: verbs if an ACTIVE IB port exists, else libfabric | `verbs` or `fabric`, forcing the transport |
| `MOJOCCL_LIBFABRIC` | unset: `libfabric.so.1`, then `/opt/cray/libfabric/2.2.0rc1/lib64/libfabric.so.1` | an exact `libfabric.so.1` to dlopen |
| `MOJOCCL_FABRIC_PROVIDER` | `cxi` | provider name asked of `fi_getinfo`; if it finds none, any provider satisfying the same hints is accepted |
| `MOJOCCL_FABRIC_DOMAIN` | affinity choice among the provider's domains (PCI proximity to the GPU, then `local_rank % n`) | exact domain to use instead (`cxi2`) — the libfabric analogue of `MOJOCCL_IB_HCA` |
| `MOJOCCL_FABRIC_HMEM` | `auto`: FI_HMEM_ROCR, then FI_HMEM_CUDA, then FI_HMEM_SYSTEM, first one the provider accepts | `system`, `rocr` or `cuda`, forcing the `fi_mr_attr.iface` the region registers under |
| `MOJOCCL_FABRIC_FLUSH` | 1 | `0`: skip the `fi_read` flush after an exchange (libfabric backend only) |
| `MOJOCCL_FABRIC_SETUP_RETRY_S` | 30 | how long `fab_setup` keeps retrying an endpoint bring-up that fails with `-FI_ENOMEM` (see the fragmentation note below); `0` fails on the first attempt |
| `MOJOCCL_REGION_MB` | 256 | staging size; single node `[signal \| stage_in cap \| stage_out cap]`, multi-node `PIPE_ARENAS` arenas of `cap/PIPE_ARENAS` halves plus a cap-sized network area. Must match on every rank — `ncclCommInitRank` checks it. On an NVLS region only the allocation is rounded up to the multicast granularity (2 MiB by default, 512 MiB under `MOJOCCL_NVLS_GRANULARITY=rec`); the halves keep the size asked for |
| `MOJOCCL_NVLS` | 1 | `0`: no multicast region and no NVLS kernel, on every rank of the communicator (it is ANDed across ranks) |
| `MOJOCCL_NVLS_MIN_MB` | 48 | single-node allreduces at or above this go through the switch; below it the unicast kernels keep the traffic. The default is the measured crossover. Must match on every rank — `ncclCommInitRank` checks it |
| `MOJOCCL_NVLS_GRANULARITY` | `min` | `rec`: size the multicast object with `CU_MULTICAST_GRANULARITY_RECOMMENDED` (NCCL's choice, 512 MiB objects on H100) instead of `MINIMUM` (2 MiB) |
| `MOJOCCL_SOCKET_DIR` | `/tmp` | where the node-local AF_UNIX sockets that carry the VMM/multicast file descriptors are bound |

Limits and failure modes: 8 ranks per node, 16 nodes; a multi-node
communicator on a machine with neither an ACTIVE InfiniBand port nor a
libfabric RMA provider fails `ncclCommInitRank` with a message naming both
and pointing at `MOJOCCL_NET`; a peer that stops
responding makes this rank's next collective fail with `ncclRemoteError` and
one printed line saying which barrier gave up (and is reported through
`ncclCommGetAsyncError` too) after `MOJOCCL_IB_TIMEOUT_S` (one variable for
every spin, intra-node and inter-node alike), or at once on `ncclCommAbort`; a stale unique id (tag `MOJOCCL2`) is rejected with
a clear message.

**Requirements.** Either rdma-core/libibverbs with active InfiniBand ports
reachable between every pair of nodes (here MLNX OFED 24.10) and GPUDirect
RDMA through `nvidia_peermem`, **or** a libfabric with an FI_EP_RDM provider
offering FI_RMA|FI_MSG|FI_HMEM and built with FI_HMEM support for the
accelerator (Adastra: `libfabric/2.2.0rc1`, provider `cxi`, four
`/dev/cxi[0-3]` NICs, ROCr HMEM). Either way: a routable interface for the
TCP bootstrap.

**Measured on Slingshot** (2 nodes x 4 MI300A, job 5393676, 512 MiB fp32
allreduce at 8 ranks, `ar_bench.py`, one size per process so the numbers are
attributable):

| | median | busbw | vs RCCL |
|---|---|---|---|
| RCCL over the same cxi NICs (aws-ofi-rccl 1.18.0) | 7 072 us | 132.8 GB/s | 1.00x |
| mojoccl | 10 568 us | 88.9 GB/s | 1.49x |
| mojoccl, `MOJOCCL_FABRIC_FLUSH=0` | 9 387 us | 100.1 GB/s | 1.33x |
| mojoccl, same size inside a 1/9/27/168/512 MiB sweep | 21 553 us | 43.6 GB/s | 3.05x |

Three things that says. **The sweep is 2x pessimistic**: the same allreduce
measured on its own is 10.6 ms against 21.6 ms as the last size of a sweep,
and the credit stalls that dominate the sweep's trace (80% of exchanges) are
2-6% (20-57 of 924) when the size runs alone -- so the credit window
(`INBOX_SLOTS`) is not what caps it and the sweep's degradation is upstream
of the transport. **The flush read costs 11%** here (1.18 ms of 10.6),
because on this fabric it is not the 1.9 us it is on InfiniBand: it queues
behind the exchange's own multi-megabyte writes on the same transmit command
queue, and the trace reads ~500 us. It stays on by default -- see
`fab_post_flush` for why the fence probably makes it unnecessary and why
"probably" is not enough to remove a memory-ordering guarantee -- but
`MOJOCCL_FABRIC_FLUSH=0` measured `correct=OK`. **The FI_FENCE is not the
cap**: exchanges do overlap despite it (the sum of per-exchange in-flight
times, 14 x ~1.9 ms, is 2.5x the 10.6 ms the allreduce takes), so it does not
serialise the pipeline. What is left is 1.33x over RCCL against a 5.2 ms wire
floor (14 chunks x 9.36 MB shard / 25 GB/s), where RCCL sits at 1.36x the
floor and mojoccl at 1.8x.

**`fi_enable` returning `-FI_ENOMEM`: the node is out of contiguous kernel
memory.** Several ranks of one node fail `ncclCommInitRank` with
`fi_enable failed, rc=-12`, the identical batch passes on other nodes or
later, and the provider says `cxil_map: write error` then "Failed to allocate
TX EQ resources, ret: -12". This was carried for a while as an unresolved
flake, described as contention with something else on the node, after
`cxi_service list` (0 of 2047 EQs in use), `RLIMIT_MEMLOCK` (unlimited), the
hugetlbfs knobs and an idle `free -g` (500 of 526 GB) had each been ruled out.

It is none of those. The kernel says what it is, in `dmesg` on the failing
node:

    python: page allocation failure: order:7,
            mode:0x40dc0(GFP_KERNEL|__GFP_COMP|__GFP_ZERO)
      cass_nta_alloc / cass_nta_init / cass_ac_alloc   [cxi_ss1]
      cxi_map / cxi_user_atu_map / ucxi_write          [cxi_user]

The cxi driver needs **512 KiB of physically contiguous kernel memory**
(order 7) for an address context's translation table, and the Mem-Info dump
printed with that failure showed no NUMA node had a single free block that
big: `0*64kB 0*128kB 0*256kB ...` on node 0, whose free total was 355 MB
against a watermark min of 353 MB.

Two measurements separate the causes:

* **Not the ranks racing.** Staggering a node's four ranks 2 s apart left 5
  runs in 8 failing, the same as unstaggered. `ib_bringup` at 8 simultaneous
  processes on the fabric path passes.
* **Not how much memory the job uses.** On a healthy node pair a nanoGPT
  2n x 4r run drives MemFree to ~14 GB of 501 — the same near-full state as
  the failing node — and initialises every time, because that node still has
  450-2350 free blocks at order >= 7 per NUMA node. The failing node had
  zero.

So it is the **node's physical fragmentation**, exposed by a workload that
legitimately uses ~97% of RAM because on an APU the GPU's memory IS system
memory. The failing node had 72 days of uptime; the pair that never failed
had been rebooted (with `drop_caches` and GPU resets) four hours earlier.

The difference is whether the fragmentation *persists*. Sampling
`/proc/buddyinfo` every 5 s through a nanoGPT run on the freshly booted pair,
with 6 GiB of ballast on top to push it further, the run does reach the same
state -- MemFree 9.3 GB, order >= 7 blocks down to 0-160 per NUMA node -- and
then comes all the way back to 29k-32k blocks at 505 GB free the moment it
exits, run after run. On a fresh node the kernel compacts what the job took;
on a node that has been up for months it does not, and `ncclCommInitRank`
finds nothing to map. That is also why the failure could not be reproduced on
demand once the fragmented node went out of allocation: eight nanoGPT runs
under that ballast took zero retries and had zero init failures.

What this repo does about it: `fab_setup` retries the endpoint bring-up while
the provider says `-FI_ENOMEM`, for `MOJOCCL_FABRIC_SETUP_RETRY_S` (30 s).
That is a mitigation, not a fix — fragmentation does not clear in 30 s — but
the pressure does ease as ranks free staging buffers, and ranks were measured
recovering on a later attempt. What actually fixes it is leaving the node
memory (`MOJOCCL_REGION_MB=64`, a smaller batch) or getting a node whose
memory is not fragmented.

**The intermittent 8-rank DDP stall: not reproduced, and what the engine now
says about it.** Roughly one nanoGPT DDP run in four at 2 nodes x 4 ranks was
seen to hang, never at 2n x 1r and never in the self-tests. One captured hang
(`repo` HEAD f0fe8ad, MI300A pair a1001/a1007) looked like this: all four
ranks of node 1 printed

    inter-node engine gave up after 60 s with no progress, at exchange 81
    (ring slot 80 of 512); engine at request 81, posted 81, done 77; ...
    credits sent 77, received 76 | blocked ms: ... arrival ~77600 ...

and all four ranks of **node 0 printed nothing at all**, with their engines
idle (`request == done`). So node 0's GPUs never reached `proxy_request(78)`:
they were stuck before it, in something with no deadline, while node 1 waited
77 s for their data. The transport on both sides was healthy.

Two things came out of chasing it:

* **Instrumentation, so the silent side speaks.** The give-up message only
  arms when `request_seq > done_seq` — i.e. only on the side that is
  *waiting* — which is why the node that was actually stuck said nothing.
  `ib_drive` now also watches for the opposite shape: nothing outstanding
  here, yet a peer has already sent data for a later exchange
  (`peer_seq_seen > request_seq`, tracked in `_consume_wc`). After the same
  `MOJOCCL_IB_TIMEOUT_S` it prints, once per episode, "THIS rank's GPU has
  not released exchange N". It is **diagnostic only** — it never touches the
  error word, because a rank whose host legitimately spends a minute between
  collectives is behind its peers for a good reason. Every exchange also
  carries which collective and which chunk it belongs to
  (`IbWork.op_kind/op_chunk/op_nchunks/op_numel`, set by `mojoccl.mojo`), so
  both messages now read "exchange 78 (allreduce chunk 3 of 14, 1703936
  elements)" instead of a bare number. Both were verified to fire and to name
  the right side by running with `MOJOCCL_IB_TIMEOUT_S=0.05`.

* **It did not reproduce on a healthy node pair.** 62 consecutive 40-step
  nanoGPT 2n x 4r runs on a1003/a1019 (24 before the instrumentation, 30
  after, and 8 more under a 6 GiB memory ballast), plus `ddp_worker.py collectives`/`ddp_parity`/`stress` at 8 ranks
  and the whole self-test suite on both nodes: no stall. The pair on which
  the hangs were seen is also the pair whose `fi_enable` failures are
  explained above by node memory fragmentation, and a stalled rank had
  previously been caught in MAX's `DeviceBuffer` release (the same place the
  VMM-allocator slowdown lives), so "the node's memory state" is the open
  suspect rather than anything found in the transport. **Unresolved**: the
  mechanism is not known, and no fix for it is claimed here.

**Running the two-node job.** `tests/multinode/run_two_node_checks.sbatch`
is a 16-rank (2 nodes × 8 GPU) SLURM job: `tests/ddp_worker.py`
(`collectives`/`ddp_parity`/`stress`; `abort` is single-node in
`tests/test_distributed.py`) under NCCL and under mojoccl, the
allreduce device-time bench (`ar_bench_gpt2.py`) in ABBA order, and a
40-step nanoGPT DDP run under both. `RUN_MOJO=0` keeps only the NCCL legs.
`tests/multinode/summarize.py <job log>` turns a log into the tables below.
`tests/multinode/selftest/` holds seven GPU-free self-tests — the bootstrap,
the RDMA transport, the pipelined transport with its credit protocol, the
region geometry, the `SCM_RIGHTS` fd transport the NVLS bring-up uses, the
socket deadlines and the libfabric ABI cross-check (the last four need no
NIC either, and the last two need no peers) — plus one that does need a GPU,
`fabric_hmem`, which registers a `driver.alloc_region` allocation with
FI_HMEM_ROCR and exchanges into it. The transport ones run against either
backend (`MOJOCCL_NET`) on a host with a NIC and no GPU — the login node
under InfiniBand, a compute node under Slingshot — with
`MOJOCCL_IB_PROXY=0`; they caught
six bugs before any GPU time was spent.

**Results**, 16 ranks on 2×8 H100, `ar_bench_gpt2.py` through the process
group, medians in µs. The pipeline against the commit before it, **in one
job on one node pair**, legs in ABBA order (before, after, after, before) so
a clock or fabric drift cancels to first order (job 234237,
`cl02s01dgx05` + `cl02s02dgx23`); K is the chunk count the rule above picks:

| MiB | K | fp32 before | fp32 after | | bf16 before | bf16 after | |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 107 | 107 | 1.00 | 100 | 113 | (noise, see below) |
| 9 | 1 | 187 | 187 | 1.00 | 186 | 187 | 1.00 |
| 27 (DDP bucket) | 2 | 342 | 282 | **0.82** | 345 | 274 | **0.80** |
| 168 (tail bucket) | 5 | 1606 | 1192 | **0.74** | 1592 | 1186 | **0.75** |
| 512 | 10 | 4762 | 3541 | **0.74** | 4739 | 3532 | **0.75** |

1 MiB is at the noise floor of this bench (per-leg medians 90–116 µs either
side, minima 90.3 vs 91.8 fp32 and 90.5 vs 91.4 bf16); K is 1 there and at
9 MiB, so those two sizes run the code they always did. NCCL 2.31.2 on the
same node pair (job 234235) reads 175 / 267 / 941 / 2359 µs fp32 and 168 /
263 / 950 / 2392 bf16 at 9 / 27 / 168 / 512 MiB, so the pipeline takes the
DDP bucket from 1.28× NCCL to 1.06× and the tail bucket from 1.71× to
1.27×. `MOJOCCL_IB_TRACE=1` over the whole bench: 0.15 µs posting, 2.8 µs
flushing, and 160–215 µs per exchange between release and retirement (which
now includes the time later chunks spend queued behind earlier ones);
4% of exchanges waited on a credit.

**Absolute numbers here are node-pair-specific and the table above is the
only fair comparison.** An earlier run of the same bench on a different pair
(job 234072) read 60 / 119 / 278 / 1532 / 4635 µs for the code in the
"before" column; two pairs measured since put that same commit at 107 / 187 /
342 / 1606 / 4762. NCCL is nearly identical on all of them (267 ± 2 µs at
the bucket), which is the tell: it pipelines its network hop, so its numbers
barely move with the fabric's per-message latency, and the unpipelined
hierarchical allreduce's numbers moved a lot. Fitting an exchange latency to
the 9 MiB point (where K is 1, and the shard's 1.1 MiB is 28 µs of wire at
the measured 40–45 GB/s) gives ~19 µs on the fast pair and ~87 µs on the
others. Hiding that latency is exactly what the pipeline does, and it is why
the gain is larger here than the exposed-transfer arithmetic alone predicts.

Splitting harder does not help: at `PIPE_SPLIT_UNIT = 320_000` (K of 3 / 8 /
14 instead of 2 / 5 / 10) the 27 MiB bucket is unchanged and 168 and 512 MiB
regress to 1291 and 3648 µs (job 234242, same ABBA design). The extra
exchange's latency is not fully hidden, so past the point where the network
is covered an extra chunk only buys launches.

`collectives`, `ddp_parity` and `stress` pass at 16 ranks under both
libraries (job 234229, `cl02s01dgx06` + `cl02s04dgx02`), including `stress`'s
200 rounds of interleaved 4-byte / 27 MiB / broadcast / allgather
collectives and its 256 MiB messages, which the pipeline cuts into 7 chunks.
The `MOJOCCL_IB_PROXY=0` fallback passes `collectives` too: it cannot
overlap (the callback blocks the stream) but the schedule and the credit
protocol are correct on it. nanoGPT-124M DDP at 16 ranks reaches the same
losses. End to end, six 40-step runs alternating NCCL and mojoccl on one
node pair (median step time over steps 3–40): on shared nodes (job 234185)
NCCL 49.7 ms vs mojoccl 51.4 ms with adjacent pairs at 1.035, 1.084 and
0.996, inside those nodes' noise; on `--exclusive` nodes (job 234455) NCCL is
tight at 44.6–44.9 ms and mojoccl reads 45.1, 45.7 and 50.1 ms — the
best-decile steps are within 1.7% of NCCL, and the slow run is a mode (steps
12–35 at a steady 51 ms, 45–46 before and after). Pinning the progress thread
by default (now the policy) did not remove it; binding every rank's whole
process to a compact eighth of the task's CPUs did (job 234658, exclusive
nodes, same six-run design, `tests/multinode/rank_bind.py` as the torchrun
entry): NCCL 46.30 ms vs mojoccl 46.37 ms pooled medians, ratio 1.001, pairs
0.968 / 1.005 / 1.030, and the remaining plateaus appear under both libraries
alike. Bind ranks to CPUs when measuring; the collective library is at parity.

What is left at the large sizes is the intra-node half, not the network: the
split reduce-scatter/all-gather pair is ~1004 µs at 168 MiB on one node
against NCCL's 751 µs NVLS multicast, and the pipeline's 1192 µs is 1.19×
that floor.

**The multi-node path deliberately stays unicast**, even on a cluster where
the single-node one takes the multicast route. Three reasons, in order of
weight:

1. *The chunking fights it.* NVLS wins only from 48 MiB up, and the pipeline
   cuts a 168 MiB bucket into K = 5 chunks of 33.6 MiB — below the crossover.
   Forcing chunks above it means K = 3, and by the model the chunk rule is
   built on (total ≈ node-local + network/K, which reproduces the measured
   1192 µs at K = 5 from a node-local 1004 µs) that trades ~125 µs of extra
   exposed network for the ~45 µs NVLS saves node-locally: 1272 µs against
   1192. At 8 nodes the geometry caps a chunk at 36 MiB, so the question does
   not even arise. Only 512 MiB at 2 nodes (K = 10, chunks of 51.2 MiB, just
   over the crossover) would gain, and only ~4%.
2. *It would need RDMA out of VMM memory*, which works but had to be
   proven. The inter-node write reads straight out of an arena's
   `stage_out`, so the staging has to be inside the registered MR.
   `ibv_reg_mr` on a `cuMemMap`'d VA first returned NULL on this cluster; the
   cause was the `cuMemCreate` prop, not the kind of memory: `nvidia_peermem`
   refuses (EFAULT) a chunk created without `allocFlags.gpuDirectRDMACapable`,
   which NCCL sets and `vmm.mojo` now sets too, after which all 12 HCAs
   register it (`docs/mojo_collectives_nvls_results.md` §4). There is still
   no dmabuf fallback (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0 on every
   device).
3. So, on the first reason alone, a multi-node communicator would pay
   150–230 ms of multicast bring-up for nothing, and it does not build a
   multicast region at all:
   multi-node allocation is byte for byte what it was, which the numbers
   confirm. Re-running this bench at 16 ranks after the NVLS work landed
   (job 234315, `cl02s01dgx24` + `cl02s02dgx23`, the same ABBA design) reads
   285 / 1185 / 3545 µs fp32 and 283 / 1184 / 3534 bf16 at 27 / 168 / 512 MiB
   against the 282 / 1192 / 3541 and 274 / 1186 / 3532 recorded above — within
   1% at every size, and `collectives`, `ddp_parity` and `stress` all pass at
   16 ranks.

The unmeasured half of point 1 is the NVLS *split* kernels themselves — a
multicast reduce-scatter and a multicast all-gather were never written, so
the ~45 µs above is estimated from the prototype's phase timings (copies
275 µs and switch 707 µs at 168 MiB) and not measured. If the chunk cap ever
rises above the crossover for the shapes that matter, this is the experiment
to run.
