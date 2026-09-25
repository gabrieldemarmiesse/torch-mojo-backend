# Distributed training on the mojo device

The mojo device supports `torch.nn.parallel.DistributedDataParallel` through
a c10d backend named `"mojo"`, registered automatically by
`register_mojo_devices()`. Collectives on mojo tensors run over the NCCL C
API — **NCCL** on NVIDIA, **RCCL** on AMD — dlopened and called from Mojo
(`torch_mojo_backend/mojo/tmb/backend/pg.mojo`); `torch_mojo_backend/distributed/
nccl.py` only resolves which library that is and the dtype/op constant maps —
no ctypes calls into NCCL/RCCL happen in Python any more. No CUDA/ROCm torch
build and no libcudart needed, in keeping with the project's "CPU-only torch
install, we bring the GPU stack" motto:

- NVIDIA: install `libnccl.so.2` with `pip install "nvidia-nccl-cu12>=2.27"`,
  or use an existing system library. Set `TORCH_MOJO_BACKEND_NCCL_LIB` to
  select a specific library path. The wheel is included in development
  dependencies only; installing this package does not directly require NCCL.
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
private gloo backend inside the same process group. This works on arm64 Macs
without NCCL/RCCL: when the current device is CPU or Metal, constructing the
group does not initialize the native GPU communicator.
Collectives on Metal tensors remain unsupported; the native GPU communicator
requires CUDA/HIP and its current unique-ID ABI requires x86-64.

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

## FSDP2

Use PyTorch's `fully_shard` directly with a mojo device mesh. FSDP1 is not
required. Create the optimizer **after** sharding, and apply `fully_shard`
to blocks before applying it to the root (shared embeddings/output weights
remain in the root group):

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# After use_local_rank_gpu(), register_mojo_devices(), and
# dist.init_process_group("mojo") as above:
mesh = init_device_mesh("mojo", (dist.get_world_size(),))
model = MyModel().to("mojo")
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
```

Set `TORCH_MOJO_BACKEND_CCL=mojo` to use MojoCCL. On one node its
reduce-scatter is a real reduce-scatter — one kernel per call, `(world-1)/
world × bytes` on the wire, nothing allocated and no stream synchronized —
and its all-gather is the unicast minimum; both are measured against NCCL at
FSDP2's sizes in "Mojo collectives" below. Across nodes the reduce-scatter
reduces node-locally and exchanges one destination shard per rank over RDMA,
and the all-gather sends one contribution per NIC; both are pipelined and
their grids are sized so the GEMMs keep their SMs (same section); both
schedules cross-compile for gfx942 but are measured on H100 only. Neither
modifies the input, except when the caller explicitly uses its own input
shard as the output (which NCCL also allows).

`demo_scripts/gpt2_fsdp2.py` exercises GPT-2 124M and XL without downloading
weights or a dataset. It uses the standard architecture, random initial
weights, and fixed rank-specific synthetic token batches; it checks finite
losses/gradient norms, parameter sharding, and loss reduction after AdamW
updates. Launch one process per GPU:

```bash
TORCH_MOJO_BACKEND_CCL=mojo uv run torchrun --standalone --nproc-per-node=2 \
    demo_scripts/gpt2_fsdp2.py --model gpt2
TORCH_MOJO_BACKEND_CCL=mojo uv run torchrun --standalone --nproc-per-node=2 \
    demo_scripts/gpt2_fsdp2.py --model gpt2-xl
```

Validated with PyTorch 2.11.0 and two H100 80GB GPUs (Slurm job 256073):
FP32, batch size 1 per rank, sequence length 64, dropout disabled, five
AdamW steps on the fixed synthetic batches above:

| Model | Parameters | First loss | Fifth loss |
|---|---:|---:|---:|
| GPT-2 | 124,439,808 | 11.028627 | 6.929632 |
| GPT-2 XL | 1,557,611,200 | 11.163113 | 4.367539 |

Both runs used MojoCCL and had finite gradient norms at every step.

`--dtype bfloat16` uses bf16 parameters for the transformer blocks, fp32
reductions, and autocast to keep normalization in fp32. The root retains
fp32 embedding/head parameters because the device's embedding backward
currently requires fp32 gradients. GPT-2 also passed five steps in this
configuration (loss 11.028791 → 6.922672), as did GPT-2 XL
(11.165338 → 4.369543). These
are training smoke tests, not performance measurements or pretrained-model
quality results. Multi-node FSDP2 on AMD is measured below ("GPT-2 XL
FSDP2 on two MI300A nodes").

The two-rank regression worker (`tests/fsdp_worker.py`) compares full
gradients and AdamW updates against CPU PyTorch on uneven layer shapes,
checks reduce-scatter dtypes/chunk boundaries/offset buffers, and saves and
reloads a sharded model/optimizer checkpoint before another reference-checked
update. The gradient/update and checkpoint checks also passed on two H100s
with a CPU-only `torch==2.11.0+cpu` wheel (Slurm job 256133): CUDA-enabled
torch is not required for FSDP2 on the mojo device.

For a throughput comparison, the same demo also supports stock CUDA and a
timed mode. Run each command on the same allocated GPUs, under the GPU
locks, using a PyTorch CUDA wheel compatible with the installed driver:

```bash
# Stock PyTorch CUDA + NCCL
uv run torchrun --standalone --nproc-per-node=2 demo_scripts/gpt2_fsdp2.py \
    --model gpt2-xl --device cuda --dtype bfloat16 --sequence-length 1024 \
    --benchmark --warmup 5 --steps 10 --windows 3 --output cuda.json
# Mojo device + vendor NCCL: use the same arguments with --device mojo
# and TORCH_MOJO_BACKEND_CCL=vendor. For MojoCCL, set CCL=mojo instead.
```

Timed windows include forward, backward, gradient clipping, AdamW, and
gradient clearing. They exclude initialization, compilation, warmup, and
loss reporting. Each window synchronizes the device before and after the
steps, and uses the slowest rank's elapsed wall time. Tokens/second is
aggregate across ranks: `world_size * batch_size * sequence_length * steps
/ elapsed_seconds`. Both devices use the same FSDP precision policy
(including the fp32 root), dropout-free model, and `foreach=False` AdamW.
This measures that explicit training configuration, without `torch.compile`
or an optimizer tuning search.
See [the two-H100 GPT-2 XL comparison](gpt2_fsdp2_throughput.md) for measured
CUDA, Mojo + NCCL, and Mojo + MojoCCL throughput.

### GPT-2 XL FSDP2 on two MI300A nodes

Measured on 2026-09-24 on Adastra (job 5447705, a1029 + a1070): 2 nodes x 4
MI300A (gfx942), Slingshot through libfabric cxi, 8 ranks, the demo's timed
mode with `--model gpt2-xl --dtype bfloat16 --sequence-length 1024
--batch-size 1 --benchmark --warmup 5 --steps 10 --windows 3`. Stock is torch
2.9.1+rocm6.4 with its RCCL and the site's `aws-ofi-rccl` plugin (`--device
cuda`); the mojo stacks use torch 2.11.0+cpu,
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` (without it four ranks per node
are OOM-killed) and either the system RCCL 2.22.3 or MojoCCL, with no
`MOJOCCL_*` variables. Every rank is NUMA-bound (`tests/multinode/rank_bind.py`).
Five interleaved rounds (ABC CBA ABC CBA ABC); median of the 15 windows per
stack:

| Stack | Tokens/s | vs stock | Window range |
|---|---:|---:|---:|
| Stock torch ROCm + RCCL | 25,365 | 100% | 23,622-26,026 |
| Torch Mojo + RCCL | 26,770 | 105.5% | 25,392-27,107 |
| Torch Mojo + MojoCCL | 26,420 | 104.2% | 24,062-26,908 |

Before the gfx942 GEMM and MojoCCL work that produced these numbers the same
configuration measured 25.6k / 23.5k / 19.7k tokens/s: the mojo device's bf16
GEMMs at 1024 rows (96 ms/step of kernels against hipBLASLt's 49, including a
12 ms scalar fallback for the odd K = 50257 of the tied head's input
gradient) and MojoCCL's unpipelined multi-node all-gather, 432-block gathers
and L2-invalidating completion polls, which also slowed the GEMMs running
beside them. Each window is 10 steps of ~0.3 s; a leg is noisy to about +-3%
on this shared fabric, so compare interleaved series only.

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
  collectives (`TORCH_MOJO_BACKEND_CCL=mojo`) implement allreduce,
  broadcast, all_gather and reduce_scatter (SUM/AVG) — see "Mojo collectives" below.
- Per-rank randomness: seed the device RNG per rank
  (`torch.mojo.manual_seed_all(seed + rank)`); weight init runs on the CPU
  RNG (`torch.manual_seed`) and DDP broadcasts rank 0's weights anyway.

## Design notes

The process group is split the way the rest of the native backend is
(`agents_docs/native_backend.md`): a thin Python adapter
(`torch_mojo_backend/distributed/process_group.py`, `MojoProcessGroup`) over
a Mojo core (`torch_mojo_backend/mojo/tmb/backend/pg.mojo`, `PG`) that owns the
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
  Every mojo device is a real accelerator, so construction always has one to
  build the communicator on; a plain `torch.device("cpu")` tensor is instead
  routed to the internal gloo group (`_is_cpu`).
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
  it is retained as a regression test. It passed with both vendor NCCL and
  MojoCCL on two H100s during the FSDP2 bring-up, but the historical
  intermittent failure has no confirmed root cause or fix.
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

- ptxas needs no configuration: the package sets
  `MODULAR_NVPTX_COMPILER_PATH` itself, to the assembler on the node that
  suits both the driver and the GPU — the `nvidia-cuda-nvcc-cu12` wheel's
  CUDA 12.8 one wherever it fits (`torch_mojo_backend/_ptxas.py`,
  agents_docs/native_backend.md "Which ptxas assembles the kernels").
  `torch-mojo-backend ptxas` prints the choice; export the variable yourself
  only to use another ptxas.
- `NCCL_DEBUG=WARN` (or `INFO` during bring-up) is the first knob for
  diagnosing init hangs; on multi-homed nodes set `NCCL_SOCKET_IFNAME` if
  NCCL's interface auto-detection picks a dead interface.
- First-run kernel builds: the JIT compile pool sizes itself per process
  from whole-node resources, so 8 cold ranks can oversubscribe a node.
  The kernel caches (`~/.cache/torch-mojo-backend`, `~/.modular`) are shared over NFS, so
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
- Put the checkout, its `.venv` and the kernel cache
  (`TORCH_MOJO_BACKEND_CACHE_DIR`; it defaults to `~/.cache`) on the fast
  parallel filesystem (scratch on Adastra, not work): a first-use kernel load makes
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
- **Earlier 124M multi-node VMM slowdown.** The measurements below
  describe an earlier nanoGPT-124M experiment. The later
  [GPT-2 XL verification](#gpt-2-xl-on-two-mi300a-nodes) runs successfully
  with VMM=1 and fused mojoccl; this is not a blanket ban for that path.
  In the earlier experiment, with
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

  **Workaround for that earlier case.** Leave the knob unset and shrink the
  communicator's
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
# Conservative recipe for the earlier 124M measurements above.
# The XL verification below uses VMM=1 with fused mojoccl.
unset MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
srun --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 -- \
    uv run torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="$SLURM_JOB_ID" demo_scripts/nanogpt_ddp.py ...
```

### GPT-2 XL on two MI300A nodes

Measured on September 15–16, 2026, with four gfx942 APUs per node (228 CUs
per APU), MAX 26.5 and the native backend using torch 2.11.0+cpu. Stock
uses torch 2.9.1+rocm6.4 and RCCL 2.22.3. The model is unchanged:
48 layers, 25 heads, width 1600, biases, bf16 autocast, batch 8 × 1024
per rank, eight ranks and 30 steps.

The comparison follows `tests/multinode/e2e_three_stacks.sbatch`: one
discarded warm-up per stack, then five interleaved ABC CBA ABC CBA ABC
rounds, with the original NUMA binder adapted to four ranks per node.
Each run contributes its mean printed tokens/s over steps 20–30; the
interval is the Student-t 95% confidence interval across rounds.

This clean-environment rerun used a[1007,1056], Adastra job 5417609,
at implementation commit `4152e03`.
All 18 runs completed with **no `MOJOCCL_*` variables**. Every log includes
rank 0's environment; all eight C ranks verified the live defaults before
timing. No compilation occurred during the series.

| Stack | Tokens/s ± 95% CI | Ratio vs A ± 95% CI | Step 1 |
|---|---:|---:|---:|
| A — Stock torch + RCCL | 129,207.27 ± 215.09 | 1.000000 | 9.48 s |
| B — Mojo backend + RCCL | 130,045.45 ± 570.36 | 1.006487 ± 0.004722 | 1.88 s |
| C — Mojo backend + mojoccl | 128,214.55 ± 100.50 | 0.992317 ± 0.001826 | 1.90 s |

Both native stacks used `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`;
stock left it unset. Both native ratios exceed the 0.95 target.

MI300A (`gfx942`) selects fused caps **8/16** and a **64 MiB** staging
capacity automatically. NVIDIA retains the H100-fitted **16/64** caps and
**256 MiB** staging capacity. The large-message threshold remains 128 MiB.
The library detects Slingshot/libfabric, chooses the nearest NIC, and runs
the progress thread automatically; no `MOJOCCL_*` exports are needed.

```bash
export TORCH_MOJO_BACKEND_CCL=mojo
export FI_CXI_DISABLE_EQ_HUGETLB=1 FI_CXI_DISABLE_CQ_HUGETLB=1
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1
```

The demo completes its own cleanup and calls `os._exit(0)`, avoiding the
MAX/ROCr interpreter-exit problem described above. These XL runs verify
VMM=1 with the fused path; they do not establish that the older 124M VMM
slowdown is fixed in every schedule or workload. Build the caches with
one process before launching four ranks per node, and keep compilation
outside GPU locks.

Full-model ABBA on a1070/a1071, job 5417296, improves from 125.1k to
128.4k tokens/s when changing caps 16/64 to 8/16 (+2.61%). Reducing the
small cap further to 4 loses 1.16%. Disabling the proxy in the historical
experiment passed the payload checks but lost 22.31% in full-model ABBA.
Those measurements are now the code defaults. The split schedule remains
the automatic fallback when a tiny staging region exceeds the fused work ring.

The fused kernel builds warning-free at 512 threads. Payload and deadline
probes pass for barrier sleep immediates 0/1/2/4/8 on all eight ranks;
every deadline arrives at 3.0 s within the 9 s bound. Reversed-order
collective timings find only a 0.89% best large-payload gain over sleep 2,
within the approximately 1% baseline reproduction spread, so the
production backoff stays 2. Raw traces verify the grids and pinned host
mailbox allocations; profiling perturbs the progress thread and is kept
separate from throughput measurements. The `ib_pipeline` self-test passes
over fabric/cxi on one node with four ranks; the full DDP runs separately
validate multinode operation. Small-region fallback and both fused/split
deadlines also pass.

The biased bf16 forward projections now use direct NT MFMA, adding bias
in the fp32 accumulator before its sole bf16 conversion. They do not use `Gemm16`,
whose reported gfx942 eight-element bf16 MMA lowering remains outside
this training path. On one a1007 APU, full-model ABBA reaches 18,904.5
tokens/s against stock's 16,890.9 (1.119×); native mean step time falls
from 641.15 to 433.35 ms, inferred from the printed throughput. Step 1
averages 1.75 s native and 8.80 s stock,
after model/DDP construction. Losses remain close, not bitwise identical.

## Mojo collectives (experimental): `TORCH_MOJO_BACKEND_CCL=mojo`

An in-repo replacement for NCCL/RCCL's intra-node collectives, written in Mojo
and exposed through **NCCL's own C ABI**: `torch_mojo_backend/mojo/tmb/ccl/`
(`entry.mojo` exports the `nccl*` functions; `distributed/mojoccl_build.py`
drives the build) builds `libmojoccl.so` on first use (into the native backend's kernel cache,
same lock/atomic-rename machinery), and `nccl.py`'s `library_path()` resolves
to it instead of `libnccl.so.2`/`librccl.so.1` when `TORCH_MOJO_BACKEND_CCL=mojo`
— `pg.mojo` dlopens whichever path comes back, so neither it nor
`process_group.py` special-cases mojoccl; NCCL/RCCL stays the default. Design
and measurements: `agents_docs/mojo_collectives_feasibility.md` (study) and
`agents_docs/mojo_collectives_kernel_results.md` (kernels).

**File layout.** `tmb/ccl/` mirrors NCCL master's `src/` tree, one file per
NCCL file, so someone porting from NCCL finds the code where NCCL keeps it.
Every file opens with `# Rewrite of: <NCCL file URL>` -- exactly one NCCL
file, or `none (mojoccl-only: ...). Closest: <url>`.
`entry.mojo` is only the export table (`src/libnccl.map`): an `@export` is
emitted only from the module being built, so it holds one C-ABI shim per
symbol forwarding to `collectives.mojo` / `group.mojo` / `init.mojo`. Two
deviations: `src/enqueue/enqueue.cc` is `enqueue.mojo` (a module named after
its directory is shadowed by the package), and the intra-node kernels are
NCCL's symmetric (LSA) kernels, `device/symmetric/*.mojo`, since mojoccl has
no ring/tree prims. The pre-split file names that older notes and journals
use map as follows:

| old | now |
|---|---|
| `entry.mojo` (a.k.a. `mojoccl.mojo`) | `entry.mojo` (exports), `init.mojo`, `enqueue.mojo`, `collectives.mojo`, `group.mojo`, `include/comm.mojo`, `nccl.mojo` |
| `collectives_kernels.mojo` | `device/symmetric/{all_reduce,all_gather,reduce_scatter,primitives,data_ops}.mojo`, `device/{broadcast,common}.mojo`, `include/device.mojo`, `include/nccl_device/lsa_barrier.mojo` |
| `nvls_kernels.mojo` | `device/all_reduce.mojo` |
| `vmm.mojo` | `transport/nvls.mojo`, `transport/multicast.mojo`, `include/transport.mojo`, `os/linux_ipcsocket.mojo` |
| `driver.mojo` | `misc/cudawrap.mojo`, `misc/strongstream.mojo`, `transport/p2p.mojo` |
| `bootstrap.mojo` | `bootstrap.mojo`, `misc/socket.mojo`, `os/linux.mojo`, `misc/utils.mojo` |
| `internode.mojo` | `transport/net.mojo`, `plugin/net.mojo`, `proxy.mojo` |
| `internode_kernels.mojo` | `device/symmetric/gin_scratch.mojo`, `include/nccl_device/gin/proxy/gin_proxy.mojo` |
| `internode_fused.mojo` | `device/symmetric/all_reduce_gin.mojo` |
| `ibverbs.mojo` | `misc/ibvwrap.mojo`, `include/{ibvwrap,ibvcore}.mojo`, `transport/net_ib/{init,connect,p2p}.mojo` |
| `libfabric.mojo` | `transport/net_ofi.mojo` |
| `netutil.mojo` | `misc/utils.mojo`, `include/plugin/nccl_net.mojo`, `graph/topo.mojo` |
| `reduce_scatter/{multinode,fused,stream}.mojo` | `device/symmetric/reduce_scatter_gin{,_fused,_stream}.mojo` |

Scope, deliberately narrow — it is an experiment showing Mojo can write
NCCL-class collectives, not a general library:

- 2–8 ranks per node (`MAX_WORLD = 8`), up to 16 nodes, one process per GPU
  under torchrun; the inter-node hop is Mojo over libibverbs — see
  "Multi-node" below;
- `ncclAllReduce` (float32/float16/bfloat16/int32/int64, SUM and AVG),
  `ncclBroadcast` and `ncclAllGather` (every dtype, byte-granular);
  `ncclReduceScatter` (the same dtypes; SUM, and floating-point AVG).
  `ncclReduce`, `ncclSend`, `ncclRecv` return
  `ncclInvalidUsage`, so DDP works and anything needing them does not;
- the rendezvous is a TCP socket that `ncclGetUniqueId` opens on rank 0;
  the 128-byte `ncclUniqueId` carries its address, port and a random magic
  (NCCL's shape), and `ncclCommInitRank` runs three relayed all-gathers over
  it (host identity, IPC handle plus IB connection data, barrier);
- every rank owns one shared staging region (`MOJOCCL_REGION_MB`, default
  64 MiB on gfx942, 256 MiB elsewhere; multiple of 4 KiB; larger requests are chunked). MAX's own
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

### Reduce-scatter: push, then reduce into the caller's output

FSDP2 asks for exactly two collectives and nothing else — a reduce-scatter of
the gradients and an all-gather of the parameters, 146 calls per GPT-2 XL
step at two ranks — so the reduce-scatter is a real one on a single node,
not the allreduce transport in a costume:

- **PUSH** rank r reads chunk s of its own input and writes it into peer s's
  staging slot. That write is the wire transfer, as in the allreduce's
  phase 1;
- **REDUCE** every rank sums its own chunk, straight out of user memory,
  with the `world-1` pushed slots, and writes the result into the caller's
  output. AVG scales each contribution as it enters the fp32 accumulator,
  never the finished sum (NCCL's PreMulSum): two fp32 ranks contributing
  2**127 average to 2**127, not to inf.

`(world-1)/world × bytes` per GPU on the wire, the unicast minimum; nothing
allocated, no stream synchronized, and one kernel per call at every size
FSDP2 asks for — the push slots are `world-1` compacted slots that may use
the whole `2 × MOJOCCL_REGION_MB` arena (512 MiB at the default region), for
the same reason the NVLS allreduce may: the start barrier is what orders a
generation's writes after every peer's previous reads.
`reduce_scatter_max_count` is that bound and a larger message is chunked
against it. In place is safe in NCCL's sense (`recvbuff == sendbuff +
rank*count`): the push never reads chunk `rank`, and in the reduce each
thread writes only the elements it just read. Any alignment is accepted —
16-byte vectors where the pointers and the input stride allow them, and the
same W-element groups moved one element at a time where they do not, so two
ranks may disagree about it and stay block-matched.

Every cross-link byte goes in the **write** direction, so unlike the
allreduce and the all-gather this schedule needs no separate AMD variant.

On one node, at the FSDP2 shapes and against NCCL on the same GPUs (H100
SXM, `--core`, CUPTI device time, median over ranks, unlocked clocks, us):

| single node, per rank | NCCL | mojoccl | ratio |
|---|---:|---:|---:|
| 2 ranks, XL block fp32 (61.4 MB) | 276 | 262 | **0.95** |
| 2 ranks, XL root fp32 (164 MB) | 677 | 688 | **1.02** |
| 8 ranks, XL block fp32 (15.4 MB) | 345 | 370 | **1.07** |
| 8 ranks, XL root fp32 (41 MB) | 864 | 1000 | 1.16 |
| 8 ranks, 357x789 at an odd offset (1.1 MB) | 51 | 104 | 2.03 |
| 8 ranks, 256 KB | 19 | 15 | **0.80** |

The 8-rank root is this same push-then-reduce shape one node down: 7 shard
pushes of 41 MB at the fabric's rate, then an 8-way HBM reduce that only
starts once every rank's push is in. At 2 ranks there is one push and the
reduce is two streams, so there is nothing to hide and it is at parity.

Across nodes, each local rank reduces the chunks destined for that local
rank on every node, using the bootstrap topology table to address input
chunks. Each remote node receives only its own partial through the existing
RDMA exchange; an add kernel combines incoming partials with the local one
directly into the caller's output. On two nodes each rank sends `count`
elements, with no all-gather phase, temporary allocation, or host stream
synchronization. Larger messages use the existing staging arenas and inbox
credits to pipeline chunks. AVG scales each input before the node-local sum.
NVIDIA fp32 runs the pipeline in one persistent kernel: local reduction,
mailbox release, bounded completion wait, output sum, and credit return.
Two such kernels exist. `reduce_scatter_gin_stream.mojo` takes every call of at least
`PIPE_SPLIT_UNIT` bytes per rank (see "Streaming reduce-scatter" below);
`reduce_scatter_gin_fused.mojo` runs everything smaller, two chunks per call so only the
second exchange is exposed (`RS_FUSED_TARGET_CHUNKS`), and stays the
fallback for every geometry the streaming kernel declines. Other dtypes and
targets use separate kernels for these phases (`reduce_scatter_gin.mojo`). Calls
whose chunk count exceeds the existing work ring also use the split
schedule.

#### Streaming reduce-scatter

`reduce_scatter_gin_fused.mojo` runs a chunk as push, 8-way barrier, reduce: no rank starts
reducing before every rank has finished pushing the whole chunk, so the
NVLink push and the HBM-bound reduce never overlap, the barrier exposes the
slowest rank's whole push, and the chunk's RDMA exchange only starts once
all of that is done. It also spends four grid barriers per chunk. NCCL's
ring has none of that: its unit of "has data arrived" is a 1 MiB slice
checked by four threads of the block against the neighbour's step counter,
with an eight-deep credit pipeline and no grid barrier anywhere
(nccl:src/device/prims_simple.h `waitPeer`/`postPeer`,
src/device/reduce_scatter.h; at these sizes NCCL 2.28 picks RING/SIMPLE,
16 CTAs of 544 threads = 512 workers plus one post warp).

`reduce_scatter_gin_stream.mojo` keeps the hierarchical schedule -- its bytes are already
NCCL's: `(local_world-1) * nnodes` shard pushes on NVLink and one shard on
the wire per rank, against a 16-rank ring's 14/15 NVLink and 1/15 network
hops, which is 287 MB and 20.5 MB for the XL root at 16 ranks either way --
and replaces every rendezvous in it with a one-directional flag:

- each block owns a contiguous range of the chunk and walks it in pieces of
  `RS_STREAM_UNROLL * RS_STREAM_SLICE_UNROLLS` 16-byte vectors per thread.
  It pushes piece `j`, publishes `DATA[peer][block][me] = ordinal(chunk, j)`
  and waits for every peer's `DATA[me][block][peer]` to reach
  `ordinal(chunk, j - RS_STREAM_DEPTH)` before reducing that piece;
- `FREE[peer][block][me]`, published once this block has reduced chunk `k`,
  is the credit that lets peers overwrite the arena at chunk `k + narenas`.
  In steady state it is already there -- it is `narenas` chunks of slack --
  so nothing waits for it;
- a rank-local arrival counter per chunk releases that chunk's RDMA exchange
  and returns the inbox credit. Blocks arrive and keep going; only the last
  one stores into the mailbox.

Both counters are generation-tagged, monotone, never reset and compared with
`>=`, like the block barrier's flags; the host reserves
`ceil(nchunks * pieces / PHASES_PER_GEN)` generations per call so a later
call's values cannot collide with them. The block-matched invariant is
unchanged: block `b` of every rank derives the same range and the same
pieces from `(chunk, grid)` alone, so it still consumes only what block `b`
of a peer produced, whatever the two ranks decide about 16-byte alignment.
In-place, `count == 0`, arbitrary alignment, bounded spins, the error word
and the status page work exactly as in the fused kernel.

**The arena layout is the same for every chunk of a call, and that is what
makes the per-block credit sound.** The slot stride and the partition come
from the full `chunk_elems`; a short last chunk leaves the tail of the
layout unwritten instead of re-cutting it. Deriving either from a chunk's
own `cnt` -- which is what the kernel first did -- moves every slot base and
every block boundary for the last chunk, so block `b`'s write lands on bytes
block `b-1` of a peer is still reducing while `b` has waited only for the
peer's block `b`. `reduce_scatter_gin_fused.mojo` gets away with a per-chunk stride because
its rank-local grid barrier makes one block's cross-rank sync transitively
cover every block of the peer; this kernel gave that up, so it owes the
invariant instead. Reachable at ordinary sizes -- the default 256 MiB region
caps a chunk at 2,097,152 fp32, so 36 MiB per rank is five chunks over four
arenas with a half-size last one -- and `tests/test_mojoccl_reduce_scatter_layout.py`
pins it in source, because the failure is a race: 1000 mojo-leg collectives
per rank at 16 ranks over 5, 9 and 35 chunks, with half-size and
single-element last chunks, at both region sizes, did not reproduce it on
the broken kernel. What keeps it shut is that every block waits on the same
`_RS_STREAM_DONE` word once per chunk and must arrive before the exchange is
released, which holds the blocks of a rank inside one chunk of each other.
Nothing promises that.

Neither benchmark shape below can reach the bug -- both plan exactly four
chunks over four arenas, so no arena is reused -- and for the block size the
two layouts are bit-identical (four chunks of exactly 480,000 elements),
while the root's differ only in a 48-byte slot stride on its last chunk.
The fixed kernel nevertheless measures about 3% slower on the root in both
leg orders (1.269/1.286 -> 1.316/1.326 against NCCL). That is either this
unlocked-clock box's build-to-build spread, which the earlier variants put
at +-4% on the mojo leg, or those 48 bytes moving the RDMA source's page
offset; it is not attributed. Rounding the slot stride to 4 KiB instead of
16 B would make every slot base page-aligned for every message and settle
the question, and it needs the arena and inbox bounds widened to match.

One thing did not survive dropping the grid barriers. With no barrier left,
every block polled the pinned mailbox for the exchange itself, and 32
threads reading host memory over the link the NIC is moving the shard on
cost far more than the barrier ever did: that one line, changed back on an
otherwise final tree, takes the isolated root fp32 reduce-scatter from
1273 us to 2195 (NCCL 982-1007). Block 0 now polls and republishes what it
saw into a device word the other blocks spin on out of L2 -- the same
transitive acquire of the NIC's writes that the grid barrier used to give
them.

What the streaming bought, measured on 2x8 H100 SXM over InfiniBand
(job 259332, 16 ranks, `--core`, CUPTI device time of the comm stream,
median over ranks, vendor/mojo ABBA and BAAB in one process,
**unlocked clocks** -- `nvidia-smi -lgc` is not permitted on these nodes):

| isolated, us | NCCL | fused | stream |
|---|---:|---:|---:|
| reduce-scatter fp32 SUM, XL block (7.68 MB/rank) | 463 | 645 (1.39x) | 512-519 (**1.10-1.12x**) |
| reduce-scatter fp32 AVG, XL block | 465 | 640 (1.38x) | 502-507 (**1.08-1.09x**) |
| reduce-scatter fp32 SUM, XL root (20.5 MB/rank) | 990-1007 | 1470 (1.48x) | 1273-1281 (1.26-1.28x) |
| reduce-scatter fp32 AVG, XL root | 1001-1007 | 1468 (1.46x) | 1283-1294 (1.28x) |

The grid is 32 CTAs and, unlike `reduce_scatter_gin_fused.mojo`'s, that is also its
isolated fit: the handoff is per block, so a larger grid multiplies the
seven remote flag stores and the 8-way rendezvous per piece while leaving
each block less to push between them. Block / root fp32 SUM in us:
**32 CTAs 515/1277**, 128 CTAs 602-628/1370-1383. The fused kernel wants
the opposite (128 CTAs 1263 us on the root, 32 CTAs 1567) and has to be
held down to 32 by the step; this one does not, and it matches the fused
kernel's 128-CTA root time on a quarter of the SMs.

`RS_STREAM_TARGET_CHUNKS = 4` is fitted on the same nodes, block / root
fp32 SUM in us: 2 chunks 617/1292, **4 chunks 500/1299**, 8 chunks
571/1413. More chunks shorten the one exposed exchange (the last chunk's)
and add about 50 us of proxy time each, and 8 is already the wrong side of
that. Two variants measured and not taken: one piece per block per chunk
(`RS_STREAM_SLICE_UNROLLS = 8`, which degenerates the streaming to the
fused schedule minus its grid barriers) 517/1279, and pushing piece `j` and
reducing piece `j-1` back to back with no barrier between them, so the
warps drift apart and one SM holds NVLink stores and HBM loads at once,
520/1274. Both sit inside this unlocked-clock box's run-to-run spread
(+-4% on the mojo leg, +-1% on NCCL's).

**The root reduce-scatter is still 1.28x NCCL, and the reduce is not most
of what is left.** Deleting both HBM passes -- the 8-way reduce and the
output sum -- from an otherwise final kernel, keeping every flag, exchange
and arrival so the schedule is unchanged, measures block 469 us and root
1155 us against 515 and 1277 with them. So the two passes cost 46 and 122
us, and the rest (the push, the handoff and the one exposed exchange) is
already 1.01x NCCL at the block size and **1.17x at the root**.

That bounds what the obvious next step buys. A node-local **ring** whose
hop fuses load, add and store -- NCCL's `recvReduceSend`, one pass that
loads the neighbour's slice, adds the local contribution and stores to the
next neighbour -- moves exactly the same NVLink bytes as this push and
makes the reduce free, so it would land near that 469 / 1155: parity at the
block size, 1.17x at the root. The other 17% is the push itself. 287 MB per
GPU at the 326 GB/s the fused kernel's push measured on this fabric would be
880 us, and 880 plus the last chunk's 122 us exchange is 1002; the
push-only build measures 1155, i.e. about 281 GB/s all in. Recovering the
fused kernel's push rate inside the streaming schedule -- the two differ in
that a block here owns a contiguous range instead of grid-striding the
chunk, and publishes a release flag per piece -- is worth as much as the
ring, and neither has been tried.

Its grid is fitted end to end, not on the isolated collective, and the two
fits disagree: 128 CTAs make the isolated root reduce-scatter 1.2x NCCL and
32 CTAs 1.5x, but every CTA holds an SM's register file for the call and the
backward's GEMMs lose those SMs, so GPT-2 XL FSDP2 on 2x8 H100 runs 62.4k
tok/s at 128 CTAs and 67.0-67.6k at 32 (`RS_FUSED_BIG_BLOCKS`, with the
sweep). The node-local gathers of the multi-node all-gather are capped the
same way (`AG_NODE_BLOCKS`, 96 against the single-node 432) and the progress
thread keeps polling between the chunks of one call instead of sleeping
(`BATCH_POLL_NS`). NCCL's kernels here are 16 CTAs of 544 threads at 96
registers.

What the SMs cost, from a torch trace of one step (`tmp/fsdp2-parity/prof0`,
both stacks): a persistent GEMM (132 CTAs, 168 registers x 384 threads,
214 KB of shared memory) cannot share an SM with any collective CTA, so the
GEMMs launched while NCCL's 16-CTA kernel is resident run at their solo
speed (41.2 vs 41.0 us), those launched under a 96-CTA gather took 1.8x and
under a 32-CTA reduce-scatter 1.46x; and both stacks' collectives are
resident about three times their isolated duration, because the kernel of
an early rank waits in its start barrier for the slowest one (our fused
reduce-scatter 1.8 ms resident for 0.65 ms of work, NCCL's 1.6 ms for
0.52). Two things follow. The start barrier is now run by one block ahead
of the grid (`ccl_rank_gate`, `_sync` on the spare flag row `GATE_ROW`), so
the skew is absorbed on one SM and the 32 (reduce-scatter) or 96 (remote
gather) CTAs behind it only ever hold their SMs while moving bytes; the
grid skips its phase-0 barrier (`GATED`), on the same guarantee -- a
rank's gate flag is published after everything before it on its stream
completed, and every peer's is awaited. And the mapped local gather
releases its chunk's RDMA exchange from inside the kernel, the moment the
last block has staged this rank's contribution (arrival counter at
`_AG_ARRIVE_OFFSET`), so the network transfer runs under the peer pulls
instead of after them: block bf16 all-gather 350 -> 249 us and one launch
fewer per chunk.

Two isolated wins that did not survive the step, recorded so they are not
retried blind: NCCL's 16-CTA grid for the gathers (16 blocks x 16 loads in
flight measure the same isolated time as 96 x 4 on the XL sizes, but a
latency-bound pull under the compute stream's HBM traffic loses far more
from 6x fewer CTAs than the GEMMs gain, 63.1-64.8k tok/s against
66.1-69.1k), and 256 threads per reduce-scatter block (no register spills,
isolated block 618 -> 576 us, but the block still owns the whole register
file and the step measured no better). The push itself is bound by the
fabric, not by the SMs: rotating the peer each block stores into by its
block index took it from 291 to 326 GB/s per GPU (`_peer_step`, now on
both vendors), and it runs at 140-170 us per 53.8 MB chunk from 16 CTAs
up; what fewer CTAs cost is the HBM-bound reduce and the phase-1 barrier's
wait on the slowest rank's push.

The former multi-node implementation all-reduced 1 MiB chunks of every
destination's slice and synchronized the host to release temporary memory
(16 ranks: 32x NCCL's device time on an XL block, 39x on the root). The
single-node measurements below compare against that former schedule; the
16-rank numbers are under "Multi-node".

Measured through the library's own exported entry points on 2×H100 SXM
(NV18), one process per GPU, 20 back-to-back calls per burst, four bursts in
ABBA order, median CUPTI kernel time, against NCCL 2.28.9 on the same node
and the same buffers (job 257033, `tmp/fsdp2-mojoccl-harness`):

| reduce-scatter | per rank out | NCCL µs | placeholder µs | this µs | ratio |
|---|---:|---:|---:|---:|---:|
| fp32, one XL block | 61.4 MB | 277.3 | 2236.9 | 263.4 | **0.95** |
| fp32, the XL root | 164.1 MB | 673.4 | 5988.9 | 689.0 | **1.02** |
| fp32, 1 MiB | 1.0 MB | 16.2 | 35.2 | 13.0 | **0.80** |
| fp32, 357×789 at an odd offset | 1.1 MB | 21.0 | 88.6 | 22.8 | 1.09 |

Host enqueue, the same call measured on the host (median of 30, queue
short): 6.4 µs, against NCCL's 12.4 µs and the placeholder's 2553 µs — it
synchronized the stream, so its enqueue was the whole collective.

The unaligned row is the one that does not clear 10%: its scalar path
reduces one element at a time where the vector path does a whole 16-byte
group with one SIMD accumulate. It reads 1.09 through the library and 1.12
through the process group, i.e. 2 µs, and FSDP2 never issues an unaligned
reduce-scatter.

### All-gather: one read of the contribution, two stores

The all-gather was already the unicast minimum (a local stage into my own
region, then every peer's slot read into my output — NVIDIA cannot do
better, because a peer may read my library region but may not write my
MAX-allocated output). Two things in it were not minimal:

- the local half copied my contribution **twice**, once into the region and
  once into my own slice of the output, so it read it twice. One read and
  two stores is a quarter less HBM traffic in that phase
  (`_copy_bytes2`) — worth 5 µs of 141 on an XL block gather and 31 of 686
  on the root's;
- the byte copy's unaligned fallback moved **one byte at a time**, sixteen
  loads and stores per 16-byte chunk. An offset view is still 4-byte
  aligned for every dtype wider than a byte, so there is now a 4-byte path
  between the two, which halves the time of an unaligned gather
  (32.3 → 18.2 µs at 1.1 MB). Both paths walk the same 16-byte chunks in the
  same grid-stride order as the vector path, which is what keeps a writer
  and a reader that disagree about alignment block-matched.

Same conditions as the reduce-scatter table above:

| all-gather | per rank in | NCCL µs | before µs | this µs | ratio |
|---|---:|---:|---:|---:|---:|
| bf16, one XL block | 30.7 MB | 147.1 | 140.8 | 135.5 | **0.92** |
| fp32, the XL root | 164.1 MB | 663.5 | 697.9 | 655.8 | **0.99** |
| bf16, 0.5 MiB | 0.5 MB | 11.5 | 11.1 | 11.3 | **0.98** |
| fp32, 357×789 at an odd offset | 1.1 MB | 28.3 | 32.2 | 18.2 | **0.65** |

Host enqueue 6.1 µs against NCCL's 12.2.

At 0.5 MiB the two libraries read the same within their own run-to-run
spread (NCCL's own 0.5 MiB number was 10.1, 11.5 and 15.6 µs in three
runs), and roughly 9 µs of our 11.3 is fixed cost: two system-scope
barriers — the start barrier, and the one between the stage and the pull —
plus the launch. NCCL needs neither, because below a few megabytes it runs
LL, whose flags travel inside the payload. Closing that would take an
LL-style protocol; the smallest all-gather GPT-2 XL FSDP2 issues is 30 MB.

For the architectures that are not present here (`gfx942`, `sm_80`, through
`scripts/compare_kernel_asm.py --kernel-dir torch_mojo_backend/distributed`
against the tree before this work): the twenty new reduce-scatter
specializations appear, the four users of `_copy_bytes` change — the
all-gather, the broadcast, and the inter-node copy and place kernels, all
for its new 4-byte path — and nothing else moves. **Unmeasured on AMD**: the
reduce-scatter's cross-link direction is the one gfx942 wants and the 4-byte
path only removes instructions, but neither of those is a measurement.

**End to end**, GPT-2 XL FSDP2 on 2×H100 (bf16 blocks, fp32 root, sequence
1024, batch 1 per rank, `demo_scripts/gpt2_fsdp2.py --benchmark`), four legs
in palindromic order mojo/vendor/vendor/mojo, median tokens/s per leg:
mojoccl 7870 and 7861, NCCL 7164 and 7786. MojoCCL was at 5134 tok/s before
this work against NCCL's 7786 — the reduce-scatter was the whole gap. The
124M five-step loss trajectory is bit-identical under the two libraries in
both precisions (11.028627 → 6.929632 fp32, 11.028791 → 6.923096 bf16).

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

**gfx942 waits acquire payloads once after observing completion.** The
intra-node barrier and the inter-node proxy wait poll with a relaxed system
atomic load, `s_sleep(1)` after a failed poll (RCCL 2.22.3's gfx942
[`waitPeer`](https://github.com/ROCm/rccl/blob/e72b592201d626f16a03a7ba22502130a2846036/src/device/prims_simple.h#L92-L128)),
and one system acquire fence on exit (`poll_acquire`). An acquire load on
gfx942 includes `buffer_inv sc0 sc1`, and issuing it on every failed poll
also invalidates the L2 of the compute running on other streams. Abort and
status loads stay acquire; every producer still executes its system release
fence before the block barrier, and flag publication is still a system
release store (the release side cannot be weakened: `hipIpcOpenMemHandle`
does not carry the uncached memory type, kernel_results §3 and §7).

This is the atomic-to-fence rule
([C++ atomics.fences p4](https://eel.is/c++draft/atomics.fences#4),
[LLVM fence](https://llvm.org/docs/LangRef.html#fence-instruction)): a
relaxed load L that reads the release store S, sequenced before the acquire
fence F, makes S synchronize with F, so the payload writes before S happen
before the reads after F. The fence runs even when the first poll succeeds.
MAX 26.5 maps relaxed to LLVM monotonic, and Atomic and fence default to
system scope. The fence sits after the polling branch has closed, so every
wave issues it under its full EXEC mask; inside the branch the gfx942
assembly issued it with EXEC = 0 (kernel_results §7 has that history). The
proxy wait's single lane polls a uniform address, so its loop never narrows
EXEC. The host publishes completion only after the receive arrivals, the
sends and the NIC flush; a transport error may release the mailbox without
data, with the communicator fault latched, as before.

Assembly ([LLVM gfx942 memory model](https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942)):
`global_load ... sc0 sc1` and `s_sleep 1` in the hot loop, `s_waitcnt` and
`buffer_inv sc0 sc1` at the exit; every barrier acquire of the CCL entry
follows the branch's `s_or_b64 exec` restore and directly precedes
`s_barrier` (309 of 309). Only gfx942 takes this route; every other target
keeps its acquire loads, and the sm_90a CCL kernels are unchanged (173 of
173).

Measured on **2 × 4 MI300A, Adastra job 5447705**, GPT-2 XL FSDP2,
bf16 parameters, fp32 reduction, sequence 1024, batch 1/rank:

| Measurement | Previous polling | Acquire once | Mojo + RCCL |
|---|---:|---:|---:|
| Profiled comm busy, ms/step | 209.5 | 188.5 | 142.9 |
| Profiled compute kernel sum, ms/step | 249 | 235 | 211 |
| Profiled GEMM sum, ms/step | 135 | 129.6 | 114 |
| Block AG, streamed device µs | 926.6 | 887.2 | 641.7 |
| Root AG, streamed device µs | 4285.4 | 4256.9 | 3228.4 |
| Block RS AVG, streamed device µs | 954.0 | 951.3 | 1235.0 |

Isolated timings: 30 queued calls, three bursts, eight ranks, ABBA legs
(the previous-polling column is one reference leg); profiles are single
diagnostic runs over three steps. End to end the change was about 1% in
paired medians (19.96k/19.49k/20.66k/19.79k tokens/s, unchanged/acquire-once
ABBA), inside the leg noise: it cuts device work and cache interference but
does not close the gap to RCCL alone. Losses reproduced each
implementation's pre-change trajectory, and the 2 × 4 and 1 × 4 worker
suites passed.

### NVLS: the large sizes go through the switch

The 168 and 512 MiB rows above are the **unicast** ceiling (~310 GB/s per
direction over NVSwitch): a push/reduce/pull allreduce moves
`2(world-1)/world × bytes` per GPU each way, and no schedule beats that
while every byte travels point to point. NCCL closes it with NVSwitch
multicast, and so does this library now.

A single-node communicator whose devices all report
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` builds its region as VMM memory
bound to a per-node multicast object, and routes allreduces of
`NVLS_MIN_BYTES` (48 MiB) or more through `multimem.ld_reduce` /
`multimem.st`: rank r pulls its 1/world slice through the switch, which sums
the `world` contributions and returns one value, and pushes the result back
into all `world` regions in one instruction. NVLink traffic falls to `bytes`
per GPU each way — 1.75× less at world 8 — paid for with ~1.5× the HBM
traffic, because the user's tensors are MAX-allocated and cannot be bound to
a multicast object, so every byte is staged in and out. That trade is why
the path has a size floor: 48 MiB is the measured crossover, sharp (4% the
wrong side at 40 MiB, 4% the right side at 48) and the same for fp32 and
bf16. int32/int64 stay unicast.

**Bring-up** (`transport/nvls.mojo` and `transport/multicast.mojo`, NCCL's
`src/transport/nvls.cc` sequence). The
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
(`/tmp`), so the rendezvous carries no extra round.
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
while at MINIMUM the same region is 514 MiB. MINIMUM is the code default.

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

**The kernel** (`device/all_reduce.mojo`) is the prototype's split-grid schedule:
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
  the same reason every other kernel in `device/symmetric/` does: the
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

A seventh leg measured `RECOMMENDED multicast granularity` — NCCL's 512 MiB
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
`device/common.mojo` reads the 100 MHz counter directly on AMD (with a
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
in a half dtype. A reduce-scatter on such a communicator (FSDP2's gradient
reduction) is its own hierarchical schedule, not this split allreduce: a
node-local reduce (`reduce_scatter_nodes`) sums the local contributions into
one aligned partial per node, each rank RDMA-writes every remote node's
partial to its counterpart there, and `inbox_sum_out` adds the received
partials to the local one in the caller's output. On
MI300A that node-local reduce runs on RCCL's 24 multi-node MI300A channels
(`RCCL_APU_NODE_CTAS`) instead of the allreduce's 128/912 (a discrete
gfx942 keeps 128/912, see below; the split allreduce keeps them everywhere):
the reduce-scatter is network-bound (15.4 MB/rank fp32 950 vs 951 µs at 128
vs 24 blocks, 2 × 4 MI300A, job 5447705), and the freed CUs go to the
backward's GEMMs -- GPT-2 XL FSDP2 ABBA legs 22.0k/22.8k -> 23.2k/23.2k
tokens/s. Broadcast uses the same RDMA path: the root's node fans
out to its counterparts, then each node broadcasts locally. In an
all-gather each rank sends only its own contribution to the same local
rank on each remote node.
After the exchange, node-local all-gathers disseminate the received
contributions, and placement follows the bootstrap global-rank table.
Each local all-gather writes directly into the mapped global output slots,
and its local staging supplies the RDMA send. Staging is reused only after
send completion. All-gathers pipeline two chunks through separate existing
arenas, overlapping a network exchange
with the next local gather and the earlier remote gather.

On AMD the local peers push into compact slots of each other's regions
(see "AMD MI300A" above), so the RDMA source cannot be the slot the peers
write: each rank stages its contribution into a separate slot after the
`local_world-1` peer slots (`allgather_nic_stage_off`), together with its own output slice, releases
it (every wave's system release fence, then the per-arena acq_rel arrival
counter), and the last block to arrive release-stores the exchange into
the proxy mailbox. The NIC therefore starts while the xGMI pushes still
run, and the two node-block placement kernels and the separate proxy
request of the older AMD schedule are gone (four launches per chunk instead
of six on two nodes); that older schedule is deleted, so every AMD target
takes this one. Multi-node MojoCCL on AMD GPUs other than MI300A is
untested: they get the same push kernel and NIC slot, the single-node copy
cap for the gathers (`_node_grids`) and unroll 4, none of it measured or
run there. The staging bound becomes `local_world *
align16(chunk) <= 2 * arena_cap` next to the inbox bound; at 4 ranks/node
and the 64 MiB region the inbox (6,709,248 B) still binds. Measured on 2 × 4 MI300A (Adastra job 5447705, 8 ranks,
streamed device time per call, ABBA; RCCL 2.22.3 through the same process
group): XL block bf16 7.68 MB/rank 887 → 737 µs (RCCL 638), fp32 root
41.0 MB/rank 4238 → 3038 µs (RCCL 3214), fp32 357×789+3 233 → 226 µs
(RCCL 286).

The MI300A gathers of that schedule then take RCCL's multi-node MI300A
geometry instead of the single-node copy cap: 24 CTAs of 256 threads,
two 16-byte vectors in flight per thread (`RCCL_APU_NODE_CTAS`,
`AG_NODE_UNROLL`; RCCL 2.22.3 forces 24 channels on multi-node MI300A and
unroll 2 on gfx94 parts with more than 80 CUs). Against the 432-block copy
cap, 24 blocks is faster in isolation too: block 742 → 650 µs, root
3034 → 2686 µs, 357×789+3 225 → 174 µs. It is not the fastest isolated
grid, though: 96 blocks measured 599 / 2669 µs. 24 wins end to end
(GPT-2 XL FSDP2 on those 8 ranks, ABBA legs, tokens/s): 432 blocks
20.9k/20.7k, 96 blocks 22.0k/21.8k, 24 blocks 23.0k/22.8k, mojo+RCCL
23.1k/22.1k. There the gathers run beside the compute stream, and every CU
they hold is one the GEMMs do not get. Unroll 2 was measured only together
with the 24-block grid.

RCCL applies its 24-channel rule only to an APU
(`hipDeviceAttributeDirectManagedMemAccessFromHost`, `init.cc:1339-1346`),
and the discrete gfx942 parts (MI300X, MI325X) are the same ISA, so the
24-block grids take RCCL's test at run time (`_node_grids`, ANDed over the
ranks at init): a discrete gfx942 keeps the single-node copy cap for the
gathers and the allreduce caps for the node reduce, unmeasured. Unroll 2 needs no such test: RCCL's
unroll-2 rule covers MI300X too. The remaining isolated all-gather time
is the network: 7.68 MB per NIC at about 16 GB/s, against RCCL's 13.4 MB
per NIC at 21 GB/s; splitting each exchange into RCCL-sized 512 KiB
writes did not change it (664 vs 653 µs). Messages at least
`PIPE_SPLIT_UNIT * local_world` bytes per rank are split into two balanced
chunks unless region capacity requires more. This threshold was measured on
2×8 H100 and leaves smaller single-chunk gathers unchanged; on 2 × 4 MI300A
the two halves (RCCL's two slices per chunk) took the XL bf16 block from 601
to 541 µs once the flush read had its own endpoint (below). Inbox credits
follow each chunk's remote consumers; the source arena is reused only after
its send and consumers complete. Broadcast remains unpipelined. Single-node
communicators keep the fused intra-node path and never touch IB.

Measured at 16 ranks (2x8 H100 SXM, InfiniBand, job 258050) through the
library's exported entry points, CUPTI device time of the comm stream,
median over ranks of complete-call sums, vendor/mojo ABBA in one process
(`tmp/fsdp2-2node/harness`, `tmp/fsdp2-parity/final/bench_abba.json`); NCCL
2.28 picked RING_LL for both collectives. The reduce-scatter grid is 32 CTAs
because the GEMMs it runs under decide the step, not this table (see
"Reduce-scatter" above): the same kernel at 128 CTAs measures root 1263 us.
"Before" is the tree before the gate, the in-kernel RDMA release and the
peer rotation (same day, same nodes).

| 16 ranks, per rank | NCCL us | before us | mojoccl us | ratio | launches | host us NCCL / mojo |
|---|---:|---:|---:|---:|---:|---:|
| all-gather bf16, XL block (3.84 MB) | 394 | 350 | 272 | **0.69** | 4 | 12.8 / 18.5 |
| all-gather fp32, XL root (20.5 MB) | 1084 | 1027 | 977 | **0.90** | 8 | 12.3 / 21.3 |
| reduce-scatter fp32 SUM, XL block (7.68 MB) | 520 | 669 | 658 | 1.27 | 2 | 12.3 / 8.5 |
| reduce-scatter fp32 AVG, XL block | 524 | 666 | 654 | 1.25 | 2 | 12.4 / 8.5 |
| reduce-scatter fp32 SUM, XL root (20.5 MB) | 1068 | 1566 | 1499 | 1.40 | 2 | 12.2 / 8.7 |
| reduce-scatter fp32, 357x789 at an odd offset | 159 | 312 | 299 | 1.88 | 2 | 13.9 / 8.2 |
| all-gather fp32, 357x789 at an odd offset | 169 | 179 | 159 | **0.94** | 4 | 12.0 / 11.7 |

The reduce-scatter rows of that table are `reduce_scatter_gin_fused.mojo`'s and are now
only what calls below `PIPE_SPLIT_UNIT` bytes per rank take; "Streaming
reduce-scatter" above has the current numbers (block 1.10x, root 1.28x) and
a fresh NCCL column measured beside them.

The reduce-scatter's device time is not where its step cost is. Its
per-phase trace (block fp32, 32 CTAs, per 3.84 MB chunk): push 165-185 us
at 326 GB/s per GPU, 8-way phase-1 barrier 5-35 us of rank skew, reduce
41-51 us, output sum 13 us, exposed last RDMA 55-60 us (42 GB/s), grid
barriers 4-7 us each. The push is bound by the fabric from 16 CTAs up, so
the isolated gap to NCCL (a 16-rank ring that never waits for a whole
node's push before reducing) is the reduce and the barriers, and closing it
with more CTAs costs the step more than it returns.

The streaming kernel confirms that from the other side. It cut the isolated
block reduce-scatter from 1.39x NCCL to 1.10x and the root from 1.48x to
1.28x, and GPT-2 XL FSDP2 on these two nodes did not notice: three six-leg
palindromes of the streaming tree against two of the tree before it, same
allocation, alternated before/after/before/after, 18 and 12 windows per
stack, median tok/s -- Mojo + MojoCCL 65,448 before and 66,007 after
(+0.9%), against -0.6% and -0.3% on the two stacks whose code is
byte-identical between the trees (Mojo + NCCL 67,571 -> 67,139, CUDA + NCCL
71,021 -> 70,789). Per-window spread is 60-71k on every stack, so +0.9% is
inside the noise and the honest reading is "nothing lost": the SM footprint
is the same 32 CTAs, and a reduce-scatter FSDP2 has already overlapped with
the next layer's backward does not get cheaper by finishing sooner.

The placeholder these replaced measured 16,623 / 41,256 us on the block /
root reduce-scatter (32x / 39x) with a host synchronize per call; the
Nsight probe of the new paths sees no synchronize or query inside any
reduce-scatter or all-gather enqueue. End to end, GPT-2 XL FSDP2 on those
two nodes, six-leg palindrome, six windows per leg, median tok/s of all
twelve windows per stack: CUDA + NCCL 70,993, Mojo + NCCL 65,444, Mojo +
MojoCCL 65,760 (`tmp/fsdp2-parity/final/xl`) -- MojoCCL and NCCL on the
Mojo device within noise of each other (the Mojo + NCCL legs of that run
spread 57.0-68.8k; the same stack measured 69,134 on job 258050 the day
before with the untouched tree, when Mojo + MojoCCL measured 66,406). The 124M
five-step fp32 loss trajectory matches NCCL's exactly at four steps and
differs by one fp32 ULP at one (9.151466 vs 9.151465): AVG's 1/16 is
applied to every input before either sum, as NCCL does, but the eight
node-local terms and two node partials associate differently from a
16-step ring, so the last bit of an fp32 sum can differ.

**The three phases overlap, inside one kernel.** The bucket is cut into K
chunks, with at most `PIPE_ARENAS` chunks alive:

```
RS(0) rel(0)  RS(1) rel(1)  RS(2) rel(2)  RS(3) rel(3)
              wait(0) add(0) AG(0)  RS(4) rel(4)
              wait(1) add(1) AG(1)  RS(5) rel(5)  …
```

so the proxy exchanges chunk k while the GPU reduce-scatters later chunks
and all-gathers earlier ones. That loop runs inside ONE persistent kernel
per allreduce (`all_reduce_gin.mojo`, `ccl_internode_allreduce_pipelined_*`):
the reduce-scatter and all-gather bodies of `device/symmetric/all_reduce.mojo`, the
inbox add, the mailbox store that releases a chunk to the progress thread
and the spin on its completion are phases of that kernel, separated by a
rank-local grid barrier where a launch boundary used to be. A grid that
waits for itself must be entirely resident, and that has two halves. The
first is capacity: the grid never exceeds the driver's occupancy answer for
the kernel times the SM count, taken at init as the minimum over every
dtype the kernel can be launched with (`fused_resident_blocks`, every
instantiation compiled there), exchanged and checked across the node's
ranks, and used as the node-agreed grid at launch -- no rank re-derives it
-- and the launch carries CUDA's cooperative attribute, so a grid past the
empty-device capacity is refused, not hung. The second is progress under
concurrency, which no launch mode guarantees: cooperative launch checks
empty-device capacity only, and under running kernels the collective's
blocks simply wait for SMs. That is progress here because everything else
on the device -- the persistent GEMM CTAs above all -- is finite and never
waits on this stream; a resident kernel polling for work that is ordered
behind this collective would starve it until the deadline. Where the
cooperative attribute is unsupported (MAX's launch attributes are
CUDA-only, so on AMD) the ordinary launch guarantees nothing beyond the
occupancy bound, and the deadline is the backstop. A launch that fails
after the call's exchanges were reserved fails the communicator from the
host (`ERR_HOST_LAUNCH`, in the status page's own host word -- the device's
record is claimed by a compare-exchange on device memory the host cannot
join, so the two never share words). `_report_fault` selects the first
fully published fault observed at host latching: an already visible device
fault wins; otherwise the host fault wins, including against a device record
still being published. That selection is latched and cannot change later.
Subsequent calls return `ncclRemoteError` and the peers'
own deadlines report the rank, instead of a hang. Collectives of one
communicator are kept in one total order across streams: an event is
recorded on the stream after every call and a call on a different
stream waits for it first. Default stream 0 participates, and the caller may
destroy a completed stream before the next call. Teardown and error polling
also use the owned completion event. Abort polls it without blocking; only
an incomplete submission needs conservative caller-stream queries. Concurrent
error polling skips the device read while submission holds the lock, but
still checks the atomic terminal-failure flag.
The SM count comes from MAX's device attribute on both vendors, so
AMD takes the fused path too. It is now measured at 512 threads on two
nodes with four MI300A APUs each; see the
[XL verification](#gpt-2-xl-on-two-mi300a-nodes). A call
the geometry cuts into more chunks than the inter-node
work ring has slots (`WORK_SLOTS`, 512: a 129 MiB allreduce at
`MOJOCCL_REGION_MB=1`) takes the split schedule, which releases chunks as it
goes, because the fused kernel files every chunk's exchange before it
launches and would wait on a slot only its own kernel could free
(`tests/multinode/small_region_probe.py`). It used to be
five launches per chunk, and on GPT-2 XL (146 allreduces per step, issued
from the autograd thread that also issues the backward's GEMMs, ~24 µs of
driver time per launch) that cost the compute stream 1179 idle gaps of
~250 µs per 5.5 s -- 90.3% busy against 96.2% under NCCL, 0.906 of stock
end to end. See "One kernel per allreduce" below for the numbers.
The split fallback and fused path launch different grids and use
block-matched intra-node barriers. The effective schedule, threads per
block, and each node's occupancy bound and multiprocessor count are
exchanged at init; incompatible builds are rejected.

`K = sqrt(bytes / (local_world × 640 KB))`, capped at 16 by choice and
raised from below by geometry when a chunk would not fit — the square root
is of what an extra chunk costs (originally two launches; now two grid
barriers and two 8-way start barriers) against the 40–45 GB/s the RDMA runs
at. It gives K = 1 up to ~10 MiB, 2 at the 27–39 MiB DDP buckets, 5 at
168 MiB and 10 at 512 MiB, and keeps a chunk's shard above 1 MiB without a
second clause. The constant is fitted in source;
it is part of the wire layout (K decides how many exchange counters a
collective consumes) and is checked equal on every rank at init.

Two things make concurrent chunks safe, and neither is stream order across
ranks. **The staging arena is replicated.** A multi-node region is
`PIPE_ARENAS` complete arenas — each its own signal area and its own
`[stage_in | stage_out]` of `cap/PIPE_ARENAS` — followed by one cap-sized
network area; chunk k uses arena `k % PIPE_ARENAS`, so concurrent chunks
cannot collide in the push slots or in the shard, and that arena's own start
barrier is what orders chunk k+`PIPE_ARENAS` behind chunk k's pulls, the
invariant one arena already had. The intra-node kernels are untouched:
the split kernels take a shifted base and a smaller cap, nothing more. The
staging total is `2 × cap` either way, so the region is the size it always
was. **The inbox is reused only against a credit** — see the transport
below. `PIPE_ARENAS` (4) and the derived `INBOX_SLOTS` (5) are source
constants in `include/comm.mojo`, not environment variables: they are part of the
wire layout and every rank has to agree on them.

The chunk cap is now one arena, and the inbox slot group, rather than the
whole region: 64 MiB at 2 nodes and 36 MiB at 8 with the default 256 MiB
region, so the large sizes are chunked by geometry as well as by choice.
`tests/multinode/selftest/geometry_test.mojo` sweeps that arithmetic over
regions of 1 MiB–1 GiB, `local_world` 1–8 and 2–16 nodes.

**Two transports, one engine.** `transport/net.mojo` is the transport-neutral
progress engine (work ring, credits, arrival tally, flush, abort,
teardown); the six operations it needs -- post a payload, post an
immediate, post the flush read, poll completions, fill the bootstrap blob,
attach a peer -- are implemented twice, in `transport/net_ib/` (InfiniBand) and
`transport/net_ofi.mojo` (HPE Slingshot through the `cxi` provider). `ib_setup`
picks one at run time: `MOJOCCL_NET=verbs|fabric` wins outright, otherwise
verbs if libibverbs opens and lists an ACTIVE InfiniBand port, else
libfabric if `libfabric.so.1` opens and `fi_getinfo` finds an FI_EP_RDM
provider with FI_RMA|FI_MSG|FI_HMEM, else the same clear error as before.
The engine's dispatch is one `st.net == NET_VERBS` branch per post and one
per poll batch -- loop-invariant and perfectly predicted -- so the verbs
path costs what it always did.

**Transport A, InfiniBand** (`torch_mojo_backend/mojo/tmb/ccl/`: `misc/ibvwrap.mojo`,
`include/{ibvwrap,ibvcore}.mojo`, `transport/net_ib/`, `transport/net.mojo`,
`proxy.mojo`, `device/symmetric/gin_scratch.mojo`,
`include/nccl_device/gin/proxy/gin_proxy.mojo`, `bootstrap.mojo`): libibverbs is dlopened; setup calls are
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

**Transport B, Slingshot / libfabric** (`transport/net_ofi.mojo`): `libfabric.so.1`
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
The flush is always enabled.

Every struct offset, size and constant `transport/net_ofi.mojo` hard-codes is dumped
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
pinned, device-mapped mailbox: the fused kernel's block 0 releases the
exchange into `MB_REQUEST` with one release store after the grid barrier
that completes the shard, the thread posts it, and the same thread spins on
`MB_DONE` (one PCIe read per poll, the abort word every 256) until the
thread has seen the N−1 arrivals and flushed; the other blocks wait in the
grid barrier behind it. After the inbox add has run on every block the
kernel stores the exchange number into `MB_CONSUMED`, which is the inbox
credit — the fact the credit asserts, published from the device rather than
inferred on the host from enqueue order (the split schedule's `credit_upto`,
which the unfused broadcast/all-gather paths still use). Several exchanges
live between request and done, so the thread is one non-blocking step
function (`ib_drive`, shared with the internal callback and the
GPU-free self-tests) that retires exchanges in sequence order — RC ordering
is per queue pair, so arrivals are not ordered across peers — and pipelines
the flush read the same way. The thread spins only while something is
outstanding; idle, it yields and then sleeps in `20 µs`
steps, because a thread spinning between exchanges competed with the
host-bound Python dispatch thread and cost ~20% of end-to-end training
throughput at 16 ranks. A `cuLaunchHostFunc`/`hipLaunchHostFunc` stream
callback did the same job in the proxy-disabled experiment and cost about
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
the run, not throughout it. The default placement policy therefore pins
automatically:
`sched_getaffinity` reads this process's mask, and if it holds at least
`2 × local_world` CPUs (torchrun gives every rank of a node the same mask),
the thread goes on that mask's CPUs taken in descending order, indexed by
`local_rank` — the top `local_world` CPUs, distinct per rank, Python's
threads left wherever the scheduler already put them. Within that, a
candidate whose SMT sibling is itself another rank's pin is deprioritized in
favor of one that isn't (`topology/thread_siblings_list`, best-effort — read
failures just skip the check). A mask smaller than `2 × local_world` leaves
the thread unpinned rather than fight Python for a scarce core.
`MOJOCCL_IB_TRACE=1` prints the chosen CPU (or why none was chosen).

**Unmeasured placement follow-up.**
Measured on Adastra (`taskset -pc` of every rank of a 2x4 job): each rank's
affinity mask is `0-47,96-143`, which is 48 physical cores with BOTH their
SMT threads -- `96+c` is the sibling of `c`. Taking the mask's CPUs in
descending order therefore pins the four progress threads to 143, 142, 141,
140, which are the siblings of cores 47, 46, 45, 44 -- cores the ranks' own
Python and autograd threads run on. `_smt_sibling_free` accepts them because
it only rejects a CPU whose sibling is ANOTHER RANK'S PIN, and 44-47 are not
pins. So on any node whose mask is a full-SMT range the policy reliably puts
every progress thread on a busy core's sibling, which is the placement it
exists to avoid; H100 measurements motivate investigating this, but do not measure its effect on MI300A.
A fix would prefer, among the mask's CPUs, ones whose sibling is not also in
the mask, and only then fall back to the descending rule. Not the cause of
any bug currently open.

### Automatic defaults and supported configuration

The measured algorithm choices live in source: fused scheduling, hardware
block caps, 512 threads/block, four-vector unrolls, one CTA/SM launch bound,
128 MiB large-message threshold, pipeline unit 640000, 48 MiB NVLS threshold,
MINIMUM multicast granularity, 20 µs idle proxy sleep, provider-negotiated
memory registration, enabled fabric flush, 30 s fabric setup retry, and
`/tmp` local descriptor sockets. They have no environment tuning overrides.
Changes require a source fit and validation on the affected hardware.

Only settings with a similar NCCL/RCCL purpose remain. The analog names below
describe purpose; their units and value syntax can differ. See the
[NCCL environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

| Variable | Default | Purpose | NCCL/RCCL analog |
|---|---|---|---|
| `MOJOCCL_SOCKET_IFNAME` | UP non-loopback IPv4 interface with a default route | Bootstrap interface; exact name | `NCCL_SOCKET_IFNAME` |
| `MOJOCCL_BOOTSTRAP_TIMEOUT_S` | 120 s | Whole rendezvous deadline | `NCCL_SOCKET_RETRY_CNT`, `NCCL_SOCKET_RETRY_SLEEP_MSEC` (connection retry budget) |
| `MOJOCCL_IB_HCA` | GPU/NIC affinity | Exact verbs adapter name | `NCCL_IB_HCA` |
| `MOJOCCL_IB_TIMEOUT_S` | 60 s | Peer/device progress deadline | `NCCL_IB_TIMEOUT` (verbs timeout; different units and scope) |
| `MOJOCCL_IB_RELAXED_ORDERING` | 1 | Relaxed-ordering memory registration | `NCCL_IB_PCI_RELAXED_ORDERING` |
| `MOJOCCL_IB_TRACE` | 0 | Transport diagnostics and timings | `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` |
| `MOJOCCL_NET` | Active verbs port, otherwise libfabric | Transport selection | `NCCL_NET` |
| `MOJOCCL_LIBFABRIC` | Loader soname, then Cray path | Transport library path | `NCCL_NET_PLUGIN` (library selection) |
| `MOJOCCL_FABRIC_PROVIDER` | Prefer cxi, then a provider satisfying the required capabilities | Fabric provider selection | `NCCL_NET` |
| `MOJOCCL_FABRIC_DOMAIN` | GPU/NIC affinity, then rank modulo domain count | Exact fabric NIC/domain | `NCCL_IB_HCA` |
| `MOJOCCL_REGION_MB` | 64 on gfx942; 256 elsewhere | Staging capacity, equal across ranks | `NCCL_BUFFSIZE` (different layout and units) |
| `MOJOCCL_NVLS` | 1, capability checked across all ranks | Enable multicast where supported | `NCCL_NVLS_ENABLE` |

The `ib_bringup`, `ib_pipeline` and `fabric_hmem` selftests pass a private
`_synchronous_test=True` argument to `ib_setup`: their calling thread drives
the transport directly, so a concurrent proxy would race it. The first two
also use libc and cannot allocate a GPU-visible mailbox. Production never
sets this argument; there is no environment switch for it. `MOJOCCL_ROOT_`
names an internal compiler-runtime global, not an environment variable.


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
| mojoccl, fabric flush disabled (historical experiment) | 9 387 us | 100.1 GB/s | 1.33x |
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
fabric flush disabled (historical experiment) measured `correct=OK`. **The FI_FENCE is not the
cap**: exchanges do overlap despite it (the sum of per-exchange in-flight
times, 14 x ~1.9 ms, is 2.5x the 10.6 ms the allreduce takes), so it does not
serialise the pipeline. What is left is 1.33x over RCCL against a 5.2 ms wire
floor (14 chunks x 9.36 MB shard / 25 GB/s), where RCCL sits at 1.36x the
floor and mojoccl at 1.8x.

**The flush read has its own endpoint.** Queued behind the data
writes, the flush of exchange e waited for exchange e+1's payload, which the
progress thread had already posted: e retired one exchange late, so the
pipelined all-gather's remote gathers of chunk e could not overlap the
network transfer of chunk e+1. A second endpoint on the same domain (same
NIC and PCIe function, so its read still orders behind that NIC's earlier
writes; same CQ and AV) issues only the flush -- the role of the verbs
path's self-connected QP and NCCL's gpuFlush QP; RCCL on AMD flushes every
512 KiB step (`net.cc`, `paths.cc` `ncclTopoNeedFlush`), so its flush never
waits behind megabytes. The flush itself stays. Measured on 2 x 4 MI300A
(job 5447705, streamed device time per call): XL bf16 block all-gather
654 -> 601 us, fp32 root 2682 -> 2365 us; reduce-scatter unchanged (its
exchanges are back to back on the wire either way).

The second endpoint is an optimization and never fails a communicator:
it asks for the least the read needs (an `fi_dupinfo` copy of the data
endpoint's info cut to `FI_RMA | FI_READ`, transmit-only CQ binding, a
local-read landing pad), and any failure while it comes up -- it is a second
cxi address context, the allocation `_check` documents failing with
-FI_ENOMEM under node memory pressure -- closes what was created, prints
`mojoccl: flush endpoint unavailable, flushing on the data endpoint` and
flushes on the data endpoint, correct and only later. It is not retried.
`MOJOCCL_IB_TRACE=1`'s closing `mojoccl net:` line says which endpoint
flushed (`flush_ep=own|data`). It is opened on every GPU: the serialisation
is a property of a libfabric endpoint's transmit queue, not of the GPU. Only
MI300A was measured; NVIDIA on libfabric (Slingshot GH200, EFA) is not
(H100 here uses verbs, whose flush QP was always separate).

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
the provider says `-FI_ENOMEM`, for the fixed 30 s setup retry budget.
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
  (`IbWork.op_kind/op_chunk/op_nchunks/op_numel`, set by `enqueue.mojo`), so
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
`fabric_hmem`, which registers a `transport/p2p.mojo` `alloc_region` allocation with
FI_HMEM_ROCR and exchanges into it. The transport ones run against either
backend (`MOJOCCL_NET`) on a host with a NIC and no GPU — the login node
under InfiniBand, a compute node under Slingshot — with
the historical proxy-disabled experiment; they caught
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

**One kernel per allreduce (2026-09-15).** GPT-2 XL (1.5B, 48 layers, bf16
autocast, batch 8×1024 per rank) under DDP on 2×8 H100, three stacks by the
protocol of `e2e_gpt2xl.sbatch` (30 steps, five interleaved rounds, mean
tok/s of steps 20–30): stock CUDA torch + NCCL 498k, mojo backend + NCCL
498k, mojo backend + mojoccl 451k = 0.906 (job 250679). nsys on rank 0
(job 250680, `ddp_prof/`): compute stream 90.3% busy under mojoccl against
96.2% under NCCL, the difference being 1179 gaps of ~250 µs per 5.5 s in the
middle of the backward, where DDP's hooks call allreduce on the autograd
thread; 127,490 `cuLaunchKernelEx` per 25 steps against 93,903 (+1,340 per
step, ~24 µs of driver time each): five kernels per chunk, two chunks per
bucket, 289 chunks per step, against NCCL's one kernel per bucket. Not
bandwidth: the comm stream was 36.6% busy against NCCL's 37.2%.

The fix is the fused kernel above, and its geometry is a trade the split
kernels never had to make: a GEMM block of this backend needs a whole SM,
so every SM holding a block of a persistent collective is lost to the
compute stream for the collective's whole life, and NVLink wants bytes in
flight, so fewer SMs need more per SM. Sweep on the same model (job 250904,
20 steps, mean tok/s of steps 10–20, two passes in palindromic order,
mojo+NCCL 495.5k in the same job; threads × 16-byte vectors in flight per
thread × blocks):

| geometry | tok/s | vs mojo+NCCL |
|---|---|---|
| split schedule (the historical forced-split experiment) | 449.1k | 0.907 |
| 256 × 2 × 32, 2 CTAs/SM (first draft) | 476.9k | 0.963 |
| 256 × 8 × 32, 2 CTAs/SM | 474.3k | 0.957 |
| 256 × 8 × 16, 2 CTAs/SM | 463.9k | 0.936 |
| 512 × 8 × 8 | 465.2k | 0.939 |
| 512 × 8 × 16 | 484.3k | 0.977 |
| 512 × 8 × 32 | 485.3k | 0.979 |
| 1024 × 8 × 16 | 484.6k | 0.978 |
| **512 × 4 × 16 (shipped)** | **487.6k** | **0.984** |

The runtime knobs around that geometry, same protocol (job 250995,
mojo+NCCL 498.0k): 16 blocks 0.982, 24 blocks 0.985, chunk rule at
320 000 (K=3 at the 39 MiB bucket) 0.975 and 160 000 (K=4) 0.971, proxy
idle quantum 2 µs 0.983, big-message grid 64 / 132 blocks 0.983 / 0.979,
1024 threads 0.981. Inside ±0.5% everything but K is noise; the big-message
grid earns its place standalone (168 / 512 MiB fp32: 1724 / 5099 µs at 16
blocks, 1329 / 3712 at 64, NCCL 941 / 2351), and 24 blocks shortens the
kernel's life (p50 712 µs, comm stream 42% busy) for the same step time, so
16 -- NCCL's own channel count -- stays the default.

An earlier sweep of the draft geometry (job 250753: 1 / 16 / 32 / 64 / 132
blocks → 450 / 462 / 481 / 482 / 448k) is where the SM argument comes from:
132 blocks is slower than the split schedule. nsys of the shipped geometry
against mojo+NCCL in the same job (rank 0, 5.5 s window): compute stream
97.0% busy against 96.0%, gaps > 150 µs 0.5% against 0.4% of the window
(the memcpy ones DDP's sync points make, identical in both), comm stream
54.9% against 37.7%; the allreduce kernel's per-call duration p50 923 µs
against `ncclDevKernel_AllReduce_f32_RING`'s 664 µs, and the last bucket
(313 MiB, the tied embedding, which nothing overlaps) 5.9 ms against
3.0–3.9 -- which is why the large-message block cap is higher on H100. What is left
against NCCL is the GEMMs: 102.6 ms/step of non-bias GEMM against 98.4,
i.e. the 16 held SMs over a longer life. Host cost per `dist.all_reduce`
(`tests/multinode/enqueue_bench.py`, 27 MiB fp32, 100 calls, no sync,
median µs on rank 0): 30.9 fused, 49.5 split, 41.3 NCCL. The standalone
device time of one allreduce (`ar_bench_gpt2.py`, 16 ranks, median µs fp32
at 27 / 168 / 512 MiB) is where the small grid shows: 355 / 1774 / 5198 at
16 blocks, 352 / 1481 / 4276 at 32, NCCL 265 / 942 / 2355 -- inside the
step that is hidden, the SMs are not.

The table (job 250996, the same protocol as job 250679 above):

| stack | mean tokens/s, steps 20–30 | ratio vs stock |
|---|---|---|
| stock CUDA torch 2.11 + NCCL | 497k ± 1k | 1.000 ± 0.002 |
| mojo backend + NCCL | 496k ± 1k | 0.999 ± 0.003 |
| mojo backend + mojoccl (all Mojo) | 488k ± 1k | **0.984 ± 0.002** |

Re-run after the review fixes (cooperative launch, occupancy bound, init
checks, ring bound; job 251179, `tests/multinode/e2e_three_stacks.sbatch`
with `E2E_TAG=e2e_gpt2xl`): stock 495k, mojo + NCCL 495k (1.001 ± 0.002),
mojo + mojoccl 487k = 0.984 ± 0.001; host cost per call unchanged at
30–31 µs (job 251253).

Validated in the same state (jobs 250997, 251023, 251056; after the review fixes 251177, 251178, 251297, 251298): `collectives`,
`ddp_parity` and `stress` at 16 ranks under both libraries and at 8 ranks
on one node, `collectives` under the historical proxy-disabled experiment and under
the historical forced-split experiment (plus `stress`), `ring_pressure.py`, nanoGPT-124M at 16
ranks to the same losses, and `tests/multinode/deadline_probe.py`: rank 0
sleeps through `MOJOCCL_IB_TIMEOUT_S=3`, its node-mates latch
`DEVICE DEADLINE in the multi-node allreduce's reduce-scatter stage` and the
other node's ranks the exchange wait, and every rank's next allreduce raises
`ncclRemoteError` within the deadline plus the 1 s grid grace instead of
hanging (measured per rank, under a watchdog). A rank alone on the split
schedule is refused at init, and a 129 MiB allreduce on a 1 MiB region --
past the work ring -- is correct (`small_region_probe.py`). The GPU-free self-tests
(`geometry_test`, `ib_pipeline` with the credit protocol) pass on the login
node.

Splitting harder does not help: at `PIPE_SPLIT_UNIT = 320_000` (K of 3 / 8 /
14 instead of 2 / 5 / 10) the 27 MiB bucket is unchanged and 168 and 512 MiB
regress to 1291 and 3648 µs (job 234242, same ABBA design). The extra
exchange's latency is not fully hidden, so past the point where the network
is covered an extra chunk only buys launches.

`collectives`, `ddp_parity` and `stress` pass at 16 ranks under both
libraries (job 234229, `cl02s01dgx06` + `cl02s04dgx02`), including `stress`'s
200 rounds of interleaved 4-byte / 27 MiB / broadcast / allgather
collectives and its 256 MiB messages, which the pipeline cuts into 7 chunks.
The historical proxy-disabled experiment fallback passes `collectives` too: it cannot
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
   which NCCL sets and `transport/nvls.mojo` now sets too, after which all 12 HCAs
   register it (`agents_docs/mojo_collectives_nvls_results.md` §4). There is still
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
