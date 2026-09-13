# Multi-GPU training on the mojo device (eager mode): design and plan

Status: **superseded.** Multi-GPU training shipped instead as the c10d
backend of the native device (`docs/distributed.md`,
`native/mojo/pg.mojo`), not as the single-process pure-Python ProcessGroup
this file proposes; the eager internals it cites (`torch_mojo_tensor.py`,
`mojo_device_aten_ops.py`) are deleted. Kept for the research it records.

Goal: data-parallel training across multiple GPUs on the `mojo` device in eager
mode, with performance comparable to CUDA (`torchrun` + DDP + NCCL), while
respecting the project's constraints: no NCCL/cuBLAS-class vendor libraries, no
C++ extensions (JIT-compiled Mojo only), works with a CPU-only PyTorch install.

## TL;DR

Build **single-process, multi-device data parallelism**: one Python process
drives all GPUs, with a **pure-Python `ProcessGroup`** (thread-per-rank) so
users get the standard `DistributedDataParallel` + `torch.distributed` UX, and
gradient allreduce running on **Modular's existing pure-Mojo P2P collective
kernels** (`max/kernels/src/comm/` in the modular repo), which are benchmarked
at-or-above NCCL intra-node. Overlap of communication with backward is done
NCCL-style — a priority side stream per device plus events — implemented inside
our Mojo extension, since stream priorities are not exposed in the Python
`max.driver` API.

## The three facts that force the architecture

These come from a survey of this repo, the pytorch repo, and the modular repo.

### 1. MAX's collectives are single-process by design

`max/kernels/src/comm/allreduce.mojo` (plus `reducescatter.mojo`,
`allgather.mojo`, `broadcast.mojo`) are pure-Mojo P2P kernels: each GPU gets
one kernel launch with its own `DeviceContext`, an array of *peer pointers*
into the other GPUs' buffers, and a per-rank `Signal` buffer
(`comm/sync.mojo`) used for cross-GPU thread-block barriers. There is no
rendezvous and no IPC handle exchange — peers are addressable only because
they live in the same address space. Even Modular's vendor-CCL escape hatch
(`comm/vendor/ccl.mojo`, compile-time gated) uses `ncclCommInitAll`, the
single-process NCCL variant.

Consequences:

- The classic `torchrun` one-process-per-GPU model **cannot** use these
  kernels. A single process driving all GPUs can.
- The kernels are tuned (1-stage / 2-stage / Lamport algorithms selected by a
  per-arch table covering H100/B200/MI300X/MI355X), AMD is first-class, and
  Modular's changelog claims they beat NCCL in benchmarks. This is the
  "similar performance as CUDA" cornerstone, and it costs us zero kernel
  writing.
- Constraints to accept: `MAX_GPUS = 8` per group, P2P required for the fast
  path (non-P2P falls back to a slow staged path), and a signal/staging buffer
  per GPU (`Signals.NUM_BYTES`, 257 MiB in the currently pinned nightly; the
  modular checkout has already bumped it to ~1 GiB — the Python-side constant
  must mirror the Mojo `Signal` struct layout exactly).
- The collectives are callable outside the MAX graph API from plain Mojo code:
  `max/kernels/test/gpu/comm/test_allreduce.mojo` launches them per-device
  directly, which is exactly the pattern our eager kernel extensions already
  use.

### 2. A pure-Python `ProcessGroup` is a first-class PyTorch citizen

`class MojoProcessGroup(torch.distributed.ProcessGroup)` in plain Python works
through a pybind trampoline already compiled into shipped libtorch
(`torch/csrc/distributed/c10d/PyProcessGroup.hpp`;
`test/distributed/test_c10d_pypg.py` runs real DDP on such groups). Register
with `Backend.register_backend("mojo", factory, devices=["mojo"])` and
`init_process_group(backend="mojo")` works.

DDP's C++ Reducer (`torch/csrc/distributed/c10d/reducer.cpp`) is
device-generic and already special-cases `is_privateuseone()` — bucketing,
comm hooks and futures all work with no C++ on our side. A synchronous PG is
correct out of the box; async `Work.get_future()` is what unlocks
compute/comm overlap. `Work` objects can be minted from
`torch.futures.Future` via `_create_work_from_future`.

### 3. PyTorch's generic stream machinery is inert for us — and we don't need it

The Python-only PrivateUse1 device guard
(`torch._C._acc.DeviceGuard`, used by
`_setup_privateuseone_for_python_backend`) forwards *only* `type()` to Python:
`deviceCount()` is hardcoded to 1, `setDevice` is a no-op, and
`getStream`/event methods return dummies. `torch.Stream`/`torch.Event` for the
mojo device are therefore no-ops, and fixing that requires a real C++
`DeviceGuardImpl` — which conflicts with the no-C++-extension rule.

The good news, from reading `ProcessGroupNCCL.cpp`: NCCL's overlap is *not*
PyTorch stream-API magic. It is simply "enqueue the collective on a second
device queue, order it with events, and make `Work.wait()` insert a GPU-side
stream-wait instead of blocking the host" (plus stashing tensor references so
the allocator can't recycle in-flight buffers). We can replicate all of that
inside our own Mojo extension: `DeviceContext.create_stream(priority=...)`,
`select_stream`, events, and `enqueue_wait_for` all exist at the Mojo level —
they are just not surfaced in Python `max.driver`.

Also ruled out: `torch.nn.parallel.DataParallel` (the old single-process
wrapper) is hard-locked to CUDA in C++ (`torch/csrc/cuda/comm.cpp`) — dead
end. Standard DDP brought into one process via thread-ranks replaces it.

## Current state of this repo (what exists / what's missing)

Exists and is directly reusable:

- `mojo:0..N-1` already address distinct physical `max.driver.Accelerator`s
  (`torch_mojo_tensor.py::find_equivalent_max_device`).
- Native ATen autograd, fused AdamW, foreach ops, per-device RNG — a
  single-GPU training step is solid to replicate per rank.
- The installed `max` package ships `driver.enable_all_peer_access()`,
  `Device.can_access()`, and one eager collective
  (`max._distributed_ops.distributed_broadcast`) — none wired up yet.

Missing (the work):

- No collectives, no `ProcessGroup`, nothing under `torch.distributed` for
  mojo.
- Cross-device copy bounces through host memory
  (`mojo_device_aten_ops.py::mojo_device__copy_from` does D2H→H2D for
  `mojo:0`→`mojo:1`), even though cross-context `enqueue_copy` D2D exists in
  MAX.
- `current_device` is a non-thread-local Python global.
- The wrapper's bookkeeping TensorImpl is pinned to `privateuseone:0` for
  *all* tensors; the autograd engine keys its worker threads on TensorImpl
  device index, so all replicas' backward passes would serialize onto one
  engine thread.

## Phased plan

### M0 — Multi-device groundwork (small, useful on its own)

- Call `enable_all_peer_access()` at device-module init when >1 GPU (guarded
  by `Device.can_access`).
- Replace the host bounce in cross-device `_copy_from` with a direct D2D
  `enqueue_copy` across contexts. This alone makes naive multi-GPU usable.
- Make `current_device` thread-local; make the wrapper TensorImpl device index
  match the real device (`privateuseone:i`) so the autograd engine gets one
  worker thread per device.
- **De-risk the dispatch budget now**: a microbench driving 2 GPUs from one
  process with a realistic training op mix, measuring whether host dispatch
  (~44 µs/op today) can keep N queues fed. This is the main threat to CUDA
  parity (see Risks).

### M1 — Eager collectives extension

- New `eager_kernels/comm_ops.mojo` importing
  `comm.allreduce`/`allgather`/`reducescatter`/`broadcast` from the modular
  repo (same import pattern as the existing `from nn import ...` kernels),
  following the launch pattern in `test_allreduce.mojo`: loop devices, launch
  per-device with peer pointer arrays.
- Signal-buffer lifecycle in Python: allocate one per GPU via the existing
  holder machinery, `init_signal_buffer` + synchronize once, cache per
  device-set (mirror `max/python/max/nn/comm/allreduce.py::Signals`).
- Python API: `torch_mojo_backend.distributed.all_reduce(tensors, op="sum")`
  etc., taking one tensor per device. Use the kernel's `output_lambda` to fuse
  the ÷world_size for gradient averaging.
- Benchmark with the nccl-tests methodology (Modular's `bench_allreduce.mojo`
  computes algbw/busbw the same way) against NCCL on the same box.

### M2 — `torch.distributed` integration (standard DDP UX)

- `MojoProcessGroup(dist.ProcessGroup)` +
  `Backend.register_backend("mojo", ..., devices=["mojo"])`, with a
  thread-per-rank runtime: N Python threads, each rank pinned to `mojo:i`;
  collectives rendezvous at a barrier and one launcher issues the single
  multi-device MAX collective; each rank gets a `Work` via
  `_create_work_from_future`.
- Implement `allreduce`, `broadcast`, `allgather`, `reduce_scatter`,
  `barrier`, `allreduce_coalesced` — DDP's full diet (param broadcast at
  construction, bucket allreduce, shape verification).
- Ship a small `torch_mojo_backend.distributed.spawn(fn, nprocs)` launcher
  (threads, not processes) so user code looks like standard DDP code.
- Correctness tests: PyTorch's threaded-PG harness patterns on 1 GPU (math +
  wiring); real tests gated on `accelerator_count() >= 2` (multi-GPU runs on
  MI300X-class hardware; MAX comm kernels support CDNA3 natively, and that is
  also where NCCL/RCCL baselines run).

### M3 — Overlap and parity tuning

- NCCL-style scheduling inside `comm_ops.mojo`: persistent high-priority comm
  stream per device; record an event on the default stream when a bucket is
  ready → comm stream waits on it → collective → end event; `Work.wait()`
  enqueues a default-stream wait on the end event and returns immediately (the
  host never blocks mid-backward).
- Lifetime: the `Work` object stashes Python references to bucket tensors
  until the end event completes. Since frees are stream-ordered on the default
  stream, ordering the free *after* the inserted stream-wait makes lifetime
  safe — same principle as NCCL's allocator stash.
- Tune DDP `bucket_cap_mb` against the allreduce tuning table (bucket size
  selects the algorithm regime; small buckets should hit the Lamport path).
- End-to-end benchmark: GPT-2/ResNet training step, 2–8 GPUs, vs `torchrun` +
  NCCL DDP on the same hardware. Target: scaling efficiency within a few
  percent of NCCL DDP, given single-GPU step parity.

### M4 — Later / optional

- **FSDP2**: it resolves `device_handle = torch.mojo` and needs
  `Stream`/`Event`/`current_stream` on that module — those can be *our*
  Python classes backed by MAX streams, dodging the C++ guard requirement.
  Substantial but no fundamental blocker.
- **Multi-node**: host-staged collectives (pinned D2H → gloo/CPU allreduce →
  H2D) as a correctness path; RDMA-class multi-node is out of scope until
  Modular ships transport.
- **Cross-process on one node** (only if the thread model binds on the GIL):
  CUDA-IPC-mapped staging buffers would let the same Mojo kernels work across
  processes. Real engineering — MAX exposes no IPC today; keep as research.

## Risks, in order of concern

1. **Host dispatch budget.** One process dispatches for N GPUs; at ~44 µs/op
   with the GIL, 8 replicas of a small model could starve the queues
   (single-GPU batch-1 decode is already CPU-bound). Training with real batch
   sizes has much fatter kernels, so it should amortize — the M0 microbench
   must confirm this before M2. Levers if it binds: keep shrinking per-op
   overhead, release the GIL inside the Mojo extension enqueue call so ranks'
   dispatch overlaps, free-threaded Python down the road.
2. **Backward exception behavior**: a raise from inside a native backward node
   aborts the process on this backend, and M2 runs N concurrent backward
   threads. Any collective error during backward must be reported without
   raising in a backward node (preflight + poisoned-future pattern in the PG).
3. **Signal struct layout coupling** to the nightly: the Python-side size
   constant must mirror the Mojo `Signal` struct (it already changed 257 MiB →
   ~1 GiB between the pinned and current nightlies). Pin and test on every MAX
   bump.
4. **8-GPU ceiling and P2P requirement** — fine for single-node
   DGX/MI300X-class boxes; document it. Non-P2P topologies fall back to the
   slow staged path.

## Why this can plausibly reach CUDA parity

- Per-GPU compute: single-GPU eager training kernels are already at or near
  CUDA parity for the models we benchmark; data parallelism multiplies that,
  it doesn't change it.
- Collective bandwidth: Modular's intra-node allreduce benchmarks at or above
  NCCL on the same fabrics (NVLink, XGMI), with the same busbw methodology.
- Overlap: the mechanism NCCL uses (side stream + events + GPU-side waits +
  reference stashing) is fully replicable with Mojo-level `DeviceContext`
  APIs; DDP's Reducer only needs futures that resolve.
- Nothing needs to be invented at the kernel level — the work is integration:
  P2P copies, a comm extension, a ProcessGroup, and stream/event scheduling.
