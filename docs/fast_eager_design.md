# Fast eager mode via Mojo Python extensions (proof of concept)

## Why

The mojo_device eager mode routes every ATen op through
`max.experimental.tensor`: each call builds a fresh MLIR graph, runs
MLIR cleanup passes, then interprets or compiles it. Measured on an
RTX 2000 Ada (WSL2), that costs **~2,200 µs per op call** regardless of
tensor size — a (64, 64) add costs the same as a (1024, 1024) one, all
fixed Python/MLIR overhead. Profiling one `add`: ~1 ms graph building,
~0.9 ms `finalize_graph` (MLIR passes), ~0.6 ms dtype promotion; the
actual kernel is noise. For reference, torch's own CUDA eager launch
overhead on the same machine is ~21 µs.

## What this PoC does

It takes the architecture of
[causal-conv1d-mojo](https://github.com/gabrieldemarmiesse/causal-conv1d-mojo)
— Mojo kernels compiled to CPython extension `.so` files, JIT-on-first-use
with a content-addressed disk cache — in the productized form that MAX
itself already uses internally for its eager interpreter
(`max/_interpreter_ops/*.mojo` + `mojo.importer`):

- `torch_mojo_backend/eager_kernels/elementwise_ops/elementwise_ops.mojo` implements
  Add/Sub/Mul/Div/Max/Min/Relu/Exp as Mojo kernels over **contiguous
  buffers with fully dynamic shapes**. Shapes and strides never enter the
  specialization key, so one `.so` serves every shape with zero
  recompilation. Dtype was a *runtime* dispatch inside Mojo in the
  original PoC; it is now a compile-time define (`DTYPE_ARG_0`,
  `DTYPE_OUT`, ...) so each `.so` carries exactly one instantiation —
  see "Compile granularity" below for the measurement that reversed that
  choice.
- Extensions are built by the loader in `eager_kernels/__init__.py`
  (`mojo build --emit shared-lib`, one gated entry point named `call`)
  and cached under
  `__mojocache__/<family>.<defines-hash>.hash-<source-hash>.so`; every
  later process just dlopens the hit. A build is **deferred to the first
  call that needs that exact specialization**, so `import
  torch_mojo_backend` and torch.compile-only workloads compile nothing,
  and neither does a process that never touches mojo_device eager mode.
  (The PoC used the `mojo.importer` hook; the loader replaced it when
  compilation moved to variant granularity — `mojo.importer` still serves
  `eager_flash_attention`.)
- Python-visible functions receive the `max.driver.Buffer` objects
  directly plus `device._device_context_ptr()`. The kernel is enqueued
  on **MAX's own DeviceContext** (same device queue the MAX driver
  uses), so ordering with copies/other MAX work needs no extra
  synchronization.
- `eager_kernels/aten_fast.py` wraps these kernels with
  ATen-compatible signatures. Supported inputs execute through Mojo
  extension calls; unsupported inputs raise an actionable
  `NotImplementedError` rather than entering the deleted graph fallback.
- `mojo_device_aten_ops.py` registers the eager implementations
  unconditionally.
  **The torch.compile backend is untouched.**

## Measured results (RTX 2000 Ada, WSL2)

Per-op call, (64, 64) float32 on the GPU mojo_device:

| path | µs/call |
|---|---|
| former graph-based eager (historical) | ~2,200 |
| fast path, bare extension call (launch only) | 6.5 |
| fast path, incl. out-alloc + wrap (MaxEagerTensor level) | ~20 |
| fast path, end-to-end `x + y` at the torch level | ~44 |
| torch native CUDA eager (reference) | ~21 |

End-to-end speedup today: **~50×**; the remaining ~24 µs over the bare
call is the PyTorch dispatcher + `TorchMojoTensor` wrapping + beartype,
which can be shaved independently.

## Compile granularity: one `.so` per variant, built in the background

This section used to be a "scaling analysis" arguing for compilation at
*module* granularity and against one `.so` per variant, on the grounds
that per-variant builds multiply total compile time ~40×. That
recommendation has been reversed by measurement and the shipped design
is the opposite one: **one `.so` per exact specialization, built on
demand in a background pool, with only the device launch waiting.**
The queue that makes the waiting invisible is documented in
[docs/kernel_call_queue.md](kernel_call_queue.md).

The old argument measured the wrong quantity — total compiler *work*
rather than the wall-clock a cold workload waits. Two things changed it:

- **Compile-time gates.** `variant_gates.mojo` reads `OP`, the
  `DTYPE_ARG_*` and the `DTYPE_OUT*` defines, and every registration and
  dtype-parametric launch sits behind a `comptime if`. A per-variant
  build therefore compiles roughly one kernel, not the family's ~150
  instantiations, so a variant build is a fraction of a module build
  rather than a repeat of it.
- **Nothing waits for the build.** Variant builds run on a background
  thread pool (`_ASYNC_BUILD_SLOTS`, sized by `_pool_size()`: available
  RAM / 5 GiB and cores / 3, capped at 16) while Python keeps
  discovering ops and requesting further compilations. Total compiler
  work went up; wall-clock went down, because the work is now parallel
  and overlapped with discovery.

### Measured (H100 PCIe, 24-core host; nanoGPT 124M, batch 12 × 1024, BF16)

Cold cache (empty `__mojocache__`):

| scenario | time |
|---|---|
| first training step, kernel-call queue + build pool (8 slots) | **11.6 s** |
| first training step, pool disabled (`TORCH_MOJO_BACKEND_KERNEL_QUEUE=0`) | 56.3 s |
| whole script, pre-PR module-granularity design | 8 min 34 s |

The third row is the whole script rather than one step, because on the old
design the module builds all land before the first step completes; it is the
"how long until anything happens on a fresh checkout" number, not a
step-for-step comparison with the first two rows.

Pool-size sweep on the same first step (slots → time): 1 → 56.6 s,
2 → 30.1 s, 4 → 16.5 s, 8 → 11.7 s, 16 → 10.5 s, 24 → 10.4 s. One slot
reproduces the pool-disabled number, as it should. Returns flatten past
~8 slots on this 24-core host, and memory rather than cores is the binding
constraint (each `mojo build` peaks around 4.5 GB RSS), which is what
`_pool_size()`'s two caps encode.

**The warm path was a regression; most of it has been recovered.** The
first measurement on this branch was **101 ms/step vs 60 ms on main**
(batch 12 × 1024). Three rounds of fixes brought it down (all H100 PCIe,
`bench_nanogpt_train.py`, warm `__mojocache__`, medians over 20–30 steady
steps, measured 2026-08-02):

| workload | main | branch before | branch after |
|---|---|---|---|
| batch 48 (default) | 193.8 ms | 195.1 ms | **193.6 ms — parity** |
| batch 12 (host-bound) | 60.0 ms | 66.5 ms | **63.6 ms** |

What was fixed, in order of measured size (py-spy at batch 12):

1. **Thread-switch device synchronize (~9 ms/step).** Rule 4 used to fully
   synchronize the device on *every* enqueuing-thread change, including
   pure direct launches — twice per training step (forward→backward,
   backward→optimizer). The barrier now applies only where the empirical
   hazard was ever observed: queue launches replaying another thread's
   work. Steady state performs no synchronizes at all.
2. **`torch.broadcast_shapes` twice per binary op (14.2 µs/call).** The
   eligibility probe and `expected_output_specs` both called it; equal
   shapes (residual adds, grad accumulation) now skip both.
3. **Defines re-canonicalization (2.4 µs/call).** Every `*SpecExtension`
   now memoizes its canonical defines per (op, dtypes) key, the same
   pattern `_CALL_DEFINES_CACHE` already used for `MojoFileExtension`.

The remaining ~3.6 ms/step at batch 12 (~6%) is the structural cost of
the Into ABI and the dispatch bracket, spread thin: Python-side output
allocation (+0.7 ms vs main's Mojo-side alloc), the per-op dispatch
bracket (+0.94 µs/op), `kernel_call_into` (+0.55 µs/launch), and
prepare/submit machinery (a warm binary add is ~13 µs of Python vs ~4 µs
on main). At batch 48 this is fully hidden under GPU time. The cold-start
numbers above still must not be read as an overall speedup.

### What still holds from the original analysis

- **Loading many `.so` files is cheap.** First extension load pays
  ~55 ms (shared Mojo runtime libraries); each additional extension
  costs **~0.07 ms and ~0.1 MB RSS**, lazily mapped. That is what makes
  one `.so` per variant affordable at load time despite there being many
  more of them than modules.
- **Sources stay grouped by family**, one directory per family under
  `eager_kernels/` (elementwise, reductions, matmul, data movement, ...),
  not one file per op — for shared helpers and reviewability. Grouping is
  now purely a source-organization decision: it no longer determines what
  a build compiles, since the defines gate everything else out.
- **Disk grows with the specializations a workload actually uses**, not
  with ATen coverage. For calibration, MAX's own interpreter op set is
  58 MB across 25 extensions, and the MAX wheel ships only the `.mojo`
  *sources* (verified: no `__mojocache__` entry in any dist-info
  RECORD) — that 58 MB is compiled locally on first import, 25 modules
  over ~10 min, the first time MAX's eager interpreter runs on a fresh
  venv. Compile-on-first-use is the deployment model MAX already
  imposes; we pay it in much smaller increments.
- **Caches are per-machine; don't plan on shipping them in wheels.**
  Modular doesn't ship prebuilt `__mojocache__/` (sources only), and
  for good reason: `mojo build` defaults to `-march=native` host
  codegen and auto-detects the GPU arch at build time, so a prebuilt
  `.so` can SIGILL on an older CPU or carry the wrong GPU target
  (causal-conv1d-mojo keys its cache by CPU brand + GPU arch for
  exactly this). Realistic options: per-machine compile on first use
  (what the loader does), an explicit warmup command for
  container/production images, and CI caching of `__mojocache__`
  keyed on source hash + toolchain version **+ runner hardware**.
  Caveat to document: the loader's cache key covers the source
  dependency closure, the canonical defines, the MAX/Mojo package
  versions, the CPython ABI and `platform.machine()` — but no CPU brand
  and, deliberately, no accelerator identity. Moving a venv/checkout
  between machines with different CPUs or GPUs can therefore load a
  stale-for-this-hardware `.so`; wipe `__mojocache__/` when that happens.

## Milestone 2: full models (resnet-18, gpt2) at CUDA-comparable latency

The op set was extended until `microsoft/resnet-18` and `gpt2` (stock
transformers models) run their full forward on the fast path. Measured on
an H100 PCIe (batch 1, fp32, end-to-end incl. bringing logits to host):

| model | graph-based eager (before) | fast eager | torch CUDA | ratio vs CUDA |
|---|---|---|---|---|
| resnet-18 | ~530 ms | **3.25 ms** | 3.03 ms | **0.93×** |
| gpt2 (6 tokens) | ~3,216 ms | **16.6 ms** | 6.75 ms | 2.5× |
| gpt2 (504 tokens) | — | **82.6 ms** | 74.2 ms | **1.11×** |

resnet-18 and long-sequence gpt2 are at CUDA parity. Short-sequence gpt2
is a pure dispatch-overhead benchmark (475 tiny-tensor aten calls); its
remaining gap is per-op Python cost plus materializing copies where CUDA
uses free strided views (transpose/split/kv-cat) — closing it needs
stride-aware tensors or C++-level registrations, not faster kernels.

Both models pass `run_hf_mojo.py`'s comparison against CPU with argmax
agreement. Numerics for float32: matmul/conv call cuBLAS with
`use_tf32=False` (full fp32, matching torch's CUDA matmul default and
*more* precise than the graph path, whose matmul dispatch hardcodes TF32).
For float16/bfloat16 the fast kernels deliberately diverge from the graph
path where torch itself does: scalar ops / exp / tanh / batch norm
accumulate in float32 (adversarial review verified the fast results match
real torch closely while the graph path's native-fp16 arithmetic is the
outlier, e.g. exp off by ~0.5% relative), `native_layer_norm` returns
float32 mean/rstd like torch CUDA, `view` aliases storage like torch
(the graph path copies), and `max_pool2d_with_indices` returns real
indices (the graph path duplicates the values).

### How the op set is organized

Each lazily compiled entry point lives in its own directory under
`eager_kernels/`: `<family>/<family>.mojo` sits beside
`<family>/<family>.py`, which owns that file's `MojoExtension` descriptor.
Family-private Mojo helpers live in the same directory; helpers shared by
multiple families and the `op_utils` Mojo package remain at the package root.
A family source file is the compilation *input*, not the compilation unit:
each `.so` built from it is gated down to one operation and one dtype tuple
(see "Compile granularity" above and
[docs/kernel_call_queue.md](kernel_call_queue.md)).

- `elementwise_ops.mojo` — binary/unary ops + Python-scalar variants
  (`x * 0.5`, `x ** 3`, int `x + 1`), tanh; contiguous, dtypes selected by
  compile-time define.
- `nn_ops.mojo` — batch-norm inference (NCHW), layer-norm (last dim, also
  emits float32 mean/rstd like `aten.native_layer_norm`), row softmax with
  fused scale + causal mask, trailing-dims mean, max-pool2d with torch
  indices, embedding gather, bool `all()`. All hand-rolled as parallel-for
  kernels (one task per output element/row), CPU + GPU.
- `data_movement_ops.mojo` — permute-copy (rank ≤ 4), narrow-copy
  (split/slice), dtype cast. Dispatch on element *size*, not dtype.
- `matmul_ops.mojo` — hand-written pure-Mojo GEMM kernels since
  Milestone 3 (see below); originally bound `linalg`'s vendor cuBLAS
  path, which is still exported as `MatmulVendor`/`BmmVendor` for A/B
  benchmarking only. The precompiled kernel packages (`.mojoc` under
  `modular/lib/mojo/`) are importable from any `mojo.importer` module
  out of the box — the importer sets `MODULAR_MOJO_MAX_IMPORT_PATH` on
  every `mojo build`.
- `conv_ops.mojo` — batched im2col + the pure GEMM since Milestone 3,
  arranged so the torch `(K,C,R,S)` weight is used as-is (viewed
  `(K, C·R·S)`) and the matmul output is already NCHW — zero layout
  permutes, and 1×1 stride-1 convs skip im2col entirely (pure matmul on
  buffer views). Originally batch>1 or grouped convs fell back to
  `nn.conv.conv_gpu` with `filter_is_fcrs=True`, which routes to cuDNN.

Some ops need no kernel at all (Python-only fast paths):

- `view` / `_unsafe_view` / `unsqueeze` alias the driver buffer via
  `Buffer.view` (zero copy — and unlike the graph path, actually matches
  torch's aliasing semantics for in-place ops after a view).
- `empty` / `empty_strided` are a bare `driver.Buffer` allocation (the old
  path launched a zeros kernel through the graph).
- `_to_copy` to CPU uses a driver-level D2H copy (`Buffer.to_numpy`),
  stream-ordered with the enqueued kernels — the graph-based
  `Tensor.to(CPU())` costs a flat ~2.2 ms.
- `arange` builds on host with exact torch semantics and does one H2D copy.
- `cat` where all-but-one input is the legacy 1-D empty (uninitialized KV
  caches) is a single narrow-copy.
- `scaled_dot_product_attention` decomposes into `bmm(q, kᵀ)` → fused
  scale+causal softmax → `bmm(probs, v)` on buffer views (4 enqueues,
  ~186 ms/call on the graph path → ~50 µs).

### How the per-op overhead was brought to CUDA level

Three structural changes, in order of impact:

1. **Fast ops receive `TorchMojoTensor` arguments directly** and return
   wrapped results (`aten_fast.NOT_HANDLED` sentinel triggers the generic
   fallback), skipping the recursive argument-conversion walk both ways.
2. **`TorchMojoTensor` stores the raw driver buffer** (`_from_buffer`) and
   only builds the `MaxEagerTensor` wrapper lazily on first `_max_data`
   access. Fast-path tensors that only ever feed other fast ops (the vast
   majority) never construct one (~1.7 µs + a sharding-mesh init each);
   slow-path fallbacks materialize it on demand.
3. **Hot functions opt out of beartype** with `@typing.no_type_check`
   (the claw hook honors it), and the device lookup is `functools.cache`d.

Resulting per-op end-to-end costs at the torch level (H100 box): view
~8 µs, relu ~10 µs, addmm ~20 µs, conv ~35 µs — at which point resnet-18
matches torch CUDA. The bare `linalg` vendor matmul call is 8.8 µs.
Remaining short-gpt2 overhead: two unavoidable sync points (the HF mask
`.all().item()` check and the final D2H) each drain the ~600-kernel-deep
queue, and transpose/split/cat run as real copy kernels where CUDA has
zero-cost strided views.

### Gotchas discovered (worth keeping in mind)

- `PythonModuleBuilder.def_function` supports at most **8 positional
  args**; pack extra scalars into a tuple (the interpreter does the same).
- A GPU kernel closure must only capture **parameters of the enclosing
  generic function** (plus pointers via `@__copy_capture`). Capturing a
  dispatcher-level `var` produced garbage on GPU (the `AllBool` bug: reads
  of a captured size ran off the buffer).
- `torch.library.impl("aten::mean", ...)` only covers the *default*
  overload: `mean.dim` silently decomposed into a chain of graph-path
  sum/div ops (~12 ms). Overloads must be registered explicitly.
- torch's `is_causal=True` means the **top-left aligned**
  `tril(ones(L, S))` mask, not the bottom-right alignment generation code
  usually wants.
- The runtime `DType` value has no size accessor in current nightlies —
  map dtypes to like-sized unsigned ints by hand for copy kernels.
- MAX's matmul dispatchers (`_matmul_gpu`, `matmul_vendor` wrapper) force
  TF32 for fp32 (max abs diff ~3e-2 vs CPU at K=768), and the graph path
  inherits that. Calling `linalg.matmul.vendor.blas.matmul` directly with
  its `use_tf32=False` default gives full-fp32 cuBLAS GEMM — and measured
  *lower* per-call overhead (8.8 µs vs 17 µs) than going through the
  dispatch heuristics.

## Milestone 3: pure-Mojo GEMM/conv — GPU eager mode with zero NVIDIA libraries

### Why

MAX's own kernels quietly depend on NVIDIA userspace libraries in exactly
the situations eager mode produces. The GPU matmul dispatch
(`linalg/matmul/gpu`) only engages Modular's native kernels when N/K are
compile-time constants (`has_static_NK`) *and* `transpose_b=True` — on
every NVIDIA arch including Blackwell. Eager ops have runtime shapes and
`transpose_b=False`, so every `m>1, n>1` matmul lands on the cuBLAS
fallback (only gemv and a naive kernel are pure Mojo). Convolution routes
to cuDNN whenever the filter is in torch's FCRS layout, which is what our
fast path uses. Neither library ships with MAX: they are found by
dlopen-by-soname only because *CUDA torch* loads them into the process
from its `nvidia/*` wheels. A pure-MAX process — or one with CPU-only
torch — hard-aborts (`std/ffi`: `symbol not found: cublasCreate_v2`,
uncatchable) on the first matmul. `MODULAR_DISABLE_VENDOR_FALLBACK` is a
compile-time `-D` define and is not honored by the runtime JIT.

The goal: run the GPU eager mode against a **CPU-only torch wheel** —
~200 MB installed instead of ~4 GB of CUDA torch + NVIDIA wheels, no
CUDA-version matching — with nothing but the kernel driver
(`libcuda.so.1`) on the NVIDIA side.

### What replaced cuBLAS/cuDNN

Three GEMM kernels in `matmul_ops.mojo`, all dynamic-shape, exact fp32
(FFMA, no TF32), fp32 accumulation, batched via `grid.z`, with edge
guards on every dimension:

- **`_gemm_pipe_kernel`** (2-stage) — 128×128 block tile, 8×8 register
  tile per thread, 256 threads. A is staged through registers and stored
  transposed in shared memory so compute reads both fragments as vectors;
  B goes straight to shared memory via `cp.async`. Used for compute-bound
  shapes (grid ≥ one wave) and all `transpose_b=True` shapes.
- **`_gemm_pipe3_kernel`** (4-stage, `cp.async` only) — 64×64 and 64×32
  tiles. Both operands are fetched with `cp.async` two-to-three slabs
  ahead of compute — enough in-flight distance to cover L2 latency on the
  L2-resident deep-K shapes convolution lowers to (A is stored row-major
  since `cp.async` cannot transpose; compute reads it as per-row scalar
  broadcasts, which warp-level broadcast makes free).
- **`_gemm_smallm_kernel`** — for `m ≤ 8` (skinny transformer GEMMs,
  gemv): one thread per output column streaming B at full bandwidth with
  8 row accumulators, k-unrolled ×4 for memory-level parallelism.

Plus **split-K with a workspace**: when the natural grid is under ~half a
wave (K-rich/MN-poor shapes like `512×49 @ k=4608`), the dispatch splits
K across up to 32 grid.z slices targeting ~3 blocks/SM. Each split writes
its own `[m, n]` slice of a stream-ordered scratch buffer
(`ctx.enqueue_create_buffer`), and a trivial reduce kernel sums the
slices into C — deterministic, no atomics. Convolution reuses all of
this: batched im2col (`(N, C·KH·KW, OH·OW)`, channel-major rows) feeds a
shared-A `Bmm` (weights broadcast across the batch), and grouped convs
run one GEMM per (sample, group) using element offsets into the same
buffers.

### Measured results (H100 PCIe, fp32)

Microbenchmark over all 22 shapes resnet-18/gpt2 actually produce plus
2048²/4096² references, against exact-fp32 cuBLAS in the same harness:
**geomean 1.02×**. Several shapes are faster than cuBLAS (skinny lm-head
0.80×, batched attention matmuls 0.12–0.32× — the vendor's strided-batch
dispatch is poor there); the only remaining >1.4× shapes are the big
compute-bound GEMMs (squares and the 504-token lm-head at ~10–12 TF/s vs
cuBLAS ~15–17 TF/s FFMA).

End-to-end (batch 1, incl. D2H), CUDA-torch environment:

| model | pure Mojo | torch CUDA | cuBLAS-backed fast path (M2) |
|---|---|---|---|
| resnet-18 | **3.43 ms** | 3.01 ms | 3.25 ms |
| gpt2 (7 tokens) | **14.3 ms** | 6.7 ms | 16.6 ms |
| gpt2 (504 tokens) | **70.4 ms** | 73.3 ms | 82.6 ms |

gpt2 at 504 tokens is now *faster than torch CUDA*, and both gpt2 rows
improved over the cuBLAS-backed path — the cached-DeviceFunction enqueue
(~2 µs) is cheaper than the vendor call's per-call handle lookup +
`cublasSetStream` (~9–16 µs).

The demo itself — `torch==2.11.0+cpu` from the PyTorch CPU index, zero
NVIDIA packages on disk, no CUDA toolkit:

```
resnet-18  MAX GPU: 3.42 ms | torch CPU: 718 ms  -> 210x
gpt2-504   MAX GPU: 70.3 ms | torch CPU: 1761 ms -> 25x
NVIDIA userspace libs mapped in the process: []
```

Same GPU numbers as with CUDA torch, correctness preserved (resnet
max_abs 3.8e-6 vs CPU, argmax match on both models).

### Performance findings (ncu-driven, worth remembering)

- **`ctx.enqueue_function[func]` re-runs `compile_function` on every
  call** (`device_context.mojo`), which costs ~60–180 µs for large
  kernels even when the runtime's module cache hits. Compile once and
  cache the `DeviceFunction` in the process-global registry
  (`_get_global_or_null` / `KGEN_CompilerRT_InsertGlobal` — the same
  pattern the vendor BLAS handle uses): `op_utils._enqueue_cached`
  brings enqueue to ~2 µs. (When benchmarking enqueue cost, beware
  stream backpressure: past ~1000 queued launches the enqueue rate
  degrades to the kernel rate and masquerades as CPU cost.)
- **Fully comptime-unrolled GEMM inner loops explode register pressure**
  (~190 regs/thread → 1 block/SM → 12.5% occupancy). Keep the K-slab
  loop a runtime loop (register-tile indexing stays comptime) and set
  launch bounds via
  `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=..., `nvvm.minctasm`=...)`
  — the Mojo equivalent of `__launch_bounds__(threads, min_blocks)`;
  ptxas then fits the register budget (128/thread at 2 blocks/SM).
- **`Atomic.fetch_add` defaults to sequentially-consistent ordering**,
  which serializes the memory system (a split-K epilogue got 30–800×
  slower). Even `Ordering.RELAXED` atomics serialize per-address on
  small outputs (22 splits × 32K-element tiles ≈ 150 µs of pure atomic
  traffic). The workspace + reduce-kernel design is 3–10× faster on
  deep-K shapes, needs no zero-fill pass, and is deterministic.
- **`cp.async` requires size-aligned source addresses.** Split-K chunk
  starts must be rounded to a BK multiple or 16-byte copies silently
  corrupt (relerr ~1.0 on any shape whose K-chunk wasn't a multiple of
  BK). The `fill=0` + `src_size` form handles all shape edges in one
  instruction; 4-byte chunks are the fallback when a leading dimension
  isn't divisible by 4.
- **Latency-bound vs compute-bound shapes want different kernels.**
  Deep-K, small-MN shapes (im2col output) are L2-resident and
  long-scoreboard-stalled: they want thin tiles, deep `cp.async`
  pipelines, and split-K for block count. Big GEMMs want fat tiles and
  vector fragment loads. One kernel tuned for both loses ~30% on each.

### Remaining gaps

- Big compute-bound GEMMs sit at ~65–75% of cuBLAS's FFMA throughput
  (fragment double-buffering in the inner loop and smarter block
  swizzling for L2 are the known next steps).
- float16/bfloat16 matmuls still use the simple single-buffer tiled
  kernel (fp32 accumulation) without split-K — fine for the fallback
  role they play today.
- `conv_transpose` upstream is cuDNN-only in MAX; nothing on the fast
  path uses it.

## MI300X GPT-2 dynamic-batch decode

The gfx942 eager path has pure-Mojo FP32 MFMA schedules for GPT-2's five
decode GEMMs, a fused single-query attention kernel with strided K/V support,
and a 512-thread vocabulary argmax. The GEMM output width, reduction
dimension, and MFMA tile are compile-time constants, but the flattened batch
dimension is runtime dynamic. Consequently, changing the decode batch does
not compile a new kernel. Partial M/N tiles are guarded, including GPT-2's
50,257-column vocabulary head and non-tile-aligned batches.

These results use the unchanged Hugging Face GPT-2 model: the backend only
optimizes the eager ATen operations it receives. Measured on one MI300X with
ROCm 7.2 and torch 2.11, using FP32 greedy generation of 200 fixed tokens
(two warmups; seven measured Mojo runs):

| batch | path | time | aggregate tokens/s | relative to ROCm |
|---:|---|---:|---:|---:|
| 512 | torch ROCm eager (rocBLAS) | 2.017 s | 50,758 | 1.00x |
| 512 | Mojo device eager | 2.064 s | 49,601 | 0.98x |

Batches 64, 257, and 512 are tested through the same compiled MFMA schedule;
257 exercises the runtime M edge guards. The implementation does not attach
operation state or reserved-capacity metadata to tensors. The eager path was
also verified with a `torch==2.11.0+cpu` install, and the generated Mojo matmul
extension has no rocBLAS/hipBLAS dependency.

## Follow-up work (not in this PoC)

- Scalar (tensor ⊕ python number) variants — one extra kernel per op
  family, removes the most common fallback.
- Broadcasting: either fall back (today), or add a stride-aware kernel
  (see `ManagedTensorSlice`/`elementwise` with `IndexList` shapes, or
  expanded strides with 0-stride broadcast dims, MAX_RANK-bounded like
  MAX's interpreter uses `MAX_RANK = 5`).
- Matmul & friends: `linalg`/`nn` Mojo packages (the same kernel
  library the MAX graph compiler uses, including cuBLAS bindings) are
  importable from these modules — coverage does not require rewriting
  kernels.
- In-place variants (`add_`, `relu_`) are trivial: write to the input
  buffer.
- First-build UX for the test suite: the loader takes a per-identity
  `flock`, so `pytest -n` workers no longer race on one variant, but they
  still each wait on it; warming the cache once up front is still faster.
- Shave the remaining per-call overhead (TorchMojoTensor creation goes
  through a meta tensor + `__init__` per op; beartype on hot wrappers),
  which is where the warm-path regression recorded under "Compile
  granularity" is being tracked down.
