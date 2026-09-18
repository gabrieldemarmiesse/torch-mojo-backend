# The native mojo device

The `mojo` torch device is a real PrivateUse1 backend: tensors are ordinary
`torch.Tensor`s whose storage is MAX device memory, and every aten op the
device supports is a Mojo function that torch's C++ dispatcher calls
directly. No Python runs on the op path.

```
torch.add(a, b)  ->  dispatcher  ->  MojoBoxedKernel (C++, native/csrc)
                                         |  TmbValue records
                                         v
                                  op_add_tensor (Mojo, libtmb_backend, native/mojo/ops_*.mojo)
                                         |  KernelCall
                                         v
                                  logic_ops.so::tmb_call  (built on first use)
```

The shim, the backend library and every op body are fixed-size and ship
prebuilt in the wheel; the kernel is one `mojo build` that happens at its
first call and is then cached on disk (see "Two builds" below).

## Two builds

Everything is compiled on demand into the user's cache directory —
`~/.cache/torch-mojo-backend/native/` on Linux (`XDG_CACHE_HOME` honored),
`~/Library/Caches/torch-mojo-backend/native/` on macOS — so the builds
outlive the venv and the checkout; `TORCH_MOJO_BACKEND_CACHE_DIR` moves the
cache (it is contents-addressed so several checkouts share it, and nothing
ever reaps it: a torch/mojo/max upgrade orphans every entry, so
`torch-mojo-backend cache clean` wipes the directory when it grows, and
`torch-mojo-backend cache dir` prints it). Each build is keyed by the hash of every
source it compiles in, the toolchain versions and its `-D` defines.

| build | when | what it holds |
|---|---|---|
| C++ shim, Mojo backend | first `register_mojo_devices()`, unless the wheel ships them (see "Prebuilt libraries and the wheel") | the runtime and every op body, both fixed-size |
| one kernel family specialization | that kernel's first call | one (OP, dtypes, flags) kernel |

The backend library holds devices, streams, events, memory, the loader, the
record ABI and every op body, registered eagerly at `tmb_native_init`.
Measured on a 22-core box, 274 registered ops: 4.6 MB, about 50 s to build
with a cold compiler module cache and about 10 s warm, against 0.4 MB and
16 s / 9 s for the runtime alone. An earlier revision kept the op bodies out
and built each one alone at its first call (6–7 s per op, about 22 minutes
for all of them); one build of 50 s that ships prebuilt replaces that. The
library still carries no device code (kernels are built per specialization
below), so it is one file per platform.

Every `mojo build` subprocess runs with `MODULAR_HOME` on node-local disk
(`native.compiler_env`, `loader.mojo`'s `_compiler_env`; an explicit value
wins). That is where the *compiler* keeps its own module cache: its default
sits in `$HOME`, which on a cluster is one NFS directory shared by every
node, and concurrent compilers evict each other's entries there — "failed to
produce an archive for the module: No such file or directory". Node-local,
the first build on a machine pays about 25 s to fill it and nothing else
touches it.

The two shims of the first row:

**C++ shim (`native/csrc/`)** — the c10 objects torch only accepts as C++
classes, each forwarding to a Mojo function pointer:

| file | what |
|---|---|
| `shim_dispatch.cpp` | `tmb_library_impl`: registers a Mojo function as a boxed kernel. `MojoBoxedKernel` converts the IValue stack to `TmbValue` records (Scalar included, no heap boxing) and back. `tmb_call_op` calls any aten op from Mojo. |
| `shim_runtime.cpp` | allocator (`c10::DeviceAllocator` forwarding allocation and memory APIs to Mojo), `PrivateUse1HooksInterface`, the device guard (devices/streams/events), the Philox generator, `ProfilerStubs`, the tensor C API (`tmb_tensor_*`, `tmb_empty_strided`, `tmb_as_strided`). The current device and per-device current stream are C++ thread-locals (`tmb_current_device/stream`). |
| `shim_autocast.cpp` | `AutocastPrivateUse1` as one boxed fallback with a policy table filled from torch's own CUDA op lists. |

Three translation units compile in parallel: about 7 s wall cold.

**Mojo backend (`native/mojo/`)** — `mojo build backend.mojo --emit shared-lib`:

| file | what |
|---|---|
| `backend.mojo` | `tmb_native_init`: hooks table + the registration list (one `_group[register_x]` per ops file) |
| `registry.mojo` | `impl[op, "name"]`: registers the op's boxed entry with torch's dispatcher — see below |
| `abi.mojo` | `Value` records, tag constants, `T` (tensor view), result setters, `new_tensor` / `view_strided`, `unsupported()` |
| `device.mojo` | `Dev` per mojo index (accelerators, then the MAX CPU device), cached MAX properties, stream views, events (MAX events for ordering, vendor driver for query/timing), memory (`Buf` boxes behind DataPtr, per-device accounting, `record_stream` fences), transfers and deferred host staging |
| `vendor.mojo` | CUDA / HIP driver calls on MAX's raw streams |
| `loader.mojo` | on-demand builds of kernel families: closure hash, cache lookup, `mojo build` in a subprocess under a flock, dlopen |
| `kernels.mojo` | `KernelCall`: defines + slots + owned specs for one kernel invocation |
| `ops_*.mojo` | the aten ops, and each file's `register_<group>` list |

## Which ptxas assembles the kernels (NVIDIA)

Every one of those builds ends in ptxas, and which one runs is not a detail:
a cubin is only loadable by a driver at least as new as the toolkit that
assembled it, and only buildable by a toolkit that knows the architecture. So
the choice is pinned between two moving bounds, and `_ptxas.py` picks inside
them:

| bound | set by | what a violation looks like |
|---|---|---|
| upper | the driver | `CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)` — at the *first op*, after every build succeeded |
| lower | the GPU | `ptxas fatal : Value 'sm_90a' is not defined for option 'gpu-name'`, inside a `mojo build` dump |

Within a CUDA major the upper bound is loose (minor version compatibility:
any 12.x cubin loads on r525+, any 13.x on r580+), so it is the major that
decides. The lower bound is not monotonic — CUDA 13 added sm_110 and dropped
everything below sm_75 and sm_101 — which is why "newest wins" is wrong and
the candidates are asked what they target (`ptxas --help`) rather than
assumed to be ordered.

MAX bundles a CUDA 13 assembler, so on an r570 driver it refuses to create a
device at all unless `MODULAR_NVPTX_COMPILER_PATH` names another one. The
package sets that variable when it finds a suitable assembler: at import
from the driver alone (`cuDriverGetVersion`
needs no `cuInit`, so it is safe before `max` loads and before any fork), then
again at `register_mojo_devices()` from the architecture, which only the
initialized driver can answer. Candidates are the `nvidia-cuda-nvcc*` wheels
(`nvidia/cuda_nvcc/bin/ptxas` for cu12, `nvidia/cu13/bin/ptxas` for CUDA 13),
torch's `torch/bin/ptxas`, Triton's, `$CUDA_HOME`, `$PATH` and
`/usr/local/cuda*`, plus MAX's own compiler, which runs when the variable is
unset. The `max-core` wheel ships it as `modular/lib/libNVPTX.so`.
We query its `nvPTXCompilerGetVersion` API through `ctypes` to read the actual
CUDA major.minor version, without importing MAX or initializing CUDA.
Its targets come from the architecture tables rather than a `--help`.
Among known versions that fit the driver and every GPU present, the newest
wins. If the library or its version API is unavailable, or the query fails,
MAX's compiler has the lowest priority: it is tried only when no known
assembler fits. The report
labels it as an unknown CUDA version, and MAX checks compatibility at runtime.
Picking the built-in means *unsetting* the variable (the mark then
reads `<max built-in>`). The nvcc wheel is optional at runtime and pinned only
in the development dependencies in `pyproject.toml`; it is one candidate
among the others.

A child process inherits the environment and nothing else, so the pick is
marked in it too (`TORCH_MOJO_BACKEND_PTXAS_AUTO`): without that, every
torchrun rank and every Inductor compile worker would read the inherited
value as a setting of the user's and refuse to move off it for the GPU it
actually has.

When nothing fits, registration raises before the first build with the
candidate table, the reason each one was rejected and the wheel to install;
`torch-mojo-backend ptxas` prints the same table on demand, and
`TORCH_MOJO_BACKEND_PTXAS_CHECK=0` downgrades the refusal to a warning for a
machine whose rules we got wrong. A `MODULAR_NVPTX_COMPILER_PATH` the user
set is never overridden — only explained, if it cannot work here.

## Registration: how an op body reaches the dispatcher

Every op body is compiled into the backend library. `tmb_native_init` walks
the `register_<group>` list of each group file, and each
`impl[op, "name"](site)` line hands `tmb_library_impl` the address of
`abi.op_entry[op]`, the boxed entry around that op, so torch calls the Mojo
function directly from the first call on. A kernel that declines its inputs
raises `NotImplementedError`.

## Kernel families: the C entry

Every family under `eager_kernels/<family>/` exports one C function per
specialization build:

```mojo
@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32
```

`argv` is an array of 64-bit **slots** (`op_utils.Arg`): ints and pointers
are the value, floats their bit pattern (`_raw_f64`), a tuple the address of
an `[len, e0, ...]` Int array, a TensorSpec its address. The `comptime if
_op_on["Name"]()` ladder picks the one op compiled into the build and calls
its `_spec_dispatcherN[go, "Name"]`; a raised `Error` comes back as
`(rc=1, message)`.

## Writing an op

An op is `def op_x(args: Values, n_args: Int, rets: Values, n_rets: Int)
raises` in one `ops_<group>.mojo`, registered at the bottom of that same file
in `register_<group>` with `impl[op_x, "x.overload"](site)`. A new group
file needs two things: the `register_<group>` list, and one
`_group[register_<group>](lib)` line in `backend.mojo`.

External operator namespaces use their own `tmb_library_new` handle and
fully qualified registration names, such as `torchvision::roi_align`.
The dispatcher accepts these implementations before the extension defining
their schemas is imported. Torchvision remains optional at runtime.

The native detection groups are `ops_roi.mojo` (ROI align/pool, their
position-sensitive variants, and backwards), `ops_nms.mojo` (non-maximum
suppression), and `ops_deform_conv.mojo` (deformable convolution). They support
float16/float32/float64 GPU inputs (float64 requires device support). ROI
inputs are made contiguous; NMS declines non-contiguous inputs. ROI backward
uses relaxed atomic scatter and
honors PyTorch's deterministic-algorithms error/warning policy. NMS uses a
stable device sort, device IoU masks, a host greedy pass, and device index
gathering.

Torchvision 0.26 CUDA autocast policies are included in the shim's generated
table: NMS casts eligible inputs to float32 and returns int64 indices; ROI
align/pool, PS-ROI align/pool, and deformable convolution compute in float32
and restore the input dtype. As in upstream 0.26, autocast also converts ROI
pool's argmax and PS-ROI's channel mapping; normal ROI/PS-ROI dispatch keeps
these in int32. Float64 inputs are not narrowed.

Deformable convolution composes deformable im2col with the existing Mojo
GEMM routes. It supports independent convolution and offset groups, optional
mask/bias, and gradients for input, weight, offset, mask, and bias. Its input
gradient scatter follows upstream's nondeterminism policy. No vendor BLAS
library is required.

Arithmetic follows torchvision 0.26 CUDA, including its intermediate rounding:

- NMS rounds half intersection widths/heights and each box's height difference
  in half, then computes areas and IoU in float32. Double boxes retain double
  IoU arithmetic. Every dtype uses a float32 threshold and strict `>`.
- ROI align/pool and PS ROI align/pool use the input dtype for scale, geometry,
  interpolation, pooling, and gradient contributions. Half products round
  before additions; host scales round through float32 before half, matching
  `c10::Half(float)`. Backward accumulates into the input dtype. ROI pool uses
  ties-away coordinate rounding; PS pool follows CUDA's `roundf`, including
  its float32 conversion for double coordinates.
- Deformable convolution uses the input dtype for coordinates, interpolation,
  mask products, and offset/mask gradient accumulation. Input-gradient weights
  follow CUDA's promoted `std::abs` expression before rounding to the input
  dtype. GEMM and bias reduction accumulate half inputs in float32; their
  stored results are half. At large half coordinates, input gradients retain
  CUDA's three-neighbor scan where adjacent integer indices round together.
  Double arithmetic remains double.

CUDA is the oracle where CPU differs: CPU NMS has no half kernel and compares
against the original double threshold; CPU PS pool uses `round`, not `roundf`.
CPU and CUDA can also differ in half gradients through accumulation order.
Parallel backward scatter has CUDA's nondeterministic accumulation order, so
general gradients need numeric comparison; the boundary regressions use
order-independent exact comparisons.

The CUDA-reference tests in `tests/native/test_torchvision_ops.py` accept
`TORCHVISION_CUDA_REFERENCE_PYTHON=/path/to/cuda-venv/bin/python`. They try that
interpreter first, then the current interpreter, `python`/`python3` on `PATH`,
and `.venv-cuda`/`torch_cu*` environments in the working directory and its parent.
Each candidate must import torch/torchvision, provide the detection CUDA kernels,
and execute on a CUDA GPU. If none works, the skip reason lists each interpreter
and its failure. Once selected, reference execution failures fail the test.

Read arguments with the `v_*` helpers by schema position, build outputs with
`new_tensor` / `new_like` / `view_strided`, set results with `ret_tensor`
(owned output), `ret_ref` (an input handed back: in-place ops),
`ret_tensor_list`, `ret_scalar_*`.

To run a kernel:

```mojo
var ctx = ctx_for(t.device)          # the device's CURRENT stream
var cp = ctx_ptr(ctx)
var call = KernelCall("logic_ops", "AddSpec")
call.arg_dtype(0, a.dtype)
call.arg_dtype(1, b.dtype)
call.out_dtype(dst.dtype)
call.spec(a.spec(cp)); call.spec(b.spec(cp)); call.spec(dst.spec(cp))
call.run()
_ = ctx
```

**Lifetimes.** Mojo destroys a value right after its last use. Never take a
local's address, hand it over as an Int and let the local die before the
call: the slot then reads freed memory (this produced the first segfault of
this backend). `KernelCall` owns specs, tuples and slots until `run()`
returns; keep the `DeviceContext` alive with `_ = ctx` after the call.
The same rule bites through `Owned`: reading `held.t` copies a *non-owning*
`T` out, so `f(held.t)` is `held`'s last use and the tensor can be released
-- its storage freed, stream-ordered at that earlier point in the stream, so
the next allocation may legally take the block -- before `f` launches
anything. Write `_ = held^` after the consuming call.

**`out=` variants.** `check_out(dest, like)` first: torch's generated
`resize_out` requires the caller's tensor to ALREADY carry the result's dtype
and device and never casts, and checking before any kernel makes a bad `out`
free. `like` is the input the structured meta function takes its
`TensorOptions` from, which is not always argument 0 (`mm` uses `self`, `bmm`
uses `mat2`, `addmm` uses `mat1`). Then `resize_out` when only the shape
differs -- a correctly shaped `out` keeps its own strides and storage offset
-- then copy into it and `ret_ref` it back.

**Declining.** `unsupported("why")` raises with a prefix the entry turns into
rc 2 = `NotImplementedError` in Python; any other `Error` is a
`RuntimeError`. A kernel that declines its inputs raises inside the family;
the op decides whether to try another route or propagate.

**Errors from C.** Every `tmb_*` call returning `int32_t` goes through
`check(rc, "what")`, which appends the shim's thread-local message.

**Streams.** Ops launch on the device's current stream (`ctx_for`), so
`with torch.Stream(...)` really moves execution. Memory is allocated on the
current stream; callers using a tensor on another stream must record that
use (`Tensor.record_stream`, or torch's internal `recordDataPtrOnStream`),
which the backend turns into an event the owner stream waits on before
the buffer is released.

### Transfers

`.to("mojo:j")` and `dst.copy_(src)` automatically use MAX
`DeviceBuffer.enqueue_copy_from` for CUDA/HIP peer-capable pairs. Peer access
is enabled lazily per ordered pair; success and failure are cached under the
shim mutex. CPU, Metal, inaccessible pairs, and enable errors use host staging.
ROCm correctness and performance remain unmeasured.

Direct copies run on the destination's current stream, with MAX events in
both directions and no host completion wait for either `non_blocking` value.
Destination consumers are ordered after the copy; `.cpu()` and `.item()` wait
for readback. Callers must order producers on unrelated streams.

Original source, staging, and destination storage are recorded on their own
device's current stream. At release, allocation-owner streams wait for those
streams; MAX's reverse event fences the remote source read. Transfer errors
drain both streams before release. A failed drain retains both devices'
allocations until exit; pinned staging is retained unless completion is known.

`.to` borrows contiguous, unchanged-dtype sources, otherwise packs/casts on
the source, then restores destination memory format. `copy_` packs and moves
before casting or copying into destination strides. Dtype pairs outside the
fast cast kernel use CPU torch to preserve exact integer conversions.

`TORCH_MOJO_BACKEND_TEST_PEER_COPY` and `TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD`
are test-only hooks cached at initialization; unset leaves no-op checks.
See `tests/native/test_peer_copy.py` for modes and usage.

### Threads and fork

**Threads.** The shim's recursive mutex serializes every call into Mojo, so
ops need no locking of their own; the autograd engine's thread and the main
thread interleave at op granularity.

**Fork.** The MAX runtime is not fork-safe, and it is up before any device
is: `import max.nn` (which `aten_functions.py` imports; the trigger is
`max._kv_cache_ops`) starts one runtime worker thread per hardware thread,
and `register_mojo_devices()` creates every device context. A forked child
inherits none of those threads, so a device call there waits forever
(measured: `DeviceContext.synchronize` under `_to_copy`, a futex that is
never signalled; on a GPU, modular/modular#5483 reports a segfault). The
shim therefore installs a `pthread_atfork` child handler at registration:
in the child `torch.mojo._is_in_bad_fork()` is True, `torch.manual_seed`
skips the device (torch's contract for that method), and an allocation or
an op raises "cannot be used in a forked subprocess ... use the 'spawn'
start method", as CUDA does. What keeps working after a fork is what
DataLoader needs: `torch.accelerator.is_available()` and `device_count()`
are reads of registration state and never reach the runtime, and forked
workers that only touch CPU tensors run normally
(`tests/native/test_fork.py`).

## Test support

`native.op_counting(True)`, `native.op_count("aten::add.Tensor")` count
boxed-kernel calls per op (the `CallChecker` in `torch_mojo_backend/testing.py`
uses them to assert that an op ran natively).

## Memory accounting

`torch.accelerator` memory APIs need torch 2.9+; `get_memory_info` needs 2.10+; this package requires torch 2.10+.

`torch.mojo.memory_allocated(device=None)` reports the bytes of live tensor
storage allocated by this backend, per device. Views, slices and
`as_strided` share a storage and count once; a zero-byte storage counts
nothing. CPU tensors, pinned host staging, foreign storage aliases and
MAX's internal workspaces/context memory do not appear in these counters.
Tensors on the MAX CPU device (the last mojo index) do count, under that
index only. Releasing a storage decrements its device's counters when the
block is handed back to MAX; MAX orders physical reuse on the owning stream.
This happens at final storage release, before device synchronization, including
when `record_stream` has fenced pending consumers; it does not wait for physical
reuse or a host retirement queue.

`memory_stats()` is a sorted `OrderedDict` with torch.cuda's keys;
`memory_stats_as_nested_dict()` returns the same data before flattening.
`memory_summary(abbreviated=False)` formats those counters. Every counter
and reset lives in Mojo, serialized by the shim mutex. The C++ allocator
only forwards and marshals torch's `DeviceStats`, so `torch.accelerator`'s
memory APIs and Inductor's `MojoInterface.memory_allocated()` read the same
accounting.

Registration eagerly initializes MAX's contexts and the allocator, before any
tensor allocation. Before `register_mojo_devices()` there is no `torch.mojo`
module; after it, `initialized()` is true and even zero-usage queries validate
their device arguments. There is no public registered-but-uninitialized window.

| statistic | meaning here |
|---|---|
| `allocated_bytes.all` | live storage bytes (`current`), their high-water mark (`peak`), and cumulative bytes allocated/freed (`allocated`/`freed`) |
| `allocation.all` | the same four counters, in nonempty storage allocations |
| `requested_bytes.all` | equals allocated bytes: we do not round or split requests |
| `reserved_bytes.all` | **equals allocated bytes**: there is no caching allocator of ours; MAX owns the arena and does not expose its reservation/slack |
| `num_device_alloc`, `num_device_free` | successful buffer allocations/releases at our boundary with MAX, not driver malloc/free calls inside its arena |
| `num_alloc_retries` | initial MAX allocation failures for which we drain the device and retry once |
| `num_ooms` | allocations that still fail on that recovery path |

`memory_reserved() - memory_allocated()` is therefore **structurally zero**.
It does not measure MAX's cached slack as the corresponding CUDA expression
measures torch's caching allocator. `small_pool`/`large_pool`, `segment`,
`active*`, `inactive_split*`, `oversize_*`, `max_split_size` and
`num_sync_all_streams` are also structurally zero: this backend has none
of those allocator concepts. All measured activity is under `all`.

`max_memory_allocated()` and `max_memory_reserved()` retain the peaks after
a free. `reset_peak_memory_stats()` resets peaks to **current usage**, not
zero. `reset_accumulated_memory_stats()` zeroes cumulative allocated/freed
values and allocation/free/retry/OOM event counts, leaving current usage and
peaks alone. Both resets apply only to the requested device.

`empty_cache()` releases **completed pinned host staging buffers** on every
device. Pending copies keep their buffers. It neither waits nor synchronizes,
and cannot return cached device memory to the OS: **MAX owns the arena and
exposes no trim API**. Calling it twice, or before any allocation, is safe;
live storage and its accounting are unchanged.

`mem_get_info()` (spelled `get_memory_info()` on `torch.accelerator`) returns
MAX's current `(free, total)` bytes, which are separate from our per-process
storage accounting. MAX may report its arena budget rather than physical
installed VRAM: on the H100 tested, its total was smaller than `nvidia-smi`'s
and a 4 MiB tensor consumed a 256 MiB arena chunk. These are MAX's numbers,
not a replacement implementation of CUDA's `cudaMemGetInfo`. Do not infer
our reserved bytes from `total - free`.
The MAX CPU device returns **host** memory information (verified with MAX
26.5 on Linux), not zeroes. A MAX implementation that supplies no memory
capacity raises an explicit `NotImplementedError` instead of fabricating it.

`get_device_properties()` returns a frozen dataclass, gathered and cached
per device in Mojo through MAX, without CUDA/HIP probing in Python.
`get_device_name()` and `get_device_capability()` read that cache. Field
names follow torch.cuda where possible; `api`, `is_cpu`, `arch_name`, shared
memory limits and `clock_rate` (kHz) provide additional MAX information.
Unavailable fields are `None`. MAX's CPU returns zero for GPU attributes;
we expose those as `None` because they do not describe CPU hardware.
Compute capability is a CUDA `(major, minor)` pair; other APIs return
`(None, None)`, with HIP's architecture in `gcnArchName` and `arch_name`.
This dataclass is separate from Inductor's Triton autotuner property contract.
`torch.accelerator.get_device_capability()` describes dtype support and raises
the default unsupported-capability error, just as CUDA does; we do not advertise
an unverified dtype support list.

There is no counterpart for CUDA allocator snapshots/history
(`memory_snapshot`, `_record_memory_history`, `_dump_snapshot`), raw caching
allocator pointers (`caching_allocator_alloc`/`caching_allocator_delete`),
per-process limits (`set_per_process_memory_fraction`), process listings
(`list_gpu_processes`), or host/pinned allocator statistics
(`host_memory_stats`, `host_memory_stats_as_nested_dict`,
`reset_accumulated_host_memory_stats`, `reset_peak_host_memory_stats`).
These names raise `NotImplementedError` with a reason: MAX does not expose
the necessary arena/process information, and we keep counters, not allocation
histories or a raw-pointer allocator.

After `fork()`, memory queries/resets and property reads reject device use
before taking the shim mutex, with the same `spawn` guidance as allocations.
`empty_cache()` is a no-op. Torch's accelerator wrappers additionally return
empty/zero statistics when `initialized()` is false, as it is in the child.
## Host memory

The device allocator owns MAX `DeviceBuffer` boxes; the pinned CPU allocator
owns MAX `HostBuffer` boxes from `enqueue_create_host_buffer`, backed by
page-locked host memory for the current mojo device's context. Mojo owns
zero-byte requests too: a real allocation handle with a null data pointer,
no host buffer and no registry entry, so `is_pinned()` stays false.
`is_pinned()` forwards to Mojo, which shares the transfer registry's binary
interval lookup, including interior pointers, then asks torch's CUDA hook
through a C accessor on a miss. Torch's CPU factories prefer CUDA's pinned
allocator when CUDA is available; `pin_memory=True` uses the Mojo allocator
with a CPU torch build. The CUDA fallback accounts for factories preferring
CUDA's allocator while queries prefer PrivateUse1.

Both `copy_` and `.to()` honor `non_blocking`. H2D copies directly from a
live Mojo-pinned source on the transfer device's selected stream, even when
blocking: this queues the read after a preceding async download or an explicit
stream dependency. A blocking upload then synchronizes that stream. Only our
live registry blocks can have a pending async host write. CPU conversion or
relayout of such a source waits for the selected stream before reading it.

With `non_blocking=True`, D2H copies directly into a live Mojo-pinned
destination on the same device without waiting for the stream. Synchronize
before reading an async download or modifying a direct upload's source.
Cross-stream producers and consumers require explicit event waits; before a
host read or an upload to another device, synchronize the producing event
on the host. A GPU-side wait alone cannot order a CPU staging memcpy.

`.to("cpu", non_blocking=True)` allocates its destination through Mojo's
pinned allocator for the **source** device, including on CUDA-enabled torch
wheels and when another device is current. The shim's `tmb_cpu_empty_pinned`
only constructs that CPU tensor; Mojo makes the allocation/routing decisions.
This reproduces upstream `_to_copy`'s strided accelerator-to-CPU pinning
condition (`isAcceleratorExcluded` is already present in torch 2.7), which
overwrites the explicit `pin_memory` option on `aten::_to_copy`. Blocking
CPU output is pageable even with `pin_memory=True`. If pinned allocation
fails, Mojo retries with pageable memory and completes the download before
returning; transfer errors still propagate.

The following conservative policies deliberately differ from CUDA:

* Pageable, foreign-pinned and other-device pinned downloads finish before
  returning, even with `non_blocking=True`. CUDA's asynchronous API does not
  make a general completion promise for these pointers. CUDA-owned pinned
  blocks have no lifetime tracking on our streams, so `is_pinned()` alone
  cannot enable Mojo's async routes.
* Uploads from those memory classes snapshot into a separate pinned buffer
  before returning. Mojo restricts direct DMA to the owning device; CUDA can
  use portable pinned allocations asynchronously across devices.
* `copy_` downloads needing host dtype conversion or relayout block. `.to()`
  performs supported conversions/relayouts on the GPU and stays asynchronous,
  but casts using the host fallback (for example float32 to float64) block.
  CUDA can also perform that conversion on the GPU and stay asynchronous.

Blocking direct uploads finish reading the caller's memory before returning.
Staged uploads take a synchronous snapshot, so the caller can reuse its source
without waiting for queued DMA or device relayout. CPU and Metal uploads always
finish synchronously; `non_blocking` is a no-op on the MAX CPU device.

Async copies record each stream on the pinned block. Free removes the block
from the live registry and queues one completion callback per recorded
stream; the shared staging/pinned drain destroys it after all callbacks
finish. Uploads, pinned allocations/frees and device/stream synchronization
drain completed blocks. Callbacks only decrement an atomic counter; no
device API runs on the callback thread. If MAX rejects host callbacks, free
synchronizes the affected stream instead.

## Streams and events

`torch.Stream(device="mojo")`, `torch.Event`, `torch.accelerator.current_stream
/ set_stream / synchronize` and `torch.mojo.stream(s)` are torch's own generic
objects driven by the shim's device guard. A stream is a MAX stream of the
device's base context (`DeviceContext.create_stream`, then a `select_stream`
view kernels launch on); events pair a MAX event (ordering: record / wait /
synchronize on every backend) with a CUDA / HIP driver event on the same raw
stream for `query()` and `elapsed_time()`. The MAX CPU device has one stream;
its events time with the host clock.

## RNG

The device generator is bit-compatible with CUDA's: `MojoGeneratorImpl`
(shim_runtime.cpp) keeps a Philox seed and an offset counted in curand's unit,
`get_rng_state()` is CUDA's 16 bytes (seed, offset), and every draw of
`ops_random.mojo` -- `uniform_`, `normal_`, `log_normal_`, `cauchy_`,
`exponential_`, `geometric_`, `bernoulli_` (scalar and tensor p), the
`random_` family, `native_dropout` -- reproduces the CUDA kernel it mirrors:
same element order (TensorIterator's), same launch geometry (which makes
draws above the grid cap of `sm_count * max_threads_per_sm / 256` blocks
depend on the GPU model, as on CUDA), same counter reservation, same
`curand4` / `curand_uniform4` / `curand_normal4` conversions
(`eager_kernels/curand_philox.mojo`), same transforms and the same math -- the CUDA fast intrinsics ATen uses for
float, the precise libdevice routines for double, both ported bit-exactly in
`eager_kernels/libdevice_port.mojo`. `torch.manual_seed(s)` therefore gives
the same `torch.rand` / `randn` / `randint` / `bernoulli` / dropout values on
the mojo device (NVIDIA; on AMD and Apple the float transforms fall back to
std.math and only the integer draws are bit-exact) and on a CUDA device of
the same model, and
`tests/native/test_random_parity.py` holds it to that against recorded CUDA
digests (`rng_golden.json`, refreshed with `rng_parity_dump.py`). Not covered:
`multinomial`, `randperm`, `poisson`, `rrelu_with_noise` (unregistered), and
draws on the MAX CPU device (same kernels' closed form, no CUDA to match).

## Distributed

`torch.distributed.init_process_group(backend="mojo")` registers
`MojoProcessGroup` (distributed/process_group.py), a thin adapter over
`native/mojo/pg.mojo`: one communicator and one dedicated comm stream per
device; every collective makes the comm stream wait for the caller's current
stream, issues the NCCL / RCCL / mojoccl call on it (the three share the NCCL
C ABI; `TORCH_MOJO_BACKEND_CCL=mojo` picks the in-repo Mojo collectives), and
the returned Work is a device-typed torch Future completed while the comm
stream is current, so `wait()` orders the waiter's stream after the collective
without blocking the host. Touched buffers are `record_stream`ed on the comm
stream; the allocator fences their release on it. CPU tensors go to a private
gloo group. Non-contiguous operands are staged through dense temporaries
allocated on the comm stream (so freeing them never fences the compute
stream); inputs are only recorded, outputs are copied back. The group
reports itself as its own backend (`_get_backend`, `supports_coalescing`)
because torch looks the backend up that way for `batch_isend_irecv` and
`_coalescing_manager`; the coalescing hooks map to one NCCL group so a
bidirectional exchange cannot deadlock. Inside such a group nothing is
submitted before `group_end`, so every copy-back and every `record_stream`
of a coalesced call is deferred to `_end_coalescing` (which also keeps the
staging buffers alive until then); a call that raises inside the block closes
the group and drops that deferred work, since torch's `_coalescing_manager`
has no `finally`. Each pg.mojo entry holds the backend mutex over its whole
body (`with Locked():`), since these calls come from Python outside the boxed
adapter.

Two limitations, both by design of the Future-based Work: `Work.is_completed()`
is true as soon as the collective is enqueued (completion is a stream event,
not a host-visible flag), and an asynchronous NCCL error surfaces at the next
call or `synchronize`, not through `Work.wait()`; `ncclCommGetAsyncError` is
exposed (`tmb_pg_async_error`) for a watchdog but nothing polls it. The
128-byte `ncclUniqueId` is passed by value the way the x86-64 SysV ABI lays it
out (16 words after the register arguments), so other architectures are
refused at construction.

## One base library for every accelerator

The Mojo base library (`libtmb_backend`) contains no device code and makes
no compile-time choice about the accelerator: `init_backend` asks MAX at run
time which api has devices (`cuda`, `hip`, `metal`, else CPU only) and
`vendor.mojo` resolves the CUDA or HIP driver entry points by name from that
answer. One api per process: the probe takes the first of `cuda`, `hip`,
`metal` that has a device, so a machine with an NVIDIA and an AMD card sees
only the NVIDIA one (MAX device ordinals, and so mojo device indices, are
per api; mixing two apis would need per-device vendor state). So it builds
on a machine with no accelerator at all, and one build
serves NVIDIA, AMD and Apple machines; its cache key deliberately excludes
the accelerators (`toolchain_identity`), while kernel specializations, which
carry device code, are keyed with them (`kernel_identity`). Checked by
building the library on the cluster's login node (no GPU) and running the
runtime tests and the two-rank collective check on an H100 with that exact
file.

## Prebuilt libraries and the wheel

The two fixed libraries of the first row above are the only thing between a
fresh install and a working device, and on a fresh install they cost ~5 s
(shim) and ~13 s (base library) of compiling plus, for the shim, a C++
compiler on the box. The wheel therefore ships them prebuilt, in
`torch_mojo_backend/native/prebuilt/`:

| file | one per |
|---|---|
| `libtmb_shim-torch<major.minor>-<platform>-<machine>-cxx11abi<0\|1>.so\|.dylib` | torch series, platform, machine, libstdc++ ABI flag |
| `libtmb_backend-max<version>-<platform>-<machine>.so\|.dylib` | MAX version, platform, machine |
| `manifest.json` | what each was built from |

The shim is ~290 KB and links only libtorch_cpu, libc10 and the C/C++
runtimes — no Python — so one file serves every Python version and both the
CPU and the CUDA wheel of its torch series; what it *does* depend on is that
series, because its C++ standard follows it (C++20 from 2.14) and its
autocast policy table is generated from its headers. The base library is
~340 KB, holds no device code and makes no compile-time accelerator choice
(previous section), so it is one file per platform per release — MAX is
pinned exactly in pyproject.

**Selection.** Before compiling, `build_shim()` / `build_backend()` look for
a manifest entry matching the running torch series (resp. MAX version),
platform, machine and ABI flag *and* whose recorded source hash equals the
hash of the sources shipped beside it — so a checkout whose `csrc/` or
`mojo/` has moved on compiles rather than loading a stale library. A shim is
taken only under a *release* torch (`2.11.0`, `2.11.0+cpu`): the key is the
series, and only releases keep a series ABI-stable; a nightly or a custom
build compiles its own. A match is copied into the on-demand cache under the
name the compiler would have written (its provenance mark first, so the copy
is never visible without it), and everything downstream, `dlopen` included,
is unchanged. One trace line says which file was used. A prebuilt library
that does not load — an ABI it was not built for — is dropped under the
build lock and that build is compiled here instead. No C++ compiler is
looked for as long as a shim matches: the "no C++ compiler found" error is
raised only when one would actually have to run.
`TORCH_MOJO_BACKEND_PREBUILT=0` ignores the directory entirely.

**CPU target.** `mojo build` targets the host CPU by default, and a base
library built on an AVX-512 machine dies with SIGILL on one without (seen on
GitHub's runner pool). The base library is runtime glue, not a kernel, so
`build_backend()` always targets the platform baseline
(`portable_target_cpu()`: `x86-64-v3` on x86, `apple-m1`, `generic` on other
arm64); kernel specializations are built on the machine that runs them and
keep the host target.

Everything else still builds on demand: kernel specializations carry device
code, so they cannot ship.

**glibc.** On Linux the package's floor is glibc 2.34, set by MAX itself:
`max-core` and `mojo-compiler` publish manylinux_2_34 wheels, so an
installation below that is not possible whatever we ship. The base library,
built on any machine MAX runs on, sits at exactly that floor (its
`dlopen`/`dlsym` moved into libc in 2.34). The shims are built in the
`quay.io/pypa/manylinux_2_28_{x86_64,aarch64}` containers — torch's own
floor, and MAX is neither needed nor installable there — because the floor of
a C++ build is decided by the glibc it compiled against, not by its sources:
a shim built on Ubuntu 22.04 imports `__libc_single_threaded@GLIBC_2.32`, and
on a 2.38 host it would pick up `__isoc23_*` too. `manifest.json` records
each artefact's floor, and `scripts/build_prebuilt.py --report` prints them.

**The wheel** stays `py3-none-any` and carries every platform's files: these
are data loaded by ctypes at run time, not extension modules, and a file that
does not match the machine is never opened. Hatchling needs them listed in
`artifacts` (both the wheel and the sdist target — plain `uv build` builds
the wheel from the sdist) because they are ignored by git. A wheel built from
a plain checkout has none of them and simply compiles, as before; the full
one comes from `.github/workflows/wheel.yml` (`publish.yml`, which uploads to
PyPI on a GitHub release, builds from a plain checkout). That workflow runs on
every pull-request commit and every push to `main`, like the unit tests, and
on `v*` tags: each run builds the shims for every supported torch series on
Linux x86_64, Linux aarch64 and macOS arm64, checks their glibc floors, builds
the wheel, then installs it in a fresh venv and runs
`scripts/smoke_prebuilt_wheel.py` against two torch versions, which fails
unless both prebuilt libraries were used and an op ran on the CPU device.
That one base library per platform can drive any GPU only because it holds
no device code, and `tests/test_backend_has_no_device_code.py` holds it to
that on every commit: it builds the library with the production command for
no accelerator, Apple M4, MI300A and H100, and requires the four shared
libraries (or their lowered LLVM programs) to be byte-identical with no
`.ptx`/`.amdgcn`/`.ll` sidecar. Mojo 26.5's Darwin host optimizer can emit
different machine code from identical LLVM for different accelerator flags. A
control in the same file builds a one-kernel module for two targets and
requires those to differ, so the equality cannot pass vacuously.

**Refreshing them** — after a change to `native/csrc/`, `native/mojo/`, or
the MAX pin:

```bash
# this machine's platform; a venv per torch version, holding that CPU wheel
uv run python scripts/build_prebuilt.py --torch 2.7 2.8 2.9 2.10 2.11 2.12 2.13 2.14

# what is there now, and what glibc each artefact needs
uv run python scripts/build_prebuilt.py --report

# CI does the same in a container, then merges every platform's upload
python3 scripts/build_prebuilt.py --merge artifacts/
```

A stale directory is harmless — the source hash no longer matches and the
build happens as usual — so refreshing is never urgent, and
`TORCH_MOJO_BACKEND_PREBUILT=0` is the way to compare the two paths.

## Supported torch versions

Checked with the CPU wheels of torch 2.7.1, 2.8.0, 2.9.1, 2.10.0, 2.11.0,
2.12.1, 2.13.0 and 2.14.0 on an H100 (MAX 26.5, Mojo 1.0, Python 3.12):
device registration, ops with autograd against CPU, autocast, streams and
events, the profiler, a fused-AdamW training step, seeded RNG,
torch.compile through the mojo backend, and the two-rank collective check
with NCCL and with the Mojo collectives. Version-specific pieces: the shim
compiles as C++20 from torch 2.14 (its headers require it) and C++17
before; the guard method behind `torch.Stream.native_handle` exists from
2.11, so code that needs a stream's vendor handle uses
`torch.mojo.stream_native_handle` instead. torch 2.6 and older cannot
import the package: MAX 26.5's torch interop (`max.experimental.torch`,
used by the torch.compile backend) references a dtype added in 2.7.

## macOS

The shim and the base library build with Xcode's clang and link with
`-undefined dynamic_lookup` (they call into each other at `dlopen`, and ld64
must be told so). Kernel families compile for Metal through MAX, which needs
Apple's Metal compiler: since Xcode 26 that is a separate download, and
without it every kernel that goes through it fails with `Metal Compiler
failed to compile metallib` (the elementwise scalar ops were the first to
hit it). Install it once with `xcodebuild -downloadComponent MetalToolchain`.
The build-failure message names the `-D` set of the specialization, so a
failing family can be rebuilt by hand with the same `mojo build` line.

What MAX 26.5's Metal backend does not have: user-created streams
(`createStream is not supported on this device`) or device events
(`eventCreate is not supported on this device`, including the default stream).
`torch.Stream(device="mojo")` therefore always identifies the default stream
(`stream_id == 0`, matching MPS — see `docs/streams.md`), while recording an
event still raises; and host callbacks, which is why host-to-device copies take
the synchronous route there (`copy_from_host`; unified memory makes the
pinned staging pointless anyway). Checked on an M4 (macOS 26.6.1): the
bring-up tests pass except the event ones, and the op groups run
against CPU torch like on CUDA.

`torch.compile(backend=mojo_backend)` exchanges allocations with the native
Metal device through DLPack without host staging. MAX does not support external
Metal streams, so the handoff synchronizes the producing queue. MAX 26.5's raw
D2D copy routine rejects imported Metal buffers that its kernels can read;
that specific failure uses the existing byte-copy kernel on the current queue.
Other copy errors still propagate.

Metal half-precision ROI/PSROI and deformable-convolution backward scatter use
one float32 workspace word per half accumulator because Metal has no 16-bit
atomic add/CAS. Each atomic update rounds to half before storing, preserving
half accumulation semantics, then a final kernel casts to the output tensor.
Cumulative sums use the portable per-line kernels, including half/bfloat16 and
rank-2 dimension 0. Operations requiring float64, such as `float_power`, remain
unavailable on Metal.

## Triton

Triton kernels run on mojo tensors with only the CPU torch wheel and the
`triton` wheel installed: Triton compiles and launches through its own GPU
backend (bundled `ptxas`, `libcuda` or `libamdhip64` from the display
driver) and asks torch only which device and stream are current, through a
driver object. `torch_mojo_backend/triton_driver.py` provides that driver
for the mojo device (a CUDA and an untested HIP variant, chosen by the
accelerator MAX drives); `register_mojo_devices()` installs it the moment
`triton.runtime.driver` is imported, or at once if it already is
(`TORCH_MOJO_BACKEND_TRITON=0` opts out; `enable_triton()` does it by
hand). Launches go to the mojo current stream's vendor handle
(`torch.mojo.stream_native_handle`), so they are ordered with our kernels;
like `torch.cuda`, a launch targets the *current* device, so select it
(`with torch.mojo.device(i):`) for tensors on another GPU. The autotuner
and `triton.testing.do_bench` time through the mojo device's events. The
CUDA device ordinal is the mojo device index. Checked with the Triton
tutorial add kernel, an autotuned kernel, a second GPU, and Liger-Kernel's
RMSNorm forward/backward (a package written against `torch.cuda`; its own
`device.type == "cuda"` branches fall back to generic paths on "mojo", e.g.
one SM's worth of partial weight gradients, and its autocast decorators
bind to the CPU device type because it infers the device from
`torch.cuda.is_available()`).

**Activation is scoped on purpose.** Triton keeps one active driver per
process and asks it for the device and the stream *before* it looks at a
launch's arguments (`triton/runtime/jit.py`'s `run`), so whichever driver is
installed answers for every Triton launch in that process; nothing in the
protocol distinguishes a launch over mojo tensors from one over CUDA tensors,
so there is no automatic answer that is right for both. `register_mojo_devices()`
therefore installs the driver only where that cannot bite — a torch with no
working CUDA/ROCm build, which is the case the feature exists for. With a
vendor build of torch, `enable_triton()` is an explicit call and takes the
whole process with it: from then on CUDA tensors must not be launched through
Triton there, or a tensor produced on a `torch.cuda` stream gets consumed on
an unordered mojo stream, on a different GPU whenever the two current devices
differ.

Both halves of a launch run under the driver context that owns the stream,
and put back the context they found. Triton's `loadBinary` loads into
whatever context is current and only retains `device`'s primary context when
there is none, and its generated `launch` does the same
(`ensureCudaContext`) — neither checks that an already-current context
belongs to the device it was handed. Inductor compiles under
`DeviceGuard(MojoInterface, i)`, which moves only mojo's TLS device, so
without the guards a kernel for `mojo:1` could be loaded into device 0's
context and then launched on device 1's stream. `_MojoCudaUtils.load_binary`
and `MojoCudaLauncher.__call__` in `triton_driver.py` are those two guards
(`cuStreamGetCtx` once per stream, then `cuCtxGetCurrent` and a set only when
it differs); HIP streams are not bound to a context and need neither.

## TorchInductor

`torch_mojo_backend.inductor.enable_inductor()` makes
`torch.compile(fn, backend="inductor")` generate and launch Triton kernels
for mojo tensors — still with a CPU torch wheel, since the kernels reach the
GPU through the Triton driver above. It calls `enable_triton()` and then the
two registrations Inductor offers out-of-tree devices, the ones Intel's XPU
backend uses:

| registry | what we give it |
|---|---|
| `torch._dynamo.device_interface.register_interface_for_device("mojo", MojoInterface)` | device / stream / event classes, `get_raw_stream`, `synchronize`, properties and compute capability (read from the CUDA driver, not from torch) |
| `torch._inductor.codegen.common.register_backend_for_device("mojo", TritonScheduling, PythonWrapperCodegen)` | Triton codegen and the Python wrapper; `MojoDeviceOpOverrides` supplies the wrapper's device lines (`from torch_mojo_backend.inductor import get_raw_stream`, `torch.mojo.set_device`, `torch.mojo.device`) |

It **needs** that CPU wheel, and raises on a torch with a working CUDA/ROCm
build rather than break it: Inductor decides "is this an accelerator?" from
one process-wide list, `torch._inductor.utils`'s `GPU_TYPES`, and
`get_gpu_type()` asserts at most one of its entries is available. With "mojo"
appended beside a working `torch.cuda` that assert fires — in autotuning's
subprocess setup and in the profiler benchmarking — for that process's CUDA
graphs as much as for ours, and nothing in that API is per-graph. So it is one
backend or the other: `enable_inductor()` says so and stops, and
`add_mojo_to_the_inductor_gpu_types` stands aside.

Everything Inductor keys off a device *registry* then works. What it keys off
a hardcoded list does not, and each of those is one function in
`monkeypatching.py`: `GPU_TYPES` (`add_mojo_to_the_inductor_gpu_types`),
`torch.utils._triton.has_triton`'s device dict
(`let_has_triton_see_the_mojo_device`), the Triton compiler backend selected
by the device-type string inside `GPUTarget`
(`register_the_mojo_triton_target`), and the compile-worker subprocess, which
imports only torch and so cannot know our device
(`compile_inductor_kernels_in_process` runs the compiles in-process instead;
`TORCHINDUCTOR_WORKER_START=fork` is the alternative). `_ptxas.py` also points
Triton at MAX's ptxas rather than the one inside the torch wheel, whose cubins
a driver older than that wheel's CUDA cannot load.

Ops Inductor does not generate — `mm`, `addmm`, `bmm`, convolution — it calls
as ATen extern kernels, through the `out=` overloads, which run as our native
ops.

Measured on one H100, a 256x1024-4096-4096-1024 fp32 MLP training step
(forward, loss, backward): 1384 us/step compiled against 1457 us/step eager,
5% faster, device time from one event pair around 50 steps. The three GEMMs
and their backwards dominate and are the same native kernels in both legs;
what Inductor buys is the fusion of everything around them.

Not production yet. `mode="max-autotune"` and `mode="reduce-overhead"` both
compile and run correctly, but neither does what it says: no Triton GEMM
template is registered for a device type outside {cuda, xpu, cpu, mtia}
(`torch/_inductor/template_heuristics/registry.py` logs "No template
heuristic found ... device_type=mojo" and falls back to the ATen mm), and
CUDA graphs are skipped (`cudagraph_skips`), cudagraph trees being written
against `torch.cuda.CUDAGraph`. Coordinate-descent autotuning of the
*generated* kernels does run, benchmarking on our device through the Triton
driver. AOTInductor is unimplemented (the C++ half of `DeviceOpOverrides`),
and this is NVIDIA only, like the Triton driver.
## Compiled CUDA extensions

A package that ships its own compiled kernels — causal-conv1d, mamba-ssm,
apex — is the opposite case from Triton: it does not launch through a driver
of its own, it calls into libtorch. `causal_conv1d_cuda.so` links
`libc10_cuda.so` and `libcudart.so.12`, its C++ opens with
`TORCH_CHECK(x.is_cuda())` and it launches on
`at::cuda::getCurrentCUDAStream()`. So it needs a **CUDA build of torch** —
on a CPU wheel `import causal_conv1d` fails at the loader, and there is no
way around that from here — and once that exists, it needs to be handed CUDA
tensors and a CUDA stream. `torch_mojo_backend/cuda_interop.py` hands it
both, as aliases rather than copies:

* **Memory.** MAX allocates on the device's *primary* CUDA context, the same
  one `torch.cuda` uses (checked: `cuStreamGetCtx` of a mojo stream equals
  `cuDevicePrimaryCtxRetain` of that device), so one device pointer is valid
  in both worlds. `as_cuda(t)` / `as_mojo(t)` take torch's own DLPack export
  and rewrite the device code in the capsule (`mojo_device/dlpack.py`'s
  `retag_capsule`: `kDLExtDev` ⟷ `kDLCUDA`), which keeps shape, strides,
  dtype and the addressed elements exactly and leaves the source tensor
  pinned by the capsule's deleter. `data_ptr()` is equal on both sides;
  nothing is copied. `storage_offset()` is *not* carried over — torch's
  DLPack export normalizes it into the pointer (`data` is `data_ptr()`,
  `byte_offset` is 0), so a view's alias starts at the view's first element
  with offset 0. The mojo device index is the CUDA ordinal, as for Triton
  (checked on a second GPU: `mojo:1` aliases to `cuda:1`), and it is not a
  parameter: relabelling a pointer as another device would copy nothing and
  hand out memory that device cannot address.
* **Ordering.** `on_mojo_stream()` installs a `torch.cuda.ExternalStream`
  over the mojo current stream's vendor handle, so the package's launches and
  ours queue on one stream and neither side synchronizes. Like `torch.cuda`'s
  own stream context it makes that device current, and restores the previous
  one on exit.

**An alias is memory and nothing else**, and the second bullet is what makes
the first safe rather than a nicety on top of it:

* it carries no stream handoff. The capsule goes out raw, bypassing torch's
  `__dlpack__(stream=...)` negotiation — the protocol half where a producer
  records an event for the consumer's stream — so an alias used on any stream
  but the one its memory was produced on races that producer;
* it carries no allocator stream tracking, and `record_stream` cannot add
  any. A mojo alias of CUDA memory has a `from_blob` deleter, which mojo's
  `recordDataPtrOnStream` (`shim_runtime.cpp`) ignores and torch 2.11's CUDA
  caching allocator ignores for foreign deleters, so recording the alias on
  another stream does not stop the original allocation being recycled while
  that stream still reads it.

So the correct use is one stream for both sides — `on_mojo_stream()`, or
`call_cuda` / the fallback, which enter it for you — plus keeping the source
tensor alive for as long as any stream still uses the alias. `as_cuda` /
`as_mojo` make that the default by refusing outside an `on_mojo_stream()`
block for their own device; `unordered=True` is the opt-out for an alias that
is only inspected (shape, dtype, `data_ptr`) or is ordered some other way,
and it is a promise, not a fix.

`call_cuda(fn, *args)` is the two together — enter the stream of the first
mojo argument's device, then convert, call and convert back inside it, one
device per call — and in-place mutation needs no conversion back, since the
alias *is* the memory. Gradients do not cross an alias (DLPack carries no
autograd history), so `cuda_autograd(fwd, bwd)` wraps a package's two entry
points as one differentiable mojo-level op; the autograd graph then stays on
mojo tensors, which matters because a CUDA backward cannot run at all once a
PrivateUse1 backend is registered (see `require_cuda_autograd` in
tests/conftest.py).

`enable_cuda_fallback()` (or `cuda_fallback()` for one block) installs the
same conversion as a `PrivateUse1` dispatcher fallback, so every op with a
CUDA kernel and no Mojo op runs this way — `index_select`, `sort`, `topk`,
… forward and backward. Two ops it cannot reach, both because the fallback
only fires where *no* kernel is registered:

* an op the backend registers and then declines at run time
  (`aten::convolution` with `transposed=True`) still raises — the dispatcher
  already found a kernel;
* `aten::convolution_backward` is CompositeExplicitAutograd and branches on
  the backend itself, sending everything that is not CPU/CUDA/MKLDNN to the
  `convolution_backward_overrideable` stub, which carries its own raising
  CompositeExplicitAutograd kernel. `_EXPLICIT_ROUTES` registers that one by
  hand, which is what makes a convolution trainable here.

Measured on an H100 (torch 2.11+cu128, causal-conv1d 1.7.0): `as_cuda` 3.6 µs,
`as_mojo` 3.5 µs, the stream context 7.4 µs, so `call_cuda` adds ~36 µs to a
three-tensor forward and ~31 µs to one op under the fallback. That is host
time only — the kernel is the package's own, unchanged — but it is larger
than a small kernel's device time, so the fallback is a correctness tool,
not a performance one.

ROCm is untested here (no AMD GPU): `torch.cuda.ExternalStream` is the same
class on a ROCm build, `retag_capsule` needs `kDLROCM` (10) instead of
`kDLCUDA` — `_vendor_dlpack_code()` already picks it off `torch.version.hip`
— and causal-conv1d publishes no ROCm wheel, so it would be built from
source with `HIP_ARCHITECTURES`.

## Profiling

The shim registers torch's PrivateUse1 `ProfilerStubs` over the backend's
timed events, so the legacy profiler reports device time per op:

```python
with torch.autograd.profiler.profile(use_device="mojo") as prof:
    ...
prof.key_averages().table(sort_by="self_device_time_total")   # "Self MOJO" columns
```

MAX 26.5 cannot create timed events on Metal. The legacy profiler reports
that unsupported operation through PyTorch's callback warnings; it does not
provide Metal device timings. Failed event creation is checked before use
so profiling cannot dereference a null event.

`torch.profiler.profile(activities=[CPU, PrivateUse1])` records the CPU-side
op timeline and exports Chrome traces; device kernel rows need the Kineto
PrivateUse1 plugin API that only exists in torch >= 2.12 (with a CUDA torch
wheel that matches the driver, CUPTI already captures MAX kernels).
