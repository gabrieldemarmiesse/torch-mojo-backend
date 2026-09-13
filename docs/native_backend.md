# The native mojo device

The `mojo` torch device is a real PrivateUse1 backend: tensors are ordinary
`torch.Tensor`s whose storage is MAX device memory, and every aten op the
device supports is a Mojo function that torch's C++ dispatcher calls
directly. No Python runs on the op path.

```
torch.add(a, b)  ->  dispatcher  ->  MojoBoxedKernel (C++, native/csrc)
                                         |  TmbValue records
                                         v
                                  op_add_tensor (Mojo, native/mojo/ops_*.mojo)
                                         |  KernelCall
                                         v
                                  logic_ops.so::tmb_call  (built on first use)
```

Nothing above is compiled ahead of time except the runtime: the op body and
the kernel are each one `mojo build` that happens at the op's first call and
is then cached on disk (see "Three builds" below).

## Three builds

Everything is compiled on demand into `eager_kernels/__mojocache__/native/`
(`TORCH_MOJO_BACKEND_CACHE_DIR` moves the cache; it is contents-addressed so
several checkouts can share it) and each build is keyed by the hash of every
source it compiles in, the toolchain versions and its `-D` defines — so
touching `abi.mojo` invalidates every op extension, not just the backend.

| build | when | what it holds |
|---|---|---|
| C++ shim, Mojo backend | first `register_mojo_devices()`, unless the wheel ships them (see "Prebuilt libraries and the wheel") | the runtime, both fixed-size |
| one op extension per aten op | that op's first call | that op's body alone |
| one kernel family specialization | that kernel's first call | one (OP, dtypes, flags) kernel |

The backend library therefore does not grow with the number of ops: it holds
devices, streams, events, memory, the loader, the record ABI and the
registration list, and nothing else. Measured on one H100 node, 226 ops:
0.33 MB and 5.5 s of build, against 3.58 MB and 6.2 s when every op body was
linked into it — the size is the number that was growing, and the rest of the
build time is the compiler and the runtime modules, which are fixed.

The price is the first call of each op: one `mojo build` of 6–7 s, once per
op per source revision per machine (0.3–0.5 MB of cache each), and
milliseconds — a `dlopen` — in every later process. Building all 226 takes
about 22 minutes, which is why `prebuild_ops` exists.

`native.prebuild_ops()` compiles every op extension up front instead, for a
test suite or a CI image that would rather not pay a compile inside the first
call of each op (and, on a shared machine, not inside a GPU lock either).

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
| `shim_dispatch.cpp` | `tmb_library_impl` / `tmb_library_impl_lazy`: registers a Mojo function (or a resolver that produces one at the first call) as a boxed kernel. `MojoBoxedKernel` converts the IValue stack to `TmbValue` records (Scalar included, no heap boxing) and back, and caches the resolved kernel pointer. `tmb_call_op` calls any aten op from Mojo. |
| `shim_runtime.cpp` | allocator (`c10::Allocator` over Mojo alloc/free), `PrivateUse1HooksInterface`, the device guard (devices/streams/events), the Philox generator, `ProfilerStubs`, the tensor C API (`tmb_tensor_*`, `tmb_empty_strided`, `tmb_as_strided`). The current device and per-device current stream are C++ thread-locals (`tmb_current_device/stream`). |
| `shim_autocast.cpp` | `AutocastPrivateUse1` as one boxed fallback with a policy table filled from torch's own CUDA op lists. |

Three translation units compile in parallel: about 7 s wall cold.

**Mojo backend (`native/mojo/`)** — `mojo build backend.mojo --emit shared-lib`:

| file | what |
|---|---|
| `backend.mojo` | `tmb_native_init`: hooks table + the registration list (one `_group[register_x]` per ops file) |
| `registry.mojo` | `impl[op, "name"]`: registers the name behind a lazy trampoline in the backend, *or* is the selected op in that op's extension — see below |
| `abi.mojo` | `Value` records, tag constants, `T` (tensor view), result setters, `new_tensor` / `view_strided`, `unsupported()` |
| `device.mojo` | `Dev` per mojo index (accelerators, then the MAX CPU device), stream views, events (MAX events for ordering, vendor driver for query/timing), memory (`Buf` boxes behind DataPtr, `record_stream` fences), transfers |
| `vendor.mojo` | CUDA / HIP driver calls on MAX's raw streams |
| `loader.mojo` | on-demand builds of op extensions and kernel families: closure hash, cache lookup, `mojo build` in a subprocess under a flock, dlopen |
| `kernels.mojo` | `KernelCall`: defines + slots + owned specs for one kernel invocation |
| `ops_*.mojo` | the aten ops, and each file's `register_<group>` list |

## Op extensions: how an op body reaches the dispatcher

`backend.mojo` registers names, not implementations: `tmb_library_impl_lazy`
gives torch one generic trampoline per aten name, carrying the group file the
op lives in. At the op's first call the trampoline asks the loader for

```
mojo build native/mojo/ops_<group>.mojo --emit shared-lib -D TMB_OP=<aten name>
```

dlopens it, calls its `tmb_op_address` for the address of that op's boxed
entry (`abi.op_entry[op]`, exactly what a non-lazy registration would have
passed), and the shim stores it in the kernel object — so every later call is
the same direct call as before, with no added indirection.

One `impl[op, "name"](site)` line does both jobs, and `registry.TARGET_OP`
(the `TMB_OP` define) picks which:

* **backend library**, no define — register the name; `op` is named only in
  the branch the compiler drops, so the body is not elaborated;
* **that op's extension**, `TMB_OP=<name>` — hand back `op_address[op]()`.
  The other ~40 lines of the group's list compile to nothing, which is what
  keeps an extension to one op rather than a whole file.

A failed build is reported as a `RuntimeError` and is *not* remembered: the
next call tries again, so a compiler that died on a full disk is not fatal
for the process. A kernel that declines its inputs still raises
`NotImplementedError`, as before.

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
in `register_<group>` with `impl[op_x, "x.overload"](site)` (the name is a
compile-time parameter: that is what lets the op's extension select it).
A new group file needs three things: the `register_<group>` list, the
`tmb_op_address` export every group file ends with, and one
`_group[register_<group>](lib, "ops_<group>", prebuild)` line in
`backend.mojo`.

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
current stream; a tensor used by another stream gets `record_stream`ed by
torch (`recordDataPtrOnStream`), which the backend turns into an event the
owner stream waits on before the buffer is released.

**Threads.** The shim's recursive mutex serializes every call into Mojo, so
ops need no locking of their own; the autograd engine's thread and the main
thread interleave at op granularity.

## Test support

`native.op_counting(True)`, `native.op_count("aten::add.Tensor")` count
boxed-kernel calls per op (the `CallChecker` in `torch_mojo_backend/testing.py`
uses them to assert that an op ran natively).

## Streams and events

`torch.Stream(device="mojo")`, `torch.Event`, `torch.accelerator.current_stream
/ set_stream / synchronize` and `torch.mojo.stream(s)` are torch's own generic
objects driven by the shim's device guard. A stream is a MAX stream of the
device's base context (`DeviceContext.create_stream`, then a `select_stream`
view kernels launch on); events pair a MAX event (ordering: record / wait /
synchronize on every backend) with a CUDA / HIP driver event on the same raw
stream for `query()` and `elapsed_time()`. The MAX CPU device has one stream;
its events time with the host clock.

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
arm64); op extensions and kernel specializations are built on the machine
that runs them and keep the host target.

Everything else still builds on demand: op extensions and kernel
specializations carry device code, so they cannot ship.

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
PyPI on a GitHub release, builds from a plain checkout).

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

`torch.profiler.profile(activities=[CPU, PrivateUse1])` records the CPU-side
op timeline and exports Chrome traces; device kernel rows need the Kineto
PrivateUse1 plugin API that only exists in torch >= 2.12 (with a CUDA torch
wheel that matches the driver, CUPTI already captures MAX kernels).
