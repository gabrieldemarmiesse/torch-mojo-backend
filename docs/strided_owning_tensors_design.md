# Strided owning tensors + killing the Python `driver.Buffer` (design)

> **Superseded.** This designs the Python eager path (`TorchMojoTensor`,
> `aten_fast.py`, the `MojoExtension` loader), which has been deleted; the
> `mojo` device is the native PrivateUse1 backend of
> `docs/native_backend.md`. Kept for the design reasoning and the
> measurements, which the Mojo host side inherited.

Status: **IMPLEMENTED** on branch `eager-strided-owning-tensors`. The plan
below is the original design; the "Implementation status" section right
after this intro records what was actually built and where it deviates.
Every non-obvious claim was verified with a runnable POC on an RTX 2000 Ada
in this repo's venv (`max==26.5.0.dev2026061806`); the POC code is embedded
below so a fresh session can re-run it without any external scratch files.

## Implementation status (what was actually built)

The eager `mojo_device` path was rewritten onto the holder. The
torch.compile backend (`aten_functions.py`, `torch_compile_backend/`) is
**unchanged** — this work is confined to eager mode.

**Key deviation from the design — metadata lives in Python, not the Mojo
struct.** `TensorHolder` (`eager_kernels/tensor_holder.mojo`) is a *pure
ownership token*: it owns one byte-typed `DeviceBuffer[uint8]` and nothing
else. All layout metadata (`_ptr`, `_shape`, `_strides` in elements,
`_offset`, `_dtype`, `_numel`, `_itemsize`, `_device`, `_is_contiguous`)
lives as plain Python attributes on `TorchMojoTensor`; views share the same
holder object and CPython's refcount on it is the ownership mechanism (last
drop → stream-ordered free). This is simpler than putting shape/strides in
the struct and downcasting, and it makes zero-copy views pure Python (no
Mojo call). The 52ns-downcast micro-optimization from the design was not
pursued; unwrap is a handful of Python attribute reads instead.

**Kernels** take raw `int` data pointers (storage offset pre-applied) plus
sizes/dtypes as ints (`op_utils._raw_dtype_int`), never `driver.Buffer`
objects. Every kernel gained a **CPU branch** (the MAX CPU device is a real
target now that the graph fallback is gone). New shared Mojo primitives in
`tensor_holder.mojo`: `alloc`/`alloc_from_host`/`copy_from_host`/
`copy_to_host`/`copy_d2d`/`read_scalar`/`CopyStrided`/`StridedFill`. A new
`reduction_ops.mojo` module holds the row reductions.

**Views are zero-copy strided** (permute/transpose/t/slice/select/split/
unbind/squeeze/unsqueeze/expand/alias/view); `reshape` is handled by a port
of ATen's `computeStride` in the `view` impl, materializing only when the
requested shape isn't a valid reinterpret. Broadcast-strided kernels
(logic_ops / WhereSelect) read the real strides, so many ops run on views
without materializing. `.contiguous()`/materialize uses the rank-4
`PermuteCopy` fast path (the generic rank-8 `CopyStrided` is ~2× slower and
is used only for rank > 4 / strided-destination copies).

**The graph fallback is deleted.** `mojo_device_aten_ops.py` binds each op
to its `aten_fast` impl or raises `NotImplementedError` naming the op. The
`_max_data`/`MaxEagerTensor`/`driver.Buffer` slots are gone from the eager
path entirely. Inputs the fast kernels don't cover (float64 — the GPU can't
do it at all; training/backward ops; SDPA with an attention mask;
isin-on-floats) raise; the eager test helpers convert that specific raise
into an `xfail` so the suite records them as expected-unsupported.

**Verified:** gpt2 runs correctly on `mojo_device:0` (GPU) — including under
a **torch-CPU-only** install (`torch==2.11.0+cpu`), proving the hard
constraint that CUDA-in-torch is not required; MAX drives the GPU via the
CUDA driver. Batch-256 decode throughput is ~1.31× torch-CUDA
(26.3k vs 20.1k tok/s on an H100 PCIe), at/above the pre-refactor
reference (26.2k re-measured on the same machine). The holder rewrite
initially cost ~25%; it was recovered by (a) running `torch.arange` as a
device kernel instead of host-build + blocking H2D (HF's logits
processors call it with vocab size twice per decoded token, and each
transfer drained the GPU queue, killing CPU/GPU overlap), (b) teaching
`AttnDecode` q batch/head strides and `cat` a strided-input gather so the
fused-qkv head-transpose views are consumed in place (36 fewer
materialize kernels/token), (c) memcpy for contiguous `clone`, and
(d) trimming per-op Python dispatch (beartype wrapper frames off the
hot helpers, cached dtype mapping, contiguity hints on row-major views).
`tests/test_eager_kernels.py` and `tests/test_max_device.py` pass;
`test_aten_functions.py` is green (508 pass / 61 xfail, 0 hard fails).

Remaining `_register_missing` raisers: `aten::_adaptive_avg_pool2d_backward`,
`aten::gelu_backward` (training-only).

## Why

The fast eager path (see `fast_eager_design.md`) works on **contiguous**
`max.driver.Buffer`s. Two problems remain:

1. **`driver.Buffer` carries no strides.** A non-contiguous tensor can't
   be represented, so any view (`permute`, `slice`, `expand`, a
   transpose that reorders memory) either materializes a full contiguous
   copy or drops to the slow graph path.
2. **The graph fallback is enormously expensive.** One fallback op pays a
   full per-call MLIR graph build + module-ASM SHA-256 + kernel-file
   re-hash + interpreter/compile (~2,200 µs/op, `fast_eager_design.md`).
   We want to **delete it and raise a clear error instead**.

The fix is a single change that also subsumes both goals: replace the
per-tensor payload — today a Python `max.driver.Buffer` (contiguous path)
or a lazy `max.experimental.tensor.Tensor` (`_max_data`, the graph path)
— with **one Mojo struct that owns the device memory and carries
`ptr + shape + strides + dtype + device-context`**. Kernels become
stride-aware and guard: they handle the strides they support and *raise*
(or materialize via `.contiguous()`) on the rest. The lazy graph slot
disappears, so "kill `driver.Buffer`" and "kill the fallback" are the
same change.

### What we are and aren't killing

- **Kill:** the Python `max.driver.Buffer` object, and the lazy
  `max.experimental.tensor` (`_max_data`) graph path.
- **Keep:** the Mojo `DeviceContext` / `DeviceBuffer` (from
  `std.gpu.host`). These are the thin (two-word) Mojo front-end to the
  AsyncRT runtime that already owns the CUDA/HIP context, the **stream**,
  and the **memory pool**, and already launches our kernels. This is not
  "MAX holding our hand" — it is the device abstraction + stream-ordered
  pooled allocator we want, and it's vendor-agnostic (NVIDIA/AMD/CPU).

### Hard constraint: CPU-only PyTorch + MAX GPU must keep working

`mojo_device` exists so a **CPU-only** PyTorch install (no CUDA/ROCm
torch build) can still do GPU work, with MAX providing the GPU. This
design preserves that **by construction**: PyTorch only ever holds an
opaque handle (the Mojo struct) inside a meta `PrivateUse1` tensor from
`torch._C._acc.create_empty_tensor`. All allocation, free, streams, and
kernels are MAX/Mojo. The entire torch-side surface —
`create_empty_tensor` + `torch.library.impl(op, "privateuseone")` + the
`__class__` swap — is core `PrivateUse1` machinery present in every torch
build, and is exactly what the current working setup already uses. No C++
compiler or CUDA-specific code is required. The Mojo tensor representation
does not depend on DLPack; a small pure-Python DLPack capsule is used only at
explicit storage-adoption boundaries, including when PyTorch adopts the pinned
CPU destination of an asynchronous D2H transfer. Each exported capsule puts a
manual Python reference to its complete export state in DLPack `manager_ctx`;
the consumer's deleter, rather than a replaceable module-global registry, is
therefore the lifetime root. The state also retains both ctypes callbacks so a
live CPU alias remains safe across module reload.

> Note: POCs below were run in a **CUDA torch** venv (`2.11.0+cu130`,
> which happened to be installed). The struct-relevant torch surface
> (`create_empty_tensor`, class swap, `privateuseone` dispatch) is
> build-agnostic and is what the current CPU-only setup already runs, so
> it is safe. Before/while implementing, re-run in a scratch
> `torch==2.11.0+cpu` venv to prove the target config.

## The one thing that decides where strides live

PyTorch's view ops (`permute`, `slice`, `expand`, `reshape`, `unsqueeze`)
are `CompositeExplicitAutograd`/`Implicit`: their C++ reads
`self.sizes()/strides()/storage_offset()/is_contiguous()` **off the
`TensorImpl` before any backend kernel runs**, then bottoms out in
`as_strided`. So a stride vector kept only in a side object is invisible
to those decompositions — they'd compute wrong geometry *silently*.

**Why this does not bite us:** the backend already registers its own
kernels for essentially every view op — `permute`, `slice.Tensor`,
`expand`, `view`, `_unsafe_view`, `transpose.int`, `t`, `squeeze.dim`,
`unsqueeze`, `select.int`, `split.Tensor`, `unbind.int`, `alias`
(see `mojo_device_aten_ops.py`). When a kernel is registered for the
`PrivateUse1` key, PyTorch dispatches straight to it and never runs its
own decomposition, so it never reads the (contiguous-looking) TensorImpl
strides. The strides live in our Mojo struct, our kernels read them, and
we intercept every op that consumes a `mojo_device` tensor. **We do not
need to put strides on the TensorImpl, and therefore do not need DLPack
or a C++ tensor constructor.** (Earlier exploration of DLPack `kDLExtDev`
and extending `torch._C._acc` was rejected for this reason.)

The residual leak — the TensorImpl keeps reporting contiguous strides —
only matters where something reads torch's metadata directly instead of
going through a registered op:

- **`reshape`** (`CompositeImplicitAutograd`, not registered) reads
  `is_contiguous()` in C++ to choose view-vs-copy, then calls our
  `view`. Handle it in the **stride-aware `view` impl**: check the
  struct's *real* strides and, if the reshape isn't a valid reinterpret
  of that layout, materialize. (Alternatively register `aten::reshape`.)
- `.is_contiguous()`/`.stride()` from user code, `torch.compile`/meta
  tracing, and autograd view-backward (training) also read TensorImpl
  metadata. They don't fire in the op-intercepted eager-inference path.
  If ever needed, mirror strides onto the TensorImpl then — not now.

## The holder

A single Mojo struct, registered as one Python type in **exactly one**
module (see single-registration rule below), stored in
`TorchMojoTensor` in place of `_buffer`/`_max_data`:

```mojo
comptime MAX_RANK = 8

struct TensorHolder(Movable, Writable):
    var buf: DeviceBuffer[DType.uint8]     # OWNS the allocation (byte-typed = dtype-erased)
    var rank: Int
    var shape: InlineArray[Int, MAX_RANK]      # no heap
    var strides: InlineArray[Int, MAX_RANK]    # in ELEMENTS
    var storage_offset: Int                    # in elements
    var dtype: DType                           # real dtype (buf is byte-typed)

    def __del__(deinit self):
        # buf's own destructor enqueues the stream-ordered free via AsyncRT.
        ...
    def write_to(self, mut writer: Some[Writer]): ...   # Writable is MANDATORY (tp_repr)
```

Design points, all verified:

- **Ownership / fire-and-forget.** `buf: DeviceBuffer` owns the memory;
  its destructor (run by the holder's `__del__` at CPython refcount 0)
  calls `AsyncRT_DeviceBuffer_release` → `cuMemFreeAsync` **on MAX's
  stream** — deferred, stream-ordered, safe as long as alloc + kernels +
  free all ride that one stream (they do; everything uses
  `device._device_context_ptr()`, the default stream).
- **Byte-typed erasure.** `DeviceBuffer[DType.uint8]` keeps the holder a
  single non-parametrized Python type (required for one registered type +
  cross-module downcast). The real `dtype` is a runtime field; kernels
  reinterpret the pointer.
- **Views** produce a new holder with adjusted `shape`/`strides`/
  `storage_offset` **sharing the same `buf`** (copy-init retains the
  AsyncRT refcount). Dropping the view frees nothing until the base also
  dies; child-then-parent order verified.
- **Allocation moves into the Mojo op** (`ctx.enqueue_create_buffer`), so
  no Python `driver.Buffer` is constructed per op — faster than today.

## Verified POCs (embedded so they're reproducible)

All run under the repo venv via `uv run python`. `mojo.importer` compiles
the `.mojo` files on first import (~30 s, cached).

### 1. Pure-Mojo owning GPU alloc + refcount free (the core primitive)

`mojo_owned.mojo`:

```mojo
from std.os import abort
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def _ctx_from_ptr(ptr: Int) -> DeviceContext:
    return DeviceContext(OpaquePointer[MutUntrackedOrigin](unsafe_from_address=ptr))


struct OwnedTensor(Movable, Writable):
    var buf: DeviceBuffer[DType.uint8]
    var nbytes: Int

    # A custom __del__ suppresses the synthesized fieldwise init — declare it.
    def __init__(out self, var buf: DeviceBuffer[DType.uint8], nbytes: Int):
        self.buf = buf^
        self.nbytes = nbytes

    def write_to(self, mut writer: Some[Writer]):
        writer.write("OwnedTensor(ptr=", Int(self.buf.unsafe_ptr()),
                     ", nbytes=", self.nbytes, ")")

    def __del__(deinit self):
        print("OWNED_DEL ptr=", Int(self.buf.unsafe_ptr()), " nbytes=", self.nbytes)

    @staticmethod
    def data_ptr(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(Int(self_ptr[].buf.unsafe_ptr()))


@export
def PyInit_mojo_owned() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_owned")
        _ = m.add_type[OwnedTensor]("OwnedTensor").def_method[OwnedTensor.data_ptr]("data_ptr")
        m.def_function[alloc]("alloc")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))


def alloc(ctx_ptr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    var ctx = _ctx_from_ptr(Int(py=ctx_ptr))
    var n = Int(py=nbytes)
    var buf = ctx.enqueue_create_buffer[DType.uint8](n)   # MAX's stream + mempool, owning
    return PythonObject(alloc=OwnedTensor(buf=buf^, nbytes=n))
```

Driver:

```python
import mojo.importer  # noqa
import gc
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators
import mojo_owned

gpu = [a for a in get_ordered_accelerators() if a.label == "gpu"][0]
h = mojo_owned.alloc(gpu._device_context_ptr(), 4096)
print(hex(h.data_ptr()))   # real GPU VA, nonzero
h2 = h; del h; gc.collect() # no free (2nd ref alive)
del h2; gc.collect()        # -> "OWNED_DEL ..." prints: stream-ordered free
```

Observed: `data_ptr` = `0x7ac34e000000` (real GPU address), destructor
fires only after the last reference drops.

### 2. Mojo struct as a Python type; `__del__` at refcount 0; cross-module downcast

A struct registered in module A (`add_type[T]` + `PythonObject(alloc=...)`)
is downcast in a **separately compiled** module B via
`obj.downcast_value_ptr[T]()` — works because the type registry is a
process-global keyed by the qualified type name. `__del__` runs exactly at
the last `Py_DECREF`. A "view" holder that stores a reference to its base
keeps the base alive; both destructors fire child-then-parent on final
drop. (See `common_holder.mojo` / `holder_a.mojo` / `holder_b.mojo` in the
PoC set; reproduce by defining a struct with a `base: PythonObject` field.)

### 3. Unwrap cost: downcast vs today's `driver.Buffer` attribute reads

Reading `ptr + numel + dtype` from a Python `driver.Buffer` via the
current `PyObject_GetAttrString` + `_data_ptr()` call path = **395 ns**.
The same via `downcast_value_ptr[TensorHolder]()` + field reads =
**52 ns** (and `unchecked_downcast_value_ptr` is ~0 ns after a one-time
`Py_TYPE` check). So the holder is a hot-path speedup, not just a
correctness change.

## `driver.Buffer` → Mojo replacement (complete touchpoint map)

| Today (Python `driver.Buffer`) | Replacement (Mojo) |
|---|---|
| Output alloc `driver.Buffer(dtype, shape, dev)` in `aten_fast._new_buffer` | `ctx.enqueue_create_buffer` in the op → returns holder |
| `TorchMojoTensor._buffer` / `_from_buffer` / lazy `_max_data` | the `TensorHolder` struct |
| Kernel unwrap (`_data_ptr()`/`num_elements`/`dtype` GetAttr in `op_utils`) | `downcast_value_ptr` + field reads |
| H2D — `Buffer.from_dlpack(cpu.detach())` in `mojo_device__copy_from` | CPU `memcpy` into an exact-size MAX `HostBuffer`, then asynchronous pinned `buf.enqueue_copy_from`; Python retains the pinned owner behind a stream event |
| D2H / `.item()` — `buffer.to_numpy()` | Blocking D2H copies into ordinary torch CPU storage and synchronizes; non-blocking GPU D2H enqueues into a MAX `HostBuffer`, exposes it as a CPU tensor through pure-Python DLPack, and retains both source and destination behind a stream event |
| dtype cast — `_to_copy` via graph | Mojo cast kernel (fast cast already exists) |
| free — `driver.Buffer` GC | holder `__del__` → `DeviceBuffer` stream-ordered free |

Relevant Mojo APIs (`std/gpu/host/device_context.mojo`, verified present):
`enqueue_create_buffer`, `enqueue_create_host_buffer`, and
`DeviceBuffer.enqueue_copy_to`/`enqueue_copy_from` (overloads for
`HostBuffer`, `DeviceBuffer`, and `Span[Scalar]`). A pageable `Span` is valid,
but a large GPU H2D from it may stage by draining earlier stream work. The
implementation therefore performs the host-side copy into MAX-owned pinned
memory first and retains that `HostBuffer` until a recorded stream event says
DMA is complete. Non-blocking GPU D2H uses the inverse arrangement: MAX owns
the pinned destination, PyTorch adopts it as CPU storage, and an event retains
both that destination and the source holder until DMA completes. Blocking D2H
and scalar reads continue to synchronize before exposing ordinary CPU memory.
The DLPack deleter clears `manager_ctx` before dropping its one manual owner,
so consumed and unconsumed capsules release exactly once; callback helpers are
captured per export to remain valid across `importlib.reload`.
This needs neither `driver.Buffer`, numpy, nor torch-cuda.
`DeviceContext.synchronize`, and the owning-from-external-pointer
`DeviceBuffer.__init__(ctx, ptr, size, *, owning=True/False)`.

## Stride-aware kernels + the guard = the fallback replacement

Each kernel declares the stride patterns it accepts (e.g. contiguous
only, or arbitrary strides via `LayoutTensor`'s runtime strides, or
"last dim contiguous"). On an unsupported pattern it either:

- materializes via a **stride-aware `.contiguous()`** gather/copy kernel
  (build this first — it's the shared materialize primitive and also what
  `.contiguous()` from user code hits), or
- **raises `NotImplementedError` naming the op** — this is where goal (2)
  "raise instead of falling back" lives. There is no graph fallback path
  anymore.

In-place ops (`add_`, `copy_`, `fill_`, `masked_fill_`) write through
strides too and need the same stride-awareness or guard.

## Build plan (suggested order)

1. **`TensorHolder`** Mojo type owning a `DeviceBuffer`, registered in one
   module (e.g. a new `tensor_holder.mojo` in `eager_kernels/`, imported
   first). Fields as above.
2. **Wire it into `TorchMojoTensor`** in place of `_buffer`/`_max_data`;
   move output allocation into the ops (`enqueue_create_buffer`). Keep the
   `create_empty_tensor` meta wrapper + `__class__` swap unchanged.
3. **Stride-aware `.contiguous()`** copy kernel + **H2D/D2H/`.item()`** on
   the holder. Blocking host reads synchronize; non-blocking GPU transfers
   use MAX-owned pinned staging/destinations with event-based lifetime
   retention. This removes the last `driver.Buffer` uses.
4. **One stride-aware elementwise kernel** + `permute`/`slice` returning
   **zero-copy strided holders**; verify a non-materializing
   `permute`/`slice` against CPU on `mojo_device`.
5. **Delete the graph fallback**; replace with the guarded raise. Audit
   the `NOT_HANDLED`/`None` sentinels in `aten_fast.py` and the
   `wrap_for_mojo_device`-only registrations (permute/expand/index/scatter/
   repeat/stack/softmax/... — the ~40 ops with no fast path today) and
   decide per op: implement, materialize, or raise.

## Gotchas discovered while prototyping (Mojo `26.5.0.dev2026061806`)

- The stdlib path is `std.python...` (not `stdlib.stdlib...`).
- A struct exposed via `add_type` **must be `Writable`** (the auto-installed
  `tp_repr` slot asserts it) and `Movable`; no properties (methods only);
  it **cannot subclass `torch.Tensor`** — it's an opaque payload the
  wrapper points at.
- A custom `__del__` **suppresses the synthesized fieldwise `__init__`** —
  declare one explicitly.
- Extension `def` functions must be marked `raises` if they call anything
  that can raise (e.g. `downcast_value_ptr`, `PythonObject(alloc=...)`).
- Single-registration rule: `add_type[T]` for the same `T` may run in only
  one module per process; a second `finalize()` raises. Put the holder +
  its registration in one module, import it first; kernel modules only
  `from tensor_holder import TensorHolder` to downcast.
- Type identity is by qualified name string (`FIXME MSTDL-1580` intends to
  switch to a compiler type id) — keep the holder's module path stable and
  all `.so`s on one toolchain.

## Reference files

- This repo: `torch_mojo_backend/eager_kernels/{__init__.py,aten_fast.py,
  op_utils/__init__.mojo,elementwise_ops.mojo}`,
  `torch_mojo_backend/mojo_device/{torch_mojo_tensor.py,mojo_device_aten_ops.py}`.
- Mojo stdlib (`/root/modular` or `.venv/.../modular`):
  `mojo/stdlib/std/python/bindings.mojo` (`add_type`/`PythonTypeBuilder`/
  `_tp_dealloc_wrapper`), `mojo/stdlib/std/python/python_object.mojo`
  (`downcast_value_ptr`), `mojo/stdlib/std/gpu/host/device_context.mojo`
  (`DeviceContext`/`DeviceBuffer`, alloc/copy/free).
- PyTorch (`/root/pytorch`): `torch/csrc/acc/Module.cpp`
  (`create_empty_tensor` — note it forces contiguous strides, no strides
  param), `aten/src/ATen/native/TensorShape.cpp` (composite view ops →
  `as_strided`).
- Prior step: `docs/fast_eager_design.md`.
