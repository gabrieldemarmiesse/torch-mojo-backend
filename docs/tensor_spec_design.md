# TensorSpec: moving the eager op prologue into Mojo

> **Superseded.** This designs the Python eager path (`TorchMojoTensor`,
> `aten_fast.py`, the `MojoExtension` loader), which has been deleted; the
> `mojo` device is the native PrivateUse1 backend of
> `docs/native_backend.md`. Kept for the design reasoning and the
> measurements, which the Mojo host side inherited.

Status: proof of concept merged on this branch (`eager-tensor-spec-poc`,
commit `49348c9`) covering `add`, `mul`, `relu` and inference `batch_norm`.
This document specifies how to extend the design to the rest of the eager
op set. It is written for an implementer with no other context; the POC
code is the reference implementation for every pattern described here.

Explicitly **out of scope** (known costs, deliberately not addressed):

- the torch wrapper mint (`create_empty_tensor` + `__class__` swap +
  attribute writes, ~1.4 µs per output tensor),
- the PyTorch dispatcher hop,
- lazy read-back of Python metadata from specs (a possible later step).

## 1. Motivation and measured results

Eager mojo-device ops used to run their whole prologue in interpreted
Python: argument type checks, dtype gates, contiguity handling, broadcast
layout (`_bcast_meta`, ~2.1 µs), output allocation, and finally a kernel
call marshalling pointers plus rank-8 int tuples. A `TensorSpec` — a
registered Mojo struct describing one tensor's layout — lets a single
boundary call do all of it in compiled code.

Measured on GPU (best-of-7 timings, two independent A/B pairs, rank-3
float32 tensors; benchmark scripts in the POC session, easily recreated):

| Op | classic | spec | delta |
|---|---|---|---|
| broadcast add (`x + rowvec`) | 13.4 µs | 9.0 µs | **−33%** |
| strided add (`x.transpose(1,2) + y`) | 13.7 µs | 9.1 µs | **−33%** |
| relu | 8.3 µs | 7.9 µs | −4% |
| add / mul, same shape | 9.1 µs | 8.8 µs | −3% |
| batch_norm inference | 20.2 µs | 19.5 µs | −3% |
| scalar add, view ops (controls) | — | — | unchanged |

Interpretation: the win concentrates where the Python prologue was heavy —
broadcast layout computation and long chains of failed `_try_*` attempts.
Paths that were already lean move only a few percent (boundary marshalling
itself is cheap: ~15 ns per int on the Mojo side). Set expectations
accordingly when picking which families to convert: **ops whose Python
routing does real work (broadcasting, geometry, multi-attempt chains) gain
the most.**

## 2. The architecture

### 2.1 TensorSpec (Mojo, `tensor_holder.mojo`)

```
struct TensorSpec(Movable, Writable):
    var ptr: Int          # data pointer, storage offset already applied
    var rank: Int
    var shape: IndexList[MAX_RANK]    # leading-padded with 1s
    var strides: IndexList[MAX_RANK]  # element strides, leading-padded with 0s
    var offset: Int       # informational
    var dtype: DType
    var itemsize: Int
    var numel: Int
    var contig: Bool
    var ctx_ptr: Int      # DeviceContext address of the tensor's device
```

Registered with `PythonModuleBuilder.add_type`. Helpers on the struct:
`dim(i)` (logical dim i, hiding the leading-pad convention: index
`MAX_RANK - rank + i`) and `ctx()` (rebuild the `DeviceContext` from
`ctx_ptr`). Leading padding is load-bearing: it aligns ranks at the
trailing edge, so Mojo-side broadcasting needs no rank bookkeeping at all
(see `_binary_spec_go`).

A spec is **effectively immutable**. Python swaps which spec a tensor
points to; nothing ever mutates a spec in place.

### 2.2 Spec lifecycle (Python, `aten_fast.py` / `torch_mojo_tensor.py`)

- `_spec_of(t)` builds a spec lazily on first use via
  `tensor_holder.make_spec(...)` and caches it as `t._spec` (read through
  `t.__dict__` — cheap, no descriptor). Inputs pay this once per tensor.
- Spec ops return a fresh spec for their output; `_wrap_spec_result`
  attaches it eagerly when minting the wrapper.
- `_rebind_payload` (the **single sanctioned metadata mutation**, used by
  the `out=` resize pattern) invalidates the destination's cached spec so
  `_spec_of` rebuilds it from the new pointer and layout on first reuse.
  Invariant for all future code: any path that changes an existing
  tensor's pointer or layout must go through `_rebind_payload`. Data-only
  mutation (`fill_`, `zero_`, in-place arithmetic, `copy_` into a tensor)
  leaves specs valid.

### 2.3 The spec op protocol

One Mojo entry per op (raw `def_py_c_function` / METH_FASTCALL dispatcher)
that does the **entire prologue and launch**:

1. Unchecked-downcast each spec argument
   (`PythonObject(from_borrowed=arg).unchecked_downcast_value_ptr[TensorSpec]()`
   — a pure pointer cast; callers are internal and guarantee the type).
2. Validate: dtype equality across operands, dtype supported (a `comptime
   for dt in ...` list per family), device (`ctx_ptr`) equality, rank
   bounds, contiguity where the kernel needs it. **A failed check raises.**
3. Compute geometry from the specs (broadcast dims/strides, channels/inner,
   reduction extents — whatever the kernel needs). Nothing geometric is
   passed in from Python.
4. Allocate the output (`ctx.enqueue_create_buffer[DType.uint8]`,
   1 byte minimum for numel == 0) and launch the kernel (skip launch when
   numel == 0).
5. Return `(holder, out_spec, shape_tuple, data_ptr)` via `_spec_result`.

Error protocol — **nothing is swallowed on spec paths**. The dispatcher's
`except e:` returns `_spec_unsupported(e)`, which sets a real Python
`NotImplementedError` via `PyErr_SetString` and returns a null
`PyObjectPtr`. Prefix every error message with `"mojo spec <op>: ..."`
(the test helper `_xfail_if_unsupported` keys on `"mojo"` in the message).

### 2.4 Python integration pattern

```python
result = _try_spec_binary("AddSpec", input, other)   # None on any exception
if result is None:
    result = <classic chain, unchanged>
```

The classic path stays as fallback during migration, so behavior is
bit-identical for anything a spec op rejects (wrong dtype, rank > 4,
promotions, bool, scalars…). Python keeps doing what Python must do:
resolving scalars/None operands, dtype promotion (`_promoted_pair`),
choosing which spec entry to call. Once a family's spec coverage provably
matches the classic gates (tests pass with the fallback deleted), the
fallback for that family may be removed in a dedicated commit — one family
at a time, never as part of the conversion commit itself.

**Status: the fallback deletion is done** (commits 151b28d..b9a164f). The
unary (incl. float64 on CPU), scalar, cast/fill, binary (scalar operands,
promotion, bool-mul, rank>4 flat, f64 via `_parallel_for_dt`), reduction
(Mojo-side permute+materialize through `_scratch_copy`) and matmul
(operand temporaries) families are spec-only; their classic kernels and
Python chains are deleted. Classic entries that remain, deliberately:
`Add` (the in-place `add_` fast path writes into an existing buffer),
`SoftmaxRows`/`Bmm`/`Matmul` (the SDPA chain and convolution
drive them with offsets/shared-A modes), `BatchNormInference` (fallback
for strided stats), `CopyStrided`/`StridedFill`/`Arange`, and the
unconverted ternary family (where/masked_fill/addcmul/addcdiv/clamp/isin
plus `AnyBool`/`AllBool`, `_bcast_meta`/`_launch_bcast`).

## 3. Module layout: where spec ops live

The Mojo type registry (`MOJO_PYTHON_TYPE_OBJECTS`) is a **process-wide**
`_Global` shared by every extension module in the process, keyed by Mojo
type id — and `_register_py_type_object` *raises* on double registration
(the PyInit `abort` turns that into SIGILL at import). Measured on the
pinned nightly during the migration; the original per-module-registration
plan crashes. The layout that works:

- **`op_utils/` owns the shared source**: `TensorSpec`, `TensorHolder`,
  `_spec_ptr` (the downcast accessor), `_row_major8`, `_spec_result` and
  `_spec_unsupported` live in `op_utils/` so every kernel module compiles
  the *same* struct definitions with the same type ids.
- **Only `tensor_holder` registers the types** (`add_type[TensorSpec]`,
  `add_type[TensorHolder]`) — exactly once per process. Every other
  module's `PyInit` registers *no* types. `PythonObject(alloc=...)` in any
  module finds the registration through the process-wide registry at
  runtime. Ordering is guaranteed: a spec op's inputs are mojo tensors, and
  creating any mojo tensor imports `tensor_holder` first.
- Each kernel module implements spec entries next to its kernels
  (`logic_ops.mojo` owns the binary/comparison specs, `elementwise_ops.mojo`
  the unary/scalar specs, `matmul_ops.mojo` gets `MatmulSpec`, …), calling
  the module's existing inner kernels directly — no duplicated kernel
  bodies. Specs flow freely across modules: `_spec_ptr` is a pure pointer
  bitcast that never consults the registry.

Keep `make_spec` (the Python-facing constructor used by `_spec_of`) in
`tensor_holder` only — one constructor is enough since specs flow freely
once created.

## 4. Conversion plan by family

Work through families in this order (highest measured/expected win first).
For each: add the Mojo spec entry (+ registration), hook the `fast_aten_*`
function with try-spec-first, run that op's tests, benchmark.

1. **Binary broadcast family** (`sub`, `div`, `maximum`, `minimum`,
   `pow`, `remainder`, `floor_divide`, comparisons `eq/ne/lt/le/gt/ge`,
   `logical_and/xor`, bitwise `and/or/xor`) — extend the POC's
   `_binary_spec_go` with the op-code table from `logic_ops.mojo`
   (`BOP_*`, `COP_*`), including its dtype gates (div/pow float-only,
   bitwise int-only, comparisons write a bool output — out dtype differs
   from input dtype; the POC's `_spec_result` already takes dtype
   explicitly). These ops all currently walk multi-attempt chains and pay
   `_bcast_meta`: expect the −33% class of win.
2. **Scalar variants** (`AddScalar`, `MulScalar`, `SubScalar`, `Div`…,
   int and float): spec + one `_raw_f64`/`_raw_int` scalar argument.
   Cheap to do once the binary template exists, removes the residual
   Python gates on the very hot `x * 2.0` shapes.
3. **Unary family** (all `UOP_*` codes in `elementwise_ops.mojo`):
   generalize `ReluSpec` into a parametrized `_unary_spec_go[op_code]` in
   `elementwise_ops.mojo`; keep the float-only gate as a comptime branch
   like `_unary_elementwise` does today.
4. **Reductions** (`sum`, `mean`, `amax/amin`, `argmax/argmin`, `any/all`,
   dim and full variants): geometry (rows/cols/kept-shape) moves from
   Python into the spec entry in `reduction_ops.mojo`. Python routing
   currently computes reduced shapes and strides — real prologue, real win.
5. **Norms and activations with state** (`layer_norm`, `softmax`
   variants, `sdpa` in `nn_ops.mojo`; `batch_norm` is done): multi-output
   ops return multiple `(holder, spec, shape, ptr)` groups in one tuple —
   extend `_spec_result` with a multi-output variant rather than calling
   it twice, so it stays one boundary call.

   **Measured limit of this family** (migration A/B, GPU, (6, 768)):
   each result group costs ~1 µs of Mojo-side object construction
   (registered-type allocation via the process-global registry). Ops whose
   classic prologue is already thin lose to that: `native_layer_norm` and
   `native_group_norm` regressed +4 µs/call as three-group spec ops and
   were reverted to the classic path. `min.dim` (two groups but a heavy
   classic prologue: permute view + `_reduce_to_rows`) wins −34%. Convert
   a multi-output op only when its classic Python routing does real work.
6. **Matmul family** (`mm`, `bmm`, `addmm`, `linear` in
   `matmul_ops.mojo`): the tier selection (GEMV vs tiled GEMM, size gates)
   stays wherever it is cheapest — tier choice on scalar fields is fine in
   Mojo; keep the *family choice* (matmul vs linear-with-bias) in Python.
7. **Data movement** (`Cast`, `Fill`, `CopyStrided` call sites,
   `_materialize_contiguous`): convert last; also implement **Mojo-side
   temporaries** here — when a spec op needs a contiguous operand, it may
   materialize into a scratch `DeviceBuffer` *inside the call* (alloc +
   `CopyStrided`-equivalent + use + drop) instead of raising. This removes
   whole wrapper mints for temporaries Python never sees. Introduce it as
   an opt-in per family after the family works with raise-and-fallback.

**Do not convert**: view ops (zero-copy pure Python, already ~free and
spec ops read strided inputs directly anyway), H2D/D2H transfers,
`.item()`/scalar reads, anything whose torch semantics need Python
(random generators, dtype-promotion decisions).

## 5. Gotchas (all hit during the POC — read before writing Mojo)

- **Pinned nightly (2026-06-18) predates the stdlib error helpers**:
  `std.python.bindings.ExceptionType` / `raise_python_exception` do NOT
  exist. Use the POC's `_spec_unsupported`:
  `cpy.get_error_global("PyExc_NotImplementedError")` +
  `cpy.PyErr_SetString(t, msg.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())`.
  `as_c_string_slice` requires a *mutable* `String` (assign to a `var`
  first). The `fn` keyword is removed in this nightly — everything is
  `def`.
- **Return-value construction** in raw dispatchers:
  `PythonObject(alloc=T(...))` then `obj^.steal_data()`; tuples via
  `Python.tuple(a^, b^, ...)`; shape tuples via `cpy.PyTuple_New` +
  `PyTuple_SetItem(t, i, cpy.PyLong_FromSsize_t(v))` (SetItem steals the
  reference — no decref).
- **Validate before allocating**: run every check that can raise before
  `enqueue_create_buffer`, so an error never leaves an enqueued alloc
  behind (the buffer's destructor would handle it, but keep the order
  clean anyway).
- **numel == 0**: allocate 1 byte (valid pointer invariant), skip the
  launch, return normally — except where the classic path deliberately
  returned `NOT_HANDLED` (e.g. batch_norm); mirror the existing tests.
- **Rank 0** works for free (shape8 all 1s, numel 1, empty shape tuple).
- **Editing any `.mojo` file invalidates the whole content-addressed
  cache** (`_mojo_sources_hash` spans the package): the next test run
  recompiles every module (~30 s each, file-lock-serialized). After a
  cache bust, run GPU test files **serially** — `pytest -n 8` makes xdist
  workers stand up concurrent MAX GPU contexts and fail with spurious
  CUDA OOM `device_context.mojo` errors. Serial passes in seconds.
- Syntax-check fast with
  `uv run mojo build torch_mojo_backend/eager_kernels/<file>.mojo --emit shared-lib -o /tmp/x.so`
  instead of waiting on the import-hook compile inside pytest.
- The spy test `test_fast_path_is_used` monkeypatches the *spec* entry
  (`eager_kernels.tensor_holder.AddSpec`) — follow that pattern when a
  conversion changes which Mojo function an op routes to. `CallChecker`
  counters are unaffected (they instrument the `fast_aten_*` Python
  functions, which remain the entry points).

## 6. Verification per family

1. Correctness: the family's tests in `tests/test_aten_functions.py`
   (`-k "<opname>"`), plus `tests/test_eager_kernels.py` and
   `tests/test_max_device.py` serially. The POC's smoke checklist is a
   good template: same-shape, broadcasting, mixed-rank, strided-view
   operands, scalars, rank-0, unsupported dtype (falls back), and one
   `out=` resize through `_rebind_payload` followed by reuse of the
   rebound tensor on a spec path.
2. Performance: A/B with `git stash` in the same session (session-to-session
   variance ~±400 ns exceeds the lean-path deltas). Measure at least one
   heavy-prologue shape and one lean shape per family; controls
   (view op, an unconverted op) must not move.
3. Lint: `uvx pre-commit run --all-files` (the Mojo formatter will
   rewrite your kernel file; rerun tests after).

Full-model sanity once several families are in: the gpt2 decode benchmark
(`bench_gpt2_504.py`) — expect the lean-path class of win there until
matmul/sdpa/layer_norm are converted.
