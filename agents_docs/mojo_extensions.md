# Mojo kernel-family extensions

> This describes the **native backend**'s on-demand kernel builds
> (`torch_mojo_backend/mojo/tmb/backend/loader.mojo`). See `agents_docs/native_backend.md`
> for the full architecture. The Python-level design this file used to
> describe (`eager_kernels/__init__.py`'s `MojoExtensionLoader`,
> `MojoExtension` descriptors, one Python-callable `call` per `.so`) was the
> old eager-mode path; that code is deleted, and its historical measurements
> live in `agents_docs/fast_eager_design.md` (marked superseded there too).

## Compiled at first call

A kernel family `torch_mojo_backend/mojo/tmb/kernels/<family>/entry.mojo`
exposes one C entry point, `tmb_call`, gated by `comptime if` on the `OP` and
`DTYPE_ARG_*`/`DTYPE_OUT`/flag defines (`tmb/kernels/common/variant_gates.mojo`). The first call
into a specialization not yet in the cache runs `mojo build --emit shared-lib`
in a subprocess, at the call site in Mojo (`loader.mojo`'s `Loader.entry`),
and the call waits for it; every later call — and every later process, as
long as the sources and the toolchain are unchanged — dlopens the cached
`.so`. `TORCH_MOJO_BACKEND_TRACE` (on by default; `0` silences it) prints a
`[TRACE]` line for each variant build, with its duration.
`TORCH_MOJO_BACKEND_WERROR=1` passes `--Werror` to every Mojo build (kernel
variants, the base library, mojoccl), so a compiler warning
fails the build instead of scrolling past in captured stderr. It is off by
default and `tests/conftest.py` turns it on, which is the repository's
no-warnings check: a warning in any Mojo source fails the tests that build it.

This mirrors the old Python loader's behavior (same "one `.so` per exact
specialization, built inline at first use" design, same rationale — see
"Compile granularity" in `agents_docs/fast_eager_design.md`), just driven from Mojo:
the caller is now `tmb/ops/*.mojo`, not a Python `aten_fast.py`
composition function.

## KernelCall: the operation-side descriptor

Where the old eager path had a stateless `MojoExtension` Python class per
operation, the native backend has `KernelCall` (`tmb/backend/kernel_call.mojo`),
built fresh per call inside the op function (`tmb/ops/*.mojo`):

```mojo
var call = KernelCall("logic", "AddSpec")       # family, OP
call.arg_dtype(0, a.dtype)                       # DTYPE_ARG_0
call.arg_dtype(1, b.dtype)                       # DTYPE_ARG_1
call.out_dtype(dst.dtype)                        # DTYPE_OUT
call.flag("INPLACE", False)                      # -D INPLACE=0
call.spec(a.spec(cp)); call.spec(b.spec(cp)); call.spec(dst.spec(cp))
call.run()
```

`arg_dtype`/`out_dtype`/`flag` build the same canonical, sorted `-D` define
set the old `make_defines()` dicts did (order never affects the cache key);
`spec`/`int`/`f64`/`tuple` append the runtime argument slots (`Argv`, see
`native_backend.md`'s "Kernel families: the C entry"). `KernelCall` owns
every spec/tuple/slot until `run()` returns, matching the old rule that all
mutable state belongs to shared infrastructure, never to a per-call
descriptor a stray concurrent call could stomp on.

A warm call allocates nothing and formats no string: the family and OP names
are copied into inline byte buffers, the defines are kept as PODs beside a
running 64-bit hash, and the specs, slots and small tuples live in
fixed-capacity inline arrays. The `-D` strings are spelled out only on a
cache miss (`Defines.sorted()`), and the loader's hot path is one dictionary
probe of that hash. The capacities are therefore fixed (`MAX_CALL_SPECS`,
`MAX_CALL_SLOTS`, `MAX_DEFINES`, `TUPLE_POOL_WORDS`; a longer tuple spills to
the heap), and a builder that does not fit records the reason for `run()` to
raise rather than raising itself — the builders are called from non-raising
helpers.

## One compiled function per variant

Unchanged in spirit: every specialized `.so` exposes one C-ABI entry with a
constant name (`tmb_call`, not the old Python `PyInit_<family>` /
`extension.call`). A family source file is the compilation *input*, not the
compilation unit — its compile-time defines select exactly one operation and
one dtype tuple for a particular `.so`.

## Compile-time definitions and cache identity

Unchanged: dtypes, operation mode, output dtype, and implementation-selecting
flags belong in the defines; shapes, strides, pointers, scalar values, and
device contexts are runtime data. The loader hashes the family's *source
closure* (every `.mojo` file its `entry.mojo` reaches through `from tmb.a.b
import`, resolved as `<root>/tmb/a/b.mojo`, plus the relative imports inside
`tmb/graph`; `native.mojo_import_closure` is the same walk for the
Python-driven builds, and `tests/test_shared_kernel_cache.py` checks the two
agree) together with the defines
and the toolchain identity (`native/__init__.py`'s `toolchain_identity()`:
torch/mojo/max/python/platform/machine versions) into the cache filename
`<family>.<defines-slug>.hash-<source-hash>.so` under
`TORCH_MOJO_BACKEND_CACHE_DIR` (default: the user's cache directory,
`~/.cache/torch-mojo-backend/native/` on Linux). A cache hit loads that exact `.so`;
a miss builds it under a per-identity `flock` and installs it with an atomic
rename, so an interrupted compiler cannot leave a partial file that looks
valid, and concurrent requests for the same identity compile it once.

The two backend shims (the C++ shim and the Mojo `tmb/backend/entry.mojo` itself) are
cached the same way, one level up, in `native/__init__.py`
(`libtmb_shim.hash-*.so`, `libtmb_backend.hash-*.so`) -- or copied there
from the ones the wheel ships prebuilt, which is the same cache entry by
another route; see `agents_docs/native_backend.md`. `tests/native/test_loader.py` exercises this cache
end to end through public behavior (env-var relocation, a second process
reusing a build, a missing/corrupt `.so`).

The ptxas 48 KiB static-shared-memory cap and the dynamic-shared-memory
workaround for kernels that need more (unchanged from the old design) are
documented in `agents_docs/fast_eager_design.md`'s "Milestone 3" section and at
each affected kernel's `shared_mem_bytes` call site.

## Argv ABI

Where the old design had an "Into-style" Python ABI
(`extension.call(input_specs..., output_specs..., runtime_parameters...)`),
the native call site builds an `Argv` — a flat array of 64-bit slots
(`op_utils.Arg`): ints and pointers are the value, floats their bit pattern,
a tuple the address of an `[len, e0, ...]` Int array, a `TensorSpec` its
address, ctx pointer last. The family's `tmb_call` reads them back by
position for whichever `(OP, dtypes)` its build was gated to. Multiple
outputs, in-place, and `out=` all pass through the same slot list — the op
function on the Mojo side decides which tensor is the output before
building the call, exactly as the old Python descriptor's
`expected_output_specs` did.


## Shared unary operations

`tmb/kernels/common/unary_math.mojo` owns the SIMD expressions used by both
native unary kernels and `torch.compile`. Half inputs are evaluated in float32
and rounded once on output. `tmb/kernels/common/math_utils.mojo` holds the
existing accurate square root and tangent helpers; `op_utils` re-exports them
for other native kernels, and `div_math.mojo` beside them holds torch.div's
three modes. They sit outside `tmb/graph` although the graph ops use them:
the graph package imports the kernel tree, so a kernel module importing
`tmb.graph.*` would reach the package twice while it is precompiled, as
`graph` and as `tmb.graph`, which Mojo 1.1 rejects. Edits to any of them
invalidate the native build cache.

Native contiguous unary operations delegate aligned GPU work to Modular's
public `max.algorithm.elementwise`, on NVIDIA, AMD, and Apple. The general
route requires both pointers to be aligned for four elements of their
respective storage dtypes (eight bytes for fp16/bf16 input, sixteen for fp32,
and four for bool output). NVIDIA also uses measured vector widths for some
operations; each route checks the alignment of both pointers before selecting
that width. More narrowly aligned views still use the general public route.
Predicates and bitwise-not use two lanes for 64-bit inputs, retaining their
previous sixteen-byte input alignment requirement (two bytes for bool output).
The launcher handles arbitrary lengths, including scalar tails, and chooses
block/grid geometry. Unaligned inputs or outputs retain the existing
fallback. Supported dtypes and CPU dispatch are unchanged.

This public launcher is an intentional exception to the usual native
`_enqueue_cached` wrapper: the earlier H100 comparison measured only about
0.1–0.2 microseconds of extra host dispatch versus a cached adapter to
Modular's private launcher. Delegating avoids maintaining that private API;
the native specialization and Modular's underlying compilation still cache.
GPU performance measurements cover H100; AMD and Apple are cross-compiled.

NVIDIA float32 math (including promoted half inputs) avoids redundant NaN
masks where the device implementation already preserves NaNs. Its `log1p`
uses compensated float32 math with a small-input polynomial instead of the
stdlib's float64 intermediate. Tests check domain boundaries, signed zero,
subnormals, infinities, and NaNs. CPU, AMD, Apple, and float64 keep their
existing math paths.

The graph backend forwards through one generic
`ElementwiseOp[kind: StaticString](ElementwiseUnaryMixedOp)`. The mixed-output
trait also supports boolean predicates. In MAX 26.5, fusion lowering did not
forward parent-struct parameters (not re-checked on 26.6), so concrete
registrations are generated from one template in
`scripts/generate_elementwise_ops.py`.
These thin forwarders preserve graph fusion while keeping the math and
registration template in one place. The generator reads the supported kinds
from the Python helper's `Literal` annotation; run
`uv run python scripts/generate_elementwise_ops.py` after adding a kind.
Pre-commit checks that the checked-in registrations are current.

GPU float64 `acos` retains its graph implementation because the pinned Mojo
compiler cannot lower a float64 GPU `acos` call. Other supported routes use
the shared SIMD implementation.


## The eager kernels in the graph backend (`tmb/graph/`)

`torch.compile(backend=mojo_backend)` builds one MAX graph per compiled
function, and MAX would run Modular's kernels for every op in it -- while the
same model in eager mode on the `mojo` device runs this repository's
(`tmb/ops/*.mojo`, usually the faster ones on an H100). For the ops
GPT-2 inference spends its time in, the graph backend now calls the eager
kernels too:

| aten op | `aten_functions` route | custom op | eager kernel |
|---|---|---|---|
| `addmm`, `mm` | `_native_matmul` | `native_gemm`, `native_gemm_bias` | `_mm_route` / `_addmm_route` ladder: gemm16 (bf16, H100), TF32 (f32, H100, precision != "highest"), SIMT / gemv, fused bias |
| `bmm` | `aten_bmm` | `native_bmm` | `_bmm_route` |
| `_softmax` (trailing dim) | `aten_softmax` | `native_softmax_rows` | `SoftmaxSpec` (`_softmax_rows`) |
| `native_layer_norm` (weight and bias given) | `aten_native_layer_norm` | `native_layer_norm` | `LayerNormForward` (`enqueue_norm_rows`), float32 mean / rstd |
| `embedding` | `aten_embedding` | `native_embedding` | `Gather0` (`_gather0`) |

`gelu` already ran the shared `tmb/kernels/common/unary_math.mojo` in both modes
(through MAX's fusible `ElementwiseUnaryOp` registration); the elementwise
glue (`add`, `mul`, `eq`, `masked_fill`, `clone`) and the view ops stay MAX's,
which fuses them into their neighbors. The routes take only operands that
share one accelerator, in float32 / float16 / bfloat16; anything else, and
`TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS=0`, keeps the MAX composition.

**How a graph op reaches an eager kernel.** `tmb/graph/gemm.mojo` and
`tmb/graph/nn.mojo` are ordinary MAX custom ops (`@compiler.register`, an `execute` taking
`InputTensor` / `OutputTensor` and the `DeviceContext`) whose bodies call the
kernels' comptime-dtype entry points -- `_gemm_transb_dispatch[dt]`,
`_matmul_bias_launch[dt]`, `_softmax_rows[dt]`, `enqueue_norm_rows[dt, ...]`,
`_gather0[dt, idx]` -- never the `_spec_*` / runtime-dtype dispatchers: those
read the `-D` defines of a kernel-family build (`variant_gates.mojo`) and gate
everything OFF in a build that has none, which a MAX-compiled package is.
`aten_functions.py` calls the ops through `custom_mojo_ops.native_*`
(`F.custom(name=...)`, parameters for `transpose_b` / `tf32` / `eps`).

One import rule: a graph that uses several of these ops compiles their
modules into one unit, and every family entry (`tmb/kernels/<family>/
entry.mojo`) carries an `@export tmb_call` -- two of them in one unit is
"invalid re-export of tmb_call". So an op imports its kernel from a module
WITHOUT the export: the softmax and gather kernels moved out of the nn entry
into `tmb/kernels/nn/softmax_rows_kernels.mojo` and
`tmb/kernels/nn/gather_kernels.mojo` for exactly this, and
`tmb/kernels/matmul/entry.mojo` is the one entry imported directly (its GEMM
dispatch is the module). A new op that needs a kernel from another entry
moves the kernel out first.

Two build facts make this work, in `native/__init__.py` (with
`compiler.kernel_extension_paths()`) and `_mojo_import_path.py`:

* `build_graph_package()` precompiles `tmb/graph/` -- these ops and the
  fusible elementwise registrations, one package -- into
  `<cache>/graph.hash-<key>/graph.mojoc`, keyed by the whole import closure
  (every kernel source it reaches) and the toolchain, with the one `-I`
  (`torch_mojo_backend/mojo`). A precompiled package is not elaborated, so the
  build takes seconds and the file stays small; MAX compiles the op bodies for
  the device when it compiles a graph that uses them. The file stem is the
  package name MAX imports, so it must stay a Mojo identifier (the key is on
  the directory). `compiler.kernel_extension_paths()` lists it -- with
  whatever `make_torch_op_from_mojo` registered -- as the `custom_extensions`
  of every `F.custom` call. Precompiling it here also retires the per-call
  source precompile MAX did for the directory (the hack `tests/conftest.py`
  carried for modular/modular#5495).
* That compile resolves `from tmb.kernels.matmul.entry import ...` along the
  Mojo import path, `MODULAR_MOJO_MAX_IMPORT_PATH` (comma-separated), which by
  default names only MAX's `lib/mojo` and which *replaces* that default when
  set. `_mojo_import_path.py` therefore puts the default back and appends the
  Mojo source root, at package import, before `max` reads the variable. The
  root holds Mojo sources only, under the one package `tmb`, so nothing on it
  can shadow a toolchain module -- the variable's entries outrank `-I` in
  every `mojo` the package runs.

**Transposed operands.** MAX materializes a transposed producer before an
opaque custom op (the op always sees row-major strides), so the GEMM handed
`weight.T` would copy every Linear's weight on every call. `_native_matmul`
looks through the `aten.t` / `transpose` / `permute` node feeding `mat2`
(`_transposed_source`, using the fx node and the compiler's tensor book) and
passes the stored weight plus `transpose_b=True`, which the eager kernels read
for free. A `bmm` whose operand is a direct `transpose(1, 2)` gets the same
treatment; the `view(expand(transpose(...)))` chains torch's `matmul`
decomposition produces are materialized, as they are in eager mode.

**Numerics.** float32 GEMMs follow `torch.get_float32_matmul_precision()`
exactly as eager mode does: "highest" (torch's default) runs the fp32 SIMT
kernels, anything else the TF32 bridge on an H100. MAX's own matmul runs TF32
for fp32 unconditionally, so switching to the native routes at the default
precision trades tensor-core speed on large prefill shapes for torch's default
numerics; decode-size shapes (m <= 32) are where the eager kernels were tuned
to beat cuBLAS.
