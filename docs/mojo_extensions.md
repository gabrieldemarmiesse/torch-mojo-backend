# Mojo kernel-family extensions

> This describes the **native backend**'s on-demand kernel builds
> (`torch_mojo_backend/mojo/tmb/backend/loader.mojo`). See `docs/native_backend.md`
> for the full architecture. The Python-level design this file used to
> describe (`eager_kernels/__init__.py`'s `MojoExtensionLoader`,
> `MojoExtension` descriptors, one Python-callable `call` per `.so`) was the
> old eager-mode path; that code is deleted, and its historical measurements
> live in `docs/fast_eager_design.md` (marked superseded there too).

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
"Compile granularity" in `docs/fast_eager_design.md`), just driven from Mojo:
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
another route; see `docs/native_backend.md`. `tests/native/test_loader.py` exercises this cache
end to end through public behavior (env-var relocation, a second process
reusing a build, a missing/corrupt `.so`).

The ptxas 48 KiB static-shared-memory cap and the dynamic-shared-memory
workaround for kernels that need more (unchanged from the old design) are
documented in `docs/fast_eager_design.md`'s "Milestone 3" section and at
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

`tmb/graph/unary_math.mojo` owns the SIMD expressions used by both native
unary kernels and `torch.compile`. Half inputs are evaluated in float32 and
rounded once on output. `tmb/graph/math_utils.mojo` holds the existing accurate square
root and tangent helpers; `op_utils` re-exports them for other native kernels.
They live in the graph package because MAX precompiles that directory on its
own, without any `-I`, so it can only import its own siblings; the kernels
import them as `from tmb.graph.unary_math import ...`. Edits to either
shared module invalidate the native build cache.

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
trait also supports boolean predicates. In pinned MAX 26.5, fusion lowering
does not forward parent-struct parameters, so concrete registrations are
generated from one template in `scripts/generate_elementwise_ops.py`.
These thin forwarders preserve graph fusion while keeping the math and
registration template in one place. The generator reads the supported kinds
from the Python helper's `Literal` annotation; run
`uv run python scripts/generate_elementwise_ops.py` after adding a kind.
Pre-commit checks that the checked-in registrations are current.

GPU float64 `acos` retains its graph implementation because the pinned Mojo
compiler cannot lower a float64 GPU `acos` call. Other supported routes use
the shared SIMD implementation.
