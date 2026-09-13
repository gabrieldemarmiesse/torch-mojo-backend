# Mojo kernel-family extensions

> This describes the **native backend**'s on-demand kernel builds
> (`torch_mojo_backend/native/mojo/loader.mojo`). See `docs/native_backend.md`
> for the full architecture. The Python-level design this file used to
> describe (`eager_kernels/__init__.py`'s `MojoExtensionLoader`,
> `MojoExtension` descriptors, one Python-callable `call` per `.so`) was the
> old eager-mode path; that code is deleted, and its historical measurements
> live in `docs/fast_eager_design.md` (marked superseded there too).

## Compiled at first call

A kernel family under `torch_mojo_backend/eager_kernels/<family>/<family>.mojo`
exposes one C entry point, `tmb_call`, gated by `comptime if` on the `OP` and
`DTYPE_ARG_*`/`DTYPE_OUT`/flag defines (`variant_gates.mojo`). The first call
into a specialization not yet in the cache runs `mojo build --emit shared-lib`
in a subprocess, at the call site in Mojo (`loader.mojo`'s `Loader.entry`),
and the call waits for it; every later call — and every later process, as
long as the sources and the toolchain are unchanged — dlopens the cached
`.so`. `TORCH_MOJO_BACKEND_TRACE` (on by default; `0` silences it) prints a
`[TRACE]` line for each variant build, with its duration.

This mirrors the old Python loader's behavior (same "one `.so` per exact
specialization, built inline at first use" design, same rationale — see
"Compile granularity" in `docs/fast_eager_design.md`), just driven from Mojo:
the caller is now `native/mojo/ops_*.mojo`, not a Python `aten_fast.py`
composition function.

## KernelCall: the operation-side descriptor

Where the old eager path had a stateless `MojoExtension` Python class per
operation, the native backend has `KernelCall` (`native/mojo/kernels.mojo`),
built fresh per call inside the op function (`native/mojo/ops_*.mojo`):

```mojo
var call = KernelCall("logic_ops", "AddSpec")   # family, OP
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
closure* (every `.mojo` file it `from X import`s, resolved family-dir-first
then package-root, plus every `op_utils/*.mojo`) together with the defines
and the toolchain identity (`native/__init__.py`'s `toolchain_identity()`:
torch/mojo/max/python/platform/machine versions) into the cache filename
`<family>.<defines-slug>.hash-<source-hash>.so` under
`TORCH_MOJO_BACKEND_CACHE_DIR` (default
`eager_kernels/__mojocache__/native/`). A cache hit loads that exact `.so`;
a miss builds it under a per-identity `flock` and installs it with an atomic
rename, so an interrupted compiler cannot leave a partial file that looks
valid, and concurrent requests for the same identity compile it once.

The two backend shims (the C++ shim and the Mojo `backend.mojo` itself) are
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
