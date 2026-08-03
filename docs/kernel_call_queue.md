# Mojo extensions and the kernel-call queue

## Why calls are prepared before an extension is compiled

Eager-mode kernels are Mojo CPython extensions compiled on demand. A cold
workload discovers new operations while it runs, so blocking at each cache miss
would serialize graph discovery behind every compiler invocation.

The Python operation descriptor therefore performs all metadata work before it
loads the extension:

1. validate the call and infer its output tensor specifications;
2. allocate those outputs in Python;
3. derive the compile-time definitions from the input and output metadata;
4. request the exact compiled extension;
5. queue its single `ext.call(...)` entry point with the runtime arguments.

This lets later operations inspect the already-known shapes and dtypes of the
outputs and request their own compilations. Only the device launch waits for
the extension build. Operations whose output metadata depends on tensor data
cannot use this asynchronous path without synchronizing first.

Requested builds run on a bounded background pool (`_ASYNC_BUILD_SLOTS` in
`eager_kernels/__init__.py`, sized by `_pool_size()`: available RAM / 5 GiB
and cores / 3, capped at 16), so a cold workload compiles many variants
concurrently while it keeps discovering operations. The measurements that
motivated this design — including the pool-size sweep and the warm-path
regression it currently costs — are recorded in
[docs/fast_eager_design.md](fast_eager_design.md) under "Compile
granularity".

## Stateless operation descriptors

Each operation is represented by a stateless `MojoExtension` subclass. Its
Mojo source path is a class attribute; per-invocation values never live on the
class or an instance. The descriptor provides three pieces of operation
semantics:

- `expected_output_specs(...)` validates metadata and describes every output;
- `make_defines(...)` returns the exact compile-time specialization;
- call preparation converts inputs and preallocated outputs into the runtime
  arguments accepted by the extension.

Output descriptions include shape, dtype, device, and layout information
needed for allocation. Operations that alias an input, mutate in place, use an
`out=` argument, or return multiple tensors must describe those facts as well.

All state belongs to the shared infrastructure: compiled-module caches, build
locks and futures, the FIFO launch queue, and tensor keep-alives. This avoids a
race where simultaneous calls with different dtypes overwrite a descriptor's
"current" specialization.

## One compiled function per variant

Every specialized `.so` exposes one Python-callable function with a constant
name:

```python
extension.call(runtime_arguments...)
```

A Mojo source file may implement many operations, but its compile-time
definitions select only one of them for a particular `.so`. There is no Python
entry-point name in the cache identity and no attribute lookup for individual
operations. `tensor_holder` remains a special process-wide module because it
owns the shared Python tensor types.

## Compile-time definitions and cache identity

`make_defines(...)` returns a dictionary such as:

```python
{
    "OP": "AddSpec",
    "DTYPE_ARG_0": "float32",
    "DTYPE_ARG_1": "bfloat16",
    "DTYPE_OUT": "float32",
    "INPLACE": False,
}
```

The loader normalizes values and sorts entries by key. The same canonical
ordered representation generates both the compiler arguments and the defines
portion of the cache key, so dictionary insertion order cannot create duplicate
variants. Argument positions remain explicit: exchanging `DTYPE_ARG_0` and
`DTYPE_ARG_1` is a different specialization.

The compiler receives the normalized entries directly:

```text
-D DTYPE_ARG_0=float32
-D DTYPE_ARG_1=bfloat16
-D DTYPE_OUT=float32
-D INPLACE=0
-D OP=AddSpec
```

Every relevant flag is always present with a canonical default. Dtypes,
operation mode, output dtype, and implementation-selecting flags belong in the
definitions. Shapes, strides, pointers, scalar values, and device contexts are
runtime data unless they truly select different generated code. Consequently,
the same operation, dtypes, and flags reuse one `.so` across different shapes.

The full build identity also includes the Mojo source and dependency hashes,
the compiler/toolchain and host ABI identity, and the loader ABI version. The
canonical defines are serialized unambiguously and hashed for filenames; raw
`key=value-key=value` strings are not safe when string values contain
separators.

Each identity is immutable. A cache hit loads that exact `.so`; a miss builds
it. An unsupported dtype or flag reported by Mojo indicates a bug in Python's
definition mapping and is surfaced directly. There is no widening, dtype
escalation, fallback build, or launch-time retry.

Builds are protected by a per-identity file lock (`flock`) and written to a
temporary file before an atomic rename. Concurrent requests for one identity,
in this process or another, compile it once, and an interrupted compiler
cannot leave a partial file that looks valid.

## Queue ordering

The deferral boundary is the extension call: everything above it—autograd
recording, version counters, view metadata, output inference, and allocation—
runs synchronously in Python. Only the device launch may wait in
`eager_kernels/call_queue.py` for its exact extension to finish compiling.

The queue follows these rules:

1. **Strict FIFO.** Once a launch is queued, subsequent launches remain behind
   it even if their extensions are already warm. `pump()` launches the longest
   ready prefix.
2. **Host reads drain.** D2H transfers, scalar reads, synchronization, and
   operations whose result is needed by Python drain pending launches first.
3. **Keep-alive.** Queued raw pointers do not own their tensors, so every
   queued item carries the tensors its pointers name and retains them until
   it launches (or until an error abandons the episode). Each enqueue site
   states its own retention — the spec route passes its output and prepared
   arguments, the raw routes name theirs explicitly — and a launched item's
   references drop on the launching thread, keeping frees stream-ordered
   behind the launch. Retention is bounded: each item's device bytes are
   metered at enqueue, and the enqueue that pushes the total past the
   budget (computed from free device memory at first use; env-overridable) drains — waiting builds out — before
   returning. A cold, read-free loop therefore stalls at the budget
   instead of retaining a whole warm-up phase of transients; warm calls
   run inline and never touch the meter. Beneath the budget sits one
   reactive layer: a device allocation that still fails drains the queue,
   synchronizes the device so the stream-ordered frees land, and retries
   exactly once before the error surfaces.
4. **Thread order.** Direct launches are ordered by their issuing thread and
   cross no barrier between threads (the same regime eager mode has always
   run forward and backward under). Queue launches replay work other threads
   enqueued, which is the empirically unsafe pattern: the queue synchronizes
   the device before launching when the launcher differs from an item's
   enqueuer or from the last thread to issue device work, and the first
   direct launch after a cross-thread queue launch synchronizes once more.
   In steady state (empty queue) no synchronization happens at all.
5. **Errors are deferred, not repaired.** A launch error is retained and
   re-raised at the next drain. The queue never changes definitions, rebuilds a
   broader module, or retries a failed call.
6. **Always-available device calls still use FIFO.** Ungated copies and other
   warm calls must retain their queue position so they cannot observe an output
   before its queued producer launches.

## Into-style ABI

Asynchronous preparation requires Python to know and allocate outputs before
the extension exists. Queueable extensions therefore use an Into-style ABI:

```python
extension.call(input_specs..., output_specs..., runtime_parameters...)
```

The call writes into the supplied outputs and returns no newly allocated tensor.
Multiple-output operations receive all output specs. In-place and `out=`
operations pass the appropriate existing tensors, and zero-sized outputs may
complete without launching a kernel.

The Python descriptor mirrors only the metadata-level eligibility checks needed
to prepare a safe call. The Mojo implementation still validates the runtime
arguments and raises on disagreement.

## Modes

| Mode | How | Behavior |
|---|---|---|
| default | — | misses build on the background pool; launches queue; host reads drain |
| kill switch | `TORCH_MOJO_BACKEND_KERNEL_QUEUE=0` | no queue: every miss blocks inline at its call site and every launch happens immediately |
| test suite | `TORCH_MOJO_BACKEND_TESTING=1` | queue off, because most tests assert synchronous contracts; `TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE=1` turns it back on for the focused queue tests |

Two further knobs:

| Variable | Effect |
|---|---|
| `TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB` | run-ahead retention bound in MB. Unset, the bound is computed from the device at the first queued launch: half the smallest free-VRAM figure across the accelerators (weights are already resident by then), floored at 1 GiB, falling back to 8 GiB when no device reports memory statistics. Set, the value wins; `0` disables the bound (the pre-budget behavior). |
| `TORCH_MOJO_BACKEND_TRACE` | timestamped `[TRACE]` lines on stderr when each variant build starts and finishes. On by default; `0` silences them. |
