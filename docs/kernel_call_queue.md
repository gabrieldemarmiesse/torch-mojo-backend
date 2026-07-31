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

Builds are protected by a process-level lock and written to temporary files
before an atomic rename. Concurrent requests for one identity compile it once,
and an interrupted compiler cannot leave a partial file that looks valid.

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
3. **Keep-alive.** Queued raw pointers do not own their tensors, so prepared
   calls retain every input, output, and intermediate allocation until the
   queue drains.
4. **Thread order.** Device work is ordered within an enqueuing thread. The
   direct path and queue executor synchronize when execution changes threads.
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
| default | — | misses build in parallel; launches queue; reads drain |
| sequential | `TMB_NO_TRIGGER_DEFER=1` | every miss blocks inline |
| kill switch | `TMB_KERNEL_QUEUE=0` | same as sequential |
| test suite | `TORCH_MOJO_BACKEND_TESTING=1` | queue off; `TMB_FORCE_KERNEL_QUEUE=1` enables it for focused queue tests |
