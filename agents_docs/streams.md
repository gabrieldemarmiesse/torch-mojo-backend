# Streams and events on the mojo device

The mojo device supports the device-generic accelerator stream API. On
CUDA and ROCm, streams and events support asynchronous coordination:

```python
s = torch.Stream(device=torch.accelerator.current_accelerator())
cur = torch.accelerator.current_stream()
s.wait_stream(cur)
with s:
    ...                      # torch.accelerator.current_stream() is now s
e = s.record_event()         # a torch.Event
cur.wait_event(e)
e.synchronize()
```

On CUDA/ROCm, `torch.Stream(device="mojo")`,
`torch.Event(device="mojo", enable_timing=True)`,
`torch.accelerator.current_stream()` / `set_stream()`, and the `torch.mojo`
module equivalents (`Stream`, `Event`, `current_stream`, `default_stream`,
`set_stream`, `stream`, all in `torch_mojo_backend/native/device_module.py`)
work, with `query` / `synchronize` / `wait_event` / `wait_stream` /
`record_event` / `elapsed_time` backed by real MAX events on real device
streams. `isinstance(s, torch.Stream)` holds without qualification: a mojo
stream is a plain `torch.Stream`/`torch._C.Stream`, not a Python subclass.
Not supported: interprocess events (`from_ipc_handle`), stream priorities
(accepted, always 0 — MAX has them, this backend does not pass the argument
through yet), and graph capture (there is no CUDA-graph equivalent; a
generic PrivateUse1 `torch.Stream` in this torch version exposes no
`is_capturing()` at all, so there is nothing to stub).

`Stream.stream_id` is an index into the device's MAX context views, with 0
identifying the default stream; the native `CUstream`/`hipStream_t` is
`Stream.native_handle`.

## Apple GPU (Metal)

`torch.Stream(device="mojo")` always identifies the device's default stream
on Apple GPUs, matching PyTorch's MPS stream behavior. Repeated construction
returns equal stream objects with `stream_id == 0`; priorities are accepted
and ignored. Context managers, current/default stream selection,
`stream.synchronize()`, and `stream.query()` work. Query currently waits for
the stream to finish before returning `True`.

MAX does not implement Metal events. Recording an event raises
`RuntimeError: mojo backend: events are not supported on Apple GPU (Metal)`.
This includes `event.record()`, `stream.record_event()`, and
`stream.wait_stream()`, which PyTorch implements using an event. Use
`stream.synchronize()` or `torch.mojo.synchronize()` for a CPU barrier.
PyTorch creates generic `torch.Event` objects lazily, so construction alone
succeeds; querying, synchronizing or waiting on an unrecorded event retains
PyTorch's no-op behavior. No event is emulated.

## Why no Python patch is needed

PyTorch's generic `torch.Stream`/`torch.Event` route through a C++ device
guard; for a PrivateUse1 device with only the default stub guard, every
stream it mints is stream id 0 and every wait/record is a silent no-op — so
a Python-implemented backend has to patch `torch.Stream`/`torch.Event` at
the class level to hand back objects of its own.

`native/csrc/shim_runtime.cpp` registers a real C++
`PrivateUse1HooksInterface` (device guard, streams, events) with torch's own
dispatcher instead, so `torch.Stream`/`torch.Event` work for the `mojo`
device through the ordinary generic path, exactly like CUDA. The Dynamo
compatibility patch below changes tracing, not these native classes.

## Execution semantics

### torch.compile

With PyTorch 2.11's generic stream tracing, `mojo_backend` preserves stream
contexts, stream/event waits and event recording on CUDA and mojo tensors.
This also works with `fullgraph=True`, streams constructed inside the
function, nested contexts, mutation, and backward.

Dynamo places stream contexts in FX node metadata, but AOTAutograd executes
ordinary stream/event methods while tracing and omits them from its tensor
graph. `torch_compile_backend/streams.py` therefore partitions the Dynamo
graph before AOT. The runtime wrapper retains stream/event operations in
program order; contiguous tensor regions are compiled to MAX and invoked
under the requested torch stream. Stream objects are resolved from Dynamo's
runtime table on every call, rather than retaining tracing-time streams.
`preserve_dynamo_stream_contexts()` retains the entire nested context stack
in extra metadata: the innermost stream alone cannot describe an active
outer context on a different device. This tracing patch is installed at
package import so it also covers CUDA-only use without device registration.

MAX still executes these regions on its own shared device queue. A tiny
per-device DLPack handoff orders the graph after the current torch stream
and orders that stream after the graph. The handoff is refreshed on every
invocation, including when tensor buffers are cached or the graph has no GPU
inputs/outputs. CUDA/ROCm waits are device-side; Metal uses its existing host
synchronization path. Tensor calculations remain compiled; stream/event
orchestration runs on the host, as it does in eager mode.

This preserves ordering, but does **not** preserve overlap between independent
compiled regions: MAX's shared queue can serialize them. We do not yet use
`ops.side_stream()` to assign compiled regions to independent MAX queues.
That optimization must also retain the external torch-stream handoffs and
runtime event operations; stream metadata alone is not sufficient.

When CUDA and mojo coexist, `fix_dynamo_current_stream_device()` in
`monkeypatching.py` repairs PyTorch 2.11's current-stream lookup: its
symbolic stack, CUDA query handler, and runtime/guard reconstruction must
respect the requested device type instead of defaulting to mojo. The patch
is installed by `register_mojo_devices()`.

### Native eager execution

Native eager kernels launch on the device's **current stream** (`ctx_for(t.device)` in
`tmb/backend/abi.mojo`/`device.mojo`). On CUDA/ROCm a new stream selects an
independent queue; on Metal it selects the same default queue.

One rule carried over from CUDA applies unchanged: a tensor produced on one
stream must be ordered (event or `wait_stream`) before another stream —
including external consumers — touches it. That includes a readback:
`t.cpu()` and `t.item()` issue their copy on the *current* stream, so
reading a tensor a side stream produced needs `torch.accelerator.synchronize()`
or a `wait_stream`, exactly as on CUDA. `tensor.record_stream(stream)`
is supported: the backend turns it into a MAX event the owning stream waits
on before the buffer is released back to the allocator (see
`agents_docs/native_backend.md`, "Streams").

`torch.accelerator.synchronize()` and `torch.mojo.synchronize()` are that
host barrier over *every* stream of the device. They only work because
`torch.mojo` defines `_lazy_init()`: torch's `_accelerator_synchronizeDevice`
returns early for a lazy-init-capable device type it has not marked
initialized, and torch marks PrivateUse1 initialized by calling
`torch.<backend>._lazy_init()` from `device_lazy_init()` on the first device
tensor. Without that method the flag never flips and both calls are silent
no-ops — the device guard is never reached — which is a readback race waiting
to happen (`tests/native/test_stream_ordering.py` pins it).
