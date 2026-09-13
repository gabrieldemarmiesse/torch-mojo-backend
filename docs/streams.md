# Streams and events on the mojo device

The mojo device supports the device-generic accelerator stream API:

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

`torch.Stream(device="mojo")`, `torch.Event(device="mojo", enable_timing=True)`,
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

`Stream.stream_id` is the stream's MAX `DeviceContext` pointer (torch treats
a stream id as an opaque int); the native `CUstream`/`hipStream_t` is
`Stream.native_handle`.

## Why no Python patch is needed

PyTorch's generic `torch.Stream`/`torch.Event` route through a C++ device
guard; for a PrivateUse1 device with only the default stub guard, every
stream it mints is stream id 0 and every wait/record is a silent no-op — so
a Python-implemented backend has to patch `torch.Stream`/`torch.Event` at
the class level to hand back objects of its own.

`native/csrc/shim_runtime.cpp` registers a real C++
`PrivateUse1HooksInterface` (device guard, streams, events) with torch's own
dispatcher instead, so `torch.Stream`/`torch.Event` work for the `mojo`
device through the ordinary generic path, exactly like CUDA, and
`monkeypatching.py` holds one unrelated patch.

## Execution semantics

Kernels launch on the device's **current stream** (`ctx_for(t.device)` in
`native/mojo/abi.mojo`/`device.mojo`), so `with torch.Stream(...):` really
moves execution, not just bookkeeping.

One rule carried over from CUDA applies unchanged: a tensor produced on one
stream must be ordered (event or `wait_stream`) before another stream —
including external consumers — touches it. That includes a readback:
`t.cpu()` and `t.item()` issue their copy on the *current* stream, so
reading a tensor a side stream produced needs `torch.accelerator.synchronize()`
or a `wait_stream`, exactly as on CUDA. `tensor.record_stream(stream)`
is supported: the backend turns it into a MAX event the owning stream waits
on before the buffer is released back to the allocator (see
`docs/native_backend.md`, "Streams").

`torch.accelerator.synchronize()` and `torch.mojo.synchronize()` are that
host barrier over *every* stream of the device. They only work because
`torch.mojo` defines `_lazy_init()`: torch's `_accelerator_synchronizeDevice`
returns early for a lazy-init-capable device type it has not marked
initialized, and torch marks PrivateUse1 initialized by calling
`torch.<backend>._lazy_init()` from `device_lazy_init()` on the first device
tensor. Without that method the flag never flips and both calls are silent
no-ops — the device guard is never reached — which is a readback race waiting
to happen (`tests/native/test_stream_ordering.py` pins it).
