# Streams and events

The backend implements PyTorch's device-agnostic stream and event API:
`torch.Stream`, `torch.Event` and the stream functions of `torch.accelerator`.
On NVIDIA and AMD GPUs, each stream is an independent device queue and events
work as they do on CUDA, so code written for CUDA streams carries over once a
few calls are renamed. Apple GPUs have a single queue and no events; see
[Apple GPUs](#apple-gpus).

This page assumes the backend is already registered, and `device` in the
examples is `torch.accelerator.current_accelerator()`.
[Accelerator API](accelerator_api.md) covers registration, device selection
and the rest of `torch.accelerator`.

## How work is ordered

- Every op on an accelerator tensor is queued on the current stream of its device,
  and the Python call returns without waiting for the kernel to run. Ops
  queued on the same stream run in the order they were issued.
- When a program starts, the current stream is the device's default stream
  (`stream_id == 0`). Each thread has its own current stream, and a new
  thread starts on the default stream.
- The CPU waits for the GPU in two cases only:
    - When it has to give you a value: `.cpu()`, `.item()`, `.tolist()`, or
      a blocking `copy_()` into a CPU tensor. These copy on the current
      stream and wait for that stream alone.
    - When you ask it to: `torch.accelerator.synchronize()` waits for every
      stream of the device, `stream.synchronize()` waits for one stream, and
      `event.synchronize()` waits for one event.
- The first time an op runs with a new combination of dtypes, its kernel is
  compiled, and the call blocks until compilation is done. The compiled
  kernel is cached on disk. Run a warm-up iteration before you time anything.

## Streams

### From `torch.cuda`

| CUDA | Device-agnostic |
|---|---|
| `torch.cuda.Stream()` | `torch.Stream(device)` |
| `torch.cuda.Event(enable_timing=True)` | `torch.Event(device, enable_timing=True)` |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` |
| `torch.cuda.set_stream(s)` | `torch.accelerator.set_stream(s)` |
| `with torch.cuda.stream(s):` | `with s:` |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` |
| `s.cuda_stream` | `s.native_handle` (torch 2.11+) |

The methods have their CUDA names: `wait_stream`, `wait_event`,
`record_event`, `query` and `synchronize` on streams, and `record`, `wait`,
`query`, `synchronize` and `elapsed_time` on events. Code written this way
also runs on stock PyTorch with CUDA.

### Creating a stream and making it current

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

main = torch.accelerator.current_stream()   # the default stream at start-up
side = torch.Stream(device)                 # a new, independent stream
print(main)  # torch.Stream device_type=mojo, device_index=0, stream_id=0
print(side)  # torch.Stream device_type=mojo, device_index=0, stream_id=1

with side:
    assert torch.accelerator.current_stream() == side
    x = torch.full((1000,), 2.0, device=device)  # queued on `side`

assert torch.accelerator.current_stream() == main  # restored on exit

side.synchronize()          # wait for everything queued on `side`
print(x.sum().item())       # 2000.0
```

`torch.accelerator.set_stream(s)` does the same without a `with` block: the
stream stays current until you set another one.

- Create streams once, before a loop, and reuse them. Every
  `torch.Stream(...)` call creates a new device stream (you can watch
  `stream_id` count up), and the stream stays alive until the process exits.
  CUDA differs here: `torch.cuda.Stream()` hands out streams from a fixed
  pool.
- `torch.Stream(device)` creates the stream on the current device, and
  `torch.Stream(torch.device(device.type, 1))` targets another GPU.
  Entering a stream that belongs to another device also makes that device
  current until the block exits, the same behavior as CUDA.
- `priority=` is passed to the driver when the stream is created. The
  convention is CUDA's: a lower number means a higher priority, and values
  outside the device's range are clamped. The generic `torch.Stream` has no
  `priority` attribute, so you cannot read the value back.
- `stream.query()` returns whether all the work queued on the stream has
  finished, without blocking. `stream.synchronize()` blocks until it has.
- `stream.native_handle` (torch 2.11 and later) returns the underlying
  `CUstream` or `hipStream_t`. Use the handle to queue work from outside
  PyTorch on the same device queue.

## Events

### Timing GPU work

=== "NVIDIA and AMD GPUs"

    ```python
    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()
    device = torch.accelerator.current_accelerator()

    a = torch.randn(2048, 2048, device=device)
    a @ a  # warm up: the first call of an op compiles its kernel

    start = torch.Event(device, enable_timing=True)
    end = torch.Event(device, enable_timing=True)

    start.record()          # records on the current stream
    for _ in range(10):
        b = a @ a
    end.record()

    end.synchronize()       # elapsed_time needs both events to have completed
    print(f"{start.elapsed_time(end) / 10:.3f} ms per matmul")
    ```

=== "Any GPU, including Apple"

    ```python
    import time

    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()
    device = torch.accelerator.current_accelerator()

    a = torch.randn(2048, 2048, device=device)
    a @ a  # warm up: the first call of an op compiles its kernel

    torch.accelerator.synchronize()   # start from an idle device
    start = time.perf_counter()
    for _ in range(10):
        b = a @ a
    torch.accelerator.synchronize()   # wait until the GPU is done
    elapsed_ms = 1e3 * (time.perf_counter() - start)
    print(f"{elapsed_ms / 10:.3f} ms per matmul")
    ```

    This measures wall time, so it includes the time the CPU spends queuing
    the ops. Events measure device time only.

### What events do

- `event.record()` records the event on the current stream, and
  `event.record(stream)` on the given stream. `stream.record_event()` creates
  a new event and records it in one call.
- `event.query()` returns whether the work before the event has finished,
  without blocking. `event.synchronize()` blocks the CPU until it has.
- `event.wait(stream)` and `stream.wait_event(event)` make a stream wait for
  the event on the GPU without blocking the CPU. With no argument,
  `event.wait()` applies to the current stream.
- `start.elapsed_time(end)` returns milliseconds, under the same contract as
  CUDA events: it raises `ValueError` unless both events were created with
  `enable_timing=True` and recorded, and `RuntimeError` unless both have
  completed.
- An event that was never recorded counts as complete: `query()` returns
  `True`, and `synchronize()` and `wait()` do nothing, as in PyTorch.

## Working with several streams

Streams run independently, so you order work across them yourself. The rules
are the same as for CUDA:

1. A stream that reads a tensor must first wait for the stream that wrote
   it. `consumer.wait_stream(producer)` waits for everything queued on
   `producer` so far, and `consumer.wait_event(event)` waits for the work
   before one recorded event. Neither blocks the CPU.
2. Reading a result back counts as reading. `.cpu()` and `.item()` copy on
   the *current* stream, so to read a tensor that a side stream produced,
   make the current stream wait for the side stream first, read it inside
   `with side:`, or synchronize the device.
3. When a stream uses a tensor that was allocated on another stream, call
   `tensor.record_stream(stream)`. Freed memory is reused by later work on
   the stream that allocated it, and `record_stream` keeps the memory from
   being reused until the work already queued on `stream` has finished.

Here a side stream computes a small reduction while the main stream runs a
matmul, and then the two results are combined:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

main = torch.accelerator.current_stream()
side = torch.Stream(device)           # create once, reuse

x = torch.randn(1024, 1024, device=device)   # produced on `main`

side.wait_stream(main)          # rule 1: `side` must not read x before it is written
with side:
    col_max = x.abs().amax(dim=0)   # runs on `side` ...
y = x @ x                           # ... while this runs on `main`

main.wait_stream(side)          # rules 1 and 2: `main` must not read col_max too early
x.record_stream(side)           # rule 3: x was used on `side`,
col_max.record_stream(main)     # and col_max is used on `main`

result = y / col_max
print(result.cpu()[0, :4])      # the copy runs on `main`, after the join
```

!!! warning "Forgetting a wait does not raise an error"
    Without `main.wait_stream(side)`, the example above still runs, but
    `y / col_max` and the readback can use `col_max` before `side` has
    written it, and the result can silently be wrong. On NVIDIA and AMD GPUs,
    every cross-stream dependency needs an explicit wait.

## Host-device copies

`tensor.to(device, non_blocking=True)`, `tensor.to("cpu", non_blocking=True)`
and `dst.copy_(src, non_blocking=True)` are supported. What
`non_blocking=True` does depends on the kind of host memory:

| `non_blocking=True` | Pinned with `tensor.pin_memory()` | Pageable host memory |
|---|---|---|
| Host to device | Queued on the current stream, and the call returns at once. The copy reads your tensor when it runs, so do not modify the tensor until the copy is done. | The data is copied into a staging buffer before the call returns, so you can reuse the source right away. The device copy itself is queued. |
| Device to host | Queued on the current stream, and the call returns at once. Wait for the copy before reading the result. | The copy completes before the call returns. |

- `tensor.to("cpu", non_blocking=True)` puts its result in pinned memory, so
  it is asynchronous and needs a wait before you read it.
- A download that has to convert the dtype or change the memory layout on the
  CPU completes before it returns.
- Pinned memory belongs to the device that was current when it was pinned.
  Copies between it and another device treat it as pageable memory.
- Without `non_blocking=True`, you can read or reuse the host tensor as soon
  as the call returns.

!!! warning "Pinned memory on a CUDA build of PyTorch"
    Pin memory with `tensor.pin_memory()`, which `DataLoader(pin_memory=True)`
    also uses. If you run a CUDA build of PyTorch on a machine with a CUDA
    GPU, factory functions called with `pin_memory=True`, such as
    `torch.empty(..., pin_memory=True)`, allocate CUDA's pinned memory
    instead. The backend treats that memory like pageable memory: copies are
    correct, but they are not asynchronous. On a CPU-only build of PyTorch,
    both spellings give the backend's pinned memory.

### Downloading without blocking

To wait for the copy alone rather than for the whole device, record an event
after it:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

x = torch.arange(1 << 20, dtype=torch.float32, device=device) * 2

host = x.to("cpu", non_blocking=True)   # returns at once, into pinned memory
copied = torch.Event(device)
copied.record()                         # marks the end of the copy on the stream
print(host.is_pinned())                 # True

# ... other CPU work here, while the copy runs ...

copied.synchronize()                    # now `host` holds the data
print(host[:4])                         # tensor([0., 2., 4., 6.])
```

### Overlapping uploads with compute

In the usual prefetching pattern, a dedicated copy stream uploads the next
batch while the main stream computes on the current one:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

main = torch.accelerator.current_stream()
copy_stream = torch.Stream(device)
weight = torch.randn(1024, 256, device=device)

# Pinned host memory lets the upload run in the background.
batches = [torch.randn(8192, 1024).pin_memory() for _ in range(8)]


def upload(batch: torch.Tensor) -> torch.Tensor:
    with copy_stream:
        return batch.to(device, non_blocking=True)


losses = []
next_batch = upload(batches[0])
for i in range(len(batches)):
    main.wait_stream(copy_stream)   # batch i has arrived
    batch = next_batch
    batch.record_stream(main)       # allocated on copy_stream, used on main
    if i + 1 < len(batches):
        next_batch = upload(batches[i + 1])   # overlaps with the compute below
    losses.append((batch @ weight).square().mean())

print(torch.stack(losses).cpu())
```

Each host batch here is a separate pinned tensor that is never modified. If
you refill a single pinned buffer instead, wait for its previous upload to
finish before you write to it again, for example with an event recorded on
`copy_stream` after the upload.

## Apple GPUs

On Apple GPUs (Metal), the backend has a single queue per device, like
PyTorch's MPS backend:

- `torch.Stream(device)` returns the default stream, whatever the priority:
  its `stream_id` is 0, and it compares equal to every other stream of the
  device. `with stream:`, `torch.accelerator.set_stream` and
  `torch.accelerator.current_stream` work.
  All the work runs in order on the one queue, so no cross-stream waits are
  needed.
- `stream.synchronize()` and `torch.accelerator.synchronize()` work. `stream.query()` waits for the
  queue to finish and then returns `True`.
- `tensor.record_stream(stream)` works.
- Host-to-device copies complete before they return, even with
  `non_blocking=True`.
- Events are not supported, because MAX does not implement Metal events.
  Creating a `torch.Event` works, but recording one raises
  `RuntimeError: mojo backend: events are not supported on Apple GPU (Metal)`.
  That covers `event.record()`, `stream.record_event()` and
  `stream.wait_stream()`, which PyTorch implements with an event. With no
  recorded events, `elapsed_time()` is not available either.

For code that runs on every GPU, skip the wait when the two streams are the
same stream (they always are on Apple GPUs), and time work with
`torch.accelerator.synchronize()` and a host clock, as in the
"Any GPU, including Apple" tab [above](#timing-gpu-work):

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()


def wait_for(waiter: torch.Stream, producer: torch.Stream):
    # On Apple GPUs every stream is the default stream: the two compare
    # equal, the queue is already in order, and wait_stream() would raise.
    if waiter != producer:
        waiter.wait_stream(producer)


main = torch.accelerator.current_stream()
side = torch.Stream(device)
x = torch.randn(1024, 1024, device=device)

wait_for(side, main)
with side:
    y = x * 2
wait_for(main, side)
print(torch.equal(y.cpu(), x.cpu() * 2))           # True
```

## Not supported

- Interprocess events: `torch.Event(interprocess=True)` is accepted but
  creates an ordinary event, and `event.ipc_handle()` and
  `torch.Event.from_ipc_handle()` raise `NotImplementedError`. For multi-GPU
  and multi-process work, see [Distributed training](distributed_training.md).
- Graph capture: there is no equivalent of `torch.cuda.graph` or
  `torch.cuda.CUDAGraph`, and no stream capture.
- External streams: nothing plays the role of `torch.cuda.ExternalStream`,
  which wraps a stream created outside PyTorch.
- Reading stream priorities: priorities are applied when a stream is
  created, but there is no `stream.priority` attribute and no
  `priority_range()`.
- Events and independent streams on Apple GPUs, as described
  [above](#apple-gpus).
