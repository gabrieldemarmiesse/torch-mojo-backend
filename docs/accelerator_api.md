# Accelerator API

Registering the backend makes the mojo device PyTorch's current accelerator.
From then on you reach it through
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html),
PyTorch's device-agnostic API, and nothing else in your code needs to know
which backend runs it. The same file runs on stock PyTorch with CUDA, ROCm
or Apple MPS once you remove the two registration lines. This page maps the
calls you already make with `torch.cuda` or `torch.mps` to `torch.accelerator`.

## Register the backend

Register once, at the start of your program, and take the device from
`torch.accelerator`:

```python
import torch
import torch_mojo_backend  # remove these two lines to run on stock PyTorch

torch_mojo_backend.register_mojo_devices()

device = torch.accelerator.current_accelerator()
print(device)                              # mojo
print(torch.accelerator.device_count())    # number of GPUs MAX sees

x = torch.ones(3, device=device)
print(x.device)                            # mojo:0
```

`current_accelerator()` returns the mojo device even when your torch build
also supports CUDA or MPS. The call is idempotent, so a library can make it
defensively. Each op's kernel is compiled the first time the op runs and then
cached on disk, as the [home page](index.md) explains. Set
`TORCH_MOJO_BACKEND_TRACE=0` to silence the build-timing lines printed on
stderr.

??? tip "Fall back to the CPU"
    For a script that must also run on machines without a GPU:

    ```python
    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()

    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    print(device)
    ```

    With `check_available=True`, `current_accelerator()` returns `None`
    when the accelerator has no usable device, for example when MAX finds no
    GPU.

## From `torch.cuda` and `torch.mps`

In this table, `device` is `torch.accelerator.current_accelerator()`.

| CUDA | MPS | Device-agnostic |
|---|---|---|
| `torch.cuda.is_available()` | `torch.backends.mps.is_available()` | `torch.accelerator.is_available()` |
| `torch.device("cuda")` | `torch.device("mps")` | `torch.accelerator.current_accelerator()` |
| `x.to("cuda")`, `x.cuda()`, `model.cuda()` | `x.to("mps")` | `x.to(device)`, `model.to(device)` |
| `x.to("cuda:1")` | | `x.to(torch.device(device.type, 1))` |
| `torch.zeros(3, device="cuda")` | `torch.zeros(3, device="mps")` | `torch.zeros(3, device=device)` |
| `x.is_cuda` | `x.is_mps` | `x.device.type == device.type` |
| `torch.cuda.device_count()` | `torch.mps.device_count()` | `torch.accelerator.device_count()` |
| `torch.cuda.current_device()` | | `torch.accelerator.current_device_index()` |
| `torch.cuda.set_device(i)` | | `torch.accelerator.set_device_index(i)` |
| `with torch.cuda.device(i):` | | `with torch.accelerator.device_index(i):` |
| `torch.cuda.synchronize()` | `torch.mps.synchronize()` | `torch.accelerator.synchronize()` |
| `torch.cuda.manual_seed_all(s)` | `torch.mps.manual_seed(s)` | `torch.manual_seed(s)` |
| `torch.autocast("cuda", ...)` | `torch.autocast("mps", ...)` | `torch.autocast(device.type, ...)` |
| `torch.cuda.memory_allocated()` | `torch.mps.current_allocated_memory()` | `torch.accelerator.memory_allocated()` |
| `torch.cuda.empty_cache()` | `torch.mps.empty_cache()` | `torch.accelerator.empty_cache()` |
| `torch.cuda.mem_get_info()` | | `torch.accelerator.get_memory_info()` |
| `torch.cuda.Stream()`, `torch.cuda.Event()` | `torch.mps.Event()` | `torch.Stream(device)`, `torch.Event(device)` |
| `torch.cuda.current_stream()` | | `torch.accelerator.current_stream()` |

`x.cpu()`, `x.item()`, `x.tolist()`, `non_blocking=True`, `pin_memory()` and
`DataLoader(pin_memory=True)` are unchanged. Streams and events have their
own page, [Streams and events](streams_and_events.md).

The `torch.accelerator` functions are part of torch, so which of them exist
depends on your torch version: the memory functions need torch 2.9 or newer,
and `get_memory_info()` needs torch 2.10.

!!! note "`torch.cuda.is_available()` says nothing about this backend"
    On a CPU-only torch wheel (the recommended install, see
    [Installation](index.md#installation)), `torch.cuda.is_available()` is
    `False` even while the backend runs on your NVIDIA GPU. On a CUDA wheel
    it is `True`, but it refers to PyTorch's own CUDA device. Code that picks
    its device with `"cuda" if torch.cuda.is_available() else "cpu"` therefore
    never lands on the accelerator. The same goes for
    `torch.backends.mps.is_available()`. Use `torch.accelerator` instead.

## A device-agnostic training loop

```python
import torch
import torch_mojo_backend  # remove these two lines to run on stock PyTorch

torch_mojo_backend.register_mojo_devices()

device = torch.accelerator.current_accelerator()
print(f"{device.type}: {torch.accelerator.device_count()} device(s), "
      f"current index {torch.accelerator.current_device_index()}")

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(32, 64, device=device)
y = torch.randint(0, 10, (32,), device=device)

for step in range(3):
    optimizer.zero_grad()
    with torch.autocast(device.type, dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

torch.accelerator.synchronize()
print(f"loss {loss.item():.4f}")
```

With the registration lines, it prints `mojo: 1 device(s), ...`. Removing
them is the only edit needed to run it on a CUDA machine, where it prints
`cuda: 1 device(s), ...`.

Kernels run asynchronously, like CUDA kernels. To time them with a wall
clock, call `torch.accelerator.synchronize()` before and after, and make a
warm-up call first, because the first call compiles the kernel.
[Timing GPU work](streams_and_events.md#timing-gpu-work) shows this
host-clock method and the event-based one.

## Moving data

`x.to(device)`, `model.to(device)` and `device=device` in factory functions
put tensors and modules on the accelerator. `non_blocking=True` works with
`.to()` and `.copy_()`, and `pin_memory()` returns page-locked memory from
the backend's own pinned allocator, so the usual overlap pattern works:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

host = torch.randn(1 << 20).pin_memory()        # page-locked host memory
dev = host.to(device, non_blocking=True)        # asynchronous upload
out = (dev * 2).to("cpu", non_blocking=True)    # asynchronous download

torch.accelerator.synchronize()                 # wait before reading `out`
print(out.is_pinned(), torch.equal(out, host * 2))  # True True
```

[Host-device copies](streams_and_events.md#host-device-copies) explains what
`non_blocking=True` does with pinned and with ordinary host memory, and how
to overlap copies with compute.

!!! warning "Don't hard-code the CUDA device"
    With a CUDA build of PyTorch, `"cuda"` still names PyTorch's own CUDA
    device, a different device from the accelerator even when it is the
    same GPU. Tensors created with a hard-coded `"cuda"` cannot be combined
    with accelerator tensors, and copying between the two raises
    `NotImplementedError`. Take every device from `torch.accelerator`.

!!! warning "Printing floating-point tensors"
    `print(t)` on a floating-point accelerator tensor currently raises
    `NotImplementedError`: PyTorch's tensor formatter calls `masked_select`,
    which the backend does not implement yet. Print `t.cpu()` instead.
    Integer and boolean tensors print directly.

## Selecting a GPU

A device without an index means the current device, as with `"cuda"`.
Select another one with `torch.accelerator.device_index(i)` (a context
manager) or `torch.accelerator.set_device_index(i)`, or build an indexed
device with `torch.device(device.type, i)`:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

for i in range(torch.accelerator.device_count()):
    with torch.accelerator.device_index(i):
        x = torch.ones(2, device=device)          # lands on GPU i
    y = torch.ones(2).to(torch.device(device.type, i))
    print(x.device, y.device)                     # mojo:0 mojo:0, then mojo:1 mojo:1, ...
```

Device indices follow the vendor's numbering, so on NVIDIA,
`CUDA_VISIBLE_DEVICES` restricts which GPUs the process sees, as it does for
CUDA. Set it before the process starts. To run one process per GPU, see
[Distributed training](distributed_training.md).

## Random numbers

`torch.manual_seed()` seeds the CPU and every accelerator device.
`torch.Generator(device=device)` and
`torch.random.fork_rng(device_type=device.type)` behave like their CUDA
counterparts:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

torch.manual_seed(0)                    # seeds the CPU and every device
print(torch.rand(4, device=device).cpu())
# tensor([0.3990, 0.5167, 0.0249, 0.9401]), the same values as on CUDA

g = torch.Generator(device=device).manual_seed(42)
state = g.get_state()                   # 16-byte uint8 tensor, like CUDA's
first = torch.rand(3, device=device, generator=g)
g.set_state(state)
assert torch.equal(first.cpu(), torch.rand(3, device=device, generator=g).cpu())

with torch.random.fork_rng(device_type=device.type):
    torch.rand(3, device=device)        # does not advance the outer stream
```

The generator is the Philox generator CUDA uses, with the same state layout.
On NVIDIA GPUs, a given seed produces the same `rand`, `randn`, `randint`,
`bernoulli` and dropout values as on a CUDA device of the same model, so a
seeded experiment can be compared across the two. On AMD and Apple GPUs the
integer draws match CUDA's bit for bit, while floating-point draws can differ
in the last bits.

## Mixed precision

`torch.autocast(device.type, ...)` works with `torch.bfloat16` or
`torch.float16`, and applies CUDA's per-op policy: matmuls, linear layers and
attention run in the lower precision, while softmax, log-softmax, sums, layer
norm and losses produce `float32`.

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

model = torch.nn.Linear(16, 4).to(device)
x = torch.randn(8, 16, device=device)
target = torch.randint(0, 4, (8,), device=device)

with torch.autocast(device.type, dtype=torch.bfloat16):
    logits = model(x)                                       # bfloat16
    loss = torch.nn.functional.cross_entropy(logits, target)  # float32
loss.backward()
print(logits.dtype, loss.dtype, model.weight.grad.dtype)
# torch.bfloat16 torch.float32 torch.float32
```

Without a `dtype`, autocast uses `torch.float16`, the same default as CUDA.

!!! warning "No `GradScaler` yet"
    `torch.amp.GradScaler(device.type)` does not work on this backend: ops it
    relies on are not implemented yet, and `scaler.step()` raises
    `NotImplementedError`. Train with `dtype=torch.bfloat16`, which has the
    range of `float32` and needs no loss scaling.

`torch.set_float32_matmul_precision("high")` (or `"medium"`) lets float32
matmuls use TF32 tensor cores where the backend has a TF32 kernel, which is
currently on H100-class GPUs (compute capability 9.0). On other GPUs, and
with the default `"highest"`, float32 matmuls stay in full float32.

## Memory

The memory functions of `torch.accelerator` use `torch.cuda`'s names and key
spellings:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

torch.accelerator.reset_peak_memory_stats()
x = torch.empty(64, 1024, 1024, device=device)          # 256 MiB of float32
print(torch.accelerator.memory_allocated() // 2**20)      # 256
del x
print(torch.accelerator.memory_allocated() // 2**20)      # 0
print(torch.accelerator.max_memory_allocated() // 2**20)  # 256

free, total = torch.accelerator.get_memory_info()         # MAX's view of the GPU
print(f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
```

What the numbers mean, and where they differ from CUDA:

- Allocated memory is the live tensor storage this process holds on each
  device. Views and slices share their base's storage and count once.
- Reserved memory equals allocated memory. On CUDA, `memory_reserved()` also
  includes the memory PyTorch's caching allocator keeps for reuse. This
  backend gets its memory from MAX's own memory manager, which does not
  report what it keeps in reserve, so `memory_reserved() - memory_allocated()`
  is always 0. For the same reason, the pool, segment and split breakdowns of
  `memory_stats()` are 0, and everything is under the `all` keys.
- `reset_peak_memory_stats()` resets the peak to the current usage, not to 0.
- `empty_cache()` does not return device memory to the system, because MAX
  offers no way to do that. It only frees pinned host buffers left over from
  finished copies, and it is safe to call at any time.
- `get_memory_info()` returns the `(free, total)` bytes that MAX reports.
  These can describe MAX's memory budget rather than the physical memory
  `nvidia-smi` shows, and they do not follow your allocations one for one. To
  measure your own usage, use `memory_allocated()`.

A failed allocation raises a plain `RuntimeError` whose message contains
`out of memory`, not `torch.OutOfMemoryError`. Since `torch.OutOfMemoryError`
is itself a subclass of `RuntimeError`, catching `RuntimeError` works on both
CUDA and this backend.

## Profiling

`torch.profiler` records ops on accelerator tensors like any others, and its
default activities already include the accelerator:

```python
import torch
from torch.profiler import profile

import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

a = torch.randn(1024, 1024, device=device)
torch.relu(a @ a)                        # compile the kernels before profiling
torch.accelerator.synchronize()

with profile() as prof:
    for _ in range(5):
        b = torch.relu(a @ a)
    torch.accelerator.synchronize()

print(prof.key_averages().table(row_limit=8))
prof.export_chrome_trace("trace.json")   # open in Perfetto or chrome://tracing
```

The CPU-side timeline (ops, their inputs, the trace export) works with any
torch build. Rows for the GPU kernels themselves, with their device times,
appear only with a CUDA build of torch on NVIDIA, because that build's
profiler sees the kernels the backend launches. Warm up before profiling,
since an op's first call includes its kernel compilation.

With a CPU-only torch build, the older profiler gives a device time per op:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()
device = torch.accelerator.current_accelerator()

a = torch.randn(1024, 1024, device=device)
torch.relu(a @ a)                        # compile the kernels before profiling
torch.accelerator.synchronize()

with torch.autograd.profiler.profile(use_device=device.type) as prof:
    for _ in range(5):
        b = torch.relu(a @ a)
    torch.accelerator.synchronize()

print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
```

It times each op with a pair of events around it, so when the GPU sits idle
waiting for the host to queue the next kernel, that wait counts as device
time too. Apple GPUs have no timing events, so neither profiler reports
device times there.

## Saving and loading

!!! warning "Save CPU copies"
    `torch.save` of an accelerator tensor, and `torch.load` with
    `map_location` set to the accelerator, currently fail with
    `NotImplementedError`, because a storage op PyTorch uses for them is not
    implemented yet. Move tensors to the CPU to save them, and load on the
    CPU:

    ```python
    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()
    device = torch.accelerator.current_accelerator()

    model = torch.nn.Linear(4, 2).to(device)

    # Save: copy the tensors to the CPU first.
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "ckpt.pt")

    # Load: read on the CPU; load_state_dict copies into the device parameters.
    restored = torch.nn.Linear(4, 2).to(device)
    restored.load_state_dict(torch.load("ckpt.pt", map_location="cpu"))
    print(restored.weight.device)            # mojo:0
    ```

## Processes and `fork`

The backend has CUDA's restriction on `fork`: a process forked after the
device was initialized cannot use it, and any op there raises a
`RuntimeError` that points to the `spawn` start method. `DataLoader` workers
that only produce CPU tensors are unaffected and run normally, with
`pin_memory=True` and a `.to(device, non_blocking=True)` in the main process.
Keep device work out of the dataset and in the main process.

## Not supported

- `torch.accelerator.get_device_capability()` raises
  `RuntimeError: Backend doesn't support getting device capabilities.`
- `torch.amp.GradScaler`, see [Mixed precision](#mixed-precision).
- CUDA graphs (`torch.cuda.graph`, `CUDAGraph`, `make_graphed_callables`) have
  no equivalent.

## Hardware

The backend runs on the GPUs Mojo supports: NVIDIA, AMD and Apple GPUs.
Most of the testing so far was on H100, MI300X and Apple M4 (see the
[home page](index.md)). The per-vendor differences that matter for this
page:

=== "NVIDIA"

    - Seeded random numbers match CUDA's bit for bit.
    - `CUDA_VISIBLE_DEVICES` selects the visible GPUs.
    - A CPU-only torch wheel is enough. With a CUDA wheel, PyTorch's own
      `cuda` device exists alongside the accelerator, as a separate device.

=== "AMD"

    - Install the CPU wheel of torch, as the [home page](index.md) explains.
    - Integer random draws match CUDA's; floating-point ones can differ in
      the last bits.

=== "Apple"

    - `float64` is not available on Apple GPUs, and neither are the ops that
      need it.
    - `torch.Stream(device)` is always the default stream (as on MPS), and
      events cannot be recorded; see
      [Apple GPUs](streams_and_events.md#apple-gpus).
    - Host-to-device copies are synchronous, and the profilers report no
      device times.
    - With Xcode 26 or newer, the Metal compiler is a separate download:
      `xcodebuild -downloadComponent MetalToolchain`. Without it, kernel
      builds fail with `Metal Compiler failed to compile metallib`.

A process uses one vendor only: the first of NVIDIA, AMD and Apple that has
a GPU. On a machine with both an NVIDIA and an AMD card, only the NVIDIA
GPUs are used.

Op coverage is still partial. An op the backend does not implement raises
`NotImplementedError` with the op's name, and never falls back to the CPU
silently.

## `torch.compile`

`torch.compile(model, backend=mojo_backend)` is documented on the
[home page](index.md). Its examples use `cuda` tensors, and compiling with
tensors on the accelerator is listed there as not supported yet. This page
covers eager execution only.
