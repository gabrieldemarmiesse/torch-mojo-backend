# Accelerator API

After registration, `mojo` is an ordinary PyTorch device. Tensors on it are
plain `torch.Tensor`s, `.to()` moves modules onto it, and autograd and
optimizers work as usual. This page maps the calls you already make on CUDA
or MPS (`torch.cuda.xxx`, `.to("cuda")`, `device="mps"`) to their
equivalents on the `mojo` device.

!!! tip "Porting from CUDA or MPS"
    Replace `"cuda"` or `"mps"` with `"mojo"`, and `torch.cuda.xxx` with
    `torch.mojo.xxx`. Better still, write your code against
    [`torch.accelerator`](#device-generic-code-with-torchaccelerator): the
    same file then runs on the `mojo` device and, unchanged, on stock PyTorch
    with CUDA, ROCm or MPS.

## Register the device

The `mojo` device does not exist until you call `register_mojo_devices()`.
Call it once, at the start of your program, before any code names `"mojo"`:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

print(torch.accelerator.current_accelerator())  # mojo
print(torch.accelerator.device_count())         # number of GPUs MAX sees

a = torch.ones(3, device="mojo")                 # the current mojo device
b = torch.ones(3, device="mojo:0")               # an explicit index
c = torch.ones(3, device=torch.device("mojo", 0))
print(a.device, b.device, c.device)              # mojo:0 mojo:0 mojo:0
```

After the call:

- The `mojo` device type is valid, with the usual spellings: `"mojo"`,
  `"mojo:0"`, `"mojo:1"`, `torch.device("mojo", 1)`. Before it,
  `torch.device("mojo")` fails with `RuntimeError: Expected one of cpu, cuda, ...`.
- `torch.mojo` exists. It is the `torch.cuda`-like module described
  [below](#the-torchmojo-module), and `torch.get_device_module("mojo")`
  returns it.
- Tensors and modules have a `.mojo()` method, and tensors an `.is_mojo`
  attribute, the counterparts of `.cuda()` and `.is_cuda`.
- `mojo` is the current accelerator:
  `torch.accelerator.current_accelerator()` returns `device(type='mojo')`,
  even when your torch build also supports CUDA or MPS.

The call is idempotent, so a library can make it defensively. Each op's
kernel is compiled the first time the op runs and then cached on disk, as the
[home page](index.md) explains. Set `TORCH_MOJO_BACKEND_TRACE=0` to silence
the build-timing lines printed on stderr.

## Moving tensors and modules

Only the device name changes:

=== "mojo"

    ```python
    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()

    model = torch.nn.Linear(4, 2).mojo()      # or .to("mojo")
    x = torch.randn(8, 4).mojo()              # or .to("mojo")
    w = torch.zeros(4, device="mojo")         # allocate directly on the device

    y = model(x)
    print(x.is_mojo, y.device)                # True mojo:0
    torch.mojo.synchronize()
    print(y.cpu().shape)                      # back to the host
    ```

=== "CUDA"

    ```python
    import torch

    model = torch.nn.Linear(4, 2).cuda()      # or .to("cuda")
    x = torch.randn(8, 4).cuda()              # or .to("cuda")
    w = torch.zeros(4, device="cuda")         # allocate directly on the device

    y = model(x)
    print(x.is_cuda, y.device)                # True cuda:0
    torch.cuda.synchronize()
    print(y.cpu().shape)                      # back to the host
    ```

=== "Apple MPS"

    ```python
    import torch

    model = torch.nn.Linear(4, 2).to("mps")
    x = torch.randn(8, 4).to("mps")
    w = torch.zeros(4, device="mps")          # allocate directly on the device

    y = model(x)
    print(x.is_mps, y.device)                 # True mps:0
    torch.mps.synchronize()
    print(y.cpu().shape)                      # back to the host
    ```

| You write today | On the `mojo` device |
|---|---|
| `x.to("cuda")`, `x.to("cuda:1")`, `x.to("mps")` | `x.to("mojo")`, `x.to("mojo:1")` |
| `x.cuda()`, `x.cuda(1)`, `model.cuda()` | `x.mojo()`, `x.mojo(1)`, `model.mojo()` |
| `x.is_cuda`, `x.is_mps` | `x.is_mojo`, or `x.device.type == "mojo"` |
| `torch.zeros(3, device="cuda")` | `torch.zeros(3, device="mojo")` |
| `torch.set_default_device("cuda")`, `with torch.device("cuda"):` | the same with `"mojo"` |
| `x.cpu()`, `x.item()`, `x.tolist()` | unchanged; they wait for the device |
| `x.to(dev, non_blocking=True)`, `x.pin_memory()`, `DataLoader(pin_memory=True)` | unchanged, see [below](#asynchronous-copies-and-pinned-memory) |
| `x.to("cuda")` on a tensor of another accelerator | not supported between `mojo` and `cuda` or `mps`: go through the CPU |

A device string without an index (`"mojo"`) means the current mojo device,
the same convention as `"cuda"`; see [Selecting a GPU](#selecting-a-gpu).

!!! warning "`cuda` and `mojo` are two different devices"
    With a CUDA build of PyTorch, the same GPU can show up both as `cuda:0`
    and as `mojo:0`, and PyTorch treats the two as separate devices:
    `x.to("cuda")` on a mojo tensor, or `x.to("mojo")` on a CUDA tensor,
    raises `NotImplementedError`. Go through the CPU (`x.cpu().to("mojo")`),
    or keep the whole program on one device.

!!! warning "Printing floating-point tensors"
    `print(t)` on a floating-point mojo tensor currently raises
    `NotImplementedError`: PyTorch's tensor formatter calls `masked_select`,
    which the mojo device does not implement yet. Print `t.cpu()` instead.
    Integer and boolean tensors print directly.

### Asynchronous copies and pinned memory

`.to()` and `.copy_()` both honor `non_blocking=True`, and `pin_memory()`
returns page-locked memory from the mojo device's own pinned allocator, so
the usual overlap pattern works:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

host = torch.randn(1 << 20).pin_memory()        # page-locked host memory
dev = host.to("mojo", non_blocking=True)        # asynchronous upload
out = (dev * 2).to("cpu", non_blocking=True)    # asynchronous download

torch.accelerator.synchronize()                 # wait before reading `out`
print(out.is_pinned(), torch.equal(out, host * 2))  # True True
```

[Host-device copies](streams_and_events.md#host-device-copies) explains what
`non_blocking=True` does with pinned and with ordinary host memory, and how
to overlap copies with compute.

## Device-generic code with `torch.accelerator`

[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
is PyTorch's device-agnostic API: it talks to whichever accelerator the
process has, and registration makes that accelerator `mojo`. Code written
against it therefore runs on the `mojo` device, and the same file runs on
stock PyTorch on CUDA, ROCm or Apple MPS. It is the recommended way to write
code for the `mojo` device.

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

With the two mojo lines, it prints `mojo: 1 device(s), ...`. Removing them
is the only edit needed to run it on a CUDA machine, where it prints
`cuda: 1 device(s), ...`.

??? tip "Make the mojo backend optional, and fall back to the CPU"
    For a script that must run whether or not `torch-mojo-backend` is
    installed, and also on machines without a GPU:

    ```python
    import torch

    try:
        import torch_mojo_backend

        torch_mojo_backend.register_mojo_devices()
    except ImportError:
        pass  # stock PyTorch: CUDA, ROCm, MPS, ... as usual

    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    print(device)
    ```

    With `check_available=True`, `current_accelerator()` returns `None`
    instead of `mojo` when the accelerator has no usable device, for example
    when MAX finds no GPU.

On the `mojo` device, `torch.accelerator` provides:

| Function | On `mojo` |
|---|---|
| `current_accelerator()`, `is_available()`, `device_count()` | `device(type='mojo')`; `is_available()` is `True` when MAX sees at least one GPU |
| `current_device_index()`, `set_device_index(i)`, `device_index(i)` (context manager) | get or select the current mojo device |
| `synchronize(device=None)` | waits for all the work queued on that device |
| `current_stream()`, `set_stream(s)` | see [Streams and events](streams_and_events.md) |
| `memory_allocated()`, `max_memory_allocated()`, `memory_reserved()`, `max_memory_reserved()`, `memory_stats()`, `reset_peak_memory_stats()`, `reset_accumulated_memory_stats()`, `empty_cache()`, `get_memory_info()` | work, with the meanings given in [Memory](#memory) |
| `get_device_capability()` | raises `RuntimeError: Backend doesn't support getting device capabilities.` Use `torch.mojo.get_device_capability()` for the CUDA compute capability. |

These functions are part of torch, so which of them exist depends on your
torch version: the memory functions need torch 2.9 or newer, and
`get_memory_info()` needs torch 2.10.

Kernels run asynchronously, like CUDA kernels. To time them with a wall
clock, call `synchronize()` before and after, and make a warm-up call first,
because the first call compiles the kernel.
[Timing GPU work](streams_and_events.md#timing-gpu-work) shows this
host-clock method and the event-based one.

### Selecting a GPU

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

for i in range(torch.accelerator.device_count()):
    with torch.accelerator.device_index(i):
        x = torch.ones(2, device="mojo")          # lands on mojo:i
        print(x.device, torch.mojo.get_device_name())
```

`torch.mojo.set_device(i)` and `with torch.mojo.device(i):` do the same
thing under their `torch.cuda` names. Mojo device indices follow the vendor's
numbering, so on NVIDIA, `CUDA_VISIBLE_DEVICES` restricts which GPUs the
process sees, for mojo as for CUDA. Set it before the process starts. To run
one process per GPU, see [Distributed training](distributed_training.md).

## The `torch.mojo` module

`torch.mojo` mirrors the parts of `torch.cuda` that make sense for the
device, under the same names. Where a `torch.cuda` function takes an optional
`device` argument, so does its `torch.mojo` counterpart: an index, a string
such as `"mojo:1"`, or a `torch.device`, defaulting to the current device.

### Device functions

| `torch.cuda` | `torch.mps` | `torch.mojo` | Notes |
|---|---|---|---|
| `is_available()` | `is_available()` | `is_available()` | at least one mojo GPU |
| `is_initialized()` | | `is_initialized()` | `True` once registered |
| `device_count()` | `device_count()` | `device_count()` | |
| `current_device()`, `set_device(i)`, `device(i)` | | `current_device()`, `set_device(i)`, `device(i)` | |
| `synchronize()` | `synchronize()` | `synchronize()` | |
| `get_device_name()` | | `get_device_name()` | |
| `get_device_properties()` | | `get_device_properties()` | see [Device properties](#device-properties) |
| `get_device_capability()` | | `get_device_capability()` | `(None, None)` on AMD and Apple GPUs |
| `is_bf16_supported()` | | `is_bf16_supported()` | |

### Random-number functions

| `torch.cuda` | `torch.mps` | `torch.mojo` |
|---|---|---|
| `manual_seed(s)`, `manual_seed_all(s)` | `manual_seed(s)` | `manual_seed(s)`, `manual_seed_all(s)` |
| `seed()`, `seed_all()`, `initial_seed()` | `seed()` | `seed()`, `seed_all()`, `initial_seed()` |
| `get_rng_state()`, `set_rng_state(t)` | `get_rng_state()`, `set_rng_state(t)` | `get_rng_state()`, `set_rng_state(t)` |
| `get_rng_state_all()`, `set_rng_state_all(ts)` | | `get_rng_state_all()`, `set_rng_state_all(ts)` |

### Memory functions

| `torch.cuda` | `torch.mps` | `torch.mojo` |
|---|---|---|
| `memory_allocated()`, `max_memory_allocated()` | `current_allocated_memory()` | `memory_allocated()`, `max_memory_allocated()` |
| `memory_reserved()`, `max_memory_reserved()` | | `memory_reserved()`, `max_memory_reserved()` (equal to allocated) |
| `memory_stats()`, `memory_stats_as_nested_dict()`, `memory_summary()` | | the same three |
| `reset_peak_memory_stats()`, `reset_accumulated_memory_stats()` | | the same two |
| `empty_cache()` | `empty_cache()` | `empty_cache()` (does not release device memory) |
| `mem_get_info()` | | `mem_get_info()` (MAX's numbers) |

[Memory](#memory) explains where their meaning differs from CUDA.

### Stream and event functions

`torch.mojo` has `current_stream()`, `default_stream()`, `set_stream(s)`,
`stream(s)` (a context manager), `Stream` and `Event`, plus
`stream_native_handle(s)` for code that launches work on a mojo stream
outside PyTorch. [Streams and events](streams_and_events.md) describes them.

### Not available

The memory manager underneath the device is MAX's, and it does not expose
the information these functions need, so they raise `NotImplementedError`
with a reason:

- `memory_snapshot()`, `_record_memory_history()`, `_dump_snapshot()` (the
  memory visualizer's inputs)
- `caching_allocator_alloc()`, `caching_allocator_delete()`
- `set_per_process_memory_fraction()`, `list_gpu_processes()`
- `host_memory_stats()` and the other pinned-host-memory statistics

These have no `torch.mojo` counterpart at all:

- CUDA graphs: `torch.cuda.graph`, `CUDAGraph`, `make_graphed_callables`.
- `torch.cuda.nvtx`, `torch.cuda.jiterator`, `torch.mps.compile_shader`, and
  the hardware monitors (`utilization`, `temperature`, `power_draw`,
  `clock_rate`).
- `torch.amp.GradScaler("mojo")`: see [Mixed precision](#mixed-precision).

`torch.backends.cudnn` settings such as `benchmark` configure cuDNN and do
nothing on the mojo device. TF32 is controlled by
`torch.set_float32_matmul_precision()`; see [Mixed precision](#mixed-precision).

!!! note "`torch.cuda.is_available()` says nothing about the mojo device"
    On a CPU-only torch wheel (the recommended install, see
    [Installation](index.md#installation)), `torch.cuda.is_available()` is
    `False` even while the mojo device runs on your NVIDIA GPU. On a CUDA
    wheel it is `True`, but it refers to the separate `cuda` device. Code that
    picks its device with `"cuda" if torch.cuda.is_available() else "cpu"`
    therefore never lands on `mojo`. Replace it with `torch.accelerator`, as
    shown [above](#device-generic-code-with-torchaccelerator). The same goes
    for `torch.backends.mps.is_available()`.

!!! note "Out-of-memory errors"
    A failed allocation raises a plain `RuntimeError` whose message contains
    `out of memory`, not `torch.OutOfMemoryError`. Since
    `torch.OutOfMemoryError` is itself a subclass of `RuntimeError`, catching
    `RuntimeError` works on both CUDA and mojo.

## Random numbers

`torch.manual_seed()` seeds every mojo device along with the CPU.
`torch.Generator(device="mojo")`, `torch.random.fork_rng(device_type="mojo")`
and the RNG state functions behave like their CUDA counterparts:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

torch.manual_seed(0)                    # seeds the CPU and every mojo device
print(torch.rand(4, device="mojo").cpu())
# tensor([0.3990, 0.5167, 0.0249, 0.9401]), the same values as on CUDA

g = torch.Generator(device="mojo").manual_seed(42)
noise = torch.randn(3, device="mojo", generator=g)

state = torch.mojo.get_rng_state()      # 16-byte uint8 tensor, like CUDA's
first = torch.rand(3, device="mojo")
torch.mojo.set_rng_state(state)
assert torch.equal(first.cpu(), torch.rand(3, device="mojo").cpu())

with torch.random.fork_rng(device_type="mojo"):
    torch.rand(3, device="mojo")        # does not advance the outer stream
```

The generator is the Philox generator CUDA uses, with the same state layout.
On NVIDIA GPUs, a given seed produces the same `rand`, `randn`, `randint`,
`bernoulli` and dropout values as on a CUDA device of the same model, so a
seeded experiment can be compared across the two. On AMD and Apple GPUs the
integer draws match CUDA's bit for bit, while floating-point draws can differ
in the last bits.

## Mixed precision

Autocast works with the device type `"mojo"`, with `torch.bfloat16` or
`torch.float16`, and applies CUDA's per-op policy: matmuls, linear layers and
attention run in the lower precision, while softmax, log-softmax, sums, layer
norm and losses produce `float32`.

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

model = torch.nn.Linear(16, 4).to("mojo")
x = torch.randn(8, 16, device="mojo")
target = torch.randint(0, 4, (8,), device="mojo")

with torch.autocast("mojo", dtype=torch.bfloat16):
    logits = model(x)                                       # bfloat16
    loss = torch.nn.functional.cross_entropy(logits, target)  # float32
loss.backward()
print(logits.dtype, loss.dtype, model.weight.grad.dtype)
# torch.bfloat16 torch.float32 torch.float32
```

In device-generic code, write `torch.autocast(device.type, dtype=...)`.
Without a `dtype`, `torch.autocast("mojo")` uses `torch.float16`, the same
default as CUDA.

!!! warning "No `GradScaler` yet"
    `torch.amp.GradScaler("mojo")` does not work: ops it relies on are not
    implemented on the mojo device, and `scaler.step()` raises
    `NotImplementedError`. Train with `dtype=torch.bfloat16`, which has the
    range of `float32` and needs no loss scaling.

`torch.set_float32_matmul_precision("high")` (or `"medium"`) lets float32
matmuls use TF32 tensor cores where the mojo device has a TF32 kernel, which
is currently on H100-class GPUs (compute capability 9.0). Setting
`torch.backends.cuda.matmul.allow_tf32 = True` sets that precision, so it has
the same effect. On other GPUs, and with the default `"highest"`, float32
matmuls stay in full float32.

## Memory

The memory functions of `torch.accelerator` and `torch.mojo` read the same
counters, with `torch.cuda`'s names and key spellings:

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

torch.accelerator.reset_peak_memory_stats()
x = torch.empty(64, 1024, 1024, device="mojo")         # 256 MiB of float32
print(torch.accelerator.memory_allocated() // 2**20)      # 256
del x
print(torch.accelerator.memory_allocated() // 2**20)      # 0
print(torch.accelerator.max_memory_allocated() // 2**20)  # 256

free, total = torch.accelerator.get_memory_info()         # MAX's view of the GPU
print(f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
print(torch.mojo.memory_summary())
```

What the numbers mean, and where they differ from CUDA:

- Allocated memory is the live tensor storage this process holds on each
  device. Views and slices share their base's storage and count once.
- Reserved memory equals allocated memory. On CUDA, `memory_reserved()` also
  includes the memory PyTorch's caching allocator keeps for reuse. The mojo
  device gets its memory from MAX's own memory manager, which does not report
  what it keeps in reserve, so `memory_reserved() - memory_allocated()` is
  always 0. For the same reason, the pool, segment and split breakdowns of
  `memory_stats()` are 0, and everything is under the `all` keys.
- `reset_peak_memory_stats()` resets the peak to the current usage, not to 0.
- `empty_cache()` does not return device memory to the system, because MAX
  offers no way to do that. It only frees pinned host buffers left over from
  finished copies, and it is safe to call at any time.
- `get_memory_info()` and `torch.mojo.mem_get_info()` return the
  `(free, total)` bytes that MAX reports. These can describe MAX's memory
  budget rather than the physical memory `nvidia-smi` shows, and they do not
  follow your allocations one for one. To measure your own usage, use
  `memory_allocated()`.

## Device properties

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

props = torch.mojo.get_device_properties(0)
print(props.name, props.api, props.arch_name)
print(props.total_memory // 2**20, "MiB,", props.multi_processor_count, "SMs/CUs")
print(torch.mojo.get_device_capability(0))   # (8, 9) on an sm_89 GPU
```

`get_device_properties()` returns a frozen dataclass. Its fields follow
`torch.cuda`'s names where they exist (`name`, `total_memory`,
`multi_processor_count`, `warp_size`, `major`, `minor`,
`max_threads_per_multi_processor`, `regs_per_multiprocessor`, and
`gcnArchName` on AMD), plus a few extras: `api` (`"cuda"`, `"hip"` or
`"metal"`), `arch_name` (for example `"sm_90a"` or `"gfx942"`), block and
shared-memory limits, and `clock_rate` in kHz. A field the GPU's driver does
not report is `None`; `major` and `minor` are only set on NVIDIA GPUs.

## Profiling

`torch.profiler` records ops on mojo tensors like any others. Add the
`PrivateUse1` activity, the generic name PyTorch gives out-of-tree devices
such as `mojo`:

```python
import torch
from torch.profiler import ProfilerActivity, profile

import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

a = torch.randn(1024, 1024, device="mojo")
torch.relu(a @ a)                        # compile the kernels before profiling
torch.accelerator.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
    for _ in range(5):
        b = torch.relu(a @ a)
    torch.accelerator.synchronize()

print(prof.key_averages().table(row_limit=8))
prof.export_chrome_trace("trace.json")   # open in Perfetto or chrome://tracing
```

The CPU-side timeline (ops, their inputs, the trace export) works with any
torch build. Rows for the GPU kernels themselves, with their device times,
appear only with a CUDA build of torch on NVIDIA, because that build's
profiler sees the kernels the mojo device launches. Warm up before
profiling, since an op's first call includes its kernel compilation.

With a CPU-only torch build, the older profiler gives a device time per op
("Self MOJO" columns):

```python
import torch
import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()

a = torch.randn(1024, 1024, device="mojo")
torch.relu(a @ a)                        # compile the kernels before profiling
torch.accelerator.synchronize()

with torch.autograd.profiler.profile(use_device="mojo") as prof:
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
    `torch.save` of a tensor on the mojo device, and `torch.load(...,
    map_location="mojo")`, currently fail with `NotImplementedError`, because
    a storage op PyTorch uses for them is not implemented yet. Move tensors
    to the CPU to save them, and load on the CPU:

    ```python
    import torch
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()

    model = torch.nn.Linear(4, 2).to("mojo")

    # Save: copy the tensors to the CPU first.
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "ckpt.pt")

    # Load: read on the CPU; load_state_dict copies into the mojo parameters.
    restored = torch.nn.Linear(4, 2).to("mojo")
    restored.load_state_dict(torch.load("ckpt.pt", map_location="cpu"))
    print(restored.weight.device)            # mojo:0
    ```

## Processes and `fork`

The mojo device has CUDA's restriction on `fork`: a process forked after the
device was initialized cannot use it, and any op there raises a
`RuntimeError` that points to the `spawn` start method. `DataLoader` workers
that only produce CPU tensors are unaffected and run normally, with
`pin_memory=True` and a `.to(device, non_blocking=True)` in the main process.
Keep device work out of the dataset and in the main process.

## Hardware

The `mojo` device runs on the GPUs Mojo supports: NVIDIA, AMD and Apple GPUs.
Most of the testing so far was on H100, MI300X and Apple M4 (see the
[home page](index.md)). The per-vendor differences that matter for this
page:

=== "NVIDIA"

    - `get_device_capability()` returns the CUDA compute capability, and
      `get_device_properties().arch_name` the target (`"sm_90a"`, ...).
    - Seeded random numbers match CUDA's bit for bit.
    - `CUDA_VISIBLE_DEVICES` selects the visible GPUs.
    - A CPU-only torch wheel is enough. With a CUDA wheel, `cuda` and `mojo`
      coexist as separate devices.

=== "AMD"

    - Install the CPU wheel of torch, as the [home page](index.md) explains.
    - `get_device_capability()` returns `(None, None)`; the architecture is
      in `get_device_properties().gcnArchName` and `.arch_name` (`"gfx942"`, ...).
    - Integer random draws match CUDA's; floating-point ones can differ in
      the last bits.

=== "Apple"

    - `float64` is not available on Apple GPUs, and neither are the ops that
      need it.
    - `torch.Stream(device="mojo")` is always the default stream (as on MPS),
      and events cannot be recorded; see
      [Apple GPUs](streams_and_events.md#apple-gpus).
    - Host-to-device copies are synchronous, and the profilers report no
      device times.
    - With Xcode 26 or newer, the Metal compiler is a separate download:
      `xcodebuild -downloadComponent MetalToolchain`. Without it, kernel
      builds fail with `Metal Compiler failed to compile metallib`.

A process uses one vendor only: the first of NVIDIA, AMD and Apple that has
a GPU. On a machine with both an NVIDIA and an AMD card, only the NVIDIA ones
become mojo devices.

Op coverage is still partial. An op the device does not implement raises
`NotImplementedError` with the op's name, and never falls back to the CPU
silently.

## `torch.compile`

`torch.compile(model, backend=mojo_backend)` is documented on the
[home page](index.md). Its examples use `cuda` tensors, and compiling with
tensors on the `mojo` device is listed there as not supported yet. This page
covers eager execution on the `mojo` device only.
