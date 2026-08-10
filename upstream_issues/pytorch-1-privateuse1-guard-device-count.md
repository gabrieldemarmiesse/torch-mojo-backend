# The Python-backend PrivateUse1 DeviceGuard hardcodes `deviceCount() == 1`, so backward on `privateuseone:i` (i >= 1) hits an autograd INTERNAL ASSERT

## 🐛 Describe the bug

`_setup_privateuseone_for_python_backend` registers a C++
`PythonDeviceGuard` in which everything except `type()` is hardcoded
(`torch/csrc/acc/Module.cpp:61-117` at v2.11.0):

```cpp
c10::Device getDevice() const override { return c10::Device(type(), 0); }
void setDevice(c10::Device) const override {}                 // no-op
c10::DeviceIndex deviceCount() const noexcept override { return 1; }
// getStream/getNewStream/exchangeStream: always (DEFAULT, index 0)
```

The autograd engine sizes its per-device ready queues from
`deviceCount()` (`Engine::start_device_threads`, `engine.cpp:1576-1603`)
and asserts the node's device index against that size:

```cpp
// engine.cpp:1548 (Engine::ready_queue)
TORCH_INTERNAL_ASSERT(
    0 <= device.index() &&
    device.index() < static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
```

So a Python-only PrivateUse1 backend that manages **multiple real devices**
and gives its tensors truthful TensorImpl device indices crashes any
backward touching `privateuseone:i` with `i >= 1`:

```
RuntimeError: 0 <= device.index() && device.index() < ... INTERNAL ASSERT FAILED
at "torch/csrc/autograd/engine.cpp":1548, please report a bug to PyTorch.
```

Truthful indices are not optional for multi-device work: DDP's C++ Reducer
allocates its gradient buckets from `params[0].options()`, i.e. from the
TensorImpl device — with a phantom index 0, every rank's buckets land on
device 0.

### Reproduction sketch (any multi-device Python backend)

```python
import torch
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend
# backend_module with device_count() -> 2, plus PrivateUse1 aten impls that
# route by tensor device index (as in the docs' python-backend example)
_setup_privateuseone_for_python_backend("myacc", backend_module=mod)

w = torch.randn(4, requires_grad=True, device="myacc:1")
(w * w).sum().backward()   # INTERNAL ASSERT at engine.cpp:1548
```

### Current workaround

`torch.autograd.set_multithreading_enabled(False)` — `Engine::ready_queue`
then returns the CPU queue unconditionally (`engine.cpp:1541-1546`) and
backward runs on the calling thread. This works (it is what
`torch/testing/_internal/distributed/multi_threaded_pg.py` does), but it is
thread-local, so every user thread that may run backward must apply it, and
a miss produces the INTERNAL ASSERT above ("please report a bug to
PyTorch") rather than an actionable error.

## Proposed fix

Let `PythonDeviceGuard` consult the registered backend module:

- `deviceCount()` → call the backend module's `device_count()` (cached; the
  module protocol already requires it — `_DummyBackendModule.device_count`
  exists precisely for this);
- `setDevice`/`exchangeDevice` → forward to the module's
  `set_device`/`current_device` (or keep as no-ops, which is safe once
  deviceCount is right — the engine guards `set_device` calls with
  `device < impl->deviceCount()`).

With a truthful `deviceCount()`, `start_device_threads` creates one worker
per device and multi-device Python backends get real per-device autograd
queues — no thread-local workaround, and single-graph multi-device backward
parallelizes like CUDA's.

Alternative, much smaller ask: turn the `ready_queue` INTERNAL ASSERT into a
`TORCH_CHECK` with a message pointing at the deviceCount limitation for
Python-registered backends.

## Versions

torch 2.11.0 (repro'd on an 8-GPU PrivateUse1 backend:
github.com/gabrieldemarmiesse/torch-mojo-backend, which runs thread-per-rank
DDP with the thread-local workaround today).
