"""torch.Stream / torch.Event on the native mojo device.

The old eager backend needed a Python-level patch of `torch.Stream`/
`torch.Event` (`monkeypatching.py`'s `_install_torch_stream_event_dispatch`,
`mojo_device/streams.py`'s `Stream`/`Event` subclasses, `device_streams.py`'s
named side streams and `record_use` fencing) because a Python-only
PrivateUse1 backend gets only the stub C++ device guard: every stream it
minted was id 0 and every wait/record a no-op. `register_mojo_devices()` no
longer calls that patch at all -- the native backend registers a real C++
`PrivateUse1HooksInterface` (`native/csrc/shim_runtime.cpp`), so
`torch.Stream(device="mojo")` is a genuine, unsubclassed `torch.Stream`
backed by a real MAX stream, through the ordinary generic path (see
`docs/streams.md`). There is no more `MojoStream`/`mojo_device.streams`
Python surface, no per-name `device_streams.get_stream` cache, and no
`record_use` to call directly: `tensor.record_stream(stream)` is a public
torch API now, handled by the C++ shim.
"""

import pytest
import torch
from torch.utils._mode_utils import no_dispatch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.native import device_module

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def test_dispatch_installed_and_cpu_delegation():
    """`torch.mojo` is the native device module; a cpu stream is untouched."""
    assert torch.mojo is device_module  # ty: ignore[unresolved-attribute]
    cpu_stream = torch.Stream(device="cpu")
    assert isinstance(cpu_stream, torch.Stream)
    assert cpu_stream.device.type != "mojo"


def test_stream_construction_and_identity(mojo_gpu: str):
    stream = torch.Stream(device=mojo_gpu)
    assert isinstance(stream, torch.Stream)
    assert stream.device == torch.device("mojo", 0)
    assert stream.device_index == 0
    assert stream.native_handle != 0  # ty: ignore[unresolved-attribute] -- torch's Stream stub lacks native_handle
    # stream_id is the MAX DeviceContext pointer (see docs/streams.md), a
    # different value from the underlying native CUstream/hipStream_t.
    assert stream.stream_id != stream.native_handle  # ty: ignore[unresolved-attribute]
    assert stream.stream_id != torch.accelerator.current_stream().stream_id
    assert stream == stream
    assert stream != torch.accelerator.current_stream()


def test_current_stream_and_context_manager(mojo_gpu: str):
    default = torch.accelerator.current_stream()
    assert device_module.current_stream() == default
    side = torch.Stream(device=mojo_gpu)
    with side:
        assert torch.accelerator.current_stream() == side
        assert device_module.current_stream() == side
    assert torch.accelerator.current_stream() == default
    device_module.set_stream(side)
    assert device_module.current_stream() == side
    device_module.set_stream(default)
    assert device_module.current_stream() == default


def test_documented_device_agnostic_pattern(mojo_gpu: str):
    stream = torch.Stream(device=torch.accelerator.current_accelerator())
    current = torch.accelerator.current_stream()
    stream.wait_stream(current)
    with stream:
        assert torch.accelerator.current_stream() == stream
    event = stream.record_event()
    assert isinstance(event, torch.Event)
    current.wait_event(event)
    stream.synchronize()
    assert stream.query()


def test_wait_stream_orders_real_work(mojo_gpu: str):
    x = torch.full((2048, 2048), 2.0, device=mojo_gpu)
    y = x * x

    side = torch.Stream(device=mojo_gpu)
    side.wait_stream(torch.accelerator.current_stream())
    event = side.record_event()
    event.synchronize()
    assert event.query()
    torch.testing.assert_close(y.cpu(), torch.full((2048, 2048), 4.0))


def test_event_semantics(mojo_gpu: str):
    unrecorded = torch.Event(device=mojo_gpu)
    assert unrecorded.query() is True
    unrecorded.synchronize()  # no-op by contract
    unrecorded.wait(torch.accelerator.current_stream())  # no-op by contract too

    start = torch.Event(device=mojo_gpu, enable_timing=True)
    end = torch.Event(device=mojo_gpu, enable_timing=True)
    stream = torch.accelerator.current_stream()
    torch.accelerator.synchronize()
    start.record(stream)
    x = torch.full((1024, 1024), 3.0, device=mojo_gpu)
    (x * x).cpu()  # forces the work through the queue and the device
    end.record(stream)
    end.synchronize()
    assert start.query() and end.query()
    assert start.elapsed_time(end) >= 0.0
    assert end.device == torch.device("mojo", 0)

    untimed = torch.Event(device=mojo_gpu)
    untimed.record(stream)
    with pytest.raises(ValueError, match="enable_timing"):
        untimed.elapsed_time(end)


def test_stream_is_a_real_torch_stream_for_cpp_argument_parsing(mojo_gpu: str):
    """THPStream_Check requires the concrete torch._C.Stream type, not duck
    typing -- automatic now that a mojo stream is a plain torch.Stream, not a
    Python subclass."""
    stream = torch.accelerator.current_stream()
    assert isinstance(stream, torch._C.Stream)


def test_record_stream_prevents_pool_reuse_corruption(mojo_gpu: str):
    """The regression this guards: MAX's buffer free is only ordered on its
    *owning* stream (see the "MAX buffer free not cross-stream fenced" memory
    note); a tensor a side stream is still reading must not have its backing
    memory handed to a new allocation on the default stream. `record_stream`
    is the public contract that prevents it. There is no more `_holder._events`
    to introspect (that bookkeeping now lives in native/mojo/device.mojo), so
    this is a stress/correctness test rather than a state-inspection one.
    """
    n = 1 << 20
    reused_any = False
    for _ in range(20):
        source = torch.full((n,), 1.0, device=mojo_gpu)
        side = torch.Stream(device=mojo_gpu)
        side.wait_stream(torch.accelerator.current_stream())
        with side:
            sink = source * source
        source.record_stream(side)
        source_ptr = source.data_ptr()
        del source  # free is fenced behind `side`'s read of it
        overwriter = torch.full((n,), 3.0, device=mojo_gpu)
        reused_any = reused_any or overwriter.data_ptr() == source_ptr
        side.synchronize()
        torch.accelerator.synchronize()
        assert (sink.cpu() != 1.0).sum().item() == 0
        del overwriter, sink
    assert reused_any, "allocator never reused the freed block; test inconclusive"


def test_record_stream_accepts_the_current_stream_as_a_no_op(mojo_gpu: str):
    """Recording the tensor's own (owning) stream needs no fence."""
    tensor = torch.full((1024,), 1.0, device=mojo_gpu)
    tensor.record_stream(torch.accelerator.current_stream())
    assert tensor.cpu().sum().item() == 1024.0


def test_record_stream_under_no_dispatch(mojo_gpu: str):
    """record_stream must work under no_dispatch(), as torch.distributed
    calls it (e.g. from the c10d reducer / process group bucket views)."""
    tensor = torch.ones(64, device=mojo_gpu)
    side = torch.Stream(device=mojo_gpu)
    with no_dispatch():
        tensor.record_stream(side)
    assert tensor.cpu().sum().item() == 64.0
