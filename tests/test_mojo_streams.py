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
`agents_docs/streams.md`). There is no more `MojoStream`/`mojo_device.streams`
Python surface, no per-name `device_streams.get_stream` cache, and no
`record_use` to call directly: `tensor.record_stream(stream)` is a public
torch API now, handled by the C++ shim.
"""

import pytest
import torch
from torch.utils._mode_utils import no_dispatch

from tests.native.conftest import side_stream_or_skip, skip_if_metal
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
    stream = side_stream_or_skip(mojo_gpu)
    assert isinstance(stream, torch.Stream)
    assert stream.device == torch.device("mojo", 0)
    assert stream.device_index == 0
    assert stream.native_handle != 0  # ty: ignore[unresolved-attribute] -- torch's Stream stub lacks native_handle
    # stream_id indexes the device's MAX context views (see agents_docs/streams.md),
    # independently of the underlying native CUstream/hipStream_t.
    assert stream.stream_id != stream.native_handle  # ty: ignore[unresolved-attribute]
    assert stream.stream_id != torch.accelerator.current_stream().stream_id
    assert stream == stream
    assert stream != torch.accelerator.current_stream()


def test_current_stream_and_context_manager(mojo_gpu: str):
    default = torch.accelerator.current_stream()
    assert device_module.current_stream() == default
    side = side_stream_or_skip(mojo_gpu)
    with side:
        assert torch.accelerator.current_stream() == side
        assert device_module.current_stream() == side
    assert torch.accelerator.current_stream() == default
    device_module.set_stream(side)
    assert device_module.current_stream() == side
    device_module.set_stream(default)
    assert device_module.current_stream() == default


@pytest.mark.parametrize("priority", [0, -1, 1])
def test_metal_streams_share_the_default_stream(mojo_gpu: str, priority: int):
    if device_module.get_device_properties(mojo_gpu).api != "metal":
        pytest.skip("Apple GPU stream semantics")
    default = device_module.default_stream(mojo_gpu)
    first = torch.Stream(device="mojo", priority=priority)
    second = device_module.Stream(device=mojo_gpu, priority=priority)
    assert type(first) is torch.Stream
    assert first == second == default
    assert first.stream_id == second.stream_id == 0
    assert first.device == torch.device(mojo_gpu)
    x = torch.ones(32, device=mojo_gpu)
    with first:
        assert torch.accelerator.current_stream() == default
        with second:
            y = x + 2
            y.record_stream(second)
        assert device_module.current_stream() == default
    first.synchronize()
    assert second.query()
    torch.testing.assert_close(y.cpu(), torch.full((32,), 3.0))
    assert device_module.current_stream() == default


@pytest.mark.parametrize("shape", [(1024,), (257, 129)])
def test_metal_independent_stream_workflows_read_back_without_explicit_sync(
    mojo_gpu: str, shape: tuple[int, ...]
):
    """Both workflows finish before CPU readback without user-inserted fences.

    Separately constructed Metal streams share the default queue, so CPU
    readback outside their contexts is ordered after both producers. This
    would require explicit ordering on a backend with independent queues.
    """
    if device_module.get_device_properties(mojo_gpu).api != "metal":
        pytest.skip("Apple GPU stream semantics")
    first = torch.Stream(device=mojo_gpu)
    second = torch.Stream(device=mojo_gpu)
    # Repeat so first-use kernel compilation cannot hide a readback race.
    for iteration in range(2):
        with first:
            x = torch.full(shape, 2.0 + iteration, device=mojo_gpu)
            first_result = (x * x + 3.0) * 0.5
        with second:
            y = torch.full(shape, -3.0 - iteration, device=mojo_gpu)
            shifted = y * 2.0 - 1.0
            second_result = shifted * shifted

        # No synchronize(), query(), events or stream waits: blocking .cpu()
        # copies on the default queue are the only readback barriers.
        first_cpu = first_result.cpu()
        second_cpu = second_result.cpu()
        torch.testing.assert_close(
            first_cpu, torch.full(shape, ((2.0 + iteration) ** 2 + 3.0) * 0.5)
        )
        torch.testing.assert_close(
            second_cpu, torch.full(shape, ((-3.0 - iteration) * 2.0 - 1.0) ** 2)
        )


@pytest.mark.parametrize("enable_timing", [False, True])
def test_metal_events_raise_on_record(mojo_gpu: str, enable_timing: bool):
    if device_module.get_device_properties(mojo_gpu).api != "metal":
        pytest.skip("Apple GPU event semantics")
    stream = device_module.default_stream(mojo_gpu)
    event = torch.Event(device=mojo_gpu, enable_timing=enable_timing)
    # Generic torch events allocate lazily; a failed recording must leave
    # the object safe to retry, query, synchronize and destroy.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="events are not supported on Apple GPU"):
            event.record(stream)
    assert event.query()
    event.synchronize()
    with pytest.raises(RuntimeError, match="events are not supported on Apple GPU"):
        stream.record_event()
    with pytest.raises(RuntimeError, match="events are not supported on Apple GPU"):
        stream.wait_stream(stream)
    stream.synchronize()


def test_documented_device_agnostic_pattern(mojo_gpu: str):
    side_stream_or_skip(mojo_gpu)
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

    side = side_stream_or_skip(mojo_gpu)
    side.wait_stream(torch.accelerator.current_stream())
    event = side.record_event()
    event.synchronize()
    assert event.query()
    torch.testing.assert_close(y.cpu(), torch.full((2048, 2048), 4.0))


def test_event_semantics(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "Apple GPU events are tested as unsupported separately")
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
    to introspect (that bookkeeping now lives in tmb/backend/device.mojo), so
    this is a stress/correctness test rather than a state-inspection one.
    """
    n = 1 << 20
    reused_any = False
    for _ in range(20):
        source = torch.full((n,), 1.0, device=mojo_gpu)
        side = side_stream_or_skip(mojo_gpu)
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
    side = side_stream_or_skip(mojo_gpu)
    with no_dispatch():
        tensor.record_stream(side)
    assert tensor.cpu().sum().item() == 64.0
