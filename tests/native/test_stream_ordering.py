"""Cross-stream ordering and the device-wide host barrier, single process.

Every case runs a producer long enough (256 MiB of float32, eight in-place
adds) that the host is many milliseconds ahead of the GPU when it reaches the
consumer, so an ordering hole shows up as a wrong value rather than as a
lucky pass. `torch.accelerator.synchronize()` is the fence before each
readback: the readback itself is issued on the default stream, and reading a
side stream's result from another stream needs a fence exactly as it does on
CUDA.
"""

import pytest
import torch

from torch_mojo_backend.native import device_module

pytestmark = pytest.mark.xdist_group(name="group1")

N = 64 * 1024 * 1024  # 256 MiB float32
ADDS = 8


def _slow_producer(device: str, stream: torch.Stream) -> torch.Tensor:
    """A buffer the GPU is still filling when this returns: 1.0 then eight
    in-place adds on `stream`, fenced after the fill on the current stream."""
    buf = torch.full((N,), 1.0, device=device)
    stream.wait_stream(torch.accelerator.current_stream(buf.device.index))
    with device_module.stream(stream):
        for _ in range(ADDS):
            buf.add_(1.0)
    return buf


def _assert_all(tensor: torch.Tensor, value: float):
    torch.accelerator.synchronize()
    got = tensor.cpu()
    wrong = int((got != value).sum())
    assert wrong == 0, (
        f"{wrong}/{got.numel()} elements differ from {value}; "
        f"values seen: {torch.unique(got)[:6].tolist()}"
    )


def test_accelerator_synchronize_drains_side_streams(mojo_gpu):
    """`torch.accelerator.synchronize()` must be a real host barrier over
    every stream of the device.

    It used to be a silent no-op: torch's `_accelerator_synchronizeDevice`
    returns early for a lazy-init device type that torch has not marked
    initialized, and nothing marked the mojo device -- `torch.mojo` had no
    `_lazy_init` for torch's `device_lazy_init()` to call. The device guard
    was never reached, so a readback could race the stream that produced it.
    """
    side = torch.Stream(device=mojo_gpu)
    buf = _slow_producer(mojo_gpu, side)
    torch.accelerator.synchronize()
    assert side.query(), "accelerator.synchronize() returned with the side stream busy"
    assert torch.equal(buf.cpu(), torch.full((N,), 1.0 + ADDS))


def test_event_orders_a_side_stream(mojo_gpu):
    """record on the producer, wait on a consumer stream."""
    prod = torch.Stream(device=mojo_gpu)
    side = torch.Stream(device=mojo_gpu)
    buf = _slow_producer(mojo_gpu, prod)
    event = torch.Event()
    event.record(prod)
    with device_module.stream(side):
        event.wait(side)
        doubled = buf + buf
    _assert_all(doubled, 2.0 * (1.0 + ADDS))


def test_wait_stream_orders_a_side_stream(mojo_gpu):
    prod = torch.Stream(device=mojo_gpu)
    side = torch.Stream(device=mojo_gpu)
    buf = _slow_producer(mojo_gpu, prod)
    side.wait_stream(prod)
    with device_module.stream(side):
        doubled = buf + buf
    _assert_all(doubled, 2.0 * (1.0 + ADDS))


def test_device_future_wait_orders_a_side_stream(mojo_gpu):
    """The shape `MojoProcessGroup` gives a collective's `Work`: a device-typed
    `torch.futures.Future` completed while the producer stream is current, so
    the completion events land there; `wait()` on another stream must block
    that stream on them (`tests/ddp_worker.py::run_stream_ordering`)."""
    work_from_future = pytest.importorskip(
        "torch._C._distributed_c10d"
    )._create_work_from_future
    prod = torch.Stream(device=mojo_gpu)
    side = torch.Stream(device=mojo_gpu)
    buf = _slow_producer(mojo_gpu, prod)
    future: torch.futures.Future[list[torch.Tensor]] = torch.futures.Future(
        devices=[buf.device]
    )
    with device_module.stream(prod):
        future.set_result([buf])
    work = work_from_future(future)
    with device_module.stream(side):
        work.wait()
        doubled = buf + buf
    _assert_all(doubled, 2.0 * (1.0 + ADDS))
