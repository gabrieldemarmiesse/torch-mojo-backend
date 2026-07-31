"""Kernel-call queue and Into-protocol tests.

The suite runs with queueing disabled (synchronous contracts); these tests
opt back in per-test via TMB_FORCE_KERNEL_QUEUE and exercise the queue's
own guarantees: FIFO ordering across cold and warm units, drain-on-read,
keep-alive of buffers referenced only by queued launches, launch-time
dtype escalation, and value parity of every Into family against the
synchronous path.
"""

import gc
import threading

import pytest
import torch

from torch_mojo_backend import eager_kernels, get_accelerators, register_mojo_devices
from torch_mojo_backend.eager_kernels import call_queue


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool):
    if not mojo_gpu_available:
        pytest.skip("mojo GPU not available")
    register_mojo_devices()
    return "mojo:0"


@pytest.fixture
def forced_queue(monkeypatch):
    """Enable kernel-call queueing for one test; leave the queue empty."""
    monkeypatch.setenv("TMB_FORCE_KERNEL_QUEUE", "1")
    assert call_queue.enabled()
    yield
    call_queue.drain()
    assert not call_queue.active()


class _StalledBuild:
    """Make one loaded unit look like it is still compiling, releasable on
    demand — a deterministic stand-in for a slow `mojo build`."""

    def __init__(self, module: str, op: str) -> None:
        unit = eager_kernels._STATES[module].unit(op)
        unit.load_blocking()  # make sure the real extension exists
        self._unit = unit
        self._ext = unit.ext
        unit.ext = None
        unit.request_async = self._request  # type: ignore[method-assign]
        self._job = _FakeJob(self)

    def _request(self, all_dtypes: bool = False) -> "_FakeJob":
        return self._job

    def release(self) -> None:
        unit = self._unit
        if unit.ext is None:
            unit.ext = self._ext
            del unit.request_async  # restore the class method
            self._job.done.set()


class _FakeJob:
    def __init__(self, stall: _StalledBuild) -> None:
        self.done = threading.Event()
        self.error: BaseException | None = None
        self._stall = stall

    def wait(self) -> None:
        # A blocking drain releases the "build" (as a finished compile would).
        self._stall.release()


def test_warm_calls_queue_behind_a_cold_unit(mojo_gpu, forced_queue):
    """FIFO: once one launch waits on a build, later launches (warm or not)
    hold their position behind it, and a host read drains in order."""
    a = torch.full((64,), 3.0, device=mojo_gpu)
    b = torch.full((64,), 4.0, device=mojo_gpu)
    stall = _StalledBuild("logic_ops", "AddSpec")
    try:
        y = a + b  # Into launch on the stalled unit: must queue
        z = y * y  # warm MulSpec: must queue BEHIND the stalled add
        assert call_queue.active()
        assert len(call_queue._QUEUE) == 2
    finally:
        stall.release()
    torch.testing.assert_close(z.cpu(), torch.full((64,), 49.0))
    assert not call_queue.active()


def test_keepalive_survives_dropped_intermediates(mojo_gpu, forced_queue):
    """Queued launches hold raw pointers; dropping every Python reference
    to an intermediate before the drain must not recycle its buffer."""
    a = torch.full((256,), 2.0, device=mojo_gpu)
    stall = _StalledBuild("logic_ops", "AddSpec")
    try:
        y = a + a
        z = y * 3.0
        del y
        gc.collect()
        # Churn the allocator while the queue still holds y's pointer.
        junk = [torch.empty(256, device=mojo_gpu) for _ in range(8)]
        del junk
        gc.collect()
    finally:
        stall.release()
    torch.testing.assert_close(z.cpu(), torch.full((256,), 12.0))


def test_sync_read_drains_the_queue(mojo_gpu, forced_queue):
    a = torch.full((8,), 5.0, device=mojo_gpu)
    stall = _StalledBuild("logic_ops", "AddSpec")
    try:
        y = a + a
        assert call_queue.active()
    finally:
        stall.release()
    assert y.sum().item() == 80.0
    assert not call_queue.active()


def test_launch_time_dtype_escalation(mojo_gpu, forced_queue):
    """A queued launch hitting a gated-out dtype rebuilds its unit with the
    full dtype set and retries — no error surfaces to the caller."""
    unit = eager_kernels._STATES["logic_ops"].unit("AddSpec")
    unit.load_blocking()
    original = unit.dtypes
    if original is None:
        pytest.skip("unit already built with every dtype")
    assert "float64" not in original  # default set has no f64
    a = torch.full((16,), 1.5, dtype=torch.float64, device=mojo_gpu)
    y = a + a  # Into launch; f64 is outside the unit's dtype gate
    torch.testing.assert_close(y.cpu(), torch.full((16,), 3.0, dtype=torch.float64))
    assert unit.dtypes is None  # escalated to the full set


@pytest.mark.parametrize(
    "case",
    [
        "binary",
        "binary_broadcast",
        "comparison",
        "mixed_dtype",
        "scalar",
        "unary",
        "reduce",
        "min_dim",
        "matmul",
        "bmm",
    ],
)
def test_into_family_parity(mojo_gpu, forced_queue, case):
    """Every Into family computes the same values as CPU eager."""
    g = torch.Generator().manual_seed(20260725)
    a = torch.randn(6, 33, generator=g)
    b = torch.randn(6, 33, generator=g)
    da, db = a.to(mojo_gpu), b.to(mojo_gpu)
    if case == "binary":
        out, ref = da + db, a + b
    elif case == "binary_broadcast":
        c = torch.randn(33, generator=g)
        out, ref = da * c.to(mojo_gpu), a * c
    elif case == "comparison":
        out, ref = da < db, a < b
    elif case == "mixed_dtype":
        hb = b.to(torch.bfloat16)
        out, ref = da + hb.to(mojo_gpu), a + hb
    elif case == "scalar":
        out, ref = da * 2.5 + 1.0, a * 2.5 + 1.0
    elif case == "unary":
        out, ref = da.abs().relu().exp(), a.abs().relu().exp()
    elif case == "reduce":
        out, ref = da.sum(dim=1), a.sum(dim=1)
    elif case == "min_dim":
        ov, oi = da.min(dim=1)
        rv, ri = a.min(dim=1)
        torch.testing.assert_close(ov.cpu(), rv)
        assert torch.equal(oi.cpu(), ri)
        return
    elif case == "matmul":
        w = torch.randn(33, 7, generator=g)
        out, ref = da @ w.to(mojo_gpu), a @ w
    elif case == "bmm":
        x = torch.randn(4, 5, 9, generator=g)
        y = torch.randn(4, 9, 3, generator=g)
        out, ref = torch.bmm(x.to(mojo_gpu), y.to(mojo_gpu)), torch.bmm(x, y)
    if out.dtype == torch.bool:
        assert torch.equal(out.cpu(), ref)
    else:
        torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)


def test_queued_and_synchronous_paths_match_bitwise(mojo_gpu, monkeypatch):
    """The Into form must be bitwise-identical to the legacy sync form."""
    g = torch.Generator().manual_seed(20260725)
    a = torch.randn(128, 64, generator=g)
    b = torch.randn(128, 64, generator=g)

    def compute():
        da, db = a.to(mojo_gpu), b.to(mojo_gpu)
        r = torch.log_softmax((da + db) * 0.5, dim=-1).sum(dim=-1)
        return r.cpu()

    monkeypatch.delenv("TMB_FORCE_KERNEL_QUEUE", raising=False)
    assert not call_queue.enabled()
    sync_result = compute()
    monkeypatch.setenv("TMB_FORCE_KERNEL_QUEUE", "1")
    queued_result = compute()
    call_queue.drain()
    assert torch.equal(sync_result, queued_result)


def test_queue_disabled_under_suite_by_default():
    assert len(list(get_accelerators())) >= 0  # touch the backend
    assert not call_queue.enabled()
