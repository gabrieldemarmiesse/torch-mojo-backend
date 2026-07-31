"""Kernel-call queue and Into-protocol tests.

The suite runs with queueing disabled (synchronous contracts); these tests
opt back in per-test via TMB_FORCE_KERNEL_QUEUE and exercise the queue's
own guarantees: FIFO ordering across cold and warm units, drain-on-read,
keep-alive of buffers referenced only by queued launches, launch-time
error propagation, and value parity of every Into family against the
synchronous path.
"""

import gc
import threading

import pytest
import torch

from torch_mojo_backend import eager_kernels, get_accelerators, register_mojo_devices
from torch_mojo_backend.eager_kernels import aten_fast, call_queue


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

    def __init__(
        self, prepared: eager_kernels.PreparedExtensionCall[object, object]
    ) -> None:
        unit = eager_kernels.MOJO_EXTENSION_LOADER.unit_canonical(
            prepared.extension.MOJO_FILE, prepared.defines
        )
        unit.load_blocking()  # make sure the real defined extension exists
        self._unit = unit
        self._module = unit.module
        unit.module = None
        unit.request_async = self._request  # type: ignore[method-assign]
        self._job = _FakeJob(self)

    def _request(self) -> "_FakeJob":
        return self._job

    def release(self) -> None:
        unit = self._unit
        if unit.module is None:
            unit.module = self._module
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


def _prepare_add(
    lhs: object, rhs: object
) -> eager_kernels.PreparedExtensionCall[object, object]:
    return aten_fast._BinarySpecExtension.prepare(
        "AddSpec", lhs, rhs, aten_fast.DType.float32
    )


def test_warm_calls_queue_behind_a_cold_unit(mojo_gpu, forced_queue):
    """FIFO: once one launch waits on a build, later launches (warm or not)
    hold their position behind it, and a host read drains in order."""
    a = torch.full((64,), 3.0, device=mojo_gpu)
    b = torch.full((64,), 4.0, device=mojo_gpu)
    call_queue.drain()
    stall = _StalledBuild(_prepare_add(a, b))
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
    call_queue.drain()
    stall = _StalledBuild(_prepare_add(a, a))
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
    call_queue.drain()
    stall = _StalledBuild(_prepare_add(a, a))
    try:
        y = a + a
        assert call_queue.active()
    finally:
        stall.release()
    assert y.sum().item() == 80.0
    assert not call_queue.active()


def test_launch_error_is_not_retried() -> None:
    """An exact variant is called once; launch errors surface unchanged."""

    class FailingExtension:
        def __init__(self) -> None:
            self.calls = 0

        def call(self) -> None:
            self.calls += 1
            raise RuntimeError("exact variant rejected its arguments")

    class LoadedUnit:
        def __init__(self) -> None:
            self.ext = FailingExtension()

        def resolve(self, attr: str) -> object:
            assert attr == "call"
            return self.ext.call

    unit = LoadedUnit()
    with pytest.raises(RuntimeError, match="exact variant rejected"):
        call_queue._exec((unit, "call", (), {}))
    assert unit.ext.calls == 1


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
        "permute_copy",
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
    elif case == "permute_copy":
        out, ref = da.t().contiguous(), a.t().contiguous()
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
