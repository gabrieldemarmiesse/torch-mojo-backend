"""Kernel-call queue and Into-protocol tests.

The suite runs with queueing disabled (synchronous contracts); these tests
opt back in per-test via ``TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE`` and
exercise the queue's own guarantees: FIFO ordering across cold and warm
units and across threads, drain-on-read, keep-alive of buffers referenced
only by queued launches, the deferred-error contract, and value parity of
every Into family against the synchronous path.

The queue-level tests (error contract, thread switch, reentrancy) drive the
public entry points — ``kernel_call_into``/``external_call`` then
``pump``/``drain`` — with fake units, so they run on any host, GPU or not.
``_exec`` is deliberately never called directly: what these tests are about
is what the queue does *around* a launch.
"""

import copy
import gc
import threading
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

from torch_mojo_backend import eager_kernels, get_accelerators
from torch_mojo_backend.eager_kernels import aten_fast, call_queue
from torch_mojo_backend.mojo_device import torch_mojo_device_module


@pytest.fixture(autouse=True)
def restore_queue_mode() -> Iterator[None]:
    """``enabled()`` is memoized, so the mode must be recomputed after any
    test that touched the environment — and after monkeypatch has put the
    environment back, which happens after this autouse fixture's setup and
    therefore before its teardown."""
    yield
    call_queue.refresh()


@pytest.fixture
def forced_queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable kernel-call queueing for one test; leave the queue empty."""
    monkeypatch.setenv("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE", "1")
    call_queue.refresh()
    assert call_queue.enabled()
    yield
    call_queue.drain()
    assert not call_queue.active()
    monkeypatch.delenv("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE")
    call_queue.refresh()


@pytest.fixture
def isolated_queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A private queue for the host-only tests: every piece of queue state is
    swapped for a fresh object, so nothing leaks into the process-wide queue
    (and no device is needed)."""
    monkeypatch.setattr(call_queue, "_QUEUE", deque())
    monkeypatch.setattr(call_queue, "_HELD_ERROR", [])
    monkeypatch.setattr(call_queue, "_DEVICE_THREAD", [None])
    monkeypatch.setattr(call_queue, "_QUEUE_LAUNCH_THREAD", [None])
    monkeypatch.setattr(call_queue, "_RETAINED_BYTES", [0])
    yield
    assert not call_queue._QUEUE
    assert not call_queue._HELD_ERROR
    assert call_queue._RETAINED_BYTES[0] == 0


class _StalledBuild:
    """Make one loaded unit look like it is still compiling, releasable on
    demand — a deterministic stand-in for a slow `mojo build`."""

    def __init__(self, unit: eager_kernels._DefinedUnit) -> None:
        unit.load_blocking()  # make sure the real defined extension exists
        self._unit = unit
        self._module = unit.module
        unit.module = None
        unit.request_async = self._request  # type: ignore[method-assign]
        self._job = _FakeJob(self)

    @classmethod
    def for_call(
        cls, prepared: eager_kernels.PreparedExtensionCall[object, object]
    ) -> "_StalledBuild":
        return cls(
            eager_kernels.MOJO_EXTENSION_LOADER.unit_canonical(
                prepared.extension.MOJO_FILE, prepared.defines
            )
        )

    def _request(self) -> "_FakeJob":
        return self._job

    def release(self) -> None:
        unit = self._unit
        if unit.module is None:
            unit.module = self._module
        if vars(unit).get("request_async") is self._request:
            del unit.request_async  # restore the class method


class _FakeJob:
    def __init__(self, stall: _StalledBuild) -> None:
        self._stall = stall

    def wait(self) -> None:
        # A blocking drain releases the "build" (as a finished compile would).
        self._stall.release()


class _StalledUnits:
    """Stall every specialization a region asks the loader for.

    ``_StalledBuild`` needs the exact prepared call; this needs nothing but
    the region, which is what a real cold start looks like — the model does
    not know which of its kernels are already compiled. Units must have been
    warmed beforehand, so releasing hands back an already-built module and
    the test never waits on a compiler.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, limit: int | None) -> None:
        loader = eager_kernels.MOJO_EXTENSION_LOADER
        original = loader.unit_canonical
        self.stalls: list[_StalledBuild] = []
        seen: set[int] = set()

        def unit_canonical(
            mojo_file: Path, defines: eager_kernels.CanonicalDefines
        ) -> eager_kernels._DefinedUnit:
            unit = original(mojo_file, defines)
            stallable = unit.module is not None  # a cold unit already stalls
            if stallable and id(unit) not in seen:
                if limit is None or len(self.stalls) < limit:
                    seen.add(id(unit))
                    self.stalls.append(_StalledBuild(unit))
            return unit

        monkeypatch.setattr(loader, "unit_canonical", unit_canonical)

    def release(self) -> None:
        for stall in self.stalls:
            stall.release()


def _prepare_add(
    lhs: object, rhs: object
) -> eager_kernels.PreparedExtensionCall[object, object]:
    return aten_fast._BinarySpecExtension.prepare(
        "AddSpec", lhs, rhs, aten_fast.DType.float32
    )


# ---------------------------------------------------------------------------
# Fake units for the host-only queue tests. They implement exactly the
# `call_queue._QueueUnit` protocol: `ext` is None until the build lands, and
# `request_async()` returns something with `wait()`.


class _FakeExtension:
    def __init__(self, log: list[str], error: BaseException | None = None) -> None:
        self._log = log
        self._error = error
        self.calls = 0

    def call(self, label: str) -> None:
        self.calls += 1
        self._log.append(label)
        if self._error is not None:
            raise self._error


class _FakeUnit:
    """A specialization whose build completes when the test says so."""

    def __init__(
        self,
        log: list[str],
        *,
        ready: bool = False,
        error: BaseException | None = None,
        build_error: BaseException | None = None,
    ) -> None:
        self.extension: object = _FakeExtension(log, error)
        self.ext: object | None = self.extension if ready else None
        self.build_error = build_error
        self.builds = 0

    def request_async(self) -> "_FakeUnit":
        self.builds += 1
        return self  # doubles as its own job: `wait()` completes the build

    def wait(self) -> None:
        if self.build_error is not None:
            raise self.build_error
        self.ready()

    def ready(self) -> None:
        self.ext = self.extension


class _Buffer:
    """A weakref-able stand-in for a tensor a queued item retains."""


# ---------------------------------------------------------------------------
# Queue contracts, host-only


def test_launch_error_ends_the_episode_at_the_next_drain(isolated_queue) -> None:
    """Rule 5, through the public entry points.

    A launch error is held (``pump()`` never raises), the rest of the queue
    is discarded rather than launched against buffers its producer never
    wrote, the keep-alive goes with it, and ``drain()`` raises exactly once.
    """
    log: list[str] = []
    failure = RuntimeError("exact variant rejected its arguments")
    first = _FakeUnit(log)
    failing = _FakeUnit(log, error=failure)
    successor = _FakeUnit(log)
    buffer = _Buffer()
    retained = weakref.ref(buffer)

    call_queue.kernel_call_into(first, ("first",), ())
    call_queue.kernel_call_into(failing, ("failing",), ())
    call_queue.kernel_call_into(successor, ("successor",), (buffer,))
    del buffer
    assert call_queue.active()
    assert retained() is not None  # the queued item retains its operands

    call_queue.pump()  # head still building: nothing launches, nothing raises
    assert log == []

    for unit in (first, failing, successor):
        unit.ready()
    call_queue.pump()  # launch error is held, not raised here
    assert log == ["first", "failing"]
    assert not call_queue.active()
    assert retained() is None  # abandoned items drop their references

    with pytest.raises(RuntimeError, match="exact variant rejected"):
        call_queue.drain()

    assert log == ["first", "failing"]  # the successor never ran
    assert failing.extension.calls == 1  # and the failure was not retried
    call_queue.drain()  # the episode is over: a clean, silent no-op
    assert log == ["first", "failing"]

    call_queue.kernel_call_into(_FakeUnit(log, ready=True), ("after",), ())
    assert log == ["first", "failing", "after"]


def test_build_failure_behind_a_queued_launch_surfaces_from_drain(
    isolated_queue,
) -> None:
    """The other way a deferred launch fails: its .so never builds."""
    log: list[str] = []
    failure = ImportError("mojo build failed for elementwise_ops")
    doomed = _FakeUnit(log, build_error=failure)
    successor = _FakeUnit(log)

    buffer = _Buffer()
    retained = weakref.ref(buffer)
    call_queue.kernel_call_into(doomed, ("doomed",), ())
    call_queue.kernel_call_into(successor, ("successor",), (buffer,))
    del buffer

    with pytest.raises(ImportError, match="mojo build failed"):
        call_queue.drain()

    assert log == []
    assert not call_queue.active()
    assert retained() is None  # the abandoned tail released its references


def test_queued_launch_out_of_memory_is_still_reported_as_such(isolated_queue) -> None:
    """OOM translation must survive deferral: the launch that exhausts the
    allocator now happens at drain time, far from the aten call that
    produced it, and it must still arrive as ``torch.OutOfMemoryError``."""
    assert call_queue._ERROR_TRANSLATOR[0] is not None  # installed by aten_fast
    log: list[str] = []
    oom = NotImplementedError(
        "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)"
    )
    unit = _FakeUnit(log, error=oom)

    call_queue.kernel_call_into(unit, ("oom",), ())
    unit.ready()
    call_queue.pump()

    with pytest.raises(torch.OutOfMemoryError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        call_queue.drain()


def test_thread_switch_synchronizes_once_and_keeps_fifo(
    isolated_queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4: device work is ordered only within one enqueuing thread, so a
    launch from another thread synchronizes first — and still launches the
    queue in enqueue order, not its own order."""
    syncs: list[threading.Thread] = []
    monkeypatch.setattr(
        torch_mojo_device_module,
        "_device_synchronize",
        lambda *_args, **_kwargs: syncs.append(threading.current_thread()),
    )
    log: list[str] = []

    warm = _FakeUnit(log, ready=True)
    call_queue.kernel_call_into(warm, ("warm",), ())  # runs inline on this thread
    assert log == ["warm"]
    assert syncs == []  # first device work of the episode: no switch yet

    first = _FakeUnit(log)
    second = _FakeUnit(log)
    call_queue.kernel_call_into(first, ("first",), ())
    call_queue.kernel_call_into(second, ("second",), ())
    first.ready()
    second.ready()

    failures: list[BaseException] = []

    def drain_from_another_thread() -> None:
        try:
            call_queue.drain()
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    worker = threading.Thread(target=drain_from_another_thread, name="drainer")
    worker.start()
    worker.join()

    assert failures == []
    assert log == ["warm", "first", "second"]
    assert syncs == [worker]
    assert call_queue._DEVICE_THREAD[0] is worker


def test_a_drain_reached_from_inside_a_launch_does_not_reorder(isolated_queue) -> None:
    """A launch can re-enter the queue: releasing a holder or synchronizing
    the device calls back into ``drain()``. That nested call must be a
    no-op, or the item already popped launches after its successors and
    against a keep-alive list the nested drain has cleared."""
    log: list[str] = []
    in_flight = weakref.ref(buffer := _Buffer())
    queued = weakref.ref(follower_buffer := _Buffer())

    class _ReentrantExtension:
        def call(self, label: str) -> None:
            log.append(f"{label}:enter")
            call_queue.drain()
            call_queue.pump()
            # This launch is still in flight: its item (a popped temporary on
            # the launching frame) must still retain its buffers, and the
            # nested no-op drain must not have released the follower's.
            assert in_flight() is not None
            assert queued() is not None
            log.append(f"{label}:exit")

    reentrant = _FakeUnit(log)
    reentrant.extension = _ReentrantExtension()  # type: ignore[assignment]
    follower = _FakeUnit(log)

    call_queue.kernel_call_into(reentrant, ("first",), (buffer,))
    call_queue.kernel_call_into(follower, ("second",), (follower_buffer,))
    del buffer, follower_buffer
    reentrant.ready()
    follower.ready()
    call_queue.drain()

    assert log == ["first:enter", "first:exit", "second"]
    assert not call_queue.active()
    assert in_flight() is None and queued() is None  # released at launch


def test_external_calls_hold_their_fifo_position(isolated_queue) -> None:
    """Rule 6: an always-loaded device call (tensor_holder, fa4) is
    launchable at once but must not overtake queued producers of its
    inputs, and its keep-alive is explicit."""
    log: list[str] = []
    producer = _FakeUnit(log)
    retained = weakref.ref(buffer := _Buffer())

    call_queue.kernel_call_into(producer, ("producer",), ())
    call_queue.external_call(log.append, ("consumer",), (buffer,))
    del buffer
    assert log == []
    assert retained() is not None  # the queued item retains it

    producer.ready()
    call_queue.drain()
    assert log == ["producer", "consumer"]
    assert retained() is None  # released once its launch ran


def test_external_call_runs_inline_when_nothing_is_queued(isolated_queue) -> None:
    log: list[str] = []
    call_queue.external_call(log.append, ("now",), ())
    assert log == ["now"]
    assert not call_queue.active()


def test_a_cold_launch_retains_the_output_it_writes_into(isolated_queue) -> None:
    """Rule 3, per item: the SDPA/flash-attention autograd nodes run above
    __torch_dispatch__ and drop their intermediates as soon as they queue a
    launch against them — the queued item itself must keep them alive."""
    log: list[str] = []
    retained = weakref.ref(out := _Buffer())

    call_queue.kernel_call_into(_FakeUnit(log), ("cold",), (out,))
    del out

    assert retained() is not None  # only the queued item holds it now
    call_queue._QUEUE[0][0].ready()
    call_queue.drain()
    assert log == ["cold"]
    assert retained() is None  # released right after its launch


def test_failed_allocation_drains_synchronizes_and_retries_once(
    isolated_queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reactive layer under the budget: a device-OOM allocation drains
    the queue (releasing what its items retain), synchronizes the device so
    the stream-ordered frees land, and retries exactly once. Non-OOM errors
    and a second failure propagate untouched."""
    from types import SimpleNamespace

    from torch_mojo_backend.mojo_device import torch_mojo_tensor

    synced: list[bool] = []
    device = SimpleNamespace(
        default_stream=SimpleNamespace(synchronize=lambda: synced.append(True))
    )
    monkeypatch.setattr(torch_mojo_tensor, "_ctx_ptr", lambda _device: 7)

    oom = Exception("CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)")

    class _FlakyHolder:
        def __init__(self, failures: list[BaseException]) -> None:
            self.failures = failures
            self.calls = 0

        def alloc(self, ctx_ptr: int, nbytes: int) -> tuple[object, int]:
            self.calls += 1
            if self.failures:
                raise self.failures.pop(0)
            return (object(), 0x1000)

    log: list[str] = []

    # OOM once -> drain (launches the queued item), sync, retry succeeds.
    holder = _FlakyHolder([oom])
    monkeypatch.setattr(torch_mojo_tensor, "_holder_mod", lambda: holder)
    call_queue.kernel_call_into(_FakeUnit(log), ("pending",), ())
    call_queue._QUEUE[0][0].ready()
    result = torch_mojo_tensor._alloc_with_recovery(device, 4096)
    assert result[1] == 0x1000
    assert holder.calls == 2
    assert log == ["pending"]  # the drain launched what the queue held
    assert synced == [True]
    assert not call_queue.active()

    # A non-OOM failure propagates immediately: no drain, no sync, no retry.
    synced.clear()
    holder = _FlakyHolder([ValueError("not a memory problem")])
    monkeypatch.setattr(torch_mojo_tensor, "_holder_mod", lambda: holder)
    with pytest.raises(ValueError, match="not a memory problem"):
        torch_mojo_tensor._alloc_with_recovery(device, 4096)
    assert holder.calls == 1
    assert synced == []

    # Two OOMs: one recovery attempt, then the second failure surfaces.
    holder = _FlakyHolder(
        [oom, Exception("CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)")]
    )
    monkeypatch.setattr(torch_mojo_tensor, "_holder_mod", lambda: holder)
    with pytest.raises(Exception, match="OUT_OF_MEMORY"):
        torch_mojo_tensor._alloc_with_recovery(device, 4096)
    assert holder.calls == 2
    assert synced == [True]


def test_budget_is_computed_from_free_device_memory(
    isolated_queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the env override, the bound adapts to the device: half the
    smallest free-VRAM figure across the accelerators, floored at 1 GiB,
    falling back to 8 GiB when nothing can report memory statistics."""
    import torch_mojo_backend

    class _FakeAccelerator:
        def __init__(self, free: int) -> None:
            self.stats = {"free_memory": free, "total_memory": free}

    gib = 1024 * 1024 * 1024
    monkeypatch.delenv("TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB", raising=False)

    monkeypatch.setattr(
        torch_mojo_backend,
        "get_accelerators",
        lambda: [_FakeAccelerator(60 * gib), _FakeAccelerator(20 * gib)],
    )
    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [None])
    assert call_queue._budget_bytes() == 10 * gib  # half the smallest

    monkeypatch.setattr(
        torch_mojo_backend, "get_accelerators", lambda: [_FakeAccelerator(gib)]
    )
    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [None])
    assert call_queue._budget_bytes() == gib  # floored at 1 GiB

    def _no_accelerators() -> list[object]:
        raise RuntimeError("no driver on this host")

    monkeypatch.setattr(torch_mojo_backend, "get_accelerators", _no_accelerators)
    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [None])
    assert call_queue._budget_bytes() == 8192 * 1024 * 1024  # fallback

    monkeypatch.setenv("TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB", "256")
    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [None])
    assert call_queue._budget_bytes() == 256 * 1024 * 1024  # override wins
    call_queue.refresh()


def test_retention_budget_bounds_cold_run_ahead(
    isolated_queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 3's bound: a read-free cold storm may not retain unbounded
    bytes. The enqueue that crosses the budget waits the builds out and
    launches everything, releasing the retention — the fix for the 70+ GB
    warm-up abort. Under budget, nothing drains and nothing blocks."""

    class _Payload:
        _numel = 512
        _itemsize = 4  # 2 KiB per queued item

    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [3 * 2048])

    log: list[str] = []
    units = [_FakeUnit(log) for _ in range(4)]
    retained = []
    for index, unit in enumerate(units):
        buf = _Payload()
        retained.append(weakref.ref(buf))
        call_queue.kernel_call_into(unit, (f"cold{index}",), (buf,))
        del buf
        if index < 3:
            # At or under budget: still queued, still retained, not blocked.
            assert call_queue.active()
            assert retained[index]() is not None

    # The fourth enqueue pushed retention to 8 KiB > 6 KiB: the budget drain
    # waited out every build and launched the whole queue in FIFO order.
    assert log == ["cold0", "cold1", "cold2", "cold3"]
    assert not call_queue.active()
    assert call_queue._RETAINED_BYTES[0] == 0
    assert all(ref() is None for ref in retained)


def test_retention_budget_holds_launch_errors_for_the_next_drain(
    isolated_queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget drain serves memory, not a value: a launch failure inside
    it is held and raised by the next real drain(), per rule 5."""

    class _Payload:
        _numel = 1024
        _itemsize = 1

    monkeypatch.setattr(call_queue, "_BUDGET_BYTES", [1024])

    log: list[str] = []
    failure = RuntimeError("exact variant rejected its arguments")
    failing = _FakeUnit(log, error=failure)
    call_queue.kernel_call_into(failing, ("failing",), (_Payload(),))
    call_queue.kernel_call_into(_FakeUnit(log), ("successor",), (_Payload(),))

    assert log == ["failing"]  # the budget drain launched and failed it
    assert not call_queue.active()  # rule 5 dropped the successor
    assert call_queue._RETAINED_BYTES[0] == 0
    with pytest.raises(RuntimeError, match="exact variant rejected"):
        call_queue.drain()


@pytest.mark.parametrize(
    ("knob", "value", "expected"),
    [
        ("TORCH_MOJO_BACKEND_KERNEL_QUEUE", "0", False),  # kill switch
        ("TORCH_MOJO_BACKEND_KERNEL_QUEUE", "1", True),  # force on, suite included
        ("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE", "1", True),
    ],
)
def test_mode_knobs_are_value_tested_not_truthiness_tested(
    monkeypatch: pytest.MonkeyPatch, knob: str, value: str, expected: bool
) -> None:
    """A knob set to "0" must never mean "on" (the deleted
    ``TMB_NO_TRIGGER_DEFER`` did exactly that), and the answer is memoized,
    so it only changes when ``refresh()`` says so."""
    call_queue.refresh()
    assert not call_queue.enabled()  # the suite default, now memoized
    monkeypatch.setenv(knob, value)
    assert not call_queue.enabled()  # unchanged: the environment is read once
    call_queue.refresh()
    assert call_queue.enabled() is expected


def test_queue_disabled_under_suite_by_default() -> None:
    assert len(list(get_accelerators())) >= 0  # touch the backend
    assert not call_queue.enabled()


# ---------------------------------------------------------------------------
# End-to-end, on a real device


def test_warm_calls_queue_behind_a_cold_unit(mojo_gpu, forced_queue):
    """FIFO: once one launch waits on a build, later launches (warm or not)
    hold their position behind it, and a host read drains in order."""
    a = torch.full((64,), 3.0, device=mojo_gpu)
    b = torch.full((64,), 4.0, device=mojo_gpu)
    call_queue.drain()
    stall = _StalledBuild.for_call(_prepare_add(a, b))
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
    stall = _StalledBuild.for_call(_prepare_add(a, a))
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


# ---------------------------------------------------------------------------
# Convolution's im2col/GEMM keep-alive (agent-C review of PR #387): the
# materialized im2col buffer -- or, for a 1x1 stride-1 conv, the input
# tensor read in place -- is never seen outside `fast_aten_convolution`.
# Nothing in these tests holds a reference to it either; its only owner is
# the stack frame that allocates (or aliases) it, which is gone by the time
# a queued GEMM launch that reads it actually runs. That is exactly the
# shape of hazard queue rule 3 exists for: a raw pointer's owning tensor
# must be in the SAME item's keepalive as the pointer, not just alive
# somewhere in the caller at enqueue time.
#
# `_StalledUnits` forces every specialization the second (post-warm-up) call
# touches to look cold, so the producer (im2col) and the GEMM both queue
# instead of launching directly. Releasing only the producer's stall and
# calling `pump()` (not `drain()`) launches it ALONE and stops with the GEMM
# item still queued -- the one moment queue rule 3 says a missing keepalive
# entry drops the buffer's last reference.
#
# The assertion is a `weakref` check, not a value comparison: whether the
# buffer's Python object is still alive right there is a direct read of
# queue rule 3, and it does not depend on whatever the device allocator
# happens to do with freed memory (which may not corrupt a value-based
# check promptly, or at all, on a given allocator/run -- a weakref going
# dead is unambiguous either way).
# ---------------------------------------------------------------------------


def _spy_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[object, ...]]]:
    """Record every ``(op, keepalive)`` pair `_call_mojo` is invoked with,
    without changing its behavior."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    orig_call_mojo = aten_fast._call_mojo

    def spy(
        extension: object,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[object, ...],
        output_dtypes: tuple[object, ...] = (),
        flags: object = None,
        keepalive: tuple[object, ...],
    ) -> object:
        calls.append((op, keepalive))
        return orig_call_mojo(
            extension,
            op,
            extension_args,
            arg_dtypes=arg_dtypes,
            output_dtypes=output_dtypes,
            flags=flags,
            keepalive=keepalive,
        )

    monkeypatch.setattr(aten_fast, "_call_mojo", spy)
    return calls


def _assert_materialized_col_survives_to_gemm(
    run: Callable[[], torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> torch.Tensor:
    """Warm up ``run`` (a zero-arg callable issuing one materialized conv2d
    -- i.e. NOT the 1x1 fast path, which has no separate im2col call), then
    replay it with every specialization forced cold. Launches the im2col
    item alone (dropping ITS OWN keepalive, `(col, a)`), then checks -- with
    the GEMM item still queued, not yet launched -- that the im2col buffer
    is still alive. It can only be alive there because the GEMM item's own
    keepalive retains it: nothing else does."""
    calls = _spy_keepalive(monkeypatch)
    warm = run()
    call_queue.drain()
    del warm
    calls.clear()

    stall = _StalledUnits(monkeypatch, limit=None)
    col_survived = None
    try:
        y = run()
        assert call_queue.active()
        assert len(stall.stalls) >= 2, "expected im2col AND a GEMM unit to stall"
        assert calls and calls[0][0] == "Im2col", (
            f"expected the first call to be Im2col, got {calls[0][0] if calls else None!r}"
        )
        col = calls[0][1][0]  # keepalive=(col, a): index 0 is `col`
        weak_col = weakref.ref(col)
        del col, calls[:]  # this frame must not become col's last owner

        gemm_stall = stall.stalls[-1]
        for earlier in stall.stalls[:-1]:
            earlier.release()
        call_queue.pump()  # launches im2col only; the GEMM item stays queued
        gc.collect()
        col_survived = weak_col() is not None
        gemm_stall.release()
    finally:
        stall.release()
    call_queue.drain()
    assert col_survived, (
        "the im2col buffer was garbage-collected after im2col's own queued "
        "item launched (and dropped its keepalive, rule 3) but BEFORE the "
        "GEMM item that reads it had launched -- the GEMM item's keepalive "
        "does not retain its own B operand, so a real cold start can free "
        "this buffer out from under the GEMM kernel (call_queue.py:18-26)"
    )
    return y


@pytest.mark.parametrize("case", ["dense_simt_bmm", "grouped_simt_matmul"])
def test_convolution_gemm_keepalive_survives_a_cold_queued_launch(
    mojo_gpu, forced_queue, monkeypatch: pytest.MonkeyPatch, case: str
):
    """Reproduces the review's use-after-free hazard for the plain SIMT
    `Bmm` fallback (materialized conv, dense) and the grouped `Matmul` loop
    (materialized conv, groups > 1, one queue item per (sample, group), all
    reading the same im2col buffer).

    float32 with matmul precision forced to "highest" (TF32 off) keeps this
    off the sm_90a-gated Bmm16/TF32 routes regardless of which GPU runs the
    suite, so it deterministically exercises the SIMT fallbacks these cases
    target -- patched on `aten_fast.torch` rather than the global, so it
    cannot leak into another test.
    """
    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    torch.manual_seed(0)
    if case == "grouped_simt_matmul":
        n, c, out_c, groups = 2, 8, 8, 2
    else:
        n, c, out_c, groups = 2, 6, 5, 1
    k, pad = 3, 1
    x = torch.randn(n, c, 9, 9, device=mojo_gpu)
    weight = torch.randn(out_c, c // groups, k, k, device=mojo_gpu) * 0.1
    x_cpu, weight_cpu = x.cpu(), weight.cpu()

    def run() -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x, weight, None, stride=1, padding=pad, groups=groups
        )

    y = _assert_materialized_col_survives_to_gemm(run, monkeypatch)

    ref = torch.nn.functional.conv2d(
        x_cpu, weight_cpu, None, stride=1, padding=pad, groups=groups
    )
    torch.testing.assert_close(y.cpu(), ref, atol=1e-4, rtol=1e-4)


def test_convolution_gemm16_keepalive_survives_a_cold_queued_launch(
    mojo_gpu, forced_queue, monkeypatch: pytest.MonkeyPatch
):
    """Same hazard as `test_convolution_gemm_keepalive_survives_a_cold_
    queued_launch`, for the sm_90a Bmm16 route `_try_gemm16_conv_bmm`
    builds (bf16, the default fast path production actually takes on this
    hardware) -- skipped off an sm_90a GPU, where it is not reachable."""
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the Bmm16 conv route requires an sm_90a GPU")
    torch.manual_seed(0)
    # C = 8 < 32 declines the implicit-GEMM route (`_try_conv_igemm`), so
    # the materialized im2col + Bmm16 route -- the one this fix touches --
    # is what actually launches.
    n, c, out_c, k, pad = 2, 8, 16, 3, 1
    x = (torch.randn(n, c, 9, 9, device=mojo_gpu)).to(torch.bfloat16)
    weight = (torch.randn(out_c, c, k, k, device=mojo_gpu) * 0.1).to(torch.bfloat16)
    x_cpu, weight_cpu = x.float().cpu(), weight.float().cpu()

    def run() -> torch.Tensor:
        return torch.nn.functional.conv2d(x, weight, None, stride=1, padding=pad)

    y = _assert_materialized_col_survives_to_gemm(run, monkeypatch)

    ref = torch.nn.functional.conv2d(x_cpu, weight_cpu, None, stride=1, padding=pad)
    torch.testing.assert_close(y.float().cpu(), ref, atol=8e-2, rtol=8e-2)


def test_convolution_1x1_alias_keepalive_survives_a_cold_queued_launch(
    mojo_gpu, forced_queue, monkeypatch: pytest.MonkeyPatch
):
    """The 1x1 stride-1 fast path never allocates a separate im2col buffer:
    `col` in `fast_aten_convolution` IS `a`, the input tensor, read in
    place. Once this test drops its own reference, the GEMM item's own
    keepalive is the ONLY thing that can keep the input alive until that
    item -- forced cold, still queued -- launches. Same TF32-off float32
    setup as the materialized cases, for the same portability reason."""
    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    torch.manual_seed(0)
    n, c, out_c = 2, 6, 5
    x = torch.randn(n, c, 9, 9, device=mojo_gpu)
    weight = torch.randn(out_c, c, 1, 1, device=mojo_gpu) * 0.1
    x_cpu, weight_cpu = x.cpu(), weight.cpu()

    def run() -> torch.Tensor:
        return torch.nn.functional.conv2d(x, weight, None, stride=1, padding=0)

    warm = run()
    call_queue.drain()
    del warm

    stall = _StalledUnits(monkeypatch, limit=None)
    x_survived = None
    try:
        y = run()
        assert call_queue.active()
        assert stall.stalls, "expected the GEMM unit to stall cold"
        weak_x = weakref.ref(x)
        x = None  # drop the caller's only reference to the input
        gc.collect()
        x_survived = weak_x() is not None
    finally:
        stall.release()
    call_queue.drain()
    assert x_survived, (
        "the input tensor was garbage-collected while the queued GEMM item "
        "that reads it in place (the 1x1 fast path, no separate im2col "
        "buffer) was still cold -- its keepalive does not retain the input "
        "(queue rule 3, call_queue.py:18-26)"
    )

    ref = torch.nn.functional.conv2d(x_cpu, weight_cpu, None, stride=1, padding=0)
    torch.testing.assert_close(y.cpu(), ref, atol=1e-4, rtol=1e-4)


def test_sync_read_drains_the_queue(mojo_gpu, forced_queue):
    a = torch.full((8,), 5.0, device=mojo_gpu)
    call_queue.drain()
    stall = _StalledBuild.for_call(_prepare_add(a, a))
    try:
        y = a + a
        assert call_queue.active()
    finally:
        stall.release()
    assert y.sum().item() == 80.0
    assert not call_queue.active()


@pytest.mark.parametrize("read", ["item", "cpu", "nonzero"])
def test_host_reads_drain_where_they_touch_bytes(mojo_gpu, forced_queue, read):
    """Rule 2. The dispatcher no longer keeps a list of "syncing" op names:
    the drain lives next to each host read, so every route that looks at
    device bytes must still land the queue first."""
    a = torch.arange(8, dtype=torch.float32, device=mojo_gpu)
    call_queue.drain()
    stall = _StalledBuild.for_call(_prepare_add(a, a))
    try:
        y = a + a
        assert call_queue.active()
        if read == "item":
            value = y[3].item()
            assert value == 6.0
        elif read == "cpu":
            torch.testing.assert_close(y.cpu(), torch.arange(8.0) * 2)
        else:
            assert torch.equal(y.nonzero().cpu(), torch.arange(1, 8).unsqueeze(1))
        assert not call_queue.active()
    finally:
        stall.release()


def test_queue_orders_across_threads(mojo_gpu, forced_queue):
    """The autograd engine is another thread; so is a dataloader doing
    `.to("mojo")`. Work enqueued on one thread and drained from another must
    still produce program-order results."""
    a = torch.full((128,), 3.0, device=mojo_gpu)
    call_queue.drain()
    stall = _StalledBuild.for_call(_prepare_add(a, a))
    results: list[torch.Tensor] = []
    failures: list[BaseException] = []

    def double_on_another_thread(y: torch.Tensor) -> None:
        try:
            results.append((y * 2.0).cpu())
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    try:
        y = a + a  # queued behind the stalled build, on this thread
        assert call_queue.active()
        worker = threading.Thread(target=double_on_another_thread, args=(y,))
        worker.start()
        worker.join()
    finally:
        stall.release()

    assert failures == []
    torch.testing.assert_close(results[0], torch.full((128,), 12.0))


def test_strided_materialization_keeps_its_place(mojo_gpu, forced_queue):
    """Materializing a strided view launches an always-loaded device call
    (tensor_holder's strided copy, via `external_call`) rather than a gated
    specialization — it is launchable immediately and must still queue
    behind the producer of the bytes it reads."""
    a = torch.arange(12, dtype=torch.float32, device=mojo_gpu).reshape(3, 4)
    call_queue.drain()
    stall = _StalledBuild.for_call(_prepare_add(a, a))
    try:
        y = a + a
        z = y.t().contiguous()
        assert len(call_queue._QUEUE) >= 2
    finally:
        stall.release()
    torch.testing.assert_close(
        z.cpu(), (torch.arange(12.0).reshape(3, 4) * 2).t().contiguous()
    )


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
        # Families whose routes never ran queued before: normalization,
        # activation, softmax, loss, conv, pooling, embedding, attention,
        # and the foreach/optimizer ops.
        "layer_norm",
        "layer_norm_no_affine",
        "gelu",
        "softmax",
        "cross_entropy",
        "conv2d",
        "max_pool2d",
        "embedding",
        "sdpa",
        "foreach",
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
    elif case == "layer_norm":
        w = torch.randn(33, generator=g)
        bias = torch.randn(33, generator=g)
        out = torch.nn.functional.layer_norm(
            da, (33,), w.to(mojo_gpu), bias.to(mojo_gpu)
        )
        ref = torch.nn.functional.layer_norm(a, (33,), w, bias)
    elif case == "layer_norm_no_affine":
        out = torch.nn.functional.layer_norm(da, (33,))
        ref = torch.nn.functional.layer_norm(a, (33,))
    elif case == "gelu":
        out, ref = torch.nn.functional.gelu(da), torch.nn.functional.gelu(a)
    elif case == "softmax":
        out, ref = torch.softmax(da, dim=-1), torch.softmax(a, dim=-1)
    elif case == "cross_entropy":
        target = torch.randint(0, 33, (6,), generator=g)
        out = torch.nn.functional.cross_entropy(da, target.to(mojo_gpu))
        ref = torch.nn.functional.cross_entropy(a, target)
    elif case == "conv2d":
        x = torch.randn(1, 3, 16, 16, generator=g)
        w = torch.randn(4, 3, 3, 3, generator=g)
        bias = torch.randn(4, generator=g)
        out = torch.nn.functional.conv2d(
            x.to(mojo_gpu), w.to(mojo_gpu), bias.to(mojo_gpu), padding=1
        )
        ref = torch.nn.functional.conv2d(x, w, bias, padding=1)
    elif case == "max_pool2d":
        x = torch.randn(1, 3, 16, 16, generator=g)
        out = torch.nn.functional.max_pool2d(x.to(mojo_gpu), 2)
        ref = torch.nn.functional.max_pool2d(x, 2)
    elif case == "embedding":
        weight = torch.randn(11, 9, generator=g)
        index = torch.randint(0, 11, (4, 5), generator=g)
        out = torch.nn.functional.embedding(index.to(mojo_gpu), weight.to(mojo_gpu))
        ref = torch.nn.functional.embedding(index, weight)
    elif case == "sdpa":
        q = torch.randn(2, 3, 5, 16, generator=g)
        k = torch.randn(2, 3, 7, 16, generator=g)
        v = torch.randn(2, 3, 7, 16, generator=g)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.to(mojo_gpu), k.to(mojo_gpu), v.to(mojo_gpu)
        )
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elif case == "foreach":
        values = [torch.randn(17, generator=g), torch.randn(4, 9, generator=g)]
        device_values = [value.to(mojo_gpu) for value in values]
        torch._foreach_mul_(device_values, 0.5)
        torch._foreach_mul_(values, 0.5)
        for actual, expected in zip(device_values, values, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)
        return
    if out.dtype == torch.bool:
        assert torch.equal(out.cpu(), ref)
    else:
        torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)


def _tiny_model() -> torch.nn.Sequential:
    """Small enough to be fast, wide enough to mix families: two GEMMs, a
    normalization with statistics, and an activation, each with a backward."""
    torch.manual_seed(20260731)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.LayerNorm(16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    )


def _step(
    model: torch.nn.Module, inputs: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    loss = (model(inputs) - target).pow(2).mean()
    loss.backward()
    return loss


@pytest.mark.parametrize("stall", ["none", "first", "every"])
def test_tiny_module_forward_backward_matches_cpu(
    mojo_gpu, forced_queue, monkeypatch: pytest.MonkeyPatch, stall: str
):
    """The headline feature, end to end: a real module's forward AND
    backward with the queue on and builds still pending.

    This is the one test that covers FIFO under a mixed op sequence,
    keep-alive of autograd-saved intermediates, the main <-> autograd-thread
    switch, and drain-on-read at the same time. Every parameter gradient is
    compared against the CPU reference, not just the output.
    """
    generator = torch.Generator().manual_seed(20260731)
    inputs = torch.randn(8, 16, generator=generator)
    target = torch.randn(8, 4, generator=generator)
    cpu_model = _tiny_model()
    device_model = copy.deepcopy(cpu_model).to(mojo_gpu)
    device_inputs = inputs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)

    expected_loss = _step(cpu_model, inputs, target)

    # Warm every specialization first: a stalled unit then releases into an
    # already-built module, so the test is deterministic and never waits on
    # a compiler subprocess.
    _step(device_model, device_inputs, device_target)
    device_model.zero_grad(set_to_none=True)
    call_queue.drain()

    stalled = None
    if stall != "none":
        stalled = _StalledUnits(monkeypatch, limit=1 if stall == "first" else None)
    try:
        loss = _step(device_model, device_inputs, device_target)
        if stalled is not None:
            assert stalled.stalls, "no specialization was stalled"
    finally:
        if stalled is not None:
            stalled.release()

    torch.testing.assert_close(loss.cpu(), expected_loss.detach(), atol=1e-4, rtol=1e-3)
    for expected, actual in zip(
        cpu_model.parameters(), device_model.parameters(), strict=True
    ):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu(), expected.grad, atol=1e-4, rtol=1e-3
        )


def test_queued_sdpa_backward_matches_cpu(mojo_gpu, forced_queue, monkeypatch):
    """The SDPA autograd nodes read payloads ABOVE __torch_dispatch__, so
    their intermediates have no op bracket to keep them alive. With a
    stalled build and a churned allocator, a dropped intermediate must not
    be recycled under a queued launch."""
    generator = torch.Generator().manual_seed(20260731)
    shape = (2, 3, 8, 16)
    q, k, v = (torch.randn(shape, generator=generator) for _ in range(3))
    grad = torch.randn(shape, generator=generator)

    def run(device: str) -> tuple[torch.Tensor, ...]:
        # detach().clone() first: `.to("cpu")` of a CPU tensor is the tensor
        # itself, and marking the shared original as a leaf that requires
        # grad would make the device copy a non-leaf.
        tensors = [t.detach().clone().to(device).requires_grad_() for t in (q, k, v)]
        out = torch.nn.functional.scaled_dot_product_attention(*tensors)
        out.backward(grad.to(device))
        return (out.detach(), *(tensor.grad for tensor in tensors))

    expected = run("cpu")
    run(mojo_gpu)  # warm every specialization the stalled region will use
    call_queue.drain()

    stalled = _StalledUnits(monkeypatch, limit=None)
    try:
        actual = run(mojo_gpu)
        junk = [torch.empty(4096, device=mojo_gpu) for _ in range(8)]
        del junk
        gc.collect()
    finally:
        stalled.release()

    for got, want in zip(actual, expected, strict=True):
        assert got is not None
        torch.testing.assert_close(got.cpu(), want, atol=2e-2, rtol=2e-2)


def test_queued_and_synchronous_paths_match_bitwise(mojo_gpu, monkeypatch):
    """The Into form must be bitwise-identical to the legacy sync form."""
    g = torch.Generator().manual_seed(20260725)
    a = torch.randn(128, 64, generator=g)
    b = torch.randn(128, 64, generator=g)

    def compute() -> torch.Tensor:
        da, db = a.to(mojo_gpu), b.to(mojo_gpu)
        r = torch.log_softmax((da + db) * 0.5, dim=-1).sum(dim=-1)
        return r.cpu()

    monkeypatch.delenv("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE", raising=False)
    call_queue.refresh()
    assert not call_queue.enabled()
    sync_result = compute()
    monkeypatch.setenv("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE", "1")
    call_queue.refresh()
    queued_result = compute()
    call_queue.drain()
    assert torch.equal(sync_result, queued_result)
