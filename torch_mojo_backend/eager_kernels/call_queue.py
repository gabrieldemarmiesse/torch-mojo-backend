"""Kernel-call-level deferral: queue extension calls, not aten ops.

The deferral boundary sits at the extension-call layer, where a call is
nothing but raw pointers / shapes / dtypes plus a DeviceContext. All
torch-level semantics (autograd graph, version counters, views, layouts,
allocations) complete synchronously at dispatch time in Python; only the
device kernel launches are delayed while their compiled units build in the
background pool — exactly CUDA's async-stream model, applied to kernel
*compilation*.

Correctness rules:

1. **Strict FIFO.** Once any call is queued (a compile miss), every
   subsequent kernel call queues behind it, warm or not: device work must
   launch in program order. `pump()` launches the longest ready prefix.
2. **Host reads drain.** D2H transfers, scalar reads and synchronization
   `drain()` before they touch bytes.
3. **Keep-alive.** Queued items reference raw pointers, not tensors, so
   every buffer touched while the queue is non-empty is retained in
   `_KEEPALIVE` until the queue empties. `dispatch()` brackets each aten op
   and feeds it that op's args, results and `_alloc`s; `note_alloc` retains
   directly for the routes that run ABOVE `__torch_dispatch__` (the SDPA /
   flash-attention autograd nodes), which have no bracket; `external_call`
   takes an explicit `keepalive` for buffers only its raw pointers name.
4. **Thread order.** Direct launches are issued by the thread that runs
   the op and are ordered against that thread's own prior work by
   construction; direct-to-direct launches across threads need no barrier
   (that is how `main` has always interleaved the forward and autograd
   threads). The empirically unsafe pattern is a *queue* launch replaying
   another thread's work (the old deferral engine's stale-read bug), so
   `_order_queue_launch_locked()` synchronizes the device before a launch
   batch whenever the launching thread differs from an item's enqueuer or
   from the last thread to issue device work — before anything is popped,
   never with an item already in flight — and `order_direct_launch()`
   places one barrier after a cross-thread queue launch before direct
   work resumes, then goes back to a single `is None` check per call.
   Both use a device-only synchronize, not the public draining one, so
   they can never re-enter the queue.
5. **Errors end the episode.** A launch error — or a failed build behind a
   queued launch — discards every item still queued and the keep-alive that
   went with it, in the same critical section: their producers never ran, so
   launching them would consume buffers nobody wrote. The error is raised by
   the next `drain()`, exactly once, and the queue resumes empty and clean.
   Nothing is retried, redefined or rebuilt: every queued item already names
   an exact dtype/flag specialization, so a Mojo dtype error is a bridge bug
   and surfaces as-is (modulo `set_error_translator`, which only retypes it).
6. **One mutex.** `_LOCK` guards the queue *and* every device touch:
   `deferred_compile._DEVICE_LOCK is _LOCK`, so a drain from any thread
   cannot overlap a direct launch, and the re-entrant `_direct` -> queue
   ordering has no second lock to invert against.
"""

import os
import threading
from collections import deque
from collections.abc import Callable
from types import ModuleType
from typing import Protocol, runtime_checkable

from torch_mojo_backend.is_running_tests import IS_RUNNING_TESTS


@runtime_checkable
class _BuildJob(Protocol):
    """The background build of one specialization."""

    def wait(self) -> None: ...


@runtime_checkable
class _QueueUnit(Protocol):
    """One immutable `.so` specialization, as the queue sees it: `ext` is
    the loaded native module once its build lands (None while building),
    and `request_async()` starts or joins that build."""

    ext: ModuleType | None

    def request_async(self) -> _BuildJob: ...


# A queued launch: exactly one of `unit` (a specialization whose constant
# `call` entry point writes into preallocated outputs) and `fn` (an
# always-loaded device call, e.g. tensor_holder's CopyStrided or fa4) is
# set; the trailing element is the enqueuing thread (rule 4).
_QueueItem = tuple[
    _QueueUnit | None, Callable[..., object] | None, tuple, threading.Thread
]

_LOCK = threading.RLock()  # queue + every device touch (see rule 6)
_QUEUE: deque[_QueueItem] = deque()
_KEEPALIVE: list = []
_HELD_ERROR: list[BaseException] = []
_DEVICE_THREAD: list = [None]  # last thread to issue device work (rule 4)
_QUEUE_LAUNCH_THREAD: list = [None]  # last thread to launch FROM the queue
_ERROR_TRANSLATOR: list = [None]
_ENABLED: list = [None]  # memoized enabled(); refresh() invalidates
_TLS = threading.local()  # .scope, .unscoped, .in_launch


def _compute_enabled() -> bool:
    knob = os.environ.get("TORCH_MOJO_BACKEND_KERNEL_QUEUE", "")
    if knob == "0":
        return False  # kill switch: every miss blocks inline at its call site
    if knob == "1":
        return True  # force on, test suite included
    if IS_RUNNING_TESTS:
        # Most tests assert synchronous contracts; the queue tests opt back in.
        return os.environ.get("TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE", "") == "1"
    return True


def enabled() -> bool:
    """Kernel-call queueing is the default execution mode. The answer is
    fixed for the process (`IS_RUNNING_TESTS` is a constant and the knobs
    are read at startup), so it is memoized: this is called several times
    per aten op on a path budgeted in microseconds. Tests that flip the
    environment mid-process call `refresh()`."""
    cached = _ENABLED[0]
    if cached is None:
        cached = _ENABLED[0] = _compute_enabled()
    return cached


def refresh() -> None:
    """Re-read the mode environment variables on the next `enabled()`."""
    _ENABLED[0] = None


def active() -> bool:
    return bool(_QUEUE)


def set_error_translator(fn: Callable[[BaseException], None]) -> None:
    """Install the hook that retypes a device error raised by a *deferred*
    launch (aten_fast installs `_raise_if_device_oom`, so an allocator
    exhaustion still surfaces as `torch.OutOfMemoryError` even when the
    launch happened at drain time). It is called with the exception and
    either raises a better one or returns to let the original stand."""
    _ERROR_TRANSLATOR[0] = fn


def _translate(exc: BaseException) -> BaseException:
    translator = _ERROR_TRANSLATOR[0]
    if translator is None:
        return exc
    try:
        translator(exc)
    except BaseException as better:
        return better
    return exc


# ---------------------------------------------------------------------------
# Keep-alive: queued items hold raw pointers, so the tensors whose buffers
# they reference must outlive the queue. `dispatch()` brackets every aten op
# and `_alloc` reports each allocation into the current scope; the routes
# that run above `__torch_dispatch__` have no bracket and are covered by
# `note_alloc`'s unscoped path.


def op_begin() -> list | None:
    scope = getattr(_TLS, "scope", None)
    _TLS.scope = []
    return scope  # previous scope (restored at op_end; dispatch can nest)


def note_alloc(tensor: object) -> None:
    scope = getattr(_TLS, "scope", None)
    if scope is not None:
        scope.append(tensor)
        return
    # No dispatch bracket. The SDPA / flash-attention autograd nodes run on
    # the autograd thread ABOVE __torch_dispatch__, allocate intermediates,
    # queue launches against their pointers and then drop them immediately.
    with _LOCK:
        if _QUEUE:
            _KEEPALIVE.append(tensor)
            return
        # The queue is empty *now*, but the very next call may be the cold
        # one that starts an episode — and its output is allocated before it
        # is queued. Park the allocation until that decision is taken
        # (`_retain_unscoped_locked`) or until another empty-queue
        # allocation shows the episode never started.
        pending = getattr(_TLS, "unscoped", None)
        if pending is None:
            pending = _TLS.unscoped = []
        elif pending and _may_release_locked():
            pending.clear()
        pending.append(tensor)


def _retain_unscoped_locked() -> None:
    """A launch is about to be queued: adopt the un-bracketed allocations
    this thread made just before it, which its raw pointers may name."""
    pending = getattr(_TLS, "unscoped", None)
    if pending:
        _KEEPALIVE.extend(pending)
        pending.clear()


def _may_release_locked() -> bool:
    """True when dropping holders here enqueues their frees behind every
    launch that could reference them: releasing is stream-ordered on the
    current thread, so only the thread that issued the launches may do it."""
    last = _DEVICE_THREAD[0]
    return last is None or last is threading.current_thread()


def op_end(prev: list | None, args: tuple, kwargs: dict, result: object) -> None:
    scope = getattr(_TLS, "scope", None)
    _TLS.scope = prev
    if not _QUEUE:
        return
    import torch
    from torch.utils._pytree import tree_flatten

    flat, _ = tree_flatten((args, kwargs, result))
    with _LOCK:
        if scope:
            _KEEPALIVE.extend(scope)
        _KEEPALIVE.extend(t for t in flat if isinstance(t, torch.Tensor))


# ---------------------------------------------------------------------------
# Execution


def _device_only_synchronize() -> None:
    """The device barrier that never drains: safe inside the launch path."""
    from torch_mojo_backend.mojo_device import torch_mojo_device_module as _dm

    _dm._device_synchronize()


def order_direct_launch() -> None:
    """A direct (non-queued) launch is about to run on the current thread.

    Direct launches are issued by the thread that runs the op, so they are
    ordered against that thread's own prior work by construction, and pure
    direct-to-direct launches across threads need no barrier: that is
    exactly how `main` has always interleaved the forward (main) and
    backward (autograd engine) threads. The one empirically unsafe pattern
    is a *queue* launch crossing threads — the old deferral engine's
    stale-read bug — so a direct launch synchronizes only while work the
    queue launched from another thread may still be in flight. The barrier
    fully drains the device, after which the tracker clears and the steady
    state pays a single `is None` check per launch.

    Uses the device-only barrier, never the public `synchronize()`: that one
    drains the queue, and this runs *inside* the launch path.
    """
    me = threading.current_thread()
    launcher = _QUEUE_LAUNCH_THREAD[0]
    if launcher is not None and launcher is not me:
        _device_only_synchronize()
        _QUEUE_LAUNCH_THREAD[0] = None
    _DEVICE_THREAD[0] = me


def _order_queue_launch_locked() -> None:
    """The queue is about to launch, on the current thread, items other
    threads may have enqueued — the replay pattern that empirically read
    stale results without a barrier (rule 4). Synchronize once, before
    anything is popped, whenever the launcher is not the thread whose work
    the items may depend on: an item's enqueuer, the last thread to issue
    a direct launch, or the last thread the queue launched from. Runs
    under `_LOCK`, and only while the queue is non-empty (the cold path).
    """
    me = threading.current_thread()
    last_direct = _DEVICE_THREAD[0]
    last_queue = _QUEUE_LAUNCH_THREAD[0]
    if (
        (last_direct is not None and last_direct is not me)
        or (last_queue is not None and last_queue is not me)
        or any(item[3] is not me for item in _QUEUE)
    ):
        _device_only_synchronize()
    _QUEUE_LAUNCH_THREAD[0] = me
    _DEVICE_THREAD[0] = me


def _exec(item: _QueueItem) -> None:
    unit, fn, args, _enqueuer = item
    if unit is None:
        fn(*args)  # type: ignore[misc]  # exactly one of unit/fn is set
        return
    unit.ext.call(*args)  # type: ignore[union-attr]  # readiness checked by caller


def _abandon_locked(exc: BaseException) -> BaseException:
    """Rule 5: a launch failed, so the episode is over. Everything still
    queued consumes buffers its producer never wrote — drop the tail and its
    keep-alive here, under the same lock, and hand back the exception to
    raise or hold."""
    _QUEUE.clear()
    _KEEPALIVE.clear()
    return _translate(exc)


def _pump_locked() -> None:
    """Launch the longest ready prefix; stop at a head whose unit is still
    building. Held errors stay held until the next drain."""
    if _HELD_ERROR or not _QUEUE:
        return
    head = _QUEUE[0][0]
    if head is not None and head.ext is None:
        return  # nothing launchable: do not force a thread-switch synchronize
    # Once, before anything is popped: an item must never be in flight
    # across a device synchronize (it would launch after its own successors).
    _order_queue_launch_locked()
    while _QUEUE:
        unit = _QUEUE[0][0]
        if unit is not None and unit.ext is None:
            return  # still building; the tail keeps its keep-alive
        try:
            _exec(_QUEUE.popleft())
        except BaseException as exc:
            _HELD_ERROR.append(_abandon_locked(exc))
            return
    # The queue emptied through pump, and reaching here means this thread
    # both launched every item and is the current device thread, so the
    # frees released now are stream-ordered behind those launches — the same
    # state drain() clears in.
    _KEEPALIVE.clear()


def pump() -> None:
    """Launch whatever is ready. Non-blocking: never waits on a build."""
    if not _QUEUE or getattr(_TLS, "in_launch", False):
        return
    _TLS.in_launch = True
    try:
        with _LOCK:
            _pump_locked()
    finally:
        _TLS.in_launch = False


def drain() -> None:
    """Launch everything (waiting out builds); re-raise launch errors.
    Called before any host read of device values.

    Guarded on the launch path as a whole, not on drain alone: a drain
    reached from inside a launch (device synchronize, holder release) must
    be a no-op, or it would run items 2..N ahead of the one already popped.
    """
    if getattr(_TLS, "in_launch", False):
        return
    _TLS.in_launch = True
    try:
        with _LOCK:
            if _HELD_ERROR:
                # Raised exactly once; anything queued since the failure
                # belongs to the dead episode and goes with it.
                exc = _HELD_ERROR.pop()
                _QUEUE.clear()
                _KEEPALIVE.clear()
                raise exc
            if _QUEUE:
                _order_queue_launch_locked()
                while _QUEUE:
                    unit = _QUEUE[0][0]
                    try:
                        if unit is not None and unit.ext is None:
                            unit.request_async().wait()  # raises on build failure
                        _exec(_QUEUE.popleft())
                    except BaseException as exc:
                        raise _abandon_locked(exc)
            _KEEPALIVE.clear()
            # Nothing is queued any more, so no raw pointer can still name a
            # parked allocation. Releasing it here is what bounds the park:
            # `note_alloc` otherwise drops it only when the *next* unscoped
            # allocation arrives, so the most recent one per thread would
            # outlive every synchronize. With the queue disabled that park is
            # the only retention there is, and `_KEEPALIVE.clear()` alone
            # would never free it.
            pending = getattr(_TLS, "unscoped", None)
            if pending:
                pending.clear()
    finally:
        _TLS.in_launch = False


# ---------------------------------------------------------------------------
# Entry points for callers


def _launch_prefix(unit: _QueueUnit | None) -> bool:
    """Launch the ready prefix, then report whether a new call may run
    inline: nothing queued ahead of it, no held error, and — for a unit
    call — its extension already loaded. Runs under `_LOCK`."""
    if getattr(_TLS, "in_launch", False):
        return False  # inside a launch: never recurse into the queue
    _TLS.in_launch = True
    try:
        _pump_locked()
    finally:
        _TLS.in_launch = False
    if _QUEUE or _HELD_ERROR:
        return False
    if unit is not None and unit.ext is None:
        return False
    order_direct_launch()
    return True


def kernel_call_into(unit: _QueueUnit, args: tuple) -> None:
    """Queue a descriptor call whose outputs were preallocated in Python.
    The call writes into them and returns nothing — always queueable
    regardless of the *Spec naming convention."""
    with _LOCK:
        if _launch_prefix(unit):
            _exec((unit, None, args, threading.current_thread()))
            return
        if unit.ext is None:
            unit.request_async()
        _retain_unscoped_locked()
        _QUEUE.append((unit, None, args, threading.current_thread()))


def external_call(
    fn: Callable[..., object], args: tuple, keepalive: tuple = ()
) -> None:
    """An ungated device call (tensor_holder, fa4): always launchable, but
    must hold its FIFO position behind queued producers of its inputs.

    `args` are raw pointers and scalars; `keepalive` names the tensors those
    pointers belong to, retained until the queue empties (callers above
    `__torch_dispatch__` have no op bracket to retain them for us)."""
    with _LOCK:
        if _launch_prefix(None):
            fn(*args)
            return
        _retain_unscoped_locked()
        if keepalive:
            _KEEPALIVE.extend(keepalive)
        _QUEUE.append((None, fn, args, threading.current_thread()))
