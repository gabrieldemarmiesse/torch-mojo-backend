"""Kernel-call-level deferral: queue extension calls, not aten ops.

The deferral boundary sits at the extension-call layer, where a call is
nothing but raw pointers / shapes / dtypes plus a DeviceContext. All
torch-level semantics (autograd graph, version counters, views, layouts,
allocations) complete synchronously at dispatch time in Python; only the
device kernel launches are delayed while their compiled units build in the
background pool — exactly CUDA's async-stream model, applied to kernel
*compilation*.

The queue is sharded per device: each mojo device owns a `_Shard` holding
its own FIFO, mutex, thread-order trackers and run-ahead budget, so the
rank threads of single-process data parallelism never contend on one
process-wide lock and a barrier for one device never synchronizes another.
Every entry point therefore names its device: enqueues pass the
`max.driver.Device` (or index) whose DeviceContext the raw args reference,
and the dispatch layer routes each op through the lock of the device(s) it
touches.

Correctness rules (each scoped to one shard = one device):

1. **Strict FIFO.** Once any call is queued on a device (a compile miss),
   every subsequent kernel call for that device queues behind it, warm or
   not: device work must launch in program order. `pump()` launches the
   longest ready prefix.
2. **Host reads drain.** D2H transfers, scalar reads and synchronization
   `drain()` before they touch bytes. Reads that know their device drain
   that shard; `drain()` with no device drains every shard.
3. **Keep-alive.** Queued items reference raw pointers, not tensors, so
   each item carries the tensors its pointers name and retains them until
   it launches (or until rule 5 abandons it). Every enqueue names its own
   retention: the spec route threads `(out, prepared.args)` at its one
   submit choke point, and the raw routes (`_call_mojo` / `external_call`)
   pass theirs explicitly — there is no global registry, no dispatch
   bracket, and no dependence on WHERE the call came from. A launched
   item's references drop on the launching thread, under the shard lock,
   right after its launch, so the frees stay stream-ordered behind it.
   Retention is bounded per device by half that device's free VRAM at the
   moment its first launch queues (not the minimum across all devices —
   rank threads must not throttle each other on the emptiest GPU).
4. **Thread order.** Direct launches are issued by the thread that runs
   the op and are ordered against that thread's own prior work by
   construction; direct-to-direct launches across threads need no barrier
   (that is how `main` has always interleaved the forward and autograd
   threads). The empirically unsafe pattern is a *queue* launch replaying
   another thread's work (the old deferral engine's stale-read bug), so
   `_order_queue_launch_locked()` synchronizes THE SHARD'S device before a
   launch batch whenever the launching thread differs from an item's
   enqueuer or from the last thread to issue that device's work — before
   anything is popped, never with an item already in flight — and
   `order_direct_launch()` places one barrier after a cross-thread queue
   launch before direct work resumes, then goes back to a single `is
   None` check per call. Both use a device-only synchronize of that
   device, not the public draining one, so they can never re-enter the
   queue.
5. **Errors end the episode.** A launch error — or a failed build behind a
   queued launch — discards every item still queued on that shard and the
   keep-alive that went with it, in the same critical section: their
   producers never ran, so launching them would consume buffers nobody
   wrote. The error is raised by the shard's next `drain()`, exactly once,
   and the queue resumes empty and clean. Nothing is retried, redefined or
   rebuilt: every queued item already names an exact dtype/flag
   specialization, so a Mojo dtype error is a bridge bug and surfaces
   as-is (modulo `set_error_translator`, which only retypes it).
6. **One mutex per device.** The shard lock guards that device's queue
   *and* every DISPATCHED touch of that device: `deferred_compile._direct`
   routes each op through `device_lock(index)` (the same re-entrant
   object), so a drain from any thread cannot overlap a direct launch on
   the same device, and the re-entrant `_direct` -> queue ordering has no
   second lock to invert against. Distinct devices proceed concurrently by
   design. Cross-device COPIES stay inside this regime: the dispatcher
   drains both sides, then holds both locks in ascending index order.
   COLLECTIVE launches (distributed.py) are the documented exception —
   they touch every device's context under their own `_launch_lock`, with
   no shard locks. Their producers-land-first guarantee is not a lock: it
   is `deferred_compile.drain()` before the launch plus the rendezvous
   parking (every rank has finished enqueueing and is parked in
   `_Collective.join` before the last-arriving rank drains and launches),
   and their concurrent-context-touch safety rests on AsyncRT enqueues
   being empirically thread-safe — the regime the pre-queue branch always
   ran collectives under. Code that lets a rank leave the rendezvous
   before the launch, or a completion callback that enqueues compute,
   breaks the first argument and must revisit this rule.
"""

import _thread
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
# set; then the enqueuing thread (rule 4), the objects whose buffers the
# raw `args` name — retained until the item launches (rule 3) — and the
# device bytes that retention holds, pre-computed for the run-ahead budget.
_QueueItem = tuple[
    _QueueUnit | None, Callable[..., object] | None, tuple, threading.Thread, tuple, int
]

# What an entry point may name as its device: a max.driver.Device or the
# plain ordered-accelerator index.
DeviceKey = object

_ERROR_TRANSLATOR: list = [None]
_ENABLED: list = [None]  # memoized enabled(); refresh() invalidates

# Run-ahead bound. During a cold storm the host can enqueue whole training
# steps ahead of the still-compiling launches, and rule 3 retains every
# buffer those launches name — unbounded, that carved 70+ GB out of an H100
# on a read-free warm-up loop. The bound is computed per shard from ITS
# device when possible (half that device's free-VRAM figure at the moment
# its first launch queues, floored at 1 GiB); these constants are the floor
# and the fallback for devices that report no memory statistics.
_FALLBACK_BUDGET_MB = 8192
_MIN_AUTO_BUDGET_MB = 1024


def _keepalive_bytes(keepalive: object) -> int:
    """Device bytes reachable from one item's keep-alive.

    Payload wrappers all expose `_numel`/`_itemsize`; containers are walked;
    scalars, specs and everything else count zero. Runs once per *queued*
    item — the warm inline path never calls it.
    """
    numel = getattr(keepalive, "_numel", None)
    if numel is not None:
        return numel * getattr(keepalive, "_itemsize", 1)
    if isinstance(keepalive, tuple | list):
        return sum(_keepalive_bytes(entry) for entry in keepalive)
    return 0


def _translate(exc: BaseException) -> BaseException:
    translator = _ERROR_TRANSLATOR[0]
    if translator is None:
        return exc
    try:
        translator(exc)
    except BaseException as better:
        return better
    return exc


def _exec(item: _QueueItem) -> None:
    unit, fn, args, _enqueuer, _keepalive, _nbytes = item
    if unit is None:
        fn(*args)  # type: ignore[misc]  # exactly one of unit/fn is set
        return
    unit.ext.call(*args)  # type: ignore[union-attr]  # readiness checked by caller


def _free_device_memory(index: int) -> int | None:
    """The free-memory figure of ONE device (by ordered-accelerator index),
    or None when it cannot report one (CPU devices, backends without
    memory statistics)."""
    try:
        from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
            get_ordered_accelerators,
        )

        return get_ordered_accelerators()[index].stats["free_memory"]
    except Exception:
        return None


class _Shard:
    """One device's queue, mutex, thread-order trackers and budget.

    Every method whose name ends in `_locked` runs under `self.lock`; the
    public methods take it themselves. `self.tls.in_launch` is per
    (thread, shard): a thread draining device 0 may still enqueue on
    device 1, but must never re-enter shard 0's queue mid-launch.
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self.lock = threading.RLock()
        self.queue: deque[_QueueItem] = deque()
        self.held_error: list[BaseException] = []
        self.device_thread: threading.Thread | None = None  # rule 4
        self.queue_launch_thread: threading.Thread | None = None
        self.retained_bytes = 0  # rule 3 budget metering
        self.budget_bytes: int | None = None  # memoized; refresh() clears
        self.tls = threading.local()  # .in_launch

    # -- device barriers ---------------------------------------------------

    def _device_only_synchronize(self) -> None:
        """The device barrier that never drains: safe inside the launch
        path. Synchronizes THIS shard's device only."""
        from torch_mojo_backend.mojo_device import torch_mojo_device_module as _dm

        _dm._device_synchronize(self.index)

    def order_direct_launch(self) -> None:
        """A direct (non-queued) launch is about to run on the current
        thread (rule 4). Must run under `self.lock`."""
        me = threading.current_thread()
        launcher = self.queue_launch_thread
        if launcher is not None and launcher is not me:
            self._device_only_synchronize()
            self.queue_launch_thread = None
        self.device_thread = me

    def _order_queue_launch_locked(self) -> None:
        """The queue is about to launch, on the current thread, items other
        threads may have enqueued — the replay pattern that empirically
        read stale results without a barrier (rule 4). Synchronize once,
        before anything is popped, whenever the launcher is not the thread
        whose work the items may depend on: an item's enqueuer, the last
        thread to issue a direct launch, or the last thread the queue
        launched from. Runs under `self.lock`, and only while the queue is
        non-empty (the cold path)."""
        me = threading.current_thread()
        last_direct = self.device_thread
        last_queue = self.queue_launch_thread
        if (
            (last_direct is not None and last_direct is not me)
            or (last_queue is not None and last_queue is not me)
            or any(item[3] is not me for item in self.queue)
        ):
            self._device_only_synchronize()
        self.queue_launch_thread = me
        self.device_thread = me

    # -- queue mechanics ---------------------------------------------------

    def _pop_locked(self) -> _QueueItem:
        """Pop the head and release its bytes from the retention counter."""
        item = self.queue.popleft()
        self.retained_bytes -= item[5]
        return item

    def _abandon_locked(self, exc: BaseException) -> BaseException:
        """Rule 5: a launch failed, so the episode is over. Everything
        still queued consumes buffers its producer never wrote — drop the
        tail (each item takes its retained references with it), and hand
        back the exception to raise or hold."""
        self.queue.clear()
        self.retained_bytes = 0
        return _translate(exc)

    def _pump_locked(self) -> None:
        """Launch the longest ready prefix; stop at a head whose unit is
        still building. Held errors stay held until the next drain."""
        if self.held_error or not self.queue:
            return
        head = self.queue[0][0]
        if head is not None and head.ext is None:
            return  # nothing launchable: don't force a thread-switch sync
        # Once, before anything is popped: an item must never be in flight
        # across a device synchronize (it would launch after its own
        # successors).
        self._order_queue_launch_locked()
        while self.queue:
            unit = self.queue[0][0]
            if unit is not None and unit.ext is None:
                return  # still building; unlaunched items keep their refs
            try:
                # The popped item is an unnamed temporary: its retained
                # references drop the moment _exec returns — on this
                # thread, under the shard lock, stream-ordered behind the
                # launch it protected.
                _exec(self._pop_locked())
            except BaseException as exc:
                self.held_error.append(self._abandon_locked(exc))
                return

    def _drain_waiting_locked(self) -> None:
        """Launch everything, waiting builds out; a failure becomes the
        held error (rule 5). Runs under `self.lock` with `in_launch` set
        by the caller."""
        self._order_queue_launch_locked()
        while self.queue:
            unit = self.queue[0][0]
            try:
                if unit is not None and unit.ext is None:
                    unit.request_async().wait()  # raises on build failure
                _exec(self._pop_locked())
            except BaseException as exc:
                self.held_error.append(self._abandon_locked(exc))
                return

    def _drain_over_budget_locked(self) -> None:
        """Rule 3's bound: queued items retain more device bytes than this
        device's budget allows, so stop running ahead — wait builds out
        and launch until the retention is released. This drain serves
        memory, not a value: a launch failure is held for the next real
        `drain()` exactly as `pump()` holds it, and when an error is
        already held the queue is a dead episode (rule 5) whose items
        would never launch — dropping them is what releases their
        retention. Runs under `self.lock` with `in_launch` set by the
        caller."""
        if self.held_error:
            self.queue.clear()
            self.retained_bytes = 0
            return
        self._drain_waiting_locked()

    def _budget(self) -> int:
        """This device's run-ahead retention bound, in bytes; 0 disables.

        ``TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB`` wins when set. Otherwise
        the bound is computed once from THIS shard's device: half its free
        VRAM at the moment its first launch queues (model weights are
        already resident by then, so this adapts to what the workload
        actually left available), floored at 1 GiB. Devices reporting no
        memory statistics fall back to 8 GiB.
        """
        cached = self.budget_bytes
        if cached is None:
            raw = os.environ.get("TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB", "")
            if raw:
                try:
                    cached = max(0, int(raw)) * 1024 * 1024
                except ValueError:
                    cached = _FALLBACK_BUDGET_MB * 1024 * 1024
            else:
                free = _free_device_memory(self.index)
                if free is None:
                    cached = _FALLBACK_BUDGET_MB * 1024 * 1024
                else:
                    cached = max(free // 2, _MIN_AUTO_BUDGET_MB * 1024 * 1024)
            self.budget_bytes = cached
        return cached

    def _enforce_budget_locked(self) -> None:
        """Drain when the just-appended item pushed retention past the
        budget. Skipped inside a launch (a reentrant enqueue must never
        re-enter the queue) — the outer launch path is already emptying
        it."""
        if (
            self.retained_bytes > self._budget() > 0  # noqa: SIM300 — 0 disables
            and not getattr(self.tls, "in_launch", False)
        ):
            self.tls.in_launch = True
            try:
                self._drain_over_budget_locked()
            finally:
                self.tls.in_launch = False

    # -- public per-shard entry points -------------------------------------

    def pump(self) -> None:
        """Launch whatever is ready. Non-blocking: never waits on a build."""
        if not self.queue or getattr(self.tls, "in_launch", False):
            return
        self.tls.in_launch = True
        try:
            with self.lock:
                self._pump_locked()
        finally:
            self.tls.in_launch = False

    def drain(self) -> None:
        """Launch everything (waiting out builds); re-raise launch errors.
        Called before any host read of this device's values.

        Guarded on the launch path as a whole, not on drain alone: a drain
        reached from inside a launch (device synchronize, holder release)
        must be a no-op, or it would run items 2..N ahead of the one
        already popped.
        """
        if getattr(self.tls, "in_launch", False):
            return
        self.tls.in_launch = True
        try:
            with self.lock:
                if self.held_error:
                    # Raised exactly once; anything queued since the
                    # failure belongs to the dead episode and goes with it.
                    exc = self.held_error.pop()
                    self.queue.clear()
                    self.retained_bytes = 0
                    raise exc
                if self.queue:
                    self._order_queue_launch_locked()
                    while self.queue:
                        unit = self.queue[0][0]
                        try:
                            if unit is not None and unit.ext is None:
                                unit.request_async().wait()  # build failure raises
                            _exec(self._pop_locked())
                        except BaseException as exc:
                            raise self._abandon_locked(exc)
        finally:
            self.tls.in_launch = False

    def _launch_prefix(self, unit: _QueueUnit | None) -> bool:
        """Launch the ready prefix, then report whether a new call may run
        inline: nothing queued ahead of it, no held error, and — for a
        unit call — its extension already loaded. Runs under `self.lock`."""
        if getattr(self.tls, "in_launch", False):
            return False  # inside a launch: never recurse into the queue
        self.tls.in_launch = True
        try:
            self._pump_locked()
        finally:
            self.tls.in_launch = False
        if self.queue or self.held_error:
            return False
        if unit is not None and unit.ext is None:
            return False
        self.order_direct_launch()
        return True

    def kernel_call_into(self, unit: _QueueUnit, args: tuple, keepalive: tuple) -> None:
        with self.lock:
            if self._launch_prefix(unit):
                _exec((unit, None, args, threading.current_thread(), keepalive, 0))
                return
            if unit.ext is None:
                unit.request_async()
            nbytes = _keepalive_bytes(keepalive)
            self.retained_bytes += nbytes
            self.queue.append(
                (unit, None, args, threading.current_thread(), keepalive, nbytes)
            )
            self._enforce_budget_locked()

    def external_call(
        self, fn: Callable[..., object], args: tuple, keepalive: tuple
    ) -> None:
        with self.lock:
            if self._launch_prefix(None):
                fn(*args)
                return
            nbytes = _keepalive_bytes(keepalive)
            self.retained_bytes += nbytes
            self.queue.append(
                (None, fn, args, threading.current_thread(), keepalive, nbytes)
            )
            self._enforce_budget_locked()


# ---------------------------------------------------------------------------
# Shard registry

_SHARDS: dict[int, _Shard] = {}
# Immutable snapshot in ascending index order, swapped whole under
# _REGISTRY_LOCK: pump()/active() run on every dispatch and iterate it
# without a lock or an allocation, and can never see a dict mid-mutation.
_SHARD_SNAPSHOT: tuple[_Shard, ...] = ()
_REGISTRY_LOCK = threading.Lock()

# max.driver.Device -> ordered-accelerator index, filled on first miss.
_DEVICE_INDEX_CACHE: dict = {}


def _shard(index: int) -> _Shard:
    global _SHARD_SNAPSHOT
    shard = _SHARDS.get(index)
    if shard is None:
        with _REGISTRY_LOCK:
            shard = _SHARDS.get(index)
            if shard is None:
                shard = _SHARDS[index] = _Shard(index)
                _SHARD_SNAPSHOT = tuple(_SHARDS[i] for i in sorted(_SHARDS))
    return shard


def device_index(device: DeviceKey) -> int:
    """The shard index for a `max.driver.Device` (an index passes
    through). The mapping is the device's position in
    `get_ordered_accelerators()` — the same numbering as torch's `mojo:N`
    and the RNG state table."""
    if isinstance(device, int):
        return device
    idx = _DEVICE_INDEX_CACHE.get(device)
    if idx is None:
        from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
            get_ordered_accelerators,
        )

        for position, accelerator in enumerate(get_ordered_accelerators()):
            _DEVICE_INDEX_CACHE.setdefault(accelerator, position)
        idx = _DEVICE_INDEX_CACHE[device]
    return idx


def shard_for(device: DeviceKey) -> _Shard:
    return _shard(device_index(device))


def device_lock(device: DeviceKey) -> _thread.RLock:
    """The re-entrant mutex serializing every touch of one device (rule
    6). `deferred_compile._direct` holds it around each op's kernels."""
    return shard_for(device).lock


# ---------------------------------------------------------------------------
# Process-wide mode and entry points


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
    """Re-read the mode environment variables on the next `enabled()`, and
    recompute every shard's budget on its next enqueue."""
    _ENABLED[0] = None
    for shard in _SHARD_SNAPSHOT:
        shard.budget_bytes = None


def active() -> bool:
    """Whether ANY device has queued launches."""
    for shard in _SHARD_SNAPSHOT:
        if shard.queue:
            return True
    return False


def set_error_translator(fn: Callable[[BaseException], None]) -> None:
    """Install the hook that retypes a device error raised by a *deferred*
    launch (aten_fast installs `_raise_if_device_oom`, so an allocator
    exhaustion still surfaces as `torch.OutOfMemoryError` even when the
    launch happened at drain time). It is called with the exception and
    either raises a better one or returns to let the original stand."""
    _ERROR_TRANSLATOR[0] = fn


def order_direct_launch(device: DeviceKey) -> None:
    """A direct (non-queued) launch is about to run on the current thread,
    on the given device. Call with that device's lock held (rule 6) —
    `deferred_compile._direct` is the production caller."""
    shard_for(device).order_direct_launch()


def pump(device: DeviceKey | None = None) -> None:
    """Launch whatever is ready. Non-blocking: never waits on a build.

    With a device, pumps only that shard — the dispatch hot path pumps the
    shard(s) of the op it is routing, so a rank never becomes another
    device's queue launcher (and never pays another device's rule-4
    synchronize) just by dispatching its own work. With None, pumps every
    shard."""
    if device is not None:
        shard_for(device).pump()
        return
    for shard in _SHARD_SNAPSHOT:
        shard.pump()


def drain(device: DeviceKey | None = None) -> None:
    """Launch everything (waiting out builds); re-raise launch errors.

    With a device, drains only that shard — the form host reads use when
    they know whose bytes they touch, so one rank's `loss.item()` never
    waits out another rank's compile storm. With None, drains every shard
    in ascending index order (public synchronize, device-crossing ops,
    collective launches)."""
    if device is not None:
        shard_for(device).drain()
        return
    for shard in _SHARD_SNAPSHOT:
        shard.drain()


def kernel_call_into(
    unit: _QueueUnit, args: tuple, keepalive: tuple, device: DeviceKey
) -> None:
    """Queue a descriptor call whose outputs were preallocated in Python.
    The call writes into them and returns nothing — always queueable
    regardless of the *Spec naming convention.

    `keepalive` names the objects whose buffers the raw `args` reference
    (outputs and inputs; containers are fine — retention chains through
    them). `device` names the device whose context those args reference.
    Both are required, not defaulted: every enqueue site must state what
    its pointers depend on and where they live."""
    shard_for(device).kernel_call_into(unit, args, keepalive)


def external_call(
    fn: Callable[..., object], args: tuple, keepalive: tuple, device: DeviceKey
) -> None:
    """An ungated device call (tensor_holder, fa4): always launchable, but
    must hold its FIFO position behind queued producers of its inputs on
    the same device.

    `args` are raw pointers and scalars; `keepalive` names the tensors
    those pointers belong to, retained on the queued item until it
    launches; `device` names the shard that owns the FIFO position."""
    shard_for(device).external_call(fn, args, keepalive)
