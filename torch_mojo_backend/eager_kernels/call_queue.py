"""Kernel-call-level deferral: queue extension calls, not aten ops.

The deferral boundary sits at the extension-call layer, where a call is
nothing but raw pointers / shapes / dtypes plus a DeviceContext. All
torch-level semantics (autograd graph, version counters, views, layouts,
allocations) complete synchronously at dispatch time in Python; only the
device kernel launches are delayed while their compiled units build in the
background pool — exactly CUDA's async-stream model, applied to kernel
*compilation*.

Correctness rules:
- Strict FIFO. Once any call is queued (a compile miss), every subsequent
  kernel call queues behind it, warm or not: device work must launch in
  program order. `pump()` launches the longest ready prefix.
- Host reads (D2H, synchronize, value-dependent ops) `drain()` first.
- Queued items reference raw pointers, not tensors, so every buffer
  touched while the queue is non-empty is retained in `_KEEPALIVE` (op
  scopes feed it: dispatch brackets each aten op and collects its args,
  results, and every `_alloc` made inside) until the queue empties.
- A call raising "unsupported dtype" at launch time rebuilds its unit
  with the full dtype set and retries that one call.
- Launch-time errors surface at the next drain, CUDA-style.
"""

import os
import threading
from collections import deque

_LOCK = threading.RLock()
_QUEUE: deque = deque()  # (unit | None, attr | callable, args, kwargs)
_KEEPALIVE: list = []
_HELD_ERROR: list = []
_EXEC_THREAD: list = [None]
_TLS = threading.local()  # .scope: per-op keepalive list, .in_drain


def enabled() -> bool:
    """Kernel-call queueing is the default execution mode; the sequential
    knob and the unit-test suite (synchronous contracts) disable it."""
    if os.environ.get("TMB_KERNEL_QUEUE", "1") != "1":
        return False
    if os.environ.get("TMB_NO_TRIGGER_DEFER"):
        return False
    from torch_mojo_backend.is_running_tests import IS_RUNNING_TESTS

    if IS_RUNNING_TESTS and not os.environ.get("TMB_FORCE_KERNEL_QUEUE"):
        return False
    return True


def active() -> bool:
    return bool(_QUEUE)


# ---------------------------------------------------------------------------
# Op-scoped keep-alive: queued items hold raw pointers, so the tensors whose
# buffers they reference must outlive the queue. dispatch() brackets every
# aten op; _alloc reports each allocation into the current scope.


def op_begin() -> list | None:
    scope = getattr(_TLS, "scope", None)
    _TLS.scope = []
    return scope  # previous scope (restored at op_end; dispatch can nest)


def note_alloc(tensor: object) -> None:
    scope = getattr(_TLS, "scope", None)
    if scope is not None:
        scope.append(tensor)


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


def _order_exec_thread() -> None:
    """Device work is ordered only within an enqueuing thread: on a thread
    switch, synchronize the device first (same rule as the aten-level
    layer; verified empirically — cross-thread enqueues read stale data)."""
    me = threading.current_thread()
    if _EXEC_THREAD[0] is not None and _EXEC_THREAD[0] is not me:
        from torch_mojo_backend.mojo_device import torch_mojo_device_module as _dm

        _dm.synchronize()
    _EXEC_THREAD[0] = me


def _exec(item: tuple) -> None:
    unit, attr, args, kwargs = item
    _order_exec_thread()
    if unit is None:
        attr(*args, **kwargs)  # external (e.g. fa4): attr is the callable
        return
    fn = getattr(unit.ext, attr)
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # Mojo errors surface as plain Exception
        if "unsupported dtype" not in str(exc) or unit.dtypes is None:
            raise
        ext = unit.load_blocking(all_dtypes=True)
        getattr(ext, attr)(*args, **kwargs)


def _pump_locked() -> None:
    """Launch the longest ready prefix; stop at a head whose unit is still
    building. Held errors stay held until the next drain."""
    if _HELD_ERROR:
        return
    while _QUEUE:
        unit = _QUEUE[0][0]
        if unit is not None and unit.ext is None:
            return
        item = _QUEUE.popleft()
        try:
            _exec(item)
        except BaseException as exc:
            _HELD_ERROR.append(exc)
            return
    # Keep-alive is NOT cleared here: releasing holders enqueues frees on
    # the CURRENT thread's stream order, which is only safe relative to
    # launches from this thread. drain() clears instead — its launches and
    # the thread-switch syncs guarantee every queued kernel precedes the
    # frees in device order.


def pump() -> None:
    with _LOCK:
        _pump_locked()


def drain() -> None:
    """Launch everything (waiting out builds); re-raise launch errors.
    Called before any host read of device values. Reentrancy-safe: the
    thread-switch synchronize inside _exec calls back into drain."""
    if getattr(_TLS, "in_drain", False):
        return
    _TLS.in_drain = True
    try:
        with _LOCK:
            if _HELD_ERROR:
                raise _HELD_ERROR.pop()
            while _QUEUE:
                unit = _QUEUE[0][0]
                if unit is not None and unit.ext is None:
                    unit.request_async().wait()  # raises on build failure
                item = _QUEUE.popleft()
                _exec(item)
            _KEEPALIVE.clear()
    finally:
        _TLS.in_drain = False


# ---------------------------------------------------------------------------
# Entry points for callers


def _returns_value(attr: str) -> bool:
    """Spec-tier entry points allocate their output inside Mojo and RETURN
    (holder, spec, ...): the caller consumes the value immediately, so the
    call cannot be queued. The raw tier writes into pre-allocated pointers
    and returns None. Naming convention: spec entries end in "Spec". A
    mis-classification is loud (the caller crashes on None), never silent.
    Making the spec tier queueable needs Into-style entry points (Python
    pre-allocates, Mojo writes into it) — the noted follow-up."""
    return attr.endswith("Spec")


def kernel_call(unit: object, attr: str, args: tuple, kwargs: dict) -> object:
    """A gated extension call (from the module proxy's wrapper)."""
    with _LOCK:
        _pump_locked()
        if _returns_value(attr):
            # Value-returning call: must execute NOW, and device ordering
            # requires every queued launch to land first.
            if _QUEUE:
                drain()
            if _HELD_ERROR:
                raise _HELD_ERROR.pop()
            if unit.ext is None:
                unit.load_blocking()
            _order_exec_thread()
            fn = getattr(unit.ext, attr)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if "unsupported dtype" not in str(exc) or unit.dtypes is None:
                    raise
                ext = unit.load_blocking(all_dtypes=True)
                return getattr(ext, attr)(*args, **kwargs)
        if not _QUEUE and unit.ext is not None and not _HELD_ERROR:
            _exec((unit, attr, args, kwargs))
            return None
        if unit.ext is None:
            unit.request_async()
        _QUEUE.append((unit, attr, args, kwargs))
    return None


def kernel_call_into(unit: object, attr: str, args: tuple) -> None:
    """A gated Into-style spec launch: Python pre-allocated the output, the
    call writes into it and returns nothing — always queueable regardless
    of the *Spec naming convention."""
    with _LOCK:
        _pump_locked()
        if not _QUEUE and unit.ext is not None and not _HELD_ERROR:
            _exec((unit, attr, args, {}))
            return None
        if unit.ext is None:
            unit.request_async()
        _QUEUE.append((unit, attr, args, {}))
    return None


def external_call(fn: object, args: tuple) -> None:
    """An ungated device call (fa4): always launchable, but must hold its
    FIFO position behind queued producers of its inputs."""
    with _LOCK:
        _pump_locked()
        if not _QUEUE and not _HELD_ERROR:
            _exec((None, fn, args, {}))
            return None
        _QUEUE.append((None, fn, args, {}))
    return None
