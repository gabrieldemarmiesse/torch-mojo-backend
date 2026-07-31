"""Periodic ObjC autorelease-pool drain for the Metal dispatch path.

The MAX Metal runtime autoreleases ~3 KB of Objective-C objects (command
buffers, encoders, ...) per kernel launch. A Python host process has no
ambient autorelease pool, so those objects accumulate forever: measured
~0.7 MB per nanoGPT training step, tens of GB over hours, ending in a
silent jetsam SIGKILL once unified memory runs out. Wrapping GPU work in a
pool reclaims all of it (measured +0.0 MB steady state vs +14.5 MB per
5,000 launches without).

Pools are per-thread and strictly LIFO, so each dispatching thread keeps
its own open pool and cycles it (pop + push) every `_DRAIN_INTERVAL` ops
at the op boundary — a point where no Apple-framework frames of ours are
live above the pool. Pool push/pop is nanoseconds; the interval only
bounds peak accumulation (~3 KB x launches-per-op x interval).

macOS only; every entry point degrades to a no-op elsewhere.

This is a workaround: the root fix belongs in the MAX Metal runtime
(scoping its own pools around command-buffer work, or explicit ownership
of the Metal objects). Once modular ships that, this module becomes a
nanosecond-cost no-op and can be deleted.
"""

import ctypes
import sys
import threading

_DRAIN_INTERVAL = 256

_libobjc = None
if sys.platform == "darwin":
    try:
        _libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        _libobjc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
        _libobjc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    except OSError:
        _libobjc = None

_state = threading.local()


def note_op_dispatched() -> None:
    """Count one dispatched op; cycle this thread's pool every interval."""
    if _libobjc is None:
        return
    count = getattr(_state, "count", 0) + 1
    if count < _DRAIN_INTERVAL:
        _state.count = count
        return
    _state.count = 0
    pool = getattr(_state, "pool", None)
    if pool is not None:
        _libobjc.objc_autoreleasePoolPop(pool)
    _state.pool = _libobjc.objc_autoreleasePoolPush()
