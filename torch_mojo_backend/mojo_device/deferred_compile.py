"""Dispatch entry for the mojo device, and its synchronization points.

Every aten op intercepted by ``TorchMojoTensor.__torch_dispatch__`` lands
here and executes SYNCHRONOUSLY at the torch level: autograd bookkeeping,
version counters, views, layouts and allocations are all complete when
``dispatch`` returns. What may lag behind is device kernel LAUNCHES whose
compiled units are still building — those wait in the kernel-call queue
(``eager_kernels.call_queue``), the CUDA async-stream model applied to
kernel compilation. This layer contributes exactly three things:

- pump the call queue at every dispatch entry (launch the ready prefix);
- drain it before host-visible reads (``_SYNC_OPS``, device crossings);
- bracket each op so every buffer it touches is kept alive while queued
  launches still hold raw pointers to it (``op_begin``/``op_end``).

Device work is ordered only within an enqueuing thread (verified
empirically: cross-thread enqueues read stale data), so `_direct`
synchronizes the device whenever the enqueuing thread changes — switches
are rare (main <-> autograd engine, per backward pass).
"""

import os
import threading

import torch
from torch.utils._pytree import tree_flatten

from torch_mojo_backend.eager_kernels import call_queue

# MAX's DeviceContext is not documented thread-safe: serialize every
# device-touching call.
_DEVICE_LOCK = threading.RLock()

# The thread whose device work was enqueued last (see module docstring).
_DEVICE_THREAD: list = [None]


def _order_device_thread() -> None:
    me = threading.current_thread()
    if _DEVICE_THREAD[0] is not None and _DEVICE_THREAD[0] is not me:
        from . import torch_mojo_device_module as _dm

        _dm.synchronize()
    _DEVICE_THREAD[0] = me


def _direct(func: object, args: tuple, kwargs: dict) -> object:
    """Execute one aten op through the PrivateUse1 kernels."""
    with _DEVICE_LOCK:
        _order_device_thread()
        with torch._C._DisableTorchDispatch():
            return func(*args, **kwargs)


# Ops whose results the host is about to look at (or that cross devices):
# they drain the call queue first so every pending launch has landed.
_SYNC_OPS = frozenset(
    {
        "aten::_local_scalar_dense",
        "aten::equal",
        "aten::allclose",
        "aten::_to_copy",
        "aten::copy_",
        "aten::item",
        "aten::cpu",
        "aten::nonzero",
        "aten::masked_select",
    }
)


def dispatch(func: object, args: tuple, kwargs: dict) -> object:
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    if not call_queue.enabled():
        return _direct(func, args, kwargs)

    call_queue.pump()
    name = func._schema.name
    sync = name in _SYNC_OPS
    if (
        sync
        and not os.environ.get("TMB_CAST_SYNC")
        and name in ("aten::_to_copy", "aten::copy_")
    ):
        # Same-device copies/casts (autocast!) are ordinary data ops —
        # only actual device crossings behave as host-visible syncs.
        flat_args, _ = tree_flatten((args, kwargs))
        devices = {a.device.type for a in flat_args if isinstance(a, torch.Tensor)}
        target = kwargs.get("device")
        if target is not None:
            devices.add(torch.device(target).type)
        sync = len(devices) > 1
    if sync and call_queue.active():
        call_queue.drain()

    prev = call_queue.op_begin()
    result = None
    try:
        result = _direct(func, args, kwargs)
        return result
    finally:
        call_queue.op_end(prev, args, kwargs, result)


def drain() -> None:
    """Public: wait for all pending kernel launches (device synchronize)."""
    call_queue.drain()


def wait_for(tensors: list) -> None:
    """Public: barrier for device code that reads tensor payloads OUTSIDE
    __torch_dispatch__ (e.g. the AutogradPrivateUse1 sdpa impl). Such
    readers bypass the dispatch layer entirely, so every queued launch —
    including the producers of `tensors` — must land before they touch
    bytes."""
    _ = tensors  # FIFO granularity: the full queue drains
    if call_queue.active():
        call_queue.drain()
