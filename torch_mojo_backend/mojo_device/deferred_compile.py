"""Dispatch entry for the mojo device, and its synchronization points.

Every aten op intercepted by ``TorchMojoTensor.__torch_dispatch__`` lands
here and executes SYNCHRONOUSLY at the torch level: autograd bookkeeping,
version counters, views, layouts and allocations are all complete when
``dispatch`` returns. What may lag behind is device kernel LAUNCHES whose
compiled units are still building — those wait in the kernel-call queue
(``eager_kernels.call_queue``), the CUDA async-stream model applied to
kernel compilation. This layer contributes exactly three things:

- pump the call queue at every dispatch entry (launch the ready prefix);
- drain it before device work that would bypass the queue (see below).

Buffer retention is not this layer's job: every queued item carries the
tensors its raw pointers name (queue rule 3), stated explicitly at each
enqueue site.

Host reads drain where they touch bytes — ``_to_cpu_tensor``,
``read_scalar`` behind ``aten::_local_scalar_dense``, ``__dlpack__`` — not
from a list of op names here. What the dispatcher must still cover is the
opposite direction: device WRITES that never enter the queue at all, namely
the H2D ``copy_from_host`` behind a cross-device ``aten::_to_copy`` /
``aten::copy_``. Those would land in a buffer a queued launch is about to
overwrite. Every other candidate is inert and deliberately absent:
``aten::item`` and ``aten::allclose`` decompose to ``_local_scalar_dense``,
which drains itself; ``aten::nonzero`` drains through ``_to_cpu_tensor``;
``aten::cpu`` is not an operator; ``aten::equal`` and ``aten::masked_select``
have no PrivateUse1 kernel and raise. Adding a fast kernel for a
value-reading op means adding its drain next to the read, not a name here.

Direct launches are ordered by their issuing thread and need no
cross-thread barrier — ``main`` has always interleaved the forward (main)
and backward (autograd engine) threads without one. What was verified
empirically as unsafe is a *queue* launch replaying another thread's work,
so ``_direct`` calls ``call_queue.order_direct_launch()``, which is a
plain tracker update unless the queue recently launched from another
thread (then it synchronizes once and clears).
"""

import torch
from torch.utils._pytree import tree_flatten

from torch_mojo_backend.eager_kernels import call_queue

# MAX's DeviceContext is not documented thread-safe: serialize every
# device-touching call. This is the SAME re-entrant mutex the queue uses, so
# a drain on one thread cannot overlap a direct launch on another — and
# there is no second lock for `_direct` -> `kernel_call_into` to invert
# against.
_DEVICE_LOCK = call_queue._LOCK


def _direct(func: object, args: tuple, kwargs: dict) -> object:
    """Execute one aten op through the PrivateUse1 kernels."""
    with _DEVICE_LOCK:
        call_queue.order_direct_launch()
        with torch._C._DisableTorchDispatch():
            return func(*args, **kwargs)


# The only ops whose PrivateUse1 kernels touch the device outside the queue.
_DEVICE_CROSSING_OPS = frozenset({"aten::_to_copy", "aten::copy_"})


def _crosses_device(args: tuple, kwargs: dict) -> bool:
    """True when a copy/cast actually moves bytes between devices. A
    same-device cast (autocast!) is an ordinary data op that stays in the
    queue; only a real crossing runs an out-of-queue H2D/D2H transfer."""
    flat_args, _ = tree_flatten((args, kwargs))
    devices = {a.device.type for a in flat_args if isinstance(a, torch.Tensor)}
    target = kwargs.get("device")
    if target is not None:
        devices.add(torch.device(target).type)
    return len(devices) > 1


def dispatch(func: object, args: tuple, kwargs: dict) -> object:
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    if not call_queue.enabled():
        return _direct(func, args, kwargs)

    call_queue.pump()
    if (
        call_queue.active()
        and func._schema.name in _DEVICE_CROSSING_OPS
        and _crosses_device(args, kwargs)
    ):
        call_queue.drain()

    return _direct(func, args, kwargs)


def drain() -> None:
    """Public: wait for all pending kernel launches (device synchronize).

    The single façade for the queue's drain — device code that reads tensor
    payloads outside ``__torch_dispatch__`` (the AutogradPrivateUse1 sdpa
    impl, ``_to_cpu_tensor``) calls this before it touches bytes. FIFO
    granularity: the whole queue drains, not just one tensor's producers.
    """
    call_queue.drain()
