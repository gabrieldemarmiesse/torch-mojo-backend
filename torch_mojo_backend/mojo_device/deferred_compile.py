"""Dispatch entry for the mojo device, and its synchronization points.

Every aten op intercepted by ``TorchMojoTensor.__torch_dispatch__`` lands
here and executes SYNCHRONOUSLY at the torch level: autograd bookkeeping,
version counters, views, layouts and allocations are all complete when
``dispatch`` returns. What may lag behind is device kernel LAUNCHES whose
compiled units are still building — those wait in the kernel-call queue
(``eager_kernels.call_queue``), the CUDA async-stream model applied to
kernel compilation. This layer contributes exactly three things:

- pump the call queue at every dispatch entry (launch the ready prefix);
- drain it before device work that would bypass the queue (see below);
- route each op through the lock of the device it touches.

Buffer retention is not this layer's job: every queued item carries the
tensors its raw pointers name (queue rule 3), stated explicitly at each
enqueue site.

Host reads drain where they touch bytes — ``_to_cpu_tensor``,
``read_scalar`` behind ``aten::_local_scalar_dense``, ``__dlpack__`` — not
from a list of op names here. What the dispatcher must still cover is the
opposite direction: device WRITES that never enter the queue at all, namely
the H2D ``copy_from_host`` and the cross-GPU ``copy_d2d_peer`` behind a
cross-device ``aten::_to_copy`` / ``aten::copy_`` / ``aten::_copy_from``.
Those would land in a buffer a queued launch is about to overwrite — or
read a buffer a queued launch has not produced yet. Every other candidate
is inert and deliberately absent: ``aten::item`` and ``aten::allclose``
decompose to ``_local_scalar_dense``, which drains itself; ``aten::nonzero``
drains through ``_to_cpu_tensor``; ``aten::cpu`` is not an operator;
``aten::equal`` and ``aten::masked_select`` have no PrivateUse1 kernel and
raise. Adding a fast kernel for a value-reading op means adding its drain
next to the read, not a name here.

The queue is sharded per device (one FIFO + one lock per mojo device), so
the mutex an op holds is the one for the device its tensors live on: rank
threads of single-process data parallelism dispatch concurrently instead
of serializing on a process mutex. An op that genuinely spans devices — a
cross-device copy — first drains every shard (its out-of-queue transfer
must not overtake queued producers on either side), then holds the locks
of every device it touches, acquired in ascending index order so two
cross-device ops can never deadlock against each other.

Direct launches are ordered by their issuing thread and need no
cross-thread barrier — ``main`` has always interleaved the forward (main)
and backward (autograd engine) threads without one. What was verified
empirically as unsafe is a *queue* launch replaying another thread's work,
so ``_direct`` calls ``call_queue.order_direct_launch()`` for its device,
which is a plain tracker update unless that device's queue recently
launched from another thread (then it synchronizes that device once and
clears).
"""

from contextlib import ExitStack

import torch
from torch.utils._pytree import tree_flatten

from torch_mojo_backend.eager_kernels import call_queue


def _op_device_indices(args: tuple, kwargs: dict) -> list[int]:
    """The mojo device indices an op's tensors live on, in first-seen
    order.

    A linear walk, not a pytree flatten: on the hot path almost every op
    is (tensor, ...) or (tensor, tensor, ...) on ONE device. Tensor lists
    (foreach ops) are represented by their first element — torch's foreach
    fast paths already require a single device per list. The `device=`
    kwarg of factory/copy ops counts as a touched device too.
    """
    indices: list[int] = []

    def note(torch_device: torch.device) -> None:
        if torch_device.type != "mojo":
            return
        index = torch_device.index
        if index is None:
            from . import torch_mojo_device_module as _dm

            index = _dm.current_device()
        if index not in indices:
            indices.append(index)

    for a in args:
        if isinstance(a, torch.Tensor):
            note(a.device)
        elif isinstance(a, list | tuple) and a and isinstance(a[0], torch.Tensor):
            note(a[0].device)
    if kwargs:
        target = kwargs.get("device")
        if target is not None:
            note(torch.device(target))
        for a in kwargs.values():
            if isinstance(a, torch.Tensor):
                note(a.device)
    if not indices:
        from . import torch_mojo_device_module as _dm

        indices.append(_dm.current_device())
    return indices


def _direct(
    func: object, args: tuple, kwargs: dict, indices: list[int] | None = None
) -> object:
    """Execute one aten op through the PrivateUse1 kernels.

    MAX's DeviceContext is not documented thread-safe: every touch of one
    device is serialized on that device's shard lock — the SAME re-entrant
    mutex its queue uses (rule 6), so a drain on one thread cannot overlap
    a direct launch on another and there is no second lock for `_direct`
    -> `kernel_call_into` to invert against. Ops spanning several devices
    take every lock, in ascending index order (see module docstring).
    """
    if indices is None:
        indices = _op_device_indices(args, kwargs)
    if len(indices) == 1:
        with call_queue.device_lock(indices[0]):
            call_queue.order_direct_launch(indices[0])
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)
    with ExitStack() as stack:
        for index in sorted(indices):
            stack.enter_context(call_queue.device_lock(index))
        for index in sorted(indices):
            call_queue.order_direct_launch(index)
        with torch._C._DisableTorchDispatch():
            return func(*args, **kwargs)


# The only ops whose PrivateUse1 kernels touch a device outside the queue:
# the host/pinned staging behind a CPU<->mojo copy and the peer path behind
# a mojo<->mojo copy (`copy_from_host`, `copy_d2d_peer`).
_DEVICE_CROSSING_OPS = frozenset({"aten::_to_copy", "aten::copy_", "aten::_copy_from"})


def _crosses_device(args: tuple, kwargs: dict) -> bool:
    """True when a copy/cast actually moves bytes between devices. A
    same-device cast (autocast!) is an ordinary data op that stays in the
    queue; only a real crossing runs an out-of-queue transfer. Devices
    compare by (type, index): a mojo:0 -> mojo:1 peer copy is a crossing
    exactly like cpu -> mojo:0."""
    flat_args, _ = tree_flatten((args, kwargs))
    devices = {
        (a.device.type, a.device.index)
        for a in flat_args
        if isinstance(a, torch.Tensor)
    }
    target = kwargs.get("device")
    if target is not None:
        torch_device = torch.device(target)
        devices.add((torch_device.type, torch_device.index))
    return len(devices) > 1


def dispatch(func: object, args: tuple, kwargs: dict) -> object:
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    if not call_queue.enabled():
        return _direct(func, args, kwargs)

    # Pump only the shards this op touches. Pumping every shard here would
    # let one rank's dispatch become ANOTHER device's queue launcher during
    # a compile storm — paying that device's rule-4 synchronize while
    # holding its lock, and flipping its launcher identity so the owning
    # rank pays a second one: exactly the cross-rank latency coupling the
    # sharding exists to remove. A shard whose owner went quiet still
    # launches at its next drain (host read, synchronize, collective) —
    # pump is opportunistic, drain is the correctness point.
    indices = _op_device_indices(args, kwargs)
    for index in indices:
        call_queue.pump(index)
    if (
        call_queue.active()
        and func._schema.name in _DEVICE_CROSSING_OPS
        and _crosses_device(args, kwargs)
    ):
        # The whole queue, not one shard: the out-of-queue transfer reads
        # one device and writes another, and both sides' queued producers
        # must land first.
        call_queue.drain()

    return _direct(func, args, kwargs, indices)


def drain(device: call_queue.DeviceKey | None = None) -> None:
    """Public: wait for pending kernel launches.

    The single façade for the queue's drain — device code that reads tensor
    payloads outside ``__torch_dispatch__`` (the AutogradPrivateUse1 sdpa
    impl, ``_to_cpu_tensor``) calls this before it touches bytes. With a
    device (a ``max.driver.Device`` or index), only that device's shard
    drains — one rank's host read never waits out another rank's compile
    storm; with None every shard drains. FIFO granularity per shard: the
    whole shard drains, not just one tensor's producers.
    """
    call_queue.drain(device)
