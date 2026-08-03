import functools
import math
import sys
import threading
from collections import deque
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import max.driver
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental.torch import max_dtype_to_torch

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.eager_kernels.data_movement_ops import DataMovementExtension
from torch_mojo_backend.eager_kernels.output_specs import (
    _submit_prepared_into,
    _TensorOutputSpec,
)
from torch_mojo_backend.mojo_device import objc_autorelease, torch_mojo_device_module

# The Mojo extension module (torch_mojo_backend.eager_kernels.tensor_holder),
# resolved lazily so that importing torch_mojo_backend never triggers a Mojo
# kernel compile.
_tensor_holder = None


def _holder_mod():
    global _tensor_holder
    if _tensor_holder is None:
        _tensor_holder = eager_kernels.tensor_holder
    return _tensor_holder


# GPU H2D copies consume a MAX-owned pinned staging allocation asynchronously.
# Keep that transfer owner alive until an event recorded behind its DMA
# completes. This mirrors the lifetime tracking performed by CUDA's pinned
# memory allocator without depending on torch-cuda.
_PENDING_H2D: dict[max.driver.Device, deque] = {}
_PENDING_H2D_LOCK = threading.Lock()

# A non-blocking D2H returns a CPU tensor that aliases a MAX-owned pinned host
# allocation. DLPack ties that owner to the returned tensor, while this queue
# also retains it until the DMA event completes if the tensor dies early.
_PENDING_D2H: dict[max.driver.Device, deque] = {}
_PENDING_D2H_LOCK = threading.Lock()

# A stream/event failure is already a fatal device condition, but raw-pointer
# lifetime must remain safe while the exception propagates. If both event
# recording and recovery synchronization fail, retain that transfer owner for
# the process lifetime rather than risk freeing memory still used by DMA.
_FAILED_TRANSFER_OWNERS: dict[max.driver.Device, list[tuple[object, object]]] = {}
_FAILED_TRANSFER_OWNERS_LOCK = threading.Lock()

# PyTorch's Python PrivateUse1 guard currently advertises one C++ autograd
# device queue, at index zero. Keep the storage-less wrapper TensorImpl on that
# bookkeeping device, as ``_acc.create_empty_tensor`` did. ``_torch_device`` and
# the public ``device`` property continue to carry the real Mojo device.
_WRAPPER_TENSORIMPL_DEVICE = torch.device("privateuseone:0")


def _ctx_ptr(device):
    # Rebinds this module-level name to the real (cached) implementation on
    # first use, so the lazy import costs one call, not one per call.
    global _ctx_ptr
    from torch_mojo_backend.eager_kernels import _ctx_ptr as real_ctx_ptr

    _ctx_ptr = real_ctx_ptr
    return real_ctx_ptr(device)


def _retain_failed_transfer_owner(device, owner: object) -> object:
    token = object()
    with _FAILED_TRANSFER_OWNERS_LOCK:
        _FAILED_TRANSFER_OWNERS.setdefault(device, []).append((token, owner))
    return token


def _forget_failed_transfer_owner(device, token: object) -> None:
    with _FAILED_TRANSFER_OWNERS_LOCK:
        retained = _FAILED_TRANSFER_OWNERS.get(device)
        if retained is None:
            return
        retained[:] = [entry for entry in retained if entry[0] is not token]
        if not retained:
            _FAILED_TRANSFER_OWNERS.pop(device, None)


def _record_h2d_source(device, source: object, non_blocking: bool) -> None:
    """Retain a pinned transfer owner until its default-stream H2D ends."""
    # MAX's CPU device uses a worker pool whose copies are not stream-ordered
    # with kernels. The Mojo helper drains it before returning; keep the Python
    # side blocking as well rather than advertising unsupported async behavior.
    if device == CPU():
        non_blocking = False

    try:
        event = device.default_stream.record_event()
    except Exception:
        # Stash before recovery: if synchronization also fails, retaining for
        # process lifetime is safer than freeing a raw DMA source prematurely.
        token = _retain_failed_transfer_owner(device, source)
        device.default_stream.synchronize()
        _forget_failed_transfer_owner(device, token)
        _release_synchronized_h2d_sources(device)
        raise

    if not non_blocking:
        event.synchronize()
        _release_synchronized_h2d_sources(device)
        return

    with _PENDING_H2D_LOCK:
        pending = _PENDING_H2D.setdefault(device, deque())
        # Retain the current DMA owner before querying older events. Event
        # queries can fail; unwinding must not free the just-enqueued source.
        pending.append((event, source))
        while pending and pending[0][0].is_ready():
            pending.popleft()
        # Do not impose a count-based wait here: a burst can legitimately have
        # many copies behind long-running GPU work. The FIFO is reaped on every
        # transfer and after explicit/blocking synchronization, while each
        # event keeps its exact source alive until DMA completion.


def _release_synchronized_h2d_sources(device) -> None:
    """Drop ready sources after the caller synchronized ``device``'s stream.

    Another thread may enqueue a transfer between the stream synchronization
    and this cleanup. Checking each event under the queue lock preserves that
    post-sync source until its own DMA completes.
    """
    with _PENDING_H2D_LOCK:
        pending = _PENDING_H2D.get(device)
        while pending and pending[0][0].is_ready():
            pending.popleft()
        if not pending:
            _PENDING_H2D.pop(device, None)


def _record_d2h_owner(device, owner: object) -> None:
    """Retain a pinned D2H allocation until its default-stream DMA ends."""
    try:
        event = device.default_stream.record_event()
    except Exception:
        token = _retain_failed_transfer_owner(device, owner)
        device.default_stream.synchronize()
        _forget_failed_transfer_owner(device, token)
        _release_synchronized_h2d_sources(device)
        _release_synchronized_d2h_owners(device)
        raise

    with _PENDING_D2H_LOCK:
        pending = _PENDING_D2H.setdefault(device, deque())
        # Retain the current source/destination before querying older events;
        # an event-query exception must not drop memory still used by DMA.
        pending.append((event, owner))
        while pending and pending[0][0].is_ready():
            pending.popleft()


def _release_synchronized_d2h_owners(device) -> None:
    """Drop pinned D2H owners whose stream events have completed."""
    with _PENDING_D2H_LOCK:
        pending = _PENDING_D2H.get(device)
        while pending and pending[0][0].is_ready():
            pending.popleft()
        if not pending:
            _PENDING_D2H.pop(device, None)


@runtime_checkable
class MojoTensorLike(Protocol):
    """Anything carrying the core Mojo eager payload metadata.

    Payload-level helpers only read these attributes, and host-contract
    tests exercise them with lightweight stand-ins, so their signatures
    declare this structural contract rather than the concrete wrapper.
    """

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _dtype: DType
    _device: object


def _dispatch_entry(func: object, args: tuple, kwargs: dict) -> object:
    """``deferred_compile.dispatch``, resolved on first use.

    ``deferred_compile`` imports the call queue this module feeds, so it
    cannot be imported at module scope. Rebinding the global on the first
    dispatch keeps that direction intact and leaves the steady state at one
    plain global lookup instead of a per-op ``sys.modules`` hit (the same
    trick as ``output_specs._alloc``).
    """
    global _dispatch_entry
    from . import deferred_compile

    _dispatch_entry = deferred_compile.dispatch
    return _dispatch_entry(func, args, kwargs)


@runtime_checkable
class _SynchronizableDevice(Protocol):
    """What allocation recovery needs from a device: a default stream it can
    synchronize. ``max.driver.Device`` satisfies it; host-contract tests use
    lightweight stand-ins."""

    default_stream: object


def _alloc_with_recovery(
    device: _SynchronizableDevice, nbytes: int
) -> tuple[object, int]:
    """One device allocation, with the reactive last resort under the budget.

    When the allocator refuses, the largest reclaimable set is whatever the
    kernel-call queue still retains: drain it (launching every pending item,
    waiting builds out), synchronize the device so the stream-ordered frees
    actually land, and retry exactly once. This cannot defeat size-class
    carve-up — the arena never splits or merges, which is what the proactive
    run-ahead budget prevents — but it converts pressure the budget cannot
    see (a model simply large for the card, external processes) into a
    stall instead of a failure. Non-OOM errors and a second failure
    propagate untouched.
    """
    holder_mod = _holder_mod()
    ctx_ptr = _ctx_ptr(device)
    try:
        return holder_mod.alloc(ctx_ptr, nbytes)
    except Exception as exc:
        from torch_mojo_backend.eager_kernels import call_queue, is_device_oom

        if not is_device_oom(exc):
            raise
        sys.stderr.write(
            f"torch-mojo-backend: device allocation of {nbytes} bytes failed; "
            f"draining {len(call_queue._QUEUE)} queued launch(es), "
            "synchronizing, and retrying once...\n"
        )
        call_queue.drain()
        device.default_stream.synchronize()
        return holder_mod.alloc(ctx_ptr, nbytes)


def _row_major_strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _compute_contiguous(shape, strides) -> bool:
    """torch's relaxed contiguity: size-1 dims never break contiguity."""
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


# max DType -> torch dtype, cached as a plain dict: max_dtype_to_torch is
# called once per tensor wrapper created (~600/decode step).
_TORCH_DTYPE_OF: dict = {}


def _torch_dtype_of(dtype):
    td = _TORCH_DTYPE_OF.get(dtype)
    if td is None:
        td = _TORCH_DTYPE_OF[dtype] = max_dtype_to_torch(dtype)
    return td


# max.driver.Device -> torch.device, cached like _TORCH_DTYPE_OF. Computed
# once per wrapper created so the `device` property is a plain attribute
# read — which also lets dynamo trace `x.device` inside compiled functions
# (the property body must not construct max.driver objects).
_TORCH_DEVICE_OF: dict = {}


def _torch_device_of(device):
    td = _TORCH_DEVICE_OF.get(device)
    if td is None:
        if device == CPU():
            td = torch_mojo_device_module.cpu()
        else:
            td = torch.device(f"mojo:{device.id}")
        _TORCH_DEVICE_OF[device] = td
    return td


# Strided kernels take shapes/strides padded to rank 8 with LEADING entries.
MAX_RANK = 8


def _pad8(values, fill: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) > MAX_RANK:
        raise NotImplementedError(
            f"mojo tensors support at most rank {MAX_RANK}, got {len(values)}"
        )
    return (fill,) * (MAX_RANK - len(values)) + values


class TorchMojoTensor(torch.Tensor):
    """Eager mojo tensor.

    A storage-less ``PrivateUse1`` wrapper subclass whose payload is:

    - `_holder`: a Mojo `TensorHolder` owning the device allocation. Views
      share the *same* holder object; CPython's refcount on it is the
      ownership mechanism, and the last drop enqueues the stream-ordered
      free (see docs/strided_owning_tensors_design.md).
    - Layout metadata as plain Python attributes (`_ptr`, `_shape`,
      `_strides` in elements, `_offset` in elements from the allocation
      start, `_dtype` as a max DType, `_numel`, `_itemsize`, `_device`,
      `_is_contiguous`).

    The wrapper's Python dispatch key only redispatches to the numerical
    ``PrivateUse1`` kernels. Autograd, autocast, and ADInplaceOrView remain
    PyTorch-owned layers above it. In particular, PyTorch can reconstruct a
    detached wrapper through ``__torch_dispatch__`` when it saves an operator
    output for backward, preserving the Python-side allocation payload.
    """

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Redispatch wrapper operations to the existing Mojo backend kernels."""
        # Give higher-priority wrappers such as FakeTensor and
        # FunctionalTensor their opportunity to handle mixed-subclass calls.
        if not all(issubclass(cls, tensor_type) for tensor_type in types):
            return NotImplemented

        # The MAX Metal runtime autoreleases ObjC objects per kernel launch;
        # without a periodically drained pool they leak until macOS SIGKILLs
        # the process (see objc_autorelease.py). No-op off macOS.
        objc_autorelease.note_op_dispatched()

        # The deferred-compile layer executes ops while kernel variants are
        # still building in the background (and is a plain pass-through to
        # the ordinary PrivateUse1 path when no compile is in flight).
        return _dispatch_entry(func, args, kwargs or {})

    @classmethod
    def _make(
        cls, holder, ptr, shape, strides, offset, dtype, device, contiguous=None
    ) -> "TorchMojoTensor":
        shape = tuple(shape)
        strides = tuple(strides)
        res = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            storage_offset=offset,
            dtype=_torch_dtype_of(dtype),
            layout=torch.strided,
            device=_WRAPPER_TENSORIMPL_DEVICE,
            requires_grad=False,
        )
        res._holder = holder
        res._ptr = ptr
        res._shape = shape
        res._strides = strides
        res._offset = offset
        res._dtype = dtype
        res._itemsize = dtype.size_in_bytes
        res._numel = math.prod(shape)
        res._device = device
        res._torch_device = _torch_device_of(device)
        res._is_contiguous = (
            _compute_contiguous(shape, strides) if contiguous is None else contiguous
        )
        return res

    @classmethod
    def _alloc(
        cls, shape, dtype: DType, device: max.driver.Device
    ) -> "TorchMojoTensor":
        """A new contiguous uninitialized tensor (one device allocation)."""
        shape = tuple(shape)
        numel = math.prod(shape)
        holder, ptr = _alloc_with_recovery(device, numel * dtype.size_in_bytes)
        result = cls._make(
            holder,
            ptr,
            shape,
            _row_major_strides(shape),
            0,
            dtype,
            device,
            contiguous=True,
        )
        return result

    @classmethod
    def _view_of(
        cls, base: "TorchMojoTensor", shape, strides, offset, contiguous=None
    ) -> "TorchMojoTensor":
        """A zero-copy view: shares base's holder, new layout metadata.

        `offset` is absolute, in elements from the allocation start.
        `contiguous` skips the contiguity rescan when the caller knows it.
        """
        ptr = base._ptr + (offset - base._offset) * base._itemsize
        return cls._make(
            base._holder,
            ptr,
            shape,
            strides,
            offset,
            base._dtype,
            base._device,
            contiguous=contiguous,
        )

    @classmethod
    def _from_cpu(
        cls,
        cpu_tensor: torch.Tensor,
        device: max.driver.Device,
        *,
        non_blocking: bool = False,
    ) -> "TorchMojoTensor":
        """H2D: allocate and enqueue a copy from a CPU torch tensor."""
        from max.experimental.torch.torch import torch_dtype_to_max

        t = cpu_tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        dtype = torch_dtype_to_max(t.dtype)
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            # Nothing to transfer; skip alloc_from_host's full queue drain.
            return cls._alloc(tuple(t.shape), dtype, device)
        holder, ptr, transfer_owner = _holder_mod().alloc_from_host(
            _ctx_ptr(device), t.data_ptr(), nbytes
        )
        _record_h2d_source(device, transfer_owner, non_blocking)
        return cls._make(
            holder,
            ptr,
            tuple(t.shape),
            _row_major_strides(t.shape),
            0,
            dtype,
            device,
            contiguous=True,
        )

    def _to_cpu_tensor(self, *, non_blocking: bool = False) -> torch.Tensor:
        """D2H into MAX-owned pinned storage exposed as a CPU tensor.

        Host read: queued kernel launches must land first (call queue).

        With ``non_blocking=True`` on a GPU, the returned tensor aliases the
        pinned destination immediately and the caller must synchronize before
        consuming it, matching PyTorch's asynchronous accelerator-to-CPU
        contract. Blocking and CPU-device copies are ready on return.
        """
        from . import deferred_compile

        src = self if self._is_contiguous else self._materialize_contiguous()
        # Reading device bytes is a host read: every queued launch must have
        # executed before the transfer is enqueued -- INCLUDING the strided
        # materialization just queued above, whose output is what this reads.
        deferred_compile.drain()
        if src._numel == 0:
            return torch.empty(self._shape, dtype=max_dtype_to_torch(self._dtype))

        nbytes = src._numel * src._itemsize
        if not non_blocking or src._device == CPU():
            out = torch.empty(src._shape, dtype=max_dtype_to_torch(src._dtype))
            _holder_mod().copy_to_host(
                _ctx_ptr(src._device), src._ptr, out.data_ptr(), nbytes
            )
            _release_synchronized_h2d_sources(src._device)
            _release_synchronized_d2h_owners(src._device)
            return out

        owner, ptr = _holder_mod().copy_to_pinned_host(
            _ctx_ptr(src._device), src._ptr, nbytes
        )
        # Record and retain both ends before DLPack adoption, which can raise.
        # A non-contiguous input may make ``src`` a temporary whose holder must
        # remain alive until the non-owning DeviceBuffer copy has completed.
        _record_d2h_owner(src._device, (owner, src._holder))
        try:
            from torch_mojo_backend.mojo_device import dlpack

            return torch.from_dlpack(
                dlpack.make_capsule(owner, ptr, src._shape, src._dtype, CPU())
            )
        except Exception:
            src._device.default_stream.synchronize()
            _release_synchronized_h2d_sources(src._device)
            _release_synchronized_d2h_owners(src._device)
            raise

    def _materialize_contiguous(self) -> "TorchMojoTensor":
        """A new contiguous tensor with this tensor's (strided) contents."""
        rank = len(self._shape)
        if self._numel > 0 and rank <= 4:
            # Hot path (attention q/k/v transposes, expand): the rank-4
            # PermuteCopy gathers a strided source into a contiguous
            # destination with no destination index math and half the
            # coordinate math of the generic rank-8 CopyStrided.
            return _submit_prepared_into(_PermuteCopyExtension.prepare(self))
        out = TorchMojoTensor._alloc(self._shape, self._dtype, self._device)
        if self._numel > 0:
            _copy_strided_into(out, self)
        return out

    def _contig(self) -> "TorchMojoTensor":
        """self if already contiguous, else a materialized copy."""
        return self if self._is_contiguous else self._materialize_contiguous()

    def __dlpack__(self, *, stream=None, **_unused):
        """Export the device allocation as a "dltensor" capsule.

        torch's inherited `__dlpack__` would export the zero-byte meta
        storage; this override exports the real allocation described by the
        Python-side metadata. Non-contiguous tensors are materialized first,
        and the capsule pins the (materialized) tensor's holder. `stream` is
        ignored: producers and consumers share the device's default stream
        (the same assumption the eager kernels make).

        Publishing a raw pointer is a payload read from outside
        `__torch_dispatch__`, so it drains the call queue: the consumer must
        not see a buffer whose producing launches -- including the copy a
        strided export just queued -- are still waiting on a compile.
        """
        from torch_mojo_backend.mojo_device import dlpack

        from . import deferred_compile

        src = self._contig()
        deferred_compile.drain()
        return dlpack.make_capsule(
            src._holder, src._ptr, src._shape, src._dtype, src._device
        )

    def __dlpack_device__(self):
        from torch_mojo_backend.mojo_device import dlpack

        return dlpack.dlpack_device(self._device)

    def __coerce_same_metadata_as_tangent__(self, expected_meta, expected_type=None):
        """Accept mojo tensors as backward tangents under torch.compile.

        AOTAutograd guesses tangent types from fake tensors, which are plain
        `torch.Tensor`s, and rejects runtime tangents of unexpected types
        unless this hook coerces them. A mojo tensor behaves exactly like a
        plain tensor for dispatch purposes, so no conversion is needed.
        """
        if expected_type not in (None, torch.Tensor):
            return None
        return self

    def __reduce_ex__(self, protocol):
        """Pickle as a portable plain CPU tensor.

        torch.Tensor's reduce would pickle this subclass's `__dict__`, which
        holds the unpicklable Mojo `TensorHolder`. Checkpoints written from
        this device therefore serialize tensor values on the CPU and can be
        loaded without this backend being installed. Moving them back is the
        caller's `.to('mojo')` or `load_state_dict` onto a Mojo model.
        """
        if hasattr(self, "_holder"):
            return self._to_cpu_tensor().__reduce_ex__(protocol)
        return super().__reduce_ex__(protocol)

    def __repr__(self):
        if hasattr(self, "_holder"):
            return f"TorchMojoTensor({self._to_cpu_tensor()!r}, device='{self.device}')"
        return super().__repr__()

    # NOTE: shape/ndim/dim/size/stride/is_contiguous/numel/storage_offset are
    # deliberately NOT overridden. ``_make_wrapper_subclass`` records the same
    # sizes, strides and storage offset on the TensorImpl that ``_make`` stores
    # in the Python payload, and ``_rebind_payload`` resizes the TensorImpl
    # alongside a payload swap, so the C++ accessors are already authoritative
    # (they were only shadowed while the wrapper was a contiguous-only
    # ``_acc.create_empty_tensor``). Re-adding a Python override would break
    # torch.compile: dynamo models a tensor subclass through
    # ``TensorWithTFOverrideVariable``, which INLINES Python-level metadata
    # overrides -- so ``x.shape[0]`` would be traced as a read of the
    # ``_shape`` tuple, specializing the shape as a constant and, once dynamic,
    # minting a symbol unrelated to the tensor's size symbol (a graph input the
    # MAX backend cannot supply). It is also slower than the C++ accessor.

    @property
    def device(self):
        # A plain attribute read so dynamo can trace `x.device` in compiled
        # functions (e.g. `torch.arange(T, device=idx.device)`).
        if hasattr(self, "_torch_device"):
            return self._torch_device
        return super().device

    __torch_function__ = torch._C._disabled_torch_function_impl


class _PermuteCopyExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, "TorchMojoTensor"]
):
    """Stateless descriptor for rank-at-most-four strided materialization.

    It borrows data_movement_ops' source but not `MojoFileExtension`'s ABI:
    that base takes the specialization apart from `(op, arg_dtypes, flags)`
    call arguments, while this descriptor names its own defines. Deriving
    from it would shadow `make_defines` with the base's
    `make_canonical_defines` and never call the override.
    """

    MOJO_FILE: ClassVar[Path] = DataMovementExtension.MOJO_FILE

    @classmethod
    def make_defines(cls, tensor: MojoTensorLike) -> dict[str, bool | int | str]:
        return {
            "OP": "PermuteCopy",
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT": tensor._dtype.name,
        }

    @classmethod
    def expected_output_specs(cls, tensor: MojoTensorLike) -> _TensorOutputSpec:
        return _TensorOutputSpec(tuple(tensor._shape), tensor._dtype, tensor._device)

    @classmethod
    def extension_args(
        cls, out: TorchMojoTensor, tensor: MojoTensorLike
    ) -> tuple[object, ...]:
        rank = len(tensor._shape)
        pad = 4 - rank
        return (
            out._ptr,
            tensor._ptr,
            (1,) * pad + tuple(tensor._shape),
            (0,) * pad + tuple(tensor._strides),
            tensor._dtype.size_in_bytes,
            _ctx_ptr(tensor._device),
        )


def _rebind_payload(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """Move ``src``'s eager payload into ``dst`` without changing identity.

    Both the Python payload and the real TensorImpl must move together. A
    manual ``__dict__`` rebind makes direct properties look right while APIs
    such as ``torch.numel``, ``mT`` and ``flatten`` continue reading the stale
    TensorImpl.

    Swapping TensorImpl pointers is not valid inside an out kernel: the boxed
    dispatcher retains the original TensorImpl to enforce the schema's alias
    return, so a swap would make the call return the discarded wrapper. The
    CPU ``resize_`` kernel only updates TensorImpl/storage metadata here (the
    wrapper's dummy storage uses the Meta allocator); redispatch it explicitly
    to retain the original TensorImpl, then move the authoritative Mojo
    payload. No CPU tensor data is allocated or read.
    """
    torch.ops.aten.resize_.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CPU),
        dst,
        src._shape,
        memory_format=None,
    )
    for name in (
        "_holder",
        "_ptr",
        "_shape",
        "_strides",
        "_offset",
        "_dtype",
        "_itemsize",
        "_numel",
        "_device",
        "_torch_device",
        "_is_contiguous",
    ):
        setattr(dst, name, getattr(src, name))
    # Any cached spec describes the old allocation or layout. Rebuild it on
    # the next spec operation instead of retaining stale pointer metadata.
    dst.__dict__.pop("_spec", None)


def _resize_payload(dst: TorchMojoTensor, shape) -> None:
    """Resize an eager out tensor and keep aliases when storage is sufficient.

    PyTorch resets a resized view to contiguous strides at its existing
    storage offset. Reuse that same allocation when the requested logical
    bytes fit; this preserves writes observed through another view such as
    ``base[:0]``. Otherwise use an ordinary context allocation. The final
    swap synchronizes Python metadata and TensorImpl metadata without changing
    ``dst``'s Python identity.
    """
    shape = tuple(shape)
    if shape == tuple(dst._shape):
        return

    required_bytes = math.prod(shape) * dst._itemsize
    allocation_bytes = int(dst._holder.get_nbytes())
    available_bytes = allocation_bytes - dst._offset * dst._itemsize
    if available_bytes < 0:
        available_bytes = 0
    if required_bytes <= available_bytes:
        replacement = TorchMojoTensor._make(
            dst._holder,
            dst._ptr,
            shape,
            _row_major_strides(shape),
            dst._offset,
            dst._dtype,
            dst._device,
            contiguous=True,
        )
    else:
        replacement = TorchMojoTensor._alloc(shape, dst._dtype, dst._device)
    _rebind_payload(dst, replacement)


def _copy_strided_enqueue(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """Queue the strided copy as an external call (tensor_holder is always
    loaded, so the item is always launch-ready; it only holds FIFO order).
    The queue holds raw pointers, so both tensors are handed over as the
    item's keep-alive: their buffers must outlive the launch."""
    from torch_mojo_backend.eager_kernels import call_queue as _cq

    holder = _holder_mod()
    args = (
        dst._ptr,
        src._ptr,
        _pad8(dst._shape, 1),
        _pad8(dst._strides, 0),
        _pad8(src._strides, 0),
        dst._itemsize,
        _ctx_ptr(dst._device),
    )
    _cq.external_call(holder.CopyStrided, args, keepalive=(dst, src))


def _copy_strided_into(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """dst[coords] = src[coords]; same shape and dtype, any strides.

    The shared materialize/copy primitive: powers .contiguous(), copy_ into
    views, and expand materialization (src strides may contain 0s).
    """
    from torch_mojo_backend.eager_kernels import call_queue as _cq

    if _cq.enabled() and _cq.active():
        # Hold FIFO position behind queued producers of src/dst.
        _copy_strided_enqueue(dst, src)
        return
    _holder_mod().CopyStrided(
        dst._ptr,
        src._ptr,
        _pad8(dst._shape, 1),
        _pad8(dst._strides, 0),
        _pad8(src._strides, 0),
        dst._itemsize,
        _ctx_ptr(dst._device),
    )


@functools.cache
def get_ordered_accelerators():
    """Get accelerators ordered with GPUs first, then CPU last"""
    from torch_mojo_backend.torch_compile_backend.compiler import get_accelerators

    accelerators = list(get_accelerators())

    # Separate GPU and CPU accelerators
    gpu_accelerators = [acc for acc in accelerators if acc.label == "gpu"]
    cpu_accelerators = [acc for acc in accelerators if acc.label == "cpu"]

    # Order: GPUs first, then CPU last
    return gpu_accelerators + cpu_accelerators


def find_equivalent_max_device(device: torch.device) -> max.driver.Device:
    """Find the equivalent MAX device for a given torch device

    Device mapping:
    - mojo (no index) -> torch.mojo.current_device()
    - mojo:0 -> First GPU (or CPU if no GPUs)
    - mojo:1, mojo:2, ... -> Additional GPUs
    - mojo:<last_index> -> CPU device
    """
    ordered_accelerators = get_ordered_accelerators()

    if device.type == "mojo":
        # mojo with specific index
        if device.index is None:
            # Match PyTorch device semantics: an indexless backend device means
            # the backend's current device, not permanently device zero.
            return ordered_accelerators[torch_mojo_device_module.current_device()]
        else:
            if device.index < len(ordered_accelerators):
                return ordered_accelerators[device.index]
            else:
                raise ValueError(f"Invalid mojo index {device.index}")
    elif device.type == "cpu":
        # Find CPU accelerator (should be last in ordered list)
        for acc in reversed(ordered_accelerators):  # Check from the end
            if acc.label == "cpu":
                return acc
        # If no CPU found, return last accelerator as fallback
        return ordered_accelerators[-1]
    elif device.type in ("cuda", "hip"):
        # Find GPU accelerator (should be first in ordered list)
        # TODO: allow setting the default device index globally like with cuda
        gpu_index = device.index if device.index is not None else 0
        gpu_accelerators = [acc for acc in ordered_accelerators if acc.label == "gpu"]
        if gpu_index < len(gpu_accelerators):
            return gpu_accelerators[gpu_index]
        raise RuntimeError(f"GPU index {gpu_index} not available in MAX")
    else:
        raise NotImplementedError(f"Cannot convert {device.type} to MAX device")
