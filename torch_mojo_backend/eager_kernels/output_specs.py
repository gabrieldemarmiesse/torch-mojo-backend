"""Output metadata inferred in Python, and the two ways to submit a call.

A descriptor whose Mojo entry point writes into a caller-allocated output can
infer that output's shape/dtype/device without loading (let alone compiling)
the module. Python then allocates it and either queues the launch — the call
queue never has to wait for the .so — or, when queueing is off, executes the
descriptor synchronously.

This lives in ``eager_kernels`` rather than in ``aten_fast`` because both the
aten layer and ``mojo_device.torch_mojo_tensor`` (strided materialization) go
through it, and the tensor module cannot import the aten layer. Nothing here
imports ``aten_fast``.
"""

from dataclasses import dataclass
from typing import TypeVar

import torch
from max.driver import Device
from max.dtype import DType

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.eager_kernels import call_queue as _call_queue

_Specs = TypeVar("_Specs")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class _TensorOutputSpec:
    """Shape/type/device metadata inferred without loading a Mojo module."""

    shape: tuple[int, ...]
    dtype: DType
    device: Device


def _alloc(shape: tuple[int, ...], dtype: DType, device: Device) -> torch.Tensor:
    """``TorchMojoTensor._alloc``, resolved on first use.

    ``mojo_device`` imports ``eager_kernels``, never the other way round, so
    the allocator cannot be imported at module scope. Rebinding this global on
    the first call keeps that direction intact and leaves the steady state at
    one plain attribute lookup (the same trick as ``_ctx_ptr``). Tensors are
    typed ``torch.Tensor`` here for the same import-direction reason.
    """
    global _alloc
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    _alloc = TorchMojoTensor._alloc
    return _alloc(shape, dtype, device)


def _allocate_output_spec(spec: _TensorOutputSpec) -> torch.Tensor:
    return _alloc(spec.shape, spec.dtype, spec.device)


def _submit_prepared_into(
    prepared: "eager_kernels.PreparedExtensionCall[_Specs, _Result]",
    *,
    force_sync: bool = False,
) -> _Result:
    """Allocate from inferred metadata and queue, or execute synchronously.

    Output arity is the descriptor's business: ``allocate_outputs`` returns
    whatever container ``extension_args`` accepts as its ``out``.
    """
    if force_sync and _call_queue.enabled():
        _call_queue.drain()
    if force_sync or not _call_queue.enabled():
        return prepared.execute()
    out = prepared.extension.allocate_outputs(prepared.output_specs)
    # (out, prepared.args) is the item's keep-alive: every tensor whose
    # pointer extension_args serialized is reachable through those two
    # references, including intermediates living only in prepared.args.
    prepared.enqueue_into(
        prepared.extension.extension_args(out, *prepared.args), (out, prepared.args)
    )
    return out
