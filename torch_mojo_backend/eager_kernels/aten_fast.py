"""ATen-signature-compatible fast implementations for mojo_device eager mode.

Each function here is registered in `mojo_device_aten_ops.py` (through
`_eager_impl`). Tensors are `TorchMojoTensor`s: a Mojo `TensorHolder`
ownership token plus Python-side layout metadata (`_ptr`, `_shape`,
`_strides` in elements, `_offset`, `_dtype`, `_device`, `_is_contiguous`).
An op runs as one or a few Mojo kernel calls — no graph building, no MLIR
passes, no interpreter, and *no graph fallback*: when a function returns
the `NOT_HANDLED` sentinel, the registration raises `NotImplementedError`
naming the op (see docs/strided_owning_tensors_design.md).

View ops (view / permute / transpose / expand / slice / select / split /
squeeze / unsqueeze / alias) are ZERO-COPY: they return new wrappers
sharing the base tensor's holder with adjusted shape/strides/offset.
Compute kernels take contiguous operands; strided inputs either feed the
broadcast-strided kernels directly (rank <= 4, real strides) or are
materialized first via the `CopyStrided` primitive.

Only the eager (mojo_device) path uses this module; the torch.compile
backend keeps using `aten_functions` directly.
"""

import math
import struct
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import torch
from max.driver import Device
from max.dtype import DType

from torch_mojo_backend import eager_kernels, is_running_tests
from torch_mojo_backend.eager_kernels import call_queue as _call_queue
from torch_mojo_backend.eager_kernels.activation_backward_ops import (
    ActivationBackwardExtension as _ActivationBackwardExtension,
)
from torch_mojo_backend.eager_kernels.activation_forward_ops import (
    ActivationForwardExtension as _ActivationForwardExtension,
)
from torch_mojo_backend.eager_kernels.bf16_matmul_ops import (
    BF16MatmulExtension as _Bf16MatmulExtension,
)
from torch_mojo_backend.eager_kernels.conv_ops import ConvExtension as _ConvExtension
from torch_mojo_backend.eager_kernels.data_movement_ops import (
    DataMovementExtension as _DataMovementExtension,
)
from torch_mojo_backend.eager_kernels.dropout_ops import (
    DropoutExtension as _DropoutExtension,
)
from torch_mojo_backend.eager_kernels.elementwise_ops import (
    ElementwiseExtension as _ElementwiseExtension,
)
from torch_mojo_backend.eager_kernels.embedding_backward_ops import (
    EmbeddingBackwardExtension as _EmbeddingBackwardExtension,
)
from torch_mojo_backend.eager_kernels.flash_attention_ops import (
    FlashAttentionExtension as _FlashAttentionExtension,
)
from torch_mojo_backend.eager_kernels.logic_ops import LogicExtension as _LogicExtension
from torch_mojo_backend.eager_kernels.loss_ops import LossExtension as _LossExtension
from torch_mojo_backend.eager_kernels.matmul_ops import (
    MatmulExtension as _MatmulExtension,
)
from torch_mojo_backend.eager_kernels.nn_ops import NNExtension as _NNExtension
from torch_mojo_backend.eager_kernels.normalization_backward_ops import (
    NormalizationBackwardExtension as _NormalizationBackwardExtension,
)
from torch_mojo_backend.eager_kernels.normalization_forward_ops import (
    NormalizationForwardExtension as _NormalizationForwardExtension,
)
from torch_mojo_backend.eager_kernels.optimizer_ops import (
    OptimizerExtension as _OptimizerExtension,
)
from torch_mojo_backend.eager_kernels.reduction_ops import (
    ReductionExtension as _ReductionExtension,
)
from torch_mojo_backend.eager_kernels.sdpa_backward_ops import (
    SDPABackwardExtension as _SdpaBackwardExtension,
)
from torch_mojo_backend.eager_kernels.softmax_backward_ops import (
    SoftmaxBackwardExtension as _SoftmaxBackwardExtension,
)
from torch_mojo_backend.eager_kernels.tf32_matmul_ops import (
    TF32MatmulExtension as _Tf32MatmulExtension,
)

_VariantFlag = bool | int | str


def _device_call(fn: object, *args: object, keepalive: tuple[object, ...]) -> object:
    """Launch an ungated device call (tensor_holder / fa4): when the call
    queue is active it must hold its FIFO position behind queued producers
    of its inputs; otherwise call directly. `keepalive` names the tensors
    whose buffers the raw `args` reference (queue rule 3)."""
    if _call_queue.enabled():
        return _call_queue.external_call(fn, args, keepalive)
    return fn(*args)


def _call_mojo(
    extension: type[eager_kernels.MojoFileExtension],
    op: str,
    extension_args: tuple[object, ...],
    *,
    arg_dtypes: tuple[DType, ...],
    output_dtypes: tuple[DType, ...] = (),
    flags: dict[str, _VariantFlag] | None = None,
    keepalive: tuple[object, ...],
) -> object:
    """Invoke one exact, shape-independent stateless Mojo extension.

    `keepalive` names every tensor whose `_spec_of(...)` / raw pointer went
    into `extension_args`; a queued launch retains them until it runs
    (queue rule 3). Required and keyword-only so no call site can forget
    it."""
    try:
        return extension.invoke(
            op,
            extension_args,
            arg_dtypes=arg_dtypes,
            output_dtypes=output_dtypes,
            flags=flags,
            keepalive=keepalive,
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        raise


from torch_mojo_backend.eager_kernels import _ctx_ptr
from torch_mojo_backend.eager_kernels.output_specs import (
    _allocate_output_spec,
    _submit_prepared_into,
    _TensorOutputSpec,
)
from torch_mojo_backend.mojo_device.torch_mojo_device_module import (
    _reserve_philox_state,
)
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    MojoTensorLike,
    TorchMojoTensor,
    _copy_strided_into,
    _pad8,
    _resize_payload,
    _row_major_strides,
)

# Returned when the inputs don't qualify; the registration then raises
# NotImplementedError naming the op (there is no fallback anymore).
NOT_HANDLED = object()

# The host/autograd route is landed before Fable's separately validated kernel
# port.  Do not ask mojo.importer to compile the thin bridge until every source
# it imports exists; a missing optional source would otherwise make ordinary
# eager SDPA backward pay for a predictably failing compiler subprocess.
_SDPA_BACKWARD_SOURCE_PATHS = (
    eager_kernels._PACKAGE_DIR / _SdpaBackwardExtension.MOJO_FILE,
    eager_kernels._PACKAGE_DIR
    / "sdpa_backward_ops/sdpa_dropout_softmax_backward_kernels.mojo",
    eager_kernels._PACKAGE_DIR / "sdpa_backward_ops/sdpa_backward_gemm_kernels.mojo",
)

# Keep the optional BF16 bridge dormant until the bridge, optimized dispatcher,
# and accepted fallback all exist. A partial dependency closure would otherwise
# launch a predictably failing compile before an ordinary eager matmul.
_BF16_SOURCE_PATHS = (
    eager_kernels._PACKAGE_DIR / _Bf16MatmulExtension.MOJO_FILE,
    eager_kernels._PACKAGE_DIR / "bf16_matmul_ops/bf16_gemm_v3_kernels.mojo",
    eager_kernels._PACKAGE_DIR / "bf16_matmul_ops/bf16_gemm_tn_v4_kernels.mojo",
    eager_kernels._PACKAGE_DIR / "bf16_matmul_ops/bf16_gemm_kernels.mojo",
)

# The TF32 host route is useful before the separately profiled Fable kernel is
# installed, but the thin bridge imports that kernel unconditionally.  Avoid a
# predictably failing lazy compile (and an unnecessary output allocation) while
# either source is absent from a source checkout or wheel.
_TF32_SOURCE_PATHS = (
    eager_kernels._PACKAGE_DIR / _Tf32MatmulExtension.MOJO_FILE,
    eager_kernels._PACKAGE_DIR / "tf32_matmul_ops/tf32_gemm_kernels.mojo",
)


@runtime_checkable
class _OptionalSource(Protocol):
    """One source file of an optional bridge (tests substitute stand-ins)."""

    def is_file(self) -> bool: ...


# One `is_file()` answer per source tuple, not one per call: these guards sit
# in front of every bf16/tf32 matmul and every SDPA backward, where re-stat'ing
# the sources costs more than all the metadata checks around them (~9 us
# measured for the four-path bf16 tuple). Keyed on the IDENTITY of the tuple,
# so a test that swaps in a synthetic tuple gets a fresh answer rather than a
# stale latch.
_BRIDGE_AVAILABILITY: dict[str, tuple[object, bool]] = {}


def _bridge_available(name: str, paths: tuple[_OptionalSource, ...]) -> bool:
    cached = _BRIDGE_AVAILABILITY.get(name)
    if cached is not None and cached[0] is paths:
        return cached[1]
    available = all(path.is_file() for path in paths)
    _BRIDGE_AVAILABILITY[name] = (paths, available)
    return available


def _bf16_bridge_available() -> bool:
    """Whether the optional BF16 bridge and all of its sources are present."""
    return _bridge_available("bf16", _BF16_SOURCE_PATHS)


def _tf32_bridge_available() -> bool:
    """Whether the optional TF32 bridge and all of its sources are present."""
    return _bridge_available("tf32", _TF32_SOURCE_PATHS)


def _sdpa_backward_bridge_available() -> bool:
    """Whether the optional SDPA backward bridge and its sources are present."""
    return _bridge_available("sdpa_backward", _SDPA_BACKWARD_SOURCE_PATHS)


# The Mojo kernels raise (instead of falling back) on dtypes they don't
# support; gate float-only ops here.
_FLOAT_DTYPES = (DType.float16, DType.bfloat16, DType.float32)

_INT_SCALAR_DTYPES = (DType.int32, DType.int64)

# Dtypes the binary elementwise kernels support (no bool: the arithmetic
# kernels don't lower for it, so bool tensors go through uint8 views or
# are rejected).
_BINARY_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)

# Dtypes the data-movement (pure copy) kernels support.
_COPYABLE_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.bool,
)

_CAST_DTYPES = (
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.uint8,
    DType.bool,
)

# What the Fill kernel dispatches on (no uint16/32/64).
_FILL_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
)

# int64 scalars round-trip through the Fill kernel's Float64 argument.
_MAX_EXACT_INT = 2**53

_BITWISE_DTYPES = (
    DType.bool,
    DType.uint8,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
)

# Dtypes the broadcast-strided logic_ops kernels dispatch on (no float64,
# no bool: bool goes through uint8 views).
_BCAST_DTYPES = _FLOAT_DTYPES + (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)

_FUSED_ADAMW_RECORD_FIELDS = 7
_FOREACH_CHUNK_ELEMENTS = 65_536
_FOREACH_NORM_RECORD_FIELDS = 3
_FOREACH_MUL_RECORD_FIELDS = 2
_FOREACH_ADD_RECORD_FIELDS = 2

# Opcodes of the batched foreach elementwise bridge (must match
# foreach_elementwise_kernels.mojo).
_FOREACH_EW_MUL = 0
_FOREACH_EW_ADD = 1
_FOREACH_EW_DIV = 2
_FOREACH_EW_ADDCMUL = 0
_FOREACH_EW_ADDCDIV = 1

# ATen Scalar arguments as they arrive at a PrivateUse1 registration.
AtenScalar = int | float | bool | complex


def _t(x) -> TorchMojoTensor | None:
    """x as a TorchMojoTensor (any layout), or None."""
    return x if isinstance(x, TorchMojoTensor) else None


def _tc(x) -> TorchMojoTensor | None:
    """x as a *contiguous* TorchMojoTensor (materializing views), or None."""
    if isinstance(x, TorchMojoTensor):
        return x if x._is_contiguous else x._materialize_contiguous()
    return None


_alloc = TorchMojoTensor._alloc
_view_of = TorchMojoTensor._view_of


def _reduce_ready_operand(
    a: TorchMojoTensor, dims: tuple[int, ...]
) -> tuple[TorchMojoTensor, tuple[int, ...]]:
    """``(operand, dims)`` with any strided or non-trailing layout
    materialized HERE, in Python.

    The Mojo reduce bridges refuse layouts they would previously have
    copied into a scratch buffer: materializing through the queued strided
    copy instead means the transient is allocated by ``_alloc`` — metered
    by the run-ahead budget, covered by the allocation retry, and retained
    per queued item like every other buffer. The permuted layout is kept
    dims ascending then reduce dims ascending (the same layout the bridge
    geometry derives), so the reduce dims become the trailing ones. The
    hot path — contiguous input, trailing dims in order — returns the pair
    unchanged, and a permutation that is already contiguous (reordered
    trailing dims) costs a zero-copy view.
    """
    rank = len(a._shape)
    trailing = tuple(range(rank - len(dims), rank))
    if a._is_contiguous and dims == trailing:
        return a, dims
    reduced = frozenset(dims)
    perm = [d for d in range(rank) if d not in reduced] + sorted(reduced)
    view = _view_of(
        a,
        tuple(a._shape[d] for d in perm),
        tuple(a._strides[d] for d in perm),
        a._offset,
    )
    return view._contig(), trailing


def _sum_middle_direct_ok(
    spec_fn_name: str, module_name: str, a: TorchMojoTensor, dims: tuple[int, ...]
) -> bool:
    """Whether the bridge's zero-copy direct kernel takes this reduction.

    Mirrors reduction_ops' early exit exactly: a contiguous fp32 SumSpec
    over one adjacent, ascending, NON-trailing dim interval on an
    accelerator. Those calls skip Python-side materialization — the direct
    kernel reads the source in place and allocates nothing."""
    if spec_fn_name != "SumSpec" or module_name != "reduction_ops":
        return False
    if a._dtype != DType.float32 or not a._is_contiguous:
        return False
    if getattr(a._device, "api", "cpu") == "cpu":
        return False
    rank = len(a._shape)
    if dims == tuple(range(rank - len(dims), rank)):
        return False  # trailing: the ordinary rows/cols path is the fast one
    return dims == tuple(range(dims[0], dims[0] + len(dims)))


def _reduce_keepdim_shape(
    result: TorchMojoTensor, shape: tuple[int, ...]
) -> TorchMojoTensor:
    """Reshape a contiguous reduce result to its keepdim shape (free view).

    Needed only when the reduce dims were re-pointed at the trailing slots:
    the buffer is identical, but keepdim's 1s belong at the ORIGINAL dim
    positions, not the trailing ones the bridge wrote."""
    return _view_of(result, shape, _row_major_strides(shape), result._offset)


def fast_aten__foreach_norm(self, ord=2, dtype=None):
    """Fast homogeneous FP32 L2 norms with one independent scalar output.

    The Mojo bridge batches runtime descriptors and uses an ordinary eager
    tensor as reduction scratch.  Destroying that local tensor enqueues its
    stream-ordered free after the kernels, so the call remains asynchronous.
    """
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if (
        isinstance(ord, bool)
        or not isinstance(ord, int | float)
        or ord != 2
        or dtype not in (None, torch.float32)
    ):
        return NOT_HANDLED

    tensors = [_t(tensor) for tensor in self]
    if any(tensor is None for tensor in tensors):
        return NOT_HANDLED
    device = tensors[0]._device
    if device.api == "cpu" or any(
        tensor._device != device
        or tensor._dtype != DType.float32
        or not tensor._is_contiguous
        for tensor in tensors
    ):
        return NOT_HANDLED

    outputs = [_alloc((), DType.float32, device) for _ in tensors]
    metadata = tuple(
        value
        for tensor, output in zip(tensors, outputs, strict=True)
        for value in (tensor._ptr, output._ptr, tensor._numel)
    )
    if len(metadata) != len(tensors) * _FOREACH_NORM_RECORD_FIELDS:
        raise AssertionError("invalid foreach norm metadata packing")
    total_chunks = sum(
        (tensor._numel + _FOREACH_CHUNK_ELEMENTS - 1) // _FOREACH_CHUNK_ELEMENTS
        for tensor in tensors
    )
    partials = _alloc((max(total_chunks, 1),), DType.float32, device)
    _call_mojo(
        _OptimizerExtension,
        "ForeachL2Norm",
        (metadata, partials._ptr, partials._numel, _ctx_ptr(device)),
        arg_dtypes=(DType.float32, DType.float32, DType.float32),
        output_dtypes=(DType.float32,),
        keepalive=(partials,),
    )
    return outputs


def foreach_norm_sequential_fallback(self, ord=2, dtype=None):
    """Device-index-correct L2 fallback using existing scalar eager ops.

    ATen's generic foreach decomposition synthesizes functional norm outputs
    from the out= overload. On PrivateUse1 that can allocate on the phantom
    default device rather than the input's real MAX context. Allocate through
    each tensor's scalar eager path instead; other norm regimes still use CEA.
    """
    if (
        isinstance(ord, bool)
        or not isinstance(ord, int | float)
        or ord != 2
        or dtype not in (None, torch.float32)
    ):
        return NOT_HANDLED
    tensors = [_t(tensor) for tensor in self]
    if any(tensor is None or tensor._dtype != DType.float32 for tensor in tensors):
        return NOT_HANDLED

    outputs = []
    for tensor in tensors:
        output = fast_aten_linalg_vector_norm(tensor, ord, None, False, dtype=None)
        if output is NOT_HANDLED:
            return NOT_HANDLED
        outputs.append(output)
    return outputs


def _foreach_tensors_overlap(tensors) -> bool:
    """Whether contiguous mutable tensor byte intervals overlap."""
    intervals = sorted(
        (tensor._ptr, tensor._ptr + tensor._numel * tensor._itemsize)
        for tensor in tensors
        if tensor._numel > 0
    )
    return any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
    )


def _is_non_overlapping_and_dense(shape, strides) -> bool:
    """Match TensorImpl's sorted-stride dense-layout classification."""
    required_stride = 1
    dimensions = sorted(
        (stride, size) for size, stride in zip(shape, strides, strict=True) if size >= 2
    )
    for stride, size in dimensions:
        if stride != required_stride:
            return False
        required_stride *= size
    return True


def _foreach_scalar_overlap_kind(tensor, scalar) -> str:
    """Classify scalar overlap as none, full, or forbidden partial overlap.

    PyTorch reports overlap as ``TooHard`` for non-dense layouts, so those
    must reach its sequential fallback rather than using a storage envelope.
    """
    if (
        tensor._device != scalar._device
        or tensor._numel == 0
        or not _is_non_overlapping_and_dense(tensor._shape, tensor._strides)
    ):
        return "none"
    tensor_begin = tensor._ptr
    tensor_end = tensor_begin + tensor._numel * tensor._itemsize
    scalar_begin = scalar._ptr
    scalar_end = scalar_begin + scalar._itemsize
    if scalar_begin >= tensor_end or tensor_begin >= scalar_end:
        return "none"
    if (
        tensor_begin == scalar_begin
        and tensor_end == scalar_end
        and tensor._strides == scalar._strides
    ):
        return "full"
    return "partial"


def _fast__foreach_add__scalar_generic(self, scalar):
    """One launch for `t += scalar` over a whole homogeneous float tensor list.

    Without this ATen runs the CompositeExplicitAutograd fallback, a sequential
    `add_.Scalar` per element of the list. nanoGPT's fused AdamW bumps 75
    one-element step counters per step, so that is 75 launches to add 75 floats.
    Anything this cannot serve -- a mixed dtype, a non-float dtype, a strided
    tensor, a list whose members alias -- returns ``NOT_HANDLED`` and keeps that
    fallback, which is also what defines the semantics being matched.
    """
    if len(self) == 0:
        return NOT_HANDLED
    if isinstance(scalar, bool) or not isinstance(scalar, int | float):
        return NOT_HANDLED
    tensors = [_t(tensor) for tensor in self]
    if any(tensor is None for tensor in tensors):
        return NOT_HANDLED
    device = tensors[0]._device
    dtype = tensors[0]._dtype
    if (
        device.api == "cpu"
        or dtype not in _FLOAT_DTYPES
        or any(
            tensor._device != device
            or tensor._dtype != dtype
            or not tensor._is_contiguous
            for tensor in tensors
        )
        # Duplicated or aliasing entries have to be applied once each, in
        # order; one grid over the concatenation would race instead.
        or _foreach_tensors_overlap(tensors)
    ):
        return NOT_HANDLED

    metadata = tuple(
        value for tensor in tensors for value in (tensor._ptr, tensor._numel)
    )
    if len(metadata) != len(tensors) * _FOREACH_ADD_RECORD_FIELDS:
        raise AssertionError("invalid foreach add metadata packing")
    _call_mojo(
        _ElementwiseExtension,
        "ForeachAddScalar",
        (metadata, float(scalar), dtype.value, _ctx_ptr(device)),
        arg_dtypes=(dtype,),
        output_dtypes=(dtype,),
        flags={"INPLACE": True},
        keepalive=(tensors,),
    )
    return None


def fast_aten__foreach_mul__tensor(self, other):
    """Fast homogeneous FP32 in-place multiply by a device scalar tensor."""
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    scalar = _t(other)
    tensors = [_t(tensor) for tensor in self]
    if scalar is None or any(tensor is None for tensor in tensors):
        return NOT_HANDLED
    device = tensors[0]._device
    if scalar._shape == ():
        overlap_kinds = [
            _foreach_scalar_overlap_kind(tensor, scalar) for tensor in tensors
        ]
        if "partial" in overlap_kinds:
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if "full" in overlap_kinds:
            return NOT_HANDLED
    if (
        device.api == "cpu"
        or scalar._device != device
        or scalar._dtype != DType.float32
        or scalar._shape != ()
        or not scalar._is_contiguous
        or any(
            tensor._device != device
            or tensor._dtype != DType.float32
            or not tensor._is_contiguous
            for tensor in tensors
        )
        or _foreach_tensors_overlap(tensors)
    ):
        return NOT_HANDLED

    metadata = tuple(
        value for tensor in tensors for value in (tensor._ptr, tensor._numel)
    )
    if len(metadata) != len(tensors) * _FOREACH_MUL_RECORD_FIELDS:
        raise AssertionError("invalid foreach multiply metadata packing")
    _call_mojo(
        _OptimizerExtension,
        "ForeachMulTensor",
        (metadata, scalar._ptr, _ctx_ptr(device)),
        arg_dtypes=(DType.float32, DType.float32),
        output_dtypes=(DType.float32,),
        flags={"INPLACE": True},
        keepalive=(scalar,),
    )
    return None


def _foreach_scalar_value(scalar: AtenScalar) -> float | None:
    """A python float for an ATen Scalar, or None when it disqualifies."""
    if isinstance(scalar, bool) or not isinstance(scalar, int | float):
        return None
    return float(scalar)


def _foreach_scalar_values(
    scalars: Sequence[AtenScalar], count: int
) -> list[float] | None:
    if len(scalars) != count:
        return None
    values = []
    for scalar in scalars:
        value = _foreach_scalar_value(scalar)
        if value is None:
            return None
        values.append(value)
    return values


def _foreach_metal_f32(
    *lists: Sequence[torch.Tensor],
) -> list[list[TorchMojoTensor]] | None:
    """Unwrap parallel TensorLists for the batched Metal foreach kernels.

    Qualifies only homogeneous lists: every entry a contiguous float32
    TorchMojoTensor on one shared Metal device, with corresponding entries
    of all lists shaped identically (the batched kernels are elementwise
    and never broadcast). Returns None otherwise so callers fall back.
    """
    count = len(lists[0])
    if count == 0:
        return None
    unwrapped = []
    for tensors in lists:
        if len(tensors) != count:
            return None
        wrapped = []
        for tensor in tensors:
            mojo_tensor = _t(tensor)
            if mojo_tensor is None:
                return None
            wrapped.append(mojo_tensor)
        unwrapped.append(wrapped)
    device = unwrapped[0][0]._device
    if device.api != "metal":
        return None
    for tensors in unwrapped:
        for index, tensor in enumerate(tensors):
            if (
                tensor._device != device
                or tensor._dtype != DType.float32
                or not tensor._is_contiguous
                or tensor._shape != unwrapped[0][index]._shape
            ):
                return None
    return unwrapped


def _foreach_mutation_hazard(
    mutated: Sequence[TorchMojoTensor], operands: Sequence[Sequence[TorchMojoTensor]]
) -> bool:
    """Whether a mutated tensor's bytes intersect any other tensor's bytes.

    One batched launch gives no ordering between slots, so any aliasing
    that the sequential per-tensor fallback would tolerate (including exact
    self-aliasing operands) is conservatively routed back to it. Operands
    may freely alias each other (addcmul_(v, g, g) is the common case).
    """
    if _foreach_tensors_overlap(mutated):
        return True
    mutated_intervals = sorted(
        (tensor._ptr, tensor._ptr + tensor._numel * tensor._itemsize)
        for tensor in mutated
        if tensor._numel > 0
    )
    operand_intervals = sorted(
        (tensor._ptr, tensor._ptr + tensor._numel * tensor._itemsize)
        for tensors in operands
        for tensor in tensors
        if tensor._numel > 0
    )
    merged: list[list[int]] = []
    for begin, end in operand_intervals:
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    mutated_index = 0
    merged_index = 0
    while mutated_index < len(mutated_intervals) and merged_index < len(merged):
        mutated_begin, mutated_end = mutated_intervals[mutated_index]
        operand_begin, operand_end = merged[merged_index]
        if mutated_end <= operand_begin:
            mutated_index += 1
        elif operand_end <= mutated_begin:
            merged_index += 1
        else:
            return True
    return False


def _foreach_scalar_inplace(
    self: Sequence[torch.Tensor], values: Sequence[float], op_code: int
) -> object:
    """Shared launch path of the in-place foreach mul/add/div fast ops."""
    lists = _foreach_metal_f32(self)
    if lists is None:
        return NOT_HANDLED
    (tensors,) = lists
    if _foreach_mutation_hazard(tensors, ()):
        return NOT_HANDLED
    metadata = tuple(
        value for tensor in tensors for value in (tensor._ptr, tensor._numel)
    )
    _call_mojo(
        _OptimizerExtension,
        "ForeachScalarOp",
        (op_code, metadata, tuple(values), _ctx_ptr(tensors[0]._device)),
        arg_dtypes=(DType.float32,),
        output_dtypes=(DType.float32,),
        flags={"INPLACE": True, "OP_CODE": op_code},
        keepalive=(tensors,),
    )
    return None


def fast_aten__foreach_mul__scalar(
    self: Sequence[torch.Tensor], scalar: AtenScalar
) -> object:
    """Batched homogeneous FP32 in-place multiply by one host scalar."""
    value = _foreach_scalar_value(scalar)
    if value is None:
        return NOT_HANDLED
    return _foreach_scalar_inplace(self, [value] * len(self), _FOREACH_EW_MUL)


def _fast__foreach_add__scalar_metal(
    self: Sequence[torch.Tensor], scalar: AtenScalar
) -> object:
    """Batched homogeneous FP32 in-place add of one host scalar."""
    value = _foreach_scalar_value(scalar)
    if value is None:
        return NOT_HANDLED
    return _foreach_scalar_inplace(self, [value] * len(self), _FOREACH_EW_ADD)


def fast_aten__foreach_add__scalar(
    self: Sequence[torch.Tensor], scalar: AtenScalar
) -> object:
    """Batched in-place `t += scalar`: Metal 8-slot path first, then the
    generic single-launch path (measured on gfx942), else NOT_HANDLED."""
    result = _fast__foreach_add__scalar_metal(self, scalar)
    if result is NOT_HANDLED:
        result = _fast__foreach_add__scalar_generic(self, scalar)
    return result


def fast_aten__foreach_div__scalarlist(
    self: Sequence[torch.Tensor], scalars: Sequence[AtenScalar]
) -> object:
    """Batched homogeneous FP32 in-place divide by one scalar per tensor."""
    values = _foreach_scalar_values(scalars, len(self))
    if values is None:
        return NOT_HANDLED
    return _foreach_scalar_inplace(self, values, _FOREACH_EW_DIV)


def fast_aten__foreach_lerp__scalar(
    self: Sequence[torch.Tensor], tensors1: Sequence[torch.Tensor], weight: AtenScalar
) -> object:
    """Batched homogeneous FP32 in-place scalar lerp toward `tensors1`.

    Mirrors `fast_aten_lerp`: the weight is narrowed to float32 first and
    selects ATen's numerically stable branch on the host, so the batched
    kernel computes exactly what the sequential composition computes.
    """
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        return NOT_HANDLED
    lists = _foreach_metal_f32(self, tensors1)
    if lists is None:
        return NOT_HANDLED
    tensors, ends = lists
    if _foreach_mutation_hazard(tensors, (ends,)):
        return NOT_HANDLED
    try:
        narrowed_weight = struct.unpack("=f", struct.pack("=f", weight))[0]
    except (OverflowError, struct.error) as exc:
        raise RuntimeError(
            "value cannot be converted to type float without overflow"
        ) from exc
    one_minus_weight = struct.unpack("=f", struct.pack("=f", 1.0 - narrowed_weight))[0]
    metadata = tuple(
        value
        for tensor, end in zip(tensors, ends, strict=True)
        for value in (tensor._ptr, end._ptr, tensor._numel)
    )
    _call_mojo(
        _OptimizerExtension,
        "ForeachLerpScalar",
        (
            metadata,
            narrowed_weight,
            one_minus_weight,
            int(abs(narrowed_weight) < 0.5),
            _ctx_ptr(tensors[0]._device),
        ),
        arg_dtypes=(DType.float32, DType.float32),
        output_dtypes=(DType.float32,),
        flags={"INPLACE": True, "SMALL_WEIGHT": abs(narrowed_weight) < 0.5},
        keepalive=(tensors, ends),
    )
    return None


def _foreach_addc_inplace(
    self: Sequence[torch.Tensor],
    tensor1: Sequence[torch.Tensor],
    tensor2: Sequence[torch.Tensor],
    values: Sequence[float],
    op_code: int,
) -> object:
    """Shared launch path of the in-place foreach addcmul/addcdiv fast ops."""
    lists = _foreach_metal_f32(self, tensor1, tensor2)
    if lists is None:
        return NOT_HANDLED
    tensors, firsts, seconds = lists
    if _foreach_mutation_hazard(tensors, (firsts, seconds)):
        return NOT_HANDLED
    metadata = tuple(
        value
        for tensor, first, second in zip(tensors, firsts, seconds, strict=True)
        for value in (tensor._ptr, first._ptr, second._ptr, tensor._numel)
    )
    _call_mojo(
        _OptimizerExtension,
        "ForeachAddcOp",
        (op_code, metadata, tuple(values), _ctx_ptr(tensors[0]._device)),
        arg_dtypes=(DType.float32, DType.float32, DType.float32),
        output_dtypes=(DType.float32,),
        flags={"INPLACE": True, "OP_CODE": op_code},
        keepalive=(tensors, firsts, seconds),
    )
    return None


def fast_aten__foreach_addcmul__scalar(
    self: Sequence[torch.Tensor],
    tensor1: Sequence[torch.Tensor],
    tensor2: Sequence[torch.Tensor],
    value: AtenScalar = 1,
) -> object:
    """Batched homogeneous FP32 in-place self += value * (t1 * t2)."""
    scalar = _foreach_scalar_value(value)
    if scalar is None:
        return NOT_HANDLED
    return _foreach_addc_inplace(
        self, tensor1, tensor2, [scalar] * len(self), _FOREACH_EW_ADDCMUL
    )


def fast_aten__foreach_addcdiv__scalarlist(
    self: Sequence[torch.Tensor],
    tensor1: Sequence[torch.Tensor],
    tensor2: Sequence[torch.Tensor],
    scalars: Sequence[AtenScalar],
) -> object:
    """Batched homogeneous FP32 in-place self += s[i] * (t1 / t2)."""
    values = _foreach_scalar_values(scalars, len(self))
    if values is None:
        return NOT_HANDLED
    return _foreach_addc_inplace(self, tensor1, tensor2, values, _FOREACH_EW_ADDCDIV)


def fast_aten__foreach_sqrt(self: Sequence[torch.Tensor]) -> object:
    """Batched homogeneous FP32 out-of-place elementwise square roots."""
    lists = _foreach_metal_f32(self)
    if lists is None:
        return NOT_HANDLED
    (tensors,) = lists
    outputs = [
        _alloc(tensor._shape, DType.float32, tensor._device) for tensor in tensors
    ]
    metadata = tuple(
        value
        for tensor, output in zip(tensors, outputs, strict=True)
        for value in (tensor._ptr, output._ptr, tensor._numel)
    )
    _call_mojo(
        _OptimizerExtension,
        "ForeachSqrt",
        (metadata, _ctx_ptr(tensors[0]._device)),
        arg_dtypes=(DType.float32,),
        output_dtypes=(DType.float32,),
        keepalive=(tensors, outputs),
    )
    return outputs


def _fused_adamw_scalar_tensor(value, name, device):
    """Validate an optional read-only scalar and return its device pointer."""
    if value is None:
        return 0
    tensor = _t(value)
    if (
        tensor is None
        or tensor._device != device
        or tensor._dtype != DType.float32
        or tensor._numel != 1
        or not tensor._is_contiguous
    ):
        raise RuntimeError(
            f"{name} must be a contiguous scalar float32 tensor on the "
            "same Mojo device as the parameters"
        )
    return tensor._ptr


def fast_aten__fused_adamw(
    parameters,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    """Runtime-dynamic, allocation-free fused FP32 AdamW TensorList route.

    Validation deliberately completes for every tensor before the first
    launch.  This preserves ATen's all-before-write failure behavior and is
    especially important for mutable TensorLists, where materializing a view
    or discovering a malformed later entry after launch would be observable.
    """
    tensor_count = len(parameters)
    expected_lengths = {
        "grads": len(grads),
        "exp_avgs": len(exp_avgs),
        "exp_avg_sqs": len(exp_avg_sqs),
        "state_steps": len(state_steps),
    }
    if amsgrad:
        expected_lengths["max_exp_avg_sqs"] = len(max_exp_avg_sqs)
    elif len(max_exp_avg_sqs) != 0:
        raise RuntimeError("max_exp_avg_sqs must be empty when amsgrad is False")
    if any(length != tensor_count for length in expected_lengths.values()):
        rendered = ", ".join(
            f"{name}={length}" for name, length in expected_lengths.items()
        )
        raise RuntimeError(
            "fused AdamW tensor lists must have the same length "
            f"(parameters={tensor_count}, {rendered})"
        )
    if tensor_count == 0:
        return None

    first = _t(parameters[0])
    if first is None:
        return NOT_HANDLED
    device = first._device
    mutable_groups = (parameters, grads, exp_avgs, exp_avg_sqs)
    if amsgrad:
        mutable_groups += (max_exp_avg_sqs,)

    # Inspect the backend's real metadata, not TensorImpl's compatibility
    # facade.  Mutable optimizer state cannot be materialized into temporaries.
    for index in range(tensor_count):
        parameter = _t(parameters[index])
        if parameter is None:
            return NOT_HANDLED
        shape = tuple(parameter._shape)
        for group in mutable_groups:
            tensor = _t(group[index])
            if (
                tensor is None
                or tensor._device != device
                or tensor._dtype != DType.float32
                or tensor._shape != shape
                or tensor._numel != parameter._numel
                or not tensor._is_contiguous
            ):
                raise RuntimeError(
                    "fused AdamW tensors must have the same dtype, device, "
                    "shape, and numel; contiguous float32 is "
                    f"required (invalid tensor index {index})"
                )
        step = _t(state_steps[index])
        if (
            step is None
            or step._device != device
            or step._dtype != DType.float32
            or step._numel != 1
            or not step._is_contiguous
        ):
            raise RuntimeError(
                "fused AdamW state_steps must be contiguous scalar float32 "
                f"tensors on the parameter device (invalid index {index})"
            )

    lr_ptr = 0
    if isinstance(lr, TorchMojoTensor):
        lr_ptr = _fused_adamw_scalar_tensor(lr, "lr", device)
        lr_scalar = 0.0
    elif isinstance(lr, torch.Tensor):
        if lr.device.type != "cpu" or lr.numel() != 1:
            raise RuntimeError("tensor lr must be a scalar CPU or Mojo tensor")
        lr_scalar = float(lr.item())
    elif isinstance(lr, int | float) and not isinstance(lr, bool):
        lr_scalar = float(lr)
    else:
        return NOT_HANDLED

    grad_scale_ptr = _fused_adamw_scalar_tensor(grad_scale, "grad_scale", device)
    found_inf_ptr = _fused_adamw_scalar_tensor(found_inf, "found_inf", device)
    metadata = tuple(
        value
        for index in range(tensor_count)
        for value in (
            parameters[index]._ptr,
            grads[index]._ptr,
            exp_avgs[index]._ptr,
            exp_avg_sqs[index]._ptr,
            max_exp_avg_sqs[index]._ptr if amsgrad else 0,
            state_steps[index]._ptr,
            parameters[index]._numel,
        )
    )
    if len(metadata) != tensor_count * _FUSED_ADAMW_RECORD_FIELDS:
        raise AssertionError("invalid fused AdamW metadata packing")

    _call_mojo(
        _OptimizerExtension,
        "FusedAdamW",
        (
            metadata,
            (lr_scalar, float(beta1), float(beta2), float(weight_decay), float(eps)),
            0,  # homogeneous FP32 parameters, gradients, and optimizer state
            int(bool(amsgrad)) | (int(bool(maximize)) << 1),
            lr_ptr,
            grad_scale_ptr,
            found_inf_ptr,
            _ctx_ptr(device),
        ),
        arg_dtypes=(
            DType.float32,
            DType.float32,
            DType.float32,
            DType.float32,
            DType.float32,
        ),
        output_dtypes=(DType.float32,),
        flags={
            "AMSGRAD": bool(amsgrad),
            "MAXIMIZE": bool(maximize),
            "TENSOR_LR": bool(lr_ptr),
            "GRAD_SCALE": bool(grad_scale_ptr),
            "FOUND_INF": bool(found_inf_ptr),
        },
        keepalive=(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            grad_scale,
            found_inf,
        ),
    )
    return None


# ---------------------------------------------------------------------------
# TensorSpec fast paths (docs/tensor_spec_design.md). A spec is the
# Mojo-side descriptor of a tensor's layout; spec ops live next to their
# kernels (binary/comparison in logic_ops, relu/batch_norm still in
# tensor_holder) and do checks, broadcast, output alloc and kernel launch
# in one boundary call, raising a real NotImplementedError when the inputs
# don't qualify — the callers below treat unsupported-input exceptions as
# "take the classic path", while allocator OOMs propagate. make_spec (the
# constructor) stays in tensor_holder.
# ---------------------------------------------------------------------------


def _spec_of(t):
    """t's cached Mojo TensorSpec, built on first use."""
    spec = t.__dict__.get("_spec")
    if spec is None:
        spec = eager_kernels.tensor_holder.make_spec(
            t._ptr,
            len(t._shape),
            _pad8(t._shape, 1),
            _pad8(t._strides, 0),
            t._offset,
            t._dtype.value,
            t._itemsize,
            t._numel,
            1 if t._is_contiguous else 0,
            _ctx_ptr(t._device),
        )
        t._spec = spec
    return spec


class _FillSpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    MOJO_FILE: ClassVar[Path] = _ElementwiseExtension.MOJO_FILE

    @classmethod
    def make_defines(
        cls, shape: Sequence[int], value: float, dtype: DType, device: Device
    ) -> dict[str, bool | int | str]:
        return {"OP": "FillSpec", "DTYPE_OUT": dtype.name}

    @classmethod
    def make_canonical_defines(
        cls, shape: Sequence[int], value: float, dtype: DType, device: Device
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines("FillSpec", (), (dtype,), ())

    @classmethod
    def expected_output_specs(
        cls, shape: Sequence[int], value: float, dtype: DType, device: Device
    ) -> _TensorOutputSpec:
        return _TensorOutputSpec(tuple(shape), dtype, device)

    @classmethod
    def extension_args(
        cls,
        out: TorchMojoTensor,
        shape: Sequence[int],
        value: float,
        dtype: DType,
        device: Device,
    ) -> tuple[object, ...]:
        return (value, _spec_of(out))


class _CastSpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    MOJO_FILE: ClassVar[Path] = _DataMovementExtension.MOJO_FILE

    @classmethod
    def make_defines(
        cls, tensor: MojoTensorLike, dtype: DType
    ) -> dict[str, bool | int | str]:
        return {
            "OP": "CastSpec",
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT": dtype.name,
        }

    @classmethod
    def make_canonical_defines(
        cls, tensor: MojoTensorLike, dtype: DType
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            "CastSpec", (tensor._dtype,), (dtype,), ()
        )

    @classmethod
    def expected_output_specs(
        cls, tensor: MojoTensorLike, dtype: DType
    ) -> _TensorOutputSpec:
        return _TensorOutputSpec(tuple(tensor._shape), dtype, tensor._device)

    @classmethod
    def extension_args(
        cls, out: TorchMojoTensor, tensor: MojoTensorLike, dtype: DType
    ) -> tuple[object, ...]:
        return (_spec_of(tensor), dtype.value, _spec_of(out))


class _BinarySpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    MOJO_FILE: ClassVar[Path] = _LogicExtension.MOJO_FILE

    @classmethod
    def make_defines(
        cls, op: str, lhs: MojoTensorLike, rhs: MojoTensorLike, out_dtype: DType
    ) -> dict[str, bool | int | str]:
        return {
            "OP": op,
            "DTYPE_ARG_0": lhs._dtype.name,
            "DTYPE_ARG_1": rhs._dtype.name,
            "DTYPE_OUT": out_dtype.name,
        }

    @classmethod
    def make_canonical_defines(
        cls, op: str, lhs: MojoTensorLike, rhs: MojoTensorLike, out_dtype: DType
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            op, (lhs._dtype, rhs._dtype), (out_dtype,), ()
        )

    @classmethod
    def expected_output_specs(
        cls, op: str, lhs: MojoTensorLike, rhs: MojoTensorLike, out_dtype: DType
    ) -> _TensorOutputSpec:
        lhs_shape = tuple(lhs._shape)
        rhs_shape = tuple(rhs._shape)
        if lhs_shape != rhs_shape:
            # Equal shapes dominate (residual adds, grad accumulation) and
            # torch.broadcast_shapes costs ~7 µs per call.
            lhs_shape = tuple(torch.broadcast_shapes(lhs_shape, rhs_shape))
        return _TensorOutputSpec(lhs_shape, out_dtype, lhs._device)

    @classmethod
    def extension_args(
        cls,
        out: TorchMojoTensor,
        op: str,
        lhs: MojoTensorLike,
        rhs: MojoTensorLike,
        out_dtype: DType,
    ) -> tuple[object, ...]:
        return (_spec_of(lhs), _spec_of(rhs), _spec_of(out))


class _UnarySpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    @classmethod
    def make_defines(
        cls, op: str, tensor: MojoTensorLike, out_dtype: DType
    ) -> dict[str, bool | int | str]:
        return {
            "OP": op,
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT": out_dtype.name,
        }

    @classmethod
    def make_canonical_defines(
        cls, op: str, tensor: MojoTensorLike, out_dtype: DType
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            op, (tensor._dtype,), (out_dtype,), ()
        )

    @classmethod
    def expected_output_specs(
        cls, op: str, tensor: MojoTensorLike, out_dtype: DType
    ) -> _TensorOutputSpec:
        return _TensorOutputSpec(tuple(tensor._shape), out_dtype, tensor._device)

    @classmethod
    def extension_args(
        cls, out: TorchMojoTensor, op: str, tensor: MojoTensorLike, out_dtype: DType
    ) -> tuple[object, ...]:
        return (_spec_of(tensor), _spec_of(out))


class _ElementwiseUnarySpecExtension(_UnarySpecExtension):
    MOJO_FILE: ClassVar[Path] = _ElementwiseExtension.MOJO_FILE


class _ReductionUnarySpecExtension(_UnarySpecExtension):
    MOJO_FILE: ClassVar[Path] = _ReductionExtension.MOJO_FILE


class _NNUnarySpecExtension(_UnarySpecExtension):
    MOJO_FILE: ClassVar[Path] = _NNExtension.MOJO_FILE


_UNARY_SPEC_EXTENSIONS = {
    "elementwise_ops": _ElementwiseUnarySpecExtension,
    "reduction_ops": _ReductionUnarySpecExtension,
    "nn_ops": _NNUnarySpecExtension,
}


# Mirrors logic_ops.mojo's SPEC_BCAST_DTYPES — the Mojo-side source of
# truth for which dtypes the Into spec kernels are compiled for.
_SPEC_INTO_DTYPES = frozenset(
    {
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    }
)
_SPEC_CMP_NAMES = frozenset(
    {
        "EqSpec",
        "NeSpec",
        "LtSpec",
        "LeSpec",
        "GtSpec",
        "GeSpec",
        "LogicalAndSpec",
        "LogicalXorSpec",
    }
)
_SPEC_FLOAT_ONLY_NAMES = frozenset({"DivSpec", "PowSpec"})
_SPEC_INT_ONLY_NAMES = frozenset({"BitwiseAndSpec", "BitwiseOrSpec", "BitwiseXorSpec"})
_SPEC_BOOL_OK_NAMES = _SPEC_CMP_NAMES | frozenset(
    {"MulSpec", "BitwiseAndSpec", "BitwiseOrSpec", "BitwiseXorSpec"}
)


def _binary_promotion(
    a_dtype: DType, b_dtype: DType
) -> tuple[bool, bool, DType] | None:
    """torch's promotion for a binary pair as (cast lhs?, cast rhs?, dtype).

    The only pairs the eager loops hit, and the single source of truth for
    both spec-binary paths (queued and drain-and-execute). None means the
    pair has no supported promotion, i.e. decline the op.
    """
    if a_dtype == b_dtype:
        return False, False, a_dtype
    if a_dtype == DType.bool and b_dtype in _CAST_DTYPES:
        return True, False, b_dtype
    if b_dtype == DType.bool and a_dtype in _CAST_DTYPES:
        return False, True, a_dtype
    if a_dtype == DType.int32 and b_dtype == DType.int64:
        return True, False, DType.int64
    if a_dtype == DType.int64 and b_dtype == DType.int32:
        return False, True, DType.int64
    if a_dtype == DType.float32 and b_dtype in (DType.float16, DType.bfloat16):
        return False, True, DType.float32
    if b_dtype == DType.float32 and a_dtype in (DType.float16, DType.bfloat16):
        return True, False, DType.float32
    if {a_dtype, b_dtype} == {DType.float16, DType.bfloat16}:
        return True, True, DType.float32
    return None


def _try_spec_binary_into(
    spec_fn_name: str, lhs: object, rhs: object, out_dtype: DType | None
) -> TorchMojoTensor | None:
    """Call-queue mode: pre-allocate the output in Python and queue the
    Into launch — allocation never forces a drain, the launch is
    fire-and-forget. Returns the output wrapper, or None to fall back to
    the legacy (drain + synchronous) spec path. Eligibility is replicated
    here CONSERVATIVELY: a queued launch cannot fall back, so anything
    uncertain declines."""
    a = _t(lhs)
    b = _t(rhs)
    if a is None and b is None:
        return None
    if a is not None and b is not None and a._device != b._device:
        return None
    anchor_t = a if a is not None else b
    device = anchor_t._device
    dtype = anchor_t._dtype

    if a is not None and b is not None:
        if len(a._shape) > 4 or len(b._shape) > 4:
            if tuple(a._shape) != tuple(b._shape):
                return None
            a = _tc(a)
            b = _tc(b)
        promotion = _binary_promotion(a._dtype, b._dtype)
        if promotion is None:
            return None
        # Same ladder as the legacy path; here the casts queue through the
        # Into cast (never a drain).
        cast_a, cast_b, dtype = promotion
        if cast_a:
            a = _cast_tensor(a, dtype)
        if cast_b:
            b = _cast_tensor(b, dtype)
    else:
        # One scalar operand: embed it as a queued 0-d fill.
        scalar = rhs if a is not None else lhs
        value = _scalar_embed(scalar, dtype)
        if value is None:
            return None
        fill = _submit_prepared_into(
            _FillSpecExtension.prepare((), value, dtype, device)
        )
        if a is not None:
            b = fill
        else:
            a = fill

    kdtype = DType.uint8 if dtype == DType.bool else dtype
    if kdtype not in _SPEC_INTO_DTYPES:
        return None
    if dtype == DType.bool and spec_fn_name not in _SPEC_BOOL_OK_NAMES:
        return None
    if spec_fn_name not in _SPEC_CMP_NAMES:
        if spec_fn_name in _SPEC_FLOAT_ONLY_NAMES and not kdtype in (
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.float64,
        ):
            return None
        if spec_fn_name in _SPEC_INT_ONLY_NAMES and kdtype in (
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.float64,
        ):
            return None
    if a._shape != b._shape:
        # Equal shapes (the residual/grad-sum hot path) skip torch's
        # broadcast machinery, which costs ~7 µs per probe.
        try:
            torch.broadcast_shapes(tuple(a._shape), tuple(b._shape))
        except RuntimeError:
            return None
    return _submit_prepared_into(
        _BinarySpecExtension.prepare(spec_fn_name, a, b, out_dtype or dtype)
    )


def _try_spec_binary(spec_fn_name, lhs, rhs, out_dtype=None):
    """Broadcast binary through a logic_ops spec op, or None.

    Python keeps what Python must do — scalar embedding, torch's promotion
    rules, dim-spec sanity — but the intermediates stay spec-to-spec: a
    scalar operand becomes a 0-d FillSpec result and a promoted operand a
    CastSpec result whose SPECS feed the binary entry directly. No wrapper
    is minted for them; their holders live in locals until the launch is
    enqueued (the stream-ordered free then lands after the kernel).
    rank>4 operands are pre-materialized (the spec's flat path needs
    contiguity there). `out_dtype` overrides the wrapper dtype for ops
    whose output differs (comparisons -> bool)."""
    if _call_queue.enabled():
        into = _try_spec_binary_into(spec_fn_name, lhs, rhs, out_dtype)
        if into is not None:
            return into
        # Ineligible for the queued Into form: the legacy call below drains
        # and runs synchronously (correct, just not overlapped).
    a = _t(lhs)
    b = _t(rhs)
    spec_a = spec_b = None
    keep_a = keep_b = None
    if a is not None and b is not None:
        if a._device != b._device:
            return None
        if len(a._shape) > 4 or len(b._shape) > 4:
            a = _tc(a)
            b = _tc(b)
        promotion = _binary_promotion(a._dtype, b._dtype)
        if promotion is None:
            return None
        # Same ladder as the queued path; here the casts drain and execute.
        cast_a, cast_b, dtype = promotion
        try:
            if cast_a:
                keep_a = _submit_prepared_into(
                    _CastSpecExtension.prepare(a, dtype), force_sync=True
                )
                spec_a = _spec_of(keep_a)
            if cast_b:
                keep_b = _submit_prepared_into(
                    _CastSpecExtension.prepare(b, dtype), force_sync=True
                )
                spec_b = _spec_of(keep_b)
        except Exception as exc:
            _raise_if_device_oom(exc)
            return None
    elif a is not None:
        device = a._device
        dtype = a._dtype
        value = _scalar_embed(rhs, dtype)
        if value is None:
            return None
        try:
            keep_b = _submit_prepared_into(
                _FillSpecExtension.prepare((), value, dtype, device), force_sync=True
            )
            spec_b = _spec_of(keep_b)
        except Exception as exc:
            _raise_if_device_oom(exc)
            return None
    elif b is not None:
        # Scalar-first calls, e.g. rsub-style `1 - tensor`.
        device = b._device
        dtype = b._dtype
        value = _scalar_embed(lhs, dtype)
        if value is None:
            return None
        try:
            keep_a = _submit_prepared_into(
                _FillSpecExtension.prepare((), value, dtype, device), force_sync=True
            )
            spec_a = _spec_of(keep_a)
        except Exception as exc:
            _raise_if_device_oom(exc)
            return None
    else:
        return None
    try:
        result = _submit_prepared_into(
            _BinarySpecExtension.prepare(
                spec_fn_name,
                keep_a if keep_a is not None else a,
                keep_b if keep_b is not None else b,
                out_dtype or dtype,
            ),
            force_sync=True,
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None
    _ = spec_a, spec_b  # keep explicit spec construction covered above
    return result


def _try_spec_add_f32_bf16(lhs, rhs):
    """One-launch contiguous FP32 + BF16 -> FP32 add, or None.

    The general binary promotion path materializes its lower-precision input.
    This hot residual-add route instead leaves both inputs in their original
    storage and lets the Mojo kernel widen BF16 values in registers.  Shape,
    dtype and contiguity checks are runtime metadata checks, so one compiled
    kernel handles every eligible shape without recompilation.
    """
    a = _t(lhs)
    b = _t(rhs)
    if (
        a is None
        or b is None
        or a._device != b._device
        or a._device.api == "cpu"
        or not a._is_contiguous
        or not b._is_contiguous
        or a._shape != b._shape
        or not (
            (a._dtype == DType.float32 and b._dtype == DType.bfloat16)
            or (a._dtype == DType.bfloat16 and b._dtype == DType.float32)
        )
    ):
        return None
    try:
        # The CPU device was already rejected above, so the metadata is always
        # queue-eligible; the submit helper runs it synchronously by itself
        # whenever the queue is disabled.
        return _submit_prepared_into(
            _BinarySpecExtension.prepare("AddF32Bf16Spec", a, b, DType.float32)
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None


_SPEC_FLOAT_DTYPES = frozenset(
    {DType.float32, DType.float16, DType.bfloat16, DType.float64}
)
_SPEC_UNARY_DIRECT_NAMES = frozenset({"ReluSpec", "AbsSpec", "NegSpec", "SignSpec"})
# Ops eligible for the queued Into form via _try_spec_unary, with the
# dtype rule the Mojo prologue enforces (a queued launch cannot fall back).
_SPEC_ROWRED_INTO = frozenset(
    {DType.float32, DType.float16, DType.bfloat16, DType.int64, DType.int32}
)
_SPEC_ANYALL_INTO = frozenset(
    {
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
        DType.bool,
    }
)
# (module, spec name) -> (operand dtype rule, output dtype override or None)
_SPEC_REDUCE_INTO = {
    ("reduction_ops", "SumSpec"): (_SPEC_ROWRED_INTO, None),
    ("reduction_ops", "AmaxSpec"): (_SPEC_ROWRED_INTO, None),
    ("reduction_ops", "AminSpec"): (_SPEC_ROWRED_INTO, None),
    ("reduction_ops", "ArgminSpec"): (_SPEC_ROWRED_INTO, DType.int64),
    ("reduction_ops", "VarSpec"): (_SPEC_FLOAT_DTYPES, None),
    ("reduction_ops", "AllSpec"): (_SPEC_ANYALL_INTO, DType.bool),
    ("reduction_ops", "AnySpec"): (_SPEC_ANYALL_INTO, DType.bool),
    ("nn_ops", "MeanSpec"): (_SPEC_FLOAT_DTYPES, None),
    ("nn_ops", "MaxSpec"): (_SPEC_ROWRED_INTO, None),
    ("nn_ops", "ArgmaxSpec"): (_SPEC_ROWRED_INTO, DType.int64),
}


def _reduced_shape(shape: tuple, rdims: object, keepdim: bool) -> tuple:
    rd = set(rdims)
    out = []
    for i, s in enumerate(shape):
        if i in rd:
            if keepdim:
                out.append(1)
        else:
            out.append(s)
    return tuple(out)


class _ReductionSpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    @classmethod
    def make_defines(
        cls,
        op: str,
        tensor: MojoTensorLike,
        rdims: tuple[int, ...],
        keepdim: bool,
        extra: tuple[object, ...],
        out_dtype: DType,
    ) -> dict[str, bool | int | str]:
        return {
            "OP": op,
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT": out_dtype.name,
        }

    @classmethod
    def make_canonical_defines(
        cls,
        op: str,
        tensor: MojoTensorLike,
        rdims: tuple[int, ...],
        keepdim: bool,
        extra: tuple[object, ...],
        out_dtype: DType,
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            op, (tensor._dtype,), (out_dtype,), ()
        )

    @classmethod
    def expected_output_specs(
        cls,
        op: str,
        tensor: MojoTensorLike,
        rdims: tuple[int, ...],
        keepdim: bool,
        extra: tuple[object, ...],
        out_dtype: DType,
    ) -> _TensorOutputSpec:
        shape = _reduced_shape(tuple(tensor._shape), rdims, keepdim)
        return _TensorOutputSpec(shape, out_dtype, tensor._device)

    @classmethod
    def extension_args(
        cls,
        out: TorchMojoTensor,
        op: str,
        tensor: MojoTensorLike,
        rdims: tuple[int, ...],
        keepdim: bool,
        extra: tuple[object, ...],
        out_dtype: DType,
    ) -> tuple[object, ...]:
        return (_spec_of(tensor), rdims, 1 if keepdim else 0, *extra, _spec_of(out))


class _ReductionOpsSpecExtension(_ReductionSpecExtension):
    MOJO_FILE: ClassVar[Path] = _ReductionExtension.MOJO_FILE


class _NNReductionSpecExtension(_ReductionSpecExtension):
    MOJO_FILE: ClassVar[Path] = _NNExtension.MOJO_FILE


_REDUCTION_SPEC_EXTENSIONS = {
    "reduction_ops": _ReductionOpsSpecExtension,
    "nn_ops": _NNReductionSpecExtension,
}


class _MinDimSpecExtension(
    eager_kernels.MojoExtension[
        tuple[_TensorOutputSpec, _TensorOutputSpec],
        tuple[TorchMojoTensor, TorchMojoTensor],
    ]
):
    MOJO_FILE: ClassVar[Path] = _ReductionExtension.MOJO_FILE

    @classmethod
    def make_defines(
        cls, tensor: MojoTensorLike, dim: int, keepdim: bool
    ) -> dict[str, bool | int | str]:
        return {
            "OP": "MinDimSpec",
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT_0": tensor._dtype.name,
            "DTYPE_OUT_1": DType.int64.name,
        }

    @classmethod
    def make_canonical_defines(
        cls, tensor: MojoTensorLike, dim: int, keepdim: bool
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            "MinDimSpec", (tensor._dtype,), (tensor._dtype, DType.int64), ()
        )

    @classmethod
    def expected_output_specs(
        cls, tensor: MojoTensorLike, dim: int, keepdim: bool
    ) -> tuple[_TensorOutputSpec, _TensorOutputSpec]:
        shape = _reduced_shape(tuple(tensor._shape), (dim,), keepdim)
        return (
            _TensorOutputSpec(shape, tensor._dtype, tensor._device),
            _TensorOutputSpec(shape, DType.int64, tensor._device),
        )

    @classmethod
    def allocate_outputs(
        cls, output_specs: tuple[_TensorOutputSpec, _TensorOutputSpec]
    ) -> tuple[TorchMojoTensor, TorchMojoTensor]:
        return (
            _allocate_output_spec(output_specs[0]),
            _allocate_output_spec(output_specs[1]),
        )

    @classmethod
    def extension_args(
        cls,
        outputs: tuple[TorchMojoTensor, TorchMojoTensor],
        tensor: MojoTensorLike,
        dim: int,
        keepdim: bool,
    ) -> tuple[object, ...]:
        return (
            _spec_of(tensor),
            (dim,),
            1 if keepdim else 0,
            _spec_of(outputs[0]),
            _spec_of(outputs[1]),
        )


def _try_spec_min_dim(
    a: TorchMojoTensor, dim: int, keepdim: bool
) -> tuple[TorchMojoTensor, TorchMojoTensor] | None:
    """min.dim through the two-output spec op, or None.

    Same shape as every sibling `_try_spec_*` route: queue-eligible metadata
    submits into the queue, everything else drains and runs synchronously.
    """
    ok = _call_queue.enabled() and a._numel > 0 and a._dtype in _SPEC_ROWRED_INTO
    original_shape = tuple(a._shape)
    a, ready_dims = _reduce_ready_operand(a, (dim,))
    try:
        outputs = _submit_prepared_into(
            _MinDimSpecExtension.prepare(a, ready_dims[0], keepdim), force_sync=not ok
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None
    if keepdim and ready_dims != (dim,):
        shape = _reduced_shape(original_shape, (dim,), True)
        outputs = (
            _reduce_keepdim_shape(outputs[0], shape),
            _reduce_keepdim_shape(outputs[1], shape),
        )
    return outputs


def _try_spec_unary(spec_fn_name, x, out_dtype=None, module_name="elementwise_ops"):
    """Contiguous unary through a spec op, or None.

    `out_dtype` overrides the wrapper dtype for the bool-output ops
    (isnan / logical_not)."""
    a = _t(x)
    if a is None:
        return None
    if not a._is_contiguous:
        # Materialize here, through the queued strided copy: metered by the
        # budget and covered by the allocation retry. The Mojo bridges no
        # longer scratch-copy strided operands.
        a = a._contig()
    kdtype = DType.uint8 if a._dtype == DType.bool else a._dtype
    if not _call_queue.enabled():
        ok = False
    elif out_dtype == DType.bool and module_name == "elementwise_ops":
        ok = kdtype in _SPEC_INTO_DTYPES
    elif module_name == "elementwise_ops":
        ok = a._dtype in (
            _SPEC_INTO_DTYPES
            if spec_fn_name in _SPEC_UNARY_DIRECT_NAMES
            else _SPEC_FLOAT_DTYPES
        )
    elif spec_fn_name in ("LogSoftmaxSpec", "SoftmaxSpec"):
        ok = a._dtype in _SPEC_FLOAT_DTYPES and len(a._shape) >= 1 and a._numel > 0
    else:
        ok = False  # e.g. CumsumSpec: constraints not mirrored, stay sync
    try:
        return _submit_prepared_into(
            _UNARY_SPEC_EXTENSIONS[module_name].prepare(
                spec_fn_name, a, out_dtype or a._dtype
            ),
            force_sync=not ok,
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None


def _try_spec_reduce(
    spec_fn_name, a, rdims, keepdim, *extra, out_dtype=None, module_name="reduction_ops"
):
    """Trailing-dims reduction through a spec op, or None. `a` is already a
    TorchMojoTensor (dtype promotion happened upstream); the spec op raises
    on non-trailing dims / strided input and the classic path takes over."""
    dims = tuple(rdims)
    extras = tuple(extra)
    ok = False
    odtype = out_dtype or a._dtype
    rank = len(a._shape)
    if _call_queue.enabled():
        rule = _SPEC_REDUCE_INTO.get((module_name, spec_fn_name))
        if (
            rule is not None
            and a._numel > 0
            and a._dtype in rule[0]
            and dims
            and len(set(dims)) == len(dims)
            and all(isinstance(d, int) and 0 <= d < rank for d in dims)
        ):
            ok = True
            odtype = rule[1] or odtype
    if not (
        dims
        and len(set(dims)) == len(dims)
        and all(isinstance(d, int) and 0 <= d < rank for d in dims)
    ):
        return None  # the bridge would reject the dim spec anyway
    original_shape = tuple(a._shape)
    if _sum_middle_direct_ok(spec_fn_name, module_name, a, dims):
        ready_dims = dims  # zero-copy direct kernel reads the source in place
    else:
        a, ready_dims = _reduce_ready_operand(a, dims)
    try:
        result = _submit_prepared_into(
            _REDUCTION_SPEC_EXTENSIONS[module_name].prepare(
                spec_fn_name, a, ready_dims, keepdim, extras, odtype
            ),
            force_sync=not ok,
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None
    if keepdim and ready_dims != dims:
        result = _reduce_keepdim_shape(
            result, _reduced_shape(original_shape, dims, True)
        )
    return result


def _raise_if_device_oom(exc: BaseException) -> None:
    """Keep TensorSpec fallbacks from disguising allocator exhaustion.

    Mojo TensorSpec dispatch reports both unsupported metadata and runtime
    launch/allocation failures as ``NotImplementedError``. Unsupported
    metadata should retain the classic fallback, but retrying after a device
    OOM only replaces the useful allocator error with a misleading
    ``aten::<op> is not supported`` message.
    """
    if eager_kernels.is_device_oom(exc):
        raise torch.OutOfMemoryError(str(exc)) from exc


# A queued launch fails inside `call_queue.drain()`, far from the `_call_mojo`
# try/except that translates allocator exhaustion into `torch.OutOfMemoryError`.
# Give the queue the same translation so the exception TYPE a caller catches
# does not depend on whether the launch happened to be deferred.
_call_queue.set_error_translator(_raise_if_device_oom)


def _spec_matmul_out_shape(
    spec_fn_name: str, ts: list, transpose_b: int, require_contiguous: bool = True
) -> tuple | None:
    """Output shape for a queueable matmul spec launch, or None when any
    Mojo-side check might fail (a queued launch cannot fall back).

    ``require_contiguous=False`` answers the *different* question "would this
    launch be eligible if every operand were materialized contiguous?".  It is
    only used to decide whether a copy is worth making; every actual launch is
    prepared with the default, so no strided operand can reach the kernel.
    """
    a = ts[0]
    b = ts[1]
    if a._dtype != b._dtype or a._dtype not in _SPEC_FLOAT_DTYPES:
        return None
    if a._device != b._device:
        return None
    if require_contiguous and not all(t._is_contiguous for t in ts):
        return None
    if spec_fn_name == "BmmSpec":
        if len(a._shape) != 3 or len(b._shape) != 3:
            return None
        batch, m, k = a._shape
        if b._shape[0] != batch:
            return None
        n, kb = (
            (b._shape[1], b._shape[2]) if transpose_b else (b._shape[2], b._shape[1])
        )
        if kb != k or 0 in (batch, m, n, k):
            return None
        return (batch, m, n)
    if len(a._shape) < 2 or len(b._shape) != 2:
        return None
    k = a._shape[-1]
    n, kb = (b._shape[0], b._shape[1]) if transpose_b else (b._shape[1], b._shape[0])
    if kb != k or k == 0 or n == 0 or a._numel == 0:
        return None
    if spec_fn_name == "MatmulBiasSpec":
        bias = ts[2]
        if bias._dtype != a._dtype or len(bias._shape) != 1 or bias._shape[0] != n:
            return None
    elif spec_fn_name != "MatmulSpec":
        return None
    return (*a._shape[:-1], n)


class _MatmulSpecExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, TorchMojoTensor]
):
    MOJO_FILE: ClassVar[Path] = _MatmulExtension.MOJO_FILE

    @classmethod
    def make_defines(
        cls, op: str, tensors: tuple[MojoTensorLike, ...], transpose_b: int
    ) -> dict[str, bool | int | str]:
        defines = {
            "OP": op,
            "DTYPE_OUT": tensors[0]._dtype.name,
            "TRANSPOSE_B": bool(transpose_b),
        }
        defines.update(
            (f"DTYPE_ARG_{index}", tensor._dtype.name)
            for index, tensor in enumerate(tensors)
        )
        return defines

    @classmethod
    def make_canonical_defines(
        cls, op: str, tensors: tuple[MojoTensorLike, ...], transpose_b: int
    ) -> "eager_kernels.CanonicalDefines":
        return eager_kernels._canonical_call_defines(
            op,
            tuple(t._dtype for t in tensors),
            (tensors[0]._dtype,),
            (("TRANSPOSE_B", bool(transpose_b)),),
        )

    @classmethod
    def expected_output_specs(
        cls, op: str, tensors: tuple[MojoTensorLike, ...], transpose_b: int
    ) -> _TensorOutputSpec:
        shape = _spec_matmul_out_shape(op, list(tensors), transpose_b)
        if shape is None:
            raise ValueError("matmul metadata is not eligible for the spec path")
        return _TensorOutputSpec(tuple(shape), tensors[0]._dtype, tensors[0]._device)

    @classmethod
    def extension_args(
        cls,
        out: TorchMojoTensor,
        op: str,
        tensors: tuple[MojoTensorLike, ...],
        transpose_b: int,
    ) -> tuple[object, ...]:
        return (*(_spec_of(tensor) for tensor in tensors), transpose_b, _spec_of(out))


def _submit_spec_matmul(
    spec_fn_name: str, ts: tuple[MojoTensorLike, ...], transpose_b: int
) -> torch.Tensor | None:
    """One matmul spec launch over already-typed operands, or None.

    Declines (ineligible metadata, a Mojo-side refusal) come back as None; a
    device allocator failure still propagates as ``torch.OutOfMemoryError``.
    """
    try:
        # No force_sync needed: _submit_prepared_into already executes
        # synchronously whenever the queue is disabled.
        return _submit_prepared_into(
            _MatmulSpecExtension.prepare(spec_fn_name, ts, transpose_b)
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None


def _try_spec_matmul(spec_fn_name, tensors, transpose_b):
    """Matmul-family spec op over already-typed operands, or None.

    The spec kernels read row-major memory, so a strided operand is declined.
    This is the last step of every matmul-family entry point (``fast_aten_mm``,
    ``fast_aten_addmm``, ``fast_aten_linear``, ``fast_aten_bmm``,
    ``_fast_aten_bmm_transpose_b``), so the copy backstop below covers all of
    them at once and only ever runs after their bridges have declined.
    """
    ts = tuple(_t(x) for x in tensors)
    if any(t is None for t in ts):
        return None
    out = _submit_spec_matmul(spec_fn_name, ts, transpose_b)
    if out is not None or all(t._is_contiguous for t in ts):
        return out

    # Last-resort correctness backstop, not the intended path: it costs a
    # materialization of the offending operand.  It runs only when every
    # layout-capable GEMM bridge has already declined and the spec path's ONLY
    # objection is that an operand is a view -- asking `_spec_matmul_out_shape`
    # again with the contiguity check removed proves nothing else is wrong.
    # Without it a strided operand is a hard NOT_HANDLED, and inside a native
    # autograd node NOT_HANDLED becomes a NotImplementedError raised under
    # `Engine::evaluate_function`, which aborts the process instead of raising.
    # A plain fp32 `nn.Linear` backward hits exactly that, because
    # `fast_aten_linear_backward` forms the weight gradient as
    # `mm(grad.transpose(0, 1), input)`.
    # Two things would remove this copy: an fp32 GEMM that reads a strided A
    # (the BF16 bridge already does, which is why bf16 autocast never hit the
    # abort), or the TF32 bridge not being gated off at the default
    # `float32_matmul_precision == "highest"`.  Do NOT take the second route by
    # relaxing that gate: TF32 drops mantissa bits, and "highest" is the user
    # asking for full fp32 -- see the note on the gate in `_try_tf32_gemm`.
    if (
        _spec_matmul_out_shape(
            spec_fn_name, list(ts), transpose_b, require_contiguous=False
        )
        is None
    ):
        return None
    contiguous = tuple(_tc(t) for t in ts)
    if any(t is None for t in contiguous):
        return None
    return _submit_spec_matmul(spec_fn_name, contiguous, transpose_b)


def _try_spec_scalar(spec_fn_name, x, scalar):
    """Contiguous tensor-with-float-scalar through a spec op, or None."""
    if not isinstance(scalar, int | float) or isinstance(scalar, bool):
        return None
    a = _t(x)
    if a is None or a._dtype not in _SPEC_FLOAT_DTYPES:
        return None
    if not a._is_contiguous:
        a = a._contig()  # queued materialize: metered + covered by the retry
    out = _alloc(a._shape, a._dtype, a._device)
    try:
        _call_mojo(
            _ElementwiseExtension,
            spec_fn_name,
            (_spec_of(a), float(scalar), _spec_of(out)),
            arg_dtypes=(a._dtype,),
            output_dtypes=(out._dtype,),
            keepalive=(a, out),
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None
    return out


def _try_spec_scalar_inplace(spec_fn_name: str, x, scalar) -> bool:
    """`x op= scalar` in place through a spec op. True when it ran.

    The functional spec plus `_copy_into` costs an allocation and a
    device-to-device copy; on a one-element tensor that copy is the whole cost
    (nanoGPT's AdamW bumps 75 scalar step counters per step).
    """
    if not isinstance(scalar, int | float) or isinstance(scalar, bool):
        return False
    a = _t(x)
    if a is None or not a._is_contiguous or a._dtype not in _FLOAT_DTYPES:
        return False
    try:
        _call_mojo(
            _ElementwiseExtension,
            spec_fn_name,
            (_spec_of(a), float(scalar)),
            arg_dtypes=(a._dtype,),
            output_dtypes=(a._dtype,),
            flags={"INPLACE": True},
            keepalive=(a,),
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return False
    return True


def _try_spec_int_scalar(spec_fn_name, x, scalar):
    """Contiguous tensor-with-int-scalar through a spec op, or None."""
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        return None
    a = _t(x)
    if a is None or a._dtype not in (DType.int32, DType.int64):
        return None
    if not a._is_contiguous:
        a = a._contig()  # queued materialize: metered + covered by the retry
    out = _alloc(a._shape, a._dtype, a._device)
    try:
        _call_mojo(
            _ElementwiseExtension,
            spec_fn_name,
            (_spec_of(a), scalar, _spec_of(out)),
            arg_dtypes=(a._dtype,),
            output_dtypes=(out._dtype,),
            keepalive=(a, out),
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None
    return out


def _on_gpu(t: MojoTensorLike) -> bool:
    return t._device.label == "gpu"


def _alert_not_deterministic(caller: str) -> None:
    """Match PyTorch's deterministic-algorithm error/warn-only contract."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{caller} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
            "You can file an issue at https://github.com/pytorch/pytorch/issues "
            "to help us prioritize adding deterministic support for this operation.",
            stacklevel=2,
        )
        return
    raise RuntimeError(
        f"{caller} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'. You can turn off "
        "determinism just for this operation, or you can use the "
        "'warn_only=True' option, if that's acceptable for your application. "
        "You can also file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation."
    )


def _copy_into(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """dst[...] = src[...] for equal shapes/dtypes, any strides on both."""
    if dst._numel == 0:
        return
    if dst._is_contiguous and src._is_contiguous:
        _device_call(
            eager_kernels.tensor_holder.copy_d2d,
            _ctx_ptr(dst._device),
            dst._ptr,
            src._ptr,
            dst._numel * dst._itemsize,
            keepalive=(dst, src),
        )
    else:
        _copy_strided_into(dst, src)


def _valid_nll_out(tensor, dtype, device) -> bool:
    """Whether an NLL out= tensor may be written or resized in place."""
    out = _t(tensor)
    return out is not None and out._dtype == dtype and out._device == device


def _prepare_nll_out(out: TorchMojoTensor, shape):
    """Return a contiguous NLL kernel destination for ``out``.

    Validation is deliberately separate: callers validate every input and
    output before entering this helper so a rejected call never partially
    resizes its supplied outputs. A wrong-shaped output is rebound to fresh
    contiguous storage (the eager out= resize convention). A correctly-shaped
    view keeps its storage and receives one ordered strided copy after the
    kernel. The common contiguous path writes directly into the supplied out.
    """
    shape = tuple(shape)
    if tuple(out._shape) != shape:
        _resize_payload(out, shape)
        return out
    if out._is_contiguous:
        return out
    return _alloc(shape, out._dtype, out._device)


# ---------------------------------------------------------------------------
# Elementwise helper layer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Broadcast helpers for the strided kernels (logic_ops / WhereSelect).
# Operands are described by the output's dims padded to rank 4 plus
# per-operand REAL element strides (0 on broadcast dims), so strided and
# expanded views feed these kernels with no materialization.
# ---------------------------------------------------------------------------


def _bcast_meta(*tensors):
    """Broadcast metadata (rank <= 4) from the tensors' real strides.

    Returns (out_shape, dims4, [strides4 per tensor]) or None when the
    shapes don't broadcast or exceed rank 4.
    """
    rank = max((len(t._shape) for t in tensors), default=0)
    if rank > 4:
        return None
    out = []
    for i in range(rank):
        size = 1
        for t in tensors:
            j = i - (rank - len(t._shape))
            if j >= 0 and t._shape[j] != 1:
                if size != 1 and t._shape[j] != size:
                    return None
                size = t._shape[j]
        out.append(size)
    dims = [1] * (4 - rank) + out
    all_strides = []
    for t in tensors:
        pad = rank - len(t._shape)
        st = []
        for i in range(rank):
            j = i - pad
            if j < 0 or t._shape[j] == 1:
                st.append(0)
            else:
                st.append(t._strides[j])
        all_strides.append([0] * (4 - rank) + st)
    return out, dims, all_strides


def _scalar_tensor_0d(value, dtype, device) -> TorchMojoTensor:
    """A 0-d tensor holding `value`, for stride-0 broadcast operands."""
    return _submit_prepared_into(
        _FillSpecExtension.prepare((), float(value), dtype, device)
    )


def _cast_tensor(x: TorchMojoTensor, dtype: DType) -> TorchMojoTensor:
    """Dtype cast through CastSpec (strided inputs materialize Mojo-side).

    Callers pre-gate on _CAST_DTYPES; anything else propagates the spec's
    NotImplementedError (the classic kernel silently wrote garbage there).
    Call-queue mode uses the Into form: Python allocates the contiguous
    output, the launch queues (no drain/sync)."""
    t = _t(x)
    return _submit_prepared_into(_CastSpecExtension.prepare(t, dtype))


def _promoted_pair(a: TorchMojoTensor, b: TorchMojoTensor):
    """Same-dtype tensor pair following torch's promotion, or None.

    Only the promotions the generation loops hit: bool combined with any
    castable dtype, and int32 with int64. Deliberately a SUBSET of
    `_binary_promotion`: its callers (where / masked_fill) must keep
    declining mixed float widths rather than silently widening them here.
    """
    if a._dtype == b._dtype:
        return a, b
    if a._dtype == DType.bool and b._dtype in _CAST_DTYPES:
        return _cast_tensor(a, b._dtype), b
    if b._dtype == DType.bool and a._dtype in _CAST_DTYPES:
        return a, _cast_tensor(b, a._dtype)
    if a._dtype == DType.int32 and b._dtype == DType.int64:
        return _cast_tensor(a, DType.int64), b
    if a._dtype == DType.int64 and b._dtype == DType.int32:
        return a, _cast_tensor(b, DType.int64)
    return None


def _scalar_embed(value, dtype: DType) -> float | None:
    """`value` validated for lossless embedding into `dtype`, as a float,
    or None."""
    if not isinstance(value, int | float):
        return None
    if dtype not in _FILL_DTYPES:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_INT:
            return None
        if dtype == DType.bool and value not in (0, 1):
            return None
    elif dtype not in _FLOAT_DTYPES + (DType.float64,):
        # float scalar against an int tensor promotes in torch; not handled.
        return None
    return float(value)


def _resolve_scalar(value, dtype: DType, device) -> TorchMojoTensor | None:
    """A 0-d stride-0 tensor holding a Python scalar in `dtype`, or None
    when the value doesn't embed losslessly."""
    v = _scalar_embed(value, dtype)
    if v is None:
        return None
    return _scalar_tensor_0d(v, dtype, device)


def _launch_where_bcast(
    out: TorchMojoTensor,
    operands: tuple[TorchMojoTensor, TorchMojoTensor, TorchMojoTensor],
    meta: tuple[list[int], list[int], list[list[int]]],
    dtype: DType,
) -> None:
    out_shape, dims, strides = meta
    params = tuple(dims) + tuple(s for st in strides for s in st)
    _call_mojo(
        _DataMovementExtension,
        "WhereSelect",
        (
            out._ptr,
            *[tensor._ptr for tensor in operands],
            params,
            dtype.value,
            _ctx_ptr(out._device),
        ),
        arg_dtypes=tuple(tensor._dtype for tensor in operands),
        output_dtypes=(out._dtype,),
        keepalive=(out, operands),
    )


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------


def _scaled_operand(other, alpha):
    """other * alpha as a tensor, for add/sub with alpha != 1, or None."""
    b = _t(other)
    if b is None:
        if isinstance(other, int | float) and not isinstance(other, bool):
            return other * alpha
        return None
    scaled = _try_spec_scalar("MulScalarSpec", b, alpha)
    if scaled is None:
        scaled = _try_spec_int_scalar("MulScalarIntSpec", b, alpha)
    return scaled


def fast_aten_add(input, other, alpha=1):
    if alpha != 1:
        other = _scaled_operand(other, alpha)
        if other is None:
            return NOT_HANDLED
    result = _try_spec_add_f32_bf16(input, other)
    if result is None:
        result = _try_spec_scalar("AddScalarSpec", input, other)
    if result is None:
        result = _try_spec_int_scalar("AddScalarIntSpec", input, other)
    if result is None:
        result = _try_spec_binary("AddSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


_fast_aten_add_default = fast_aten_add


def fast_aten_add_apple(input, other, alpha=1):
    """Metal specialization for equal-shape contiguous tensor addition."""
    a = _t(input)
    b = _t(other)
    if (
        alpha == 1
        and a is not None
        and b is not None
        and a._device.api == "metal"
        and a._device == b._device
        and a._is_contiguous
        and b._is_contiguous
        and a._dtype == b._dtype
        and a._dtype in _FLOAT_DTYPES
        and a._shape == b._shape
    ):
        out = _alloc(a._shape, a._dtype, a._device)
        if a._numel > 0:
            _call_mojo(
                _ElementwiseExtension,
                "Add",
                (
                    out._ptr,
                    a._ptr,
                    b._ptr,
                    a._numel,
                    a._dtype.value,
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype, b._dtype),
                output_dtypes=(out._dtype,),
                flags={"INPLACE": False},
                keepalive=(out, a, b),
            )
        return out
    return _fast_aten_add_default(input, other, alpha)


def fast_aten_add_(input, other, alpha=1):
    """In-place add into input. Returns None when unavailable."""
    dst = _t(input)
    if dst is None:
        return None
    # Direct in-place kernel when everything lines up.
    b = _t(other)
    if (
        alpha == 1
        and b is not None
        and dst._is_contiguous
        and b._is_contiguous
        and dst._dtype in _BINARY_DTYPES
        and dst._dtype == b._dtype
        and dst._shape == b._shape
        and dst._device == b._device
    ):
        if dst._numel > 0:
            _call_mojo(
                _ElementwiseExtension,
                "Add",
                (
                    dst._ptr,
                    dst._ptr,
                    b._ptr,
                    dst._numel,
                    dst._dtype.value,
                    _ctx_ptr(dst._device),
                ),
                arg_dtypes=(dst._dtype, b._dtype),
                output_dtypes=(dst._dtype,),
                flags={"INPLACE": True},
                keepalive=(dst, b),
            )
        return input
    # A float scalar goes straight into `input`, with no output buffer and no
    # copy back. `alpha` folds into the scalar exactly.
    if (
        b is None
        and isinstance(other, int | float)
        and not isinstance(other, bool)
        and isinstance(alpha, int | float)
        and not isinstance(alpha, bool)
        and _try_spec_scalar_inplace("AddScalarInplace", input, other * alpha)
    ):
        return input
    # General path: functional result, then a (strided-safe) copy back.
    result = fast_aten_add(input, other, alpha)
    if (
        result is NOT_HANDLED
        or result._shape != dst._shape
        or result._dtype != dst._dtype
    ):
        return None
    _copy_into(dst, result)
    return input


def fast_aten_sub(input, other, alpha=1):
    if alpha != 1:
        other = _scaled_operand(other, alpha)
        if other is None:
            return NOT_HANDLED
    result = None
    if isinstance(other, int | float) and not isinstance(other, bool):
        # sub-by-scalar reuses the AddScalar spec with a negated scalar.
        result = _try_spec_scalar("AddScalarSpec", input, -other)
        if result is None and isinstance(other, int):
            result = _try_spec_int_scalar("AddScalarIntSpec", input, -other)
    if result is None:
        result = _try_spec_binary("SubSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


def fast_aten_mul(input, other):
    result = _try_spec_scalar("MulScalarSpec", input, other)
    if result is None:
        result = _try_spec_int_scalar("MulScalarIntSpec", input, other)
    if result is None:
        result = _try_spec_binary("MulSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


def fast_aten_mul_(input, other):
    """In-place multiply used by the foreach gradient-clipping slow path."""
    dst = _t(input)
    if dst is None:
        return None
    if _t(other) is None and _try_spec_scalar_inplace("MulScalarInplace", input, other):
        return input
    result = fast_aten_mul(input, other)
    if (
        result is NOT_HANDLED
        or result._shape != dst._shape
        or result._dtype != dst._dtype
        or result._device != dst._device
    ):
        return None
    _copy_into(dst, result)
    return input


def fast_aten_div(input, other, *, rounding_mode=None):
    if rounding_mode is not None:
        return NOT_HANDLED
    a = _t(input)
    if a is not None and a._dtype in _FLOAT_DTYPES:
        result = _try_spec_binary("DivSpec", input, other)
        if result is not None:
            return result
        return NOT_HANDLED
    # Integer (and int-scalar) division promotes to float32 in torch.
    if a is not None and a._dtype in _INT_SCALAR_DTYPES + (
        DType.int8,
        DType.int16,
        DType.uint8,
    ):
        lhs = _cast_tensor(a, DType.float32)
        b = _t(other)
        if b is not None:
            if b._device != a._device:
                return NOT_HANDLED
            rhs = _cast_tensor(b, DType.float32) if b._dtype != DType.float32 else b
        elif isinstance(other, int | float) and not isinstance(other, bool):
            rhs = other
        else:
            return NOT_HANDLED
        return fast_aten_div(lhs, rhs)
    return NOT_HANDLED


def fast_aten_lerp(self, end, weight):
    """FP32 scalar lerp composed from the existing asynchronous fast ops.

    Match ATen's numerically stable branch from ``native/Lerp.h``.  Keeping
    this as a host composition is sufficient for the foreach AdamW slow path:
    tensor values stay on the accelerator and every constituent launch uses
    the tensors' current device context.
    """
    start = _t(self)
    finish = _t(end)
    if (
        start is None
        or finish is None
        or start._device != finish._device
        or start._dtype != DType.float32
        or finish._dtype != DType.float32
        or not isinstance(weight, int | float)
        or isinstance(weight, bool)
    ):
        return NOT_HANDLED

    # ATen narrows Scalar to the tensor's opmath type before choosing the
    # stable formula.  In particular, values just below 0.5 can round to
    # exactly 0.5 and must take the second branch.
    try:
        narrowed_weight = struct.unpack("=f", struct.pack("=f", weight))[0]
    except (OverflowError, struct.error) as exc:
        # Scalar conversion is part of the ATen contract: finite values that
        # do not fit the tensor's opmath type are rejected rather than
        # silently becoming +/-inf.  Actual infinity remains representable
        # and therefore reaches the kernel normally.
        raise RuntimeError(
            "value cannot be converted to type float without overflow"
        ) from exc

    difference = fast_aten_sub(finish, start)
    if difference is NOT_HANDLED:
        return NOT_HANDLED
    if abs(narrowed_weight) < 0.5:
        return fast_aten_add(start, difference, alpha=narrowed_weight)
    one_minus_weight = struct.unpack("=f", struct.pack("=f", 1.0 - narrowed_weight))[0]
    return fast_aten_sub(finish, difference, alpha=one_minus_weight)


def fast_aten_fill_scalar(input, value):
    """Functional fill: new tensor, same shape/dtype, all elements = value."""
    # ``bool`` is a Python ``int`` subclass and ATen accepts it for every
    # scalar-fill dtype (most importantly for mutating saved bool masks).
    if not isinstance(value, int | float):
        return NOT_HANDLED
    a = _t(input)
    if a is None or a._dtype not in _FILL_DTYPES:
        return NOT_HANDLED
    result = fast_filled(a._shape, value, a._dtype, a._device)
    return result if result is not None else NOT_HANDLED


def fast_aten_fill__scalar(input, value):
    """In-place fill of input (any strides). Returns None when unavailable."""
    if not isinstance(value, int | float):
        return None
    a = _t(input)
    if a is None:
        return None
    if a._dtype == DType.float64 and a._device.api == "metal":
        return None
    if a._numel > 0:
        _device_call(
            eager_kernels.tensor_holder.StridedFill,
            a._ptr,
            float(value),
            _pad8(a._shape, 1),
            _pad8(a._strides, 0),
            a._dtype.value,
            _ctx_ptr(a._device),
            keepalive=(a,),
        )
    return input


def fast_aten_maximum(x, y):
    result = _try_spec_binary("MaximumSpec", x, y)
    return result if result is not None else NOT_HANDLED


def fast_aten_minimum(x, y):
    result = _try_spec_binary("MinimumSpec", x, y)
    return result if result is not None else NOT_HANDLED


def fast_aten_relu(tensor):
    return _unary_spec_op("ReluSpec", tensor)


def fast_aten_exp(input):
    return _unary_spec_op("ExpSpec", input)


def fast_aten_tanh(x):
    return _unary_spec_op("TanhSpec", x)


def fast_aten_pow(x, y):
    result = _try_spec_scalar("PowScalarSpec", x, y)
    return result if result is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Unary elementwise suite
#
# Spec-only (no classic fallback, design doc §2.4): the spec entries cover
# the full classic gates — float64 included (CPU device; the kernels
# comptime-refuse f64 on GPU), strided inputs via Mojo-side temporaries,
# int dtypes for the direct ops, bool via uint8 for the bool-output ops.
# ---------------------------------------------------------------------------


def _unary_spec_op(spec, x):
    """Float-only unary with no classic fallback: the spec entry provably
    covers the classic gate (_FLOAT_DTYPES, strided via Mojo-side
    temporaries), so the classic chain was deleted (design doc §2.4)."""
    result = _try_spec_unary(spec, x)
    return result if result is not None else NOT_HANDLED


def fast_aten_abs(x):
    return _unary_spec_op("AbsSpec", x)


def fast_aten_neg(x):
    return _unary_spec_op("NegSpec", x)


def fast_aten_sign(x):
    return _unary_spec_op("SignSpec", x)


def _int_unary_identity(x):
    """ceil/floor on integer tensors is the identity in torch; return a copy."""
    a = _t(x)
    if a is not None and a._dtype in _BITWISE_DTYPES and a._dtype != DType.bool:
        return a._materialize_contiguous()
    return None


def fast_aten_ceil(x):
    result = _int_unary_identity(x)
    if result is not None:
        return result
    return _unary_spec_op("CeilSpec", x)


def fast_aten_floor(x):
    result = _int_unary_identity(x)
    if result is not None:
        return result
    return _unary_spec_op("FloorSpec", x)


def fast_aten_acos(x):
    return _unary_spec_op("AcosSpec", x)


def fast_aten_asinh(x):
    return _unary_spec_op("AsinhSpec", x)


def fast_aten_atanh(x):
    return _unary_spec_op("AtanhSpec", x)


def fast_aten_cos(x):
    return _unary_spec_op("CosSpec", x)


def fast_aten_cosh(x):
    return _unary_spec_op("CoshSpec", x)


def fast_aten_erf(x):
    return _unary_spec_op("ErfSpec", x)


def fast_aten_log(x):
    return _unary_spec_op("LogSpec", x)


def fast_aten_log1p(x):
    return _unary_spec_op("Log1pSpec", x)


def fast_aten_reciprocal(x):
    return _unary_spec_op("ReciprocalSpec", x)


def fast_aten_rsqrt(x):
    return _unary_spec_op("RsqrtSpec", x)


def fast_aten_sigmoid(x):
    return _unary_spec_op("SigmoidSpec", x)


def fast_aten_silu(x):
    return _unary_spec_op("SiluSpec", x)


def fast_aten_sin(x):
    return _unary_spec_op("SinSpec", x)


def fast_aten_sinh(x):
    return _unary_spec_op("SinhSpec", x)


def fast_aten_sqrt(x):
    return _unary_spec_op("SqrtSpec", x)


def fast_aten_tan(x):
    return _unary_spec_op("TanSpec", x)


def fast_aten_gelu(input, approximate="none"):
    if approximate == "none":
        spec = "GeluNoneSpec"
    elif approximate == "tanh":
        spec = "GeluTanhSpec"
    else:
        return NOT_HANDLED

    a = _t(input)
    if a is not None and a._dtype == DType.bfloat16 and _on_gpu(a) and a._is_contiguous:
        out = _alloc(a._shape, a._dtype, a._device)
        if out._numel > 0:
            _call_mojo(
                _ActivationForwardExtension,
                "GeluForwardBF16",
                (
                    out._ptr,
                    a._ptr,
                    out._numel,
                    int(approximate == "tanh"),
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype,),
                output_dtypes=(out._dtype,),
                flags={"APPROXIMATE": approximate},
                keepalive=(out, a),
            )
        return out
    return _unary_spec_op(spec, input)


def fast_aten_gelu_backward(grad_output, self, *, approximate="none"):
    """Float32/BFloat16 GPU GELU backward through Fable-owned Mojo kernels."""
    grad = _t(grad_output)
    input = _t(self)
    dtype = input._dtype if input is not None else None
    if (
        approximate not in ("none", "tanh")
        or grad is None
        or input is None
        or not _on_gpu(input)
        or grad._device != input._device
        or grad._dtype != dtype
        or dtype not in (DType.float32, DType.bfloat16)
        or tuple(grad._shape) != tuple(input._shape)
    ):
        return NOT_HANDLED

    grad = _tc(grad)
    input = _tc(input)
    out = _alloc(input._shape, dtype, input._device)
    if out._numel > 0:
        op = "GeluBackwardBF16" if dtype == DType.bfloat16 else "GeluBackwardF32"
        _call_mojo(
            _ActivationBackwardExtension,
            op,
            (
                out._ptr,
                grad._ptr,
                input._ptr,
                out._numel,
                int(approximate == "tanh"),
                _ctx_ptr(input._device),
            ),
            arg_dtypes=(grad._dtype, input._dtype),
            output_dtypes=(out._dtype,),
            flags={"APPROXIMATE": approximate},
            keepalive=(out, grad, input),
        )
    return out


def fast_aten_isnan(x):
    result = _try_spec_unary("IsNanSpec", x, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_logical_not(x):
    result = _try_spec_unary("LogicalNotSpec", x, DType.bool)
    return result if result is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Comparisons and bitwise/logic ops (broadcast-strided kernels). These are
# the generation-loop bookkeeping ops: stopping criteria, attention-mask
# prep, position ids.
# ---------------------------------------------------------------------------


def fast_aten_eq(input, other):
    result = _try_spec_binary("EqSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_ne(input, other):
    result = _try_spec_binary("NeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_lt(input, other):
    result = _try_spec_binary("LtSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_le(input, other):
    result = _try_spec_binary("LeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_gt(input, other):
    result = _try_spec_binary("GtSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_ge(input, other):
    result = _try_spec_binary("GeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


def fast_aten_bitwise_and(input, other):
    result = _try_spec_binary("BitwiseAndSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_bitwise_or(input, other):
    result = _try_spec_binary("BitwiseOrSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_bitwise_xor(input, other):
    result = _try_spec_binary("BitwiseXorSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_bitwise_not(input):
    a = _tc(input)
    if a is None or a._dtype not in _BITWISE_DTYPES:
        return NOT_HANDLED
    # bitwise_not on bool is logical negation (0<->1), NOT a byte complement
    # (~0 == 255 in uint8), so route bool through the logical-not path.
    if a._dtype == DType.bool:
        return fast_aten_logical_not(a)
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        _call_mojo(
            _LogicExtension,
            "BitwiseNot",
            (out._ptr, a._ptr, out._numel, a._dtype.value, _ctx_ptr(a._device)),
            arg_dtypes=(a._dtype,),
            output_dtypes=(out._dtype,),
            keepalive=(out, a),
        )
    return out


def fast_aten_isin(elements, test_elements, *, assume_unique=False, invert=False):
    del assume_unique
    el = _tc(elements)
    te = _tc(test_elements)
    if (
        el is None
        or te is None
        or el._device != te._device
        or el._dtype != te._dtype
        or el._dtype not in (DType.int64, DType.int32)
    ):
        return NOT_HANDLED
    if el._numel > 0 and te._numel == 0:
        # x in {} is always False (True under invert).
        return fast_filled(el._shape, 1.0 if invert else 0.0, DType.bool, el._device)
    out = _alloc(el._shape, DType.bool, el._device)
    if el._numel > 0:
        _call_mojo(
            _LogicExtension,
            "IsIn",
            (
                out._ptr,
                el._ptr,
                te._ptr,
                el._numel,
                te._numel,
                1 if invert else 0,
                el._dtype.value,
                _ctx_ptr(el._device),
            ),
            arg_dtypes=(el._dtype, te._dtype),
            output_dtypes=(out._dtype,),
            flags={"INVERT": bool(invert)},
            keepalive=(out, el, te),
        )
    return out


# ---------------------------------------------------------------------------
# Binary/ternary extras: remainder, floor_divide, pow(Tensor,Tensor),
# logical_and/xor, clamp, addcmul/addcdiv. All ride the broadcast-strided
# kernels (real strides, no materialization) except clamp, which is a
# contiguous unary.
# ---------------------------------------------------------------------------


def fast_aten_remainder(input, other):
    # Divisor-signed remainder (Python/torch `%`), float and int dtypes.
    result = _try_spec_binary("RemainderSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_floor_divide(input, other):
    # floor(input / other), float and int dtypes.
    result = _try_spec_binary("FloorDivSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_pow_tensor_tensor(input, exponent):
    # Float-only (the kernel raises on ints, which would leave the output
    # unwritten); gate here so unsupported dtypes fall through cleanly.
    a = _t(input)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    result = _try_spec_binary("PowSpec", input, exponent)
    return result if result is not None else NOT_HANDLED


def _try_logical(spec_fn_name, input, other):
    """Mixed-dtype logical_and / logical_xor: reduce each operand to bool
    (the nonzero test) spec-to-spec via CastSpec, then the bool spec path
    (uint8 dispatch). Same-dtype pairs already went through the spec
    directly; no wrappers are minted for the bool intermediates."""
    a = _t(input)
    b = _t(other)
    if a is None or b is None or a._device != b._device:
        return None
    if a._dtype not in _CAST_DTYPES or b._dtype not in _CAST_DTYPES:
        return None
    try:
        bool_a = a if a._dtype == DType.bool else _cast_tensor(a, DType.bool)
        bool_b = b if b._dtype == DType.bool else _cast_tensor(b, DType.bool)
        return _submit_prepared_into(
            _BinarySpecExtension.prepare(spec_fn_name, bool_a, bool_b, DType.bool)
        )
    except Exception as exc:
        _raise_if_device_oom(exc)
        return None


def fast_aten_logical_and(input, other):
    result = _try_spec_binary("LogicalAndSpec", input, other, DType.bool)
    if result is None:
        result = _try_logical("LogicalAndSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_logical_xor(input, other):
    result = _try_spec_binary("LogicalXorSpec", input, other, DType.bool)
    if result is None:
        result = _try_logical("LogicalXorSpec", input, other)
    return result if result is not None else NOT_HANDLED


def fast_aten_clamp(input, min=None, max=None):
    a = _tc(input)
    if a is None or a._dtype not in _BCAST_DTYPES:
        return NOT_HANDLED
    if min is None and max is None:
        return NOT_HANDLED
    for bound in (min, max):
        if bound is not None and not isinstance(bound, int | float):
            return NOT_HANDLED
    has_min = min is not None
    has_max = max is not None
    lo = float(min) if has_min else 0.0
    hi = float(max) if has_max else 0.0
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        _call_mojo(
            _LogicExtension,
            "ClampScalar",
            (
                out._ptr,
                a._ptr,
                lo,
                hi,
                1 if has_min else 0,
                1 if has_max else 0,
                out._numel,
                a._dtype.value,
                _ctx_ptr(a._device),
            ),
            arg_dtypes=(a._dtype,),
            output_dtypes=(out._dtype,),
            flags={"HAS_MIN": has_min, "HAS_MAX": has_max},
            keepalive=(out, a),
        )
    return out


def _try_addc(kernel_name, self, tensor1, tensor2, value, allow_int):
    a = _t(self)
    b = _t(tensor1)
    c = _t(tensor2)
    if a is None or b is None or c is None:
        return NOT_HANDLED
    if a._device != b._device or a._device != c._device:
        return NOT_HANDLED
    dtype = a._dtype
    if b._dtype != dtype or c._dtype != dtype:
        return NOT_HANDLED
    if dtype in _FLOAT_DTYPES:
        pass
    elif allow_int and dtype in (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ):
        pass
    else:
        return NOT_HANDLED
    if not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_HANDLED
    meta = _bcast_meta(a, b, c)
    if meta is None:
        return NOT_HANDLED
    out_shape, dims, strides = meta
    out = _alloc(out_shape, dtype, a._device)
    if out._numel > 0:
        params = tuple(dims) + tuple(s for st in strides for s in st)
        _call_mojo(
            _LogicExtension,
            kernel_name,
            (
                out._ptr,
                a._ptr,
                b._ptr,
                c._ptr,
                params,
                float(value),
                dtype.value,
                _ctx_ptr(a._device),
            ),
            arg_dtypes=(a._dtype, b._dtype, c._dtype),
            output_dtypes=(out._dtype,),
            keepalive=(out, a, b, c),
        )
    return out


def fast_aten_addcmul(self, tensor1, tensor2, value=1):
    return _try_addc("AddcmulBcast", self, tensor1, tensor2, value, True)


def fast_aten_addcdiv(self, tensor1, tensor2, value=1):
    return _try_addc("AddcdivBcast", self, tensor1, tensor2, value, False)


def _binary_operands(input, other):
    """Resolve (lhs, rhs) TorchMojoTensors with equal dtypes for the ternary
    broadcast kernels (where / masked_fill). Either operand may be a Python
    scalar (which becomes a 0-d stride-0 tensor of the tensor operand's
    dtype), or None if unresolvable."""
    a = _t(input)
    b = _t(other)
    if a is not None and b is not None:
        if a._device != b._device:
            return None
        return _promoted_pair(a, b)
    if a is not None:
        scalar = _resolve_scalar(other, a._dtype, a._device)
        return None if scalar is None else (a, scalar)
    if b is not None:
        scalar = _resolve_scalar(input, b._dtype, b._device)
        return None if scalar is None else (scalar, b)
    return None


def fast_aten_where(condition, input, other):
    cond = _t(condition)
    if cond is None or cond._dtype != DType.bool:
        return NOT_HANDLED
    pair = _binary_operands(input, other)
    if pair is None:
        return NOT_HANDLED
    a, b = pair
    if a._device != cond._device:
        return NOT_HANDLED
    meta = _bcast_meta(cond, a, b)
    if meta is None:
        return NOT_HANDLED
    out = _alloc(meta[0], a._dtype, a._device)
    if out._numel > 0:
        _launch_where_bcast(out, (cond, a, b), meta, a._dtype)
    return out


def _masked_fill_operands(input, mask, value):
    """Resolve (in_t, mask_t, value_t, meta) for masked_fill, or None.

    `value` may be a Python scalar or a 1-element tensor of input's dtype.
    The output shape must equal input's shape (mask broadcasts up to input).
    """
    a = _t(input)
    m = _t(mask)
    if a is None or m is None or m._dtype != DType.bool or m._device != a._device:
        return None
    val = _t(value)
    if val is not None:
        if val._dtype != a._dtype or val._device != a._device or val._numel != 1:
            return None
        if len(val._shape) > 0:
            val = _view_of(val, (), (), val._offset)
    elif isinstance(value, int | float):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
            return None
        if (
            isinstance(value, float)
            and a._dtype not in _FLOAT_DTYPES
            and not value.is_integer()
        ):
            return None
        if a._dtype not in _FILL_DTYPES:
            return None
        val = _scalar_tensor_0d(value, a._dtype, a._device)
    else:
        return None
    meta = _bcast_meta(m, val, a)
    if meta is None or tuple(meta[0]) != tuple(a._shape):
        return None
    return a, m, val, meta


def fast_aten_masked_fill(input, mask, value):
    resolved = _masked_fill_operands(input, mask, value)
    if resolved is None:
        return NOT_HANDLED
    a, m, val, meta = resolved
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        _launch_where_bcast(out, (m, val, a), meta, a._dtype)
    return out


def fast_aten_masked_fill_(input, mask, value):
    """In-place masked fill into input. Returns None when unavailable."""
    resolved = _masked_fill_operands(input, mask, value)
    if resolved is None:
        return None
    a, m, val, meta = resolved
    if a._numel > 0:
        if a._is_contiguous:
            # Writing out == a is safe: each element reads and writes the
            # same index (a's strides are the output layout).
            _launch_where_bcast(a, (m, val, a), meta, a._dtype)
        else:
            result = fast_aten_masked_fill(input, mask, value)
            if result is NOT_HANDLED:
                return None
            _copy_into(a, result)
    return input


# ---------------------------------------------------------------------------
# View / shape-metadata ops. ALL ZERO-COPY: new wrappers over the same
# holder with adjusted shape/strides/offset.
# ---------------------------------------------------------------------------


def _resolve_sizes(shape, numel: int) -> list[int] | None:
    # Single pass; view is the hottest op in transformer forwards.
    prod = 1
    neg = -1
    for i, s in enumerate(shape):
        if s >= 0:
            prod *= s
        elif s == -1:
            if neg >= 0:
                return None
            neg = i
        else:
            return None
    if neg < 0:
        return list(shape) if prod == numel else None
    if prod == 0 or numel % prod != 0:
        return None
    sizes = list(shape)
    sizes[neg] = numel // prod
    return sizes


def _compute_view_strides(old_shape, old_strides, new_shape):
    """Port of ATen's computeStride: strides for viewing `old` as
    `new_shape` without a copy, or None when impossible."""
    if len(old_shape) == 0:
        return (1,) * len(new_shape)
    numel = 1
    for s in old_shape:
        numel *= s
    if numel == 0:
        if tuple(old_shape) == tuple(new_shape):
            return tuple(old_strides)
        return _row_major_strides(new_shape)

    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base_stride = old_strides[-1]
    tensor_numel = 1
    view_numel = 1
    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]
        if tensor_d == 0 or (
            old_shape[tensor_d - 1] != 1
            and old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride
        ):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = old_strides[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return tuple(new_strides)


def _fast_view(tensor, shape):
    if len(shape) == 1 and isinstance(shape[0], list | tuple):
        shape = shape[0]
    if not all(isinstance(s, int) for s in shape):
        return NOT_HANDLED
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    sizes = _resolve_sizes(shape, t._numel)
    if sizes is None:
        return NOT_HANDLED
    if t._is_contiguous:
        return _view_of(t, sizes, _row_major_strides(sizes), t._offset, contiguous=True)
    new_strides = _compute_view_strides(t._shape, t._strides, sizes)
    if new_strides is None:
        # PyTorch's reshape reads the TensorImpl's (fake-contiguous) strides
        # and routes copy-requiring reshapes here too — materialize.
        c = t._materialize_contiguous()
        return _view_of(c, sizes, _row_major_strides(sizes), 0, contiguous=True)
    return _view_of(t, sizes, new_strides, t._offset)


def fast_aten_view(tensor, *shape):
    return _fast_view(tensor, shape)


def fast_aten__unsafe_view(tensor, *shape):
    return _fast_view(tensor, shape)


def fast_aten_unsqueeze(tensor, dim):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    rank = len(t._shape)
    if not -rank - 1 <= dim <= rank:
        return NOT_HANDLED
    if dim < 0:
        dim += rank + 1
    new_shape = t._shape[:dim] + (1,) + t._shape[dim:]
    inserted = t._shape[dim] * t._strides[dim] if dim < rank else 1
    new_strides = t._strides[:dim] + (inserted,) + t._strides[dim:]
    return _view_of(t, new_shape, new_strides, t._offset)


def fast_aten_squeeze_dim(tensor, dim):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    if t._shape[dim] != 1:
        return _view_of(t, t._shape, t._strides, t._offset)
    new_shape = t._shape[:dim] + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + t._strides[dim + 1 :]
    return _view_of(t, new_shape, new_strides, t._offset)


def fast_aten_alias(tensor):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    # PyTorch dispatches alias on a saved tensor that was an OUTPUT of an
    # autograd Function, which is how a saved-tensor unpack hook's result first
    # reaches an op. A hook may hand back a TorchMojoTensor stripped of its
    # allocation holder, and every accessor below would then fail with a bare
    # AttributeError from deep inside a dispatch. Name the invariant here, at
    # the boundary where such a tensor arrives, rather than on the hot accessor
    # paths: alias is rare, so this check is free where it matters.
    if not hasattr(t, "_holder"):
        raise RuntimeError(
            "saved-tensor hook returned an unusable Mojo tensor without "
            "a TorchMojoTensor allocation holder; its unpack hook must "
            "return a complete TorchMojoTensor or a host tensor"
        )
    return _view_of(t, t._shape, t._strides, t._offset)


fast_aten_detach = fast_aten_alias


def fast_aten_permute(input, dims):
    t = _t(input)
    if t is None or not isinstance(dims, list | tuple):
        return NOT_HANDLED
    rank = len(t._shape)
    if len(dims) != rank:
        return NOT_HANDLED
    perm = [d % rank if -rank <= d < rank else None for d in dims]
    if None in perm or len(set(perm)) != rank:
        return NOT_HANDLED
    new_shape = tuple(t._shape[p] for p in perm)
    new_strides = tuple(t._strides[p] for p in perm)
    return _view_of(t, new_shape, new_strides, t._offset)


def fast_aten_t(input):
    t = _t(input)
    if t is None or len(t._shape) > 2:
        return NOT_HANDLED
    if len(t._shape) < 2:
        return _view_of(t, t._shape, t._strides, t._offset)
    return fast_aten_transpose(input, 0, 1)


def fast_aten_transpose(input, dim0, dim1):
    t = _t(input)
    if t is None or not isinstance(dim0, int) or not isinstance(dim1, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0:
        return _view_of(t, t._shape, t._strides, t._offset)
    if not (-rank <= dim0 < rank and -rank <= dim1 < rank):
        return NOT_HANDLED
    dim0 %= rank
    dim1 %= rank
    shape = list(t._shape)
    strides = list(t._strides)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    return _view_of(t, shape, strides, t._offset)


def fast_aten_expand(tensor, sizes, *, implicit=False):
    t = _t(tensor)
    if t is None or not isinstance(sizes, list | tuple):
        return NOT_HANDLED
    rank = len(t._shape)
    new_rank = len(sizes)
    if new_rank < rank:
        return NOT_HANDLED
    pad = new_rank - rank
    new_shape = []
    new_strides = []
    for i, s in enumerate(sizes):
        j = i - pad
        if j < 0:
            if s == -1:
                return NOT_HANDLED
            new_shape.append(s)
            new_strides.append(0)
        else:
            old_size = t._shape[j]
            if s == -1 or s == old_size:
                new_shape.append(old_size)
                new_strides.append(t._strides[j])
            elif old_size == 1:
                new_shape.append(s)
                new_strides.append(0)
            else:
                return NOT_HANDLED
    return _view_of(t, new_shape, new_strides, t._offset)


def fast_aten_slice(input, dim=0, start=None, end=None, step=1):
    t = _t(input)
    if t is None or not isinstance(dim, int) or not isinstance(step, int) or step < 1:
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
    start = 0 if start is None else start
    end = size if end is None else end
    if not isinstance(start, int) or not isinstance(end, int):
        return NOT_HANDLED
    if start < 0:
        start += size
    if end < 0:
        end += size
    start = min(max(start, 0), size)
    end = min(max(end, 0), size)
    length = max(end - start, 0)
    length = -(-length // step)  # ceil div for step > 1
    new_shape = t._shape[:dim] + (length,) + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + (t._strides[dim] * step,) + t._strides[dim + 1 :]
    new_offset = t._offset + start * t._strides[dim]
    return _view_of(t, new_shape, new_strides, new_offset)


def fast_aten_select(input, dim, index):
    t = _t(input)
    if t is None or not isinstance(dim, int) or not isinstance(index, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
    if index < 0:
        index += size
    if not 0 <= index < size:
        return NOT_HANDLED
    new_shape = t._shape[:dim] + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + t._strides[dim + 1 :]
    new_offset = t._offset + index * t._strides[dim]
    return _view_of(t, new_shape, new_strides, new_offset)


def fast_aten_split(input, split_size, dim=0):
    t = _t(input)
    if t is None or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
    if isinstance(split_size, int):
        if split_size <= 0:
            return NOT_HANDLED
        lengths = [
            min(split_size, size - start) for start in range(0, size, split_size)
        ]
        if not lengths:
            lengths = [0]
    else:
        lengths = list(split_size)
        if (
            not all(isinstance(x, int) and x >= 0 for x in lengths)
            or sum(lengths) != size
        ):
            return NOT_HANDLED
    results = []
    offset = 0
    for length in lengths:
        results.append(fast_aten_slice(t, dim, offset, offset + length))
        offset += length
    if any(r is NOT_HANDLED for r in results):
        return NOT_HANDLED
    return results


def fast_aten_split_with_sizes(input, split_sizes, dim=0):
    return fast_aten_split(input, list(split_sizes), dim)


def fast_aten_unbind(input, dim=0):
    t = _t(input)
    if t is None or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    results = [fast_aten_select(t, dim, i) for i in range(t._shape[dim])]
    if any(r is NOT_HANDLED for r in results):
        return NOT_HANDLED
    return results


def fast_aten_clone(input, *, memory_format=None):
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    if t._is_contiguous:
        # Straight D2D memcpy; the strided gather kernel below costs ~2x
        # the bandwidth (per-element index math on 50k-column logits).
        out = _alloc(t._shape, t._dtype, t._device)
        _copy_into(out, t)
        return out
    return t._materialize_contiguous()


# ---------------------------------------------------------------------------
# Concatenation along any dim: one destination-strided narrow copy per
# input into the output's slot for that input.
# ---------------------------------------------------------------------------


def _is_legacy_empty(t) -> bool:
    x = _t(t)
    return x is not None and len(x._shape) == 1 and x._numel == 0


def fast_aten_cat(tensors, dim=0):
    # PyTorch's cat skips legacy "empty" (1-D, size-0) tensors, e.g.
    # uninitialized KV-caches.
    real = [x for x in tensors if not _is_legacy_empty(x)]
    if not real or not isinstance(dim, int):
        return NOT_HANDLED
    ins = [_t(x) for x in real]
    first = ins[0]
    if first is None or first._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    rank = len(first._shape)
    if rank == 0 or not -rank <= dim < rank or rank > _MAX_RANK:
        return NOT_HANDLED
    dim %= rank
    for b in ins:
        if (
            b is None
            or b._dtype != first._dtype
            or b._device != first._device
            or len(b._shape) != rank
            or any(i != dim and b._shape[i] != first._shape[i] for i in range(rank))
        ):
            return NOT_HANDLED
    out_shape = list(first._shape)
    out_shape[dim] = sum(b._shape[dim] for b in ins)
    inner = math.prod(out_shape[dim + 1 :])
    outer = math.prod(out_shape[:dim])
    out = _alloc(out_shape, first._dtype, first._device)
    dst_stride = out_shape[dim] * inner
    ctx = _ctx_ptr(first._device)
    if (
        len(ins) == 2
        and ins[0]._is_contiguous
        and ins[1]._is_contiguous
        and outer > 0
        and inner > 0
    ):
        # Both source rows contiguous: one fused kernel fills each output
        # row from both inputs (no second launch, no bubble between the
        # two copies). Raises (e.g. row lengths not vector-aligned, outer
        # over the grid cap, CPU device) -> per-input copy loop below.
        len1 = ins[0]._shape[dim] * inner
        len2 = ins[1]._shape[dim] * inner
        if (
            first._device.api != "cpu"
            and 0 < outer <= 65535
            and len1 > 0
            and len2 > 0
            and len1 % 4 == 0
            and len2 % 4 == 0
        ):
            _call_mojo(
                _DataMovementExtension,
                "Cat2",
                (
                    out._ptr,
                    ins[0]._ptr,
                    ins[1]._ptr,
                    outer,
                    len1,
                    len2,
                    out._itemsize,
                    ctx,
                ),
                arg_dtypes=(ins[0]._dtype, ins[1]._dtype),
                output_dtypes=(out._dtype,),
                keepalive=(out, ins[0], ins[1]),
            )
            return out
    if (
        len(ins) == 3
        and first._device.api == "metal"
        and all(b._is_contiguous for b in ins)
        and outer > 0
        and inner > 0
    ):
        # Metal: one fused three-source kernel (the split-backward
        # reassembly) instead of three narrow copies and two pipeline
        # bubbles. Raises (row lengths not vector-aligned, outer over the
        # grid cap) -> per-input copy loop below.
        lens = [b._shape[dim] * inner for b in ins]
        if (
            0 < outer <= 65535
            and all(n > 0 for n in lens)
            and all(n % 4 == 0 for n in lens)
        ):
            _call_mojo(
                _DataMovementExtension,
                "Cat3",
                (
                    out._ptr,
                    ins[0]._ptr,
                    ins[1]._ptr,
                    ins[2]._ptr,
                    outer,
                    lens[0],
                    lens[1],
                    lens[2],
                    out._itemsize,
                    ctx,
                ),
                arg_dtypes=tuple(tensor._dtype for tensor in ins),
                output_dtypes=(out._dtype,),
                keepalive=(out, ins[0], ins[1], ins[2]),
            )
            return out
    offset = 0
    for b in ins:
        copy_len = b._shape[dim] * inner
        if copy_len > 0 and outer > 0:
            if b._is_contiguous:
                _call_mojo(
                    _DataMovementExtension,
                    "NarrowCopyDst",
                    (
                        out._ptr,
                        b._ptr,
                        outer,
                        dst_stride,
                        copy_len,
                        offset,
                        out._itemsize,
                        ctx,
                    ),
                    arg_dtypes=(b._dtype,),
                    output_dtypes=(out._dtype,),
                    keepalive=(out, b),
                )
            else:
                # Strided input (e.g. the new-token K/V head-transpose in a
                # KV-cache append): gather it straight into its output slot
                # instead of materializing a contiguous copy first. The slot
                # is out[..., pos:pos+len, ...]: same strides as out, offset
                # pos * out_strides[dim] == the accumulated flat offset.
                slot = _view_of(out, b._shape, out._strides, offset)
                _copy_strided_into(slot, b)
        offset += copy_len
    return out


# ---------------------------------------------------------------------------
# Movement tail: stack / repeat / tril / triu / index / scatter /
# select_scatter / nonzero.
# ---------------------------------------------------------------------------

# Max tensor rank the strided (rank-8-padded) kernels accept.
_MAX_RANK = 8

# Dtypes ScatterDim dispatches on (mirrors SCATTER_DTYPES in
# data_movement_ops.mojo): the scalar value overload casts through Float64.
_SCATTER_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
)


def _try_stack_scalars(
    tensors: Sequence[torch.Tensor], dim: int
) -> TorchMojoTensor | None:
    """Batched Metal path for stacking 0-d float32 scalars (the foreach-norm
    outputs gradient clipping aggregates): one gather launch per 8 inputs
    instead of one copy launch per input. None -> take the generic path."""
    if dim not in (0, -1):
        return None
    unwrapped = []
    for tensor in tensors:
        mojo_tensor = _t(tensor)
        if mojo_tensor is None or mojo_tensor._shape != ():
            return None
        unwrapped.append(mojo_tensor)
    device = unwrapped[0]._device
    if device.api != "metal" or any(
        tensor._device != device or tensor._dtype != DType.float32
        for tensor in unwrapped
    ):
        return None
    out = _alloc((len(unwrapped),), DType.float32, device)
    _call_mojo(
        _OptimizerExtension,
        "ForeachGatherScalars",
        (tuple(tensor._ptr for tensor in unwrapped), out._ptr, _ctx_ptr(device)),
        arg_dtypes=(DType.float32,),
        output_dtypes=(out._dtype,),
        keepalive=(tensor, out),
    )
    return out


def fast_aten_stack(tensors, dim=0):
    # stack = unsqueeze each input at `dim`, then concatenate along `dim`.
    # Both helpers normalize `dim` against the SAME (rank + 1) base, so a raw
    # (possibly negative) `dim` stays consistent between them.
    if not isinstance(tensors, list | tuple) or len(tensors) == 0:
        return NOT_HANDLED
    if not isinstance(dim, int):
        return NOT_HANDLED
    scalar_stack = _try_stack_scalars(tensors, dim)
    if scalar_stack is not None:
        return scalar_stack
    unsqueezed = []
    for x in tensors:
        u = fast_aten_unsqueeze(x, dim)
        if u is NOT_HANDLED:
            return NOT_HANDLED
        unsqueezed.append(u)
    return fast_aten_cat(unsqueezed, dim)


def fast_aten_repeat(input, repeats):
    t = _tc(input)
    if t is None or t._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    if not isinstance(repeats, list | tuple):
        return NOT_HANDLED
    repeats = list(repeats)
    if not all(isinstance(r, int) and r >= 0 for r in repeats):
        return NOT_HANDLED
    rank = len(t._shape)
    # torch left-pads the input shape with 1s when len(repeats) > rank; fewer
    # repeats than dims is invalid.
    if len(repeats) < rank or len(repeats) > _MAX_RANK:
        return NOT_HANDLED
    n_out = len(repeats)
    padded_shape = (1,) * (n_out - rank) + tuple(t._shape)
    out_shape = tuple(padded_shape[i] * repeats[i] for i in range(n_out))
    padded_strides = _row_major_strides(padded_shape)
    out = _alloc(out_shape, t._dtype, t._device)
    if out._numel > 0:
        _call_mojo(
            _DataMovementExtension,
            "TileCopy",
            (
                out._ptr,
                t._ptr,
                _pad8(out_shape, 1),
                _pad8(padded_shape, 1),
                _pad8(padded_strides, 0),
                out._itemsize,
                _ctx_ptr(t._device),
            ),
            arg_dtypes=(t._dtype,),
            output_dtypes=(out._dtype,),
            keepalive=(out, t),
        )
    return out


def _fast_triangular(input, diagonal, upper):
    if not isinstance(diagonal, int):
        return NOT_HANDLED
    t = _tc(input)
    if t is None or t._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    if len(t._shape) < 2:
        return NOT_HANDLED
    out = _alloc(t._shape, t._dtype, t._device)
    if out._numel > 0:
        rows = t._shape[-2]
        cols = t._shape[-1]
        batch = t._numel // (rows * cols)
        _call_mojo(
            _DataMovementExtension,
            "TriangularCopy",
            (
                out._ptr,
                t._ptr,
                batch,
                rows,
                cols,
                diagonal,
                upper,
                out._itemsize,
                _ctx_ptr(t._device),
            ),
            arg_dtypes=(t._dtype,),
            output_dtypes=(out._dtype,),
            flags={"UPPER": bool(upper)},
            keepalive=(out, t),
        )
    return out


def fast_aten_tril(input, diagonal=0):
    return _fast_triangular(input, diagonal, 0)


def fast_aten_triu(input, diagonal=0):
    return _fast_triangular(input, diagonal, 1)


def fast_aten_index(input, indices):
    t = _t(input)
    if t is None or not isinstance(indices, list | tuple):
        return NOT_HANDLED
    non_none = [(i, x) for i, x in enumerate(indices) if x is not None]
    # Only the single-index-on-dim-0 cases are handled fast.
    if len(non_none) != 1 or non_none[0][0] != 0:
        return NOT_HANDLED
    idx = _t(non_none[0][1])
    if idx is None or idx._device != t._device:
        return NOT_HANDLED

    if idx._dtype in (DType.int32, DType.int64):
        # Gather whole rows along dim 0.
        src = _tc(t)
        if src is None or src._dtype not in _COPYABLE_DTYPES or len(src._shape) < 1:
            return NOT_HANDLED
        idx_c = _tc(idx)
        row_len = 1
        for s in src._shape[1:]:
            row_len *= s
        out_shape = tuple(idx_c._shape) + tuple(src._shape[1:])
        out = _alloc(out_shape, src._dtype, src._device)
        if out._numel > 0:
            _call_mojo(
                _DataMovementExtension,
                "GatherRows",
                (
                    out._ptr,
                    src._ptr,
                    idx_c._ptr,
                    idx_c._dtype.value,
                    idx_c._numel,
                    row_len,
                    src._shape[0],
                    out._itemsize,
                    _ctx_ptr(src._device),
                ),
                arg_dtypes=(src._dtype, idx_c._dtype),
                output_dtypes=(out._dtype,),
                keepalive=(out, src, idx_c),
            )
        return out

    if idx._dtype == DType.bool:
        # Boolean mask: data-dependent output shape -> host bounce (syncs).
        cpu_self = t._to_cpu_tensor()
        cpu_indices = tuple(
            slice(None) if x is None else _t(x)._to_cpu_tensor() for x in indices
        )
        result = cpu_self[cpu_indices]
        return TorchMojoTensor._from_cpu(result, t._device)

    return NOT_HANDLED


def _fast_scatter(input, dim, index, src, value):
    a = _t(input)
    idx = _t(index)
    if a is None or idx is None or not isinstance(dim, int):
        return NOT_HANDLED
    if idx._device != a._device or idx._dtype != DType.int64:
        return NOT_HANDLED
    if a._dtype not in _SCATTER_DTYPES:
        return NOT_HANDLED
    if a._dtype == DType.float64 and a._device.api == "metal":
        return NOT_HANDLED
    rank = len(a._shape)
    if rank == 0 or rank > 4 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    if len(idx._shape) != rank:
        return NOT_HANDLED

    out = a._materialize_contiguous()  # clone(self)
    idx_c = _tc(idx)

    is_value = 0
    value_f = 0.0
    src_ptr = out._ptr  # unused in value mode; a valid pointer for the kernel
    src_strides4 = [0, 0, 0, 0]
    if src is not None:
        s = _tc(src)
        if (
            s is None
            or s._dtype != a._dtype
            or s._device != a._device
            or len(s._shape) != rank
        ):
            return NOT_HANDLED
        src_ptr = s._ptr
        src_strides4 = [0] * (4 - rank) + list(_row_major_strides(s._shape))
    else:
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int | float):
            return NOT_HANDLED
        is_value = 1
        if a._dtype == DType.bool:
            value_f = 1.0 if value != 0 else 0.0
        else:
            value_f = float(value)

    dims4 = [1] * (4 - rank) + list(idx_c._shape)
    out_strides4 = [0] * (4 - rank) + list(out._strides)
    idx_strides4 = [0] * (4 - rank) + list(idx_c._strides)
    dim_padded = dim + (4 - rank)
    params = (
        tuple(dims4)
        + tuple(out_strides4)
        + tuple(src_strides4)
        + tuple(idx_strides4)
        + (dim_padded,)
    )
    if idx_c._numel > 0:
        _call_mojo(
            _DataMovementExtension,
            "ScatterDim",
            (
                out._ptr,
                idx_c._ptr,
                src_ptr,
                params,
                is_value,
                value_f,
                a._dtype.value,
                _ctx_ptr(a._device),
            ),
            arg_dtypes=(
                a._dtype,
                idx_c._dtype,
                s._dtype if src is not None else a._dtype,
            ),
            output_dtypes=(out._dtype,),
            flags={"VALUE_MODE": bool(is_value)},
            keepalive=(out, idx_c),
        )
    return out


def fast_aten_scatter_src(input, dim, index, src):
    return _fast_scatter(input, dim, index, src, None)


def fast_aten_scatter_value(input, dim, index, value):
    return _fast_scatter(input, dim, index, None, value)


def fast_aten_select_scatter(input, src, dim, index):
    a = _t(input)
    s = _t(src)
    if a is None or s is None or not isinstance(dim, int) or not isinstance(index, int):
        return NOT_HANDLED
    if s._device != a._device:
        return NOT_HANDLED
    out = fast_aten_clone(a)
    if out is NOT_HANDLED:
        return NOT_HANDLED
    view = fast_aten_select(out, dim, index)
    if view is NOT_HANDLED:
        return NOT_HANDLED
    if s._dtype != out._dtype:
        s = _cast_tensor(s, out._dtype)
    if tuple(s._shape) != tuple(view._shape):
        s = fast_aten_expand(s, view._shape)
        if s is NOT_HANDLED:
            return NOT_HANDLED
    _copy_into(view, s)
    return out


def fast_aten_nonzero(input):
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    # Data-dependent output shape -> host bounce (syncs). `.nonzero()` returns
    # an int64 (N, ndim) tensor in C-order over the input's coordinates.
    result = t._to_cpu_tensor().nonzero()
    return TorchMojoTensor._from_cpu(result, t._device)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _fast_batch_norm_inference(input, weight, bias, running_mean, running_var, eps):
    a = _t(input)
    stats = [_t(x) for x in (running_mean, running_var, weight, bias)]
    if (
        a is not None
        and all(stat is not None for stat in stats)
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) >= 2
        and a._numel > 0
        and a._is_contiguous
        and all(
            stat._device == a._device
            and stat._dtype == a._dtype
            and stat._is_contiguous
            for stat in stats
        )
    ):
        out = _alloc(a._shape, a._dtype, a._device)
        _call_mojo(
            _NNExtension,
            "BatchNormSpec",
            (
                _spec_of(a),
                _spec_of(stats[0]),
                _spec_of(stats[1]),
                _spec_of(stats[2]),
                _spec_of(stats[3]),
                float(eps),
                _spec_of(out),
            ),
            arg_dtypes=(a._dtype, *(stat._dtype for stat in stats)),
            output_dtypes=(out._dtype,),
            keepalive=(a, stats, out),
        )
        # Inference mode returns empty (0,) tensors for the saved stats.
        return (
            out,
            _alloc((0,), a._dtype, a._device),
            _alloc((0,), a._dtype, a._device),
        )
    a = _tc(input)
    if a is None or a._dtype not in _FLOAT_DTYPES or len(a._shape) < 2:
        return NOT_HANDLED
    if a._numel == 0:
        return NOT_HANDLED
    params = [_tc(x) for x in (running_mean, running_var, weight, bias)]
    if any(p is None or p._dtype != a._dtype for p in params):
        return NOT_HANDLED
    mean_t, var_t, gamma_t, beta_t = params
    channels = a._shape[1]
    inner = math.prod(a._shape[2:])
    out = _alloc(a._shape, a._dtype, a._device)
    _call_mojo(
        _NNExtension,
        "BatchNormInference",
        (
            out._ptr,
            a._ptr,
            mean_t._ptr,
            var_t._ptr,
            gamma_t._ptr,
            beta_t._ptr,
            (float(eps), channels, inner, a._numel),
            a._dtype.value,
            _ctx_ptr(a._device),
        ),
        arg_dtypes=(
            a._dtype,
            mean_t._dtype,
            var_t._dtype,
            gamma_t._dtype,
            beta_t._dtype,
        ),
        output_dtypes=(out._dtype,),
        keepalive=(out, a, mean_t, var_t, gamma_t, beta_t),
    )
    # Inference mode returns empty (0,) tensors for the saved stats.
    return (out, _alloc((0,), a._dtype, a._device), _alloc((0,), a._dtype, a._device))


def fast_aten_native_batch_norm(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
    if not training:
        return _fast_batch_norm_inference(
            input, weight, bias, running_mean, running_var, eps
        )
    return NOT_HANDLED


def fast_aten__native_batch_norm_legit_no_training(
    input, weight, bias, running_mean, running_var, momentum, eps
):
    return _fast_batch_norm_inference(
        input, weight, bias, running_mean, running_var, eps
    )


def fast_aten_native_dropout(input, p, train):
    """Contiguous float32 GPU native dropout with host-owned RNG state.

    The Fable-owned Mojo kernel is stateless: this host path atomically
    reserves the exact Philox4 interval and passes its full seed/counter as
    four 32-bit limbs.  Keeping every Python integer below ``2**32`` avoids
    ``Py_ssize_t`` overflow in the raw CPython bridge while preserving all 64
    bits of both values.
    """
    a = _t(input)
    if (
        a is None
        or not _on_gpu(a)
        or a._dtype != DType.float32
        or (train is not None and type(train) is not bool)
    ):
        return NOT_HANDLED

    # Match native_dropout's inference shortcut: p is deliberately ignored,
    # including an otherwise invalid or NaN value, and RNG state is untouched.
    if train is False:
        output = fast_aten_clone(a)
        mask = fast_filled(a._shape, True, DType.bool, a._device)
        if output is NOT_HANDLED or mask is None:
            return NOT_HANDLED
        return output, mask

    if not isinstance(p, int | float) or not 0.0 <= p <= 1.0:
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    p = float(p)

    # Return distinct tensors even for an empty input, without a zero-grid
    # launch or a zero-length generator reservation.
    if a._numel == 0:
        return (
            _alloc(a._shape, DType.float32, a._device),
            _alloc(a._shape, DType.bool, a._device),
        )

    # PyTorch's GPU endpoint shortcut neither divides by zero nor consumes
    # generator state.
    if p == 1.0:
        output = fast_filled(a._shape, 0.0, DType.float32, a._device)
        mask = fast_filled(a._shape, False, DType.bool, a._device)
        if output is None or mask is None:
            return NOT_HANDLED
        return output, mask

    # Validate and allocate before changing generator state.  No operation
    # after the reservation reads the host or synchronizes the device queue.
    a = _tc(a)
    output = _alloc(a._shape, DType.float32, a._device)
    mask = _alloc(a._shape, DType.bool, a._device)
    seed, base_offset = _reserve_philox_state(a._torch_device, (a._numel + 3) // 4)
    word_mask = (1 << 32) - 1
    _call_mojo(
        _DropoutExtension,
        "NativeDropoutF32",
        (
            output._ptr,
            mask._ptr,
            a._ptr,
            a._numel,
            p,
            seed & word_mask,
            (seed >> 32) & word_mask,
            base_offset & word_mask,
            (base_offset >> 32) & word_mask,
            _ctx_ptr(a._device),
        ),
        arg_dtypes=(a._dtype,),
        output_dtypes=(output._dtype, mask._dtype),
        keepalive=(output, mask, a),
    )
    return output, mask


def fast_aten_native_dropout_backward(grad_output, mask, scale):
    """Float32 GPU native-dropout backward through the saved bool mask."""
    grad = _t(grad_output)
    keep = _t(mask)
    if (
        grad is None
        or keep is None
        or not _on_gpu(grad)
        or grad._dtype != DType.float32
        or keep._dtype != DType.bool
        or keep._device != grad._device
        or tuple(keep._shape) != tuple(grad._shape)
        or not isinstance(scale, int | float)
    ):
        return NOT_HANDLED

    grad = _tc(grad)
    keep = _tc(keep)
    grad_input = _alloc(grad._shape, DType.float32, grad._device)
    if grad._numel > 0:
        _call_mojo(
            _DropoutExtension,
            "NativeDropoutBackwardF32",
            (
                grad_input._ptr,
                grad._ptr,
                keep._ptr,
                grad._numel,
                float(scale),
                _ctx_ptr(grad._device),
            ),
            arg_dtypes=(grad._dtype, keep._dtype),
            output_dtypes=(grad_input._dtype,),
            keepalive=(grad_input, grad, keep),
        )
    return grad_input


def fast_aten_native_layer_norm(input, normalized_shape, weight, bias, eps):
    a = _t(input)
    normalized_shape = tuple(normalized_shape)
    k = len(normalized_shape)
    if (
        a is None
        or a._numel == 0
        or a._dtype not in _FLOAT_DTYPES
        or k < 1
        or len(a._shape) < k
        or tuple(a._shape[-k:]) != normalized_shape
    ):
        return NOT_HANDLED
    cols = 1
    for s in normalized_shape:
        cols *= s
    rows = a._numel // cols
    # weight/bias are optional (no-affine layer norm): default to 1s / 0s.
    gamma = None
    beta = None
    if weight is not None:
        gamma = _t(weight)
        if (
            gamma is None
            or gamma._dtype != a._dtype
            or gamma._device != a._device
            or tuple(gamma._shape) != normalized_shape
        ):
            return NOT_HANDLED
    if bias is not None:
        beta = _t(bias)
        if (
            beta is None
            or beta._dtype != a._dtype
            or beta._device != a._device
            or tuple(beta._shape) != normalized_shape
        ):
            return NOT_HANDLED

    # Materialize only after all metadata has been validated, so a rejected
    # cross-device or wrong-shape call cannot enqueue partial work.
    eps_value = float(eps)
    a = _tc(a)
    if weight is not None:
        gamma = _tc(gamma)
    if bias is not None:
        beta = _tc(beta)
    # Not spec-converted: the classic prologue here is already thin, and
    # building three (holder, spec, shape, ptr) result groups measurably
    # costs more than the removed Python work (+4us/call measured A/B on
    # (6, 768); contrast min.dim, whose heavy classic prologue makes the
    # two-group spec a -34% win).
    out = _alloc(a._shape, a._dtype, a._device)
    stat_shape = tuple(a._shape[:-k]) + (1,) * k
    mean = _alloc(stat_shape, DType.float32, a._device)
    rstd = _alloc(stat_shape, DType.float32, a._device)

    # The training-hot FP32 GPU route has a direct optional-affine ABI.  A
    # zero pointer is safe because the runtime flags prevent any corresponding
    # device load; this avoids allocating and filling synthetic ones/zeros.
    if a._dtype == DType.float32 and _on_gpu(a):
        _call_mojo(
            _NormalizationForwardExtension,
            "LayerNormForwardF32",
            (
                out._ptr,
                mean._ptr,
                rstd._ptr,
                a._ptr,
                gamma._ptr if weight is not None else 0,
                beta._ptr if bias is not None else 0,
                rows,
                cols,
                eps_value,
                int(weight is not None),
                int(bias is not None),
                _ctx_ptr(a._device),
            ),
            arg_dtypes=(
                a._dtype,
                gamma._dtype if gamma is not None else a._dtype,
                beta._dtype if beta is not None else a._dtype,
            ),
            output_dtypes=(out._dtype, mean._dtype, rstd._dtype),
            flags={"HAS_WEIGHT": weight is not None, "HAS_BIAS": bias is not None},
            keepalive=(out, mean, rstd, a, gamma, beta),
        )
        return out, mean, rstd

    if weight is None:
        gamma = fast_filled((cols,), 1.0, a._dtype, a._device)
    if bias is None:
        beta = fast_filled((cols,), 0.0, a._dtype, a._device)
    _call_mojo(
        _NNExtension,
        "LayerNorm",
        (
            out._ptr,
            mean._ptr,
            rstd._ptr,
            a._ptr,
            gamma._ptr,
            beta._ptr,
            (eps_value, rows, cols),
            a._dtype.value,
            _ctx_ptr(a._device),
        ),
        arg_dtypes=(a._dtype, gamma._dtype, beta._dtype),
        output_dtypes=(out._dtype, mean._dtype, rstd._dtype),
        flags={"HAS_WEIGHT": weight is not None, "HAS_BIAS": bias is not None},
        keepalive=(out, mean, rstd, a, gamma, beta),
    )
    return out, mean, rstd


def fast_aten_native_layer_norm_backward(
    grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
):
    """Direct eager LayerNorm backward for the pure-Mojo f32 GPU kernels.

    Validate every tensor from Python-side metadata before materializing a
    view or allocating an output. Besides keeping rejected calls free of
    partial device work, this lets the Fable-owned kernels keep a compact
    pointer-only ABI while preserving ATen's optional-output contract.
    """
    grad = _t(grad_out)
    a = _t(input)
    saved_mean = _t(mean)
    saved_rstd = _t(rstd)
    gamma = _t(weight) if weight is not None else None
    beta = _t(bias) if bias is not None else None
    normalized_shape = tuple(normalized_shape)
    mask = tuple(output_mask)
    k = len(normalized_shape)

    if (
        len(mask) != 3
        or any(not isinstance(requested, bool) for requested in mask)
        or grad is None
        or a is None
        or saved_mean is None
        or saved_rstd is None
        or not _on_gpu(a)
        or a._dtype != DType.float32
        or grad._dtype != DType.float32
        or saved_mean._dtype != DType.float32
        or saved_rstd._dtype != DType.float32
        or grad._device != a._device
        or saved_mean._device != a._device
        or saved_rstd._device != a._device
        or tuple(grad._shape) != tuple(a._shape)
        or k < 1
        or len(a._shape) < k
        or tuple(a._shape[-k:]) != normalized_shape
    ):
        return NOT_HANDLED

    cols = math.prod(normalized_shape)
    if cols <= 0:
        return NOT_HANDLED
    rows = a._numel // cols
    if saved_mean._numel != rows or saved_rstd._numel != rows:
        return NOT_HANDLED

    if weight is not None and (
        gamma is None
        or gamma._dtype != DType.float32
        or gamma._device != a._device
        or tuple(gamma._shape) != normalized_shape
    ):
        return NOT_HANDLED
    if bias is not None and (
        beta is None
        or beta._dtype != DType.float32
        or beta._device != a._device
        or tuple(beta._shape) != normalized_shape
    ):
        return NOT_HANDLED
    if (mask[1] and gamma is None) or (mask[2] and beta is None):
        return NOT_HANDLED

    if not any(mask):
        return None, None, None

    if rows == 0:
        # ATen defines the two affine reductions over an empty outer extent as
        # zero. Avoid a zero-grid LayerNorm launch; Fill handles only the
        # requested nonempty affine outputs.
        grad_input = _alloc(a._shape, DType.float32, a._device) if mask[0] else None
        grad_weight = (
            fast_filled(normalized_shape, 0.0, DType.float32, a._device)
            if mask[1]
            else None
        )
        grad_bias = (
            fast_filled(normalized_shape, 0.0, DType.float32, a._device)
            if mask[2]
            else None
        )
        return grad_input, grad_weight, grad_bias

    # All metadata validation is complete. The kernels consume dense row-major
    # buffers, so materialize arbitrary ATen views only now.
    grad = _tc(grad)
    a = _tc(a)
    saved_mean = _tc(saved_mean)
    saved_rstd = _tc(saved_rstd)
    # The affine-parameter reductions do not consume weight.  Materialize it
    # only for the grad-input kernel, which needs gamma in the dx formula.
    gamma = _tc(gamma) if gamma is not None and mask[0] else None

    grad_input = _alloc(a._shape, DType.float32, a._device) if mask[0] else None
    grad_weight = (
        _alloc(normalized_shape, DType.float32, a._device) if mask[1] else None
    )
    grad_bias = _alloc(normalized_shape, DType.float32, a._device) if mask[2] else None
    mask_bits = int(mask[0]) | (int(mask[1]) << 1) | (int(mask[2]) << 2)
    _call_mojo(
        _NormalizationBackwardExtension,
        "LayerNormBackwardF32",
        (
            grad_input._ptr if grad_input is not None else 0,
            grad_weight._ptr if grad_weight is not None else 0,
            grad_bias._ptr if grad_bias is not None else 0,
            grad._ptr,
            a._ptr,
            saved_mean._ptr,
            saved_rstd._ptr,
            gamma._ptr if gamma is not None else 0,
            rows,
            cols,
            mask_bits,
            _ctx_ptr(a._device),
        ),
        arg_dtypes=(
            grad._dtype,
            a._dtype,
            saved_mean._dtype,
            saved_rstd._dtype,
            gamma._dtype if gamma is not None else a._dtype,
        ),
        output_dtypes=(
            grad_input._dtype if grad_input is not None else a._dtype,
            grad_weight._dtype if grad_weight is not None else a._dtype,
            grad_bias._dtype if grad_bias is not None else a._dtype,
        ),
        flags={"OUTPUT_MASK": mask_bits},
        keepalive=(
            grad_input,
            grad_weight,
            grad_bias,
            grad,
            a,
            saved_mean,
            saved_rstd,
            gamma,
        ),
    )
    return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


_ROW_REDUCE_DTYPES = _FLOAT_DTYPES + (DType.int64, DType.int32)

# Dtypes any()/all() accept as input (the nonzero test works for all of them;
# the output is always bool). Matches reduction_ops' AnyRows/AllRows dispatch.
_ANYALL_DTYPES = _FLOAT_DTYPES + (
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
)


def _torch_dtype_to_max(dtype):
    from max.experimental.torch.torch import torch_dtype_to_max

    try:
        return torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        return None


def _norm_reduce_dims(dim, rank, empty_is_all):
    """Sorted unique normalized reduce dims, or None if the spec is invalid.

    `dim=None` always reduces every dim. An empty dim list reduces every dim
    when `empty_is_all` (sum/amax/amin/mean/var semantics) and nothing when
    not (any.dims/all.dims semantics). Duplicate or out-of-range dims -> None.
    """
    if dim is None:
        return list(range(rank))
    if isinstance(dim, int) and not isinstance(dim, bool):
        dims = [dim]
    elif isinstance(dim, list | tuple):
        dims = list(dim)
    else:
        return None
    if len(dims) == 0:
        return list(range(rank)) if empty_is_all else []
    seen = set()
    out = []
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool):
            return None
        if not -rank <= d < rank:
            return None
        d %= rank
        if d in seen:
            return None
        seen.add(d)
        out.append(d)
    return sorted(out)


def _reduced_shape(shape, reduce_dims, keepdim):
    """The reduction output shape (keepdim already applied)."""
    rset = set(reduce_dims)
    if keepdim:
        return tuple(1 if i in rset else s for i, s in enumerate(shape))
    return tuple(s for i, s in enumerate(shape) if i not in rset)


def fast_aten_mean(input, dim=None, keepdim=False, *, dtype=None):
    a = _t(input)
    if a is None:
        return NOT_HANDLED
    if dtype is not None:
        target = _torch_dtype_to_max(dtype)
        if target is None or target not in _FLOAT_DTYPES:
            return NOT_HANDLED
        if a._dtype != target:
            if a._dtype not in _CAST_DTYPES or target not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, target)
    if a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce("MeanSpec", a, rdims, keepdim, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


def fast_aten_sum(input, dim=None, keepdim=False, *, dtype=None):
    a = _t(input)
    if a is None:
        return NOT_HANDLED
    if dtype is not None:
        target = _torch_dtype_to_max(dtype)
        if target is None:
            return NOT_HANDLED
        if a._dtype != target:
            if a._dtype not in _CAST_DTYPES or target not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, target)
    elif not a._dtype.is_float():
        # torch promotes bool/integer sums to int64.
        if a._dtype != DType.int64:
            if a._dtype not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, DType.int64)
    if a._dtype not in (DType.float32, DType.float16, DType.bfloat16, DType.int64):
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce("SumSpec", a, rdims, keepdim)
    if result is not None:
        return result
    if a._numel == 0:
        # Empty reduction (the spec raises): torch defines it as 0.
        out_shape = _reduced_shape(a._shape, rdims, keepdim)
        filled = fast_filled(out_shape, 0, a._dtype, a._device)
        return NOT_HANDLED if filled is None else filled
    return NOT_HANDLED


def fast_aten_linalg_vector_norm(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    """FP32 L2 norm composed from existing eager elementwise/reduction ops."""
    input = _t(self)
    if (
        input is None
        or input._dtype != DType.float32
        or not isinstance(ord, int | float)
        or isinstance(ord, bool)
        or ord != 2
        or dtype is not None
        or not isinstance(keepdim, bool)
    ):
        return NOT_HANDLED

    squared = fast_aten_mul(input, input)
    if squared is NOT_HANDLED:
        return NOT_HANDLED
    summed = fast_aten_sum(squared, dim, keepdim)
    if summed is NOT_HANDLED:
        return NOT_HANDLED
    return fast_aten_sqrt(summed)


def _amax_amin(input, dim, keepdim, spec_name):
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce(spec_name, a, rdims, keepdim)
    return result if result is not None else NOT_HANDLED


def fast_aten_amax(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "AmaxSpec")


def fast_aten_amin(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "AminSpec")


def fast_aten_min(input):
    # Values-only full reduction: aten::min(Tensor) -> Tensor.
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    result = _try_spec_reduce("AminSpec", t, range(len(t._shape)), False)
    return result if result is not None else NOT_HANDLED


def fast_aten_min_dim(input, dim, keepdim=False):
    """aten::min.dim -> (values, indices) along `dim` (first-min-wins)."""
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(a._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    rdim = dim % rank
    result = _try_spec_min_dim(a, rdim, bool(keepdim))
    return result if result is not None else NOT_HANDLED


def _argreduce(input, dim, keepdim, is_min):
    a = _t(input)
    if a is None or a._numel == 0 or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = list(range(rank)) if dim is None else None
    if rdims is None and isinstance(dim, int) and rank > 0 and -rank <= dim < rank:
        rdims = [dim % rank]
    if rdims is not None:
        spec_name = "ArgminSpec" if is_min else "ArgmaxSpec"
        module_name = "reduction_ops" if is_min else "nn_ops"
        result = _try_spec_reduce(
            spec_name, a, rdims, keepdim, out_dtype=DType.int64, module_name=module_name
        )
        if result is not None:
            return result
    return NOT_HANDLED


def fast_aten_argmax(input, dim=None, keepdim=False):
    return _argreduce(input, dim, keepdim, is_min=False)


def fast_aten_argmin(input, dim=None, keepdim=False):
    return _argreduce(input, dim, keepdim, is_min=True)


def fast_aten_max(input, *args, **kwargs):
    # Only the values-only overload max(Tensor) -> Tensor.
    if args or kwargs:
        return NOT_HANDLED
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    result = _try_spec_reduce(
        "MaxSpec", t, range(len(t._shape)), False, module_name="nn_ops"
    )
    return result if result is not None else NOT_HANDLED


def fast_aten_var(input, dim=None, *, correction=1, keepdim=False):
    a = _t(input)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    if correction is None:
        correction = 1
    if not isinstance(correction, int | float) or isinstance(correction, bool):
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce("VarSpec", a, rdims, keepdim, float(correction))
    return result if result is not None else NOT_HANDLED


def _any_all(input, dim, keepdim, is_all):
    a = _t(input)
    if a is None or a._dtype not in _ANYALL_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    # Fast full-reduce bool path (the existing single-block AllBool/AnyBool).
    if dim is None and not keepdim and a._dtype == DType.bool:
        c = _tc(a)
        if 0 < c._numel < (1 << 22):
            out = _alloc((), DType.bool, a._device)
            _call_mojo(
                _NNExtension,
                "AllBool" if is_all else "AnyBool",
                (out._ptr, c._ptr, c._numel, _ctx_ptr(a._device)),
                arg_dtypes=(c._dtype,),
                output_dtypes=(out._dtype,),
                keepalive=(out, c),
            )
            return out
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=False)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce(
        "AllSpec" if is_all else "AnySpec", a, rdims, keepdim, out_dtype=DType.bool
    )
    return result if result is not None else NOT_HANDLED


def fast_aten_all(input, dim=None, keepdim=False):
    return _any_all(input, dim, keepdim, is_all=True)


def fast_aten_any(input, dim=None, keepdim=False):
    return _any_all(input, dim, keepdim, is_all=False)


def fast_aten__log_softmax(input, dim, half_to_float=False):
    t = _t(input)
    if (
        t is None
        or t._numel == 0
        or t._dtype not in _FLOAT_DTYPES
        or not isinstance(dim, int)
        or isinstance(dim, bool)
    ):
        return NOT_HANDLED
    # half_to_float: half input, float32 output (torch computes in fp32).
    if half_to_float:
        if t._dtype != DType.float16:
            return NOT_HANDLED
        t = _cast_tensor(t, DType.float32)
    rank = len(t._shape)
    if rank == 0:
        if dim not in (-1, 0):
            return NOT_HANDLED
        return fast_filled((), 0.0, t._dtype, t._device)
    if not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    if dim != rank - 1:
        # log_softmax(x, d) = log_softmax(x.transpose(d, -1), -1).T; both
        # transposes are zero-copy, the inner one materializes once.
        # (t is already fp32 here if half_to_float was set, so pass False.)
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__log_softmax(swapped, rank - 1, False)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    result = _try_spec_unary("LogSoftmaxSpec", t, module_name="reduction_ops")
    return result if result is not None else NOT_HANDLED


def fast_aten__log_softmax_backward_data(
    grad_output: object, output: object, dim: int, input_dtype: object
) -> object:
    # `object` params: besides real (TorchMojoTensor, torch.dtype) calls, the
    # composed path's lifetime contract is regression-tested with duck-typed
    # doubles (tests/test_autograd_memory_lifetime.py).
    grad = _t(grad_output)
    saved_output = _t(output)
    target_dtype = _torch_dtype_to_max(input_dtype)
    if (
        grad is None
        or saved_output is None
        or grad._shape != saved_output._shape
        or grad._dtype != saved_output._dtype
        or grad._device != saved_output._device
        or grad._dtype not in _FLOAT_DTYPES
        or target_dtype not in _FLOAT_DTYPES
        or not isinstance(dim, int)
        or isinstance(dim, bool)
    ):
        return NOT_HANDLED
    if grad._dtype != target_dtype and not (
        grad._dtype == DType.float32 and target_dtype == DType.float16
    ):
        return NOT_HANDLED

    rank = len(grad._shape)
    if rank > _MAX_RANK:
        return NOT_HANDLED
    if rank == 0:
        if dim not in (-1, 0):
            return NOT_HANDLED
        summed = grad
    else:
        if not -rank <= dim < rank:
            return NOT_HANDLED
        dim %= rank

        # ATen promises a fresh contiguous result. No kernel is needed when
        # there are no elements, and allocating directly also avoids reduction
        # specs whose empty-axis behavior varies across MAX releases.
        if grad._numel == 0:
            return _alloc(grad._shape, target_dtype, grad._device)

        # Fused single-kernel GPU path (softmax_backward_kernels.mojo): the
        # reduction dim is trailing, so any rank flattens to [rows, cols]
        # without a view. Everything else (non-trailing dim, the
        # f32-grad -> f16-target promotion, strided or unaligned tensors,
        # CPU device) keeps the composed sum -> exp -> addcmul path below.
        if (
            dim == rank - 1
            and isinstance(grad, TorchMojoTensor)
            and isinstance(saved_output, TorchMojoTensor)
            and grad._dtype == target_dtype
            and grad._dtype in _FLOAT_DTYPES
            and _on_gpu(grad)
            and grad._is_contiguous
            and saved_output._is_contiguous
            and grad._ptr % 16 == 0
            and saved_output._ptr % 16 == 0
        ):
            cols = grad._shape[dim]
            rows = grad._numel // cols
            if rows < 2**31:
                grad_input = _alloc(grad._shape, target_dtype, grad._device)
                if grad_input._ptr % 16 == 0:
                    _call_mojo(
                        _SoftmaxBackwardExtension,
                        "LogSoftmaxBackwardData",
                        (
                            grad_input._ptr,
                            grad._ptr,
                            saved_output._ptr,
                            rows,
                            cols,
                            grad._dtype.value,
                            _ctx_ptr(grad._device),
                        ),
                        arg_dtypes=(grad._dtype, saved_output._dtype),
                        output_dtypes=(grad_input._dtype,),
                        keepalive=(grad_input, grad, saved_output),
                    )
                    return grad_input
                # Unaligned fresh allocation (never expected): fall through
                # to the composed path rather than launching a misaligned
                # vector kernel.
                del grad_input

        summed = None

    work_grad = grad
    work_output = saved_output
    work_dim = dim
    restore_shape = None
    if rank > 4:
        work_grad = _tc(grad)
        work_output = _tc(saved_output)
        if work_grad is None or work_output is None:
            return NOT_HANDLED
        flat_shape = (
            math.prod(grad._shape[:dim]),
            grad._shape[dim],
            math.prod(grad._shape[dim + 1 :]),
        )
        work_grad = fast_aten_view(work_grad, flat_shape)
        work_output = fast_aten_view(work_output, flat_shape)
        if work_grad is NOT_HANDLED or work_output is NOT_HANDLED:
            return NOT_HANDLED
        work_dim = 1
        restore_shape = grad._shape

    if summed is None:
        summed = fast_aten_sum(work_grad, dim=[work_dim], keepdim=True)
        if summed is NOT_HANDLED:
            return NOT_HANDLED
    probabilities = fast_aten_exp(work_output)
    if probabilities is NOT_HANDLED:
        return NOT_HANDLED

    grad_input = fast_aten_addcmul(work_grad, probabilities, summed, value=-1.0)
    if grad_input is NOT_HANDLED:
        return NOT_HANDLED

    # The launches above have captured their inputs on this device stream.
    # Release the vocabulary-sized temporaries promptly so the stream-ordered
    # allocator can recycle them without introducing a host synchronization.
    del probabilities, summed
    if restore_shape is not None:
        grad_input = fast_aten_view(grad_input, restore_shape)
        if grad_input is NOT_HANDLED:
            return NOT_HANDLED
    if grad_input._dtype != target_dtype:
        grad_input = _cast_tensor(grad_input, target_dtype)
    return grad_input


def _nll_loss_inputs(self, target, weight, reduction, ignore_index):
    """Validate the f32/i64 two-dimensional NLL kernel contract.

    This stays enqueue-only, so it cannot inspect target values on the host.
    The current kernel contract therefore assumes each label is either in
    ``[0, classes)`` or exactly ``ignore_index``; a future device-side assert
    is needed to match PyTorch's error for other labels without synchronizing.
    """
    log_probs = _t(self)
    labels = _t(target)
    if (
        weight is not None
        or not isinstance(reduction, int)
        or isinstance(reduction, bool)
        or reduction not in (0, 1, 2)
        or not isinstance(ignore_index, int)
        or isinstance(ignore_index, bool)
        or log_probs is None
        or labels is None
        or not _on_gpu(log_probs)
        or log_probs._dtype != DType.float32
        or not log_probs._is_contiguous
        or len(log_probs._shape) != 2
    ):
        return None

    rows, classes = log_probs._shape
    if (
        classes <= 0
        or labels._dtype != DType.int64
        or labels._device != log_probs._device
        or tuple(labels._shape) != (rows,)
    ):
        return None
    return log_probs, labels, rows, classes


def fast_aten_nll_loss_forward_output(
    self, target, weight, reduction, ignore_index, *, output, total_weight
):
    """Direct out= NLL forward for the pure-Mojo f32 GPU kernel."""
    inputs = _nll_loss_inputs(self, target, weight, reduction, ignore_index)
    if inputs is None:
        return NOT_HANDLED
    log_probs, labels, rows, classes = inputs
    if (
        output is total_weight
        or not _valid_nll_out(output, DType.float32, log_probs._device)
        or not _valid_nll_out(total_weight, DType.float32, log_probs._device)
    ):
        return NOT_HANDLED

    output_shape = (rows,) if reduction == 0 else ()
    # Every supplied out is known-valid before either can be resized.
    write_output = _prepare_nll_out(output, output_shape)
    write_total_weight = _prepare_nll_out(total_weight, ())

    if rows == 0:
        # Avoid a zero-grid launch. PyTorch defines empty reduced NLL as NaN
        # for mean and zero for sum; reduction=none already has no elements.
        fast_aten_fill__scalar(write_total_weight, 0.0)
        if reduction != 0:
            fast_aten_fill__scalar(write_output, math.nan if reduction == 1 else 0.0)
    else:
        labels_c = _tc(labels)
        _call_mojo(
            _LossExtension,
            "NllLossForwardF32",
            (
                write_output._ptr,
                write_total_weight._ptr,
                log_probs._ptr,
                labels_c._ptr,
                rows,
                classes,
                reduction,
                ignore_index,
                _ctx_ptr(log_probs._device),
            ),
            arg_dtypes=(log_probs._dtype, labels_c._dtype),
            output_dtypes=(write_output._dtype, write_total_weight._dtype),
            flags={"REDUCTION": reduction},
            keepalive=(write_output, write_total_weight, log_probs, labels_c),
        )

    if write_output is not output:
        _copy_into(output, write_output)
    if write_total_weight is not total_weight:
        _copy_into(total_weight, write_total_weight)
    return output, total_weight


def fast_aten_nll_loss_backward_grad_input(
    grad_output,
    self,
    target,
    weight,
    reduction,
    ignore_index,
    total_weight,
    *,
    grad_input,
):
    """Direct out= NLL backward for the pure-Mojo f32 GPU kernel."""
    inputs = _nll_loss_inputs(self, target, weight, reduction, ignore_index)
    if inputs is None:
        return NOT_HANDLED
    log_probs, labels, rows, classes = inputs
    grad = _t(grad_output)
    weight_sum = _t(total_weight)
    expected_grad_shape = (rows,) if reduction == 0 else None
    if (
        grad is None
        or grad._dtype != DType.float32
        or grad._device != log_probs._device
        or (
            tuple(grad._shape) != expected_grad_shape
            if expected_grad_shape is not None
            else tuple(grad._shape) not in ((), (1,))
        )
        or weight_sum is None
        or weight_sum._dtype != DType.float32
        or weight_sum._device != log_probs._device
        or weight_sum._numel != 1
        or not _valid_nll_out(grad_input, DType.float32, log_probs._device)
    ):
        return NOT_HANDLED

    # Validation above completes before the sanctioned out= resize.
    write_grad_input = _prepare_nll_out(grad_input, log_probs._shape)
    if rows != 0:
        labels_c = _tc(labels)
        grad_c = _tc(grad)
        weight_sum_c = _tc(weight_sum)
        _call_mojo(
            _LossExtension,
            "NllLossBackwardF32",
            (
                write_grad_input._ptr,
                grad_c._ptr,
                labels_c._ptr,
                weight_sum_c._ptr,
                rows,
                classes,
                reduction,
                ignore_index,
                _ctx_ptr(log_probs._device),
            ),
            arg_dtypes=(grad_c._dtype, labels_c._dtype, weight_sum_c._dtype),
            output_dtypes=(write_grad_input._dtype,),
            flags={"REDUCTION": reduction},
            keepalive=(write_grad_input, grad_c, labels_c, weight_sum_c),
        )

    if write_grad_input is not grad_input:
        _copy_into(grad_input, write_grad_input)
    return grad_input


def fast_aten_cumsum(input, dim, *, dtype=None):
    a = _t(input) if dtype is None else None
    if (
        a is None
        or a._numel == 0
        or a._dtype not in (DType.int64, DType.int32, DType.float32)
    ):
        return NOT_HANDLED
    rank = len(a._shape)
    if not isinstance(dim, int) or rank == 0 or dim % rank != rank - 1:
        return NOT_HANDLED
    result = _try_spec_unary("CumsumSpec", a, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _pair(x) -> tuple[int, int] | None:
    if isinstance(x, int):
        return (x, x)
    if (
        isinstance(x, list | tuple)
        and len(x) == 2
        and all(isinstance(i, int) for i in x)
    ):
        return (x[0], x[1])
    if isinstance(x, list | tuple) and len(x) == 1 and isinstance(x[0], int):
        return (x[0], x[0])
    return None


def fast_aten_max_pool2d_with_indices(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    a = _tc(input)
    kernel = _pair(kernel_size)
    strides = _pair(stride) if stride not in (None, []) else kernel
    pads = _pair(padding)
    dils = _pair(dilation)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and not ceil_mode
        and None not in (kernel, strides, pads, dils)
    ):
        n, c, in_h, in_w = a._shape
        kh, kw = kernel
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            indices = _alloc((n, c, out_h, out_w), DType.int64, a._device)
            _call_mojo(
                _NNExtension,
                "MaxPool2dWithIndices",
                (
                    out._ptr,
                    indices._ptr,
                    a._ptr,
                    (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, n * c),
                    a._dtype.value,
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype,),
                output_dtypes=(out._dtype, indices._dtype),
                keepalive=(out, indices, a),
            )
            return out, indices
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# NN tail: average / adaptive-average pool, group norm, bilinear upsample, and
# the lowered SDPA variants (flash / efficient / math), all inference forward.
# ---------------------------------------------------------------------------


def fast_aten_avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    a = _tc(input)
    kernel = _pair(kernel_size)
    strides = _pair(stride) if stride not in (None, []) else kernel
    pads = _pair(padding)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and not ceil_mode
        and None not in (kernel, strides, pads)
        and (
            divisor_override is None
            or (isinstance(divisor_override, int) and divisor_override != 0)
        )
    ):
        n, c, in_h, in_w = a._shape
        kh, kw = kernel
        sh, sw = strides
        ph, pw = pads
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            div = divisor_override if divisor_override is not None else 0
            _call_mojo(
                _NNExtension,
                "AvgPool2d",
                (
                    out._ptr,
                    a._ptr,
                    (
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        1 if count_include_pad else 0,
                        div,
                        n * c,
                    ),
                    a._dtype.value,
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype,),
                output_dtypes=(out._dtype,),
                flags={
                    "COUNT_INCLUDE_PAD": count_include_pad,
                    "HAS_DIVISOR_OVERRIDE": divisor_override is not None,
                },
                keepalive=(out, a),
            )
            return out
    return NOT_HANDLED


def fast_aten__adaptive_avg_pool2d(input, output_size):
    a = _tc(input)
    osize = _pair(output_size)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and osize is not None
    ):
        n, c, in_h, in_w = a._shape
        out_h, out_w = osize
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            _call_mojo(
                _NNExtension,
                "AdaptiveAvgPool2d",
                (
                    out._ptr,
                    a._ptr,
                    (in_h, in_w, out_h, out_w, n * c),
                    a._dtype.value,
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype,),
                output_dtypes=(out._dtype,),
                keepalive=(out, a),
            )
            return out
    return NOT_HANDLED


def fast_aten_native_group_norm(input, weight, bias, N, C, HxW, group, eps):
    a = _tc(input)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and isinstance(group, int)
        and group > 0
        and isinstance(C, int)
        and C % group == 0
        and a._numel == N * C * HxW
    ):
        cpg = C // group
        cols = cpg * HxW
        rows = N * group
        gamma = (
            _tc(weight)
            if weight is not None
            else fast_filled((C,), 1.0, a._dtype, a._device)
        )
        beta = (
            _tc(bias)
            if bias is not None
            else fast_filled((C,), 0.0, a._dtype, a._device)
        )
        if (
            gamma is None
            or beta is None
            or gamma._dtype != a._dtype
            or beta._dtype != a._dtype
            or tuple(gamma._shape) != (C,)
            or tuple(beta._shape) != (C,)
        ):
            return NOT_HANDLED
        # Not spec-converted for the same reason as native_layer_norm
        # above (three-group result construction outweighs the thin
        # classic prologue).
        out = _alloc(a._shape, a._dtype, a._device)
        mean = _alloc((N, group), DType.float32, a._device)
        rstd = _alloc((N, group), DType.float32, a._device)
        _call_mojo(
            _NNExtension,
            "GroupNorm",
            (
                out._ptr,
                mean._ptr,
                rstd._ptr,
                a._ptr,
                gamma._ptr,
                beta._ptr,
                (float(eps), rows, cols, HxW, group, cpg),
                a._dtype.value,
                _ctx_ptr(a._device),
            ),
            arg_dtypes=(a._dtype, gamma._dtype, beta._dtype),
            output_dtypes=(out._dtype, mean._dtype, rstd._dtype),
            flags={"HAS_WEIGHT": weight is not None, "HAS_BIAS": bias is not None},
            keepalive=(out, mean, rstd, a, gamma, beta),
        )
        return out, mean, rstd
    return NOT_HANDLED


def _area_pixel_scale(in_size, out_size, align_corners, scale):
    """torch area_pixel_compute_scale for one axis."""
    if align_corners:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return in_size / out_size


def fast_aten_upsample_bilinear2d(
    input, output_size, align_corners, scales_h=None, scales_w=None
):
    a = _tc(input)
    osize = _pair(output_size)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and osize is not None
    ):
        n, c, in_h, in_w = a._shape
        out_h, out_w = osize
        if out_h > 0 and out_w > 0:
            ratio_h = _area_pixel_scale(in_h, out_h, align_corners, scales_h)
            ratio_w = _area_pixel_scale(in_w, out_w, align_corners, scales_w)
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            _call_mojo(
                _NNExtension,
                "UpsampleBilinear2d",
                (
                    out._ptr,
                    a._ptr,
                    (
                        float(ratio_h),
                        float(ratio_w),
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        n * c,
                        1 if align_corners else 0,
                    ),
                    a._dtype.value,
                    _ctx_ptr(a._device),
                ),
                arg_dtypes=(a._dtype,),
                output_dtypes=(out._dtype,),
                flags={"ALIGN_CORNERS": bool(align_corners)},
                keepalive=(out, a),
            )
            return out
    return NOT_HANDLED


# Causal regimes of `matmul_ops.CausalBmm`; see the CAUSAL_* aliases there.
SDPA_CAUSAL_NONE = 0
SDPA_CAUSAL_OUT = 1
SDPA_CAUSAL_A_ROWS = 2
SDPA_CAUSAL_B_COLS = 3


def _try_sdpa_causal_bmm(
    a: MojoTensorLike, b: MojoTensorLike, transpose_b: bool, causal_mode: int
) -> object:
    """Batched GEMM that skips the contraction indices a causal mask kills.

    ``a`` is ``(batch, m, k)`` and ``b`` is ``(batch, n, k)`` when
    ``transpose_b`` else ``(batch, k, n)``; both must be dense row-major.
    Returns ``None`` when the operands do not qualify, so the caller keeps its
    ordinary dense chain.

    ``SDPA_CAUSAL_OUT`` leaves the masked half of the output unwritten, so only
    ask for it when the consumer reads each row's live prefix.  The other two
    regimes are exact for any consumer: the contraction indices they skip
    multiply exact zeros.
    """
    if (
        not _on_gpu(a)
        or a._dtype != b._dtype
        or a._dtype not in (DType.float32, DType.bfloat16)
        or a._device != b._device
        or len(a._shape) != 3
        or len(b._shape) != 3
        or not a._is_contiguous
        or not b._is_contiguous
        or a._shape[0] != b._shape[0]
    ):
        return None
    batch, m, k = a._shape
    n = b._shape[1] if transpose_b else b._shape[2]
    inner = b._shape[2] if transpose_b else b._shape[1]
    mode_supported = causal_mode == SDPA_CAUSAL_OUT or (
        not transpose_b and causal_mode in (SDPA_CAUSAL_A_ROWS, SDPA_CAUSAL_B_COLS)
    )
    if (
        inner != k
        or batch <= 1
        or m < 64
        or n < 64
        or n % 32 != 0
        or k % 32 != 0
        or a._device.architecture_name != "gfx942"
        or not mode_supported
    ):
        return None
    out = _alloc((batch, m, n), a._dtype, a._device)
    _call_mojo(
        _MatmulExtension,
        "CausalBmm",
        (
            out._ptr,
            a._ptr,
            b._ptr,
            (batch, m, n, k, int(bool(transpose_b)), int(causal_mode)),
            a._dtype.value,
            _ctx_ptr(a._device),
        ),
        arg_dtypes=(a._dtype, b._dtype),
        output_dtypes=(out._dtype,),
        flags={"TRANSPOSE_B": transpose_b, "CAUSAL_MODE": causal_mode},
        keepalive=(out, a, b),
    )
    return out


def _sdpa_math_forward_with_dropout(query, key, value, is_causal, scale, dropout_p):
    """Decomposed SDPA returning output, pre-dropout probabilities, and mask.

    Dropout is deliberately composed between softmax and the value BMM.  The
    returned probabilities stay pre-dropout because softmax backward needs
    values at every position, including positions dropped in the forward.
    """
    q = _t(query)
    k = _t(key)
    v = _t(value)
    if (
        q is None
        or k is None
        or v is None
        or q._device != k._device
        or q._device != v._device
        or q._dtype != k._dtype
        or q._dtype != v._dtype
        or q._dtype not in _FLOAT_DTYPES
        or len(q._shape) != 4
        or tuple(k._shape) != tuple(v._shape)
        or tuple(q._shape[:2]) != tuple(k._shape[:2])
        or q._shape[3] != k._shape[3]
        or 0 in q._shape
        or 0 in k._shape
    ):
        return NOT_HANDLED
    if not isinstance(dropout_p, int | float) or not 0.0 <= dropout_p <= 1.0:
        return NOT_HANDLED
    dropout_p = float(dropout_p)
    if dropout_p != 0.0 and (q._dtype != DType.float32 or not _on_gpu(q)):
        return NOT_HANDLED

    q = _tc(q)
    k = _tc(k)
    v = _tc(v)
    b, h, q_len, head_dim = q._shape
    kv_len = k._shape[2]
    scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    ctx = _ctx_ptr(q._device)
    dt = q._dtype.value
    q3_shape = (b * h, q_len, head_dim)
    kv3_shape = (b * h, kv_len, head_dim)
    q3 = _view_of(q, q3_shape, _row_major_strides(q3_shape), q._offset, contiguous=True)
    k3 = _view_of(
        k, kv3_shape, _row_major_strides(kv3_shape), k._offset, contiguous=True
    )
    v3 = _view_of(
        v, kv3_shape, _row_major_strides(kv3_shape), v._offset, contiguous=True
    )
    # Apple f32 causal specialization: BmmCausalF32 mode 1 skips (and leaves
    # unwritten) score tiles strictly above the diagonal, which is safe only
    # because the Metal SoftmaxRows/SoftmaxRowsDropoutF32 kernels never read
    # past the causal boundary; mode 2 cuts the P @ V reduction at the
    # boundary, where P's columns are exactly zero.
    on_metal = q._dtype == DType.float32 and getattr(q._device, "api", None) == "metal"
    metal_causal = bool(is_causal) and on_metal
    if metal_causal:
        scores = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
        _call_mojo(
            _MatmulExtension,
            "BmmCausalF32",
            (
                scores._ptr,
                q._ptr,
                k._ptr,
                (b * h, q_len, kv_len, head_dim, 1, 1),
                dt,
                ctx,
            ),
            arg_dtypes=(q._dtype, k._dtype),
            output_dtypes=(scores._dtype,),
            flags={"TRANSPOSE_B": True, "CAUSAL_MODE": SDPA_CAUSAL_OUT},
            keepalive=(scores, q, k),
        )
    else:
        # Off Metal, SDPA_CAUSAL_OUT would skip the fully masked output tiles
        # of the score matrix here, and it is correct, but it is measurably
        # *not* worth it: at these shapes the batched GEMM is bound by
        # workgroup dispatch rather than by the work inside a workgroup, so
        # the 44% of tiles it removes buy nothing (895.05 -> 901.77 us,
        # against 488.42 us for a dense half-width control with the same
        # live-tile count). See optimization_journal.md, experiment AA.
        scores = _try_bf16_bmm(q3, k3, transpose_b=True)
        if scores is None:
            scores = _try_tf32_bmm(q3, k3, transpose_b=True)
        if scores is None:
            scores = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
            _call_mojo(
                _MatmulExtension,
                "Bmm",
                (
                    scores._ptr,
                    q._ptr,
                    k._ptr,
                    (b * h, q_len, kv_len, head_dim, 1),
                    dt,
                    ctx,
                ),
                arg_dtypes=(q._dtype, k._dtype),
                output_dtypes=(scores._dtype,),
                flags={"TRANSPOSE_B": True},
                keepalive=(scores, q, k),
            )
    probs = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
    # Fused softmax + dropout (Apple f32): one launch writes the pre-dropout
    # probabilities, the dropped/rescaled probabilities, and the keep-mask,
    # consuming the exact same Philox interval as the composed
    # SoftmaxRows + NativeDropoutF32 path (byte-identical outputs).
    fused_dropout = None
    if (
        0.0 < dropout_p < 1.0
        and on_metal
        and kv_len % 4 == 0
        and kv_len <= 1024
        and probs._ptr % 16 == 0
        and scores._ptr % 16 == 0
    ):
        pdrop = _alloc((b * h, q_len, kv_len), DType.float32, q._device)
        drop_mask = _alloc((b * h, q_len, kv_len), DType.bool, q._device)
        if pdrop._ptr % 16 == 0 and drop_mask._ptr % 4 == 0:
            fused_dropout = (pdrop, drop_mask)
    if fused_dropout is not None:
        pdrop, drop_mask = fused_dropout
        numel = b * h * q_len * kv_len
        seed, base_offset = _reserve_philox_state(q._torch_device, (numel + 3) // 4)
        word_mask = (1 << 32) - 1
        _call_mojo(
            _NNExtension,
            "SoftmaxRowsDropoutF32",
            (
                probs._ptr,
                pdrop._ptr,
                drop_mask._ptr,
                scores._ptr,
                b * h * q_len,
                kv_len,
                float(scale_val),
                1 if is_causal else 0,
                q_len,
                float(dropout_p),
                seed & word_mask,
                (seed >> 32) & word_mask,
                base_offset & word_mask,
                (base_offset >> 32) & word_mask,
                ctx,
            ),
            arg_dtypes=(scores._dtype,),
            output_dtypes=(probs._dtype, pdrop._dtype, drop_mask._dtype),
            flags={"CAUSAL": bool(is_causal)},
            keepalive=(probs, pdrop, drop_mask, scores),
        )
        del scores
        effective_probs = pdrop
        dropout_mask = drop_mask
    else:
        _call_mojo(
            _NNExtension,
            "SoftmaxRows",
            (
                probs._ptr,
                scores._ptr,
                b * h * q_len,
                kv_len,
                float(scale_val),
                1 if is_causal else 0,
                q_len,
                dt,
                ctx,
            ),
            arg_dtypes=(scores._dtype,),
            output_dtypes=(probs._dtype,),
            flags={"CAUSAL": bool(is_causal)},
            keepalive=(probs, scores),
        )
        # All allocations use stream-ordered lifetime management, so releasing
        # the host reference here cannot recycle scores before SoftmaxRows
        # consumes it.
        del scores
        effective_probs = probs
        dropout_mask = None
    if fused_dropout is not None:
        pass
    elif dropout_p == 1.0:
        # SDPA's math implementation composes ``torch.dropout`` rather than
        # exposing CUDA native_dropout directly.  At the full-drop endpoint
        # that is arithmetic ``P * 0`` semantics: nonfinite probabilities
        # remain nonfinite instead of being overwritten by zeros.  The mask
        # is still saved so backward applies the same mask/scale arithmetic,
        # and neither path reserves RNG state at this endpoint.
        effective_probs = fast_aten_mul(probs, 0.0)
        dropout_mask = fast_filled(probs._shape, False, DType.bool, probs._device)
        if effective_probs is NOT_HANDLED or dropout_mask is None:
            return NOT_HANDLED
    elif dropout_p > 0.0:
        dropout_result = fast_aten_native_dropout(probs, dropout_p, True)
        if dropout_result is NOT_HANDLED:
            return NOT_HANDLED
        effective_probs, dropout_mask = dropout_result
        del dropout_result

    if metal_causal:
        out = _alloc((b * h, q_len, head_dim), q._dtype, q._device)
        _call_mojo(
            _MatmulExtension,
            "BmmCausalF32",
            (
                out._ptr,
                effective_probs._ptr,
                v._ptr,
                (b * h, q_len, head_dim, kv_len, 0, 2),
                dt,
                ctx,
            ),
            arg_dtypes=(effective_probs._dtype, v._dtype),
            output_dtypes=(out._dtype,),
            flags={"TRANSPOSE_B": False, "CAUSAL_MODE": SDPA_CAUSAL_A_ROWS},
            keepalive=(out, effective_probs, v),
        )
    else:
        out = None
        if is_causal:
            # P is exactly zero above the diagonal, so output row block r
            # only needs contraction indices below its last row.  Exact for
            # any consumer.
            out = _try_sdpa_causal_bmm(effective_probs, v3, False, SDPA_CAUSAL_A_ROWS)
        if out is None:
            out = _try_bf16_bmm(effective_probs, v3)
        if out is None:
            out = _try_tf32_bmm(effective_probs, v3)
        if out is None:
            out = _alloc((b * h, q_len, head_dim), q._dtype, q._device)
            _call_mojo(
                _MatmulExtension,
                "Bmm",
                (
                    out._ptr,
                    effective_probs._ptr,
                    v._ptr,
                    (b * h, q_len, head_dim, kv_len, 0),
                    dt,
                    ctx,
                ),
                arg_dtypes=(effective_probs._dtype, v._dtype),
                output_dtypes=(out._dtype,),
                flags={"TRANSPOSE_B": False},
                keepalive=(out, effective_probs, v),
            )
    # P_drop is not saved: backward cheaply reconstructs it from P and the bool
    # mask, avoiding one persistent f32 (B,H,L,S) allocation per layer.
    del effective_probs

    out_shape = (b, h, q_len, head_dim)
    out4 = _view_of(
        out, out_shape, _row_major_strides(out_shape), out._offset, contiguous=True
    )
    probs_shape = (b, h, q_len, kv_len)
    probs4 = _view_of(
        probs, probs_shape, _row_major_strides(probs_shape), probs._offset
    )
    mask4 = None
    if dropout_mask is not None:
        mask4 = _view_of(
            dropout_mask,
            probs_shape,
            _row_major_strides(probs_shape),
            dropout_mask._offset,
            contiguous=True,
        )
    return out4, probs4, mask4


def _sdpa_masked_math_forward(query, key, value, attn_mask, is_causal, scale):
    """Decomposed SDPA with an explicit attention mask -> ``(output, probs)``.

    ATen turns the mask into an additive bias on the *already scaled* scores: a
    float mask is added as-is, a bool mask keeps its ``True`` positions and
    drives the rest to ``-inf`` before the row softmax.  Both forms broadcast up
    to ``(batch, heads, query, key)``, including the per-head ``(H, L, S)`` and
    per-row ``(L, S)`` shapes, so the bias is applied on the 4-D view of the
    score matrix rather than the folded 3-D one the batched GEMM works on.

    ``is_causal`` is rejected because ATen itself refuses an explicit mask
    together with causal masking, so the combination has no reference to match.
    """
    q = _t(query)
    k = _t(key)
    v = _t(value)
    mask = _t(attn_mask)
    if (
        q is None
        or k is None
        or v is None
        or mask is None
        or is_causal
        or q._device != k._device
        or q._device != v._device
        or q._device != mask._device
        or q._dtype != k._dtype
        or q._dtype != v._dtype
        or q._dtype not in _FLOAT_DTYPES
        or len(q._shape) != 4
        or tuple(k._shape) != tuple(v._shape)
        or tuple(q._shape[:2]) != tuple(k._shape[:2])
        or q._shape[3] != k._shape[3]
        or 0 in q._shape
        or 0 in k._shape
    ):
        return NOT_HANDLED
    # A promoting mask dtype would widen the result behind the caller's back,
    # so only an exact float bias or a bool keep-mask is served here.
    if mask._dtype != DType.bool and mask._dtype != q._dtype:
        return NOT_HANDLED

    b, h, q_len, head_dim = q._shape
    kv_len = k._shape[2]
    q = _tc(q)
    k = _tc(k)
    v = _tc(v)
    q3_shape = (b * h, q_len, head_dim)
    kv3_shape = (b * h, kv_len, head_dim)
    q3 = _view_of(q, q3_shape, _row_major_strides(q3_shape), q._offset, contiguous=True)
    k3 = _view_of(
        k, kv3_shape, _row_major_strides(kv3_shape), k._offset, contiguous=True
    )
    v3 = _view_of(
        v, kv3_shape, _row_major_strides(kv3_shape), v._offset, contiguous=True
    )
    scores3 = _fast_aten_bmm_transpose_b(q3, k3)
    if scores3 is NOT_HANDLED:
        return NOT_HANDLED
    score_shape = (b, h, q_len, kv_len)
    scores4 = _view_of(
        scores3, score_shape, _row_major_strides(score_shape), scores3._offset
    )
    scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    scaled = fast_aten_mul(scores4, scale_val)
    del scores3, scores4
    if scaled is NOT_HANDLED:
        return NOT_HANDLED
    if mask._dtype == DType.bool:
        biased = fast_aten_where(mask, scaled, float("-inf"))
    else:
        biased = fast_aten_add(scaled, mask)
    del scaled
    if biased is NOT_HANDLED:
        return NOT_HANDLED
    # A mask that broadcasts to anything other than the score matrix is not a
    # valid SDPA mask; the elementwise kernels would silently expand instead.
    if tuple(biased._shape) != score_shape:
        return NOT_HANDLED
    probs4 = fast_aten__softmax(biased, -1, False)
    del biased
    if probs4 is NOT_HANDLED or not probs4._is_contiguous:
        return NOT_HANDLED
    probs3_shape = (b * h, q_len, kv_len)
    probs3 = _view_of(
        probs4,
        probs3_shape,
        _row_major_strides(probs3_shape),
        probs4._offset,
        contiguous=True,
    )
    out3 = fast_aten_bmm(probs3, v3)
    del probs3, q3, k3, v3
    if out3 is NOT_HANDLED:
        return NOT_HANDLED
    out_shape = (b, h, q_len, head_dim)
    out4 = _view_of(
        out3, out_shape, _row_major_strides(out_shape), out3._offset, contiguous=True
    )
    return out4, probs4


def _fa4_bf16_d64_causal_inputs(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Return eligible public BHTD inputs without doing any device work."""
    q = _t(query)
    k = _t(key)
    v = _t(value)
    if (
        q is None
        or k is None
        or v is None
        or attn_mask is not None
        or enable_gqa
        or not isinstance(dropout_p, int | float)
        or isinstance(dropout_p, bool)
        or float(dropout_p) != 0.0
        or is_causal is not True
        or q._device != k._device
        or q._device != v._device
        or q._device.api != "cuda"
        or q._device.architecture_name != "sm_90a"
        or q._dtype != DType.bfloat16
        or k._dtype != DType.bfloat16
        or v._dtype != DType.bfloat16
        or len(q._shape) != 4
        or tuple(q._shape) != tuple(k._shape)
        or tuple(q._shape) != tuple(v._shape)
    ):
        return None
    batch, heads, seqlen, head_dim = q._shape
    if batch <= 0 or heads <= 0 or seqlen <= 0 or seqlen % 128 != 0 or head_dim != 64:
        return None
    if scale is not None and (
        not isinstance(scale, int | float)
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
    ):
        return None
    return q, k, v


def _fa4_strided_bthd_layout(tensor) -> bool:
    """Whether a physical BTHD view is safe for FA4's strided TMA ABI.

    The pointer is already adjusted to logical ``[0, 0, 0, 0]`` and strides
    are element strides.  Keep this predicate in lockstep with the defensive
    validation in the Mojo bridge so an unsupported view is materialized
    before any FA4 launch is selected.
    """
    if (
        tensor is None
        or tensor._dtype != DType.bfloat16
        or tensor._itemsize != DType.bfloat16.size_in_bytes
        or len(tensor._shape) != 4
        or len(tensor._strides) != 4
    ):
        return False
    batch, seqlen, heads, head_dim = tensor._shape
    b_stride, s_stride, h_stride, d_stride = tensor._strides
    if (
        batch <= 0
        or seqlen <= 0
        or heads <= 0
        or seqlen % 128 != 0
        or head_dim != 64
        or min(b_stride, s_stride, h_stride, d_stride) <= 0
        or d_stride != 1
        or h_stride != 64
        or s_stride < heads * 64
        or b_stride != seqlen * s_stride
        or tensor._ptr % 16 != 0
    ):
        return False
    return all(
        stride * tensor._itemsize % 16 == 0 for stride in (b_stride, s_stride, h_stride)
    )


def _fa4_native_bthd(tensor):
    """Expose public BHTD storage as FA4-native BTHD, copying if required."""
    physical = fast_aten_transpose(tensor, 1, 2)
    if physical is NOT_HANDLED:
        return None
    if _fa4_strided_bthd_layout(physical):
        return physical
    return _tc(physical)


def _fa4_prepare_qkv_bridge(q_native, k_native, v_native):
    """Return prepared Q/K/V plus whether their strided ABI is required."""
    qkv = (q_native, k_native, v_native)
    needs_strided_bridge = any(not tensor._is_contiguous for tensor in qkv)
    if needs_strided_bridge and all(_fa4_strided_bthd_layout(tensor) for tensor in qkv):
        return qkv, True
    if needs_strided_bridge:
        # Defensive path for callers of the private backward helper.  Normal
        # forward preparation has already copied every unsupported view.
        qkv = tuple(_tc(tensor) for tensor in qkv)
        if any(tensor is None for tensor in qkv):
            return None
    return qkv, False


def fast_fa4_bf16_d64_causal_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Run vendored FA4 and return output/LSE plus saved physical inputs.

    Loading the bridge happens before materialization or allocation, so a
    packaging/compiler error cannot leave unnecessary device work queued.
    """
    inputs = _fa4_bf16_d64_causal_inputs(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    if inputs is None:
        return NOT_HANDLED

    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    fa4_ops = load_fa4_ops()
    q, k, v = inputs
    q_native = _fa4_native_bthd(q)
    k_native = _fa4_native_bthd(k)
    v_native = _fa4_native_bthd(v)
    if q_native is None or k_native is None or v_native is None:
        return NOT_HANDLED
    prepared = _fa4_prepare_qkv_bridge(q_native, k_native, v_native)
    if prepared is None:
        return NOT_HANDLED
    (q_native, k_native, v_native), use_strided_qkv = prepared

    batch, heads, seqlen, head_dim = q._shape
    physical_shape = (batch, seqlen, heads, head_dim)
    out_native = _alloc(physical_shape, DType.bfloat16, q._device)
    logsumexp = _alloc((batch, heads, seqlen), DType.float32, q._device)
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    if use_strided_qkv:
        _device_call(
            fa4_ops.flash_attention_fwd_bf16_d64_causal_strided_qkv,
            q_native._ptr,
            *q_native._strides,
            k_native._ptr,
            *k_native._strides,
            v_native._ptr,
            *v_native._strides,
            out_native._ptr,
            logsumexp._ptr,
            batch,
            seqlen,
            heads,
            scale_value,
            _ctx_ptr(q._device),
            keepalive=(q_native, k_native, v_native, out_native, logsumexp),
        )
    else:
        _device_call(
            fa4_ops.flash_attention_fwd_bf16_d64_causal,
            q_native._ptr,
            k_native._ptr,
            v_native._ptr,
            out_native._ptr,
            logsumexp._ptr,
            batch,
            seqlen,
            heads,
            scale_value,
            _ctx_ptr(q._device),
            keepalive=(q_native, k_native, v_native, out_native, logsumexp),
        )
    output = fast_aten_transpose(out_native, 1, 2)
    if output is NOT_HANDLED:
        raise RuntimeError("FA4 output could not be exposed as a BHTD view")
    return output, logsumexp, q_native, k_native, v_native


def fast_fa4_bf16_d64_causal_backward(
    q_native, k_native, v_native, output, logsumexp, grad_output, scale
):
    """Enqueue FA4 preprocess/main/convert and return public BHTD grads."""
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    fa4_ops = load_fa4_ops()
    batch, seqlen, heads, head_dim = q_native._shape
    prepared = _fa4_prepare_qkv_bridge(q_native, k_native, v_native)
    if prepared is None:
        return NOT_HANDLED
    (q_native, k_native, v_native), use_strided_qkv = prepared
    # The new ABI broadens only Q/K/V.  Output and dO retain the original
    # contiguous BTHD contract even if a caller supplies a descriptor-safe
    # gapped public view.
    out_native = _fa4_native_bthd(output)
    dout_native = _fa4_native_bthd(grad_output)
    if out_native is not None and not out_native._is_contiguous:
        out_native = _tc(out_native)
    if dout_native is not None and not dout_native._is_contiguous:
        dout_native = _tc(dout_native)
    if out_native is None or dout_native is None:
        return NOT_HANDLED

    physical_shape = (batch, seqlen, heads, head_dim)
    dq_native = _alloc(physical_shape, DType.bfloat16, q_native._device)
    dk_native = _alloc(physical_shape, DType.bfloat16, q_native._device)
    dv_native = _alloc(physical_shape, DType.bfloat16, q_native._device)
    seqlen_padded = ((seqlen + 127) // 128) * 128
    stats_shape = (batch, heads, seqlen_padded)
    dpsum = _alloc(stats_shape, DType.float32, q_native._device)
    lse_log2 = _alloc(stats_shape, DType.float32, q_native._device)
    dq_accum = _alloc(
        (batch * heads * seqlen_padded * head_dim,), DType.float32, q_native._device
    )
    if use_strided_qkv:
        _device_call(
            fa4_ops.flash_attention_bwd_bf16_d64_causal_strided_qkv,
            q_native._ptr,
            *q_native._strides,
            k_native._ptr,
            *k_native._strides,
            v_native._ptr,
            *v_native._strides,
            out_native._ptr,
            dout_native._ptr,
            logsumexp._ptr,
            dq_native._ptr,
            dk_native._ptr,
            dv_native._ptr,
            dpsum._ptr,
            lse_log2._ptr,
            dq_accum._ptr,
            batch,
            seqlen,
            heads,
            float(scale),
            _ctx_ptr(q_native._device),
            keepalive=(
                q_native,
                k_native,
                v_native,
                out_native,
                dout_native,
                logsumexp,
                dq_native,
                dk_native,
                dv_native,
                dpsum,
                lse_log2,
                dq_accum,
            ),
        )
    else:
        _device_call(
            fa4_ops.flash_attention_bwd_bf16_d64_causal,
            q_native._ptr,
            k_native._ptr,
            v_native._ptr,
            out_native._ptr,
            dout_native._ptr,
            logsumexp._ptr,
            dq_native._ptr,
            dk_native._ptr,
            dv_native._ptr,
            dpsum._ptr,
            lse_log2._ptr,
            dq_accum._ptr,
            batch,
            seqlen,
            heads,
            float(scale),
            _ctx_ptr(q_native._device),
            keepalive=(
                q_native,
                k_native,
                v_native,
                out_native,
                dout_native,
                logsumexp,
                dq_native,
                dk_native,
                dv_native,
                dpsum,
                lse_log2,
                dq_accum,
            ),
        )
    # TensorHolder destruction enqueues frees after the three kernels on the
    # same context; releasing scratch here never synchronizes the CPU.
    del dpsum, lse_log2, dq_accum, out_native, dout_native
    grad_query = fast_aten_transpose(dq_native, 1, 2)
    grad_key = fast_aten_transpose(dk_native, 1, 2)
    grad_value = fast_aten_transpose(dv_native, 1, 2)
    if any(grad is NOT_HANDLED for grad in (grad_query, grad_key, grad_value)):
        raise RuntimeError("FA4 gradients could not be exposed as BHTD views")
    return grad_query, grad_key, grad_value


# ---------------------------------------------------------------------------
# Fused flash attention for gfx942 (CDNA3).
#
# The decomposition below costs 29.854 ms/step forward and 46.153 backward on
# nanoGPT 124M at batch 48 / block 1024; the fused kernels measure 6.100 and
# 28.896.  They are gated narrowly and everything they decline falls through to
# the decomposition unchanged.
# ---------------------------------------------------------------------------

_FUSED_FA_MAX_HEAD_DIM = 256


def _fa_strides(t: MojoTensorLike) -> tuple[int, int, int]:
    """``t``'s (batch, head, seq) element strides for the fused kernels.

    The head_dim stride is not passed: it must be 1, and ``_fused_fa_inputs`` is
    what guarantees that.
    """
    strides = t._strides
    return (strides[0], strides[1], strides[2])


def _alloc_bthd(
    batch: int, heads: int, seq: int, head_dim: int, dtype: DType, device: Device
) -> TorchMojoTensor:
    """A ``[batch, heads, seq, head_dim]`` tensor STORED ``[batch, seq, heads,
    head_dim]``.

    The layout PyTorch's own flash attention returns, and the reason its
    ``transpose(1, 2)`` is free: ``o.transpose(1, 2)`` over these strides is
    exactly a dense ``[batch, seq, heads, head_dim]``, so the universal
    re-assembly idiom ``y.transpose(1, 2).contiguous().view(B, T, C)`` reduces to
    two views. Returning a dense ``[B, H, T, D]`` instead cost nanoGPT 48
    ``clone`` kernels and 1.84 ms/step (journal D11).

    One dense allocation with a strided view over it, not a strided allocation:
    the view spans every element exactly once, so it costs no extra memory and
    the Mojo bridge's zeroing by element count is still exactly the buffer.
    ``_compute_contiguous`` then reports ``False`` for it -- except when
    ``heads == 1`` or ``seq == 1``, where the two layouts genuinely coincide and
    ``True`` is the right answer.
    """
    dense = _alloc((batch, seq, heads, head_dim), dtype, device)
    return _view_of(
        dense,
        (batch, heads, seq, head_dim),
        (seq * heads * head_dim, head_dim, heads * head_dim, 1),
        0,
    )


def _fused_fa_inputs(
    query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
):
    """Eligible BHTD inputs for the fused gfx942 kernels, or None.

    The kernels read Q, K and V through their own (batch, head, seq) strides, so
    the transposed view of BTHD storage that a transformer's
    ``q.view(B, T, H, D).transpose(1, 2)`` produces -- and that nanoGPT hands
    over -- is taken as it stands rather than copied. The one layout that cannot
    be expressed is a strided head_dim axis: that is the axis every vectorized
    load and every LDS row fill runs along. It is declined here rather than
    guessed at, and the caller falls through to the decomposition.

    No device work: this only inspects shape, stride, dtype and device, which
    is all an eager op is allowed to look at.
    """
    q = _t(query)
    k = _t(key)
    v = _t(value)
    if (
        q is None
        or k is None
        or v is None
        or attn_mask is not None
        or enable_gqa
        or not isinstance(dropout_p, int | float)
        or isinstance(dropout_p, bool)
        or float(dropout_p) != 0.0
        or q._device != k._device
        or q._device != v._device
        or q._device.api != "hip"
        or q._device.architecture_name != "gfx942"
        or q._dtype not in _FLOAT_DTYPES
        or q._dtype != k._dtype
        or q._dtype != v._dtype
        or len(q._shape) != 4
        or len(k._shape) != 4
        or len(v._shape) != 4
    ):
        return None
    # Q and the output share seq_q; K and V share seq_kv. Batch, heads and
    # head_dim must agree across all three.
    if (
        tuple(q._shape[:2]) != tuple(k._shape[:2])
        or tuple(k._shape) != tuple(v._shape)
        or q._shape[3] != k._shape[3]
    ):
        return None
    batch, heads, seq_q, head_dim = q._shape
    seq_kv = k._shape[2]
    if (
        batch <= 0
        or heads <= 0
        or seq_q <= 0
        or seq_kv <= 0
        or head_dim <= 0
        or head_dim > _FUSED_FA_MAX_HEAD_DIM
    ):
        return None
    # The innermost axis must be contiguous; every other stride is free.
    if q._strides[3] != 1 or k._strides[3] != 1 or v._strides[3] != 1:
        return None
    if scale is not None and (
        not isinstance(scale, int | float)
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
    ):
        return None
    return q, k, v


def fast_fused_flash_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Fused forward returning ``(output, lse, q, k, v)``, or NOT_HANDLED.

    ``lse`` is the per-row log-sum-exp the fused backward consumes; the three
    returned tensors are exactly the tensors the kernel read, which the backward
    must be handed rather than re-deriving, because it recomputes the scores from
    the same bytes. Nothing is copied: each is read through its own
    (batch, head, seq) strides.

    ``output`` is ``[B, H, T, D]`` shaped and ``[B, T, H, D]`` stored, matching
    what PyTorch's flash attention returns, so the caller's
    ``y.transpose(1, 2).contiguous()`` is a no-op rather than a gather.
    """
    eligible = _fused_fa_inputs(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    if eligible is None:
        return NOT_HANDLED
    q, k, v = eligible
    batch, heads, seq_q, head_dim = q._shape
    seq_kv = k._shape[2]
    scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    output = _alloc_bthd(batch, heads, seq_q, head_dim, q._dtype, q._device)
    lse = _alloc((batch, heads, seq_q), DType.float32, q._device)
    _call_mojo(
        _FlashAttentionExtension,
        "FlashAttentionForward",
        (
            output._ptr,
            lse._ptr,
            q._ptr,
            k._ptr,
            v._ptr,
            (batch, heads, seq_q, seq_kv, head_dim),
            _fa_strides(q) + _fa_strides(k) + _fa_strides(v) + _fa_strides(output),
            scale_val,
            1 if is_causal else 0,
            q._dtype.value,
            _ctx_ptr(q._device),
        ),
        arg_dtypes=(q._dtype, k._dtype, v._dtype),
        output_dtypes=(output._dtype, lse._dtype),
        flags={"CAUSAL": bool(is_causal)},
        keepalive=(output, lse, q, k, v),
    )
    return output, lse, q, k, v


def fast_fused_flash_attention_backward(
    grad_output, query, key, value, output, lse, is_causal, scale
):
    """Fused backward returning ``(dq, dk, dv)``, or NOT_HANDLED.

    ``query``/``key``/``value``/``output``/``lse`` must be exactly what the
    fused forward read and wrote. Like the forward, the five read operands are
    addressed through their own strides; only a ``grad_output`` whose head_dim
    axis is strided has to be materialized, because that is the axis the
    vectorized loads run along.

    The three gradients come back ``[B, H, T, D]`` shaped and ``[B, T, H, D]``
    stored, so the ``transpose(1, 2)`` autograd runs on the way back out to
    ``x.view(B, T, H, D)`` is a view and the reshape behind it needs no copy.
    """
    g = _t(grad_output)
    q = _t(query)
    k = _t(key)
    v = _t(value)
    o = _t(output)
    l = _t(lse)
    if g is None or q is None or k is None or v is None or o is None or l is None:
        return NOT_HANDLED
    if g._dtype != q._dtype or tuple(g._shape) != tuple(q._shape):
        return NOT_HANDLED
    if g._strides[3] != 1:
        g = _tc(g)
        if g is None:
            return NOT_HANDLED
    if (
        q._strides[3] != 1
        or k._strides[3] != 1
        or v._strides[3] != 1
        or o._strides[3] != 1
    ):
        return NOT_HANDLED
    batch, heads, seq_q, head_dim = q._shape
    seq_kv = k._shape[2]
    scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    grad_query = _alloc_bthd(batch, heads, seq_q, head_dim, q._dtype, q._device)
    grad_key = _alloc_bthd(batch, heads, seq_kv, head_dim, q._dtype, q._device)
    grad_value = _alloc_bthd(batch, heads, seq_kv, head_dim, q._dtype, q._device)
    _call_mojo(
        _FlashAttentionExtension,
        "FlashAttentionBackward",
        (
            grad_query._ptr,
            grad_key._ptr,
            grad_value._ptr,
            g._ptr,
            q._ptr,
            k._ptr,
            v._ptr,
            o._ptr,
            l._ptr,
            (batch, heads, seq_q, seq_kv, head_dim),
            _fa_strides(g)
            + _fa_strides(q)
            + _fa_strides(k)
            + _fa_strides(v)
            + _fa_strides(o)
            + _fa_strides(grad_query)
            + _fa_strides(grad_key)
            + _fa_strides(grad_value),
            scale_val,
            1 if is_causal else 0,
            q._dtype.value,
            _ctx_ptr(q._device),
        ),
        arg_dtypes=(g._dtype, q._dtype, k._dtype, v._dtype, o._dtype, l._dtype),
        output_dtypes=(grad_query._dtype, grad_key._dtype, grad_value._dtype),
        flags={"CAUSAL": bool(is_causal)},
        keepalive=(grad_query, grad_key, grad_value, g, q, k, v, o, l),
    )
    return grad_query, grad_key, grad_value


def _sdpa_math_forward(query, key, value, is_causal, scale):
    """Dropout-free compatibility wrapper returning ``(output, probs)``."""
    result = _sdpa_math_forward_with_dropout(query, key, value, is_causal, scale, 0.0)
    if result is NOT_HANDLED:
        return NOT_HANDLED
    output, probabilities, _ = result
    return output, probabilities


def fast_aten__scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    *,
    scale=None,
    enable_gqa=False,
):
    if dropout_p != 0.0 or dropout_mask is not None or enable_gqa:
        return NOT_HANDLED
    if attn_mask is not None:
        return _sdpa_masked_math_forward(query, key, value, attn_mask, is_causal, scale)
    result = _sdpa_math_forward(query, key, value, is_causal, scale)
    if result is NOT_HANDLED:
        return NOT_HANDLED
    out, probs = result
    return out, probs


def fast_aten__scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    if dropout_p != 0.0 or return_debug_mask:
        return NOT_HANDLED
    result = fast_fa4_bf16_d64_causal_forward(
        query, key, value, None, 0.0, is_causal, scale, False
    )
    if result is NOT_HANDLED:
        return NOT_HANDLED
    out, logsumexp, q_native, k_native, v_native = result
    # This direct ATen forward does not own the autograd saves used by the
    # high-level SDPA custom Function. The physical input copies only need to
    # survive their already-enqueued forward launch, so release them in stream
    # order instead of retaining three full activations.
    del q_native, k_native, v_native
    q = _t(query)
    sq = q._shape[2]
    sk = _t(key)._shape[2]
    dev = q._device
    # Dense CUDA returns undefined cumulative-sequence tensors, uint64 RNG
    # state/offset tensors, and an empty debug mask when dropout/debugging are
    # disabled. FA4's real FP32 LSE must be returned for backward semantics.
    # Dropout is disabled, so the RNG payload values are unobserved; only the
    # CUDA-compatible uint64 shapes/dtypes are part of this forward contract.
    rng_state = _alloc((2,), DType.uint64, dev)
    unused = _alloc((), DType.uint64, dev)
    debug_attn_mask = _alloc((0,), q._dtype, dev)
    return (out, logsumexp, None, None, sq, sk, rng_state, unused, debug_attn_mask)


def fast_aten__scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    """Dense lower-op autograd bridge for the vendored BF16 FA4 kernel."""
    inputs = _fa4_bf16_d64_causal_inputs(
        query, key, value, None, dropout_p, is_causal, scale, False
    )
    grad = _t(grad_out)
    output = _t(out)
    lse = _t(logsumexp)
    if inputs is None:
        return NOT_HANDLED
    q, k, v = inputs
    batch, heads, seqlen, _ = q._shape
    if (
        cum_seq_q is not None
        or cum_seq_k is not None
        or max_q != seqlen
        or max_k != seqlen
        or grad is None
        or grad._device != q._device
        or grad._dtype != DType.bfloat16
        or tuple(grad._shape) != tuple(q._shape)
        or (
            output is not None
            and (
                output._device != q._device
                or output._dtype != DType.bfloat16
                or tuple(output._shape) != tuple(q._shape)
            )
        )
        or (
            lse is not None
            and (
                lse._device != q._device
                or lse._dtype != DType.float32
                or tuple(lse._shape) != (batch, heads, seqlen)
            )
        )
    ):
        return NOT_HANDLED
    # Dropout is excluded above, so the RNG payload values are irrelevant.
    _ = philox_seed, philox_offset
    if output is None or lse is None:
        # PyTorch's generated SavedVariable path can unpack our returned
        # subclass output as a base Tensor without the Python allocation
        # payload. Recompute this direct lower-op forward so backward never
        # reads an inaccessible pointer. The high-level SDPA custom autograd
        # path retains its payload and does not take this compatibility path.
        recomputed = fast_fa4_bf16_d64_causal_forward(
            q, k, v, None, 0.0, True, scale, False
        )
        if recomputed is NOT_HANDLED:
            return NOT_HANDLED
        recomputed_output, recomputed_lse, q_native, k_native, v_native = recomputed
        output = recomputed_output
        lse = recomputed_lse
    else:
        q_native = _fa4_native_bthd(q)
        k_native = _fa4_native_bthd(k)
        v_native = _fa4_native_bthd(v)
        if q_native is None or k_native is None or v_native is None:
            return NOT_HANDLED
    # The lower ATen backward accepts arbitrary shape-compatible views, while
    # FA4 consumes LSE as contiguous (B, H, S).  Keep the generated/high-level
    # path zero-copy and materialize only a genuinely strided direct caller.
    lse = _tc(lse)
    if lse is None:
        return NOT_HANDLED
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(q._shape[-1])
    return fast_fa4_bf16_d64_causal_backward(
        q_native, k_native, v_native, output, lse, grad, scale_value
    )


def fast_aten__scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias,
    compute_log_sumexp,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    if dropout_p != 0.0 or attn_bias is not None:
        return NOT_HANDLED
    out = fast_aten_scaled_dot_product_attention(
        query, key, value, None, 0.0, is_causal, scale, False
    )
    if out is NOT_HANDLED:
        return NOT_HANDLED
    q = _t(query)
    b, h, sq, _ = q._shape
    dev = q._device
    lse_len = sq if compute_log_sumexp else 0
    log_sumexp = fast_filled((b, h, lse_len), 0.0, DType.float32, dev)
    philox_seed = fast_filled((), 0, DType.int64, dev)
    philox_offset = fast_filled((), 0, DType.int64, dev)
    return (out, log_sumexp, philox_seed, philox_offset)


def fast_sdpa_dropout_softmax_backward(
    probabilities: object,
    grad_after_dropout: object,
    dropout_mask: object,
    dropout_scale: object,
    score_scale: object,
    is_causal: bool = False,
    query_length: int = 0,
) -> object:
    """Fuse SDPA dropout backward, softmax backward, and score scaling.

    The helper owns public-tensor validation, ordinary output allocation, and
    the pointer-only bridge call.  Its device-kernel body remains isolated in
    the Fable-owned module.  Unsupported inputs return ``NOT_HANDLED`` before
    any operand is materialized or output is allocated.

    ``is_causal`` lets the kernel skip the fully masked half of every row: the
    forward softmax makes ``P`` exactly zero there, so ``dScores`` is exactly
    zero and is written as such.  ``query_length`` is the row period of the
    top-left-aligned mask (rows are ``batch * heads * query_length``).
    """
    probs = _t(probabilities)
    grad = _t(grad_after_dropout)
    mask = _t(dropout_mask) if dropout_mask is not None else None
    if (
        probs is None
        or grad is None
        or (dropout_mask is not None and mask is None)
        or not _on_gpu(probs)
        or probs._dtype not in _FLOAT_DTYPES
        or grad._dtype != probs._dtype
        or probs._device != grad._device
        or tuple(probs._shape) != tuple(grad._shape)
        or len(probs._shape) < 1
        or not isinstance(score_scale, int | float)
        or isinstance(score_scale, bool)
        or not math.isfinite(float(score_scale))
    ):
        return NOT_HANDLED
    if is_causal and (
        not isinstance(query_length, int)
        or isinstance(query_length, bool)
        or query_length <= 0
        or math.prod(probs._shape[:-1]) % query_length != 0
    ):
        return NOT_HANDLED

    has_mask = mask is not None
    if has_mask and (
        mask._device != probs._device
        or mask._dtype != DType.bool
        or tuple(mask._shape) != tuple(probs._shape)
        or not isinstance(dropout_scale, int | float)
        or isinstance(dropout_scale, bool)
        or not math.isfinite(float(dropout_scale))
    ):
        return NOT_HANDLED

    # The Fable-owned production kernel is ported separately from this host
    # wiring. Keep the eager decomposition usable while that optional module
    # is absent.
    if not _sdpa_backward_bridge_available():
        return NOT_HANDLED

    # In the no-dropout path the scale is semantically dead.  Canonicalizing
    # it also keeps nonnumeric or nonfinite caller metadata away from the raw
    # Float64 bridge without rejecting an otherwise valid operation.
    bridge_dropout_scale = float(dropout_scale) if has_mask else 1.0
    bridge_score_scale = float(score_scale)

    probs = _tc(probs)
    grad = _tc(grad)
    if has_mask:
        mask = _tc(mask)
    out = _alloc(probs._shape, probs._dtype, probs._device)
    if out._numel == 0:
        return out

    rows = math.prod(probs._shape[:-1])
    cols = probs._shape[-1]
    _call_mojo(
        _SdpaBackwardExtension,
        "SDPADropoutSoftmaxBackward",
        (
            out._ptr,
            probs._ptr,
            grad._ptr,
            mask._ptr if has_mask else 0,
            rows,
            cols,
            int(query_length) if is_causal else 0,
            int(has_mask),
            int(bool(is_causal)),
            bridge_dropout_scale,
            bridge_score_scale,
            probs._dtype.value,
            _ctx_ptr(probs._device),
        ),
        arg_dtypes=(probs._dtype, grad._dtype) + ((mask._dtype,) if has_mask else ()),
        output_dtypes=(out._dtype,),
        flags={"HAS_MASK": has_mask, "CAUSAL": bool(is_causal)},
        keepalive=(out, probs, grad, mask),
    )
    return out


def fast_sdpa_backward(
    probabilities: MojoTensorLike,
    grad_output: MojoTensorLike,
    dropout_mask: MojoTensorLike | None,
    query: MojoTensorLike | None,
    key: MojoTensorLike | None,
    value: MojoTensorLike | None,
    need_query: bool,
    need_key: bool,
    need_value: bool,
    is_causal: bool,
    dropout_scale: float,
    score_scale: float,
) -> (
    tuple[TorchMojoTensor | None, TorchMojoTensor | None, TorchMojoTensor | None]
    | object  # the NOT_HANDLED sentinel
):
    """Fused Apple-GPU FP32 SDPA backward over (batch*heads, L, S) operands.

    Five launches at most, no permute copies and no dropout-backward pass:

        dP = dO @ V^T          (causal mode 1: upper tiles skipped, unwritten)
        dS = fused causal dropout+softmax+scale backward
        dQ = dS @ K            (causal mode 2: reduction cut at the boundary)
        dV = (P*mask*ds)^T @ dO  (transposed-A GEMM, dropout fused in A load)
        dK = dS^T @ Q          (same transposed-A GEMM, unmasked)

    The causal specializations are exact because the Apple causal forward
    zero-fills P past the boundary and the causal dS kernel zero-fills its
    64-aligned band. Non-causal inputs run the same sequence unspecialized.
    Unsupported inputs return ``NOT_HANDLED`` before any device work.
    """
    probs = _t(probabilities)
    grad = _t(grad_output)
    mask = _t(dropout_mask) if dropout_mask is not None else None
    if (
        probs is None
        or grad is None
        or (dropout_mask is not None and mask is None)
        or getattr(probs._device, "api", None) != "metal"
        or probs._dtype != DType.float32
        or grad._dtype != DType.float32
        or probs._device != grad._device
        or len(probs._shape) != 3
        or len(grad._shape) != 3
        or probs._shape[:2] != grad._shape[:2]
        or not isinstance(score_scale, int | float)
        or isinstance(score_scale, bool)
        or not math.isfinite(float(score_scale))
        or 0 in probs._shape
        or 0 in grad._shape
    ):
        return NOT_HANDLED
    has_mask = mask is not None
    if has_mask and (
        mask._device != probs._device
        or mask._dtype != DType.bool
        or tuple(mask._shape) != tuple(probs._shape)
        or not isinstance(dropout_scale, int | float)
        or isinstance(dropout_scale, bool)
        or not math.isfinite(float(dropout_scale))
    ):
        return NOT_HANDLED

    batch_heads, q_len, kv_len = probs._shape
    head_dim = grad._shape[2]
    kv3_shape = (batch_heads, kv_len, head_dim)
    q3_shape = (batch_heads, q_len, head_dim)
    if need_query or need_key:
        v = _t(value)
        if (
            v is None
            or v._device != probs._device
            or v._dtype != DType.float32
            or tuple(v._shape) != kv3_shape
        ):
            return NOT_HANDLED
    if need_query:
        k = _t(key)
        if (
            k is None
            or k._device != probs._device
            or k._dtype != DType.float32
            or tuple(k._shape) != kv3_shape
        ):
            return NOT_HANDLED
    if need_key:
        q = _t(query)
        if (
            q is None
            or q._device != probs._device
            or q._dtype != DType.float32
            or tuple(q._shape) != q3_shape
        ):
            return NOT_HANDLED

    if not _sdpa_backward_bridge_available():
        return NOT_HANDLED

    bridge_dropout_scale = float(dropout_scale) if has_mask else 1.0
    causal = 1 if is_causal else 0
    probs = _tc(probs)
    grad = _tc(grad)
    if has_mask:
        mask = _tc(mask)
    device = probs._device
    ctx = _ctx_ptr(device)
    dt = DType.float32.value

    grad_value = None
    if need_value:
        grad_value = _alloc(kv3_shape, DType.float32, device)
        _call_mojo(
            _SdpaBackwardExtension,
            "SDPATransAGemmF32",
            (
                grad_value._ptr,
                probs._ptr,
                grad._ptr,
                mask._ptr if has_mask else 0,
                (batch_heads, kv_len, head_dim, q_len, int(has_mask), causal),
                bridge_dropout_scale,
                ctx,
            ),
            arg_dtypes=(probs._dtype, grad._dtype)
            + ((mask._dtype,) if has_mask else ()),
            output_dtypes=(grad_value._dtype,),
            flags={"HAS_MASK": has_mask, "CAUSAL": bool(is_causal)},
            keepalive=(grad_value, probs, grad, mask),
        )

    grad_query = None
    grad_key = None
    if need_query or need_key:
        v = _tc(_t(value))
        grad_probs = _alloc((batch_heads, q_len, kv_len), DType.float32, device)
        if causal:
            _call_mojo(
                _MatmulExtension,
                "BmmCausalF32",
                (
                    grad_probs._ptr,
                    grad._ptr,
                    v._ptr,
                    (batch_heads, q_len, kv_len, head_dim, 1, 1),
                    dt,
                    ctx,
                ),
                arg_dtypes=(grad._dtype, v._dtype),
                output_dtypes=(grad_probs._dtype,),
                flags={"TRANSPOSE_B": True, "CAUSAL_MODE": SDPA_CAUSAL_OUT},
                keepalive=(grad_probs, grad, v),
            )
        else:
            _call_mojo(
                _MatmulExtension,
                "Bmm",
                (
                    grad_probs._ptr,
                    grad._ptr,
                    v._ptr,
                    (batch_heads, q_len, kv_len, head_dim, 1),
                    dt,
                    ctx,
                ),
                arg_dtypes=(grad._dtype, v._dtype),
                output_dtypes=(grad_probs._dtype,),
                flags={"TRANSPOSE_B": True},
                keepalive=(grad_probs, grad, v),
            )
        del v
        grad_scores = _alloc((batch_heads, q_len, kv_len), DType.float32, device)
        _call_mojo(
            _SdpaBackwardExtension,
            "SDPADropoutSoftmaxBackwardF32",
            (
                grad_scores._ptr,
                probs._ptr,
                grad_probs._ptr,
                mask._ptr if has_mask else 0,
                batch_heads * q_len,
                kv_len,
                int(has_mask),
                bridge_dropout_scale,
                float(score_scale),
                causal,
                q_len,
                ctx,
            ),
            arg_dtypes=(probs._dtype, grad_probs._dtype)
            + ((mask._dtype,) if has_mask else ()),
            output_dtypes=(grad_scores._dtype,),
            flags={"HAS_MASK": has_mask, "CAUSAL": bool(is_causal)},
            keepalive=(grad_scores, probs, grad_probs, mask),
        )
        del grad_probs
        if need_query:
            k = _tc(_t(key))
            grad_query = _alloc(q3_shape, DType.float32, device)
            if causal:
                _call_mojo(
                    _MatmulExtension,
                    "BmmCausalF32",
                    (
                        grad_query._ptr,
                        grad_scores._ptr,
                        k._ptr,
                        (batch_heads, q_len, head_dim, kv_len, 0, 2),
                        dt,
                        ctx,
                    ),
                    arg_dtypes=(grad_scores._dtype, k._dtype),
                    output_dtypes=(grad_query._dtype,),
                    flags={"TRANSPOSE_B": False, "CAUSAL_MODE": SDPA_CAUSAL_A_ROWS},
                    keepalive=(grad_query, grad_scores, k),
                )
            else:
                _call_mojo(
                    _MatmulExtension,
                    "Bmm",
                    (
                        grad_query._ptr,
                        grad_scores._ptr,
                        k._ptr,
                        (batch_heads, q_len, head_dim, kv_len, 0),
                        dt,
                        ctx,
                    ),
                    arg_dtypes=(grad_scores._dtype, k._dtype),
                    output_dtypes=(grad_query._dtype,),
                    flags={"TRANSPOSE_B": False},
                    keepalive=(grad_query, grad_scores, k),
                )
            del k
        if need_key:
            q = _tc(_t(query))
            grad_key = _alloc(kv3_shape, DType.float32, device)
            _call_mojo(
                _SdpaBackwardExtension,
                "SDPATransAGemmF32",
                (
                    grad_key._ptr,
                    grad_scores._ptr,
                    q._ptr,
                    0,
                    (batch_heads, kv_len, head_dim, q_len, 0, causal),
                    1.0,
                    ctx,
                ),
                arg_dtypes=(grad_scores._dtype, q._dtype),
                output_dtypes=(grad_key._dtype,),
                flags={"HAS_MASK": False, "CAUSAL": bool(is_causal)},
                keepalive=(grad_key, grad_scores, q),
            )
            del q
        del grad_scores

    return grad_query, grad_key, grad_value


# ---------------------------------------------------------------------------
# Matmul family (GPU: pure-Mojo GEMM; CPU: correctness-grade loops)
# ---------------------------------------------------------------------------


def _tf32_dense_2d_layout(tensor: MojoTensorLike) -> bool | None:
    """Return the physical-transpose flag for an exact dense 2-D layout."""
    if len(tensor._shape) != 2:
        return None
    rows, cols = tensor._shape
    strides = tuple(tensor._strides)
    if strides == (cols, 1):
        return False
    if strides == (1, rows):
        return True
    return None


def _tf32_dense_batched_layout(tensor: MojoTensorLike) -> tuple[bool, int] | None:
    """Classify dense matrices separated by a non-overlapping batch stride.

    The boolean is the physical per-matrix transpose flag and the integer is
    the runtime batch stride in elements.  Padding between matrices is valid;
    arbitrary inner strides, broadcasts, and overlapping batches are not.
    """
    if len(tensor._shape) != 3:
        return None
    batch, rows, cols = tensor._shape
    batch_stride, row_stride, col_stride = tuple(tensor._strides)
    row_major = col_stride == 1 and (rows == 1 or row_stride == cols)
    transposed = row_stride == 1 and (cols == 1 or col_stride == rows)
    if row_major:
        physical_transpose = False
    elif transposed:
        physical_transpose = True
    else:
        return None
    matrix_elements = rows * cols
    if min(batch, rows, cols) <= 0 or batch_stride < matrix_elements:
        return None
    return physical_transpose, batch_stride


def _try_bf16_gemm(a, b, bias=None, *, transpose_b=False, output_shape=None):
    """Enqueue the dense H100 BF16 GEMM, or return ``None``.

    The bridge consumes and produces BF16 while the accepted device kernel
    accumulates in FP32.  This host helper only validates metadata, resolves
    the optional bridge before allocation, and launches it without a retry.
    """
    lhs = _t(a)
    rhs = _t(b)
    bias_tensor = _t(bias) if bias is not None else None
    if (
        lhs is None
        or rhs is None
        or (bias is not None and bias_tensor is None)
        or lhs._dtype != DType.bfloat16
        or rhs._dtype != DType.bfloat16
        or lhs._device != rhs._device
        or lhs._device.label != "gpu"
        or lhs._device.api != "cuda"
        or lhs._device.architecture_name != "sm_90a"
    ):
        return None
    lhs_layout = _tf32_dense_2d_layout(lhs)
    rhs_layout = _tf32_dense_2d_layout(rhs)
    if lhs_layout is None or rhs_layout is None:
        return None

    m, k = lhs._shape
    rhs_k = rhs._shape[1] if transpose_b else rhs._shape[0]
    n = rhs._shape[0] if transpose_b else rhs._shape[1]
    if min(m, n, k) <= 0 or rhs_k != k:
        return None
    if bias_tensor is not None and (
        bias_tensor._device != lhs._device
        or bias_tensor._dtype != DType.bfloat16
        or tuple(bias_tensor._shape) != (n,)
        or not bias_tensor._is_contiguous
    ):
        return None

    logical_output_shape = (m, n) if output_shape is None else tuple(output_shape)
    if (
        not logical_output_shape
        or logical_output_shape[-1] != n
        or math.prod(logical_output_shape) != m * n
    ):
        return None
    if not _bf16_bridge_available():
        return None
    out = _alloc(logical_output_shape, DType.bfloat16, lhs._device)
    _call_mojo(
        _Bf16MatmulExtension,
        "Bf16GemmBF16",
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            bias_tensor._ptr if bias_tensor is not None else out._ptr,
            m,
            n,
            k,
            int(lhs_layout),
            int(rhs_layout) ^ int(bool(transpose_b)),
            int(bias_tensor is not None),
            _ctx_ptr(lhs._device),
        ),
        arg_dtypes=(lhs._dtype, rhs._dtype)
        + ((bias_tensor._dtype,) if bias_tensor is not None else ()),
        output_dtypes=(out._dtype,),
        flags={"TRANSPOSE_B": bool(transpose_b), "HAS_BIAS": bias_tensor is not None},
        keepalive=(out, lhs, rhs, bias_tensor),
    )
    return out


def _try_tf32_gemm(a, b, bias=None, *, transpose_b=False, output_shape=None):
    """Enqueue the opt-in dense H100 TF32 GEMM, or return ``None``.

    This helper owns only host validation/allocation and the raw bridge call;
    the Fable-owned module owns every device-kernel body.  Unsupported layouts
    and strict FP32 retain the existing pure-Mojo SIMT path.
    """
    # This gate is a numerics decision, not a capability one: TF32 drops
    # mantissa bits, and "highest" (PyTorch's default) is the user asking for
    # full FP32.  Do not relax it to widen layout support -- that would change
    # results silently.  It does mean the one bridge that accepts an arbitrary
    # 2D layout is off by default for FP32; `_try_spec_matmul` owns the
    # copy-and-retry backstop that keeps strided FP32 operands working.
    if torch.get_float32_matmul_precision() == "highest":
        return None
    lhs = _t(a)
    rhs = _t(b)
    bias_tensor = _t(bias) if bias is not None else None
    if (
        lhs is None
        or rhs is None
        or (bias is not None and bias_tensor is None)
        or lhs._dtype != DType.float32
        or rhs._dtype != DType.float32
        or lhs._device != rhs._device
        or lhs._device.label != "gpu"
        or lhs._device.api != "cuda"
        or lhs._device.architecture_name != "sm_90a"
    ):
        return None
    lhs_layout = _tf32_dense_2d_layout(lhs)
    rhs_layout = _tf32_dense_2d_layout(rhs)
    if lhs_layout is None or rhs_layout is None:
        return None

    m, k = lhs._shape
    rhs_k = rhs._shape[1] if transpose_b else rhs._shape[0]
    n = rhs._shape[0] if transpose_b else rhs._shape[1]
    if min(m, n, k) <= 0 or rhs_k != k:
        return None
    if bias_tensor is not None and (
        bias_tensor._device != lhs._device
        or bias_tensor._dtype != DType.float32
        or tuple(bias_tensor._shape) != (n,)
        or not bias_tensor._is_contiguous
    ):
        return None

    logical_output_shape = (m, n) if output_shape is None else tuple(output_shape)
    if (
        not logical_output_shape
        or logical_output_shape[-1] != n
        or math.prod(logical_output_shape) != m * n
    ):
        return None
    if not _tf32_bridge_available():
        return None
    out = _alloc(logical_output_shape, DType.float32, lhs._device)
    _call_mojo(
        _Tf32MatmulExtension,
        "Tf32GemmF32",
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            bias_tensor._ptr if bias_tensor is not None else out._ptr,
            m,
            n,
            k,
            int(lhs_layout),
            int(rhs_layout) ^ int(bool(transpose_b)),
            int(bias_tensor is not None),
            _ctx_ptr(lhs._device),
        ),
        arg_dtypes=(lhs._dtype, rhs._dtype)
        + ((bias_tensor._dtype,) if bias_tensor is not None else ()),
        output_dtypes=(out._dtype,),
        flags={"TRANSPOSE_B": bool(transpose_b), "HAS_BIAS": bias_tensor is not None},
        keepalive=(out, lhs, rhs, bias_tensor),
    )
    return out


def _try_bf16_bmm(a, b, *, transpose_b=False):
    """Enqueue dense H100 BF16 BMM over packed or padded batches."""
    lhs = _t(a)
    rhs = _t(b)
    if (
        lhs is None
        or rhs is None
        or lhs._dtype != DType.bfloat16
        or rhs._dtype != DType.bfloat16
        or lhs._device != rhs._device
        or lhs._device.label != "gpu"
        or lhs._device.api != "cuda"
        or lhs._device.architecture_name != "sm_90a"
    ):
        return None
    lhs_layout = _tf32_dense_batched_layout(lhs)
    rhs_layout = _tf32_dense_batched_layout(rhs)
    if lhs_layout is None or rhs_layout is None:
        return None

    batch, m, k = lhs._shape
    rhs_batch = rhs._shape[0]
    rhs_k = rhs._shape[2] if transpose_b else rhs._shape[1]
    n = rhs._shape[1] if transpose_b else rhs._shape[2]
    if min(batch, m, n, k) <= 0 or rhs_batch != batch or rhs_k != k:
        return None

    lhs_transposed, lhs_batch_stride = lhs_layout
    rhs_transposed, rhs_batch_stride = rhs_layout
    output_batch_stride = m * n
    if not _bf16_bridge_available():
        return None
    out = _alloc((batch, m, n), DType.bfloat16, lhs._device)
    _call_mojo(
        _Bf16MatmulExtension,
        "Bf16BmmBF16",
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            batch,
            m,
            n,
            k,
            output_batch_stride,
            lhs_batch_stride,
            rhs_batch_stride,
            int(lhs_transposed),
            int(rhs_transposed) ^ int(bool(transpose_b)),
            _ctx_ptr(lhs._device),
        ),
        arg_dtypes=(lhs._dtype, rhs._dtype),
        output_dtypes=(out._dtype,),
        flags={"TRANSPOSE_B": bool(transpose_b)},
        keepalive=(out, lhs, rhs),
    )
    return out


def _try_tf32_bmm(a, b, *, transpose_b=False):
    """Enqueue the dormant dense H100 TF32 BMM, or return ``None``.

    This bias-free ABI accepts packed or independently padded batches of
    dense row-major/per-matrix-transpose operands.  ``transpose_b`` is the
    logical RHS transpose used by the SDPA autograd path; it is folded with
    B's physical layout without materializing a transposed tensor.
    """
    if torch.get_float32_matmul_precision() == "highest":
        return None
    lhs = _t(a)
    rhs = _t(b)
    if (
        lhs is None
        or rhs is None
        or lhs._dtype != DType.float32
        or rhs._dtype != DType.float32
        or lhs._device != rhs._device
        or lhs._device.label != "gpu"
        or lhs._device.api != "cuda"
        or lhs._device.architecture_name != "sm_90a"
    ):
        return None
    lhs_layout = _tf32_dense_batched_layout(lhs)
    rhs_layout = _tf32_dense_batched_layout(rhs)
    if lhs_layout is None or rhs_layout is None:
        return None

    batch, m, k = lhs._shape
    rhs_batch = rhs._shape[0]
    rhs_k = rhs._shape[2] if transpose_b else rhs._shape[1]
    n = rhs._shape[1] if transpose_b else rhs._shape[2]
    if min(batch, m, n, k) <= 0 or rhs_batch != batch or rhs_k != k:
        return None

    lhs_transposed, lhs_batch_stride = lhs_layout
    rhs_transposed, rhs_batch_stride = rhs_layout
    output_batch_stride = m * n
    if not _tf32_bridge_available():
        return None
    out = _alloc((batch, m, n), DType.float32, lhs._device)
    _call_mojo(
        _Tf32MatmulExtension,
        "Tf32BmmF32",
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            batch,
            m,
            n,
            k,
            output_batch_stride,
            lhs_batch_stride,
            rhs_batch_stride,
            int(lhs_transposed),
            int(rhs_transposed) ^ int(bool(transpose_b)),
            _ctx_ptr(lhs._device),
        ),
        arg_dtypes=(lhs._dtype, rhs._dtype),
        output_dtypes=(out._dtype,),
        flags={"TRANSPOSE_B": bool(transpose_b)},
        keepalive=(out, lhs, rhs),
    )
    return out


def _try_bf16_linear(input, weight, bias=None):
    """Route a dense rank >= 2 BF16 projection through GEMM without copies."""
    a = _t(input)
    w = _t(weight)
    if (
        a is None
        or w is None
        or len(a._shape) < 2
        or len(w._shape) != 2
        or (len(a._shape) > 2 and not a._is_contiguous)
    ):
        return None

    output_shape = tuple(a._shape[:-1]) + (w._shape[0],)
    if len(a._shape) == 2:
        matrix = a
    else:
        matrix_shape = (math.prod(a._shape[:-1]), a._shape[-1])
        matrix = _view_of(
            a,
            matrix_shape,
            _row_major_strides(matrix_shape),
            a._offset,
            contiguous=True,
        )
    return _try_bf16_gemm(
        matrix, weight, bias, transpose_b=True, output_shape=output_shape
    )


def _try_tf32_linear(input, weight, bias=None):
    """Route a dense rank >= 2 linear projection through TF32 without copies.

    The TF32 GEMM ABI is two-dimensional.  A contiguous higher-rank linear
    input is already the same row-major matrix after its leading dimensions
    are flattened, so only a metadata view is needed.  Non-contiguous inputs
    retain the TensorSpec path, which owns any required materialization.
    """
    if torch.get_float32_matmul_precision() == "highest":
        return None
    a = _t(input)
    w = _t(weight)
    if (
        a is None
        or w is None
        or len(a._shape) < 2
        or len(w._shape) != 2
        or (len(a._shape) > 2 and not a._is_contiguous)
    ):
        return None

    output_shape = tuple(a._shape[:-1]) + (w._shape[0],)
    if len(a._shape) == 2:
        matrix = a
    else:
        matrix_shape = (math.prod(a._shape[:-1]), a._shape[-1])
        matrix = _view_of(
            a,
            matrix_shape,
            _row_major_strides(matrix_shape),
            a._offset,
            contiguous=True,
        )
    return _try_tf32_gemm(
        matrix, weight, bias, transpose_b=True, output_shape=output_shape
    )


def fast_aten_mm(x, y):
    out = _try_bf16_gemm(x, y)
    if out is not None:
        return out
    out = _try_tf32_gemm(x, y)
    if out is not None:
        return out
    out = _try_spec_matmul("MatmulSpec", (x, y), 0)
    return out if out is not None else NOT_HANDLED


def fast_aten_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    # beta/alpha scaling isn't implemented by the fast path (falls through).
    if beta == 1 and alpha == 1:
        out = _try_bf16_gemm(mat1, mat2, input)
        if out is not None:
            return out
        out = _try_tf32_gemm(mat1, mat2, input)
        if out is not None:
            return out
        out = _try_spec_matmul("MatmulBiasSpec", (mat1, mat2, input), 0)
        if out is not None:
            return out
    return NOT_HANDLED


def fast_aten_linear(input, weight, bias=None):
    # Keep linear as a concrete backend op alongside fast_aten_linear_backward.
    # The GEMM kernel reads B transposed for free, so the weight is never
    # materialized in transposed layout.
    out = _try_bf16_linear(input, weight, bias)
    if out is not None:
        return out
    out = _try_tf32_linear(input, weight, bias)
    if out is not None:
        return out
    if bias is None:
        out = _try_spec_matmul("MatmulSpec", (input, weight), 1)
    else:
        out = _try_spec_matmul("MatmulBiasSpec", (input, weight, bias), 1)
    if out is not None:
        return out

    # TensorSpec deliberately owns the ordinary fallback above.  Inspect the
    # input only after it declines so strict FP32 can reach its SIMT path
    # without touching TF32 metadata, while rank-1 linear can still reuse the
    # rank-2 implementation (the current TensorSpec ABI accepts rank >= 2).
    vector = _t(input)
    if vector is None or len(vector._shape) != 1:
        return NOT_HANDLED
    matrix_weight = _t(weight)
    vector_bias = _t(bias) if bias is not None else None
    if (
        matrix_weight is None
        or len(matrix_weight._shape) != 2
        or matrix_weight._shape[1] != vector._shape[0]
        or matrix_weight._dtype != vector._dtype
        or matrix_weight._device != vector._device
        or (
            bias is not None
            and (
                vector_bias is None
                or vector_bias._shape != (matrix_weight._shape[0],)
                or vector_bias._dtype != vector._dtype
                or vector_bias._device != vector._device
            )
        )
    ):
        return NOT_HANDLED
    output_features = matrix_weight._shape[0]
    if output_features == 0:
        return _alloc((0,), vector._dtype, vector._device)
    if vector._shape[0] == 0:
        if vector_bias is not None:
            return fast_aten_clone(vector_bias)
        return fast_filled((output_features,), 0.0, vector._dtype, vector._device)

    matrix = fast_aten_view(_tc(vector), (1, vector._shape[0]))
    if matrix is NOT_HANDLED:
        return NOT_HANDLED
    out = fast_aten_linear(matrix, weight, bias)
    if out is NOT_HANDLED:
        return NOT_HANDLED
    return fast_aten_view(out, (out._shape[-1],))


def fast_aten_linear_backward(self, grad_output, weight, output_mask):
    input = _t(self)
    grad = _t(grad_output)
    matrix_weight = _t(weight)
    mask = tuple(output_mask)
    if (
        len(mask) != 3
        or any(not isinstance(requested, bool) for requested in mask)
        or input is None
        or grad is None
        or matrix_weight is None
        or len(input._shape) < 1
        or len(matrix_weight._shape) != 2
        or input._dtype not in _FLOAT_DTYPES
        or grad._dtype != input._dtype
        or matrix_weight._dtype != input._dtype
        or grad._device != input._device
        or matrix_weight._device != input._device
        or input._shape[-1] != matrix_weight._shape[1]
        or grad._shape != input._shape[:-1] + (matrix_weight._shape[0],)
    ):
        return NOT_HANDLED
    if not any(mask):
        return None, None, None

    rows = math.prod(input._shape[:-1]) if len(input._shape) > 1 else 1
    input_features = input._shape[-1]
    output_features = matrix_weight._shape[0]
    # PyTorch's registered Meta/MPS contract defines both parameter outputs
    # whenever either one is requested. An unrequested bias result is only an
    # allocation; requesting bias also requires the weight-gradient GEMM.
    need_parameter_grads = mask[1] or mask[2]

    def zeros(shape):
        if math.prod(shape) == 0:
            return _alloc(shape, input._dtype, input._device)
        return fast_filled(shape, 0.0, input._dtype, input._device)

    if rows == 0 or output_features == 0:
        return (
            zeros(input._shape) if mask[0] else None,
            zeros(matrix_weight._shape) if need_parameter_grads else None,
            (
                zeros((output_features,))
                if mask[2]
                else _alloc((output_features,), input._dtype, input._device)
            )
            if need_parameter_grads
            else None,
        )

    grad = _tc(grad)
    grad_matrix = fast_aten_view(grad, (rows, output_features))
    if grad_matrix is NOT_HANDLED:
        return NOT_HANDLED

    grad_input = None
    if mask[0]:
        if input_features == 0:
            grad_input = _alloc(input._shape, input._dtype, input._device)
        else:
            grad_input = fast_aten_mm(grad_matrix, matrix_weight)
            if grad_input is NOT_HANDLED:
                return NOT_HANDLED
            grad_input = fast_aten_view(grad_input, input._shape)
            if grad_input is NOT_HANDLED:
                return NOT_HANDLED

    grad_weight = None
    if need_parameter_grads:
        if input_features == 0:
            grad_weight = _alloc(matrix_weight._shape, input._dtype, input._device)
        else:
            input_matrix = fast_aten_view(_tc(input), (rows, input_features))
            if input_matrix is NOT_HANDLED:
                return NOT_HANDLED
            grad_transposed = fast_aten_transpose(grad_matrix, 0, 1)
            if grad_transposed is NOT_HANDLED:
                return NOT_HANDLED
            grad_weight = fast_aten_mm(grad_transposed, input_matrix)
            if grad_weight is NOT_HANDLED:
                return NOT_HANDLED

    grad_bias = None
    if need_parameter_grads:
        if mask[2]:
            grad_bias = fast_aten_sum(grad_matrix, dim=[0], keepdim=False)
            if grad_bias is NOT_HANDLED:
                return NOT_HANDLED
        else:
            grad_bias = _alloc((output_features,), input._dtype, input._device)

    return grad_input, grad_weight, grad_bias


def fast_aten_bmm(input, mat2):
    out = _try_bf16_bmm(input, mat2)
    if out is not None:
        return out
    out = _try_tf32_bmm(input, mat2)
    if out is not None:
        return out
    out = _try_spec_matmul("BmmSpec", (input, mat2), 0)
    return out if out is not None else NOT_HANDLED


def _fast_aten_bmm_transpose_b(input, mat2):
    """Batched ``input @ mat2.transpose(-2, -1)`` without a transpose copy.

    This is an internal eager-autograd helper rather than an ATen registration:
    ``BmmSpec`` passes the logical RHS-transpose flag directly to the existing
    GEMM kernel while both physical operands remain dense row-major tensors.
    """
    out = _try_bf16_bmm(input, mat2, transpose_b=True)
    if out is not None:
        return out
    out = _try_tf32_bmm(input, mat2, transpose_b=True)
    if out is not None:
        return out
    out = _try_spec_matmul("BmmSpec", (input, mat2), 1)
    return out if out is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Convolution, pure Mojo: batched im2col + the pure GEMM with the torch
# (K,C,R,S) weight used as-is and NCHW output — no layout permutes.
# Grouped convolutions slice the channel-major im2col rows and weights per
# group with element offsets.
# ---------------------------------------------------------------------------


def fast_aten_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    a = _tc(input)
    w = _tc(weight)
    bias_t = _tc(bias) if bias is not None else None
    strides = _pair(list(stride))
    pads = _pair(list(padding))
    dils = _pair(list(dilation))
    if (
        a is not None
        and w is not None
        and not transposed
        and (bias is None or bias_t is not None)
        and a._dtype == w._dtype
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and len(w._shape) == 4
        and isinstance(groups, int)
        and groups >= 1
        and None not in (strides, pads, dils)
        and 0 not in a._shape
    ):
        n, c, in_h, in_w = a._shape
        out_c, c_per_group, kh, kw = w._shape
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if c_per_group * groups == c and out_h > 0 and out_w > 0:
            if bias_t is not None and (
                bias_t._dtype != a._dtype or tuple(bias_t._shape) != (out_c,)
            ):
                return NOT_HANDLED
            ctx = _ctx_ptr(a._device)
            cols = out_h * out_w
            ckk = c * kh * kw
            if (kh, kw, sh, sw, ph, pw, dh, dw) == (1, 1, 1, 1, 0, 0, 1, 1):
                # 1x1 stride-1 conv: NCHW input already is the col matrix.
                col_ptr = a._ptr
            else:
                col = _alloc((n, ckk, cols), a._dtype, a._device)
                _call_mojo(
                    _ConvExtension,
                    "Im2col",
                    (
                        col._ptr,
                        a._ptr,
                        (
                            in_h,
                            in_w,
                            out_h,
                            out_w,
                            kh,
                            kw,
                            sh,
                            sw,
                            ph,
                            pw,
                            dh,
                            dw,
                            c,
                            n,
                        ),
                        a._dtype.value,
                        ctx,
                    ),
                    arg_dtypes=(a._dtype,),
                    output_dtypes=(col._dtype,),
                    keepalive=(col, a),
                )
                col_ptr = col._ptr
            out = _alloc((n, out_c, cols), a._dtype, a._device)
            if groups == 1:
                _call_mojo(
                    _MatmulExtension,
                    "Bmm",
                    (
                        out._ptr,
                        w._ptr,
                        col_ptr,
                        (n, out_c, cols, ckk, 0, 1),
                        a._dtype.value,
                        ctx,
                    ),
                    arg_dtypes=(w._dtype, a._dtype),
                    output_dtypes=(out._dtype,),
                    # Same define shape as every other Bmm site: the shared-A
                    # broadcast is RUNTIME data (the trailing 1 in the params
                    # tuple, matmul_ops._bmm_go), so naming it here would only
                    # fork this call site onto a second .so of identical code.
                    flags={"TRANSPOSE_B": False},
                    keepalive=(out, w),
                )
            else:
                # Channel-major im2col rows make each group a contiguous
                # (crs_g, cols) slice; run one offset GEMM per (sample, group).
                crs_g = c_per_group * kh * kw
                oc_g = out_c // groups
                for s in range(n):
                    for g in range(groups):
                        _call_mojo(
                            _MatmulExtension,
                            "Matmul",
                            (
                                out._ptr,
                                w._ptr,
                                col_ptr,
                                (
                                    oc_g,
                                    cols,
                                    crs_g,
                                    0,
                                    (s * out_c + g * oc_g) * cols,
                                    g * oc_g * crs_g,
                                    (s * c + g * c_per_group) * kh * kw * cols,
                                ),
                                a._dtype.value,
                                ctx,
                            ),
                            arg_dtypes=(w._dtype, a._dtype),
                            output_dtypes=(out._dtype,),
                            flags={"TRANSPOSE_B": False},
                            keepalive=(out, w),
                        )
            if bias_t is not None:
                _call_mojo(
                    _ConvExtension,
                    "BiasAddChan",
                    (
                        out._ptr,
                        bias_t._ptr,
                        (cols, out_c, n * out_c * cols),
                        a._dtype.value,
                        ctx,
                    ),
                    arg_dtypes=(out._dtype, bias_t._dtype),
                    output_dtypes=(out._dtype,),
                    keepalive=(out, bias_t),
                )
            return _view_of(
                out,
                (n, out_c, out_h, out_w),
                _row_major_strides((n, out_c, out_h, out_w)),
                out._offset,
            )
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Attention: decomposed scaled dot product attention (bmm + fused
# scale/causal softmax + bmm), with a fused single-kernel decode path.
# ---------------------------------------------------------------------------


def fast_aten_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if attn_mask is not None:
        # None of the fused attention kernels take a mask operand, so a masked
        # call goes straight to the decomposed masked forward.
        if enable_gqa or dropout_p != 0.0:
            return NOT_HANDLED
        result = _sdpa_masked_math_forward(
            query, key, value, attn_mask, is_causal, scale
        )
        if result is NOT_HANDLED:
            return NOT_HANDLED
        out, _ = result
        return out

    fa4_result = fast_fa4_bf16_d64_causal_forward(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    if fa4_result is not NOT_HANDLED:
        output, logsumexp, q_native, k_native, v_native = fa4_result
        del logsumexp, q_native, k_native, v_native
        return output

    fused = fast_fused_flash_attention_forward(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    if fused is not NOT_HANDLED:
        output, lse, q_used, k_used, v_used = fused
        del lse, q_used, k_used, v_used
        return output

    q = _t(query)
    k = _t(key)
    v = _t(value)
    if (
        q is not None
        and k is not None
        and v is not None
        and attn_mask is None
        and (
            dropout_p == 0.0
            or (
                isinstance(dropout_p, int | float)
                and 0.0 < dropout_p <= 1.0
                and q._dtype == DType.float32
                and _on_gpu(q)
            )
        )
        and q._device == k._device == v._device
        and q._dtype == k._dtype == v._dtype
        and q._dtype in _FLOAT_DTYPES
        and len(q._shape) == 4
        and tuple(k._shape) == tuple(v._shape)
        and tuple(q._shape[:2]) == tuple(k._shape[:2])
        and q._shape[3] == k._shape[3]
        and 0 not in q._shape
        and 0 not in k._shape
    ):
        b, h, q_len, head_dim = q._shape
        kv_len = k._shape[2]
        scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        if (
            dropout_p == 0.0
            and _on_gpu(q)
            and q_len == 1
            and not is_causal
            and head_dim % 4 == 0
            and head_dim <= 256
            and kv_len <= 4096
            and q._strides[3] == 1
            and k._strides[3] == v._strides[3] == 1
            and k._strides[2] == v._strides[2] == head_dim
        ):
            # Decode step: one fused kernel instead of bmm+softmax+bmm
            # (single launch, no scratch buffers, coalesced K/V reads).
            # q reads through its (batch, head) strides, so the per-head
            # transpose view of the fused qkv projection is NOT
            # materialized first.
            out = _alloc((b, h, 1, head_dim), q._dtype, q._device)
            _call_mojo(
                _NNExtension,
                "AttnDecodeSpec",
                (
                    _spec_of(q),
                    _spec_of(k),
                    _spec_of(v),
                    float(scale_val),
                    _spec_of(out),
                ),
                arg_dtypes=(q._dtype, k._dtype, v._dtype),
                output_dtypes=(out._dtype,),
                keepalive=(q, k, v, out),
            )
            return out
        result = _sdpa_math_forward_with_dropout(
            q, k, v, is_causal, scale, float(dropout_p)
        )
        if result is NOT_HANDLED:
            return NOT_HANDLED
        out, _, _ = result
        return out
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Softmax (the SDPA SoftmaxRows kernel with scale=1, no causal mask).
# Non-trailing dims go through a zero-copy transpose + materialize.
# ---------------------------------------------------------------------------


def fast_aten__softmax(input, dim, half_to_float=False):
    t = _t(input)
    if (
        t is None
        or t._numel == 0
        or t._dtype not in _FLOAT_DTYPES
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    # half_to_float: half input, float32 output (torch computes in fp32).
    if half_to_float:
        if t._dtype not in (DType.float16, DType.bfloat16):
            return NOT_HANDLED
        t = _cast_tensor(t, DType.float32)
    rank = len(t._shape)
    if rank == 0:
        return fast_filled((), 1.0, t._dtype, t._device)
    dim %= rank
    if dim != rank - 1:
        # softmax(x, d) = softmax(x.transpose(d, -1), -1).transpose(d, -1);
        # both transposes are zero-copy, the inner one materializes once.
        # (t is already fp32 here if half_to_float was set, so pass False.)
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__softmax(swapped, rank - 1, False)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    result = _try_spec_unary("SoftmaxSpec", t, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


def fast_aten_softmax(input, dim=-1, dtype=None):
    if dtype is not None:
        return NOT_HANDLED
    return fast_aten__softmax(input, dim, False)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def fast_aten_embedding(
    input, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    # `input` is the weight table, `weight` the indices (aten naming).
    table = _t(input)
    idx = _t(weight)
    if (
        table is None
        or idx is None
        or table._device != idx._device
        or table._dtype not in _FLOAT_DTYPES
        or idx._dtype not in (DType.int32, DType.int64)
        or len(table._shape) != 2
    ):
        return NOT_HANDLED

    # Validate before either operand can be materialized.  Besides avoiding
    # needless work, this keeps cross-device inputs from being enqueued on the
    # table's context.
    table = _tc(table)
    idx = _tc(idx)
    row_len = table._shape[1]
    out_shape = tuple(idx._shape) + (row_len,)
    out = _alloc(out_shape, table._dtype, table._device)
    if out._numel > 0:
        _call_mojo(
            _NNExtension,
            "Gather0",
            (
                out._ptr,
                table._ptr,
                idx._ptr,
                idx._dtype.value,
                idx._numel,
                row_len,
                table._dtype.value,
                _ctx_ptr(table._device),
            ),
            arg_dtypes=(table._dtype, idx._dtype),
            output_dtypes=(out._dtype,),
            keepalive=(out, table, idx),
        )
    return out


def fast_aten_embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
):
    """FP32/int64 GPU embedding backward through the Fable-owned kernel."""
    grad = _t(grad_output)
    idx = _t(indices)

    # Reject unsupported public modes here before materialization, allocation,
    # or launch.  The raw bridge also propagates defensive candidate-side
    # errors, but those must not be the normal validation path.
    if scale_grad_by_freq is not False:
        raise NotImplementedError(
            "Mojo eager embedding_dense_backward does not yet support "
            "scale_grad_by_freq=True"
        )
    _alert_not_deterministic("embedding_dense_backward on Mojo")

    if (
        grad is None
        or idx is None
        or not _on_gpu(grad)
        or grad._device != idx._device
        or grad._dtype != DType.float32
        or idx._dtype != DType.int64
        or len(grad._shape) < 1
        or not isinstance(num_weights, int)
        or isinstance(num_weights, bool)
        or num_weights < 0
        or not isinstance(padding_idx, int)
        or isinstance(padding_idx, bool)
    ):
        return NOT_HANDLED

    embedding_dim = grad._shape[-1]
    if grad._numel != idx._numel * embedding_dim:
        return NOT_HANDLED

    grad = _tc(grad)
    idx = _tc(idx)
    grad_weight = _alloc((num_weights, embedding_dim), DType.float32, grad._device)
    if grad_weight._numel > 0:
        # This call includes complete output zeroing and accumulation, stays on
        # the tensor's supplied context, and returns asynchronously.
        _call_mojo(
            _EmbeddingBackwardExtension,
            "EmbeddingDenseBackwardF32I64",
            (
                grad_weight._ptr,
                grad._ptr,
                idx._ptr,
                idx._numel,
                embedding_dim,
                num_weights,
                padding_idx,
                0,
                _ctx_ptr(grad._device),
            ),
            arg_dtypes=(grad._dtype, idx._dtype),
            output_dtypes=(grad_weight._dtype,),
            keepalive=(grad_weight, grad, idx),
        )
    return grad_weight


# ---------------------------------------------------------------------------
# Filled factories (full / ones / zeros / scalar_tensor): one allocation
# plus one Fill kernel. The registrations resolve torch dtype/device.
# ---------------------------------------------------------------------------


def fast_filled(shape, value, dtype: DType, device):
    """A TorchMojoTensor of `shape` filled with `value`, or None."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int | float):
        return None
    if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
        return None
    if dtype not in _FILL_DTYPES:
        return None
    if dtype == DType.float64 and device.api == "metal":
        return None
    shape = tuple(shape)
    return _submit_prepared_into(
        _FillSpecExtension.prepare(shape, float(value), dtype, device)
    )


# What the Arange kernel dispatches on (_FILL_DTYPES minus bool, which
# torch.arange rejects anyway).
_ARANGE_DTYPES = _FILL_DTYPES[:-1]


def fast_arange(numel, start, step, dtype: DType, device):
    """A 1-D TorchMojoTensor holding start + i*step (device kernel), or None.

    The caller resolves torch's arange semantics (numel, dtype). Inputs arrive
    through Float64, while the kernel uses PyTorch's device-specific
    accumulator type; integers beyond the exact-f64 range must not reach here.
    """
    if dtype not in _ARANGE_DTYPES:
        return None
    if dtype == DType.float64 and device.api == "metal":
        return None
    out = _alloc((numel,), dtype, device)
    if numel > 0:
        _call_mojo(
            _ElementwiseExtension,
            "Arange",
            (out._ptr, float(start), float(step), numel, dtype.value, _ctx_ptr(device)),
            arg_dtypes=(),
            output_dtypes=(out._dtype,),
            keepalive=(out,),
        )
    return out


# ---------------------------------------------------------------------------
# Scalar readback
# ---------------------------------------------------------------------------


def fast_aten__local_scalar_dense(tensor):
    t = _t(tensor)
    if t is None or t._numel != 1:
        return NOT_HANDLED
    _call_queue.drain()  # host read: queued launches must land first
    return eager_kernels.tensor_holder.read_scalar(
        _ctx_ptr(t._device), t._ptr, t._dtype.value
    )


def _instrument_call_counts():
    """Give every fast op a test-only call counter, mirroring what
    `aten_functions.map_to` does, so `CallChecker` can assert that an op
    was handled by either implementation."""
    import functools

    for name, func in list(globals().items()):
        if not name.startswith("fast_aten"):
            continue

        def make_wrapper(wrapped):
            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                wrapper.call_count += 1
                return wrapped(*args, **kwargs)

            wrapper.call_count = 0
            return wrapper

        globals()[name] = make_wrapper(func)


if is_running_tests.IS_RUNNING_TESTS:
    _instrument_call_counts()
