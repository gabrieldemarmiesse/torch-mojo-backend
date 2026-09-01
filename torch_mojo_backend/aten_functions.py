"""How to execute Pytorch's Aten functions using Max's backend.

The only ressources I could find on the subject are:
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
- https://docs.pytorch.org/docs/stable/torch.compiler_ir.html
"""

import contextvars
import functools
import itertools
import math
import operator
import os
from typing import Literal

import max.driver as max_driver
import max.graph.ops as max_ops
import max.graph.type as max_type
import torch
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.random import gaussian as max_gaussian
from max.experimental.torch import max_dtype_to_torch
from max.experimental.torch.torch import max_device_ref, torch_dtype_to_max
from max.graph import Dim, StaticDim
from max.graph.type import DeviceRef
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from torch._decomp import core_aten_decompositions
from torch._ops import OpOverload, OpOverloadPacket
from torch.ops import aten

import torch_mojo_backend.is_running_tests
from torch_mojo_backend import custom_mojo_ops
from torch_mojo_backend.flags import verbose_enabled
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators
from torch_mojo_backend.types import MaxTensor, Scalar, SymIntType

flash_attention_gpu = F.functional(flash_attention_gpu)

# MAX made the `functional` reduction helpers (mean/sum/argmax/argmin and the
# reduction forms of max/min) eager-only: they assert their input is an eager
# Tensor and so reject the graph TensorValues used by the torch.compile backend.
# Re-wrap the bare graph ops with `functional` (which accepts both eager Tensors
# and graph values) to restore dual-path support.
_reduce_mean = F.functional(max_ops.mean)
_reduce_sum = F.functional(max_ops.sum)
_reduce_max = F.functional(max_ops.reduction.max)
_reduce_min = F.functional(max_ops.reduction.min)
_reduce_argmax = F.functional(max_ops.argmax)
_reduce_argmin = F.functional(max_ops.argmin)
# F.transfer_to is eager-only (reads Tensor.real); re-wrap the graph op so it
# also works on graph TensorValues in the torch.compile backend.
_transfer_to = F.functional(max_ops.transfer_to)


def find_broadcast_shape(shape_a: list[Dim], shape_b: list[Dim]) -> list[Dim]:
    # A rank-0 shape broadcasts like a scalar: zip_longest fills its
    # missing dims with None, which resolve to the other shape's dims.
    result = []
    for dim_a, dim_b in itertools.zip_longest(reversed(shape_a), reversed(shape_b)):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a in (Dim(1), None):
            result.append(dim_b)
        elif dim_b in (Dim(1), None):
            result.append(dim_a)
        else:
            raise ValueError(
                f"Broadcast is not possible between shapes {shape_a} and {shape_b}"
            )
    return list(reversed(result))


def torch_device_to_max_device(x: torch.device) -> DeviceRef:
    if x.type == "mojo":
        # For mojo, use ordered accelerators (GPU first, CPU last)
        # index None or 0 = first accelerator (first GPU or CPU if no GPU)
        # higher indices = additional GPUs, with CPU at the highest index
        index = x.index if x.index is not None else 0

        accelerators = get_ordered_accelerators()
        if index >= len(accelerators):
            raise ValueError(f"Invalid mojo index {index}")

        device = accelerators[index]
        if device.label == "cpu":
            return DeviceRef.CPU()
        else:
            return DeviceRef.GPU(device.id)  # Use the actual GPU ID
    else:
        return max_device_ref(x)


# The fx node currently being converted to MAX ops, set by the compiler's
# handle_call_function. Lets multi-output mappings check which outputs are
# actually consumed (e.g. native_layer_norm's mean/rstd are dead in
# inference graphs, enabling the fused kernel).
CURRENT_FX_NODE: contextvars.ContextVar = contextvars.ContextVar(
    "CURRENT_FX_NODE", default=None
)


def _only_first_output_used() -> bool:
    node = CURRENT_FX_NODE.get()
    if node is None:
        return False
    return all(
        user.target is operator.getitem and user.args[1] == 0 for user in node.users
    )


# Ops that need to be decomposed.
DECOMPOSITION_TABLE = core_aten_decompositions()
original_decomposition_table_size = len(DECOMPOSITION_TABLE)
# Initialize the mapping dictionary
MAPPING_TORCH_ATEN_TO_MOJO = {}


IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    operator.matmul,
    operator.neg,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.ifloordiv,
    operator.ipow,
    operator.imod,
    operator.getitem,
    str,
    max,
    min,
]

# Map identical functions
for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_ATEN_TO_MOJO[func] = func

number_of_decompositions_removed = 0


def map_to(func):
    def decorator(func_to_map):
        if os.environ.get("TORCH_MOJO_BACKEND_BEARTYPE", "0") == "1":
            from beartype import beartype

            func_to_map = beartype(func_to_map)

        # We count the number of calls here because otherwise it's hard to
        # get it with mock since the functions are all gathered at import time
        # into dicts.
        if torch_mojo_backend.is_running_tests.IS_RUNNING_TESTS:

            @functools.wraps(func_to_map)
            def wrapped_func(*args, **kwargs):
                wrapped_func.call_count += 1
                return func_to_map(*args, **kwargs)

            wrapped_func.call_count = 0

            MAPPING_TORCH_ATEN_TO_MOJO[func] = wrapped_func
        else:
            MAPPING_TORCH_ATEN_TO_MOJO[func] = func_to_map
        if isinstance(func, OpOverload):
            DECOMPOSITION_TABLE.pop(func, None)
        elif isinstance(func, OpOverloadPacket):
            # We assume we cover all overloads in the packet
            for overload_name in func:
                popped = DECOMPOSITION_TABLE.pop(getattr(func, overload_name), None)
                if verbose_enabled() and popped is not None:
                    global number_of_decompositions_removed
                    number_of_decompositions_removed += 1

        else:
            raise TypeError(
                f"Expected OpOverload or OpOverloadPacket, got {type(func)}"
            )
        if torch_mojo_backend.is_running_tests.IS_RUNNING_TESTS:
            return wrapped_func
        else:
            return func_to_map

    return decorator


# Add direct mappings with decorators


def get_float_dtype(x, y):
    for t in (x, y):
        if t.dtype.is_float():
            return t.dtype


def get_int_dtype(x, y):
    for t in (x, y):
        if t.dtype.is_integral():
            return t.dtype


def _materialize_dim(dim: Dim, like=None):
    """A symbolic Dim as a scalar tensor, usable in tensor arithmetic.

    torch treats SymInt operands like python scalars: the result keeps the
    tensor operand's dtype, so cast to `like`'s dtype (and move to its
    device) when a companion tensor is given.
    """
    t = max_ops.shape_to_tensor([dim])
    t = F.reshape(t, [])
    if like is not None:
        t = F.cast(t, dtype=like.dtype)
        if t.device != like.device:
            t = _transfer_to(t, like.device)
    return t


def type_promotion(x, y):
    if isinstance(x, Dim) and isinstance(y, Dim):
        x = _materialize_dim(x)
        y = _materialize_dim(y, x)
    elif isinstance(x, Dim):
        x = _materialize_dim(x, y)
    elif isinstance(y, Dim):
        y = _materialize_dim(y, x)
    if isinstance(x, int | float) or isinstance(y, int | float):
        # case not handled yet
        return x, y

    float_dtype = get_float_dtype(x, y)
    int_dtype = get_int_dtype(x, y)
    if float_dtype is not None and int_dtype is not None:
        # If both are float and int, promote to float
        x = F.cast(x, dtype=float_dtype)
        y = F.cast(y, dtype=float_dtype)

    return x, y


_SEARCHSORTED_DTYPES = (
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.int32,
    DType.int64,
)


def _searchsorted_tensor_dtype(left: DType, right: DType) -> DType:
    """PyTorch's result-type promotion for two comparison tensors."""
    promoted = torch.promote_types(max_dtype_to_torch(left), max_dtype_to_torch(right))
    return torch_dtype_to_max(promoted)


def _searchsorted_scalar_dtype(boundary_dtype: DType, value: Scalar) -> DType:
    """Wrapped-number promotion used by ATen's Scalar overloads."""
    promoted = torch.result_type(
        torch.empty((), dtype=max_dtype_to_torch(boundary_dtype)), value
    )
    return torch_dtype_to_max(promoted)


def _searchsorted_impl(
    sorted_sequence: MaxTensor,
    values: MaxTensor | Scalar,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: MaxTensor | None,
) -> MaxTensor:
    """MAX graph composition shared by searchsorted and bucketize."""
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(
                "torch.searchsorted(): side can only be 'left' or 'right' but "
                f"got {side}"
            )
        if right and side != "right":
            raise RuntimeError(
                "torch.searchsorted(): side and right can't be set to opposites, "
                f"got side of {side} while right was True"
            )
        right = side == "right"

    if len(sorted_sequence.shape) == 0:
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should have positive "
            "dimension, but got 0 dimension"
        )

    scalar_values = isinstance(values, int | float | Dim)
    if scalar_values:
        if len(sorted_sequence.shape) != 1:
            raise RuntimeError(
                "torch.searchsorted(): input value can be a scalar only when "
                "boundaries tensor dimension is 1"
            )
        if isinstance(values, Dim):
            raise NotImplementedError(
                "symbolic scalar values are not supported by searchsorted"
            )
        common_dtype = _searchsorted_scalar_dtype(sorted_sequence.dtype, values)
        values = F.constant(values, dtype=common_dtype, device=sorted_sequence.device)
    else:
        if values.device != sorted_sequence.device:
            raise RuntimeError(
                "torch.searchsorted(): boundaries and input value tensors "
                "should have same device type"
            )
        if len(values.shape) == 0 and len(sorted_sequence.shape) != 1:
            raise RuntimeError(
                "torch.searchsorted(): input value can be a scalar only when "
                "boundaries tensor dimension is 1"
            )
        if len(sorted_sequence.shape) != 1 and (
            len(values.shape) != len(sorted_sequence.shape)
            or tuple(values.shape[:-1]) != tuple(sorted_sequence.shape[:-1])
        ):
            raise RuntimeError(
                "torch.searchsorted(): boundaries tensor should be 1 dimension "
                "or the first N-1 dimensions of boundaries tensor and input "
                "value tensor must match"
            )
        common_dtype = _searchsorted_tensor_dtype(sorted_sequence.dtype, values.dtype)

    if common_dtype not in _SEARCHSORTED_DTYPES:
        raise NotImplementedError(
            f"searchsorted comparison dtype {common_dtype} is not supported"
        )

    if sorter is not None:
        if sorter.device != sorted_sequence.device:
            raise RuntimeError(
                "torch.searchsorted(): sorter and boundary tensors should have "
                "same device type"
            )
        if tuple(sorter.shape) != tuple(sorted_sequence.shape):
            raise RuntimeError(
                "torch.searchsorted(): boundary and sorter must have the same size"
            )
        if sorter.dtype != DType.int64:
            raise RuntimeError(
                "torch.searchsorted(): sorter must be a tensor of long dtype"
            )
        # MAX graphs cannot validate data-dependent sorter bounds before a
        # gather_nd. Passing unchecked indices through silently returned wrong
        # answers for out-of-range sorters, so decline the graph sorter path.
        raise NotImplementedError(
            "searchsorted sorter is not supported by the MAX graph backend "
            "because sorter bounds cannot be validated"
        )

    if sorted_sequence.dtype != common_dtype:
        sorted_sequence = F.cast(sorted_sequence, dtype=common_dtype)
    if values.dtype != common_dtype:
        values = F.cast(values, dtype=common_dtype)

    output_dtype = DType.int32 if out_int32 else DType.int64
    boundary_size = sorted_sequence.shape[-1]
    if isinstance(boundary_size, StaticDim) and int(boundary_size) == 0:
        zero = F.constant(0, dtype=output_dtype, device=values.device)
        return F.broadcast_to(zero, values.shape)

    values = F.unsqueeze(values, axis=-1)
    if len(sorted_sequence.shape) != 1:
        sorted_sequence = F.unsqueeze(sorted_sequence, axis=-2)
    comparison = (
        F.greater(sorted_sequence, values)
        if right
        else F.greater_equal(sorted_sequence, values)
    )
    count_mask = F.logical_not(comparison)
    if common_dtype.is_float():
        # For a sorted numeric prefix followed by NaNs, ATen's
        # !(boundary >/>= value) search skips the NaN tail only when the
        # numeric insertion point reaches it. A plain count treats every NaN
        # as an advance for every value, so exclude them from the numeric
        # count and remap that terminal insertion point to the full size.
        boundary_is_nan = F.is_nan(sorted_sequence)
        boundary_is_numeric = F.logical_not(boundary_is_nan)
        count_mask = F.logical_and(count_mask, boundary_is_numeric)
        numeric_boundary_count = _reduce_sum(
            F.cast(boundary_is_numeric, dtype=DType.int32), axis=-1
        )
        total_boundary_count = numeric_boundary_count + _reduce_sum(
            F.cast(boundary_is_nan, dtype=DType.int32), axis=-1
        )
    # MAX has no binary-search primitive, so this graph composition necessarily
    # materializes the broadcast N x K mask. Accumulate it as int32 to halve
    # peak memory versus an int64 mask, then cast only the reduced result.
    counts = _reduce_sum(F.cast(count_mask, dtype=DType.int32), axis=-1)
    if common_dtype.is_float():
        counts = F.where(
            F.equal(counts, numeric_boundary_count), total_boundary_count, counts
        )
    return F.squeeze(F.cast(counts, dtype=output_dtype), axis=-1)


@map_to(aten.floordiv)
def aten_floordiv(x, y):
    return operator.floordiv(x, y)


# _local_scalar_dense(Tensor self) -> Scalar
@map_to(aten._local_scalar_dense)
def aten__local_scalar_dense(tensor: MaxTensor) -> Scalar:
    if tensor.num_elements() != 1:
        raise ValueError(
            f"_local_scalar_dense requires a tensor with a single element, got {tensor.num_elements()} elements"
        )
    # We need to convert the result to a Python scalar for PyTorch to recognize it as a valid return type for this operator
    return tensor.item()


@map_to(aten.all)
def aten_all(
    input: MaxTensor, dim: list[int] | int | None = None, keepdim: bool = False
) -> MaxTensor:
    if input.dtype == DType.bool:
        input_bool = input
    else:
        input_bool = F.not_equal(input, 0)
    if isinstance(dim, int):
        dim = [dim]
    if dim is None:
        # Return True if any element is True (reduce all dimensions)
        dim = tuple(range(len(input.shape)))

    # Handle negative dimensions
    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    result = input_bool.cast(DType.int32)
    # Use max() to implement any() since True > False
    for axis in sorted(dim, reverse=True):
        result = _reduce_min(result, axis=axis)

    # Handle keepdim=False
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = F.squeeze(result, axis=axis)

    return result.cast(DType.bool)


# _adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
@map_to(aten._adaptive_avg_pool2d)
def aten__adaptive_avg_pool2d(
    input: MaxTensor, output_size: list[SymIntType]
) -> MaxTensor:
    # For now, we'll implement this using global average pooling for (1, 1) output
    # and regular avg pooling for other sizes
    if output_size == (1, 1) or output_size == 1:
        # Global average pooling - take mean over spatial dimensions
        return aten_mean(input, dim=(2, 3), keepdim=True)
    else:
        # For other output sizes, we'll use avg_pool2d with calculated kernel size and stride
        # Get input spatial dimensions (assuming NCHW format)
        input_h, input_w = input.shape[2], input.shape[3]
        try:
            input_h = int(input_h)
            input_w = int(input_w)
        except TypeError:
            pass

        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size

        # Calculate kernel size and stride to achieve the desired output size
        kernel_h = input_h // output_h
        kernel_w = input_w // output_w
        stride_h = input_h // output_h
        stride_w = input_w // output_w

        # Convert input from NCHW to NHWC for MAX
        input_nhwc = input.permute([0, 2, 3, 1])

        result = F.avg_pool2d(
            input_nhwc,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(0, 0),
            ceil_mode=False,
            count_boundary=True,
        )

        # Convert result back from NHWC to NCHW
        return result.permute([0, 3, 1, 2])


# _adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
@map_to(aten._adaptive_avg_pool2d_backward)
def aten__adaptive_avg_pool2d_backward(
    grad_output: MaxTensor, input_tensor: MaxTensor
) -> MaxTensor:
    """Compute gradient for adaptive average pooling 2d backward pass.

    Args:
        grad_output: Gradient from the output, shape (N, C, H_out, W_out) or (C, H_out, W_out)
        input_tensor: Original input tensor, shape (N, C, H_in, W_in) or (C, H_in, W_in)

    Returns:
        Gradient with respect to input, same shape as input_tensor
    """
    raise NotImplementedError()


# _adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
# _cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
# _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
# _fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
# _local_scalar_dense(Tensor self) -> Scalar
# _log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
# _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# _native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# _native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten._native_batch_norm_legit_no_training)
def aten__native_batch_norm_legit_no_training(
    input: MaxTensor,
    weight: MaxTensor | None,
    bias: MaxTensor | None,
    running_mean: MaxTensor,
    running_var: MaxTensor,
    momentum: float,
    eps: float,
) -> tuple[MaxTensor, NotImplementedError, NotImplementedError]:
    """
    Implements batch normalization for inference (no training).

    Args:
        input: Input tensor of shape (N, C, H, W) or (N, C, ...)
        weight: Optional gamma parameter tensor of shape (C,)
        bias: Optional beta parameter tensor of shape (C,)
        running_mean: Running mean statistics tensor of shape (C,)
        running_var: Running variance statistics tensor of shape (C,)
        momentum: Momentum factor (unused in no-training mode)
        eps: Small value for numerical stability

    Returns:
        Tuple of (normalized_output, save_mean, save_var)
        where save_mean and save_var are empty tensors in no-training mode
    """
    # Get input dimensions
    input_shape = input.shape
    num_channels = int(input_shape[1])  # Channel dimension is always 1 in NCHW format

    # Reshape running statistics to broadcast properly: (C,) -> (1, C, 1, 1, ...)
    # Create broadcast shape with 1s for all dims except channel dim
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[1] = num_channels  # Set channel dimension

    # Reshape running mean and variance for broadcasting
    running_mean_reshaped = F.reshape(running_mean, broadcast_shape)
    running_var_reshaped = F.reshape(running_var, broadcast_shape)

    # Compute normalization: (input - mean) / sqrt(var + eps)
    normalized = (input - running_mean_reshaped) / F.sqrt(running_var_reshaped + eps)

    # Apply weight (gamma) and bias (beta) if provided
    if weight is not None:
        weight_reshaped = F.reshape(weight, broadcast_shape)
        normalized = normalized * weight_reshaped

    if bias is not None:
        bias_reshaped = F.reshape(bias, broadcast_shape)
        normalized = normalized + bias_reshaped

    # It's not sure we'll ever support returning those, notably because of
    # https://github.com/pytorch/pytorch/issues/85960
    return (
        normalized,
        NotImplementedError(
            "We don't support returning the saved mean "
            "in aten._native_batch_norm_legit_no_training yet"
        ),
        NotImplementedError(
            "We don't support returning the saved variance "
            "in aten._native_batch_norm_legit_no_training yet"
        ),
    )


# _pdist_forward(Tensor self, float p=2) -> Tensor


# _scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)
@map_to(aten._scaled_dot_product_attention_math)
def aten__scaled_dot_product_attention_math(
    query: MaxTensor,
    key: MaxTensor,
    value: MaxTensor,
    attn_mask: MaxTensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: MaxTensor | None = None,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
):
    if dropout_mask is not None:
        raise NotImplementedError(
            "dropout_mask is not supported in aten._scaled_dot_product_attention_math yet"
        )
    if enable_gqa:
        # Grouped-query attention: broadcast each KV head over its query
        # group so the math below sees matching head counts (the same
        # repeat_interleave(dim=1) torch's reference implementation does).
        q_heads = query.shape[1]
        kv_heads = key.shape[1]
        if q_heads != kv_heads:
            group = int(q_heads) // int(kv_heads)

            def _repeat_kv(x: MaxTensor) -> MaxTensor:
                b, h, s, d = x.shape
                x = F.reshape(x, (b, h, 1, s, d))
                x = F.broadcast_to(x, (b, h, group, s, d))
                return F.reshape(x, (b, int(h) * group, s, d))

            key = _repeat_kv(key)
            value = _repeat_kv(value)
    if attn_mask is None:
        # No custom mask: delegate to flash attention (handles is_causal) and use its
        # first two return values as (output, logsumexp).
        output, logsumexp, *_ = aten__scaled_dot_product_flash_attention(
            query, key, value, dropout_p, is_causal, False, scale
        )
        return output, logsumexp

    # Custom attn_mask path: compute attention manually so the mask can be applied
    # to the pre-softmax scores.
    head_dim = query.shape[-1]
    head_dim_val = int(head_dim) if isinstance(head_dim, Dim) else head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(float(head_dim_val))

    key_transposed = F.transpose(key, -2, -1)
    scores = F.matmul(query, key_transposed) * scale

    # Match PyTorch's math reference: the mask is always added to scores.
    # A bool mask is promoted to the scores dtype (True -> 1.0, False -> 0.0).
    if attn_mask.dtype != scores.dtype:
        attn_mask = F.cast(attn_mask, dtype=scores.dtype)
    scores = scores + attn_mask

    attention = aten_softmax(scores, dim=-1)
    output = F.matmul(attention, value)
    return output, attention


# _scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)
@map_to(aten._scaled_dot_product_efficient_attention)
def aten__scaled_dot_product_efficient_attention(
    query: MaxTensor,
    key: MaxTensor,
    value: MaxTensor,
    attn_bias: MaxTensor | None,
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
):
    # The overload cuda picks for fp32 sdpa. Only `output` is consumed at
    # inference (the philox seed/offset outputs exist for dropout replay in
    # training).
    if attn_bias is None and not is_causal:
        # Plain attention: compute manually with matmuls instead of the
        # flash kernel, whose numerics are noticeably looser.
        head_dim = query.shape[-1]
        head_dim_val = int(head_dim) if isinstance(head_dim, Dim) else head_dim
        if scale is None:
            scale = 1.0 / math.sqrt(float(head_dim_val))
        scores = F.matmul(query, F.transpose(key, -2, -1)) * scale
        attention = aten_softmax(scores, dim=-1)
        return F.matmul(attention, value), attention, None, None
    output, weights_or_logsumexp = aten__scaled_dot_product_attention_math(
        query, key, value, attn_bias, dropout_p, is_causal, scale=scale
    )
    return output, weights_or_logsumexp, None, None


# _scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor, SymInt, SymInt, Tensor, Tensor, Tensor)
@map_to(aten._scaled_dot_product_flash_attention)
def aten__scaled_dot_product_flash_attention(
    query: MaxTensor,
    key: MaxTensor,
    value: MaxTensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
):
    # We return only the first element for now because we don't support training yet.
    # PyTorch provides tensors in shape [batch, num_heads, seq_len, head_dim]
    # MAX expects tensors in shape [batch, seq_len, num_heads, head_dim]

    # Transpose from PyTorch format to MAX format
    q = F.permute(query, [0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
    k = F.permute(key, [0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
    v = F.permute(value, [0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]

    # Calculate scale if not provided
    if scale is None:
        head_dim = query.shape[-1]
        if isinstance(head_dim, Dim):
            head_dim_value = int(head_dim)
        else:
            head_dim_value = head_dim
        scale = 1.0 / math.sqrt(float(head_dim_value))

    # Choose mask variant based on is_causal flag
    mask_variant = MHAMaskVariant.CAUSAL_MASK if is_causal else MHAMaskVariant.NULL_MASK

    # Flash attention kernels are currently only valid on GPU. Use a fake matmul-based
    # implementation on CPU to keep correctness on CPU-backed execution.
    query_device = query.device
    use_fake_flash_attention = False
    if isinstance(query_device, DeviceRef):
        use_fake_flash_attention = query_device.is_cpu()
    elif isinstance(query_device, max_driver.Device):
        use_fake_flash_attention = query_device.label == "cpu"
    if use_fake_flash_attention:
        # Fake path: compute attention with basic matmuls.
        q_heads = F.permute(q, [0, 2, 1, 3])  # [batch, heads, seq_len, head_dim]
        k_heads = F.permute(k, [0, 2, 1, 3])  # [batch, heads, seq_len, head_dim]
        v_heads = F.permute(v, [0, 2, 1, 3])  # [batch, heads, seq_len, head_dim]
        if q_heads.dtype == DType.bfloat16:
            q_heads = F.cast(q_heads, dtype=DType.float32)
            k_heads = F.cast(k_heads, dtype=DType.float32)
            v_heads = F.cast(v_heads, dtype=DType.float32)

        scores = F.matmul(q_heads, F.transpose(k_heads, 2, 3))
        scores = F.mul(scores, scale)
        if is_causal:
            raise NotImplementedError(
                "Causal attention is not implemented in fake matmul fallback for "
                "aten::_scaled_dot_product_flash_attention."
            )
        attention = aten_softmax(scores, dim=-1)
        attn_out = F.permute(F.matmul(attention, v_heads), [0, 2, 1, 3])
        if q.dtype == DType.bfloat16:
            attn_out = F.cast(attn_out, dtype=DType.bfloat16)
    else:
        # Call flash attention on GPU.
        attn_out = flash_attention_gpu(q, k, v, mask_variant=mask_variant, scale=scale)

    # Transpose back to PyTorch format [batch, num_heads, seq_len, head_dim]
    result = F.permute(attn_out, [0, 2, 1, 3])

    # Create additional outputs to match PyTorch's 9-value return contract for
    # aten::_scaled_dot_product_flash_attention.
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    seq_len = query.shape[2]
    seq_len_k = key.shape[2]
    head_dim = query.shape[3]

    batch_size_int = int(batch_size if isinstance(batch_size, Dim) else batch_size)
    num_heads_int = int(num_heads if isinstance(num_heads, Dim) else num_heads)
    seq_len_int = int(seq_len if isinstance(seq_len, Dim) else seq_len)
    seq_len_k_int = int(seq_len_k if isinstance(seq_len_k, Dim) else seq_len_k)
    head_dim_int = int(head_dim if isinstance(head_dim, Dim) else head_dim)

    logsumexp = F.broadcast_to(
        F.constant(0, dtype=DType.float32, device=result.device),
        [batch_size_int, num_heads_int, seq_len_int],
    )

    # cum_seq_q and cum_seq_k are cumulative sequence length vectors for packed
    # layouts; for dense attention the real backend returns None, but this backend
    # currently returns symbolic zero placeholders.
    cum_seq_q = F.broadcast_to(
        F.constant(0, dtype=DType.int32, device=result.device), [batch_size_int]
    )
    cum_seq_k = F.broadcast_to(
        F.constant(0, dtype=DType.int32, device=result.device), [batch_size_int]
    )

    # max_q and max_k are computed from input sequence dimensions.
    max_q = seq_len
    max_k = seq_len_k

    # RNG state placeholders: seed + offset tensors used by flash attention kernels.
    rng_state = F.broadcast_to(
        F.constant(0, dtype=DType.int64, device=result.device), [2]
    )
    unused = F.broadcast_to(F.constant(0, dtype=DType.int64, device=result.device), [])

    # debug_attn_mask is a compressed debug representation when requested; otherwise
    # it is returned as an empty tensor.
    if return_debug_mask:
        block_size = 128 if head_dim_int > 64 else 256
        if seq_len_k_int <= block_size:
            max_seqlen_k = block_size
        elif seq_len_k_int <= 2 * block_size:
            max_seqlen_k = 2 * block_size
        else:
            max_seqlen_k = math.ceil(seq_len_k_int / block_size)
        debug_attn_mask = F.broadcast_to(
            F.constant(0, dtype=result.dtype, device=result.device),
            [batch_size_int, num_heads_int, seq_len_int, max_seqlen_k],
        )
    else:
        debug_attn_mask = F.broadcast_to(
            F.constant(0, dtype=result.dtype, device=result.device), []
        )

    # Return tuple as expected by PyTorch:
    # output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, unused, debug_attn_mask
    return (
        result,  # output
        logsumexp,  # logsumexp
        cum_seq_q,  # cum_seq_q
        cum_seq_k,  # cum_seq_k
        max_q,  # max_q
        max_k,  # max_k
        rng_state,  # rng_state
        unused,  # unused
        debug_attn_mask,  # debug_attn_mask
    )


# _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
@map_to(aten._softmax)
def aten__softmax(self: MaxTensor, dim: int, half_to_float: bool):
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = None
    return aten_softmax(self, dim=dim, dtype=dtype)


@map_to(aten.softmax)
def aten_softmax(input, dim=-1, dtype=None):
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        input = F.cast(input, dtype=max_dtype)

    # Handle negative dim
    if dim < 0:
        dim = len(input.shape) + dim

    # Manual implementation
    # Compute max along the specified axis for numerical stability, keeping dimensions
    x_max = aten_amax(input, dim=[dim], keepdim=True)

    # Subtract max for numerical stability
    x_shifted = input - x_max

    # Compute exponential
    x_exp = F.exp(x_shifted)

    # Sum along the axis, keeping dimensions for broadcasting
    x_sum = aten_sum(x_exp, dim=[dim], keepdim=True)

    # Divide to get softmax
    return x_exp / x_sum


# aten._log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
@map_to(aten._log_softmax)
def aten__log_softmax(input: MaxTensor, dim: int, half_to_float: bool) -> MaxTensor:
    """Compute log(softmax(x)) in a numerically stable way.

    Formula: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

    This is more numerically stable than computing log(softmax(x)) directly,
    as it prevents overflow from exp(large_values) and underflow from log(small_values).

    Based on PyTorch's implementation:
    pytorch/aten/src/ATen/native/SoftMax.cpp (TORCH_META_FUNC(_log_softmax))
    pytorch/aten/src/ATen/native/cuda/SoftMax.cu (log_softmax_cuda_out)
    pytorch/torch/_decomp/decompositions.py (_log_softmax decomposition)

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
        half_to_float: If True, input must be float16 and output will be float32.
                      If False, output dtype matches input dtype.

    Returns:
        Log-softmax of input along the specified dimension.
        Output dtype is float32 if half_to_float=True, otherwise matches input dtype.
    """
    # PyTorch checks: half_to_float conversion is only supported for Half type
    # See: pytorch/aten/src/ATen/native/cuda/SoftMax.cu
    if half_to_float:
        if input.dtype != DType.float16:
            raise ValueError(
                f"half_to_float conversion is supported for Half (float16) type only, got {input.dtype}"
            )

    # Store original dtype for potential conversion back
    original_dtype = input.dtype

    # Convert to float32 for computation if input is float16
    # PyTorch's type promotion automatically uses float32 for float16 computation
    if input.dtype == DType.float16:
        input = F.cast(input, dtype=DType.float32)

    # Handle negative dim
    if dim < 0:
        dim = len(input.shape) + dim

    # Compute max along the specified axis for numerical stability, keeping dimensions
    x_max = aten_amax(input, dim=[dim], keepdim=True)

    # Subtract max for numerical stability: x - max(x)
    x_shifted = input - x_max

    # Compute exp(x - max(x))
    x_exp = F.exp(x_shifted)

    # Sum exponentials along the axis, keeping dimensions for broadcasting
    x_sum = aten_sum(x_exp, dim=[dim], keepdim=True)

    # Compute log(sum(exp(x - max(x))))
    log_sum = F.log(x_sum)

    # Compute result: (x - max(x)) - log(sum(exp(x - max(x))))
    result = x_shifted - log_sum

    # Convert back to original dtype if half_to_float=False and we converted from float16
    # When half_to_float=True with float16 input, output stays in float32
    if not half_to_float and original_dtype == DType.float16:
        result = F.cast(result, dtype=original_dtype)

    return result


# aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
@map_to(aten._log_softmax_backward_data.default)
def aten__log_softmax_backward_data(
    grad_output: MaxTensor, output: MaxTensor, dim: int, input_dtype: torch.dtype
) -> MaxTensor:
    """Differentiate log-softmax using its saved output.

    PyTorch's generated ``LogSoftmaxBackward0`` node calls this ATen op.  Keep
    the implementation as ordinary MAX operations so it works for symbolic
    graph values and eager tensors alike.
    """
    if grad_output.dtype != output.dtype:
        raise ValueError(
            "grad_output and output must have the same dtype for "
            "_log_softmax_backward_data"
        )
    if tuple(grad_output.shape) != tuple(output.shape):
        raise ValueError(
            "grad_output and output must have the same shape for "
            "_log_softmax_backward_data"
        )

    target_dtype = torch_dtype_to_max(input_dtype)
    if grad_output.dtype != target_dtype and not (
        grad_output.dtype == DType.float32 and target_dtype == DType.float16
    ):
        raise ValueError(
            "input and grad dtypes must match, or input must be float16 and "
            "grad must be float32"
        )

    rank = len(grad_output.shape)
    if rank == 0:
        if dim not in (-1, 0):
            raise ValueError("dim must be -1 or 0 for a scalar input")
        summed = grad_output
    else:
        if not -rank <= dim < rank:
            raise ValueError("dim must refer to an input dimension")
        dim %= rank
        summed = aten_sum(grad_output, dim=[dim], keepdim=True)
    result = grad_output - F.exp(output) * summed

    if result.dtype != target_dtype:
        result = F.cast(result, dtype=target_dtype)
    return result


# _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten._to_copy)
def aten__to_copy(
    tensor: MaxTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
):
    result = tensor
    if device is not None:
        result = _transfer_to(result, torch_device_to_max_device(device))
    if dtype is not None:
        result = F.cast(result, dtype=torch_dtype_to_max(dtype))
    return result


# abs(Tensor self) -> Tensor
@map_to(aten.abs)
def aten_abs(x: MaxTensor):
    return F.abs(x)


# acos(Tensor self) -> Tensor
@map_to(aten.acos)
def aten_acos(x: MaxTensor) -> MaxTensor:
    """Computes the arccosine (inverse cosine) of the input tensor.

    Returns values in the range [0, π] for inputs in [-1, 1].
    Uses polynomial approximation based on the Mojo stdlib implementation.

    Args:
        x: Input tensor with values in [-1, 1]

    Returns:
        Arccosine of the input in radians [0, π]
    """
    # Create constants as tensors for use in F.where()
    zero = F.constant(0.0, dtype=x.dtype, device=x.device)
    one = F.constant(1.0, dtype=x.dtype, device=x.device)
    neg_one = F.constant(-1.0, dtype=x.dtype, device=x.device)

    # Clamp input to valid domain [-1, 1]
    x_clamped = F.max(F.min(x, 1.0), -1.0)
    x_abs = F.abs(x_clamped)

    # Domain split at 0.5
    small_domain = x_abs < 0.5

    # Compute x_squared and d based on domain
    # Small domain: x_squared = x², d = |x|
    # Large domain: x_squared = (1 - |x|) / 2, d = sqrt(x_squared)
    x_squared_small = x_clamped * x_clamped
    x_squared_large = (1.0 - x_abs) * 0.5
    x_squared = F.where(small_domain, x_squared_small, x_squared_large)

    d_small = x_abs
    d_large = F.sqrt(x_squared_large)
    d = F.where(small_domain, d_small, d_large)

    # Handle special case |x| = 1 (d should be 0)
    is_one = x_abs >= 1.0
    d = F.where(is_one, zero, d)

    # Polynomial evaluation using Horner's method
    # Coefficients from Mojo stdlib (Remez approximation)
    poly = 0.4197454825e-1
    poly = poly * x_squared + 0.2424046025e-1
    poly = poly * x_squared + 0.4547423869e-1
    poly = poly * x_squared + 0.7495029271e-1
    poly = poly * x_squared + 0.1666677296
    poly = poly * x_squared * d

    # Small domain: π/2 - (d + poly) with sign preservation
    # copysign(d, x) is implemented as d * sign(x)
    is_negative = x_clamped < 0.0
    sign_x = F.where(is_negative, neg_one, one)
    d_signed = d * sign_x
    poly_signed = poly * sign_x
    result_small = (math.pi * 0.5) - (d_signed + poly_signed)

    # Large domain: 2 * (d + poly)
    result_large = 2.0 * (d + poly)

    # For large domain with negative x: π - result
    result_large = F.where(is_negative, math.pi - result_large, result_large)

    # Select based on domain
    result = F.where(small_domain, result_small, result_large)

    return result


# acosh(Tensor self) -> Tensor
# adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor


# add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
# add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
@map_to(aten.add)
def aten_add(input: MaxTensor, other: MaxTensor | Scalar, alpha: Scalar = 1):
    input, other = type_promotion(input, other)
    if alpha != 1:
        other = aten_mul(other, alpha)
    return input + other


# addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
@map_to(aten.addcdiv)
def aten_addcdiv(
    input: MaxTensor, tensor1: MaxTensor, tensor2: MaxTensor, *, value: Scalar = 1
) -> MaxTensor:
    # addcdiv computes: input + value * tensor1 / tensor2
    div_result = aten_div(tensor1, tensor2)
    if value != 1:
        div_result = aten_mul(div_result, value)
    return aten_add(input, div_result)


# addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
@map_to(aten.addcmul)
def aten_addcmul(
    input: MaxTensor, tensor1: MaxTensor, tensor2: MaxTensor, *, value: Scalar = 1
) -> MaxTensor:
    # addcmul computes: input + value * tensor1 * tensor2
    mul_result = aten_mul(tensor1, tensor2)
    if value != 1:
        mul_result = aten_mul(mul_result, value)
    return aten_add(input, mul_result)


# addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
@map_to(aten.addmm)
def aten_addmm(
    input: MaxTensor,
    mat1: MaxTensor,
    mat2: MaxTensor,
    *,
    beta: Scalar = 1.0,
    alpha: Scalar = 1.0,
) -> MaxTensor:
    # addmm computes: beta * input + alpha * mat1 @ mat2
    matmul_result = operator.matmul(mat1, mat2)

    # Apply scaling factors
    if alpha != 1.0:
        matmul_result = operator.mul(matmul_result, alpha)

    if beta != 1.0:
        scaled_input = operator.mul(input, beta)
    else:
        scaled_input = input

    return operator.add(scaled_input, matmul_result)


# alias(Tensor(a) self) -> Tensor(a)
@map_to(aten.alias)
def aten_alias(input: MaxTensor) -> MaxTensor:
    return input


# amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
@map_to(aten.amax)
def aten_amax(
    input: MaxTensor, dim: list[int] = [], keepdim: bool = False
) -> MaxTensor:
    # If empty dim list is provided, reduce over all dimensions
    if not dim:
        dim = [i for i in range(len(input.shape))]

    # Reduce each dimension one by one, similar to aten_mean
    result = input
    for axis in dim:
        result = _reduce_max(result, axis=axis)

    if not keepdim:
        # Squeeze the reduced dimensions - sort in reverse order to avoid index shifting
        for axis in sorted(dim, reverse=True):
            result = F.squeeze(result, axis=axis)

    return result


# amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
@map_to(aten.amin)
def aten_amin(
    input: MaxTensor, dim: list[int] = [], keepdim: bool = False
) -> MaxTensor:
    # If empty dim list is provided, reduce over all dimensions
    if not dim:
        dim = [i for i in range(len(input.shape))]

    # Reduce each dimension one by one, similar to aten_mean
    result = input
    for axis in dim:
        result = _reduce_min(result, axis=axis)

    if not keepdim:
        # Squeeze the reduced dimensions - sort in reverse order to avoid index shifting
        for axis in sorted(dim, reverse=True):
            result = F.squeeze(result, axis=axis)

    return result


# any(Tensor self) -> Tensor
# any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
# any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
@map_to(aten.any)
def aten_any(
    input: MaxTensor, dim: int | list[int] | None = None, keepdim: bool = False
) -> MaxTensor:
    """
    Equivalent to torch.any.
    Tests if any elements in the input are True (non-zero).
    Uses max() on boolean tensor since True > False.
    """
    # Convert input to boolean first (non-zero values become True)
    if input.dtype == DType.bool:
        input_bool = input
    else:
        input_bool = F.not_equal(input, 0)

    if dim is None:
        # Return True if any element is True (reduce all dimensions)
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    # Handle negative dimensions
    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Remove this cast when https://github.com/modular/modular/issues/6067 is fixed
    result = input_bool.cast(DType.int32)

    # Use max() to implement any() since True > False
    for axis in sorted(dim, reverse=True):
        result = _reduce_max(result, axis=axis)

    # Handle keepdim=False
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = F.squeeze(result, axis=axis)

    # Remove this cast when https://github.com/modular/modular/issues/6067 is fixed
    return result.cast(DType.bool)


# arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.arange)
def aten_arange(
    start: Scalar,
    end: Scalar | None = None,
    step: Scalar = 1,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> MaxTensor:
    if isinstance(start, float):
        raise ValueError("We don't support float start values for torch.arange")
    if isinstance(step, float):
        raise ValueError("We don't support float step values for torch.arange")
    if isinstance(end, float):
        raise ValueError("We don't support float end values for torch.arange")
    if dtype is None:
        dtype = torch.int64
    dtype = torch_dtype_to_max(dtype)

    if device is None:
        device = torch.get_default_device()
    device = torch_device_to_max_device(device)

    if end is None:
        # Single argument form: torch.arange(end)
        end = start
        start = 0

    # Calculate output dimension for F.range
    # The length is ceil((end - start) / step) as per PyTorch docs
    out_dim = end - start
    if step != 1:
        out_dim = int(math.ceil(out_dim / step))

    # Use F.range to create the sequence
    result = F.arange(
        Dim(start),
        Dim(end),
        Dim(step),
        out_dim=Dim(out_dim),
        device=device,
        dtype=dtype,
    )
    # TODO: Remove this when the bug is addressed in MAX, range doesn't produce the correct dtype
    # https://github.com/modular/modular/issues/5178
    return F.cast(result, dtype=dtype)


# argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
@map_to(aten.argmax)
def aten_argmax(
    input: MaxTensor, dim: int | None = None, keepdim: bool = False
) -> MaxTensor:
    # If dim is None, return argmax of flattened tensor
    if dim is None:
        # Flatten the tensor and compute argmax along axis 0
        flattened = F.reshape(input, [-1])
        result = _reduce_argmax(flattened, axis=0)
        if keepdim:
            # Return tensor with same number of dimensions as input, all size 1
            result_shape = [1] * len(input.shape)
            result = F.reshape(result, result_shape)
        else:
            # Return scalar (0-dimensional tensor)
            result = F.squeeze(result, axis=0)
    else:
        # Compute argmax along specified dimension
        # MAX only supports argmax on innermost axis, so we need to transpose
        ndim = len(input.shape)

        # Normalize negative axis
        if dim < 0:
            dim = ndim + dim

        # If dim is not the last dimension, transpose to make it last
        if dim != ndim - 1:
            # Swap target dimension with last dimension
            transposed_input = F.transpose(input, dim, ndim - 1)

            # Perform argmax on last axis
            result = _reduce_argmax(transposed_input, axis=-1)

            # Swap back if needed
            if not keepdim:
                # The result has one fewer dimension, so we need to be careful about indexing
                result_ndim = ndim - 1
                if dim < result_ndim:
                    # Swap back: what was at position dim is now at position (result_ndim - 1)
                    # We want to move it back to position dim
                    result = F.transpose(result, dim, result_ndim - 1)
            else:
                # For keepdim=True, the result still has the same number of dimensions
                # Swap the dimensions back
                result = F.transpose(result, dim, ndim - 1)
        else:
            # Target axis is already the last dimension
            result = _reduce_argmax(input, axis=dim)

        if not keepdim:
            # Find the dimension with size 1 and squeeze it
            for i, size in enumerate(result.shape):
                if size == 1:
                    result = F.squeeze(result, axis=i)
                    break
    return result


# argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
@map_to(aten.argmin)
def aten_argmin(
    input: MaxTensor, dim: int | None = None, keepdim: bool = False
) -> MaxTensor:
    # If dim is None, return argmin of flattened tensor
    if dim is None:
        # Flatten the tensor and compute argmin along axis 0
        flattened = F.reshape(input, [-1])
        result = _reduce_argmin(flattened, axis=0)
        if keepdim:
            # Return tensor with same number of dimensions as input, all size 1
            result_shape = [1] * len(input.shape)
            result = F.reshape(result, result_shape)
        else:
            # Return scalar (0-dimensional tensor)
            result = F.squeeze(result, axis=0)
    else:
        # Compute argmin along specified dimension
        # MAX only supports argmin on innermost axis, so we need to transpose
        ndim = len(input.shape)

        # Normalize negative axis
        if dim < 0:
            dim = ndim + dim

        # If dim is not the last dimension, transpose to make it last
        if dim != ndim - 1:
            # Swap target dimension with last dimension
            transposed_input = F.transpose(input, dim, ndim - 1)

            # Perform argmin on last axis
            result = _reduce_argmin(transposed_input, axis=-1)

            # Swap back if needed
            if not keepdim:
                # The result has one fewer dimension, so we need to be careful about indexing
                result_ndim = ndim - 1
                if dim < result_ndim:
                    # Swap back: what was at position dim is now at position (result_ndim - 1)
                    # We want to move it back to position dim
                    result = F.transpose(result, dim, result_ndim - 1)
            else:
                # For keepdim=True, the result still has the same number of dimensions
                # Swap the dimensions back
                result = F.transpose(result, dim, ndim - 1)
        else:
            # Target axis is already the last dimension
            result = _reduce_argmin(input, axis=dim)

        if not keepdim:
            # Find the dimension with size 1 and squeeze it
            for i, size in enumerate(result.shape):
                if size == 1:
                    result = F.squeeze(result, axis=i)
                    break
    return result


# as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
# asin(Tensor self) -> Tensor


# asinh(Tensor self) -> Tensor
@map_to(aten.asinh)
def aten_asinh(x: MaxTensor) -> MaxTensor:
    """Computes inverse hyperbolic sine using asinh(x) = log(x + sqrt(x² + 1))"""
    return F.log(x + F.sqrt(x * x + 1))


# atan(Tensor self) -> Tensor
# atan2(Tensor self, Tensor other) -> Tensor
# atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)


# atanh(Tensor self) -> Tensor
@map_to(aten.atanh)
def aten_atanh(x: MaxTensor) -> MaxTensor:
    return F.atanh(x)


# avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor


# avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
@map_to(aten.avg_pool2d)
def aten_avg_pool2d(
    input: MaxTensor,
    kernel_size: list[int],
    stride: list[int] | None = None,
    padding: list[int] = [0, 0],
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    """
    Applies a 2D average pooling over an input signal composed of several input planes.

    Args:
        input: input tensor (N, C, H_in, W_in)
        kernel_size: size of the pooling window
        stride: stride of the pooling window. Default value is kernel_size
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use ceil instead of floor to compute output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used
    """
    if divisor_override is not None:
        raise NotImplementedError("divisor_override is not supported yet in avg_pool2d")

    # Handle default stride
    if stride is None:
        stride = kernel_size

    # Ensure kernel_size, stride, and padding are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, list):
        stride = tuple(stride)

    if isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, list):
        padding = tuple(padding)

    # Convert padding from PyTorch format (pad_h, pad_w) to MAX format (pad_h_before, pad_h_after, pad_w_before, pad_w_after)
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    input_nhwc = input.permute([0, 2, 3, 1])

    # Apply average pooling using MAX
    result = F.avg_pool2d(
        input_nhwc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_boundary=count_include_pad,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    return result.permute([0, 3, 1, 2])


# avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
# avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor


# bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
@map_to(aten.bitwise_and.Scalar)
def aten_bitwise_and_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    return custom_mojo_ops.bitwise_and_scalar(input, other)


# bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.bitwise_and.Tensor)
def aten_bitwise_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    # For the moment we only support tensors of the same dimension

    final_shape = find_broadcast_shape(input.shape, other.shape)
    input = F.broadcast_to(input, final_shape)
    other = F.broadcast_to(other, final_shape)

    return custom_mojo_ops.bitwise_and(input, other)


# bitwise_not(Tensor self) -> Tensor
@map_to(aten.bitwise_not)
def aten_bitwise_not(input: MaxTensor) -> MaxTensor:
    return custom_mojo_ops.bitwise_not(input)


# bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
@map_to(aten.bitwise_or.Scalar)
def aten_bitwise_or_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    return custom_mojo_ops.bitwise_or_scalar(input, other)


# bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.bitwise_or.Tensor)
def aten_bitwise_or(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    # For the moment we only support tensors of the same dimension

    final_shape = find_broadcast_shape(input.shape, other.shape)
    input = F.broadcast_to(input, final_shape)
    other = F.broadcast_to(other, final_shape)

    return custom_mojo_ops.bitwise_or(input, other)


# bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
@map_to(aten.bitwise_xor.Scalar)
def aten_bitwise_xor_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    return custom_mojo_ops.bitwise_xor_scalar(input, other)


# bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.bitwise_xor.Tensor)
def aten_bitwise_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    # For the moment we only support tensors of the same dimension

    final_shape = find_broadcast_shape(input.shape, other.shape)
    input = F.broadcast_to(input, final_shape)
    other = F.broadcast_to(other, final_shape)

    return custom_mojo_ops.bitwise_xor(input, other)


# bmm(Tensor self, Tensor mat2) -> Tensor
@map_to(aten.bmm)
def aten_bmm(input: MaxTensor, mat2: MaxTensor) -> MaxTensor:
    """
    Batch matrix multiplication equivalent to torch.bmm.

    Args:
        input: 3D tensor of shape [batch_size, n, m]
        mat2: 3D tensor of shape [batch_size, m, p]

    Returns:
        3D tensor of shape [batch_size, n, p]
    """
    # MAX's matmul handles batch dimensions automatically through broadcasting
    return F.matmul(input, mat2)


# aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
# aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
@map_to(aten.bucketize)
def aten_bucketize(
    input: MaxTensor | Scalar,
    boundaries: MaxTensor,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> MaxTensor:
    if len(boundaries.shape) != 1:
        raise RuntimeError(
            "boundaries tensor must be 1 dimension, but got "
            f"dim({len(boundaries.shape)})"
        )
    return _searchsorted_impl(
        boundaries, input, out_int32=out_int32, right=right, side=None, sorter=None
    )


# cat(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.cat)
def aten_cat(tensors: list[MaxTensor], dim: int = 0) -> MaxTensor:
    # PyTorch's cat skips the legacy "empty" tensor (1-D with static size 0),
    # which doesn't need to match the rank of the other inputs. This shows up
    # e.g. as uninitialized KV-caches in transformers. MAX's concat requires all
    # inputs to share the same rank, so drop those sentinels to match PyTorch.
    def _is_legacy_empty(t: MaxTensor) -> bool:
        return (
            len(t.shape) == 1
            and isinstance(t.shape[0], StaticDim)
            and int(t.shape[0]) == 0
        )

    filtered = [t for t in tensors if not _is_legacy_empty(t)]
    if not filtered:
        # Everything was a legacy-empty tensor; return one as-is (shape [0]).
        return tensors[0]
    return F.concat(filtered, axis=dim)


# ceil(Tensor self) -> Tensor
@map_to(aten.ceil)
def aten_ceil(input: MaxTensor) -> MaxTensor:
    """
    Ceiling of the input tensor, element-wise.

    For floating-point inputs: Uses MAX's native ceil op.
    For integer inputs: Returns it (no mathematical change needed, following PyTorch behavior).
    """
    if input.type.dtype.is_integral():
        return input
    else:
        return F.ceil(input)


# clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
# clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
@map_to(aten.clamp)
def aten_clamp(
    input: MaxTensor,
    min: MaxTensor | Scalar | None = None,
    max: MaxTensor | Scalar | None = None,
) -> MaxTensor:
    """
    Implements torch.clamp by clamping all elements in input to the range [min, max].
    Uses F.max and F.min to implement clamp as:
    clamp(x, min, max) = min(max(x, min), max)
    """
    result = input

    # Apply lower bound if min is provided
    if min is not None:
        result = F.max(result, min)

    # Apply upper bound if max is provided
    if max is not None:
        result = F.min(result, max)

    return result


# clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.clone)
def aten_clone(
    input: MaxTensor, *, memory_format: torch.memory_format | None = None
) -> MaxTensor:
    return input


# col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
def _torch_pad_to_max_paddings(rank: int, pad: list[SymIntType]) -> list[SymIntType]:
    """Torch's pad list to MAX's `ops.pad` paddings list.

    Torch's pad list covers the trailing `len(pad) // 2` dims, LAST dim
    first: `[last_before, last_after, second_to_last_before, ...]`. MAX
    wants every dim, in forward order: `[before_dim0, after_dim0,
    before_dim1, after_dim1, ...]`. Shared by constant_pad_nd,
    reflection_pad2d and replication_pad2d, whose pad/padding arguments all
    use this same torch convention.
    """
    paddings: list[SymIntType] = [0] * (2 * rank)
    for i in range(len(pad) // 2):
        dim = rank - 1 - i
        paddings[2 * dim] = pad[2 * i]
        paddings[2 * dim + 1] = pad[2 * i + 1]
    return paddings


# constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
@map_to(aten.constant_pad_nd)
def aten_constant_pad_nd(
    input: MaxTensor, pad: list[int | Dim], value: Scalar = 0
) -> MaxTensor:
    rank = len(input.shape)
    if any(isinstance(p, int) and p < 0 for p in pad):
        # A negative entry crops rather than pads. Mirrors ATen's own
        # constant_pad_nd (PadNd.cpp): narrow the input away from the
        # negative amount first, then pad only what remains non-negative --
        # MAX's ops.pad rejects negative paddings outright.
        slices = [slice(None)] * rank
        cropped_pad = list(pad)
        for i in range(len(pad) // 2):
            dim = rank - 1 - i
            before, after = pad[2 * i], pad[2 * i + 1]
            if isinstance(before, int) and before < 0:
                slices[dim] = slice(-before, slices[dim].stop)
                cropped_pad[2 * i] = 0
            if isinstance(after, int) and after < 0:
                slices[dim] = slice(slices[dim].start, after)
                cropped_pad[2 * i + 1] = 0
        input = input[*slices]
        pad = cropped_pad
    paddings = _torch_pad_to_max_paddings(rank, pad)
    return max_ops.pad(input, paddings, mode="constant", value=value)


def _add_bias_transposed(
    transposed: bool, result: MaxTensor, bias: MaxTensor | None, shape_suffix: list[int]
) -> MaxTensor:
    if transposed and bias is not None:
        bias_reshaped = F.reshape(bias, (1, bias.shape[0], *shape_suffix))
        result = result + bias_reshaped
    return result


def _normalize_2d_params(stride, padding, dilation, output_padding):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, tuple | list) and len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, tuple | list) and len(padding) == 4:
        padding = tuple(padding)
    elif isinstance(padding, str):
        raise ValueError("Padding must be an int or a tuple of ints.")
    else:
        raise ValueError(f"Unsupported padding length: {len(padding)}")
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    elif isinstance(output_padding, tuple | list) and len(output_padding) == 2:
        output_padding = tuple(output_padding)
    else:
        output_padding = (0, 0)
    return stride, padding, dilation, output_padding


# convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
@map_to(aten.convolution)
def aten_convolution(
    input: MaxTensor,
    weight: MaxTensor,
    bias: MaxTensor | None,
    stride: list[SymIntType],
    padding: list[SymIntType],
    dilation: list[SymIntType],
    transposed: bool,
    output_padding: list[SymIntType],
    groups: SymIntType,
) -> MaxTensor:
    groups = int(groups)
    if groups != 1 and transposed:
        # MAX's conv2d_transpose has no native `groups` support (only conv2d /
        # conv3d do), so emulate a grouped transposed conv by splitting the
        # input channels and the weights into `groups` independent groups,
        # running a regular (groups=1) transposed conv on each, and
        # concatenating the results along the channel dimension.
        # A transposed conv weight has shape (C_in, C_out/groups, *K), so it is
        # split along dim 0 (input channels), same as the input.
        in_channels = int(input.shape[1])
        if in_channels % groups != 0:
            raise ValueError(
                f"Input channels ({in_channels}) must be divisible by groups ({groups})."
            )
        weight_in = int(weight.shape[0])
        if weight_in % groups != 0:
            raise ValueError(
                f"Weight dim 0 ({weight_in}) must be divisible by groups ({groups})."
            )
        input_groups = F.split(input, [in_channels // groups] * groups, axis=1)
        weight_groups = F.split(weight, [weight_in // groups] * groups, axis=0)
        partial_results = [
            aten_convolution(
                input_group,
                weight_group,
                None,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                1,
            )
            for input_group, weight_group in zip(input_groups, weight_groups)
        ]
        result = F.concat(partial_results, axis=1)
        if bias is not None:
            shape_suffix = [1] * (len(result.shape) - 2)
            bias_reshaped = F.reshape(bias, (1, bias.shape[0], *shape_suffix))
            result = result + bias_reshaped
        return result

    input_rank = len(input.shape)

    if input_rank == 3:
        if dilation[0] != 1:
            raise NotImplementedError(
                "Non-unit dilation is not supported for conv1d yet."
            )

        stride_2d = (stride[0], 1)
        dilation_2d = (dilation[0], 1)
        padding_2d = (padding[0], padding[0], 0, 0)
        output_padding_2d = (output_padding[0] if output_padding else 0, 0)

        input_2d = F.unsqueeze(input, axis=-1)
        input_nlwc = input_2d.permute([0, 2, 3, 1])

        if transposed:
            weight_2d = F.unsqueeze(weight, axis=2)
            weight_k1oi = weight_2d.permute([3, 2, 1, 0])
            result = F.conv2d_transpose(
                input_nlwc,
                weight_k1oi,
                bias=None,
                stride=stride_2d,
                padding=padding_2d,
                dilation=dilation_2d,
                output_paddings=output_padding_2d,
                input_layout=max_type.ConvInputLayout.NHWC,
                filter_layout=max_type.FilterLayout.RSCF,
            )
        else:
            weight_2d = F.unsqueeze(weight, axis=2)
            weight_k1io = weight_2d.permute([3, 2, 1, 0])
            result = F.conv2d(
                input_nlwc,
                weight_k1io,
                bias=bias,
                stride=stride_2d,
                padding=padding_2d,
                dilation=dilation_2d,
                groups=groups,
                input_layout=max_type.ConvInputLayout.NHWC,
                filter_layout=max_type.FilterLayout.RSCF,
            )

        result_nclw = result.permute([0, 3, 1, 2])
        result_ncl = F.squeeze(result_nclw, axis=-1)
        return _add_bias_transposed(transposed, result_ncl, bias, [1])

    elif input_rank == 4:
        stride, padding, dilation, output_padding = _normalize_2d_params(
            stride, padding, dilation, output_padding
        )
        input_nhwc = input.permute([0, 2, 3, 1])
        weight_rscf = weight.permute([2, 3, 1, 0])

        if transposed:
            result = F.conv2d_transpose(
                input_nhwc,
                weight_rscf,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_paddings=output_padding,
                input_layout=max_type.ConvInputLayout.NHWC,
                filter_layout=max_type.FilterLayout.RSCF,
            )
        else:
            result = F.conv2d(
                input_nhwc,
                weight_rscf,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                input_layout=max_type.ConvInputLayout.NHWC,
                filter_layout=max_type.FilterLayout.RSCF,
            )

        result_nchw = result.permute([0, 3, 1, 2])
        return _add_bias_transposed(transposed, result_nchw, bias, [1, 1])

    else:
        raise ValueError(
            f"Unsupported input rank for convolution: {input_rank}. Expected 3 (1D) or 4 (2D)."
        )


# convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
# copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
@map_to(aten.copy)
def aten_copy(
    input: MaxTensor, src: MaxTensor, non_blocking: bool = False
) -> MaxTensor:
    return src


# cos(Tensor self) -> Tensor
@map_to(aten.cos)
def aten_cos(x: MaxTensor) -> MaxTensor:
    return F.cos(x)


# cosh(Tensor self) -> Tensor
@map_to(aten.cosh)
def aten_cosh(x: MaxTensor) -> MaxTensor:
    """Computes hyperbolic cosine using cosh(x) = (exp(x) + exp(-x)) / 2"""
    return (F.exp(x) + F.exp(-x)) / 2


# cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
@map_to(aten.cumsum)
def aten_cumsum(
    input: MaxTensor, dim: int, *, dtype: torch.dtype | None = None
) -> MaxTensor:
    """
    Returns the cumulative sum of elements of input in the dimension dim.

    Args:
        input: the input tensor
        dim: the dimension to do the operation over
        dtype: the desired data type of returned tensor
    """
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        input = F.cast(input, dtype=max_dtype)

    # MAX's cumsum handles negative dimensions automatically, so no need to convert
    return F.cumsum(input, axis=dim)


# TODO: handle inplace?
# detach(Tensor(a) self) -> Tensor(a)
@map_to(aten.detach)
def aten_detach(input: MaxTensor) -> MaxTensor:
    return input


# diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)


# div.Scalar(Tensor self, Scalar other) -> Tensor
# div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
# div.Tensor(Tensor self, Tensor other) -> Tensor
# div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
@map_to(aten.div)
def aten_div(
    input: MaxTensor, other: MaxTensor | Scalar, *, rounding_mode: str | None = None
) -> MaxTensor:
    # Handle torch.div with different rounding modes
    if rounding_mode is None:
        return operator.truediv(input, other)
    elif rounding_mode == "floor":
        return operator.floordiv(input, other)
    elif rounding_mode == "trunc":
        # Truncation towards zero (not implemented in operator, need custom logic)
        result = operator.truediv(input, other)
        return F.trunc(result)
    else:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")


# elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor


# embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
@map_to(aten.embedding)
def aten_embedding(
    input: MaxTensor,
    weight: MaxTensor,
    padding_idx: SymIntType = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    # For some reason with aten, input and weight are inverted.
    return torch_embedding_equivalent(
        weight,
        input,
        padding_idx=padding_idx,
        max_norm=None,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )


def torch_embedding_equivalent(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    scale_grad_by_freq=False,
    sparse=False,
):
    if max_norm is not None:
        raise NotImplementedError(
            "max_norm is not supported yet in this embedding implementation"
        )
    if scale_grad_by_freq:
        raise NotImplementedError(
            "scale_grad_by_freq is not supported yet in this embedding implementation"
        )
    if sparse:
        raise NotImplementedError(
            "sparse gradients are not supported yet in this embedding implementation"
        )

    # Handle scalar indices by reshaping to have at least one dimension
    # PyTorch embedding returns the selected row directly for scalar input
    # but MAX gather may need proper shape handling
    original_shape = input.shape
    if len(original_shape) == 0:  # Scalar tensor
        input_reshaped = F.unsqueeze(input, axis=0)
        result = F.gather(weight, input_reshaped, axis=0)
        # Remove the added dimension: [1, embedding_dim] -> [embedding_dim]
        return F.squeeze(result, axis=0)
    else:
        # Use gather to select rows from weight matrix based on input indices
        # axis=0 means we're gathering along the first dimension (vocab dimension)
        return F.gather(weight, input, axis=0)


# embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor


# empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.empty.memory_format)
def aten_empty_memory_format(
    size: list[SymIntType],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> MaxTensor:
    return aten_full(
        size, 0, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.empty_strided)
def aten_empty_strided(
    size: list[SymIntType],
    stride: list[SymIntType],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> MaxTensor:
    return aten_full(
        size, 0, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.empty_like)
def aten_empty_like(
    self: MaxTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> MaxTensor:
    return aten_full_like(
        self, 0, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.empty_permuted)
def aten_empty_permuted(
    size: list[SymIntType],
    physical_layout: list[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> MaxTensor:
    return aten_full(
        size, 0, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# eq.Scalar(Tensor self, Scalar other) -> Tensor
# eq.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.eq)
def aten_eq(x: MaxTensor, y: MaxTensor | Scalar) -> MaxTensor:
    return operator.eq(x, y)


# erf(Tensor self) -> Tensor
@map_to(aten.erf)
def aten_erf(input: MaxTensor) -> MaxTensor:
    return F.erf(input)


# exp(Tensor self) -> Tensor
@map_to(aten.exp)
def aten_exp(input: MaxTensor) -> MaxTensor:
    return F.exp(input)


# expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
@map_to(aten.expand)
def aten_expand(
    tensor: MaxTensor, size: list[SymIntType], *, implicit: bool = False
) -> MaxTensor:
    target_shape = []

    # Get current tensor shape - we need this to handle -1 values
    current_shape = tensor.shape

    # Pad the current shape with 1s if target has more dimensions
    if len(size) > len(current_shape):
        padded_current_shape = [1] * (len(size) - len(current_shape)) + list(
            current_shape
        )
    else:
        padded_current_shape = list(current_shape)

    # Process each dimension in the target size
    for i, dim_size in enumerate(size):
        if dim_size == -1:
            # Keep current dimension size
            if i < len(padded_current_shape):
                target_shape.append(padded_current_shape[i])
            else:
                # This shouldn't happen in well-formed expand calls
                target_shape.append(1)
        else:
            target_shape.append(dim_size)

    return F.broadcast_to(tensor, target_shape)


# expm1(Tensor self) -> Tensor
# fill.Scalar(Tensor self, Scalar value) -> Tensor
@map_to(aten.fill)
def aten_fill_scalar(input: MaxTensor, value: Scalar) -> MaxTensor:
    """
    Returns a tensor filled with the scalar value, with the same shape as the input tensor.
    This creates a new tensor (functional version, not in-place).
    """
    # Use the input tensor's dtype and device
    target_dtype = input.dtype
    target_device = input.device
    target_shape = input.shape

    # Create a scalar constant with the fill value
    scalar = F.constant(value, dtype=target_dtype, device=target_device)

    # Broadcast the scalar to the target shape
    return F.broadcast_to(scalar, target_shape)


# fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
@map_to(aten.fill_)
def aten_fill__scalar(input: MaxTensor, value: Scalar) -> MaxTensor:
    return aten_fill_scalar(input, value)


# flip(Tensor self, int[] dims) -> Tensor


# floor(Tensor self) -> Tensor
@map_to(aten.floor)
def aten_floor(input: MaxTensor) -> MaxTensor:
    """
    Returns a new tensor with the floor of the elements of input,
    the largest integer less than or equal to each element.
    """
    return F.floor(input)


# fmod.Scalar(Tensor self, Scalar other) -> Tensor
# fmod.Tensor(Tensor self, Tensor other) -> Tensor


# full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.full)
def aten_full(
    size: list[SymIntType],
    fill_value: Scalar,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
):
    if dtype is None:
        dtype = torch.float32
    dtype = torch_dtype_to_max(dtype)

    if device is None:
        device = torch.get_default_device()
    device = torch_device_to_max_device(device)

    # Create a scalar constant with the fill value
    scalar = F.constant(fill_value, dtype=dtype, device=device)
    return F.broadcast_to(scalar, size)


# full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.full_like)
def aten_full_like(
    input: MaxTensor,
    fill_value: Scalar,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> MaxTensor:
    # If dtype is not specified, use the input tensor's dtype
    if dtype is None:
        target_dtype = input.dtype
    else:
        target_dtype = torch_dtype_to_max(dtype)

    # If device is not specified, use the input tensor's device
    if device is None:
        target_device = input.device
    else:
        target_device = torch_device_to_max_device(device)

    # Get the shape from the input tensor
    target_shape = input.shape

    # Create a scalar constant with the fill value
    scalar = F.constant(fill_value, dtype=target_dtype, device=target_device)
    return F.broadcast_to(scalar, target_shape)


# gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor


# ge.Scalar(Tensor self, Scalar other) -> Tensor
# ge.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.ge)
def aten_ge(input: MaxTensor, other: MaxTensor | Scalar) -> MaxTensor:
    return input >= other


# gelu(Tensor self, *, str approximate='none') -> Tensor
@map_to(aten.gelu)
def aten_gelu(
    input: MaxTensor, approximate: Literal["tanh", "none"] = "none"
) -> MaxTensor:
    return F.gelu(input, approximate=approximate)


# gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor
@map_to(aten.gelu_backward)
def aten_gelu_backward(
    grad_output: MaxTensor,
    input: MaxTensor,
    *,
    approximate: Literal["tanh", "none"] = "none",
) -> MaxTensor:
    """
    Computes the gradient of the GELU activation function.

    Based on PyTorch C++ implementation:
    - CUDA: pytorch/aten/src/ATen/native/cuda/ActivationGeluKernel.cu (lines 44-86)
    - CPU: pytorch/aten/src/ATen/native/cpu/Activation.cpp (lines 345-521)

    Args:
        grad_output: Gradient flowing back from the next layer (∂L/∂y)
        input: Original input to the forward GELU operation (x)
        approximate: "none" for exact erf-based GELU, "tanh" for tanh approximation

    Returns:
        Gradient with respect to input (∂L/∂x)
    """
    if approximate not in ("none", "tanh"):
        raise ValueError(
            f"Invalid approximate argument '{approximate}'. "
            f"Supported values are:\n"
            f"  - 'none': Exact GELU using error function (erf)\n"
            f"  - 'tanh': Tanh approximation of GELU (faster, slightly less accurate)\n"
            f"See PyTorch documentation for details on approximation trade-offs."
        )

    return custom_mojo_ops.gelu_backward(grad_output, input, approximate=approximate)


# grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor


# gt.Scalar(Tensor self, Scalar other) -> Tensor
# gt.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.gt)
def aten_gt(x: MaxTensor, y: Scalar | MaxTensor) -> MaxTensor:
    return operator.gt(x, y)


# hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor


# index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
@map_to(aten.index)
def aten_index(input: MaxTensor, indices: list[MaxTensor | None]) -> MaxTensor:
    if not indices:
        raise NotImplementedError("We don't yet support aten.index without indices")

    indices = indices + [None] * (len(input.shape) - len(indices))

    result = input

    # Step 1 — group consecutive index tensors into blocks
    i = 0
    while i < len(indices):
        if indices[i] is None:
            i += 1
            continue

        # Found start of an advanced indexing block
        start = i
        while i < len(indices) and indices[i] is not None:
            i += 1
        end = i

        block_tensors = indices[start:end]

        if end - start == 1:
            # Single-axis indexing — use gather
            idx = block_tensors[0]
            result = F.gather(result, idx, axis=start)
        else:
            # Multi-axis indexing — use gather_nd
            # First broadcast indices to same shape
            final_shape = broadcast_shape([t.shape for t in block_tensors])

            b_indices = [F.broadcast_to(t, final_shape) for t in block_tensors]

            # Stack into shape [..., num_axes]
            stacked = F.stack(b_indices, axis=-1)

            # We still have to broadcast them so that they match the starting dimensions
            for j in range(start - 1, -1, -1):
                stacked = F.broadcast_to(
                    stacked[None, ...], [input.shape[j]] + list(stacked.shape)
                )

            # batch_dims = start
            result = F.gather_nd(result, stacked, batch_dims=start)

    return result


def broadcast_shape(shapes):
    # Normalize: extract raw tuples/lists of dims
    norm_shapes = []
    for s in shapes:
        if hasattr(s, "shape"):
            s = s.shape
        # convert Shape-like to list if needed
        norm_shapes.append(list(s))

    if not norm_shapes:
        return []

    # Determine max rank and left-pad with 1s
    max_rank = max(len(s) for s in norm_shapes)
    padded = []
    for s in norm_shapes:
        pad = [1] * (max_rank - len(s))
        padded.append(pad + list(s))

    # Helper: recognize "dimension == 1"
    def is_one(d):
        # Covers ints == 1 and Dim-like objects that compare equal to 1
        return d == 1

    # Walk from left to right over aligned dims (already padded)
    out = []
    for col in zip(*padded):
        # Keep only the non-1 candidates
        non_ones = [d for d in col if not is_one(d)]
        if not non_ones:
            out.append(1)
            continue
        # All non-1 must be equal
        first = non_ones[0]
        if any(d != first for d in non_ones[1:]):
            raise ValueError(f"Shapes are not broadcastable at a dimension: {col}")
        out.append(first)

    return out


# index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
# index_select(Tensor self, int dim, Tensor index) -> Tensor
# isinf(Tensor self) -> Tensor


# isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
@map_to(aten.isin)
def aten_isin(
    elements: MaxTensor,
    test_elements: MaxTensor,
    assume_unique: bool = False,
    invert: bool = False,
) -> MaxTensor:
    """
    Tests whether each element of elements tensor is in test_elements tensor.

    For each element in `elements`, returns True if that element is in
    `test_elements`, False otherwise. The result has the same shape as `elements`.

    Args:
        elements: Input tensor to check elements from
        test_elements: Tensor of values to test against (typically 1D)
        assume_unique: If True, assumes both inputs contain unique elements
            for potential performance optimization (currently unused)
        invert: If True, inverts the result - returns True for elements NOT in test_elements

    Returns:
        Boolean tensor of same shape as elements
    """
    # Special case: if test_elements is empty, return False for all elements
    # if elements is empty, return empty boolean tensor
    # Check by flattening and checking size
    test_flat = F.reshape(test_elements, [-1])
    elements_flat = F.reshape(elements, [-1])
    num_test = test_flat.shape[0]
    num_elements = elements_flat.shape[0]

    if num_elements == 0:
        # Empty elements means empty result
        elements_bool = F.cast(elements_flat, DType.bool)
        return F.reshape(elements_bool, elements.shape)

    if num_test == 0:
        # Empty test_elements means no matches
        # Create all-False tensor by negating an all-True tensor
        # True > False, so comparing any element with itself gives True
        all_true = F.equal(elements_flat, elements_flat)
        result_flat = F.logical_not(all_true)
        result = F.reshape(result_flat, elements.shape)
        if invert:
            result = F.logical_not(result)
        return result

    # Cast test_flat to same dtype as elements_flat for comparison
    # This ensures F.equal compares compatible types
    test_flat = F.cast(test_flat, elements_flat.dtype)

    # Broadcast shapes: (num_elements, 1) vs (1, num_test) -> (num_elements, num_test)
    # This compares every element against every test_element
    num_elements = elements_flat.shape[0]

    # Add broadcasting dimensions
    elements_expanded = F.reshape(elements_flat, [num_elements, 1])
    test_expanded = F.reshape(test_flat, [1, num_test])

    # Compare all pairs: result shape (num_elements, num_test)
    comparisons = F.equal(elements_expanded, test_expanded)

    # Reduce across test_elements dimension using logical_or
    # Any=True means at least one test element matched
    # Need to cast to numeric type first since reduce doesn't support bool
    comparisons_numeric = F.cast(comparisons, DType.int32)
    result_flat_numeric = _reduce_max(comparisons_numeric, axis=-1)
    result_flat = F.cast(result_flat_numeric, DType.bool)

    # Reshape back to original elements shape
    result = F.reshape(result_flat, elements.shape)

    # Handle invert if needed
    if invert:
        result = F.logical_not(result)

    return result


# isnan(Tensor self) -> Tensor
@map_to(aten.isnan)
def aten_isnan(input: MaxTensor) -> MaxTensor:
    """
    Returns a new tensor with boolean elements representing if each element is NaN or not.
    """
    return F.is_nan(input)


# le.Scalar(Tensor self, Scalar other) -> Tensor
# le.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.le)
def aten_le(input: MaxTensor, other: Scalar | MaxTensor) -> MaxTensor:
    return input <= other


# leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
# linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
@map_to(aten.linear)
def aten_linear(
    input: MaxTensor, weight: MaxTensor, bias: MaxTensor | None = None
) -> MaxTensor:
    result = operator.matmul(input, F.transpose(weight, -1, -2))
    if bias is not None:
        result = result + bias
    return result


# linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
@map_to(aten.linear_backward.default)
def aten_linear_backward(
    self: MaxTensor, grad_output: MaxTensor, weight: MaxTensor, output_mask: list[bool]
) -> tuple[MaxTensor | None, MaxTensor | None, MaxTensor | None]:
    """Compute the requested dense linear gradients with dynamic shapes."""
    mask = tuple(output_mask)
    if len(mask) != 3 or any(not isinstance(requested, bool) for requested in mask):
        raise ValueError("linear_backward output_mask must contain three booleans")
    if len(self.shape) < 1 or len(weight.shape) != 2:
        raise ValueError("linear_backward expects rank >= 1 input and rank 2 weight")

    input_features = weight.shape[1]
    output_features = weight.shape[0]
    rows = math.prod(self.shape[:-1])
    input_matrix = F.reshape(self, [rows, input_features])
    grad_matrix = F.reshape(grad_output, [rows, output_features])

    def zeros(shape: list[Dim] | tuple[Dim, ...]) -> MaxTensor:
        zero = F.constant(0, dtype=grad_output.dtype, device=grad_output.device)
        return F.broadcast_to(zero, shape)

    grad_input = None
    if mask[0]:
        # MAX's matmul leaves C uninitialized when K is statically zero.
        grad_input = (
            zeros(self.shape)
            if output_features == 0
            else F.reshape(operator.matmul(grad_matrix, weight), self.shape)
        )

    grad_weight = None
    # Match PyTorch's registered Meta/MPS structure: the two parameter
    # outputs are both defined whenever either one is requested.
    need_parameter_grads = mask[1] or mask[2]
    if need_parameter_grads:
        grad_weight = (
            zeros(weight.shape)
            if rows == 0
            else operator.matmul(F.transpose(grad_matrix, 0, 1), input_matrix)
        )

    grad_bias = None
    if need_parameter_grads:
        if mask[2]:
            grad_bias = (
                zeros([output_features])
                if rows == 0
                else aten_sum(grad_matrix, dim=[0])
            )
        else:
            grad_bias = zeros([output_features])
    return grad_input, grad_weight, grad_bias


# lift_fresh_copy(Tensor self) -> Tensor
@map_to(aten.lift_fresh_copy)
def aten_lift_fresh_copy(input: MaxTensor) -> MaxTensor:
    # Graph constants (dynamo-lifted tensors created inside the traced
    # function) are already fresh values in the MAX graph.
    return input


# log(Tensor self) -> Tensor
@map_to(aten.log)
def aten_log(input: MaxTensor) -> MaxTensor:
    """
    Returns a new tensor with the natural logarithm of the elements of input.
    """
    return F.log(input)


# log10(Tensor self) -> Tensor


# log1p(Tensor self) -> Tensor
@map_to(aten.log1p)
def aten_log1p(input: MaxTensor) -> MaxTensor:
    """
    Returns a new tensor with the natural logarithm of (1 + input).
    This function is more numerically stable than log(1 + input) for small values of input.
    """
    return F.log1p(input)


# log2(Tensor self) -> Tensor


# logical_and(Tensor self, Tensor other) -> Tensor
@map_to(aten.logical_and)
def aten_logical_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Computes element-wise logical AND of two tensors.
    Both inputs are converted to boolean first if they aren't already.
    """
    # Convert both inputs to boolean if they aren't already
    if input.dtype != DType.bool:
        input_bool = F.not_equal(input, 0)
    else:
        input_bool = input

    if other.dtype != DType.bool:
        other_bool = F.not_equal(other, 0)
    else:
        other_bool = other

    # Apply logical and
    return F.logical_and(input_bool, other_bool)


# logical_not(Tensor self) -> Tensor
@map_to(aten.logical_not)
def aten_logical_not(input: MaxTensor) -> MaxTensor:
    """
    PyTorch's logical_not treats any non-zero value as True and returns the logical negation.
    MAX's logical_not requires boolean input, so we need to convert first.
    """
    # Convert input to boolean (non-zero -> True, zero -> False)
    input_bool = F.not_equal(input, 0)
    # Apply logical not
    return F.logical_not(input_bool)


# logical_or(Tensor self, Tensor other) -> Tensor


# logical_xor(Tensor self, Tensor other) -> Tensor
@map_to(aten.logical_xor)
def aten_logical_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Computes element-wise logical XOR of two tensors.
    Both inputs are converted to boolean first if they aren't already.
    """
    # Convert both inputs to boolean if they aren't already
    if input.dtype != DType.bool:
        input_bool = F.not_equal(input, 0)
    else:
        input_bool = input

    if other.dtype != DType.bool:
        other_bool = F.not_equal(other, 0)
    else:
        other_bool = other

    # Apply logical xor
    return F.logical_xor(input_bool, other_bool)


# lt.Scalar(Tensor self, Scalar other) -> Tensor
# lt.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.lt)
def aten_lt(input: MaxTensor, other: Scalar | MaxTensor) -> MaxTensor:
    return input < other


# masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor


# max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
@map_to(aten.max)
def aten_max(
    input: MaxTensor, dim: int | None = None, keepdim: bool = False
) -> MaxTensor | tuple[MaxTensor, MaxTensor]:
    """
    Implements torch.max with dimension-based reduction.
    Returns (values, indices) tuple when dim is specified, single value otherwise.
    """
    if dim is None:
        # Variant 1: torch.max(input) - single maximum value across all elements
        return aten_amax(input, dim=list(range(len(input.shape))), keepdim=False)
    else:
        # Variant 2: torch.max(input, dim) - (values, indices) tuple along dimension
        values = aten_amax(input, dim=[dim], keepdim=keepdim)
        indices = aten_argmax(input, dim=dim, keepdim=keepdim)
        return (values, indices)


# TODO: re-enable notimplementederror
# max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
@map_to(aten.max_pool2d_with_indices)
def aten_max_pool2d_with_indices(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
) -> tuple[MaxTensor, MaxTensor]:
    # the first output is the values, the second output is the indices
    # most of the time people just want the values so we'll implement that
    # for now.
    if not stride:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    input_nhwc = input.permute([0, 2, 3, 1])

    result = F.max_pool2d(
        input_nhwc,
        kernel_size=kernel_size,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        ceil_mode=ceil_mode,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    forward_result = result.permute([0, 3, 1, 2])
    return (
        forward_result,
        # NotImplementedError(
        #    "The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet."
        # ),
        forward_result,  # This is wrong but needed for eager mode
    )


# max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
# max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)


# maximum(Tensor self, Tensor other) -> Tensor
@map_to(aten.maximum)
def aten_maximum(x: MaxTensor, y: MaxTensor) -> MaxTensor:
    return F.max(x, y)


# mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
# mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
@map_to(aten.mean)
def aten_mean(
    input: MaxTensor,
    dim=None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> MaxTensor:
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        input = F.cast(input, dtype=max_dtype)

    result = input

    if dim is None:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Multiple dimensions reduction - reduce each dimension one by one
    # Sort dimensions in descending order to avoid index shifting issues
    for axis in dim:
        result = _reduce_mean(result, axis=axis)

    # Handle keepdim=False - MAX's mean keeps dimensions by default, so we need to squeeze
    if not keepdim:
        # Remove multiple dimensions - need to be careful about index shifting
        # Sort original dimensions and squeeze from highest to lowest
        dims_to_squeeze = sorted(dim, reverse=True)
        for axis in dims_to_squeeze:
            result = F.squeeze(result, axis=axis)

    return result


# mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
@map_to(aten.mean.out)
def aten_mean_out(
    input: MaxTensor,
    dim,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: MaxTensor,
) -> MaxTensor:
    return aten_mean(input, dim=dim, keepdim=keepdim, dtype=dtype)


# min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
@map_to(aten.min)
def aten_min(
    input: MaxTensor, dim: int | None = None, keepdim: bool = False
) -> MaxTensor | tuple[MaxTensor, MaxTensor]:
    """
    Implements torch.min with dimension-based reduction.
    Returns (values, indices) tuple when dim is specified, single value otherwise.
    """
    if dim is None:
        # Variant 1: torch.min(input) - single minimum value across all elements
        return aten_amin(input, dim=list(range(len(input.shape))), keepdim=False)
    else:
        # Variant 2: torch.min(input, dim) - (values, indices) tuple along dimension
        values = aten_amin(input, dim=[dim], keepdim=keepdim)
        indices = aten_argmin(input, dim=dim, keepdim=keepdim)
        return (values, indices)


# minimum(Tensor self, Tensor other) -> Tensor
@map_to(aten.minimum)
def aten_minimum(x: MaxTensor, y: MaxTensor) -> MaxTensor:
    return F.min(x, y)


# mm(Tensor self, Tensor mat2) -> Tensor
@map_to(aten.mm)
def aten_mm(x: MaxTensor, y: MaxTensor) -> MaxTensor:
    return operator.matmul(x, y)


# mul.Scalar(Tensor self, Scalar other) -> Tensor
# mul.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.mul)
def aten_mul(input: MaxTensor, other: MaxTensor | Scalar) -> MaxTensor:
    input, other = type_promotion(input, other)
    if input.dtype == DType.bool and getattr(other, "dtype", None) == DType.bool:
        # MAX's mul doesn't lower for bool; torch defines it as logical AND.
        return F.logical_and(input, other)
    return input * other


# native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten.native_batch_norm)
def aten_native_batch_norm(
    input: MaxTensor,
    weight: MaxTensor | None,
    bias: MaxTensor | None,
    running_mean: MaxTensor | None,
    running_var: MaxTensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[MaxTensor, MaxTensor, MaxTensor]:
    """
    Implements batch normalization for both training and inference.

    In inference mode (training=False) the provided running statistics are used.
    In training mode (training=True) per-batch statistics are computed over every
    dimension except the channel dimension (dim 1), using the biased variance,
    exactly as PyTorch does for the normalization itself.

    Args:
        input: Input tensor of shape (N, C, ...)
        weight: Optional gamma parameter tensor of shape (C,)
        bias: Optional beta parameter tensor of shape (C,)
        running_mean: Running mean of shape (C,) (used when training=False)
        running_var: Running variance of shape (C,) (used when training=False)
        training: Whether to compute statistics from the batch
        momentum: Momentum factor (unused, running stats are not updated here)
        eps: Small value for numerical stability

    Returns:
        Tuple of (normalized_output, save_mean, save_invstd). Matching PyTorch,
        save_mean / save_invstd are the per-channel (C,) batch statistics when
        training=True and empty (0,) tensors when training=False.
    """
    input_shape = input.shape
    num_channels = int(input_shape[1])

    # Broadcast shape for per-channel statistics: (1, C, 1, 1, ...)
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[1] = num_channels

    if training:
        # Reduce over every dimension except the channel dimension, keeping dims
        # so the statistics broadcast back over the input.
        reduce_axes = [i for i in range(len(input_shape)) if i != 1]
        mean = input
        for axis in reduce_axes:
            mean = _reduce_mean(mean, axis=axis)
        centered = input - mean
        var = centered * centered
        for axis in reduce_axes:
            var = _reduce_mean(var, axis=axis)
        invstd = 1.0 / F.sqrt(var + eps)
        normalized = centered * invstd
        # PyTorch returns the per-channel batch mean and inverse-std (shape (C,)).
        save_mean = F.reshape(mean, [num_channels])
        save_invstd = F.reshape(invstd, [num_channels])
    else:
        if running_mean is None or running_var is None:
            raise ValueError(
                "running_mean and running_var are required when training=False "
                "in aten.native_batch_norm"
            )
        running_mean_reshaped = F.reshape(running_mean, broadcast_shape)
        running_var_reshaped = F.reshape(running_var, broadcast_shape)
        normalized = (input - running_mean_reshaped) / F.sqrt(
            running_var_reshaped + eps
        )
        # In inference mode PyTorch returns empty (0,) tensors for the saved stats.
        save_mean = running_mean[0:0]
        save_invstd = running_var[0:0]

    if weight is not None:
        normalized = normalized * F.reshape(weight, broadcast_shape)

    if bias is not None:
        normalized = normalized + F.reshape(bias, broadcast_shape)

    return (normalized, save_mean, save_invstd)


# native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)


# native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten.native_group_norm)
def aten_native_group_norm(
    input: MaxTensor,
    weight: MaxTensor | None,
    bias: MaxTensor | None,
    N: SymIntType,
    C: SymIntType,
    HxW: SymIntType,
    group: int,
    eps: float,
) -> tuple[MaxTensor, NotImplementedError, NotImplementedError]:
    """
    This is the low-level operation that F.group_norm gets compiled to.
    Returns (normalized_output, mean, rstd) tuple but we only return the first element for simplicity.
    """
    # Reshape input from [N*C, HxW] back to [N, C, H, W] format
    # First, calculate H and W from HxW
    HW = int(HxW)
    # For simplicity, assume square spatial dimensions
    H = W = int(HW**0.5)
    if H * W != HW:
        # If not square, try to factor HxW into reasonable H and W
        # For now, use 1D spatial dimension
        H, W = HW, 1

    # Reshape input to [N, C, H, W]
    input_reshaped = F.reshape(input, [int(N), int(C), H, W])

    # Use the regular group_norm implementation
    result = torch_group_norm_equivalent(input_reshaped, group, weight, bias, eps)

    # Return just the normalized output (native_group_norm returns a tuple)
    return (
        result,
        NotImplementedError(
            "The implementation of aten.native_group_norm doesn't support returning mean yet."
        ),
        NotImplementedError(
            "The implementation of aten.native_group_norm doesn't support returning rstd yet."
        ),
    )


def torch_group_norm_equivalent(input, num_groups, weight=None, bias=None, eps=1e-5):
    # input shape: [N, C, H, W]
    N, C, H, W = input.shape

    # Ensure number of channels is divisible by number of groups
    if int(C) % num_groups != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by number of groups ({num_groups})"
        )

    channels_per_group = int(C) // num_groups

    # Reshape input to [N, num_groups, channels_per_group, H, W]
    reshaped = F.reshape(
        input, [int(N), num_groups, channels_per_group, int(H), int(W)]
    )

    # Calculate mean and variance for each group
    # Normalize over dimensions: channels_per_group, H, W (dims 2, 3, 4)
    axis_to_reduce = [2, 3, 4]

    # Calculate mean
    mean = aten_mean(reshaped, dim=axis_to_reduce, keepdim=True)

    # Calculate variance: Var(X) = E[(X - mean)^2]
    centered = reshaped - mean
    variance = aten_mean(centered * centered, dim=axis_to_reduce, keepdim=True)

    # Normalize: (x - mean) / sqrt(variance + eps)
    normalized = centered / F.sqrt(variance + eps)

    # Reshape back to original shape [N, C, H, W]
    normalized = F.reshape(normalized, [int(N), int(C), int(H), int(W)])

    # Apply scale and shift if provided
    if weight is not None:
        # weight shape: [C] - broadcast to [N, C, H, W]
        weight_reshaped = F.reshape(weight, [1, int(C), 1, 1])
        normalized = normalized * weight_reshaped

    if bias is not None:
        # bias shape: [C] - broadcast to [N, C, H, W]
        bias_reshaped = F.reshape(bias, [1, int(C), 1, 1])
        normalized = normalized + bias_reshaped

    return normalized


# native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)


# native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten.native_layer_norm)
def aten_native_layer_norm(
    input: MaxTensor,
    normalized_shape: list[SymIntType],
    weight: MaxTensor | None,
    bias: MaxTensor | None,
    eps: float,
) -> tuple[MaxTensor, MaxTensor | None, MaxTensor | None]:
    # Fused MAX kernel for the common last-dim case. The two aten_mean
    # reductions below lower to a generic reduction kernel that costs
    # ~150us per launch on GPU (7.5ms/step on gpt2 decode, 73% of GPU
    # time). Only usable when the mean/rstd outputs are dead (inference
    # graphs after DCE), since the fused kernel doesn't expose the
    # statistics.
    if (
        len(normalized_shape) == 1
        and weight is not None
        and bias is not None
        and _only_first_output_used()
    ):
        return max_ops.layer_norm(input, weight, bias, eps), None, None

    # Layer norm normalizes over the last len(normalized_shape) dimensions
    axis_to_reduce = list(
        range(len(input.shape) - len(normalized_shape), len(input.shape))
    )

    mean = aten_mean(input, dim=axis_to_reduce, keepdim=True)
    centered = input - mean
    variance = aten_mean(centered * centered, dim=axis_to_reduce, keepdim=True)
    rstd = 1 / F.sqrt(variance + eps)

    normalized = centered * rstd

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias

    return normalized, mean, rstd


# native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)


# normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
@map_to(aten.normal_)
def aten_normal_(
    self: MaxTensor, mean: float = 0.0, std: float = 1.0, generator=None
) -> MaxTensor:
    if generator is not None:
        raise NotImplementedError(
            "aten::normal_ does not support the generator argument"
        )
    return max_gaussian(
        self.shape, mean=mean, std=std, dtype=self.dtype, device=self.device
    )


# ne.Scalar(Tensor self, Scalar other) -> Tensor
# ne.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.ne)
def aten_ne(x: MaxTensor, y: MaxTensor | Scalar) -> MaxTensor:
    return operator.ne(x, y)


# neg(Tensor self) -> Tensor
@map_to(aten.neg)
def aten_neg(x: MaxTensor) -> MaxTensor:
    return operator.neg(x)


# nonzero(Tensor self) -> Tensor
@map_to(aten.nonzero)
def aten_nonzero(input: MaxTensor) -> MaxTensor:
    """
    Returns the indices of the elements that are non-zero.
    Returns a 2D tensor where each row is the indices of a non-zero element.
    """
    return F.nonzero(input)


# ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.ones_like)
def aten_ones_like(
    self: MaxTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> MaxTensor:
    return aten_full_like(
        self, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# permute(Tensor(a) self, int[] dims) -> Tensor(a)
@map_to(aten.permute)
def aten_permute(x: MaxTensor, dims: list[int]) -> MaxTensor:
    return F.permute(x, dims)


# pow.Scalar(Scalar self, Tensor exponent) -> Tensor
# pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
# pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
@map_to(aten.pow)
def aten_pow(x: Scalar | MaxTensor, y: Scalar | MaxTensor) -> MaxTensor:
    return operator.pow(x, y)


# prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
# prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
# rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# reciprocal(Tensor self) -> Tensor
@map_to(aten.reciprocal)
def aten_reciprocal(tensor: MaxTensor) -> MaxTensor:
    return 1.0 / tensor


# reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
# reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
@map_to(aten.reflection_pad2d)
def aten_reflection_pad2d(input: MaxTensor, padding: list[SymIntType]) -> MaxTensor:
    pad_l, pad_r, pad_t, pad_b = padding
    input_h, input_w = input.shape[-2], input.shape[-1]
    # Match ATen's own validation (ReflectionPad.cpp): a reflected index
    # needs a pivot strictly inside the dimension. Only checked when both
    # the padding and the dimension are concrete -- a dynamic dim trusts the
    # graph, same as everywhere else in this file.
    if (
        isinstance(pad_l, int)
        and isinstance(pad_r, int)
        and isinstance(input_w, StaticDim)
        and not (pad_l < int(input_w) and pad_r < int(input_w))
    ) or (
        isinstance(pad_t, int)
        and isinstance(pad_b, int)
        and isinstance(input_h, StaticDim)
        and not (pad_t < int(input_h) and pad_b < int(input_h))
    ):
        raise RuntimeError(
            "Padding size should be less than the corresponding input "
            f"dimension, but got: padding ({pad_l}, {pad_r}, {pad_t}, {pad_b}) "
            f"for input of size {tuple(input.shape)}"
        )
    paddings = _torch_pad_to_max_paddings(len(input.shape), padding)
    return max_ops.pad(input, paddings, mode="reflect")


# reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor


# relu(Tensor self) -> Tensor
@map_to(aten.relu)
def aten_relu(tensor: MaxTensor) -> MaxTensor:
    # inplace has no meaning in max since it's graph-based
    return F.relu(tensor)


# remainder.Scalar(Tensor self, Scalar other) -> Tensor
# remainder.Tensor(Tensor self, Tensor other) -> Tensor
@map_to(aten.remainder)
def aten_remainder(x: MaxTensor, y: MaxTensor | Scalar) -> MaxTensor:
    return operator.mod(x, y)


# repeat(Tensor self, SymInt[] repeats) -> Tensor
@map_to(aten.repeat)
def aten_repeat(input: MaxTensor, repeats: list[SymIntType]) -> MaxTensor:
    """
    Equivalent to torch.repeat - repeats the tensor along each dimension.
    Each dimension is repeated the number of times specified in repeats.
    When len(repeats) exceeds the input rank, torch prepends size-1 dims
    to the input first; MAX's tile requires matching ranks, so do the
    same reshape here.
    """
    rank = len(input.shape)
    if len(repeats) > rank:
        input = F.reshape(input, [1] * (len(repeats) - rank) + list(input.shape))
    return F.tile(input, repeats)


# replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
@map_to(aten.replication_pad2d)
def aten_replication_pad2d(input: MaxTensor, padding: list[SymIntType]) -> MaxTensor:
    # Unlike reflection padding, ATen's replication_pad2d (ReplicationPadding.cpp)
    # places no upper bound on the padding size relative to the input dimension --
    # every padded cell just repeats the nearest edge element.
    paddings = _torch_pad_to_max_paddings(len(input.shape), padding)
    return max_ops.pad(input, paddings, mode="edge")


# replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor


# TODO: handle in-place correctly
# relu_(Tensor(a!) self) -> Tensor(a!)
@map_to(aten.relu_)
def aten_relu_(tensor: MaxTensor) -> MaxTensor:
    # inplace has no meaning in max since it's graph-based
    return F.relu(tensor)


# resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
# round(Tensor self) -> Tensor


# rsqrt(Tensor self) -> Tensor
@map_to(aten.rsqrt)
def aten_rsqrt(x: MaxTensor) -> MaxTensor:
    return F.rsqrt(x)


# scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.scalar_tensor)
def aten_scalar_tensor(
    value: float | int,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    device: torch.device = None,
) -> MaxTensor:
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.get_default_device()

    return F.constant(
        value,
        dtype=torch_dtype_to_max(dtype),
        device=torch_device_to_max_device(device),
    )


# scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor
@map_to(aten.scaled_dot_product_attention)
def aten_scaled_dot_product_attention(
    query: MaxTensor,
    key: MaxTensor,
    value: MaxTensor,
    attn_mask: MaxTensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> MaxTensor:
    if attn_mask is not None and attn_mask.dtype == DType.bool:
        neg_inf = F.constant(float("-inf"), dtype=query.dtype, device=query.device)
        zero = F.constant(0.0, dtype=query.dtype, device=query.device)
        attn_mask = F.where(attn_mask, zero, neg_inf)

    query_device = query.device
    is_gpu = (
        not query_device.is_cpu()
        if isinstance(query_device, DeviceRef)
        else query_device.label != "cpu"
    )

    if (
        is_gpu
        and attn_mask is not None
        and dropout_p == 0.0
        and not is_causal
        and scale in (None, 0.125)
        and not enable_gqa
        and query.dtype == key.dtype == value.dtype == attn_mask.dtype == DType.float32
        and len(query.shape)
        == len(key.shape)
        == len(value.shape)
        == len(attn_mask.shape)
        == 4
        and query.shape[2] == 1
        and query.shape[3] == 64
        and query.shape[0] == key.shape[0] == value.shape[0]
        and query.shape[1] == key.shape[1] == value.shape[1]
        and key.shape == value.shape
        and attn_mask.shape[-1] == key.shape[2]
        and attn_mask.shape[0] in (1, query.shape[0])
        and attn_mask.shape[1] == attn_mask.shape[2] == 1
    ):
        return custom_mojo_ops.gpt2_decode_attention(query, key, value, attn_mask)

    output, _ = aten__scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return output


# scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
@map_to(aten.scatter.src)
def aten_scatter_src(
    input: MaxTensor, dim: int, index: MaxTensor, src: MaxTensor
) -> MaxTensor:
    """Scatters values from src tensor into input at positions specified by index along dimension dim.

    For a 3D tensor with dim=0, this performs:
        output[index[i][j][k]][j][k] = src[i][j][k]

    For a 3D tensor with dim=1, this performs:
        output[i][index[i][j][k]][k] = src[i][j][k]
    """
    return F.scatter(input, src, index, axis=dim)


# scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
@map_to(aten.scatter.value)
def aten_scatter_value(
    input: MaxTensor, dim: int, index: MaxTensor, value: Scalar
) -> MaxTensor:
    """Scatters a scalar value into input tensor at positions specified by index along dimension dim.

    For a 3D tensor with dim=0, this performs:
        output[index[i][j][k]][j][k] = value

    For a 3D tensor with dim=1, this performs:
        output[i][index[i][j][k]][k] = value
    """
    # Broadcast the scalar value to match the index shape
    # We need to create a tensor filled with the value in the same shape as index
    updates = F.broadcast_to(
        F.constant(value, dtype=input.dtype, device=input.device), index.shape
    )
    return F.scatter(input, updates, index, axis=dim)


# scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
# scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor


# aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
# aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
@map_to(aten.searchsorted)
def aten_searchsorted(
    sorted_sequence: MaxTensor,
    input: MaxTensor | Scalar,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: MaxTensor | None = None,
) -> MaxTensor:
    return _searchsorted_impl(
        sorted_sequence,
        input,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


# select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
@map_to(aten.select)
def aten_select(input: MaxTensor, dim: int, index: SymIntType) -> MaxTensor:
    """
    Equivalent to torch.select - selects a slice of the tensor along the given dimension at the given index.
    """
    nb_dims = len(input.shape)
    slices = [slice(None)] * nb_dims
    slices[dim] = index
    return input[slices]


# select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
@map_to(aten.select_scatter)
def aten_select_scatter(
    input: MaxTensor, src: MaxTensor, dim: int, index: int
) -> MaxTensor:
    """Embeds src values into input at the specified index along dimension dim.

    This is the inverse of select: given input with shape (d0, d1, ..., d_dim, ..., dn),
    and src with shape (d0, d1, ..., d_{dim-1}, d_{dim+1}, ..., dn),
    returns a tensor with input's shape where input[..., index, ...] = src.

    Implementation using where + mask:
        1. Create mask that is True only at the target index along dim
        2. Unsqueeze src to add back the dimension
        3. Broadcast src to match input's shape
        4. Use where to selectively replace values
    """
    # Handle negative dimension
    if dim < 0:
        dim = dim + len(input.shape)

    # Handle negative index
    dim_size = input.shape[dim]
    if index < 0:
        index = index + dim_size.dim

    # Step 1: Create a range tensor for the dimension to build the mask
    indices = F.arange(0, dim_size, 1, dtype=DType.int64, device=input.device)

    # Step 2: Create 1D boolean mask where indices == index
    index_tensor = F.constant(index, dtype=DType.int64, device=input.device)
    mask_1d = F.equal(indices, index_tensor)

    # Step 3: Reshape mask to have correct broadcasting shape
    # All dimensions except 'dim' should be 1
    mask_shape = [StaticDim(1)] * len(input.shape)
    mask_shape[dim] = dim_size
    mask = F.reshape(mask_1d, mask_shape)

    # Step 4: Broadcast mask to input's shape
    mask_expanded = F.broadcast_to(mask, input.shape)

    # Step 5: Unsqueeze src to add back the dimension
    src_unsqueezed = F.unsqueeze(src, dim)

    # Step 6: Broadcast src to match input's shape
    src_expanded = F.broadcast_to(src_unsqueezed, input.shape)

    # Step 7: Use where to select: where mask is True, use src, else use input
    return F.where(mask_expanded, src_expanded, input)


# sigmoid(Tensor self) -> Tensor
@map_to(aten.sigmoid)
def aten_sigmoid(input: MaxTensor) -> MaxTensor:
    return F.sigmoid(input)


# sign(Tensor self) -> Tensor
@map_to(aten.sign)
def aten_sign(x: MaxTensor) -> MaxTensor:
    # sign(x) = (x > 0) + (x < 0) * (-1)
    # This returns 1.0 for positive, -1.0 for negative, 0.0 for zero
    positive = F.cast(x > 0, dtype=x.dtype)
    negative = F.cast(x < 0, dtype=x.dtype)
    return positive + negative * (-1)


# silu(Tensor self) -> Tensor
@map_to(aten.silu)
def aten_silu(input: MaxTensor) -> MaxTensor:
    return input * F.sigmoid(input)


# sin(Tensor self) -> Tensor
@map_to(aten.sin)
def aten_sin(x: MaxTensor) -> MaxTensor:
    return F.sin(x)


# tan(Tensor self) -> Tensor
@map_to(aten.tan)
def aten_tan(x: MaxTensor) -> MaxTensor:
    return F.sin(x) / F.cos(x)


# tanh(Tensor self) -> Tensor
@map_to(aten.tanh)
def aten_tanh(x: MaxTensor) -> MaxTensor:
    return F.tanh(x)


# sinh(Tensor self) -> Tensor
@map_to(aten.sinh)
def aten_sinh(x: MaxTensor) -> MaxTensor:
    """Computes hyperbolic sine using sin(x) = (exp(x) - exp(-x)) / 2"""
    return (F.exp(x) - F.exp(-x)) / 2


# slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
@map_to(aten.slice)
def aten_slice(
    input: MaxTensor,
    dim: int,
    start: SymIntType | None = None,
    end: SymIntType | None = None,
    step: SymIntType = 1,
) -> MaxTensor:
    if end == 2**63 - 1:  # MAX_INT64
        end = None
    slices = [slice(None)] * len(input.shape)
    slices[dim] = slice(start, end, step)
    return input[*slices]


# slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
# sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)


# split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
@map_to(aten.split_with_sizes)
def aten_split_with_sizes(
    input: MaxTensor, split_sizes: list[SymIntType], dim: int = 0
) -> list[MaxTensor]:
    result = []
    start = 0
    for size in split_sizes:
        end = start + size
        result.append(aten_slice(input, dim, start, end))
        start = end
    return result


# sqrt(Tensor self) -> Tensor
@map_to(aten.sqrt)
def aten_sqrt(x: MaxTensor) -> MaxTensor:
    return F.sqrt(x)


# squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
# squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
@map_to(aten.squeeze)
def aten_squeeze(input: MaxTensor, dim: int | list[int]) -> MaxTensor:
    if isinstance(dim, int):
        dim = [dim]
    result = input
    for d in sorted(dim, reverse=True):
        # Handle negative dimensions
        actual_dim = d if d >= 0 else len(result.shape) + d
        # Only squeeze if the dimension has size 1
        if actual_dim < len(result.shape) and result.shape[actual_dim] == 1:
            result = F.squeeze(result, axis=actual_dim)
    return result


# sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
# sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
@map_to(aten.sub)
def aten_sub(
    input: MaxTensor | int | float, other: MaxTensor | Scalar, alpha: Scalar = 1
) -> MaxTensor:
    input, other = type_promotion(input, other)
    if alpha != 1:
        other = aten_mul(other, alpha)
    return input - other


# sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
@map_to(aten.sum)
def aten_sum(
    input: MaxTensor,
    dim: list[int] | int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> MaxTensor:
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        input = F.cast(input, dtype=max_dtype)
    elif input.dtype in (DType.bool, DType.uint8, DType.int8, DType.int16, DType.int32):
        # torch promotes bool/small-int sums to int64; MAX's reduction
        # kernel also rejects bool outright ("SIMD type must be numeric").
        input = F.cast(input, dtype=DType.int64)

    result = input

    if not dim:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Sum over each dimension
    for axis in sorted(dim, reverse=True):
        result = _reduce_sum(result, axis=axis)

    # Handle keepdim=False - squeeze the reduced dimensions
    if not keepdim:
        # MAX's sum keeps dimensions by default, so we need to squeeze
        for axis in sorted(dim, reverse=True):
            result = F.squeeze(result, axis=axis)

    return result


# sym_numel(Tensor self) -> SymInt
# sym_size.int(Tensor self, int dim) -> SymInt
# sym_storage_offset(Tensor self) -> SymInt
# sym_stride.int(Tensor self, int dim) -> SymInt
# tan(Tensor self) -> Tensor
# tanh(Tensor self) -> Tensor
# topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
# trunc(Tensor self) -> Tensor


# unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
@map_to(aten.unsqueeze)
def aten_unsqueeze(tensor: MaxTensor, dim: int) -> MaxTensor:
    return F.unsqueeze(tensor, axis=dim)


# upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
@map_to(aten.upsample_bilinear2d)
def aten_upsample_bilinear2d(
    input: MaxTensor,
    output_size: list[SymIntType],
    align_corners: bool,
    scale_factors: list[float] | None = None,
) -> MaxTensor:
    if scale_factors is not None:
        raise NotImplementedError(
            "The implementation of aten.upsample_bilinear2d doesn't support scale_factors yet. Please provide output_size instead."
        )
    if len(input.shape) == 4:
        # F.interpolate passes only spatial dimensions for 2D interpolation.
        # Max resize ops expect the full output shape.
        output_shape = [input.shape[0], input.shape[1], output_size[0], output_size[1]]
    else:
        raise ValueError(
            "The Mojo backend currently only supports upsample_bilinear2d for 4D tensors"
        )

    coordinate_transform_mode = 1 if align_corners else 0
    return F.resize_linear(
        input, output_shape, coordinate_transform_mode=coordinate_transform_mode
    )


# upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
# var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
# var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
@map_to(aten.var)
def aten_var(
    input: MaxTensor,
    dim: list[int] | None = None,
    *,
    correction: int | float | None = None,
    keepdim: bool = False,
) -> MaxTensor:
    if correction is None:
        correction = 1
    mean = aten_mean(input, dim=dim, keepdim=True)
    centered = input - mean
    var = aten_mean(centered * centered, dim=dim, keepdim=keepdim)
    if correction != 0:
        # Apply Bessel's correction: var * N / (N - correction)
        reduced_dims = range(len(input.shape)) if dim is None else dim
        n = 1
        for d in reduced_dims:
            n *= int(input.shape[d])
        var = var * (n / (n - correction))
    return var


# view(Tensor(a) self, SymInt[] size) -> Tensor(a)
@map_to(aten.view)
def aten_view(tensor: MaxTensor, *shape) -> MaxTensor:
    if len(shape) == 1 and isinstance(shape[0], tuple | list):
        target_shape = list(shape[0])
    else:
        target_shape = list(shape)
    return F.reshape(tensor, target_shape)


# _unsafe_view(Tensor self, SymInt[] size) -> Tensor
# Same as view but skips the safety check on strides. Used internally by PyTorch
# in decompositions (e.g. matmul 3D×2D) where the view is known to be valid.
@map_to(aten._unsafe_view)
def aten__unsafe_view(tensor: MaxTensor, *shape) -> MaxTensor:
    return aten_view(tensor, *shape)


# where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
@map_to(aten.where)
def aten_where(input: MaxTensor, condition: MaxTensor, other: MaxTensor) -> MaxTensor:
    return F.where(input, condition, other)


# stack(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.stack)
def aten_stack(tensors: list[MaxTensor], dim: int = 0) -> MaxTensor:
    return F.stack(tensors, axis=dim)


# tril(Tensor self, int diagonal=0) -> Tensor
@map_to(aten.tril)
def aten_tril(input: MaxTensor, diagonal: int = 0) -> MaxTensor:
    # tril keeps the lower triangular part of the matrix
    # diagonal=0: keep main diagonal and below
    # diagonal>0: include k diagonals above main (shift cutoff down)
    # diagonal<0: exclude |k| diagonals below main (shift cutoff up)
    if diagonal >= 0:
        # Include diagonal bands above the main diagonal and all below
        num_upper = diagonal

        # Only apply bounds check if we have static dimensions
        shape = input.shape
        if len(shape) >= 2:
            dim_n = shape[-1]
            # Check if dimension is static by trying to convert to int
            try:
                dim_n_val = int(dim_n)
                # Dimension can be converted to int, it is static
                # Clamp num_upper to avoid out of bounds error
                # num_upper can't be larger than the number of columns - 1
                num_upper = min(num_upper, dim_n_val - 1)
            except (TypeError, ValueError):
                # Dimension is dynamic, don't apply bounds check
                pass

        return F.band_part(input, num_lower=None, num_upper=num_upper)
    else:
        # Exclude diagonal bands by using exclude with the inverse band
        # We want to zero out everything above and including (-diagonal-1) diagonals below main
        # This is equivalent to keeping only bands starting from -diagonal below main
        # band_part doesn't directly support this, so we need a workaround
        # We can use exclude=True to invert the selection

        # Only apply bounds check if we have static dimensions
        shape = input.shape
        lower_limit = -diagonal - 1

        # Check if the last two dimensions are static (not dynamic)
        if len(shape) >= 2:
            dim_m = shape[-2]
            dim_n = shape[-1]
            # Check if both dimensions are static
            # Dim objects with a value set are static, check if we can convert to int
            try:
                dim_m_val = int(dim_m)
                dim_n_val = int(dim_n)
                # Both dimensions can be converted to int, they are static
                min_dim = min(dim_m_val, dim_n_val)
                # Clamp lower_limit to avoid out of bounds error
                if -diagonal >= min_dim:
                    # If -diagonal >= min_dim, the result is all zeros
                    lower_limit = min_dim - 1
                else:
                    lower_limit = -diagonal - 1
            except (TypeError, ValueError):
                # At least one dimension is dynamic, use original lower_limit
                pass

        return F.band_part(input, num_lower=lower_limit, num_upper=None, exclude=True)


# triu(Tensor self, int diagonal=0) -> Tensor
@map_to(aten.triu)
def aten_triu(input: MaxTensor, diagonal: int = 0) -> MaxTensor:
    # triu keeps the upper triangular part of the matrix
    # diagonal=0: keep main diagonal and above
    # diagonal>0: exclude k diagonals starting from main (shift cutoff up)
    # diagonal<0: include |k| diagonals below main (shift cutoff down)
    if diagonal <= 0:
        # Include |diagonal| bands below the main diagonal and all above
        num_lower = -diagonal

        # Only apply bounds check if we have static dimensions
        shape = input.shape
        if len(shape) >= 2:
            dim_m = shape[-2]
            # Check if dimension is static by trying to convert to int
            try:
                dim_m_val = int(dim_m)
                # Dimension can be converted to int, it is static
                # Clamp num_lower to avoid out of bounds error
                # num_lower can't be larger than the number of rows - 1
                num_lower = min(num_lower, dim_m_val - 1)
            except (TypeError, ValueError):
                # Dimension is dynamic, don't apply bounds check
                pass

        return F.band_part(input, num_lower=num_lower, num_upper=None)
    else:
        # Exclude diagonal bands by using exclude with the inverse band
        # We want to zero out everything below and including (diagonal-1) diagonals above main
        # This is equivalent to keeping only bands starting from diagonal above main
        # band_part doesn't directly support this, so we need a workaround
        # We can use exclude=True to invert the selection

        # Only apply bounds check if we have static dimensions
        shape = input.shape
        upper_limit = diagonal - 1

        # Check if the last two dimensions are static (not dynamic)
        if len(shape) >= 2:
            dim_m = shape[-2]
            dim_n = shape[-1]
            # Check if both dimensions are static
            # Dim objects with a value set are static, check if we can convert to int
            try:
                dim_m_val = int(dim_m)
                dim_n_val = int(dim_n)
                # Both dimensions can be converted to int, they are static
                min_dim = min(dim_m_val, dim_n_val)
                # Clamp upper_limit to avoid out of bounds error
                if diagonal >= min_dim:
                    # If diagonal >= min_dim, the result is all zeros
                    upper_limit = min_dim - 1
                else:
                    upper_limit = diagonal - 1
            except (TypeError, ValueError):
                # At least one dimension is dynamic, use original upper_limit
                pass

        return F.band_part(input, num_lower=None, num_upper=upper_limit, exclude=True)


# split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]
# split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]
@map_to(aten.split)
def aten_split(
    input: MaxTensor, split_size: int | list[int], dim: int = 0
) -> list[MaxTensor]:
    if isinstance(split_size, int):
        shape = int(input.shape[dim])
        new_split_size = [split_size] * (shape // split_size)
        if shape % split_size != 0:
            new_split_size.append(shape % split_size)
    else:
        new_split_size = split_size
    return F.split(input, new_split_size, dim)


@map_to(aten.unbind)
def aten_unbind(input: MaxTensor, dim: int = 0) -> list[MaxTensor]:
    """
    Equivalent to torch.unbind - removes a tensor dimension and returns a tuple of all slices along that dimension.
    """
    # Get the size of the dimension to unbind
    shape = input.shape
    if dim < 0:
        dim = len(shape) + dim

    size = int(shape[dim])

    # Use split with size 1 to get individual slices, then squeeze
    split_sizes = [1] * size
    split_tensors = F.split(input, split_sizes, dim)

    # Squeeze each tensor to remove the dimension we split along
    result = []
    for tensor in split_tensors:
        squeezed = F.squeeze(tensor, axis=dim)
        result.append(squeezed)

    return result


# For some reason, aot_autograd always decomposes repeat_interleave. No need to have an
# implementation here if it's never used.
# repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
# repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
# repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor


# t(Tensor(a) self) -> Tensor(a)
@map_to(aten.t.default)
def aten_t(input: MaxTensor) -> MaxTensor:
    return torch_transpose_equivalent(input, 0, 1)


def torch_transpose_equivalent(tensor, dim0, dim1):
    # Get the current tensor dimensions
    ndim = len(tensor.shape)

    # Handle negative dimensions
    if dim0 < 0:
        dim0 = ndim + dim0
    if dim1 < 0:
        dim1 = ndim + dim1

    # Validate dimensions
    if dim0 < 0 or dim0 >= ndim:
        raise ValueError(
            f"Dimension {dim0} out of range for tensor with {ndim} dimensions"
        )
    if dim1 < 0 or dim1 >= ndim:
        raise ValueError(
            f"Dimension {dim1} out of range for tensor with {ndim} dimensions"
        )

    # If dimensions are the same, no change needed
    if dim0 == dim1:
        return tensor

    # Create permutation list - swap dim0 and dim1
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

    return F.permute(tensor, perm)


# _foreach_add.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
@map_to(aten._foreach_add.Scalar)
def aten__foreach_add_scalar(
    self: list[MaxTensor],
    other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor,
    alpha: Scalar = 1,
) -> list[MaxTensor]:
    return [aten_add(x, other, alpha=alpha) for x in self]


# _foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_add.ScalarList)
def aten__foreach_add_scalar_list(
    self: list[MaxTensor], other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(scalars), but got {len(self)} and {len(other)}"
        )
    return [aten_add(tensor, scalar) for tensor, scalar in zip(self, other)]


# _foreach_add.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) -> Tensor[]
@map_to(aten._foreach_add.Tensor)
def aten__foreach_add_tensor(
    self: list[MaxTensor], other: MaxTensor, alpha: Scalar = 1
) -> list[MaxTensor]:
    return [aten_add(x, other, alpha=alpha) for x in self]


# _foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
@map_to(aten._foreach_add.List)
def aten__foreach_add_list(
    self: list[MaxTensor], other: list[MaxTensor], alpha: Scalar = 1
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(other), but got {len(self)} and {len(other)}"
        )
    return [aten_add(x, y, alpha=alpha) for x, y in zip(self, other)]


# _foreach_sub.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
@map_to(aten._foreach_sub.Scalar)
def aten__foreach_sub_scalar(
    self: list[MaxTensor],
    other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor,
    alpha: Scalar = 1,
) -> list[MaxTensor]:
    return [aten_sub(x, other, alpha=alpha) for x in self]


# _foreach_sub.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_sub.ScalarList)
def aten__foreach_sub_scalar_list(
    self: list[MaxTensor], other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(scalars), but got {len(self)} and {len(other)}"
        )
    return [aten_sub(tensor, scalar) for tensor, scalar in zip(self, other)]


# _foreach_sub.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
@map_to(aten._foreach_sub.List)
def aten__foreach_sub_list(
    self: list[MaxTensor], other: list[MaxTensor], alpha: Scalar = 1
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(other), but got {len(self)} and {len(other)}"
        )
    return [aten_sub(x, y, alpha=alpha) for x, y in zip(self, other)]


# _foreach_mul.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
@map_to(aten._foreach_mul.Scalar)
def aten__foreach_mul_scalar(
    self: list[MaxTensor], other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor
) -> list[MaxTensor]:
    return [aten_mul(x, other) for x in self]


# _foreach_mul.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_mul.ScalarList)
def aten__foreach_mul_scalar_list(
    self: list[MaxTensor], other: Scalar | list[MaxTensor] | list[Scalar] | MaxTensor
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(scalars), but got {len(self)} and {len(other)}"
        )
    return [aten_mul(tensor, scalar) for tensor, scalar in zip(self, other)]


# _foreach_mul.Tensor(Tensor[] self, Tensor other) -> Tensor[]
@map_to(aten._foreach_mul.Tensor)
def aten__foreach_mul_tensor(
    self: list[MaxTensor], other: MaxTensor
) -> list[MaxTensor]:
    return [aten_mul(x, other) for x in self]


# _foreach_mul.List(Tensor[] self, Tensor[] other) -> Tensor[]
@map_to(aten._foreach_mul.List)
def aten__foreach_mul_list(
    self: list[MaxTensor], other: list[MaxTensor]
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(other), but got {len(self)} and {len(other)}"
        )
    return [aten_mul(x, y) for x, y in zip(self, other)]


# _foreach_neg(Tensor[] self) -> Tensor[]
@map_to(aten._foreach_neg)
def aten__foreach_neg(self: list[MaxTensor]) -> list[MaxTensor]:
    return [aten_neg(x) for x in self]


# _foreach_pow.Scalar(Tensor[] self, Scalar exponent) -> Tensor[]
@map_to(aten._foreach_pow.Scalar)
def aten__foreach_pow_scalar(
    self: list[MaxTensor], exponent: Scalar
) -> list[MaxTensor]:
    return [aten_pow(x, exponent) for x in self]


# _foreach_pow.ScalarList(Tensor[] self, Scalar[] exponent) -> Tensor[]
@map_to(aten._foreach_pow.ScalarList)
def aten__foreach_pow_scalar_list(
    self: list[MaxTensor], exponent: list[Scalar]
) -> list[MaxTensor]:
    if len(self) != len(exponent):
        raise ValueError(
            f"Expected len(self) == len(exponent), but got {len(self)} and {len(exponent)}"
        )
    return [aten_pow(tensor, exp) for tensor, exp in zip(self, exponent)]


# _foreach_pow.List(Tensor[] self, Tensor[] exponent) -> Tensor[]
@map_to(aten._foreach_pow.List)
def aten__foreach_pow_list(
    self: list[MaxTensor], exponent: list[MaxTensor]
) -> list[MaxTensor]:
    if len(self) != len(exponent):
        raise ValueError(
            f"Expected len(self) == len(exponent), but got {len(self)} and {len(exponent)}"
        )
    return [aten_pow(x, exp) for x, exp in zip(self, exponent)]


# _foreach_pow.ScalarAndTensor(Scalar self, Tensor[] exponent) -> Tensor[]
@map_to(aten._foreach_pow.ScalarAndTensor)
def aten__foreach_pow_scalarandtensor(
    self: Scalar, exponent: list[MaxTensor]
) -> list[MaxTensor]:
    return [aten_pow(self, exp) for exp in exponent]


# _foreach_div.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
@map_to(aten._foreach_div.Scalar)
def aten__foreach_div_scalar(self: list[MaxTensor], other: Scalar) -> list[MaxTensor]:
    return [aten_div(x, other) for x in self]


# _foreach_div.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_div.ScalarList)
def aten__foreach_div_scalar_list(
    self: list[MaxTensor], other: list[Scalar]
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(scalars), but got {len(self)} and {len(other)}"
        )
    return [aten_div(tensor, scalar) for tensor, scalar in zip(self, other)]


# _foreach_div.Tensor(Tensor[] self, Tensor other) -> Tensor[]
@map_to(aten._foreach_div.Tensor)
def aten__foreach_div_tensor(
    self: list[MaxTensor], other: MaxTensor
) -> list[MaxTensor]:
    return [aten_div(x, other) for x in self]


# _foreach_div.List(Tensor[] self, Tensor[] other) -> Tensor[]
@map_to(aten._foreach_div.List)
def aten__foreach_div_list(
    self: list[MaxTensor], other: list[MaxTensor]
) -> list[MaxTensor]:
    if len(self) != len(other):
        raise ValueError(
            f"Expected len(self) == len(other), but got {len(self)} and {len(other)}"
        )
    return [aten_div(x, y) for x, y in zip(self, other)]


# _foreach_reciprocal(Tensor[] self) -> Tensor[]
@map_to(aten._foreach_reciprocal)
def aten__foreach_reciprocal(self: list[MaxTensor]) -> list[MaxTensor]:
    return [aten_reciprocal(x) for x in self]


# _foreach_sqrt(Tensor[] self) -> Tensor[]
@map_to(aten._foreach_sqrt)
def aten__foreach_sqrt(self: list[MaxTensor]) -> list[MaxTensor]:
    return [aten_sqrt(x) for x in self]


# _foreach_addcdiv.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
@map_to(aten._foreach_addcdiv.Scalar)
def aten__foreach_addcdiv_scalar(
    self: list[MaxTensor],
    tensor1: list[MaxTensor],
    tensor2: list[MaxTensor],
    value: Scalar = 1,
) -> list[MaxTensor]:
    if len(self) != len(tensor1) or len(self) != len(tensor2):
        raise ValueError(
            f"Expected len(self) == len(tensor1) == len(tensor2), but got {len(self)}, {len(tensor1)}, {len(tensor2)}"
        )
    return [
        aten_addcdiv(s, t1, t2, value=value)
        for s, t1, t2 in zip(self, tensor1, tensor2)
    ]


# _foreach_addcdiv.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_addcdiv.ScalarList)
def aten__foreach_addcdiv_scalarlist(
    self: list[MaxTensor],
    tensor1: list[MaxTensor],
    tensor2: list[MaxTensor],
    scalars: list[Scalar],
) -> list[MaxTensor]:
    if (
        len(self) != len(tensor1)
        or len(self) != len(tensor2)
        or len(self) != len(scalars)
    ):
        raise ValueError(
            f"Expected len(self) == len(tensor1) == len(tensor2) == len(scalars), but got {len(self)}, {len(tensor1)}, {len(tensor2)}, {len(scalars)}"
        )
    return [
        aten_addcdiv(s, t1, t2, value=val)
        for s, t1, t2, val in zip(self, tensor1, tensor2, scalars)
    ]


# NOTE: _foreach_addcdiv.Tensor is NOT supported
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# Use .Scalar or .ScalarList variants instead.
# See: https://github.com/pytorch/pytorch/issues/139795


# _foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
@map_to(aten._foreach_addcmul.Scalar)
def aten__foreach_addcmul_scalar(
    self: list[MaxTensor],
    tensor1: list[MaxTensor],
    tensor2: list[MaxTensor],
    value: Scalar = 1,
) -> list[MaxTensor]:
    if len(self) != len(tensor1) or len(self) != len(tensor2):
        raise ValueError(
            f"Expected len(self) == len(tensor1) == len(tensor2), but got {len(self)}, {len(tensor1)}, {len(tensor2)}"
        )
    return [
        aten_addcmul(s, t1, t2, value=value)
        for s, t1, t2 in zip(self, tensor1, tensor2)
    ]


# _foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
@map_to(aten._foreach_addcmul.ScalarList)
def aten__foreach_addcmul_scalarlist(
    self: list[MaxTensor],
    tensor1: list[MaxTensor],
    tensor2: list[MaxTensor],
    scalars: list[Scalar],
) -> list[MaxTensor]:
    if (
        len(self) != len(tensor1)
        or len(self) != len(tensor2)
        or len(self) != len(scalars)
    ):
        raise ValueError(
            f"Expected len(self) == len(tensor1) == len(tensor2) == len(scalars), but got {len(self)}, {len(tensor1)}, {len(tensor2)}, {len(scalars)}"
        )
    return [
        aten_addcmul(s, t1, t2, value=val)
        for s, t1, t2, val in zip(self, tensor1, tensor2, scalars)
    ]


# NOTE: _foreach_addcmul.Tensor is NOT supported
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# Use .Scalar or .ScalarList variants instead.
# See: https://github.com/pytorch/pytorch/issues/139795


# masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
# masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
@map_to(aten.masked_fill)
def aten_masked_fill(
    input: MaxTensor, mask: MaxTensor, value: Scalar | MaxTensor
) -> MaxTensor:
    return F.where(mask, value, input)


# transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
# transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
@map_to(aten.transpose)
def aten_transpose(input: MaxTensor, dim0: int | Dim, dim1: int | Dim) -> MaxTensor:
    return F.transpose(input, dim0, dim1)


if verbose_enabled():
    print(
        f"Removed  {number_of_decompositions_removed}/{original_decomposition_table_size} decomposition functions."
    )
