"""ATen op registrations for mojo_device eager mode.

Every op is either bound to its fast implementation in
`eager_kernels/aten_fast.py` (Mojo kernels over raw pointers) or raises
`NotImplementedError` with an actionable message. The old graph fallback
(per-op MAX graph build + interpret, ~2.2 ms/op) is gone — see
docs/strided_owning_tensors_design.md.

This module is the registration list and nothing else: the implementations
that need real Python code live in the `aten_ops/` package, grouped by family
(`transfer`, `factories`, `inplace`, `foreach`, ...), with their shared
plumbing in `aten_ops/support.py`. Adding an op means adding one line to the
alphabetical list at the bottom — plus, when it needs more than a name, one
function in the matching `aten_ops/` module.
"""

import functools
from collections.abc import Callable

import torch_mojo_backend.is_running_tests

from .aten_ops import foreach
from .aten_ops.autograd_preflight import (
    mojo_device__adaptive_avg_pool2d,
    mojo_device__scaled_dot_product_efficient_attention,
    mojo_device__softmax,
    mojo_device_avg_pool2d,
    mojo_device_cumsum,
    mojo_device_embedding,
    mojo_device_index,
    mojo_device_linear,
    mojo_device_max_pool2d_with_indices,
    mojo_device_native_batch_norm,
    mojo_device_native_group_norm,
    mojo_device_relu,
    mojo_device_scatter_src,
    mojo_device_sigmoid,
    mojo_device_softmax,
    mojo_device_tanh,
    mojo_device_upsample_bilinear2d,
)
from .aten_ops.blas import mojo_device_addr
from .aten_ops.factories import (
    empty_strided,
    mojo_device_arange,
    mojo_device_arange_start_out,
    mojo_device_empty_like,
    mojo_device_empty_memory_format,
    mojo_device_empty_permuted,
    mojo_device_full,
    mojo_device_full_like,
    mojo_device_new_empty,
    mojo_device_new_full,
    mojo_device_new_ones,
    mojo_device_new_zeros,
    mojo_device_ones,
    mojo_device_ones_like,
    mojo_device_scalar_tensor,
    mojo_device_zeros,
    mojo_device_zeros_like,
)
from .aten_ops.foreach import (
    mojo_device__foreach_mul__tensor,
    mojo_device__foreach_norm_scalar,
    mojo_device__foreach_sqrt,
)
from .aten_ops.inplace import (
    mojo_device_add_,
    mojo_device_fill__scalar,
    mojo_device_masked_fill_,
    mojo_device_mul_,
    mojo_device_relu_,
    mojo_device_zero_,
)
from .aten_ops.reductions import mojo_device_min_dim, mojo_device_min_dim_min
from .aten_ops.rng import mojo_device_normal_
from .aten_ops.support import _eager_impl, _not_implemented, _out_variant
from .aten_ops.transfer import mojo_device__copy_from, mojo_device__to_copy

# Global registry for functions to register
_aten_ops_registry: list[tuple[str, Callable]] = []

# Under tests, each registered op's dispatcher is wrapped with a call
# counter so `CallChecker` can assert the backend's impl for a given op ran
# — uniformly for fast, custom, and out-variant registrations (see
# torch_mojo_backend/testing.py). Keyed by the aten op name.
EAGER_CALL_COUNTERS: dict[str, Callable] = {}


def register_aten_op(op_name: str):
    """Decorator to mark a function for aten op registration.

    Args:
        op_name: The aten operation name (e.g., "aten::add.Tensor")
    """

    def decorator(func: Callable) -> Callable:
        if torch_mojo_backend.is_running_tests.IS_RUNNING_TESTS:

            @functools.wraps(func)
            def counted(*args, **kwargs):
                counted.call_count += 1
                return func(*args, **kwargs)

            counted.call_count = 0
            EAGER_CALL_COUNTERS[op_name] = counted
            _aten_ops_registry.append((op_name, counted))
            return counted
        _aten_ops_registry.append((op_name, func))
        return func

    return decorator


def _register_fast(op_name: str, fast_name: str) -> None:
    """Bind an op to `aten_fast.<fast_name>`."""
    register_aten_op(op_name)(_eager_impl(fast_name, op_name))


def _register_out(
    op_name: str, fast_name: str, *, dtype_policy: str = "safe_cast"
) -> None:
    """Bind an `out=` overload to a functional `aten_fast.<fast_name>`."""
    register_aten_op(op_name)(
        _out_variant(op_name, fast_name, dtype_policy=dtype_policy)
    )


def _register_foreach_inplace(op_name: str, fast_name: str) -> None:
    """Bind a mutable ()-returning foreach op, ATen's sequential semantics as
    the fallback."""
    register_aten_op(op_name)(foreach.inplace_dispatcher(op_name, fast_name))


def _register_missing(op_name: str) -> None:
    """Register an explicit raiser for an op with no fast implementation yet,
    so users get an actionable message and the remaining surface stays
    greppable."""
    register_aten_op(op_name)(_not_implemented(op_name))


# ----------------------------------------------------------------------------------
# The registrations, one aten name per entry, alphabetical.
#
#   _register_fast              -> aten_fast.fast_* directly
#   _register_out               -> a functional fast impl wrapped as out=
#   _register_foreach_inplace   -> batched foreach, sequential ATen fallback
#   _register_missing           -> explicit raiser, no fast impl yet
#   register_aten_op(...)(fn)   -> a hand-written impl from aten_ops/
# ----------------------------------------------------------------------------------

register_aten_op("aten::_adaptive_avg_pool2d")(mojo_device__adaptive_avg_pool2d)
_register_missing("aten::_adaptive_avg_pool2d_backward")
register_aten_op("aten::_copy_from")(mojo_device__copy_from)
_register_foreach_inplace(
    "aten::_foreach_add_.Scalar", "fast_aten__foreach_add__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_addcdiv_.ScalarList", "fast_aten__foreach_addcdiv__scalarlist"
)
_register_foreach_inplace(
    "aten::_foreach_addcmul_.Scalar", "fast_aten__foreach_addcmul__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_div_.ScalarList", "fast_aten__foreach_div__scalarlist"
)
_register_foreach_inplace(
    "aten::_foreach_lerp_.Scalar", "fast_aten__foreach_lerp__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_mul_.Scalar", "fast_aten__foreach_mul__scalar"
)
register_aten_op("aten::_foreach_mul_.Tensor")(mojo_device__foreach_mul__tensor)
register_aten_op("aten::_foreach_norm.Scalar")(mojo_device__foreach_norm_scalar)
register_aten_op("aten::_foreach_sqrt")(mojo_device__foreach_sqrt)
_register_fast("aten::_fused_adamw_", "fast_aten__fused_adamw")
_register_fast("aten::_fused_adamw_.tensor_lr", "fast_aten__fused_adamw")
_register_fast("aten::_local_scalar_dense", "fast_aten__local_scalar_dense")
_register_fast("aten::_log_softmax", "fast_aten__log_softmax")
_register_fast(
    "aten::_log_softmax_backward_data", "fast_aten__log_softmax_backward_data"
)
_register_fast(
    "aten::_native_batch_norm_legit_no_training",
    "fast_aten__native_batch_norm_legit_no_training",
)
_register_fast(
    "aten::_scaled_dot_product_attention_math",
    "fast_aten__scaled_dot_product_attention_math",
)
register_aten_op("aten::_scaled_dot_product_efficient_attention")(
    mojo_device__scaled_dot_product_efficient_attention
)
_register_fast(
    "aten::_scaled_dot_product_flash_attention",
    "fast_aten__scaled_dot_product_flash_attention",
)
_register_fast(
    "aten::_scaled_dot_product_flash_attention_backward",
    "fast_aten__scaled_dot_product_flash_attention_backward",
)
register_aten_op("aten::_softmax")(mojo_device__softmax)
register_aten_op("aten::_to_copy")(mojo_device__to_copy)
_register_fast("aten::_unsafe_view", "fast_aten__unsafe_view")
_register_fast("aten::abs", "fast_aten_abs")
_register_fast("aten::acos", "fast_aten_acos")
register_aten_op("aten::add_.Tensor")(mojo_device_add_)
_register_fast("aten::add.Tensor", "fast_aten_add")
_register_fast("aten::addcdiv", "fast_aten_addcdiv")
_register_out("aten::addcdiv.out", "fast_aten_addcdiv")
_register_fast("aten::addcmul", "fast_aten_addcmul")
_register_out("aten::addcmul.out", "fast_aten_addcmul")
_register_fast("aten::addmm", "fast_aten_addmm")
register_aten_op("aten::addr")(mojo_device_addr)
_register_fast("aten::alias", "fast_aten_alias")
_register_fast("aten::all", "fast_aten_all")
_register_fast("aten::all.dim", "fast_aten_all")
_register_fast("aten::all.dims", "fast_aten_all")
_register_fast("aten::amax", "fast_aten_amax")
_register_fast("aten::amin", "fast_aten_amin")
_register_fast("aten::any", "fast_aten_any")
_register_fast("aten::any.dim", "fast_aten_any")
_register_fast("aten::any.dims", "fast_aten_any")
_register_out("aten::any.out", "fast_aten_any", dtype_policy="bool_or_uint8")
register_aten_op("aten::arange")(mojo_device_arange)
register_aten_op("aten::arange.start_out")(mojo_device_arange_start_out)
_register_fast("aten::argmax", "fast_aten_argmax")
_register_fast("aten::argmin", "fast_aten_argmin")
_register_fast("aten::as_strided", "fast_aten_as_strided")
_register_fast("aten::asinh", "fast_aten_asinh")
_register_fast("aten::atanh", "fast_aten_atanh")
register_aten_op("aten::avg_pool2d")(mojo_device_avg_pool2d)
_register_fast("aten::bitwise_and.Scalar", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_and.Tensor", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_not", "fast_aten_bitwise_not")
_register_fast("aten::bitwise_or.Scalar", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_or.Tensor", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_xor.Scalar", "fast_aten_bitwise_xor")
_register_fast("aten::bitwise_xor.Tensor", "fast_aten_bitwise_xor")
_register_fast("aten::bmm", "fast_aten_bmm")
_register_fast("aten::bucketize.Scalar", "fast_aten_bucketize")
_register_out("aten::bucketize.Scalar_out", "fast_aten_bucketize", dtype_policy="exact")
_register_fast("aten::bucketize.Tensor", "fast_aten_bucketize")
_register_out("aten::bucketize.Tensor_out", "fast_aten_bucketize", dtype_policy="exact")
_register_fast("aten::cat", "fast_aten_cat")
_register_fast("aten::ceil", "fast_aten_ceil")
_register_fast("aten::clamp", "fast_aten_clamp")
_register_fast("aten::clone", "fast_aten_clone")
_register_fast("aten::convolution", "fast_aten_convolution")
_register_fast("aten::convolution_backward", "fast_aten_convolution_backward")
_register_fast("aten::cos", "fast_aten_cos")
_register_fast("aten::cosh", "fast_aten_cosh")
register_aten_op("aten::cumsum")(mojo_device_cumsum)
_register_fast("aten::detach", "fast_aten_detach")
_register_out("aten::div.out", "fast_aten_div")
_register_out("aten::div.out_mode", "fast_aten_div")
_register_fast("aten::div.Tensor", "fast_aten_div")
_register_fast("aten::div.Tensor_mode", "fast_aten_div")
register_aten_op("aten::embedding")(mojo_device_embedding)
_register_fast("aten::embedding_dense_backward", "fast_aten_embedding_dense_backward")
register_aten_op("aten::empty_like")(mojo_device_empty_like)
register_aten_op("aten::empty_permuted")(mojo_device_empty_permuted)
register_aten_op("aten::empty_strided")(empty_strided)
register_aten_op("aten::empty_strided.memory_format")(empty_strided)
register_aten_op("aten::empty.memory_format")(mojo_device_empty_memory_format)
_register_fast("aten::eq", "fast_aten_eq")
_register_fast("aten::eq.Scalar", "fast_aten_eq")
_register_fast("aten::eq.Tensor", "fast_aten_eq")
_register_fast("aten::erf", "fast_aten_erf")
_register_fast("aten::exp", "fast_aten_exp")
_register_fast("aten::expand", "fast_aten_expand")
register_aten_op("aten::fill_.Scalar")(mojo_device_fill__scalar)
_register_fast("aten::fill.Scalar", "fast_aten_fill_scalar")
_register_fast("aten::floor", "fast_aten_floor")
_register_fast("aten::floor_divide", "fast_aten_floor_divide")
_register_fast("aten::floor_divide.Scalar", "fast_aten_floor_divide")
_register_fast("aten::floordiv", "fast_aten_floor_divide")
register_aten_op("aten::full")(mojo_device_full)
register_aten_op("aten::full_like")(mojo_device_full_like)
_register_fast("aten::ge", "fast_aten_ge")
_register_fast("aten::ge.Scalar", "fast_aten_ge")
_register_fast("aten::ge.Tensor", "fast_aten_ge")
_register_fast("aten::gelu", "fast_aten_gelu")
_register_fast("aten::gelu_backward", "fast_aten_gelu_backward")
_register_fast("aten::gt", "fast_aten_gt")
_register_fast("aten::gt.Scalar", "fast_aten_gt")
_register_fast("aten::gt.Tensor", "fast_aten_gt")
register_aten_op("aten::index.Tensor")(mojo_device_index)
_register_fast("aten::isin.Tensor_Tensor", "fast_aten_isin")
_register_out("aten::isin.Tensor_Tensor_out", "fast_aten_isin", dtype_policy="exact")
_register_fast("aten::isnan", "fast_aten_isnan")
_register_fast("aten::le", "fast_aten_le")
_register_fast("aten::le.Scalar", "fast_aten_le")
_register_fast("aten::le.Tensor", "fast_aten_le")
_register_fast("aten::lerp.Scalar", "fast_aten_lerp")
_register_out("aten::lerp.Scalar_out", "fast_aten_lerp")
_register_out(
    "aten::linalg_vector_norm.out", "fast_aten_linalg_vector_norm", dtype_policy="exact"
)
register_aten_op("aten::linear")(mojo_device_linear)
_register_fast("aten::linear_backward", "fast_aten_linear_backward")
_register_fast("aten::log", "fast_aten_log")
_register_fast("aten::log1p", "fast_aten_log1p")
_register_fast("aten::logical_and", "fast_aten_logical_and")
_register_fast("aten::logical_not", "fast_aten_logical_not")
_register_fast("aten::logical_xor", "fast_aten_logical_xor")
_register_fast("aten::lt", "fast_aten_lt")
_register_fast("aten::lt.Scalar", "fast_aten_lt")
_register_fast("aten::lt.Tensor", "fast_aten_lt")
register_aten_op("aten::masked_fill_.Scalar")(mojo_device_masked_fill_)
register_aten_op("aten::masked_fill_.Tensor")(mojo_device_masked_fill_)
_register_fast("aten::masked_fill.Scalar", "fast_aten_masked_fill")
_register_fast("aten::masked_fill.Tensor", "fast_aten_masked_fill")
_register_fast("aten::max", "fast_aten_max")
register_aten_op("aten::max_pool2d_with_indices")(mojo_device_max_pool2d_with_indices)
_register_fast("aten::maximum", "fast_aten_maximum")
_register_fast("aten::mean", "fast_aten_mean")
# Registering the base name only covers the default overload; mean.dim would
# otherwise get decomposed by PyTorch into a chain of sum/div/... ops.
_register_fast("aten::mean.dim", "fast_aten_mean")
_register_out("aten::mean.out", "fast_aten_mean")
_register_fast("aten::min", "fast_aten_min")
register_aten_op("aten::min.dim")(mojo_device_min_dim)
register_aten_op("aten::min.dim_min")(mojo_device_min_dim_min)
_register_fast("aten::minimum", "fast_aten_minimum")
_register_fast("aten::mm", "fast_aten_mm")
register_aten_op("aten::mul_.Tensor")(mojo_device_mul_)
_register_out("aten::mul.out", "fast_aten_mul")
_register_fast("aten::mul.Tensor", "fast_aten_mul")
register_aten_op("aten::native_batch_norm")(mojo_device_native_batch_norm)
_register_fast("aten::native_dropout", "fast_aten_native_dropout")
_register_fast("aten::native_dropout_backward", "fast_aten_native_dropout_backward")
register_aten_op("aten::native_group_norm")(mojo_device_native_group_norm)
_register_fast("aten::native_layer_norm", "fast_aten_native_layer_norm")
_register_fast(
    "aten::native_layer_norm_backward", "fast_aten_native_layer_norm_backward"
)
_register_fast("aten::ne", "fast_aten_ne")
_register_fast("aten::ne.Scalar", "fast_aten_ne")
_register_fast("aten::ne.Tensor", "fast_aten_ne")
_register_fast("aten::neg", "fast_aten_neg")
register_aten_op("aten::new_empty")(mojo_device_new_empty)
register_aten_op("aten::new_full")(mojo_device_new_full)
register_aten_op("aten::new_ones")(mojo_device_new_ones)
register_aten_op("aten::new_zeros")(mojo_device_new_zeros)
_register_fast(
    "aten::nll_loss_backward.grad_input", "fast_aten_nll_loss_backward_grad_input"
)
_register_fast("aten::nll_loss_forward.output", "fast_aten_nll_loss_forward_output")
_register_fast("aten::nonzero", "fast_aten_nonzero")
register_aten_op("aten::normal_")(mojo_device_normal_)
register_aten_op("aten::ones")(mojo_device_ones)
register_aten_op("aten::ones_like")(mojo_device_ones_like)
_register_fast("aten::permute", "fast_aten_permute")
_register_fast("aten::pow.Tensor_Scalar", "fast_aten_pow")
_register_fast("aten::pow.Tensor_Tensor", "fast_aten_pow_tensor_tensor")
_register_fast("aten::reciprocal", "fast_aten_reciprocal")
register_aten_op("aten::relu")(mojo_device_relu)
register_aten_op("aten::relu_")(mojo_device_relu_)
_register_fast("aten::remainder.Scalar", "fast_aten_remainder")
_register_fast("aten::remainder.Scalar_Tensor", "fast_aten_remainder")
_register_fast("aten::remainder.Tensor", "fast_aten_remainder")
_register_fast("aten::repeat", "fast_aten_repeat")
_register_fast("aten::rsqrt", "fast_aten_rsqrt")
register_aten_op("aten::scalar_tensor")(mojo_device_scalar_tensor)
_register_fast(
    "aten::scaled_dot_product_attention", "fast_aten_scaled_dot_product_attention"
)
register_aten_op("aten::scatter.src")(mojo_device_scatter_src)
_register_fast("aten::scatter.value", "fast_aten_scatter_value")
_register_fast("aten::searchsorted.Scalar", "fast_aten_searchsorted")
_register_out(
    "aten::searchsorted.Scalar_out", "fast_aten_searchsorted", dtype_policy="exact"
)
_register_fast("aten::searchsorted.Tensor", "fast_aten_searchsorted")
_register_out(
    "aten::searchsorted.Tensor_out", "fast_aten_searchsorted", dtype_policy="exact"
)
_register_fast("aten::select_scatter", "fast_aten_select_scatter")
_register_fast("aten::select.int", "fast_aten_select")
register_aten_op("aten::sigmoid")(mojo_device_sigmoid)
_register_fast("aten::sign", "fast_aten_sign")
_register_fast("aten::silu", "fast_aten_silu")
_register_fast("aten::sin", "fast_aten_sin")
_register_fast("aten::sinh", "fast_aten_sinh")
_register_fast("aten::slice.Tensor", "fast_aten_slice")
register_aten_op("aten::softmax.int")(mojo_device_softmax)
_register_fast("aten::split_with_sizes", "fast_aten_split_with_sizes")
_register_fast("aten::split.Tensor", "fast_aten_split")
_register_fast("aten::sqrt", "fast_aten_sqrt")
_register_fast("aten::squeeze.dim", "fast_aten_squeeze_dim")
_register_fast("aten::stack", "fast_aten_stack")
_register_out("aten::sub.out", "fast_aten_sub")
_register_fast("aten::sub.Tensor", "fast_aten_sub")
_register_fast("aten::sum.dim_IntList", "fast_aten_sum")
_register_fast("aten::t", "fast_aten_t")
_register_fast("aten::tan", "fast_aten_tan")
register_aten_op("aten::tanh")(mojo_device_tanh)
_register_fast("aten::transpose.int", "fast_aten_transpose")
_register_fast("aten::tril", "fast_aten_tril")
_register_fast("aten::triu", "fast_aten_triu")
_register_fast("aten::unbind.int", "fast_aten_unbind")
_register_fast("aten::uniform_", "fast_aten_uniform_")
_register_fast("aten::unsqueeze", "fast_aten_unsqueeze")
register_aten_op("aten::upsample_bilinear2d")(mojo_device_upsample_bilinear2d)
_register_fast("aten::var.correction", "fast_aten_var")
_register_fast("aten::view", "fast_aten_view")
_register_fast("aten::where.self", "fast_aten_where")
register_aten_op("aten::zero_")(mojo_device_zero_)
register_aten_op("aten::zeros")(mojo_device_zeros)
register_aten_op("aten::zeros_like")(mojo_device_zeros_like)
