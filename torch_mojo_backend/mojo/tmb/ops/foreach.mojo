"""ATen ops: foreach group (see agents_docs/native_backend.md).

Every elementwise `aten::_foreach_*` overload is a row of FOREACH_OVERLOADS
(below) and runs through one op body, `op_foreach`: the row says which
arguments the schema has, which element math the generic multi-tensor apply
(tmb/kernels/foreach) runs, and which per-tensor op ATen's own slow kernel
calls. When CUDA's fast-route conditions hold the whole list is one batched
launch; otherwise the op falls back to ATen's sequential semantics: the
per-tensor op once per list element through `tmb_call_op`, exactly what
ATen's CompositeExplicitAutograd "slow" kernels do
(`aten/src/ATen/native/ForeachOpsKernels.cpp`). That fallback is what makes
declining safe here, unlike the general porting rule of raising
`unsupported(...)`: this native op runs as a real PrivateUse1 kernel, so
calling the *same* op name from inside itself would re-enter our own
registration (infinite recursion) -- calling the *per-tensor* op name
instead (`add_.Scalar`, `addcmul_`, ...) does not, and matches the exact
semantics ATen's own slow kernels implement.

GradScaler's two ops, `_amp_foreach_non_finite_check_and_unscale_` (a kind of
the same multi-tensor apply) and `_amp_update_scale_` (one thread), have no
CompositeExplicitAutograd registration either, so they take no sequential
fallback: what the batched kernel cannot take goes through a contiguous
temporary, and what no kernel supports is declined.

The fused optimizers (`_fused_{adam,adamw,sgd,adagrad}_`, what
torch.optim runs with `fused=True`) are the same: no CompositeExplicitAutograd
registration, so no sequential fallback. They validate like stock CUDA and
raise its errors; see their section below.
"""
from std.builtin.sort import sort
from std.math import ceildiv
from std.utils import IndexList

from tmb.backend.abi import (
    ST_FLOAT16,
    ST_INT64,
    ST_COMPLEX32,
    ST_COMPLEX64,
    ST_COMPLEX128,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_UINT8,
    TAG_BOOL,
    TAG_DOUBLE,
    TAG_INT,
    max_dtype,
    new_scalar,
    TAG_BOOL_LIST,
    TAG_COMPLEX,
    TAG_DOUBLE_LIST,
    TAG_INT_LIST,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    new_strided,
    v_scalar_is_integral,
    T,
    bool_arg,
    is_dense,
    Value,
    Values,
    call_op,
    dtype_code,
    f64_bits,
    new_like,
    new_tensor,
    own,
    Owned,
    ret_ref,
    ret_tensor_list,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_int,
    v_scalar_is_bool,
    v_tensor,
    v_tensor_list,
    TAG_NONE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev, copy_d2d
from tmb.kernels.optimizer.foreach_clip_contract import FOREACH_CHUNK_ELEMENTS
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.float8 import is_float8
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.backend.registry import Site, impl
from tmb.kernels.foreach.contract import (
    FE_REDUCE_CHUNK,
    FR_L0,
    FR_L1,
    FR_L2,
    FR_LINF,
    FR_LNEGINF,
    FR_LP,
    FR_MAX,
    FE_ROLE_ALPHA,
    FE_ROLE_B,
    FE_ROLE_C,
    FE_ROLE_NONE,
    fe_variant,
    foreach_apple_supported,
)
from tmb.ops.common import (
    cast_into,
    cast_to,
    copy_strided_into,
    fill_value,
    device_str,
    is_float_stype,
    known_stype,
    promote_types,
    scalar_to_float,
    scalar_to_int,
)
from tmb.ops.data_movement import _scalar_type_name


# --- the sequential per-tensor fallback ---------------------------------
#
# A foreach op that declines its batched fast path replicates ATen's own
# sequential decomposition by calling the underlying per-tensor op through
# the full dispatcher (never the SAME op name, which would re-enter this very
# registration). Every such call allocates a fresh result handle -- the shim
# wraps even the tensor an in-place op hands back -- so the results go through
# `Results`, which releases whatever is not taken.


def _tensor_value(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _none_value() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


def _seq_vector_norm(
    op: String, t: T, ord_v: Value, dtype_v: Value
) raises -> T:
    """`linalg_vector_norm` / `linalg__powsum` of the whole tensor, what
    ATen's slow `_foreach_norm` / `_foreach_powsum` call per tensor."""
    var args = List[Value](capacity=5)
    args.append(_tensor_value(t))
    args.append(ord_v.copy())
    args.append(_none_value())  # dim=None
    args.append(Value(TAG_BOOL, 0, 0, 0))  # keepdim=False
    args.append(dtype_v.copy())
    var rets = call_op(op, "", args^, 1)
    return rets.take_tensor(0)


# --- overlap / aliasing (aten_fast._foreach_tensors_overlap /
# _foreach_mutation_hazard, simplified to pairwise interval checks: foreach
# lists are not so long that an O(n^2) sweep matters next to the kernel
# launches around it) ----------------------------------------------------


def _overlaps(a: T, b: T) -> Bool:
    if a.numel == 0 or b.numel == 0:
        return False
    var a_begin = a.ptr
    var a_end = a_begin + a.numel * a.itemsize
    var b_begin = b.ptr
    var b_end = b_begin + b.numel * b.itemsize
    return a_begin < b_end and b_begin < a_end


def _self_overlaps(ts: List[T]) -> Bool:
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if _overlaps(ts[i], ts[j]):
                return True
    return False


# --- fast-path qualification (aten_fast._foreach_lists): every tensor across
# every list mojo-resident, contiguous, on one shared device, one shared
# dtype, with index-aligned tensors sharing a shape ------------


def _tensor_qualifies(t: T, device: Int, dtype: DType) -> Bool:
    return t.on_mojo() and t.device == device and t.dtype == dtype and t.contig


def _qualifies1(a: List[T], allow_half: Bool) raises -> Bool:
    if len(a) == 0:
        return False
    var first = a[0].copy()
    if not first.on_mojo():
        return False
    # Metal's batched kernels accept only float32. Half lists must use the
    # existing per-tensor scalar operations instead of entering that kernel.
    if dev(first.device)[].api == "metal" and first.dtype != DType.float32:
        return False
    if allow_half:
        if (
            first.dtype != DType.float32
            and first.dtype != DType.float16
            and first.dtype != DType.bfloat16
        ):
            return False
    elif first.dtype != DType.float32:
        return False
    for t in a:
        if not _tensor_qualifies(t, first.device, first.dtype):
            return False
    return True


# --- Apple's _foreach_norm: the optimizer family's float32 ForeachL2Norm ------


def _foreach_norm_launch(tensors: List[T]) raises -> List[T]:
    var outs = List[Owned]()
    var total_chunks = 0
    for t in tensors:
        outs.append(
            own(new_tensor(IndexList[MAX_RANK](1), 0, t.stype, t.device))
        )
        total_chunks += ceildiv(t.numel, FOREACH_CHUNK_ELEMENTS)
    var metadata = List[Int]()
    for i in range(len(tensors)):
        metadata.append(tensors[i].ptr)
        metadata.append(outs[i].t.ptr)
        metadata.append(tensors[i].numel)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = max(total_chunks, 1)
    var partials = own(new_tensor(shape, 1, ST_FLOAT32, tensors[0].device))
    var ctx = ctx_for(tensors[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachL2Norm")
    call.arg_dtype(0, DType.float32)
    call.arg_dtype(1, DType.float32)
    call.arg_dtype(2, DType.float32)
    call.out_dtype(DType.float32)
    call.tuple(metadata)
    call.int(partials.t.ptr)
    call.int(partials.t.numel)
    call.int(cp)
    call.run()
    _ = partials  # alive past the launch (its last use above is the pointer read)
    _ = ctx
    var result = List[T]()
    for i in range(len(outs)):
        result.append(outs[i].take())
    return result^


# --- the foreach overload table ----------------------------------------------
#
# Every elementwise `aten::_foreach_*` overload is one row below and nothing
# else: `op_foreach[row, inplace]` is the only op body, registered for each
# row's functional and in-place schema by `register_foreach`. A row names
#
#   * the schemas (functional, in-place; "" when the overload has none);
#   * `kind`, the element math of the generic multi-tensor apply
#     (tmb/kernels/foreach, `functors.foreach_math`);
#   * `shape`, which arguments the schema has (S_*), hence the kernel's
#     operand variant (`_variant`);
#   * the per-tensor op ATen's own CompositeExplicitAutograd slow kernel
#     calls for each element (ForeachOpsKernels.cpp), which is what runs when
#     the fast route does not apply;
#   * the dtypes CUDA's fast route takes (its AT_DISPATCH set and its
#     `does_op_promote_integer_inputs_to_float` flag, D_*), and whether this
#     kind's float64 math lowers on the GPU (`f64`).
#
# The fast route is CUDA's `can_use_fast_route` (ForeachUtils.h): every
# tensor of every list on one mojo device, non-overlapping and dense, of one
# dtype, index-aligned tensors sharing sizes and strides, and no Scalar that
# would promote the result dtype -- plus the one rule CUDA leaves to chance:
# an in-place list that overlaps itself or another operand (other than the
# same tensor at the same index) runs sequentially. Anything else falls back
# to the per-tensor op through the dispatcher, exactly what ATen's slow
# kernel does: calling the per-tensor op name (`add_.Scalar`, `addcmul_`, ...)
# never re-enters this registration, so declining is always safe.

# Argument shapes of the foreach schemas.
comptime S_UNARY = 0  # (Tensor[] self)
comptime S_SCALAR = 1  # (Tensor[] self, Scalar scalar)
comptime S_SCALARLIST = 2  # (Tensor[] self, Scalar[] scalars)
comptime S_LIST = 3  # (Tensor[] self, Tensor[] other[, *, Scalar alpha])
comptime S_TENSOR = 4  # (Tensor[] self, Tensor other[, *, Scalar alpha])
comptime S_PW_SCALAR = 5  # (self, Tensor[] tensor1, Tensor[] tensor2, Scalar)
comptime S_PW_SCALARLIST = 6  # (self, tensor1, tensor2, Scalar[] scalars)
comptime S_PW_TENSOR = 7  # (self, tensor1, tensor2, Tensor scalars) (CPU, 1-D)
comptime S_LERP_LIST = 8  # (self, Tensor[] tensors1, Tensor[] weights)
comptime S_LERP_SCALAR = 9  # (self, Tensor[] tensors1, Scalar weight)
comptime S_LERP_SCALARLIST = 10  # (self, Tensor[] tensors1, Scalar[] weight)
comptime S_RPOW = 11  # (Scalar self, Tensor[] exponent)

# Fast-route dtypes (CUDA's dispatch set for the overload).
comptime D_FLOAT = 0  # float32 / float16 / bfloat16 (+ float64 if `f64`)
comptime D_INT = 1  # and the integer dtypes
comptime D_ALL = 2  # and bool


@fieldwise_init
struct ForeachOverload(ImplicitlyCopyable, Movable):
    var name: StaticString  # the functional schema, "" if none
    var name_: StaticString  # the in-place schema, "" if none
    var kind: StaticString
    var shape: Int
    var op: StaticString  # per-tensor op of the functional slow path
    var overload: StaticString
    var op_: StaticString  # per-tensor op of the in-place slow path
    var overload_: StaticString
    var dtypes: Int
    var f64: Bool
    var alpha: Bool  # the per-tensor op takes an alpha (add, sub)
    var reciprocal: Bool  # div.Scalar: multiply by 1 / scalar, as CUDA does
    var apple: Bool  # the in-place / sqrt route Apple's fixed-arity kernels take


def _unary(
    name: StaticString,
    name_: StaticString,
    kind: StaticString,
    op: StaticString,
    op_: StaticString,
    dtypes: Int,
    f64: Bool,
    apple: Bool = False,
) -> ForeachOverload:
    return ForeachOverload(
        name,
        name_,
        kind,
        S_UNARY,
        op,
        "",
        op_,
        "",
        dtypes,
        f64,
        False,
        False,
        apple,
    )


def _binary(
    name: StaticString,
    name_: StaticString,
    kind: StaticString,
    shape: Int,
    op: StaticString,
    overload: StaticString,
    op_: StaticString,
    overload_: StaticString,
    dtypes: Int,
    alpha: Bool = False,
    reciprocal: Bool = False,
    apple: Bool = False,
) -> ForeachOverload:
    return ForeachOverload(
        name,
        name_,
        kind,
        shape,
        op,
        overload,
        op_,
        overload_,
        dtypes,
        True,
        alpha,
        reciprocal,
        apple,
    )


# One row per overload pair, one line per row: a table read across.
# fmt: off
comptime FOREACH_OVERLOADS: List[ForeachOverload] = [
    # unary: ForeachUnaryOp.cu. Floating lists only, except abs (every
    # dtype), neg (not bool) and zero_; float64 where the GPU lowers it.
    _unary("_foreach_abs", "_foreach_abs_", "abs", "aten::abs", "aten::abs_", D_ALL, True),
    _unary("_foreach_acos", "_foreach_acos_", "acos", "aten::acos", "aten::acos_", D_FLOAT, False),
    _unary("_foreach_asin", "_foreach_asin_", "asin", "aten::asin", "aten::asin_", D_FLOAT, False),
    _unary("_foreach_atan", "_foreach_atan_", "atan", "aten::atan", "aten::atan_", D_FLOAT, False),
    _unary("_foreach_ceil", "_foreach_ceil_", "ceil", "aten::ceil", "aten::ceil_", D_FLOAT, True),
    _unary("_foreach_cos", "_foreach_cos_", "cos", "aten::cos", "aten::cos_", D_FLOAT, False),
    _unary("_foreach_cosh", "_foreach_cosh_", "cosh", "aten::cosh", "aten::cosh_", D_FLOAT, False),
    _unary("_foreach_erf", "_foreach_erf_", "erf", "aten::erf", "aten::erf_", D_FLOAT, False),
    _unary("_foreach_erfc", "_foreach_erfc_", "erfc", "aten::erfc", "aten::erfc_", D_FLOAT, False),
    _unary("_foreach_exp", "_foreach_exp_", "exp", "aten::exp", "aten::exp_", D_FLOAT, False),
    _unary("_foreach_expm1", "_foreach_expm1_", "expm1", "aten::expm1", "aten::expm1_", D_FLOAT, False),
    _unary("_foreach_floor", "_foreach_floor_", "floor", "aten::floor", "aten::floor_", D_FLOAT, True),
    _unary("_foreach_frac", "_foreach_frac_", "frac", "aten::frac", "aten::frac_", D_FLOAT, True),
    _unary("_foreach_lgamma", "_foreach_lgamma_", "lgamma", "aten::lgamma", "aten::lgamma_", D_FLOAT, False),
    _unary("_foreach_log", "_foreach_log_", "log", "aten::log", "aten::log_", D_FLOAT, False),
    _unary("_foreach_log10", "_foreach_log10_", "log10", "aten::log10", "aten::log10_", D_FLOAT, False),
    _unary("_foreach_log1p", "_foreach_log1p_", "log1p", "aten::log1p", "aten::log1p_", D_FLOAT, False),
    _unary("_foreach_log2", "_foreach_log2_", "log2", "aten::log2", "aten::log2_", D_FLOAT, False),
    _unary("_foreach_neg", "_foreach_neg_", "neg", "aten::neg", "aten::neg_", D_INT, True),
    _unary("_foreach_reciprocal", "_foreach_reciprocal_", "reciprocal", "aten::reciprocal", "aten::reciprocal_", D_FLOAT, True),
    _unary("_foreach_round", "_foreach_round_", "round", "aten::round", "aten::round_", D_FLOAT, True),
    _unary("_foreach_rsqrt", "_foreach_rsqrt_", "rsqrt", "aten::rsqrt", "aten::rsqrt_", D_FLOAT, False),
    _unary("_foreach_sigmoid", "_foreach_sigmoid_", "sigmoid", "aten::sigmoid", "aten::sigmoid_", D_FLOAT, False),
    _unary("_foreach_sign", "_foreach_sign_", "sign", "aten::sign", "aten::sign_", D_FLOAT, True),
    _unary("_foreach_sin", "_foreach_sin_", "sin", "aten::sin", "aten::sin_", D_FLOAT, False),
    _unary("_foreach_sinh", "_foreach_sinh_", "sinh", "aten::sinh", "aten::sinh_", D_FLOAT, False),
    _unary("_foreach_sqrt", "_foreach_sqrt_", "sqrt", "aten::sqrt", "aten::sqrt_", D_FLOAT, True, apple=True),
    _unary("_foreach_tan", "_foreach_tan_", "tan", "aten::tan", "aten::tan_", D_FLOAT, False),
    _unary("_foreach_tanh", "_foreach_tanh_", "tanh", "aten::tanh", "aten::tanh_", D_FLOAT, False),
    _unary("_foreach_trunc", "_foreach_trunc_", "trunc", "aten::trunc", "aten::trunc_", D_FLOAT, True),
    _unary("", "_foreach_zero_", "zero", "", "aten::zero_", D_ALL, True),
    # binary: ForeachBinaryOp{Scalar,ScalarList,List,ScalarTensor}.cu.
    _binary("_foreach_add.Scalar", "_foreach_add_.Scalar", "add", S_SCALAR, "aten::add", "Scalar", "aten::add_", "Scalar", D_ALL, alpha=True, apple=True),
    _binary("_foreach_add.ScalarList", "_foreach_add_.ScalarList", "add", S_SCALARLIST, "aten::add", "Scalar", "aten::add_", "Scalar", D_ALL, alpha=True),
    _binary("_foreach_add.List", "_foreach_add_.List", "add", S_LIST, "aten::add", "Tensor", "aten::add_", "Tensor", D_ALL, alpha=True),
    _binary("_foreach_add.Tensor", "_foreach_add_.Tensor", "add", S_TENSOR, "aten::add", "Tensor", "aten::add_", "Tensor", D_ALL, alpha=True),
    _binary("_foreach_sub.Scalar", "_foreach_sub_.Scalar", "sub", S_SCALAR, "aten::sub", "Scalar", "aten::sub_", "Scalar", D_ALL, alpha=True),
    _binary("_foreach_sub.ScalarList", "_foreach_sub_.ScalarList", "sub", S_SCALARLIST, "aten::sub", "Scalar", "aten::sub_", "Scalar", D_ALL, alpha=True),
    _binary("_foreach_sub.List", "_foreach_sub_.List", "sub", S_LIST, "aten::sub", "Tensor", "aten::sub_", "Tensor", D_ALL, alpha=True),
    _binary("_foreach_mul.Scalar", "_foreach_mul_.Scalar", "mul", S_SCALAR, "aten::mul", "Scalar", "aten::mul_", "Scalar", D_ALL, apple=True),
    _binary("_foreach_mul.ScalarList", "_foreach_mul_.ScalarList", "mul", S_SCALARLIST, "aten::mul", "Scalar", "aten::mul_", "Scalar", D_ALL),
    _binary("_foreach_mul.List", "_foreach_mul_.List", "mul", S_LIST, "aten::mul", "Tensor", "aten::mul_", "Tensor", D_ALL),
    _binary("_foreach_mul.Tensor", "_foreach_mul_.Tensor", "mul", S_TENSOR, "aten::mul", "Tensor", "aten::mul_", "Tensor", D_ALL, apple=True),
    # div by a Scalar is CUDA's multiply by its reciprocal, slow path too.
    _binary("_foreach_div.Scalar", "_foreach_div_.Scalar", "mul", S_SCALAR, "aten::mul", "Scalar", "aten::mul_", "Scalar", D_FLOAT, reciprocal=True),
    _binary("_foreach_div.ScalarList", "_foreach_div_.ScalarList", "div", S_SCALARLIST, "aten::div", "Scalar", "aten::div_", "Scalar", D_FLOAT),
    _binary("_foreach_div.List", "_foreach_div_.List", "div", S_LIST, "aten::div", "Tensor", "aten::div_", "Tensor", D_FLOAT),
    _binary("_foreach_div.Tensor", "_foreach_div_.Tensor", "div", S_TENSOR, "aten::div", "Tensor", "aten::div_", "Tensor", D_FLOAT),
    # clamp_max / minimum are CUDA's `minimum`, clamp_min / maximum its
    # `maximum`; the slow path is clamp_max / clamp_min for all four.
    _binary("_foreach_clamp_max.Scalar", "_foreach_clamp_max_.Scalar", "minimum", S_SCALAR, "aten::clamp_max", "", "aten::clamp_max_", "", D_FLOAT),
    _binary("_foreach_clamp_max.ScalarList", "_foreach_clamp_max_.ScalarList", "minimum", S_SCALARLIST, "aten::clamp_max", "", "aten::clamp_max_", "", D_FLOAT),
    _binary("_foreach_clamp_max.List", "_foreach_clamp_max_.List", "minimum", S_LIST, "aten::clamp_max", "Tensor", "aten::clamp_max_", "Tensor", D_FLOAT),
    _binary("_foreach_minimum.Scalar", "_foreach_minimum_.Scalar", "minimum", S_SCALAR, "aten::clamp_max", "", "aten::clamp_max_", "", D_FLOAT),
    _binary("_foreach_minimum.ScalarList", "_foreach_minimum_.ScalarList", "minimum", S_SCALARLIST, "aten::clamp_max", "", "aten::clamp_max_", "", D_FLOAT),
    _binary("_foreach_minimum.List", "_foreach_minimum_.List", "minimum", S_LIST, "aten::clamp_max", "Tensor", "aten::clamp_max_", "Tensor", D_FLOAT),
    _binary("_foreach_clamp_min.Scalar", "_foreach_clamp_min_.Scalar", "maximum", S_SCALAR, "aten::clamp_min", "", "aten::clamp_min_", "", D_FLOAT),
    _binary("_foreach_clamp_min.ScalarList", "_foreach_clamp_min_.ScalarList", "maximum", S_SCALARLIST, "aten::clamp_min", "", "aten::clamp_min_", "", D_FLOAT),
    _binary("_foreach_clamp_min.List", "_foreach_clamp_min_.List", "maximum", S_LIST, "aten::clamp_min", "Tensor", "aten::clamp_min_", "Tensor", D_FLOAT),
    _binary("_foreach_maximum.Scalar", "_foreach_maximum_.Scalar", "maximum", S_SCALAR, "aten::clamp_min", "", "aten::clamp_min_", "", D_FLOAT),
    _binary("_foreach_maximum.ScalarList", "_foreach_maximum_.ScalarList", "maximum", S_SCALARLIST, "aten::clamp_min", "", "aten::clamp_min_", "", D_FLOAT),
    _binary("_foreach_maximum.List", "_foreach_maximum_.List", "maximum", S_LIST, "aten::clamp_min", "Tensor", "aten::clamp_min_", "Tensor", D_FLOAT),
    _binary("_foreach_pow.Scalar", "_foreach_pow_.Scalar", "pow", S_SCALAR, "aten::pow", "Tensor_Scalar", "aten::pow_", "Scalar", D_FLOAT),
    _binary("_foreach_pow.ScalarList", "_foreach_pow_.ScalarList", "pow", S_SCALARLIST, "aten::pow", "Tensor_Scalar", "aten::pow_", "Scalar", D_FLOAT),
    _binary("_foreach_pow.List", "_foreach_pow_.List", "pow", S_LIST, "aten::pow", "Tensor_Tensor", "aten::pow_", "Tensor", D_FLOAT),
    # An integer exponent list promotes, as result_type and CPU torch say;
    # stock CUDA truncates the base instead (ForeachBinaryOpScalar.cu:184
    # takes its fast route without looking at the scalar, :47 converts it
    # with `scalar.to<opmath_t>()`): _foreach_pow(1.5, [int [1, 2]]) is
    # int64 [1, 1] there, float32 [1.5, 2.25] here.
    _binary("_foreach_pow.ScalarAndTensor", "", "rpow", S_RPOW, "aten::pow", "Scalar", "", "", D_FLOAT),
    # pointwise: ForeachPointwiseOp.cu (floating lists only).
    _binary("_foreach_addcmul.Scalar", "_foreach_addcmul_.Scalar", "addcmul", S_PW_SCALAR, "aten::addcmul", "", "aten::addcmul_", "", D_FLOAT, apple=True),
    _binary("_foreach_addcmul.ScalarList", "_foreach_addcmul_.ScalarList", "addcmul", S_PW_SCALARLIST, "aten::addcmul", "", "aten::addcmul_", "", D_FLOAT),
    _binary("_foreach_addcmul.Tensor", "_foreach_addcmul_.Tensor", "addcmul", S_PW_TENSOR, "aten::addcmul", "", "aten::addcmul_", "", D_FLOAT),
    _binary("_foreach_addcdiv.Scalar", "_foreach_addcdiv_.Scalar", "addcdiv", S_PW_SCALAR, "aten::addcdiv", "", "aten::addcdiv_", "", D_FLOAT),
    _binary("_foreach_addcdiv.ScalarList", "_foreach_addcdiv_.ScalarList", "addcdiv", S_PW_SCALARLIST, "aten::addcdiv", "", "aten::addcdiv_", "", D_FLOAT),
    _binary("_foreach_addcdiv.Tensor", "_foreach_addcdiv_.Tensor", "addcdiv", S_PW_TENSOR, "aten::addcdiv", "", "aten::addcdiv_", "", D_FLOAT),
    # ternary: ForeachTernaryOp.cu (floating lists only).
    _binary("_foreach_lerp.List", "_foreach_lerp_.List", "lerp", S_LERP_LIST, "aten::lerp", "Tensor", "aten::lerp_", "Tensor", D_FLOAT),
    _binary("_foreach_lerp.Scalar", "_foreach_lerp_.Scalar", "lerp", S_LERP_SCALAR, "aten::lerp", "Scalar", "aten::lerp_", "Scalar", D_FLOAT, apple=True),
    _binary("_foreach_lerp.ScalarList", "_foreach_lerp_.ScalarList", "lerp", S_LERP_SCALARLIST, "aten::lerp", "Scalar", "aten::lerp_", "Scalar", D_FLOAT),
]
# fmt: on


def _variant(e: ForeachOverload, inplace: Bool) -> Int:
    """The kernel's operand variant (tmb/kernels/foreach/contract.mojo) of a
    row: where each operand of `foreach_math` comes from."""
    var s = e.shape
    var out = not inplace
    if s == S_SCALAR or s == S_SCALARLIST or s == S_RPOW:
        return fe_variant(1, out, FE_ROLE_B, False)
    if s == S_LIST:
        return fe_variant(
            2, out, FE_ROLE_ALPHA if e.alpha else FE_ROLE_NONE, False
        )
    if s == S_TENSOR:
        return fe_variant(
            1, out, FE_ROLE_ALPHA if e.alpha else FE_ROLE_NONE, True
        )
    if s == S_PW_SCALAR or s == S_PW_SCALARLIST or s == S_PW_TENSOR:
        return fe_variant(3, out, FE_ROLE_ALPHA, False)
    if s == S_LERP_LIST:
        return fe_variant(3, out, FE_ROLE_NONE, False)
    if s == S_LERP_SCALAR or s == S_LERP_SCALARLIST:
        return fe_variant(2, out, FE_ROLE_C, False)
    return fe_variant(1, out, FE_ROLE_NONE, False)


# --- arguments ------------------------------------------------------------------


def _int_scalar(x: Int) -> Value:
    return Value(TAG_SCALAR_INT, 0, Int64(x), 0)


def _scalar_list(v: Value) raises -> List[Value]:
    """A `Scalar[]` argument, element by element (see `to_record` in
    native/csrc/shim_dispatch.cpp: the list is tagged by the type every
    element agrees on, so a mixed int/float list arrives as doubles)."""
    var out = List[Value](capacity=Int(v.len))
    if v.tag == TAG_INT_LIST:
        var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
        for i in range(Int(v.len)):
            out.append(Value(TAG_SCALAR_INT, 0, p[unsafe_offset=i], 0))
    elif v.tag == TAG_DOUBLE_LIST:
        var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
        for i in range(Int(v.len)):
            out.append(Value(TAG_SCALAR_DOUBLE, 0, p[unsafe_offset=i], 0))
    elif v.tag == TAG_BOOL_LIST:
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
        for i in range(Int(v.len)):
            out.append(
                Value(TAG_SCALAR_BOOL, 0, Int64(p[unsafe_offset=i] != 0), 0)
            )
    else:
        raise Error("expected a Scalar[] argument, got record tag ", v.tag)
    return out^


def _cpu_element(t: T, i: Int) raises -> Value:
    """Element `i` of a contiguous CPU tensor as the Scalar
    `Tensor::item()` / `scalar_t` makes of it."""
    if t.dtype == DType.float32:
        var p = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return Value(
            TAG_SCALAR_DOUBLE, 0, f64_bits(Float64(p[unsafe_offset=i])), 0
        )
    if t.dtype == DType.float64:
        var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return Value(TAG_SCALAR_DOUBLE, 0, f64_bits(p[unsafe_offset=i]), 0)
    if t.dtype == DType.float16:
        var p = Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return Value(
            TAG_SCALAR_DOUBLE,
            0,
            f64_bits(p[unsafe_offset=i].cast[DType.float64]()),
            0,
        )
    if t.dtype == DType.bfloat16:
        var p = Pointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return Value(
            TAG_SCALAR_DOUBLE,
            0,
            f64_bits(p[unsafe_offset=i].cast[DType.float64]()),
            0,
        )
    if t.dtype == DType.bool or t.dtype == DType.uint8:
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        var x = Int64(p[unsafe_offset=i])
        if t.dtype == DType.bool:
            return Value(TAG_SCALAR_BOOL, 0, Int64(x != 0), 0)
        return _int_scalar(Int(x))
    if t.dtype == DType.int8:
        var p = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return _int_scalar(Int(p[unsafe_offset=i]))
    if t.dtype == DType.int16:
        var p = Pointer[Int16, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return _int_scalar(Int(p[unsafe_offset=i]))
    if t.dtype == DType.int32:
        var p = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return _int_scalar(Int(p[unsafe_offset=i]))
    if t.dtype == DType.int64:
        var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=t.ptr)
        return _int_scalar(Int(p[unsafe_offset=i]))
    unsupported("a CPU scalar of dtype " + String(t.dtype))
    return _int_scalar(0)


def _packed_scalars(v: Value, expect: Int) raises -> List[Value]:
    """ForeachUtils.h's `convert_tensor_to_scalar_list`: the pointwise
    `.Tensor` overloads' scalars, a 1-D CPU tensor."""
    var t = v_tensor(v)
    if not t.on_cpu():
        raise Error(
            "Expected scalars to be on CPU, got ", device_str(t), " instead."
        )
    if not t.contig:
        raise Error("Expected scalars to be contiguous.")
    if t.rank != 1:
        raise Error(
            "Expected packed scalar Tensor to be of dimension 1. Got ",
            t.rank,
            " instead.",
        )
    if t.dim(0) != expect:
        raise Error(
            "Expected length of scalars to match input of length ",
            expect,
            " but got ",
            t.dim(0),
            " instead.",
        )
    var out = List[Value](capacity=expect)
    for i in range(expect):
        out.append(_cpu_element(t, i))
    return out^


def _reciprocal(v: Value) raises -> Value:
    """ForeachBinaryOpScalar.cu's `scalar_reciprocal`: 1 / x in double."""
    if v.tag == TAG_COMPLEX:
        unsupported("a complex Scalar for _foreach_div")
    return Value(TAG_SCALAR_DOUBLE, 0, f64_bits(1.0 / v_f64(v)), 0)


struct _ForeachArgs(Movable):
    """One call's operands, read by the row's shape: the tensor lists (self
    or the exponents first), the Scalar or the per-tensor Scalars, the 0-d
    `other` tensor, alpha."""

    var lists: List[List[T]]
    var scalar: Value
    var scalars: List[Value]
    var other: List[T]  # the `.Tensor` overloads' operand, else empty
    var alpha: Value

    def __init__(out self, e: ForeachOverload, args: Values) raises:
        self.lists = List[List[T]]()
        self.scalar = _int_scalar(1)
        self.scalars = List[Value]()
        self.other = List[T]()
        self.alpha = _int_scalar(1)
        var s = e.shape
        if s == S_RPOW:
            self.scalar = args[unsafe_offset=0].copy()
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            return
        self.lists.append(v_tensor_list(args[unsafe_offset=0]))
        if s == S_SCALAR:
            self.scalar = args[unsafe_offset=1].copy()
        elif s == S_SCALARLIST:
            self.scalars = _scalar_list(args[unsafe_offset=1])
        elif s == S_LIST:
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            if e.alpha:
                self.alpha = args[unsafe_offset=2].copy()
        elif s == S_TENSOR:
            self.other.append(v_tensor(args[unsafe_offset=1]))
            if e.alpha:
                self.alpha = args[unsafe_offset=2].copy()
        elif s == S_LERP_LIST:
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            self.lists.append(v_tensor_list(args[unsafe_offset=2]))
        elif s == S_LERP_SCALAR:
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            self.scalar = args[unsafe_offset=2].copy()
        elif s == S_LERP_SCALARLIST:
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            self.scalars = _scalar_list(args[unsafe_offset=2])
        elif s != S_UNARY:  # the pointwise shapes
            self.lists.append(v_tensor_list(args[unsafe_offset=1]))
            self.lists.append(v_tensor_list(args[unsafe_offset=2]))
            if s == S_PW_SCALAR:
                self.scalar = args[unsafe_offset=3].copy()
            elif s == S_PW_SCALARLIST:
                self.scalars = _scalar_list(args[unsafe_offset=3])
            else:
                self.scalars = _packed_scalars(
                    args[unsafe_offset=3], len(self.lists[0])
                )

    def check(self, e: ForeachOverload) raises:
        """ForeachUtils.h's `check_foreach_api_restrictions`."""
        for k in range(len(self.lists)):
            if len(self.lists[k]) == 0:
                raise Error("Tensor list must have at least one tensor.")
        for k in range(1, len(self.lists)):
            if len(self.lists[k]) != len(self.lists[0]):
                raise Error(
                    "Tensor lists must have the same number of tensors, got ",
                    len(self.lists[0]),
                    " and ",
                    len(self.lists[k]),
                )
        var s = e.shape
        if (
            s == S_SCALARLIST
            or s == S_PW_SCALARLIST
            or s == S_PW_TENSOR
            or s == S_LERP_SCALARLIST
        ) and len(self.scalars) != len(self.lists[0]):
            raise Error(
                "Tensor list must have same number of elements as scalar list."
            )

    def scalar_at(self, e: ForeachOverload, i: Int) -> Value:
        """The Scalar tensor `i` is combined with (the reciprocal already
        taken for div)."""
        if len(self.scalars) != 0:
            return self.scalars[i].copy()
        return self.scalar.copy()


# --- the fast route ---------------------------------------------------------------


def _scalar_keeps_dtype(v: Value, dt: DType) -> Bool:
    """`result_type(scalar, tensor) == tensor.scalar_type()`: a Scalar of a
    lower category than the list's dtype does not promote it."""
    if v.tag == TAG_COMPLEX:
        return False
    if dt.is_floating_point():
        return True
    if dt == DType.bool:
        return v_scalar_is_bool(v)
    return v_scalar_is_integral(v)


def _dtype_on_route(e: ForeachOverload, dt: DType) -> Bool:
    if dt == DType.float32 or dt == DType.float16 or dt == DType.bfloat16:
        return True
    if dt == DType.float64:
        return e.f64
    if (
        dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
    ):
        return e.dtypes != D_FLOAT
    if dt == DType.bool:
        return e.dtypes == D_ALL
    return False


def _same_layout(a: T, b: T) -> Bool:
    """Equal sizes, and equal strides on every dim of size > 1
    (`_check_tensors_share_sizes_and_strides`)."""
    if not a.same_shape(b):
        return False
    for i in range(a.rank):
        if a.dim(i) != 1 and a.stride(i) != b.stride(i):
            return False
    return True


def _lists_on_route(lists: List[List[T]]) raises -> Bool:
    """`_check_tensors_share_device_and_dtype` and
    `_check_tensors_share_sizes_and_strides`, over every list."""
    var first = lists[0][0].copy()
    if not first.on_mojo():
        return False
    for k in range(len(lists)):
        for i in range(len(lists[k])):
            ref t = lists[k][i]
            if (
                not t.on_mojo()
                or t.device != first.device
                or t.dtype != first.dtype
                or not (t.contig or is_dense(t.shape, t.strides, t.rank))
            ):
                return False
            if k > 0 and not _same_layout(lists[0][i], t):
                return False
    return True


def _same_span(a: T, b: T) -> Bool:
    return a.ptr == b.ptr and _same_layout(a, b)


def _inplace_hazard(p: _ForeachArgs) -> Bool:
    """An in-place launch has no order between its blocks, so a written
    tensor may share bytes with nothing but itself at its own index."""
    ref written = p.lists[0]
    var read = List[T]()
    for k in range(1, len(p.lists)):
        for i in range(len(p.lists[k])):
            if not _same_span(written[i], p.lists[k][i]):
                read.append(p.lists[k][i].copy())
    for t in p.other:
        read.append(t.copy())
    return _writes_overlap(written, read)


def _opmath_stype(dt: DType) -> Int32:
    return ST_FLOAT64 if dt == DType.float64 else ST_FLOAT32


def _scalar_slot(v: Value, t: T) raises -> Int:
    """A Scalar as the kernel's per-tensor slot (`Scalar.to<opmath_t>()`,
    with c10's overflow check): float64 bits for a floating list, the value
    for an integer or bool one."""
    if t.dtype.is_floating_point():
        return Int(f64_bits(scalar_to_float(v, _opmath_stype(t.dtype))))
    if t.dtype == DType.bool:
        return 1 if v_bool(v) else 0
    return scalar_to_int(v, t.stype)


def _fast_route(
    e: ForeachOverload, inplace: Bool, p: _ForeachArgs
) raises -> Bool:
    if not _lists_on_route(p.lists):
        return False
    ref first = p.lists[0][0]
    var dt = first.dtype
    if not _dtype_on_route(e, dt):
        return False
    if dev(first.device)[].api == "metal":
        if (
            dt != DType.float32
            or not e.apple
            or not foreach_apple_supported(e.kind, _variant(e, inplace))
        ):
            return False
    var s = e.shape
    # The Scalars that must not promote: the overload's scalar(s), and add /
    # sub's alpha where the schema has one.
    if s == S_SCALAR or s == S_PW_SCALAR or s == S_LERP_SCALAR:
        if not _scalar_keeps_dtype(p.scalar, dt):
            return False
    elif s == S_RPOW:
        if p.scalar.tag == TAG_COMPLEX:
            return False
    for v in p.scalars:
        if not _scalar_keeps_dtype(v, dt):
            return False
    if (s == S_LIST or s == S_TENSOR) and e.alpha:
        if not _scalar_keeps_dtype(p.alpha, dt):
            return False
    if s == S_TENSOR:
        ref other = p.other[0]
        if other.dtype != dt:
            return False
        if not other.on_mojo() or other.device != first.device:
            # CUDA's fast route does not look at the scalar's device; its
            # kernel then refuses it.
            raise Error(
                "scalar tensor is expected to be on " if inplace else (
                    "scalar tensor expected to be on "
                ),
                device_str(first),
                " but is on ",
                device_str(other),
            )
    if inplace and _inplace_hazard(p):
        return False
    return True


def _mta_launch(
    kind: StringSlice,
    variant: Int,
    lists: List[List[T]],
    outs: List[T],
    scalars: List[Int],
    device_addr: Int,
    aux_addr: Int,
) raises:
    """One multi-tensor apply over index-aligned lists (tmb/kernels/foreach):
    five ints per tensor, the operand addresses then the numel."""
    ref a = lists[0]
    var metadata = List[Int](capacity=5 * len(a))
    for i in range(len(a)):
        metadata.append(a[i].ptr)
        metadata.append(lists[1][i].ptr if len(lists) > 1 else 0)
        metadata.append(lists[2][i].ptr if len(lists) > 2 else 0)
        metadata.append(outs[i].ptr if len(outs) > 0 else 0)
        metadata.append(a[i].numel)
    var ctx = ctx_for(a[0].device)
    var call = KernelCall("foreach", kind)
    call.flag("FE_VARIANT", variant)
    call.arg_dtype(0, a[0].dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.int(device_addr)
    call.int(aux_addr)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _launch(
    e: ForeachOverload, inplace: Bool, p: _ForeachArgs, rets: Values
) raises:
    ref selves = p.lists[0]
    var n = len(selves)
    var role_scalars = List[Int]()
    var s = e.shape
    if (s == S_LIST or s == S_TENSOR) and e.alpha:
        var slot = _scalar_slot(p.alpha, selves[0])
        for _ in range(n):
            role_scalars.append(slot)
    elif s != S_UNARY and s != S_LIST and s != S_TENSOR and s != S_LERP_LIST:
        for i in range(n):
            role_scalars.append(_scalar_slot(p.scalar_at(e, i), selves[i]))
    var device_addr = p.other[0].ptr if len(p.other) > 0 else 0
    var outs = List[Owned]()
    var out_ts = List[T]()
    if not inplace:
        for t in selves:
            outs.append(
                own(new_strided(t.shape, t.strides, t.rank, t.stype, t.device))
            )
            out_ts.append(outs[len(outs) - 1].t.copy())
    _mta_launch(
        e.kind,
        _variant(e, inplace),
        p.lists,
        out_ts,
        role_scalars,
        device_addr,
        0,
    )
    if inplace:
        # A `Tensor(a!)[]` gets no ADInplaceOrView bump: one per entry here.
        for t in selves:
            t.bump_version()
        return
    var result = List[T]()
    for i in range(len(outs)):
        result.append(outs[i].take())
    ret_tensor_list(rets, 0, result)


# --- the slow route: ATen's own per-tensor composition ------------------------


def _per_tensor_args(
    e: ForeachOverload, p: _ForeachArgs, i: Int
) -> List[Value]:
    var s = e.shape
    var out = List[Value](capacity=4)
    if s == S_RPOW:
        out.append(p.scalar.copy())
        out.append(_tensor_value(p.lists[0][i]))
        return out^
    out.append(_tensor_value(p.lists[0][i]))
    if s == S_SCALAR or s == S_SCALARLIST:
        out.append(p.scalar_at(e, i))
        if e.alpha:
            out.append(_int_scalar(1))
    elif s == S_LIST:
        out.append(_tensor_value(p.lists[1][i]))
        if e.alpha:
            out.append(p.alpha.copy())
    elif s == S_TENSOR:
        out.append(_tensor_value(p.other[0]))
        if e.alpha:
            out.append(p.alpha.copy())
    elif s == S_LERP_LIST:
        out.append(_tensor_value(p.lists[1][i]))
        out.append(_tensor_value(p.lists[2][i]))
    elif s == S_LERP_SCALAR or s == S_LERP_SCALARLIST:
        out.append(_tensor_value(p.lists[1][i]))
        out.append(p.scalar_at(e, i))
    elif s != S_UNARY:  # pointwise
        out.append(_tensor_value(p.lists[1][i]))
        out.append(_tensor_value(p.lists[2][i]))
        out.append(p.scalar_at(e, i))
    return out^


def _per_tensor(
    e: ForeachOverload, inplace: Bool, p: _ForeachArgs, rets: Values
) raises:
    var op = String(e.op_ if inplace else e.op)
    var overload = String(e.overload_ if inplace else e.overload)
    var results = List[Owned]()
    for i in range(len(p.lists[0])):
        var r = call_op(op, overload, _per_tensor_args(e, p, i), 1)
        if not inplace:
            results.append(own(r.take_tensor(0)))
    if inplace:
        return
    var result = List[T]()
    for i in range(len(results)):
        result.append(results[i].take())
    ret_tensor_list(rets, 0, result)


def _sub_check(t: T, v: Value) raises:
    """BinaryOps.h's `sub_check(self, scalar)`: no bool on either side."""
    var bool_self = t.dtype == DType.bool
    var bool_scalar = v_scalar_is_bool(v)
    if bool_self and bool_scalar:
        raise Error(
            "Subtraction, the `-` operator, with two bool tensors is not"
            " supported. Use the `^` or `logical_xor()` operator instead."
        )
    if bool_self or bool_scalar:
        raise Error(
            "Subtraction, the `-` operator, with a bool tensor is not"
            " supported. If you are trying to invert a mask, use the `~` or"
            " `logical_not()` operator instead."
        )


def _scalar_twin(e: ForeachOverload) -> ForeachOverload:
    """The `.Scalar` overload a `.Tensor` row (mul, div) runs when its
    operand is a CPU tensor: ForeachBinaryOpScalarTensor.cu calls
    `foreach_tensor_<op>_scalar_kernel_cuda(tensors, scalar.item())`, and
    div's Scalar overload is the multiply by the reciprocal. Built from the
    row itself, so a functional and an in-place row both map to their own
    form."""
    var div = e.kind == "div"
    return ForeachOverload(
        e.name,
        e.name_,
        "mul" if div else e.kind,
        S_SCALAR,
        "aten::mul" if div else e.op,
        "Scalar",
        "aten::mul_" if div else e.op_,
        "Scalar",
        e.dtypes,
        e.f64,
        e.alpha,
        div,
        e.apple,
    )


def _foreach(
    e: ForeachOverload, inplace: Bool, args: Values, rets: Values
) raises:
    var p = _ForeachArgs(e, args)
    if e.shape == S_TENSOR and p.other[0].on_cpu() and not e.alpha:
        # mul / div by a CPU tensor: CUDA takes its `item()` and runs the
        # Scalar overload (div's being the reciprocal multiply).
        ref other = p.other[0]
        if other.numel != 1:
            raise Error(
                "a Tensor with ",
                other.numel,
                " elements cannot be converted to Scalar",
            )
        var scalar_row = _scalar_twin(e)
        p.scalar = _cpu_element(other, 0)
        p.other = List[T]()
        if scalar_row.reciprocal:
            p.scalar = _reciprocal(p.scalar)
        p.check(scalar_row)
        if _fast_route(scalar_row, inplace, p):
            _launch(scalar_row, inplace, p, rets)
        else:
            _per_tensor(scalar_row, inplace, p, rets)
        return
    p.check(e)
    if e.kind == "sub":
        # ForeachBinaryOpScalar{,List}.cu check before anything runs: the
        # Scalar overload the first tensor, the ScalarList one every pair.
        # The List overload checks nothing up front: a bool pair on the
        # fast route (bool alpha) computes, any other raises in the
        # per-tensor sub, as on CUDA.
        if e.shape == S_SCALAR:
            _sub_check(p.lists[0][0], p.scalar)
        elif e.shape == S_SCALARLIST:
            for i in range(len(p.lists[0])):
                _sub_check(p.lists[0][i], p.scalars[i])
    if e.shape == S_TENSOR:
        ref other = p.other[0]
        if other.rank != 0 or other.numel != 1:
            # The in-place fast kernel words it without "it"
            # (ForeachBinaryOpScalarTensor.cu's foreach_binary_op_).
            var fast_inplace = (
                inplace
                and other.dtype == p.lists[0][0].dtype
                and _lists_on_route(p.lists)
            )
            raise Error(
                "scalar tensor expected to be 0 dim but has " if fast_inplace else (
                    "scalar tensor expected to be 0 dim but it has "
                ),
                other.rank,
                " dimensions and ",
                other.numel,
                " elements.",
            )
    if e.reciprocal:
        p.scalar = _reciprocal(p.scalar)
    if _fast_route(e, inplace, p):
        _launch(e, inplace, p, rets)
    else:
        _per_tensor(e, inplace, p, rets)


def op_foreach[
    row: Int, inplace: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """Every row of FOREACH_OVERLOADS, functional and in place."""
    comptime e = FOREACH_OVERLOADS[row]
    _foreach(e, inplace, args, rets)


def foreach_addc_inplace(
    dest: T, t1: T, t2: T, value: Float64, multiplies: Bool
) raises:
    """`dest += value * (t1 * t2)` (or `/`) as a one-tensor multi-tensor
    apply: the per-tensor addcmul_ / addcdiv_ route for same-dtype dense
    operands of one layout (tmb/ops/binary.mojo). Its version bump is the
    caller's dispatcher's."""
    var lists = List[List[T]]()
    lists.append([dest.copy()])
    lists.append([t1.copy()])
    lists.append([t2.copy()])
    var scalars = List[Int]()
    scalars.append(Int(f64_bits(value)))
    _mta_launch(
        "addcmul" if multiplies else "addcdiv",
        fe_variant(3, False, FE_ROLE_ALPHA, False),
        lists,
        List[T](),
        scalars,
        0,
        0,
    )


def foreach_lerp_inplace(dest: T, end: T, weight: Float64) raises:
    """`dest = lerp(dest, end, weight)` as a one-tensor multi-tensor apply
    (the per-tensor lerp_.Scalar route of tmb/ops/binary.mojo)."""
    var lists = List[List[T]]()
    lists.append([dest.copy()])
    lists.append([end.copy()])
    var scalars = List[Int]()
    scalars.append(Int(f64_bits(weight)))
    _mta_launch(
        "lerp",
        fe_variant(2, False, FE_ROLE_C, False),
        lists,
        List[T](),
        scalars,
        0,
        0,
    )


# --- ops -------------------------------------------------------------------


def _batch_copy_dtype(dt: DType) -> Bool:
    """Mirrors tmb/kernels/data_movement/entry.mojo's `COPY_BATCH_DTYPES`."""
    return (
        dt == DType.float64
        or dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint64
        or dt == DType.uint32
        or dt == DType.uint16
        or dt == DType.uint8
        or dt == DType.bool
    )


def _bits_dtype(itemsize: Int) -> DType:
    """The unsigned dtype of this width: a same-dtype copy only moves bits, so
    every dtype of one width shares one compiled variant."""
    if itemsize == 1:
        return DType.uint8
    if itemsize == 2:
        return DType.uint16
    if itemsize == 4:
        return DType.uint32
    return DType.uint64


def _writes_overlap(dsts: List[T], srcs: List[T]) -> Bool:
    """Whether a destination's bytes meet another destination's or any
    source's, the pairs a one-launch copy cannot order like sequential copy_.

    O(n log n) on plain integer sorts. Half-open ranges are pairwise disjoint
    exactly when, with starts and ends sorted independently, every start is
    at or past the previous end; the sorted destinations then pair up in
    order, and a binary search finds the one each source could meet.
    """
    var begins = List[Int](capacity=len(dsts))
    var ends = List[Int](capacity=len(dsts))
    for t in dsts:
        if t.numel > 0:
            begins.append(t.ptr)
            ends.append(t.ptr + t.numel * t.itemsize)
    sort(begins)
    sort(ends)
    for k in range(1, len(begins)):
        if begins[k] < ends[k - 1]:
            return True
    for t in srcs:
        if t.numel == 0:
            continue
        var end = t.ptr + t.numel * t.itemsize
        # The last destination starting before this source ends.
        var lo = 0
        var hi = len(begins)
        while lo < hi:
            var mid = (lo + hi) // 2
            if begins[mid] < end:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and ends[lo - 1] > t.ptr:
            return True
    return False


def _batched_copy_device(device: Int) raises -> Bool:
    """Where the batched rectangle copy runs (CopyBatched).

    TODO: enable on Metal. Tried on an M4 (macOS 26.6): 142 of the 173 copy
    tests in tests/native/test_foreach.py fail. float64 does not build
    (Apple GPUs have none; it must stay excluded), some variants hit "Failed
    to verify LLVM IR for Metal", and the rest return wrong elements. The
    descriptors are an array passed by value, and indexing a copied array of
    pointers is what miscompiles on Metal (see `_cat_pick` in
    tmb/kernels/data_movement/entry.mojo).
    """
    var api = dev(device)[].api
    return api == "cuda" or api == "hip"


def _copy_batch_qualifies(dsts: List[T], srcs: List[T]) raises -> Bool:
    """CUDA's `_foreach_copy_` fast-path rule (one device, contiguous, one
    dtype per list, index-aligned shapes) plus the cross-pair overlap check
    it lacks, so aliased lists keep sequential copy_ semantics."""
    var first = dsts[0].copy()
    if not first.on_mojo() or not _batched_copy_device(first.device):
        return False
    var src_dtype = srcs[0].dtype
    if not _batch_copy_dtype(first.dtype) or not _batch_copy_dtype(src_dtype):
        return False
    for i in range(len(dsts)):
        if not _tensor_qualifies(dsts[i], first.device, first.dtype):
            return False
        if not _tensor_qualifies(srcs[i], first.device, src_dtype):
            return False
        if not dsts[i].same_shape(srcs[i]):
            return False
    # Empty occurrences have no byte-range hazard. The caller still bumps
    # each occurrence, including repeated references to the same empty tensor.
    return not _writes_overlap(dsts, srcs)


def _copy_batch_launch(dsts: List[T], srcs: List[T]) raises:
    var src_dtype = srcs[0].dtype
    var dst_dtype = dsts[0].dtype
    if src_dtype == dst_dtype:
        src_dtype = _bits_dtype(dsts[0].itemsize)
        dst_dtype = src_dtype
    var ctx = ctx_for(dsts[0].device)
    var call = KernelCall("data_movement", "CopyBatched")
    call.arg_dtype(0, src_dtype)
    call.out_dtype(dst_dtype)
    # One-row rectangles: source, destination, rows, cols and both pitches.
    var metadata = List[Int](capacity=6 * len(dsts))
    for i in range(len(dsts)):
        metadata.append(srcs[i].ptr)
        metadata.append(dsts[i].ptr)
        metadata.append(1)
        metadata.append(dsts[i].numel)
        metadata.append(dsts[i].numel)
        metadata.append(dsts[i].numel)
    call.tuple(metadata)
    call.int(ctx_ptr(ctx))
    call.run()
    # This schema has no ADInplaceOrView wrapper, unlike scalar copy_.
    for t in dsts:
        t.bump_version()
    _ = ctx


def _copy_adjacent_views(dsts: List[T], srcs: List[T]) raises -> Bool:
    """Copy matching contiguous partitions of two buffers with one DMA.

    Adjacency is proved from current pointers, shapes and strides. Require
    one storage base on each side so the DMA never crosses allocations.
    Overlapping source/destination spans retain sequential copy_ semantics.
    """
    var first = dsts[0].copy()
    if not first.on_mojo() or dev(first.device)[].api != "cuda":
        return False
    var dst_begin = 0
    var src_begin = 0
    var dst_storage = 0
    var src_storage = 0
    var nbytes = 0
    for i in range(len(dsts)):
        var d = dsts[i].copy()
        var s = srcs[i].copy()
        if (
            not d.on_mojo()
            or not s.on_mojo()
            or d.device != first.device
            or s.device != first.device
            or d.stype != first.stype
            or s.stype != first.stype
            or not d.contig
            or not s.contig
            or not d.same_shape(s)
        ):
            return False
        if d.numel == 0:
            continue
        if nbytes == 0:
            dst_begin, src_begin = d.ptr, s.ptr
            dst_storage, src_storage = d.storage_ptr(), s.storage_ptr()
            if dst_storage == 0 or src_storage == 0:
                return False
        if (
            d.ptr != dst_begin + nbytes
            or s.ptr != src_begin + nbytes
            or d.storage_ptr() != dst_storage
            or s.storage_ptr() != src_storage
        ):
            return False
        nbytes += d.numel * d.itemsize
    if nbytes != 0 and dst_begin != src_begin:
        if dst_begin < src_begin + nbytes and src_begin < dst_begin + nbytes:
            return False
        copy_d2d(ctx_for(first.device), dst_begin, src_begin, nbytes)
    # _foreach_copy_ owns its version bumps (no ADInplaceOrView wrapper).
    # Repeated empty destinations still receive one bump per occurrence.
    for d in dsts:
        d.bump_version()
    return True


def _foreach_copy_dtype_check(dt: DType) raises:
    if (
        dt == DType.float8_e8m0fnu
        or dt == DType.uint16
        or dt == DType.uint32
        or dt == DType.uint64
    ):
        unsupported(
            '"foreach_tensor_copy" not implemented for \''
            + _scalar_type_name(dt)
            + "'"
        )


def _cuda_copy_fast_route(dsts: List[T], srcs: List[T]) -> Bool:
    """The condition of CUDA's `_foreach_copy_` multi_tensor_apply route
    (ForeachBinaryOpList.cu): one device, dense strided tensors, one dtype
    per list, index-aligned sizes and strides."""
    var first = dsts[0].copy()
    if not first.on_mojo():
        return False
    var src_dtype = srcs[0].dtype
    for i in range(len(dsts)):
        ref d = dsts[i]
        ref s = srcs[i]
        if (
            not d.on_mojo()
            or d.device != first.device
            or d.dtype != first.dtype
        ):
            return False
        if not s.on_mojo() or s.device != first.device or s.dtype != src_dtype:
            return False
        # An empty tensor is contiguous, hence non-overlapping and dense.
        if (d.numel > 0 and not is_dense(d.shape, d.strides, d.rank)) or (
            s.numel > 0 and not is_dense(s.shape, s.strides, s.rank)
        ):
            return False
        if not d.same_shape(s):
            return False
        var pad = MAX_RANK - d.rank
        for k in range(d.rank):
            if (
                d.shape[pad + k] != 1
                and d.strides[pad + k] != s.strides[pad + k]
            ):
                return False
    return True


def _copy_e4m3fn_unsaturated(dsts: List[T], srcs: List[T]) raises:
    """CUDA's fast route converts with c10's static_cast: an out-of-range
    value becomes NaN, not copy_'s saturated 448 (sm_89+ SATFINITE)."""
    for i in range(len(dsts)):
        if dsts[i].numel == 0:
            dsts[i].bump_version()  # CUDA's increment_version, empty too
            continue
        if dsts[i].contig:
            cast_into(dsts[i], srcs[i], False)
            dsts[i].bump_version()
            continue
        var tmp = own(cast_to(srcs[i], dsts[i].stype, False))
        var copy_args = List[Value]()
        copy_args.append(_tensor_value(dsts[i]))
        copy_args.append(_tensor_value(tmp.t))
        copy_args.append(bool_arg(False))
        _ = call_op("aten::copy_", "", copy_args^, 1)
        _ = tmp^


# aten::_foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ()
def op_foreach_copy_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dsts = v_tensor_list(args[unsafe_offset=0])
    var srcs = v_tensor_list(args[unsafe_offset=1])
    if len(dsts) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if len(srcs) != len(dsts):
        raise Error("Tensor lists must have the same number of tensors.")
    var f8 = is_float8(dsts[0].dtype) or is_float8(srcs[0].dtype)
    if f8 and _cuda_copy_fast_route(dsts, srcs):
        # CUDA's multi_tensor_apply route dispatches the destination, then
        # the source, over the standard dtypes and four float8s: e8m0fnu and
        # uint16/32/64 refuse there, empty lists included, before writing.
        _foreach_copy_dtype_check(dsts[0].dtype)
        _foreach_copy_dtype_check(srcs[0].dtype)
    if _copy_adjacent_views(dsts, srcs):
        return
    if _copy_batch_qualifies(dsts, srcs):
        _copy_batch_launch(dsts, srcs)
        return
    if (
        dsts[0].dtype == DType.float8_e4m3fn
        and srcs[0].dtype != DType.float8_e4m3fn
        and _cuda_copy_fast_route(dsts, srcs)
    ):
        _copy_e4m3fn_unsaturated(dsts, srcs)
        return
    for i in range(len(dsts)):
        var copy_args = List[Value]()
        copy_args.append(_tensor_value(dsts[i]))
        copy_args.append(_tensor_value(srcs[i]))
        copy_args.append(args[unsafe_offset=2].copy())
        _ = call_op("aten::copy_", "", copy_args^, 1)


# --- reductions: _foreach_norm, _foreach_powsum, _foreach_max ---------------
#
# ForeachReduceOp.cu: one 0-d result per tensor, two launches of the foreach
# family's reduction kernels (tmb/kernels/foreach/reduce.mojo) for a list on
# the fast route, the per-tensor reduction otherwise.


def _reduce_launch(
    tensors: List[T],
    norm: Int,
    root: Bool,
    out_stype: Int32,
    p: Float64 = 0.0,
    one_element_abs: Bool = False,
) raises -> List[T]:
    var outs = List[Owned]()
    var total_chunks = 0
    var metadata = List[Int](capacity=3 * len(tensors))
    for t in tensors:
        outs.append(own(new_scalar(out_stype, t.device)))
        total_chunks += ceildiv(t.numel, FE_REDUCE_CHUNK)
    for i in range(len(tensors)):
        metadata.append(tensors[i].ptr)
        metadata.append(outs[i].t.ptr)
        metadata.append(tensors[i].numel)
    var dtype = tensors[0].dtype
    var acc = ST_FLOAT64 if out_stype == ST_FLOAT64 else ST_FLOAT32
    if norm == FR_MAX:
        acc = ST_UINT8 if dtype == DType.bool else tensors[0].stype
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = max(total_chunks, 1)
    var partials = own(new_tensor(shape, 1, acc, tensors[0].device))
    var ctx = ctx_for(tensors[0].device)
    var call = KernelCall("foreach", "max" if norm == FR_MAX else "norm")
    call.flag("FE_NORM", norm)
    call.flag("FE_ROOT", 1 if root else 0)
    call.arg_dtype(0, dtype)
    call.out_dtype(max_dtype(out_stype))
    call.tuple(metadata)
    call.int(partials.t.ptr)
    call.f64(p)
    call.int(1 if one_element_abs else 0)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = partials  # alive past the launch (its last use above is the pointer read)
    _ = ctx
    var result = List[T]()
    for i in range(len(outs)):
        result.append(outs[i].take())
    return result^


def _ord_value(ord_v: Value, name: String) raises -> Float64:
    """`ord.isIntegral(false) ? ord.to<int64_t>() : ord.to<double>()`."""
    if ord_v.tag == TAG_SCALAR_INT or ord_v.tag == TAG_INT:
        return Float64(v_int(ord_v))
    if ord_v.tag == TAG_SCALAR_DOUBLE or ord_v.tag == TAG_DOUBLE:
        return v_f64(ord_v)
    raise Error(name, " expects ord to be integer or float")


def _check_norm_dtype(dtype_v: Value, self_stype: Int32, name: String) raises:
    """ForeachReduceOp.cu's `check_foreach_norm_dtype`."""
    if dtype_v.tag == TAG_NONE:
        return
    var dt = v_dtype_or(dtype_v, ST_FLOAT32)
    if dt == ST_COMPLEX32 or dt == ST_COMPLEX64 or dt == ST_COMPLEX128:
        raise Error(
            name,
            ": dtype should be real for real inputs, but got ",
            "ComplexHalf" if dt
            == ST_COMPLEX32 else (
                "ComplexFloat" if dt == ST_COMPLEX64 else "ComplexDouble"
            ),
        )
    if not is_float_stype(dt):
        raise Error(
            name,
            ": dtype should be floating point or complex, but got ",
            _scalar_type_name(max_dtype(dt)),
        )
    if promote_types(self_stype, dt) != dt:
        raise Error(
            name,
            ": the dtype of the input (",
            _scalar_type_name(max_dtype(self_stype)),
            (
                ") should be convertible without narrowing to the specified"
                " dtype ("
            ),
            _scalar_type_name(max_dtype(dt)),
            ")",
        )


def _reduce_on_route(tensors: List[T], floating: Bool) raises -> Bool:
    var lists = List[List[T]]()
    lists.append(tensors.copy())
    if not _lists_on_route(lists):
        return False
    var dt = tensors[0].dtype
    var api = dev(tensors[0].device)[].api
    if api == "metal":
        return False
    if floating:
        return (
            dt == DType.float32
            or dt == DType.float16
            or dt == DType.bfloat16
            or dt == DType.float64
        )
    # max: the dtypes the reduction builds (float8 and the wide unsigned
    # types take the per-tensor max).
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
        or dt == DType.bool
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
    )


def _foreach_lp(args: Values, rets: Values, root: Bool) raises:
    """`_foreach_norm` (root) and `_foreach_powsum`."""
    var name = String(
        "_foreach_tensor_norm_cuda" if root else "_foreach_tensor_powsum_cuda"
    )
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var ord_v = args[unsafe_offset=1].copy()
    var dtype_v = args[unsafe_offset=2].copy()
    var p = _ord_value(
        ord_v,
        "foreach_tensor_norm_cuda" if root else "foreach_tensor_powsum_cuda",
    )
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    var inf = Float64.MAX * 2
    if root and p == inf:
        for t in self_list:
            if t.numel == 0:
                raise Error(
                    "_foreach_norm cannot compute the infinity norm on an"
                    " empty tensor because the operation does not have an"
                    " identity"
                )
    # Stock CUDA (torch 2.11) takes ord 0 off its fast route: the slow
    # route's linalg_vector_norm runs it (measured: its dtype errors are
    # linalg.vector_norm's).
    var norm = -1
    if p == 1:
        norm = FR_L1
    elif p == 2:
        norm = FR_L2
    elif p == inf and root:
        norm = FR_LINF
    if norm >= 0 and _reduce_on_route(self_list, True):
        _check_norm_dtype(dtype_v, self_list[0].stype, name)
        var out_stype = v_dtype_or(dtype_v, self_list[0].stype)
        ret_tensor_list(
            rets, 0, _reduce_launch(self_list, norm, root, out_stype)
        )
        return
    if (
        root
        and p == 2
        and _qualifies1(self_list, False)
        and dev(self_list[0].device)[].api == "metal"
        and (
            dtype_v.tag == TAG_NONE
            or v_dtype_or(dtype_v, ST_FLOAT32) == ST_FLOAT32
        )
    ):
        # Apple: the float32 L2 norm's fixed-arity kernels.
        ret_tensor_list(rets, 0, _foreach_norm_launch(self_list))
        return
    var result = List[Owned]()
    for t in self_list:
        result.append(own(_lp_one(t, root, ord_v, p, dtype_v)))
    var out = List[T]()
    for i in range(len(result)):
        out.append(result[i].take())
    ret_tensor_list(rets, 0, out)


def _digits(var n: Int) -> String:
    """The decimal digits of n >= 0."""
    if n == 0:
        return "0"
    var out = String()
    while n > 0:
        out = String(chr(48 + n % 10)) + out
        n //= 10
    return out^


def _format_g(x: Float64) -> String:
    """`std::ostream << double` with the default precision: printf's %g
    with 6 significant digits, how c10::Scalar prints a double."""
    if x != x:
        return "nan"
    if x == Float64.MAX * 2:
        return "inf"
    if x == -Float64.MAX * 2:
        return "-inf"
    if x == 0:
        return "-0" if String(x).startswith("-") else "0"
    var sign = String("-") if x < 0 else String()
    var a = abs(x)
    # The decimal exponent of the value rounded to 6 significant digits.
    var e = 0
    var m = a
    while m >= 10:
        m /= 10
        e += 1
    while m < 1:
        m *= 10
        e -= 1
    var digits = Int(m * 100000 + 0.5)  # 6 significant digits
    if digits >= 1000000:
        digits //= 10
        e += 1
    var text = _digits(digits)  # d.ddddd without the point
    if e < -4 or e >= 6:
        var mant = String(text[byte=0:1])
        var frac = String(text[byte=1:])
        while frac.byte_length() > 0 and frac.endswith("0"):
            var shorter = String(frac[byte = 0 : frac.byte_length() - 1])
            frac = shorter^
        if frac.byte_length() > 0:
            mant += "." + frac
        var ee = abs(e)
        var es = _digits(ee)
        if ee < 10:
            es = "0" + es
        return sign + mant + "e" + ("-" if e < 0 else "+") + es
    var int_part: String
    var frac_part: String
    if e >= 0:
        int_part = String(text[byte = 0 : e + 1])
        frac_part = String(text[byte = e + 1 :])
    else:
        int_part = "0"
        frac_part = "0" * (-e - 1) + text
    while frac_part.byte_length() > 0 and frac_part.endswith("0"):
        var shorter = String(frac_part[byte = 0 : frac_part.byte_length() - 1])
        frac_part = shorter^
    if frac_part.byte_length() > 0:
        return sign + int_part + "." + frac_part
    return sign + int_part


def _ord_text(ord_v: Value, p: Float64) -> String:
    """The order as c10::Scalar prints it in linalg.vector_norm's messages:
    an integer as itself, a double as %g with 6 digits ("-0.333333",
    "-1e+100")."""
    if ord_v.tag == TAG_SCALAR_INT or ord_v.tag == TAG_INT:
        return String(Int(ord_v.a))  # exact, not through the double
    return _format_g(p)


def _reduce_input_ok(dt: DType) -> Bool:
    """The input dtypes the norm reduction builds (bool as uint8)."""
    return (
        dt.is_floating_point()
        and dt != DType.float8_e4m3fn
        and dt != DType.float8_e5m2
        or dt == DType.bool
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
        or dt == DType.uint16
        or dt == DType.uint32
        or dt == DType.uint64
    )


def _lp_one(
    t: T, root: Bool, ord_v: Value, p: Float64, dtype_v: Value
) raises -> T:
    """One tensor of CUDA's slow route: `linalg_vector_norm` (root) or
    `linalg__powsum` of the whole tensor, with their checks and their
    arithmetic (LinearAlgebra.cpp, ReduceNormKernel.cu), measured on stock
    CUDA: the elements converted to the result dtype, then accumulated in
    its opmath (float32 for the half types) and rounded once."""
    var inf = Float64.MAX * 2
    if not t.on_mojo():
        # A list may mix devices (CUDA's fast route then declines): a
        # tensor elsewhere is that device's own per-tensor reduction.
        return _seq_vector_norm(
            String(
                "aten::linalg_vector_norm" if root else "aten::linalg__powsum"
            ),
            t,
            ord_v,
            dtype_v,
        )
    var out_stype: Int32
    if root:
        if not t.dtype.is_floating_point():
            raise Error(
                (
                    "linalg.vector_norm: Expected a floating point or"
                    " complex tensor as input. Got "
                ),
                _scalar_type_name(t.dtype),
            )
        if t.numel == 0 and (p < 0 or p == inf):
            raise Error(
                "linalg.vector_norm cannot compute the ",
                _ord_text(ord_v, p),
                (
                    " norm on an empty tensor because the operation does not"
                    " have an identity"
                ),
            )
        _check_norm_dtype(dtype_v, t.stype, "linalg.vector_norm")
        out_stype = v_dtype_or(dtype_v, t.stype)
    else:
        # get_dtype_from_self(promote_integers=True): `dtype`, else int64
        # for an integer or bool tensor. Only a floating computation dtype
        # has a powsum kernel: an empty tensor is a 0 of the computation
        # dtype (no kernel runs), any other raises.
        out_stype = v_dtype_or(
            dtype_v, t.stype if t.dtype.is_floating_point() else ST_INT64
        )
        # A complex computation dtype computes |x| ** p in its real value
        # dtype (`toRealValueType`, measured on stock CUDA: dtype=complex64
        # gives a float32 result).
        if out_stype == ST_COMPLEX32:
            out_stype = ST_FLOAT16
        elif out_stype == ST_COMPLEX64:
            out_stype = ST_FLOAT32
        elif out_stype == ST_COMPLEX128:
            out_stype = ST_FLOAT64
        if not is_float_stype(out_stype):
            if t.numel == 0 and known_stype(out_stype):
                var zero = new_scalar(out_stype, t.device)
                fill_value(zero, 0.0)
                return zero^
            unsupported(
                '"powsum_cuda" not implemented for \''
                + _scalar_type_name(max_dtype(out_stype))
                + "'"
            )
    if dev(t.device)[].api == "metal" or not _reduce_input_ok(t.dtype):
        var op = String(
            "aten::linalg_vector_norm" if root else "aten::linalg__powsum"
        )
        return _seq_vector_norm(op, t, ord_v, dtype_v)
    # ReduceNormKernel.cu: the norm specializes 0, 1, 2, inf and -inf, the
    # powsum only 2 (its order 1 is NormOps' pow(|x|, 1), which CUDA's
    # double pow does not keep exact: pow(3., 1.) = 2.9999999999999996).
    var norm = FR_LP
    if root and p == 0:
        norm = FR_L0
    elif root and p == 1:
        norm = FR_L1
    elif p == 2:
        norm = FR_L2
    elif root and p == inf:
        norm = FR_LINF
    elif root and p == -inf:
        norm = FR_LNEGINF
    var one = List[T]()
    if t.contig or is_dense(t.shape, t.strides, t.rank):
        one.append(t.copy())
        var r = _reduce_launch(
            one, norm, root, out_stype, p, one_element_abs=root
        )
        return r[0].copy()
    var tmp = own(new_like(t))
    copy_strided_into(tmp.t, t)
    one.append(tmp.t.copy())
    var r = _reduce_launch(one, norm, root, out_stype, p, one_element_abs=root)
    _ = tmp  # alive past the launch
    return r[0].copy()


# aten::_foreach_norm.Scalar(Tensor[] self, Scalar ord=2, ScalarType? dtype=None) -> Tensor[]
def op_foreach_norm_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _foreach_lp(args, rets, True)


# aten::_foreach_powsum.Scalar(Tensor[] self, Scalar ord=2, ScalarType? dtype=None) -> Tensor[]
def op_foreach_powsum_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _foreach_lp(args, rets, False)


# aten::_foreach_max(Tensor[] self) -> Tensor[]
def op_foreach_max(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    for t in self_list:
        if t.numel == 0:
            # Stock CUDA (torch 2.11) takes a list with an empty tensor to
            # the per-tensor max, which refuses it (the v2.14 source has a
            # foreach-specific message instead).
            raise Error(
                "max(): Expected reduction dim to be specified for"
                " input.numel() == 0. Specify the reduction dim with the"
                " 'dim' argument."
            )
    var lists = List[List[T]]()
    lists.append(self_list.copy())
    var dt = self_list[0].dtype
    if (
        dt == DType.uint16 or dt == DType.uint32 or dt == DType.uint64
    ) and _lists_on_route(lists):
        # On CUDA's fast route the kernel's dispatch has no wide unsigned
        # type.
        unsupported(
            '"foreach_tensor_max_cuda_scalar_type" not implemented for \''
            + _scalar_type_name(dt)
            + "'"
        )
    if _reduce_on_route(self_list, False):
        ret_tensor_list(
            rets,
            0,
            _reduce_launch(self_list, FR_MAX, False, self_list[0].stype),
        )
        return
    var result = List[Owned]()
    for t in self_list:
        if t.on_mojo() and (
            t.dtype == DType.uint16
            or t.dtype == DType.uint32
            or t.dtype == DType.uint64
        ):
            # The per-tensor max of CUDA's slow route has no wide unsigned
            # kernel either.
            unsupported(
                '"max_all_cuda" not implemented for \''
                + _scalar_type_name(t.dtype)
                + "'"
            )
        var one = List[Value](capacity=1)
        one.append(_tensor_value(t))
        var r = call_op("aten::max", "", one^, 1)
        result.append(own(r.take_tensor(0)))
    var out = List[T]()
    for i in range(len(result)):
        out.append(result[i].take())
    ret_tensor_list(rets, 0, out)


def _scalar_f32_tensor(
    v: Value, name: StaticString, device: Int, dtype: DType = DType.float32
) raises -> Int:
    """A validated one-element device scalar (`grad_scale`/`found_inf`/...):
    its device pointer, or 0 for None."""
    if v.tag == TAG_NONE:
        return 0
    var t = v_tensor(v)
    if (
        t.device != device
        or not t.on_mojo()
        or t.dtype != dtype
        or t.numel != 1
        or not t.contig
    ):
        raise Error(
            name,
            " must be a contiguous one-element ",
            dtype,
            " tensor on the same mojo device as the other operands",
        )
    return t.ptr


# --- GradScaler (torch/amp/grad_scaler.py) ----------------------------------


def _unscale_dtype_ok(t: T) raises -> Bool:
    """The batched kernel's dtypes; Metal's fixed-arity arm is float32 only."""
    if dev(t.device)[].api == "metal":
        return t.dtype == DType.float32
    return (
        t.dtype == DType.float32
        or t.dtype == DType.float16
        or t.dtype == DType.bfloat16
    )


def _unscale_launch(tensors: List[T], inv_scale: T, found_inf: T) raises:
    """One multi-tensor apply over same-dtype tensors whose elements are
    dense: the math is elementwise, so any stride permutation of a dense
    tensor is the same `numel` elements from its data pointer."""
    var lists = List[List[T]]()
    lists.append(tensors.copy())
    _mta_launch(
        "nonfinite_unscale",
        fe_variant(1, False, FE_ROLE_NONE, False),
        lists,
        List[T](),
        List[Int](),
        inv_scale.ptr,
        found_inf.ptr,
    )
    for t in tensors:
        t.bump_version()


# aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self,
#   Tensor(b!) found_inf, Tensor inv_scale) -> ()
def op_amp_foreach_non_finite_check_and_unscale_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """ATen's CUDA semantics: found_inf = 1 if any element is inf/NaN, and
    every element is multiplied by inv_scale unless it is exactly 1. One
    launch when the list shares a dtype and every tensor is dense, as
    GradScaler's per-(device, dtype) lists do; anything else goes tensor by
    tensor, through a contiguous temporary when the tensor is not dense."""
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var found_inf = v_tensor(args[unsafe_offset=1])
    if len(self_list) == 0:
        return
    if not found_inf.on_mojo():
        unsupported("found_inf is not on a mojo device")
    var device = found_inf.device
    _ = _scalar_f32_tensor(args[unsafe_offset=1], "found_inf", device)
    _ = _scalar_f32_tensor(args[unsafe_offset=2], "inv_scale", device)
    var inv_scale = v_tensor(args[unsafe_offset=2])
    var batched = True
    for t in self_list:
        if not t.on_mojo() or t.device != device:
            raise Error(
                "_amp_foreach_non_finite_check_and_unscale_: every tensor"
                " must be on found_inf's device"
            )
        if not t.dtype.is_floating_point():
            raise Error(
                "_amp_foreach_non_finite_check_and_unscale_ only supports"
                " floating-point tensors"
            )
        if not _unscale_dtype_ok(t):
            unsupported(
                "_amp_foreach_non_finite_check_and_unscale_: dtype "
                + String(t.dtype)
                + " is not supported on this GPU"
            )
        if t.dtype != self_list[0].dtype or not is_dense(
            t.shape, t.strides, t.rank
        ):
            batched = False
    if batched:
        _unscale_launch(self_list, inv_scale, found_inf)
        found_inf.bump_version()
        return
    for t in self_list:
        if t.numel == 0:
            continue
        var one = List[T]()
        if is_dense(t.shape, t.strides, t.rank):
            one.append(t.copy())
            _unscale_launch(one, inv_scale, found_inf)
            continue
        var tmp = own(new_like(t))
        copy_strided_into(tmp.t, t)
        one.append(tmp.t.copy())
        _unscale_launch(one, inv_scale, found_inf)
        copy_strided_into(t, tmp.t)
        t.bump_version()
    found_inf.bump_version()


# aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker,
#   Tensor found_inf, float scale_growth_factor, float scale_backoff_factor,
#   int growth_interval) -> Tensor(a!)
def op_amp_update_scale_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var scale = v_tensor(args[unsafe_offset=0])
    if not scale.on_mojo():
        unsupported("_amp_update_scale_: the scale is not on a mojo device")
    var device = scale.device
    _ = _scalar_f32_tensor(args[unsafe_offset=0], "current_scale", device)
    var tracker_ptr = _scalar_f32_tensor(
        args[unsafe_offset=1], "growth_tracker", device, DType.int32
    )
    var found_inf_ptr = _scalar_f32_tensor(
        args[unsafe_offset=2], "found_inf", device
    )
    var factors = List[Int]()
    factors.append(Int(f64_bits(v_f64(args[unsafe_offset=3]))))
    factors.append(Int(f64_bits(v_f64(args[unsafe_offset=4]))))
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "AmpUpdateScale")
    call.int(scale.ptr)
    call.int(tracker_ptr)
    call.int(found_inf_ptr)
    call.tuple(factors)
    call.int(v_int(args[unsafe_offset=5]))
    call.int(cp)
    call.run()
    v_tensor(args[unsafe_offset=1]).bump_version()
    ret_ref(rets, 0, scale)
    _ = ctx


# --- fused optimizers (torch.optim's fused=True) ----------------------------
#
# `_fused_{adam,adamw,sgd,adagrad}_` have no CompositeExplicitAutograd
# registration (native_functions.yaml), so there is no sequential fallback:
# what the kernel cannot take raises, with the message stock CUDA raises
# (aten/src/ATen/native/cuda/Fused*Kernel.cu). Their autogen functional and
# `.out` variants are CompositeExplicitAutograd over these in-place ones.


# `flags` bits, as tmb/kernels/optimizer/kernels.mojo reads them.
comptime _FO_AMSGRAD = 1
comptime _FO_MAXIMIZE = 2
comptime _FO_NESTEROV = 4
comptime _FO_FIRST_STEP = 8
comptime _FO_MOMENTUM = 16


def _fused_same_sizes_and_strides(p: T, t: T) -> Bool:
    """`_check_tensors_share_sizes_and_strides`: a size-1 dim's stride is
    free."""
    if p.rank != t.rank:
        return False
    for i in range(p.rank):
        if p.dim(i) != t.dim(i):
            return False
        if p.dim(i) != 1 and p.stride(i) != t.stride(i):
            return False
    return True


def _fused_fast_path_ok(
    lists: List[List[T]], skip_cross_list_dtype: Bool
) -> Bool:
    """ATen's `check_fast_path_restrictions` (ForeachUtils.h): one device,
    non-overlapping and dense, one dtype per list (and across lists unless
    `skip_cross_list_dtype`), and every list laid out like the first. Dense
    tensors sharing strides are walked in storage order, as flat arrays."""
    var first = lists[0][0].copy()
    for i in range(len(lists)):
        if len(lists[i]) == 0:
            continue
        var list_dtype = lists[i][0].dtype
        for j in range(len(lists[i])):
            var t = lists[i][j].copy()
            if (
                not t.on_mojo()
                or t.device != first.device
                or not is_dense(t.shape, t.strides, t.rank)
                or t.dtype != list_dtype
                or (not skip_cross_list_dtype and t.dtype != first.dtype)
            ):
                return False
            if i > 0 and not _fused_same_sizes_and_strides(lists[0][j], t):
                return False
    return True


def _fused_check_device(
    t: T, name: StaticString, device: Int, dtype: DType = DType.float32
) raises -> Int:
    """A device scalar the kernel dereferences (grad_scale, found_inf, a
    device lr): stock CUDA's same-device check, then its `data_ptr<float>`
    dtype check. Its first element's address."""
    if not t.on_mojo() or t.device != device:
        raise Error(name, " must be on the same GPU device as the params")
    if t.dtype != dtype:
        raise Error(
            "expected scalar type Float but found ",
            _scalar_type_name(t.dtype),
        )
    if t.numel < 1:
        raise Error(name, " must have at least one element")
    return t.ptr


@always_inline
def _read_as_f64[dt: DType](addr: Int) -> Float64:
    return Pointer[Scalar[dt], MutUntrackedOrigin](
        unsafe_from_address=addr
    )[].cast[DType.float64]()


def _fused_cpu_lr(lr: T) raises -> Float64:
    """`lr.item<double>()` of a CPU tensor lr, whatever its real dtype."""
    if lr.numel != 1:
        raise Error(
            "a Tensor with ",
            lr.numel,
            " elements cannot be converted to Scalar",
        )
    comptime for dt in [
        DType.float32,
        DType.float64,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]:
        if lr.dtype == dt:
            return _read_as_f64[dt](lr.ptr)
    if lr.dtype == DType.bool:
        return 1.0 if _read_as_f64[DType.uint8](lr.ptr) != 0 else 0.0
    unsupported(
        "fused optimizers: a CPU lr tensor of dtype " + String(lr.dtype)
    )
    return 0.0


def _fused_optimizer(
    op: String,
    kernel_name: String,
    mixed_label: String,
    layout_msg: String,
    lists: List[List[T]],
    steps: List[T],
    needs_steps: Bool,
    lr_v: Value,
    var hyper: List[Float64],
    flags: Int,
    grad_scale_v: Value,
    found_inf_v: Value,
    mutated: List[List[T]],
) raises:
    """Validate like stock CUDA, then one batched launch.

    `lists` is what the fast-path check covers: params, grads, then the
    state lists the kernel reads (their order is the kernel's state0..2).
    `hyper[0]` is overwritten with the float lr when `lr_v` is a tensor on
    the CPU. `mutated` is every `Tensor(x!)[]` argument, for the version
    counters.
    """
    var params = lists[0].copy()
    var count = len(params)
    for i in range(1, len(lists)):
        if len(lists[i]) != count:
            raise Error(
                "Tensor lists must have the same number of tensors, got ",
                count,
                " and ",
                len(lists[i]),
            )
    # Adam and Adagrad read one step per tensor: an empty or short
    # state_steps list is a length mismatch like any other.
    if (needs_steps or len(steps) != 0) and len(steps) != count:
        raise Error(
            "Tensor lists must have the same number of tensors, got ",
            count,
            " and ",
            len(steps),
        )
    if count == 0:
        return
    var first = params[0].copy()
    var device = first.device

    # A CPU lr tensor is `lr.item<double>()`: the float-lr overload.
    var lr_ptr = 0
    var lr_is_tensor = lr_v.tag == TAG_TENSOR or lr_v.tag == TAG_TENSOR_REF
    var lr_t = Optional[T](None)
    if lr_is_tensor:
        var t = v_tensor(lr_v)
        if t.on_cpu():
            hyper[0] = _fused_cpu_lr(t)
        else:
            lr_t = t^
    var grad_scale_ptr = 0
    var found_inf_ptr = 0
    if grad_scale_v.tag != TAG_NONE:
        grad_scale_ptr = _fused_check_device(
            v_tensor(grad_scale_v), "grad_scale", device
        )
    if found_inf_v.tag != TAG_NONE:
        found_inf_ptr = _fused_check_device(
            v_tensor(found_inf_v), "found_inf", device
        )
    if lr_t:
        lr_ptr = _fused_check_device(lr_t.value(), "lr", device)

    var mixed = (
        mixed_label != ""
        and len(lists) > 2
        and first.dtype != lists[2][0].dtype
    )
    if not _fused_fast_path_ok(lists, mixed):
        raise Error(layout_msg)
    var state_dtype = first.dtype
    if mixed:
        # validate_mixed_precision_dtypes (fused_adam_utils.cuh): float32
        # params and grads, bfloat16 states.
        var what = List[String]()
        what.append("params")
        what.append("grads")
        what.append("optimizer states")
        what.append("optimizer states")
        what.append("max_exp_avg_sqs")
        for i in range(len(lists)):
            var want = DType.float32 if i < 2 else DType.bfloat16
            var got = lists[i][0].dtype
            if got != want:
                raise Error(
                    mixed_label,
                    " requires ",
                    "float32 " if i < 2 else "bfloat16 ",
                    what[i],
                    ", got ",
                    _scalar_type_name(got),
                )
        state_dtype = DType.bfloat16
    elif (
        first.dtype != DType.float32
        and first.dtype != DType.float16
        and first.dtype != DType.bfloat16
        and first.dtype != DType.float64
    ):
        raise Error(
            '"',
            kernel_name,
            "\" not implemented for '",
            _scalar_type_name(first.dtype),
            "'",
        )
    if first.dtype == DType.float64 and dev(device)[].api == "metal":
        unsupported("fused optimizers: Apple GPUs have no float64")

    for i in range(len(steps)):
        var step = steps[i].copy()
        if (
            not step.on_mojo()
            or step.device != device
            or step.dtype != DType.float32
            or step.numel < 1
        ):
            raise Error(
                (
                    "fused optimizer state_steps must be float32 tensors on the"
                    " params' device (invalid index "
                ),
                i,
                ")",
            )

    var metadata = List[Int](capacity=count * 7)
    for j in range(count):
        metadata.append(lists[0][j].ptr)
        metadata.append(lists[1][j].ptr)
        for s in range(3):
            metadata.append(lists[2 + s][j].ptr if 2 + s < len(lists) else 0)
        metadata.append(steps[j].ptr if len(steps) != 0 else 0)
        metadata.append(lists[0][j].numel)
    var scalars = List[Int](capacity=5)
    for i in range(5):
        scalars.append(Int(f64_bits(hyper[i] if i < len(hyper) else 0.0)))

    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", op)
    call.arg_dtype(0, first.dtype)
    call.arg_dtype(1, state_dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.int(flags)
    call.int(lr_ptr)
    call.int(grad_scale_ptr)
    call.int(found_inf_ptr)
    call.int(cp)
    call.run()
    _ = ctx
    for group in mutated:
        for t in group:
            t.bump_version()


def _fused_adam_family(args: Values, adamw: Bool) raises:
    var amsgrad = v_bool(args[unsafe_offset=11])
    var maximize = v_bool(args[unsafe_offset=12])
    var lists = List[List[T]]()
    for i in range(4):
        lists.append(v_tensor_list(args[unsafe_offset=i]))
    var max_exp_avg_sqs = v_tensor_list(args[unsafe_offset=4])
    if amsgrad:
        lists.append(max_exp_avg_sqs.copy())
    var hyper = List[Float64]()
    var lr_v = args[unsafe_offset=6].copy()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    for i in range(7, 11):
        hyper.append(v_f64(args[unsafe_offset=i]))
    var mutated = lists.copy()
    if not amsgrad:
        mutated.append(max_exp_avg_sqs^)
    _fused_optimizer(
        "FusedAdamW" if adamw else "FusedAdam",
        ("fused_adamw_kernel_cuda" if adamw else "fused_adam_kernel_cuda"),
        (
            "Mixed-precision fused AdamW" if adamw else "Mixed-precision fused Adam"
        ),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout" if amsgrad else (
            "params, grads, exp_avgs, and exp_avg_sqs must have same dtype,"
            " device, and layout"
        ),
        lists,
        v_tensor_list(args[unsafe_offset=5]),
        True,
        lr_v,
        hyper^,
        (_FO_AMSGRAD if amsgrad else 0) | (_FO_MAXIMIZE if maximize else 0),
        args[unsafe_offset=13],
        args[unsafe_offset=14],
        mutated,
    )


# aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs,
#   Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *,
#   float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad,
#   bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_adam_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adam_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _fused_adam_family(args, False)


# aten::_fused_adamw_(...same as _fused_adam_...) -> ()
# aten::_fused_adamw_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adamw_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _fused_adam_family(args, True)


# aten::_fused_sgd_(Tensor(a!)[] self, Tensor(b!)[] grads,
#   Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum,
#   float lr, float dampening, bool nesterov, bool maximize, bool is_first_step,
#   Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_sgd_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_sgd_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lists = List[List[T]]()
    lists.append(v_tensor_list(args[unsafe_offset=0]))
    lists.append(v_tensor_list(args[unsafe_offset=1]))
    var buffers = v_tensor_list(args[unsafe_offset=2])
    var weight_decay = v_f64(args[unsafe_offset=3])
    var momentum = v_f64(args[unsafe_offset=4])
    var lr_v = args[unsafe_offset=5].copy()
    var dampening = v_f64(args[unsafe_offset=6])
    var nesterov = v_bool(args[unsafe_offset=7])
    var maximize = v_bool(args[unsafe_offset=8])
    var is_first_step = v_bool(args[unsafe_offset=9])
    var has_momentum = len(buffers) != 0
    var layout_msg: String
    if has_momentum:
        if not momentum > 0:
            raise Error("Check failed: momentum > 0 (", momentum, " vs. 0). ")
        lists.append(buffers.copy())
        layout_msg = (
            "Expected at::native::check_fast_path_restrictions( {params,"
            " grads, momentum_buffer_list}) to be true, but got false."
        )
    else:
        if momentum != 0:
            raise Error("Check failed: momentum == 0 (", momentum, " vs. 0). ")
        layout_msg = (
            "Expected at::native::check_fast_path_restrictions({params,"
            " grads}) to be true, but got false."
        )
    var hyper = List[Float64]()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    hyper.append(weight_decay)
    hyper.append(momentum)
    hyper.append(dampening)
    hyper.append(0.0)
    var flags = (
        (_FO_MAXIMIZE if maximize else 0)
        | (_FO_NESTEROV if nesterov else 0)
        | (_FO_FIRST_STEP if is_first_step and has_momentum else 0)
        | (_FO_MOMENTUM if has_momentum else 0)
    )
    var mutated = lists.copy()
    if not has_momentum:
        mutated.append(buffers^)
    _fused_optimizer(
        "FusedSgd",
        (
            "fused_sgd_with_momentum_kernel_cuda" if has_momentum else "fused_sgd_kernel_cuda"
        ),
        "",
        layout_msg,
        lists,
        List[T](),
        False,
        lr_v,
        hyper^,
        flags,
        args[unsafe_offset=10],
        args[unsafe_offset=11],
        mutated,
    )


# aten::_fused_adagrad_(Tensor(a!)[] self, Tensor(b!)[] grads,
#   Tensor(c!)[] state_sums, Tensor(d!)[] state_steps, *, float lr,
#   float lr_decay, float weight_decay, float eps, bool maximize,
#   Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_adagrad_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adagrad_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var lists = List[List[T]]()
    for i in range(3):
        lists.append(v_tensor_list(args[unsafe_offset=i]))
    var steps = v_tensor_list(args[unsafe_offset=3])
    var lr_v = args[unsafe_offset=4].copy()
    var hyper = List[Float64]()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    for i in range(5, 8):
        hyper.append(v_f64(args[unsafe_offset=i]))
    hyper.append(0.0)
    var mutated = lists.copy()
    mutated.append(steps.copy())
    _fused_optimizer(
        "FusedAdagrad",
        "fused_adagrad_kernel_cuda",
        "",
        (
            "params, grads, and state_sums must have same dtype, device, and"
            " layout"
        ),
        lists,
        steps,
        True,
        lr_v,
        hyper^,
        _FO_MAXIMIZE if v_bool(args[unsafe_offset=8]) else 0,
        args[unsafe_offset=9],
        args[unsafe_offset=10],
        mutated,
    )


def register_foreach(site: Site) raises:
    impl[
        op_amp_foreach_non_finite_check_and_unscale_,
        "_amp_foreach_non_finite_check_and_unscale_",
    ](site)
    impl[op_amp_update_scale_, "_amp_update_scale_"](site)
    impl[op_foreach_copy_, "_foreach_copy_"](site)
    impl[op_foreach_norm_scalar, "_foreach_norm.Scalar"](site)
    impl[op_foreach_powsum_scalar, "_foreach_powsum.Scalar"](site)
    impl[op_foreach_max, "_foreach_max"](site)
    comptime for row in range(len(FOREACH_OVERLOADS)):
        comptime e = FOREACH_OVERLOADS[row]
        comptime if e.name != "":
            impl[op_foreach[row, False], e.name](site)
        comptime if e.name_ != "":
            impl[op_foreach[row, True], e.name_](site)
    impl[op_fused_adam_, "_fused_adam_"](site)
    impl[op_fused_adam_, "_fused_adam_.tensor_lr"](site)
    impl[op_fused_adamw_, "_fused_adamw_"](site)
    impl[op_fused_adamw_, "_fused_adamw_.tensor_lr"](site)
    impl[op_fused_sgd_, "_fused_sgd_"](site)
    impl[op_fused_sgd_, "_fused_sgd_.tensor_lr"](site)
    impl[op_fused_adagrad_, "_fused_adagrad_"](site)
    impl[op_fused_adagrad_, "_fused_adagrad_.tensor_lr"](site)
