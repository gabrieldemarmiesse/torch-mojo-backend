"""ATen ops: unary group (see agents_docs/native_backend.md).

Ported from `eager_kernels/aten_fast.py`'s unary-elementwise suite
(`_unary_spec_op` / `_try_spec_unary`) and `activation_backward` (GELU
backward). Every op here materializes its input contiguous (the elementwise
spec kernels do not scratch-copy strided operands) and dispatches one kernel
call; `.out` variants compute directly into the caller's tensor when it is
already contiguous with the right dtype/shape, else compute into a fresh
temporary and `copy_strided_into` the result — the same rule for every op,
factored into `_unary_out`.

Dtype gates mirror the old Python bridge exactly: the "direct" ops (abs, neg,
sign, relu) accept SPEC_UNARY_DTYPES (float32/float16/bfloat16/float64/
int8/16/32/64/uint8, see `elementwise.SPEC_UNARY_DTYPES`); every
transcendental op accepts only FLOAT_DTYPES (float32/float16/bfloat16, see
`op_utils.FLOAT_DTYPES` — no float64, the kernels comptime-refuse it on GPU);
isnan/logical_not accept the same broad set plus bool (bool read through its
uint8 storage). Anything else declines with `unsupported(...)`, matching the
old NOT_HANDLED convention.
"""
from std.ffi import external_call

from tmb.kernels.common.float8 import is_float8
from tmb.backend.abi import (
    Owned,
    ST_FLOAT32,
    ST_FLOAT64,
    T,
    Value,
    Values,
    default_dtype,
    dtype_code,
    dtype_name,
    is_dense,
    max_dtype,
    new_like,
    new_tensor,
    own,
    own_if_new,
    release,
    retain,
    ret_owned,
    ret_ref,
    torch_dtype,
    unsupported,
    v_f64,
    v_f64_or,
    v_int,
    v_is_none,
    v_string,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.ops.common import (
    assert_no_internal_overlap,
    can_cast,
    cast_to,
    contiguous,
    copy_strided_into,
    elementwise_direct,
    fill_value,
    one_device,
    resize_out,
    same_view,
    shares_storage,
)
from tmb.ops.f8_iter import (
    f8_iter,
    generated_out,
    MODE_CAST,
    MODE_FN,
    MODE_INPLACE,
    MODE_NOCHECK,
    MODE_OUTDT,
    MODE_SAME,
    PRE_BOOL_OUT,
    PRE_COMPLEX_NOT_IMPL,
    PRE_COMPLEX_TO_FLOAT,
    PRE_DISPATCH_OUT,
    PRE_DIV_MODE,
    PRE_HEAVISIDE,
    PRE_LDEXP,
    PRE_NAN_TO_NUM,
    PRE_NONE,
    PRE_RELU,
    PRE_SUB,
)
from tmb.backend.registry import Site, impl
from tmb.ops.complex import (
    cx_conj_physical_out,
    cx_fill,
    cx_via_bool,
    routed,
)
from tmb.ops.complex_math import (
    cx_abs,
    cx_abs_out,
    cx_angle,
    cx_angle_out,
    cx_exp,
    cx_exp_out,
    cx_isfinite,
    cx_isinf,
    cx_isinf_out,
    cx_isnan,
    cx_isnan_out,
    cx_log,
    cx_log_out,
    cx_neg,
    cx_neg_out,
    cx_reciprocal,
    cx_reciprocal_out,
    cx_rsqrt,
    cx_rsqrt_out,
    cx_sgn,
    cx_sgn_out,
    cx_sqrt,
    cx_sqrt_out,
)
from tmb.ops.core import cast_for_copy
from tmb.ops.data_movement import _scalar_type_name


# ---------------------------------------------------------------------------
# Dtype gates (mirrors aten_fast.py's _FLOAT_DTYPES / elementwise.mojo's
# SPEC_UNARY_DTYPES — see the module docstring).
# ---------------------------------------------------------------------------


def _is_float_dtype(dt: DType) -> Bool:
    return dt == DType.float32 or dt == DType.float16 or dt == DType.bfloat16


def _is_spec_unary_dtype(dt: DType) -> Bool:
    return (
        _is_float_dtype(dt)
        or dt == DType.float64
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
    )


def _is_bool_spec_dtype(dt: DType) -> Bool:
    return _is_spec_unary_dtype(dt) or dt == DType.bool


def _is_bitwise_dtype(dt: DType) -> Bool:
    return (
        dt == DType.bool
        or dt == DType.uint8
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
    )


def _f8_refuse(op: String, dt: DType) raises:
    """CUDA's unary kernels dispatch no float8: `"<op>_cuda" not implemented
    for '<dtype>'`, the op named from its spec (`ExpSpec` -> exp_cuda)."""
    if is_float8(dt):
        var name = String(op.removesuffix("Spec")).lower()
        unsupported(
            '"'
            + name
            + "_cuda\" not implemented for '"
            + _scalar_type_name(dt)
            + "'"
        )


def _f8_out_refuse(op: String, t: T, mut dst: T, same_dtype: Bool) raises:
    """A float8 `out=` route in TensorIterator's order: the out dtype check
    (a `unary_op` wants the input's own dtype, a `unary_float_op` one the
    result can cast to) leaves `out` untouched; then `out` is resized; then
    the kernel dispatch refuses."""
    if is_float8(t.dtype):
        if same_dtype and dst.stype != t.stype:
            raise Error(
                "Found dtype ",
                _scalar_type_name(dst.dtype),
                " but expected ",
                _scalar_type_name(t.dtype),
            )
        if not same_dtype and not can_cast(t.stype, dst.stype):
            raise Error(
                "result type ",
                _scalar_type_name(t.dtype),
                " can't be cast to the desired output type ",
                _scalar_type_name(dst.dtype),
            )
        if not dst.same_shape(t):
            resize_out(dst, t.shape, t.rank)
        _f8_refuse(op, t.dtype)


def _require_float(op: String, dt: DType) raises:
    _f8_refuse(op, dt)
    if not _is_float_dtype(dt):
        unsupported(
            op
            + ": dtype "
            + String(dt)
            + " is not supported (float32/float16/bfloat16 only)"
        )


def _require_float_or_f64(op: String, t: T) raises:
    """The float dtypes plus float64, which Apple GPUs do not have:
    reciprocal (GradScaler's inverse scale is float64) and ceil / floor
    (torch's tensor printer ceils float64 values)."""
    if t.dtype == DType.float64:
        if dev(t.device)[].api == "metal":
            unsupported(op + " float64 is unavailable on Apple GPUs")
        return
    _require_float(op, t.dtype)


def _require_direct(op: String, dt: DType) raises:
    _f8_refuse(op, dt)
    if not _is_spec_unary_dtype(dt):
        unsupported(op + ": dtype " + String(dt) + " is not supported")


def _require_bool_spec(op: String, dt: DType) raises:
    if not _is_bool_spec_dtype(dt):
        unsupported(op + ": dtype " + String(dt) + " is not supported")


# ---------------------------------------------------------------------------
# Shared spec-kernel plumbing (elementwise family: one TensorSpec in, one
# TensorSpec out — the calling convention `_unary_spec_into_go` /
# `_unary_bool_spec_into_go` read).
# ---------------------------------------------------------------------------


def _dense_enough(t: T) -> Bool:
    """`TensorImpl::is_non_overlapping_and_dense` (see binary.mojo's
    `_b_dense_enough`): any other view is `TooHard` for ATen's check."""
    return t.contig or is_dense(t.shape, t.strides, t.rank)


def _no_partial_overlap(written: T, other: T) raises:
    """`at::assert_no_partial_overlap`: an `out=` that shares storage with
    the input without being the same view of it is a read/write race
    (`torch.neg(x[:-1], out=x[1:])`). The identical view is fine -- that is
    how the in-place variants reuse this path. Private to this file until the
    port is merged; ops_binary.mojo has the same helper and both belong in
    tmb/ops/common.mojo.
    """
    if written.h == other.h or written.numel == 0 or other.numel == 0:
        return
    var storage = written.storage_ptr()
    if storage == 0 or storage != other.storage_ptr():
        return
    if not _dense_enough(written) or not _dense_enough(other):
        return
    var a_begin = written.ptr
    var a_end = a_begin + written.numel * written.itemsize
    var b_begin = other.ptr
    var b_end = b_begin + other.numel * other.itemsize
    if a_begin == b_begin and a_end == b_end:
        if written.rank == other.rank:
            var same = True
            for i in range(written.rank):
                if written.stride(i) != other.stride(i):
                    same = False
            if same:
                return
    elif not (a_begin < b_end and b_begin < a_end):
        return
    raise Error(
        "unsupported operation: some elements of the input tensor and the"
        " written-to tensor refer to a single memory location. Please clone()"
        " the tensor before performing the operation."
    )


def _unary_direct(
    family: String, op: String, src_c: T, mut dst: T, out_dtype: DType
) raises:
    """dst[...] = f(src_c[...]); src_c must already be contiguous, dst must
    already be the right shape/dtype/contiguity. Shared with the vector-norm
    size-one-reduce fast path (`common.elementwise_direct`)."""
    elementwise_direct(family, op, src_c, dst, out_dtype)


def _unary(family: String, op: String, t_in: T, out_dtype: DType) raises -> T:
    """The functional route: a fresh contiguous output."""
    var src = contiguous(t_in)
    var out = own(
        new_tensor(src.shape, src.rank, torch_dtype(out_dtype), src.device)
    )
    _unary_direct(family, op, src, out.t, out_dtype)
    if src.h != t_in.h:
        release(src.h)
    return out.take()


def _out_prologue(t_in: T, mut dst: T, result_stype: Int32) raises:
    """TensorIterator's `out=` meta for a one-input op whose `out` must have
    the result dtype itself: same device, no internal overlap, no partial
    overlap with the input, then the dtype and the resize (a correctly
    shaped `out` keeps its own strides and storage offset)."""
    one_device(t_in, dst)
    assert_no_internal_overlap(dst)
    _no_partial_overlap(dst, t_in)
    if dst.stype != result_stype:
        # TensorIterator's check_dtype wording (stock CUDA, every dtype).
        raise Error(
            "Found dtype ",
            _scalar_type_name(dst.dtype),
            " but expected ",
            _scalar_type_name(max_dtype(result_stype)),
        )
    if not dst.same_shape(t_in):
        resize_out(dst, t_in.shape, t_in.rank)


def _unary_out(
    family: String,
    op: String,
    t_in: T,
    mut dst: T,
    out_dtype: DType,
    cast_ok: Bool = False,
) raises:
    """The `.out` / in-place route: compute straight into dst when it is
    ready, else compute into a temporary and copy (also correct when dst
    aliases t_in, which is how the in-place ops reuse this).

    An `out=` of the wrong shape is resized first, the way every other `out=`
    op in ATen does (`resize_output`); without that the copy below would face
    a shape it cannot satisfy. A correctly shaped one keeps its own strides
    and storage offset, so `out=base[4:8]` writes where the caller asked.

    `cast_ok` is the rule of the `unary_float_op` iterators (asin, exp,
    sqrt, ...): the result is cast into an `out` of any dtype it `canCast`s
    to. Without it (`unary_op`: abs, neg, sign; the bool predicates) the
    `out` must have the result dtype.
    """
    var result_stype = torch_dtype(out_dtype)
    if (
        cast_ok
        and dst.stype != result_stype
        and can_cast(result_stype, dst.stype)
    ):
        one_device(t_in, dst)
        assert_no_internal_overlap(dst)
        _no_partial_overlap(dst, t_in)
        if not dst.same_shape(t_in):
            resize_out(dst, t_in.shape, t_in.rank)
        var result = own(_unary(family, op, t_in, out_dtype))
        var casted = own(cast_for_copy(result.t, dst.stype, False))
        copy_strided_into(dst, casted.t)
        _ = casted^  # alive past the launch
        _ = result^
        return
    if cast_ok and dst.stype != result_stype:
        raise Error(
            "result type ",
            dtype_name(result_stype),
            " can't be cast to the desired output type ",
            dtype_name(dst.stype),
        )
    _out_prologue(t_in, dst, result_stype)
    var src = contiguous(t_in)
    if dst.contig:
        _unary_direct(family, op, src, dst, out_dtype)
    else:
        var out = own(
            new_tensor(src.shape, src.rank, torch_dtype(out_dtype), src.device)
        )
        _unary_direct(family, op, src, out.t, out_dtype)
        copy_strided_into(dst, out.t)
        _ = out^  # alive past the launch
    if src.h != t_in.h:
        release(src.h)


def _float_unary(op: String, t: T) raises -> T:
    _require_float(op, t.dtype)
    return _unary("elementwise", op, t, t.dtype)


def _float_unary_out(op: String, t: T, mut dst: T) raises:
    _f8_out_refuse(op, t, dst, op == "SiluSpec")
    _require_float(op, t.dtype)
    # silu is a `unary_op` structured kernel (the out dtype is self's); the
    # rest are `unary_float_op`s, which cast.
    _unary_out("elementwise", op, t, dst, t.dtype, op != "SiluSpec")


def _direct_unary(op: String, t: T) raises -> T:
    _require_direct(op, t.dtype)
    return _unary("elementwise", op, t, t.dtype)


def _direct_unary_out(op: String, t: T, mut dst: T) raises:
    _f8_out_refuse(op, t, dst, True)
    _require_direct(op, t.dtype)
    _unary_out("elementwise", op, t, dst, t.dtype)


def _bool_unary(op: String, t: T) raises -> T:
    _require_bool_spec(op, t.dtype)
    return _unary("elementwise", op, t, DType.bool)


def _bool_unary_out(op: String, t: T, mut dst: T) raises:
    _require_bool_spec(op, t.dtype)
    _unary_out("elementwise", op, t, dst, DType.bool)


# ---------------------------------------------------------------------------
# abs / neg / sign (direct dtypes: floats, float64, every signed/unsigned int
# up to 64 bits and uint8 — no bool).
# ---------------------------------------------------------------------------


# aten::abs(Tensor self) -> Tensor
def op_abs(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_direct_unary("AbsSpec", t))
    ret_owned(rets, 0, out)


# aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_abs_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _direct_unary_out("AbsSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::neg(Tensor self) -> Tensor
def op_neg(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_direct_unary("NegSpec", t))
    ret_owned(rets, 0, out)


# aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_neg_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _direct_unary_out("NegSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_conj_physical_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """conj_physical_out on a real tensor: UnaryComplexKernels.cu's
    conj_kernel is a copy for every non-complex dtype (the complex dtypes
    never reach a mojo tensor)."""
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    var src = own(T(retain(t)))
    if not dst.same_shape(t) and shares_storage(dst, t):
        # The resize below may reallocate the storage `t` reads from.
        src = own(new_like(t))
        copy_strided_into(src.t, t)
    _out_prologue(t, dst, t.stype)
    if not same_view(dst, src.t):
        copy_strided_into(dst, src.t)
    _ = src^
    ret_ref(rets, 0, dst)


# aten::sign(Tensor self) -> Tensor
def op_sign(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    if t.dtype == DType.bool:
        # UnarySignKernels.cu's sign_kernel_cuda: the identity on bool.
        var out = own(new_like(t))
        copy_strided_into(out.t, t)
        ret_owned(rets, 0, out)
        return
    var out = own(_direct_unary("SignSpec", t))
    ret_owned(rets, 0, out)


# aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sign_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    if t.dtype == DType.bool:
        _out_prologue(t, dst, t.stype)
        if dst.h != t.h:
            copy_strided_into(dst, t)
        ret_ref(rets, 0, dst)
        return
    _direct_unary_out("SignSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# relu (direct dtypes too) + its in-place variant.
# ---------------------------------------------------------------------------


# aten::relu(Tensor self) -> Tensor
def _relu_check(t: T) raises:
    if t.dtype == DType.bool:
        raise Error("Boolean inputs not supported for relu")


def op_relu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _relu_check(t)
    var out = own(_direct_unary("ReluSpec", t))
    ret_owned(rets, 0, out)


# aten::relu_(Tensor(a!) self) -> Tensor(a!)
def op_relu_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _relu_check(t)
    # `_unary_out` with dst == t_in: computes straight into self when self is
    # already contiguous, else materializes and copies back into self's
    # (possibly strided) storage — same as the old `fast_aten_relu` +
    # `_copy_into_tensor(self, result)` two-step. The second handle is a
    # separate `T` because `_unary_out` takes its destination `mut` (it may
    # resize an out= of the wrong shape); self is never that case, so no
    # resize happens here and the two views stay in step.
    var dst = t.copy()
    _direct_unary_out("ReluSpec", t, dst)
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# Transcendental unary ops: `unary_float_op`s, so an integer or bool input
# computes in the default float dtype (`_promote`), float32/float16/bfloat16
# otherwise.
# ---------------------------------------------------------------------------


# aten::acos(Tensor self) -> Tensor
def op_acos(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AcosSpec", t))
    ret_owned(rets, 0, out)


# aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_acos_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AcosSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::acosh(Tensor self) -> Tensor
def op_acosh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AcoshSpec", t))
    ret_owned(rets, 0, out)


# aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_acosh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AcoshSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::asinh(Tensor self) -> Tensor
def op_asinh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AsinhSpec", t))
    ret_owned(rets, 0, out)


# aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_asinh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AsinhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::atanh(Tensor self) -> Tensor
def op_atanh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AtanhSpec", t))
    ret_owned(rets, 0, out)


# aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_atanh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AtanhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::cos(Tensor self) -> Tensor
def op_cos(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("CosSpec", t))
    ret_owned(rets, 0, out)


# aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_cos_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("CosSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::cosh(Tensor self) -> Tensor
def op_cosh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("CoshSpec", t))
    ret_owned(rets, 0, out)


# aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_cosh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("CoshSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::erf(Tensor self) -> Tensor
def op_erf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ErfSpec", t))
    ret_owned(rets, 0, out)


# aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_erf_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ErfSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::exp(Tensor self) -> Tensor
def op_exp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ExpSpec", t))
    ret_owned(rets, 0, out)


# aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_exp_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ExpSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::log(Tensor self) -> Tensor
def op_log(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("LogSpec", t))
    ret_owned(rets, 0, out)


# aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("LogSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::log1p(Tensor self) -> Tensor
def op_log1p(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("Log1pSpec", t))
    ret_owned(rets, 0, out)


# aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log1p_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("Log1pSpec", t, dst)
    ret_ref(rets, 0, dst)


def _log2_input(t: T) raises -> Owned:
    if not t.on_mojo():
        unsupported("log2 requires an input on the mojo device")
    if (
        not _is_float_dtype(t.dtype)
        and t.dtype != DType.float64
        and not _is_bool_spec_dtype(t.dtype)
    ):
        unsupported(
            "log2 supports real floating point, integer, and bool inputs"
        )
    var target = t.stype if t.dtype.is_floating_point() else default_dtype()
    if target == ST_FLOAT64 and dev(t.device)[].api == "metal":
        unsupported("log2 float64 is unavailable on Apple GPUs")
    return own_if_new(cast_to(t, target), t)


# aten::log2(Tensor self) -> Tensor
def op_log2(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var source = _log2_input(t)
    var out = own(_unary("elementwise", "Log2Spec", source.t, source.t.dtype))
    ret_owned(rets, 0, out)
    _ = source^


# aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log2_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    var source = _log2_input(t)
    _unary_out("elementwise", "Log2Spec", source.t, dst, source.t.dtype, True)
    ret_ref(rets, 0, dst)
    _ = source^


def _reciprocal_check(t: T) raises:
    """Floats and float64: GradScaler takes its inverse scale in float64."""
    _require_float_or_f64("reciprocal", t)


# aten::reciprocal(Tensor self) -> Tensor
def op_reciprocal(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var src = own_if_new(_promote(t), t)
    _reciprocal_check(src.t)
    var out = own(_unary("elementwise", "ReciprocalSpec", src.t, src.t.dtype))
    ret_owned(rets, 0, out)
    _ = src^


# aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_reciprocal_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    var src = own_if_new(_promote(t), t)
    _reciprocal_check(src.t)
    _unary_out("elementwise", "ReciprocalSpec", src.t, dst, src.t.dtype, True)
    _ = src^
    ret_ref(rets, 0, dst)


# aten::rsqrt(Tensor self) -> Tensor
def op_rsqrt(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("RsqrtSpec", t))
    ret_owned(rets, 0, out)


# aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_rsqrt_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("RsqrtSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sigmoid(Tensor self) -> Tensor
def op_sigmoid(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SigmoidSpec", t))
    ret_owned(rets, 0, out)


# aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sigmoid_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SigmoidSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::silu(Tensor self) -> Tensor
def op_silu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("SiluSpec", t))
    ret_owned(rets, 0, out)


# aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_silu_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("SiluSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sin(Tensor self) -> Tensor
def op_sin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SinSpec", t))
    ret_owned(rets, 0, out)


# aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sin_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SinSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sinh(Tensor self) -> Tensor
def op_sinh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SinhSpec", t))
    ret_owned(rets, 0, out)


# aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sinh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SinhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sqrt(Tensor self) -> Tensor
def op_sqrt(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SqrtSpec", t))
    ret_owned(rets, 0, out)


# aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sqrt_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SqrtSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::tan(Tensor self) -> Tensor
def op_tan(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("TanSpec", t))
    ret_owned(rets, 0, out)


# aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_tan_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("TanSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::tanh(Tensor self) -> Tensor
def op_tanh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("TanhSpec", t))
    ret_owned(rets, 0, out)


# aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_tanh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("TanhSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# Float-only math and special functions whose kernels are scalar ports of the
# CUDA routines stock torch runs (kernels/common/cuda_math.mojo,
# special_math.mojo). float32/float16/bfloat16 only: halves compute in float,
# and float64 has no port.
# ---------------------------------------------------------------------------


# aten::asin(Tensor self) -> Tensor
def op_asin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AsinSpec", t))
    ret_owned(rets, 0, out)


# aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_asin_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AsinSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::atan(Tensor self) -> Tensor
def op_atan(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AtanSpec", t))
    ret_owned(rets, 0, out)


# aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_atan_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AtanSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::digamma(Tensor self) -> Tensor
def op_digamma(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("DigammaSpec", t))
    ret_owned(rets, 0, out)


# aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_digamma_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("DigammaSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::erfc(Tensor self) -> Tensor
def op_erfc(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ErfcSpec", t))
    ret_owned(rets, 0, out)


# aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_erfc_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ErfcSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::erfinv(Tensor self) -> Tensor
def op_erfinv(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ErfinvSpec", t))
    ret_owned(rets, 0, out)


# aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_erfinv_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ErfinvSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::exp2(Tensor self) -> Tensor
def op_exp2(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("Exp2Spec", t))
    ret_owned(rets, 0, out)


# aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_exp2_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("Exp2Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::expm1(Tensor self) -> Tensor
def op_expm1(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("Expm1Spec", t))
    ret_owned(rets, 0, out)


# aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_expm1_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("Expm1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::i0(Tensor self) -> Tensor
def op_i0(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("I0Spec", t))
    ret_owned(rets, 0, out)


# aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_i0_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("I0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::lgamma(Tensor self) -> Tensor
def op_lgamma(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("LgammaSpec", t))
    ret_owned(rets, 0, out)


# aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_lgamma_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("LgammaSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::log10(Tensor self) -> Tensor
def op_log10(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("Log10Spec", t))
    ret_owned(rets, 0, out)


# aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log10_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("Log10Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sinc(Tensor self) -> Tensor
def op_sinc(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SincSpec", t))
    ret_owned(rets, 0, out)


# aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sinc_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SincSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_airy_ai(Tensor x) -> Tensor
def op_special_airy_ai(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AiryAiSpec", t))
    ret_owned(rets, 0, out)


# aten::special_airy_ai.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
def op_special_airy_ai_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AiryAiSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_bessel_j0(Tensor self) -> Tensor
def op_special_bessel_j0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("BesselJ0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_bessel_j0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_bessel_j0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("BesselJ0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_bessel_j1(Tensor self) -> Tensor
def op_special_bessel_j1(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("BesselJ1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_bessel_j1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_bessel_j1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("BesselJ1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_bessel_y0(Tensor self) -> Tensor
def op_special_bessel_y0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("BesselY0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_bessel_y0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_bessel_y0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("BesselY0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_bessel_y1(Tensor self) -> Tensor
def op_special_bessel_y1(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("BesselY1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_bessel_y1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_bessel_y1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("BesselY1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_entr(Tensor self) -> Tensor
def op_special_entr(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("EntrSpec", t))
    ret_owned(rets, 0, out)


# aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_entr_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("EntrSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_erfcx(Tensor self) -> Tensor
def op_special_erfcx(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ErfcxSpec", t))
    ret_owned(rets, 0, out)


# aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_erfcx_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ErfcxSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_i0e(Tensor self) -> Tensor
def op_special_i0e(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("I0eSpec", t))
    ret_owned(rets, 0, out)


# aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_i0e_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("I0eSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_i1(Tensor self) -> Tensor
def op_special_i1(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("I1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_i1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("I1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_i1e(Tensor self) -> Tensor
def op_special_i1e(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("I1eSpec", t))
    ret_owned(rets, 0, out)


# aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_i1e_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("I1eSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_log_ndtr(Tensor self) -> Tensor
def op_special_log_ndtr(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("LogNdtrSpec", t))
    ret_owned(rets, 0, out)


# aten::special_log_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_log_ndtr_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("LogNdtrSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_modified_bessel_i0(Tensor self) -> Tensor
def op_special_modified_bessel_i0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ModifiedBesselI0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_modified_bessel_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_modified_bessel_i0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ModifiedBesselI0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_modified_bessel_i1(Tensor self) -> Tensor
def op_special_modified_bessel_i1(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ModifiedBesselI1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_modified_bessel_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_modified_bessel_i1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ModifiedBesselI1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_modified_bessel_k0(Tensor self) -> Tensor
def op_special_modified_bessel_k0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ModifiedBesselK0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_modified_bessel_k0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_modified_bessel_k0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ModifiedBesselK0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_modified_bessel_k1(Tensor self) -> Tensor
def op_special_modified_bessel_k1(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ModifiedBesselK1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_modified_bessel_k1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_modified_bessel_k1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ModifiedBesselK1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_ndtri(Tensor self) -> Tensor
def op_special_ndtri(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("NdtriSpec", t))
    ret_owned(rets, 0, out)


# aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_special_ndtri_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("NdtriSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_scaled_modified_bessel_k0(Tensor x) -> Tensor
def op_special_scaled_modified_bessel_k0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ScaledModifiedBesselK0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_scaled_modified_bessel_k0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
def op_special_scaled_modified_bessel_k0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ScaledModifiedBesselK0Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_scaled_modified_bessel_k1(Tensor x) -> Tensor
def op_special_scaled_modified_bessel_k1(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("ScaledModifiedBesselK1Spec", t))
    ret_owned(rets, 0, out)


# aten::special_scaled_modified_bessel_k1.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
def op_special_scaled_modified_bessel_k1_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("ScaledModifiedBesselK1Spec", t, dst)
    ret_ref(rets, 0, dst)


# aten::special_spherical_bessel_j0(Tensor x) -> Tensor
def op_special_spherical_bessel_j0(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("SphericalBesselJ0Spec", t))
    ret_owned(rets, 0, out)


# aten::special_spherical_bessel_j0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
def op_special_spherical_bessel_j0_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("SphericalBesselJ0Spec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# angle / frac (real inputs; float64 too) and trunc / round (the integer
# identity of ceil / floor, float64 too). sgn is sign for a real tensor.
# ---------------------------------------------------------------------------


def _float_or_f64_unary(op: String, t: T) raises -> T:
    _require_float_or_f64(op, t)
    return _unary("elementwise", op, t, t.dtype)


def _float_or_f64_unary_out(
    op: String, t: T, mut dst: T, cast_ok: Bool = False
) raises:
    _require_float_or_f64(op, t)
    _unary_out("elementwise", op, t, dst, t.dtype, cast_ok)


def _promote(t: T) raises -> T:
    """unary_float_op: an integer or bool input computes in the default float
    dtype (a new tensor; the float input itself otherwise)."""
    if _is_bitwise_dtype(t.dtype):
        return cast_to(t, default_dtype())
    return t.copy()


def _promoting_unary(op: String, t: T) raises -> T:
    _f8_refuse(op, t.dtype)
    var src = own_if_new(_promote(t), t)
    if op == "AngleSpec":  # the one op of these that takes float64
        return _float_or_f64_unary(op, src.t)
    return _float_unary(op, src.t)


def _promoting_unary_out(op: String, t: T, mut dst: T) raises:
    # Every op here is a `unary_float_op`: its result casts into `out`.
    _f8_out_refuse(op, t, dst, False)
    var src = own_if_new(_promote(t), t)
    if op == "AngleSpec":
        _float_or_f64_unary_out(op, src.t, dst, True)
    else:
        _require_float(op, src.t.dtype)
        _unary_out("elementwise", op, src.t, dst, src.t.dtype, True)


# aten::angle(Tensor self) -> Tensor
def op_angle(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_promoting_unary("AngleSpec", t))
    ret_owned(rets, 0, out)


# aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_angle_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _promoting_unary_out("AngleSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::frac(Tensor self) -> Tensor
def op_frac(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_or_f64_unary("FracSpec", t))
    ret_owned(rets, 0, out)


# aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_frac_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_or_f64_unary_out("FracSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::round(Tensor self) -> Tensor
def op_round(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_ceil_or_floor("RoundSpec", t))
    ret_owned(rets, 0, out)


# aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_round_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _ceil_or_floor_into("RoundSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sgn(Tensor self) -> Tensor
def op_sgn(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    if t.dtype == DType.bool:  # sgn of a bool is the bool itself
        var copy = own(_int_identity(t))
        ret_owned(rets, 0, copy)
        return
    var out = own(_direct_unary("SignSpec", t))
    ret_owned(rets, 0, out)


# aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sgn_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    if t.dtype == DType.bool:
        _identity_into(t, dst)
        ret_ref(rets, 0, dst)
        return
    _direct_unary_out("SignSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::trunc(Tensor self) -> Tensor
def op_trunc(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_ceil_or_floor("TruncSpec", t))
    ret_owned(rets, 0, out)


# aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_trunc_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _ceil_or_floor_into("TruncSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::signbit(Tensor self) -> Tensor
def op_signbit(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    if t.dtype == DType.bool:  # no bool is negative
        var zeros = own(new_like(t))
        fill_value(zeros.t, 0.0)
        ret_owned(rets, 0, zeros)
        return
    _require_direct("signbit", t.dtype)
    var out = own(_unary("elementwise", "SignbitSpec", t, DType.bool))
    ret_owned(rets, 0, out)


# aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_signbit_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    if t.dtype == DType.bool:
        _out_prologue(t, dst, torch_dtype(DType.bool))
        fill_value(dst, 0.0)
        ret_ref(rets, 0, dst)
        return
    _require_direct("signbit", t.dtype)
    _unary_out("elementwise", "SignbitSpec", t, dst, DType.bool)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# Unary ops with scalar arguments: the elementwise family's parameterized
# route (three float64 slots after the input spec, see
# `unary_math.elementwise_unary_param` for what each op reads).
# ---------------------------------------------------------------------------


def _param_direct(
    op: String, src_c: T, dst: T, p0: Float64, p1: Float64, p2: Float64
) raises:
    one_device(src_c, dst)
    if src_c.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise", op)
    call.arg_dtype(0, src_c.dtype)
    call.out_dtype(src_c.dtype)
    call.spec(src_c.spec(cp))
    call.f64(p0)
    call.f64(p1)
    call.f64(p2)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _param_unary(
    op: String, t_in: T, p0: Float64, p1: Float64, p2: Float64
) raises -> T:
    var src = contiguous(t_in)
    var out = own(new_like(src))
    _param_direct(op, src, out.t, p0, p1, p2)
    if src.h != t_in.h:
        release(src.h)
    return out.take()


def _param_unary_out(
    op: String,
    t_in: T,
    mut dst: T,
    p0: Float64,
    p1: Float64,
    p2: Float64,
    cast_ok: Bool = False,
) raises:
    """`_unary_out` for the parameterized route (also the in-place one: dst
    may be t_in itself). `cast_ok` is `_unary_out`'s: the unary_float_op
    iterators (logit, polygamma) and mvlgamma's `copy_` cast the result into
    an `out` of any dtype it `canCast`s to; round.decimals and nan_to_num
    want the input dtype itself."""
    if cast_ok and dst.stype != t_in.stype:
        if not can_cast(t_in.stype, dst.stype):
            raise Error(
                "result type ",
                dtype_name(t_in.stype),
                " can't be cast to the desired output type ",
                dtype_name(dst.stype),
            )
        one_device(t_in, dst)
        assert_no_internal_overlap(dst)
        _no_partial_overlap(dst, t_in)
        if not dst.same_shape(t_in):
            resize_out(dst, t_in.shape, t_in.rank)
        var result = own(_param_unary(op, t_in, p0, p1, p2))
        var casted = own(cast_for_copy(result.t, dst.stype, False))
        copy_strided_into(dst, casted.t)
        _ = casted^  # alive past the launch
        _ = result^
        return
    _out_prologue(t_in, dst, t_in.stype)
    var src = contiguous(t_in)
    if dst.contig:
        _param_direct(op, src, dst, p0, p1, p2)
    else:
        var out = own(new_like(src))
        _param_direct(op, src, out.t, p0, p1, p2)
        copy_strided_into(dst, out.t)
        _ = out^  # alive past the launch
    if src.h != t_in.h:
        release(src.h)


def _round_decimals_params(
    t: T, decimals: Int
) raises -> Tuple[Float64, Float64, Float64]:
    """round_decimals_kernel_cuda's ten_pow_decimals = pow(10, |decimals|) (in
    double; the kernel launch rounds it to the compute type) and its negative
    flag, then decimals itself (the MPS kernel's exp10(decimals)).
    """
    _require_float_or_f64("round", t)
    var n = decimals if decimals >= 0 else -decimals
    # The C library's pow, the very call torch makes: a loop of `*= 10`
    # drifts by ulps past 10^22 (10^25 came out 9.999999999999999e24, so
    # round(5e24, decimals=-25) gave 1e25 instead of 0).
    var ten_pow = external_call["pow", Float64](Float64(10.0), Float64(n))
    return (
        ten_pow,
        Float64(1.0) if decimals < 0 else Float64(0.0),
        Float64(decimals),
    )


# aten::round.decimals(Tensor self, *, int decimals) -> Tensor
def op_round_decimals(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var decimals = v_int(args[unsafe_offset=1])
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        unsupported("round.decimals: integer tensors are not supported")
    var p = _round_decimals_params(t, decimals)
    var out = own(_param_unary("RoundDecimalsSpec", t, p[0], p[1], p[2]))
    ret_owned(rets, 0, out)


# aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
def op_round_decimals_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var decimals = v_int(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=2])
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        unsupported("round.decimals: integer tensors are not supported")
    var p = _round_decimals_params(t, decimals)
    _param_unary_out("RoundDecimalsSpec", t, dst, p[0], p[1], p[2])
    ret_ref(rets, 0, dst)


# aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)
def op_round__decimals(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var decimals = v_int(args[unsafe_offset=1])
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        unsupported("round.decimals: integer tensors are not supported")
    var p = _round_decimals_params(t, decimals)
    var dst = t.copy()
    _param_unary_out("RoundDecimalsSpec", t, dst, p[0], p[1], p[2])
    ret_ref(rets, 0, t)


def _logit_eps(v: Value) raises -> Float64:
    """`float? eps=None`: None is a negative eps (no clamping)."""
    if v_is_none(v):
        return -1.0
    return v_f64(v)


def _logit_bounds(eps: Float64, dtype: DType) -> Tuple[Float64, Float64]:
    """logit_mps_impl's clamp bounds: lo = eps and hi = 1 - lo, each a graph
    constant of the tensor type. Only the Apple kernel reads them (slots p1 /
    p2); logit_kernel_cuda clamps to the float eps itself."""
    var lo = _round_to_dtype(eps, dtype)
    return (lo, _round_to_dtype(1.0 - lo, dtype))


def _round_to_dtype(v: Float64, dtype: DType) -> Float64:
    if dtype == DType.float16:
        return v.cast[DType.float16]().cast[DType.float64]()
    if dtype == DType.bfloat16:
        return v.cast[DType.bfloat16]().cast[DType.float64]()
    if dtype == DType.float32:
        return v.cast[DType.float32]().cast[DType.float64]()
    return v


# aten::logit(Tensor self, float? eps=None) -> Tensor
def op_logit(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var eps = _logit_eps(args[unsafe_offset=1])
    var src = own_if_new(_promote(t), t)
    _require_float("logit", src.t.dtype)
    var b = _logit_bounds(eps, src.t.dtype)
    var out = own(_param_unary("LogitSpec", src.t, eps, b[0], b[1]))
    ret_owned(rets, 0, out)


# aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
def op_logit_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var eps = _logit_eps(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=2])
    var src = own_if_new(_promote(t), t)
    _require_float("logit", src.t.dtype)
    var b = _logit_bounds(eps, src.t.dtype)
    _param_unary_out("LogitSpec", src.t, dst, eps, b[0], b[1], cast_ok=True)
    ret_ref(rets, 0, dst)


# aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
#
# Unlike every other unary_float_op in this file, logit has no generic
# dispatch fallback for an out-of-tree backend: native_functions.yaml gives
# logit_ its own per-backend `dispatch: CPU, CUDA, XPU: logit_` instead of
# leaving it structured, so there is no CompositeExplicitAutogradNonFunctional
# (or CompositeImplicitAutograd) alias for PrivateUse1 to fall through to --
# confirmed with `torch._C._dispatch_dump("aten::logit_")`, which lists only
# the three named backends. Every other op in this file's in-place family
# (acos_, sin_, sigmoid_, ...) already works on this device without a
# registration of its own, through exactly that alias; logit_ is the one
# exception, so it alone needs `_param_unary_out` wired up directly, the same
# way `op_round__decimals` does for round_.decimals.
def op_logit_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var eps = _logit_eps(args[unsafe_offset=1])
    if _is_bitwise_dtype(t.dtype):
        # CUDA's order: the cast check on self fires before any dtype support
        # check of the promoted result (a float64 default dtype included), so
        # it runs before the promotion casts anything (Apple GPUs would
        # otherwise refuse the float64 cast first).
        raise Error(
            "result type ",
            _scalar_type_name(max_dtype(default_dtype())),
            " can't be cast to the desired output type ",
            _scalar_type_name(t.dtype),
        )
    var src = own_if_new(_promote(t), t)
    _require_float("logit", src.t.dtype)
    var b = _logit_bounds(eps, src.t.dtype)
    var dst = t.copy()
    _param_unary_out("LogitSpec", src.t, dst, eps, b[0], b[1], cast_ok=True)
    _ = src^
    ret_ref(rets, 0, t)


def _polygamma_check(n: Int, t: T) raises:
    if n < 0:
        raise Error("polygamma(n, x) does not support negative n.")
    _require_float("polygamma", t.dtype)


def _polygamma_params(n: Int) -> Tuple[Float64, Float64, Float64]:
    """The order as the kernels take it (`unary_math.polygamma_lane`): its
    int32 truncation in two halves exact in float32 (the slots cross the
    launch in float; the order itself would round past 2^24), then 0 for
    digamma / trigamma (orders 0 and 1), else `float(n)` (Apple's series
    reads the int64 order), negated when n is even."""
    var n32 = Int(Int32(n)) if n >= 2 else n  # C's int conversion: wraps
    var hi = n32 >> 16
    var nf = Float64(Float32(n))  # int64 -> float in one rounding, as Metal
    return (
        Float64(hi),
        Float64(n32 - hi * 65536),
        Float64(0.0) if n < 2 else (nf if n % 2 != 0 else -nf),
    )


# aten::polygamma(int n, Tensor self) -> Tensor
def op_polygamma(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var n = v_int(args[unsafe_offset=0])
    var t = v_tensor(args[unsafe_offset=1])
    var src = own_if_new(_promote(t), t)
    _polygamma_check(n, src.t)
    var p = _polygamma_params(n)
    var out = own(_param_unary("PolygammaSpec", src.t, p[0], p[1], p[2]))
    ret_owned(rets, 0, out)


# aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_polygamma_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var n = v_int(args[unsafe_offset=0])
    var t = v_tensor(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=2])
    var src = own_if_new(_promote(t), t)
    _polygamma_check(n, src.t)
    var p = _polygamma_params(n)
    _param_unary_out(
        "PolygammaSpec", src.t, dst, p[0], p[1], p[2], cast_ok=True
    )
    ret_ref(rets, 0, dst)


def _mvlgamma_constant(p: Int) -> Float64:
    """The `add_(p * (p - 1) * log(pi) / 4)` of UnaryOps.cpp `mvlgamma`, in
    double on the host (the add kernel rounds it to float)."""
    return Float64(p) * Float64(p - 1) * 1.1447298858494002 / 4


def _mvlgamma_no_bool(t: T) raises:
    """UnaryOps.cpp `mvlgamma_check` refuses bool before any promotion."""
    if t.dtype == DType.bool:
        raise Error("The input tensor may not be a boolean tensor.")


def _mvlgamma_check(p: Int, t: T) raises:
    # ATen also rejects an input <= (p - 1) / 2, which needs a device-to-host
    # read of the whole tensor; below that bound this returns lgamma's values.
    if p < 1:
        raise Error("p has to be greater than or equal to 1")
    _require_float("mvlgamma", t.dtype)


# aten::mvlgamma(Tensor self, int p) -> Tensor
def op_mvlgamma(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var p = v_int(args[unsafe_offset=1])
    _mvlgamma_no_bool(t)
    var src = own_if_new(_promote(t), t)
    _mvlgamma_check(p, src.t)
    var out = own(
        _param_unary(
            "MvlgammaSpec", src.t, Float64(p), _mvlgamma_constant(p), 0.0
        )
    )
    ret_owned(rets, 0, out)


# aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
def op_mvlgamma_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var p = v_int(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=2])
    _mvlgamma_no_bool(t)
    one_device(t, dst)
    var src = own_if_new(_promote(t), t)
    _mvlgamma_check(p, src.t)
    # UnaryOps.cpp `mvlgamma_out` is not an iterator: it computes
    # `self.mvlgamma(p)` into a new tensor, then `result.copy_(out)`. So an
    # `out` overlapping the input (x[1:] for x[:-1]) is fine, and any dtype
    # the result can cast to.
    if not can_cast(src.t.stype, dst.stype):
        raise Error(
            "mvlgamma: result type ",
            dtype_name(t.stype),
            " can't be cast to the desired output type ",
            dtype_name(src.t.stype),
        )
    var result = own(
        _param_unary(
            "MvlgammaSpec", src.t, Float64(p), _mvlgamma_constant(p), 0.0
        )
    )
    if not dst.same_shape(result.t):
        resize_out(dst, result.t.shape, result.t.rank)
    assert_no_internal_overlap(dst)
    var casted = own_if_new(cast_for_copy(result.t, dst.stype, False), result.t)
    copy_strided_into(dst, casted.t)
    _ = casted^  # alive past the launch
    _ = result^
    ret_ref(rets, 0, dst)


def _nan_to_num_params(
    t: T, nan: Value, posinf: Value, neginf: Value
) raises -> Tuple[Float64, Float64, Float64]:
    """The replacements, None resolved to 0 / the dtype's max / its lowest
    (all exact in float64, so the kernel's cast back to the dtype is too)."""
    _require_float_or_f64("nan_to_num", t)
    var big = Float64(3.4028234663852886e38)
    if t.dtype == DType.float16:
        big = 65504.0
    elif t.dtype == DType.bfloat16:
        big = 3.3895313892515355e38
    elif t.dtype == DType.float64:
        big = 1.7976931348623157e308
    return (
        v_f64_or(nan, 0.0),
        big if v_is_none(posinf) else v_f64(posinf),
        -big if v_is_none(neginf) else v_f64(neginf),
    )


# aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
def op_nan_to_num(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    if _is_bitwise_dtype(t.dtype):
        # Integers and bool hold no NaN or infinity: a copy.
        var copy = own(_int_identity(t))
        ret_owned(rets, 0, copy)
        return
    var p = _nan_to_num_params(
        t, args[unsafe_offset=1], args[unsafe_offset=2], args[unsafe_offset=3]
    )
    var out = own(_param_unary("NanToNumSpec", t, p[0], p[1], p[2]))
    ret_owned(rets, 0, out)


# aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
def op_nan_to_num_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=4])
    if is_float8(t.dtype):
        # nan_to_num_out's own dtype check, then the resize, then
        # nan_to_num_cuda's dispatch (no float8 case).
        if dst.stype != t.stype:
            raise Error(
                "nan_to_num: dtype of out: ",
                _scalar_type_name(dst.dtype),
                " should be same as input: ",
                _scalar_type_name(t.dtype),
            )
        _f8_out_refuse("nan_to_num", t, dst, True)
    if _is_bitwise_dtype(t.dtype):
        _identity_into(t, dst)
        ret_ref(rets, 0, dst)
        return
    var p = _nan_to_num_params(
        t, args[unsafe_offset=1], args[unsafe_offset=2], args[unsafe_offset=3]
    )
    _param_unary_out("NanToNumSpec", t, dst, p[0], p[1], p[2])
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# ceil / floor: identity (as a fresh copy, functional semantics) on integer
# dtypes, the float spec kernel (float64 included) otherwise — matches
# aten_fast._int_unary_identity.
# ---------------------------------------------------------------------------


def _int_identity(t: T) raises -> T:
    """A fresh tensor holding a copy of t's values (unlike `contiguous`,
    always allocates — ceil/floor are functional even on the identity path).
    """
    var out = new_like(t)
    if t.numel > 0:
        copy_strided_into(out, t)
    return out^


def _ceil_or_floor(op: String, t: T) raises -> T:
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        return _int_identity(t)
    _require_float_or_f64(op, t)
    return _unary("elementwise", op, t, t.dtype)


def _identity_into(t: T, mut dst: T) raises:
    """The `.out` form of `_int_identity`: dst = a copy of t's values."""
    _out_prologue(t, dst, t.stype)
    copy_strided_into(dst, t)


def _ceil_or_floor_into(op: String, t: T, mut dst: T) raises:
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        _identity_into(t, dst)
        return
    _require_float_or_f64(op, t)
    _unary_out("elementwise", op, t, dst, t.dtype)


# aten::ceil(Tensor self) -> Tensor
def op_ceil(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_ceil_or_floor("CeilSpec", t))
    ret_owned(rets, 0, out)


# aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_ceil_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _ceil_or_floor_into("CeilSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::floor(Tensor self) -> Tensor
def op_floor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_ceil_or_floor("FloorSpec", t))
    ret_owned(rets, 0, out)


# aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_floor_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _ceil_or_floor_into("FloorSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# gelu / gelu_backward.
#
# Forward always goes through the generic elementwise spec (GeluNoneSpec /
# GeluTanhSpec, float32/float16/bfloat16): the old bridge additionally had a
# BF16-contiguous-GPU fast path straight to `activation_forward`
# (GeluForwardBF16) that this port drops — the spec kernel already covers
# that dtype, so it is a performance-only gap, not a correctness one (see
# the report).
#
# Backward has real device kernels only for float32/bfloat16 on GPU
# (activation_backward.GeluBackwardF32/BF16), ported faithfully below
# with the same dtype/device/contiguity gates as `fast_aten_gelu_backward`.
# ---------------------------------------------------------------------------


def _gelu_spec(approximate: String) raises -> String:
    if approximate == "none":
        return "GeluNoneSpec"
    if approximate == "tanh":
        return "GeluTanhSpec"
    unsupported("gelu: unknown approximate mode '" + approximate + "'")
    return ""


# aten::gelu(Tensor self, *, str approximate="none") -> Tensor
def op_gelu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var approximate = v_string(args[unsafe_offset=1])
    var spec = _gelu_spec(approximate)
    var out = own(_float_unary(spec, t))
    ret_owned(rets, 0, out)


# aten::gelu.out(Tensor self, *, str approximate="none", Tensor(a!) out) -> Tensor(a!)
def op_gelu_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var approximate = v_string(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=2])
    var spec = _gelu_spec(approximate)
    # gelu is a `unary_op`-style structured kernel: the out dtype is self's
    # (no cast, unlike the `unary_float_op`s).
    _require_float(spec, t.dtype)
    _unary_out("elementwise", spec, t, dst, t.dtype)
    ret_ref(rets, 0, dst)


# aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate="none") -> Tensor
def op_gelu_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var self_t = v_tensor(args[unsafe_offset=1])
    var approximate = v_string(args[unsafe_offset=2])
    if approximate != "none" and approximate != "tanh":
        unsupported(
            "gelu_backward: unknown approximate mode '" + approximate + "'"
        )
    if (
        not grad.on_mojo()
        or not self_t.on_mojo()
        or grad.device != self_t.device
    ):
        unsupported(
            "gelu_backward requires grad_output and self on the same mojo"
            " device"
        )
    if self_t.dtype != DType.float32 and self_t.dtype != DType.bfloat16:
        unsupported(
            "gelu_backward: dtype "
            + String(self_t.dtype)
            + " is not supported (float32/bfloat16 only)"
        )
    if grad.dtype != self_t.dtype:
        unsupported("gelu_backward: grad_output and self must share one dtype")
    if not grad.same_shape(self_t):
        unsupported(
            "gelu_backward: grad_output and self must have the same shape"
        )
    var g = contiguous(grad)
    var s = contiguous(self_t)
    var out = own(new_like(s))
    if s.numel > 0:
        var ctx = ctx_for(s.device)
        var cp = ctx_ptr(ctx)
        var op_name = (
            "GeluBackwardBF16" if s.dtype
            == DType.bfloat16 else "GeluBackwardF32"
        )
        var call = KernelCall("activation_backward", op_name)
        call.arg_dtype(0, g.dtype)
        call.arg_dtype(1, s.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(g.ptr)
        call.int(s.ptr)
        call.int(s.numel)
        call.int(1 if approximate == "tanh" else 0)
        call.int(cp)
        call.run()
        _ = ctx
    if g.h != grad.h:
        release(g.h)
    if s.h != self_t.h:
        release(s.h)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# isnan / logical_not (bool output, broad input dtype incl. bool).
# ---------------------------------------------------------------------------


# aten::isnan(Tensor self) -> Tensor
def op_isnan(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    # float8: ATen's isnan is `self != self`, which CUDA runs for float8;
    # its exact float32 copy has the same NaNs.
    var src = own_if_new(
        cast_to(t, ST_FLOAT32) if is_float8(t.dtype) else t.copy(), t
    )
    var out = own(_bool_unary("IsNanSpec", src.t))
    _ = src^
    ret_owned(rets, 0, out)


# aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_isnan_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    var src = own_if_new(
        cast_to(t, ST_FLOAT32) if is_float8(t.dtype) else t.copy(), t
    )
    _bool_unary_out("IsNanSpec", src.t, dst)
    _ = src^
    ret_ref(rets, 0, dst)


# aten::isinf(Tensor self) -> Tensor
def op_isinf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_inf_predicate("IsInfSpec", t))
    ret_owned(rets, 0, out)


# aten::isinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_isinf_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _inf_predicate_out("IsInfSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::isfinite(Tensor self) -> Tensor
# CompositeImplicitAutograd upstream (abs + eq + ne + mul); its bool output
# is not differentiable, so one kernel here loses no gradient.
def op_isfinite(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_inf_predicate("IsFiniteSpec", t))
    ret_owned(rets, 0, out)


def _inf_predicate(op: String, t: T) raises -> T:
    """isinf / isfinite / isposinf / isneginf: one kernel for a floating
    tensor; any other dtype (every integer width, unsigned ones included,
    and bool) is always finite and never infinite, so a fill without a
    kernel of its own -- as ATen's composites give for any dtype."""
    if t.dtype.is_floating_point():
        return _bool_unary(op, t)
    var out = own(
        new_tensor(t.shape, t.rank, torch_dtype(DType.bool), t.device)
    )
    fill_value(out.t, 1.0 if op == "IsFiniteSpec" else 0.0)
    return out.take()


def _inf_predicate_out(op: String, t: T, mut dst: T) raises:
    if t.dtype.is_floating_point():
        _bool_unary_out(op, t, dst)
        return
    _out_prologue(t, dst, torch_dtype(DType.bool))
    fill_value(dst, 1.0 if op == "IsFiniteSpec" else 0.0)


# aten::isposinf(Tensor self) -> Tensor
def op_isposinf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_inf_predicate("IsPosInfSpec", t))
    ret_owned(rets, 0, out)


# aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_isposinf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _inf_predicate_out("IsPosInfSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::isneginf(Tensor self) -> Tensor
def op_isneginf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_inf_predicate("IsNegInfSpec", t))
    ret_owned(rets, 0, out)


# aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_isneginf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _inf_predicate_out("IsNegInfSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::logical_not(Tensor self) -> Tensor
def op_logical_not(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_bool_unary("LogicalNotSpec", t))
    ret_owned(rets, 0, out)


# aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_logical_not_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    # logical_not_out computes into an `out` of any dtype (0 / 1).
    _require_bool_spec("LogicalNotSpec", t.dtype)
    _unary_out("elementwise", "LogicalNotSpec", t, dst, DType.bool, True)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# bitwise_not: logic.BitwiseNot (raw-pointer slots, no TensorSpec), bool
# routed to logical_not (~True must be False, not a byte complement) —
# matches fast_aten_bitwise_not exactly.
# ---------------------------------------------------------------------------


def _bitwise_not_kernel(src: T, dst: T) raises:
    one_device(src, dst)
    if src.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic", "BitwiseNot")
    call.arg_dtype(0, src.dtype)
    call.int(dst.ptr)
    call.int(src.ptr)
    call.int(src.numel)
    call.int(dtype_code(src.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _bitwise_not(t_in: T) raises -> T:
    if t_in.dtype == DType.bool:
        return _bool_unary("LogicalNotSpec", t_in)
    if not _is_bitwise_dtype(t_in.dtype):
        unsupported(
            "bitwise_not: dtype " + String(t_in.dtype) + " is not supported"
        )
    var src = contiguous(t_in)
    var out = own(new_like(src))
    _bitwise_not_kernel(src, out.t)
    if src.h != t_in.h:
        release(src.h)
    return out.take()


def _bitwise_not_into(t_in: T, mut dst: T) raises:
    if t_in.dtype == DType.bool:
        _bool_unary_out("LogicalNotSpec", t_in, dst)
        return
    if not _is_bitwise_dtype(t_in.dtype):
        unsupported(
            "bitwise_not: dtype " + String(t_in.dtype) + " is not supported"
        )
    _out_prologue(t_in, dst, t_in.stype)
    var src = contiguous(t_in)
    if dst.contig:
        _bitwise_not_kernel(src, dst)
    else:
        var out = own(new_like(src))
        _bitwise_not_kernel(src, out.t)
        copy_strided_into(dst, out.t)
        _ = out^  # alive past the launch
    if src.h != t_in.h:
        release(src.h)


# aten::bitwise_not(Tensor self) -> Tensor
def op_bitwise_not(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_bitwise_not(t))
    ret_owned(rets, 0, out)


# aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_bitwise_not_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _bitwise_not_into(t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# fill.Scalar: the functional fill (fill_.Scalar, the in-place form, is
# already registered in tmb/ops/core.mojo).
# ---------------------------------------------------------------------------


# aten::fill.Scalar(Tensor self, Scalar value) -> Tensor
def op_fill_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var value = v_f64(args[unsafe_offset=1])
    var out = own(new_like(t))
    fill_value(out.t, value)
    ret_owned(rets, 0, out)


comptime _relu_fn = f8_iter[
    op_relu, "clamp_scalar_cuda", 1, -1, MODE_FN, False, PRE_RELU
]


def register_unary(site: Site) raises:
    impl[
        f8_iter[
            routed[op_abs, cx_abs, 1],
            "abs_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "abs",
    ](site)
    impl[
        f8_iter[
            routed[op_abs_out, cx_abs_out, 2],
            "abs_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_COMPLEX_TO_FLOAT,
        ],
        "abs.out",
    ](site)
    impl[
        f8_iter[op_acos, "acos_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "acos",
    ](site)
    impl[
        f8_iter[op_acos_out, "acos_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "acos.out",
    ](site)
    impl[
        f8_iter[op_acosh, "acosh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "acosh",
    ](site)
    impl[
        f8_iter[op_acosh_out, "acosh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "acosh.out",
    ](site)
    impl[
        f8_iter[
            routed[op_angle, cx_angle, 1],
            "angle_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "angle",
    ](site)
    impl[
        f8_iter[
            routed[op_angle_out, cx_angle_out, 2],
            "angle_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_COMPLEX_TO_FLOAT,
        ],
        "angle.out",
    ](site)
    impl[
        f8_iter[op_asin, "asin_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "asin",
    ](site)
    impl[
        f8_iter[op_asin_out, "asin_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "asin.out",
    ](site)
    impl[
        f8_iter[op_asinh, "asinh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "asinh",
    ](site)
    impl[
        f8_iter[op_asinh_out, "asinh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "asinh.out",
    ](site)
    impl[
        f8_iter[op_atan, "atan_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "atan",
    ](site)
    impl[
        f8_iter[op_atan_out, "atan_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "atan.out",
    ](site)
    impl[
        f8_iter[op_atanh, "atanh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "atanh",
    ](site)
    impl[
        f8_iter[op_atanh_out, "atanh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "atanh.out",
    ](site)
    impl[
        f8_iter[
            op_ceil,
            "ceil_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "ceil is not supported for complex inputs",
        ],
        "ceil",
    ](site)
    impl[
        f8_iter[
            op_ceil_out,
            "ceil_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "ceil is not supported for complex inputs",
        ],
        "ceil.out",
    ](site)
    impl[
        f8_iter[op_cos, "cos_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "cos",
    ](site)
    impl[
        f8_iter[op_cos_out, "cos_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "cos.out",
    ](site)
    impl[
        f8_iter[op_cosh, "cosh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "cosh",
    ](site)
    impl[
        f8_iter[op_cosh_out, "cosh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "cosh.out",
    ](site)
    impl[
        f8_iter[op_digamma, "digamma_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "digamma",
    ](site)
    impl[
        f8_iter[
            op_digamma_out, "digamma_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "digamma.out",
    ](site)
    impl[
        f8_iter[op_erf, "erf_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "erf",
    ](site)
    impl[
        f8_iter[op_erf_out, "erf_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "erf.out",
    ](site)
    impl[
        f8_iter[op_erfc, "erfc_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "erfc",
    ](site)
    impl[
        f8_iter[op_erfc_out, "erfc_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "erfc.out",
    ](site)
    impl[
        f8_iter[op_erfinv, "erfinv_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "erfinv",
    ](site)
    impl[
        f8_iter[op_erfinv_out, "erfinv_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "erfinv.out",
    ](site)
    impl[
        f8_iter[
            routed[op_exp, cx_exp, 1],
            "exp_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "exp",
    ](site)
    impl[
        f8_iter[
            routed[op_exp_out, cx_exp_out, 2],
            "exp_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "exp.out",
    ](site)
    impl[
        f8_iter[op_exp2, "exp2_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "exp2",
    ](site)
    impl[
        f8_iter[op_exp2_out, "exp2_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "exp2.out",
    ](site)
    impl[
        f8_iter[op_expm1, "expm1_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "expm1",
    ](site)
    impl[
        f8_iter[op_expm1_out, "expm1_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "expm1.out",
    ](site)
    impl[
        f8_iter[
            op_floor,
            "floor_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "floor is not supported for complex inputs",
        ],
        "floor",
    ](site)
    impl[
        f8_iter[
            op_floor_out,
            "floor_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "floor is not supported for complex inputs",
        ],
        "floor.out",
    ](site)
    impl[
        f8_iter[op_frac, "frac_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "frac",
    ](site)
    impl[
        f8_iter[op_frac_out, "frac_cuda", 1, 1, MODE_SAME, False, PRE_NONE],
        "frac.out",
    ](site)
    impl[
        f8_iter[op_i0, "i0_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "i0",
    ](site)
    impl[
        f8_iter[op_i0_out, "i0_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "i0.out",
    ](site)
    impl[
        f8_iter[op_lgamma, "lgamma_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "lgamma",
    ](site)
    impl[
        f8_iter[op_lgamma_out, "lgamma_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "lgamma.out",
    ](site)
    impl[
        f8_iter[
            routed[op_log, cx_log, 1],
            "log_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "log",
    ](site)
    impl[
        f8_iter[
            routed[op_log_out, cx_log_out, 2],
            "log_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "log.out",
    ](site)
    impl[
        f8_iter[op_log10, "log10_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "log10",
    ](site)
    impl[
        f8_iter[op_log10_out, "log10_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "log10.out",
    ](site)
    impl[
        f8_iter[op_log1p, "log1p_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "log1p",
    ](site)
    impl[
        f8_iter[op_log1p_out, "log1p_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "log1p.out",
    ](site)
    impl[
        f8_iter[op_log2, "log2_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "log2",
    ](site)
    impl[
        f8_iter[op_log2_out, "log2_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "log2.out",
    ](site)
    impl[
        f8_iter[op_logit, "logit_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "logit",
    ](site)
    impl[
        f8_iter[op_logit_out, "logit_cuda", 1, 2, MODE_CAST, False, PRE_NONE],
        "logit.out",
    ](site)
    impl[
        f8_iter[op_logit_, "logit_cuda", 1, 0, MODE_INPLACE, False, PRE_NONE],
        "logit_",
    ](site)
    impl[op_mvlgamma, "mvlgamma"](site)
    impl[op_mvlgamma_out, "mvlgamma.out"](site)
    impl[
        f8_iter[
            op_nan_to_num, "nan_to_num_cuda", 1, -1, MODE_FN, False, PRE_NONE
        ],
        "nan_to_num",
    ](site)
    impl[
        f8_iter[
            op_nan_to_num_out,
            "nan_to_num_cuda",
            1,
            4,
            MODE_SAME,
            False,
            PRE_NAN_TO_NUM,
        ],
        "nan_to_num.out",
    ](site)
    impl[
        f8_iter[
            routed[op_neg, cx_neg, 1],
            "neg_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "neg",
    ](site)
    impl[
        f8_iter[
            routed[op_neg_out, cx_neg_out, 2],
            "neg_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_NONE,
        ],
        "neg.out",
    ](site)
    impl[
        f8_iter[
            routed[op_conj_physical_out, cx_conj_physical_out, 2],
            "conj_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_NONE,
        ],
        "conj_physical.out",
    ](site)
    impl[op_polygamma, "polygamma"](site)
    impl[op_polygamma_out, "polygamma.out"](site)
    impl[
        f8_iter[
            routed[op_reciprocal, cx_reciprocal, 1],
            "reciprocal_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "reciprocal",
    ](site)
    impl[
        f8_iter[
            routed[op_reciprocal_out, cx_reciprocal_out, 2],
            "reciprocal_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "reciprocal.out",
    ](site)
    impl[
        f8_iter[op_round, "round_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "round",
    ](site)
    impl[op_round_decimals, "round.decimals"](site)
    impl[op_round_decimals_out, "round.decimals_out"](site)
    impl[
        f8_iter[op_round_out, "round_cuda", 1, 1, MODE_SAME, False, PRE_NONE],
        "round.out",
    ](site)
    impl[op_round__decimals, "round_.decimals"](site)
    impl[
        f8_iter[
            routed[op_rsqrt, cx_rsqrt, 1],
            "rsqrt_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "rsqrt",
    ](site)
    impl[
        f8_iter[
            routed[op_rsqrt_out, cx_rsqrt_out, 2],
            "rsqrt_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "rsqrt.out",
    ](site)
    impl[
        f8_iter[
            routed[op_sgn, cx_sgn, 1],
            "sign_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "sgn",
    ](site)
    impl[
        f8_iter[
            routed[op_sgn_out, cx_sgn_out, 2],
            "sign_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_NONE,
        ],
        "sgn.out",
    ](site)
    impl[
        f8_iter[op_sigmoid, "sigmoid_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "sigmoid",
    ](site)
    impl[
        f8_iter[
            op_sigmoid_out, "sigmoid_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "sigmoid.out",
    ](site)
    impl[
        f8_iter[
            op_sign,
            "sign_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_COMPLEX_NOT_IMPL,
            (
                "Unlike NumPy, torch.sign is not intended to support complex"
                " numbers. Please use torch.sgn instead."
            ),
        ],
        "sign",
    ](site)
    impl[
        f8_iter[
            op_sign_out,
            "sign_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_COMPLEX_NOT_IMPL,
            (
                "Unlike NumPy, torch.sign is not intended to support complex"
                " numbers. Please use torch.sgn instead."
            ),
        ],
        "sign.out",
    ](site)
    impl[
        f8_iter[
            op_signbit,
            "signbit_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_BOOL_OUT,
            "signbit",
        ],
        "signbit",
    ](site)
    impl[
        f8_iter[
            op_signbit_out,
            "signbit_cuda",
            1,
            1,
            MODE_NOCHECK,
            False,
            PRE_BOOL_OUT,
            "signbit",
        ],
        "signbit.out",
    ](site)
    impl[
        f8_iter[op_silu, "silu_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "silu",
    ](site)
    impl[
        f8_iter[op_silu_out, "silu_cuda", 1, 1, MODE_SAME, False, PRE_NONE],
        "silu.out",
    ](site)
    impl[
        f8_iter[op_sin, "sin_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "sin",
    ](site)
    impl[
        f8_iter[op_sin_out, "sin_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "sin.out",
    ](site)
    impl[
        f8_iter[op_sinc, "sinc_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "sinc",
    ](site)
    impl[
        f8_iter[op_sinc_out, "sinc_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "sinc.out",
    ](site)
    impl[
        f8_iter[op_sinh, "sinh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "sinh",
    ](site)
    impl[
        f8_iter[op_sinh_out, "sinh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "sinh.out",
    ](site)
    impl[
        f8_iter[
            op_special_airy_ai, "airy_ai_cuda", 1, -1, MODE_FN, False, PRE_NONE
        ],
        "special_airy_ai",
    ](site)
    impl[
        f8_iter[
            op_special_airy_ai_out,
            "airy_ai_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_airy_ai.out",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_j0,
            "bessel_j0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_bessel_j0",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_j0_out,
            "bessel_j0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_bessel_j0.out",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_j1,
            "bessel_j1_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_bessel_j1",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_j1_out,
            "bessel_j1_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_bessel_j1.out",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_y0,
            "bessel_y0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_bessel_y0",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_y0_out,
            "bessel_y0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_bessel_y0.out",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_y1,
            "bessel_y1_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_bessel_y1",
    ](site)
    impl[
        f8_iter[
            op_special_bessel_y1_out,
            "bessel_y1_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_bessel_y1.out",
    ](site)
    impl[
        f8_iter[op_special_entr, "entr_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "special_entr",
    ](site)
    impl[
        f8_iter[
            op_special_entr_out, "entr_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "special_entr.out",
    ](site)
    impl[
        f8_iter[
            op_special_erfcx, "erfcx_cuda", 1, -1, MODE_FN, False, PRE_NONE
        ],
        "special_erfcx",
    ](site)
    impl[
        f8_iter[
            op_special_erfcx_out, "erfcx_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "special_erfcx.out",
    ](site)
    impl[
        f8_iter[op_special_i0e, "i0e_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "special_i0e",
    ](site)
    impl[
        f8_iter[
            op_special_i0e_out, "i0e_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "special_i0e.out",
    ](site)
    impl[
        f8_iter[op_special_i1, "i1_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "special_i1",
    ](site)
    impl[
        f8_iter[op_special_i1_out, "i1_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "special_i1.out",
    ](site)
    impl[
        f8_iter[op_special_i1e, "i1e_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "special_i1e",
    ](site)
    impl[
        f8_iter[
            op_special_i1e_out, "i1e_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "special_i1e.out",
    ](site)
    impl[
        f8_iter[
            op_special_log_ndtr,
            "log_ndtr_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_log_ndtr",
    ](site)
    impl[
        f8_iter[
            op_special_log_ndtr_out,
            "log_ndtr_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_log_ndtr.out",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_i0,
            "modified_bessel_i0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_i0",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_i0_out,
            "modified_bessel_i0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_i0.out",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_i1,
            "modified_bessel_i1_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_i1",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_i1_out,
            "modified_bessel_i1_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_i1.out",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_k0,
            "modified_bessel_k0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_k0",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_k0_out,
            "modified_bessel_k0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_k0.out",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_k1,
            "modified_bessel_k1_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_k1",
    ](site)
    impl[
        f8_iter[
            op_special_modified_bessel_k1_out,
            "modified_bessel_k1_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_modified_bessel_k1.out",
    ](site)
    impl[
        f8_iter[
            op_special_ndtri, "ndtri_cuda", 1, -1, MODE_FN, False, PRE_NONE
        ],
        "special_ndtri",
    ](site)
    impl[
        f8_iter[
            op_special_ndtri_out, "ndtri_cuda", 1, 1, MODE_CAST, False, PRE_NONE
        ],
        "special_ndtri.out",
    ](site)
    impl[
        f8_iter[
            op_special_scaled_modified_bessel_k0,
            "scaled_modified_bessel_k0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_scaled_modified_bessel_k0",
    ](site)
    impl[
        f8_iter[
            op_special_scaled_modified_bessel_k0_out,
            "scaled_modified_bessel_k0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_scaled_modified_bessel_k0.out",
    ](site)
    impl[
        f8_iter[
            op_special_scaled_modified_bessel_k1,
            "scaled_modified_bessel_k1_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_scaled_modified_bessel_k1",
    ](site)
    impl[
        f8_iter[
            op_special_scaled_modified_bessel_k1_out,
            "scaled_modified_bessel_k1_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_scaled_modified_bessel_k1.out",
    ](site)
    impl[
        f8_iter[
            op_special_spherical_bessel_j0,
            "spherical_bessel_j0_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "special_spherical_bessel_j0",
    ](site)
    impl[
        f8_iter[
            op_special_spherical_bessel_j0_out,
            "spherical_bessel_j0_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "special_spherical_bessel_j0.out",
    ](site)
    impl[
        f8_iter[
            routed[op_sqrt, cx_sqrt, 1],
            "sqrt_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "sqrt",
    ](site)
    impl[
        f8_iter[
            routed[op_sqrt_out, cx_sqrt_out, 2],
            "sqrt_cuda",
            1,
            1,
            MODE_CAST,
            False,
            PRE_NONE,
        ],
        "sqrt.out",
    ](site)
    impl[
        f8_iter[op_tan, "tan_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "tan",
    ](site)
    impl[
        f8_iter[op_tan_out, "tan_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "tan.out",
    ](site)
    impl[
        f8_iter[op_tanh, "tanh_cuda", 1, -1, MODE_FN, False, PRE_NONE],
        "tanh",
    ](site)
    impl[
        f8_iter[op_tanh_out, "tanh_cuda", 1, 1, MODE_CAST, False, PRE_NONE],
        "tanh.out",
    ](site)
    impl[
        f8_iter[
            op_trunc_out,
            "trunc_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "trunc is not supported for complex inputs",
        ],
        "trunc.out",
    ](site)
    impl[
        f8_iter[
            op_trunc,
            "trunc_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_COMPLEX_NOT_IMPL,
            "trunc is not supported for complex inputs",
        ],
        "trunc",
    ](site)
    impl[_relu_fn, "relu"](site)
    # relu.out is a generated wrapper (relu has no out kernel): the
    # functional result, then resize, copy_arg's checks, the copy
    impl[generated_out[_relu_fn, 1, 1], "relu.out"](site)
    impl[
        f8_iter[
            op_relu_, "clamp_scalar_cuda", 1, 0, MODE_INPLACE, False, PRE_RELU
        ],
        "relu_",
    ](site)
    impl[
        f8_iter[op_gelu, "GeluCUDAKernelImpl", 1, -1, MODE_FN, False, PRE_NONE],
        "gelu",
    ](site)
    impl[
        f8_iter[
            op_gelu_out, "GeluCUDAKernelImpl", 1, 2, MODE_SAME, False, PRE_NONE
        ],
        "gelu.out",
    ](site)
    impl[routed[op_isnan, cx_isnan, 1], "isnan"](site)
    impl[routed[op_isnan_out, cx_isnan_out, 2], "isnan.out"](site)
    impl[
        f8_iter[
            routed[op_isinf, cx_isinf, 1],
            "isinf",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "isinf",
    ](site)
    impl[
        f8_iter[
            routed[op_isinf_out, cx_isinf_out, 2],
            "isinf",
            1,
            1,
            MODE_NOCHECK,
            False,
            PRE_NONE,
        ],
        "isinf.out",
    ](site)
    impl[
        f8_iter[
            routed[op_isfinite, cx_isfinite, 1],
            "isfinite",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "isfinite",
    ](site)
    impl[
        f8_iter[
            op_isposinf,
            "isposinf_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_BOOL_OUT,
            "isposinf",
        ],
        "isposinf",
    ](site)
    impl[
        f8_iter[
            op_isposinf_out,
            "isposinf_cuda",
            1,
            1,
            MODE_NOCHECK,
            False,
            PRE_BOOL_OUT,
            "isposinf",
        ],
        "isposinf.out",
    ](site)
    impl[
        f8_iter[
            op_isneginf,
            "isneginf_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_BOOL_OUT,
            "isneginf",
        ],
        "isneginf",
    ](site)
    impl[
        f8_iter[
            op_isneginf_out,
            "isneginf_cuda",
            1,
            1,
            MODE_NOCHECK,
            False,
            PRE_BOOL_OUT,
            "isneginf",
        ],
        "isneginf.out",
    ](site)
    impl[
        f8_iter[
            routed[op_logical_not, cx_via_bool["logical_not", "", 1], 1],
            "logical_not_cuda",
            1,
            -1,
            MODE_FN,
            False,
            PRE_NONE,
        ],
        "logical_not",
    ](site)
    impl[
        f8_iter[
            routed[op_logical_not_out, cx_via_bool["logical_not", "out", 1], 2],
            "logical_not_cuda",
            1,
            1,
            MODE_NOCHECK,
            False,
            PRE_DISPATCH_OUT,
        ],
        "logical_not.out",
    ](site)
    impl[
        f8_iter[
            op_bitwise_not, "bitwise_not_cuda", 1, -1, MODE_FN, False, PRE_NONE
        ],
        "bitwise_not",
    ](site)
    impl[
        f8_iter[
            op_bitwise_not_out,
            "bitwise_not_cuda",
            1,
            1,
            MODE_SAME,
            False,
            PRE_NONE,
        ],
        "bitwise_not.out",
    ](site)
    impl[routed[op_fill_scalar, cx_fill, 1], "fill.Scalar"](site)
