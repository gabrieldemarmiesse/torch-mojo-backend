"""aten ops: unary group (see docs/native_backend.md).

Ported from `eager_kernels/aten_fast.py`'s unary-elementwise suite
(`_unary_spec_op` / `_try_spec_unary`) and `activation_backward_ops` (GELU
backward). Every op here materializes its input contiguous (the elementwise
spec kernels do not scratch-copy strided operands) and dispatches one kernel
call; `.out` variants compute directly into the caller's tensor when it is
already contiguous with the right dtype/shape, else compute into a fresh
temporary and `copy_strided_into` the result — the same rule for every op,
factored into `_unary_out`.

Dtype gates mirror the old Python bridge exactly: the "direct" ops (abs, neg,
sign, relu) accept SPEC_UNARY_DTYPES (float32/float16/bfloat16/float64/
int8/16/32/64/uint8, see `elementwise_ops.SPEC_UNARY_DTYPES`); every
transcendental op accepts only FLOAT_DTYPES (float32/float16/bfloat16, see
`op_utils.FLOAT_DTYPES` — no float64, the kernels comptime-refuse it on GPU);
isnan/logical_not accept the same broad set plus bool (bool read through its
uint8 storage). Anything else declines with `unsupported(...)`, matching the
old NOT_HANDLED convention.
"""
from abi import (
    T,
    Values,
    dtype_code,
    new_like,
    new_tensor,
    own,
    release,
    ret_owned,
    ret_ref,
    torch_dtype,
    unsupported,
    v_f64,
    v_string,
    v_tensor,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from ops_common import (
    contiguous,
    copy_strided_into,
    fill_value,
    resize_out,
)
from registry import Site, impl, op_address_of


# ---------------------------------------------------------------------------
# Dtype gates (mirrors aten_fast.py's _FLOAT_DTYPES / elementwise_ops.mojo's
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


def _require_float(op: String, dt: DType) raises:
    if not _is_float_dtype(dt):
        unsupported(
            op
            + ": dtype "
            + String(dt)
            + " is not supported (float32/float16/bfloat16 only)"
        )


def _require_direct(op: String, dt: DType) raises:
    if not _is_spec_unary_dtype(dt):
        unsupported(op + ": dtype " + String(dt) + " is not supported")


def _require_bool_spec(op: String, dt: DType) raises:
    if not _is_bool_spec_dtype(dt):
        unsupported(op + ": dtype " + String(dt) + " is not supported")


# ---------------------------------------------------------------------------
# Shared spec-kernel plumbing (elementwise_ops family: one TensorSpec in, one
# TensorSpec out — the calling convention `_unary_spec_into_go` /
# `_unary_bool_spec_into_go` read).
# ---------------------------------------------------------------------------


def _dense_enough(t: T) -> Bool:
    """A weak stand-in for `TensorImpl::is_non_overlapping_and_dense`: false
    for a view that repeats elements, which is the case ATen's own overlap
    check calls `TooHard` and declines to judge."""
    for i in range(t.rank):
        if t.stride(i) <= 0 and t.dim(i) > 1:
            return False
    return True


def _no_partial_overlap(written: T, other: T) raises:
    """`at::assert_no_partial_overlap`: an `out=` that shares storage with
    the input without being the same view of it is a read/write race
    (`torch.neg(x[:-1], out=x[1:])`). The identical view is fine -- that is
    how the in-place variants reuse this path. Private to this file until the
    port is merged; ops_binary.mojo has the same helper and both belong in
    ops_common.mojo.
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


def _one_device(a: T, b: T) raises:
    """Both operands of a raw-pointer launch on the same mojo device.

    A kernel gets bare pointers and one stream: a pointer belonging to
    another device -- or to no mojo device at all -- would be dereferenced
    against the wrong context. The fields are cached on `T`, so this costs
    nothing. Private to this file until the port is merged; it belongs in
    ops_common.mojo.
    """
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected every operand on the same mojo device")


def _unary_direct(
    family: String, op: String, src_c: T, dst: T, out_dtype: DType
) raises:
    """dst[...] = f(src_c[...]); src_c must already be contiguous, dst must
    already be the right shape/dtype/contiguity."""
    _one_device(src_c, dst)
    if src_c.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall(family, op)
    call.arg_dtype(0, src_c.dtype)
    call.out_dtype(out_dtype)
    call.spec(src_c.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


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


def _unary_out(
    family: String, op: String, t_in: T, mut dst: T, out_dtype: DType
) raises:
    """The `.out` / in-place route: compute straight into dst when it is
    ready, else compute into a temporary and copy (also correct when dst
    aliases t_in, which is how the in-place ops reuse this).

    An `out=` of the wrong shape is resized first, the way every other `out=`
    op in ATen does (`resize_output`); without that the copy below would face
    a shape it cannot satisfy. A correctly shaped one keeps its own strides
    and storage offset, so `out=base[4:8]` writes where the caller asked.
    """
    _one_device(t_in, dst)
    _no_partial_overlap(dst, t_in)
    if dst.stype != torch_dtype(out_dtype):
        raise Error(
            "expected an out= tensor of dtype ",
            torch_dtype(out_dtype),
            ", got ",
            dst.stype,
        )
    if not dst.same_shape(t_in):
        resize_out(dst, t_in.shape, t_in.rank)
    var src = contiguous(t_in)
    if dst.contig:
        _unary_direct(family, op, src, dst, out_dtype)
    else:
        var out = own(
            new_tensor(src.shape, src.rank, torch_dtype(out_dtype), src.device)
        )
        _unary_direct(family, op, src, out.t, out_dtype)
        copy_strided_into(dst, out.t)
    if src.h != t_in.h:
        release(src.h)


def _float_unary(op: String, t: T) raises -> T:
    _require_float(op, t.dtype)
    return _unary("elementwise_ops", op, t, t.dtype)


def _float_unary_out(op: String, t: T, mut dst: T) raises:
    _require_float(op, t.dtype)
    _unary_out("elementwise_ops", op, t, dst, t.dtype)


def _direct_unary(op: String, t: T) raises -> T:
    _require_direct(op, t.dtype)
    return _unary("elementwise_ops", op, t, t.dtype)


def _direct_unary_out(op: String, t: T, mut dst: T) raises:
    _require_direct(op, t.dtype)
    _unary_out("elementwise_ops", op, t, dst, t.dtype)


def _bool_unary(op: String, t: T) raises -> T:
    _require_bool_spec(op, t.dtype)
    return _unary("elementwise_ops", op, t, DType.bool)


def _bool_unary_out(op: String, t: T, mut dst: T) raises:
    _require_bool_spec(op, t.dtype)
    _unary_out("elementwise_ops", op, t, dst, DType.bool)


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


# aten::sign(Tensor self) -> Tensor
def op_sign(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_direct_unary("SignSpec", t))
    ret_owned(rets, 0, out)


# aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sign_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _direct_unary_out("SignSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# relu (direct dtypes too) + its in-place variant.
# ---------------------------------------------------------------------------


# aten::relu(Tensor self) -> Tensor
def op_relu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_direct_unary("ReluSpec", t))
    ret_owned(rets, 0, out)


# aten::relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_relu_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _direct_unary_out("ReluSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::relu_(Tensor(a!) self) -> Tensor(a!)
def op_relu_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
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
# Transcendental / float-only unary ops.
# ---------------------------------------------------------------------------


# aten::acos(Tensor self) -> Tensor
def op_acos(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("AcosSpec", t))
    ret_owned(rets, 0, out)


# aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_acos_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("AcosSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::asinh(Tensor self) -> Tensor
def op_asinh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("AsinhSpec", t))
    ret_owned(rets, 0, out)


# aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_asinh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("AsinhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::atanh(Tensor self) -> Tensor
def op_atanh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("AtanhSpec", t))
    ret_owned(rets, 0, out)


# aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_atanh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("AtanhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::cos(Tensor self) -> Tensor
def op_cos(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("CosSpec", t))
    ret_owned(rets, 0, out)


# aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_cos_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("CosSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::cosh(Tensor self) -> Tensor
def op_cosh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("CoshSpec", t))
    ret_owned(rets, 0, out)


# aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_cosh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("CoshSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::erf(Tensor self) -> Tensor
def op_erf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("ErfSpec", t))
    ret_owned(rets, 0, out)


# aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_erf_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("ErfSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::exp(Tensor self) -> Tensor
def op_exp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("ExpSpec", t))
    ret_owned(rets, 0, out)


# aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_exp_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("ExpSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::log(Tensor self) -> Tensor
def op_log(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("LogSpec", t))
    ret_owned(rets, 0, out)


# aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("LogSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::log1p(Tensor self) -> Tensor
def op_log1p(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("Log1pSpec", t))
    ret_owned(rets, 0, out)


# aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_log1p_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("Log1pSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::reciprocal(Tensor self) -> Tensor
def op_reciprocal(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("ReciprocalSpec", t))
    ret_owned(rets, 0, out)


# aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_reciprocal_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("ReciprocalSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::rsqrt(Tensor self) -> Tensor
def op_rsqrt(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("RsqrtSpec", t))
    ret_owned(rets, 0, out)


# aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_rsqrt_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("RsqrtSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sigmoid(Tensor self) -> Tensor
def op_sigmoid(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("SigmoidSpec", t))
    ret_owned(rets, 0, out)


# aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sigmoid_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("SigmoidSpec", t, dst)
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
    var out = own(_float_unary("SinSpec", t))
    ret_owned(rets, 0, out)


# aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sin_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("SinSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sinh(Tensor self) -> Tensor
def op_sinh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("SinhSpec", t))
    ret_owned(rets, 0, out)


# aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sinh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("SinhSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::sqrt(Tensor self) -> Tensor
def op_sqrt(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("SqrtSpec", t))
    ret_owned(rets, 0, out)


# aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_sqrt_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("SqrtSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::tan(Tensor self) -> Tensor
def op_tan(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("TanSpec", t))
    ret_owned(rets, 0, out)


# aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_tan_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("TanSpec", t, dst)
    ret_ref(rets, 0, dst)


# aten::tanh(Tensor self) -> Tensor
def op_tanh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var out = own(_float_unary("TanhSpec", t))
    ret_owned(rets, 0, out)


# aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_tanh_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _float_unary_out("TanhSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# ceil / floor: identity (as a fresh copy, functional semantics) on integer
# dtypes, the float-only spec kernel otherwise — matches
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
    return _float_unary(op, t)


def _ceil_or_floor_into(op: String, t: T, mut dst: T) raises:
    if _is_bitwise_dtype(t.dtype) and t.dtype != DType.bool:
        _one_device(t, dst)
        _no_partial_overlap(dst, t)
        if dst.stype != t.stype:
            raise Error(
                "expected an out= tensor of dtype ",
                t.stype,
                ", got ",
                dst.stype,
            )
        if not dst.same_shape(t):
            resize_out(dst, t.shape, t.rank)
        copy_strided_into(dst, t)
        return
    _float_unary_out(op, t, dst)


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
# BF16-contiguous-GPU fast path straight to `activation_forward_ops`
# (GeluForwardBF16) that this port drops — the spec kernel already covers
# that dtype, so it is a performance-only gap, not a correctness one (see
# the report).
#
# Backward has real device kernels only for float32/bfloat16 on GPU
# (activation_backward_ops.GeluBackwardF32/BF16), ported faithfully below
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
    _float_unary_out(spec, t, dst)
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
    if dev(self_t.device)[].is_cpu:
        unsupported("gelu_backward requires an accelerator device")
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
        var call = KernelCall("activation_backward_ops", op_name)
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
    var out = own(_bool_unary("IsNanSpec", t))
    ret_owned(rets, 0, out)


# aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_isnan_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    _bool_unary_out("IsNanSpec", t, dst)
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
    _bool_unary_out("LogicalNotSpec", t, dst)
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# bitwise_not: logic_ops.BitwiseNot (raw-pointer slots, no TensorSpec), bool
# routed to logical_not (~True must be False, not a byte complement) —
# matches fast_aten_bitwise_not exactly.
# ---------------------------------------------------------------------------


def _bitwise_not_kernel(src: T, dst: T) raises:
    _one_device(src, dst)
    if src.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic_ops", "BitwiseNot")
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
    _one_device(t_in, dst)
    _no_partial_overlap(dst, t_in)
    if dst.stype != t_in.stype:
        raise Error(
            "expected an out= tensor of dtype ", t_in.stype, ", got ", dst.stype
        )
    if not dst.same_shape(t_in):
        resize_out(dst, t_in.shape, t_in.rank)
    var src = contiguous(t_in)
    if dst.contig:
        _bitwise_not_kernel(src, dst)
    else:
        var out = own(new_like(src))
        _bitwise_not_kernel(src, out.t)
        copy_strided_into(dst, out.t)
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
# already registered in ops_core.mojo).
# ---------------------------------------------------------------------------


# aten::fill.Scalar(Tensor self, Scalar value) -> Tensor
def op_fill_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var value = v_f64(args[unsafe_offset=1])
    var out = own(new_like(t))
    fill_value(out.t, value)
    ret_owned(rets, 0, out)


def register_unary(site: Site) raises:
    impl[op_abs, "abs"](site)
    impl[op_abs_out, "abs.out"](site)
    impl[op_acos, "acos"](site)
    impl[op_acos_out, "acos.out"](site)
    impl[op_asinh, "asinh"](site)
    impl[op_asinh_out, "asinh.out"](site)
    impl[op_atanh, "atanh"](site)
    impl[op_atanh_out, "atanh.out"](site)
    impl[op_ceil, "ceil"](site)
    impl[op_ceil_out, "ceil.out"](site)
    impl[op_cos, "cos"](site)
    impl[op_cos_out, "cos.out"](site)
    impl[op_cosh, "cosh"](site)
    impl[op_cosh_out, "cosh.out"](site)
    impl[op_erf, "erf"](site)
    impl[op_erf_out, "erf.out"](site)
    impl[op_exp, "exp"](site)
    impl[op_exp_out, "exp.out"](site)
    impl[op_floor, "floor"](site)
    impl[op_floor_out, "floor.out"](site)
    impl[op_log, "log"](site)
    impl[op_log_out, "log.out"](site)
    impl[op_log1p, "log1p"](site)
    impl[op_log1p_out, "log1p.out"](site)
    impl[op_neg, "neg"](site)
    impl[op_neg_out, "neg.out"](site)
    impl[op_reciprocal, "reciprocal"](site)
    impl[op_reciprocal_out, "reciprocal.out"](site)
    impl[op_rsqrt, "rsqrt"](site)
    impl[op_rsqrt_out, "rsqrt.out"](site)
    impl[op_sigmoid, "sigmoid"](site)
    impl[op_sigmoid_out, "sigmoid.out"](site)
    impl[op_sign, "sign"](site)
    impl[op_sign_out, "sign.out"](site)
    impl[op_silu, "silu"](site)
    impl[op_silu_out, "silu.out"](site)
    impl[op_sin, "sin"](site)
    impl[op_sin_out, "sin.out"](site)
    impl[op_sinh, "sinh"](site)
    impl[op_sinh_out, "sinh.out"](site)
    impl[op_sqrt, "sqrt"](site)
    impl[op_sqrt_out, "sqrt.out"](site)
    impl[op_tan, "tan"](site)
    impl[op_tan_out, "tan.out"](site)
    impl[op_tanh, "tanh"](site)
    impl[op_tanh_out, "tanh.out"](site)
    impl[op_relu, "relu"](site)
    impl[op_relu_out, "relu.out"](site)
    impl[op_relu_, "relu_"](site)
    impl[op_gelu, "gelu"](site)
    impl[op_gelu_out, "gelu.out"](site)
    impl[op_gelu_backward, "gelu_backward"](site)
    impl[op_isnan, "isnan"](site)
    impl[op_isnan_out, "isnan.out"](site)
    impl[op_logical_not, "logical_not"](site)
    impl[op_logical_not_out, "logical_not.out"](site)
    impl[op_bitwise_not, "bitwise_not"](site)
    impl[op_bitwise_not_out, "bitwise_not.out"](site)
    impl[op_fill_scalar, "fill.Scalar"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_unary]()
