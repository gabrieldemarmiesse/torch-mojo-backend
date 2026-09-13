"""aten ops: compare group (see docs/native_backend.md).

Ports the old eager path's comparisons / isin / where / masked_fill /
searchsorted / bucketize (`eager_kernels/aten_fast.py`,
`mojo_device/aten_ops/inplace.py`) onto three kernel families:
`logic_ops` (the comparison specs, IsIn), `data_movement_ops` (WhereSelect,
MaskedFillScalar) and `searchsorted_ops` (Searchsorted). The comparison specs
follow the generic TensorSpec convention (arbitrary rank/strides, like
AddSpec/MulSpec in ops_binary.mojo); WhereSelect/MaskedFillScalar/IsIn
predate that convention and take raw pointers plus fixed-rank-4 dim/stride
tuples, so operands there are capped at rank 4 (declined above that, matching
the old `_bcast_meta`'s `rank > 4: return None`).
"""
from std.utils import IndexList

from abi import (
    Owned,
    T,
    Value,
    Values,
    ST_BOOL,
    default_dtype,
    dtype_code,
    max_dtype,
    new_like,
    new_scalar,
    new_tensor,
    own,
    release,
    ret_owned,
    ret_ref,
    torch_dtype,
    unsupported,
    v_bool_or,
    v_is_none,
    v_opt_tensor,
    v_scalar_is_integral,
    v_string,
    v_tensor,
)
from device import ctx_for, ctx_ptr
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import (
    binary_promotion,
    cast_to,
    contiguous,
    copy_strided_into,
    fill_value,
    promoted_pair,
    resize_out,
    scalar_embed,
)
from registry import Site, impl, op_address_of


def _release_if_new(t: T, orig: T):
    """Release `t` if it is a fresh allocation distinct from `orig` (the
    original, caller-owned operand) -- the standard idiom this backend uses
    for a helper's "input itself, or a materialized copy" return."""
    if t.h != orig.h:
        release(t.h)


def _shape_eq(t: T, shape: IndexList[MAX_RANK]) -> Bool:
    """Whether t's padded shape exactly equals `shape` (both MAX_RANK-wide,
    leading-padded with 1s): the eligibility check for writing an out=
    kernel result directly into `t`."""
    for i in range(MAX_RANK):
        if t.shape[i] != shape[i]:
            return False
    return True


def _prepare_out(
    mut out_arg: T,
    shape: IndexList[MAX_RANK],
    rank: Int,
    stype: Int32,
    device: Int,
) raises -> Bool:
    """Get `out_arg` ready to receive a `shape`/`stype` result: raises if
    its dtype doesn't match (fixed per op, never promoted -- matches torch's
    own strict out= dtype check), resizes it in place if its shape doesn't
    (`resize_out`: no `aten::resize_` kernel exists to do this for us), and
    reports whether the (now correctly shaped) tensor is contiguous.

    False means the caller must compute into a temporary and
    `copy_strided_into` the result instead of writing directly into
    `out_arg`; that only happens when `out_arg` already had the right shape
    but a non-contiguous layout, since a resize always leaves it contiguous.
    """
    if not out_arg.on_mojo() or out_arg.device != device:
        raise Error("expected the out= tensor on the operands' mojo device")
    if out_arg.stype != stype:
        raise Error(
            "expected an out= tensor of dtype ", stype, ", got ", out_arg.stype
        )
    if not _shape_eq(out_arg, shape):
        resize_out(out_arg, shape, rank)
        return True
    return out_arg.contig


def _ensure_out_shape(
    mut out_arg: T,
    shape: IndexList[MAX_RANK],
    rank: Int,
    stype: Int32,
    device: Int,
) raises:
    """Like `_prepare_out`, for callers that always compute into a
    temporary and copy (searchsorted/bucketize): only the dtype/shape
    checks matter, not contiguity."""
    if not out_arg.on_mojo() or out_arg.device != device:
        raise Error("expected the out= tensor on the operands' mojo device")
    if out_arg.stype != stype:
        raise Error(
            "expected an out= tensor of dtype ", stype, ", got ", out_arg.stype
        )
    if not _shape_eq(out_arg, shape):
        resize_out(out_arg, shape, rank)


# ---------------------------------------------------------------------------
# eq / ne / lt / le / gt / ge: broadcast-strided comparison specs
# (logic_ops: EqSpec/NeSpec/LtSpec/LeSpec/GtSpec/GeSpec), same TensorSpec
# convention and dtype-promotion rule as add/mul in ops_binary.mojo, but the
# output dtype is always bool. Ported from `fast_aten_eq` et al., which all
# go through `_try_spec_binary(..., out_dtype=bool)`.
# ---------------------------------------------------------------------------


def _cmp_broadcast_shape(a: T, b: T) raises -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        var x = a.shape[i]
        var y = b.shape[i]
        if x == y or y == 1:
            shape[i] = x
        elif x == 1:
            shape[i] = y
        else:
            raise Error("shapes are not broadcastable")
    return shape


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


def _compare_spec(op: StaticString, a: T, b: T, dst: T) raises:
    _one_device(a, dst)
    _one_device(b, dst)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    call.out_dtype(dst.dtype)
    call.spec(a.spec(cp))
    call.spec(b.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _compare_functional(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected both operands on the same mojo device")
    var dtype = binary_promotion(a.dtype, b.dtype)
    var stype = torch_dtype(dtype)
    var pa = cast_to(a, stype)
    var pb = cast_to(b, stype)
    var shape = _cmp_broadcast_shape(pa, pb)
    var rank = max(pa.rank, pb.rank)
    var out = own(new_tensor(shape, rank, ST_BOOL, a.device))
    _compare_spec(op, pa, pb, out.t)
    _release_if_new(pa, a)
    _release_if_new(pb, b)
    ret_owned(rets, 0, out)


def _compare_functional_out(
    op: StaticString, args: Values, rets: Values
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=2])
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected both operands on the same mojo device")
    var dtype = binary_promotion(a.dtype, b.dtype)
    var stype = torch_dtype(dtype)
    var pa = cast_to(a, stype)
    var pb = cast_to(b, stype)
    var shape = _cmp_broadcast_shape(pa, pb)
    var rank = max(pa.rank, pb.rank)
    if _prepare_out(out_arg, shape, rank, ST_BOOL, a.device):
        _compare_spec(op, pa, pb, out_arg)
    else:
        var tmp = own(new_tensor(shape, rank, ST_BOOL, a.device))
        _compare_spec(op, pa, pb, tmp.t)
        copy_strided_into(out_arg, tmp.t)
    _release_if_new(pa, a)
    _release_if_new(pb, b)
    ret_ref(rets, 0, out_arg)


def _compare_scalar(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not a.on_mojo():
        raise Error("expected the mojo device")
    var value = scalar_embed(args[unsafe_offset=1], a.dtype)
    var fill = own(new_scalar(a.stype, a.device))
    fill_value(fill.t, value)
    var out = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    _compare_spec(op, a, fill.t, out.t)
    ret_owned(rets, 0, out)


def _compare_scalar_out(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not a.on_mojo():
        raise Error("expected the mojo device")
    var out_arg = v_tensor(args[unsafe_offset=2])
    var value = scalar_embed(args[unsafe_offset=1], a.dtype)
    var fill = own(new_scalar(a.stype, a.device))
    fill_value(fill.t, value)
    if _prepare_out(out_arg, a.shape, a.rank, ST_BOOL, a.device):
        _compare_spec(op, a, fill.t, out_arg)
    else:
        var tmp = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
        _compare_spec(op, a, fill.t, tmp.t)
        copy_strided_into(out_arg, tmp.t)
    ret_ref(rets, 0, out_arg)


# aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
def op_eq_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("EqSpec", args, rets)


# aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_eq_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("EqSpec", args, rets)


# aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
def op_eq_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("EqSpec", args, rets)


# aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_eq_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("EqSpec", args, rets)


# aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
def op_ne_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("NeSpec", args, rets)


# aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_ne_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("NeSpec", args, rets)


# aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
def op_ne_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("NeSpec", args, rets)


# aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_ne_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("NeSpec", args, rets)


# aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
def op_lt_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("LtSpec", args, rets)


# aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_lt_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("LtSpec", args, rets)


# aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
def op_lt_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("LtSpec", args, rets)


# aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_lt_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("LtSpec", args, rets)


# aten::le.Tensor(Tensor self, Tensor other) -> Tensor
def op_le_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("LeSpec", args, rets)


# aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_le_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("LeSpec", args, rets)


# aten::le.Scalar(Tensor self, Scalar other) -> Tensor
def op_le_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("LeSpec", args, rets)


# aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_le_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("LeSpec", args, rets)


# aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
def op_gt_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("GtSpec", args, rets)


# aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_gt_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("GtSpec", args, rets)


# aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
def op_gt_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("GtSpec", args, rets)


# aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_gt_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("GtSpec", args, rets)


# aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
def op_ge_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_functional("GeSpec", args, rets)


# aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_ge_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_functional_out("GeSpec", args, rets)


# aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
def op_ge_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _compare_scalar("GeSpec", args, rets)


# aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_ge_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _compare_scalar_out("GeSpec", args, rets)


# ---------------------------------------------------------------------------
# isin.Tensor_Tensor: elementwise membership test (logic_ops IsIn). Ported
# from `fast_aten_isin` -- int32/int64 operands only, `assume_unique` is
# ignored (the kernel always does a linear scan), an empty test_elements
# short-circuits to a constant fill.
# ---------------------------------------------------------------------------


def _isin_validate(el: T, te: T) raises -> Bool:
    if not el.on_mojo() or not te.on_mojo() or el.device != te.device:
        raise Error("expected both operands on the same mojo device")
    return el.dtype == te.dtype and (
        el.dtype == DType.int64 or el.dtype == DType.int32
    )


def _isin_launch(el: T, te: T, invert: Bool, dst: T) raises:
    _one_device(el, dst)
    _one_device(te, dst)
    var elc = contiguous(el)
    var tec = contiguous(te)
    var ctx = ctx_for(el.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic_ops", "IsIn")
    call.arg_dtype(0, el.dtype)
    call.arg_dtype(1, te.dtype)
    call.out_dtype(DType.bool)
    call.int(dst.ptr)
    call.int(elc.ptr)
    call.int(tec.ptr)
    call.int(el.numel)
    call.int(te.numel)
    call.int(1 if invert else 0)
    call.int(dtype_code(el.dtype))
    call.int(cp)
    call.run()
    _ = ctx
    _release_if_new(elc, el)
    _release_if_new(tec, te)


def _isin_into(el: T, te: T, invert: Bool, dst: T) raises:
    if el.numel == 0:
        return
    if te.numel == 0:
        fill_value(dst, 1.0 if invert else 0.0)
        return
    _isin_launch(el, te, invert, dst)


# aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
def op_isin_tensor_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var te = v_tensor(args[unsafe_offset=1])
    var invert = v_bool_or(args[unsafe_offset=3], False)
    if not _isin_validate(el, te):
        unsupported("isin: only matching int32/int64 operands are supported")
    var out = own(new_tensor(el.shape, el.rank, ST_BOOL, el.device))
    _isin_into(el, te, invert, out.t)
    ret_owned(rets, 0, out)


# aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
def op_isin_tensor_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var te = v_tensor(args[unsafe_offset=1])
    var invert = v_bool_or(args[unsafe_offset=3], False)
    var out_arg = v_tensor(args[unsafe_offset=4])
    if not _isin_validate(el, te):
        unsupported("isin: only matching int32/int64 operands are supported")
    if _prepare_out(out_arg, el.shape, el.rank, ST_BOOL, el.device):
        _isin_into(el, te, invert, out_arg)
    else:
        var tmp = own(new_tensor(el.shape, el.rank, ST_BOOL, el.device))
        _isin_into(el, te, invert, tmp.t)
        copy_strided_into(out_arg, tmp.t)
    ret_ref(rets, 0, out_arg)


# ---------------------------------------------------------------------------
# where.self / masked_fill(_): the rank<=4 raw-pointer broadcast kernels
# (data_movement_ops WhereSelect, MaskedFillScalar). These predate the
# TensorSpec convention: dims/strides travel as a fixed `[d0..d3, ...]`
# tuple, so operands with true rank > 4 are declined (matching the old
# `_bcast_meta`'s `rank > 4: return None`). Ported from `fast_aten_where`,
# `fast_aten_masked_fill(_)`, `_launch_where_bcast`,
# `_launch_masked_fill_scalar`.
# ---------------------------------------------------------------------------


def _where_broadcast_shape(a: T, b: T, c: T) raises -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        var dim = 1
        if a.shape[i] != 1:
            dim = a.shape[i]
        if b.shape[i] != 1:
            if dim != 1 and b.shape[i] != dim:
                raise Error("shapes are not broadcastable")
            dim = b.shape[i]
        if c.shape[i] != 1:
            if dim != 1 and c.shape[i] != dim:
                raise Error("shapes are not broadcastable")
            dim = c.shape[i]
        shape[i] = dim
    return shape


def _bcast_strides(
    t: T, out_shape: IndexList[MAX_RANK]
) raises -> IndexList[MAX_RANK]:
    """t's per-dim strides broadcast against `out_shape`: 0 on a dim where t
    is size-1 and out_shape isn't (real strides can be anything there, not
    necessarily 0), t's own stride where the dims already match."""
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        if t.shape[i] == out_shape[i]:
            strides[i] = t.strides[i]
        elif t.shape[i] == 1:
            strides[i] = 0
        else:
            raise Error("shapes are not broadcastable")
    return strides


def _where_select(
    cond: T,
    cond_s: IndexList[MAX_RANK],
    a: T,
    a_s: IndexList[MAX_RANK],
    b: T,
    b_s: IndexList[MAX_RANK],
    dst: T,
) raises:
    """dst[i] = cond[i] ? a[i] : b[i], for rank<=4 operands (each stride
    array already the correct broadcast strides against dst's shape)."""
    _one_device(cond, dst)
    _one_device(a, dst)
    _one_device(b, dst)
    if dst.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var params = List[Int](capacity=16)
    for i in range(4):
        params.append(dst.shape[MAX_RANK - 4 + i])
    for i in range(4):
        params.append(cond_s[MAX_RANK - 4 + i])
    for i in range(4):
        params.append(a_s[MAX_RANK - 4 + i])
    for i in range(4):
        params.append(b_s[MAX_RANK - 4 + i])
    var call = KernelCall("data_movement_ops", "WhereSelect")
    call.arg_dtype(0, cond.dtype)
    call.arg_dtype(1, a.dtype)
    call.arg_dtype(2, b.dtype)
    call.out_dtype(dst.dtype)
    call.int(dst.ptr)
    call.int(cond.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.tuple(params)
    call.int(dtype_code(dst.dtype))
    call.int(cp)
    call.run()
    _ = ctx


# aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
def op_where_self(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var cond = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var b = v_tensor(args[unsafe_offset=2])
    if cond.dtype != DType.bool:
        unsupported("where: condition must be a bool tensor")
    if (
        not cond.on_mojo()
        or not a.on_mojo()
        or not b.on_mojo()
        or cond.device != a.device
        or a.device != b.device
    ):
        raise Error("expected all operands on the same mojo device")
    if cond.rank > 4 or a.rank > 4 or b.rank > 4:
        unsupported(
            "where: operand rank exceeds the fast broadcast kernel's limit of 4"
        )
    var pair = promoted_pair(a, b)
    var pa = pair[0].copy()
    var pb = pair[1].copy()
    var shape = _where_broadcast_shape(cond, pa, pb)
    var rank = max(cond.rank, max(pa.rank, pb.rank))
    var cond_s = _bcast_strides(cond, shape)
    var a_s = _bcast_strides(pa, shape)
    var b_s = _bcast_strides(pb, shape)
    var out = own(new_tensor(shape, rank, pa.stype, pa.device))
    _where_select(cond, cond_s, pa, a_s, pb, b_s, out.t)
    _release_if_new(pa, a)
    _release_if_new(pb, b)
    ret_owned(rets, 0, out)


def _masked_fill_validate(a: T, mask: T) raises:
    if mask.dtype != DType.bool:
        unsupported("masked_fill: mask must be a bool tensor")
    if not a.on_mojo() or not mask.on_mojo() or a.device != mask.device:
        raise Error("expected both operands on the same mojo device")


def _is_masked_fill_scalar_dtype(dtype: DType) -> Bool:
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
    )


def _masked_fill_scalar_launch(
    mask: T, self_t: T, value: Float64, dst: T
) raises:
    """The fast path: `value` is baked into the launch (no separate Fill
    kernel first). Only float32/float16/bfloat16 `self` is wired to this;
    every other dtype goes through `_masked_fill_where` below."""
    _one_device(mask, dst)
    _one_device(self_t, dst)
    var mask_s = _bcast_strides(mask, self_t.shape)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var params = List[Int](capacity=12)
    for i in range(4):
        params.append(self_t.shape[MAX_RANK - 4 + i])
    for i in range(4):
        params.append(mask_s[MAX_RANK - 4 + i])
    for i in range(4):
        params.append(self_t.strides[MAX_RANK - 4 + i])
    var call = KernelCall("data_movement_ops", "MaskedFillScalar")
    call.arg_dtype(0, mask.dtype)
    call.arg_dtype(1, self_t.dtype)
    call.out_dtype(dst.dtype)
    call.int(dst.ptr)
    call.int(mask.ptr)
    call.int(self_t.ptr)
    call.tuple(params)
    call.f64(value)
    call.int(dtype_code(dst.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _masked_fill_where(mask: T, value: T, a: T, dst: T) raises:
    """The generic path: dst = WhereSelect(mask, value, a) -- any dtype, any
    value (a real tensor or a synthesized 0-d fill)."""
    if dst.numel == 0:
        return
    if mask.rank > 4 or value.rank > 4 or a.rank > 4:
        unsupported(
            "masked_fill: operand rank exceeds the fast broadcast kernel's"
            " limit of 4"
        )
    var mask_s = _bcast_strides(mask, a.shape)
    var value_s = _bcast_strides(value, a.shape)
    _where_select(mask, mask_s, value, value_s, a, a.strides, dst)


def _masked_fill_scalar_dispatch(
    mask: T, a: T, value_arg: Value, dst: T
) raises:
    if dst.numel == 0:
        return
    if _is_masked_fill_scalar_dtype(a.dtype) and mask.rank <= 4 and a.rank <= 4:
        var fast_value = scalar_embed(value_arg, a.dtype)
        _masked_fill_scalar_launch(mask, a, fast_value, dst)
        return
    var value = scalar_embed(value_arg, a.dtype)
    var fill = own(new_scalar(a.stype, a.device))
    fill_value(fill.t, value)
    _masked_fill_where(mask, fill.t, a, dst)


# aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
def op_masked_fill_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var out = own(new_like(a))
    _masked_fill_scalar_dispatch(mask, a, args[unsafe_offset=2], out.t)
    ret_owned(rets, 0, out)


# aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
def op_masked_fill_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=3])
    _masked_fill_validate(a, mask)
    if _prepare_out(out_arg, a.shape, a.rank, a.stype, a.device):
        _masked_fill_scalar_dispatch(mask, a, args[unsafe_offset=2], out_arg)
    else:
        var tmp = own(new_like(a))
        _masked_fill_scalar_dispatch(mask, a, args[unsafe_offset=2], tmp.t)
        copy_strided_into(out_arg, tmp.t)
    ret_ref(rets, 0, out_arg)


# aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
def op_masked_fill__scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    if a.contig:
        # Writing dst == a is safe: each element reads and writes the same
        # flat index (a's own strides are the output layout).
        _masked_fill_scalar_dispatch(mask, a, args[unsafe_offset=2], a)
    else:
        var tmp = own(new_like(a))
        _masked_fill_scalar_dispatch(mask, a, args[unsafe_offset=2], tmp.t)
        copy_strided_into(a, tmp.t)
    ret_ref(rets, 0, a)


def _masked_fill_tensor_value(a: T, value_arg: Value) raises -> T:
    var val = v_tensor(value_arg)
    if val.rank != 0:
        raise Error(
            (
                "masked_fill_ only supports a 0-dimensional value tensor, but"
                " got tensor with "
            ),
            val.rank,
            " dimension(s).",
        )
    if val.dtype != a.dtype or val.device != a.device:
        unsupported("masked_fill.Tensor: value dtype/device mismatch")
    return val^


# aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
def op_masked_fill_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_tensor_value(a, args[unsafe_offset=2])
    var out = own(new_like(a))
    _masked_fill_where(mask, val, a, out.t)
    ret_owned(rets, 0, out)


# aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
def op_masked_fill_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=3])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_tensor_value(a, args[unsafe_offset=2])
    if _prepare_out(out_arg, a.shape, a.rank, a.stype, a.device):
        _masked_fill_where(mask, val, a, out_arg)
    else:
        var tmp = own(new_like(a))
        _masked_fill_where(mask, val, a, tmp.t)
        copy_strided_into(out_arg, tmp.t)
    ret_ref(rets, 0, out_arg)


# aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
def op_masked_fill__tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_tensor_value(a, args[unsafe_offset=2])
    if a.contig:
        _masked_fill_where(mask, val, a, a)
    else:
        var tmp = own(new_like(a))
        _masked_fill_where(mask, val, a, tmp.t)
        copy_strided_into(a, tmp.t)
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# searchsorted / bucketize: one binary search per value (searchsorted_ops
# Searchsorted), shared by CPU/GPU. Ported from `_fast_searchsorted`,
# `fast_aten_searchsorted`, `fast_aten_bucketize`.
#
# `sorter` (searchsorted only) is checked the way `searchsorted_pre_check`
# (BucketizationUtils.h) checks it statically -- device, shape, dtype -- but
# NOT the way it checks values: torch's own check there is a device-to-host
# `aminmax().item()`, a sync we won't pay on every call. An out-of-range
# sorter entry is instead made harmless in the kernel itself (clamped into
# the valid boundary range), see `_binary_search_position` in
# `searchsorted_ops.mojo`.
# ---------------------------------------------------------------------------

# _SS_ALLOWED (float32/bfloat16/float16/int32/int64): float64, and integers
# narrower than int32, are deliberately declined below -- matches the old
# `_SEARCHSORTED_DTYPES` (their kernels aren't supported).


def _ss_allowed(dt: DType) -> Bool:
    return (
        dt == DType.float32
        or dt == DType.bfloat16
        or dt == DType.float16
        or dt == DType.int32
        or dt == DType.int64
    )


def _ss_int_like(dt: DType) -> Bool:
    return (
        dt == DType.bool
        or dt == DType.uint8
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
    )


def _ss_float_like(dt: DType) -> Bool:
    return (
        dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float32
        or dt == DType.float64
    )


def _searchsorted_dtype(a: DType, b: DType) raises -> DType:
    """torch's `result_type(a, b)`, restricted to `_SS_ALLOWED`: declines
    (NotImplementedError) any pair whose true torch result would be float64
    or an integer narrower than int32, since the kernel doesn't cover those.
    Ported from `_searchsorted_tensor_dtype`.
    """
    if a == b:
        if _ss_allowed(a):
            return a
        unsupported("searchsorted: dtype " + String(a) + " is not supported")
    if _ss_float_like(a) and _ss_float_like(b):
        if a == DType.float64 or b == DType.float64:
            unsupported("searchsorted: float64 is not supported")
        return DType.float32  # float32/float16/bfloat16 all promote here
    if _ss_float_like(a) and _ss_int_like(b):
        if a == DType.float64:
            unsupported("searchsorted: float64 is not supported")
        return a
    if _ss_float_like(b) and _ss_int_like(a):
        if b == DType.float64:
            unsupported("searchsorted: float64 is not supported")
        return b
    if _ss_int_like(a) and _ss_int_like(b):
        if a == DType.int64 or b == DType.int64:
            return DType.int64
        if a == DType.int32 or b == DType.int32:
            return DType.int32
        unsupported("searchsorted: dtype pair is narrower than int32")
    unsupported(
        "searchsorted: unsupported dtype pair " + String(a) + "/" + String(b)
    )
    return DType.float32


def _searchsorted_scalar_dtype(boundary_dtype: DType, v: Value) raises -> DType:
    """ATen's wrapped-number promotion for a Python scalar search value: an
    int/bool scalar never upgrades the boundary tensor's dtype category; a
    float scalar against a floating boundary stays that dtype; a float
    scalar against an int/bool boundary promotes to the default float
    dtype. Ported from `_searchsorted_scalar_dtype`.
    """
    if v_scalar_is_integral(v):
        if not _ss_allowed(boundary_dtype):
            unsupported(
                "searchsorted: dtype "
                + String(boundary_dtype)
                + " is not supported"
            )
        return boundary_dtype
    if _ss_float_like(boundary_dtype):
        if boundary_dtype == DType.float64:
            unsupported("searchsorted: float64 is not supported")
        return boundary_dtype
    return max_dtype(default_dtype())


def _prep_flat(t: T, stype: Int32) raises -> T:
    """A contiguous tensor of dtype `stype`, in at most one fresh
    allocation beyond `t` (the Searchsorted kernel indexes flat storage)."""
    if t.stype != stype:
        return cast_to(t, stype)  # cast_to's result is already contiguous
    return contiguous(t)


def _check_sorter(sorter: T, boundaries: T) raises:
    """torch's static `sorter` checks (`searchsorted_pre_check`): device,
    shape, dtype. Index VALUES are not checked here -- see the header
    comment above."""
    if not sorter.on_mojo() or sorter.device != boundaries.device:
        raise Error(
            "torch.searchsorted(): sorter and boundary tensors should have"
            " same device type"
        )
    if not sorter.same_shape(boundaries):
        raise Error(
            "torch.searchsorted(): boundary and sorter must have the same"
            " size"
        )
    if sorter.dtype != DType.int64:
        raise Error(
            "torch.searchsorted(): sorter must be a tensor of long dtype"
        )


def _searchsorted_common(
    boundaries0: T,
    values0: T,
    common: DType,
    out_int32: Bool,
    right0: Bool,
    has_side: Bool,
    side_str: String,
    sorter0: Optional[T],
) raises -> Owned:
    var right = right0
    if has_side:
        if side_str != "left" and side_str != "right":
            raise Error(
                (
                    "torch.searchsorted(): side can only be 'left' or 'right'"
                    " but got "
                ),
                side_str,
            )
        if right0 and side_str != "right":
            raise Error(
                "torch.searchsorted(): side and right can't be set to opposites"
            )
        right = side_str == "right"

    if (
        not boundaries0.on_mojo()
        or not values0.on_mojo()
        or boundaries0.device != values0.device
    ):
        raise Error(
            "torch.searchsorted(): boundaries and input value tensors should"
            " have same device type"
        )
    if boundaries0.rank == 0:
        raise Error(
            "torch.searchsorted(): boundaries tensor should have positive"
            " dimension, but got 0 dimension"
        )
    if values0.rank == 0 and boundaries0.rank != 1:
        raise Error(
            "torch.searchsorted(): input value can be a scalar only when"
            " boundaries tensor dimension is 1"
        )
    if boundaries0.rank != 1:
        if values0.rank != boundaries0.rank:
            raise Error(
                "torch.searchsorted(): boundaries tensor should be 1"
                " dimension or the first N-1 dimensions of boundaries"
                " tensor and input value tensor must match"
            )
        for i in range(boundaries0.rank - 1):
            if values0.dim(i) != boundaries0.dim(i):
                raise Error(
                    "torch.searchsorted(): boundaries tensor should be 1"
                    " dimension or the first N-1 dimensions of boundaries"
                    " tensor and input value tensor must match"
                )

    var has_sorter = Bool(sorter0)
    if has_sorter:
        _check_sorter(sorter0.value(), boundaries0)

    var boundary_size = boundaries0.dim(-1)
    if out_int32 and boundary_size >= 2147483647:
        raise Error(
            (
                "torch.searchsorted(): the size of boundaries' last dimension"
                " should be less than 2147483647, but we got "
            ),
            boundary_size,
        )

    var stype = torch_dtype(common)
    var boundaries = _prep_flat(boundaries0, stype)
    var values = _prep_flat(values0, stype)
    # The kernel indexes it as flat int64 storage, same as boundaries/values.
    var sorter_c: Optional[T] = None
    if has_sorter:
        sorter_c = contiguous(sorter0.value())
    var sorter_ptr = sorter_c.value().ptr if has_sorter else 0

    var out_dtype = DType.int32 if out_int32 else DType.int64
    var out = own(
        new_tensor(
            values0.shape, values0.rank, torch_dtype(out_dtype), values0.device
        )
    )
    if out.t.numel > 0:
        var values_per_batch = 1
        if values0.rank > 0:
            values_per_batch = values0.dim(-1)
        var ctx = ctx_for(values0.device)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("searchsorted_ops", "Searchsorted")
        call.arg_dtype(0, common)
        call.out_dtype(out_dtype)
        call.int(out.t.ptr)
        call.int(boundaries.ptr)
        call.int(values.ptr)
        call.int(sorter_ptr)
        call.int(values0.numel)
        call.int(boundary_size)
        call.int(values_per_batch)
        call.int(1 if boundaries0.rank == 1 else 0)
        call.int(1 if has_sorter else 0)
        call.int(1 if right else 0)
        call.int(dtype_code(common))
        call.int(dtype_code(out_dtype))
        call.int(cp)
        call.run()
        _ = ctx
    _release_if_new(boundaries, boundaries0)
    _release_if_new(values, values0)
    if has_sorter:
        _release_if_new(sorter_c.value(), sorter0.value())
    return out^


def _side_of(v: Value) raises -> Tuple[Bool, String]:
    if v_is_none(v):
        return (False, String(""))
    return (True, v_string(v))


# aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
def op_searchsorted_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=0])
    var values = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var side = _side_of(args[unsafe_offset=4])
    var sorter = v_opt_tensor(args[unsafe_offset=5])
    var common = _searchsorted_dtype(boundaries.dtype, values.dtype)
    var out = _searchsorted_common(
        boundaries, values, common, out_int32, right, side[0], side[1], sorter
    )
    ret_owned(rets, 0, out)


# aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
def op_searchsorted_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=0])
    var values = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var side = _side_of(args[unsafe_offset=4])
    var sorter = v_opt_tensor(args[unsafe_offset=5])
    var out_arg = v_tensor(args[unsafe_offset=6])
    var common = _searchsorted_dtype(boundaries.dtype, values.dtype)
    var computed = _searchsorted_common(
        boundaries, values, common, out_int32, right, side[0], side[1], sorter
    )
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    ret_ref(rets, 0, out_arg)


# aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
def op_searchsorted_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=0])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var side = _side_of(args[unsafe_offset=4])
    var sorter = v_opt_tensor(args[unsafe_offset=5])
    if boundaries.rank != 1:
        raise Error(
            "torch.searchsorted(): input value can be a scalar only when"
            " boundaries tensor dimension is 1"
        )
    var common = _searchsorted_scalar_dtype(
        boundaries.dtype, args[unsafe_offset=1]
    )
    var value = scalar_embed(args[unsafe_offset=1], common)
    var values = own(new_scalar(torch_dtype(common), boundaries.device))
    fill_value(values.t, value)
    var out = _searchsorted_common(
        boundaries, values.t, common, out_int32, right, side[0], side[1], sorter
    )
    ret_owned(rets, 0, out)


# aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
def op_searchsorted_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=0])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var side = _side_of(args[unsafe_offset=4])
    var sorter = v_opt_tensor(args[unsafe_offset=5])
    var out_arg = v_tensor(args[unsafe_offset=6])
    if boundaries.rank != 1:
        raise Error(
            "torch.searchsorted(): input value can be a scalar only when"
            " boundaries tensor dimension is 1"
        )
    var common = _searchsorted_scalar_dtype(
        boundaries.dtype, args[unsafe_offset=1]
    )
    var value = scalar_embed(args[unsafe_offset=1], common)
    var values = own(new_scalar(torch_dtype(common), boundaries.device))
    fill_value(values.t, value)
    var computed = _searchsorted_common(
        boundaries, values.t, common, out_int32, right, side[0], side[1], sorter
    )
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    ret_ref(rets, 0, out_arg)


# aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
def op_bucketize_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_t = v_tensor(args[unsafe_offset=0])
    var boundaries = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    if boundaries.rank != 1:
        raise Error(
            "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
            boundaries.rank,
            ")",
        )
    var common = _searchsorted_dtype(boundaries.dtype, self_t.dtype)
    var out = _searchsorted_common(
        boundaries, self_t, common, out_int32, right, False, String(""), None
    )
    ret_owned(rets, 0, out)


# aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
def op_bucketize_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_t = v_tensor(args[unsafe_offset=0])
    var boundaries = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var out_arg = v_tensor(args[unsafe_offset=4])
    if boundaries.rank != 1:
        raise Error(
            "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
            boundaries.rank,
            ")",
        )
    var common = _searchsorted_dtype(boundaries.dtype, self_t.dtype)
    var computed = _searchsorted_common(
        boundaries, self_t, common, out_int32, right, False, String(""), None
    )
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    ret_ref(rets, 0, out_arg)


# aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
def op_bucketize_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    if boundaries.rank != 1:
        raise Error(
            "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
            boundaries.rank,
            ")",
        )
    var common = _searchsorted_scalar_dtype(
        boundaries.dtype, args[unsafe_offset=0]
    )
    var value = scalar_embed(args[unsafe_offset=0], common)
    var values = own(new_scalar(torch_dtype(common), boundaries.device))
    fill_value(values.t, value)
    var out = _searchsorted_common(
        boundaries, values.t, common, out_int32, right, False, String(""), None
    )
    ret_owned(rets, 0, out)


# aten::bucketize.Scalar_out(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
def op_bucketize_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var boundaries = v_tensor(args[unsafe_offset=1])
    var out_int32 = v_bool_or(args[unsafe_offset=2], False)
    var right = v_bool_or(args[unsafe_offset=3], False)
    var out_arg = v_tensor(args[unsafe_offset=4])
    if boundaries.rank != 1:
        raise Error(
            "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
            boundaries.rank,
            ")",
        )
    var common = _searchsorted_scalar_dtype(
        boundaries.dtype, args[unsafe_offset=0]
    )
    var value = scalar_embed(args[unsafe_offset=0], common)
    var values = own(new_scalar(torch_dtype(common), boundaries.device))
    fill_value(values.t, value)
    var computed = _searchsorted_common(
        boundaries, values.t, common, out_int32, right, False, String(""), None
    )
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    ret_ref(rets, 0, out_arg)


def register_compare(site: Site) raises:
    impl[op_eq_tensor, "eq.Tensor"](site)
    impl[op_eq_tensor_out, "eq.Tensor_out"](site)
    impl[op_eq_scalar, "eq.Scalar"](site)
    impl[op_eq_scalar_out, "eq.Scalar_out"](site)
    impl[op_ne_tensor, "ne.Tensor"](site)
    impl[op_ne_tensor_out, "ne.Tensor_out"](site)
    impl[op_ne_scalar, "ne.Scalar"](site)
    impl[op_ne_scalar_out, "ne.Scalar_out"](site)
    impl[op_lt_tensor, "lt.Tensor"](site)
    impl[op_lt_tensor_out, "lt.Tensor_out"](site)
    impl[op_lt_scalar, "lt.Scalar"](site)
    impl[op_lt_scalar_out, "lt.Scalar_out"](site)
    impl[op_le_tensor, "le.Tensor"](site)
    impl[op_le_tensor_out, "le.Tensor_out"](site)
    impl[op_le_scalar, "le.Scalar"](site)
    impl[op_le_scalar_out, "le.Scalar_out"](site)
    impl[op_gt_tensor, "gt.Tensor"](site)
    impl[op_gt_tensor_out, "gt.Tensor_out"](site)
    impl[op_gt_scalar, "gt.Scalar"](site)
    impl[op_gt_scalar_out, "gt.Scalar_out"](site)
    impl[op_ge_tensor, "ge.Tensor"](site)
    impl[op_ge_tensor_out, "ge.Tensor_out"](site)
    impl[op_ge_scalar, "ge.Scalar"](site)
    impl[op_ge_scalar_out, "ge.Scalar_out"](site)
    impl[op_isin_tensor_tensor, "isin.Tensor_Tensor"](site)
    impl[op_isin_tensor_tensor_out, "isin.Tensor_Tensor_out"](site)
    impl[op_where_self, "where.self"](site)
    impl[op_masked_fill_scalar, "masked_fill.Scalar"](site)
    impl[op_masked_fill_scalar_out, "masked_fill.Scalar_out"](site)
    impl[op_masked_fill_tensor, "masked_fill.Tensor"](site)
    impl[op_masked_fill_tensor_out, "masked_fill.Tensor_out"](site)
    impl[op_masked_fill__scalar, "masked_fill_.Scalar"](site)
    impl[op_masked_fill__tensor, "masked_fill_.Tensor"](site)
    impl[op_searchsorted_tensor, "searchsorted.Tensor"](site)
    impl[op_searchsorted_tensor_out, "searchsorted.Tensor_out"](site)
    impl[op_searchsorted_scalar, "searchsorted.Scalar"](site)
    impl[op_searchsorted_scalar_out, "searchsorted.Scalar_out"](site)
    impl[op_bucketize_tensor, "bucketize.Tensor"](site)
    impl[op_bucketize_tensor_out, "bucketize.Tensor_out"](site)
    impl[op_bucketize_scalar, "bucketize.Scalar"](site)
    impl[op_bucketize_scalar_out, "bucketize.Scalar_out"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_compare]()
