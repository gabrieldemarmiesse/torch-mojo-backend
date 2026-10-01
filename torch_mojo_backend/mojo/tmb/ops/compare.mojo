"""ATen ops: compare group (see agents_docs/native_backend.md).

Ports the old eager path's comparisons / isin / where / masked_fill /
searchsorted / bucketize (`eager_kernels/aten_fast.py`,
`mojo_device/aten_ops/inplace.py`) onto three kernel families:
`logic` (the comparison specs, IsIn), `data_movement` (WhereSelect,
MaskedFillScalar) and `searchsorted` (Searchsorted). The comparison specs
follow the generic TensorSpec convention (arbitrary rank/strides, like
AddSpec/MulSpec in ops_binary.mojo); WhereSelect/MaskedFillScalar/IsIn
predate that convention and take raw pointers plus fixed-rank-4 dim/stride
tuples, so operands there are capped at rank 4 (declined above that, matching
the old `_bcast_meta`'s `rank > 4: return None`).
"""
from std.utils import IndexList

from tmb.backend.abi import (
    bits_f64,
    Owned,
    T,
    TAG_BOOL,
    TAG_COMPLEX,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    Value,
    Values,
    ST_BOOL,
    ST_FLOAT64,
    default_dtype,
    dtype_code,
    dtype_name,
    max_dtype,
    new_like,
    new_scalar,
    new_tensor,
    none_arg,
    own,
    own_if_new,
    release,
    ret_bool,
    ret_owned,
    ret_ref,
    int_arg,
    tensor_arg,
    torch_dtype,
    unsupported,
    v_bool_or,
    v_f64,
    v_int,
    v_is_none,
    v_opt_tensor,
    v_scalar_is_integral,
    v_string,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    binary_promotion,
    broadcast_shape,
    call_op,
    cast_to,
    contiguous,
    copy_strided_into,
    fill_value,
    promote_types,
    promoted_pair,
    resize_out,
    scalar_embed,
    scalar_to_float,
    scalar_to_int,
)
from tmb.backend.registry import Site, impl
from tmb.ops.core import cast_for_copy
from tmb.ops.binary import _b_out_guard, _b_tside


def _release_if_new(t: T, orig: T):
    """Release `t` if it is a fresh allocation distinct from `orig` (the
    original, caller-owned operand) -- the standard idiom this backend uses
    for a helper's "input itself, or a materialized copy" return."""
    if t.h != orig.h:
        release(t.h)


def _shape_eq(t: T, shape: IndexList[MAX_RANK], rank: Int) -> Bool:
    """Whether t's padded shape exactly equals `shape` (both MAX_RANK-wide,
    leading-padded with 1s) at `rank`: the eligibility check for writing an
    out= kernel result directly into `t`. The rank matters: a [1] `out` of a
    0-d result is resized to 0-d."""
    if t.rank != rank:
        return False
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
    (`resize_out`: a boxed kernel gets no resize before dispatch), and
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
    if not _shape_eq(out_arg, shape, rank):
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
    if not _shape_eq(out_arg, shape, rank):
        resize_out(out_arg, shape, rank)


# ---------------------------------------------------------------------------
# eq / ne / lt / le / gt / ge: broadcast-strided comparison specs
# (logic: EqSpec/NeSpec/LtSpec/LeSpec/GtSpec/GeSpec), same TensorSpec
# convention and dtype-promotion rule as add/mul in ops_binary.mojo, but the
# output dtype is always bool. Ported from `fast_aten_eq` et al., which all
# go through `_try_spec_binary(..., out_dtype=bool)`.
# ---------------------------------------------------------------------------


def _one_device(a: T, b: T) raises:
    """Both operands of a raw-pointer launch on the same mojo device.

    A kernel gets bare pointers and one stream: a pointer belonging to
    another device -- or to no mojo device at all -- would be dereferenced
    against the wrong context. The fields are cached on `T`, so this costs
    nothing. Private to this file until the port is merged; it belongs in
    tmb/ops/common.mojo.
    """
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected every operand on the same mojo device")


def _compare_spec(op: StaticString, a: T, b: T, dst: T) raises:
    _one_device(a, dst)
    _one_device(b, dst)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic", String(op))
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    call.out_dtype(dst.dtype)
    call.spec(a.spec(cp))
    call.spec(b.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _compare_out_guard(out_arg: T, a: T, b: T) raises:
    """TensorIterator's `out=` meta (the #606 contract every other `out=`
    path runs): no internal overlap in `out`, no partial overlap with an
    input, and an `out` that IS an input (the in-place `x.eq_(y)`) never
    resized -- a broadcast shape larger than it raises."""
    if not out_arg.on_mojo():
        raise Error("expected the out= tensor on the operands' mojo device")
    assert_no_internal_overlap(out_arg)
    _b_out_guard(out_arg, _b_tside(a), _b_tside(b))


def _cast_into_out(
    mut out_arg: T,
    res: T,
    shape: IndexList[MAX_RANK],
    rank: Int,
    device: Int,
) raises:
    """A comparison's `out=` of a non-bool dtype: TensorIterator's
    comparison ops (`build_borrowing_comparison_op`) cast the bool result
    into an `out` of any dtype (`torch.eq(x, 0.5, out=float_t)` is 0./1.),
    which is also how the in-place `x.eq_(0.5)` on a float `x` works. `res`
    is the bool result, computed apart so an `out` aliasing an operand is
    only written once every element has been read."""
    if not out_arg.on_mojo() or out_arg.device != device:
        raise Error("expected the out= tensor on the operands' mojo device")
    if not _shape_eq(out_arg, shape, rank):
        resize_out(out_arg, shape, rank)
    var casted = own(cast_for_copy(res, out_arg.stype))
    copy_strided_into(out_arg, casted.t)
    _ = casted^  # alive past the launch


def _compare_functional(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected both operands on the same mojo device")
    var dtype = binary_promotion(a, b)
    var stype = torch_dtype(dtype)
    var pa = cast_to(a, stype)
    var pb = cast_to(b, stype)
    var shape = broadcast_shape(pa, pb)
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
    _compare_out_guard(out_arg, a, b)
    var dtype = binary_promotion(a, b)
    var stype = torch_dtype(dtype)
    var pa = cast_to(a, stype)
    var pb = cast_to(b, stype)
    var shape = broadcast_shape(pa, pb)
    var rank = max(pa.rank, pb.rank)
    if out_arg.stype != ST_BOOL:
        var res = own(new_tensor(shape, rank, ST_BOOL, a.device))
        _compare_spec(op, pa, pb, res.t)
        _cast_into_out(out_arg, res.t, shape, rank, a.device)
        _ = res^  # alive past the launch
    elif _prepare_out(out_arg, shape, rank, ST_BOOL, a.device):
        _compare_spec(op, pa, pb, out_arg)
    else:
        var tmp = own(new_tensor(shape, rank, ST_BOOL, a.device))
        _compare_spec(op, pa, pb, tmp.t)
        copy_strided_into(out_arg, tmp.t)
        _ = tmp^  # alive past the launch
    _release_if_new(pa, a)
    _release_if_new(pb, b)
    ret_ref(rets, 0, out_arg)


def _scalar_fill(a: T, v: Value) raises -> Owned:
    """The Scalar operand as a 0-d tensor of `a`'s dtype. An integer scalar
    against an integer tensor is filled from its int64 bits (the #606
    integer-Scalar path, exact past 2**53); everything else goes through
    `scalar_embed`'s Float64."""
    var fill = own(new_scalar(a.stype, a.device))
    if (
        v_scalar_is_integral(v)
        and not a.dtype.is_floating_point()
        and a.dtype != DType.bool
    ):
        fill_value(fill.t, v)
    else:
        fill_value(fill.t, scalar_embed(v, a.dtype))
    return fill^


def _compare_scalar(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not a.on_mojo():
        raise Error("expected the mojo device")
    var fill = _scalar_fill(a, args[unsafe_offset=1])
    var out = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    _compare_spec(op, a, fill.t, out.t)
    _ = fill^  # alive past the launch
    ret_owned(rets, 0, out)


def _compare_scalar_out(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not a.on_mojo():
        raise Error("expected the mojo device")
    var out_arg = v_tensor(args[unsafe_offset=2])
    _compare_out_guard(out_arg, a, a)
    var fill = _scalar_fill(a, args[unsafe_offset=1])
    if out_arg.stype != ST_BOOL:
        var res = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
        _compare_spec(op, a, fill.t, res.t)
        _ = fill^  # alive past the launch
        _cast_into_out(out_arg, res.t, a.shape, a.rank, a.device)
        _ = res^  # alive past the launch
    elif _prepare_out(out_arg, a.shape, a.rank, ST_BOOL, a.device):
        _compare_spec(op, a, fill.t, out_arg)
        _ = fill^  # alive past the launch
    else:
        var tmp = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
        _compare_spec(op, a, fill.t, tmp.t)
        _ = fill^  # alive past the launch
        copy_strided_into(out_arg, tmp.t)
        _ = tmp^  # alive past the launch
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
# isin: elementwise membership test (logic IsIn, a linear scan of the test
# elements per element: CUDA's `isin_default_kernel`; its sorting route for
# large test sets answers identically, NaN never being a member). Operands
# promote to their common dtype first; `assume_unique` changes nothing here.
# The Scalar overloads are ATen's redispatches: Tensor_Scalar is eq / ne,
# Scalar_Tensor is a 0-d membership test of the scalar.
# ---------------------------------------------------------------------------


def _isin_check_dtype(dt: DType) raises:
    """`check_for_unsupported_isin_dtype` (bool and complex are refused)."""
    if dt == DType.bool:
        raise Error("Unsupported input type encountered for isin(): Bool")


def _isin_kernel_dtype(dt: DType) -> Bool:
    return (
        dt == DType.int64
        or dt == DType.int32
        or dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
    )


def _isin_validate(el: T, te: T) raises -> Int32:
    """The common dtype both operands are scanned in (declines what the
    kernel lacks)."""
    if not el.on_mojo() or not te.on_mojo() or el.device != te.device:
        raise Error("expected both operands on the same mojo device")
    _isin_check_dtype(el.dtype)
    _isin_check_dtype(te.dtype)
    var common = promote_types(el.stype, te.stype)
    var cdt = max_dtype(common)
    if not _isin_kernel_dtype(cdt):
        unsupported("isin of dtype " + String(cdt))
    if cdt == DType.float64 and ctx_for(el.device).api() == "metal":
        unsupported("isin: float64 is unavailable on Apple GPUs")
    return common


def _isin_launch(el: T, te: T, invert: Bool, dst: T) raises:
    _one_device(el, dst)
    _one_device(te, dst)
    var elc = contiguous(el)
    var tec = contiguous(te)
    var ctx = ctx_for(el.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic", "IsIn")
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


def _isin_into(el: T, te: T, common: Int32, invert: Bool, dst: T) raises:
    if el.numel == 0:
        return
    if te.numel == 0:
        fill_value(dst, 1.0 if invert else 0.0)
        return
    var elp = own_if_new(cast_to(el, common), el)
    var tep = own_if_new(cast_to(te, common), te)
    _isin_launch(elp.t, tep.t, invert, dst)
    _ = elp^  # alive past the launch
    _ = tep^


# aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
def op_isin_tensor_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var te = v_tensor(args[unsafe_offset=1])
    var invert = v_bool_or(args[unsafe_offset=3], False)
    var common = _isin_validate(el, te)
    var out = own(new_tensor(el.shape, el.rank, ST_BOOL, el.device))
    _isin_into(el, te, common, invert, out.t)
    ret_owned(rets, 0, out)


# aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
def op_isin_tensor_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var te = v_tensor(args[unsafe_offset=1])
    var invert = v_bool_or(args[unsafe_offset=3], False)
    var out_arg = v_tensor(args[unsafe_offset=4])
    var common = _isin_validate(el, te)
    var tmp = own(new_tensor(el.shape, el.rank, ST_BOOL, el.device))
    _isin_into(el, te, common, invert, tmp.t)
    _ = _prepare_out(out_arg, el.shape, el.rank, ST_BOOL, el.device)
    assert_no_internal_overlap(out_arg)
    copy_strided_into(out_arg, tmp.t)
    _ = tmp^  # alive past the launch
    ret_ref(rets, 0, out_arg)


def _scalar_type_of(v: Value) -> String:
    if v_scalar_is_integral(v):
        if v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL:
            return "Bool"
        return "Long"
    return "Double"


def _isin_scalar_test(el: T, test: Value) raises:
    if not el.on_mojo():
        raise Error("expected a tensor on the mojo device")
    _isin_check_dtype(el.dtype)
    if _scalar_type_of(test) == "Bool":
        raise Error("Unsupported input type encountered for isin(): Bool")


# aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
def op_isin_tensor_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var test = args[unsafe_offset=1].copy()
    _isin_scalar_test(el, test)
    var invert = v_bool_or(args[unsafe_offset=3], False)
    var r = call_op(
        "aten::ne" if invert else "aten::eq",
        "Scalar",
        [tensor_arg(el), test^],
        1,
    )
    var out = own(r.take_tensor(0))
    ret_owned(rets, 0, out)


# aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
def op_isin_tensor_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var el = v_tensor(args[unsafe_offset=0])
    var test = args[unsafe_offset=1].copy()
    var out_arg = v_tensor(args[unsafe_offset=4])
    _isin_scalar_test(el, test)
    if out_arg.stype != ST_BOOL:
        raise Error(
            "Expected out tensor to have dtype bool, but got ",
            dtype_name(out_arg.stype),
            " instead",
        )
    var invert = v_bool_or(args[unsafe_offset=3], False)
    _ = call_op(
        "aten::ne" if invert else "aten::eq",
        "Scalar_out",
        [tensor_arg(el), test^, tensor_arg(out_arg)],
        1,
    )
    ret_ref(rets, 0, out_arg)


def _float_holds(dt: DType, x: Float64) -> Bool:
    """Is `x` exactly a value of the float dtype `dt` (NaN counts: it
    compares unequal either way)?"""
    comptime for fdt in [DType.float16, DType.bfloat16, DType.float32]:
        if dt == fdt:
            var r = Scalar[fdt](x).cast[DType.float64]()
            return r == x or x != x
    return True  # float64


def _int_range(dt: DType) -> Tuple[Float64, Float64]:
    comptime for idt in [
        DType.int8,
        DType.uint8,
        DType.int16,
        DType.int32,
        DType.int64,
    ]:
        if dt == idt:
            return (
                Float64(Scalar[idt].MIN_FINITE),
                Float64(Scalar[idt].MAX_FINITE),
            )
    return (Float64(0), Float64(1))  # bool


def _isin_scalar_tensor(el: Value, te: T, invert: Bool) raises -> Owned:
    """The scalar as a 0-d member test: any(test_elements == element).

    ATen wraps the scalar into a 0-d float64 (int64) tensor that its isin
    kernels treat as an ordinary operand, so the comparison runs in
    promote_types(float64 or int64, test dtype) -- not in the test dtype,
    as `eq.Scalar`'s wrapped number would: `isin(1 + 1e-8, float32 [1])` is
    False. Comparing in the test dtype is the same test once the scalar is
    known to be one of its values exactly; a scalar no test value can equal
    finds nothing."""
    if not te.on_mojo():
        raise Error("expected a tensor on the mojo device")
    if _scalar_type_of(el) == "Bool":
        raise Error("Unsupported input type encountered for isin(): Bool")
    _isin_check_dtype(te.dtype)
    var tc = own_if_new(te.copy(), te)
    var test = el.copy()
    var possible = True
    var x = v_f64(el)
    if te.dtype.is_floating_point():
        # An integer scalar promotes to the float test dtype (int64 tensor
        # with a float32 tensor is float32): its rounding is torch's too.
        if not v_scalar_is_integral(el):
            possible = _float_holds(te.dtype, x)
    else:
        var rng = _int_range(te.dtype)
        if v_scalar_is_integral(el):
            # int64 comparison: a scalar outside the dtype matches nothing.
            var v = v_int(el)
            possible = Float64(v) >= rng[0] and Float64(v) <= rng[1]
            if not possible:  # (a stand-in: nothing can match)
                test = int_arg(0)
        elif te.dtype == DType.int64 and abs(x) >= 9007199254740992.0:
            # Past 2**53 several int64 values round to the same double.
            tc = own(cast_to(te, ST_FLOAT64))
        else:
            # float64 comparison: only an integral double in range matches.
            possible = x == x.__floor__() and x >= rng[0] and x <= rng[1]
            # (any integer stands in when nothing can match)
            test = int_arg(Int(x) if possible else 0)
    var eq = call_op("aten::eq", "Scalar", [tensor_arg(tc.t), test^], 1)
    _ = tc^  # alive past the call that reads it
    var eq_t = own(eq.take_tensor(0))
    var any = call_op("aten::any", "", [tensor_arg(eq_t.t)], 1)
    _ = eq_t^  # alive past the call that reads it
    var found = own(any.take_tensor(0))
    if not possible:
        # A 0-d False on the device: a bool never differs from itself.
        var none = call_op(
            "aten::ne", "Tensor", [tensor_arg(found.t), tensor_arg(found.t)], 1
        )
        _ = found^  # alive past the call that reads it
        found = own(none.take_tensor(0))
    if not invert:
        return found^
    var r = call_op("aten::logical_not", "", [tensor_arg(found.t)], 1)
    _ = found^  # alive past the call that reads it
    return own(r.take_tensor(0))


# aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
def op_isin_scalar_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out = _isin_scalar_tensor(
        args[unsafe_offset=0],
        v_tensor(args[unsafe_offset=1]),
        v_bool_or(args[unsafe_offset=3], False),
    )
    ret_owned(rets, 0, out)


# aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
def op_isin_scalar_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var te = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=4])
    var res = _isin_scalar_tensor(
        args[unsafe_offset=0], te, v_bool_or(args[unsafe_offset=3], False)
    )
    _ = _prepare_out(out_arg, res.t.shape, 0, ST_BOOL, te.device)
    copy_strided_into(out_arg, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, out_arg)


# ---------------------------------------------------------------------------
# where.self / masked_fill(_): the rank<=4 raw-pointer broadcast kernels
# (data_movement WhereSelect, MaskedFillScalar). These predate the
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
    var call = KernelCall("data_movement", "WhereSelect")
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
    var call = KernelCall("data_movement", "MaskedFillScalar")
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


def _c10_name(dt: DType) -> String:
    """The C++ type `c10::checked_convert` names in its overflow error."""
    if dt == DType.float32:
        return "float"
    if dt == DType.float64:
        return "double"
    if dt == DType.float16:
        return "c10::Half"
    if dt == DType.bfloat16:
        return "c10::BFloat16"
    if dt == DType.bool:
        return "bool"
    if dt == DType.int64:
        return "int64_t"
    if dt == DType.int32:
        return "int"
    if dt == DType.int16:
        return "int16_t"
    if dt == DType.int8:
        return "int8_t"
    return "uint8_t"


def scalar_as_fill(v: Value, dtype: DType) raises -> Float64:
    """`Scalar::to<scalar_t>()` for a fill value (c10's checked_convert): a
    bool destination takes the scalar's truth; an integer one goes through
    `scalar_to_int` (range-checked, a float truncates, an integer wraps into
    uint8) and a floating one through `scalar_to_float`; a complex scalar
    into a real type is the overflow error ATen raises. The result is
    exactly representable as a Float64 or declined."""
    if v.tag == TAG_COMPLEX:
        # c10::overflows<real, complex>: a nonzero imaginary part overflows,
        # otherwise the real part converts like a double Scalar.
        if bits_f64(v.b) != 0.0:
            raise Error(
                "value cannot be converted to type ",
                _c10_name(dtype),
                " without overflow",
            )
        return scalar_as_fill(Value(TAG_SCALAR_DOUBLE, 0, v.a, 0), dtype)
    if dtype == DType.bool:
        return 1.0 if v_f64(v) != 0.0 else 0.0
    var st = torch_dtype(dtype)
    if dtype.is_integral():
        var i = scalar_to_int(v, st)
        if dtype == DType.uint8:
            i = i & 0xFF
        if abs(i) > 9007199254740992:
            unsupported("scalar magnitude exceeds the exact float64 range")
        return Float64(i)
    if dtype.is_floating_point():
        return scalar_to_float(v, st)
    return scalar_embed(v, dtype)


def _masked_fill_scalar_dispatch(
    mask: T, a: T, value_arg: Value, dst: T
) raises:
    var value = scalar_as_fill(value_arg, a.dtype)
    if dst.numel == 0:
        return
    if _is_masked_fill_scalar_dtype(a.dtype) and mask.rank <= 4 and a.rank <= 4:
        _masked_fill_scalar_launch(mask, a, value, dst)
        return
    var fill = own(new_scalar(a.stype, a.device))
    fill_value(fill.t, value)
    _masked_fill_where(mask, fill.t, a, dst)
    _ = fill^  # alive past the launch


def _prepare_out_checked(
    mut out_arg: T,
    shape: IndexList[MAX_RANK],
    rank: Int,
    stype: Int32,
    device: Int,
) raises -> Bool:
    """`_prepare_out`, then copy_'s `assert_no_internal_overlap` on the
    resized `out` (an expanded `out` would collapse distinct results)."""
    var direct = _prepare_out(out_arg, shape, rank, stype, device)
    assert_no_internal_overlap(out_arg)
    return direct


def _shares_storage(dest: T, t: T) -> Bool:
    """`dest` and `t` live in one storage (resizing `dest` can move `t`)."""
    var sp = dest.storage_ptr()
    return sp != 0 and sp == t.storage_ptr()


def _masked_fill_expanded(a: T, mask: T) raises -> T:
    """`expand_outplace(mask, self)`'s self: the out-of-place masked_fill
    result has the broadcast shape of self and mask, so self is read through
    a 0-stride view over the dims it broadcasts along (a kernel-only view:
    it keeps self's handle)."""
    var shape = broadcast_shape(a, mask)
    var v = a.copy()
    v.strides = _bcast_strides(a, shape)
    v.shape = shape
    v.rank = max(a.rank, mask.rank)
    var n = 1
    for i in range(MAX_RANK):
        n *= shape[i]
    v.numel = n
    v.contig = False
    return v^


# aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
def op_masked_fill_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var ae = _masked_fill_expanded(a, mask)
    var out = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
    _masked_fill_scalar_dispatch(mask, ae, args[unsafe_offset=2], out.t)
    ret_owned(rets, 0, out)


# aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
def op_masked_fill_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=3])
    _masked_fill_validate(a, mask)
    var ae = _masked_fill_expanded(a, mask)
    if _shares_storage(out_arg, a) or _shares_storage(out_arg, mask):
        # The autogenerated out= overload's order: the functional result
        # first, then resize + copy. A resize of an `out` sharing storage
        # with an input may move that storage, so nothing reads an input
        # pointer after it.
        var res = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
        _masked_fill_scalar_dispatch(mask, ae, args[unsafe_offset=2], res.t)
        _ensure_out_shape(out_arg, ae.shape, ae.rank, a.stype, a.device)
        assert_no_internal_overlap(out_arg)
        copy_strided_into(out_arg, res.t)
        _ = res^  # alive past the launch
    elif _prepare_out_checked(out_arg, ae.shape, ae.rank, a.stype, a.device):
        _masked_fill_scalar_dispatch(mask, ae, args[unsafe_offset=2], out_arg)
    else:
        var tmp = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
        _masked_fill_scalar_dispatch(mask, ae, args[unsafe_offset=2], tmp.t)
        copy_strided_into(out_arg, tmp.t)
        _ = tmp^  # alive past the launch
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
        _ = tmp^  # alive past the launch
    ret_ref(rets, 0, a)


def _masked_fill_check_value(value_arg: Value) raises -> T:
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
    return val^


def _value_is_direct(a: T, val: T) -> Bool:
    """A 0-d value of self's dtype on self's device is read in place by the
    kernel; any other one (a CPU scalar tensor, another dtype) goes through
    `value.item()` like ATen's masked_fill does."""
    return val.dtype == a.dtype and val.on_mojo() and val.device == a.device


def _value_item(val: T) raises -> Value:
    var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(val)], 1)
    return r[0]


# aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
def op_masked_fill_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_check_value(args[unsafe_offset=2])
    var ae = _masked_fill_expanded(a, mask)
    var out = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
    if _value_is_direct(a, val):
        _masked_fill_where(mask, val, ae, out.t)
    else:
        _masked_fill_scalar_dispatch(mask, ae, _value_item(val), out.t)
    ret_owned(rets, 0, out)


def _masked_fill_value_into(mask: T, a: T, val: T, dst: T) raises:
    if _value_is_direct(a, val):
        _masked_fill_where(mask, val, a, dst)
    else:
        _masked_fill_scalar_dispatch(mask, a, _value_item(val), dst)


# aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
def op_masked_fill_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    var out_arg = v_tensor(args[unsafe_offset=3])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_check_value(args[unsafe_offset=2])
    var ae = _masked_fill_expanded(a, mask)
    if (
        _shares_storage(out_arg, a)
        or _shares_storage(out_arg, mask)
        or _shares_storage(out_arg, val)
    ):
        # Result first, then resize + copy (see op_masked_fill_scalar_out).
        var res = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
        _masked_fill_value_into(mask, ae, val, res.t)
        _ensure_out_shape(out_arg, ae.shape, ae.rank, a.stype, a.device)
        assert_no_internal_overlap(out_arg)
        copy_strided_into(out_arg, res.t)
        _ = res^  # alive past the launch
    elif _prepare_out_checked(out_arg, ae.shape, ae.rank, a.stype, a.device):
        _masked_fill_value_into(mask, ae, val, out_arg)
    else:
        var tmp = own(new_tensor(ae.shape, ae.rank, a.stype, a.device))
        _masked_fill_value_into(mask, ae, val, tmp.t)
        copy_strided_into(out_arg, tmp.t)
        _ = tmp^  # alive past the launch
    ret_ref(rets, 0, out_arg)


# aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
def op_masked_fill__tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    _masked_fill_validate(a, mask)
    var val = _masked_fill_check_value(args[unsafe_offset=2])
    if a.contig:
        _masked_fill_value_into(mask, a, val, a)
    else:
        var tmp = own(new_like(a))
        _masked_fill_value_into(mask, a, val, tmp.t)
        copy_strided_into(a, tmp.t)
        _ = tmp^  # alive past the launch
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# equal -- ATen's `cuda_equal` (native/cuda/Equal.cpp): different shapes are
# unequal, empty tensors equal, an identical view of one storage equal
# without a launch; otherwise `eq(self, other).all()` read back to the host
# (the op returns a host bool, so that one sync is inherent).
# ---------------------------------------------------------------------------


# aten::equal(Tensor self, Tensor other) -> bool
def op_equal(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    if not a.same_shape(b):
        ret_bool(rets, 0, False)
        return
    if a.numel == 0:
        ret_bool(rets, 0, True)
        return
    var same_view = (
        a.storage_ptr() != 0
        and a.storage_ptr() == b.storage_ptr()
        and a.offset == b.offset
        and a.stype == b.stype
        and a.contig == b.contig
    )
    if same_view:
        for i in range(a.rank):
            if a.stride(i) != b.stride(i):
                same_view = False
    if same_view:
        ret_bool(rets, 0, True)
        return
    var eq = call_op("aten::eq", "Tensor", [tensor_arg(a), tensor_arg(b)], 1)
    var all = call_op("aten::all", "", [eq[0]], 1)
    _ = eq^  # its handle was read by `all`
    var item = call_op("aten::_local_scalar_dense", "", [all[0]], 1)
    _ = all^
    ret_bool(rets, 0, v_f64(item[0]) != 0.0)


# ---------------------------------------------------------------------------
# searchsorted / bucketize: one binary search per value (searchsorted
# Searchsorted), shared by CPU/GPU. Ported from `_fast_searchsorted`,
# `fast_aten_searchsorted`, `fast_aten_bucketize`.
#
# `sorter` (searchsorted only) is checked the way `searchsorted_pre_check`
# (BucketizationUtils.h) checks it statically -- device, shape, dtype -- but
# NOT the way it checks values: torch's own check there is a device-to-host
# `aminmax().item()`, a sync we won't pay on every call. An out-of-range
# sorter entry is instead made harmless in the kernel itself (clamped into
# the valid boundary range), see `_binary_search_position` in
# `searchsorted.mojo`.
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
            "torch.searchsorted(): boundary and sorter must have the same size"
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
        var call = KernelCall("searchsorted", "Searchsorted")
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
    _ = computed^  # alive past the launch
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
    _ = values^  # alive past the launch
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
    _ = values^  # alive past the launch
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    _ = computed^  # alive past the launch
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
    _ = computed^  # alive past the launch
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
    _ = values^  # alive past the launch
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
    _ = values^  # alive past the launch
    _ensure_out_shape(
        out_arg,
        computed.t.shape,
        computed.t.rank,
        computed.t.stype,
        computed.t.device,
    )
    copy_strided_into(out_arg, computed.t)
    _ = computed^  # alive past the launch
    ret_ref(rets, 0, out_arg)


# ---------------------------------------------------------------------------
# _assert_async: TensorCompare.cpp's CPU form (a synchronous check). CUDA
# launches a device assert instead, which aborts the context; raising from a
# 1-element readback is the recoverable equivalent.
# ---------------------------------------------------------------------------


def _assert_nonzero(t: T, msg: String) raises:
    if t.numel == 0:
        raise Error("Boolean value of Tensor with no values is ambiguous")
    if t.numel > 1:
        raise Error(
            "Boolean value of Tensor with more than one value is ambiguous"
        )
    var item = call_op("aten::_local_scalar_dense", "", [tensor_arg(t)], 1)
    if not (v_f64(item[0]) != 0.0):
        raise Error(msg)


# aten::_assert_async(Tensor self) -> ()
def op_assert_async(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _assert_nonzero(
        v_tensor(args[unsafe_offset=0]),
        "Expected Tensor with single nonzero value, but got zero",
    )


# aten::_assert_async.msg(Tensor self, str assert_msg) -> ()
def op_assert_async_msg(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var msg = v_string(args[unsafe_offset=1])
    _assert_nonzero(
        v_tensor(args[unsafe_offset=0]),
        msg if msg.byte_length() > 0 else String("Assertion is failed"),
    )


# aten::_functional_assert_async.msg(Tensor self, str assert_msg,
#   Tensor dep_token) -> Tensor
def op_functional_assert_async_msg(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """ATen's CPU kernel (the only one upstream has): `_assert_async.msg`,
    then a fresh copy of the dependency token."""
    var msg = v_string(args[unsafe_offset=1])
    _assert_nonzero(
        v_tensor(args[unsafe_offset=0]),
        msg if msg.byte_length() > 0 else String("Assertion is failed"),
    )
    var token = call_op(
        "aten::clone",
        "",
        [tensor_arg(v_tensor(args[unsafe_offset=2])), none_arg()],
        1,
    )
    var out = own(token.take_tensor(0))
    ret_owned(rets, 0, out)


def register_compare(site: Site) raises:
    impl[op_assert_async, "_assert_async"](site)
    impl[op_assert_async_msg, "_assert_async.msg"](site)
    impl[op_functional_assert_async_msg, "_functional_assert_async.msg"](site)
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
    impl[op_isin_scalar_tensor, "isin.Scalar_Tensor"](site)
    impl[op_isin_scalar_tensor_out, "isin.Scalar_Tensor_out"](site)
    impl[op_isin_tensor_scalar, "isin.Tensor_Scalar"](site)
    impl[op_isin_tensor_scalar_out, "isin.Tensor_Scalar_out"](site)
    impl[op_isin_tensor_tensor, "isin.Tensor_Tensor"](site)
    impl[op_isin_tensor_tensor_out, "isin.Tensor_Tensor_out"](site)
    impl[op_where_self, "where.self"](site)
    impl[op_masked_fill_scalar, "masked_fill.Scalar"](site)
    impl[op_masked_fill_scalar_out, "masked_fill.Scalar_out"](site)
    impl[op_masked_fill_tensor, "masked_fill.Tensor"](site)
    impl[op_masked_fill_tensor_out, "masked_fill.Tensor_out"](site)
    impl[op_masked_fill__scalar, "masked_fill_.Scalar"](site)
    impl[op_masked_fill__tensor, "masked_fill_.Tensor"](site)
    impl[op_equal, "equal"](site)
    impl[op_searchsorted_tensor, "searchsorted.Tensor"](site)
    impl[op_searchsorted_tensor_out, "searchsorted.Tensor_out"](site)
    impl[op_searchsorted_scalar, "searchsorted.Scalar"](site)
    impl[op_searchsorted_scalar_out, "searchsorted.Scalar_out"](site)
    impl[op_bucketize_tensor, "bucketize.Tensor"](site)
    impl[op_bucketize_tensor_out, "bucketize.Tensor_out"](site)
    impl[op_bucketize_scalar, "bucketize.Scalar"](site)
    impl[op_bucketize_scalar_out, "bucketize.Scalar_out"](site)
