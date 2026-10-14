"""TensorIterator's pre-dispatch sequence for an elementwise op that sees a
float8 tensor, in stock CUDA's order, shared by every such op.

Each registration of a TensorIterator-built op (unary, binary, comparison,
logical, pointwise; functional, `out=` and in-place) wraps its kernel in
`f8_iter[op, kernel, n_in, out_idx, mode, int_only, pre]`. A call with no
float8 operand or output goes straight to `op`. Otherwise, in order:

(a) the common dtype of the tensor operands (c10's ResultTypeState; a 0-dim
    or CPU 0-dim operand ranks below a dimensioned one), which raises c10's
    "Promotion for Float8 Types is not supported" RuntimeError;
(b) the out= checks: at::assert_no_internal_overlap on out and
    at::assert_no_partial_overlap against every input;
(c) the out dtype check: `mode` says which (see the MODE_* constants);
(d) out is resized to the broadcast shape (never for an in-place op);
(e) the kernel dispatch: a float8 computation dtype -- the common dtype, or
    the out dtype for kernels dispatched on it -- refuses as
    `"<kernel>" not implemented for '<dtype>'` (NotImplementedError);
    otherwise every float8 device operand is converted to the computation
    dtype (exactly: a promotion, never saturating) and `op` computes.

The modes and kernel names were measured on stock CUDA (torch 2.11, H100)
for every registered op; tests/native/test_float8.py pins them.
"""
from std.ffi import external_call

from tmb.backend.abi import (
    TAG_BOOL,
    TAG_SCALAR_BOOL,
    OpFn,
    Owned,
    T,
    TAG_TENSOR,
    TAG_TENSOR_REF,
    Value,
    Values,
    is_float8_stype,
    max_dtype,
    own,
    tensor_arg,
    unsupported,
    v_is_none,
    v_string,
)
from std.utils import IndexList
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    ST_UNDEFINED,
    assert_no_internal_overlap,
    assert_no_partial_overlap,
    can_cast,
    cast_to,
    combine_categories,
    is_float_stype,
    promote_types,
    resize_out,
    scalar_type_name,
)

# out= dtype rule of the op's TensorIterator (MODE_FN: no out, functional).
comptime MODE_FN = 0
comptime MODE_CAST = 1  # canCast(common, out); computes in common
comptime MODE_SAME = 2  # out must hold the common dtype ("Found dtype")
comptime MODE_NOCHECK = 3  # any out (a bool result casts anywhere)
comptime MODE_OUTDT = 4  # canCast; the kernel dispatches on out's dtype
comptime MODE_INPLACE = 5  # self is out: canCast(common, self)

# Checks an op makes before its TensorIterator.
comptime PRE_NONE = 0
comptime PRE_SUB = 1  # sub_check: no bool operand
comptime PRE_HEAVISIDE = 2  # equal operand (and out) dtypes
comptime PRE_BOOL_OUT = 3  # signbit / isposinf / isneginf: bool out only
comptime PRE_NAN_TO_NUM = 4  # out dtype == input dtype
comptime PRE_LDEXP = 5  # ldexp = self * pow(2, other), composite
comptime PRE_DIV_MODE = 6  # div's rounding_mode picks the kernel


def _rec_tensor(v: Value) -> Bool:
    return v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF


def _f8_rec(v: Value) -> Bool:
    """A float8 tensor record: one dtype query, not a full tensor read (this
    runs on every call of every wrapped op)."""
    if not _rec_tensor(v):
        return False
    return is_float8_stype(external_call["tmb_tensor_dtype", Int32](Int(v.a)))


def _tname(st: Int32) raises -> String:
    return scalar_type_name(max_dtype(st))


def _refuse(kernel: String, st: Int32) raises:
    unsupported('"' + kernel + "\" not implemented for '" + _tname(st) + "'")


@fieldwise_init
struct _Iter(Movable):
    var common: Int32
    var shape: IndexList[MAX_RANK]
    var rank: Int
    var shape_ok: Bool


def _iter_of(args: Values, n_in: Int) raises -> _Iter:
    """(a): the common dtype of the tensor operands and their broadcast
    shape. A wrapped Python number cannot change a float8 result, so only
    tensors take part."""
    var dim = ST_UNDEFINED
    var zero = ST_UNDEFINED
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    var ok = True
    for i in range(n_in):
        var v = args[unsafe_offset=i].copy()
        if not _rec_tensor(v):
            continue
        var t = T(Int(v.a))
        if t.rank == 0:
            if not t.is_wrapped_number():
                zero = promote_types(zero, t.stype)
        else:
            dim = promote_types(dim, t.stype)
        rank = max(rank, t.rank)
        for k in range(MAX_RANK):
            if shape[k] == 1:
                shape[k] = t.shape[k]
            elif t.shape[k] != 1 and t.shape[k] != shape[k]:
                ok = False
    return _Iter(combine_categories(dim, zero), shape, rank, ok)


def _shape_is(t: T, shape: IndexList[MAX_RANK], rank: Int) -> Bool:
    if t.rank != rank:
        return False
    for i in range(MAX_RANK):
        if t.shape[i] != shape[i]:
            return False
    return True


def _pre[
    pre: Int, n_in: Int, out_idx: Int, label: StaticString
](args: Values) raises:
    """The checks an op makes before building its iterator."""
    comptime if pre == PRE_SUB:
        for i in range(2):
            var v = args[unsafe_offset=i].copy()
            var is_bool: Bool
            if _rec_tensor(v):
                is_bool = T(Int(v.a)).dtype == DType.bool
            else:
                is_bool = v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL
            if is_bool:
                raise Error(
                    "Subtraction, the `-` operator, with a bool tensor is not"
                    " supported. If you are trying to invert a mask, use the"
                    " `~` or `logical_not()` operator instead."
                )
    elif pre == PRE_HEAVISIDE:
        var a = T(Int(args[unsafe_offset=0].a))
        var b = T(Int(args[unsafe_offset=1].a))
        var same = a.stype == b.stype
        comptime if out_idx >= 0:
            same = (
                same and T(Int(args[unsafe_offset=out_idx].a)).stype == a.stype
            )
        if not same:
            raise Error(
                "heaviside is not yet implemented for tensors with different"
                " dtypes."
            )
    elif pre == PRE_BOOL_OUT:
        comptime if out_idx >= 0:
            if T(Int(args[unsafe_offset=out_idx].a)).dtype != DType.bool:
                raise Error(
                    String(label), " does not support non-boolean outputs."
                )
    elif pre == PRE_NAN_TO_NUM:
        comptime if out_idx >= 0:
            var a = T(Int(args[unsafe_offset=0].a))
            var o = T(Int(args[unsafe_offset=out_idx].a))
            if o.stype != a.stype:
                raise Error(
                    "nan_to_num: dtype of out: ",
                    _tname(o.stype),
                    " should be same as input: ",
                    _tname(a.stype),
                )
    elif pre == PRE_LDEXP:
        # ldexp(self, other) = self * pow(2, other): pow(2, other) runs in
        # other's dtype first; a dimensioned float8 self then meets that
        # result's dtype in mul's promotion.
        var a = T(Int(args[unsafe_offset=0].a))
        var b = T(Int(args[unsafe_offset=1].a))
        comptime if out_idx >= 0:
            var o = T(Int(args[unsafe_offset=out_idx].a))
            var it = _iter_of(args, 2)
            if not can_cast(it.common, o.stype):
                raise Error(
                    "ldexp can't be cast to the desired output type ",
                    _tname(o.stype),
                )
        if is_float8_stype(b.stype):
            _refuse("pow_cuda", b.stype)
        if is_float8_stype(a.stype) and a.rank > 0 and b.stype != a.stype:
            _ = promote_types(a.stype, b.stype)


def f8_iter[
    op: OpFn,
    kernel: StaticString,
    n_in: Int,
    out_idx: Int,
    mode: Int,
    int_only: Bool = False,
    pre: Int = PRE_NONE,
    label: StaticString = "",
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var touched = False
    for i in range(n_in):
        if _f8_rec(args[unsafe_offset=i]):
            touched = True
    comptime if out_idx >= 0:
        if _f8_rec(args[unsafe_offset=out_idx]):
            touched = True
    if not touched:
        op(args, n_args, rets, n_rets)
        return
    _pre[pre, n_in, out_idx, label](args)
    var it = _iter_of(args, n_in)  # (a), raises c10's promotion error
    var compute = it.common
    comptime if out_idx >= 0:
        var o = T(Int(args[unsafe_offset=out_idx].a))
        comptime inplace = out_idx < n_in
        # (b)
        assert_no_internal_overlap(o)
        for i in range(n_in):
            comptime if inplace:
                if i == out_idx:
                    continue
            var v = args[unsafe_offset=i].copy()
            if _rec_tensor(v):
                var t = T(Int(v.a))
                if not t.on_cpu():
                    assert_no_partial_overlap(o, t)
        # (c)
        comptime if mode == MODE_SAME:
            if o.stype != it.common:
                raise Error(
                    "Found dtype ",
                    _tname(o.stype),
                    " but expected ",
                    _tname(it.common),
                )
        elif mode != MODE_NOCHECK:
            if not can_cast(it.common, o.stype):
                raise Error(
                    "result type ",
                    _tname(it.common),
                    " can't be cast to the desired output type ",
                    _tname(o.stype),
                )
        # (d)
        comptime if not inplace:
            if it.shape_ok and not _shape_is(o, it.shape, it.rank):
                resize_out(o, it.shape, it.rank)
        comptime if mode == MODE_OUTDT:
            compute = o.stype
    # (e)
    var k = String(kernel)
    comptime if pre == PRE_DIV_MODE:
        var m = args[unsafe_offset=2].copy()
        if not v_is_none(m):
            var name = v_string(m)
            if name == "floor":
                k = "div_floor_cuda"
            elif name == "trunc":
                k = "div_trunc_cuda"
    if is_float8_stype(compute):
        _refuse(k, compute)
    comptime if int_only:
        if is_float_stype(compute):
            _refuse(k, compute)
    # A float8 operand of a wider computation: converted exactly first.
    var recs = List[Value](capacity=n_args)
    var held = List[Owned]()
    for i in range(n_args):
        var v = args[unsafe_offset=i].copy()
        if i < n_in and _rec_tensor(v):
            var t = T(Int(v.a))
            var swap: Bool
            comptime if out_idx >= 0 and out_idx < n_in:
                swap = i != out_idx
            else:
                swap = True
            if swap and is_float8_stype(t.stype) and not t.on_cpu():
                held.append(own(cast_to(t, compute, False)))
                recs.append(tensor_arg(held[len(held) - 1].t))
                continue
        recs.append(v^)
    op(
        recs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        n_args,
        rets,
        n_rets,
    )
    _ = held^
    _ = recs^
