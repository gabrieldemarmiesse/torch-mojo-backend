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
    ST_FLOAT32,
    Owned,
    T,
    TAG_TENSOR,
    TAG_TENSOR_REF,
    Value,
    Values,
    dtype_name,
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
    device_str,
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
comptime MODE_COPY_OUT = 6  # functional result, then resize_out + exact dtype

# Checks an op makes before its TensorIterator.
comptime PRE_NONE = 0
comptime PRE_SUB = 1  # sub_check: no bool operand
comptime PRE_HEAVISIDE = 2  # equal operand (and out) dtypes
comptime PRE_BOOL_OUT = 3  # signbit / isposinf / isneginf: bool out only
comptime PRE_NAN_TO_NUM = 4  # out dtype == input dtype
comptime PRE_LDEXP = 5  # ldexp = self * pow(2, other), composite
comptime PRE_DIV_MODE = 6  # div's rounding_mode picks the kernel
comptime PRE_LERP = 7  # end / weight dtypes, the weight's rank picks the mode
comptime PRE_ALL_SAME = 8  # grad_output must have self's dtype
comptime PRE_DISPATCH_SELF = 9  # the kernel dispatches on self's dtype
comptime PRE_SELF_FIRST = 10  # the iterator adds self (arg 1) before arg 0
comptime PRE_SOFTSHRINK = 11  # softshrink_check dispatches on self first
comptime PRE_RELU = 12  # relu's own check: no bool self
comptime PRE_THRESHOLD = 13  # self is added twice (self and other)

comptime _MAX_ARGS = 16


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
    var shape: IndexList[MAX_RANK]
    var rank: Int
    var shape_ok: Bool
    var err: String  # the broadcast error, when not shape_ok


def _shape_of(args: Values, ins: Int) raises -> _Iter:
    """The operands' broadcast shape (`ins`: bitmask of operand args)."""
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    var ok = True
    var err = String("")
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 0:
            continue
        var v = args[unsafe_offset=i].copy()
        if not _rec_tensor(v):
            continue
        var t = T(Int(v.a))
        rank = max(rank, t.rank)
        # infer_size: the trailing dims first, as c10 reports them.
        for j in range(MAX_RANK):
            var k = MAX_RANK - 1 - j
            if shape[k] == 1:
                shape[k] = t.shape[k]
            elif t.shape[k] != 1 and t.shape[k] != shape[k]:
                if ok:
                    err = (
                        "The size of tensor a ("
                        + String(shape[k])
                        + ") must match the size of tensor b ("
                        + String(t.shape[k])
                        + ") at non-singleton dimension "
                        + String(k - (MAX_RANK - rank))
                    )
                ok = False
    return _Iter(shape, rank, ok, err)


def _common_of(args: Values, ins: Int, rev: Bool = False) raises -> Int32:
    """(a): c10's ResultTypeState over the tensor operands. A wrapped Python
    number cannot change a float8 result, so only tensors take part. Raises
    c10's float8 promotion error."""
    var dim = ST_UNDEFINED
    var zero = ST_UNDEFINED
    for j in range(_MAX_ARGS):
        var i = _MAX_ARGS - 1 - j if rev else j
        if (ins >> i) & 1 == 0:
            continue
        var v = args[unsafe_offset=i].copy()
        if not _rec_tensor(v):
            continue
        var t = T(Int(v.a))
        if t.rank == 0:
            if not t.is_wrapped_number():
                zero = promote_types(zero, t.stype)
        else:
            dim = promote_types(dim, t.stype)
    return combine_categories(dim, zero)


def _shape_is(t: T, shape: IndexList[MAX_RANK], rank: Int) -> Bool:
    if t.rank != rank:
        return False
    for i in range(MAX_RANK):
        if t.shape[i] != shape[i]:
            return False
    return True


def _common_device(args: Values, ins: Int, o: T) raises -> String:
    """TensorIterator's common device: the first operand, out first, that
    is not on the CPU."""
    if not o.on_cpu():
        return device_str(o)
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 0:
            continue
        var v = args[unsafe_offset=i].copy()
        if _rec_tensor(v):
            var t = T(Int(v.a))
            if not t.on_cpu():
                return device_str(t)
    return String("cpu")


def _first_device(args: Values, ins: Int) raises -> String:
    """The first operand not on the CPU ("" when all are)."""
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 0:
            continue
        var v = args[unsafe_offset=i].copy()
        if _rec_tensor(v):
            var t = T(Int(v.a))
            if not t.on_cpu():
                return device_str(t)
    return String("")


def _inputs_on_device(
    args: Values, ins: Int, dev: String, cpu_scalars: Bool = True
) raises:
    """Every input on `dev`, except one CPU 0-d scalar of an op with several
    operands (TensorIterator's allow_cpu_scalars); more than one such
    scalar is refused."""
    var n_ops = 0
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 1:
            n_ops += 1
    var scalars = 0
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 0:
            continue
        var v = args[unsafe_offset=i].copy()
        if not _rec_tensor(v):
            continue
        var t = T(Int(v.a))
        var d = device_str(t)
        if d == dev:
            continue
        if cpu_scalars and t.on_cpu() and t.rank == 0 and n_ops > 1:
            scalars += 1
            continue
        _device_error(dev, d)
    if scalars > 1:
        raise Error("Trying to pass too many CPU scalars to non-CPU kernel!")


def _device_error(a: String, b: String) raises:
    raise Error(
        (
            "Expected all tensors to be on the same device, but found at least"
            " two devices, "
        ),
        a,
        " and ",
        b,
        "!",
    )


def _shape_str(shape: IndexList[MAX_RANK], rank: Int) -> String:
    var s = String("[")
    for i in range(rank):
        if i > 0:
            s += ", "
        s += String(shape[MAX_RANK - rank + i])
    return s + "]"


@fieldwise_init
struct _Pre(Movable):
    var mode: Int  # -1: the op's own
    var ins: Int  # the operand mask after the op's own rules
    var rev: Bool  # promote the operands last to first
    var kernel: String  # "" or the kernel the op dispatches instead
    var shape_ins: Int  # the iterator's operands: shape, overlap, devices
    var cpu_scalars: Bool  # the iterator allows one CPU 0-d operand


def _pre[
    pre: Int, ins: Int, out_idx: Int, label: StaticString
](args: Values) raises -> _Pre:
    """The checks an op makes before building its iterator, on the
    operands' ORIGINAL dtypes."""
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
        # BinaryOps.cpp: ldexp_out rejects an integral out, then computes
        # mul(self, _pow2(self, other)); _pow2 is pow(2.0, other) for an
        # integral or float32 self (other's floating dtype, or float32), else
        # full({}, 2.0, self.dtype).pow(other) with the usual promotion.
        var a = T(Int(args[unsafe_offset=0].a))
        var b = T(Int(args[unsafe_offset=1].a))
        comptime if out_idx >= 0:
            var o = T(Int(args[unsafe_offset=out_idx].a))
            if not is_float_stype(o.stype) and not is_float8_stype(o.stype):
                raise Error(
                    "ldexp can't be cast to the desired output type ",
                    _tname(o.stype),
                )
        var b_float = is_float_stype(b.stype) or is_float8_stype(b.stype)
        if not b_float and is_float8_stype(a.stype):
            # _ldexp_int_exponent: an integral (or bool) exponent of a
            # floating self runs ldexp_cuda's own iterator (out checked and
            # resized first), which has no float8 case.
            # Its iterator (check_all_same_dtype(false), no CPU scalars)
            # promotes nothing: self's dtype dispatches, both operands
            # broadcast and must share a device.
            return _Pre(-1, 1, False, "ldexp_cuda", ins, False)
        var pdt: Int32
        if not (is_float_stype(a.stype) or is_float8_stype(a.stype)) or (
            a.dtype == DType.float32
        ):
            pdt = b.stype if b_float else ST_FLOAT32
        elif b.rank == 0:
            pdt = promote_types(a.stype, b.stype)
        else:
            pdt = combine_categories(b.stype, a.stype)
        if is_float8_stype(pdt):
            _refuse("pow_cuda", pdt)
        # mul(self, p), p of other's rank: a read-write out's shape first.
        comptime if out_idx == 0:
            var it = _shape_of(args, 3)
            if it.shape_ok and not _shape_is(a, it.shape, it.rank):
                raise Error(
                    "output with shape ",
                    _shape_str(a.shape, a.rank),
                    " doesn't match the broadcast shape ",
                    _shape_str(it.shape, it.rank),
                )
        if a.rank > 0 and b.rank > 0:
            _ = promote_types(a.stype, pdt)
        elif a.rank == 0 and b.rank == 0:
            _ = promote_types(a.stype, pdt)
        elif a.rank > 0:
            _ = combine_categories(a.stype, pdt)
        else:
            _ = combine_categories(pdt, a.stype)
    elif pre == PRE_LERP:
        # lerp's meta: `end` (and a dimensioned `weight`) must have self's
        # dtype; such a weight joins the all-same-dtype iterator (out must
        # match), a 0-d one promotes like any operand (out casts).
        var a = T(Int(args[unsafe_offset=0].a))
        var b = T(Int(args[unsafe_offset=1].a))
        if b.stype != a.stype:
            raise Error(
                "expected dtype ",
                dtype_name(a.stype),
                " for `end` but got dtype ",
                dtype_name(b.stype),
            )
        var w = args[unsafe_offset=2].copy()
        if _rec_tensor(w):
            var wt = T(Int(w.a))
            if wt.rank > 0:
                if wt.stype != a.stype:
                    raise Error(
                        "expected dtype ",
                        dtype_name(a.stype),
                        " for `weight` but got dtype ",
                        dtype_name(wt.stype),
                    )
                return _Pre(MODE_SAME, ins, False, "", ins, True)
        return _Pre(MODE_CAST, ins, False, "", ins, True)
    elif pre == PRE_ALL_SAME:
        # log_sigmoid_backward: grad_output must have self's dtype (the
        # buffer is not an operand on CUDA).
        var g = T(Int(args[unsafe_offset=0].a))
        var a = T(Int(args[unsafe_offset=1].a))
        comptime if out_idx >= 0:
            # The iterator's operands: out, self, grad_output.
            var o = T(Int(args[unsafe_offset=out_idx].a))
            if o.stype != a.stype:
                raise Error(
                    "Found dtype ",
                    _tname(o.stype),
                    " but expected ",
                    _tname(a.stype),
                )
        if g.stype != a.stype:
            raise Error(
                "Found dtype ",
                _tname(g.stype),
                " but expected ",
                _tname(a.stype),
            )
    elif pre == PRE_SOFTSHRINK:
        # softshrink_check: an AT_DISPATCH over self's dtype, before the
        # iterator (and its out= checks) exists.
        # AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16).
        var a = T(Int(args[unsafe_offset=0].a))
        if not is_float_stype(a.stype):
            _refuse("softshrink_check", a.stype)
    elif pre == PRE_THRESHOLD:
        # threshold's iterator adds self twice (self, other): a CPU 0-d self
        # is two CPU scalars for a device out.
        comptime if out_idx >= 0:
            var a = T(Int(args[unsafe_offset=0].a))
            var o = T(Int(args[unsafe_offset=out_idx].a))
            if a.on_cpu() and a.rank == 0 and not o.on_cpu():
                raise Error(
                    "Trying to pass too many CPU scalars to non-CPU kernel!"
                )
    elif pre == PRE_RELU:
        if T(Int(args[unsafe_offset=0].a)).dtype == DType.bool:
            raise Error("Boolean inputs not supported for relu")
    elif pre == PRE_SELF_FIRST:
        return _Pre(-1, ins, True, "", ins, True)
    return _Pre(-1, ins, False, "", ins, True)


def f8_iter[
    op: OpFn,
    kernel: StaticString,
    ins: Int,
    out_idx: Int,
    mode: Int,
    int_only: Bool = False,
    pre: Int = PRE_NONE,
    label: StaticString = "",
    overlap: Bool = True,
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`ins`: bitmask of the operand args; `out_idx`: the out= (or, inside
    `ins`, the in-place self) arg, -1 for none."""
    var touched = False
    for i in range(_MAX_ARGS):
        if (ins >> i) & 1 == 1 and _f8_rec(args[unsafe_offset=i]):
            touched = True
    comptime if out_idx >= 0:
        if _f8_rec(args[unsafe_offset=out_idx]):
            touched = True
    if not touched:
        op(args, n_args, rets, n_rets)
        return
    var p = _pre[pre, ins, out_idx, label](args)
    var eff_mode = mode if p.mode < 0 else p.mode
    var it = _shape_of(args, p.shape_ins)
    var compute: Int32
    comptime if mode == MODE_COPY_OUT:
        # A composite out=: the functional result first (its dispatch
        # refuses with out untouched), then resize_output, then the exact
        # dtype check of the copy.
        if not it.shape_ok:
            raise Error(it.err)
        compute = _common_of(args, p.ins, p.rev)
        if is_float8_stype(compute):
            _refuse(String(kernel), compute)
        var o = T(Int(args[unsafe_offset=out_idx].a))
        if not _shape_is(o, it.shape, it.rank):
            resize_out(o, it.shape, it.rank)
        if o.stype != compute:
            raise Error(
                "Expected out tensor to have dtype ",
                dtype_name(compute),
                ", but got ",
                dtype_name(o.stype),
                " instead",
            )
    elif out_idx >= 0:
        var o = T(Int(args[unsafe_offset=out_idx].a))
        comptime inplace = (ins >> out_idx) & 1 == 1
        # (b) memory overlap, then the shape of a read-write out: one that
        # is also an operand is never resized.
        comptime if overlap:
            assert_no_internal_overlap(o)
        var read_write = inplace
        for i in range(_MAX_ARGS):
            if (p.shape_ins >> i) & 1 == 0:
                continue
            comptime if inplace:
                if i == out_idx:
                    continue
            var v = args[unsafe_offset=i].copy()
            if _rec_tensor(v):
                var t = T(Int(v.a))
                if t.impl() == o.impl():
                    read_write = True
                elif not t.on_cpu():
                    comptime if overlap:
                        assert_no_partial_overlap(o, t)
        # compute_shape: a broadcast error, then a read-write out's shape.
        if not it.shape_ok:
            raise Error(it.err)
        if read_write and not _shape_is(o, it.shape, it.rank):
            raise Error(
                "output with shape ",
                _shape_str(o.shape, o.rank),
                " doesn't match the broadcast shape ",
                _shape_str(it.shape, it.rank),
            )
        # (a) + (c), with the device check where TensorIterator's
        # compute_types makes it: after an exact-dtype out's check, before
        # canCast's.
        var common = _common_of(args, p.ins, p.rev)
        if eff_mode == MODE_SAME:
            if o.stype != common:
                raise Error(
                    "Found dtype ",
                    _tname(o.stype),
                    " but expected ",
                    _tname(common),
                )
        var cdev = _common_device(args, p.shape_ins, o)
        if device_str(o) != cdev:
            _device_error(cdev, device_str(o))
        if eff_mode != MODE_SAME and eff_mode != MODE_NOCHECK:
            if not can_cast(common, o.stype):
                raise Error(
                    "result type ",
                    _tname(common),
                    " can't be cast to the desired output type ",
                    _tname(o.stype),
                )
        _inputs_on_device(args, p.shape_ins, cdev, p.cpu_scalars)
        # (d)
        if (
            not read_write
            and it.shape_ok
            and not _shape_is(o, it.shape, it.rank)
        ):
            resize_out(o, it.shape, it.rank)
        compute = o.stype if eff_mode == MODE_OUTDT else common
    else:
        if not it.shape_ok:
            raise Error(it.err)
        compute = _common_of(args, p.ins, p.rev)
        var fdev = _first_device(args, p.shape_ins)
        if fdev.byte_length() > 0:
            _inputs_on_device(args, p.shape_ins, fdev, p.cpu_scalars)
    # (e)
    var k = String(kernel) if p.kernel.byte_length() == 0 else p.kernel
    comptime if pre == PRE_DIV_MODE:
        var m = args[unsafe_offset=2].copy()
        if not v_is_none(m):
            var name = v_string(m)
            if name == "floor":
                k = "div_floor_cuda"
            elif name == "trunc":
                k = "div_trunc_cuda"
    comptime if pre == PRE_DISPATCH_SELF:
        # The kernel dispatches on self's dtype (arg 1), not the common one.
        var a = T(Int(args[unsafe_offset=1].a))
        if is_float8_stype(a.stype):
            _refuse(k, a.stype)
    if is_float8_stype(compute):
        _refuse(k, compute)
    comptime if int_only:
        if is_float_stype(compute):
            _refuse(k, compute)
    # A float8 operand of a wider computation: converted exactly first. A
    # kernel dispatched on the out dtype computes in it: every operand is
    # converted to it, as TensorIterator's common-dtype cast does.
    var all_ops = eff_mode == MODE_OUTDT
    var recs = List[Value](capacity=n_args)
    var held = List[Owned]()
    for i in range(n_args):
        var v = args[unsafe_offset=i].copy()
        if i < _MAX_ARGS and (p.ins >> i) & 1 == 1 and _rec_tensor(v):
            var t = T(Int(v.a))
            var swap = i != out_idx
            var convert = is_float8_stype(t.stype) or (
                all_ops and t.stype != compute
            )
            if swap and convert and not t.on_cpu():
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
