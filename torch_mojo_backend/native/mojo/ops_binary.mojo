"""aten ops: binary arithmetic — add/sub/mul/div, pow, maximum/minimum,
remainder, floor_divide, lerp, addcmul/addcdiv, clamp and the logical and
bitwise ops, with their in-place and `out=` variants.

Ported from the old Python fast path (`eager_kernels/aten_fast.py`): the same
route cascade (scalar spec, int-scalar spec, the fp32+bf16 add spec, the
broadcast binary spec), the same dtype-promotion table, the same scalar
embedding rules and the same declines. Every route is gated on tensor
metadata *here*, so a kernel only ever sees inputs it accepts and nothing has
to be retried after a raise; the last route of a cascade calls
`unsupported(...)` exactly where the old path returned NOT_HANDLED.

Two families do the work: `logic_ops` (the broadcast-strided binary and
ternary kernels) and `elementwise_ops` (the contiguous tensor-with-scalar
kernels and the raw contiguous `Add`).

The `_b_`-prefixed helpers (scalar records, dtype promotion, dtype
predicates, temporaries) are private to this file only to keep the port
conflict-free; they are generic and belong in ops_common.mojo.
"""
from std.utils import IndexList

from abi import (
    DEVICE_TYPE_CPU,
    ST_BFLOAT16,
    ST_BOOL,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT8,
    ST_INT16,
    ST_INT32,
    ST_INT64,
    ST_UINT8,
    T,
    TAG_BOOL,
    TAG_COMPLEX,
    TAG_DOUBLE,
    TAG_INT,
    TAG_NONE,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    TAG_TENSOR_REF,
    Value,
    Values,
    default_dtype,
    dtype_code,
    f64_bits,
    new_like,
    new_scalar,
    new_tensor,
    own,
    release,
    ret_ref,
    ret_tensor,
    unsupported,
    v_f64,
    v_is_none,
    v_string,
    v_tensor,
)
from device import copy_d2d, ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import cast_to, contiguous, copy_strided_into, resize_out
from registry import Site, impl, op_address_of

# ---------------------------------------------------------------------------
# dtype predicates: the gates of the kernels this file calls
# ---------------------------------------------------------------------------


def _b_float3(st: Int32) -> Bool:
    """op_utils.FLOAT_DTYPES: what the scalar/elementwise kernels dispatch on.
    """
    return st == ST_FLOAT32 or st == ST_FLOAT16 or st == ST_BFLOAT16


def _b_castable(st: Int32) -> Bool:
    """data_movement_ops CAST_DTYPES (the old `_CAST_DTYPES`)."""
    return (
        _b_float3(st)
        or st == ST_INT64
        or st == ST_INT32
        or st == ST_UINT8
        or st == ST_BOOL
    )


def _b_bcast_dtype(st: Int32) -> Bool:
    """logic_ops SPEC_BCAST_DTYPES (bool travels as uint8, see `_b_binary`)."""
    return (
        _b_float3(st)
        or st == ST_FLOAT64
        or st == ST_INT8
        or st == ST_INT16
        or st == ST_INT32
        or st == ST_INT64
        or st == ST_UINT8
    )


def _b_fillable(st: Int32) -> Bool:
    """elementwise_ops SPEC_FILL_DTYPES (the old `_FILL_DTYPES`)."""
    return _b_bcast_dtype(st) or st == ST_BOOL


def _b_int_dtype(st: Int32) -> Bool:
    return (
        st == ST_INT8
        or st == ST_INT16
        or st == ST_INT32
        or st == ST_INT64
        or st == ST_UINT8
    )


def _b_category(st: Int32) -> Int:
    """torch's promotion category: bool < integral < floating."""
    if st == ST_BOOL:
        return 0
    if _b_float3(st) or st == ST_FLOAT64:
        return 2
    return 1


def _b_can_cast(src: Int32, dst: Int32) -> Bool:
    """torch.can_cast: a cast that does not narrow the dtype category."""
    return _b_category(src) <= _b_category(dst)


def _b_promote(a: Int32, b: Int32) -> Int32:
    """The old `_binary_promotion`, as the promoted dtype (-1 = decline).

    An operand whose dtype differs from the result is cast; the table covers
    only the pairs the eager loops hit, everything else declines.
    """
    if a == b:
        return a
    if a == ST_BOOL and _b_castable(b):
        return b
    if b == ST_BOOL and _b_castable(a):
        return a
    if (a == ST_INT32 and b == ST_INT64) or (a == ST_INT64 and b == ST_INT32):
        return ST_INT64
    if a == ST_FLOAT32 and (b == ST_FLOAT16 or b == ST_BFLOAT16):
        return ST_FLOAT32
    if b == ST_FLOAT32 and (a == ST_FLOAT16 or a == ST_BFLOAT16):
        return ST_FLOAT32
    if (a == ST_FLOAT16 and b == ST_BFLOAT16) or (
        a == ST_BFLOAT16 and b == ST_FLOAT16
    ):
        return ST_FLOAT32
    return Int32(-1)


# ---------------------------------------------------------------------------
# operands: a Scalar record, a mojo tensor, or torch's 0-d CPU wrapped number
# ---------------------------------------------------------------------------


@fieldwise_init
struct Scal(Copyable, Movable):
    """A Scalar operand: its value both ways, plus torch's isIntegral(true)."""

    var f: Float64
    var i: Int
    var is_int: Bool
    var is_bool: Bool


comptime MAX_EXACT_INT = 9007199254740992.0  # 2**53, the old `_MAX_EXACT_INT`


def _b_host_scalar(t: T) raises -> Scal:
    """The value of a 0-d CPU tensor, read straight out of host memory."""
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=t.ptr)
    if t.dtype == DType.bool:
        var on = p[] != 0
        return Scal(1.0 if on else 0.0, 1 if on else 0, True, True)
    if t.dtype == DType.float32:
        var v32 = Float64(p.unsafe_bitcast[Float32]()[])
        return Scal(v32, Int(v32), False, False)
    if t.dtype == DType.float64:
        var v64 = p.unsafe_bitcast[Float64]()[]
        return Scal(v64, Int(v64), False, False)
    if t.dtype == DType.bfloat16:
        var vb = Float64(p.unsafe_bitcast[BFloat16]()[])
        return Scal(vb, Int(vb), False, False)
    if t.dtype == DType.float16:
        var vh = Float64(p.unsafe_bitcast[Float16]()[])
        return Scal(vh, Int(vh), False, False)
    var k: Int = 0
    if t.dtype == DType.int64:
        k = Int(p.unsafe_bitcast[Int64]()[])
    elif t.dtype == DType.int32:
        k = Int(p.unsafe_bitcast[Int32]()[])
    elif t.dtype == DType.int16:
        k = Int(p.unsafe_bitcast[Int16]()[])
    elif t.dtype == DType.int8:
        k = Int(p.unsafe_bitcast[Int8]()[])
    elif t.dtype == DType.uint8:
        k = Int(p[])
    elif t.dtype == DType.uint16:
        k = Int(p.unsafe_bitcast[UInt16]()[])
    elif t.dtype == DType.uint32:
        k = Int(p.unsafe_bitcast[UInt32]()[])
    elif t.dtype == DType.uint64:
        k = Int(p.unsafe_bitcast[UInt64]()[])
    else:
        unsupported("a scalar tensor of dtype " + String(t.dtype))
    return Scal(Float64(k), k, True, False)


@fieldwise_init
struct Side(Copyable, Movable):
    """One operand of a binary op: a mojo tensor or a Scalar, never both.

    `is_t` says which; the Optionals are only unwrapped through it (testing
    one directly would copy it, and neither payload is ImplicitlyCopyable).
    """

    var is_t: Bool
    var t: Optional[T]
    var s: Optional[Scal]


def _b_tside(t: T) -> Side:
    return Side(True, Optional[T](t.copy()), Optional[Scal]())


def _b_sside(s: Scal) -> Side:
    return Side(False, Optional[T](), Optional[Scal](s.copy()))


def _b_side(v: Value) raises -> Side:
    """Read one operand record.

    A `Tensor` argument that is a 0-d CPU tensor is torch's *wrapped number*:
    `x + 2` reaches a backend kernel as `add.Tensor(x, tensor(2))`, because
    the python arg parser wraps a number passed where the schema says Tensor.
    The old path never saw that (a `__torch_dispatch__` subclass is handed the
    number itself), so the scalar routes below would never fire if this were
    taken for a real operand. Reading it back as a Scalar also reproduces
    torch's promotion of a wrapped number: it takes the other operand's
    dtype, which is exactly what `_scalar_embed` did.
    """
    if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
        var t = v_tensor(v)
        if t.on_mojo():
            return _b_tside(t)
        if t.device_type == DEVICE_TYPE_CPU and t.numel == 1 and t.rank == 0:
            return _b_sside(_b_host_scalar(t))
        unsupported("an operand that is neither a mojo tensor nor a scalar")
    if v.tag == TAG_COMPLEX:
        unsupported("complex scalars")
    if v.tag == TAG_NONE:
        raise Error("missing operand")
    if v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL:
        var on = v.a != 0
        return _b_sside(Scal(1.0 if on else 0.0, 1 if on else 0, True, True))
    if v.tag == TAG_SCALAR_INT or v.tag == TAG_INT:
        return _b_sside(Scal(Float64(v.a), Int(v.a), True, False))
    if v.tag == TAG_SCALAR_DOUBLE or v.tag == TAG_DOUBLE:
        var f = v_f64(v)
        return _b_sside(Scal(f, Int(f), False, False))
    raise Error("unexpected operand record tag ", v.tag)


def _b_scalar_is_int(v: Value) -> Bool:
    return (
        v.tag == TAG_SCALAR_INT
        or v.tag == TAG_INT
        or v.tag == TAG_SCALAR_BOOL
        or v.tag == TAG_BOOL
    )


def _b_embed(s: Scal, st: Int32) raises -> Float64:
    """The old `_scalar_embed`: `s` validated for lossless embedding in `st`."""
    if not _b_fillable(st):
        unsupported("a scalar operand against a tensor of dtype " + String(st))
    if s.is_int:
        if abs(s.f) > MAX_EXACT_INT:
            unsupported("an integer scalar too large to embed exactly")
        if st == ST_BOOL and s.f != 0.0 and s.f != 1.0:
            unsupported("a non-boolean scalar against a bool tensor")
    elif not (_b_float3(st) or st == ST_FLOAT64):
        # A float scalar promotes an integer tensor to float in torch.
        unsupported("a float scalar against an integer tensor")
    return s.f


# ---------------------------------------------------------------------------
# temporaries and small tensor plumbing
# ---------------------------------------------------------------------------


struct Held(Movable):
    """An operand that may be a fresh temporary (materialized contiguous,
    cast, or a 0-d scalar fill): released when the op returns, unless it is
    one of the caller's own tensors."""

    var t: T
    var tmp: Bool

    def __init__(out self, var t: T, tmp: Bool):
        self.t = t^
        self.tmp = tmp

    def __deinit__(deinit self):
        if self.tmp:
            release(self.t.h)


def _b_hold(t: T) -> Held:
    return Held(t.copy(), False)


def _b_cast(t: T, st: Int32) raises -> Held:
    if t.stype == st:
        return Held(t.copy(), False)
    if not (_b_castable(t.stype) and _b_castable(st)):
        unsupported(
            "a cast from dtype "
            + String(t.stype)
            + " to "
            + String(st)
            + " on the mojo device"
        )
    return Held(cast_to(t, st), True)


def _b_ready(t: T, st: Int32, need_contig: Bool) raises -> Held:
    """`t` as the kernel wants it: in dtype `st` (a cast materializes a
    contiguous copy on its own) and contiguous when the caller needs it."""
    if t.stype != st:
        return _b_cast(t, st)
    if need_contig and not t.contig:
        return Held(contiguous(t), True)
    return Held(t.copy(), False)


def _b_fill_spec(dst: T, value: Float64) raises:
    """elementwise_ops FillSpec into a caller-allocated contiguous output."""
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise_ops", "FillSpec")
    call.out_dtype(dst.dtype)
    call.f64(value)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _b_scalar_tensor(value: Float64, st: Int32, device: Int) raises -> Held:
    """A 0-d tensor holding `value`: what a scalar operand becomes.

    Written with the FillSpec kernel, like the old path, and deliberately
    not with a device memset: on the MAX CPU device a memset enqueued on the
    context does not order against the `elementwise` launch that reads it,
    and the binary kernel then saw uninitialized memory (flaky wrong results
    on `mojo:cpu` only). Both halves have to be the same kind of launch.
    """
    var t = new_scalar(st, device)
    try:
        _b_fill_spec(t, value)
    except e:
        release(t.h)
        raise e
    return Held(t^, True)


def _b_copy_into(dst: T, src: T) raises:
    """dst[...] = src[...] for equal shapes and dtypes, any strides."""
    if dst.numel == 0:
        return
    if dst.contig and src.contig:
        copy_d2d(
            ctx_for(dst.device), dst.ptr, src.ptr, dst.numel * dst.itemsize
        )
    else:
        copy_strided_into(dst, src)


def _b_broadcast(a: T, b: T) raises -> IndexList[MAX_RANK]:
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


def _b_broadcast3(a: T, b: T, c: T) raises -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        var size = 1
        for extent in [a.shape[i], b.shape[i], c.shape[i]]:
            if extent != 1:
                if size != 1 and extent != size:
                    raise Error("shapes are not broadcastable")
                size = extent
        shape[i] = size
    return shape


def _b_fits(t: T, shape: IndexList[MAX_RANK]) -> Bool:
    for i in range(MAX_RANK):
        if t.shape[i] != shape[i]:
            return False
    return True


# ---------------------------------------------------------------------------
# kernel calls
# ---------------------------------------------------------------------------


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


def _b_binary_spec(op: StaticString, a: T, b: T, dst: T) raises:
    """logic_ops broadcast binary into a caller-allocated contiguous output."""
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


def _b_scalar_spec(op: StaticString, a: T, value: Float64, dst: T) raises:
    """elementwise_ops contiguous tensor-with-float-scalar."""
    _one_device(a, dst)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.out_dtype(dst.dtype)
    call.spec(a.spec(cp))
    call.f64(value)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _b_int_scalar_spec(op: StaticString, a: T, value: Int, dst: T) raises:
    """elementwise_ops contiguous tensor-with-int-scalar."""
    _one_device(a, dst)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.out_dtype(dst.dtype)
    call.spec(a.spec(cp))
    call.int(value)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _b_scalar_inplace(op: StaticString, a: T, value: Float64) raises:
    """`a op= scalar`: no output buffer, no copy back."""
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.out_dtype(a.dtype)
    call.spec(a.spec(cp))
    call.f64(value)
    call.run()
    _ = ctx


def _b_raw_add(dst: T, a: T, b: T) raises:
    """elementwise_ops `Add`: contiguous, equal shapes, one dtype. `dst` may
    be `a` — that is the in-place route, and the kernel is a flat elementwise
    loop, so aliasing the destination with an operand is exact."""
    _one_device(a, dst)
    _one_device(b, dst)
    if a.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var call = KernelCall("elementwise_ops", "Add")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    call.out_dtype(dst.dtype)
    call.int(dst.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.int(a.numel)
    call.int(dtype_code(a.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


# ---------------------------------------------------------------------------
# the binary route
# ---------------------------------------------------------------------------


@fieldwise_init
struct Res(Copyable, Movable):
    """A route's result: a fresh tensor this op owns, or the caller's `out`
    written in place."""

    var t: T
    var owned: Bool


def _b_op_dtype_ok(op: StaticString, st: Int32, is_cmp: Bool) raises:
    """The dtype checks `_binary_spec_into_go` makes, made here instead so a
    declined pair never triggers a kernel build."""
    if st == ST_BOOL:
        if not (
            is_cmp
            or op == "MulSpec"
            or op == "BitwiseAndSpec"
            or op == "BitwiseOrSpec"
            or op == "BitwiseXorSpec"
        ):
            unsupported("bool operands for " + String(op))
        return
    if not _b_bcast_dtype(st):
        unsupported("dtype " + String(st) + " for " + String(op))
    if op == "DivSpec" or op == "PowSpec":
        if not (_b_float3(st) or st == ST_FLOAT64):
            unsupported(String(op) + " requires a float dtype")
    if (
        op == "BitwiseAndSpec"
        or op == "BitwiseOrSpec"
        or op == "BitwiseXorSpec"
    ):
        if _b_float3(st) or st == ST_FLOAT64:
            unsupported(String(op) + " requires an integer dtype")


def _b_binary(
    op: StaticString,
    lhs: Side,
    rhs: Side,
    out_stype: Int32,
    dst: Optional[T],
) raises -> Res:
    """The old `_try_spec_binary`: promotion, scalar embedding, rank>4
    pre-materialization, output allocation and one logic_ops launch.

    `out_stype` overrides the output dtype (-1 keeps the promoted one; only
    the comparison-shaped ops pass bool). `dst`, when given and eligible, is
    written directly instead of allocating (the `out=` variants).
    """
    var is_cmp = out_stype == ST_BOOL
    if not lhs.is_t and not rhs.is_t:
        unsupported("two scalar operands")
    var a_h: Held
    var b_h: Held
    var device: Int
    var dtype: Int32
    if lhs.is_t and rhs.is_t:
        var a = lhs.t.value().copy()
        var b = rhs.t.value().copy()
        if a.device != b.device:
            raise Error("expected both operands on the same mojo device")
        device = a.device
        var promoted = _b_promote(a.stype, b.stype)
        if promoted < 0:
            unsupported(
                "no supported promotion for dtypes "
                + String(a.stype)
                + " and "
                + String(b.stype)
            )
        dtype = promoted
        _b_op_dtype_ok(op, dtype, is_cmp)
        # Above rank 4 the spec entry takes a flat pass: equal shapes and
        # contiguous operands only.
        var flat = a.rank > 4 or b.rank > 4
        if flat and (a.rank != b.rank or not _b_fits(a, b.shape)):
            unsupported("broadcasting operands of rank > 4")
        a_h = _b_ready(a, dtype, flat)
        b_h = _b_ready(b, dtype, flat)
    else:
        var t = lhs.t.value().copy() if lhs.is_t else rhs.t.value().copy()
        var s = rhs.s.value().copy() if lhs.is_t else lhs.s.value().copy()
        device = t.device
        dtype = t.stype
        _b_op_dtype_ok(op, dtype, is_cmp)
        var fill = _b_scalar_tensor(_b_embed(s, dtype), dtype, device)
        if lhs.is_t:
            a_h = _b_hold(t)
            b_h = fill^
        else:
            a_h = fill^
            b_h = _b_hold(t)
    var shape = _b_broadcast(a_h.t, b_h.t)
    var rank = max(a_h.t.rank, b_h.t.rank)
    var result_stype = out_stype if out_stype >= 0 else dtype
    if dst.__bool__():
        var o = dst.value().copy()
        if (
            o.contig
            and o.stype == result_stype
            and o.device == device
            and o.rank == rank
            and _b_fits(o, shape)
            and a_h.t.contig
            and b_h.t.contig
            and _b_fits(a_h.t, shape)
            and _b_fits(b_h.t, shape)
        ):
            # Nothing broadcast and everything dense: writing out[i] from
            # a[i]/b[i] stays exact even when `out` aliases an operand.
            _b_binary_spec(op, a_h.t, b_h.t, o)
            _ = a_h
            _ = b_h
            return Res(o^, False)
    var out = own(new_tensor(shape, rank, result_stype, device))
    _b_binary_spec(op, a_h.t, b_h.t, out.t)
    _ = a_h
    _ = b_h
    return Res(out.take(), True)


def _b_try_scalar(
    op: StaticString, lhs: Side, rhs: Side, negate: Bool
) raises -> Optional[Res]:
    """The old `_try_spec_scalar`: contiguous float tensor with a numeric
    scalar in one elementwise launch.

    float64 is excluded because the kernel's FLOAT_DTYPES has no entry for it
    — the old code let the kernel raise and fell through to the broadcast
    route, which is what declining here does, minus a wasted build.
    """
    if not lhs.is_t or rhs.is_t:
        return None
    var a = lhs.t.value().copy()
    if not _b_float3(a.stype):
        return None
    var s = rhs.s.value().copy()
    if s.is_bool:
        return None
    var value = -s.f if negate else s.f
    var src = _b_ready(a, a.stype, True)
    var out = own(new_like(src.t))
    _b_scalar_spec(op, src.t, value, out.t)
    _ = src
    return Res(out.take(), True)


def _b_try_int_scalar(
    op: StaticString, lhs: Side, rhs: Side, negate: Bool
) raises -> Optional[Res]:
    """The old `_try_spec_int_scalar`: contiguous int32/int64 tensor with an
    integer scalar."""
    if not lhs.is_t or rhs.is_t:
        return None
    var a = lhs.t.value().copy()
    if not (a.stype == ST_INT32 or a.stype == ST_INT64):
        return None
    var s = rhs.s.value().copy()
    if not s.is_int or s.is_bool:
        return None
    var value = -s.i if negate else s.i
    var src = _b_ready(a, a.stype, True)
    var out = own(new_like(src.t))
    _b_int_scalar_spec(op, src.t, value, out.t)
    _ = src
    return Res(out.take(), True)


def _b_try_add_f32_bf16(lhs: Side, rhs: Side) raises -> Optional[Res]:
    """The old `_try_spec_add_f32_bf16`: one contiguous FP32 + BF16 -> FP32
    launch that widens the BF16 operand in registers instead of
    materializing it (the hot residual add)."""
    if not lhs.is_t or not rhs.is_t:
        return None
    var a = lhs.t.value().copy()
    var b = rhs.t.value().copy()
    if (
        a.device != b.device
        or dev(a.device)[].is_cpu
        or not a.contig
        or not b.contig
        or not _b_fits(a, b.shape)
    ):
        return None
    if not (
        (a.stype == ST_FLOAT32 and b.stype == ST_BFLOAT16)
        or (a.stype == ST_BFLOAT16 and b.stype == ST_FLOAT32)
    ):
        return None
    var out = own(new_tensor(a.shape, a.rank, ST_FLOAT32, a.device))
    _b_binary_spec("AddF32Bf16Spec", a, b, out.t)
    return Res(out.take(), True)


def _b_try_apple_add(
    lhs: Side, rhs: Side, alpha: Float64
) raises -> Optional[Res]:
    """The old `_apple_contiguous_add`: Metal's equal-shape contiguous add."""
    if alpha != 1.0 or not lhs.is_t or not rhs.is_t:
        return None
    var a = lhs.t.value().copy()
    var b = rhs.t.value().copy()
    if (
        a.device != b.device
        or not a.contig
        or not b.contig
        or a.stype != b.stype
        or not _b_float3(a.stype)
        or not _b_fits(a, b.shape)
    ):
        return None
    var out = own(new_like(a))
    _b_raw_add(out.t, a, b)
    return Res(out.take(), True)


def _b_is_reduced(st: Int32) -> Bool:
    return st == ST_FLOAT16 or st == ST_BFLOAT16


def _b_alpha_result(lhs: Side, rhs: Side) raises -> Int32:
    """The result dtype of `a +/- alpha*b` when it is a REDUCED-precision one
    (float16/bfloat16), else -1.

    That is exactly the case ATen computes in a wider type: its add/sub
    functor runs in `opmath_type<scalar_t>` -- float32 for both half types --
    and rounds once, at the store. Scaling the operand in its own dtype first
    rounds twice, and the second rounding is against a value `alpha` may have
    shrunk by orders of magnitude.
    """
    if not rhs.is_t or not lhs.is_t:
        return Int32(-1)
    var result = _b_promote(lhs.t.value().stype, rhs.t.value().stype)
    if result < 0:
        return Int32(-1)
    if _b_is_reduced(result):
        return result
    return Int32(-1)


def _b_scale(t: T, alpha: Float64, alpha_is_int: Bool) raises -> Held:
    """`t * alpha` as an owned temporary: the tensor half of the old
    `_scaled_operand`."""
    var side = _b_tside(t)
    var a_side = _b_sside(Scal(alpha, Int(alpha), alpha_is_int, False))
    var scaled = _b_try_scalar("MulScalarSpec", side, a_side, False)
    if not scaled.__bool__():
        scaled = _b_try_int_scalar("MulScalarIntSpec", side, a_side, False)
    if not scaled.__bool__():
        unsupported("alpha != 1 with an operand of this dtype")
        return _b_hold(t)
    return Held(scaled.value().t.copy(), True)


# ---------------------------------------------------------------------------
# results: functional, in-place and out=
# ---------------------------------------------------------------------------


def _b_ret(rets: Values, var r: Res) raises:
    if not r.owned:
        raise Error("internal: a functional op produced a borrowed result")
    ret_tensor(rets, 0, r.t)


def _b_store_out(rets: Values, dest: T, var res: Res) raises:
    """Finish an `out=` variant: `res` is either `dest` itself (already
    written) or a fresh result to copy — and cast — into it."""
    if not res.owned:
        ret_ref(rets, 0, dest)
        return
    var held = own(res.t.copy())
    if not _b_can_cast(held.t.stype, dest.stype):
        raise Error(
            "result type ",
            held.t.stype,
            " can't be cast to the desired output type ",
            dest.stype,
        )
    var dst = dest.copy()
    if not dst.same_shape(held.t):
        # Only a MISMATCHING out= is resized: a resize re-lays the tensor out
        # contiguously, so doing it unconditionally would throw away a
        # correctly-shaped out's own strides and storage offset.
        resize_out(dst, held.t.shape, held.t.rank)
    if dst.stype == held.t.stype:
        _b_copy_into(dst, held.t)
    else:
        var casted = own(cast_to(held.t, dst.stype))
        _b_copy_into(dst, casted.t)
        _ = casted^  # alive past the launch
    _ = held^
    ret_ref(rets, 0, dst)


def _b_store_inplace(rets: Values, self: T, var res: Res) raises:
    """Finish an in-place variant: copy the functional result back over
    `self`, declining a result that does not fit (as the old path did)."""
    if not res.owned:
        ret_ref(rets, 0, self)
        return
    var held = own(res.t.copy())
    if held.t.stype != self.stype or not self.same_shape(held.t):
        unsupported("an in-place result that changes dtype or shape")
    _b_copy_into(self, held.t)
    _ = held^  # alive past the launch
    ret_ref(rets, 0, self)


def _b_out_tensor(v: Value, device: Int) raises -> T:
    var out = v_tensor(v)
    if not out.on_mojo():
        raise Error("expected `out` to be a mojo tensor")
    if out.device != device:
        raise Error("expected `out` and the inputs on the same mojo device")
    return out^


def _b_device_of(lhs: Side, rhs: Side) raises -> Int:
    if lhs.is_t:
        return lhs.t.value().device
    if rhs.is_t:
        return rhs.t.value().device
    unsupported("two scalar operands")
    return 0


def _b_dense_enough(t: T) -> Bool:
    """A weak stand-in for `TensorImpl::is_non_overlapping_and_dense`: false
    for a view that repeats elements (a broadcast stride of 0 over a real
    extent), which is the case ATen's own overlap check calls `TooHard` and
    declines to judge."""
    for i in range(t.rank):
        if t.stride(i) <= 0 and t.dim(i) > 1:
            return False
    return True


def _b_no_partial_overlap(written: T, other: T) raises:
    """`at::assert_no_partial_overlap` (c10/core/MemOverlap.cpp).

    An in-place or `out=` op whose input shares storage with the tensor being
    written, WITHOUT being the same view of it, is a read/write race: the
    kernel reads and writes the same bytes from different threads in an order
    nothing fixes. `x[1:].add_(x[:-1])` is the canonical case. The same view
    (identical span AND identical strides) is fine -- element i is written
    from element i -- and that is how `x.add_(x)` works.
    """
    if written.h == other.h or written.numel == 0 or other.numel == 0:
        return
    var storage = written.storage_ptr()
    if storage == 0 or storage != other.storage_ptr():
        return
    if not _b_dense_enough(written) or not _b_dense_enough(other):
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


def _b_no_overlap_side(written: T, side: Side) raises:
    if side.is_t:
        _b_no_partial_overlap(written, side.t.value())


def _b_self(v: Value, what: StaticString) raises -> T:
    var self = v_tensor(v)
    if not self.on_mojo():
        unsupported(String(what) + " on a tensor outside the mojo device")
    return self^


# ---------------------------------------------------------------------------
# add / sub / mul
# ---------------------------------------------------------------------------


def _b_add_routes(lhs: Side, rhs: Side, dst: Optional[T]) raises -> Res:
    var r = _b_try_add_f32_bf16(lhs, rhs)
    if r.__bool__():
        return r.value().copy()
    r = _b_try_scalar("AddScalarSpec", lhs, rhs, False)
    if r.__bool__():
        return r.value().copy()
    r = _b_try_int_scalar("AddScalarIntSpec", lhs, rhs, False)
    if r.__bool__():
        return r.value().copy()
    return _b_binary("AddSpec", lhs, rhs, Int32(-1), dst)


def _b_alpha_opmath(
    lhs: Side,
    rhs: Side,
    alpha: Float64,
    alpha_int: Bool,
    result_stype: Int32,
    subtract: Bool,
) raises -> Res:
    """`a +/- alpha*b` for reduced-precision operands: every step in float32,
    one rounding back to `result_stype` at the end (ATen's `opmath_type`
    contract, see `_b_alpha_result`). Five launches instead of two, on a
    route `alpha == 1` never reaches."""
    var a32 = _b_cast(lhs.t.value(), ST_FLOAT32)
    var b32 = _b_cast(rhs.t.value(), ST_FLOAT32)
    var scaled = _b_scale(b32.t, -alpha if subtract else alpha, alpha_int)
    var sum32 = _b_add_routes(_b_tside(a32.t), _b_tside(scaled.t), None)
    _ = a32
    _ = b32
    _ = scaled
    var held = own(sum32.t.copy())
    var out = cast_to(held.t, result_stype)
    _ = held  # the cast reads held.t's pointer inside a launch
    return Res(out^, True)


def _b_add(
    lhs: Side, rhs: Side, alpha_v: Value, dst: Optional[T]
) raises -> Res:
    """The `fast_aten_add` cascade."""
    var alpha = v_f64(alpha_v)
    var alpha_int = _b_scalar_is_int(alpha_v)
    if lhs.is_t and dev(lhs.t.value().device)[].api == "metal":
        var metal = _b_try_apple_add(lhs, rhs, alpha)
        if metal.__bool__():
            return metal.value().copy()
    if alpha == 1.0:
        return _b_add_routes(lhs, rhs, dst)
    if not rhs.is_t:
        var s = rhs.s.value().copy()
        if s.is_bool:
            unsupported("a bool scalar operand with alpha != 1")
        var v = s.f * alpha
        return _b_add_routes(
            lhs, _b_sside(Scal(v, Int(v), s.is_int and alpha_int, False)), dst
        )
    var reduced = _b_alpha_result(lhs, rhs)
    if reduced >= 0:
        return _b_alpha_opmath(lhs, rhs, alpha, alpha_int, reduced, False)
    var scaled = _b_scale(rhs.t.value(), alpha, alpha_int)
    var res = _b_add_routes(lhs, _b_tside(scaled.t), dst)
    _ = scaled
    return res^


def _b_sub_routes(lhs: Side, rhs: Side, dst: Optional[T]) raises -> Res:
    # sub-by-scalar reuses the AddScalar specs with a negated scalar.
    var r = _b_try_scalar("AddScalarSpec", lhs, rhs, True)
    if r.__bool__():
        return r.value().copy()
    r = _b_try_int_scalar("AddScalarIntSpec", lhs, rhs, True)
    if r.__bool__():
        return r.value().copy()
    return _b_binary("SubSpec", lhs, rhs, Int32(-1), dst)


def _b_sub(
    lhs: Side, rhs: Side, alpha_v: Value, dst: Optional[T]
) raises -> Res:
    """The `fast_aten_sub` cascade."""
    var alpha = v_f64(alpha_v)
    var alpha_int = _b_scalar_is_int(alpha_v)
    if alpha == 1.0:
        return _b_sub_routes(lhs, rhs, dst)
    if not rhs.is_t:
        var s = rhs.s.value().copy()
        if s.is_bool:
            unsupported("a bool scalar operand with alpha != 1")
        var v = s.f * alpha
        return _b_sub_routes(
            lhs, _b_sside(Scal(v, Int(v), s.is_int and alpha_int, False)), dst
        )
    var reduced = _b_alpha_result(lhs, rhs)
    if reduced >= 0:
        return _b_alpha_opmath(lhs, rhs, alpha, alpha_int, reduced, True)
    var scaled = _b_scale(rhs.t.value(), alpha, alpha_int)
    var res = _b_sub_routes(lhs, _b_tside(scaled.t), dst)
    _ = scaled
    return res^


def _b_mul(lhs: Side, rhs: Side, dst: Optional[T]) raises -> Res:
    var r = _b_try_scalar("MulScalarSpec", lhs, rhs, False)
    if r.__bool__():
        return r.value().copy()
    r = _b_try_int_scalar("MulScalarIntSpec", lhs, rhs, False)
    if r.__bool__():
        return r.value().copy()
    return _b_binary("MulSpec", lhs, rhs, Int32(-1), dst)


# aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
def op_add_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_add(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            args[unsafe_offset=2],
            None,
        ),
    )


# aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
def op_add_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = _b_self(args[unsafe_offset=0], "add_")
    var rhs = _b_side(args[unsafe_offset=1])
    _b_no_overlap_side(self, rhs)
    var alpha = v_f64(args[unsafe_offset=2])
    if alpha == 1.0 and rhs.is_t:
        # Direct in-place kernel when every layout lines up.
        var b = rhs.t.value().copy()
        if (
            self.contig
            and b.contig
            and self.stype == b.stype
            and self.device == b.device
            and _b_bcast_dtype(self.stype)
            and _b_fits(self, b.shape)
        ):
            _b_raw_add(self, self, b)
            ret_ref(rets, 0, self)
            return
    if not rhs.is_t and self.contig and _b_float3(self.stype):
        # A float scalar goes straight into `self`, alpha folded in exactly:
        # no output buffer and no copy back.
        var s = rhs.s.value().copy()
        if not s.is_bool:
            _b_scalar_inplace("AddScalarInplace", self, s.f * alpha)
            ret_ref(rets, 0, self)
            return
    _b_store_inplace(
        rets, self, _b_add(_b_tside(self), rhs, args[unsafe_offset=2], None)
    )


# aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_add_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var dest = _b_out_tensor(args[unsafe_offset=3], _b_device_of(lhs, rhs))
    _b_no_overlap_side(dest, lhs)
    _b_no_overlap_side(dest, rhs)
    _b_store_out(
        rets,
        dest,
        _b_add(lhs, rhs, args[unsafe_offset=2], Optional[T](dest.copy())),
    )


# aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
def op_sub_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_sub(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            args[unsafe_offset=2],
            None,
        ),
    )


# aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
def op_sub_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = _b_self(args[unsafe_offset=0], "sub_")
    var rhs = _b_side(args[unsafe_offset=1])
    _b_no_overlap_side(self, rhs)
    var alpha = v_f64(args[unsafe_offset=2])
    if not rhs.is_t and self.contig and _b_float3(self.stype):
        var s = rhs.s.value().copy()
        if not s.is_bool:
            _b_scalar_inplace("AddScalarInplace", self, -(s.f * alpha))
            ret_ref(rets, 0, self)
            return
    _b_store_inplace(
        rets, self, _b_sub(_b_tside(self), rhs, args[unsafe_offset=2], None)
    )


# aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_sub_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var dest = _b_out_tensor(args[unsafe_offset=3], _b_device_of(lhs, rhs))
    _b_no_overlap_side(dest, lhs)
    _b_no_overlap_side(dest, rhs)
    _b_store_out(
        rets,
        dest,
        _b_sub(lhs, rhs, args[unsafe_offset=2], Optional[T](dest.copy())),
    )


# aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
def op_mul_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_mul(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            None,
        ),
    )


# aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
def op_mul_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = _b_self(args[unsafe_offset=0], "mul_")
    var rhs = _b_side(args[unsafe_offset=1])
    _b_no_overlap_side(self, rhs)
    if not rhs.is_t and self.contig and _b_float3(self.stype):
        var s = rhs.s.value().copy()
        if not s.is_bool:
            _b_scalar_inplace("MulScalarInplace", self, s.f)
            ret_ref(rets, 0, self)
            return
    _b_store_inplace(rets, self, _b_mul(_b_tside(self), rhs, None))


# aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_mul_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var dest = _b_out_tensor(args[unsafe_offset=2], _b_device_of(lhs, rhs))
    _b_no_overlap_side(dest, lhs)
    _b_no_overlap_side(dest, rhs)
    _b_store_out(rets, dest, _b_mul(lhs, rhs, Optional[T](dest.copy())))


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------


def _b_is_floating(st: Int32) -> Bool:
    return _b_float3(st) or st == ST_FLOAT64


def _b_true_div_dtype(a_stype: Int32, rhs: Side) raises -> Int32:
    """`torch.result_type` for TRUE division.

    ATen builds the divide iterator with `promote_integer_inputs_to_float`,
    so an otherwise-integral result becomes `torch.get_default_dtype()` --
    not float32 unconditionally: `torch.set_default_dtype(torch.float64)`
    moves it. A floating operand keeps its own dtype (int64 / float16 is
    float16), and a float SCALAR against an integral tensor lands on the
    default dtype too (`torch.result_type(int_tensor, 0.5)`).
    """
    var common = a_stype
    if rhs.is_t:
        var b_stype = rhs.t.value().stype
        var a_float = _b_is_floating(a_stype)
        var b_float = _b_is_floating(b_stype)
        if a_float and not b_float:
            common = a_stype
        elif b_float and not a_float:
            common = b_stype
        elif a_float and b_float:
            common = _b_promote(a_stype, b_stype)
            if common < 0:
                unsupported(
                    "no dtype promotion for "
                    + String(a_stype)
                    + " and "
                    + String(b_stype)
                )
        else:
            return default_dtype()
    elif not rhs.s.value().is_int and not _b_is_floating(a_stype):
        return default_dtype()
    if not _b_is_floating(common):
        return default_dtype()
    return common


def _b_div(lhs: Side, rhs: Side, mode: Value, dst: Optional[T]) raises -> Res:
    """The `fast_aten_div` cascade, rounding modes included.

    div.Tensor_mode keeps the operands' promoted dtype ("floor" is
    torch.floor_divide's own convention, "trunc" is C-style); plain division
    always promotes to float.
    """
    if not v_is_none(mode):
        var name = v_string(mode)
        if name == "floor":
            return _b_binary("FloorDivSpec", lhs, rhs, Int32(-1), dst)
        if name == "trunc":
            return _b_binary("TruncDivSpec", lhs, rhs, Int32(-1), dst)
        unsupported("div rounding_mode " + name)
    if not lhs.is_t:
        unsupported("div with a scalar numerator")
    var a = lhs.t.value().copy()
    var common = _b_true_div_dtype(a.stype, rhs)
    # float64 included: DivSpec dispatches on logic_ops SPEC_BCAST_DTYPES,
    # which has it, and the kernel only asks that the dtype be floating.
    if not _b_is_floating(common):
        unsupported(
            "true division producing dtype "
            + String(common)
            + " (the divide kernel covers the float dtypes only)"
        )
    # An integer numerator still has to be CAST into `common`, and
    # data_movement_ops' CAST_DTYPES stops before float64: `_b_cast` declines
    # that pair on its own.
    var num = _b_cast(a, common)
    if not rhs.is_t:
        var r1 = _b_binary("DivSpec", _b_tside(num.t), rhs, Int32(-1), dst)
        _ = num
        return r1^
    var b = rhs.t.value().copy()
    if not a.on_mojo() or not b.on_mojo() or b.device != a.device:
        raise Error("expected both operands on the same mojo device")
    var den = _b_cast(b, common)
    var r2 = _b_binary(
        "DivSpec", _b_tside(num.t), _b_tside(den.t), Int32(-1), dst
    )
    _ = num
    _ = den
    return r2^


def _b_no_mode() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


# aten::div.Tensor(Tensor self, Tensor other) -> Tensor
def op_div_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_div(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            _b_no_mode(),
            None,
        ),
    )


# aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
def op_div_mode(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_div(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            args[unsafe_offset=2],
            None,
        ),
    )


# aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_div_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var dest = _b_out_tensor(args[unsafe_offset=2], _b_device_of(lhs, rhs))
    _b_no_overlap_side(dest, lhs)
    _b_no_overlap_side(dest, rhs)
    _b_store_out(
        rets, dest, _b_div(lhs, rhs, _b_no_mode(), Optional[T](dest.copy()))
    )


# aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
def op_div_out_mode(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var dest = _b_out_tensor(args[unsafe_offset=3], _b_device_of(lhs, rhs))
    _b_no_overlap_side(dest, lhs)
    _b_no_overlap_side(dest, rhs)
    _b_store_out(
        rets,
        dest,
        _b_div(lhs, rhs, args[unsafe_offset=2], Optional[T](dest.copy())),
    )


# ---------------------------------------------------------------------------
# pow, maximum/minimum, remainder, floor_divide, bitwise
# ---------------------------------------------------------------------------


def _b_simple(op: StaticString, args: Values, rets: Values) raises:
    _b_ret(
        rets,
        _b_binary(
            op,
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            Int32(-1),
            None,
        ),
    )


# aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
def op_pow_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    var r = _b_try_scalar("PowScalarSpec", lhs, rhs, False)
    if r.__bool__():
        _b_ret(rets, r.value().copy())
        return
    # PowScalarSpec is FLOAT_DTYPES only and narrows the exponent to float32.
    # float64 takes the broadcast route instead, which embeds the exponent in
    # a 0-d tensor of the operand's dtype and whose PowSpec kernel covers the
    # dtype (logic_ops SPEC_BCAST_DTYPES); `_b_binary` declines the rest.
    _b_ret(rets, _b_binary("PowSpec", lhs, rhs, Int32(-1), None))


# aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
def op_pow_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lhs = _b_side(args[unsafe_offset=0])
    # The kernel raises on integers, which would leave the output unwritten.
    if not lhs.is_t or not _b_is_floating(lhs.t.value().stype):
        unsupported("pow.Tensor_Tensor on a tensor that is not float")
    _b_simple("PowSpec", args, rets)


# aten::maximum(Tensor self, Tensor other) -> Tensor
def op_maximum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_simple("MaximumSpec", args, rets)


# aten::minimum(Tensor self, Tensor other) -> Tensor
def op_minimum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_simple("MinimumSpec", args, rets)


# aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
# aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
# aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
def op_remainder(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    # Divisor-signed remainder (Python/torch `%`), float and int dtypes.
    _b_simple("RemainderSpec", args, rets)


# aten::floor_divide(Tensor self, Tensor other) -> Tensor
# aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
def op_floor_divide(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _b_simple("FloorDivSpec", args, rets)


# aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
# aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
def op_bitwise_and(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_simple("BitwiseAndSpec", args, rets)


# aten::bitwise_or.Scalar / .Tensor
def op_bitwise_or(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_simple("BitwiseOrSpec", args, rets)


# aten::bitwise_xor.Scalar / .Tensor
def op_bitwise_xor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_simple("BitwiseXorSpec", args, rets)


# ---------------------------------------------------------------------------
# logical_and / logical_xor
# ---------------------------------------------------------------------------


def _b_logical(op: StaticString, args: Values, rets: Values) raises:
    """The promoted route when torch's promotion covers the pair, else each
    operand reduced to bool first (the old `_try_logical`)."""
    var lhs = _b_side(args[unsafe_offset=0])
    var rhs = _b_side(args[unsafe_offset=1])
    if lhs.is_t and rhs.is_t:
        var a = lhs.t.value().copy()
        var b = rhs.t.value().copy()
        if _b_promote(a.stype, b.stype) < 0:
            if not (_b_castable(a.stype) and _b_castable(b.stype)):
                unsupported(
                    "a logical op on dtypes "
                    + String(a.stype)
                    + " and "
                    + String(b.stype)
                )
            var ba = _b_cast(a, ST_BOOL)
            var bb = _b_cast(b, ST_BOOL)
            var res = _b_binary(
                op, _b_tside(ba.t), _b_tside(bb.t), ST_BOOL, None
            )
            _ = ba
            _ = bb
            _b_ret(rets, res^)
            return
    _b_ret(rets, _b_binary(op, lhs, rhs, ST_BOOL, None))


# aten::logical_and(Tensor self, Tensor other) -> Tensor
def op_logical_and(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_logical("LogicalAndSpec", args, rets)


# aten::logical_xor(Tensor self, Tensor other) -> Tensor
def op_logical_xor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_logical("LogicalXorSpec", args, rets)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def _b_clamp_dtype(self_stype: Int32, lo: Value, hi: Value) -> Int32:
    """`torch.result_type(self, min, max)`: ATen's clamp iterator promotes
    its inputs to a common dtype, so a FLOAT bound against an integral (or
    bool) tensor lands on the default floating dtype --
    `torch.clamp(int_tensor, min=0.5)` is a float tensor. An integral bound
    keeps the tensor's own dtype (and wraps into it, as ATen does)."""
    var float_bound = (not v_is_none(lo) and not _b_scalar_is_int(lo)) or (
        not v_is_none(hi) and not _b_scalar_is_int(hi)
    )
    if not float_bound or _b_is_floating(self_stype):
        return self_stype
    return default_dtype()


def _b_clamp(self: T, lo: Value, hi: Value) raises -> Res:
    var has_min = not v_is_none(lo)
    var has_max = not v_is_none(hi)
    if not has_min and not has_max:
        unsupported("clamp with neither min nor max")
    var result_stype = _b_clamp_dtype(self.stype, lo, hi)
    if not _b_bcast_dtype(result_stype) or result_stype == ST_FLOAT64:
        unsupported("clamp producing dtype " + String(result_stype))
    var lo_v = v_f64(lo) if has_min else 0.0
    var hi_v = v_f64(hi) if has_max else 0.0
    var src = _b_ready(self, result_stype, True)
    var out = own(new_like(src.t))
    if out.t.numel > 0:
        var ctx = ctx_for(self.device)
        var call = KernelCall("logic_ops", "ClampScalar")
        call.arg_dtype(0, src.t.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(src.t.ptr)
        call.f64(lo_v)
        call.f64(hi_v)
        call.int(1 if has_min else 0)
        call.int(1 if has_max else 0)
        call.int(out.t.numel)
        call.int(dtype_code(src.t.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = src
    return Res(out.take(), True)


# aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
def op_clamp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(
        rets,
        _b_clamp(
            _b_self(args[unsafe_offset=0], "clamp"),
            args[unsafe_offset=1],
            args[unsafe_offset=2],
        ),
    )


# ---------------------------------------------------------------------------
# addcmul / addcdiv
# ---------------------------------------------------------------------------


def _b_strides4(t: T, shape: IndexList[MAX_RANK]) -> List[Int]:
    """The operand's rank-4 strides for the ternary kernel: 0 on every axis
    it broadcasts over (the old `_bcast_meta`)."""
    var out = List[Int](capacity=4)
    for k in range(4):
        var i = MAX_RANK - 4 + k
        out.append(0 if t.shape[i] == 1 else t.strides[i])
    return out^


def _b_addc(op: StaticString, args: Values, allow_int: Bool) raises -> Res:
    """self + value * (tensor1 * tensor2) — or / — in one broadcast launch."""
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var c = v_tensor(args[unsafe_offset=2])
    if not a.on_mojo() or not b.on_mojo() or not c.on_mojo():
        unsupported("addc* operands that are not all on the mojo device")
    if a.device != b.device or a.device != c.device:
        raise Error("expected every operand on the same mojo device")
    if a.stype != b.stype or a.stype != c.stype:
        unsupported("addc* operands of different dtypes")
    if not _b_float3(a.stype) and not (allow_int and _b_int_dtype(a.stype)):
        unsupported("addc* of dtype " + String(a.stype))
    if a.rank > 4 or b.rank > 4 or c.rank > 4:
        unsupported("addc* operands of rank > 4")
    var value = _b_side(args[unsafe_offset=3])
    if value.is_t:
        unsupported("a tensor `value` for addc*")
    var s = value.s.value().copy()
    if s.is_bool:
        unsupported("a bool `value` for addc*")
    var merged = _b_broadcast3(a, b, c)
    var rank = max(a.rank, max(b.rank, c.rank))
    var numel = 1
    for i in range(MAX_RANK):
        numel *= merged[i]
    var out = own(new_tensor(merged, rank, a.stype, a.device))
    if numel > 0:
        var params = List[Int](capacity=16)
        for k in range(4):
            params.append(merged[MAX_RANK - 4 + k])
        params.extend(_b_strides4(a, merged))
        params.extend(_b_strides4(b, merged))
        params.extend(_b_strides4(c, merged))
        var ctx = ctx_for(a.device)
        var call = KernelCall("logic_ops", String(op))
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, b.dtype)
        call.arg_dtype(2, c.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(a.ptr)
        call.int(b.ptr)
        call.int(c.ptr)
        call.tuple(params)
        call.f64(s.f)
        call.int(dtype_code(a.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    return Res(out.take(), True)


# aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
def op_addcmul(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(rets, _b_addc("AddcmulBcast", args, True))


# aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
def op_addcmul_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = _b_out_tensor(
        args[unsafe_offset=4], v_tensor(args[unsafe_offset=0]).device
    )
    _b_store_out(rets, dest, _b_addc("AddcmulBcast", args, True))


# aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
def op_addcdiv(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    # addcdiv is float-only: torch errors for integer inputs.
    _b_ret(rets, _b_addc("AddcdivBcast", args, False))


# aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
def op_addcdiv_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = _b_out_tensor(
        args[unsafe_offset=4], v_tensor(args[unsafe_offset=0]).device
    )
    _b_store_out(rets, dest, _b_addc("AddcdivBcast", args, False))


# ---------------------------------------------------------------------------
# lerp
# ---------------------------------------------------------------------------


def _b_lerp(args: Values) raises -> Res:
    """ATen's numerically stable scalar lerp (native/Lerp.h), composed from
    the ops above exactly as the old `fast_aten_lerp` did."""
    var start = _b_self(args[unsafe_offset=0], "lerp")
    var finish = _b_self(args[unsafe_offset=1], "lerp")
    if start.device != finish.device:
        raise Error("expected both operands on the same mojo device")
    if start.stype != ST_FLOAT32 or finish.stype != ST_FLOAT32:
        unsupported("lerp of a dtype other than float32")
    var w = _b_side(args[unsafe_offset=2])
    if w.is_t:
        unsupported("a tensor weight for lerp.Scalar")
    var weight = w.s.value().f
    # ATen narrows the Scalar to the tensor's opmath type before choosing the
    # stable formula: a value just below 0.5 can round to exactly 0.5 and
    # must take the second branch. Conversion is part of the ATen contract:
    # a finite value that does not fit float32 is rejected, not silently
    # turned into an infinity.
    var narrowed = Float64(Float32(weight))
    if (weight - weight) == 0.0 and (narrowed - narrowed) != 0.0:
        raise Error("value cannot be converted to type float without overflow")
    var delta = _b_binary(
        "SubSpec", _b_tside(finish), _b_tside(start), Int32(-1), None
    )
    var diff = own(delta.t.copy())
    var res: Res
    if abs(narrowed) < 0.5:
        res = _b_add(
            _b_tside(start),
            _b_tside(diff.t),
            Value(TAG_SCALAR_DOUBLE, 0, f64_bits(narrowed), 0),
            None,
        )
    else:
        var rest = Float64(Float32(1.0 - narrowed))
        res = _b_sub(
            _b_tside(finish),
            _b_tside(diff.t),
            Value(TAG_SCALAR_DOUBLE, 0, f64_bits(rest), 0),
            None,
        )
    _ = diff
    return res^


# aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
def op_lerp_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _b_ret(rets, _b_lerp(args))


# aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
def op_lerp_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dest = _b_out_tensor(
        args[unsafe_offset=3], v_tensor(args[unsafe_offset=0]).device
    )
    _b_store_out(rets, dest, _b_lerp(args))


# ---------------------------------------------------------------------------


def register_binary(site: Site) raises:
    impl[op_add_tensor, "add.Tensor"](site)
    impl[op_add_, "add_.Tensor"](site)
    impl[op_add_out, "add.out"](site)
    impl[op_addcdiv, "addcdiv"](site)
    impl[op_addcdiv_out, "addcdiv.out"](site)
    impl[op_addcmul, "addcmul"](site)
    impl[op_addcmul_out, "addcmul.out"](site)
    impl[op_bitwise_and, "bitwise_and.Scalar"](site)
    impl[op_bitwise_and, "bitwise_and.Tensor"](site)
    impl[op_bitwise_or, "bitwise_or.Scalar"](site)
    impl[op_bitwise_or, "bitwise_or.Tensor"](site)
    impl[op_bitwise_xor, "bitwise_xor.Scalar"](site)
    impl[op_bitwise_xor, "bitwise_xor.Tensor"](site)
    impl[op_clamp, "clamp"](site)
    impl[op_div_tensor, "div.Tensor"](site)
    impl[op_div_mode, "div.Tensor_mode"](site)
    impl[op_div_out, "div.out"](site)
    impl[op_div_out_mode, "div.out_mode"](site)
    impl[op_floor_divide, "floor_divide"](site)
    impl[op_floor_divide, "floor_divide.Scalar"](site)
    impl[op_lerp_scalar, "lerp.Scalar"](site)
    impl[op_lerp_scalar_out, "lerp.Scalar_out"](site)
    impl[op_logical_and, "logical_and"](site)
    impl[op_logical_xor, "logical_xor"](site)
    impl[op_maximum, "maximum"](site)
    impl[op_minimum, "minimum"](site)
    impl[op_mul_tensor, "mul.Tensor"](site)
    impl[op_mul_, "mul_.Tensor"](site)
    impl[op_mul_out, "mul.out"](site)
    impl[op_pow_scalar, "pow.Tensor_Scalar"](site)
    impl[op_pow_tensor, "pow.Tensor_Tensor"](site)
    impl[op_remainder, "remainder.Scalar"](site)
    impl[op_remainder, "remainder.Scalar_Tensor"](site)
    impl[op_remainder, "remainder.Tensor"](site)
    impl[op_sub_tensor, "sub.Tensor"](site)
    impl[op_sub_, "sub_.Tensor"](site)
    impl[op_sub_out, "sub.out"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_binary]()
