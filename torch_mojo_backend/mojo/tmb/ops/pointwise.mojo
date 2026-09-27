"""ATen ops: pointwise math with up to three operands and scalar parameters.

The binary and ternary math ops (atan2, hypot, copysign, fmod, frexp,
nextafter, gcd/lcm, the shifts, logaddexp, xlogy, zeta, the special
polynomials, lerp.Tensor, clamp.Tensor, pow.Scalar, ...) and the
parameterized activations with their backwards (elu, hardtanh, softplus,
leaky_relu, ...), all on one kernel family: `pointwise`
(tmb/kernels/pointwise/entry.mojo), whose per-kind math is
`tmb.kernels.common.pointwise_math`.

Every op goes through `_pw_run`: ATen's type promotion (`_pw_result_type`,
the ResultTypeState rules of c10/core/ScalarType + TensorIterator), then each
operand made flat for the kernel -- cast to the computation dtype, a full
dense buffer of the output's extent, a single device element, or a host
scalar passed by value -- and one launch into a fresh contiguous output or
straight into a fitting `out=`.
"""

from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    ST_BFLOAT16,
    ST_BOOL,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT16,
    ST_INT32,
    ST_INT64,
    ST_INT8,
    ST_UINT8,
    T,
    TAG_BOOL,
    TAG_DOUBLE,
    TAG_NONE,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    Value,
    bits_f64,
    Values,
    default_dtype,
    dtype_name,
    max_dtype,
    new_tensor,
    own,
    own_if_new,
    release,
    ret_ref,
    ret_tensor,
    unsupported,
    v_bool,
    v_f64,
    v_generator,
    v_int,
    v_is_none,
    v_string,
    v_tensor,
    view_strided,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.binary import (
    Res,
    Scal,
    Side,
    _b_copy_into,
    _b_inplace_destination,
    _b_inplace_operand,
    _b_no_overlap_side,
    _b_out_guard3,
    _b_no_partial_overlap,
    _b_ret,
    _b_self,
    _b_side,
    _b_sside,
    _b_store_inplace,
    _b_store_out,
    _b_tside,
    op_lerp_scalar_,
    op_pow_scalar,
    op_pow_scalar_out,
    op_pow_tensor,
    op_pow_tensor_out,
    op_rsub,
    op_rsub_out,
)
from tmb.ops.common import (
    ST_UNDEFINED,
    assert_no_internal_overlap,
    combine_categories as _pw_combine,
    contiguous,
    copy_strided_into,
    is_float_stype as _pw_is_float,
    is_int_stype as _pw_is_int,
    known_stype as _pw_known,
    promote_types,
    resize_out,
    scalar_to_float,
    scalar_to_int,
)
from tmb.ops.random import _draw
from tmb.ops.core import cast_for_copy
from tmb.ops.reductions import _scalar_reduction
from tmb.ops.unary import op_gelu_backward

# ---------------------------------------------------------------------------
# type promotion
# ---------------------------------------------------------------------------


@fieldwise_init
struct _TypeState(Copyable, Movable):
    var dim: Int32
    var zero: Int32
    var wrapped: Int32


def _pw_update(mut state: _TypeState, side: Side) raises:
    """at::native::update_result_type_state for one operand. A Scalar (or
    torch's 0-d CPU wrapped number) is a wrapped number: bool, int64, or the
    default floating dtype."""
    if side.is_t:
        var t = side.t.value().copy()
        if t.rank == 0:
            state.zero = promote_types(state.zero, t.stype)
        else:
            state.dim = promote_types(state.dim, t.stype)
        return
    if side.zero_st >= 0:
        # An explicit 0-d CPU tensor (not a wrapped Python number).
        state.zero = promote_types(state.zero, side.zero_st)
        return
    var s = side.s.value().copy()
    var st = ST_BOOL if s.is_bool else (
        ST_INT64 if s.is_int else default_dtype()
    )
    state.wrapped = promote_types(state.wrapped, st)


def _pw_result_type(a: Side, b: Side, c: Side, arity: Int) raises -> Int32:
    var state = _TypeState(ST_UNDEFINED, ST_UNDEFINED, ST_UNDEFINED)
    _pw_update(state, a)
    if arity >= 2:
        _pw_update(state, b)
    if arity >= 3:
        _pw_update(state, c)
    return _pw_combine(state.dim, _pw_combine(state.zero, state.wrapped))


# Promotion policies: how an op maps the promoted dtype to its compute dtype.
comptime P_FLOAT = 0  # integral and bool promote to the default float dtype
comptime P_FLOAT_ONLY = 1  # floating dtypes only (ATen's kernel has no ints)
comptime P_ALL = 2  # every dtype, bool included
comptime P_INT = 3  # integral dtypes only, no bool
comptime P_NUMERIC = 4  # every dtype but bool


def _pw_compute_dtype(
    kind: StaticString, common: Int32, policy: Int, f64_ok: Bool
) raises -> Int32:
    var st = common
    if policy == P_FLOAT and not _pw_is_float(st):
        st = default_dtype()
    if policy == P_FLOAT_ONLY and not _pw_is_float(st):
        unsupported(String(kind) + " on dtype " + String(st) + " (floats only)")
    if policy == P_INT and not _pw_is_int(st):
        unsupported(
            String(kind) + " on dtype " + String(st) + " (integers only)"
        )
    if policy == P_NUMERIC and st == ST_BOOL:
        unsupported(String(kind) + " on bool operands")
    if not _pw_known(st):
        unsupported(String(kind) + " on dtype " + String(st))
    if st == ST_FLOAT64 and not f64_ok:
        unsupported(
            String(kind)
            + " on float64 (its math has no double lowering on the GPU)"
        )
    return st


# ---------------------------------------------------------------------------
# the launch
# ---------------------------------------------------------------------------


comptime MODE_FULL = 0
comptime MODE_DEVICE = 1
comptime MODE_HOST = 2


def _pw_broadcast(
    a: Side, b: Side, c: Side, arity: Int
) raises -> Tuple[IndexList[MAX_RANK], Int]:
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    for k in range(arity):
        var side = a.copy() if k == 0 else (b.copy() if k == 1 else c.copy())
        if not side.is_t:
            continue
        var t = side.t.value().copy()
        rank = max(rank, t.rank)
        for i in range(MAX_RANK):
            var x = shape[i]
            var y = t.shape[i]
            if x == y or y == 1:
                continue
            if x == 1:
                shape[i] = y
            else:
                raise Error("shapes are not broadcastable")
    return (shape, rank)


def _pw_numel(shape: IndexList[MAX_RANK]) -> Int:
    var n = 1
    for i in range(MAX_RANK):
        n *= shape[i]
    return n


struct _Flat(Movable):
    """One kernel operand: its mode, address and host value, plus the
    temporary (cast / materialized copy) that must outlive the launch."""

    var mode: Int
    var addr: Int
    var value: Float64
    var hold: List[T]

    def __init__(out self, mode: Int, addr: Int, value: Float64):
        self.mode = mode
        self.addr = addr
        self.value = value
        self.hold = List[T]()

    def __deinit__(deinit self):
        for t in self.hold:
            release(t.h)


def _pw_flat(
    side: Side, compute: Int32, shape: IndexList[MAX_RANK], rank: Int
) raises -> _Flat:
    if not side.is_t:
        var s = side.s.value().copy()
        if _pw_is_int(compute) or compute == ST_BOOL:
            # An integer (or bool: 0 / 1) operand travels as its int64 bits
            # in the float64 slot (see `_host_value`), exact across int64.
            var i = s.i if s.is_int else Int(s.f)
            return _Flat(MODE_HOST, 0, bits_f64(Int64(i)))
        return _Flat(MODE_HOST, 0, s.f)
    var t = side.t.value().copy()
    var flat = _Flat(MODE_FULL, 0, 0.0)
    if t.stype != compute:
        var casted = cast_for_copy(t, compute)
        if casted.h != t.h:
            flat.hold.append(casted.copy())
        t = casted^
    var out_numel = _pw_numel(shape)
    if t.numel == out_numel and t.contig:
        flat.addr = t.ptr
        return flat^
    if t.numel == 1:
        flat.mode = MODE_DEVICE
        flat.addr = t.ptr
        return flat^
    var src = t.copy()
    if t.numel != out_numel:
        # Broadcast: a zero-stride view over the output's extent, then a
        # dense copy of it.
        var strides = IndexList[MAX_RANK](0)
        for i in range(MAX_RANK):
            strides[i] = 0 if t.shape[i] == 1 else t.strides[i]
        src = view_strided(t, shape, strides, rank, t.offset)
        flat.hold.append(src.copy())
    var dense = contiguous(src)
    if dense.h != src.h:
        flat.hold.append(dense.copy())
    flat.addr = dense.ptr
    return flat^


def _pw_launch(
    kind: StaticString,
    dst: T,
    compute: Int32,
    fa: _Flat,
    fb: _Flat,
    fc: _Flat,
    params: SIMD[DType.float64, 4],
) raises:
    if dst.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var call = KernelCall("pointwise", String(kind))
    call.arg_dtype(0, max_dtype(compute))
    call.out_dtype(dst.dtype)
    call.int(dst.ptr)
    call.int(fa.addr)
    call.int(fb.addr)
    call.int(fc.addr)
    call.int(fa.mode | (fb.mode << 2) | (fc.mode << 4))
    call.f64(fa.value)
    call.f64(fb.value)
    call.f64(fc.value)
    call.int(dst.numel)
    call.f64(params[0])
    call.f64(params[1])
    call.f64(params[2])
    call.f64(params[3])
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _none_side() -> Side:
    return _b_sside(Scal(0.0, 0, False, False))


def _pw_device(a: Side, b: Side, c: Side) raises -> Int:
    var device = -1
    for k in range(3):
        var side = a.copy() if k == 0 else (b.copy() if k == 1 else c.copy())
        if side.is_t:
            var t = side.t.value().copy()
            if not t.on_mojo():
                unsupported("an operand outside the mojo device")
            if device >= 0 and t.device != device:
                raise Error("expected every operand on the same mojo device")
            device = t.device
    if device < 0:
        unsupported("no tensor operand")
    return device


def _pw_run(
    kind: StaticString,
    arity: Int,
    a: Side,
    b: Side,
    c: Side,
    compute: Int32,
    out_stype: Int32,
    params: SIMD[DType.float64, 4],
    dst: Optional[T],
) raises -> Res:
    """One launch of `kind` over (a, b, c) computed in `compute`, stored as
    `out_stype`: into `dst` when it already has the result's dtype, shape
    and a dense layout, else into a fresh tensor."""
    var device = _pw_device(a, b, c)
    var bs = _pw_broadcast(a, b, c, arity)
    var shape = bs[0]
    var rank = bs[1]
    var fa = _pw_flat(a, compute, shape, rank)
    var fb = _pw_flat(b, compute, shape, rank) if arity >= 2 else _Flat(
        MODE_HOST, 0, 0.0
    )
    var fc = _pw_flat(c, compute, shape, rank) if arity >= 3 else _Flat(
        MODE_HOST, 0, 0.0
    )
    if fa.mode != MODE_FULL and fb.mode != MODE_FULL and fc.mode != MODE_FULL:
        # Every operand is a single element: the first device one is the
        # (one-element) full operand.
        if fa.mode == MODE_DEVICE:
            fa.mode = MODE_FULL
        elif fb.mode == MODE_DEVICE:
            fb.mode = MODE_FULL
        elif fc.mode == MODE_DEVICE:
            fc.mode = MODE_FULL
    if dst:
        var d = dst.value().copy()
        var fits = d.contig and d.stype == out_stype and d.device == device
        if fits and d.rank == rank:
            for i in range(MAX_RANK):
                if d.shape[i] != shape[i]:
                    fits = False
        else:
            fits = False
        if fits:
            # Elementwise and dense: writing out[i] from operand[i] is exact
            # even when `d` is one of the operands (same view).
            _pw_launch(kind, d, compute, fa, fb, fc, params)
            return Res(d^, False)
    var out = own(new_tensor(shape, rank, out_stype, device))
    _pw_launch(kind, out.t, compute, fa, fb, fc, params)
    return Res(out.take(), True)


def _p(
    p0: Float64 = 0.0, p1: Float64 = 0.0, p2: Float64 = 0.0, p3: Float64 = 0.0
) -> SIMD[DType.float64, 4]:
    return SIMD[DType.float64, 4](p0, p1, p2, p3)


def _pw_out_exact(kind: StaticString) -> Bool:
    """Whether `kind`'s `out=` must already have the result dtype, where
    every other kind's result is cast into any `out` it `canCast`s to.

    Torch's rule per op (v2.14): the activations are structured kernels
    over `build_unary_op`-style iterators (`check_all_same_dtype`), and
    heaviside / the loss kernels (lerp.Tensor: see `_pw_lerp`) below set their output from
    `self`'s options; threshold, the other backwards and the math ops are
    TensorIterator binary/ternary ops that cast.
    """
    return (
        kind == "elu"
        or kind == "hardshrink"
        or kind == "softshrink"
        or kind == "hardsigmoid"
        or kind == "hardswish"
        or kind == "hardtanh"
        or kind == "leaky_relu"
        or kind == "softplus"
        or kind == "mish"
        or kind == "log_sigmoid"
        or kind == "log_sigmoid_backward"
        or kind == "heaviside"
        or kind == "mse_backward"
        or kind == "huber_backward"
        or kind == "bce"
        or kind == "bce_backward"
        or kind == "bce_logits"
    )


def _pw_finish(
    rets: Values, dest: Optional[T], var res: Res, exact_dtype: Bool = False
) raises:
    if dest:
        _b_store_out(rets, dest.value(), res^, exact_dtype)
    else:
        _b_ret(rets, res^)


def _pw_out_of(v: Value, a: Side, b: Side, c: Side) raises -> T:
    """The caller's `out=`, checked like TensorIterator's meta: no internal
    overlap, no partial overlap with an input, and never resized when it is
    one of the inputs (see `_b_out_guard`)."""
    var dest = v_tensor(v)
    if not dest.on_mojo():
        raise Error("expected `out` to be a mojo tensor")
    for k in range(3):
        var side = a.copy() if k == 0 else (b.copy() if k == 1 else c.copy())
        if side.is_t and side.t.value().device != dest.device:
            raise Error("expected `out` and the inputs on the same mojo device")
    assert_no_internal_overlap(dest)
    _b_out_guard3(dest, a, b, c)
    return dest^


def _pw_inplace_self(
    args: Values, what: StaticString, n_operands: Int
) raises -> T:
    """`self` of an in-place op over args[0 .. n_operands), checked like
    TensorIterator's in-place meta: an operand that would broadcast self to
    a larger shape, or a self whose elements share memory, raises. An
    in-place op never resizes self (only an `out=` does)."""
    var self = _b_self(args[unsafe_offset=0], what)
    _b_inplace_destination(self)
    for i in range(1, n_operands):
        var side = _b_side(args[unsafe_offset=i])
        if side.is_t:
            _b_inplace_operand(self, side.t.value())
    return self^


def _pw_math(
    kind: StaticString,
    policy: Int,
    f64_ok: Bool,
    arity: Int,
    args: Values,
    rets: Values,
    out_index: Int,
    params: SIMD[DType.float64, 4] = _p(),
    exact_dtype: Bool = False,
) raises:
    """A math op over args[0 .. arity): promoted, computed, stored.
    `exact_dtype` (or `_pw_out_exact(kind)`) demands an `out` of the result
    dtype instead of casting into it."""
    var a = _b_side(args[unsafe_offset=0])
    var b = _b_side(args[unsafe_offset=1]) if arity >= 2 else _none_side()
    var c = _b_side(args[unsafe_offset=2]) if arity >= 3 else _none_side()
    var common = _pw_result_type(a, b, c, arity)
    var compute = _pw_compute_dtype(kind, common, policy, f64_ok)
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(args[unsafe_offset=out_index], a, b, c)
    _pw_finish(
        rets,
        dest,
        _pw_run(kind, arity, a, b, c, compute, compute, params, dest),
        exact_dtype or _pw_out_exact(kind),
    )


def _pw_math_inplace(
    kind: StaticString,
    policy: Int,
    f64_ok: Bool,
    arity: Int,
    args: Values,
    rets: Values,
    params: SIMD[DType.float64, 4] = _p(),
) raises:
    """`op_(self, ...)` of a same-dtype math op over args[0 .. arity):
    computed into self (straight into it when it is dense), never resizing
    it."""
    var self = _pw_inplace_self(args, kind, arity)
    var a = _b_tside(self)
    var b = _b_side(args[unsafe_offset=1]) if arity >= 2 else _none_side()
    var c = _b_side(args[unsafe_offset=2]) if arity >= 3 else _none_side()
    var common = _pw_result_type(a, b, c, arity)
    var compute = _pw_compute_dtype(kind, common, policy, f64_ok)
    _b_store_inplace(
        rets,
        self,
        _pw_run(kind, arity, a, b, c, compute, compute, params, self.copy()),
    )


# ---------------------------------------------------------------------------
# binary math
# ---------------------------------------------------------------------------


# aten::atan2(Tensor self, Tensor other) -> Tensor
def op_atan2(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("atan2", P_FLOAT, True, 2, args, rets, -1)


# aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_atan2_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("atan2", P_FLOAT, True, 2, args, rets, 2)


# aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
# aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
def op_copysign(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("copysign", P_FLOAT, True, 2, args, rets, -1)


# aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_copysign_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("copysign", P_FLOAT, True, 2, args, rets, 2)


# aten::fmax(Tensor self, Tensor other) -> Tensor
def op_fmax(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmax", P_ALL, True, 2, args, rets, -1)


# aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_fmax_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmax", P_ALL, True, 2, args, rets, 2)


# aten::fmin(Tensor self, Tensor other) -> Tensor
def op_fmin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmin", P_ALL, True, 2, args, rets, -1)


# aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_fmin_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmin", P_ALL, True, 2, args, rets, 2)


# aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
# aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
def op_fmod(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmod", P_NUMERIC, True, 2, args, rets, -1)


# aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
# aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
def op_fmod_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("fmod", P_NUMERIC, True, 2, args, rets, 2)


# aten::gcd(Tensor self, Tensor other) -> Tensor
def op_gcd(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("gcd", P_INT, True, 2, args, rets, -1)


# aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_gcd_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("gcd", P_INT, True, 2, args, rets, 2)


# aten::lcm(Tensor self, Tensor other) -> Tensor
def op_lcm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("lcm", P_INT, True, 2, args, rets, -1)


# aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_lcm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("lcm", P_INT, True, 2, args, rets, 2)


def _pw_side_stype(side: Side) -> Int32:
    """The dtype of an operand's tensor: a CPU 0-d tensor read back as a
    Scalar keeps its own (a wrapped Python number is double / long / bool)."""
    if side.is_t:
        return side.t.value().stype
    if side.zero_st >= 0:
        return side.zero_st
    if side.s.value().is_bool:
        return ST_BOOL
    return ST_INT64 if side.s.value().is_int else ST_FLOAT64


def _pw_same_dtype(name: StaticString, args: Values) raises:
    """heaviside's meta check: both tensors of one dtype, a CPU 0-d
    `values` included."""
    var a = _pw_side_stype(_b_side(args[unsafe_offset=0]))
    var b = _pw_side_stype(_b_side(args[unsafe_offset=1]))
    if a != b:
        raise Error(
            String(name)
            + " is not yet implemented for tensors with different dtypes."
        )


# aten::heaviside(Tensor self, Tensor values) -> Tensor
def op_heaviside(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_same_dtype("heaviside", args)
    _pw_math("heaviside", P_ALL, True, 2, args, rets, -1)


# aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
def op_heaviside_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_same_dtype("heaviside", args)
    _pw_math("heaviside", P_ALL, True, 2, args, rets, 2)


# aten::hypot(Tensor self, Tensor other) -> Tensor
def op_hypot(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("hypot", P_FLOAT_ONLY, True, 2, args, rets, -1)


# aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_hypot_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("hypot", P_FLOAT_ONLY, True, 2, args, rets, 2)


# aten::igamma(Tensor self, Tensor other) -> Tensor
def op_igamma(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("igamma", P_FLOAT, False, 2, args, rets, -1)


# aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_igamma_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("igamma", P_FLOAT, False, 2, args, rets, 2)


# aten::igammac(Tensor self, Tensor other) -> Tensor
def op_igammac(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("igammac", P_FLOAT, False, 2, args, rets, -1)


# aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_igammac_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("igammac", P_FLOAT, False, 2, args, rets, 2)


# aten::logaddexp(Tensor self, Tensor other) -> Tensor
def op_logaddexp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("logaddexp", P_FLOAT_ONLY, True, 2, args, rets, -1)


# aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_logaddexp_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("logaddexp", P_FLOAT_ONLY, True, 2, args, rets, 2)


# aten::logaddexp2(Tensor self, Tensor other) -> Tensor
def op_logaddexp2(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("logaddexp2", P_FLOAT_ONLY, True, 2, args, rets, -1)


# aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_logaddexp2_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("logaddexp2", P_FLOAT_ONLY, True, 2, args, rets, 2)


# aten::nextafter(Tensor self, Tensor other) -> Tensor
def op_nextafter(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("nextafter", P_FLOAT_ONLY, True, 2, args, rets, -1)


# aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_nextafter_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("nextafter", P_FLOAT_ONLY, True, 2, args, rets, 2)


# aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
def op_xlogy(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("xlogy", P_FLOAT, True, 2, args, rets, -1)


# aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_xlogy_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("xlogy", P_FLOAT, True, 2, args, rets, 2)


# aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
def op_special_xlog1py(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("xlog1py", P_FLOAT, True, 2, args, rets, -1)


# aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_special_xlog1py_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("xlog1py", P_FLOAT, True, 2, args, rets, 2)


# aten::special_zeta(Tensor self, Tensor other) -> Tensor
def op_special_zeta(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("zeta", P_FLOAT, True, 2, args, rets, -1)


# aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_special_zeta_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_math("zeta", P_FLOAT, True, 2, args, rets, 2)


# The special polynomials: aten::special_<name>(Tensor x, Tensor n) -> Tensor
# and aten::special_<name>.out(Tensor x, Tensor n, *, Tensor(a!) out).
def op_poly[
    kind: StaticString, out_index: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math(kind, P_FLOAT, True, 2, args, rets, out_index)


# ---------------------------------------------------------------------------
# shifts
# ---------------------------------------------------------------------------


def _pw_shift(
    kind: StaticString, args: Values, rets: Values, out_index: Int
) raises:
    """bitwise_left/right_shift and __lshift__/__rshift__: integral only.
    torch shifts the integral result type, so a float operand declines."""
    _pw_math(kind, P_INT, True, 2, args, rets, out_index)


# aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
# aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
# aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
def op_lshift(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift("lshift", args, rets, -1)


# aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_lshift_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift("lshift", args, rets, 2)


# aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
# aten::__rshift__.Tensor / .Scalar
def op_rshift(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift("rshift", args, rets, -1)


# aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_rshift_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift("rshift", args, rets, 2)


def _pw_shift_inplace(kind: StaticString, args: Values, rets: Values) raises:
    """__ilshift__ / __irshift__: computed in the promoted dtype, then cast
    into self when that cast is category-safe (int8 `<<=` int64 is int8),
    as TensorIterator does for any in-place op (`_b_store_inplace`)."""
    var self = _pw_inplace_self(args, "an in-place shift", 2)
    var a = _b_tside(self)
    var b = _b_side(args[unsafe_offset=1])
    var common = _pw_result_type(a, b, _none_side(), 2)
    var compute = _pw_compute_dtype(kind, common, P_INT, True)
    var res = _pw_run(
        kind, 2, a, b, _none_side(), compute, compute, _p(), self.copy()
    )
    _b_store_inplace(rets, self, res^)


# aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
# aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
def op_ilshift(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift_inplace("lshift", args, rets)


# aten::__irshift__.Tensor / .Scalar
def op_irshift(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_shift_inplace("rshift", args, rets)


# ---------------------------------------------------------------------------
# frexp
# ---------------------------------------------------------------------------


def _pw_frexp(self: T) raises -> Tuple[Res, Res]:
    if not self.on_mojo():
        unsupported("frexp of a tensor outside the mojo device")
    if not _pw_is_float(self.stype):
        raise Error("torch.frexp() only supports floating-point dtypes")
    var a = _b_tside(self)
    var mant = _pw_run(
        "frexp_mantissa",
        1,
        a,
        _none_side(),
        _none_side(),
        self.stype,
        self.stype,
        _p(),
        None,
    )
    var expo = _pw_run(
        "frexp_exponent",
        1,
        a,
        _none_side(),
        _none_side(),
        self.stype,
        ST_INT32,
        _p(),
        None,
    )
    return (mant^, expo^)


# aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
def op_frexp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var r = _pw_frexp(v_tensor(args[unsafe_offset=0]))
    ret_tensor(rets, 0, r[0].t)
    ret_tensor(rets, 1, r[1].t)


# aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
def op_frexp_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mant = v_tensor(args[unsafe_offset=1])
    var expo = v_tensor(args[unsafe_offset=2])
    if mant.stype != self.stype:
        raise Error(
            "torch.frexp() expects mantissa to have dtype " + String(self.stype)
        )
    if expo.stype != ST_INT32:
        raise Error("torch.frexp() expects exponent to have int dtype")
    var r = _pw_frexp(self)
    _pw_store_slot(rets, 0, mant, r[0].copy())
    _pw_store_slot(rets, 1, expo, r[1].copy())


def _pw_store_slot(rets: Values, slot: Int, dest: T, var res: Res) raises:
    """Copy a same-dtype result into the caller's `dest` (resized when its
    shape differs) and return it in slot `slot`."""
    var held = own(res.t.copy())
    held.live = res.owned
    var d = dest.copy()
    assert_no_internal_overlap(d)
    if not d.same_shape(held.t):
        resize_out(d, held.t.shape, held.t.rank)
    _b_copy_into(d, held.t)
    _ = held^
    ret_ref(rets, slot, d)


# ---------------------------------------------------------------------------
# ternary: lerp.Tensor, clamp.Tensor; pow with a Scalar base
# ---------------------------------------------------------------------------


def _pw_lerp(args: Values, rets: Values, out_index: Int) raises:
    var start = _b_side(args[unsafe_offset=0])
    var end = _b_side(args[unsafe_offset=1])
    var weight = _b_side(args[unsafe_offset=2])
    if start.is_t and end.is_t:
        if start.t.value().stype != end.t.value().stype:
            raise Error(
                "expected dtype "
                + dtype_name(start.t.value().stype)
                + " for `end` but got dtype "
                + dtype_name(end.t.value().stype)
            )
    # TORCH_META_FUNC(lerp_Tensor): a 0-dim weight is promoted with the
    # other operands (and the result cast into `out`); any other weight must
    # have self's dtype, and so must `out`.
    var promote_weight = not weight.is_t or weight.t.value().rank == 0
    if (
        not promote_weight
        and start.is_t
        and weight.t.value().stype != start.t.value().stype
    ):
        raise Error(
            "expected dtype "
            + dtype_name(start.t.value().stype)
            + " for `weight` but got dtype "
            + dtype_name(weight.t.value().stype)
        )
    _pw_math(
        "lerp",
        P_FLOAT_ONLY,
        True,
        3,
        args,
        rets,
        out_index,
        exact_dtype=not promote_weight,
    )


# aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
def op_lerp_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_lerp(args, rets, -1)


# aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
def op_lerp_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_lerp(args, rets, 3)


def _pw_clamp_tensor(args: Values, rets: Values, out_index: Int) raises:
    """clamp.Tensor with both bounds: ATen's clamp_stub (NaN value, then a
    NaN bound, wins). One bound is maximum_stub / minimum_stub, which
    binary.mojo's clamp_min.Tensor / clamp_max.Tensor already run; torch
    only reaches this op with both, or with one for the out= form."""
    var has_min = args[unsafe_offset=1].tag != TAG_NONE
    var has_max = args[unsafe_offset=2].tag != TAG_NONE
    if not has_min and not has_max:
        raise Error(
            "torch.clamp: At least one of 'min' or 'max' must not be None"
        )
    var a = _b_side(args[unsafe_offset=0])
    if has_min and has_max:
        _pw_math("clamp", P_ALL, True, 3, args, rets, out_index)
        return
    # One bound: ATen's maximum_stub / minimum_stub over the promoted pair.
    var bound = _b_side(args[unsafe_offset=1 if has_min else 2])
    var common = _pw_result_type(a, bound, _none_side(), 2)
    var compute = _pw_compute_dtype("clamp", common, P_ALL, True)
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(args[unsafe_offset=out_index], a, bound, _none_side())
    if has_min:
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "maximum",
                2,
                a,
                bound,
                _none_side(),
                compute,
                compute,
                _p(),
                dest,
            ),
        )
    else:
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "minimum",
                2,
                a,
                bound,
                _none_side(),
                compute,
                compute,
                _p(),
                dest,
            ),
        )


# aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
def op_clamp_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_clamp_tensor(args, rets, -1)


# aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
def op_clamp_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_clamp_tensor(args, rets, 3)


def _pw_pow_scalar_base(args: Values, rets: Values, out_index: Int) raises:
    """pow(Scalar self, Tensor exponent): the base is the parameter, rounded
    to the result dtype as PowKernel.cu's cpu-scalar base
    (`iter.scalar_value<scalar_t>(1)`; MPS wraps it in a tensor of that
    dtype); floating results only."""
    var base = _b_side(args[unsafe_offset=0])
    var expo = _b_side(args[unsafe_offset=1])
    if base.is_t:
        unsupported("pow.Scalar with a tensor base")
    if _pw_int_side(base) and _pw_int_side(expo):
        _pw_ipow(args, rets, out_index, False)
        return
    var common = _pw_result_type(base, expo, _none_side(), 2)
    var compute = _pw_compute_dtype(
        "pow_scalar_base", common, P_FLOAT_ONLY, True
    )
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(
            args[unsafe_offset=out_index], expo, _none_side(), _none_side()
        )
    _pw_finish(
        rets,
        dest,
        _pw_run(
            "pow_scalar_base",
            1,
            expo,
            _none_side(),
            _none_side(),
            compute,
            compute,
            _p(_round_to(base.s.value().f, compute)),
            dest,
        ),
    )


# aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
def op_pow_scalar_base(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_pow_scalar_base(args, rets, -1)


# aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
def op_pow_scalar_base_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_pow_scalar_base(args, rets, 2)


def _pw_lerp_scalar(
    args: Values, rets: Values, out_index: Int, in_place: Bool = False
) raises:
    """lerp.Scalar: the weight is a parameter, converted to opmath like
    Lerp.cu. `in_place` is lerp_ (self written, never resized)."""
    var start = _b_side(args[unsafe_offset=0])
    var end = _b_side(args[unsafe_offset=1])
    if start.is_t and end.is_t and start.t.value().stype != end.t.value().stype:
        raise Error(
            "expected dtype "
            + String(start.t.value().stype)
            + " for `end` but got dtype "
            + String(end.t.value().stype)
        )
    var params = _p(v_f64(args[unsafe_offset=2]))
    if in_place:
        _pw_math_inplace(
            "lerp_scalar", P_FLOAT_ONLY, True, 2, args, rets, params
        )
    else:
        _pw_math(
            "lerp_scalar",
            P_FLOAT_ONLY,
            True,
            2,
            args,
            rets,
            out_index,
            params,
        )


# aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
def op_lerp_scalar_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    # Every dtype, float32 included: binary.mojo's float32 composite (sub,
    # then a scaled add) rounded alpha * diff before the add, where Lerp.cu
    # fuses it (1 - 2 ulp off CUDA on a quarter of the elements).
    _pw_lerp_scalar(args, rets, -1)


# aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
def op_lerp_scalar_out_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_lerp_scalar(args, rets, 3)


def _is_f32_pair(args: Values) raises -> Bool:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    return a.stype == ST_FLOAT32 and b.stype == ST_FLOAT32


# aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
def op_lerp_scalar__any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    # float32 in place keeps binary.mojo's route: its fused foreach kernel
    # (contracted like Lerp.cu) on dense operands, and the in-place layout
    # contract (expanded self, overlaps) on the rest.
    if _is_f32_pair(args):
        op_lerp_scalar_(args, n_args, rets, n_rets)
    else:
        _pw_lerp_scalar(args, rets, -1, in_place=True)


def _pw_int_side(side: Side) -> Bool:
    if side.is_t:
        return _pw_is_int(side.t.value().stype)
    return side.s.value().is_int and not side.s.value().is_bool


def _pw_ipow(
    args: Values, rets: Values, out_index: Int, scalar_exponent: Bool
) raises:
    """Integer pow (Pow.h powi) for two integral operands."""
    if scalar_exponent:
        var e = _b_side(args[unsafe_offset=1])
        if e.s.value().i < 0:
            raise Error("Integers to negative integer powers are not allowed.")
    _pw_math("ipow", P_INT, True, 2, args, rets, out_index)


def _pw_both_int(args: Values) raises -> Bool:
    return _pw_int_side(_b_side(args[unsafe_offset=0])) and _pw_int_side(
        _b_side(args[unsafe_offset=1])
    )


def _pw_rsub_alpha(args: Values, rets: Values, out_index: Int) raises -> Bool:
    """rsub with alpha != 1 on a floating result: `other - alpha * self` as
    one fma in opmath, as CUDA's add kernel computes sub(other, self, alpha)
    (the generic alpha cascade rounds alpha * self to the dtype first).
    False for alpha 1 and integral results, which binary.mojo keeps."""
    var alpha = v_f64(args[unsafe_offset=2])
    if alpha == 1.0:
        return False
    var a = _b_side(args[unsafe_offset=0])
    var b = _b_side(args[unsafe_offset=1])
    if not a.is_t:
        return False
    var common = _pw_result_type(a, b, _none_side(), 2)
    if not _pw_is_float(common):
        return False
    # `alpha.to<scalar_t>()`: a finite alpha past the result dtype's range
    # raises, as ATen's checked conversion does.
    alpha = scalar_to_float(args[unsafe_offset=2], common)
    var compute = _pw_compute_dtype("rsub_alpha", common, P_FLOAT, True)
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(args[unsafe_offset=out_index], a, b, _none_side())
    if b.is_t:
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "rsub_alpha",
                2,
                a,
                b,
                _none_side(),
                compute,
                compute,
                _p(alpha),
                dest,
            ),
            True,  # rsub: `out` of the result dtype only
        )
    else:
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "rsub_alpha_scalar",
                1,
                a,
                _none_side(),
                _none_side(),
                compute,
                compute,
                _p(alpha, b.s.value().f),
                dest,
            ),
            True,  # rsub: `out` of the result dtype only
        )
    return True


def _pw_is_bool_side(side: Side) -> Bool:
    if side.is_t:
        return side.t.value().stype == ST_BOOL
    return side.s.value().is_bool


def _pw_rsub_check(args: Values) raises:
    """BinaryOps.cpp's sub_check and alpha_check (rsub is sub(other, self,
    alpha)): no bool operand, no bool alpha (it would need a bool result,
    which sub_check already refused), no floating alpha for an integral
    result."""
    var a = _pw_is_bool_side(_b_side(args[unsafe_offset=0]))
    var b = _pw_is_bool_side(_b_side(args[unsafe_offset=1]))
    if a and b:
        raise Error(
            "Subtraction, the `-` operator, with two bool tensors is not"
            " supported. Use the `^` or `logical_xor()` operator instead."
        )
    if a or b:
        raise Error(
            "Subtraction, the `-` operator, with a bool tensor is not"
            " supported. If you are trying to invert a mask, use the `~` or"
            " `logical_not()` operator instead."
        )
    var alpha = args[unsafe_offset=2].tag
    if alpha == TAG_SCALAR_BOOL or alpha == TAG_BOOL:
        raise Error("Boolean alpha only supported for Boolean results.")
    if alpha == TAG_SCALAR_DOUBLE or alpha == TAG_DOUBLE:
        var common = _pw_result_type(
            _b_side(args[unsafe_offset=0]),
            _b_side(args[unsafe_offset=1]),
            _none_side(),
            2,
        )
        if not _pw_is_float(common):
            raise Error(
                "For integral input tensors, argument alpha must not be a"
                " floating point number."
            )


# aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
def op_rsub_any(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_rsub_check(args)
    if not _pw_rsub_alpha(args, rets, -1):
        op_rsub(args, n_args, rets, n_rets)


# aten::rsub.Tensor_out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
# aten::rsub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
def op_rsub_out_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_rsub_check(args)
    if not _pw_rsub_alpha(args, rets, 3):
        op_rsub_out(args, n_args, rets, n_rets)


# aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
def op_pow_tensor_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _pw_both_int(args):
        _pw_ipow(args, rets, -1, False)
    else:
        op_pow_tensor(args, n_args, rets, n_rets)


# aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
def op_pow_tensor_out_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _pw_both_int(args):
        _pw_ipow(args, rets, 2, False)
    else:
        op_pow_tensor_out(args, n_args, rets, n_rets)


def _pw_pow_f64_scalar(
    args: Values, rets: Values, out_index: Int
) raises -> Bool:
    """A float64 tensor to a real scalar power: PowKernel.cu's special
    exponents (products, sqrt, rsqrt, reciprocal) before the full pow, which
    the elementwise family's PowScalarSpec does for the float types only."""
    var base = _b_side(args[unsafe_offset=0])
    var expo = _b_side(args[unsafe_offset=1])
    if not base.is_t or expo.is_t:
        return False
    if base.t.value().stype != ST_FLOAT64:
        return False
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(
            args[unsafe_offset=out_index], base, _none_side(), _none_side()
        )
    _pw_finish(
        rets,
        dest,
        _pw_run(
            "pow_tensor_scalar",
            1,
            base,
            _none_side(),
            _none_side(),
            ST_FLOAT64,
            ST_FLOAT64,
            _p(expo.s.value().f),
            dest,
        ),
    )
    return True


# aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
def op_pow_scalar_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _pw_both_int(args):
        _pw_ipow(args, rets, -1, True)
    elif not _pw_pow_f64_scalar(args, rets, -1):
        op_pow_scalar(args, n_args, rets, n_rets)


# aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
def op_pow_scalar_out_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _pw_both_int(args):
        _pw_ipow(args, rets, 2, True)
    elif not _pw_pow_f64_scalar(args, rets, 2):
        op_pow_scalar_out(args, n_args, rets, n_rets)


# ---------------------------------------------------------------------------
# activations and their backwards
# ---------------------------------------------------------------------------


def _round_to(v: Float64, st: Int32) -> Float64:
    """A Scalar converted to the tensor's own dtype (`value.to<scalar_t>()`
    in the kernels that compare in scalar_t). c10::Half / c10::BFloat16 are
    built from a float, so a double reaches them rounded twice."""
    if st == ST_FLOAT16:
        return (
            v.cast[DType.float32]().cast[DType.float16]().cast[DType.float64]()
        )
    if st == ST_BFLOAT16:
        return (
            v.cast[DType.float32]().cast[DType.bfloat16]().cast[DType.float64]()
        )
    if st == ST_FLOAT32:
        return v.cast[DType.float32]().cast[DType.float64]()
    return v


def _pw_act(
    kind: StaticString,
    args: Values,
    rets: Values,
    a_index: Int,
    b_index: Int,
    out_index: Int,
    params: SIMD[DType.float64, 4],
    f64_ok: Bool = True,
    policy: Int = P_FLOAT_ONLY,
) raises:
    """An activation (b_index < 0) or its backward: floating operands of one
    dtype, operand a = args[a_index], b = args[b_index]."""
    var a = _b_side(args[unsafe_offset=a_index])
    var arity = 1 if b_index < 0 else 2
    var b = _b_side(args[unsafe_offset=b_index]) if arity == 2 else _none_side()
    var common = _pw_result_type(a, b, _none_side(), arity)
    var compute = _pw_compute_dtype(kind, common, policy, f64_ok)
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(args[unsafe_offset=out_index], a, b, _none_side())
    _pw_finish(
        rets,
        dest,
        _pw_run(
            kind, arity, a, b, _none_side(), compute, compute, params, dest
        ),
        _pw_out_exact(kind),
    )


def _pw_act_inplace(
    kind: StaticString,
    args: Values,
    rets: Values,
    params: SIMD[DType.float64, 4],
    policy: Int = P_FLOAT_ONLY,
) raises:
    """`op_(self, ...)`: computed straight into self (flat, so exact)."""
    var self = _pw_inplace_self(args, kind, 1)
    var a = _b_tside(self)
    var compute = _pw_compute_dtype(kind, self.stype, policy, True)
    var res = _pw_run(
        kind,
        1,
        a,
        _none_side(),
        _none_side(),
        compute,
        compute,
        params,
        self.copy(),
    )
    _b_store_out(rets, self, res^)


def _self_stype(args: Values, i: Int) raises -> Int32:
    return v_tensor(args[unsafe_offset=i]).stype


def _elu_p(args: Values, first: Int) raises -> SIMD[DType.float64, 4]:
    return _p(
        v_f64(args[unsafe_offset=first]),
        v_f64(args[unsafe_offset=first + 1]),
        v_f64(args[unsafe_offset=first + 2]),
    )


# aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
def op_elu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("elu", args, rets, 0, -1, -1, _elu_p(args, 1))


# aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
def op_elu_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("elu", args, rets, 0, -1, 4, _elu_p(args, 1))


def _elu_backward_p(args: Values) raises -> SIMD[DType.float64, 4]:
    return _p(
        v_f64(args[unsafe_offset=1]),
        v_f64(args[unsafe_offset=2]),
        v_f64(args[unsafe_offset=3]),
        1.0 if v_bool(args[unsafe_offset=4]) else 0.0,
    )


def _elu_backward_check(args: Values) raises:
    # ATen's elu_backward meta: a negative alpha cannot be recovered from
    # the result.
    if v_bool(args[unsafe_offset=4]) and v_f64(args[unsafe_offset=1]) < 0:
        raise Error(
            "In-place elu backward calculation is triggered with a negative"
            " slope which is not supported. This is caused by calling"
            " in-place forward function with a negative slope, please call"
            " out-of-place version instead."
        )


# aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor
def op_elu_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _elu_backward_check(args)
    _pw_act("elu_backward", args, rets, 0, 5, -1, _elu_backward_p(args))


# aten::elu_backward.grad_input(..., Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_elu_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _elu_backward_check(args)
    _pw_act("elu_backward", args, rets, 0, 5, 6, _elu_backward_p(args))


def _lambd_p(args: Values, i: Int, st: Int32) raises -> SIMD[DType.float64, 4]:
    return _p(_round_to(v_f64(args[unsafe_offset=i]), st))


# aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
def op_hardshrink(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act(
        "hardshrink",
        args,
        rets,
        0,
        -1,
        -1,
        _lambd_p(args, 1, _self_stype(args, 0)),
    )


# aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
def op_hardshrink_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "hardshrink",
        args,
        rets,
        0,
        -1,
        2,
        _lambd_p(args, 1, _self_stype(args, 0)),
    )


# aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
def op_softshrink(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if v_f64(args[unsafe_offset=1]) < 0:
        raise Error(
            "lambda must be greater or equal to 0, but found to be "
            + String(v_f64(args[unsafe_offset=1]))
            + "."
        )
    _pw_act(
        "softshrink",
        args,
        rets,
        0,
        -1,
        -1,
        _lambd_p(args, 1, _self_stype(args, 0)),
    )


# aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
def op_softshrink_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if v_f64(args[unsafe_offset=1]) < 0:
        raise Error(
            "lambda must be greater or equal to 0, but found to be "
            + String(v_f64(args[unsafe_offset=1]))
            + "."
        )
    _pw_act(
        "softshrink",
        args,
        rets,
        0,
        -1,
        2,
        _lambd_p(args, 1, _self_stype(args, 0)),
    )


# aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
# aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
def op_shrink_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "shrink_backward",
        args,
        rets,
        0,
        1,
        -1,
        _lambd_p(args, 2, _self_stype(args, 1)),
    )


# aten::hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
# aten::softshrink_backward.grad_input(..., *, Tensor(a!) grad_input) -> Tensor(a!)
def op_shrink_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "shrink_backward",
        args,
        rets,
        0,
        1,
        3,
        _lambd_p(args, 2, _self_stype(args, 1)),
    )


# aten::hardsigmoid(Tensor self) -> Tensor
def op_hardsigmoid(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("hardsigmoid", args, rets, 0, -1, -1, _p())


# aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_hardsigmoid_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("hardsigmoid", args, rets, 0, -1, 1, _p())


# aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
def op_hardsigmoid_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act_inplace("hardsigmoid", args, rets, _p())


# aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
def op_hardsigmoid_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("hardsigmoid_backward", args, rets, 0, 1, -1, _p())


# aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_hardsigmoid_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("hardsigmoid_backward", args, rets, 0, 1, 2, _p())


# aten::hardswish(Tensor self) -> Tensor
def op_hardswish(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("hardswish", args, rets, 0, -1, -1, _p())


# aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_hardswish_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("hardswish", args, rets, 0, -1, 1, _p())


# aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
def op_hardswish_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("hardswish", args, rets, _p())


# aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
def op_hardswish_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("hardswish_backward", args, rets, 0, 1, -1, _p())


def _int_exact_param(v: Value, st: Int32) raises -> Float64:
    """A Scalar parameter of an op that applies it in scalar_t. On an
    integer tensor it travels as the int64's bits in the float64 slot (the
    kernel reads it back as `param_dtype`, int64): exact across int64, never
    through a float, and checked against the dtype as
    `Scalar::to<scalar_t>()` checks it."""
    if _pw_is_int(st):
        return bits_f64(Int64(scalar_to_int(v, st)))
    return v_f64(v)


def _hardtanh_p(args: Values) raises -> SIMD[DType.float64, 4]:
    """hardtanh's bounds (args 1, 2) in self's dtype. On an integer self
    ATen's hardtanh_out first takes each as `toLong()` (a float truncates:
    -0.5 is 0), refuses a negative one on uint8, and clamps with those
    integers, which `Scalar::to<scalar_t>()` then checks against the dtype."""
    var st = _self_stype(args, 0)
    if not _pw_is_int(st):
        return _p(v_f64(args[unsafe_offset=1]), v_f64(args[unsafe_offset=2]))
    var lo = scalar_to_int(args[unsafe_offset=1], ST_INT64)
    var hi = scalar_to_int(args[unsafe_offset=2], ST_INT64)
    if st == ST_UINT8 and (lo < 0 or hi < 0):
        raise Error(
            "cannot do hardtanh on an unsigned type with negative limits"
        )
    return _p(
        _int_exact_param(Value(TAG_SCALAR_INT, 0, Int64(lo), 0), st),
        _int_exact_param(Value(TAG_SCALAR_INT, 0, Int64(hi), 0), st),
    )


def _hardtanh_backward_p(args: Values) raises -> SIMD[DType.float64, 4]:
    """hardtanh_backward's bounds (args 2, 3): a floating binary op over
    (grad_output, self), so they are applied in the promoted floating
    dtype -- never encoded in grad_output's own (possibly integer) one."""
    return _p(v_f64(args[unsafe_offset=2]), v_f64(args[unsafe_offset=3]))


# aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
def op_hardtanh(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act(
        "hardtanh", args, rets, 0, -1, -1, _hardtanh_p(args), True, P_NUMERIC
    )


# aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
def op_hardtanh_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "hardtanh", args, rets, 0, -1, 3, _hardtanh_p(args), True, P_NUMERIC
    )


# aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
def op_hardtanh_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("hardtanh", args, rets, _hardtanh_p(args), P_NUMERIC)


# aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
def op_hardtanh_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "hardtanh_backward", args, rets, 0, 1, -1, _hardtanh_backward_p(args)
    )


# aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_hardtanh_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "hardtanh_backward", args, rets, 0, 1, 4, _hardtanh_backward_p(args)
    )


# aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
def op_leaky_relu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act(
        "leaky_relu", args, rets, 0, -1, -1, _p(v_f64(args[unsafe_offset=1]))
    )


# aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
def op_leaky_relu_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "leaky_relu", args, rets, 0, -1, 2, _p(v_f64(args[unsafe_offset=1]))
    )


# aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
def op_leaky_relu_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("leaky_relu", args, rets, _p(v_f64(args[unsafe_offset=1])))


def _leaky_backward_check(args: Values) raises:
    if v_bool(args[unsafe_offset=3]) and v_f64(args[unsafe_offset=2]) < 0:
        raise Error(
            "In-place leakyReLu backward calculation is triggered with a"
            " negative slope which is not supported. This is caused by"
            " calling in-place forward function with a negative slope, please"
            " call out-of-place version instead. File an issue at"
            " https://github.com/pytorch/pytorch if you do require supporting"
            " in-place leakRelu backward calculation with negative slope"
        )


# aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor
def op_leaky_relu_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _leaky_backward_check(args)
    # The CUDA iterator's operand order: (self, grad_output).
    _pw_act(
        "leaky_relu_backward",
        args,
        rets,
        1,
        0,
        -1,
        _p(v_f64(args[unsafe_offset=2])),
    )


# aten::leaky_relu_backward.grad_input(..., *, Tensor(a!) grad_input) -> Tensor(a!)
def op_leaky_relu_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _leaky_backward_check(args)
    _pw_act(
        "leaky_relu_backward",
        args,
        rets,
        1,
        0,
        4,
        _p(v_f64(args[unsafe_offset=2])),
    )


def _softplus_p(args: Values, i: Int) raises -> SIMD[DType.float64, 4]:
    return _p(v_f64(args[unsafe_offset=i]), v_f64(args[unsafe_offset=i + 1]))


# aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
def op_softplus(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("softplus", args, rets, 0, -1, -1, _softplus_p(args, 1))


# aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
def op_softplus_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("softplus", args, rets, 0, -1, 3, _softplus_p(args, 1))


# aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor
def op_softplus_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("softplus_backward", args, rets, 0, 1, -1, _softplus_p(args, 2))


# aten::softplus_backward.grad_input(..., *, Tensor(a!) grad_input) -> Tensor(a!)
def op_softplus_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("softplus_backward", args, rets, 0, 1, 4, _softplus_p(args, 2))


# aten::mish(Tensor self) -> Tensor
def op_mish(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("mish", args, rets, 0, -1, -1, _p())


# aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_mish_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act("mish", args, rets, 0, -1, 1, _p())


# aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor
def op_mish_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("mish_backward", args, rets, 0, 1, -1, _p())


# aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor
def op_silu_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("silu_backward", args, rets, 0, 1, -1, _p())


# aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_silu_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("silu_backward", args, rets, 0, 1, 2, _p())


# aten::gelu_backward.grad_input(Tensor grad_output, Tensor self, *, str approximate='none', Tensor(a!) grad_input) -> Tensor(a!)
def op_gelu_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var approximate = v_string(args[unsafe_offset=2])
    if approximate == "none":
        _pw_act("gelu_backward_none", args, rets, 0, 1, 3, _p())
    elif approximate == "tanh":
        _pw_act("gelu_backward_tanh", args, rets, 0, 1, 3, _p())
    else:
        raise Error("approximate argument must be either none or tanh.")


# aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor
def op_gelu_backward_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """The tuned activation_backward kernels for same-shape float32 /
    bfloat16 operands on one device; everything else (float16, float64 in
    tanh mode, broadcasting, mixed dtypes) through the pointwise kinds, which
    run the same GeluBackwardCUDAKernelImpl formulas."""
    var grad = v_tensor(args[unsafe_offset=0])
    var self_t = v_tensor(args[unsafe_offset=1])
    var approximate = v_string(args[unsafe_offset=2])
    if approximate != "none" and approximate != "tanh":
        raise Error("approximate argument must be either none or tanh.")
    if (
        grad.on_mojo()
        and self_t.on_mojo()
        and grad.device == self_t.device
        and grad.dtype == self_t.dtype
        and (self_t.dtype == DType.float32 or self_t.dtype == DType.bfloat16)
        and grad.same_shape(self_t)
    ):
        op_gelu_backward(args, n_args, rets, n_rets)
        return
    if approximate == "none":
        _pw_act("gelu_backward_none", args, rets, 0, 1, -1, _p())
    else:
        _pw_act("gelu_backward_tanh", args, rets, 0, 1, -1, _p())


def _threshold_p(args: Values) raises -> SIMD[DType.float64, 4]:
    var st = _self_stype(args, 0)
    return _p(
        _round_to(_int_exact_param(args[unsafe_offset=1], st), st),
        _round_to(_int_exact_param(args[unsafe_offset=2], st), st),
    )


# aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
def op_threshold(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act(
        "threshold", args, rets, 0, -1, -1, _threshold_p(args), True, P_NUMERIC
    )


# aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
def op_threshold_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act(
        "threshold", args, rets, 0, -1, 3, _threshold_p(args), True, P_NUMERIC
    )


# aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
def op_threshold_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("threshold", args, rets, _threshold_p(args), P_NUMERIC)


# aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
def op_log_sigmoid_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    _pw_act("log_sigmoid", args, rets, 0, -1, -1, _p())
    # CUDA's buffer is an empty tensor: only the CPU kernel fills it.
    ret_tensor(rets, 1, _empty_1d(self))


def _empty_1d(like: T) raises -> T:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = 0
    return new_tensor(shape, 1, like.stype, like.device)


# aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
def op_log_sigmoid_forward_output(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("log_sigmoid", args, rets, 0, -1, 1, _p())
    ret_ref(rets, 1, v_tensor(args[unsafe_offset=2]))


# aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
def op_log_sigmoid_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    # The CUDA iterator's operand order: (self, grad_output); buffer unused.
    _pw_act("log_sigmoid_backward", args, rets, 1, 0, -1, _p())


# aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_log_sigmoid_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("log_sigmoid_backward", args, rets, 1, 0, 3, _p())


def _logit_eps(args: Values) raises -> SIMD[DType.float64, 4]:
    if v_is_none(args[unsafe_offset=2]):
        return _p(-1.0)
    return _p(v_f64(args[unsafe_offset=2]))


# aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
def op_logit_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("logit_backward", args, rets, 0, 1, -1, _logit_eps(args))


# aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_logit_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _pw_act("logit_backward", args, rets, 0, 1, 3, _logit_eps(args))


def _rrelu_check(args: Values) raises:
    var lower = v_f64(args[unsafe_offset=2])
    var upper = v_f64(args[unsafe_offset=3])
    if not (upper >= lower):
        raise Error(
            "Lower bound should be less than or equal to the upper bound"
        )


def _rrelu(args: Values, rets: Values, out_index: Int, in_place: Bool) raises:
    """rrelu_with_noise (RreluWithNoise.cu). Eval mode is leaky_relu with the
    mean slope. Training draws one uniform per element from the generator,
    exactly as `uniform_` does (same Philox offsets, same draw per index:
    ATen's rrelu kernel shares distribution_nullary_kernel's launch policy),
    then slope = draw * (upper - lower) + lower for x <= 0, written into
    `noise` (1 elsewhere)."""
    _rrelu_check(args)
    var lower = v_f64(args[unsafe_offset=2])
    var upper = v_f64(args[unsafe_offset=3])
    if not v_bool(args[unsafe_offset=4]):
        var slope = _p((lower + upper) / 2)
        if in_place:
            _pw_act_inplace("leaky_relu", args, rets, slope)
        else:
            _pw_act("leaky_relu", args, rets, 0, -1, out_index, slope)
        return
    var self = v_tensor(args[unsafe_offset=0])
    var noise = v_tensor(args[unsafe_offset=1])
    if not self.on_mojo() or not noise.on_mojo():
        unsupported("rrelu_with_noise with a tensor outside the mojo device")
    if not _pw_is_float(self.stype):
        unsupported("rrelu_with_noise on a non-floating tensor")
    if noise.stype != self.stype or not noise.same_shape(self):
        raise Error(
            "rrelu_with_noise: noise must match the input's dtype and shape"
        )
    var draw = own(new_tensor(self.shape, self.rank, self.stype, self.device))
    _draw(draw.t, "Uniform", 0.0, 1.0, 0, 0, v_generator(args[unsafe_offset=5]))
    var a = _b_tside(self)
    var b = _b_tside(draw.t)
    # (lower, range) as RreluWithNoise.cu's doubles: for float operands, the
    # float each rounds to plus the float remainder (the kernel sums the two
    # in double).
    var span = upper - lower
    var params = _p(lower, span)
    if self.stype != ST_FLOAT64:
        params = _p(
            lower,
            span,
            lower - lower.cast[DType.float32]().cast[DType.float64](),
            span - span.cast[DType.float32]().cast[DType.float64](),
        )
    var noise_res = _pw_run(
        "rrelu_noise",
        2,
        a,
        b,
        _none_side(),
        self.stype,
        self.stype,
        params,
        noise.copy(),
    )
    if noise_res.owned:
        # A non-dense noise: copy the dense result into it.
        var held = own(noise_res.t.copy())
        copy_strided_into(noise, held.t)
        _ = held^
    var dest = Optional[T]()
    if in_place:
        _b_inplace_destination(self)
        dest = self.copy()
    elif out_index >= 0:
        dest = _pw_out_of(
            args[unsafe_offset=out_index], a, _none_side(), _none_side()
        )
    var res = _pw_run(
        "rrelu_train",
        2,
        a,
        b,
        _none_side(),
        self.stype,
        self.stype,
        params,
        dest,
    )
    _ = draw^
    _pw_finish(rets, dest, res^)


# aten::rrelu_with_noise(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
def op_rrelu_with_noise(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _rrelu(args, rets, -1, False)


# aten::rrelu_with_noise.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_rrelu_with_noise_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _rrelu(args, rets, 6, False)


# aten::rrelu_with_noise_(Tensor(a!) self, Tensor(b!) noise, ...) -> Tensor(a!)
def op_rrelu_with_noise_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _rrelu(args, rets, -1, True)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# deg2rad / rad2deg / ldexp: ATen composes them from other ops (empty + mul,
# mul + pow); each is one pointwise kind here.
# ---------------------------------------------------------------------------

comptime M_PI_180 = 0.017453292519943295769236907684886127134428718885417
comptime M_180_PI = 57.295779513082320876798154814105170332405472466564


# aten::deg2rad(Tensor self) -> Tensor
def op_deg2rad(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("scale", P_FLOAT, True, 1, args, rets, -1, _p(M_PI_180))


# aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_deg2rad_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("scale", P_FLOAT, True, 1, args, rets, 1, _p(M_PI_180))


# aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
def op_deg2rad_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("scale", args, rets, _p(M_PI_180))


# aten::rad2deg(Tensor self) -> Tensor
def op_rad2deg(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("scale", P_FLOAT, True, 1, args, rets, -1, _p(M_180_PI))


# aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_rad2deg_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_math("scale", P_FLOAT, True, 1, args, rets, 1, _p(M_180_PI))


# aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
def op_rad2deg_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_act_inplace("scale", args, rets, _p(M_180_PI))


def _pw_narrow_code(narrow: Int32, compute: Int32) -> Float64:
    """`ldexp_pow2`'s p0: the dtype pow(2, other) rounds to before the
    product, when it is narrower than the compute dtype."""
    # The kernel computes a half compute dtype in float32, so a half pow
    # result is rounded even when it is the compute dtype itself (the pow
    # kernel stores it as c10::Half before the mul reads it back).
    if narrow == ST_FLOAT16:
        return 1.0
    if narrow == ST_BFLOAT16:
        return 2.0
    if narrow == ST_FLOAT32 and compute == ST_FLOAT64:
        return 3.0
    return 0.0


def _pw_ldexp(args: Values, rets: Values, out_index: Int) raises:
    """BinaryOps.cpp's ldexp / ldexp_out. A floating `self` with an integral
    exponent is its `_ldexp_int_exponent` route (::ldexp, the result in
    self's dtype); everything else is `self * _pow2(self, other)`, fused.
    (torch MPS registers no ldexp_stub and multiplies by pow(2, other) for an
    integral exponent too, which under- and overflows where ::ldexp does
    not: Apple GPUs keep the exact route.)"""
    var a = _b_side(args[unsafe_offset=0])
    var b = _b_side(args[unsafe_offset=1])
    if not a.is_t or not b.is_t:
        unsupported("ldexp with a scalar operand")
    var x = a.t.value().copy()
    var e = b.t.value().copy()
    var dest = Optional[T]()
    if out_index >= 0:
        dest = _pw_out_of(args[unsafe_offset=out_index], a, b, _none_side())
        var d = dest.value().copy()
        if not _pw_is_float(d.stype):
            raise Error(
                "ldexp can't be cast to the desired output type ", d.stype
            )
    var e_integral = _pw_is_int(e.stype) or e.stype == ST_BOOL
    if (
        e_integral
        and _pw_is_float(x.stype)
        and (not dest or dest.value().stype == x.stype)
    ):
        # bfloat16 computes in float32: its 8-bit mantissa cannot carry
        # every exponent that still changes the result.
        var compute = ST_FLOAT32 if x.stype == ST_BFLOAT16 else x.stype
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "ldexp", 2, a, b, _none_side(), compute, x.stype, _p(), dest
            ),
        )
        return
    # _pow2: pow(2.0, other) for an integral or float32 self (a Python
    # float base: an integral exponent gives the default dtype), else
    # full({}, 2.0, self.dtype).pow(other), promoted as a 0-d tensor.
    var pdt: Int32
    if _pw_is_float(e.stype):
        pdt = e.stype
        if not (_pw_is_int(x.stype) or x.stype == ST_BOOL) and (
            x.stype != ST_FLOAT32 and e.rank == 0
        ):
            pdt = promote_types(x.stype, e.stype)
    elif _pw_is_int(x.stype) or x.stype == ST_BOOL or x.stype == ST_FLOAT32:
        pdt = default_dtype()
    else:
        pdt = x.stype
    # mul(self, pow2): the pow result has other's rank.
    var state = _TypeState(ST_UNDEFINED, ST_UNDEFINED, ST_UNDEFINED)
    _pw_update(state, a)
    if e.rank == 0:
        state.zero = promote_types(state.zero, pdt)
    else:
        state.dim = promote_types(state.dim, pdt)
    var compute = _pw_combine(state.dim, _pw_combine(state.zero, state.wrapped))
    if not _pw_is_float(compute):
        unsupported("ldexp on dtype " + String(compute))
    var params = _p(_pw_narrow_code(pdt, compute))
    _pw_finish(
        rets,
        dest,
        _pw_run(
            "ldexp_pow2", 2, a, b, _none_side(), compute, compute, params, dest
        ),
    )


# aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
def op_ldexp(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_ldexp(args, rets, -1)


# aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
def op_ldexp_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _pw_ldexp(args, rets, 2)


# aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
def op_ldexp_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """ATen's ldexp_ is ldexp_out(self, self, other): exact in place, since
    the kernel is elementwise and dense (`_pw_run` writes self directly).
    Unlike ldexp_out, self is never resized: an `other` that broadcasts it
    to a larger shape raises."""
    var self = _pw_inplace_self(args, "ldexp_", 2)
    var a = _b_tside(self)
    var b = _b_side(args[unsafe_offset=1])
    _b_no_overlap_side(self, b)
    if not b.is_t:
        unsupported("ldexp_ with a scalar exponent")
    var e = b.t.value().copy()
    if not _pw_is_float(self.stype):
        raise Error(
            "ldexp can't be cast to the desired output type ", self.stype
        )
    var dest = Optional[T](self.copy())
    if _pw_is_int(e.stype) or e.stype == ST_BOOL:
        var compute = ST_FLOAT32 if self.stype == ST_BFLOAT16 else self.stype
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "ldexp", 2, a, b, _none_side(), compute, self.stype, _p(), dest
            ),
        )
        return
    var pdt = e.stype
    if self.stype != ST_FLOAT32 and e.rank == 0:
        pdt = promote_types(self.stype, e.stype)
    var compute = promote_types(self.stype, pdt)
    _pw_finish(
        rets,
        dest,
        _pw_run(
            "ldexp_pow2",
            2,
            a,
            b,
            _none_side(),
            compute,
            compute,
            _p(_pw_narrow_code(pdt, compute)),
            dest,
        ),
    )


# ---------------------------------------------------------------------------
# elementwise losses and their backwards (Loss.cpp / Loss.cu): one kind per
# formula, computed in scalar_t like the CUDA kernels; 'mean' / 'sum' reduce
# that result with the reduction group's own full reductions.
# ---------------------------------------------------------------------------

comptime REDUCTION_NONE = 0
comptime REDUCTION_MEAN = 1
comptime REDUCTION_SUM = 2


def _loss_reduction(v: Value) raises -> Int:
    var r = v_int(v)
    if r != REDUCTION_NONE and r != REDUCTION_MEAN and r != REDUCTION_SUM:
        raise Error(String(r) + " is not a valid value for reduction")
    return r


def _loss_reduce(var res: Res, reduction: Int) raises -> Res:
    """apply_loss_reduction: `mean()` / `sum()` over every element."""
    if reduction == REDUCTION_NONE or res.t.rank == 0:
        return res^
    if not res.owned:
        raise Error("internal: a reduced loss computed into its out=")
    var loss = own(res.t.copy())
    var dims = List[Int]()
    for i in range(loss.t.rank):
        dims.append(i)
    var out: Owned
    if reduction == REDUCTION_MEAN:
        if loss.t.dtype == DType.float64:
            unsupported("a float64 loss with reduction='mean'")
        out = _scalar_reduction(
            "nn", "MeanSpec", loss.t, dims, False, loss.t.stype, False, 0.0
        )
    else:
        out = _scalar_reduction(
            "reduction",
            "SumSpec",
            loss.t,
            dims,
            False,
            loss.t.stype,
            False,
            0.0,
        )
    _ = loss^
    return Res(out.take(), True)


def _loss_forward(
    kind: StaticString,
    arity: Int,
    a: Side,
    b: Side,
    c: Side,
    reduction: Int,
    params: SIMD[DType.float64, 4],
    out_v: Optional[Value],
    rets: Values,
    weight: Optional[T] = None,
) raises:
    """One loss kind over (a, b[, c]), times `weight` when given (its own
    rounding, as `loss.mul_(weight)`), then the reduction."""
    if not a.is_t or not b.is_t:
        unsupported(String(kind) + " with a scalar operand")
    var common = _pw_result_type(a, b, c, arity)
    var compute = _pw_compute_dtype(kind, common, P_FLOAT_ONLY, True)
    var dest = Optional[T]()
    if out_v:
        dest = _pw_out_of(out_v.value(), a, b, c)
    var direct = (
        dest.copy() if reduction == REDUCTION_NONE
        and not weight else (Optional[T]())
    )
    var res = _pw_run(kind, arity, a, b, c, compute, compute, params, direct)
    if weight:
        var loss = res.t.copy()
        var w = _b_tside(weight.value())
        var target = (
            dest.copy() if reduction == REDUCTION_NONE else Optional[T]()
        )
        var weighted = _pw_run(
            "mul_scale",
            2,
            _b_tside(loss),
            w,
            _none_side(),
            compute,
            compute,
            _p(1.0),
            target,
        )
        if res.owned:
            release(loss.h)
        res = weighted^
    _pw_finish(rets, dest, _loss_reduce(res^, reduction), _pw_out_exact(kind))


def _loss_backward(
    kind: StaticString,
    grad: Side,
    input: Side,
    target: Side,
    norm: Float64,
    p1: Float64,
    out_v: Optional[Value],
    rets: Values,
) raises:
    """(input, target, grad) through one backward kind; p0 = the norm and
    p1 = beta / delta, both rounded to scalar_t as `Scalar::to<scalar_t>`
    and `scalar_t(double)` do."""
    if not grad.is_t or not input.is_t or not target.is_t:
        unsupported(String(kind) + " with a scalar operand")
    var common = _pw_result_type(input, target, grad, 3)
    var compute = _pw_compute_dtype(kind, common, P_FLOAT_ONLY, True)
    var out_st = input.t.value().stype
    var dest = Optional[T]()
    if out_v:
        dest = _pw_out_of(out_v.value(), input, target, grad)
    var params = _p(_round_to(norm, compute), _round_to(p1, compute))
    _pw_finish(
        rets,
        dest,
        _pw_run(kind, 3, input, target, grad, compute, out_st, params, dest),
        _pw_out_exact(kind),
    )


def _loss_norm(input: Side, reduction: Int, scale: Float64) raises -> Float64:
    if reduction == REDUCTION_MEAN:
        return scale / Float64(input.t.value().numel)
    return scale


# aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
def op_mse_loss(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _loss_forward(
        "mse",
        2,
        _b_side(args[unsafe_offset=0]),
        _b_side(args[unsafe_offset=1]),
        _none_side(),
        _loss_reduction(args[unsafe_offset=2]),
        _p(),
        None,
        rets,
    )


# aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
def op_mse_loss_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _loss_forward(
        "mse",
        2,
        _b_side(args[unsafe_offset=0]),
        _b_side(args[unsafe_offset=1]),
        _none_side(),
        _loss_reduction(args[unsafe_offset=2]),
        _p(),
        args[unsafe_offset=3].copy(),
        rets,
    )


def _mse_backward(args: Values, rets: Values, out_v: Optional[Value]) raises:
    var input = _b_side(args[unsafe_offset=1])
    var reduction = _loss_reduction(args[unsafe_offset=3])
    _loss_backward(
        "mse_backward",
        _b_side(args[unsafe_offset=0]),
        input,
        _b_side(args[unsafe_offset=2]),
        _loss_norm(input, reduction, 2.0),
        0.0,
        out_v,
        rets,
    )


# aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
def op_mse_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _mse_backward(args, rets, None)


# aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_mse_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _mse_backward(args, rets, args[unsafe_offset=4].copy())


def _smooth_l1(args: Values, rets: Values, out_v: Optional[Value]) raises:
    var beta = v_f64(args[unsafe_offset=3])
    if beta < 0:
        raise Error("smooth_l1_loss does not support negative values for beta.")
    var a = _b_side(args[unsafe_offset=0])
    var st = a.t.value().stype if a.is_t else ST_FLOAT32
    _loss_forward(
        "smooth_l1",
        2,
        a,
        _b_side(args[unsafe_offset=1]),
        _none_side(),
        _loss_reduction(args[unsafe_offset=2]),
        _p(_round_to(beta, st)),
        out_v,
        rets,
    )


# aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor
def op_smooth_l1_loss(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _smooth_l1(args, rets, None)


# aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)
def op_smooth_l1_loss_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _smooth_l1(args, rets, args[unsafe_offset=4].copy())


def _smooth_l1_backward(
    args: Values, rets: Values, out_v: Optional[Value]
) raises:
    var input = _b_side(args[unsafe_offset=1])
    var reduction = _loss_reduction(args[unsafe_offset=3])
    _loss_backward(
        "smooth_l1_backward",
        _b_side(args[unsafe_offset=0]),
        input,
        _b_side(args[unsafe_offset=2]),
        _loss_norm(input, reduction, 1.0),
        v_f64(args[unsafe_offset=4]),
        out_v,
        rets,
    )


# aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
def op_smooth_l1_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _smooth_l1_backward(args, rets, None)


# aten::smooth_l1_loss_backward.grad_input(..., float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_smooth_l1_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _smooth_l1_backward(args, rets, args[unsafe_offset=5].copy())


def _huber(args: Values, rets: Values, out_v: Optional[Value]) raises:
    var delta = v_f64(args[unsafe_offset=3])
    if not (delta > 0):
        raise Error(
            "huber_loss does not support non-positive values for delta."
        )
    var a = _b_side(args[unsafe_offset=0])
    var st = a.t.value().stype if a.is_t else ST_FLOAT32
    _loss_forward(
        "huber",
        2,
        a,
        _b_side(args[unsafe_offset=1]),
        _none_side(),
        _loss_reduction(args[unsafe_offset=2]),
        _p(_round_to(delta, st)),
        out_v,
        rets,
    )


# aten::huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
def op_huber_loss(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _huber(args, rets, None)


# aten::huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)
def op_huber_loss_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _huber(args, rets, args[unsafe_offset=4].copy())


def _huber_backward(args: Values, rets: Values, out_v: Optional[Value]) raises:
    var input = _b_side(args[unsafe_offset=1])
    var reduction = _loss_reduction(args[unsafe_offset=3])
    _loss_backward(
        "huber_backward",
        _b_side(args[unsafe_offset=0]),
        input,
        _b_side(args[unsafe_offset=2]),
        _loss_norm(input, reduction, 1.0),
        v_f64(args[unsafe_offset=4]),
        out_v,
        rets,
    )


# aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor
def op_huber_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _huber_backward(args, rets, None)


# aten::huber_loss_backward.out(..., float delta, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_huber_loss_backward_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _huber_backward(args, rets, args[unsafe_offset=5].copy())


def _opt_weight(v: Value) raises -> Optional[T]:
    if v_is_none(v):
        return None
    var w = v_tensor(v)
    if not w.on_mojo():
        unsupported("a weight outside the mojo device")
    return w^


def _bce(args: Values, rets: Values, out_v: Optional[Value]) raises:
    var weight = _opt_weight(args[unsafe_offset=2])
    # The weight rides as the third operand (the product is the kernel's
    # last rounding, like `loss.mul_(weight)`); 1 without one.
    var c = _b_tside(weight.value()) if weight else _b_sside(
        Scal(1.0, 1, True, False)
    )
    _loss_forward(
        "bce",
        3,
        _b_side(args[unsafe_offset=0]),
        _b_side(args[unsafe_offset=1]),
        c,
        _loss_reduction(args[unsafe_offset=3]),
        _p(),
        out_v,
        rets,
    )


# aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
def op_binary_cross_entropy(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce(args, rets, None)


# aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
def op_binary_cross_entropy_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce(args, rets, args[unsafe_offset=4].copy())


def _bce_backward(args: Values, rets: Values, out_v: Optional[Value]) raises:
    """Loss.cu binary_cross_entropy_backward_out_cuda: the gradient kernel,
    then `mul_(weight)` and the mean's `div_(numel)` -- folded into the
    gradient kernel when there is no weight, a second kind when there is."""
    var grad = _b_side(args[unsafe_offset=0])
    var input = _b_side(args[unsafe_offset=1])
    var target = _b_side(args[unsafe_offset=2])
    var weight = _opt_weight(args[unsafe_offset=3])
    var reduction = _loss_reduction(args[unsafe_offset=4])
    if not grad.is_t or not input.is_t or not target.is_t:
        unsupported("binary_cross_entropy_backward with a scalar operand")
    var common = _pw_result_type(grad, input, target, 3)
    var compute = _pw_compute_dtype("bce_backward", common, P_FLOAT_ONLY, True)
    var inv = 1.0
    if reduction == REDUCTION_MEAN:
        # div_true's CPU-scalar route multiplies by the opmath reciprocal.
        var n = input.t.value().numel
        if compute == ST_FLOAT64:
            inv = 1.0 / Float64(n)
        else:
            inv = (Float32(1.0) / Float32(n)).cast[DType.float64]()
    var dest = Optional[T]()
    if out_v:
        dest = _pw_out_of(out_v.value(), grad, input, target)
    var out_st = input.t.value().stype
    if not weight:
        _pw_finish(
            rets,
            dest,
            _pw_run(
                "bce_backward",
                3,
                grad,
                input,
                target,
                compute,
                out_st,
                _p(inv),
                dest,
            ),
            _pw_out_exact("bce_backward"),
        )
        return
    var gi = _pw_run(
        "bce_backward", 3, grad, input, target, compute, compute, _p(1.0), None
    )
    var res = _pw_run(
        "mul_scale",
        2,
        _b_tside(gi.t),
        _b_tside(weight.value()),
        _none_side(),
        compute,
        out_st,
        _p(inv),
        dest,
    )
    release(gi.t.h)
    _pw_finish(rets, dest, res^, _pw_out_exact("bce_backward"))


# aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
def op_binary_cross_entropy_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce_backward(args, rets, None)


# aten::binary_cross_entropy_backward.grad_input(..., int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_binary_cross_entropy_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce_backward(args, rets, args[unsafe_offset=5].copy())


def _bce_logits(args: Values, rets: Values, out_v: Optional[Value]) raises:
    """Loss.cpp binary_cross_entropy_with_logits, whose CompositeExplicit
    body (log_sigmoid, sub, mul, rsub, ...) is one kind here; autograd keeps
    its own derivatives.yaml formula."""
    var weight = _opt_weight(args[unsafe_offset=2])
    var pos_weight = _opt_weight(args[unsafe_offset=3])
    var c = _b_tside(pos_weight.value()) if pos_weight else _b_sside(
        Scal(1.0, 1, True, False)
    )
    _loss_forward(
        "bce_logits",
        3,
        _b_side(args[unsafe_offset=0]),
        _b_side(args[unsafe_offset=1]),
        c,
        _loss_reduction(args[unsafe_offset=4]),
        _p(),
        out_v,
        rets,
        weight,
    )


# aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
def op_binary_cross_entropy_with_logits(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce_logits(args, rets, None)


# aten::binary_cross_entropy_with_logits.out(..., int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
def op_binary_cross_entropy_with_logits_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bce_logits(args, rets, args[unsafe_offset=5].copy())


def register_pointwise(site: Site) raises:
    impl[op_ilshift, "__ilshift__.Scalar"](site)
    impl[op_ilshift, "__ilshift__.Tensor"](site)
    impl[op_irshift, "__irshift__.Scalar"](site)
    impl[op_irshift, "__irshift__.Tensor"](site)
    impl[op_lshift, "__lshift__.Scalar"](site)
    impl[op_lshift, "__lshift__.Tensor"](site)
    impl[op_rshift, "__rshift__.Scalar"](site)
    impl[op_rshift, "__rshift__.Tensor"](site)
    impl[op_atan2, "atan2"](site)
    impl[op_atan2_out, "atan2.out"](site)
    impl[op_lshift, "bitwise_left_shift.Tensor"](site)
    impl[op_lshift_out, "bitwise_left_shift.Tensor_out"](site)
    impl[op_rshift, "bitwise_right_shift.Tensor"](site)
    impl[op_rshift_out, "bitwise_right_shift.Tensor_out"](site)
    impl[op_clamp_tensor, "clamp.Tensor"](site)
    impl[op_clamp_tensor_out, "clamp.Tensor_out"](site)
    impl[op_copysign, "copysign.Scalar"](site)
    impl[op_copysign, "copysign.Tensor"](site)
    impl[op_copysign_out, "copysign.Scalar_out"](site)
    impl[op_copysign_out, "copysign.out"](site)
    impl[op_fmax, "fmax"](site)
    impl[op_fmax_out, "fmax.out"](site)
    impl[op_fmin, "fmin"](site)
    impl[op_fmin_out, "fmin.out"](site)
    impl[op_fmod, "fmod.Scalar"](site)
    impl[op_fmod, "fmod.Tensor"](site)
    impl[op_fmod_out, "fmod.Scalar_out"](site)
    impl[op_fmod_out, "fmod.Tensor_out"](site)
    impl[op_gcd, "gcd"](site)
    impl[op_gcd_out, "gcd.out"](site)
    impl[op_gelu_backward_any, "gelu_backward"](site)
    impl[op_heaviside, "heaviside"](site)
    impl[op_heaviside_out, "heaviside.out"](site)
    impl[op_hypot, "hypot"](site)
    impl[op_hypot_out, "hypot.out"](site)
    impl[op_lcm, "lcm"](site)
    impl[op_lcm_out, "lcm.out"](site)
    impl[op_lerp_scalar_any, "lerp.Scalar"](site)
    impl[op_lerp_scalar_out_any, "lerp.Scalar_out"](site)
    impl[op_lerp_scalar__any, "lerp_.Scalar"](site)
    impl[op_lerp_tensor, "lerp.Tensor"](site)
    impl[op_lerp_tensor_out, "lerp.Tensor_out"](site)
    impl[op_logaddexp, "logaddexp"](site)
    impl[op_logaddexp_out, "logaddexp.out"](site)
    impl[op_logaddexp2, "logaddexp2"](site)
    impl[op_logaddexp2_out, "logaddexp2.out"](site)
    impl[op_nextafter, "nextafter"](site)
    impl[op_nextafter_out, "nextafter.out"](site)
    impl[op_rsub_any, "rsub.Tensor"](site)
    impl[op_rsub_any, "rsub.Scalar"](site)
    impl[op_rsub_out_any, "rsub.Tensor_out"](site)
    impl[op_rsub_out_any, "rsub.Scalar_out"](site)
    impl[op_pow_scalar_base, "pow.Scalar"](site)
    impl[op_pow_scalar_base_out, "pow.Scalar_out"](site)
    impl[op_pow_scalar_any, "pow.Tensor_Scalar"](site)
    impl[op_pow_scalar_out_any, "pow.Tensor_Scalar_out"](site)
    impl[op_pow_tensor_any, "pow.Tensor_Tensor"](site)
    impl[op_pow_tensor_out_any, "pow.Tensor_Tensor_out"](site)
    _register_special(site)
    impl[op_xlogy, "xlogy.Tensor"](site)
    impl[op_xlogy_out, "xlogy.OutTensor"](site)


comptime op_special_chebyshev_polynomial_t = op_poly[
    "chebyshev_polynomial_t", -1
]
comptime op_special_chebyshev_polynomial_t_out = op_poly[
    "chebyshev_polynomial_t", 2
]
comptime op_special_chebyshev_polynomial_u = op_poly[
    "chebyshev_polynomial_u", -1
]
comptime op_special_chebyshev_polynomial_u_out = op_poly[
    "chebyshev_polynomial_u", 2
]
comptime op_special_chebyshev_polynomial_v = op_poly[
    "chebyshev_polynomial_v", -1
]
comptime op_special_chebyshev_polynomial_v_out = op_poly[
    "chebyshev_polynomial_v", 2
]
comptime op_special_chebyshev_polynomial_w = op_poly[
    "chebyshev_polynomial_w", -1
]
comptime op_special_chebyshev_polynomial_w_out = op_poly[
    "chebyshev_polynomial_w", 2
]
comptime op_special_hermite_polynomial_h = op_poly["hermite_polynomial_h", -1]
comptime op_special_hermite_polynomial_h_out = op_poly[
    "hermite_polynomial_h", 2
]
comptime op_special_hermite_polynomial_he = op_poly["hermite_polynomial_he", -1]
comptime op_special_hermite_polynomial_he_out = op_poly[
    "hermite_polynomial_he", 2
]
comptime op_special_laguerre_polynomial_l = op_poly["laguerre_polynomial_l", -1]
comptime op_special_laguerre_polynomial_l_out = op_poly[
    "laguerre_polynomial_l", 2
]
comptime op_special_legendre_polynomial_p = op_poly["legendre_polynomial_p", -1]
comptime op_special_legendre_polynomial_p_out = op_poly[
    "legendre_polynomial_p", 2
]
comptime op_special_shifted_chebyshev_polynomial_t = op_poly[
    "shifted_chebyshev_polynomial_t", -1
]
comptime op_special_shifted_chebyshev_polynomial_t_out = op_poly[
    "shifted_chebyshev_polynomial_t", 2
]
comptime op_special_shifted_chebyshev_polynomial_u = op_poly[
    "shifted_chebyshev_polynomial_u", -1
]
comptime op_special_shifted_chebyshev_polynomial_u_out = op_poly[
    "shifted_chebyshev_polynomial_u", 2
]
comptime op_special_shifted_chebyshev_polynomial_v = op_poly[
    "shifted_chebyshev_polynomial_v", -1
]
comptime op_special_shifted_chebyshev_polynomial_v_out = op_poly[
    "shifted_chebyshev_polynomial_v", 2
]
comptime op_special_shifted_chebyshev_polynomial_w = op_poly[
    "shifted_chebyshev_polynomial_w", -1
]
comptime op_special_shifted_chebyshev_polynomial_w_out = op_poly[
    "shifted_chebyshev_polynomial_w", 2
]


def _register_special(site: Site) raises:
    impl[op_special_xlog1py, "special_xlog1py"](site)
    impl[op_special_xlog1py_out, "special_xlog1py.out"](site)
