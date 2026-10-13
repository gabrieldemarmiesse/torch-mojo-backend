"""The complex dtypes (complex64, complex128) on the mojo device.

Mojo has no complex DType, so a complex tensor is, to the kernels, its
interleaved float32 / float64 storage (torch's layout): `T.dtype` is the
component dtype, `T.itemsize` the pair's, `T.pairs()` / `T.component(i)`
the `view_as_real` geometry (tmb/backend/abi.mojo). Every op that is not
written for complex declines a complex tensor (`v_tensor`); the ones here
read theirs with `v_tensor_c`.

Three ways a complex op is computed:

* elementwise kernels on the pairs: the complex kernel family
  (tmb/kernels/complex), through `cx_launch`;
* an existing real op on `view_as_real` (cat, indexing, the random
  fills): a complex element is two adjacent reals, so the real op over the
  trailing pair dimension is the complex op;
* the byte movers (copy, clone, views) need nothing: they move `itemsize`
  bytes per element (tmb/ops/core.mojo, tmb/ops/data_movement.mojo).

The ops that already exist for the real dtypes keep their registration; it
is wrapped in `routed[real_op, complex_op, n]`, which sends the call to the
complex implementation when one of the first `n` arguments is complex.

Type promotion is torch's (`c10::promoteTypes` with the complex category
on top, ResultTypeState's dim / zero-dim / wrapped-number ranks).
complex32 (ComplexHalf) is not held, so a promotion that lands on it is
declined.
"""
from std.ffi import external_call
from std.utils import IndexList

from tmb.backend.abi import (
    ST_BFLOAT16,
    ST_BOOL,
    ST_COMPLEX128,
    ST_COMPLEX64,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT16,
    ST_INT32,
    ST_INT64,
    ST_INT8,
    ST_UINT8,
    TAG_BOOL,
    TAG_COMPLEX,
    TAG_DOUBLE,
    TAG_DTYPE,
    TAG_INT,
    TAG_INT_LIST,
    TAG_MEMORY_FORMAT,
    TAG_NONE,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    TAG_TENSOR_LIST,
    TAG_OPT_TENSOR_LIST,
    TAG_TENSOR_REF,
    OpFn,
    Owned,
    Results,
    T,
    Value,
    Values,
    bits_f64,
    call_op,
    complex_of,
    contiguous_strides,
    default_dtype,
    dtype_name,
    f64_bits,
    is_complex_stype,
    new_tensor,
    own,
    own_if_new,
    real_of,
    release,
    retain,
    ret_owned,
    ret_ref,
    unsupported,
    index_error,
    v_bool_or,
    v_complex,
    v_dtype_or,
    v_f64,
    v_is_complex,
    v_is_none,
    v_tensor_c,
    v_tensor_list_c,
    v_opt_tensor_list_present,
    view_strided,
    IntList,
)
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    CX_DEVICE,
    CX_FULL,
    CX_HOST,
    ST_UNDEFINED,
    cast_complex,
    contiguous,
    copy_strided_into,
    cx_launch,
    conj_copy,
    device_str,
    fill_complex,
    promote_types,
    resize_out,
    resized_geometry,
    assert_no_overlap,
    shape_str,
    assert_no_internal_overlap,
    assert_no_partial_overlap,
    store_out,
)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def routed[
    real_op: OpFn, complex_op: OpFn, n_check: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`real_op`, unless one of the first `n_check` arguments is a complex
    tensor or a complex Scalar: then `complex_op`. What the op groups
    register in place of an op that has a complex implementation here."""
    comptime for i in range(n_check):
        if i < n_args and v_is_complex(args[unsafe_offset=i]):
            complex_op(args, n_args, rets, n_rets)
            return
    real_op(args, n_args, rets, n_rets)


# ---------------------------------------------------------------------------
# calling other aten ops
# ---------------------------------------------------------------------------


struct _Call(Movable):
    """Arguments of one `call_op`, built in schema order; tensors are held
    by a reference of our own until the call is made."""

    var op: String
    var overload: String
    var vals: List[Value]
    var lists: List[List[Int64]]
    var held: List[Int]

    def __init__(out self, op: String, overload: String = ""):
        self.op = op
        self.overload = overload
        self.vals = List[Value]()
        self.lists = List[List[Int64]]()
        self.held = List[Int]()

    def __deinit__(deinit self):
        for h in self.held:
            release(h)

    def t(mut self, x: T):
        var h = retain(x)
        self.held.append(h)
        self.vals.append(Value(TAG_TENSOR, 0, Int64(h), 0))

    def v(mut self, x: Value):
        self.vals.append(x.copy())

    def i(mut self, x: Int):
        self.vals.append(Value(TAG_INT, 0, Int64(x), 0))

    def b(mut self, x: Bool):
        self.vals.append(Value(TAG_BOOL, 0, Int64(1) if x else Int64(0), 0))

    def s(mut self, x: Float64):
        self.vals.append(Value(TAG_SCALAR_DOUBLE, 0, f64_bits(x), 0))

    def f(mut self, x: Float64):
        self.vals.append(Value(TAG_DOUBLE, 0, f64_bits(x), 0))

    def dtype(mut self, st: Int32):
        if st < 0:
            self.none()
        else:
            self.vals.append(Value(TAG_DTYPE, 0, Int64(st), 0))

    def none(mut self):
        self.vals.append(Value(TAG_NONE, 0, 0, 0))

    def ints(mut self, xs: List[Int]):
        var l = List[Int64](capacity=max(len(xs), 1))
        for x in xs:
            l.append(Int64(x))
        var addr = Int(l.unsafe_ptr())
        self.lists.append(l^)
        self.vals.append(Value(TAG_INT_LIST, Int32(len(xs)), Int64(addr), 0))

    def run(mut self, n_rets: Int) raises -> Results:
        return call_op(self.op, self.overload, self.vals.copy(), n_rets)

    def one(mut self) raises -> Owned:
        var r = self.run(1)
        return own(r.take_tensor(0))


def as_real(t: T) raises -> Owned:
    """`torch.view_as_real(t)` (a real view sharing t's storage)."""
    var c = _Call("aten::view_as_real")
    c.t(t)
    return c.one()


def as_complex(t: T) raises -> Owned:
    """`torch.view_as_complex(t)` of a real [..., 2] tensor, made a dense
    copy first when its layout cannot be viewed (a trailing stride other
    than 1, an odd stride or offset -- an empty result of a real op can
    carry any strides)."""
    var viewable = t.rank >= 1 and t.stride(-1) == 1 and t.offset % 2 == 0
    for i in range(t.rank - 1):
        if t.stride(i) % 2 != 0:
            viewable = False
    if not viewable:
        var cl = _Call("aten::clone")
        cl.t(t)
        cl.v(Value(TAG_MEMORY_FORMAT, 0, 0, 0))  # torch.contiguous_format
        var dense = cl.one()
        var v = _Call("aten::view_as_complex")
        v.t(dense.t)
        var r = v.one()
        _ = dense^
        return r^
    var c = _Call("aten::view_as_complex")
    c.t(t)
    return c.one()


def _st_name(st: Int32) -> String:
    """`c10::toString(ScalarType)`, as TensorIterator's cast errors spell it."""
    if st == ST_FLOAT32:
        return "Float"
    if st == ST_FLOAT64:
        return "Double"
    if st == ST_FLOAT16:
        return "Half"
    if st == ST_BFLOAT16:
        return "BFloat16"
    if st == ST_COMPLEX64:
        return "ComplexFloat"
    if st == ST_COMPLEX128:
        return "ComplexDouble"
    if st == ST_INT64:
        return "Long"
    if st == ST_INT32:
        return "Int"
    if st == ST_INT16:
        return "Short"
    if st == ST_INT8:
        return "Char"
    if st == ST_UINT8:
        return "Byte"
    if st == ST_BOOL:
        return "Bool"
    return String("ScalarType ") + String(st)


def _category(st: Int32) -> Int:
    """bool < integral < floating < complex (c10's canCast order)."""
    if st == ST_BOOL:
        return 0
    if is_complex_stype(st):
        return 3
    if (
        st == ST_FLOAT32
        or st == ST_FLOAT64
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
    ):
        return 2
    return 1


def _check_cast(result: Int32, dest: Int32) raises:
    """TensorIterator's `can't be cast to the desired output type`."""
    if _category(result) > _category(dest):
        raise Error(
            "result type ",
            _st_name(result),
            " can't be cast to the desired output type ",
            _st_name(dest),
        )


def _store(mut dest: T, var res: Owned, rets: Values) raises:
    """A fresh result into a caller's `out=` (or an in-place `self`):
    `resize_output` first, then the internal-overlap check on the resized
    out (an expanded out of the wrong shape is resized, not refused), then
    the copy, converting when the dtypes differ (after TensorIterator's
    cast check)."""
    _same_device(dest, res.t)
    resize_out(dest, res.t.shape, res.t.rank)
    assert_no_internal_overlap(dest)
    if dest.stype == res.t.stype:
        copy_strided_into(dest, res.t)
        _ = res^
    else:
        _check_cast(res.t.stype, dest.stype)
        var c = _Call("aten::copy_")
        c.t(dest)
        c.t(res.t)
        c.b(False)
        _ = c.run(1)
        _ = res^
    ret_ref(rets, 0, dest)


def _exact_dtype(dest: T, st: Int32) raises:
    """An `out=` that must already hold the result's dtype (structured
    kernels' resize_out: no cast)."""
    if dest.stype != st:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(st),
            ", but got ",
            dtype_name(dest.stype),
            " instead",
        )


def _require_same_device(t: T, device: Int, what: StaticString) raises:
    """ATen's device check of an input against the op's device."""
    if not t.on_mojo() or t.device != device:
        raise Error(
            "Expected all tensors to be on the same device, but got ",
            what,
            " is on ",
            device_str(t),
            ", different from other tensors on mojo:",
            device,
        )


def _real_out(
    op: String,
    overload: String,
    args: Values,
    n_in: Int,
    out_idx: Int,
    rets: Values,
) raises -> Bool:
    """An elementwise `out=` op reaching the complex route only for its
    complex `out` (no complex input): ATen computes in the inputs' common
    dtype and casts into the out, so run the real functional op and store
    its result (real -> complex is a valid cast). False when an input is
    complex (the caller's complex route applies)."""
    for i in range(n_in):
        if v_is_complex(args[unsafe_offset=i]):
            return False
    var c = _Call(op, overload)
    for i in range(n_in):
        c.v(args[unsafe_offset=i])
    var r = c.one()
    var dest = v_tensor_c(args[unsafe_offset=out_idx])
    _store_ew_n(args, n_in, dest, r^, rets)
    return True


def _tensor_or_scalar(v: Value, tensor: String, scalar: String) -> String:
    return tensor if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF else scalar


def _same_device(dest: T, res: T) raises:
    """TensorIterator's device check on an `out=` / in-place tensor."""
    if dest.device_type != res.device_type or dest.device != res.device:
        raise Error(
            (
                "Expected all tensors to be on the same device, but found at"
                " least two devices, "
            ),
            device_str(res),
            " and ",
            device_str(dest),
            "!",
        )


def _store_ew[
    n_in: Int
](args: Values, mut dest: T, var res: Owned, rets: Values) raises:
    _store_ew_n(args, n_in, dest, res^, rets)


def _store_ew_n(
    args: Values, n_in: Int, mut dest: T, var res: Owned, rets: Values
) raises:
    """`_store` for an elementwise op (a TensorIterator in ATen): the out's
    device, then its memory-overlap checks against itself and against each
    of the first `n_in` arguments that is a tensor -- made on the out as
    given, before it is resized, as TensorIterator's compute_mem_overlaps
    is -- then the cast check, resize and copy."""
    _same_device(dest, res.t)
    assert_no_internal_overlap(dest)
    for i in range(n_in):
        var v = args[unsafe_offset=i].copy()
        if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
            var t = v_tensor_c(v)
            if t.on_mojo() and t.device == dest.device:
                assert_no_partial_overlap(dest, t)
    _store(dest, res^, rets)


# ---------------------------------------------------------------------------
# type promotion with the complex category
# ---------------------------------------------------------------------------


def _default_complex() -> Int32:
    """`get_default_complex_dtype()`: complex128 under a float64 default."""
    return ST_COMPLEX128 if default_dtype() == ST_FLOAT64 else ST_COMPLEX64


def cx_promote(a: Int32, b: Int32) raises -> Int32:
    """`c10::promoteTypes` where either may be complex: the complex type of
    the promoted value types (complex64 + float64 is complex128)."""
    if a == ST_UNDEFINED:
        return b
    if b == ST_UNDEFINED or a == b:
        return a
    if not is_complex_stype(a) and not is_complex_stype(b):
        return promote_types(a, b)
    return complex_of(promote_types(real_of(a), real_of(b)))


def _is_float(st: Int32) -> Bool:
    return (
        st == ST_FLOAT32
        or st == ST_FLOAT64
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
    )


def cx_combine(higher: Int32, lower: Int32) raises -> Int32:
    """ATen's `combine_categories` (TypeProperties.cpp), complex included."""
    if is_complex_stype(higher):
        return higher
    if is_complex_stype(lower):
        if _is_float(higher):
            return complex_of(higher)
        return lower
    if _is_float(higher):
        return higher
    if higher == ST_BOOL or _is_float(lower):
        if higher == ST_UNDEFINED:
            return lower
        if lower == ST_UNDEFINED:
            return higher
        return promote_types(higher, lower)
    if higher != ST_UNDEFINED:
        return higher
    return lower


# ---------------------------------------------------------------------------
# operands of the elementwise ops
# ---------------------------------------------------------------------------

comptime _RANK_DIM = 0
comptime _RANK_ZERO = 1
comptime _RANK_WRAPPED = 2


struct _Operand(Movable):
    """One argument of a binary op: a mojo tensor, or a host value (a Scalar,
    a wrapped number, a 0-d CPU tensor) as (re, im)."""

    var is_t: Bool
    var t: T
    var held: Owned  # a materialized operand, released with this
    var re: Float64
    var im: Float64
    var stype: Int32  # the dtype it promotes as
    var rank_kind: Int

    def __init__(out self, t: T) raises:
        self.is_t = True
        self.t = t.copy()
        self.held = own(t.copy())
        self.held.live = False
        self.re = 0.0
        self.im = 0.0
        self.stype = t.stype
        self.rank_kind = _RANK_DIM if t.rank > 0 else _RANK_ZERO

    def __init__(
        out self,
        like: T,
        re: Float64,
        im: Float64,
        stype: Int32,
        rank_kind: Int,
    ) raises:
        self.is_t = False
        self.t = like.copy()
        self.held = own(like.copy())
        self.held.live = False
        self.re = re
        self.im = im
        self.stype = stype
        self.rank_kind = rank_kind


def _host_value(t: T) raises -> Tuple[Float64, Float64]:
    """The one element of a CPU tensor, as (re, im)."""
    var p = t.ptr
    var dt = t.dtype
    if t.is_complex():
        if dt == DType.float32:
            var q = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=p)
            return (Float64(q[]), Float64(q[unsafe_offset=1]))
        var q = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=p)
        return (q[], q[unsafe_offset=1])
    if dt == DType.float32:
        return (
            Float64(
                Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.float64:
        return (
            Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=p)[],
            0.0,
        )
    if dt == DType.float16:
        return (
            Float64(
                Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.bfloat16:
        return (
            Float64(
                Pointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.int64:
        return (
            Float64(
                Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.int32:
        return (
            Float64(
                Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.int16:
        return (
            Float64(
                Pointer[Int16, MutUntrackedOrigin](unsafe_from_address=p)[]
            ),
            0.0,
        )
    if dt == DType.int8:
        return (
            Float64(Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=p)[]),
            0.0,
        )
    var b = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=p)[]
    return (Float64(b), 0.0)


def _operand(v: Value, like: T) raises -> _Operand:
    """Read one operand record (`like` is any mojo tensor of the call, a
    placeholder for the host kinds)."""
    if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
        var t = v_tensor_c(v)
        if t.on_mojo():
            return _Operand(t)
        if t.on_cpu() and t.rank == 0:
            var hv = _host_value(t)
            if t.is_wrapped_number():
                var st = t.stype
                if is_complex_stype(st):
                    st = _default_complex()
                elif _is_float(st):
                    st = default_dtype()
                return _Operand(like, hv[0], hv[1], st, _RANK_WRAPPED)
            return _Operand(like, hv[0], hv[1], t.stype, _RANK_ZERO)
        raise Error(
            (
                "Expected all tensors to be on the same device, but found at"
                " least two devices, "
            ),
            device_str(like),
            " and ",
            device_str(t),
            "!",
        )
    if v.tag == TAG_COMPLEX:
        return _Operand(
            like,
            bits_f64(v.a),
            bits_f64(v.b),
            _default_complex(),
            _RANK_WRAPPED,
        )
    if v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL:
        return _Operand(
            like, 1.0 if v.a != 0 else 0.0, 0.0, ST_BOOL, _RANK_WRAPPED
        )
    if v.tag == TAG_SCALAR_INT or v.tag == TAG_INT:
        return _Operand(like, Float64(v.a), 0.0, ST_INT64, _RANK_WRAPPED)
    if v.tag == TAG_SCALAR_DOUBLE or v.tag == TAG_DOUBLE:
        return _Operand(like, v_f64(v), 0.0, default_dtype(), _RANK_WRAPPED)
    raise Error("unexpected operand record tag ", v.tag)


def _first_tensor(a: Value, b: Value) raises -> T:
    """The mojo tensor of a binary call (its device, a placeholder)."""
    if a.tag == TAG_TENSOR or a.tag == TAG_TENSOR_REF:
        var t = v_tensor_c(a)
        if t.on_mojo():
            return t^
    if b.tag == TAG_TENSOR or b.tag == TAG_TENSOR_REF:
        var t = v_tensor_c(b)
        if t.on_mojo():
            return t^
    unsupported("a complex op with no mojo tensor operand")
    return v_tensor_c(a)


def _result_type(a: _Operand, b: _Operand) raises -> Int32:
    var dim = ST_UNDEFINED
    var zero = ST_UNDEFINED
    var wrapped = ST_UNDEFINED
    for k in range(2):
        var st = a.stype if k == 0 else b.stype
        var rk = a.rank_kind if k == 0 else b.rank_kind
        if rk == _RANK_DIM:
            dim = cx_promote(dim, st)
        elif rk == _RANK_ZERO:
            zero = cx_promote(zero, st)
        else:
            wrapped = cx_promote(wrapped, st)
    return cx_combine(dim, cx_combine(zero, wrapped))


def _bshape(a: _Operand, b: _Operand) raises -> Tuple[IndexList[MAX_RANK], Int]:
    """Broadcast shape and rank of the tensor operands."""
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    for k in range(2):
        var is_t = a.is_t if k == 0 else b.is_t
        if not is_t:
            continue
        var t = a.t.copy() if k == 0 else b.t.copy()
        rank = max(rank, t.rank)
        for i in range(MAX_RANK):
            var x = shape[i]
            var y = t.shape[i]
            if x == y or y == 1:
                continue
            if x == 1:
                shape[i] = y
            else:
                raise Error(
                    "The size of tensor a must match the size of tensor b at"
                    " non-singleton dimension"
                )
    return (shape, rank)


@fieldwise_init
struct _Flat(Copyable, Movable):
    var mode: Int
    var addr: Int
    var re: Float64
    var im: Float64


def _flatten(
    mut x: _Operand,
    st: Int32,
    shape: IndexList[MAX_RANK],
    rank: Int,
) raises -> _Flat:
    """The operand as the complex family takes it: cast to `st`, then a dense
    buffer of `shape`, one device element (a 0-d tensor) or a host value."""
    if not x.is_t:
        return _Flat(CX_HOST, 0, x.re, x.im)
    var cast = own_if_new(cast_complex(x.t, st), x.t)
    if cast.t.rank == 0:
        var flat = _Flat(CX_DEVICE, cast.t.ptr, 0.0, 0.0)
        x.held = cast^
        return flat^
    var same = cast.t.rank == rank and cast.t.contig
    if same:
        for i in range(MAX_RANK):
            if cast.t.shape[i] != shape[i]:
                same = False
    if same:
        var flat = _Flat(CX_FULL, cast.t.ptr, 0.0, 0.0)
        x.held = cast^
        return flat^
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        strides[i] = cast.t.strides[i] if cast.t.shape[i] == shape[i] else 0
    var expanded = own(
        view_strided(cast.t, shape, strides, rank, cast.t.offset)
    )
    _ = cast^  # the view holds the storage
    var dense = own_if_new(contiguous(expanded.t), expanded.t)
    var flat = _Flat(CX_FULL, dense.t.ptr, 0.0, 0.0)
    if dense.live:
        x.held = dense^
        _ = expanded^
    else:
        # an expansion torch calls contiguous (an empty one): it is the data
        x.held = expanded^
    return flat^


def _one_device(a: _Operand, b: _Operand) raises -> Int:
    if a.is_t and b.is_t and a.t.device != b.t.device:
        raise Error(
            (
                "Expected all tensors to be on the same device, but found at"
                " least two devices, "
            ),
            device_str(a.t),
            " and ",
            device_str(b.t),
            "!",
        )
    return a.t.device if a.is_t else b.t.device


def _binary(kind: StaticString, av: Value, bv: Value) raises -> Owned:
    """`kind(a, b)` (eq / ne) where one side is complex: promoted to the
    common complex dtype, broadcast, one launch of the complex family."""
    var like = _first_tensor(av, bv)
    var a = _operand(av, like)
    var b = _operand(bv, like)
    var st = _result_type(a, b)
    if not is_complex_stype(st):
        unsupported("a complex op whose result type is not complex")
    var device = _one_device(a, b)
    var geo = _bshape(a, b)
    var fa = _flatten(a, st, geo[0], geo[1])
    var fb = _flatten(b, st, geo[0], geo[1])
    var out = own(new_tensor(geo[0], geo[1], ST_BOOL, device))
    cx_launch(
        kind,
        out.t.ptr,
        fa.addr,
        fb.addr,
        fa.mode,
        fb.mode,
        fa.re,
        fa.im,
        fb.re,
        fb.im,
        out.t.numel,
        DType.float32 if st == ST_COMPLEX64 else DType.float64,
        device,
    )
    _ = a^
    _ = b^
    return out^


def _ret(rets: Values, var o: Owned):
    ret_owned(rets, 0, o)


# eq / ne --------------------------------------------------------------------


def cx_eq(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """eq.Tensor / eq.Scalar(self, other)"""
    _ret(
        rets,
        _binary("eq", args[unsafe_offset=0], args[unsafe_offset=1]),
    )


def cx_eq_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if _real_out(
        "aten::eq",
        _tensor_or_scalar(args[unsafe_offset=1], "Tensor", "Scalar"),
        args,
        2,
        2,
        rets,
    ):
        return
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_ew[2](
        args,
        dest,
        _binary("eq", args[unsafe_offset=0], args[unsafe_offset=1]),
        rets,
    )


def cx_ne(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(
        rets,
        _binary("ne", args[unsafe_offset=0], args[unsafe_offset=1]),
    )


def cx_ne_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if _real_out(
        "aten::ne",
        _tensor_or_scalar(args[unsafe_offset=1], "Tensor", "Scalar"),
        args,
        2,
        2,
        rets,
    ):
        return
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_ew[2](
        args,
        dest,
        _binary("ne", args[unsafe_offset=0], args[unsafe_offset=1]),
        rets,
    )


# ---------------------------------------------------------------------------
# unary
# ---------------------------------------------------------------------------

comptime _TO_CX = 0
comptime _TO_RE = 1
comptime _TO_BOOL = 2


def _unary(kind: StaticString, to: Int, x: T) raises -> Owned:
    if not x.is_complex():
        raise Error("complex ", kind, ": not a complex tensor")
    var st = x.stype
    if to == _TO_RE:
        st = real_of(x.stype)
    elif to == _TO_BOOL:
        st = ST_BOOL
    var dense = own_if_new(contiguous(x), x)
    var out = own(new_tensor(x.shape, x.rank, st, x.device))
    cx_launch(
        kind,
        out.t.ptr,
        dense.t.ptr,
        0,
        CX_FULL,
        CX_FULL,
        0.0,
        0.0,
        0.0,
        0.0,
        x.numel,
        x.dtype,
        x.device,
    )
    _ = dense^
    return out^


def cx_unary_out[
    kind: StaticString, to: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`op.out(self, *, out)` of a complex self (also the in-place form,
    which torch routes here with out = self)."""
    if _real_out(String("aten::") + kind, "", args, 1, 1, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=1])
    _store_ew[1](
        args, dest, _unary(kind, to, v_tensor_c(args[unsafe_offset=0])), rets
    )


comptime cx_conj_physical_out = cx_unary_out["conj", _TO_CX]


# ---------------------------------------------------------------------------
# reductions and data movement through view_as_real
# ---------------------------------------------------------------------------


def _dims_of(v: Value, rank: Int) raises -> List[Int]:
    """A reduction's `int[1]? dim` over a rank-`rank` complex tensor, made
    non-negative; None / [] is every dim (of the complex tensor, never the
    trailing pair dim of its real view)."""
    var out = List[Int]()
    if v.tag == TAG_NONE:
        for d in range(rank):
            out.append(d)
        return out^
    var dims = IntList(v)
    if len(dims) == 0:
        for d in range(rank):
            out.append(d)
        return out^
    for i in range(len(dims)):
        var d = dims[i]
        out.append(_wrap_dim(d, rank))
    return out^


def _wrap_dim(d: Int, rank: Int) raises -> Int:
    """`c10::maybe_wrap_dim` on the COMPLEX tensor's rank (a 0-d tensor
    takes 0 / -1, both the tensor itself): an IndexError, as ATen raises."""
    var lim = max(rank, 1)
    if d < -lim or d >= lim:
        index_error(
            "Dimension out of range (expected to be in range of ["
            + String(-lim)
            + ", "
            + String(lim - 1)
            + "], but got "
            + String(d)
            + ")"
        )
    return d + lim if d < 0 else d


def _common_complex(ts: List[T]) raises -> Int32:
    var st = ST_UNDEFINED
    for t in ts:
        st = cx_promote(st, t.stype)
    return st


def _real_views(
    ts: List[T], st: Int32, mut keep: List[Owned]
) raises -> List[T]:
    """Each tensor converted to complex dtype `st`, then viewed as real."""
    var out = List[T]()
    for t in ts:
        var c = own_if_new(cast_complex(t, st), t)
        var rv = as_real(c.t)
        out.append(rv.t.copy())
        keep.append(c^)
        keep.append(rv^)
    return out^


def _tensor_list_arg(mut c: _Call, ts: List[T]):
    """A `Tensor[]` argument: an array of handles (held by `c`)."""
    var l = List[Int64](capacity=max(len(ts), 1))
    for t in ts:
        var h = retain(t)
        c.held.append(h)
        l.append(Int64(h))
    var addr = Int(l.unsafe_ptr())
    c.lists.append(l^)
    c.vals.append(Value(TAG_TENSOR_LIST, Int32(len(ts)), Int64(addr), 0))


def _cat(args: Values, stack: Bool, out_st: Int32 = -1) raises -> Owned:
    """cat / stack of tensors where one is complex: every input converted
    straight to the result dtype (the out's for cat.out, else the common
    one -- no detour through a narrower complex type), the real op on their
    real views (the pair dim trails, so a non-negative dim is the same dim),
    viewed back. cat skips the legacy empty 1-D tensors, as ATen does."""
    var all_ts = v_tensor_list_c(args[unsafe_offset=0])
    var dim = Int(args[unsafe_offset=1].a)
    if len(all_ts) == 0:
        raise Error("torch.cat(): expected a non-empty list of Tensors")
    var device = all_ts[0].device
    for i in range(len(all_ts)):
        _require_same_device(all_ts[i], device, "tensors")
        if not stack and all_ts[i].rank == 0:
            raise Error(
                "zero-dimensional tensor (at position ",
                i,
                ") cannot be concatenated",
            )
    var st = _common_complex(all_ts)
    if out_st >= 0 and is_complex_stype(out_st):
        _check_cast(st, out_st)
        st = out_st
    var ts = List[T]()
    for t in all_ts:
        if stack or not (t.rank == 1 and t.numel == 0):
            ts.append(t.copy())
    if len(ts) == 0:
        # only legacy empties: an empty result of the common dtype
        var shape = IndexList[MAX_RANK](1)
        shape[MAX_RANK - 1] = 0
        return own(new_tensor(shape, 1, st, device))
    var rank = 0
    for t in ts:
        rank = max(rank, t.rank)
    var keep = List[Owned]()
    var rvs = _real_views(ts, st, keep)
    var c = _Call("aten::stack" if stack else "aten::cat")
    _tensor_list_arg(c, rvs)
    c.i(_wrap_dim(dim, rank + 1 if stack else rank))
    var r = c.one()
    keep.clear()
    var cv = as_complex(r.t)
    _ = r^
    return cv^


def cx_cat(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cat(Tensor[] tensors, int dim=0)"""
    _ret(rets, _cat(args, False))


def cx_cat_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cat.out(Tensor[] tensors, int dim=0, *, out): resize, then the out
    may not overlap itself or any input (TensorShape.cpp's cat meta)."""
    var dest = v_tensor_c(args[unsafe_offset=2])
    var res = _cat(args, False, dest.stype)
    _same_device(dest, res.t)
    var geo = resized_geometry(dest, res.t.shape, res.t.rank)
    assert_no_internal_overlap(geo)
    for t in v_tensor_list_c(args[unsafe_offset=0]):
        assert_no_overlap(geo, t)
    _store(dest, res^, rets)


def cx_stack(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """stack(Tensor[] tensors, int dim=0)"""
    _ret(rets, _cat(args, True))


def _list_has_complex(v: Value) -> Bool:
    """Whether a `Tensor[]` record holds a complex tensor: one dtype read
    per handle, no tensor views built (the real cat / stack pay only this)."""
    if v.tag != TAG_TENSOR_LIST and v.tag != TAG_OPT_TENSOR_LIST:
        return False
    var ptrs = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        var h = ptrs[unsafe_offset=i]
        if h != 0 and is_complex_stype(
            external_call["tmb_tensor_dtype", Int32](h)
        ):
            return True
    return False


def routed_list[
    real_op: OpFn, complex_op: OpFn
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`routed` for an op whose first argument is a `Tensor[]`."""
    if _list_has_complex(args[unsafe_offset=0]):
        complex_op(args, n_args, rets, n_rets)
        return
    real_op(args, n_args, rets, n_rets)


def _on_real_view(
    op: StaticString, overload: StaticString, args: Values, n_args: Int
) raises -> Owned:
    """`op(view_as_real(self), <the other arguments>)`, viewed back as
    complex: for the indexing ops whose arguments only address the leading
    dims (index, index_select with a non-negative dim, repeat's extra 1)."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var rv = as_real(x)
    var c = _Call(op, overload)
    c.t(rv.t)
    for i in range(1, n_args):
        c.v(args[unsafe_offset=i])
    var r = c.one()
    _ = rv^
    var cv = as_complex(r.t)
    _ = r^
    return cv^


def _check_index_count(x: T, indices: Value) raises:
    """ATen's `too many indices` check, on the COMPLEX tensor's dims: its
    real view has one more (the pair), which an index must never reach. A
    bool / byte mask indexes as many dims as it has."""
    var used = 0
    var present = v_opt_tensor_list_present(indices)
    var ts = v_tensor_list_c(indices)
    var k = 0
    for i in range(len(present)):
        if not present[i]:
            used += 1  # a None index consumes a dim too (ATen counts it)
            continue
        var t = ts[k].copy()
        k += 1
        if t.stype == ST_BOOL or t.stype == ST_UINT8:
            used += t.rank
        else:
            used += 1
    if used > x.rank:
        index_error(
            "too many indices for tensor of dimension "
            + String(x.rank)
            + " (got "
            + String(used)
            + ")"
        )


def cx_index(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """index.Tensor(self, Tensor?[] indices)"""
    _check_index_count(v_tensor_c(args[unsafe_offset=0]), args[unsafe_offset=1])
    _ret(rets, _on_real_view("aten::index", "Tensor", args, n_args))


def _dim_args(
    op: StaticString, overload: StaticString, args: Values, dim_at: Int
) raises -> Owned:
    """`op` with argument `dim_at` an int dim made non-negative first."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var rv = as_real(x)
    var c = _Call(op, overload)
    c.t(rv.t)
    c.i(_wrap_dim(Int(args[unsafe_offset=dim_at].a), x.rank))
    c.v(args[unsafe_offset=2])
    var r = c.one()
    _ = rv^
    var cv = as_complex(r.t)
    _ = r^
    return cv^


def _index_select(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    _require_same_device(v_tensor_c(args[unsafe_offset=2]), x.device, "index")
    if x.rank > 0:
        return _dim_args("aten::index_select", "", args, 1)
    # A 0-d self: its real view's only dim is the pair. Select from it as a
    # one-element vector (ATen's own range check on the index included).
    _ = _wrap_dim(Int(args[unsafe_offset=1].a), 0)
    var index = v_tensor_c(args[unsafe_offset=2])
    var one = IndexList[MAX_RANK](1)
    var vec = own(view_strided(x, one, IndexList[MAX_RANK](0), 1, x.offset))
    var rv = as_real(vec.t)
    var c = _Call("aten::index_select")
    c.t(rv.t)
    c.i(0)
    c.t(index)
    var r = c.one()
    _ = rv^
    _ = vec^
    # ATen gives a 0-d self's selection the self's (0-d) shape
    var shape = List[Int]()
    shape.append(2)
    var re = _Call("aten::reshape")
    re.t(r.t)
    re.ints(shape)
    var shaped = re.one()
    _ = r^
    var cv = as_complex(shaped.t)
    _ = shaped^
    return cv^


def cx_index_select(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """index_select(self, int dim, Tensor index)"""
    _ret(rets, _index_select(args))


def cx_index_select_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """index_select.out(self, dim, index, *, out): the out's exact dtype,
    and on the out as given (Indexing.cu checks before resizing): no
    internal overlap, no overlap with self or the index."""
    var dest = v_tensor_c(args[unsafe_offset=3])
    var x = v_tensor_c(args[unsafe_offset=0])
    if dest.stype != x.stype:
        raise Error(
            "expected scalar type ",
            _st_name(dest.stype),
            " but found ",
            _st_name(x.stype),
        )
    _same_device(dest, x)
    assert_no_internal_overlap(dest)
    assert_no_overlap(dest, x)
    assert_no_overlap(dest, v_tensor_c(args[unsafe_offset=2]))
    _store(dest, _index_select(args), rets)


def _int_list_dims(v: Value, rank: Int) raises -> List[Int]:
    var dims = IntList(v)
    var out = List[Int]()
    for i in range(len(dims)):
        out.append(_wrap_dim(dims[i], rank))
    return out^


def cx_flip(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """flip(self, int[] dims)"""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dims = _int_list_dims(args[unsafe_offset=1], x.rank)
    for i in range(len(dims)):
        for j in range(i):
            if dims[i] == dims[j]:
                raise Error(
                    "dim ",
                    dims[i],
                    " appears multiple times in the list of dims",
                )
    if x.rank == 0:
        # a 0-d tensor's only "dim" is itself: the real view's would be the
        # pair, which a flip must never reach
        var copy = own(new_tensor(x.shape, 0, x.stype, x.device))
        copy_strided_into(copy.t, x)
        _ret(rets, copy^)
        return
    var rv = as_real(x)
    var c = _Call("aten::flip")
    c.t(rv.t)
    c.ints(dims)
    var r = c.one()
    _ = rv^
    _ret(rets, as_complex(r.t))
    _ = r^


def cx_roll(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """roll(self, SymInt[1] shifts, int[1] dims=[]); no dims rolls the
    flattened tensor, which is dim 0 of the [numel, 2] pairs."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dims = IntList(args[unsafe_offset=2])
    if x.rank == 0 and len(dims) > 0:
        # roll_common: a 0-d tensor has no dim to roll along
        index_error(
            "Dimension specified as "
            + String(dims[0])
            + " but tensor has no dimensions"
        )
    var src = own_if_new(contiguous(x), x)
    var rv = as_real(src.t)
    var c = _Call("aten::roll")
    if len(dims) == 0:
        var shape = List[Int]()
        shape.append(x.numel)
        shape.append(2)
        var flat_c = _Call("aten::reshape")
        flat_c.t(rv.t)
        flat_c.ints(shape)
        var flat = flat_c.one()
        c.t(flat.t)
        c.v(args[unsafe_offset=1])
        var zero = List[Int]()
        zero.append(0)
        c.ints(zero)
        var r = c.one()
        _ = flat^
        var back_c = _Call("aten::reshape")
        var full = x.logical_shape()
        full.append(2)
        back_c.t(r.t)
        back_c.ints(full)
        var back = back_c.one()
        _ = r^
        _ret(rets, as_complex(back.t))
        _ = back^
    else:
        c.t(rv.t)
        c.v(args[unsafe_offset=1])
        c.ints(_int_list_dims(args[unsafe_offset=2], x.rank))
        var r = c.one()
        _ret(rets, as_complex(r.t))
        _ = r^
    _ = rv^
    _ = src^


def cx_repeat(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """repeat(self, SymInt[] repeats): the pair dim repeats once."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var reps = IntList(args[unsafe_offset=1])
    if len(reps) < x.rank:
        raise Error(
            "Number of dimensions of repeat dims can not be smaller than"
            " number of dimensions of tensor"
        )
    var rv = as_real(x)
    var r2 = reps.to_list()
    r2.append(1)
    var c = _Call("aten::repeat")
    c.t(rv.t)
    c.ints(r2)
    var r = c.one()
    _ = rv^
    _ret(rets, as_complex(r.t))
    _ = r^


def cx_index_put_impl_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """_index_put_impl_(self, Tensor?[] indices, Tensor values, bool
    accumulate, bool unsafe) with ONE leading index (an integer tensor, or a
    bool / byte mask over the leading dims, turned into its nonzero
    positions as ATen does), the values of self's own dtype. Self is
    collapsed to [leading, rest] (a contiguous working copy when it is not
    contiguous), so the real op sees at most a rank-3 real view whatever
    self's rank."""
    var self = v_tensor_c(args[unsafe_offset=0])
    var values = v_tensor_c(args[unsafe_offset=2])
    _check_index_count(self, args[unsafe_offset=1])
    if self.stype != values.stype:
        # CUDA's index_put_ converts nothing (Indexing.cu's index_put_with_sort
        # and TensorAdvancedIndexing's index_put_ both check the dtypes)
        raise Error(
            "Index put requires the source and destination dtypes match, got ",
            _st_name(self.stype),
            " for the destination and ",
            _st_name(values.stype),
            " for the source.",
        )
    var present = v_opt_tensor_list_present(args[unsafe_offset=1])
    var given = v_tensor_list_c(args[unsafe_offset=1])
    if len(given) != 1 or not present[0] or self.rank == 0:
        unsupported("complex index_put_ with other than one leading index")
    var index = given[0].copy()
    _require_same_device(values, self.device, "values")
    var mask = index.stype == ST_BOOL or index.stype == ST_UINT8
    var lead_dims = index.rank if mask else 1
    if mask:
        for d in range(index.rank):
            if index.dim(d) != self.dim(d):
                index_error(
                    "The shape of the mask "
                    + shape_str(index)
                    + " at index "
                    + String(d)
                    + " does not match the shape of the indexed tensor "
                    + shape_str(self)
                    + " at index "
                    + String(d)
                )
    elif index.rank != 1:
        unsupported("complex index_put_ with a multi-dimensional index")
    var lead = 1
    for d in range(lead_dims):
        lead *= self.dim(d)
    var rest = List[Int]()
    var tail = 1
    for d in range(lead_dims, self.rank):
        rest.append(self.dim(d))
        tail *= self.dim(d)
    # the integer positions
    var pos: Owned
    if mask:
        var flat_shape = List[Int]()
        flat_shape.append(-1)
        var rm = _Call("aten::reshape")
        rm.t(index)
        rm.ints(flat_shape)
        var flat = rm.one()
        var nz = _Call("aten::nonzero")
        nz.t(flat.t)
        var nzr = nz.one()
        _ = flat^
        var rp = _Call("aten::reshape")
        rp.t(nzr.t)
        rp.ints(flat_shape)
        pos = rp.one()
        _ = nzr^
    else:
        pos = own(index.copy())
        pos.live = False
    var k = pos.t.numel
    # the values, broadcast to [k, rest...] and laid out as [k, tail]
    var vshape = List[Int]()
    vshape.append(k)
    for e in rest:
        vshape.append(e)
    var ex = _Call("aten::expand")
    ex.t(values)
    ex.ints(vshape)
    ex.b(False)
    var vexp = ex.one()
    var vdense = own_if_new(contiguous(vexp.t), vexp.t)
    var two = List[Int]()
    two.append(k)
    two.append(tail)
    var vr = _Call("aten::reshape")
    vr.t(vdense.t)
    vr.ints(two)
    var v2 = vr.one()
    # self as [lead, tail], a working copy when not contiguous
    var work = own_if_new(contiguous(self), self)
    var tshape = List[Int]()
    tshape.append(lead)
    tshape.append(tail)
    var tv = _Call("aten::view")
    tv.t(work.t)
    tv.ints(tshape)
    var t2 = tv.one()
    var rt = as_real(t2.t)
    var rv = as_real(v2.t)
    var idx_list = List[T]()
    idx_list.append(pos.t.copy())
    var c = _Call("aten::_index_put_impl_")
    c.t(rt.t)
    _tensor_list_arg(c, idx_list)
    c.vals[len(c.vals) - 1].tag = TAG_OPT_TENSOR_LIST
    c.t(rv.t)
    c.v(args[unsafe_offset=3])
    c.v(args[unsafe_offset=4])
    _ = c.run(1)
    _ = rt^
    _ = rv^
    _ = t2^
    _ = v2^
    _ = vdense^
    _ = vexp^
    _ = pos^
    if work.live:
        copy_strided_into(self, work.t)
    _ = work^
    ret_ref(rets, 0, self)


def cx_fill_tensor_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """fill_.Tensor(self, Tensor value) of a complex self (or value): the
    value read once on the host, as the real route does."""
    var t = v_tensor_c(args[unsafe_offset=0])
    var value = v_tensor_c(args[unsafe_offset=1])
    if value.rank != 0:
        raise Error(
            "fill_ only supports 0-dimension value tensor but got tensor with ",
            value.rank,
            " dimensions.",
        )
    if not t.is_complex():
        unsupported("fill_ of a real tensor with a complex value")
    var c = _Call("aten::_local_scalar_dense")
    c.t(value)
    var r = c.run(1)
    fill_complex(t, r[0])
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# ops computed on a bool view of the complex values (`z != 0`)
# ---------------------------------------------------------------------------


def cx_via_bool[
    op: StaticString, overload: StaticString, k: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`op` with each complex tensor among the first `k` arguments replaced
    by its truth (`re != 0 or im != 0`, torch's complex-to-bool): the
    logical ops, any / all, nonzero and count_nonzero see nothing else of a
    complex value. An `out=` overload (the out is the last argument) with a complex out: the logical ops compute into a bool out
    and cast into it (the in-place forms reach here so); any / all require
    a bool result, as ATen."""
    var keep = List[Owned]()
    var vals = List[Value]()
    var cx_out = False
    var out_idx = n_args - 1
    comptime is_out = overload.endswith("out")
    comptime if is_out:
        cx_out = v_is_complex(args[unsafe_offset=out_idx])
        if cx_out and (op == "any" or op == "all"):
            raise Error(
                op,
                " only supports bool tensor for result, got: ",
                _st_name(v_tensor_c(args[unsafe_offset=out_idx]).stype),
            )
    for i in range(n_args):
        var v = args[unsafe_offset=i].copy()
        if cx_out and i == out_idx:
            var d = v_tensor_c(v)
            var tmp = own(
                new_tensor(IndexList[MAX_RANK](1), 1, ST_BOOL, d.device)
            )
            vals.append(Value(TAG_TENSOR, 0, Int64(tmp.t.h), 0))
            keep.append(tmp^)
        elif (
            i < k
            and (v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF)
            and v_is_complex(v)
        ):
            var b = own(cast_complex(v_tensor_c(v), ST_BOOL))
            vals.append(Value(TAG_TENSOR, 0, Int64(b.t.h), 0))
            keep.append(b^)
        else:
            vals.append(v^)
    var r = call_op(String("aten::") + op, overload, vals^, n_rets)
    if cx_out:
        var dest = v_tensor_c(args[unsafe_offset=out_idx])
        _store_ew_n(args, k, dest, own(r.take_tensor(0)), rets)
        _ = keep^
        return
    for i in range(n_rets):
        var rec = r[i]
        if rec.tag == TAG_TENSOR or rec.tag == TAG_TENSOR_REF:
            var o = own(r.take_tensor(i))
            ret_owned(rets, i, o)
        else:
            rets[unsafe_offset=i] = rec.copy()
    _ = keep^


def cx_equal(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """equal(self, other): same shape and every element equal."""
    var a = v_tensor_c(args[unsafe_offset=0])
    var b = v_tensor_c(args[unsafe_offset=1])
    if not a.same_shape(b):
        rets[unsafe_offset=0] = Value(TAG_BOOL, 0, 0, 0)
        return
    var eq = _binary("eq", args[unsafe_offset=0], args[unsafe_offset=1])
    var c = _Call("aten::all")
    c.t(eq.t)
    var every = c.one()
    var s = _Call("aten::_local_scalar_dense")
    s.t(every.t)
    var r = s.run(1)
    rets[unsafe_offset=0] = Value(TAG_BOOL, 0, r[0].a, 0)


# ---------------------------------------------------------------------------
# where / masked_fill / rsub
# ---------------------------------------------------------------------------


def _device_complex(v: Value, st: Int32, like: T) raises -> Owned:
    """An operand of `where` as a mojo tensor of complex dtype `st`: a mojo
    tensor converted, a host 0-d tensor or a Scalar as a device 0-d one."""
    if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
        var t = v_tensor_c(v)
        if t.on_mojo():
            _require_same_device(t, like.device, "self")
            return own_if_new(cast_complex(t, st), t)
    var op = _operand(v, like)
    var one = own(new_tensor(IndexList[MAX_RANK](1), 0, st, like.device))
    fill_complex(one.t, Value(TAG_COMPLEX, 0, f64_bits(op.re), f64_bits(op.im)))
    return one^


def _where(args: Values) raises -> Owned:
    """where.self(condition, self, other) with a complex self or other: both
    sides in the common complex dtype, then `where` on each part (real /
    imaginary views, the tensors' own shapes: no pair dim to exceed a
    kernel's rank limit). The device is the condition's when self and other
    are both host scalars."""
    var cond = v_tensor_c(args[unsafe_offset=0])
    var like: T
    if cond.on_mojo():
        like = cond.copy()
    else:
        like = _first_tensor(args[unsafe_offset=1], args[unsafe_offset=2])
    var a = _operand(args[unsafe_offset=1], like)
    var b = _operand(args[unsafe_offset=2], like)
    var st = _result_type(a, b)
    var ta = _device_complex(args[unsafe_offset=1], st, like)
    var tb = _device_complex(args[unsafe_offset=2], st, like)
    var parts = List[Owned]()
    for which in range(2):
        var pa = _part(ta.t, which)
        var pb = _part(tb.t, which)
        var w = _Call("aten::where", "self")
        w.t(cond)
        w.t(pa.t)
        w.t(pb.t)
        parts.append(w.one())
        _ = pa^
        _ = pb^
    _ = ta^
    _ = tb^
    return _from_parts(parts[0].t, parts[1].t, st)


def cx_where(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """where.self(Tensor condition, Tensor self, Tensor other)"""
    _ret(rets, _where(args))


def cx_where_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """where.self_out(condition, self, other, *, out)"""
    var dest = v_tensor_c(args[unsafe_offset=3])
    var res = _where(args)
    if dest.stype != res.t.stype:  # where_self_out: no cast into the out
        raise Error(
            "Expected out type to be ",
            _st_name(res.t.stype),
            " but got ",
            _st_name(dest.stype),
        )
    _store_ew[3](args, dest, res^, rets)


def _masked_fill_into(target: T, mask: Value, value: Value) raises:
    """target[mask] = value, in place, a complex target: one real
    masked_fill_ per component view."""
    var val = value.copy()
    if value.tag == TAG_TENSOR or value.tag == TAG_TENSOR_REF:
        var vt = v_tensor_c(value)
        if vt.rank != 0:
            raise Error(
                (
                    "masked_fill_ only supports a 0-dimensional value tensor,"
                    " but got tensor with "
                ),
                vt.rank,
                " dimension(s).",
            )
        var s = _Call("aten::_local_scalar_dense")
        s.t(vt)
        var r = s.run(1)
        val = r[0].copy()
    var parts = v_complex(val)
    var rv = as_real(target)
    for j in range(2):
        var sel = _Call("aten::select", "int")
        sel.t(rv.t)
        sel.i(-1)
        sel.i(j)
        var part = sel.one()
        var m = _Call("aten::masked_fill_", "Scalar")
        m.t(part.t)
        m.v(mask)
        m.s(parts[0] if j == 0 else parts[1])
        _ = m.run(1)
        _ = part^
    _ = rv^


def _masked_fill(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    if not x.is_complex():
        unsupported("masked_fill of a real tensor with a complex value")
    # the result has the broadcast shape of self and mask
    var mask = v_tensor_c(args[unsafe_offset=1])
    var shape = IndexList[MAX_RANK](1)
    var rank = max(x.rank, mask.rank)
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        var a = x.shape[i]
        var b = mask.shape[i]
        if a != b and a != 1 and b != 1:
            raise Error("masked_fill: shapes are not broadcastable")
        shape[i] = max(a, b) if a != 0 and b != 0 else 0
        strides[i] = x.strides[i] if a == shape[i] else 0
    var expanded = own(view_strided(x, shape, strides, rank, x.offset))
    var out = own(new_tensor(shape, rank, x.stype, x.device))
    copy_strided_into(out.t, expanded.t)
    _ = expanded^
    _masked_fill_into(out.t, args[unsafe_offset=1], args[unsafe_offset=2])
    return out^


def cx_masked_fill(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """masked_fill.Scalar / .Tensor(self, mask, value)"""
    _ret(rets, _masked_fill(args))


def cx_masked_fill_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """masked_fill.Scalar_out / .Tensor_out(self, mask, value, *, out)"""
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](args, dest, _masked_fill(args), rets)


def cx_masked_fill_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """masked_fill_.Scalar / .Tensor(self, mask, value)"""
    var self = v_tensor_c(args[unsafe_offset=0])
    if not self.is_complex():
        raise Error(
            "value cannot be converted to type ",
            _st_name(self.stype),
            " without overflow",
        )
    _masked_fill_into(self, args[unsafe_offset=1], args[unsafe_offset=2])
    ret_ref(rets, 0, self)


def cx_fill(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """fill.Scalar(self, value): a new tensor like self, filled."""
    var x = v_tensor_c(args[unsafe_offset=0])
    if not x.is_complex():
        unsupported("fill of a real tensor with a complex value")
    var out = own(new_tensor(x.shape, x.rank, x.stype, x.device))
    fill_complex(out.t, args[unsafe_offset=1].copy())
    _ret(rets, out^)


# ---------------------------------------------------------------------------
# more data movement on the real views
# ---------------------------------------------------------------------------


def _part(t: T, which: Int) raises -> Owned:
    """The real (0) or imaginary (1) part of a complex tensor as a strided
    real VIEW of it (`view_as_real(t)[..., which]`): writable, of t's own
    shape, so an op on it never meets the real view's extra pair dim (rank
    limits, 0-d tensors)."""
    var rv = as_real(t)
    var c = _Call("aten::select", "int")
    c.t(rv.t)
    c.i(-1)
    c.i(which)
    var r = c.one()
    _ = rv^
    return r^


def _from_parts(re: T, im: T, st: Int32) raises -> Owned:
    """A fresh complex tensor of dtype `st` from its real and imaginary
    parts (same shape, `st`'s component dtype)."""
    var out = own(new_tensor(re.shape, re.rank, st, re.device))
    copy_strided_into(out.t.component(0), re)
    copy_strided_into(out.t.component(1), im)
    return out^


def _gather(args: Values) raises -> Owned:
    """gather on each part (strided real views of the tensor's own shape)."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var index = v_tensor_c(args[unsafe_offset=2])
    _require_same_device(index, x.device, "index")
    _ = _wrap_dim(Int(args[unsafe_offset=1].a), x.rank)
    var parts = List[Owned]()
    for which in range(2):
        var p = _part(x, which)
        var c = _Call("aten::gather")
        c.t(p.t)
        c.v(args[unsafe_offset=1])
        c.t(index)
        c.v(args[unsafe_offset=3])
        parts.append(c.one())
        _ = p^
    return _from_parts(parts[0].t, parts[1].t, x.stype)


def cx_gather(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """gather(self, int dim, Tensor index, *, bool sparse_grad)"""
    _ret(rets, _gather(args))


def cx_gather_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """gather.out(self, dim, index, *, sparse_grad, out): the out's exact
    dtype; resize, then no internal overlap, no overlap with self, no
    partial overlap with the index (TensorAdvancedIndexing's gather meta)."""
    var dest = v_tensor_c(args[unsafe_offset=4])
    var x = v_tensor_c(args[unsafe_offset=0])
    _exact_dtype(dest, x.stype)
    var res = _gather(args)
    _same_device(dest, res.t)
    var geo = resized_geometry(dest, res.t.shape, res.t.rank)
    assert_no_internal_overlap(geo)
    assert_no_overlap(geo, x)
    assert_no_partial_overlap(geo, v_tensor_c(args[unsafe_offset=2]))
    _store(dest, res^, rets)


def _masked_select(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    var mask = v_tensor_c(args[unsafe_offset=1])
    var rv = as_real(x)
    var u = _Call("aten::unsqueeze")
    u.t(mask)
    u.i(-1)
    var mu = u.one()
    var c = _Call("aten::masked_select")
    c.t(rv.t)
    c.t(mu.t)
    var r = c.one()
    _ = rv^
    _ = mu^
    var shape = List[Int]()
    shape.append(-1)
    shape.append(2)
    var re = _Call("aten::reshape")
    re.t(r.t)
    re.ints(shape)
    var pairs = re.one()
    _ = r^
    var cv = as_complex(pairs.t)
    _ = pairs^
    return cv^


def cx_masked_select(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """masked_select(self, mask)"""
    _ret(rets, _masked_select(args))


def cx_masked_select_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """masked_select.out(self, mask, *, out)"""
    var dest = v_tensor_c(args[unsafe_offset=2])
    if dest.stype != v_tensor_c(args[unsafe_offset=0]).stype:
        raise Error(
            "masked_select(): self and result must have the same scalar type"
        )
    _store(dest, _masked_select(args), rets)


def _index_into(op: StaticString, target: T, args: Values, n_args: Int) raises:
    """index_add_ / index_copy_ (self, dim, index, source[, alpha]) into a
    complex `target` (self or a copy of it), one call per part (real /
    imaginary views of their own shapes): add and copy are per part."""
    var source = v_tensor_c(args[unsafe_offset=3])
    var index = v_tensor_c(args[unsafe_offset=2])
    _require_same_device(index, target.device, "index")
    _require_same_device(source, target.device, "source")
    _ = _wrap_dim(Int(args[unsafe_offset=1].a), target.rank)
    var alpha = 1.0
    if n_args > 4:
        var al = v_complex(args[unsafe_offset=4])
        if al[1] != 0.0:
            unsupported("index_add with a complex alpha")
        alpha = al[0]
    for which in range(2):
        var tp = _part(target, which)
        var sp = _part(source, which)
        var c = _Call(op)
        c.t(tp.t)
        c.v(args[unsafe_offset=1])
        c.t(index)
        c.t(sp.t)
        if n_args > 4:
            c.s(alpha)
        _ = c.run(1)
        _ = sp^
        _ = tp^


def _index_func(op: StaticString, args: Values, n_args: Int) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    var source_st = v_tensor_c(args[unsafe_offset=3]).stype
    if x.stype != source_st:
        # TensorAdvancedIndexing.cpp's index_func_meta_impl / index_copy
        if op == "aten::index_copy_":
            raise Error(
                (
                    "index_copy_(): self and source expected to have the same"
                    " dtype, but got (self) "
                ),
                _st_name(x.stype),
                " and (source) ",
                _st_name(source_st),
            )
        raise Error(
            "index_add_(): self (",
            _st_name(x.stype),
            ") and source (",
            _st_name(source_st),
            ") must have the same scalar type",
        )
    var out = own(new_tensor(x.shape, x.rank, x.stype, x.device))
    copy_strided_into(out.t, x)
    _index_into(op, out.t, args, n_args)
    return out^


def _index_func_store(
    op: StaticString,
    args: Values,
    n_args: Int,
    dest_idx: Int,
    rets: Values,
) raises:
    """index_add / index_copy into an out (or self, in place): the out's
    exact dtype; resize, then no internal overlap and no overlap with the
    index or the source (index_func_meta_impl), and a partial overlap with
    self only when the out is not self."""
    var dest = v_tensor_c(args[unsafe_offset=dest_idx])
    var x = v_tensor_c(args[unsafe_offset=0])
    _exact_dtype(dest, x.stype)
    var geo = resized_geometry(dest, x.shape, x.rank)
    assert_no_internal_overlap(geo)
    assert_no_overlap(geo, v_tensor_c(args[unsafe_offset=2]))
    assert_no_overlap(geo, v_tensor_c(args[unsafe_offset=3]))
    if dest.impl() != x.impl():
        assert_no_partial_overlap(geo, x)
    _store(dest, _index_func(op, args, n_args), rets)


def cx_index_add(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """index_add(self, dim, index, source, *, alpha)"""
    _ret(rets, _index_func("aten::index_add_", args, 5))


def cx_index_add_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _index_func_store("aten::index_add_", args, 5, 0, rets)


def cx_index_add_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _index_func_store("aten::index_add_", args, 5, 5, rets)


def cx_index_copy(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """index_copy(self, dim, index, source)"""
    _ret(rets, _index_func("aten::index_copy_", args, 4))


def cx_index_copy_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _index_func_store("aten::index_copy_", args, 4, 0, rets)


def cx_index_copy_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _index_func_store("aten::index_copy_", args, 4, 4, rets)


def _tri(op: StaticString, args: Values) raises -> Owned:
    """tril / triu of a complex matrix batch: the pair dim moved to the
    front of the real view, so the triangle is of the last two dims."""
    var x = v_tensor_c(args[unsafe_offset=0])
    if x.rank < 2:
        # checked on the complex shape: the real view's pair dim would make
        # a 1-D input look like a matrix
        raise Error(
            op.removeprefix("aten::"),
            ": input tensor must have at least 2 dimensions",
        )
    var rv = as_real(x)
    var m = _Call("aten::movedim", "int")
    m.t(rv.t)
    m.i(-1)
    m.i(0)
    var front = m.one()
    var c = _Call(op)
    c.t(front.t)
    c.v(args[unsafe_offset=1])
    var r = c.one()
    _ = front^
    _ = rv^
    var b = _Call("aten::movedim", "int")
    b.t(r.t)
    b.i(0)
    b.i(-1)
    var back = b.one()
    _ = r^
    var dense = own_if_new(contiguous(back.t), back.t)
    var cv = as_complex(dense.t)
    _ = dense^
    _ = back^
    return cv^


def cx_tril(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(rets, _tri("aten::tril", args))


def cx_triu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(rets, _tri("aten::triu", args))


def cx_tril_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor_c(args[unsafe_offset=0])
    _store(self, _tri("aten::tril", args), rets)


def cx_triu_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor_c(args[unsafe_offset=0])
    _store(self, _tri("aten::triu", args), rets)


def cx_tril_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor_c(args[unsafe_offset=2])
    _exact_dtype(dest, v_tensor_c(args[unsafe_offset=0]).stype)
    _store(dest, _tri("aten::tril", args), rets)


def cx_triu_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor_c(args[unsafe_offset=2])
    _exact_dtype(dest, v_tensor_c(args[unsafe_offset=0]).stype)
    _store(dest, _tri("aten::triu", args), rets)


# ---------------------------------------------------------------------------
# random fills (ATen's normal_impl_ / uniform_impl_ for complex)
# ---------------------------------------------------------------------------


def cx_normal_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """normal_(self, mean, std, *, generator): the real view filled with
    std / sqrt(2) (DistributionTemplates.h's normal_impl_), so the draws are
    CUDA's."""
    var self = v_tensor_c(args[unsafe_offset=0])
    var mean = v_f64(args[unsafe_offset=1])
    var std = v_f64(args[unsafe_offset=2])
    if std < 0.0:
        raise Error("normal expects std >= 0.0, but found std ", std)
    var rv = as_real(self)
    var c = _Call("aten::normal_")
    c.t(rv.t)
    c.f(mean)
    c.f(std / 1.4142135623730951)
    c.v(args[unsafe_offset=3])
    _ = c.run(1)
    _ = rv^
    ret_ref(rets, 0, self)


def cx_uniform_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """uniform_(self, from, to, *, generator) on the real view."""
    var self = v_tensor_c(args[unsafe_offset=0])
    var rv = as_real(self)
    var c = _Call("aten::uniform_")
    c.t(rv.t)
    c.v(args[unsafe_offset=1])
    c.v(args[unsafe_offset=2])
    c.v(args[unsafe_offset=3])
    _ = c.run(1)
    _ = rv^
    ret_ref(rets, 0, self)


# aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
def op_conj_physical_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """ATen's composite calls the conj stub directly, which has no mojo
    kernel: conjugate the stored values in place (a real tensor is left
    as is)."""
    var self = v_tensor_c(args[unsafe_offset=0])
    if self.is_complex() and self.numel > 0:
        var conj = own(conj_copy(self))
        copy_strided_into(self, conj.t)
        _ = conj^
    ret_ref(rets, 0, self)


def register_complex(site: Site) raises:
    impl[op_conj_physical_, "conj_physical_"](site)
