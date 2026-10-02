"""Dense linear algebra on the mojo device: Cholesky, LU, triangular solves,
Householder QR, symmetric eigendecomposition, SVD, LDL^T, least squares,
determinants and the matrix exponential.

The factorizations run on the `linalg` kernel family
(`tmb/kernels/linalg/entry.mojo`, one thread block per matrix, no vendor
solver); everything else is the composition torch's own structured kernels
make (BatchLinearAlgebra.cpp / LinearAlgebra.cpp), reproduced here with the
same checks, output layouts and error messages: outputs that LAPACK fills
are batched column-major ("F-contiguous"), info tensors int32, pivots
1-based int32.

Real float32 / float64 only: a complex input is declined, as is float64 on
an Apple GPU (Metal has no float64).
"""
from std.utils import IndexList
from std.utils.numerics import nan

from tmb.backend.abi import (
    ST_BFLOAT16,
    ST_BOOL,
    ST_COMPLEX128,
    ST_COMPLEX32,
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
    TAG_DOUBLE,
    TAG_INT,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_STRING,
    TAG_TENSOR,
    Owned,
    Results,
    T,
    Value,
    Values,
    call_op,
    contiguous_strides,
    dtype_code,
    f64_bits,
    new_strided,
    new_tensor,
    own,
    own_if_new,
    release,
    retain,
    ret_owned,
    ret_ref,
    set_sizes_strides,
    unsupported,
    v_bool,
    v_bool_or,
    v_f64,
    v_is_none,
    v_int,
    v_string,
    v_tensor,
    view_strided,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    can_cast,
    cast_to,
    check_out_as,
    copy_strided_into,
    device_str,
    fill_value,
    resize_storage_bytes,
)


# ---------------------------------------------------------------------------
# Small helpers: names, shapes, layouts
# ---------------------------------------------------------------------------


def _st_name(st: Int32) -> String:
    """`c10::ScalarType` as `operator<<` prints it (`Float`, `Long`...)."""
    if st == ST_FLOAT32:
        return "Float"
    if st == ST_FLOAT64:
        return "Double"
    if st == ST_FLOAT16:
        return "Half"
    if st == ST_BFLOAT16:
        return "BFloat16"
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
    if st == ST_COMPLEX64:
        return "ComplexFloat"
    if st == ST_COMPLEX128:
        return "ComplexDouble"
    if st == ST_COMPLEX32:
        return "ComplexHalf"
    return String("ScalarType(") + String(st) + ")"


def _is_complex(st: Int32) -> Bool:
    return st == ST_COMPLEX32 or st == ST_COMPLEX64 or st == ST_COMPLEX128


def _is_float(st: Int32) -> Bool:
    return (
        st == ST_FLOAT32
        or st == ST_FLOAT64
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
    )


def _sizes_str(dims: List[Int]) -> String:
    var s = String("[")
    for i in range(len(dims)):
        if i:
            s += ", "
        s += String(dims[i])
    return s + "]"


def _dims(t: T) -> List[Int]:
    return t.logical_shape()


def _padded(dims: List[Int]) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    var r = len(dims)
    for i in range(r):
        shape[MAX_RANK - r + i] = dims[i]
    return shape


def _batch_dims(t: T) -> List[Int]:
    var out = List[Int]()
    for i in range(t.rank - 2):
        out.append(t.dim(i))
    return out^


def _prod(dims: List[Int]) -> Int:
    var p = 1
    for d in dims:
        p *= d
    return p


def _batch_count(t: T) -> Int:
    var p = 1
    for i in range(t.rank - 2):
        p *= t.dim(i)
    return p


def _with(var dims: List[Int], a: Int, b: Int) -> List[Int]:
    dims.append(a)
    dims.append(b)
    return dims^


def _with1(var dims: List[Int], a: Int) -> List[Int]:
    dims.append(a)
    return dims^


def _fstrides(dims: List[Int]) -> IndexList[MAX_RANK]:
    """`batched_matrix_contiguous_strides(sizes, f_contig=True)`: C-ordered
    batches of column-major matrices."""
    var r = len(dims)
    var shape = _padded(dims)
    var strides = contiguous_strides(shape, r)
    if r >= 2:
        strides[MAX_RANK - 1] = max(dims[r - 2], 1)
        strides[MAX_RANK - 2] = 1
    return strides


def _new_f(dims: List[Int], st: Int32, device: Int) raises -> T:
    return new_strided(_padded(dims), _fstrides(dims), len(dims), st, device)


def _new_c(dims: List[Int], st: Int32, device: Int) raises -> T:
    return new_tensor(_padded(dims), len(dims), st, device)


def _empty0(st: Int32, device: Int) raises -> T:
    """`at::empty({0})`, what torch returns for an output it does not
    compute."""
    var d = List[Int]()
    d.append(0)
    return _new_c(d, st, device)


def _clone_f(t: T) raises -> T:
    """`cloneBatchedColumnMajor`."""
    var out = own(_new_f(_dims(t), t.stype, t.device))
    copy_strided_into(out.t, t)
    return out.take()


def _clone_c(t: T) raises -> T:
    var out = own(_new_c(_dims(t), t.stype, t.device))
    copy_strided_into(out.t, t)
    return out.take()


def _mT(t: T) raises -> T:
    """The transpose of the last two dims, as a view (an owned handle)."""
    var shape = t.shape
    var strides = t.strides
    shape[MAX_RANK - 1] = t.shape[MAX_RANK - 2]
    shape[MAX_RANK - 2] = t.shape[MAX_RANK - 1]
    strides[MAX_RANK - 1] = t.strides[MAX_RANK - 2]
    strides[MAX_RANK - 2] = t.strides[MAX_RANK - 1]
    return view_strided(t, shape, strides, t.rank, t.offset)


def _narrow(t: T, dim: Int, start: Int, length: Int) raises -> T:
    """`t.narrow(dim, start, length)` as a view (owned handle)."""
    var d = dim + t.rank if dim < 0 else dim
    var shape = t.shape
    shape[MAX_RANK - t.rank + d] = length
    var off = t.offset + start * t.stride(d)
    return view_strided(t, shape, t.strides, t.rank, off)


def _diagonal(t: T) raises -> T:
    """`t.diagonal(0, -2, -1)` as a view (owned handle)."""
    var k = min(t.dim(-2), t.dim(-1))
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    for i in range(t.rank - 2):
        shape[MAX_RANK - (t.rank - 1) + i] = t.dim(i)
        strides[MAX_RANK - (t.rank - 1) + i] = t.stride(i)
    shape[MAX_RANK - 1] = k
    strides[MAX_RANK - 1] = t.stride(-2) + t.stride(-1)
    return view_strided(t, shape, strides, t.rank - 1, t.offset)


def _expand(t: T, dims: List[Int]) raises -> T:
    """`t.expand(dims)` (same rank or broadcast from the left) as a view."""
    var r = len(dims)
    var strides = IndexList[MAX_RANK](0)
    for i in range(r):
        var ti = i - (r - t.rank)
        if ti >= 0 and t.dim(ti) == dims[i]:
            strides[MAX_RANK - r + i] = t.stride(ti)
        elif ti >= 0 and t.dim(ti) != 1:
            raise Error(
                "The expanded size of the tensor (",
                dims[i],
                ") must match the existing size (",
                t.dim(ti),
                ") at non-singleton dimension ",
                i,
            )
    return view_strided(t, _padded(dims), strides, r, t.offset)


def _broadcast(a: List[Int], b: List[Int]) raises -> List[Int]:
    """`infer_size` of two batch shapes."""
    var r = max(len(a), len(b))
    var out = List[Int]()
    for i in range(r):
        var ia = i - (r - len(a))
        var ib = i - (r - len(b))
        var x = a[ia] if ia >= 0 else 1
        var y = b[ib] if ib >= 0 else 1
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise Error(
                "The size of tensor a (",
                x,
                ") must match the size of tensor b (",
                y,
                ") at non-singleton dimension ",
                i,
            )
    return out^


def _bstride(t: T) -> Int:
    """The stride between consecutive matrices of `t`'s flattened batch, or
    -1 when its batch dims do not flatten to a single stride."""
    var s = -1
    var expect = 0
    var i = t.rank - 3
    while i >= 0:
        var d = t.dim(i)
        if d != 1:
            if s == -1:
                s = t.stride(i)
                expect = s * d
            elif t.stride(i) != expect:
                return -1
            else:
                expect = t.stride(i) * d
        i -= 1
    return 0 if s == -1 else s


def _flat(t: T) raises -> Owned:
    """`t` itself (not owned) when its batch flattens to one stride, else a
    column-major copy (owned)."""
    if _bstride(t) >= 0:
        return own_if_new(t.copy(), t)
    return own(_clone_f(t))


def _is_metal(t: T) raises -> Bool:
    return t.on_mojo() and dev(t.device)[].api == "metal"


def _check_float(t: T, name: String, allow_low: Bool = True) raises:
    """`checkFloatingOrComplex`, then what this device computes in: real
    float32 / float64 (float64 not on Metal)."""
    if not _is_float(t.stype) and not _is_complex(t.stype):
        raise Error(
            name,
            ": Expected a floating point or complex tensor as input. Got ",
            _st_name(t.stype),
        )
    if not allow_low and (t.stype == ST_FLOAT16 or t.stype == ST_BFLOAT16):
        raise Error(
            name,
            ": Low precision dtypes not supported. Got ",
            _st_name(t.stype),
        )
    _check_compute(t, name)


def _check_compute(t: T, name: String) raises:
    if _is_complex(t.stype):
        unsupported(name + ": complex inputs are not supported on mojo")
    if t.stype != ST_FLOAT32 and t.stype != ST_FLOAT64:
        unsupported(
            name + ": " + _st_name(t.stype) + " is not supported on mojo"
        )
    if t.stype == ST_FLOAT64 and _is_metal(t):
        unsupported(name + ": float64 is not supported on an Apple GPU")


def _check_matrix(t: T, name: String, arg: String = "A") raises:
    if t.rank < 2:
        raise Error(
            name,
            ": The input tensor ",
            arg,
            " must have at least 2 dimensions.",
        )


def _check_square(t: T, name: String, arg: String = "A") raises:
    _check_matrix(t, name, arg)
    if t.dim(-1) != t.dim(-2):
        raise Error(
            name,
            ": ",
            arg,
            " must be batches of square matrices, but they are ",
            t.dim(-2),
            " by ",
            t.dim(-1),
            " matrices",
        )


def _same_device(a: T, b: T) -> Bool:
    return a.device_type == b.device_type and a.device == b.device


def _all_on(t: T, other: T) raises:
    """The kernels dereference every operand on `t`'s device: anything on
    another device (a CPU pivots tensor, say) is refused, with the message
    of torch's TensorIterator / dispatcher device check."""
    if not _same_device(t, other):
        raise Error(
            (
                "Expected all tensors to be on the same device, but found at"
                " least two devices, "
            ),
            device_str(t),
            " and ",
            device_str(other),
            "!",
        )


# ---------------------------------------------------------------------------
# Calling other aten ops
# ---------------------------------------------------------------------------


struct _Op(Movable):
    """Arguments of one `call_op`, built in schema order."""

    var op: String
    var overload: String
    var vals: List[Value]
    var lists: List[List[Int64]]
    var held: List[Int]

    def __init__(out self, op: String, overload: String):
        self.op = op
        self.overload = overload
        self.vals = List[Value]()
        self.lists = List[List[Int64]]()
        self.held = List[Int]()

    def __deinit__(deinit self):
        for h in self.held:
            release(h)

    def t(mut self, x: T):
        # A reference of our own: the caller's handle may die (Mojo ends a
        # value at its last use) before `run`.
        var h = retain(x)
        self.held.append(h)
        self.vals.append(Value(TAG_TENSOR, 0, Int64(h), 0))

    def i(mut self, x: Int):
        self.vals.append(Value(TAG_INT, 0, Int64(x), 0))

    def b(mut self, x: Bool):
        self.vals.append(Value(TAG_BOOL, 0, Int64(1) if x else Int64(0), 0))

    def s(mut self, x: Float64):
        """A `Scalar`."""
        self.vals.append(Value(TAG_SCALAR_DOUBLE, 0, f64_bits(x), 0))

    def si(mut self, x: Int):
        """An integral `Scalar`."""
        self.vals.append(Value(TAG_SCALAR_INT, 0, Int64(x), 0))

    def f(mut self, x: Float64):
        """A `float`."""
        self.vals.append(Value(TAG_DOUBLE, 0, f64_bits(x), 0))

    def str(mut self, x: StaticString):
        self.vals.append(
            Value(
                TAG_STRING,
                Int32(x.byte_length()),
                Int64(Int(x.unsafe_ptr())),
                0,
            )
        )

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


def _check_errors(info: T, api: StaticString, is_matrix: Bool) raises:
    """`at::_linalg_check_errors`: torch's own messages, from the info
    tensor (a device-to-host read, as on CUDA)."""
    var c = _Op("aten::_linalg_check_errors", "")
    c.t(info)
    c.str(api)
    c.b(is_matrix)
    _ = c.run(0)


def _tri(t: T, upper: Bool, diagonal: Int) raises -> Owned:
    if upper:
        var c = _Op("aten::triu", "")
        c.t(t)
        c.i(diagonal)
        return c.one()
    var c = _Op("aten::tril", "")
    c.t(t)
    c.i(diagonal)
    return c.one()


def _unary(op: StaticString, t: T) raises -> Owned:
    var c = _Op(op, "")
    c.t(t)
    return c.one()


def _binary(op: StaticString, a: T, b: T) raises -> Owned:
    var c = _Op(op, "Tensor")
    c.t(a)
    c.t(b)
    if op == "aten::add" or op == "aten::sub":
        c.s(1.0)
    return c.one()


def _matmul(a: T, b: T) raises -> Owned:
    var c = _Op("aten::matmul", "")
    c.t(a)
    c.t(b)
    return c.one()


def _reduce_last(op: StaticString, t: T) raises -> Owned:
    """`sum` / `prod` over the last dim (`dim_IntList` / `dim_int`)."""
    if op == "aten::prod":
        var c = _Op(op, "dim_int")
        c.t(t)
        c.i(-1)
        c.b(False)
        c.none()
        return c.one()
    var c = _Op(op, "dim_IntList")
    c.t(t)
    var d = List[Int]()
    d.append(-1)
    c.ints(d)
    c.b(False)
    c.none()
    return c.one()


def _eye_into(t: T) raises:
    """`t.zero_(); t.diagonal(0, -2, -1).fill_(1)`."""
    fill_value(t, 0.0)
    var d = own(_diagonal(t))
    fill_value(d.t, 1.0)


# ---------------------------------------------------------------------------
# out= plumbing
# ---------------------------------------------------------------------------


def _check_out(dest: T, st: Int32, like: T) raises:
    check_out_as(dest, st, like)


def _store(mut dst: T, src: T) raises:
    """Structured `set_output_strided` for a caller's `out=`: an `out` of
    the right shape keeps its layout; any other is resized to the result's
    own strides (column-major where the result is). Then the copy."""
    assert_no_internal_overlap(dst)
    if not dst.same_shape(src):
        var numel = src.numel
        var nbytes = dst.offset * dst.itemsize
        if numel > 0:
            var span = 0
            for i in range(src.rank):
                span += (src.dim(i) - 1) * src.stride(i)
            nbytes = (dst.offset + span + 1) * dst.itemsize
        if numel > 0 and nbytes > dst.storage_nbytes():
            resize_storage_bytes(dst, nbytes)
        set_sizes_strides(dst, src.shape, src.strides, src.rank, dst.offset)
        dst = T(dst.h)
    copy_strided_into(dst, src)


# ---------------------------------------------------------------------------
# Kernel launches (tmb/kernels/linalg)
# ---------------------------------------------------------------------------


def _launch(
    op: StaticString,
    dt: DType,
    device: Int,
    p0: Int,
    p1: Int,
    p2: Int,
    p3: Int,
    p4: Int,
    ints: List[Int],
) raises:
    var ctx = ctx_for(device)
    var call = KernelCall("linalg", op)
    call.arg_dtype(0, dt)
    call.int(p0)
    call.int(p1)
    call.int(p2)
    call.int(p3)
    call.int(p4)
    var l = List[Int](capacity=len(ints) + 1)
    l.append(dtype_code(dt))
    for x in ints:
        l.append(x)
    call.tuple(l)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _k_potrf(a: T, info: T) raises:
    """Lower Cholesky of `a` in place (any strides; a flattenable batch)."""
    var n = a.dim(-1)
    var batch = _batch_count(a)
    if batch == 0:
        return
    var l = List[Int]()
    l.append(n)
    l.append(a.stride(-2))
    l.append(a.stride(-1))
    l.append(_bstride(a))
    l.append(batch)
    _launch("Potrf", a.dtype, a.device, a.ptr, info.ptr, 0, 0, 0, l)


def _k_getrf(a: T, piv: T, info: T, pivot: Bool) raises:
    var batch = _batch_count(a)
    if batch == 0:
        return
    var l = List[Int]()
    l.append(a.dim(-2))
    l.append(a.dim(-1))
    l.append(a.stride(-2))
    l.append(a.stride(-1))
    l.append(_bstride(a))
    l.append(batch)
    l.append(1 if pivot else 0)
    _launch("Getrf", a.dtype, a.device, a.ptr, piv.ptr, info.ptr, 0, 0, l)


def _k_trsm(a: T, b: T, upper: Bool, trans: Bool, unit: Bool) raises:
    """op(a) X = b in place on b (left side), op = transpose when `trans`;
    `upper` names the triangle of `a` itself."""
    var batch = _batch_count(b)
    if batch == 0 or b.numel == 0:
        return
    var a_rs = a.stride(-2)
    var a_cs = a.stride(-1)
    if trans:
        a_rs = a.stride(-1)
        a_cs = a.stride(-2)
    var l = List[Int]()
    l.append(b.dim(-2))
    l.append(b.dim(-1))
    l.append(a_rs)
    l.append(a_cs)
    l.append(_bstride(a))
    l.append(b.stride(-2))
    l.append(b.stride(-1))
    l.append(_bstride(b))
    l.append(batch)
    l.append(1 if upper == trans else 0)
    l.append(1 if unit else 0)
    _launch("Trsm", b.dtype, b.device, a.ptr, b.ptr, 0, 0, 0, l)


def _k_laswp(b: T, piv: T, forward: Bool) raises:
    """The row interchanges of `piv` (int32, contiguous, k per matrix) on
    b's rows, in order (`forward`) or reversed."""
    var batch = _batch_count(b)
    if batch == 0 or b.numel == 0:
        return
    var k = piv.dim(-1)
    var l = List[Int]()
    l.append(k)
    l.append(b.dim(-1))
    l.append(b.stride(-2))
    l.append(b.stride(-1))
    l.append(_bstride(b))
    l.append(k)
    l.append(batch)
    l.append(1 if forward else 0)
    l.append(b.dim(-2))
    _launch("Laswp", b.dtype, b.device, b.ptr, piv.ptr, 0, 0, 0, l)


def _k_geqrf(a: T, tau: T) raises:
    var batch = _batch_count(a)
    if batch == 0 or a.numel == 0:
        return
    var l = List[Int]()
    l.append(a.dim(-2))
    l.append(a.dim(-1))
    l.append(a.stride(-2))
    l.append(a.stride(-1))
    l.append(_bstride(a))
    l.append(batch)
    _launch("Geqrf", a.dtype, a.device, a.ptr, tau.ptr, 0, 0, 0, l)


def _k_ormqr(a: T, tau: T, c: T, trans: Bool) raises:
    """c = Q c (Q^T c when `trans`), Q = H(0)...H(k-1) from the reflectors
    in `a`'s columns and `tau` (contiguous, k per matrix); left side."""
    var batch = _batch_count(c)
    var k = tau.dim(-1)
    if batch == 0 or c.numel == 0 or k == 0:
        return
    var l = List[Int]()
    l.append(c.dim(-2))
    l.append(c.dim(-1))
    l.append(k)
    l.append(a.stride(-2))
    l.append(a.stride(-1))
    l.append(_bstride(a))
    l.append(k)
    l.append(c.stride(-2))
    l.append(c.stride(-1))
    l.append(_bstride(c))
    l.append(batch)
    l.append(1 if trans else 0)
    _launch("Ormqr", c.dtype, c.device, a.ptr, tau.ptr, c.ptr, 0, 0, l)


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------


def _cholesky(A: T, upper: Bool) raises -> Tuple[Owned, Owned]:
    """(L, info): `cholesky_stub` on a column-major copy of A, with the
    unused triangle zeroed (torch's non-CPU path)."""
    var L = own(_clone_f(A))
    var info = own(_new_c(_batch_dims(A), ST_INT32, A.device))
    if L.t.numel == 0:
        fill_value(info.t, 0.0)
        return (L^, info^)
    if upper:
        # U^T U = A from A's upper triangle is the lower factor of A^T.
        var w = own(_mT(L.t))
        _k_potrf(w.t, info.t)
    else:
        _k_potrf(L.t, info.t)
    return (L^, info^)


def _cholesky_ex_args(args: Values) raises -> Tuple[T, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var upper = v_bool_or(args[unsafe_offset=1], False)
    var check_errors = v_bool_or(args[unsafe_offset=2], False)
    _check_square(A, "linalg.cholesky")
    _check_float(A, "linalg.cholesky")
    return (A^, upper, check_errors)


# aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
def op_linalg_cholesky_ex(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _cholesky_ex_args(args)
    var r = _cholesky(a[0], a[1])
    if a[2]:
        _check_errors(r[1].t, "linalg.cholesky_ex", a[0].rank == 2)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
def op_linalg_cholesky_ex_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _cholesky_ex_args(args)
    var L_out = v_tensor(args[unsafe_offset=3])
    var info_out = v_tensor(args[unsafe_offset=4])
    _check_out(L_out, a[0].stype, a[0])
    _check_out(info_out, ST_INT32, a[0])
    var r = _cholesky(a[0], a[1])
    _store(L_out, r[0].t)
    _store(info_out, r[1].t)
    if a[2]:
        _check_errors(info_out, "linalg.cholesky_ex", a[0].rank == 2)
    ret_ref(rets, 0, L_out)
    ret_ref(rets, 1, info_out)


def _cholesky_legacy(A: T, upper: Bool) raises -> Owned:
    if A.numel == 0:
        return own(_new_c(_dims(A), A.stype, A.device))
    _check_square(A, "cholesky")
    _check_compute(A, "cholesky")
    var r = _cholesky(A, upper)
    _check_errors(r[1].t, "cholesky", A.rank == 2)
    return own(r[0].take())


# aten::cholesky(Tensor self, bool upper=False) -> Tensor
def op_cholesky(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var upper = v_bool_or(args[unsafe_offset=1], False)
    var L = _cholesky_legacy(A, upper)
    ret_owned(rets, 0, L)


def _check_linalg_out(fname: String, dest: T, input: T, name: String) raises:
    """`checkSameDevice` + `checkLinalgCompatibleDtype` (canCast)."""
    if not _same_device(dest, input):
        raise Error(
            fname,
            ": Expected ",
            name,
            " and input tensors to be on the same device, but got ",
            name,
            " on ",
            device_str(dest),
            " and input on ",
            device_str(input),
        )
    var ok = _is_float(dest.stype) or _is_complex(dest.stype)
    if _is_complex(input.stype) and not _is_complex(dest.stype):
        ok = False
    if not ok:
        raise Error(
            fname,
            ": Expected ",
            name,
            " to be safely castable from ",
            _st_name(input.stype),
            " dtype, but got ",
            name,
            " with dtype ",
            _st_name(dest.stype),
        )


def _store_cast(mut dst: T, src: T) raises:
    """`resize_output(out, src.sizes()); out.copy_(src)` -- a copy_ casts."""
    if dst.stype == src.stype:
        _store(dst, src)
        return
    var cast = own(cast_to(src, dst.stype))
    _store(dst, cast.t)


# aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
def op_cholesky_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var upper = v_bool_or(args[unsafe_offset=1], False)
    var out = v_tensor(args[unsafe_offset=2])
    _check_linalg_out("cholesky", out, A, "result")
    var L = _cholesky_legacy(A, upper)
    _store_cast(out, L.t)
    ret_ref(rets, 0, out)


def _cholesky_inverse(A: T, upper: Bool) raises -> Owned:
    """(L L^T)^-1 (or (U^T U)^-1) by two triangular solves on the
    identity."""
    var X = own(_new_f(_dims(A), A.stype, A.device))
    if X.t.numel == 0:
        return X^
    _eye_into(X.t)
    var Af = _flat(A)
    # lower: L Y = I, then L^T X = Y; upper: U^T Y = I, then U X = Y.
    _k_trsm(Af.t, X.t, upper, upper, False)
    _k_trsm(Af.t, X.t, upper, not upper, False)
    return X^


# aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
def op_cholesky_inverse(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var upper = v_bool_or(args[unsafe_offset=1], False)
    _check_square(A, "cholesky_inverse")
    _check_compute(A, "cholesky_inverse")
    var X = _cholesky_inverse(A, upper)
    ret_owned(rets, 0, X)


# aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
def op_cholesky_inverse_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var upper = v_bool_or(args[unsafe_offset=1], False)
    var out = v_tensor(args[unsafe_offset=2])
    _check_square(A, "cholesky_inverse")
    _check_linalg_out("cholesky_inverse", out, A, "result")
    _check_compute(A, "cholesky_inverse")
    var X = _cholesky_inverse(A, upper)
    _store_cast(out, X.t)
    ret_ref(rets, 0, out)


# aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
def op_cholesky_solve_helper(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var B = v_tensor(args[unsafe_offset=0])
    var A = v_tensor(args[unsafe_offset=1])
    var upper = v_bool(args[unsafe_offset=2])
    _linear_solve_check(B, A, "cholesky_solve")
    _check_compute(A, "cholesky_solve")
    if B.stype != A.stype:
        raise Error(
            "Expected b and A to have the same dtype, but found b of type ",
            _st_name(B.stype),
            " and A of type ",
            _st_name(A.stype),
            " instead.",
        )
    var X = own(_clone_f(B))
    var Af = own(_clone_f(A))
    # lower: L L^T X = B; upper: U^T U X = B.
    _k_trsm(Af.t, X.t, upper, upper, False)
    _k_trsm(Af.t, X.t, upper, not upper, False)
    _ = Af^
    ret_owned(rets, 0, X)


# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------


def _lu_factor(A: T, pivot: Bool) raises -> Tuple[Owned, Owned, Owned]:
    """(LU, pivots, info) as `linalg_lu_factor_ex` returns them."""
    if A.rank < 2:
        raise Error(
            (
                "torch.lu_factor: Expected tensor with 2 or more dimensions."
                " Got size: "
            ),
            _sizes_str(_dims(A)),
            " instead",
        )
    _check_compute(A, "linalg.lu_factor")
    var m = A.dim(-2)
    var n = A.dim(-1)
    var LU = own(_clone_f(A))
    var piv = own(_new_c(_with1(_batch_dims(A), min(m, n)), ST_INT32, A.device))
    var info = own(_new_c(_batch_dims(A), ST_INT32, A.device))
    if A.numel == 0:
        fill_value(info.t, 0.0)
        return (LU^, piv^, info^)
    _k_getrf(LU.t, piv.t, info.t, pivot)
    return (LU^, piv^, info^)


def _lu_factor_args(args: Values) raises -> Tuple[T, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var pivot = v_bool_or(args[unsafe_offset=1], True)
    var check_errors = v_bool_or(args[unsafe_offset=2], False)
    return (A^, pivot, check_errors)


# aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
def op_linalg_lu_factor_ex(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _lu_factor_args(args)
    var r = _lu_factor(a[0], a[1])
    if a[2]:
        _check_errors(r[2].t, "torch.linalg.lu_factor_ex", a[0].rank == 2)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)
def op_linalg_lu_factor_ex_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _lu_factor_args(args)
    var LU_out = v_tensor(args[unsafe_offset=3])
    var piv_out = v_tensor(args[unsafe_offset=4])
    var info_out = v_tensor(args[unsafe_offset=5])
    _check_out(LU_out, a[0].stype, a[0])
    _check_out(piv_out, ST_INT32, a[0])
    _check_out(info_out, ST_INT32, a[0])
    var r = _lu_factor(a[0], a[1])
    _store(LU_out, r[0].t)
    _store(piv_out, r[1].t)
    _store(info_out, r[2].t)
    if a[2]:
        _check_errors(info_out, "torch.linalg.lu_factor_ex", a[0].rank == 2)
    ret_ref(rets, 0, LU_out)
    ret_ref(rets, 1, piv_out)
    ret_ref(rets, 2, info_out)


def _perm_matrix(
    piv: T, m: Int, batch_dims: List[Int], like: T
) raises -> Owned:
    """P with A = P L U from getrf pivots: the row interchanges applied to
    the identity give P^T."""
    var Pt = own(
        _new_f(_with(batch_dims.copy(), m, m), like.stype, like.device)
    )
    if Pt.t.numel == 0:
        return own(
            _new_c(_with(batch_dims.copy(), m, m), like.stype, like.device)
        )
    _eye_into(Pt.t)
    var pc = own_if_new(_contig_i32(piv), piv)
    _k_laswp(Pt.t, pc.t, True)
    var P = own(_new_c(_with(batch_dims.copy(), m, m), like.stype, like.device))
    var v = own(_mT(Pt.t))
    copy_strided_into(P.t, v.t)
    return P^


def _contig_i32(t: T) raises -> T:
    if t.stype != ST_INT32:
        return cast_to(t, ST_INT32)
    if t.contig:
        return t.copy()
    return _clone_c(t)


def _unpack_lu(LU: T) raises -> Tuple[Owned, Owned]:
    """(L, U) from a packed LU: L (m x k) unit lower, U (k x n) upper,
    contiguous."""
    var m = LU.dim(-2)
    var n = LU.dim(-1)
    var k = min(m, n)
    var lsrc = own(_narrow(LU, -1, 0, k))
    var L = _tri(lsrc.t, False, -1)
    var dl = own(_diagonal(L.t))
    fill_value(dl.t, 1.0)
    var usrc = own(_narrow(LU, -2, 0, k))
    var U = _tri(usrc.t, True, 0)
    return (L^, U^)


def _lu_unpack(
    LU: T, piv: T, unpack_data: Bool, unpack_pivots: Bool
) raises -> Tuple[Owned, Owned, Owned]:
    if LU.rank < 2:
        raise Error(
            (
                "torch.lu_unpack: Expected tensor with 2 or more dimensions."
                " Got size: "
            ),
            _sizes_str(_dims(LU)),
            " instead",
        )
    var m = LU.dim(-2)
    var n = LU.dim(-1)
    var k = min(m, n)
    if unpack_pivots:
        if piv.stype != ST_INT32:
            raise Error(
                "torch.lu_unpack: LU_pivots is expected to be a contiguous"
                " tensor of torch.int32 dtype.\nNote: this function is"
                " intended to be used with the output produced by"
                " torch.linalg.lu_factor"
            )
        var expected = _with1(_batch_dims(LU), k)
        var got = _dims(piv)
        var same = len(expected) == len(got)
        if same:
            for i in range(len(got)):
                if expected[i] != got[i]:
                    same = False
        if not same:
            raise Error(
                "Expected LU_pivots to have shape ",
                _sizes_str(expected),
                " but got ",
                _sizes_str(got),
                " instead.",
            )
    if unpack_pivots:
        _all_on(LU, piv)
    _check_compute(LU, "torch.lu_unpack")
    var P: Owned
    if unpack_pivots:
        P = _perm_matrix(piv, m, _batch_dims(LU), LU)
    else:
        P = own(_empty0(LU.stype, LU.device))
    var L: Owned
    var U: Owned
    if unpack_data:
        if LU.numel == 0:
            L = own(_new_c(_with(_batch_dims(LU), m, k), LU.stype, LU.device))
            U = own(_new_c(_with(_batch_dims(LU), k, n), LU.stype, LU.device))
        else:
            var r = _unpack_lu(LU)
            L = own(r[0].take())
            U = own(r[1].take())
    else:
        L = own(_empty0(LU.stype, LU.device))
        U = own(_empty0(LU.stype, LU.device))
    return (P^, L^, U^)


# aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
def op_lu_unpack(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var LU = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var ud = v_bool_or(args[unsafe_offset=2], True)
    var up = v_bool_or(args[unsafe_offset=3], True)
    var r = _lu_unpack(LU, piv, ud, up)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
def op_lu_unpack_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var LU = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var ud = v_bool_or(args[unsafe_offset=2], True)
    var up = v_bool_or(args[unsafe_offset=3], True)
    var P_out = v_tensor(args[unsafe_offset=4])
    var L_out = v_tensor(args[unsafe_offset=5])
    var U_out = v_tensor(args[unsafe_offset=6])
    _check_out(P_out, LU.stype, LU)
    _check_out(L_out, LU.stype, LU)
    _check_out(U_out, LU.stype, LU)
    var r = _lu_unpack(LU, piv, ud, up)
    _store(P_out, r[0].t)
    _store(L_out, r[1].t)
    _store(U_out, r[2].t)
    ret_ref(rets, 0, P_out)
    ret_ref(rets, 1, L_out)
    ret_ref(rets, 2, U_out)


def _linalg_lu(A: T, pivot: Bool) raises -> Tuple[Owned, Owned, Owned]:
    if A.rank < 2:
        raise Error(
            "linalg.lu: Expected tensor with 2 or more dimensions. Got size: ",
            _sizes_str(_dims(A)),
            " instead",
        )
    var f = _lu_factor(A, pivot)
    var r = _lu_unpack(f[0].t, f[1].t, True, pivot)
    return r^


# aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
def op_linalg_lu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var pivot = v_bool_or(args[unsafe_offset=1], True)
    var r = _linalg_lu(A, pivot)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
def op_linalg_lu_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var pivot = v_bool_or(args[unsafe_offset=1], True)
    var P_out = v_tensor(args[unsafe_offset=2])
    var L_out = v_tensor(args[unsafe_offset=3])
    var U_out = v_tensor(args[unsafe_offset=4])
    _check_out(P_out, A.stype, A)
    _check_out(L_out, A.stype, A)
    _check_out(U_out, A.stype, A)
    var r = _linalg_lu(A, pivot)
    _store(P_out, r[0].t)
    _store(L_out, r[1].t)
    _store(U_out, r[2].t)
    ret_ref(rets, 0, P_out)
    ret_ref(rets, 1, L_out)
    ret_ref(rets, 2, U_out)


def _lu_solve_inplace(LU: T, piv: T, X: T, trans: Bool) raises:
    """X <- A^-1 X (A^-T X when `trans`) for A = P L U, X column-major with
    LU's (flattened) batch."""
    if not trans:
        _k_laswp(X, piv, True)
        _k_trsm(LU, X, False, False, True)
        _k_trsm(LU, X, True, False, False)
    else:
        _k_trsm(LU, X, True, True, False)
        _k_trsm(LU, X, False, True, True)
        _k_laswp(X, piv, False)


def _lu_solve(LU: T, piv: T, B: T, left: Bool, adjoint: Bool) raises -> Owned:
    """`linalg_lu_solve`: the result, batched column-major when `left`,
    row-major otherwise (its meta's layout)."""
    _check_float(LU, "torch.linalg.lu_solve")
    if LU.stype != B.stype:
        raise Error(
            (
                "linalg.lu_solve: Expected LU and B to have the same dtype, but"
                " found LU of type "
            ),
            _st_name(LU.stype),
            " and B of type ",
            _st_name(B.stype),
            " instead",
        )
    if piv.stype != ST_INT32:
        raise Error(
            "linalg.lu_solve: pivots should be a Tensor of scalar type"
            " torch.int32"
        )
    _check_square(LU, "torch.linalg.lu_solve")
    _check_matrix(B, "linalg.lu_solve", "B")
    var n = LU.dim(-1)
    var ok = LU.dim(-2) == B.dim(-2) if left else LU.dim(-1) == B.dim(-1)
    if not ok:
        raise Error(
            "linalg.lu_solve: Incompatible shapes of A and B for the equation ",
            "AX = B" if left else "XA = B",
            " (",
            LU.dim(-2),
            "x",
            LU.dim(-1),
            " and ",
            B.dim(-2),
            "x",
            B.dim(-1),
            ")",
        )
    if piv.rank == 0 or piv.dim(-1) != n:
        raise Error(
            "linalg.lu_solve: Number of pivots per batch should be same as the"
            " dimension of the matrix"
        )
    var lub = _with1(_batch_dims(LU), n)
    var pd = _dims(piv)
    var same = len(lub) == len(pd)
    if same:
        for i in range(len(pd)):
            if lub[i] != pd[i]:
                same = False
    if not same:
        raise Error(
            (
                "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be"
                " the same, but got pivots with shape "
            ),
            _sizes_str(pd),
            " instead",
        )
    _all_on(LU, piv)
    _all_on(LU, B)
    var batch = _broadcast(_batch_dims(B), _batch_dims(LU))
    var rows = B.dim(-2)
    var cols = B.dim(-1)
    # Work on X = B (left) or B^T (right: A^T X^T = B^T), column-major.
    var wr = rows if left else cols
    var wc = cols if left else rows
    var X = own(_new_f(_with(batch.copy(), wr, wc), B.stype, B.device))
    if X.t.numel == 0:
        if left:
            return X^
        return own(_new_c(_with(batch.copy(), rows, cols), B.stype, B.device))
    var Bsrc = own(_mT(B)) if not left else own_if_new(B.copy(), B)
    var Bx = own(_expand(Bsrc.t, _with(batch.copy(), wr, wc)))
    copy_strided_into(X.t, Bx.t)
    var LUx = own(_expand(LU, _with(batch.copy(), n, n)))
    var LUf = _flat(LUx.t)
    var px = own(_expand(piv, _with1(batch.copy(), n)))
    var pc = own(_clone_c(px.t))
    var trans = adjoint if left else not adjoint
    _lu_solve_inplace(LUf.t, pc.t, X.t, trans)
    _ = LUf^
    _ = pc^
    if left:
        return X^
    return own(_mT(X.t))


# aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor
def op_linalg_lu_solve(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var LU = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var B = v_tensor(args[unsafe_offset=2])
    var left = v_bool_or(args[unsafe_offset=3], True)
    var adjoint = v_bool_or(args[unsafe_offset=4], False)
    var X = _lu_solve(LU, piv, B, left, adjoint)
    ret_owned(rets, 0, X)


# aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!)
def op_linalg_lu_solve_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var LU = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var B = v_tensor(args[unsafe_offset=2])
    var left = v_bool_or(args[unsafe_offset=3], True)
    var adjoint = v_bool_or(args[unsafe_offset=4], False)
    var out = v_tensor(args[unsafe_offset=5])
    _check_out(out, B.stype, B)
    var X = _lu_solve(LU, piv, B, left, adjoint)
    _store(out, X.t)
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# inv, solve, det, slogdet
# ---------------------------------------------------------------------------


def _inv(A: T) raises -> Tuple[Owned, Owned]:
    _check_square(A, "linalg.inv")
    _check_float(A, "linalg.inv", False)
    var f = _lu_factor(A, True)
    var X = own(_new_f(_dims(A), A.stype, A.device))
    if X.t.numel > 0:
        _eye_into(X.t)
        _lu_solve_inplace(f[0].t, f[1].t, X.t, False)
    return (X^, own(f[2].take()))


# aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
def op_linalg_inv_ex(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var check_errors = v_bool_or(args[unsafe_offset=1], False)
    var r = _inv(A)
    if check_errors:
        _check_errors(r[1].t, "linalg.inv_ex", A.rank == 2)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
def op_linalg_inv_ex_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var check_errors = v_bool_or(args[unsafe_offset=1], False)
    var inv_out = v_tensor(args[unsafe_offset=2])
    var info_out = v_tensor(args[unsafe_offset=3])
    _check_out(inv_out, A.stype, A)
    _check_out(info_out, ST_INT32, A)
    var r = _inv(A)
    _store(inv_out, r[0].t)
    _store(info_out, r[1].t)
    if check_errors:
        _check_errors(info_out, "linalg.inv_ex", A.rank == 2)
    ret_ref(rets, 0, inv_out)
    ret_ref(rets, 1, info_out)


def _is_vector_rhs(A: T, B: T) -> Bool:
    """`linalg_solve_is_vector_rhs`: B.shape == A.shape[:-1] (or 1-D)."""
    if B.rank == 1:
        return True
    if A.rank - 1 != B.rank:
        return False
    for i in range(B.rank):
        if B.dim(i) != A.dim(i):
            return False
    return True


def _unsqueeze_last(t: T) raises -> T:
    var d = _dims(t)
    d.append(1)
    var strides = IndexList[MAX_RANK](0)
    for i in range(t.rank):
        strides[MAX_RANK - (t.rank + 1) + i] = t.stride(i)
    strides[MAX_RANK - 1] = 1
    return view_strided(t, _padded(d), strides, t.rank + 1, t.offset)


def _squeeze_last(t: T) raises -> T:
    var d = _dims(t)
    _ = d.pop()
    var strides = IndexList[MAX_RANK](0)
    for i in range(t.rank - 1):
        strides[MAX_RANK - (t.rank - 1) + i] = t.stride(i)
    return view_strided(t, _padded(d), strides, t.rank - 1, t.offset)


def _solve_ex(
    A: T, B: T, left: Bool, check_errors: Bool
) raises -> Tuple[Owned, Owned, Owned, Owned]:
    _check_float(A, "linalg.solve")
    if A.stype != B.stype:
        raise Error(
            (
                "linalg.solve: Expected A and B to have the same dtype, but"
                " found A of type "
            ),
            _st_name(A.stype),
            " and B of type ",
            _st_name(B.stype),
            " instead",
        )
    var vector_case = _is_vector_rhs(A, B)
    var B_ = own(_unsqueeze_last(B)) if vector_case else own_if_new(B.copy(), B)
    _check_square(A, "linalg.solve")
    _check_matrix(B_.t, "linalg.solve", "B")
    var ok = A.dim(-2) == B_.t.dim(-2) if left else A.dim(-1) == B_.t.dim(-1)
    if not ok:
        raise Error(
            "linalg.solve: Incompatible shapes of A and B for the equation ",
            "AX = B" if left else "XA = B",
            " (",
            A.dim(-2),
            "x",
            A.dim(-1),
            " and ",
            B_.t.dim(-2),
            "x",
            B_.t.dim(-1),
            ")",
        )
    _ = _broadcast(_batch_dims(B_.t), _batch_dims(A))
    if not left and vector_case:
        raise Error(
            "linalg.solve: Vector broadcasting of the left hand side is not"
            " supported for left=False. In this case linalg.solve is"
            " equivalent to B / A.squeeze(-1)"
        )
    _all_on(A, B)
    var f = _lu_factor(A, True)
    if check_errors:
        _check_errors(f[2].t, "torch.linalg.solve_ex", A.rank == 2)
    var X = _lu_solve(f[0].t, f[1].t, B_.t, left, False)
    if vector_case:
        X = own(_squeeze_last(X.t))
    return (X^, own(f[0].take()), own(f[1].take()), own(f[2].take()))


def _solve_args(args: Values) raises -> Tuple[T, T, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var B = v_tensor(args[unsafe_offset=1])
    var left = v_bool_or(args[unsafe_offset=2], True)
    var check_errors = v_bool_or(args[unsafe_offset=3], False)
    return (A^, B^, left, check_errors)


# aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)
def op_linalg_solve_ex_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _solve_args(args)
    var r = _solve_ex(a[0], a[1], a[2], a[3])
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])
    ret_owned(rets, 3, r[3])


# aten::_linalg_solve_ex.result(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info) -> (...)
def op_linalg_solve_ex_out_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _solve_args(args)
    var res_out = v_tensor(args[unsafe_offset=4])
    var LU_out = v_tensor(args[unsafe_offset=5])
    var piv_out = v_tensor(args[unsafe_offset=6])
    var info_out = v_tensor(args[unsafe_offset=7])
    _check_out(res_out, a[1].stype, a[1])
    _check_out(LU_out, a[0].stype, a[0])
    _check_out(piv_out, ST_INT32, a[0])
    _check_out(info_out, ST_INT32, a[0])
    var r = _solve_ex(a[0], a[1], a[2], a[3])
    _store(res_out, r[0].t)
    _store(LU_out, r[1].t)
    _store(piv_out, r[2].t)
    _store(info_out, r[3].t)
    ret_ref(rets, 0, res_out)
    ret_ref(rets, 1, LU_out)
    ret_ref(rets, 2, piv_out)
    ret_ref(rets, 3, info_out)


def _det_lu(A: T, name: String) raises -> Tuple[Owned, Owned, Owned]:
    """`linalg_lu_factor_ex` of A^T when A is contiguous (det(A^T) =
    det(A)), of A otherwise -- what `_linalg_det_out` factors."""
    _check_square(A, name)
    if A.contig:
        var At = own(_mT(A))
        var f = _lu_factor(At.t, True)
        return f^
    return _lu_factor(A, True)


def _pivot_sign(piv: T, like: T) raises -> Owned:
    """det(P) per matrix, in `like`'s dtype (torch's `lu_det_P`)."""
    var out = own(_new_c(_batch_dims(like), like.stype, like.device))
    var batch = _batch_count(like)
    if batch == 0:
        return out^
    var l = List[Int]()
    l.append(piv.dim(-1))
    l.append(batch)
    _launch("PivSign", like.dtype, like.device, piv.ptr, out.t.ptr, 0, 0, 0, l)
    return out^


def _det(A: T) raises -> Tuple[Owned, Owned, Owned]:
    _check_square(A, "linalg.det")
    _check_float(A, "linalg.det")
    var f = _det_lu(A, "linalg.det")
    var sign = _pivot_sign(f[1].t, A)
    var d = own(_diagonal(f[0].t))
    var p = _reduce_last("aten::prod", d.t)
    var det = _binary("aten::mul", sign.t, p.t)
    return (det^, own(f[0].take()), own(f[1].take()))


# aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)
def op_linalg_det_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var r = _det(A)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots)
def op_linalg_det_out_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var det_out = v_tensor(args[unsafe_offset=1])
    var LU_out = v_tensor(args[unsafe_offset=2])
    var piv_out = v_tensor(args[unsafe_offset=3])
    _check_out(det_out, A.stype, A)
    _check_out(LU_out, A.stype, A)
    _check_out(piv_out, ST_INT32, A)
    var r = _det(A)
    _store(det_out, r[0].t)
    _store(LU_out, r[1].t)
    _store(piv_out, r[2].t)
    ret_ref(rets, 0, det_out)
    ret_ref(rets, 1, LU_out)
    ret_ref(rets, 2, piv_out)


def _slogdet(A: T) raises -> Tuple[Owned, Owned, Owned, Owned]:
    _check_square(A, "linalg.slogdet")
    _check_float(A, "linalg.slogdet", False)
    var f = _det_lu(A, "linalg.slogdet")
    var sign = own(_new_c(_batch_dims(A), A.stype, A.device))
    var logabs = own(_new_c(_batch_dims(A), A.stype, A.device))
    var batch = _batch_count(A)
    if batch > 0:
        var LU = f[0].t.copy()
        var l = List[Int]()
        l.append(A.dim(-1))
        l.append(LU.stride(-2))
        l.append(LU.stride(-1))
        l.append(_bstride(LU))
        l.append(batch)
        _launch(
            "Slogdet",
            A.dtype,
            A.device,
            LU.ptr,
            f[1].t.ptr,
            sign.t.ptr,
            logabs.t.ptr,
            0,
            l,
        )
    return (sign^, logabs^, own(f[0].take()), own(f[1].take()))


# aten::_linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
def op_linalg_slogdet_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var r = _slogdet(A)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])
    ret_owned(rets, 3, r[3])


# aten::_linalg_slogdet.sign(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots) -> (...)
def op_linalg_slogdet_out_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var s_out = v_tensor(args[unsafe_offset=1])
    var l_out = v_tensor(args[unsafe_offset=2])
    var LU_out = v_tensor(args[unsafe_offset=3])
    var piv_out = v_tensor(args[unsafe_offset=4])
    _check_out(s_out, A.stype, A)
    _check_out(l_out, A.stype, A)
    _check_out(LU_out, A.stype, A)
    _check_out(piv_out, ST_INT32, A)
    var r = _slogdet(A)
    _store(s_out, r[0].t)
    _store(l_out, r[1].t)
    _store(LU_out, r[2].t)
    _store(piv_out, r[3].t)
    ret_ref(rets, 0, s_out)
    ret_ref(rets, 1, l_out)
    ret_ref(rets, 2, LU_out)
    ret_ref(rets, 3, piv_out)


# ---------------------------------------------------------------------------
# Triangular solves
# ---------------------------------------------------------------------------


def _solve_triangular(
    A: T, B: T, upper: Bool, left: Bool, unit: Bool
) raises -> Owned:
    _check_square(A, "linalg.solve_triangular")
    _check_matrix(B, "linalg.solve_triangular", "B")
    var ok = A.dim(-2) == B.dim(-2) if left else A.dim(-1) == B.dim(-1)
    if not ok:
        raise Error(
            (
                "linalg.solve_triangular: Incompatible shapes of A and B for"
                " the equation "
            ),
            "AX = B" if left else "XA = B",
            " (",
            A.dim(-2),
            "x",
            A.dim(-1),
            " and ",
            B.dim(-2),
            "x",
            B.dim(-1),
            ")",
        )
    _all_on(A, B)
    _check_compute(A, "linalg.solve_triangular")
    var batch = _broadcast(_batch_dims(B), _batch_dims(A))
    var n = A.dim(-1)
    var X = own(
        _new_f(_with(batch.copy(), B.dim(-2), B.dim(-1)), A.stype, A.device)
    )
    if X.t.numel == 0:
        return X^
    var Bc = own(cast_to(B, A.stype)) if B.stype != A.stype else own_if_new(
        B.copy(), B
    )
    var Bx = own(_expand(Bc.t, _with(batch.copy(), B.dim(-2), B.dim(-1))))
    copy_strided_into(X.t, Bx.t)
    var Ax = own(_expand(A, _with(batch.copy(), n, n)))
    var Af = _flat(Ax.t)
    if left:
        _k_trsm(Af.t, X.t, upper, False, unit)
    else:
        # X A = B  <=>  A^T X^T = B^T
        var Xt = own(_mT(X.t))
        _k_trsm(Af.t, Xt.t, upper, True, unit)
    _ = Af^
    return X^


def _st_args(args: Values) raises -> Tuple[T, T, Bool, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var B = v_tensor(args[unsafe_offset=1])
    var upper = v_bool(args[unsafe_offset=2])
    var left = v_bool_or(args[unsafe_offset=3], True)
    var unit = v_bool_or(args[unsafe_offset=4], False)
    return (A^, B^, upper, left, unit)


# aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor
def op_linalg_solve_triangular(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _st_args(args)
    var X = _solve_triangular(a[0], a[1], a[2], a[3], a[4])
    ret_owned(rets, 0, X)


# aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!)
def op_linalg_solve_triangular_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _st_args(args)
    var out = v_tensor(args[unsafe_offset=5])
    _all_on(a[0], out)
    var X = _solve_triangular(a[0], a[1], a[2], a[3], a[4])
    _store_cast(out, X.t)
    ret_ref(rets, 0, out)


def _triangular_solve(
    B: T, A: T, upper: Bool, transpose: Bool, unit: Bool
) raises -> Tuple[Owned, Owned]:
    if B.rank < 2:
        raise Error(
            (
                "torch.triangular_solve: Expected b to have at least 2"
                " dimensions, but it has "
            ),
            B.rank,
            " dimensions instead",
        )
    if A.rank < 2:
        raise Error(
            (
                "torch.triangular_solve: Expected A to have at least 2"
                " dimensions, but it has "
            ),
            A.rank,
            " dimensions instead",
        )
    _linear_solve_check(B, A, "triangular_solve")
    _check_compute(A, "triangular_solve")
    var batch = _broadcast(_batch_dims(B), _batch_dims(A))
    var n = A.dim(-1)
    var X = own(
        _new_f(_with(batch.copy(), B.dim(-2), B.dim(-1)), B.stype, B.device)
    )
    var Ac = own(_new_f(_with(batch.copy(), n, n), A.stype, A.device))
    var Ax = own(_expand(A, _with(batch.copy(), n, n)))
    copy_strided_into(Ac.t, Ax.t)
    if X.t.numel > 0:
        var Bx = own(_expand(B, _with(batch.copy(), B.dim(-2), B.dim(-1))))
        copy_strided_into(X.t, Bx.t)
        _k_trsm(Ac.t, X.t, upper, transpose, unit)
    return (X^, Ac^)


def _linear_solve_check(B: T, A: T, name: String) raises:
    """`linearSolveCheckInputs(self=b, A, name)`."""
    if not _same_device(B, A):
        raise Error(
            "Expected b and A to be on the same device, but found b on ",
            device_str(B),
            " and A on ",
            device_str(A),
            " instead.",
        )
    if B.stype != A.stype:
        raise Error(
            "Expected b and A to have the same dtype, but found b of type ",
            _st_name(B.stype),
            " and A of type ",
            _st_name(A.stype),
            " instead.",
        )
    if A.dim(-1) != A.dim(-2):
        raise Error(
            "A must be batches of square matrices, but they are ",
            A.dim(-2),
            " by ",
            A.dim(-1),
            " matrices",
        )
    if A.dim(-1) != B.dim(-2):
        raise Error(
            "Incompatible matrix sizes for ",
            name,
            ": each A matrix is ",
            A.dim(-1),
            " by ",
            A.dim(-1),
            " but each b matrix is ",
            B.dim(-2),
            " by ",
            B.dim(-1),
        )


def _ts_args(args: Values) raises -> Tuple[T, T, Bool, Bool, Bool]:
    var B = v_tensor(args[unsafe_offset=0])
    var A = v_tensor(args[unsafe_offset=1])
    var upper = v_bool_or(args[unsafe_offset=2], True)
    var transpose = v_bool_or(args[unsafe_offset=3], False)
    var unit = v_bool_or(args[unsafe_offset=4], False)
    return (B^, A^, upper, transpose, unit)


# aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
def op_triangular_solve(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _ts_args(args)
    var r = _triangular_solve(a[0], a[1], a[2], a[3], a[4])
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
def op_triangular_solve_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _ts_args(args)
    var X_out = v_tensor(args[unsafe_offset=5])
    var M_out = v_tensor(args[unsafe_offset=6])
    _check_out(X_out, a[0].stype, a[0])
    _check_out(M_out, a[1].stype, a[1])
    var r = _triangular_solve(a[0], a[1], a[2], a[3], a[4])
    _store(X_out, r[0].t)
    _store(M_out, r[1].t)
    ret_ref(rets, 0, X_out)
    ret_ref(rets, 1, M_out)


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------


def _geqrf(A: T) raises -> Tuple[Owned, Owned]:
    if A.rank < 2:
        raise Error("torch.geqrf: input must have at least 2 dimensions.")
    _check_compute(A, "torch.geqrf")
    var QR = own(_clone_f(A))
    var tau = own(
        _new_c(
            _with1(_batch_dims(A), min(A.dim(-2), A.dim(-1))), A.stype, A.device
        )
    )
    _k_geqrf(QR.t, tau.t)
    return (QR^, tau^)


# aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
def op_geqrf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var r = _geqrf(A)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
def op_geqrf_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var a_out = v_tensor(args[unsafe_offset=1])
    var tau_out = v_tensor(args[unsafe_offset=2])
    if A.rank < 2:
        raise Error("torch.geqrf: input must have at least 2 dimensions.")
    _check_linalg_out("torch.geqrf", a_out, A, "a")
    _check_linalg_out("torch.geqrf", tau_out, A, "tau")
    var r = _geqrf(A)
    _store_cast(a_out, r[0].t)
    _store_cast(tau_out, r[1].t)
    ret_ref(rets, 0, a_out)
    ret_ref(rets, 1, tau_out)


def _apply_q(
    QR: T, tau: T, rows: Int, cols: Int, batch: List[Int]
) raises -> Owned:
    """The first `cols` columns of Q (rows x rows) from geqrf's reflectors:
    Q applied to the rows x cols identity, column-major."""
    var Q = own(_new_f(_with(batch.copy(), rows, cols), QR.stype, QR.device))
    if Q.t.numel == 0:
        return Q^
    _eye_into(Q.t)
    var tc = own_if_new(_contig_f(tau), tau)
    var Af = _flat(QR)
    _k_ormqr(Af.t, tc.t, Q.t, False)
    _ = Af^
    _ = tc^
    return Q^


def _contig_f(t: T) raises -> T:
    if t.contig:
        return t.copy()
    return _clone_c(t)


def _householder_product(input: T, tau: T) raises -> Owned:
    if input.rank < 2:
        raise Error(
            "torch.linalg.householder_product: input must have at least 2"
            " dimensions."
        )
    if input.dim(-2) < input.dim(-1):
        raise Error(
            "torch.linalg.householder_product: input.shape[-2] must be greater"
            " than or equal to input.shape[-1]"
        )
    if tau.rank == 0 or input.dim(-1) < tau.dim(-1):
        raise Error(
            "torch.linalg.householder_product: input.shape[-1] must be greater"
            " than or equal to tau.shape[-1]"
        )
    if input.rank - tau.rank != 1:
        raise Error(
            (
                "torch.linalg.householder_product: Expected tau to have one"
                " dimension less than input, but got tau.ndim equal to "
            ),
            tau.rank,
            " and input.ndim is equal to ",
            input.rank,
        )
    if input.rank > 2:
        var tb = List[Int]()
        var same = True
        for i in range(tau.rank - 1):
            tb.append(tau.dim(i))
            if tau.dim(i) != input.dim(i):
                same = False
        if not same:
            raise Error(
                (
                    "torch.linalg.householder_product: Expected batch"
                    " dimensions of tau to be equal to input.shape[:-2], but"
                    " got "
                ),
                _sizes_str(tb),
            )
    if tau.stype != input.stype:
        raise Error(
            "torch.linalg.householder_product: tau dtype ",
            _st_name(tau.stype),
            " does not match input dtype ",
            _st_name(input.stype),
        )
    if not _same_device(tau, input):
        raise Error(
            (
                "torch.linalg.householder_product: Expected tau and input"
                " tensors to be on the same device, but got tau on "
            ),
            device_str(tau),
            " and input on ",
            device_str(input),
        )
    _check_compute(input, "torch.linalg.householder_product")
    return _apply_q(
        input, tau, input.dim(-2), input.dim(-1), _batch_dims(input)
    )


# aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
def op_linalg_householder_product(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var tau = v_tensor(args[unsafe_offset=1])
    var Q = _householder_product(input, tau)
    ret_owned(rets, 0, Q)


# aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
def op_linalg_householder_product_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var tau = v_tensor(args[unsafe_offset=1])
    var out = v_tensor(args[unsafe_offset=2])
    _check_linalg_out("torch.linalg.householder_product", out, input, "result")
    var Q = _householder_product(input, tau)
    _store_cast(out, Q.t)
    ret_ref(rets, 0, out)


def _ormqr(
    input: T, tau: T, other: T, left: Bool, transpose: Bool, result_st: Int32
) raises -> Owned:
    if input.rank < 2:
        raise Error("torch.ormqr: input must have at least 2 dimensions.")
    if other.rank < 2:
        raise Error("torch.ormqr: other must have at least 2 dimensions.")
    var cond = -2 if left else -1
    if other.dim(cond) != input.dim(-2):
        raise Error(
            "torch.ormqr: other.shape[",
            cond,
            "] must be equal to input.shape[-2]",
        )
    if tau.rank == 0 or min(other.dim(cond), input.dim(-1)) != tau.dim(-1):
        raise Error(
            "torch.ormqr: tau.shape[-1] must be equal to min(other.shape[",
            cond,
            "], input.shape[-1])",
        )
    if input.rank - tau.rank != 1:
        raise Error(
            (
                "torch.ormqr: Expected tau to have one dimension less than"
                " input, but got tau.ndim equal to "
            ),
            tau.rank,
            " and input.ndim is equal to ",
            input.rank,
        )
    if input.rank != other.rank:
        raise Error(
            (
                "torch.ormqr: Expected other to have the same number of"
                " dimensions as input, but got other.ndim equal to "
            ),
            other.rank,
            " and input.ndim is equal to ",
            input.rank,
        )
    if input.rank > 2:
        var tb = List[Int]()
        var ob = List[Int]()
        var same_t = True
        var same_o = True
        for i in range(input.rank - 2):
            tb.append(tau.dim(i))
            ob.append(other.dim(i))
            if tau.dim(i) != input.dim(i):
                same_t = False
            if other.dim(i) != input.dim(i):
                same_o = False
        if not same_t:
            raise Error(
                (
                    "torch.ormqr: Expected batch dimensions of tau to be equal"
                    " to input.shape[:-2], but got "
                ),
                _sizes_str(tb),
            )
        if not same_o:
            raise Error(
                (
                    "torch.ormqr: Expected batch dimensions of other to be"
                    " equal to input.shape[:-2], but got "
                ),
                _sizes_str(ob),
            )
    if tau.stype != input.stype:
        raise Error(
            (
                "torch.ormqr: Expected input and tau to have the same dtype,"
                " but input has dtype"
            ),
            _st_name(input.stype),
            " and tau has dtype ",
            _st_name(tau.stype),
        )
    if other.stype != input.stype:
        raise Error(
            (
                "torch.ormqr: Expected input and other to have the same dtype,"
                " but input has dtype"
            ),
            _st_name(input.stype),
            " and other has dtype ",
            _st_name(other.stype),
        )
    if result_st != input.stype:
        raise Error(
            (
                "torch.ormqr: Expected input and result to have the same dtype,"
                " but input has dtype"
            ),
            _st_name(input.stype),
            " and result has dtype ",
            _st_name(result_st),
        )
    if not _same_device(tau, input):
        raise Error(
            (
                "torch.ormqr: Expected tau and input tensors to be on the same"
                " device, but got tau on "
            ),
            device_str(tau),
            " and input on ",
            device_str(input),
        )
    if not _same_device(other, input):
        raise Error(
            (
                "torch.ormqr: Expected other and input tensors to be on the"
                " same device, but got other on "
            ),
            device_str(other),
            " and input on ",
            device_str(input),
        )
    _check_compute(input, "torch.ormqr")
    var C = own(_clone_f(other))
    if C.t.numel == 0:
        return C^
    var tc = own_if_new(_contig_f(tau), tau)
    var Af = _flat(input)
    if left:
        _k_ormqr(Af.t, tc.t, C.t, transpose)
    else:
        # C Q = (Q^T C^T)^T
        var Ct = own(_mT(C.t))
        _k_ormqr(Af.t, tc.t, Ct.t, not transpose)
    _ = Af^
    _ = tc^
    return C^


# aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
def op_ormqr(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var tau = v_tensor(args[unsafe_offset=1])
    var other = v_tensor(args[unsafe_offset=2])
    var left = v_bool_or(args[unsafe_offset=3], True)
    var transpose = v_bool_or(args[unsafe_offset=4], False)
    var C = _ormqr(input, tau, other, left, transpose, input.stype)
    ret_owned(rets, 0, C)


# aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
def op_ormqr_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var tau = v_tensor(args[unsafe_offset=1])
    var other = v_tensor(args[unsafe_offset=2])
    var left = v_bool_or(args[unsafe_offset=3], True)
    var transpose = v_bool_or(args[unsafe_offset=4], False)
    var out = v_tensor(args[unsafe_offset=5])
    var C = _ormqr(input, tau, other, left, transpose, out.stype)
    if not _same_device(out, input):
        raise Error(
            (
                "torch.ormqr: Expected result and input tensors to be on the"
                " same device, but got result on "
            ),
            device_str(out),
            " and input on ",
            device_str(input),
        )
    _store(out, C.t)
    ret_ref(rets, 0, out)


def _qr(A: T, mode: String) raises -> Tuple[Owned, Owned]:
    _check_matrix(A, "linalg.qr")
    _check_float(A, "linalg.qr")
    var compute_q: Bool
    var reduced: Bool
    if mode == "reduced":
        compute_q = True
        reduced = True
    elif mode == "complete":
        compute_q = True
        reduced = False
    elif mode == "r":
        compute_q = False
        reduced = True
    else:
        raise Error(
            "qr received unrecognized mode '",
            mode,
            "' but expected one of 'reduced' (default), 'r', or 'complete'",
        )
    var m = A.dim(-2)
    var n = A.dim(-1)
    var k = min(m, n)
    var batch = _batch_dims(A)
    var g = _geqrf(A)
    var r_rows = k if (reduced or not compute_q) else m
    var R = own(_new_f(_with(batch.copy(), r_rows, n), A.stype, A.device))
    if R.t.numel > 0:
        var src = own(_narrow(g[0].t, -2, 0, r_rows))
        var tr = _tri(src.t, True, 0)
        copy_strided_into(R.t, tr.t)
    var Q: Owned
    if compute_q:
        Q = _apply_q(g[0].t, g[1].t, m, k if reduced else m, batch)
    else:
        Q = own(_empty0(A.stype, A.device))
    return (Q^, R^)


# aten::linalg_qr(Tensor A, str mode='reduced') -> (Tensor Q, Tensor R)
def op_linalg_qr(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var mode = String("reduced")
    if not v_is_none(args[unsafe_offset=1]):
        mode = v_string(args[unsafe_offset=1])
    var r = _qr(A, mode)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::linalg_qr.out(Tensor A, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
def op_linalg_qr_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var mode = String("reduced")
    if not v_is_none(args[unsafe_offset=1]):
        mode = v_string(args[unsafe_offset=1])
    var Q_out = v_tensor(args[unsafe_offset=2])
    var R_out = v_tensor(args[unsafe_offset=3])
    _check_out(Q_out, A.stype, A)
    _check_out(R_out, A.stype, A)
    var r = _qr(A, mode)
    _store(Q_out, r[0].t)
    _store(R_out, r[1].t)
    ret_ref(rets, 0, Q_out)
    ret_ref(rets, 1, R_out)


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition
# ---------------------------------------------------------------------------


def _eigh(A: T, uplo: String, compute_v: Bool) raises -> Tuple[Owned, Owned]:
    _check_square(A, "linalg.eigh")
    var u = uplo.upper()
    if uplo.byte_length() != 1 or (u != "U" and u != "L"):
        raise Error("Expected UPLO argument to be 'L' or 'U', but got ", uplo)
    _check_compute(A, "linalg.eigh")
    var n = A.dim(-1)
    var batch = _batch_dims(A)
    var W = own(_new_c(_with1(batch.copy(), n), A.stype, A.device))
    var V: Owned
    if compute_v:
        V = own(_new_f(_dims(A), A.stype, A.device))
    else:
        V = own(_empty0(A.stype, A.device))
    var nb = _batch_count(A)
    if A.numel == 0:
        return (W^, V^)
    var Af = _flat(A)
    var work = own(_new_c(_with(batch.copy(), n, n), A.stype, A.device))
    var vw = own(_new_c(_with(batch.copy(), n, n), A.stype, A.device))
    var ws = own(_new_c(_with1(batch.copy(), 2 * n + 2), A.stype, A.device))
    var info = own(_new_c(batch.copy(), ST_INT32, A.device))
    var l = List[Int]()
    l.append(V.t.ptr if compute_v else vw.t.ptr)
    l.append(n)
    l.append(nb)
    l.append(1 if compute_v else 0)
    l.append(Af.t.stride(-2))
    l.append(Af.t.stride(-1))
    l.append(_bstride(Af.t))
    l.append(1 if u == "L" else 0)
    l.append(info.t.ptr)
    _launch(
        "Syevj",
        A.dtype,
        A.device,
        Af.t.ptr,
        work.t.ptr,
        vw.t.ptr,
        ws.t.ptr,
        W.t.ptr,
        l,
    )
    _ = Af^
    _ = work^
    _ = vw^
    _ = ws^
    _check_errors(info.t, "linalg.eigh", A.rank == 2)
    return (W^, V^)


def _uplo_arg(v: Value) raises -> String:
    if v_is_none(v):
        return String("L")
    return v_string(v)


# aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)
def op_linalg_eigh_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var uplo = _uplo_arg(args[unsafe_offset=1])
    var compute_v = v_bool_or(args[unsafe_offset=2], True)
    var r = _eigh(A, uplo, compute_v)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (...)
def op_linalg_eigh_out_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var uplo = _uplo_arg(args[unsafe_offset=1])
    var compute_v = v_bool_or(args[unsafe_offset=2], True)
    var w_out = v_tensor(args[unsafe_offset=3])
    var v_out = v_tensor(args[unsafe_offset=4])
    _check_out(w_out, A.stype, A)
    _check_out(v_out, A.stype, A)
    var r = _eigh(A, uplo, compute_v)
    _store(w_out, r[0].t)
    _store(v_out, r[1].t)
    ret_ref(rets, 0, w_out)
    ret_ref(rets, 1, v_out)


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------


def _orthonormal_completion(Ut: T, cols: Int) raises -> Owned:
    """An orthonormal `rows x cols` basis whose first k columns are Ut's
    (rows x k, orthonormal columns or zero ones): Householder QR of Ut,
    then Q's columns flipped to R's diagonal signs. A zero column of Ut
    becomes a unit vector orthogonal to the others."""
    var rows = Ut.dim(-2)
    var k = Ut.dim(-1)
    var batch = _batch_dims(Ut)
    var g = _geqrf(Ut)
    var Q = _apply_q(g[0].t, g[1].t, rows, cols, batch)
    if k == 0 or Q.t.numel == 0:
        return Q^
    var d = own(_diagonal(g[0].t))
    var sg = _unary("aten::sign", d.t)
    # sign(0) -> +1: s - |s| + 1
    var ab = _unary("aten::abs", sg.t)
    var c = _Op("aten::sub", "Tensor")
    c.t(sg.t)
    c.t(ab.t)
    c.s(1.0)
    var df = c.one()
    _ = sg^  # the records above hold raw handles: keep them alive
    _ = ab^
    var a = _Op("aten::add", "Scalar")
    a.t(df.t)
    a.s(1.0)
    a.s(1.0)
    var fix = a.one()
    _ = df^
    var fv = own(_unsqueeze_at_minus2(fix.t))
    var Qk = own(_narrow(Q.t, -1, 0, k))
    var m = _Op("aten::mul_", "Tensor")
    m.t(Qk.t)
    m.t(fv.t)
    _ = m.run(1)
    _ = Qk^
    _ = fv^
    _ = fix^
    return Q^


def _unsqueeze_at_minus2(t: T) raises -> T:
    """(*, k) -> (*, 1, k) view."""
    var d = _dims(t)
    var last = d.pop()
    d.append(1)
    d.append(last)
    var strides = IndexList[MAX_RANK](0)
    for i in range(t.rank - 1):
        strides[MAX_RANK - (t.rank + 1) + i] = t.stride(i)
    strides[MAX_RANK - 2] = 0
    strides[MAX_RANK - 1] = t.stride(-1)
    return view_strided(t, _padded(d), strides, t.rank + 1, t.offset)


def _svd(
    A: T, full_matrices: Bool, compute_uv: Bool, has_driver: Bool
) raises -> Tuple[Owned, Owned, Owned]:
    _check_matrix(A, "linalg.svd")
    _check_float(A, "linalg.svd")
    if has_driver:
        raise Error(
            "torch.linalg.svd: keyword argument `driver=` is only supported on"
            " CUDA inputs with cuSOLVER backend."
        )
    var m = A.dim(-2)
    var n = A.dim(-1)
    var k = min(m, n)
    var batch = _batch_dims(A)
    var S = own(_new_c(_with1(batch.copy(), k), A.stype, A.device))
    var U = own(_empty0(A.stype, A.device))
    var Vh = own(_empty0(A.stype, A.device))
    if A.numel == 0:
        if compute_uv:
            U = own(
                _new_f(
                    _with(batch.copy(), m, m if full_matrices else k),
                    A.stype,
                    A.device,
                )
            )
            Vh = own(
                _new_f(
                    _with(batch.copy(), n if full_matrices else k, n),
                    A.stype,
                    A.device,
                )
            )
            if full_matrices:
                if U.t.numel != 0:
                    _eye_into(U.t)
                if Vh.t.numel != 0:
                    _eye_into(Vh.t)
        return (U^, S^, Vh^)
    # One-sided Jacobi on the tall orientation X = A (m >= n) or A^T.
    var tall = m >= n
    var mm = max(m, n)
    var X = own(_mT(A)) if not tall else own_if_new(A.copy(), A)
    var Xf = _flat(X.t)
    var nb = _batch_count(A)
    var u = own(_new_c(_with(batch.copy(), k, mm), A.stype, A.device))
    var uo = own(_new_c(_with(batch.copy(), k, mm), A.stype, A.device))
    var vw = own(_new_c(_with(batch.copy(), k, k), A.stype, A.device))
    var vo = own(_new_c(_with(batch.copy(), k, k), A.stype, A.device))
    var ws = own(_new_c(_with1(batch.copy(), 3 * k + 2), A.stype, A.device))
    var info = own(_new_c(batch.copy(), ST_INT32, A.device))
    var l = List[Int]()
    l.append(uo.t.ptr)
    l.append(vo.t.ptr)
    l.append(mm)
    l.append(k)
    l.append(nb)
    l.append(1 if compute_uv else 0)
    l.append(Xf.t.stride(-2))
    l.append(Xf.t.stride(-1))
    l.append(_bstride(Xf.t))
    l.append(info.t.ptr)
    _launch(
        "Gesvdj",
        A.dtype,
        A.device,
        Xf.t.ptr,
        u.t.ptr,
        vw.t.ptr,
        ws.t.ptr,
        S.t.ptr,
        l,
    )
    _ = Xf^
    _ = u^
    _ = vw^
    _ = ws^
    _check_errors(info.t, "linalg.svd", A.rank == 2)
    if not compute_uv:
        return (U^, S^, Vh^)
    # uo / vo hold, per matrix, column-major mm x k and k x k: view them so.
    var Ut = own(_cm_view(uo.t, batch, mm, k))
    var V = own(_cm_view(vo.t, batch, k, k))
    var full_cols = mm if full_matrices else k
    var Uc = _orthonormal_completion(Ut.t, full_cols)
    if tall:
        # A = U S V^T
        U = own(_new_f(_with(batch.copy(), m, full_cols), A.stype, A.device))
        copy_strided_into(U.t, Uc.t)
        Vh = own(_new_f(_with(batch.copy(), k, n), A.stype, A.device))
        var Vt = own(_mT(V.t))
        copy_strided_into(Vh.t, Vt.t)
    else:
        # A^T = U' S V'^T  =>  A = V' S U'^T
        U = own(_new_f(_with(batch.copy(), m, k), A.stype, A.device))
        copy_strided_into(U.t, V.t)
        Vh = own(_new_f(_with(batch.copy(), full_cols, n), A.stype, A.device))
        var Ut2 = own(_mT(Uc.t))
        copy_strided_into(Vh.t, Ut2.t)
    _ = uo^
    _ = vo^
    return (U^, S^, Vh^)


def _cm_view(t: T, batch: List[Int], rows: Int, cols: Int) raises -> T:
    """A dense (*, cols, rows) row-major buffer seen as (*, rows, cols)
    column-major matrices."""
    var d = _with(batch.copy(), rows, cols)
    var strides = contiguous_strides(_padded(d), len(d))
    strides[MAX_RANK - 2] = 1
    strides[MAX_RANK - 1] = rows
    return view_strided(t, _padded(d), strides, len(d), t.offset)


def _svd_args(args: Values) raises -> Tuple[T, Bool, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var full = v_bool_or(args[unsafe_offset=1], False)
    var compute_uv = v_bool_or(args[unsafe_offset=2], True)
    var has_driver = not v_is_none(args[unsafe_offset=3])
    return (A^, full, compute_uv, has_driver)


# aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
def op_linalg_svd_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = _svd_args(args)
    var r = _svd(a[0], a[1], a[2], a[3])
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (...)
def op_linalg_svd_out_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _svd_args(args)
    var U_out = v_tensor(args[unsafe_offset=4])
    var S_out = v_tensor(args[unsafe_offset=5])
    var Vh_out = v_tensor(args[unsafe_offset=6])
    _check_out(U_out, a[0].stype, a[0])
    _check_out(S_out, a[0].stype, a[0])
    _check_out(Vh_out, a[0].stype, a[0])
    var r = _svd(a[0], a[1], a[2], a[3])
    _store(U_out, r[0].t)
    _store(S_out, r[1].t)
    _store(Vh_out, r[2].t)
    ret_ref(rets, 0, U_out)
    ret_ref(rets, 1, S_out)
    ret_ref(rets, 2, Vh_out)


# ---------------------------------------------------------------------------
# LDL^T (Bunch-Kaufman)
# ---------------------------------------------------------------------------


def _ldl_factor(A: T, hermitian: Bool) raises -> Tuple[Owned, Owned, Owned]:
    _check_square(A, "torch.linalg.ldl_factor_ex")
    _check_float(A, "torch.linalg.ldl_factor_ex")
    var n = A.dim(-1)
    var batch = _batch_dims(A)
    var LD = own(_new_f(_dims(A), A.stype, A.device))
    var piv = own(_new_c(_with1(batch.copy(), n), ST_INT32, A.device))
    var info = own(_new_c(batch.copy(), ST_INT32, A.device))
    if A.numel == 0:
        fill_value(info.t, 0.0)
        return (LD^, piv^, info^)
    # torch's tril_out(LD, A): the upper triangle of LD is zero.
    var tr = _tri(A, False, 0)
    copy_strided_into(LD.t, tr.t)
    var l = List[Int]()
    l.append(n)
    l.append(_batch_count(A))
    _launch(
        "Sytf2", A.dtype, A.device, LD.t.ptr, piv.t.ptr, info.t.ptr, 0, 0, l
    )
    _ = tr^
    return (LD^, piv^, info^)


def _ldl_args(args: Values) raises -> Tuple[T, Bool, Bool]:
    var A = v_tensor(args[unsafe_offset=0])
    var hermitian = v_bool_or(args[unsafe_offset=1], False)
    var check_errors = v_bool_or(args[unsafe_offset=2], False)
    return (A^, hermitian, check_errors)


# aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)
def op_linalg_ldl_factor_ex(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _ldl_args(args)
    var r = _ldl_factor(a[0], a[1])
    if a[2]:
        _check_errors(r[2].t, "torch.linalg.ldl_factor_ex", a[0].rank == 2)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    ret_owned(rets, 2, r[2])


# aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (...)
def op_linalg_ldl_factor_ex_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = _ldl_args(args)
    var LD_out = v_tensor(args[unsafe_offset=3])
    var piv_out = v_tensor(args[unsafe_offset=4])
    var info_out = v_tensor(args[unsafe_offset=5])
    _check_out(LD_out, a[0].stype, a[0])
    _check_out(piv_out, ST_INT32, a[0])
    _check_out(info_out, ST_INT32, a[0])
    var r = _ldl_factor(a[0], a[1])
    _store(LD_out, r[0].t)
    _store(piv_out, r[1].t)
    _store(info_out, r[2].t)
    if a[2]:
        _check_errors(info_out, "torch.linalg.ldl_factor_ex", a[0].rank == 2)
    ret_ref(rets, 0, LD_out)
    ret_ref(rets, 1, piv_out)
    ret_ref(rets, 2, info_out)


def _is_int_st(st: Int32) -> Bool:
    return (
        st == ST_INT64
        or st == ST_INT32
        or st == ST_INT16
        or st == ST_INT8
        or st == ST_UINT8
    )


def _ldl_solve(LD: T, piv: T, B: T, hermitian: Bool) raises -> Owned:
    _check_square(LD, "torch.linalg.ldl_solve")
    _check_float(LD, "torch.linalg.ldl_solve")
    _linear_solve_check(B, LD, "torch.linalg.ldl_solve")
    if B.rank < 2:
        raise Error(
            (
                "torch.linalg.ldl_solve: Expected B to have at least 2"
                " dimensions, but it has "
            ),
            B.rank,
            " dimensions instead",
        )
    var n = LD.dim(-1)
    var expected = _with1(_batch_dims(LD), n)
    var got = _dims(piv)
    var same = len(expected) == len(got)
    if same:
        for i in range(len(got)):
            if expected[i] != got[i]:
                same = False
    if not same:
        raise Error(
            (
                "torch.linalg.ldl_solve: Expected LD.shape[:-1] and"
                " pivots.shape to be the same, but got pivots with shape "
            ),
            _sizes_str(got),
            " instead",
        )
    if not _is_int_st(piv.stype):
        raise Error(
            "torch.linalg.ldl_solve: Expected pivots to be integers. Got ",
            _st_name(piv.stype),
        )
    if LD.stype != B.stype:
        raise Error(
            "torch.linalg.ldl_solve: LD dtype",
            _st_name(LD.stype),
            " does not match b dtype ",
            _st_name(B.stype),
        )
    _all_on(LD, piv)
    var batch = _broadcast(_batch_dims(B), _batch_dims(LD))
    var X = own(
        _new_f(_with(batch.copy(), B.dim(-2), B.dim(-1)), B.stype, B.device)
    )
    if LD.numel == 0 or piv.numel == 0 or X.t.numel == 0:
        return X^
    var Bx = own(_expand(B, _with(batch.copy(), B.dim(-2), B.dim(-1))))
    copy_strided_into(X.t, Bx.t)
    var LDx = own(_expand(LD, _with(batch.copy(), n, n)))
    var LDf = _flat(LDx.t)
    var pi = own_if_new(_contig_i32(piv), piv)
    var px = own(_expand(pi.t, _with1(batch.copy(), n)))
    var pc = own(_clone_c(px.t))
    var l = List[Int]()
    l.append(n)
    l.append(X.t.dim(-1))
    l.append(LDf.t.stride(-2))
    l.append(LDf.t.stride(-1))
    l.append(_bstride(LDf.t))
    l.append(n)
    l.append(X.t.stride(-2))
    l.append(X.t.stride(-1))
    l.append(_bstride(X.t))
    l.append(_batch_count(X.t))
    _launch("Sytrs", LD.dtype, LD.device, LDf.t.ptr, pc.t.ptr, X.t.ptr, 0, 0, l)
    _ = LDf^
    _ = pc^
    return X^


# aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor
def op_linalg_ldl_solve(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var LD = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var B = v_tensor(args[unsafe_offset=2])
    var hermitian = v_bool_or(args[unsafe_offset=3], False)
    var X = _ldl_solve(LD, piv, B, hermitian)
    ret_owned(rets, 0, X)


# aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
def op_linalg_ldl_solve_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var LD = v_tensor(args[unsafe_offset=0])
    var piv = v_tensor(args[unsafe_offset=1])
    var B = v_tensor(args[unsafe_offset=2])
    var hermitian = v_bool_or(args[unsafe_offset=3], False)
    var out = v_tensor(args[unsafe_offset=4])
    _check_out(out, B.stype, B)
    var X = _ldl_solve(LD, piv, B, hermitian)
    _store(out, X.t)
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------


def _item(t: T) raises -> Float64:
    """`t.item()` (a device-to-host read)."""
    var c = _Op("aten::item", "")
    c.t(t)
    var r = c.run(1)
    return v_f64(r[0])


def _scalar_op(
    op: StaticString, overload: StaticString, t: T, x: Float64
) raises -> Owned:
    var c = _Op(op, overload)
    c.t(t)
    c.s(x)
    if op == "aten::add" or op == "aten::sub":
        c.s(1.0)
    return c.one()


def _reduce_dim(
    op: StaticString, t: T, dim: Int, keepdim: Bool
) raises -> Owned:
    """`sum` / `amax` over one dim."""
    var c = _Op(op, "dim_IntList" if op == "aten::sum" else "")
    c.t(t)
    var d = List[Int]()
    d.append(dim)
    c.ints(d)
    c.b(keepdim)
    if op == "aten::sum":
        c.none()
    return c.one()


def _lstsq_driver(v: Value) raises -> String:
    """`get_default_lstsq_driver` as on CUDA: 'gels' by default and the only
    driver accepted."""
    if v_is_none(v):
        return String("gels")
    var d = v_string(v).lower()
    if d != "gels":
        raise Error(
            "torch.linalg.lstsq: `driver` other than `gels` is not supported"
            " on CUDA"
        )
    return d


def _lstsq(
    A: T, B: T, rcond_v: Value, driver_v: Value
) raises -> Tuple[Owned, Owned, Owned, Owned]:
    """`linalg_lstsq` as torch runs it on CUDA: the 'gels' driver only,
    Householder QR of A (m >= n) or of A^T (m < n, the minimum-norm
    solution), as CUDA's `linalg_lstsq_gels`; rank and singular values come
    back empty, residuals for an overdetermined system."""
    if A.rank < 2:
        raise Error(
            "torch.linalg.lstsq: input must have at least 2 dimensions."
        )
    if B.rank < 1:
        raise Error("torch.linalg.lstsq: other must have at least 1 dimension.")
    if A.stype != B.stype:
        raise Error(
            (
                "torch.linalg.lstsq: Expected input and other to have the same"
                " dtype, but got input's dtype "
            ),
            _st_name(A.stype),
            " and other's dtype ",
            _st_name(B.stype),
        )
    var dim_diff = A.rank - B.rank
    if dim_diff < 0 or dim_diff > 1:
        raise Error(
            "torch.linalg.lstsq: input.dim() must be greater or equal to"
            " other.dim() and (input.dim() - other.dim()) <= 1"
        )
    var vector_case = _is_vector_rhs(A, B)
    var B2 = own(_unsqueeze_last(B)) if vector_case else own_if_new(B.copy(), B)
    if A.dim(-2) != B2.t.dim(-2):
        if vector_case:
            raise Error(
                "torch.linalg.lstsq: input.size(-2) should match other.size(-1)"
            )
        raise Error(
            "torch.linalg.lstsq: input.size(-2) should match other.size(-2)"
        )
    if not _same_device(A, B):
        raise Error(
            (
                "torch.linalg.lstsq: Expected other and input tensors to be on"
                " the same device, but got other on "
            ),
            device_str(B),
            " and input on ",
            device_str(A),
        )
    _ = _lstsq_driver(driver_v)
    _ = rcond_v  # rcond only matters to the rank-revealing drivers
    _check_compute(A, "torch.linalg.lstsq")
    var m = A.dim(-2)
    var n = A.dim(-1)
    var nrhs = B2.t.dim(-1)
    var batch = _broadcast(_batch_dims(A), _batch_dims(B2.t))
    var Ax = own(_expand(A, _with(batch.copy(), m, n)))
    var Bx = own(_expand(B2.t, _with(batch.copy(), m, nrhs)))
    var X = own(_new_f(_with(batch.copy(), n, nrhs), A.stype, A.device))
    var residuals = own(_empty0(A.stype, A.device))
    var rank = own(_empty0(ST_INT64, A.device))
    var sv = own(_empty0(A.stype, A.device))
    if m == 0 or n == 0 or X.t.numel == 0:
        fill_value(X.t, 0.0)
    else:
        var Acp = own(_clone_f(Ax.t))
        if m >= n:
            var tau = own(_new_c(_with1(batch.copy(), n), A.stype, A.device))
            _k_geqrf(Acp.t, tau.t)
            var C = own(_clone_f(Bx.t))
            _k_ormqr(Acp.t, tau.t, C.t, True)
            var Ctop = own(_narrow(C.t, -2, 0, n))
            var R = own(_narrow(Acp.t, -2, 0, n))
            _k_trsm(R.t, Ctop.t, True, False, False)
            copy_strided_into(X.t, Ctop.t)
            if m > n:
                var tail = own(_narrow(C.t, -2, n, m - n))
                var sq = _binary("aten::mul", tail.t, tail.t)
                residuals = _reduce_dim("aten::sum", sq.t, -2, False)
                _ = tail^
            _ = tau^
            _ = R^
            _ = Ctop^
            _ = C^
        else:
            # A^T = Q R: A = R^T Q^T, so R^T y = B and X = Q [y; 0].
            var At = own(_mT(Ax.t))
            var Q = own(_clone_f(At.t))
            var tau = own(_new_c(_with1(batch.copy(), m), A.stype, A.device))
            _k_geqrf(Q.t, tau.t)
            fill_value(X.t, 0.0)
            var Xtop = own(_narrow(X.t, -2, 0, m))
            copy_strided_into(Xtop.t, Bx.t)
            var R = own(_narrow(Q.t, -2, 0, m))
            _k_trsm(R.t, Xtop.t, True, True, False)
            _k_ormqr(Q.t, tau.t, X.t, False)
            _ = tau^
            _ = R^
            _ = Xtop^
            _ = Q^
        _ = Acp^
    var sol: Owned
    if vector_case:
        sol = own(_squeeze_last(X.t))
    else:
        sol = X^
    _ = Ax^
    _ = Bx^
    _ = B2^
    return (sol^, residuals^, rank^, sv^)


def _lstsq_out_dtype(dest: T, from_st: Int32, name: StaticString) raises:
    """`checkLinalgCompatibleDtype`: `from_st` must cast safely into the
    out tensor's dtype (c10::canCast)."""
    if not can_cast(from_st, dest.stype) or (
        _is_complex(dest.stype) != _is_complex(from_st) and _is_complex(from_st)
    ):
        raise Error(
            "torch.linalg.lstsq: Expected ",
            name,
            " to be safely castable from ",
            _st_name(from_st),
            " dtype, but got ",
            name,
            " with dtype ",
            _st_name(dest.stype),
        )


def _lstsq_out_check(dest: T, input: T, name: StaticString) raises:
    if not _same_device(dest, input):
        raise Error(
            "torch.linalg.lstsq: Expected ",
            name,
            " and input tensors to be on the same device, but got ",
            name,
            " on ",
            device_str(dest),
            " and input on ",
            device_str(input),
        )


# aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (...)
def op_linalg_lstsq_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var A = v_tensor(args[unsafe_offset=0])
    var B = v_tensor(args[unsafe_offset=1])
    var sol_out = v_tensor(args[unsafe_offset=4])
    var res_out = v_tensor(args[unsafe_offset=5])
    var rank_out = v_tensor(args[unsafe_offset=6])
    var sv_out = v_tensor(args[unsafe_offset=7])
    _lstsq_out_check(sol_out, A, "solution")
    _lstsq_out_check(res_out, A, "residuals")
    _lstsq_out_check(rank_out, A, "rank")
    _lstsq_out_check(sv_out, A, "singular_values")
    _lstsq_out_dtype(sol_out, A.stype, "solution")
    # torch names the residuals "solution" in this message too
    _lstsq_out_dtype(res_out, A.stype, "solution")
    _lstsq_out_dtype(rank_out, ST_INT64, "rank")
    _lstsq_out_dtype(sv_out, A.stype, "singular_values")
    var r = _lstsq(A, B, args[unsafe_offset=2], args[unsafe_offset=3])
    _store_cast(sol_out, r[0].t)
    _store_cast(res_out, r[1].t)
    _store_cast(rank_out, r[2].t)
    _store_cast(sv_out, r[3].t)
    ret_ref(rets, 0, sol_out)
    ret_ref(rets, 1, res_out)
    ret_ref(rets, 2, rank_out)
    ret_ref(rets, 3, sv_out)


# ---------------------------------------------------------------------------
# Matrix exponential (Bader, Blanes, Casas optimized Taylor polynomials with
# scaling and squaring: torch's `mexp`, LinearAlgebra.cpp)
# ---------------------------------------------------------------------------


def _coef(c: Float64, f32: Bool) -> Float64:
    """A coefficient as torch's `scalar_t` holds it."""
    if f32:
        return Float64(Float32(c))
    return c


def _coef_in(c: Float64, st: Int32) -> Float64:
    """A coefficient as a tensor of dtype `st` holds it (torch moves its
    coefficient blobs into the matrix's dtype)."""
    if st == ST_FLOAT32:
        return Float64(Float32(c))
    if st == ST_FLOAT16:
        return Float64(Float16(c))
    if st == ST_BFLOAT16:
        return Float64(BFloat16(c))
    return c


def _lincomb(mats: List[Int], coefs: List[Float64], like: T) raises -> Owned:
    """sum_j coefs[j] * mats[j] (handles of `like`-shaped tensors),
    accumulated in order from zero, as `_compute_linear_combination`."""
    var acc = own(_new_c(_dims(like), like.stype, like.device))
    fill_value(acc.t, 0.0)
    for j in range(len(mats)):
        # A zero coefficient still adds 0 * M (NaN for an infinite M), as
        # torch's kernel does.
        var c = _Op("aten::add_", "Tensor")
        c.t(acc.t)
        c.t(T(mats[j]))
        c.s(_coef_in(coefs[j], like.stype))
        _ = c.run(1)
    return acc^


def _powers(A: T, count: Int) raises -> List[Owned]:
    """[I, A, A^2, A^3, A^6][:count] (torch's `_fill_matrix_powers`)."""
    var out = List[Owned]()
    var I = own(_new_c(_dims(A), A.stype, A.device))
    _eye_into(I.t)
    out.append(I^)
    out.append(own(_clone_c(A)))
    if count > 2:
        out.append(_matmul(out[1].t, out[1].t))
    if count > 3:
        out.append(_matmul(out[1].t, out[2].t))
    if count > 4:
        out.append(_matmul(out[3].t, out[3].t))
    return out^


def _handles(ps: List[Owned], start: Int, count: Int) -> List[Int]:
    var hs = List[Int]()
    for i in range(start, start + count):
        hs.append(ps[i].t.h)
    return hs^


def _add_into(dst: T, src: T) raises:
    var c = _Op("aten::add_", "Tensor")
    c.t(dst)
    c.t(src)
    c.s(1.0)
    _ = c.run(1)


def _mexp_T(A: T, deg: Int) raises -> Owned:
    """The degree-`deg` approximant of exp(A) (deg in 1, 2, 4, 8, 12, 18)."""
    var f32 = A.stype != ST_FLOAT64
    if deg == 1:
        var ps = _powers(A, 2)
        var r = _lincomb(_handles(ps, 0, 2), [1.0, 1.0], A)
        _ = ps^
        return r^
    if deg == 2:
        var ps = _powers(A, 3)
        var r = _lincomb(_handles(ps, 0, 3), [1.0, 1.0, 0.5], A)
        _ = ps^
        return r^
    if deg == 4:
        var ps = _powers(A, 3)
        var inner = _lincomb(
            _handles(ps, 0, 3),
            [_coef(1 / 2.0, f32), _coef(1 / 6.0, f32), _coef(1 / 24.0, f32)],
            A,
        )
        var a3 = _matmul(ps[2].t, inner.t)
        var hs = _handles(ps, 0, 3)
        hs.append(a3.t.h)
        var r = _lincomb(hs, [1.0, 1.0, 0.0, 1.0], A)
        _ = a3^
        _ = inner^
        _ = ps^
        return r^
    if deg == 8:
        var sqrt_177 = 0.1330413469565007072504e2
        var x3 = 2.0 / 3.0
        var x1 = x3 * ((1.0 + sqrt_177) / 88.0)
        var x2 = x3 * ((1.0 + sqrt_177) / 352.0)
        var x4 = (-271.0 + 29.0 * sqrt_177) / (315.0 * x3)
        var x5 = (-11.0 + 11.0 * sqrt_177) / (1260.0 * x3)
        var x6 = (-99.0 + 11.0 * sqrt_177) / (5040.0 * x3)
        var x7 = (89.0 - sqrt_177) / (5040.0 * x3)
        var y2 = (857.0 - 58.0 * sqrt_177) / 630.0
        if f32:
            # constexpr float arithmetic, as torch's compute_T8<float>
            var s32 = Float32(sqrt_177)
            var t3 = Float32(2.0) / Float32(3.0)
            x3 = Float64(t3)
            x1 = Float64(t3 * ((Float32(1.0) + s32) / Float32(88.0)))
            x2 = Float64(t3 * ((Float32(1.0) + s32) / Float32(352.0)))
            x4 = Float64(
                (Float32(-271.0) + Float32(29.0) * s32) / (Float32(315.0) * t3)
            )
            x5 = Float64(
                (Float32(-11.0) + Float32(11.0) * s32) / (Float32(1260.0) * t3)
            )
            x6 = Float64(
                (Float32(-99.0) + Float32(11.0) * s32) / (Float32(5040.0) * t3)
            )
            x7 = Float64((Float32(89.0) - s32) / (Float32(5040.0) * t3))
            y2 = Float64(
                (Float32(857.0) - Float32(58.0) * s32) / Float32(630.0)
            )
        var ps = _powers(A, 3)
        var c1 = _lincomb(_handles(ps, 1, 2), [x1, x2], A)
        var a4 = _matmul(ps[2].t, c1.t)
        var hs = _handles(ps, 2, 1)
        hs.append(a4.t.h)
        var left = _lincomb(hs, [x3, 1.0], A)
        var hs2 = _handles(ps, 0, 3)
        hs2.append(a4.t.h)
        var right = _lincomb(hs2, [x4, x5, x6, x7], A)
        var a8 = _matmul(left.t, right.t)
        var hs3 = _handles(ps, 0, 3)
        hs3.append(a8.t.h)
        var r = _lincomb(hs3, [1.0, 1.0, y2, 1.0], A)
        _ = c1^
        _ = a4^
        _ = left^
        _ = right^
        _ = a8^
        _ = ps^
        return r^
    if deg == 12:
        var b = [
            [
                9.0198e-16,
                0.46932117595418237389,
                -0.20099424927047284052,
                -0.04623946134063071740,
            ],
            [
                5.31597895759871264183,
                1.19926790417132231573,
                0.01179296240992997031,
                0.01108844528519167989,
            ],
            [
                0.18188869982170434744,
                0.05502798439925399070,
                0.09351590770535414968,
                0.00610700528898058230,
            ],
            [
                -2.0861320e-13,
                -0.13181061013830184015,
                -0.02027855540589259079,
                -0.00675951846863086359,
            ],
        ]
        var ps = _powers(A, 4)
        var Bs = List[Owned]()
        for i in range(4):
            var cs = List[Float64]()
            for j in range(4):
                cs.append(_coef(b[i][j], f32))
            Bs.append(_lincomb(_handles(ps, 0, 4), cs, A))
        var a6 = _matmul(Bs[3].t, Bs[3].t)
        _add_into(Bs[2].t, a6.t)
        _add_into(Bs[1].t, Bs[2].t)
        var prod = _matmul(Bs[1].t, Bs[2].t)
        _add_into(Bs[0].t, prod.t)
        _ = a6^
        _ = prod^
        _ = ps^
        return own(Bs[0].take())
    # deg 18
    var b = [
        [
            0.0,
            -1.00365581030144618291e-01,
            -8.02924648241156932449e-03,
            -8.92138498045729985177e-04,
            0.0,
        ],
        [
            0.0,
            3.97849749499645077844e-01,
            1.36783778460411720168e00,
            4.98289622525382669416e-01,
            -6.37898194594723280150e-04,
        ],
        [
            -1.09676396052962061844e01,
            1.68015813878906206114e00,
            5.71779846478865511061e-02,
            -6.98210122488052056106e-03,
            3.34975017086070470649e-05,
        ],
        [
            -9.04316832390810593223e-02,
            -6.76404519071381882256e-02,
            6.75961301770459654925e-02,
            2.95552570429315521194e-02,
            -1.39180257516060693404e-05,
        ],
        [
            0.0,
            0.0,
            -9.23364619367118555360e-02,
            -1.69364939002081722752e-02,
            -1.40086798182036094347e-05,
        ],
    ]
    var ps = _powers(A, 5)
    var Bs = List[Owned]()
    for i in range(5):
        var cs = List[Float64]()
        for j in range(5):
            cs.append(_coef(b[i][j], f32))
        Bs.append(_lincomb(_handles(ps, 0, 5), cs, A))
    var a9 = _matmul(Bs[0].t, Bs[4].t)
    _add_into(Bs[3].t, a9.t)
    _add_into(Bs[2].t, Bs[3].t)
    var prod = _matmul(Bs[2].t, Bs[3].t)
    _add_into(Bs[1].t, prod.t)
    _ = a9^
    _ = prod^
    _ = ps^
    return own(Bs[1].take())


def _mexp_scale_square(A: T, norm: T, theta: Float64) raises -> Owned:
    """`compute_T18_scale_square`: scale each matrix by 2^-s, s =
    max(ceil(log2(norm / theta)), 0), take T18, square it back s times."""
    var q = _scalar_op("aten::div", "Scalar", norm, theta)
    var lg = _unary("aten::log2", q.t)
    var ce = _unary("aten::ceil", lg.t)
    var cm = _Op("aten::clamp_min", "")
    cm.t(ce.t)
    cm.s(0.0)
    var s_raw = cm.one()
    # A matrix with a non-finite norm squares zero times and comes out NaN
    # (torch's result for it); it must not set every other matrix's count.
    var fin = _unary("aten::isfinite", norm)
    var sw = _Op("aten::where", "ScalarOther")
    sw.t(fin.t)
    sw.t(s_raw.t)
    sw.s(0.0)
    var s = sw.one()
    var ng = _unary("aten::neg", s.t)
    var pw = _Op("aten::pow", "Scalar")
    pw.s(2.0)
    pw.t(ng.t)
    var p2 = pw.one()
    var p3 = own(_view_b11(p2.t))
    var scaled = _binary("aten::mul", A, p3.t)
    var E = _mexp_T(scaled.t, 18)
    var mx = _unary("aten::max", s.t)
    var smax = _item(mx.t)
    var s3 = own(_view_b11(s.t))
    var it = 0
    while Float64(it) < smax:
        var sq = _matmul(E.t, E.t)
        var g = _scalar_op("aten::gt", "Scalar", s3.t, Float64(it))
        var w = _Op("aten::where", "self")
        w.t(g.t)
        w.t(sq.t)
        w.t(E.t)
        E = w.one()
        _ = sq^
        _ = g^
        it += 1
    var fin3 = own(_view_b11(fin.t))
    var nw = _Op("aten::where", "ScalarOther")
    nw.t(fin3.t)
    nw.t(E.t)
    nw.s(nan[DType.float64]())
    E = nw.one()
    _ = s_raw^
    _ = q^
    _ = lg^
    _ = ce^
    _ = ng^
    _ = p2^
    _ = p3^
    _ = s3^
    return E^


def _view_b11(t: T) raises -> T:
    """(b,) -> (b, 1, 1) view."""
    var d = List[Int]()
    d.append(t.dim(0))
    d.append(1)
    d.append(1)
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 3] = t.stride(0)
    return view_strided(t, _padded(d), strides, 3, t.offset)


def _matrix_exp(a: T) raises -> Owned:
    _check_square(a, "linalg.matrix_exp")
    if not _is_float(a.stype) and not _is_complex(a.stype):
        raise Error(
            (
                "linalg.matrix_exp: Expected a floating point or complex tensor"
                " as input. Got "
            ),
            _st_name(a.stype),
        )
    if _is_complex(a.stype):
        unsupported(
            "linalg.matrix_exp: complex inputs are not supported on mojo"
        )
    if a.stype == ST_FLOAT64 and _is_metal(a):
        unsupported(
            "linalg.matrix_exp: float64 is not supported on an Apple GPU"
        )
    var n = a.dim(-1)
    if n == 0:
        return own(_clone_c(a))
    if n == 1:
        return _unary("aten::exp", a)
    if a.numel == 0:  # an empty batch
        return own(_clone_c(a))
    var ac = own(_clone_c(a))
    var d3 = List[Int]()
    d3.append(_batch_count(a))
    d3.append(n)
    d3.append(n)
    var a3 = own(
        view_strided(
            ac.t, _padded(d3), contiguous_strides(_padded(d3), 3), 3, 0
        )
    )
    # operator_1_norm: abs().sum(-2).max(-1)
    var ab = _unary("aten::abs", a3.t)
    var cs = _reduce_dim("aten::sum", ab.t, -2, False)
    var norm = _reduce_dim("aten::amax", cs.t, -1, False)
    var f32 = a.stype == ST_FLOAT32
    var th: List[Float64]
    if f32:
        th = [
            1.192092800768788e-07,
            5.978858893805233e-04,
            5.116619363445086e-02,
            5.800524627688768e-01,
            1.461661507209034e00,
            3.010066362817634e00,
        ]
    else:
        th = [
            2.220446049250313e-16,
            2.580956802971767e-08,
            3.397168839976962e-04,
            4.991228871115323e-02,
            2.996158913811580e-01,
            1.090863719290036e00,
        ]
    var res: Owned
    if d3[0] > 1:
        res = _mexp_scale_square(a3.t, norm.t, _coef(th[5], f32))
    else:
        # One matrix: the lowest degree whose bound covers its norm.
        var nv = _item(norm.t)
        if f32:
            nv = Float64(Float32(nv))
        var degs = [1, 2, 4, 8, 12]
        res = own(_new_c(d3.copy(), a.stype, a.device))
        fill_value(res.t, nan[DType.float64]())
        var lower = -1.0
        for i in range(5):
            var upper = _coef(th[i], f32)
            if lower < nv and nv <= upper:
                res = _mexp_T(a3.t, degs[i])
            lower = upper
        if nv >= _coef(th[4], f32):
            res = _mexp_scale_square(a3.t, norm.t, _coef(th[5], f32))
    var out = own(
        view_strided(
            res.t, a.shape, contiguous_strides(a.shape, a.rank), a.rank, 0
        )
    )
    _ = ab^
    _ = cs^
    _ = norm^
    _ = res^
    return out^


# aten::linalg_matrix_exp(Tensor self) -> Tensor
def op_linalg_matrix_exp(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var r = _matrix_exp(a)
    ret_owned(rets, 0, r)


# aten::linalg_matrix_exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_linalg_matrix_exp_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=1])
    _check_out(out, a.stype, a)
    var r = _matrix_exp(a)
    _store(out, r.t)
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_linalg(site: Site) raises:
    impl[op_linalg_cholesky_ex, "linalg_cholesky_ex"](site)
    impl[op_linalg_cholesky_ex_out, "linalg_cholesky_ex.L"](site)
    impl[op_cholesky, "cholesky"](site)
    impl[op_cholesky_out, "cholesky.out"](site)
    impl[op_cholesky_inverse, "cholesky_inverse"](site)
    impl[op_cholesky_inverse_out, "cholesky_inverse.out"](site)
    impl[op_cholesky_solve_helper, "_cholesky_solve_helper"](site)
    impl[op_linalg_lu_factor_ex, "linalg_lu_factor_ex"](site)
    impl[op_linalg_lu_factor_ex_out, "linalg_lu_factor_ex.out"](site)
    impl[op_linalg_lu, "linalg_lu"](site)
    impl[op_linalg_lu_out, "linalg_lu.out"](site)
    impl[op_lu_unpack, "lu_unpack"](site)
    impl[op_lu_unpack_out, "lu_unpack.out"](site)
    impl[op_linalg_lu_solve, "linalg_lu_solve"](site)
    impl[op_linalg_lu_solve_out, "linalg_lu_solve.out"](site)
    impl[op_linalg_inv_ex, "linalg_inv_ex"](site)
    impl[op_linalg_inv_ex_out, "linalg_inv_ex.inverse"](site)
    impl[op_linalg_solve_ex_, "_linalg_solve_ex"](site)
    impl[op_linalg_solve_ex_out_, "_linalg_solve_ex.result"](site)
    impl[op_linalg_det_, "_linalg_det"](site)
    impl[op_linalg_det_out_, "_linalg_det.result"](site)
    impl[op_linalg_slogdet_, "_linalg_slogdet"](site)
    impl[op_linalg_slogdet_out_, "_linalg_slogdet.sign"](site)
    impl[op_linalg_solve_triangular, "linalg_solve_triangular"](site)
    impl[op_linalg_solve_triangular_out, "linalg_solve_triangular.out"](site)
    impl[op_triangular_solve, "triangular_solve"](site)
    impl[op_triangular_solve_out, "triangular_solve.X"](site)
    impl[op_geqrf, "geqrf"](site)
    impl[op_geqrf_out, "geqrf.a"](site)
    impl[op_linalg_householder_product, "linalg_householder_product"](site)
    impl[op_linalg_householder_product_out, "linalg_householder_product.out"](
        site
    )
    impl[op_ormqr, "ormqr"](site)
    impl[op_ormqr_out, "ormqr.out"](site)
    impl[op_linalg_qr, "linalg_qr"](site)
    impl[op_linalg_qr_out, "linalg_qr.out"](site)
    impl[op_linalg_eigh_, "_linalg_eigh"](site)
    impl[op_linalg_eigh_out_, "_linalg_eigh.eigenvalues"](site)
    impl[op_linalg_svd_, "_linalg_svd"](site)
    impl[op_linalg_svd_out_, "_linalg_svd.U"](site)
    impl[op_linalg_ldl_factor_ex, "linalg_ldl_factor_ex"](site)
    impl[op_linalg_ldl_factor_ex_out, "linalg_ldl_factor_ex.out"](site)
    impl[op_linalg_ldl_solve, "linalg_ldl_solve"](site)
    impl[op_linalg_ldl_solve_out, "linalg_ldl_solve.out"](site)
    impl[op_linalg_lstsq_out, "linalg_lstsq.out"](site)
    impl[op_linalg_matrix_exp, "linalg_matrix_exp"](site)
    impl[op_linalg_matrix_exp_out, "linalg_matrix_exp.out"](site)
