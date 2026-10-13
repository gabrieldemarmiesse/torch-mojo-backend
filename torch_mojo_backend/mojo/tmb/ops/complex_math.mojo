"""Complex arithmetic and math on the mojo device: add / sub / mul / div
/ pow / rsub, the unary math (exp, log, sqrt, abs, angle, sgn, ...),
complex() / polar(), and the sum / mean / cumsum reductions.

Built on tmb/ops/complex.mojo (the complex dtypes' routing, type
promotion, operand flattening and `out=` handling) and the complex kernel
family (tmb/kernels/complex, whose math follows the CUDA kernels stock
PyTorch runs; tmb/kernels/common/complex_math.mojo).
"""
from std.utils import IndexList

from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.complex import (
    _Call,
    _Flat,
    _Operand,
    _TO_BOOL,
    _TO_CX,
    _TO_RE,
    _binary,
    _dims_of,
    _host_value,
    _inplace,
    _is_float,
    _no_alpha,
    _operand,
    _real_out,
    _result_type,
    _ret,
    _st_name,
    _store,
    _store_ew,
    _store_red,
    _tensor_or_scalar,
    _wrap_dim,
    as_complex,
    as_real,
    cx_unary,
    cx_unary_out,
)
from tmb.ops.common import (
    CX_DEVICE,
    CX_FULL,
    CX_HOST,
    cast_complex,
    contiguous,
    copy_strided_into,
    cx_launch,
    fill_complex,
)
from tmb.backend.abi import (
    Owned,
    ST_COMPLEX64,
    T,
    TAG_COMPLEX,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_TENSOR,
    TAG_TENSOR_REF,
    Value,
    Values,
    complex_of,
    dtype_name,
    f64_bits,
    is_complex_stype,
    new_tensor,
    own,
    own_if_new,
    real_of,
    release,
    retain,
    unsupported,
    v_bool_or,
    v_complex,
    v_dtype_or,
    v_tensor_c,
    view_strided,
)


# add / sub ------------------------------------------------------------------


def cx_add(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """add.Tensor(self, other, *, alpha)"""
    _ret(
        rets,
        _binary(
            "add",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
    )


def cx_add_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """add_.Tensor(self, other, *, alpha)"""
    var self = v_tensor_c(args[unsafe_offset=0])
    _inplace(
        args,
        self,
        _binary(
            "add",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
        rets,
    )


def cx_add_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """add.out(self, other, *, alpha, out)"""
    if _real_out("aten::add", "Tensor", args, 3, 3, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](
        args,
        dest,
        _binary(
            "add",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
        rets,
    )


def cx_sub(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(
        rets,
        _binary(
            "sub",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
    )


def cx_sub_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor_c(args[unsafe_offset=0])
    _inplace(
        args,
        self,
        _binary(
            "sub",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
        rets,
    )


def cx_sub_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if _real_out("aten::sub", "Tensor", args, 3, 3, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](
        args,
        dest,
        _binary(
            "sub",
            args[unsafe_offset=0],
            args[unsafe_offset=1],
            args[unsafe_offset=2].copy(),
        ),
        rets,
    )


# mul / div ------------------------------------------------------------------


def cx_mul(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(
        rets,
        _binary(
            "mul", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
    )


def cx_mul_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor_c(args[unsafe_offset=0])
    _inplace(
        args,
        self,
        _binary(
            "mul", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
        rets,
    )


def cx_mul_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if _real_out("aten::mul", "Tensor", args, 2, 2, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_ew[2](
        args,
        dest,
        _binary(
            "mul", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
        rets,
    )


def cx_div(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(
        rets,
        _binary(
            "div", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
    )


def cx_div_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    if _real_out("aten::div", "Tensor", args, 2, 2, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_ew[2](
        args,
        dest,
        _binary(
            "div", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
        rets,
    )


def _no_rounding(mode: Value) raises:
    if mode.tag != TAG_NONE:
        unsupported("div with a rounding_mode is not defined for complex")


def cx_div_mode(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """div.Tensor_mode(self, other, *, rounding_mode)"""
    _no_rounding(args[unsafe_offset=2].copy())
    cx_div(args, n_args, rets, n_rets)


def cx_div_out_mode(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """div.out_mode(self, other, *, rounding_mode, out)"""
    _no_rounding(args[unsafe_offset=2].copy())
    if _real_out("aten::div", "Tensor_mode", args, 3, 3, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](
        args,
        dest,
        _binary(
            "div", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
        rets,
    )


# pow ------------------------------------------------------------------------


def _ones(
    shape: IndexList[MAX_RANK], rank: Int, st: Int32, device: Int
) raises -> Owned:
    var ones = own(new_tensor(shape, rank, st, device))
    fill_complex(ones.t, Value(TAG_SCALAR_DOUBLE, 0, f64_bits(1.0), 0))
    return ones^


def _pow_tensor_scalar(x: T, ev: Value) raises -> Owned:
    """pow.Tensor_Scalar as ATen runs it. The result dtype is
    `result_type(x, exponent)` (a complex Scalar is a wrapped number of the
    default complex dtype: it never widens a complex64 tensor). Pow.cpp's
    structured impl: an exponent EQUAL to 0 fills ones, to 1 copies
    (`Scalar::equal`, so the complex 0j / 1+0j count, and false / true);
    PowKernel.cu: a non-complex 0.5 / -0.5 / -1 is sqrt / rsqrt /
    reciprocal, anything equal to 2 squares; else exp(log(x) * e)."""
    var e = v_complex(ev)
    var cplx = ev.tag == TAG_COMPLEX
    var st = _result_type(_Operand(x), _operand(ev, x))
    if not is_complex_stype(st):
        unsupported("pow of a complex tensor with a non-complex result")
    if e[0] == 0.0 and e[1] == 0.0:
        return _ones(x.shape, x.rank, st, x.device)
    var dense = own_if_new(cast_complex(x, st), x)
    var out = own(new_tensor(x.shape, x.rank, st, x.device))
    if e[0] == 1.0 and e[1] == 0.0:
        copy_strided_into(out.t, dense.t)
        _ = dense^
        return out^
    if not dense.t.contig:
        dense = own(contiguous(dense.t))
    var kind: StaticString = "pow"
    if not cplx and e[0] == 0.5:
        kind = "sqrt"
    elif not cplx and e[0] == -0.5:
        kind = "rsqrt"
    elif not cplx and e[0] == -1.0:
        kind = "reciprocal"
    elif e[0] == 2.0 and e[1] == 0.0:
        kind = "square"
    cx_launch(
        kind,
        out.t.ptr,
        dense.t.ptr,
        0,
        CX_FULL,
        CX_HOST,
        0.0,
        0.0,
        e[0],
        e[1],
        x.numel,
        dense.t.dtype,
        x.device,
    )
    _ = dense^
    return out^


def cx_pow_tensor_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """pow.Tensor_Scalar(self, exponent)"""
    _ret(
        rets,
        _pow_tensor_scalar(
            v_tensor_c(args[unsafe_offset=0]), args[unsafe_offset=1].copy()
        ),
    )


def cx_pow_tensor_scalar_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _real_out("aten::pow", "Tensor_Scalar", args, 2, 2, rets):
        return
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_ew[1](
        args,
        dest,
        _pow_tensor_scalar(
            v_tensor_c(args[unsafe_offset=0]), args[unsafe_offset=1].copy()
        ),
        rets,
    )


def _pow(av: Value, bv: Value) raises -> Owned:
    """pow.Tensor_Tensor / pow.Scalar. Pow.cpp's pow_Scalar_out: a base
    EQUAL to 1 fills ones (of the result dtype)."""
    if av.tag != TAG_TENSOR and av.tag != TAG_TENSOR_REF:
        var base = v_complex(av)
        if base[0] == 1.0 and base[1] == 0.0:
            var e = v_tensor_c(bv)
            var st = _result_type(_operand(av, e), _Operand(e))
            return _ones(e.shape, e.rank, st, e.device)
    return _binary("pow", av, bv, _no_alpha())


def cx_pow_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """pow.Tensor_Tensor(self, exponent) and pow.Scalar(self, exponent)."""
    _ret(rets, _pow(args[unsafe_offset=0], args[unsafe_offset=1]))


def cx_pow_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if _real_out(
        "aten::pow",
        _tensor_or_scalar(args[unsafe_offset=0], "Tensor_Tensor", "Scalar"),
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
        _pow(args[unsafe_offset=0], args[unsafe_offset=1]),
        rets,
    )


comptime cx_abs = cx_unary["abs", _TO_RE]
comptime cx_abs_out = cx_unary_out["abs", _TO_RE]
comptime cx_angle = cx_unary["angle", _TO_RE]
comptime cx_angle_out = cx_unary_out["angle", _TO_RE]
comptime cx_exp = cx_unary["exp", _TO_CX]
comptime cx_exp_out = cx_unary_out["exp", _TO_CX]
comptime cx_log = cx_unary["log", _TO_CX]
comptime cx_log_out = cx_unary_out["log", _TO_CX]
comptime cx_neg = cx_unary["neg", _TO_CX]
comptime cx_neg_out = cx_unary_out["neg", _TO_CX]
comptime cx_reciprocal = cx_unary["reciprocal", _TO_CX]
comptime cx_reciprocal_out = cx_unary_out["reciprocal", _TO_CX]
comptime cx_rsqrt = cx_unary["rsqrt", _TO_CX]
comptime cx_rsqrt_out = cx_unary_out["rsqrt", _TO_CX]
comptime cx_sgn = cx_unary["sgn", _TO_CX]
comptime cx_sgn_out = cx_unary_out["sgn", _TO_CX]
comptime cx_sqrt = cx_unary["sqrt", _TO_CX]
comptime cx_sqrt_out = cx_unary_out["sqrt", _TO_CX]
comptime cx_isnan = cx_unary["isnan", _TO_BOOL]
comptime cx_isnan_out = cx_unary_out["isnan", _TO_BOOL]
comptime cx_isinf = cx_unary["isinf", _TO_BOOL]
comptime cx_isinf_out = cx_unary_out["isinf", _TO_BOOL]
comptime cx_isfinite = cx_unary["isfinite", _TO_BOOL]

# ---------------------------------------------------------------------------
# complex(real, imag) / polar(abs, angle)
# ---------------------------------------------------------------------------


def _real_flat(
    t: T, shape: IndexList[MAX_RANK], rank: Int, mut keep: List[Int]
) raises -> _Flat:
    """A real operand of `complex` / `polar` as the family takes it."""
    if t.on_cpu() and t.rank == 0:
        var hv = _host_value(t)
        return _Flat(CX_HOST, 0, hv[0], 0.0)
    if t.rank == 0:
        return _Flat(CX_DEVICE, t.ptr, 0.0, 0.0)
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        strides[i] = t.strides[i] if t.shape[i] == shape[i] else 0
    var expanded = own(view_strided(t, shape, strides, rank, t.offset))
    var dense = contiguous(expanded.t)
    if dense.h != expanded.t.h:
        keep.append(dense.h)
    else:
        keep.append(retain(dense))
    _ = expanded^
    return _Flat(CX_FULL, dense.ptr, 0.0, 0.0)


def _make(kind: StaticString, re: T, im: T, out_st: Int32) raises -> Owned:
    var device = re.device if re.on_mojo() else im.device
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    for k in range(2):
        var t = re.copy() if k == 0 else im.copy()
        if t.on_cpu() and t.rank == 0:
            continue
        rank = max(rank, t.rank)
        for i in range(MAX_RANK):
            if shape[i] == 1:
                shape[i] = t.shape[i]
            elif t.shape[i] != 1 and t.shape[i] != shape[i]:
                raise Error(
                    "The size of tensor a must match the size of tensor b at"
                    " non-singleton dimension"
                )
    var keep = List[Int]()
    var fa = _real_flat(re, shape, rank, keep)
    var fb = _real_flat(im, shape, rank, keep)
    var out = own(new_tensor(shape, rank, out_st, device))
    cx_launch(
        kind,
        out.t.ptr,
        fa.addr,
        fb.addr,
        fa.mode,
        fb.mode,
        fa.re,
        0.0,
        fb.re,
        0.0,
        out.t.numel,
        DType.float32 if out_st == ST_COMPLEX64 else DType.float64,
        device,
    )
    for h in keep:
        release(h)
    return out^


def _make_dtype(name: StaticString, re: T, im: T) raises -> Int32:
    """`complex_check_floating`: both parts of one floating dtype; that
    dtype's complex type is the result."""
    if not _is_float(re.stype) or not _is_float(im.stype):
        raise Error(
            "Expected both inputs to be Half, Float or Double tensors but got ",
            _st_name(re.stype),
            " and ",
            _st_name(im.stype),
        )
    if re.stype != im.stype:
        raise Error(
            "Expected object of scalar type ",
            _st_name(re.stype),
            " but got scalar type ",
            _st_name(im.stype),
            " for second argument",
        )
    _ = name
    return complex_of(re.stype)


def _make_op(
    kind: StaticString, args: Values, rets: Values, out_index: Int
) raises:
    var re = v_tensor_c(args[unsafe_offset=0])
    var im = v_tensor_c(args[unsafe_offset=1])
    var st = _make_dtype(kind, re, im)
    if out_index < 0:
        _ret(rets, _make(kind, re, im, st))
        return
    var dest = v_tensor_c(args[unsafe_offset=out_index])
    if dest.stype != st:
        raise Error(
            "Expected object of scalar type ",
            _st_name(st),
            " but got scalar type ",
            _st_name(dest.stype),
            " for argument 'out'",
        )
    _store_ew[2](args, dest, _make(kind, re, im, st), rets)


def op_complex(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::complex(Tensor real, Tensor imag) -> Tensor"""
    _make_op("make", args, rets, -1)


def op_complex_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out)"""
    _make_op("make", args, rets, 2)


def op_polar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::polar(Tensor abs, Tensor angle) -> Tensor"""
    _make_op("polar", args, rets, -1)


def op_polar_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out)"""
    _make_op("polar", args, rets, 2)


def _reduce_dtype(x: T, dtype_v: Value) raises -> Int32:
    """The real dtype a complex scan accumulates in: the requested complex
    dtype's value type (a real dtype= for a complex input is declined)."""
    var want = v_dtype_or(dtype_v, x.stype)
    if not is_complex_stype(want):
        unsupported(
            "a scan of a complex tensor to the real dtype " + dtype_name(want)
        )
    return real_of(want)


def _reduce(
    name: StaticString,
    x: T,
    dim_v: Value,
    keepdim: Bool,
    dtype_v: Value,
    out_st: Int32 = -1,
) raises -> Owned:
    """`name`.dim of a complex x. The accumulation dtype is dtype= if given,
    else the `out=`'s dtype (ATen's infer_dtype_from_optional), else x's.
    A complex one: the same reduction of the real view, with the pair dim
    moved to the front so each part reduces as rows (the trailing-pair
    column reduction rounded 7x worse on a 1M-element sum), viewed back as
    complex. A real one: x converted to it first (keeping the real part,
    with torch's warning), then the real reduction."""
    var want = v_dtype_or(dtype_v, out_st if out_st >= 0 else x.stype)
    if out_st >= 0 and dtype_v.tag != TAG_NONE and want != out_st:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(want),
            ", but got ",
            dtype_name(out_st),
            " instead",
        )
    var dims = _dims_of(dim_v, x.rank)  # validated on the complex rank
    var overload = String("dim_IntList") if name == "sum" else String("dim")
    if not is_complex_stype(want):
        var xr = own_if_new(cast_complex(x, want), x)
        var rc = _Call(String("aten::") + name, overload)
        rc.t(xr.t)
        rc.ints(dims)
        rc.b(keepdim)
        rc.none()
        var r = rc.one()
        _ = xr^
        return r^
    var real_st = real_of(want)
    if x.rank == 0:
        # nothing to reduce: a fresh copy (in the requested dtype)
        if want != x.stype:
            return own(cast_complex(x, want))
        var copy = own(new_tensor(x.shape, 0, want, x.device))
        copy_strided_into(copy.t, x)
        return copy^
    var rv = as_real(x)
    var m = _Call("aten::movedim", "int")
    m.t(rv.t)
    m.i(-1)
    m.i(0)
    var front = m.one()
    _ = rv^
    var shifted = List[Int]()
    for d in dims:
        shifted.append(d + 1)
    var c = _Call(String("aten::") + name, overload)
    c.t(front.t)
    c.ints(shifted)
    c.b(keepdim)
    c.dtype(real_st)
    var r = c.one()
    _ = front^
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


def cx_sum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """sum(self, *, dtype)"""
    _ret(
        rets,
        _reduce(
            "sum",
            v_tensor_c(args[unsafe_offset=0]),
            Value(TAG_NONE, 0, 0, 0),
            False,
            args[unsafe_offset=1].copy(),
        ),
    )


def cx_sum_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """sum.dim_IntList(self, dim, keepdim, *, dtype)"""
    _ret(
        rets,
        _reduce(
            "sum",
            v_tensor_c(args[unsafe_offset=0]),
            args[unsafe_offset=1].copy(),
            v_bool_or(args[unsafe_offset=2], False),
            args[unsafe_offset=3].copy(),
        ),
    )


def cx_sum_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """sum.IntList_out(self, dim, keepdim, *, dtype, out)"""
    var dest = v_tensor_c(args[unsafe_offset=4])
    _store_red(
        dest,
        _reduce(
            "sum",
            v_tensor_c(args[unsafe_offset=0]),
            args[unsafe_offset=1].copy(),
            v_bool_or(args[unsafe_offset=2], False),
            args[unsafe_offset=3].copy(),
            dest.stype,
        ),
        rets,
    )


def cx_mean(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """mean(self, *, dtype)"""
    _ret(
        rets,
        _reduce(
            "mean",
            v_tensor_c(args[unsafe_offset=0]),
            Value(TAG_NONE, 0, 0, 0),
            False,
            args[unsafe_offset=1].copy(),
        ),
    )


def cx_mean_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """mean.dim(self, dim, keepdim, *, dtype)"""
    _ret(
        rets,
        _reduce(
            "mean",
            v_tensor_c(args[unsafe_offset=0]),
            args[unsafe_offset=1].copy(),
            v_bool_or(args[unsafe_offset=2], False),
            args[unsafe_offset=3].copy(),
        ),
    )


def cx_mean_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """mean.out(self, dim, keepdim, *, dtype, out)"""
    var dest = v_tensor_c(args[unsafe_offset=4])
    _store_red(
        dest,
        _reduce(
            "mean",
            v_tensor_c(args[unsafe_offset=0]),
            args[unsafe_offset=1].copy(),
            v_bool_or(args[unsafe_offset=2], False),
            args[unsafe_offset=3].copy(),
            dest.stype,
        ),
        rets,
    )


def cx_mean_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """mean.dtype_out(self, *, dtype, out)"""
    var dest = v_tensor_c(args[unsafe_offset=2])
    _store_red(
        dest,
        _reduce(
            "mean",
            v_tensor_c(args[unsafe_offset=0]),
            Value(TAG_NONE, 0, 0, 0),
            False,
            args[unsafe_offset=1].copy(),
            dest.stype,
        ),
        rets,
    )


def cx_rsub(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """rsub.Tensor / rsub.Scalar(self, other, *, alpha): other - alpha * self.
    """
    _ret(
        rets,
        _binary(
            "sub",
            args[unsafe_offset=1],
            args[unsafe_offset=0],
            args[unsafe_offset=2].copy(),
        ),
    )


def cx_rsub_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """rsub.Tensor_out / rsub.Scalar_out(self, other, *, alpha, out)"""
    if _real_out(
        "aten::rsub",
        _tensor_or_scalar(args[unsafe_offset=1], "Tensor", "Scalar"),
        args,
        3,
        3,
        rets,
    ):
        return
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](
        args,
        dest,
        _binary(
            "sub",
            args[unsafe_offset=1],
            args[unsafe_offset=0],
            args[unsafe_offset=2].copy(),
        ),
        rets,
    )


def _cumsum(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    var real_st = _reduce_dtype(x, args[unsafe_offset=2].copy())
    if x.rank == 0:
        _ = _wrap_dim(Int(args[unsafe_offset=1].a), 0)
        var st = complex_of(real_st)
        if st != x.stype:
            return own(cast_complex(x, st))
        var copy = own(new_tensor(x.shape, 0, x.stype, x.device))
        copy_strided_into(copy.t, x)
        return copy^
    var dim = _wrap_dim(Int(args[unsafe_offset=1].a), x.rank)
    var rv = as_real(x)
    var c = _Call("aten::cumsum")
    c.t(rv.t)
    c.i(dim)
    c.dtype(real_st)
    var r = c.one()
    _ = rv^
    var cv = as_complex(r.t)
    _ = r^
    return cv^


def cx_cumsum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum(self, int dim, *, ScalarType? dtype)"""
    _ret(rets, _cumsum(args))


def cx_cumsum_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum_(self, int dim, *, ScalarType? dtype)"""
    var self = v_tensor_c(args[unsafe_offset=0])
    _store(self, _cumsum(args), rets)


def cx_cumsum_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum.out(self, dim, *, dtype, out)"""
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store(dest, _cumsum(args), rets)


def register_complex_math(site: Site) raises:
    impl[op_complex, "complex"](site)
    impl[op_complex_out, "complex.out"](site)
    impl[op_polar, "polar"](site)
    impl[op_polar_out, "polar.out"](site)
