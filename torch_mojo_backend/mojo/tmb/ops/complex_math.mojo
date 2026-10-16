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
    _bshape,
    _broadcast_into,
    _dims_of,
    _first_tensor,
    _flatten,
    _host_value,
    _one_device,
    _part,
    _require_same_device,
    _operand,
    _real_out,
    _result_type,
    _ret,
    _same_device,
    _store,
    _store_ew,
    _store_ew_n,
    _exact_dtype,
    _check_cast,
    _tensor_or_scalar,
    _unary,
    _wrap_dim,
    as_complex,
    as_real,
    cx_unary_out,
)
from tmb.ops.common import (
    assert_no_internal_overlap,
    can_cast,
    is_float_stype,
    scalar_type_name,
    scalar_parts,
    scalar_type_token,
    device_str,
    check_scalar_fits,
    resize_out,
    shape_str,
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
    OpFn,
    TAG_DTYPE,
    TAG_INT,
    v_is_complex,
    Owned,
    ST_BOOL,
    ST_INT64,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_COMPLEX128,
    TAG_BOOL,
    TAG_SCALAR_BOOL,
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
    is_float8_stype,
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
    ret_ref,
)


# ---------------------------------------------------------------------------
# the arithmetic launch and the out= / in-place stores
# ---------------------------------------------------------------------------


def _no_alpha() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


def _is_bool_value(v: Value) raises -> Bool:
    if v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL:
        return True
    if v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
        return v_tensor_c(v).stype == ST_BOOL
    return False


def _host_reciprocal[
    dt: DType
](re: Float64, im: Float64) -> Tuple[Float64, Float64]:
    """`opmath_t(1) / b` as CUDA's div_true_kernel computes it on the HOST
    for a CPU-scalar divisor: c10::complex's scaled division in the
    component precision (gcc's std::abs)."""
    var c = Scalar[dt](re)
    var d = Scalar[dt](im)
    var one = Scalar[dt](1)
    var zero = Scalar[dt](0)
    var abs_c = abs(c)
    var abs_d = abs(d)
    if abs_c >= abs_d:
        if abs_c == zero and abs_d == zero:
            return (Float64(one / abs_c), Float64(zero / abs_d))
        var rat = d / c
        var scl = one / (c + d * rat)
        return (Float64(scl), Float64((zero - rat) * scl))
    var rat = c / d
    var scl = one / (d + c * rat)
    return (Float64(rat * scl), Float64(-scl))


def _arith(
    kind: StaticString, av: Value, bv: Value, alpha: Value
) raises -> Owned:
    """`kind(a, b)` (add / sub / mul / div / pow) where one side is complex:
    promoted, broadcast, one launch. `alpha` scales `b` (add, sub)."""
    if kind == "sub" and (_is_bool_value(av) or _is_bool_value(bv)):
        # BinaryOps.cpp sub_check
        raise Error(
            "Subtraction, the `-` operator, with a bool tensor is not"
            " supported. If you are trying to invert a mask, use the `~` or"
            " `logical_not()` operator instead."
        )
    if alpha.tag == TAG_SCALAR_BOOL or alpha.tag == TAG_BOOL:
        # alpha_check: the result here is complex, never bool
        raise Error("Boolean alpha only supported for Boolean results.")
    var like = _first_tensor(av, bv)
    var a = _operand(av, like)
    var b = _operand(bv, like)
    var st = _result_type(a, b)
    if alpha.tag == TAG_COMPLEX and not is_complex_stype(st):
        raise Error(
            "For non-complex input tensors, argument alpha must not be a"
            " complex number."
        )
    if not is_complex_stype(st):
        unsupported("a complex op whose result type is not complex")
    if alpha.tag != TAG_NONE:
        # alpha.to<scalar_t>(), checked
        var al0 = v_complex(alpha)
        check_scalar_fits(st, al0[0], al0[1])
    var device = _one_device(a, b)
    var geo = _bshape(a, b)
    var fa = _flatten(a, st, geo[0], geo[1])
    var fb = _flatten(b, st, geo[0], geo[1])
    var out = own(new_tensor(geo[0], geo[1], st, device))
    var component = DType.float32 if st == ST_COMPLEX64 else DType.float64
    # add / sub: CUDA's a + alpha * b (sub is add with -alpha: a Scalar's
    # negation, so a real alpha keeps a +0 imaginary part)
    var p_re = 1.0
    var p_im = 0.0
    if alpha.tag != TAG_NONE:
        var al = scalar_parts(alpha, st)  # an integer: one rounding
        p_re = al[0]
        p_im = al[1]
    var launch_kind = kind
    if kind == "div" and fb.mode == CX_HOST:
        # div_true_kernel_cuda: a CPU-scalar divisor becomes a multiply by
        # its reciprocal, computed once on the host (1 / (0 + 0i) is
        # inf + NaN i, so x / 0j is NaN, as on CUDA)
        # (in double, as CUDA's host-side Scalar math is, then narrowed by
        # the kernel: 1 / 1e-40 is 1e40 -> inf, not inf + NaN i)
        var inv = _host_reciprocal[DType.float64](fb.re, fb.im)
        fb = _Flat(CX_HOST, 0, inv[0], inv[1])
        launch_kind = "mul"
    if kind == "pow" and fb.mode == CX_HOST:
        # the CPU-scalar exponent becomes a scalar_t first
        # (iter.scalar_value<scalar_t>), so it is compared once rounded
        if st == ST_COMPLEX64:
            fb.re = Float64(Float32(fb.re))
            fb.im = Float64(Float32(fb.im))
    if kind == "pow" and fb.mode == CX_HOST and fb.re == 2.0 and fb.im == 0.0:
        # pow_tensor_tensor_kernel: a CPU-scalar exponent goes through
        # pow_tensor_scalar_kernel as a complex Scalar, whose only shortcut
        # is `exp.equal(2.0)` -> square (no 0 / 1 / 0.5 / -1 ones there)
        launch_kind = "square"
    if kind == "sub":
        launch_kind = "add"
        p_re = -p_re
        p_im = -p_im if alpha.tag == TAG_COMPLEX else 0.0
    cx_launch(
        launch_kind,
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
        component,
        device,
        p_re,
        p_im,
    )
    _ = a^
    _ = b^
    return out^


def _inplace(args: Values, mut self: T, var res: Owned, rets: Values) raises:
    """`self op= other`: the broadcast result must keep self's shape."""
    if not res.t.same_shape(self):
        raise Error(
            "output with shape ",
            shape_str(self),
            " doesn't match the broadcast shape ",
            shape_str(res.t),
        )
    _store_ew[2](args, self, res^, rets)


def _store_red(mut dest: T, var res: Owned, rets: Values) raises:
    """`_store` for a reduction's `out=` (its dtype already checked):
    `resize_output` first -- an expanded out of the wrong shape is resized,
    not refused -- then the copy."""
    _same_device(dest, res.t)
    resize_out(dest, res.t.shape, res.t.rank)
    _store(dest, res^, rets, False)  # reductions check no overlap


def cx_unary[
    kind: StaticString, to: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`op(self)` of a complex self."""
    _ret(rets, _unary(kind, to, v_tensor_c(args[unsafe_offset=0])))


def cx_unary_same_out[
    kind: StaticString
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`op.out(self, *, out)` of an op ATen builds with
    build_borrowing_unary_op (neg, sgn): the out must already hold self's
    dtype (TensorIterator's check_all_same_dtype)."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dest = v_tensor_c(args[unsafe_offset=1])
    if dest.stype != x.stype:
        raise Error(
            "Found dtype ",
            scalar_type_name(dest.stype),
            " but expected ",
            scalar_type_name(x.stype),
        )
    _store_ew[1](args, dest, _unary(kind, _TO_CX, x), rets)


def routed_red[
    real_op: OpFn,
    complex_op: OpFn,
    n_check: Int,
    dtype_idx: Int,
    out_idx: Int,
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`routed` for a reduction or scan: the complex route also when its
    `dtype=` (argument `dtype_idx`) or its `out=` (argument `out_idx`) is
    complex, as ATen accumulates in that dtype (-1: no such argument)."""
    var cx = False
    comptime for i in range(n_check):
        if i < n_args and v_is_complex(args[unsafe_offset=i]):
            cx = True
    comptime if dtype_idx >= 0:
        var d = args[unsafe_offset=dtype_idx].copy()
        if (d.tag == TAG_DTYPE or d.tag == TAG_INT) and is_complex_stype(
            Int32(d.a)
        ):
            cx = True
    comptime if out_idx >= 0:
        if v_is_complex(args[unsafe_offset=out_idx]):
            cx = True
    if cx:
        complex_op(args, n_args, rets, n_rets)
    else:
        real_op(args, n_args, rets, n_rets)


# add / sub ------------------------------------------------------------------


def cx_add(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """add.Tensor(self, other, *, alpha)"""
    _ret(
        rets,
        _arith(
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
        _arith(
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
        _arith(
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
        _arith(
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
        _arith(
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
        _arith(
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
        _arith(
            "mul", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
    )


def cx_mul_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor_c(args[unsafe_offset=0])
    _inplace(
        args,
        self,
        _arith(
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
        _arith(
            "mul", args[unsafe_offset=0], args[unsafe_offset=1], _no_alpha()
        ),
        rets,
    )


def cx_div(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ret(
        rets,
        _arith(
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
        _arith(
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
    # real inputs into a complex out: the real op (which rounds), cast
    if _real_out("aten::div", "Tensor_mode", args, 3, 3, rets):
        return
    _no_rounding(args[unsafe_offset=2].copy())
    var dest = v_tensor_c(args[unsafe_offset=3])
    _store_ew[2](
        args,
        dest,
        _arith(
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
    var ek = scalar_parts(ev, st)  # the exponent in st's parts, one rounding
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
    if kind == "pow":
        check_scalar_fits(st, e[0], e[1])  # exp_scalar.to<scalar_t>()
    cx_launch(
        kind,
        out.t.ptr,
        dense.t.ptr,
        0,
        CX_FULL,
        CX_HOST,
        0.0,
        0.0,
        ek[0],
        ek[1],
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


def _base_is_one(av: Value) raises -> Bool:
    """pow_Scalar_out's `base.equal(1.0)`: a Scalar base equal to 1 (a bool
    True is not: Scalar::equal compares a bool to a bool only)."""
    if av.tag == TAG_TENSOR or av.tag == TAG_TENSOR_REF:
        return False
    if av.tag == TAG_SCALAR_BOOL or av.tag == TAG_BOOL:
        return False
    var base = v_complex(av)
    return base[0] == 1.0 and base[1] == 0.0


def _pow(av: Value, bv: Value) raises -> Owned:
    """pow.Tensor_Tensor / pow.Scalar. Pow.cpp's pow_Scalar_out: a base
    EQUAL to 1 fills ones (of the result dtype)."""
    if _base_is_one(av):
        var e = v_tensor_c(bv)
        var st = _result_type(_operand(av, e), _Operand(e))
        return _ones(e.shape, e.rank, st, e.device)
    return _arith("pow", av, bv, _no_alpha())


def cx_pow_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """pow.Tensor_Tensor(self, exponent) and pow.Scalar(self, exponent)."""
    _ret(rets, _pow(args[unsafe_offset=0], args[unsafe_offset=1]))


def cx_pow_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dest = v_tensor_c(args[unsafe_offset=2])
    var av = args[unsafe_offset=0].copy()
    if av.tag != TAG_TENSOR and av.tag != TAG_TENSOR_REF:
        # pow.Scalar is structured: its meta resizes the out to the
        # exponent's shape before anything checks it (a real exponent too)
        var ex = v_tensor_c(args[unsafe_offset=1])
        _same_device(dest, ex)
        resize_out(dest, ex.shape, ex.rank)
        dest = v_tensor_c(args[unsafe_offset=2])
    if _real_out(
        "aten::pow",
        _tensor_or_scalar(args[unsafe_offset=0], "Tensor_Tensor", "Scalar"),
        args,
        2,
        2,
        rets,
    ):
        return
    if _base_is_one(av):
        # pow_Scalar_out: `out.fill_(1)` in the out's own dtype, no
        # result-type cast check
        var f = _Call("aten::fill_", "Scalar")
        f.t(dest)
        f.s(1.0)
        _ = f.run(1)
        ret_ref(rets, 0, dest)
        return
    _store_ew[2](
        args,
        dest,
        _pow(args[unsafe_offset=0], args[unsafe_offset=1]),
        rets,
    )


comptime cx_abs = cx_unary["abs", _TO_RE]


def cx_to_real_out[
    kind: StaticString
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """abs.out / angle.out with a complex self or out
    (unary_op_impl_with_complex_to_float_out). A real out of a complex self:
    the result computed, the out resized (resize_output) and copy_'d into
    -- any device, no overlap check before the resize. A complex out: a
    TensorIterator unary op, so self's own dtype. A real self (a complex
    out): abs refuses; angle casts the real op's result."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dest = v_tensor_c(args[unsafe_offset=1])
    if not x.is_complex():
        comptime if kind == "angle":
            if _real_out(String("aten::") + kind, "", args, 1, 1, rets):
                return
        raise Error(
            "Found dtype ",
            scalar_type_name(dest.stype),
            " but expected ",
            scalar_type_name(x.stype),
        )
    if is_complex_stype(dest.stype):
        # (the dtype check first, then the device check, as CUDA orders them)
        if kind == "abs" and dest.stype != x.stype:
            raise Error(
                "Found dtype ",
                scalar_type_name(dest.stype),
                " but expected ",
                scalar_type_name(x.stype),
            )
        if not x.on_mojo():
            raise Error(
                (
                    "Expected all tensors to be on the same device, but found"
                    " at least two devices, "
                ),
                device_str(dest),
                " and cpu!",
            )
        _store_ew[1](args, dest, _unary(kind, _TO_RE, x), rets)
        return
    var res: Owned
    if x.on_mojo():
        res = _unary(kind, _TO_RE, x)
    else:
        # a host self: the functional op runs where self is (CUDA's
        # at::abs(self)), then the result is resized into and copied to
        # the out, wherever that is
        var hc = _Call(String("aten::") + kind)
        hc.t(x)
        res = hc.one()
    _check_cast(res.t.stype, dest.stype)
    var sizes = List[Int]()
    for d in range(res.t.rank):
        sizes.append(res.t.dim(d))
    var rz = _Call("aten::resize_")
    rz.t(dest)
    rz.ints(sizes)
    rz.none()
    _ = rz.run(1)
    var dest2 = v_tensor_c(args[unsafe_offset=1])
    # copy_'s own first check, raised here without the nested call's
    # "aten::copy_:" prefix
    assert_no_internal_overlap(dest2)
    var cp = _Call("aten::copy_")
    cp.t(dest2)
    cp.t(res.t)
    cp.b(False)
    _ = cp.run(1)
    _ = res^
    ret_ref(rets, 0, dest2)


comptime cx_abs_out = cx_to_real_out["abs"]
comptime cx_angle = cx_unary["angle", _TO_RE]
comptime cx_angle_out = cx_to_real_out["angle"]


def cx_bool_out[
    kind: StaticString
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """isnan.out / isinf.out (generated out= wrappers): a bool out only,
    then resize and copy."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dest = v_tensor_c(args[unsafe_offset=1])
    _exact_dtype(dest, ST_BOOL)
    _store(dest, _unary(kind, _TO_BOOL, x), rets)


comptime cx_exp = cx_unary["exp", _TO_CX]
comptime cx_exp_out = cx_unary_out["exp", _TO_CX]
comptime cx_log = cx_unary["log", _TO_CX]
comptime cx_log_out = cx_unary_out["log", _TO_CX]
comptime cx_neg = cx_unary["neg", _TO_CX]
comptime cx_neg_out = cx_unary_same_out["neg"]
comptime cx_reciprocal = cx_unary["reciprocal", _TO_CX]
comptime cx_reciprocal_out = cx_unary_out["reciprocal", _TO_CX]
comptime cx_rsqrt = cx_unary["rsqrt", _TO_CX]
comptime cx_rsqrt_out = cx_unary_out["rsqrt", _TO_CX]
comptime cx_sgn = cx_unary["sgn", _TO_CX]
comptime cx_sgn_out = cx_unary_same_out["sgn"]
comptime cx_sqrt = cx_unary["sqrt", _TO_CX]
comptime cx_sqrt_out = cx_unary_out["sqrt", _TO_CX]
comptime cx_isnan = cx_unary["isnan", _TO_BOOL]
comptime cx_isnan_out = cx_bool_out["isnan"]
comptime cx_isinf = cx_unary["isinf", _TO_BOOL]
comptime cx_isinf_out = cx_bool_out["isinf"]
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
        _broadcast_into(shape, rank, t)
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
    """`complex_check_floating`: both parts of one of half / float / double;
    that dtype's complex type is the result (complex32, half's, is not a
    dtype of this device)."""
    if not _make_float(re.stype) or not _make_float(im.stype):
        raise Error(
            "Expected both inputs to be Half, Float or Double tensors but got ",
            scalar_type_name(re.stype),
            " and ",
            scalar_type_name(im.stype),
        )
    if re.stype != im.stype:
        raise Error(
            "Expected object of scalar type ",
            scalar_type_name(re.stype),
            " but got scalar type ",
            scalar_type_name(im.stype),
            " for second argument",
        )
    if re.stype == ST_FLOAT16:
        unsupported("complex32 (the complex type of half) on the mojo device")
    _ = name
    return complex_of(re.stype)


def _make_float(st: Int32) -> Bool:
    return st == ST_FLOAT16 or st == ST_FLOAT32 or st == ST_FLOAT64


def _make_op(
    kind: StaticString, args: Values, rets: Values, out_index: Int
) raises:
    var re = v_tensor_c(args[unsafe_offset=0])
    var im = v_tensor_c(args[unsafe_offset=1])
    # TensorIterator's device check: a CPU input, even a 0-d one, is not
    # a scalar here
    var device = re.device if re.on_mojo() else im.device
    var polar = kind == "polar"
    if polar:
        _require_same_device(re, device, "abs")
        _require_same_device(im, device, "angle")
    else:
        _require_same_device(re, device, "real")
        _require_same_device(im, device, "imag")
    var st = _make_dtype(kind, re, im)
    if out_index < 0:
        _ret(rets, _make(kind, re, im, st))
        return
    var dest = v_tensor_c(args[unsafe_offset=out_index])
    if dest.stype != st:
        raise Error(
            "Expected object of scalar type ",
            scalar_type_name(st),
            " but got scalar type ",
            scalar_type_name(dest.stype),
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


def _reduced_count(x: T, dims: List[Int]) -> Int:
    """How many elements each output of a reduction over `dims` takes (1
    for a 0-d x, whose dims are none)."""
    if x.rank == 0:
        return 1
    var count = 1
    for d in dims:
        count *= x.dim(d)
    return count


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
    # float8 is a floating dtype for these checks (c10's isFloatingType)
    if (
        name == "mean"
        and dtype_v.tag == TAG_NONE
        and not is_float_stype(x.stype)
        and not is_float8_stype(x.stype)
        and not x.is_complex()
    ):
        # mean_dtype_out / meta: no dtype= to compute an integer mean in
        raise Error(
            (
                "mean(): could not infer output dtype. Input dtype must be"
                " either a floating point or complex dtype. Got: "
            ),
            scalar_type_name(x.stype),
        )
    if name == "mean" and dtype_v.tag != TAG_NONE:
        var asked = v_dtype_or(dtype_v, x.stype)
        if (
            not is_float_stype(asked)
            and not is_float8_stype(asked)
            and not is_complex_stype(asked)
        ):
            raise Error(
                (
                    "mean(): could not infer output dtype. Optional dtype must"
                    " be either a floating point or complex dtype. Got: "
                ),
                scalar_type_name(asked),
            )
    var dims = _dims_of(dim_v, x.rank)  # validated on the complex rank
    for i in range(len(dims)):
        for j in range(i):
            if dims[i] == dims[j]:
                raise Error(
                    "dim ",
                    dims[i],
                    " appears multiple times in the list of dims",
                )
    var want = v_dtype_or(dtype_v, out_st if out_st >= 0 else x.stype)
    if out_st >= 0 and dtype_v.tag != TAG_NONE and want != out_st:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(want),
            ", but got ",
            dtype_name(out_st),
            " instead",
        )
    if not x.is_complex():
        # a real self reaching here accumulates in a complex dtype= / out=
        var xc = own(cast_complex(x, want))
        var r = _reduce(name, xc.t, dim_v, keepdim, Value(TAG_NONE, 0, 0, 0))
        _ = xc^
        return r^
    var overload = String("dim_IntList") if name == "sum" else String("dim")
    if not is_complex_stype(want):
        if is_float8_stype(want):
            # the reduction kernel's dtype dispatch (sum_cuda, mean_cuda)
            unsupported(
                '"'
                + name
                + "_cuda\" not implemented for '"
                + scalar_type_name(want)
                + "'"
            )
        if not is_float_stype(want):
            # a complex reduction into a bool / integer dtype or out: not
            # emulated (declined after CUDA's own checks, before the out is
            # touched)
            unsupported(
                String("complex ")
                + name
                + " into the non-floating dtype "
                + dtype_name(want)
            )
        # a floating accumulation dtype (from dtype= or the out): x's real
        # part in it, then the real reduction in it
        var xr = own_if_new(cast_complex(x, want), x)
        var rc = _Call(String("aten::") + name, overload)
        rc.t(xr.t)
        rc.ints(dims)
        rc.b(keepdim)
        rc.dtype(want)
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
    if name == "mean" and cv.t.numel > 0:
        if _reduced_count(x, dims) == 0:
            # CUDA's mean of nothing is NaN + 0i (each part's mean, ours,
            # would make the imaginary part NaN too)
            var im = _part(cv.t, 1)
            var z = _Call("aten::zero_")
            z.t(im.t)
            _ = z.run(1)
            _ = im^
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
    var x = v_tensor_c(args[unsafe_offset=0])
    if not can_cast(x.stype, dest.stype):
        raise Error(
            (
                "mean.dtype_out(): input types can't be cast to the desired"
                " output type "
            ),
            scalar_type_name(dest.stype),
        )
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
        _arith(
            "sub",
            args[unsafe_offset=1],
            args[unsafe_offset=0],
            args[unsafe_offset=2].copy(),
        ),
    )


def _cumsum(args: Values, out_st: Int32 = -1) raises -> Owned:
    """cumsum(self, dim, *, dtype) with a complex self, dtype= or out=. The
    accumulation dtype is dtype=, else the out='s dtype, else self's
    (meta_func_cum_ops); a real one keeps self's real part (torch's
    warning), then the real scan."""
    var x = v_tensor_c(args[unsafe_offset=0])
    var dtype_v = args[unsafe_offset=2].copy()
    # CUDA's order: the dim, then the out's dtype against dtype=
    var dim = _wrap_dim(Int(args[unsafe_offset=1].a), x.rank)
    var want = v_dtype_or(dtype_v, out_st if out_st >= 0 else x.stype)
    if out_st >= 0 and want != out_st:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(want),
            ", but got ",
            dtype_name(out_st),
            " instead",
        )
    if is_float8_stype(want):
        unsupported(
            '"cumsum_cuda" not implemented for \''
            + scalar_type_name(want)
            + "'"
        )
    if not is_complex_stype(want) and not is_float_stype(want):
        # a complex scan into a bool / integer dtype or out: not emulated
        # (declined before the out is touched)
        unsupported(
            "complex cumsum into the non-floating dtype " + dtype_name(want)
        )
    if x.rank == 0:
        if want != x.stype:
            return own(cast_complex(x, want))
        var copy = own(new_tensor(x.shape, 0, x.stype, x.device))
        copy_strided_into(copy.t, x)
        return copy^
    var xc = own_if_new(cast_complex(x, want), x)
    if not is_complex_stype(want):
        var rc = _Call("aten::cumsum")
        rc.t(xc.t)
        rc.i(dim)
        rc.dtype(want)
        var rr = rc.one()
        _ = xc^
        return rr^
    var rv = as_real(xc.t)
    var c = _Call("aten::cumsum")
    c.t(rv.t)
    c.i(dim)
    c.dtype(real_of(want))
    var r = c.one()
    _ = rv^
    _ = xc^
    var cv = as_complex(r.t)
    _ = r^
    return cv^


def cx_cumsum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum(self, int dim, *, ScalarType? dtype)"""
    _ret(rets, _cumsum(args))


def cx_cumsum_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum_(self, int dim, *, ScalarType? dtype)"""
    var self = v_tensor_c(args[unsafe_offset=0])
    var want = v_dtype_or(args[unsafe_offset=2], self.stype)
    if want != self.stype:
        raise Error(
            "Bad in-place call: input tensor dtype ",
            dtype_name(self.stype),
            " and output tensor dtype ",
            dtype_name(want),
            " should match",
        )
    _store(self, _cumsum(args, self.stype), rets)


def cx_cumsum_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """cumsum.out(self, dim, *, dtype, out): resized to self's shape first
    (set_output), then checked and written."""
    var dest = v_tensor_c(args[unsafe_offset=3])
    var res = _cumsum(args, dest.stype)
    _same_device(dest, res.t)
    resize_out(dest, res.t.shape, res.t.rank)
    _store(dest, res^, rets)  # a scan's out: internal overlap checked


def register_complex_math(site: Site) raises:
    impl[op_complex, "complex"](site)
    impl[op_complex_out, "complex.out"](site)
    impl[op_polar, "polar"](site)
    impl[op_polar_out, "polar.out"](site)
