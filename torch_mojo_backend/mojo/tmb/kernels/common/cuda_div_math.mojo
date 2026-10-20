"""Stock CUDA's division kernels (BinaryDivTruncKernel.cu,
BinaryDivFloorKernel.cu, BinaryRemainderKernel.cu, c10/util/generic_math.h)
operation for operation, each at the C++ type it runs in: a c10::Half or
BFloat16 operand meeting a float yields a float, `scalar_t` locals round back
to the half type. The eager div / floor_divide / remainder / fmod kernels;
`div_math` keeps the torch.compile lowering's own forms."""

from std.math import floor
from std.memory import bitcast
from std.sys import is_nvidia_gpu
from std.sys._assembly import inlined_assembly
from std.sys.info import size_of

from tmb.kernels.common.dtype_arithmetic import cast_rounded
from tmb.kernels.common.nan_order import _bits_dtype, is_nan_bits, nan_first2
from tmb.kernels.common.op_utils import _fmod_f64_exact_scalar, _fmod_float32


@always_inline
def _opmath[dtype: DType]() -> DType:
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        return DType.float32
    else:
        return dtype


@always_inline
def _mag_bits[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[_bits_dtype[dtype](), w]:
    comptime bt = _bits_dtype[dtype]()
    comptime sign = Scalar[bt](1) << Scalar[bt](size_of[dtype]() * 8 - 1)
    return bitcast[bt, w](x) & ~sign


@always_inline
def _nonzero[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    """`x != 0` with NaN nonzero, from the bits (the GPU build's fast-math
    flags may fold a float NaN compare)."""
    return _mag_bits(x).ne(0)


@always_inline
def _negative[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    """`x < 0`: false for a zero and for a NaN."""
    comptime bt = _bits_dtype[dtype]()
    comptime sign = Scalar[bt](1) << Scalar[bt](size_of[dtype]() * 8 - 1)
    var b = bitcast[bt, w](x)
    return b.ge(sign) & _nonzero(x) & ~is_nan_bits(x)


@always_inline
def _mul_rn[w: DType, n: Int](a: SIMD[w, n], b: SIMD[w, n]) -> SIMD[w, n]:
    """A rounded product no later add may absorb (nvcc rounds `(a - mod) *
    inv_b` before `div -= 1`; ptxas would contract a plain `mul`)."""
    comptime if is_nvidia_gpu() and (w == DType.float32 or w == DType.float64):
        comptime asm = "mul.rn.f32 $0, $1, $2;" if w == DType.float32 else (
            "mul.rn.f64 $0, $1, $2;"
        )
        comptime cons = "=f,f,f" if w == DType.float32 else "=d,d,d"
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = inlined_assembly[
                asm, Scalar[w], constraints=cons, has_side_effect=False
            ](a[i], b[i])
        return r
    else:
        return a * b


@always_inline
def nv_fmod(x: Float64, y: Float64) -> Float64:
    """`__nv_fmod`: the exact remainder; x's NaN quieted, then y's, else the
    negative default NaN 0xFFF8000000000000 for a zero y or an infinite x."""
    comptime if is_nvidia_gpu():
        var ux = bitcast[DType.uint64, 1](x)
        var uy = bitcast[DType.uint64, 1](y)
        comptime mag = UInt64(0x7FFFFFFFFFFFFFFF)
        comptime inf = UInt64(0x7FF0000000000000)
        comptime quiet = UInt64(0x0008000000000000)
        if (ux & mag) > inf:
            return bitcast[DType.float64, 1](ux | quiet)
        if (uy & mag) > inf:
            return bitcast[DType.float64, 1](uy | quiet)
        if (uy & mag) == 0 or (ux & mag) == inf:
            return bitcast[DType.float64, 1](UInt64(0xFFF8000000000000))
    return _fmod_f64_exact_scalar(x, y)


@always_inline
def cuda_fmod[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[_opmath[dtype](), w]:
    """C's `::fmod` per lane, in opmath (a half pair runs `fmodf`)."""
    comptime om = _opmath[dtype]()
    var r = SIMD[om, w]()
    comptime for i in range(w):
        comptime if dtype == DType.float64:
            r[i] = nv_fmod(
                a[i].cast[DType.float64](), b[i].cast[DType.float64]()
            ).cast[om]()
        else:
            r[i] = _fmod_float32(
                a[i].cast[DType.float32](), b[i].cast[DType.float32]()
            ).cast[om]()
    return r


@always_inline
def cuda_int_divrem[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> Tuple[
    SIMD[dtype, w], SIMD[dtype, w]
]:
    """C's truncating `a / b` and `a % b` as stock CUDA's integer kernels get
    them from nvcc: the narrow types promote to int, INT_MIN / -1 wraps, and
    a zero divisor yields what the GPU's division sequence does on NVIDIA:
    -1 for both, except 2**32 - 1 when an int64 pair fits in 32 unsigned bits
    (the 32-bit fast path of the 64-bit division). Elsewhere a zero divisor
    gives 0."""
    comptime wide = DType.int32 if size_of[dtype]() < 4 else dtype
    comptime uwide = DType.uint32 if size_of[wide]() == 4 else DType.uint64
    var x = a.cast[wide]()
    var y = b.cast[wide]()
    var zero = SIMD[wide, w](0)
    var xneg = x.lt(zero)
    var yneg = y.lt(zero)
    var ux = xneg.select(zero - x, x).cast[uwide]()
    var uy = yneg.select(zero - y, y).cast[uwide]()
    var by_zero = uy.eq(0)
    var uys = by_zero.select(SIMD[uwide, w](1), uy)
    var uq = ux // uys
    var ur = ux - uq * uys
    var q = (xneg ^ yneg).select(SIMD[uwide, w](0) - uq, uq).cast[wide]()
    var r = xneg.select(SIMD[uwide, w](0) - ur, ur).cast[wide]()
    var undef = SIMD[wide, w](0)
    comptime if is_nvidia_gpu():
        comptime if wide == DType.int64:
            var fits = (x.cast[DType.uint64]() >> 32).eq(0)
            undef = fits.select(SIMD[wide, w](0xFFFFFFFF), SIMD[wide, w](-1))
        else:
            undef = SIMD[wide, w](-1)
    q = by_zero.select(undef, q)
    r = by_zero.select(undef, r)
    return (q.cast[dtype](), r.cast[dtype]())


@always_inline
def cuda_trunc_div[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """div_trunc_kernel_cuda over two tensors: `trunc(a / b)` with the
    quotient rounded to `scalar_t` first; C's `/` for integers."""
    comptime if dtype.is_floating_point():
        return (a / b).__trunc__()
    else:
        return cuda_int_divrem(a, b)[0]


@always_inline
def cuda_floor_div[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """div_floor_kernel_cuda over two tensors: c10::div_floor_floating /
    div_floor_integer."""
    comptime if dtype.is_floating_point():
        comptime om = _opmath[dtype]()
        var x = a.cast[om]()
        var y = b.cast[om]()
        var mod = cuda_fmod(a, b)
        var div = (x - mod) / y
        var fixed = _nonzero(mod) & (_negative(y) ^ _negative(mod))
        div = fixed.select(div - 1, div)
        var r = _round_floor[dtype](div)
        # div == 0: copysign(scalar_t(0), a / b), the quotient in scalar_t
        var q = a / b
        r = _nonzero(div).select(r, _signed_zero[dtype](q))
        return _nonzero(b).select(r, q)
    else:
        var qr = cuda_int_divrem(a, b)
        var q = qr[0]
        comptime if dtype.is_signed():
            var zero = SIMD[dtype, w](0)
            var adjust = (a.lt(zero) ^ b.lt(zero)) & qr[1].ne(zero)
            return adjust.select(q - 1, q)
        else:
            return q


@always_inline
def _round_floor[
    dtype: DType, om: DType, w: Int
](div: SIMD[om, w]) -> SIMD[dtype, w]:
    """`floordiv = floor(div); if (div - floordiv > 0.5) floordiv += 1;`
    with `floordiv` a `scalar_t` local: rounded to the half type at each
    assignment."""
    var f = cast_rounded[dtype](floor(div))
    var fo = f.cast[om]()
    var up = (div - fo).gt(SIMD[om, w](0.5))
    return up.select(cast_rounded[dtype](fo + 1), f)


@always_inline
def _signed_zero[
    dtype: DType, sd: DType, w: Int
](s: SIMD[sd, w]) -> SIMD[dtype, w]:
    """A zero of dtype `dtype` with `s`'s sign bit (`copysign(0, s)`)."""
    comptime sbt = _bits_dtype[sd]()
    comptime ssign = Scalar[sbt](1) << Scalar[sbt](size_of[sd]() * 8 - 1)
    var neg = (bitcast[sbt, w](s) & ssign).ne(0)
    return neg.select(SIMD[dtype, w](-0.0), SIMD[dtype, w](0.0))


@always_inline
def cuda_remainder[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """remainder_kernel_cuda: C's remainder, plus the divisor when it is
    nonzero and its sign differs from the divisor's (in opmath, rounded
    once, for the half types)."""
    comptime if dtype.is_floating_point():
        comptime om = _opmath[dtype]()
        var mod = cuda_fmod(a, b)
        var y = b.cast[om]()
        var fixed = _nonzero(mod) & (_negative(y) ^ _negative(mod))
        return cast_rounded[dtype](fixed.select(mod + y, mod))
    else:
        var r = cuda_int_divrem(a, b)[1]
        var zero = SIMD[dtype, w](0)
        var fixed = r.ne(zero) & (r.lt(zero) ^ b.lt(zero))
        return fixed.select(r + b, r)


@always_inline
def cuda_fmod_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """fmod_kernel_cuda: C's `::fmod` (a half pair in float, rounded once),
    C's `%` for integers."""
    comptime if dtype.is_floating_point():
        return cuda_fmod(a, b).cast[dtype]()
    else:
        return cuda_int_divrem(a, b)[1]


@always_inline
def cuda_mul_by_inverse[
    dtype: DType, w: Int
](a: SIMD[dtype, w], inv: Float64) -> SIMD[dtype, w]:
    """div_true_kernel_cuda by a CPU scalar: `a * inv_b` in opmath, `inv_b`
    the reciprocal computed in double by the host (its float64 NaN first:
    a uniform operand sits in DMUL's B slot)."""
    comptime om = _opmath[dtype]()
    var x = a.cast[om]()
    var iv = SIMD[om, w](inv.cast[om]())
    return cast_rounded[dtype](nan_first2(x * iv, iv, x))


@always_inline
def cuda_trunc_div_by_scalar[
    dtype: DType, om: DType, w: Int
](x: SIMD[om, w], inv: Scalar[om]) -> SIMD[dtype, w]:
    """div_trunc_kernel_cuda by a CPU scalar: `trunc(a * inv_b)`, `inv_b`
    = acc(1) / acc(b), `x` the numerator in opmath `om`."""
    var iv = SIMD[om, w](inv)
    return cast_rounded[dtype](nan_first2(x * iv, iv, x).__trunc__())


@always_inline
def cuda_floor_div_by_scalar[
    dtype: DType, om: DType, w: Int
](x: SIMD[om, w], b: Scalar[om], inv: Scalar[om]) -> SIMD[dtype, w]:
    """div_floor_kernel_cuda by a nonzero CPU scalar `b` (`inv` = acc(1) /
    b; a zero `b` takes the true-division route), `x` the numerator in
    opmath `om`."""
    var y = SIMD[om, w](b)
    var iv = SIMD[om, w](inv)
    var mod = rebind[SIMD[om, w]](cuda_fmod(x, y))
    var div = nan_first2(_mul_rn(x - mod, iv), iv, iv)
    var fixed = _nonzero(mod) & (_negative(y) ^ _negative(mod))
    div = fixed.select(div - 1, div)
    var r = _round_floor[dtype](div)
    return _nonzero(div).select(r, _signed_zero[dtype](x * iv))
