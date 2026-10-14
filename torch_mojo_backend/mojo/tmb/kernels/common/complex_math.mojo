"""Complex arithmetic on (re, im) pairs of a real dtype (float32 / float64).

Mojo has no complex DType, so a complex64 / complex128 tensor is, to the
kernels, its interleaved float32 / float64 storage (torch's layout), and
every function here takes and returns the two components as scalars. Each
follows the CUDA code stock PyTorch runs for the complex dtypes
(c10/util/complex.h's operators, c10/util/complex_math.h, which forwards to
thrust's complex math, itself the FreeBSD msun routines): two implementations
coexist there, and so here:

* the jiterated CUDA kernels (exp, log, sqrt, rsqrt, sgn: the default on
  CUDA builds; mul and div only for complex32) compile the complex math of
  aten/src/ATen/cuda/llvm_complex.cpp, libc++'s <complex>: the `_jit`
  functions (Annex G recovery in /, polar-form sqrt, plain log|z|);
* the gpu_kernel ones (mul, div, reciprocal, pow, x * x, polar) use
  c10::complex's operators and thrust's math (FreeBSD msun's cexp / clog):
  `cmul`, `cdiv` (c10's scaled division, numpy's), `cexp`, `clog`, `cpow`.

Checked against stock torch on CUDA over a grid of signed zeros, ones,
1e30, infinities and NaNs (bit-level special values, both dtypes).

The real building blocks are the accurate per-lane ones the pointwise
family uses (`_exp` / `_log` / `_log1p`, the libdevice ports; `_pcos` /
`_psin`; `pointwise["hypot"]` / `pointwise["atan2"]`, which are what
torch.hypot / torch.atan2 compute on this device). The GPU build assumes no
NaN in comparisons, so every special case tests `isnan` / `isinf` (bit
tests that survive the fast-math flags) rather than a comparison.
"""

from std.memory import bitcast
from std.utils.numerics import inf, isinf, isnan, nan

from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.pointwise_math import (
    _exp,
    _log,
    _log1p,
    _pcos,
    _psin,
    param_dtype,
    pointwise,
)


@fieldwise_init
struct Cx[dt: DType](Copyable, Movable):
    """One complex value as its two components."""

    var re: Scalar[Self.dt]
    var im: Scalar[Self.dt]


@always_inline
def _copysign[dt: DType](mag: Scalar[dt], sgn: Scalar[dt]) -> Scalar[dt]:
    """C's copysign on the bits (the sign of a NaN or a -0.0 included)."""
    comptime if dt == DType.float64:
        var m = bitcast[DType.uint64](mag) & 0x7FFFFFFFFFFFFFFF
        var s = bitcast[DType.uint64](sgn) & 0x8000000000000000
        return bitcast[dt](m | s)
    else:
        var m = bitcast[DType.uint32](mag.cast[DType.float32]()) & 0x7FFFFFFF
        var s = bitcast[DType.uint32](sgn.cast[DType.float32]()) & 0x80000000
        return bitcast[DType.float32](m | s).cast[dt]()


@always_inline
def _signbit[dt: DType](x: Scalar[dt]) -> Bool:
    comptime if dt == DType.float64:
        return (bitcast[DType.uint64](x) >> 63) != 0
    else:
        return (bitcast[DType.uint32](x.cast[DType.float32]()) >> 31) != 0


@always_inline
def _is_zero[dt: DType](x: Scalar[dt]) -> Bool:
    """x == +-0.0, on the bits (no NaN-folding comparison)."""
    comptime if dt == DType.float64:
        return (bitcast[DType.uint64](x) & 0x7FFFFFFFFFFFFFFF) == 0
    else:
        return (
            bitcast[DType.uint32](x.cast[DType.float32]()) & 0x7FFFFFFF
        ) == 0


@always_inline
def _rexp[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return _exp[dt, 1](x)


@always_inline
def _rlog[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return _log[dt, 1](x)


@always_inline
def _rlog1p[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return _log1p[dt, 1](x)


@always_inline
def _rcos[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return _pcos[DType.float64](x.cast[DType.float64]()).cast[dt]()
    else:
        return _pcos[DType.float32](x.cast[DType.float32]()).cast[dt]()


@always_inline
def _rsin[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return _psin[DType.float64](x.cast[DType.float64]()).cast[dt]()
    else:
        return _psin[DType.float32](x.cast[DType.float32]()).cast[dt]()


@always_inline
def chypot[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """`abs(a + bi)`: thrust's `abs` is `hypot(re, im)`, and this is the
    device's torch.hypot."""
    var z = Scalar[dt](0)
    return pointwise["hypot", dt, dt, 1](a, b, z, SIMD[param_dtype[dt](), 4](0))


@always_inline
def catan2[dt: DType](y: Scalar[dt], x: Scalar[dt]) -> Scalar[dt]:
    """`arg(x + yi)` = atan2(y, x), the device's torch.atan2."""
    var z = Scalar[dt](0)
    return pointwise["atan2", dt, dt, 1](y, x, z, SIMD[param_dtype[dt](), 4](0))


@always_inline
def cmul[dt: DType](a: Cx[dt], b: Cx[dt]) -> Cx[dt]:
    """c10::complex operator*: (ac - bd) + (ad + bc)i, with the fused
    multiply-adds nvcc forms for it (measured bit-exact against stock CUDA
    on 200k complex64 / 5k complex128 pairs): re = fma(a, c, -(b d)),
    im = fma(a, d, b c). Spelled out so no contraction choice of this
    compiler can differ -- it decides overflow signs (inf - inf)."""
    return Cx[dt](a.re.fma(b.re, -(a.im * b.im)), a.re.fma(b.im, a.im * b.re))


@always_inline
def cdiv[dt: DType](x: Cx[dt], y: Cx[dt]) -> Cx[dt]:
    """c10::complex operator/= (numpy's scaled division): divide through by
    the larger of |c|, |d| so the squares never overflow."""
    var a = x.re
    var b = x.im
    var c = y.re
    var d = y.im
    # std::abs, as CUDA (and gcc builds of torch) compile it: +inf from a
    # division by -0.0 (clang builds spell it `x < 0 ? -x : x` and get -inf).
    var abs_c = abs(c)
    var abs_d = abs(d)
    # `abs_c >= abs_d` is false for a NaN in either: the else branch, as in
    # C++, which then propagates the NaN.
    var c_ge = not (isnan(c) or isnan(d)) and abs_c >= abs_d
    if c_ge:
        if _is_zero(c) and _is_zero(d):
            # divide by zeros should yield a complex inf or nan
            return Cx[dt](a / abs_c, b / abs_d)
        var rat = d / c
        # each multiply-add one fma, as nvcc contracts c10's expressions
        var scl = Scalar[dt](1) / d.fma(rat, c)
        return Cx[dt](b.fma(rat, a) * scl, (-a).fma(rat, b) * scl)
    var rat = c / d
    var scl = Scalar[dt](1) / c.fma(rat, d)
    return Cx[dt](a.fma(rat, b) * scl, b.fma(rat, -a) * scl)


@always_inline
def _overflow_threshold[dt: DType]() -> Scalar[dt]:
    """Where exp(x) overflows (msun's cexp_ovfl)."""
    comptime if dt == DType.float64:
        return Scalar[dt](709.782712893383973096)
    else:
        return Scalar[dt](88.7228391116729996)


@always_inline
def cexp[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """msun's cexp / cexpf (thrust's complex exp), Annex G special values."""
    var x = z.re
    var y = z.im
    if _is_zero(y):
        # exp(x + 0i) = exp(x) + 0i, the zero's sign kept
        return Cx[dt](_rexp(x), y)
    if _is_zero(x):
        return Cx[dt](_rcos(y), _rsin(y))
    if isinf(y) or isnan(y):
        # y is not finite
        if isinf(x) and not _signbit(x):
            # cexp(+inf + i y) = +inf + NaN i  (y +-inf or NaN)
            return Cx[dt](x, y - y)
        if isinf(x):
            # cexp(-inf + i y) = +-0 +- 0i
            return Cx[dt](Scalar[dt](0), Scalar[dt](0))
        # cexp(finite|NaN +- i inf|NaN) = NaN + NaN i
        return Cx[dt](nan[dt](), nan[dt]())
    var c = _rcos(y)
    var s = _rsin(y)
    if not isnan(x) and not isinf(x) and x > _overflow_threshold[dt]():
        # exp(x) alone overflows while exp(x) * cos(y) may not: halve it.
        var h = _rexp(x * Scalar[dt](0.5))
        return Cx[dt]((h * c) * h, (h * s) * h)
    var e = _rexp(x)
    if isinf(e):
        # x = +inf (y finite, nonzero): inf * cos(y) / inf * sin(y)
        return Cx[dt](e * c, e * s)
    return Cx[dt](e * c, e * s)


@always_inline
def clog[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """log|z| + i arg(z). Near |z| = 1, where log|z| is small and the plain
    log(hypot) loses its relative precision, log|z| comes from log1p of
    |z|^2 - 1 computed as (x - 1)(x + 1) + y^2 about the larger component."""
    var x = z.re
    var y = z.im
    var ang = catan2(y, x)
    if isnan(x) or isnan(y):
        if isinf(x) or isinf(y):
            return Cx[dt](inf[dt](), ang)
        return Cx[dt](nan[dt](), nan[dt]())
    var h = chypot(x, y)
    var ax = abs(x)
    var ay = abs(y)
    var big = max(ax, ay)
    var small = min(ax, ay)
    if (
        big > Scalar[dt](0.5)
        and big < Scalar[dt](1.5)
        and h > Scalar[dt](0.70710678)
        and h < Scalar[dt](1.25)
    ):
        var t = (big - 1) * (big + 1) + small * small
        return Cx[dt](_rlog1p(t) * Scalar[dt](0.5), ang)
    if isinf(h) and not isinf(x) and not isinf(y):
        # |z| overflows the component type though z is finite (thrust's
        # clog scales): log|z| = log|z / 2| + log 2, so [3e38+3e38j] ** 0.1
        # stays finite as on CUDA
        var half = Scalar[dt](0.5)
        return Cx[dt](
            _rlog(chypot(x * half, y * half))
            + Scalar[dt](0.69314718055994530942),
            ang,
        )
    return Cx[dt](_rlog(h), ang)


# ---------------------------------------------------------------------------
# libc++ <complex> (the jiterator's), ported literally
# ---------------------------------------------------------------------------


@always_inline
def _inf[dt: DType]() -> Scalar[dt]:
    return inf[dt]()


@always_inline
def _unit_sign[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """copysign(isinf(x) ? 1 : 0, x)."""
    return _copysign(Scalar[dt](1) if isinf(x) else Scalar[dt](0), x)


@always_inline
def _logb[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """C's logb for x >= 0: the unbiased exponent (subnormals included),
    -inf at 0, inf at inf, NaN at NaN."""
    if isnan(x):
        return x
    if isinf(x):
        return _inf[dt]()
    if _is_zero(x):
        return -_inf[dt]()
    comptime if dt == DType.float64:
        var bits = bitcast[DType.uint64](x)
        var e = Int((bits >> 52) & 0x7FF)
        if e == 0:
            # subnormal: normalize
            var m = bits & 0xFFFFFFFFFFFFF
            var k = 0
            while (m & 0x10000000000000) == 0:
                m <<= 1
                k += 1
            return Scalar[dt](-1022 - k)
        return Scalar[dt](e - 1023)
    else:
        var bits = bitcast[DType.uint32](x.cast[DType.float32]())
        var e = Int((bits >> 23) & 0xFF)
        if e == 0:
            var m = bits & 0x7FFFFF
            var k = 0
            while (m & 0x800000) == 0:
                m <<= 1
                k += 1
            return Scalar[dt](-126 - k)
        return Scalar[dt](e - 127)


@always_inline
def _scalbn[dt: DType](x: Scalar[dt], n: Int) -> Scalar[dt]:
    """x * 2^n, exact (in up to three in-range power-of-two steps)."""
    var r = x
    var k = n
    comptime lim = 1000 if dt == DType.float64 else 120
    while k > lim:
        r = r * _pow2[dt](lim)
        k -= lim
    while k < -lim:
        r = r * _pow2[dt](-lim)
        k += lim
    return r * _pow2[dt](k)


@always_inline
def _pow2[dt: DType](k: Int) -> Scalar[dt]:
    """2^k for a normal-range k."""
    comptime if dt == DType.float64:
        return bitcast[dt](UInt64(k + 1023) << 52)
    else:
        return bitcast[DType.float32](UInt32(k + 127) << 23).cast[dt]()


@always_inline
def cdiv_jit[dt: DType](z: Cx[dt], w: Cx[dt]) -> Cx[dt]:
    """libc++'s operator/(complex, complex): scaled by the larger exponent
    of the divisor (logb / scalbn), Annex G recovery."""
    var ilogbw = 0
    var a = z.re
    var b = z.im
    var c = w.re
    var d = w.im
    var big = abs(c)
    var ad_ = abs(d)
    if isnan(big) or (not isnan(ad_) and ad_ > big):
        big = ad_  # fmax: the non-NaN one
    var logbw = _logb(big)
    var finite_logbw = not isnan(logbw) and not isinf(logbw)
    if finite_logbw:
        ilogbw = Int(logbw)
        c = _scalbn(c, -ilogbw)
        d = _scalbn(d, -ilogbw)
    var denom = c * c + d * d
    var x = _scalbn((a * c + b * d) / denom, -ilogbw)
    var y = _scalbn((b * c - a * d) / denom, -ilogbw)
    if isnan(x) and isnan(y):
        if _is_zero(denom) and (not isnan(a) or not isnan(b)):
            var s = _copysign(_inf[dt](), c)
            x = s * a
            y = s * b
        elif (
            (isinf(a) or isinf(b))
            and not (isinf(c) or isnan(c))
            and not (isinf(d) or isnan(d))
        ):
            a = _unit_sign(a)
            b = _unit_sign(b)
            x = _inf[dt]() * (a * c + b * d)
            y = _inf[dt]() * (b * c - a * d)
        elif (
            isinf(logbw)
            and not _signbit(logbw)
            and not (isinf(a) or isnan(a))
            and not (isinf(b) or isnan(b))
        ):
            c = _unit_sign(c)
            d = _unit_sign(d)
            x = Scalar[dt](0) * (a * c + b * d)
            y = Scalar[dt](0) * (b * c - a * d)
    return Cx[dt](x, y)


@always_inline
def cexp_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """libc++'s exp(complex)."""
    var i = z.im
    if _is_zero(i):
        return Cx[dt](_rexp(z.re), _copysign(Scalar[dt](0), i))
    if isinf(z.re):
        if _signbit(z.re):
            if isinf(i) or isnan(i):
                i = Scalar[dt](1)
        elif isinf(i) or isnan(i):
            if isinf(i):
                i = nan[dt]()
            return Cx[dt](z.re, i)
    var e = _rexp(z.re)
    return Cx[dt](e * _rcos(i), e * _rsin(i))


@always_inline
def clog_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """libc++'s log(complex): log(abs(z)) + i arg(z)."""
    return Cx[dt](_rlog(chypot(z.re, z.im)), catan2(z.im, z.re))


@always_inline
def _polar[dt: DType](rho: Scalar[dt], theta: Scalar[dt]) -> Cx[dt]:
    """libc++'s polar(rho, theta)."""
    if isnan(rho) or _signbit(rho):
        return Cx[dt](nan[dt](), nan[dt]())
    if isnan(theta):
        if isinf(rho):
            return Cx[dt](rho, theta)
        return Cx[dt](theta, theta)
    if isinf(theta):
        if isinf(rho):
            return Cx[dt](rho, nan[dt]())
        return Cx[dt](nan[dt](), nan[dt]())
    var x = rho * _rcos(theta)
    if isnan(x):
        x = Scalar[dt](0)
    var y = rho * _rsin(theta)
    if isnan(y):
        y = Scalar[dt](0)
    return Cx[dt](x, y)


@always_inline
def csqrt_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """libc++'s sqrt(complex): the polar form, Annex G's infinities."""
    if isinf(z.im):
        return Cx[dt](_inf[dt](), z.im)
    if isinf(z.re):
        if not _signbit(z.re):
            return Cx[dt](
                z.re, z.im if isnan(z.im) else _copysign(Scalar[dt](0), z.im)
            )
        return Cx[dt](
            z.im if isnan(z.im) else Scalar[dt](0), _copysign(z.re, z.im)
        )
    return _polar(
        ieee_sqrt(chypot(z.re, z.im)), catan2(z.im, z.re) / Scalar[dt](2)
    )


@always_inline
def creciprocal[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """UnaryOpsKernel.cu's complex reciprocal: `scalar_t(1) / a`."""
    return cdiv(Cx[dt](Scalar[dt](1), Scalar[dt](0)), z)


@always_inline
def csgn[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """`sgn_impl`: 0 at 0, else z / |z| (a complex over a real)."""
    if _is_zero(z.re) and _is_zero(z.im):
        return Cx[dt](Scalar[dt](0), Scalar[dt](0))
    var r = chypot(z.re, z.im)
    if isinf(r):
        # x / inf: NaN for an infinite part, a signed zero for a finite one
        # (spelled out: the build may assume no infinities in a division)
        var re = nan[dt]() if isinf(z.re) or isnan(z.re) else _copysign(
            Scalar[dt](0), z.re
        )
        var im = nan[dt]() if isinf(z.im) or isnan(z.im) else _copysign(
            Scalar[dt](0), z.im
        )
        return Cx[dt](re, im)
    return Cx[dt](z.re / r, z.im / r)


@always_inline
def cpow[dt: DType](x: Cx[dt], y: Cx[dt]) -> Cx[dt]:
    """thrust's `pow(complex, complex)` (c10_complex_math::pow on CUDA):
    exp(log(x) * y), special values included (a zero base gives
    exp(-inf + NaN i) = 0 for a nonzero real exponent)."""
    return cexp(cmul(clog(x), y))
