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


# thrust's exp / log of complex (CCCL 3.0, as stock torch 2.11+cu130 runs
# them in pow: thrust/detail/complex/cexp(f).h, clog(f).h -- FreeBSD msun),
# ported literally, bit tricks included.
#
# Third-party code (full notices and licence texts: THIRD_PARTY_NOTICES.txt
# at the repository root):
#   Copyright (c) 2011 David Schultz <das@FreeBSD.ORG> (cexp)
#   Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG> (clog)
#   SPDX-License-Identifier: BSD-2-Clause
#   Copyright 2008-2021 NVIDIA Corporation; Copyright 2013 Filipe RNC Maia
#   SPDX-License-Identifier: Apache-2.0


@always_inline
def _hi_word[dt: DType](x: Scalar[dt]) -> UInt32:
    """get_float_word (float) / get_high_word (double)."""
    comptime if dt == DType.float64:
        return UInt32(bitcast[DType.uint64](x.cast[DType.float64]()) >> 32)
    else:
        return bitcast[DType.uint32](x.cast[DType.float32]())


@always_inline
def _lo_word[dt: DType](x: Scalar[dt]) -> UInt32:
    comptime if dt == DType.float64:
        return UInt32(
            bitcast[DType.uint64](x.cast[DType.float64]()) & 0xFFFFFFFF
        )
    else:
        return 0


@always_inline
def _from_words[dt: DType](hi: UInt32, lo: UInt32) -> Scalar[dt]:
    """set_float_word (float) / insert_words (double)."""
    comptime if dt == DType.float64:
        return bitcast[DType.float64]((UInt64(hi) << 32) | UInt64(lo)).cast[
            dt
        ]()
    else:
        return bitcast[DType.float32](hi).cast[dt]()


@always_inline
def _frexp_exp[dt: DType](x: Scalar[dt], mut expt: Int) -> Scalar[dt]:
    """frexp_exp(f): exp(x - k ln2) with its exponent field reset."""
    # Modified from frexp_exp / frexp_expf, thrust/detail/complex/cexp.h,
    # cexpf.h (nvidia-cuda-cccl 13.0.85; FreeBSD lib/msun/src/k_exp.c).
    comptime if dt == DType.float64:
        var exp_x = _rexp(x - Scalar[dt](1246.97177782734161156))
        var hx = _hi_word(exp_x)
        expt = Int(hx >> 20) - (0x3FF + 1023) + 1799
        return _from_words[dt](
            (hx & 0xFFFFF) | UInt32((0x3FF + 1023) << 20), _lo_word(exp_x)
        )
    else:
        var exp_x = _rexp(x - Scalar[dt](162.88958740))
        var hx = _hi_word(exp_x)
        expt = Int(hx >> 23) - (0x7F + 127) + 235
        return _from_words[dt]((hx & 0x7FFFFF) | UInt32((0x7F + 127) << 23), 0)


@always_inline
def _ldexp_cexp[dt: DType](z: Cx[dt], expt_in: Int) -> Cx[dt]:
    # Modified from ldexp_cexp / ldexp_cexpf, thrust/detail/complex/cexp.h,
    # cexpf.h (nvidia-cuda-cccl 13.0.85; FreeBSD lib/msun/src/k_exp.c).
    var ex_expt = 0
    var exp_x = _frexp_exp(z.re, ex_expt)
    var expt = expt_in + ex_expt
    var half = expt // 2 if expt >= 0 else -((-expt) // 2)  # C's / truncates
    var scale1: Scalar[dt]
    var scale2: Scalar[dt]
    comptime if dt == DType.float64:
        scale1 = _from_words[dt](UInt32((0x3FF + half) << 20), 0)
        scale2 = _from_words[dt](UInt32((0x3FF + (expt - half)) << 20), 0)
    else:
        scale1 = _from_words[dt](UInt32((0x7F + half) << 23), 0)
        scale2 = _from_words[dt](UInt32((0x7F + (expt - half)) << 23), 0)
    var y = z.im
    return Cx[dt](
        _rcos(y) * exp_x * scale1 * scale2, _rsin(y) * exp_x * scale1 * scale2
    )


@always_inline
def cexp[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """thrust's cexp / cexpf."""
    # Modified from cexp / cexpf, thrust/detail/complex/cexp.h, cexpf.h
    # (nvidia-cuda-cccl 13.0.85; FreeBSD lib/msun/src/s_cexp.c, s_cexpf.c).
    var x = z.re
    var y = z.im
    var hy = _hi_word(y) & 0x7FFFFFFF
    var ly = _lo_word(y)
    if (hy | ly) == 0:
        return Cx[dt](_rexp(x), y)
    var hx = _hi_word(x)
    var lx = _lo_word(x)
    if ((hx & 0x7FFFFFFF) | lx) == 0:
        return Cx[dt](_rcos(y), _rsin(y))
    comptime INF_HI = UInt32(0x7FF00000) if dt == DType.float64 else UInt32(
        0x7F800000
    )
    if hy >= INF_HI:
        # (double: thrust tests lx here, not ly -- kept)
        if lx != 0 or (hx & 0x7FFFFFFF) != INF_HI:
            return Cx[dt](y - y, y - y)
        elif (hx & 0x80000000) != 0:
            return Cx[dt](Scalar[dt](0), Scalar[dt](0))
        else:
            return Cx[dt](x, y - y)
    comptime EXP_OVFL = UInt32(0x40862E42) if dt == DType.float64 else UInt32(
        0x42B17218
    )
    comptime CEXP_OVFL = UInt32(0x4096B8E4) if dt == DType.float64 else UInt32(
        0x43400074
    )
    if hx >= EXP_OVFL and hx <= CEXP_OVFL:
        return _ldexp_cexp(z, 0)
    var exp_x = _rexp(x)
    return Cx[dt](exp_x * _rcos(y), exp_x * _rsin(y))


@always_inline
def _trim[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """Round down to a third of the mantissa bits."""
    # Modified from trim, thrust/detail/complex/clog.h, clogf.h
    # (nvidia-cuda-cccl 13.0.85; FreeBSD lib/msun/src/s_clog.c).
    comptime if dt == DType.float64:
        return _from_words[dt](_hi_word(x) & 0xFFFFFFF8, 0)
    else:
        return _from_words[dt](_hi_word(x) & 0xFFFF0000, 0)


@always_inline
def clog[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """thrust's clog / clogf. Every a*b + c*d is the fused multiply-add nvcc
    forms for it: fma(a, b, c*d)."""
    # Modified from clog / clogf, thrust/detail/complex/clog.h, clogf.h
    # (nvidia-cuda-cccl 13.0.85; FreeBSD lib/msun/src/s_clog.c, s_clogf.c).
    var x = z.re
    var y = z.im
    if isnan(x) or isnan(y):
        return Cx[dt](_rlog(x.fma(x, y * y)), catan2(y, x))
    var ax = abs(x)
    var ay = abs(y)
    if ax < ay:
        var t = ax
        ax = ay
        ay = t
    comptime BIG = Scalar[dt](5e307) if dt == DType.float64 else Scalar[dt](
        1e34
    )
    comptime TINY1 = Scalar[dt](1e-150) if dt == DType.float64 else Scalar[dt](
        1e-19
    )
    comptime LO = Scalar[dt](1e-50) if dt == DType.float64 else Scalar[dt](1e-6)
    comptime HI = Scalar[dt](1e50) if dt == DType.float64 else Scalar[dt](1e6)
    var one = Scalar[dt](1)
    var half = Scalar[dt](0.5)
    if ay > BIG:
        var e = Scalar[dt](2.7182818284590452354)
        return Cx[dt](_rlog(chypot(x / e, y / e)) + one, catan2(y, x))
    if ax == one:
        if ay < TINY1:
            return Cx[dt]((ay * half) * ay, catan2(y, x))
        return Cx[dt](_rlog1p(ay * ay) * half, catan2(y, x))
    if ax < LO or ay < LO or ax > HI or ay > HI:
        return Cx[dt](_rlog(chypot(x, y)), catan2(y, x))
    if ax >= one:
        return Cx[dt](
            _rlog1p(ay.fma(ay, (ax - one) * (ax + one))) * half, catan2(y, x)
        )
    if ax.fma(ax, ay * ay) <= Scalar[dt](0.7):
        return Cx[dt](_rlog(ax.fma(ax, ay * ay)) * half, catan2(y, x))
    var x0 = _trim(ax)
    ax = ax - x0
    var x1 = _trim(ax)
    var x2 = ax - x1
    var y0 = _trim(ay)
    ay = ay - y0
    var y1 = _trim(ay)
    var y2 = ay - y1
    var two = Scalar[dt](2)
    var val = SIMD[dt, 16](0)
    val[0] = x0 * x0
    val[1] = y0 * y0
    val[2] = two * x0 * x1
    val[3] = two * y0 * y1
    val[4] = x1 * x1
    val[5] = y1 * y1
    val[6] = two * x0 * x2
    val[7] = two * y0 * y2
    val[8] = two * x1 * x2
    val[9] = two * y1 * y2
    val[10] = x2 * x2
    val[11] = y2 * y2
    var sorted = False
    while not sorted:
        sorted = True
        for i in range(11):
            if val[i] < val[i + 1]:
                sorted = False
                var t = val[i]
                val[i] = val[i + 1]
                val[i + 1] = t
    var hm1 = -one
    for i in range(12):
        hm1 += val[i]
    return Cx[dt](half * _rlog1p(hm1), catan2(y, x))


# ---------------------------------------------------------------------------
# libc++ <complex> (the jiterator's), ported literally
#
# Third-party code: LLVM libc++ <complex>, as copied with modifications into
# PyTorch's aten/src/ATen/cuda/llvm_complex.cpp (v2.11.0). Part of the LLVM
# Project, under the Apache License v2.0 with LLVM Exceptions; see
# THIRD_PARTY_NOTICES.txt. SPDX-License-Identifier: Apache-2.0 WITH
# LLVM-exception. PyTorch's modifications: Copyright (c) 2016- Facebook,
# Inc. and the PyTorch contributors, SPDX-License-Identifier: BSD-3-Clause
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
    # Modified from libc++ logb (libc++ <complex> helper, llvm_complex.cpp) (PyTorch v2.11.0; LLVM licence).
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
    # Modified from libc++ scalbn (libc++ <complex> helper, llvm_complex.cpp) (PyTorch v2.11.0; LLVM licence).
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
    # Modified from libc++ operator/(const complex&, const complex&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
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
    # Modified from libc++ exp(const complex&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
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
    # Modified from libc++ log(const complex&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
    return Cx[dt](_rlog(chypot(z.re, z.im)), catan2(z.im, z.re))


@always_inline
def _polar[dt: DType](rho: Scalar[dt], theta: Scalar[dt]) -> Cx[dt]:
    """libc++'s polar(rho, theta)."""
    # Modified from libc++ polar(const _Tp&, const _Tp&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
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
    # Modified from libc++ sqrt(const complex&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
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
    """UnaryFractionKernels.cu's reciprocal_wrapper: NaN + NaN i for a NaN
    part or two infinite ones, 0 for one infinite part, else 1 / z (c10's
    division)."""
    # Modified from reciprocal_wrapper, PyTorch v2.11.0
    # aten/src/ATen/native/cuda/UnaryFractionKernels.cu (BSD-3-Clause; see
    # THIRD_PARTY_NOTICES.txt).
    if isnan(z.re) or isnan(z.im) or (isinf(z.re) and isinf(z.im)):
        return Cx[dt](nan[dt](), nan[dt]())
    if isinf(z.re) or isinf(z.im):
        return Cx[dt](Scalar[dt](0), Scalar[dt](0))
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
