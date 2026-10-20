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

from std.sys.info import is_nvidia_gpu

from std.sys._assembly import inlined_assembly

from tmb.kernels.common.cuda_math import (
    cuda_acosf,
    cuda_atan2f,
    nv_asinf,
    nv_coshf,
    nv_cosf,
    nv_sinf,
    nv_sinhf,
)
from tmb.kernels.common.libdevice_port import (
    nv_acos,
    nv_asin,
    nv_atan2,
    nv_cosh,
    nv_sincos,
    nv_sinh,
    nv_tan,
    nv_tanf,
)
from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.pointwise_math import (
    _exp,
    _expm1,
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
def _jnan[dt: DType]() -> Scalar[dt]:
    """The jiterator's `NAN`, `__int_as_float(0x7fffffff)`: a float even
    where it becomes a double."""
    return bitcast[DType.float32](UInt32(0x7FFFFFFF)).cast[dt]()


@always_inline
def _hwnan[dt: DType]() -> Scalar[dt]:
    """The NaN an NVIDIA invalid operation (inf / inf, 0 * inf) produces."""
    comptime if dt == DType.float64:
        return bitcast[DType.float64](UInt64(0xFFF8000000000000)).cast[dt]()
    else:
        return bitcast[DType.float32](UInt32(0x7FFFFFFF)).cast[dt]()


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
    comptime if dt == DType.float64 and is_nvidia_gpu():
        return rebind[Scalar[dt]](nv_sincos(rebind[Float64](x))[1])
    elif dt == DType.float64:
        return _pcos[DType.float64](x.cast[DType.float64]()).cast[dt]()
    else:
        return _pcos[DType.float32](x.cast[DType.float32]()).cast[dt]()


@always_inline
def _rsin[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64 and is_nvidia_gpu():
        return rebind[Scalar[dt]](nv_sincos(rebind[Float64](x))[0])
    elif dt == DType.float64:
        return _psin[DType.float64](x.cast[DType.float64]()).cast[dt]()
    else:
        return _psin[DType.float32](x.cast[DType.float32]()).cast[dt]()


@always_inline
def chypot[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """`abs(a + bi)` = hypot(re, im), the device's torch.hypot, except for a
    NaN on NVIDIA, which follows libdevice's `__nv_hypot(f)`: its |.| are
    adds with an |x| operand (a float NaN becomes 0x7FFFFFFF, a double one
    is quieted, its sign kept) and the bitwise smaller magnitude wins when it
    is not finite (hypot(inf, NaN) = inf)."""
    var z = Scalar[dt](0)
    var r = pointwise["hypot", dt, dt, 1](
        a, b, z, SIMD[param_dtype[dt](), 4](0)
    )
    comptime if is_nvidia_gpu():
        if isnan(a) or isnan(b):
            comptime if dt == DType.float64:
                var ua = _habs_bits(a)
                var ub = _habs_bits(b)
                var mn = min(ua, ub)
                if mn >= 0x7FF0000000000000:
                    return bitcast[dt](mn)
                return bitcast[dt](max(ua, ub))
            else:
                if isinf(a) or isinf(b):
                    return _inf[dt]()
                return bitcast[DType.float32](UInt32(0x7FFFFFFF)).cast[dt]()
    return r


@always_inline
def _habs_bits[dt: DType](x: Scalar[dt]) -> UInt64:
    """The bits of a float64 |x| as an add with an |x| operand makes them."""
    var b = bitcast[DType.uint64](x.cast[DType.float64]())
    if (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000:
        return b | 0x0008000000000000
    return b & 0x7FFFFFFFFFFFFFFF


@always_inline
def catan2_nvcc[dt: DType](y: Scalar[dt], x: Scalar[dt]) -> Scalar[dt]:
    """atan2 as nvcc's math library returns it in the gpu_kernel complex ops
    (log1p, thrust's catrig): a float64 NaN comes out positive."""
    comptime if dt == DType.float64 and is_nvidia_gpu():
        return rebind[Scalar[dt]](
            nv_atan2[keep_nan_sign=False](
                rebind[Float64](y), rebind[Float64](x)
            )
        )
    else:
        return catan2(y, x)


@always_inline
def catan2[dt: DType](y: Scalar[dt], x: Scalar[dt]) -> Scalar[dt]:
    """`arg(x + yi)` = atan2(y, x): libdevice's `__nv_atan2` for float64 on
    NVIDIA (what the jiterated kernels call), else the device's
    torch.atan2."""
    comptime if dt == DType.float64 and is_nvidia_gpu():
        return rebind[Scalar[dt]](
            nv_atan2(rebind[Float64](y), rebind[Float64](x))
        )
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
            x = _hmul(s, a)
            y = _hmul(s, b)
        elif (
            (isinf(a) or isinf(b))
            and not (isinf(c) or isnan(c))
            and not (isinf(d) or isnan(d))
        ):
            a = _unit_sign(a)
            b = _unit_sign(b)
            x = _hmul(_inf[dt](), a * c + b * d)
            y = _hmul(_inf[dt](), b * c - a * d)
        elif (
            isinf(logbw)
            and not _signbit(logbw)
            and not (isinf(a) or isnan(a))
            and not (isinf(b) or isnan(b))
        ):
            c = _unit_sign(c)
            d = _unit_sign(d)
            x = _hmul(Scalar[dt](0), a * c + b * d)
            y = _hmul(Scalar[dt](0), b * c - a * d)
    return Cx[dt](x, y)


@always_inline
def _quiet_f64[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """A float64 NaN quieted with its sign and payload kept, as an add with an
    operand NaN leaves it on NVIDIA."""
    return bitcast[dt](bitcast[DType.uint64](x) | 0x0008000000000000)


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
                i = _jnan[dt]()
            return Cx[dt](z.re, i)
    comptime if dt == DType.float64 and is_nvidia_gpu():
        # A NaN real part with an imaginary part that is not zero: stock CUDA
        # returns that NaN, quieted with its sign and payload, in both parts
        # (the product of NaNs picks the other factor's NaN here).
        if isnan(z.re):
            var q = _quiet_f64(z.re)
            return Cx[dt](q, q)
    var e = _rexp(z.re)
    return Cx[dt](_hmul(e, _rcos(i)), _hmul(e, _rsin(i)))


@always_inline
def cangle[dt: DType](z: Cx[dt]) -> Scalar[dt]:
    """arg(z) = atan2(im, re); on NVIDIA double a NaN imaginary part is the
    result, quieted, as CUDA's atan2 returns its y operand's NaN."""
    comptime if dt == DType.float64 and is_nvidia_gpu():
        if isnan(z.im):
            return _quiet_f64(z.im)
    return catan2(z.im, z.re)


@always_inline
def clog_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """libc++'s log(complex): log(abs(z)) + i arg(z)."""
    # Modified from libc++ log(const complex&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
    comptime if dt == DType.float64 and is_nvidia_gpu():
        # A NaN imaginary part is the result's imaginary part (CUDA's atan2
        # returns its y operand's NaN), quieted.
        if isnan(z.im):
            return Cx[dt](_rlog(chypot(z.re, z.im)), _quiet_f64(z.im))
    return Cx[dt](_rlog(chypot(z.re, z.im)), catan2(z.im, z.re))


@always_inline
def _polar[dt: DType](rho: Scalar[dt], theta: Scalar[dt]) -> Cx[dt]:
    """libc++'s polar(rho, theta)."""
    # Modified from libc++ polar(const _Tp&, const _Tp&), llvm_complex.cpp (PyTorch v2.11.0; LLVM licence).
    if isnan(rho) or _signbit(rho):
        return Cx[dt](_jnan[dt](), _jnan[dt]())
    if isnan(theta):
        if isinf(rho):
            return Cx[dt](rho, theta)
        return Cx[dt](theta, theta)
    if isinf(theta):
        if isinf(rho):
            return Cx[dt](rho, _jnan[dt]())
        return Cx[dt](_jnan[dt](), _jnan[dt]())
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
def _sgn_part_nan[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """A part of sgn of an infinite magnitude: a NaN part comes back quieted
    on NVIDIA double (CUDA's x / |z| keeps it), an infinite one is the
    hardware NaN."""
    comptime if dt == DType.float64 and is_nvidia_gpu():
        if isnan(x):
            return _quiet_f64(x)
    return _hwnan[dt]()


@always_inline
def csgn[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """`sgn_impl`: 0 at 0, else z / |z| (a complex over a real)."""
    if _is_zero(z.re) and _is_zero(z.im):
        return Cx[dt](Scalar[dt](0), Scalar[dt](0))
    var r = chypot(z.re, z.im)
    if isinf(r):
        # x / inf: NaN for an infinite part, a signed zero for a finite one
        # (spelled out: the build may assume no infinities in a division)
        var re = _sgn_part_nan[dt](z.re) if isinf(z.re) or isnan(
            z.re
        ) else _copysign(Scalar[dt](0), z.re)
        var im = _sgn_part_nan[dt](z.im) if isinf(z.im) or isnan(
            z.im
        ) else _copysign(Scalar[dt](0), z.im)
        return Cx[dt](re, im)
    return Cx[dt](z.re / r, z.im / r)


@always_inline
def cpow[dt: DType](x: Cx[dt], y: Cx[dt]) -> Cx[dt]:
    """thrust's `pow(complex, complex)` (c10_complex_math::pow on CUDA):
    exp(log(x) * y), special values included (a zero base gives
    exp(-inf + NaN i) = 0 for a nonzero real exponent)."""
    return cexp(cmul(clog(x), y))


# --------------------------------------------------------------------------- #
# The jiterated complex transcendentals (cos, sin, tan, their hyperbolic and
# inverse forms, log2 / log10, exp2): libc++'s <complex> of
# aten/src/ATen/cuda/llvm_complex.cpp, compiled by NVRTC (--fmad=true), over
# the libdevice real routines. NVIDIA only (`_nv_*` real helpers). `NAN` is
# the jiterator's `__int_as_float(0x7fffffff)`, a float even for double.
# --------------------------------------------------------------------------- #


@always_inline
def _isfin[dt: DType](x: Scalar[dt]) -> Bool:
    return not (isinf(x) or isnan(x))


@always_inline
def neg_like_cuda[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """C's -x as CUDA compiles it: a sign-bit flip for float (a NaN's sign
    flips too), the hardware negation for double."""
    comptime if dt == DType.float64:
        return -x
    else:
        return bitcast[dt](
            bitcast[DType.uint32](x.cast[DType.float32]()) ^ 0x80000000
        ).cast[dt]()


@always_inline
def _jabs[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """fabs as CUDA's SASS computes it, an add with an |x| operand: a float
    NaN comes out 0x7FFFFFFF, a double one quieted with its sign kept
    (NVIDIA only; elsewhere the plain bit mask)."""
    comptime if not is_nvidia_gpu():
        comptime if dt == DType.float64:
            return bitcast[dt](bitcast[DType.uint64](x) & 0x7FFFFFFFFFFFFFFF)
        else:
            return bitcast[dt](
                bitcast[DType.uint32](x.cast[DType.float32]()) & 0x7FFFFFFF
            ).cast[dt]()
    comptime if dt == DType.float64:
        var b = bitcast[DType.uint64](x)
        if (b & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000:
            return bitcast[dt](b | 0x0008000000000000)
        return bitcast[dt](b & 0x7FFFFFFFFFFFFFFF)
    else:
        var b = bitcast[DType.uint32](x.cast[DType.float32]())
        if (b & 0x7FFFFFFF) > 0x7F800000:
            return bitcast[DType.float32](UInt32(0x7FFFFFFF)).cast[dt]()
        return bitcast[DType.float32](b & 0x7FFFFFFF).cast[dt]()


@always_inline
def _jpi[dt: DType]() -> Scalar[dt]:
    """`_Tp(atan2(+0., -0.))`: double pi, rounded to the component type."""
    return Float64(3.141592653589793).cast[dt]()


@always_inline
def _nv_sinh[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](nv_sinh(rebind[Float64](x)))
    else:
        return rebind[Scalar[dt]](nv_sinhf(rebind[Float32](x)))


@always_inline
def _nv_cosh[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](nv_cosh(rebind[Float64](x)))
    else:
        return rebind[Scalar[dt]](nv_coshf(rebind[Float32](x)))


@always_inline
def csinh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    if isinf(z.re) and not _isfin(z.im):
        return Cx[dt](z.re, _jnan[dt]())
    if _is_zero(z.re) and not _isfin(z.im):
        return Cx[dt](z.re, _jnan[dt]())
    if _is_zero(z.im) and not _isfin(z.re):
        return z.copy()
    return Cx[dt](_nv_sinh(z.re) * _rcos(z.im), _nv_cosh(z.re) * _rsin(z.im))


@always_inline
def ccosh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    if isinf(z.re) and not _isfin(z.im):
        return Cx[dt](_jabs(z.re), _jnan[dt]())
    if _is_zero(z.re) and not _isfin(z.im):
        return Cx[dt](_jnan[dt](), z.re)
    if _is_zero(z.re) and _is_zero(z.im):
        return Cx[dt](Scalar[dt](1), z.im)
    if _is_zero(z.im) and not _isfin(z.re):
        return Cx[dt](_jabs(z.re), z.im)
    return Cx[dt](_nv_cosh(z.re) * _rcos(z.im), _nv_sinh(z.re) * _rsin(z.im))


@always_inline
def _rot_in[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """complex(-im, re): i z."""
    return Cx[dt](neg_like_cuda(z.im), z.re)


@always_inline
def _rot_out[dt: DType](w: Cx[dt]) -> Cx[dt]:
    """complex(w.im, -w.re): -i w."""
    return Cx[dt](w.im, neg_like_cuda(w.re))


@always_inline
def csin_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    return _rot_out(csinh_jit(_rot_in(z)))


@always_inline
def ccos_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    return ccosh_jit(_rot_in(z))


@always_inline
def catanh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var pi = _jpi[dt]()
    if isinf(z.im):
        return Cx[dt](
            _copysign(Scalar[dt](0), z.re), _copysign(pi / Scalar[dt](2), z.im)
        )
    if isnan(z.im):
        if isinf(z.re) or _is_zero(z.re):
            return Cx[dt](_copysign(Scalar[dt](0), z.re), z.im)
        return Cx[dt](z.im, z.im)
    if isnan(z.re):
        return Cx[dt](z.re, z.re)
    if isinf(z.re):
        return Cx[dt](
            _copysign(Scalar[dt](0), z.re), _copysign(pi / Scalar[dt](2), z.im)
        )
    if _jabs(z.re) == Scalar[dt](1) and _is_zero(z.im):
        return Cx[dt](
            _copysign(_inf[dt](), z.re), _copysign(Scalar[dt](0), z.im)
        )
    var q = cdiv_jit(
        Cx[dt](Scalar[dt](1) + z.re, z.im),
        Cx[dt](Scalar[dt](1) - z.re, neg_like_cuda(z.im)),
    )
    var w = clog_jit(q)
    return Cx[dt](
        _copysign(w.re / Scalar[dt](2), z.re),
        _copysign(w.im / Scalar[dt](2), z.im),
    )


@always_inline
def catan_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    # the -imag of the rotation is an arithmetic negation in stock torch's
    # kernel (a float NaN comes out canonical)
    return _rot_out(catanh_jit(Cx[dt](_hsub(_flipz[dt](), z.im), z.re)))


@always_inline
def clog_base_jit[dt: DType, base: Int](z: Cx[dt]) -> Cx[dt]:
    """log(z) / log(_Tp(base)) (log10, log2)."""
    var w = clog_jit(z)
    var l = _rlog(Scalar[dt](base))
    return Cx[dt](w.re / l, w.im / l)


@always_inline
def cexp2_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """Math.cuh's exp2_impl of a complex: exp(_Tp(ln 2) * z)."""
    var l2 = Float64(0.693147180559945309417232121458176).cast[dt]()
    return cexp_jit(Cx[dt](l2 * z.re, l2 * z.im))


@always_inline
def cexpm1_c10[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """c10_complex_math::expm1 (a gpu_kernel, nvcc): (expm1(x) cos(y) -
    2 sin(y / 2)^2) + i e^x sin(y), the subtraction fused into the product
    as stock torch's SASS has it."""
    comptime if dt == DType.float64 and is_nvidia_gpu():
        # As cexp_jit: a NaN real part with an imaginary part that is not zero.
        if isnan(z.re) and not _is_zero(z.im):
            var q = _quiet_f64(z.re)
            return Cx[dt](q, q)
    var a = _rsin(z.im * Scalar[dt](0.5))
    var er = _hfma(_expm1[dt, 1](z.re), _rcos(z.im), (a * Scalar[dt](-2)) * a)
    return Cx[dt](er, _hmul(_rexp(z.re), _rsin(z.im)))


@always_inline
def clog1p_c10[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """c10_complex_math::log1p (a gpu_kernel, nvcc): theta = atan2(y, x + 1);
    below |z| = 0.5 log1p(x (2 + x) + y^2) / 2 (x itself when that is 0),
    else log(hypot(x + 1, y))."""
    var x = z.re
    var y = z.im
    var zabs = chypot(x, y)
    var theta = catan2_nvcc(y, x + Scalar[dt](1))
    comptime if dt == DType.float64 and is_nvidia_gpu():
        if isnan(x):
            theta = _quiet_f64(x)
        elif isnan(y):
            theta = _quiet_f64(y)
    if not isnan(zabs) and zabs < Scalar[dt](0.5):
        var r = y.fma(y, x * (Scalar[dt](2) + x))
        if _is_zero(r):
            return Cx[dt](x, theta)
        return Cx[dt](Scalar[dt](0.5) * _rlog1p(r), theta)
    return Cx[dt](_rlog(chypot(x + Scalar[dt](1), y)), theta)


@always_inline
def _jsqr_plus[dt: DType](z: Cx[dt], c: Scalar[dt]) -> Cx[dt]:
    """__sqr(z) + c: ((re - im)(re + im) + c, 2 re im), the product and the
    add contracted into one fma, as NVRTC compiles it."""
    return Cx[dt](
        (z.re - z.im).fma(z.re + z.im, c), (Scalar[dt](2) * z.re) * z.im
    )


@always_inline
def ctanh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    if isinf(z.re):
        if not _isfin(z.im):
            return Cx[dt](_copysign(Scalar[dt](1), z.re), Scalar[dt](0))
        return Cx[dt](
            _copysign(Scalar[dt](1), z.re),
            _copysign(Scalar[dt](0), _rsin(Scalar[dt](2) * z.im)),
        )
    if isnan(z.re) and _is_zero(z.im):
        return z.copy()
    var r2 = Scalar[dt](2) * z.re
    var i2 = Scalar[dt](2) * z.im
    var d = _nv_cosh(r2) + _rcos(i2)
    var rsh = _nv_sinh(r2)
    if isinf(rsh) and isinf(d):
        return Cx[dt](
            Scalar[dt](-1) if _signbit(rsh) else Scalar[dt](1),
            Scalar[dt](0) if (
                not isnan(i2) and not _signbit(i2) and not _is_zero(i2)
            ) else neg_like_cuda(Scalar[dt](0)),
        )
    return Cx[dt](rsh / d, _rsin(i2) / d)


@always_inline
def ctan_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    return _rot_out(ctanh_jit(_rot_in(z)))


@always_inline
def casinh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var pi = _jpi[dt]()
    if isinf(z.re):
        if isnan(z.im):
            return z.copy()
        if isinf(z.im):
            return Cx[dt](z.re, _copysign(pi * Scalar[dt](0.25), z.im))
        return Cx[dt](z.re, _copysign(Scalar[dt](0), z.im))
    if isnan(z.re):
        if isinf(z.im):
            return Cx[dt](z.im, z.re)
        if _is_zero(z.im):
            return z.copy()
        return Cx[dt](z.re, z.re)
    if isinf(z.im):
        return Cx[dt](
            _copysign(z.im, z.re), _copysign(pi / Scalar[dt](2), z.im)
        )
    var s = csqrt_jit(_jsqr_plus(z, Scalar[dt](1)))
    var w = clog_jit(Cx[dt](z.re + s.re, z.im + s.im))
    return Cx[dt](_copysign(w.re, z.re), _copysign(w.im, z.im))


@always_inline
def casin_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    return _rot_out(casinh_jit(_rot_in(z)))


@always_inline
def cacosh_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var pi = _jpi[dt]()
    if isinf(z.re):
        if isnan(z.im):
            return Cx[dt](_jabs(z.re), z.im)
        if isinf(z.im):
            if not _signbit(z.re):
                return Cx[dt](z.re, _copysign(pi * Scalar[dt](0.25), z.im))
            return Cx[dt](
                neg_like_cuda(z.re), _copysign(pi * Scalar[dt](0.75), z.im)
            )
        if _signbit(z.re):
            return Cx[dt](neg_like_cuda(z.re), _copysign(pi, z.im))
        return Cx[dt](z.re, _copysign(Scalar[dt](0), z.im))
    if isnan(z.re):
        if isinf(z.im):
            return Cx[dt](_jabs(z.im), z.re)
        return Cx[dt](z.re, z.re)
    if isinf(z.im):
        return Cx[dt](_jabs(z.im), _copysign(pi / Scalar[dt](2), z.im))
    var s = csqrt_jit(_jsqr_plus(z, Scalar[dt](-1)))
    var w = clog_jit(Cx[dt](z.re + s.re, z.im + s.im))
    return Cx[dt](_copysign(w.re, Scalar[dt](0)), _copysign(w.im, z.im))


@always_inline
def cacos_jit[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var pi = _jpi[dt]()
    if isinf(z.re):
        if isnan(z.im):
            return Cx[dt](z.im, z.re)
        if isinf(z.im):
            if _signbit(z.re):
                return Cx[dt](Scalar[dt](0.75) * pi, neg_like_cuda(z.im))
            return Cx[dt](Scalar[dt](0.25) * pi, neg_like_cuda(z.im))
        if _signbit(z.re):
            return Cx[dt](pi, neg_like_cuda(z.re) if _signbit(z.im) else z.re)
        return Cx[dt](
            Scalar[dt](0), z.re if _signbit(z.im) else neg_like_cuda(z.re)
        )
    if isnan(z.re):
        if isinf(z.im):
            return Cx[dt](z.re, neg_like_cuda(z.im))
        return Cx[dt](z.re, z.re)
    if isinf(z.im):
        return Cx[dt](pi / Scalar[dt](2), neg_like_cuda(z.im))
    if _is_zero(z.re) and (_is_zero(z.im) or isnan(z.im)):
        return Cx[dt](pi / Scalar[dt](2), neg_like_cuda(z.im))
    var s = csqrt_jit(_jsqr_plus(z, Scalar[dt](-1)))
    var w = clog_jit(Cx[dt](z.re + s.re, z.im + s.im))
    if _signbit(z.im):
        return Cx[dt](_jabs(w.im), _jabs(w.re))
    return Cx[dt](_jabs(w.im), neg_like_cuda(_jabs(w.re)))


# --------------------------------------------------------------------------- #
# thrust's ctanh / ctan and catrig (casinh, casin, cacos, cacosh): what
# stock torch's complex tanh / tan / asinh / asin / acos / acosh run (their
# jiterated kernels are disabled for accuracy, so these are gpu_kernels over
# c10_complex_math, which forwards to thrust; FreeBSD msun's s_ctanh.c and
# catrig.c), with the fused multiply-adds nvcc forms.
#
# Third-party code (full notices and licence texts: THIRD_PARTY_NOTICES.txt):
#   Copyright (c) 2011 David Schultz (ctanh); Copyright (c) 2012 Stephen
#   Montgomery-Smith (catrig). SPDX-License-Identifier: BSD-2-Clause
#   Copyright 2008-2021 NVIDIA Corporation. SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #


@always_inline
def _hadd[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """a + b as one hardware add (NaN propagation included), opaque to the
    fast-math folds of `x + 0` / `x - x`."""
    comptime if is_nvidia_gpu() and dt == DType.float64:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "add.rn.f64 $0, $1, $2;",
                Float64,
                constraints="=d,d,d",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    elif is_nvidia_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "add.rn.f32 $0, $1, $2;",
                Float32,
                constraints="=f,f,f",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    else:
        return a + b


@always_inline
def _flipz[dt: DType]() -> Scalar[dt]:
    """-0.0, built from the bits (a literal may fold to +0.0)."""
    return neg_like_cuda(Scalar[dt](0))


@always_inline
def _hfma[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    """fma(a, b, c) with its operand order kept (which NaN propagates)."""
    comptime if is_nvidia_gpu() and dt == DType.float64:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "fma.rn.f64 $0, $1, $2, $3;",
                Float64,
                constraints="=d,d,d,d",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b), rebind[Float64](c))
        )
    else:
        return a.fma(b, c)


@always_inline
def _hsub[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if is_nvidia_gpu() and dt == DType.float64:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "sub.rn.f64 $0, $1, $2;",
                Float64,
                constraints="=d,d,d",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    elif is_nvidia_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "sub.rn.f32 $0, $1, $2;",
                Float32,
                constraints="=f,f,f",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    else:
        return a - b


@always_inline
def _hmul[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if is_nvidia_gpu() and dt == DType.float64:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "mul.rn.f64 $0, $1, $2;",
                Float64,
                constraints="=d,d,d",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    elif is_nvidia_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "mul.rn.f32 $0, $1, $2;",
                Float32,
                constraints="=f,f,f",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    else:
        return a * b


@always_inline
def _rtan[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](nv_tan(rebind[Float64](x)))
    else:
        return rebind[Scalar[dt]](nv_tanf(rebind[Float32](x)))


@always_inline
def _rasin[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](nv_asin(rebind[Float64](x)))
    else:
        return rebind[Scalar[dt]](nv_asinf(rebind[Float32](x)))


@always_inline
def _racos[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](nv_acos(rebind[Float64](x)))
    else:
        return rebind[Scalar[dt]](cuda_acosf(rebind[Float32](x)))


@always_inline
def _absbits[dt: DType](x: Scalar[dt]) -> UInt64:
    comptime if dt == DType.float64:
        return bitcast[DType.uint64](x) & 0x7FFFFFFFFFFFFFFF
    else:
        return UInt64(
            bitcast[DType.uint32](x.cast[DType.float32]()) & 0x7FFFFFFF
        )


@always_inline
def _lt[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Bool:
    """a < b, false for a NaN (spelled with an isnan the fast-math build
    keeps)."""
    return not isnan(a) and not isnan(b) and a < b


@always_inline
def ctanh_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var x = z.re
    var y = z.im
    comptime f64 = dt == DType.float64
    var ix = _absbits(x)
    comptime INF_BITS = UInt64(0x7FF0000000000000) if f64 else UInt64(
        0x7F800000
    )
    if ix >= INF_BITS:
        if ix != INF_BITS:  # x is NaN
            return Cx[dt](x, y if _is_zero(y) else _hmul(x, y))
        var one = _copysign(Scalar[dt](1), x)
        var t = y if isinf(y) else _hmul(_rsin(y), _rcos(y))
        return Cx[dt](one, _copysign(Scalar[dt](0), t))
    if isinf(y) or isnan(y):
        var d = _hsub(y, y)
        return Cx[dt](d, d)
    comptime BIG = UInt64(0x4036000000000000) if f64 else UInt64(0x41300000)
    if ix >= BIG:  # |x| >= 22 (float: 11)
        var exp_mx = _rexp(-_jabs(x))
        return Cx[dt](
            _copysign(Scalar[dt](1), x),
            Scalar[dt](4) * _rsin(y) * _rcos(y) * exp_mx * exp_mx,
        )
    var t = _rtan(y)
    var beta = t.fma(t, Scalar[dt](1))
    var s = _nv_sinh(x)
    var rho = ieee_sqrt(s.fma(s, Scalar[dt](1)))
    var denom = s.fma(beta * s, Scalar[dt](1))
    return Cx[dt]((beta * rho) * s / denom, t / denom)


@always_inline
def ctan_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var w = ctanh_thrust(Cx[dt](neg_like_cuda(z.im), z.re))
    return Cx[dt](w.im, neg_like_cuda(w.re))


@always_inline
def _cat_f[
    dt: DType
](a: Scalar[dt], b: Scalar[dt], h: Scalar[dt]) -> Scalar[dt]:
    if _lt(b, Scalar[dt](0)):
        return (h - b) / Scalar[dt](2)
    if b == Scalar[dt](0):
        return a / Scalar[dt](2)
    return a * a / (h + b) / Scalar[dt](2)


@always_inline
def _eps[dt: DType]() -> Scalar[dt]:
    comptime if dt == DType.float64:
        return Scalar[dt](2.220446049250313e-16)
    else:
        return Scalar[dt](1.1920928955078125e-07)


@fieldwise_init
struct _HardWork[dt: DType](Copyable, Movable):
    var rx: Scalar[Self.dt]
    var b_usable: Bool
    var b: Scalar[Self.dt]
    var sqrt_a2my2: Scalar[Self.dt]
    var new_y: Scalar[Self.dt]


@always_inline
def _do_hard_work[dt: DType](x: Scalar[dt], y: Scalar[dt]) -> _HardWork[dt]:
    """catrig's do_hard_work (Hull et al.), x, y >= 0 finite."""
    comptime f64 = dt == DType.float64
    var eps = _eps[dt]()
    var four_sqrt_min = Scalar[dt](
        5.966672584960165394632772e-154 if f64 else 4.336808689942017736029811e-19
    )
    var one = Scalar[dt](1)
    var r = chypot(x, y + one)
    var s = chypot(x, y - one)
    var a = (r + s) / Scalar[dt](2)
    if _lt(a, one):
        a = one
    var rx: Scalar[dt]
    if _lt(a, Scalar[dt](10)):
        if y == one and _lt(x, eps * eps / Scalar[dt](128)):
            rx = ieee_sqrt(x)
        elif not _lt(x, eps * _jabs(y - one)):
            var am1 = _cat_f(x, one + y, r) + _cat_f(x, one - y, s)
            rx = _rlog1p(am1 + ieee_sqrt(am1 * (a + one)))
        elif _lt(y, one):
            rx = x / ieee_sqrt((one - y) * (one + y))
        else:
            rx = _rlog1p((y - one) + ieee_sqrt((y - one) * (y + one)))
    else:
        rx = _rlog(a + ieee_sqrt(a.fma(a, -one)))
    var w = _HardWork[dt](rx, False, Scalar[dt](0), Scalar[dt](0), y)
    if _lt(y, four_sqrt_min):
        w.sqrt_a2my2 = a * (Scalar[dt](2) / eps)
        w.new_y = y * (Scalar[dt](2) / eps)
        return w^
    w.b = y / a
    w.b_usable = True
    if not isnan(w.b) and w.b > Scalar[dt](0.6417):
        w.b_usable = False
        if y == one and _lt(x, eps / Scalar[dt](128)):
            w.sqrt_a2my2 = ieee_sqrt(x) * ieee_sqrt((a + y) / Scalar[dt](2))
        elif not _lt(x, eps * _jabs(y - one)):
            var amy = _cat_f(x, y + one, r) + _cat_f(x, y - one, s)
            w.sqrt_a2my2 = ieee_sqrt(amy * (a + y))
        elif not isnan(y) and y > one:
            var k = Scalar[dt](4) / eps / eps
            w.sqrt_a2my2 = x * k * y / ieee_sqrt((y + one) * (y - one))
            w.new_y = y * k
        else:
            w.sqrt_a2my2 = ieee_sqrt((one - y) * (one + y))
    return w^


@always_inline
def _clog_large[dt: DType](z: Cx[dt]) -> Cx[dt]:
    """catrig's clog_for_large_values (thrust's double QUARTER_SQRT_MAX is
    0x1p-509, not 0x1p509: kept)."""
    comptime f64 = dt == DType.float64
    var x = z.re
    var y = z.im
    var ax = _jabs(x)
    var ay = _jabs(y)
    if _lt(ax, ay):
        var t = ax
        ax = ay
        ay = t
    var half_max = Scalar[dt](
        8.988465674311579e307 if f64 else 1.7014117331926443e38
    )
    var theta = catan2_nvcc(y, x)
    if not isnan(ax) and ax > half_max:
        var m_e = Scalar[dt](2.7182818284590452)
        return Cx[dt](_rlog(chypot(x / m_e, y / m_e)) + Scalar[dt](1), theta)
    var quarter_sqrt_max = Scalar[dt](
        5.966672584960165394632772e-154 if f64 else 2.305843009213693952e18
    )
    var sqrt_min = Scalar[dt](
        1.491668146240041348658193e-154 if f64 else 1.084202172485504434007453e-19
    )
    if (not isnan(ax) and ax > quarter_sqrt_max) or _lt(ay, sqrt_min):
        return Cx[dt](_rlog(chypot(x, y)), theta)
    return Cx[dt](_rlog(ax.fma(ax, ay * ay)) / Scalar[dt](2), theta)


@always_inline
def casinh_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    comptime f64 = dt == DType.float64
    var x = z.re
    var y = z.im
    var ax = _jabs(x)
    var ay = _jabs(y)
    var zero = Scalar[dt](0)
    if isnan(x) or isnan(y):
        if isinf(x):
            return Cx[dt](x, _hadd(y, y))
        if isinf(y):
            return Cx[dt](y, _hadd(x, x))
        if _is_zero(y):
            return Cx[dt](_hadd(x, x), y)
        var n = _hadd(_hadd(x, zero), _hadd(y, zero))
        return Cx[dt](n, n)
    var recip_eps = Scalar[dt](1) / _eps[dt]()
    var m_ln2 = Scalar[dt](6.9314718055994531e-1)
    if ax > recip_eps or ay > recip_eps:
        var w = _clog_large(
            z.copy() if not _signbit(x) else Cx[dt](
                neg_like_cuda(x), neg_like_cuda(y)
            )
        )
        return Cx[dt](_copysign(w.re + m_ln2, x), _copysign(w.im, y))
    if _is_zero(x) and _is_zero(y):
        return z.copy()
    var sqrt_6_eps = Scalar[dt](
        3.6500241499888571e-8 if f64 else 8.4572793338e-4
    )
    if ax < sqrt_6_eps / Scalar[dt](4) and ay < sqrt_6_eps / Scalar[dt](4):
        return z.copy()
    var w = _do_hard_work(ax, ay)
    var ry = _rasin(w.b) if w.b_usable else catan2_nvcc(w.new_y, w.sqrt_a2my2)
    return Cx[dt](_copysign(w.rx, x), _copysign(ry, y))


@always_inline
def casin_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var w = casinh_thrust(Cx[dt](z.im, z.re))
    return Cx[dt](w.im, w.re)


@always_inline
def cacos_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    comptime f64 = dt == DType.float64
    var x = z.re
    var y = z.im
    var sx = _signbit(x)
    var sy = _signbit(y)
    var ax = _jabs(x)
    var ay = _jabs(y)
    var zero = Scalar[dt](0)
    var pio2_hi = Scalar[dt](1.5707963267948966)
    var pio2_lo = Scalar[dt](6.1232339957367659e-17)
    if isnan(x) or isnan(y):
        if isinf(x):
            return Cx[dt](_hadd(y, y), -_inf[dt]())
        if isinf(y):
            return Cx[dt](_hadd(x, x), neg_like_cuda(y))
        if _is_zero(x):
            return Cx[dt](_hadd(pio2_hi, pio2_lo), _hadd(y, y))
        var n = _hadd(_hadd(x, zero), _hadd(y, zero))
        return Cx[dt](n, n)
    var recip_eps = Scalar[dt](1) / _eps[dt]()
    var m_ln2 = Scalar[dt](6.9314718055994531e-1)
    if ax > recip_eps or ay > recip_eps:
        var w = _clog_large(z)
        var ry = w.re + m_ln2
        if not sy:
            ry = neg_like_cuda(ry)
        return Cx[dt](_jabs(w.im), ry)
    if x == Scalar[dt](1) and _is_zero(y):
        return Cx[dt](zero, neg_like_cuda(y))
    var sqrt_6_eps = Scalar[dt](
        3.6500241499888571e-8 if f64 else 8.4572793338e-4
    )
    if ax < sqrt_6_eps / Scalar[dt](4) and ay < sqrt_6_eps / Scalar[dt](4):
        return Cx[dt](_hsub(pio2_hi, _hsub(x, pio2_lo)), neg_like_cuda(y))
    var w = _do_hard_work(ay, ax)
    var rx: Scalar[dt]
    if w.b_usable:
        rx = _racos(w.b if not sx else neg_like_cuda(w.b))
    else:
        rx = catan2_nvcc(
            w.sqrt_a2my2, w.new_y if not sx else neg_like_cuda(w.new_y)
        )
    var ry = w.rx
    if not sy:
        ry = neg_like_cuda(ry)
    return Cx[dt](rx, ry)


@always_inline
def cacosh_thrust[dt: DType](z: Cx[dt]) -> Cx[dt]:
    var w = cacos_thrust(z)
    var rx = w.re
    var ry = w.im
    if isnan(rx) and isnan(ry):
        return Cx[dt](ry, rx)
    if isnan(rx):
        return Cx[dt](_jabs(ry), rx)
    if isnan(ry):
        return Cx[dt](ry, ry)
    return Cx[dt](_jabs(ry), _copysign(rx, z.im))
