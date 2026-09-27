"""Pointwise SIMD expressions shared by eager (tmb/kernels/pointwise) and
torch.compile (tmb/graph/pointwise.mojo) execution: the binary and ternary
math ops (atan2, hypot, copysign, nextafter, gcd, the special polynomials,
...) and the activations with scalar parameters (elu, softplus, hardtanh,
...) with their backwards.

One entry, `pointwise[kind]`, takes up to three operands of one dtype plus
four scalar parameters and returns the result in `out_dtype`. Each kind
follows the CUDA kernel stock PyTorch runs (named next to it); half-precision
operands are widened to float32 (ATen's opmath type), computed, and rounded
once, except where ATen works on the half value itself (copysign, nextafter,
heaviside, fmax/fmin, clamp).

The exponential and logarithm are the bit-exact libdevice ports
(`nv_expf` / `nv_logf` / `nv_log1pf`, `nv_exp` / `nv_log` / `nv_log1p` for
float64): std.math's GPU `exp` is `ex2.approx` of `x * log2(e)`, whose
relative error grows with |x| (5e-6 near 88) where CUDA's `expf` stays within
2 ulp. They run once per lane.

The GPU build compiles with fast-math flags that let LLVM assume no NaN, so a
comparison with NaN may fold either way: every kind whose result depends on
NaN handling tests `isnan` (bit-based `llvm.is.fpclass`, which survives the
flags) and selects explicitly.
"""

from std.math import acos, cos, floor, fma, sin
from std.bit import count_leading_zeros
from std.collections import Array
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_apple_gpu, is_nvidia_gpu, size_of
from std.utils.numerics import inf, isinf, isnan, nan

from tmb.kernels.common.cuda_math import (
    cuda_acosf,
    cuda_atan2f,
    cuda_hypotf,
    nv_cosf,
    nv_erf,
    nv_erff,
    nv_exp2f,
    nv_expm1,
    nv_expm1f,
    nv_sinf,
    nv_tanh,
    nv_tanhf,
)
from tmb.kernels.common.libdevice_port import (
    _trig_reduction_slowpath_d,
    nv_exp,
    nv_expf,
    nv_log,
    nv_log1p,
    nv_log1pf,
    nv_logf,
)
from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import _fmod_f64_exact_scalar, _fmod_float32
from tmb.kernels.common.pow_math import torch_pow
from tmb.kernels.common.special_math import igamma_f, igammac_f, zeta_f


@always_inline
def wide_dtype[dtype: DType]() -> DType:
    """ATen's opmath type of a floating dtype: float64 stays, the rest
    compute in float32 (integers too, for the kinds that promote them)."""
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def param_dtype[dtype: DType]() -> DType:
    """The dtype the scalar parameters travel in to the device: float64 only
    for float64 operands (Metal has no double at all); int64 for integer
    operands, whose parameters (threshold, hardtanh's bounds) torch applies
    in scalar_t -- a float32 would round 16777217 to 16777216."""
    comptime if dtype.is_integral():
        return DType.int64
    else:
        return wide_dtype[dtype]()


# ---------------------------------------------------------------------------
# accurate exp / log family (per lane, bit-exact libdevice ports)
# ---------------------------------------------------------------------------


@always_inline
def _exp[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_exp(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_expf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _log[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_log(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_logf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _log1p[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_log1p(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_log1pf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _expm1[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    """expm1 by Kahan's correction, (e^x - 1) * x / log(e^x): a few ulp
    everywhere, exact 0 at 0, -1 below the float range, NaN/inf through.
    float32 on NVIDIA is instead libdevice's expm1f (what `std::expm1` runs
    in the CUDA kernels): one ex2 and a polynomial, where the Kahan form
    costs an exp, a log and an IEEE division (elu f32 16M: 1.35x torch)."""
    comptime if w == DType.float32 and is_nvidia_gpu():
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = nv_expm1f(x[i].cast[DType.float32]()).cast[w]()
        return r
    elif w == DType.float64:
        # libdevice's expm1 (what `std::expm1(double)` runs in CUDA).
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = nv_expm1(x[i].cast[DType.float64]()).cast[w]()
        return r
    var u = _exp(x)
    var um1 = u - 1
    var corrected = um1 * (x / _log(u))
    var r = um1.eq(0).select(x, corrected)
    r = um1.eq(-1).select(um1, r)
    r = isinf(u).select(u, r)
    return isnan(x).select(x, r)


@always_inline
def _tanh[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    """tanh(|x|) = e / (e + 2), e = expm1(2|x|); sign restored, saturated to
    1 past 2|x| = 40 where e overflows float32's ratio."""
    var ax = abs(x)
    var e = _expm1(ax + ax)
    var t = e / (e + 2)
    t = ax.gt(20).select(SIMD[w, n](1), t)
    var r = x.lt(0).select(-t, t)
    return isnan(x).select(x, r)


@always_inline
def _tanh_cuda[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    """`c10::cuda::compat::tanh`: libdevice's tanhf / tanh (ported
    literally)."""
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_tanh(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_tanhf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _sigmoid[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    return 1 / (1 + _exp(-x))


@always_inline
def _one_sixth[w: DType, n: Int]() -> SIMD[w, n]:
    """The CUDA activation kernels' `opmath_t(1.0f / 6.0f)`: float's 1/6,
    widened as is for double."""
    return SIMD[w, n](Float32(1.0 / 6.0).cast[w]())


@always_inline
def _atan_f64(x: Float64) -> Float64:
    """fdlibm's `atan` for a finite x >= 0 (four reduction intervals and an
    odd degree-23 fit, within an ulp)."""
    comptime HI = [
        4.63647609000806093515e-01,
        7.85398163397448278999e-01,
        9.82793723247329054082e-01,
        1.57079632679489655800e00,
    ]
    comptime LO = [
        2.26987774529616870924e-17,
        3.06161699786838301793e-17,
        1.39033110312309984516e-17,
        6.12323399573676603587e-17,
    ]
    if x >= 7.378697629483821e19:  # 2**66
        comptime h3 = HI[3]
        comptime l3 = LO[3]
        return h3 + l3
    var t = x
    var hi = 0.0
    var lo = 0.0
    var reduced = True
    if x < 0.4375:
        if x < 7.450580596923828e-09:  # 2**-27
            return x
        reduced = False
    elif x < 0.6875:
        t = (2.0 * x - 1.0) / (2.0 + x)
        comptime h = HI[0]
        comptime l = LO[0]
        hi = h
        lo = l
    elif x < 1.1875:
        t = (x - 1.0) / (x + 1.0)
        comptime h = HI[1]
        comptime l = LO[1]
        hi = h
        lo = l
    elif x < 2.4375:
        t = (x - 1.5) / (1.0 + 1.5 * x)
        comptime h = HI[2]
        comptime l = LO[2]
        hi = h
        lo = l
    else:
        t = -1.0 / x
        comptime h = HI[3]
        comptime l = LO[3]
        hi = h
        lo = l
    var z = t * t
    var w = z * z
    var s1 = z * (
        3.33333333333329318027e-01
        + w
        * (
            1.42857142725034663711e-01
            + w
            * (
                9.09088713343650656196e-02
                + w
                * (
                    6.66107313738753120669e-02
                    + w
                    * (
                        4.97687799461593236017e-02
                        + w * 1.62858201153657823623e-02
                    )
                )
            )
        )
    )
    var s2 = w * (
        -1.99999999998764832476e-01
        + w
        * (
            -1.11111104054623557880e-01
            + w
            * (
                -7.69187620504482999495e-02
                + w
                * (
                    -5.83357013379057348645e-02
                    + w * -3.65315727442169155270e-02
                )
            )
        )
    )
    if not reduced:
        return t - t * (s1 + s2)
    return hi - ((t * (s1 + s2) - lo) - t)


@always_inline
def _atan2_f64(y: Float64, x: Float64) -> Float64:
    """fdlibm's `__ieee754_atan2`: C99's special values, atan(|y / x|)
    moved to its quadrant with the low part of pi (std.math has no float64
    atan2 on NVIDIA)."""
    comptime PI = 3.1415926535897931160e00
    comptime PI_LO = 1.2246467991473531772e-16
    comptime PIO2 = 1.57079632679489655800e00
    comptime PIO4 = 7.8539816339744827900e-01
    if isnan(x) or isnan(y):
        return nan[DType.float64]()
    var yneg = bitcast[DType.int64](y) < 0
    var xneg = bitcast[DType.int64](x) < 0
    if y == 0.0:
        if not xneg:
            return y
        return -PI if yneg else PI
    if x == 0.0:
        return -PIO2 if yneg else PIO2
    if isinf(x):
        var r: Float64
        if isinf(y):
            r = 3.0 * PIO4 if xneg else PIO4
        else:
            r = PI if xneg else 0.0
        return -r if yneg else r
    if isinf(y):
        return -PIO2 if yneg else PIO2
    var ey = Int((bitcast[DType.uint64](y) >> UInt64(52)) & UInt64(0x7FF))
    var ex = Int((bitcast[DType.uint64](x) >> UInt64(52)) & UInt64(0x7FF))
    var k = ey - ex
    var z: Float64
    if k > 60:
        z = PIO2 + 0.5 * PI_LO
        xneg = False
    elif xneg and k < -60:
        z = 0.0
    else:
        z = _atan_f64(abs(y / x))
    if not xneg:
        return -z if yneg else z
    var r = PI - (z - PI_LO)
    return -r if yneg else r


@always_inline
def _exp2_f64(d: Float64) -> Float64:
    """2^d: d = k + f, f in [-1/2, 1/2], 2^f = exp(f ln 2) with the product
    kept in double-double, scaled by 2^k in one rounding (`_ldexp`)."""
    if isnan(d):
        return d
    if d > 1024.0:
        return inf[DType.float64]()
    if d < -1080.0:
        return 0.0
    comptime LN2 = 6.93147180559945286227e-01
    comptime LN2_LO = 2.319046813846299558e-17
    var k = llvm_intrinsic["llvm.roundeven", Float64, has_side_effect=False](d)
    var f = d - k  # exact
    var hi = f * LN2
    var lo = fma(f, LN2, -hi) + f * LN2_LO
    var r = nv_exp(hi)
    r = fma(r, lo, r)
    return _ldexp[DType.float64, 1](r, k)[0]


@always_inline
def _hypot_root_f64(v: Float64) -> Float64:
    """sqrt(v) as CUDA's double hypot takes it (stock torch's
    `hypot_kernel_cuda<double>` SASS, libtorch_cuda.so sm_90): the rsqrt
    approximation of the high word (MUFU.RSQ64H), one quadratic Newton
    step, times v -- not the IEEE root. Other GPUs: the IEEE root."""
    comptime if is_nvidia_gpu():
        var vc = min(v, 1.7976931348623157e308)
        var hi = bitcast[DType.float64](
            bitcast[DType.uint64](vc) & UInt64(0xFFFFFFFF00000000)
        )
        var y = inlined_assembly[
            "rsqrt.approx.ftz.f64 $0, $1;",
            Float64,
            constraints="=d,d",
            has_side_effect=False,
        ](hi)
        y = bitcast[DType.float64](
            bitcast[DType.uint64](y) & UInt64(0xFFFFFFFF00000000)
        )
        var e = vc.fma(-(y * y), 1.0)
        var y2 = e.fma(0.375, 0.5).fma(y * e, y)
        return v * y2
    else:
        return ieee_sqrt(v)


@always_inline
def _hypot_f64(x: Float64, y: Float64) -> Float64:
    """CUDA's double hypot (the SASS above): the magnitudes ordered by their
    bits (so NaN ranks above inf: hypot(inf, NaN) = inf), both scaled by a
    power of two taken from the larger one's exponent, one fma-summed square,
    `_hypot_root_f64`, the scale restored; a zero smaller magnitude returns
    the larger."""
    var a = bitcast[DType.uint64](abs(x))
    var b = bitcast[DType.uint64](abs(y))
    var mx = max(a, b)
    var mn = min(a, b)
    var fmx = bitcast[DType.float64](mx)
    var fmn = bitcast[DType.float64](mn)
    if (mn >> UInt64(32)) >= UInt64(0x7FF00000):
        return fmn
    if fmn == 0:
        return fmx
    var e = mx & UInt64(0xFFC0000000000000)
    var scale = bitcast[DType.float64](UInt64(0x7FD0000000000000) - e)
    var t = fmn * scale
    var u = fmx * scale
    var r = _hypot_root_f64(u.fma(u, t * t))
    return r * bitcast[DType.float64](e | UInt64(0x0010000000000000))


@always_inline
def _zeta_scalar(x: Float64, q: Float64) -> Float64:
    """Hurwitz zeta, Cephes (aten/src/ATen/native/Math.h `zeta`), in double
    for every input dtype: CPU torch accumulates float in double too."""
    comptime MACHEP = 1.11022302462515654042e-16
    var A = Array[Float64, 12](fill=0.0)
    A[0] = 12.0
    A[1] = -720.0
    A[2] = 30240.0
    A[3] = -1209600.0
    A[4] = 47900160.0
    A[5] = -1.8924375803183791606e9
    A[6] = 7.47242496e10
    A[7] = -2.950130727918164224e12
    A[8] = 1.1646782814350067249e14
    A[9] = -4.5979787224074726105e15
    A[10] = 1.8152105401943546773e17
    A[11] = -7.1661652561756670113e18
    # C's comparisons with NaN are all false (zeta_f spells out why).
    var x_nan = isnan(x)
    var q_nan = isnan(q)
    if not x_nan and x == 1.0:
        return inf[DType.float64]()
    if not x_nan and x < 1.0:
        return nan[DType.float64]()
    if not q_nan and q <= 0.0:
        if q == floor(q):
            return inf[DType.float64]()
        if x_nan or x != floor(x):
            return nan[DType.float64]()
    if x_nan or q_nan:
        return nan[DType.float64]()
    var s = _zeta_pow(q, -x)
    var a = q
    var i = 0
    var b = 0.0
    while i < 9 or a <= 9.0:
        i += 1
        a += 1.0
        b = _zeta_pow(a, -x)
        s += b
        if -MACHEP * s < b and b < MACHEP * s:
            return s
    var w = a
    s += b * w / (x - 1.0)
    s -= 0.5 * b
    a = 1.0
    var k = 0.0
    for j in range(12):
        a *= x + k
        b /= w
        var t = a * b / A[j]
        s = s + t
        t = abs(t / s)
        if t < MACHEP:
            return s
        k += 1.0
        a *= x + k
        b /= w
        k += 1.0
    return s


@always_inline
def _zeta_pow(a: Float64, b: Float64) -> Float64:
    """C's pow for zeta's terms (the jiterator's `pow` for T = double):
    pow_math's special cases (pow(1, -inf) = 1, zero and negative bases)
    around its double-double core."""
    return torch_pow[DType.float64, 1](a, b)[0]


@always_inline
def _poly_n[w: DType](n: Scalar[w]) -> Int where w.is_floating_point():
    """`static_cast<int64_t>(n)` for the polynomial degree (truncation), in
    the operand's own float type (no float64 on Apple GPUs); NaN and
    out-of-range degrees read as -1, which every polynomial maps to 0
    (hermite does exactly this; the others would be UB in C++)."""
    if isnan(n) or n >= 9.2e18 or n <= -9.2e18:
        return -1
    return Int(n)


@always_inline
def _sincos_f64(a: Float64) -> Tuple[Float64, Float64]:
    """(sin a, cos a) in float64 (std.math has neither on NVIDIA or AMD):
    libdevice's reduction by pi/2 (three-part Cody-Waite, Payne-Hanek past
    2^31, `_trig_reduction_slowpath_d`), then fdlibm's sin and cos kernels
    (within an ulp)."""
    if isnan(a) or isinf(a):
        return (nan[DType.float64](), nan[DType.float64]())
    var qf = llvm_intrinsic["llvm.roundeven", Float64, has_side_effect=False](
        a * Float64(from_bits=UInt64(0x3FE45F306DC9C883))
    )
    var t = fma(-qf, Float64(from_bits=UInt64(0x3FF921FB54442D18)), a)
    t = fma(-qf, Float64(from_bits=UInt64(0x3C91A62633145C00)), t)
    t = fma(-qf, Float64(from_bits=UInt64(0x397B839A252049C0)), t)
    var q = Int(qf) if abs(a) < 2147483648.0 else 0
    if abs(a) >= 2147483648.0:
        var rq = _trig_reduction_slowpath_d(a, Int32(0))
        t = rq[0]
        q = Int(rq[1])
    var z = t * t
    var v = z * t
    var rs = 8.33333333332248946124e-03 + z * (
        -1.98412698298579493134e-04
        + z
        * (
            2.75573137070700676789e-06
            + z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10)
        )
    )
    var sn = t + v * (-1.66666666666666324348e-01 + z * rs)
    var w = z * z
    var rc = z * (
        4.16666666666666019037e-02
        + z * (-1.38888888888741095749e-03 + z * 2.48015872894767294178e-05)
    ) + w * w * (
        -2.75573143513906633035e-07
        + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11)
    )
    var hz = 0.5 * z
    var ww = 1.0 - hz
    var cs = ww + (((1.0 - ww) - hz) + z * rc)
    var k = q & 3
    if k == 0:
        return (sn, cs)
    elif k == 1:
        return (cs, -sn)
    elif k == 2:
        return (-sn, -cs)
    return (-cs, sn)


@always_inline
def _acos_f64(x: Float64) -> Float64:
    """fdlibm's `__ieee754_acos` (a rational fit of asin on [0, 1/2], within
    an ulp): std.math has no float64 acos on NVIDIA or AMD (`facos` has no
    libcall there)."""
    comptime PIO2_HI = 1.57079632679489655800e00
    comptime PIO2_LO = 6.12323399573676603587e-17

    @always_inline
    def rat(z: Float64) -> Float64:
        var p = z * (
            1.66666666666666657415e-01
            + z
            * (
                -3.25565818622400915405e-01
                + z
                * (
                    2.01212532134862925881e-01
                    + z
                    * (
                        -4.00555345006794114027e-02
                        + z
                        * (
                            7.91534994289814532176e-04
                            + z * 3.47933107596021167570e-05
                        )
                    )
                )
            )
        )
        var q = 1.0 + z * (
            -2.40339491173441421878e00
            + z
            * (
                2.02094576023350569471e00
                + z
                * (-6.88283971605453293030e-01 + z * 7.70381505559019352791e-02)
            )
        )
        return p / q

    if isnan(x) or abs(x) > 1.0:
        return nan[DType.float64]()
    if abs(x) < 0.5:
        return PIO2_HI - (x - (PIO2_LO - x * rat(x * x)))
    if x < 0.0:
        var z = (1.0 + x) * 0.5
        var s = ieee_sqrt(z)
        var w = rat(z) * s - PIO2_LO
        return 2.0 * (PIO2_HI - (s + w))
    var z = (1.0 - x) * 0.5
    var s = ieee_sqrt(z)
    var df = bitcast[DType.float64](
        bitcast[DType.uint64](s) & UInt64(0xFFFFFFFF00000000)
    )
    var c = (z - df * df) / (s + df)
    var w = rat(z) * s + c
    return 2.0 * (df + w)


@always_inline
def _pacos[w: DType](x: Scalar[w]) -> Scalar[w] where w.is_floating_point():
    """The polynomials' `acos`: the CUDA acosf on NVIDIA (n acos(x) carries
    its last-ulp differences n-fold), `_acos_f64` for float64."""
    comptime if w == DType.float32 and is_nvidia_gpu():
        return cuda_acosf(x.cast[DType.float32]()).cast[w]()
    elif w == DType.float64:
        return _acos_f64(x.cast[DType.float64]()).cast[w]()
    else:
        return acos(x)


@always_inline
def _pcos[w: DType](x: Scalar[w]) -> Scalar[w] where w.is_floating_point():
    """The polynomials' `cos`: libdevice's cosf on NVIDIA (what the float
    jiterator kernels call; std.math's is `cos.approx.ftz`, whose error
    grows with the n * acos(x) argument), Metal's own cos on Apple (MPS's
    `precise::cos`), std.math elsewhere."""
    comptime if w == DType.float32 and is_nvidia_gpu():
        return nv_cosf(x.cast[DType.float32]()).cast[w]()
    elif w == DType.float64:
        # std.math has no float64 cos on NVIDIA or AMD (`fcos` has no
        # libcall there).
        return _sincos_f64(x.cast[DType.float64]())[1].cast[w]()
    else:
        return cos(x)


@always_inline
def _psin[w: DType](x: Scalar[w]) -> Scalar[w] where w.is_floating_point():
    """The polynomials' `sin`, as `_pcos`."""
    comptime if w == DType.float32 and is_nvidia_gpu():
        return nv_sinf(x.cast[DType.float32]()).cast[w]()
    elif w == DType.float64:
        return _sincos_f64(x.cast[DType.float64]())[0].cast[w]()
    else:
        return sin(x)


@always_inline
def _polynomial[
    kind: StaticString, w: DType
](x: Scalar[w], n_f: Scalar[w]) -> Scalar[w] where w.is_floating_point():
    """The orthogonal polynomials of aten/src/ATen/native/Math.h
    (`*_polynomial_*_forward`), statement for statement, in the opmath type."""
    var n = _poly_n(n_f)
    comptime one = Scalar[w](1.0)
    if n < 0:
        return 0
    if isnan(x):
        # Every special case compares x and fails for NaN, so the result is
        # the degree-0 constant or NaN; tested up front because the GPU's
        # fast-math flags may fold those comparisons either way.
        return one if n == 0 else x
    comptime if kind == "chebyshev_polynomial_t":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return one
            return -one
        if n > 6 and abs(x) < one:
            return _pcos(Scalar[w](n) * _pacos(x))
        if n == 0:
            return one
        if n == 1:
            return x
        var p = one
        var q = x
        var r = x
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_u":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return Scalar[w](n + 1)
            return -Scalar[w](n + 1)
        if n > 8 and abs(x) < one:
            var t = _pacos(x)
            if _psin(t) != 0:
                return _psin(Scalar[w](n + 1) * t) / _psin(t)
            return Scalar[w](n + 1) * _pcos(Scalar[w](n + 1) * t) / x
        if n == 0:
            return one
        if n == 1:
            return x + x
        var p = one
        var q = x + x
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_v":
        if abs(x) == one:
            if x > 0:
                return one
            if n % 2 == 0:
                return Scalar[w](n + n + 1)
            return -Scalar[w](n + n + 1)
        if n > 8 and abs(x) < one:
            var t = _pacos(x)
            if _psin(t / 2) != one:
                return _pcos((Scalar[w](n) + 0.5) * t) / _pcos(t / 2)
            if n % 2 == 0:
                return Scalar[w](n + n + 1)
            return -Scalar[w](n + n + 1)
        if n == 0:
            return one
        if n == 1:
            return x + x - one
        var p = one
        var q = x + x - one
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_w":
        if abs(x) == one:
            if x > 0:
                return Scalar[w](n + n + 1)
            if n % 2 == 0:
                return one
            return -one
        if n > 8 and abs(x) < one:
            var t = _pacos(x)
            if _pcos(t / 2) != one:
                return _psin((Scalar[w](n) + 0.5) * t) / _psin(t / 2)
            if x > 0:
                return Scalar[w](n + n + 1)
            if n % 2 == 0:
                return one
            return -one
        if n == 0:
            return one
        if n == 1:
            return x + x + one
        var p = one
        var q = x + x + one
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "hermite_polynomial_h":
        if n == 0:
            return one
        if n == 1:
            return x + x
        comptime limit = 512 if w == DType.float64 else 128
        if n > limit:
            return nan[w]()
        var p = one
        var q = x + x
        var r = Scalar[w](0)
        var k = 2
        while k < n + n:
            r = (x + x) * q - Scalar[w](k) * p
            p = q
            q = r
            k += 2
        return r
    elif kind == "hermite_polynomial_he":
        if n == 0:
            return one
        if n == 1:
            return x
        comptime limit = 512 if w == DType.float64 else 128
        if n > limit:
            return nan[w]()
        var p = one
        var q = x
        var r = Scalar[w](0)
        for k in range(1, n):
            r = x * q - Scalar[w](k) * p
            p = q
            q = r
        return r
    elif kind == "laguerre_polynomial_l":
        if abs(x) == 0:
            return one
        if n == 0:
            return one
        if n == 1:
            return one - x
        var p = one
        var q = one - x
        var r = q
        var k = 1
        while k < n and not isnan(q):
            r = (
                (Scalar[w](k + k) + (one - x)) * q - Scalar[w](k) * p
            ) / Scalar[w](k + 1)
            p = q
            q = r
            k += 1
        return r
    elif kind == "legendre_polynomial_p":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return one
            return -one
        if n == 0:
            return one
        if n == 1:
            return x
        var p = one
        var q = x
        var r = q
        var k = 1
        while k < n and not isnan(q):
            r = (Scalar[w](k + k + 1) * x * q - Scalar[w](k) * p) / Scalar[w](
                k + 1
            )
            p = q
            q = r
            k += 1
        return r
    else:
        # The shifted Chebyshev polynomials: T*(x) = T(2x - 1) and friends,
        # with their own boundary cases at x = 0 and x = 1.
        var y = x + x - one
        comptime if kind == "shifted_chebyshev_polynomial_t":
            if x == one:
                return one
            if x == 0:
                return one if n % 2 == 0 else -one
            if n > 6 and abs(y) < one:
                return _pcos(Scalar[w](n) * _pacos(y))
            if n == 0:
                return one
            if n == 1:
                return y
            var p = one
            var q = y
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        elif kind == "shifted_chebyshev_polynomial_u":
            if x == one:
                return Scalar[w](n + 1)
            if x == 0:
                return Scalar[w](n + 1) if n % 2 == 0 else -Scalar[w](n + 1)
            if n > 6 and abs(y) < one:
                var t = _pacos(y)
                if _psin(t) != 0:
                    return _psin(Scalar[w](n + 1) * t) / _psin(t)
                return Scalar[w](n + 1) * _pcos(Scalar[w](n + 1) * t) / y
            if n == 0:
                return one
            if n == 1:
                return y + y
            var p = one
            var q = y + y
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        elif kind == "shifted_chebyshev_polynomial_v":
            if x == one:
                return one
            if x == 0:
                return Scalar[w](n + n + 1) if n % 2 == 0 else -Scalar[w](
                    n + n + 1
                )
            if n > 6 and abs(y) < one:
                var t = _pacos(y)
                if _psin(t / 2) != one:
                    return _pcos((Scalar[w](n) + 0.5) * t) / _pcos(t / 2)
                return Scalar[w](n + n + 1) if n % 2 == 0 else -Scalar[w](
                    n + n + 1
                )
            if n == 0:
                return one
            if n == 1:
                return y + y - one
            var p = one
            var q = y + y - one
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        else:
            comptime assert (
                kind == "shifted_chebyshev_polynomial_w"
            ), "unknown polynomial kind"
            if x == one:
                return Scalar[w](n + n + 1)
            if x == 0:
                return one if n % 2 == 0 else -one
            if n > 4 and abs(y) < one:
                var t = _pacos(y)
                if _pcos(t / 2) != one:
                    return _psin((Scalar[w](n) + 0.5) * t) / _psin(t / 2)
                return one if n % 2 == 0 else -one
            if n == 0:
                return one
            if n == 1:
                return y + y + one
            var p = one
            var q = y + y + one
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r


@always_inline
def is_polynomial[kind: StaticString]() -> Bool:
    return (
        kind == "chebyshev_polynomial_t"
        or kind == "chebyshev_polynomial_u"
        or kind == "chebyshev_polynomial_v"
        or kind == "chebyshev_polynomial_w"
        or kind == "shifted_chebyshev_polynomial_t"
        or kind == "shifted_chebyshev_polynomial_u"
        or kind == "shifted_chebyshev_polynomial_v"
        or kind == "shifted_chebyshev_polynomial_w"
        or kind == "hermite_polynomial_h"
        or kind == "hermite_polynomial_he"
        or kind == "laguerre_polynomial_l"
        or kind == "legendre_polynomial_p"
    )


# ---------------------------------------------------------------------------
# kinds computed on the operand dtype itself
# ---------------------------------------------------------------------------


@always_inline
def is_native_kind[kind: StaticString]() -> Bool:
    """Kinds ATen evaluates on the storage value, never widened: they take
    every dtype their op accepts (integers, bool) and keep it."""
    return (
        kind == "copysign"
        or kind == "nextafter"
        or kind == "heaviside"
        or kind == "fmax"
        or kind == "fmin"
        or kind == "fmod"
        or kind == "gcd"
        or kind == "lcm"
        or kind == "lshift"
        or kind == "rshift"
        or kind == "clamp"
        or kind == "maximum"
        or kind == "minimum"
        or kind == "ipow"
        or kind == "frexp_mantissa"
        or kind == "frexp_exponent"
    )


@always_inline
def _uint_of[dtype: DType]() -> DType:
    comptime if size_of[dtype]() == 8:
        return DType.uint64
    elif size_of[dtype]() == 4:
        return DType.uint32
    elif size_of[dtype]() == 2:
        return DType.uint16
    else:
        return DType.uint8


@always_inline
def _mant_bits[dtype: DType]() -> Int:
    comptime if dtype == DType.float64:
        return 52
    elif dtype == DType.float32:
        return 23
    elif dtype == DType.float16:
        return 10
    else:
        return 7


@always_inline
def _nextafter[
    dtype: DType, n: Int
](a: SIMD[dtype, n], b: SIMD[dtype, n]) -> SIMD[dtype, n]:
    """C's nextafter on the bits of `dtype` (c10's Half/BFloat16 overloads
    do the same on their 16-bit patterns)."""
    comptime u = _uint_of[dtype]()
    comptime sign = Scalar[u](1) << Scalar[u](size_of[dtype]() * 8 - 1)
    var ua = bitcast[u, n](a)
    var ub = bitcast[u, n](b)
    var mag_a = ua & ~sign
    var mag_b = ub & ~sign
    var both_zero = mag_a.eq(0) & mag_b.eq(0)
    # From +-0 toward nonzero b: the smallest subnormal with b's sign.
    var from_zero = (ub & sign) | 1
    # Away from zero when a < b for positive a (or a > b for negative a):
    # one step up in magnitude, else one step down.
    var a_neg = (ua & sign).ne(0)
    var up = a.lt(b) ^ a_neg
    var stepped = up.select(ua + 1, ua - 1)
    var r = mag_a.eq(0).select(from_zero, stepped)
    r = a.eq(b).select(ub, r)
    r = both_zero.select(ub, r)
    # The NaN select on the bits too: a float select of a half type may go
    # through float32 and flush a subnormal result (bfloat16 on Apple GPUs).
    comptime nan_bits = bitcast[u](nan[dtype]())
    r = (isnan(a) | isnan(b)).select(SIMD[u, n](nan_bits), r)
    return bitcast[dtype, n](r)


@always_inline
def _frexp_parts[
    dtype: DType, n: Int
](a: SIMD[dtype, n]) -> Tuple[SIMD[dtype, n], SIMD[DType.int32, n]]:
    """frexp on the bits of `dtype` itself: a = m * 2^e with 0.5 <= |m| < 1;
    m = a and e = 0 for zero, inf and NaN (glibc's and CUDA's frexp).
    Integer operations only, so a subnormal is normalized exactly even
    where the GPU flushes subnormal floats (Apple GPUs), and no widening
    conversion can flush it either."""
    comptime u = _uint_of[dtype]()
    comptime W = size_of[dtype]() * 8
    comptime mb = _mant_bits[dtype]()
    comptime emax = (1 << (W - 1 - mb)) - 1
    comptime bias = emax >> 1
    comptime sign_mask = Scalar[u](1) << Scalar[u](W - 1)
    comptime mant_mask = (Scalar[u](1) << Scalar[u](mb)) - 1
    var bits = bitcast[u, n](a)
    var sign = bits & sign_mask
    var mag = bits & ~sign_mask
    var e = (mag >> Scalar[u](mb)).cast[DType.int32]()
    var m = mag & mant_mask
    # A subnormal's leading one moves up to the implicit bit's position.
    var p = Int32(W - 1) - count_leading_zeros(m).cast[DType.int32]()
    var shift = Int32(mb) - p
    var sub = e.eq(0) & m.ne(0)
    var frac = sub.select((m << shift.cast[u]()) & mant_mask, m)
    var unbiased = sub.select(Int32(1 - bias) - shift, e - Int32(bias))
    var mant = sign | (Scalar[u](bias - 1) << Scalar[u](mb)) | frac
    var special = e.eq(Int32(emax)) | mag.eq(0)
    var res_m = special.select(bits, mant)
    var res_e = special.select(SIMD[DType.int32, n](0), unbiased + 1)
    return (bitcast[dtype, n](res_m), res_e)


@always_inline
def _frexp_exponent[
    dtype: DType, n: Int
](a: SIMD[dtype, n]) -> SIMD[DType.int32, n]:
    return _frexp_parts(a)[1]


@always_inline
def _frexp_mantissa[dtype: DType, n: Int](a: SIMD[dtype, n]) -> SIMD[dtype, n]:
    return _frexp_parts(a)[0]


@always_inline
def _gcd_scalar[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """calc_gcd (aten/src/ATen/native/Math.h): Euclid on the magnitudes."""
    comptime if dtype.is_signed() and is_nvidia_gpu():
        # Unsigned remainders on the magnitudes when neither operand is the
        # minimum value (whose C `abs` stays negative): the same values
        # without the sign fix-up Mojo's floored `%` costs per step (gcd
        # int32 4096^2: 1.17x torch).
        comptime u = _uint_of[dtype]()
        if x != Scalar[dtype].MIN and y != Scalar[dtype].MIN:
            var ua = abs(x).cast[u]()
            var ub = abs(y).cast[u]()
            while ua != 0:
                var uc = ua
                ua = ub % ua
                ub = uc
            return ub.cast[dtype]()
    # C's truncating `%` (Mojo's floors): with the minimum value, whose
    # `abs` wraps back to itself, the operands stay negative and the sign of
    # each remainder matters (gcd(int8 -128, -20) is 4 on CUDA and MPS).
    var a = abs(x)
    var b = abs(y)
    while a != 0:
        var c = a
        var r = b % a
        comptime if dtype.is_signed():
            if r != 0 and ((r < 0) != (b < 0)):
                r -= a
        a = r
        b = c
    return b


@always_inline
def _powi[
    dtype: DType
](base: Scalar[dtype], exponent: Scalar[dtype]) -> Scalar[dtype]:
    var a = base
    var b = exponent
    comptime if dtype.is_signed():
        if b < 0:
            if a == 1:
                return 1
            if a == -1:
                return -1 if (b % 2) != 0 else 1
            return 0
    var result = Scalar[dtype](1)
    while b != 0:
        if (b & 1) != 0:
            result *= a
        b = b >> 1
        a *= a
    return result


@always_inline
def _native[
    kind: StaticString, dtype: DType, n: Int
](a: SIMD[dtype, n], b: SIMD[dtype, n], c: SIMD[dtype, n]) -> SIMD[dtype, n]:
    comptime if kind == "copysign":
        # CopysignKernel.cu: the sign bit of b on the magnitude bits of a.
        comptime u = _uint_of[dtype]()
        comptime sign = Scalar[u](1) << Scalar[u](size_of[dtype]() * 8 - 1)
        return bitcast[dtype, n](
            (bitcast[u, n](a) & ~sign) | (bitcast[u, n](b) & sign)
        )
    elif kind == "nextafter":
        return _nextafter(a, b)
    elif kind == "heaviside":
        # StepKernel.cu: a == 0 ? b : (a > 0).
        comptime if dtype == DType.bool:
            return a | b
        elif dtype.is_floating_point():
            # On the bits: a subnormal a is nonzero (Apple GPUs flush it in
            # a float compare; MPS has no heaviside kernel, torch falls back
            # to the CPU's), NaN is neither zero nor positive.
            comptime u = _uint_of[dtype]()
            comptime sign = Scalar[u](1) << Scalar[u](size_of[dtype]() * 8 - 1)
            var bits = bitcast[u, n](a)
            var zero = (bits & ~sign).eq(0)
            var positive = bits.lt(sign) & ~zero & ~isnan(a)
            return zero.select(
                b, positive.select(SIMD[dtype, n](1), SIMD[dtype, n](0))
            )
        else:
            return a.eq(0).select(b, a.gt(0).cast[dtype]())
    elif kind == "fmax" or kind == "fmin":
        # MaxMinElementwiseKernel.cu: C's fmax/fmin for floats (a NaN
        # operand yields the other one), maximum/minimum otherwise.
        comptime if dtype == DType.bool:
            comptime if kind == "fmax":
                return a | b
            else:
                return a & b
        else:
            var r = max(a, b) if kind == "fmax" else min(a, b)
            comptime if dtype.is_floating_point():
                r = isnan(a).select(b, r)
                r = isnan(b).select(a, r)
            return r
    elif kind == "fmod":
        # BinaryRemainderKernel.cu: C's fmod (the dividend's sign), exact:
        # libdevice's fmodf on NVIDIA (its approximate-quotient fast path
        # beats the bit-at-a-time long division: fmod bf16 4096^2 was 3.2x
        # torch before), the exact long division elsewhere (Metal's fmod,
        # ROCm's ocml fmod are exact too); float64 by musl's exact fmod.
        comptime if dtype.is_floating_point():
            var r = SIMD[dtype, n]()
            comptime for i in range(n):
                comptime if dtype == DType.float64:
                    r[i] = _fmod_f64_exact_scalar(
                        a[i].cast[DType.float64](), b[i].cast[DType.float64]()
                    ).cast[dtype]()
                else:
                    r[i] = _fmod_float32(
                        a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                    ).cast[dtype]()
            return r
        else:
            # C's %, truncating (Mojo's `%` floors: move a remainder whose
            # sign differs from the dividend's back by one divisor); 0 for
            # a zero divisor (CPU raises, CUDA leaves it undefined).
            var safe = b.eq(0).select(SIMD[dtype, n](1), b)
            var r = a % safe
            comptime if dtype.is_signed():
                var fix = r.ne(0) & (r.lt(0) ^ a.lt(0))
                r = fix.select(r - safe, r)
            return b.eq(0).select(SIMD[dtype, n](0), r)
    elif kind == "gcd" or kind == "lcm":
        var r = SIMD[dtype, n]()
        comptime for i in range(n):
            var g = _gcd_scalar(a[i], b[i])
            comptime if kind == "gcd":
                r[i] = g
            else:
                # GcdLcmKernel.cu: (g == 0) ? 0 : abs(a / g * b), with C++'s
                # promotion of the narrow integers to int before the abs.
                comptime wide = DType.int64 if size_of[
                    dtype
                ]() == 8 else DType.int32
                var prod = (a[i] // g).cast[wide]() * b[i].cast[wide]()
                r[i] = 0 if g == 0 else abs(prod).cast[dtype]()
        return r
    elif kind == "lshift" or kind == "rshift":
        # BinaryShiftOpsKernels.cu: a negative or too-wide shift gives 0
        # (left) or the sign fill (right) instead of C's UB.
        comptime width_bits = size_of[dtype]() * 8
        comptime u = _uint_of[dtype]()
        comptime if kind == "lshift":
            var bad = bitcast[u, n](b).ge(Scalar[u](width_bits))
            comptime if dtype.is_signed():
                bad = bad | b.lt(0)
            var sh = bad.select(SIMD[u, n](0), bitcast[u, n](b))
            var r = bitcast[dtype, n](bitcast[u, n](a) << sh)
            return bad.select(SIMD[dtype, n](0), r)
        else:
            comptime max_shift = width_bits - (1 if dtype.is_signed() else 0)
            var bad = bitcast[u, n](b).ge(Scalar[u](max_shift))
            comptime if dtype.is_signed():
                bad = bad | b.lt(0)
            var sh = bad.select(SIMD[dtype, n](max_shift), b)
            return a >> sh
    elif kind == "clamp":
        # ClampKernel (clamp_kernel_cuda): NaN value, then NaN bounds win,
        # else min(max(v, lo), hi).
        var r = min(max(a, b), c)
        comptime if dtype.is_floating_point():
            r = isnan(c).select(c, r)
            r = isnan(b).select(b, r)
            r = isnan(a).select(a, r)
        return r
    elif kind == "maximum" or kind == "minimum":
        # MaxMinElementwiseKernel.cu: or/and for bool, NaN propagating.
        comptime if dtype == DType.bool:
            return (a | b) if kind == "maximum" else (a & b)
        else:
            var r = max(a, b) if kind == "maximum" else min(a, b)
            comptime if dtype.is_floating_point():
                r = isnan(b).select(b, r)
                r = isnan(a).select(a, r)
            return r
    elif kind == "ipow":
        # Pow.h powi: square-and-multiply; a negative exponent gives 1 for
        # base 1, +-1 for base -1 by parity, 0 otherwise.
        comptime if dtype.is_integral():
            var r = SIMD[dtype, n]()
            comptime for i in range(n):
                r[i] = _powi(a[i], b[i])
            return r
        else:
            comptime assert False, "ipow takes integers"
    elif kind == "frexp_mantissa":
        return _frexp_mantissa(a)
    else:
        comptime assert False, "unknown native pointwise kind"


@always_inline
def _ldexp[
    w: DType, n: Int
](x: SIMD[w, n], e: SIMD[w, n]) -> SIMD[w, n] where w.is_floating_point():
    """x * 2^e, correctly rounded (musl's scalbnf / scalbn): at most two
    exact power-of-two steps bring the exponent into the normal range, and
    the last product is the only rounding, so a subnormal result is rounded
    once. `e` is integral; beyond +-3 times the exponent range the result
    has saturated, so it is clamped there first. Zero, inf and NaN pass
    through the products unchanged."""
    comptime f64 = w == DType.float64
    comptime emax = 1023 if f64 else 127
    comptime emin = -1022 if f64 else -126
    comptime mbits = 53 if f64 else 24
    comptime lim = 3 * emax
    var y = x
    var k = min(max(e, SIMD[w, n](-lim)), SIMD[w, n](lim)).cast[DType.int32]()
    comptime up = SIMD[w, n](2.0**emax)
    comptime down = SIMD[w, n](2.0 ** (emin + mbits))
    comptime step_down = -emin - mbits  # the exponent one down step removes
    comptime EMAX = SIMD[DType.int32, n](emax)
    comptime EMIN = SIMD[DType.int32, n](emin)
    comptime STEP = SIMD[DType.int32, n](step_down)
    var m = k.gt(EMAX)
    y = m.select(y * up, y)
    k = m.select(k - EMAX, k)
    m = k.gt(EMAX)
    y = m.select(y * up, y)
    k = m.select(k - EMAX, k)
    k = min(k, EMAX)
    m = k.lt(EMIN)
    y = m.select(y * down, y)
    k = m.select(k + STEP, k)
    m = k.lt(EMIN)
    y = m.select(y * down, y)
    k = m.select(k + STEP, k)
    k = max(k, EMIN)
    comptime if f64:
        var scale = bitcast[DType.float64](
            (k.cast[DType.int64]() + SIMD[DType.int64, n](1023))
            << SIMD[DType.int64, n](52)
        )
        return y * scale.cast[w]()
    else:
        var scale = bitcast[DType.float32](
            (k + SIMD[DType.int32, n](127)) << SIMD[DType.int32, n](23)
        )
        return y * scale.cast[w]()


@always_inline
def _opaque[w: DType, n: Int](v: SIMD[w, n]) -> SIMD[w, n]:
    """`v`, through a copy the optimizer cannot see into (NVIDIA only)."""
    comptime if is_nvidia_gpu() and w == DType.float32:
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = inlined_assembly[
                "mov.b32 $0, $1;",
                Scalar[w],
                constraints="=f,f",
                has_side_effect=False,
            ](v[i])
        return r
    else:
        return v


@always_inline
def _mul_rn[w: DType, n: Int](a: SIMD[w, n], b: SIMD[w, n]) -> SIMD[w, n]:
    """A rounded product that no later add may absorb: `mul.rn` on NVIDIA
    (ptxas contracts a plain `mul` with the add consuming it, even through
    a `mov`), for a product that is its own kernel -- or that nvcc rounded
    before the add -- in the CUDA code being matched."""
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
def is_scalar_t_kind[kind: StaticString]() -> Bool:
    """The loss kinds, whose CUDA kernels compute in scalar_t: each
    operation of the C++ expression is rounded to the operand dtype (a no-op
    for float32 / float64), so half results match stock torch's."""
    return (
        kind == "mse"
        or kind == "mse_backward"
        or kind == "smooth_l1"
        or kind == "smooth_l1_backward"
        or kind == "huber"
        or kind == "huber_backward"
        or kind == "bce"
        or kind == "bce_backward"
        or kind == "bce_logits"
        or kind == "mul_scale"
    )


@always_inline
def _st_round[
    dtype: DType, w: DType, n: Int
](v: SIMD[w, n], fence: Bool) -> SIMD[w, n]:
    """A c10::Half / c10::BFloat16 result of the scalar_t loss kernels:
    rounded, and kept rounded -- LLVM narrows the float32 arithmetic to f16
    and then contracts a product into the next add (fma.rn.f16), a rounding
    CUDA's scalar_t code never skips (nor MPS, where every step of a
    composite is its own kernel). The opaque copy stops that on NVIDIA; on
    Apple GPUs (no inline assembly) an integer XOR with `fence`, a runtime
    False, does. float32 / float64 keep nvcc's own default contraction."""
    comptime if dtype == w:
        return v
    elif is_apple_gpu():
        var bits = bitcast[DType.uint16, n](v.cast[dtype]())
        bits = bits ^ SIMD[DType.uint16, n](UInt16(1 if fence else 0))
        return bitcast[dtype, n](bits).cast[w]()
    else:
        return _opaque(v.cast[dtype]().cast[w]())


@always_inline
def _scalar_t[
    kind: StaticString, dtype: DType, n: Int
](
    a: SIMD[dtype, n],
    b: SIMD[dtype, n],
    c: SIMD[dtype, n],
    p: SIMD[param_dtype[dtype](), 4],
) -> SIMD[dtype, n] where dtype.is_floating_point():
    comptime w = wide_dtype[dtype]()

    # Apple GPUs: the rounding fence of `_st_round` (p3 is never a NaN).
    var fence = (
        p[3].cast[DType.float32]().to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)
    ) > UInt32(0x7F800000)

    var x = a.cast[w]()
    var y = b.cast[w]()
    var z = c.cast[w]()
    var q = p.cast[w]()
    comptime zero = SIMD[w, n](0)
    comptime one = SIMD[w, n](1)
    comptime half = SIMD[w, n](0.5)
    var res: SIMD[w, n]
    comptime if kind == "mse":
        # BinaryMiscOpsKernels.cu mse_kernel_cuda: diff = a - b; diff * diff.
        var d = _st_round[dtype](x - y, fence)
        res = d * d
    elif kind == "mse_backward":
        # PointwiseOpsKernel.cu: alpha * (a - b) * c, a = input, b = target,
        # c = grad; p0 = alpha (norm rounded to scalar_t by the host).
        res = _st_round[dtype](q[0] * _st_round[dtype](x - y, fence), fence) * z
    elif kind == "smooth_l1":
        # p0 = beta (scalar_t): z < beta ? 0.5 * z * z / beta : z - 0.5 * beta.
        # z = ::abs(a - b) is a float (the half difference, widened), so
        # everything after it is float math rounded once, but 0.5 * beta
        # (scalar_t * scalar_t).
        var d = abs(_st_round[dtype](x - y, fence))
        var beta = SIMD[w, n](q[0])
        var quad = half * d * d / beta
        res = d.lt(beta).select(quad, d - _st_round[dtype](half * beta, fence))
    elif kind == "smooth_l1_backward":
        # a = input, b = target, c = grad; p0 = norm, p1 = beta (scalar_t).
        var d = _st_round[dtype](x - y, fence)
        var norm = SIMD[w, n](q[0])
        var beta = SIMD[w, n](q[1])
        var inner = (
            _st_round[dtype](_st_round[dtype](norm * d, fence) * z, fence)
            / beta
        )
        res = d.lt(-beta).select(-norm * z, d.gt(beta).select(norm * z, inner))
        res = isnan(d).select(inner, res)
    elif kind == "huber":
        # p0 = delta: z < delta ? 0.5 * z * z : delta * (z - 0.5 * delta),
        # in float after z = ::abs(a - b) as smooth_l1.
        var d = abs(_st_round[dtype](x - y, fence))
        var delta = SIMD[w, n](q[0])
        var lin = delta * (d - _st_round[dtype](half * delta, fence))
        res = d.lt(delta).select(half * d * d, lin)
    elif kind == "huber_backward":
        # a = input, b = target, c = grad; p0 = norm, p1 = delta.
        var d = _st_round[dtype](x - y, fence)
        var norm = SIMD[w, n](q[0])
        var delta = SIMD[w, n](q[1])
        var g = _st_round[dtype](norm * z, fence)
        var mid = _st_round[dtype](norm * d, fence) * z
        res = d.lt(-delta).select(
            _st_round[dtype](-norm * z, fence) * delta,
            d.gt(delta).select(g * delta, mid),
        )
        res = isnan(d).select(mid, res)
    elif kind == "bce":
        # Loss.cu binary_cross_entropy_out_cuda, a = input, b = target, c =
        # weight (1 without one; `loss.mul_(weight)` is its own rounding).
        var log_x = max(_st_round[dtype](_log(x), fence), SIMD[w, n](-100))
        var log_1mx = max(_st_round[dtype](_log1p(-x), fence), SIMD[w, n](-100))
        var loss = _st_round[dtype](
            _st_round[dtype](_st_round[dtype](y - one, fence) * log_1mx, fence)
            - _st_round[dtype](y * log_x, fence),
            fence,
        )
        res = loss * z
    elif kind == "bce_backward":
        # a = grad, b = input, c = target; p0 = the mean's 1 / numel as
        # `div_` computes it (a product with the opmath reciprocal), 1 when
        # there is none: grad * (x - t) / max((1 - x) * x, eps).
        # EPSILON is the float 1e-12 converted to scalar_t (0 in half).
        var eps = _st_round[dtype](SIMD[w, n](Float32(1e-12).cast[w]()), fence)
        var denom = max(
            _st_round[dtype](_st_round[dtype](one - y, fence) * y, fence), eps
        )
        res = (
            _st_round[dtype](
                _st_round[dtype](x * _st_round[dtype](y - z, fence), fence)
                / denom,
                fence,
            )
            * q[0]
        )
    elif kind == "bce_logits":
        # Loss.cpp binary_cross_entropy_with_logits, a = input, b = target,
        # c = pos_weight (1 without one): log_sigmoid(x) (*= (pw - 1) * t + 1),
        # then (1 - t) * x - that.
        # Each step is its own ATen kernel, so float32 products are rounded
        # before the next add too (no contraction across kernels).
        var ls = _st_round[dtype](min(zero, x) - _log1p(_exp(-abs(x))), fence)
        var lw = _st_round[dtype](
            _st_round[dtype](
                _mul_rn(_st_round[dtype](z - one, fence), y), fence
            )
            + one,
            fence,
        )
        ls = _st_round[dtype](_mul_rn(ls, lw), fence)
        res = (
            _st_round[dtype](
                _mul_rn(_st_round[dtype](one - y, fence), x), fence
            )
            - ls
        )
    elif kind == "mul_scale":
        # `t.mul_(w)` then a mean's `div_(numel)` (p0 = 1 / numel, or 1).
        res = _st_round[dtype](x * y, fence) * q[0]
    else:
        comptime assert False, "unknown scalar_t kind"
    return res.cast[dtype]()


# ---------------------------------------------------------------------------
# kinds computed in the opmath type
# ---------------------------------------------------------------------------


@always_inline
def _wide[
    kind: StaticString, w: DType, n: Int
](a: SIMD[w, n], b: SIMD[w, n], c: SIMD[w, n], p: SIMD[w, 4]) -> SIMD[
    w, n
] where w.is_floating_point():
    comptime zero = SIMD[w, n](0)
    comptime one = SIMD[w, n](1)
    # --- binary math -------------------------------------------------------
    comptime if kind == "atan2":
        # BinaryGeometricKernels.cu's ::atan2. float32 (and the half types,
        # widened): the CUDA atan2f, whose special values are selected on
        # the bits (it also stands for ROCm's ocml atan2f and MPS's
        # `precise::atan2(float, float)`); float64: fdlibm's atan2.
        var r = SIMD[w, n]()
        comptime for i in range(n):
            comptime if w == DType.float64:
                r[i] = _atan2_f64(
                    a[i].cast[DType.float64](), b[i].cast[DType.float64]()
                ).cast[w]()
            else:
                r[i] = cuda_atan2f(
                    a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                ).cast[w]()
        return r
    elif kind == "hypot":
        # BinaryGeometricKernels.cu (::hypot). Float32 squares in float64,
        # which is exact, and rounds the root once; float64 scales by the
        # larger magnitude.
        comptime if w == DType.float32 and is_nvidia_gpu():
            # The CUDA hypotf itself: float-only, bit for bit.
            var r = SIMD[w, n]()
            comptime for i in range(n):
                r[i] = cuda_hypotf(
                    a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                ).cast[w]()
            return r
        elif w == DType.float32 and is_apple_gpu():
            # c10::metal::hypot(|a|, |b|) (BinaryKernel.metal hypot_functor,
            # float for every input type): a sqrt(1 + (b/a)^2) with its a == b
            # and 1 + r == 1 cases; unordered max/min give NaN.
            var ax = abs(a)
            var bx = abs(b)
            var mx = max(ax, bx)
            var mn = min(ax, bx)
            var q = mn / mx
            var rr = q * q
            var s1 = ieee_sqrt(rr + 1)
            var h1 = mx * 1.41421356237309504880
            var h2 = mx + mx * rr / 2
            var h3 = mx * s1
            var res = (s1.eq(1) & rr.gt(0)).select(h2, h3)
            res = mx.eq(mn).select(h1, res)
            res = (isnan(a) | isnan(b)).select(SIMD[w, n](nan[w]()), res)
            return (isinf(a) | isinf(b)).select(SIMD[w, n](inf[w]()), res)
        elif w == DType.float32:
            # ROCm's ocml hypotf is correctly rounded but for an ulp: the
            # squares in float64 (exact) and one rounded root.
            var ad = a.cast[DType.float64]()
            var bd = b.cast[DType.float64]()
            var r = ieee_sqrt(ad * ad + bd * bd).cast[w]()
            return (isinf(a) | isinf(b)).select(SIMD[w, n](inf[w]()), r)
        else:
            # float64: cuda_hypotf's scheme on the double bits (both scaled
            # by a power of two from the larger exponent, one fma-summed
            # square, an IEEE root, the scale restored).
            var r = SIMD[w, n]()
            comptime for i in range(n):
                r[i] = _hypot_f64(
                    a[i].cast[DType.float64](), b[i].cast[DType.float64]()
                ).cast[w]()
            return r
    elif kind == "logaddexp" or kind == "logaddexp2":
        # LogAddExpKernel.cu: inf == inf keeps it, else
        # m + log1p(exp(-|a - b|)) (exp2 and a 1/ln2 factor for base 2).
        var m = max(a, b)
        var d = -abs(a - b)
        var r: SIMD[w, n]
        comptime if kind == "logaddexp":
            r = m + _log1p(_exp(d))
        else:
            comptime ln2 = 0.693147180559945309417232121458
            comptime inv_ln2 = 1.44269504088896340735992468100
            # ::exp2: libdevice's exp2f (one ex2.approx.f32 on NVIDIA), and
            # for float64 an exp2 exact at integers (exp(d ln 2) is not:
            # 2^-1070 came out thousands of subnormal ulps off).
            var e2 = SIMD[w, n]()
            comptime for i in range(n):
                comptime if w == DType.float32:
                    e2[i] = nv_exp2f(d[i].cast[DType.float32]()).cast[w]()
                else:
                    e2[i] = _exp2_f64(d[i].cast[DType.float64]()).cast[w]()
            r = m + _log1p(e2) * inv_ln2
        r = (isinf(a) & a.eq(b)).select(a, r)
        return (isnan(a) | isnan(b)).select(SIMD[w, n](nan[w]()), r)
    elif kind == "xlogy" or kind == "xlog1py":
        # BinaryMiscOpsKernels.cu: NaN y -> NaN, x == 0 -> 0, else x log(y).
        # MPS (c10::metal::xlogy) returns x itself for x == 0, keeping -0.
        var l = _log(b) if kind == "xlogy" else _log1p(b)
        var r = a * l
        comptime if is_apple_gpu():
            r = a.eq(0).select(a, r)
        else:
            r = a.eq(0).select(zero, r)
        r = isnan(a).select(a, r)
        return isnan(b).select(SIMD[w, n](nan[w]()), r)
    elif kind == "igamma" or kind == "igammac":
        # IGammaKernel.cu calc_igamma / calc_igammac, float accscalar_t
        # (special_math.mojo; stock torch has no float64 route to match here).
        var r = SIMD[w, n]()
        comptime for i in range(n):
            var ai = a[i].cast[DType.float32]()
            var xi = b[i].cast[DType.float32]()
            comptime if kind == "igamma":
                r[i] = igamma_f(ai, xi).cast[w]()
            else:
                r[i] = igammac_f(ai, xi).cast[w]()
        return r
    elif kind == "zeta":
        var r = SIMD[w, n]()
        comptime for i in range(n):
            comptime if w == DType.float32:
                # CUDA's zeta_string runs in float (the jiterator's T, on
                # ROCm as well), MPS's c10::metal::zeta too:
                # special_math.zeta_f, about 3x cheaper than the double
                # loop (special_zeta f32 357x789: 1.59x torch before).
                r[i] = zeta_f(
                    a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                ).cast[w]()
            else:
                r[i] = _zeta_scalar(
                    a[i].cast[DType.float64](), b[i].cast[DType.float64]()
                ).cast[w]()
        return r
    elif is_polynomial[kind]():
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = _polynomial[kind, w](a[i], b[i])
        return r
    elif kind == "lerp_scalar":
        # lerp.Scalar: the weight is a parameter in opmath (Lerp.cu's
        # lerp_scalar_kernel converts it with weight.to<opmath_t>()).
        # (MPS's lerp_alpha drops the small-weight branch, a + w * (b - a)
        # throughout, which is less accurate near w = 1: kept everywhere.)
        var diff = b - a
        var weight = SIMD[w, n](p[0])
        var small = abs(weight).lt(0.5)
        return small.select(weight.fma(diff, a), b - diff * (one - weight))
    elif kind == "lerp":
        # Lerp.h: self + weight * (end - self) for |weight| < 0.5, else
        # end - (end - self) * (1 - weight).
        # (MPS's lerp_tensor is fma(w, e - s, s) for every weight, less
        # accurate near w = 1: the branch is kept on Apple GPUs too.)
        var diff = b - a
        var small = abs(c).lt(0.5)
        return small.select(c.fma(diff, a), b - diff * (one - c))
    elif kind == "pow_scalar_base":
        # pow(Scalar base, Tensor exponent): PowKernel.cu's cpu-scalar base,
        # base ** exponent in opmath (the base rounded to the tensor's own
        # tensor dtype, which the host already did: `scalar_value<scalar_t>`).
        # pow_math.torch_pow: C's special cases (std.math.pow's GPU lowering
        # returns 1 for an infinite exponent and inf for a NaN one) around
        # the CUDA float powf (float32 and the half types widened, on
        # NVIDIA and Apple) or a double-double float64 core, what stock
        # torch runs.
        return torch_pow(SIMD[w, n](p[0]), a)
    elif kind == "pow_tensor_scalar":
        # PowKernel.cu's pow_tensor_scalar_kernel (the float64 route; the
        # float types take the elementwise family's PowScalarSpec): p0 = the
        # exponent. 2, 3 and -2 are products, 0.5 / -0.5 / -1 the sqrt /
        # rsqrt / reciprocal kernels, the rest the full pow.
        var e = p[0]
        if e == 2:
            return a * a
        elif e == 3:
            return (a * a) * a
        elif e == -2:
            return 1 / (a * a)
        elif e == 0.5:
            return ieee_sqrt(a)
        elif e == -0.5:
            return 1 / ieee_sqrt(a)
        elif e == -1:
            return 1 / a
        return torch_pow(a, SIMD[w, n](e))
    elif kind == "rsub_alpha" or kind == "rsub_alpha_scalar":
        # rsub(self, other, alpha) is sub(other, self, alpha), CUDA's add
        # kernel with -alpha: other + (-alpha) * self in opmath, one fma
        # (a = self, b = other or p1 = a Python-number other, kept in opmath
        # like the kernel's cpu scalar; p0 = alpha.to<opmath_t>()).
        var other = b
        comptime if kind == "rsub_alpha_scalar":
            other = SIMD[w, n](p[1])
        return SIMD[w, n](-p[0]).fma(a, other)
    elif kind == "scale":
        # deg2rad / rad2deg: `mul_out(result, self, wrapped_scalar_tensor(c))`,
        # the constant rounded to opmath like any CPU-scalar operand of mul.
        return a * p[0]
    elif kind == "ldexp":
        # BinaryMiscOpsKernels.cu's ldexp_kernel_cuda: ::ldexp(x, int exp)
        # (half operands widened, rounded once by the caller's cast). b holds
        # the integer exponent, exact in the compute dtype up to where the
        # result saturates anyway.
        return _ldexp(a, b)
    elif kind == "ldexp_pow2":
        # BinaryOps.cpp's general ldexp: self * _pow2(self, other), where
        # _pow2 is pow(2, other) in its own dtype; p0 names that dtype when it
        # is narrower than the compute dtype (1 half, 2 bfloat16, 3 float32).
        # pow(2, other), as PowKernel.cu's scalar-base kernel runs it
        # (float for the float types, never float64: Apple GPUs have none).
        var pw = torch_pow(SIMD[w, n](2), b)
        if p[0] == 1:
            pw = pw.cast[DType.float16]().cast[w]()
        elif p[0] == 2:
            pw = pw.cast[DType.bfloat16]().cast[w]()
        elif p[0] == 3:
            pw = pw.cast[DType.float32]().cast[w]()
        return a * pw
    # --- activations (forward: a = x) ----------------------------------------
    elif kind == "elu":
        # ActivationEluKernel.cu, p = (alpha, scale, input_scale):
        # x > 0 ? x * scale : expm1(x * input_scale) * alpha * scale. (MPS's
        # kernel spells expm1 as exp(x) - 1, which loses the small-x digits
        # CPU and CUDA keep; expm1 is kept on every GPU.)
        var negcoef = p[0] * p[1]
        var r = a.gt(0).select(a * p[1], _expm1(a * p[2]) * negcoef)
        return isnan(a).select(a, r)
    elif kind == "hardshrink":
        # p0 = lambd rounded to the input dtype by the host; NaN is kept.
        var keep = a.ge(-p[0]) & a.le(p[0])
        return (keep & ~isnan(a)).select(zero, a)
    elif kind == "softshrink":
        var r = a.gt(p[0]).select(a - p[0], a.lt(-p[0]).select(a + p[0], zero))
        return isnan(a).select(a, r)
    elif kind == "hardsigmoid":
        # CUDA: min(max(x + 3, 0), 6) * one_sixth, one_sixth being
        # `opmath_t(1.0f / 6.0f)` (float's 1/6 even for double); MPS divides
        # (x + 3) / 6 and clamps to [0, 1].
        var r: SIMD[w, n]
        comptime if is_apple_gpu():
            r = min(max((a + 3) / 6, zero), one)
        else:
            r = min(max(a + 3, zero), SIMD[w, n](6)) * _one_sixth[w, n]()
        return isnan(a).select(a, r)
    elif kind == "hardswish":
        # CUDA: x * min(max(x + 3, 0), 6) * one_sixth; MPS: ... / 6.
        var r: SIMD[w, n]
        comptime if is_apple_gpu():
            r = a * min(max(a + 3, zero), SIMD[w, n](6)) / 6
        else:
            r = a * min(max(a + 3, zero), SIMD[w, n](6)) * _one_sixth[w, n]()
        return isnan(a).select(a, r)
    elif kind == "hardtanh":
        # hardtanh is clamp(x, min_val, max_val); NaN passes through.
        var r = min(max(a, SIMD[w, n](p[0])), SIMD[w, n](p[1]))
        return isnan(a).select(a, r)
    elif kind == "leaky_relu":
        var r = a.gt(0).select(a, a * p[0])
        return isnan(a).select(a, r)
    elif kind == "softplus":
        # p = (beta, threshold): x * beta > threshold ? x
        #                       : log1p(exp(x * beta)) / beta.
        var xb = a * p[0]
        var r = xb.gt(p[1]).select(a, _log1p(_exp(xb)) / p[0])
        return isnan(a).select(a, r)
    elif kind == "mish":
        # ActivationMishKernel.cu: x * tanh(log1p(exp(x))), tanh being
        # c10::cuda::compat::tanh (libdevice tanhf) on NVIDIA.
        comptime if is_nvidia_gpu():
            return a * _tanh_cuda(_log1p(_exp(a)))
        else:
            return a * _tanh(_log1p(_exp(a)))
    elif kind == "threshold":
        # p = (threshold, value), both rounded to the input dtype by the
        # host (ActivationThresholdKernel.cu computes in scalar_t).
        var r = a.le(p[0]).select(SIMD[w, n](p[1]), a)
        return isnan(a).select(a, r)
    elif kind == "log_sigmoid":
        # ActivationLogSigmoidKernel.cu: min(0, x) - log1p(exp(-|x|)).
        var r = min(zero, a) - _log1p(_exp(-abs(a)))
        return isnan(a).select(a, r)
    # --- activation backwards ------------------------------------------------
    elif kind == "elu_backward":
        # a = grad, b = self or result; p = (alpha, scale, input_scale,
        # is_result). A NaN b takes the positive branch (`b <= 0` is false).
        var negcoef = p[0] * p[1]
        var neg: SIMD[w, n]
        comptime if is_apple_gpu():
            # ActivationKernel.metal elu_backward_functor's association.
            if p[3] != 0:
                neg = a * (p[2] * (b + negcoef))
            else:
                neg = a * (p[2] * p[0] * p[1] * _exp(b * p[2]))
        else:
            if p[3] != 0:
                neg = a * p[2] * (b + negcoef)
            else:
                neg = a * p[2] * negcoef * _exp(b * p[2])
        return (b.le(0) & ~isnan(b)).select(neg, a * p[1])
    elif kind == "shrink_backward":
        # a = grad, b = self; hardshrink_backward and softshrink_backward.
        var zeroed = b.ge(-p[0]) & b.le(p[0]) & ~isnan(b)
        return zeroed.select(zero, a)
    elif kind == "hardsigmoid_backward":
        var inside = b.gt(-3) & b.lt(3) & ~isnan(b)
        return inside.select(a * _one_sixth[w, n](), zero)
    elif kind == "hardswish_backward":
        # x <= -3 ? 0 : (x < 3 ? g * (x / 3 + 0.5) : g); NaN x gives g.
        var mid = a * (b / 3 + 0.5)
        var r = b.le(-3).select(zero, b.lt(3).select(mid, a))
        return isnan(b).select(a, r)
    elif kind == "hardtanh_backward":
        var clipped = (b.le(p[0]) | b.ge(p[1])) & ~isnan(b)
        return clipped.select(zero, a)
    elif kind == "leaky_relu_backward":
        # a = self, b = grad (the CUDA iterator's operand order).
        return (a.gt(0) & ~isnan(a)).select(b, b * p[0])
    elif kind == "softplus_backward":
        # a = grad, b = self; p = (beta, threshold).
        var xb = b * p[0]
        var z = _exp(xb)
        var soft = a * z / (z + 1)
        return (xb.gt(p[1]) & ~isnan(xb)).select(a, soft)
    elif kind == "mish_backward":
        var s = _sigmoid(b)
        var t: SIMD[w, n]
        comptime if is_nvidia_gpu():
            t = _tanh_cuda(_log1p(_exp(b)))
        else:
            t = _tanh(_log1p(_exp(b)))
        return a * (t + b * s * (1 - t * t))
    elif kind == "silu_backward":
        var s = _sigmoid(b)
        comptime if is_apple_gpu():
            # ActivationKernel.metal: g * sig * (1 + x - x * sig).
            return a * s * (1 + b - b * s)
        else:
            return a * s * (1 + b * (1 - s))
    elif kind == "log_sigmoid_backward":
        # a = self, b = grad.
        var neg = a.lt(0)
        var max_deriv = neg.select(one, zero)
        var sign = neg.select(one, -one)
        var z = _exp(-abs(a))
        return b * (max_deriv - sign * (z / (1 + z)))
    elif kind == "logit_backward":
        # a = grad, b = self; p0 = eps, < 0 for none (BinaryMiscBackward-
        # OpsKernels.cu: lo = eps, hi = 1 - eps in the accumulate type).
        var d = a / (b * (1 - b))
        var outside: SIMD[DType.bool, n]
        if p[0] < 0:
            outside = (b.lt(0) | b.gt(1)) & ~isnan(b)
            return outside.select(SIMD[w, n](nan[w]()), d)
        outside = (b.lt(p[0]) | b.gt(1 - p[0])) & ~isnan(b)
        return outside.select(zero, d)
    elif kind == "gelu_backward_none" or kind == "gelu_backward_tanh":
        # ActivationGeluKernel.cu, a = grad, b = self; erf and exp are
        # libdevice's (erff / expf, erf / exp for double).
        # nvcc contracts `cdf + x * pdf` as fma(1 + erf, 0.5, x * pdf): the
        # product x * pdf is rounded first, the exact half product fused.
        comptime if kind == "gelu_backward_none":
            comptime kAlpha = Scalar[w](0.70710678118654752440)
            comptime kBeta = Scalar[w](0.39894228040143267794)
            var cdf = 0.5 * (1 + _erf(b * kAlpha))
            var pdf = _exp(-0.5 * b * b) * kBeta
            return a * (cdf + _mul_rn(b, pdf))
        else:
            comptime kBeta = Scalar[w](0.79788456080286535588)
            comptime kKappa = Scalar[w](0.044715)
            var x_sq = b * b
            var x_cube = x_sq * b
            var inner = kBeta * (b + kKappa * x_cube)
            var t = _tanh_cuda(inner)
            var left = 0.5 * b
            var right = 1 + t
            var left_derivative = 0.5 * right
            var tanh_derivative = 1 - t * t
            var inner_derivative = kBeta * (1 + Scalar[w](3) * kKappa * x_sq)
            var right_derivative = left * tanh_derivative * inner_derivative
            return a * (left_derivative + right_derivative)
    else:
        comptime assert False, "unknown pointwise kind"


@always_inline
def _erf[
    w: DType, n: Int
](x: SIMD[w, n]) -> SIMD[w, n] where w.is_floating_point():
    """`::erf` of the CUDA kernels: libdevice's erff / erf, per lane."""
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_erf(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_erff(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _rrelu[
    kind: StaticString, dtype: DType, n: Int
](
    a: SIMD[dtype, n], b: SIMD[dtype, n], p: SIMD[param_dtype[dtype](), 4]
) -> SIMD[dtype, n] where dtype.is_floating_point():
    """RreluWithNoise.cu, a = x, b = the uniform draw in scalar_t; p =
    (lower, range, lower - p0, range - p1) with range = upper - lower, split
    in two floats for float32 and the half types. CUDA computes the slope
    `r * range + lower` in double (range and lower are doubles; one DFMA)
    and assigns it to scalar_t; x <= 0 ? x * r : x in scalar_t, noise r or 1.
    Metal has no double: the slope is one float32 fma there."""
    var r: SIMD[dtype, n]
    comptime if is_apple_gpu():
        var bf = b.cast[DType.float32]()
        r = bf.fma(
            SIMD[DType.float32, n](p[1].cast[DType.float32]()),
            SIMD[DType.float32, n](p[0].cast[DType.float32]()),
        ).cast[dtype]()
    else:
        var lower = p[0].cast[DType.float64]() + p[2].cast[DType.float64]()
        var span = p[1].cast[DType.float64]() + p[3].cast[DType.float64]()
        var rd = b.cast[DType.float64]().fma(
            SIMD[DType.float64, n](span), SIMD[DType.float64, n](lower)
        )
        # A double assigned to c10::Half / c10::BFloat16 goes through float.
        comptime if dtype == DType.float64:
            r = rd.cast[dtype]()
        else:
            r = rd.cast[DType.float32]().cast[dtype]()
    var keep = a.le(0) & ~isnan(a)
    comptime if kind == "rrelu_noise":
        return keep.select(r, SIMD[dtype, n](1))
    else:
        comptime w = wide_dtype[dtype]()
        return keep.select((a.cast[w]() * r.cast[w]()).cast[dtype](), a)


@always_inline
def pointwise[
    kind: StaticString, dtype: DType, out_dtype: DType, n: Int
](
    a: SIMD[dtype, n],
    b: SIMD[dtype, n],
    c: SIMD[dtype, n],
    p: SIMD[param_dtype[dtype](), 4],
) -> SIMD[out_dtype, n]:
    """`kind` of (a, b, c) with scalar parameters `p`, rounded once into
    `out_dtype`. Unused operands and parameters are ignored."""
    comptime if kind == "frexp_exponent":
        return _frexp_exponent(a).cast[out_dtype]()
    elif kind == "hardtanh" and dtype.is_integral():
        # hardtanh is clamp_out(min_val, max_val): on integers, in scalar_t.
        var lo = SIMD[dtype, n](p[0].cast[dtype]())
        var hi = SIMD[dtype, n](p[1].cast[dtype]())
        return min(max(a, lo), hi).cast[out_dtype]()
    elif kind == "threshold" and dtype.is_integral():
        # ActivationThresholdKernel.cu on integers: in scalar_t.
        var thr = SIMD[dtype, n](p[0].cast[dtype]())
        return (
            a.le(thr)
            .select(SIMD[dtype, n](p[1].cast[dtype]()), a)
            .cast[out_dtype]()
        )
    elif kind == "rrelu_train" or kind == "rrelu_noise":
        comptime assert dtype.is_floating_point(), "rrelu takes floats"
        return _rrelu[kind](a, b, p).cast[out_dtype]()
    elif is_native_kind[kind]():
        return _native[kind](a, b, c).cast[out_dtype]()
    elif is_scalar_t_kind[kind]():
        comptime assert dtype.is_floating_point(), "losses take floats"
        return _scalar_t[kind](a, b, c, p).cast[out_dtype]()
    else:
        var q = p
        comptime if dtype == DType.float64:
            return _wide[kind](
                a.cast[DType.float64](),
                b.cast[DType.float64](),
                c.cast[DType.float64](),
                q.cast[DType.float64](),
            ).cast[out_dtype]()
        else:
            return _wide[kind](
                a.cast[DType.float32](),
                b.cast[DType.float32](),
                c.cast[DType.float32](),
                q.cast[DType.float32](),
            ).cast[out_dtype]()


@always_inline
def metal_rounds_params[kind: StaticString]() -> Bool:
    """The kinds whose parameters MPS hands its Metal kernels as the
    tensor's own type (ActivationKernel.mm: `ELUParams<scalar_t>{alpha.to<
    scalar_t>(), ...}`, and the scalar_t alpha of leaky_relu's
    REGISTER_UNARY_ALPHA_OP) before widening them in the kernel; CUDA keeps
    them in opmath (float). The launcher rounds them on the host."""
    return (
        kind == "elu"
        or kind == "elu_backward"
        or kind == "leaky_relu"
        or kind == "leaky_relu_backward"
    )
