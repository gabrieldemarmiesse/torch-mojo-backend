# ===----------------------------------------------------------------------=== #
# Scalar ports of the Metal routines stock torch runs on MPS for the unary
# math ops of the elementwise family (Apple GPUs only).
#
# torch v2.14 computes these in float32 on MPS (Metal has no double):
# `aten/src/ATen/native/mps/kernels/UnaryKernel.metal` calls Metal's
# `precise::` library (asin, atan, log10, pow, exp) or the float ports of
# `c10/metal/special_math.h` / `c10/metal/expm1f.h` (erfc, erfinv, i0,
# lgamma, digamma, trigamma, polygamma, sinc, expm1), compiled with
# `-fno-fast-math` (cmake/Metal.cmake), so unqualified `::metal::exp` / `log`
# / `sin` / `tan` / `pow` are the precise variants too. The `llvm.air.*`
# intrinsics below are those same Metal library functions (`air.exp.f32` is
# `precise::exp`; the fast ones are `air.fast_*`).
#
# Mojo builds Apple kernels with no-NaNs / no-infs fast-math flags, so a
# comparison against NaN or an infinity may be folded away; the special
# values the C++ reaches through such a comparison are selected here from
# the bit pattern instead.
#
# Where the C++ template computes in the tensor's own type T (erfinv's
# `y * y`, i0's `fabs` / `exp` / `sqrt` on T), `_rt[dtype]` rounds the float
# intermediate to that type, as the half / bfloat arithmetic does.
# ===----------------------------------------------------------------------=== #

from std.collections import Array
from std.sys import llvm_intrinsic

comptime _INF = Float32(from_bits=UInt32(0x7F800000))
comptime _NAN = Float32(from_bits=UInt32(0x7FC00000))
comptime M_PI_F = Float32(3.14159265358979323846)


@always_inline
def _bits(x: Float32) -> UInt32:
    return x.to_bits[DType.uint32]()


@always_inline
def is_nan_f(x: Float32) -> Bool:
    return (_bits(x) & 0x7FFFFFFF) > 0x7F800000


@always_inline
def is_inf_f(x: Float32) -> Bool:
    return (_bits(x) & 0x7FFFFFFF) == 0x7F800000


@always_inline
def _rt[dtype: DType](x: Float32) -> Float32:
    """Round a float intermediate to the tensor type T and back."""
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        return x.cast[dtype]().cast[DType.float32]()
    else:
        return x


@always_inline
def _air[name: StaticString](x: Float32) -> Float32:
    return llvm_intrinsic[name, Float32, has_side_effect=False](x)


@always_inline
def air_exp(x: Float32) -> Float32:
    return _air["llvm.air.exp"](x)


@always_inline
def air_log(x: Float32) -> Float32:
    return _air["llvm.air.log"](x)


@always_inline
def air_log10(x: Float32) -> Float32:
    return _air["llvm.air.log10"](x)


@always_inline
def air_sin(x: Float32) -> Float32:
    return _air["llvm.air.sin"](x)


@always_inline
def air_tan(x: Float32) -> Float32:
    return _air["llvm.air.tan"](x)


@always_inline
def air_asin(x: Float32) -> Float32:
    return _air["llvm.air.asin"](x)


@always_inline
def air_atan(x: Float32) -> Float32:
    return _air["llvm.air.atan"](x)


@always_inline
def air_sqrt(x: Float32) -> Float32:
    return _air["llvm.air.sqrt"](x)


@always_inline
def air_sinpi(x: Float32) -> Float32:
    return _air["llvm.air.sinpi"](x)


@always_inline
def air_exp10(x: Float32) -> Float32:
    return _air["llvm.air.exp10"](x)


@always_inline
def air_pow(x: Float32, y: Float32) -> Float32:
    return llvm_intrinsic["llvm.air.pow", Float32, has_side_effect=False](x, y)


@always_inline
def _abs(x: Float32) -> Float32:
    return Float32(from_bits=_bits(x) & 0x7FFFFFFF)


@always_inline
def _copysign(mag: Float32, sgn: Float32) -> Float32:
    return Float32(
        from_bits=(_bits(mag) & 0x7FFFFFFF) | (_bits(sgn) & 0x80000000)
    )


@always_inline
def _trunc(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.trunc", Float32, has_side_effect=False](x)


@always_inline
def _floor(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.floor", Float32, has_side_effect=False](x)


@always_inline
def _rint(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.roundeven", Float32, has_side_effect=False](x)


@always_inline
def _fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return llvm_intrinsic["llvm.fma", Float32, has_side_effect=False](a, b, c)


@always_inline
def _fract(x: Float32) -> Float32:
    """`metal::fract`: x - floor(x), clamped below one."""
    return min(x - _floor(x), Float32(from_bits=UInt32(0x3F7FFFFF)))


# --------------------------------------------------------------------------- #
# UnaryKernel.metal functors
# --------------------------------------------------------------------------- #


@always_inline
def mps_exp2(x: Float32) -> Float32:
    """`exp2_functor`: precise::pow(2, x)."""
    return air_pow(Float32(2.0), x)


@always_inline
def mps_expm1f(a: Float32) -> Float32:
    """`c10/metal/expm1f.h` (Juffa's expm1f)."""
    var j = _fma(Float32(1.442695), a, Float32(12582912.0))
    j = j - Float32(12582912.0)
    var i = Int(j)
    var f = _fma(j, Float32(-6.93145752e-1), a)
    var s = f * f
    if a == Float32(0.0):
        s = a  # ensure -0 is passed through
    var r = Float32(1.97350979e-4)
    r = _fma(r, f, Float32(1.39309070e-3))
    r = _fma(r, f, Float32(8.33343994e-3))
    r = _fma(r, f, Float32(4.16668020e-2))
    r = _fma(r, f, Float32(1.66666716e-1))
    r = _fma(r, f, Float32(4.99999970e-1))
    var u = (f + Float32(0.5)) if j == Float32(1.0) else f
    var v = _fma(r, s, u)
    s = Float32(0.5)
    # ldexp(s, i), i within the exponent range for |a| <= 89 (larger a
    # takes the pow branch of `mps_expm1`).
    var t = s * air_pow(Float32(2.0), Float32(i))
    var y = t - s
    var x = (t - y) - s
    r = _fma(v, t, x) + y
    r = r + r
    if j == Float32(0.0):
        r = v
    if j == Float32(1.0):
        r = v + v
    return r


@always_inline
def mps_expm1(x: Float32) -> Float32:
    """`expm1_functor`: expm1f below 1e-5 in magnitude, exp(x) - 1 above."""
    if is_nan_f(x):
        return x
    if _abs(x) < Float32(1e-5):
        # expm1f's overflow / underflow branch is unreachable this close to 0.
        return mps_expm1f(x)
    if is_inf_f(x):
        return _INF if x > 0 else Float32(-1.0)
    return air_exp(x) - Float32(1.0)


# --------------------------------------------------------------------------- #
# c10/metal/special_math.h
# --------------------------------------------------------------------------- #


@always_inline
def _mps_erf(a: Float32) -> Float32:
    var t = _abs(a)
    var s = a * a
    if t > Float32(0.927734375):
        var r = _fma(Float32(-1.72853470e-5), t, Float32(3.83197126e-4))
        var u = _fma(Float32(-3.88396438e-3), t, Float32(2.42546219e-2))
        r = _fma(r, s, u)
        r = _fma(r, t, Float32(-1.06777877e-1))
        r = _fma(r, t, Float32(-6.34846687e-1))
        r = _fma(r, t, Float32(-1.28717512e-1))
        r = _fma(r, t, -t)
        r = Float32(1.0) - air_exp(r)
        return _copysign(r, a)
    var r = Float32(-5.96761703e-4)
    r = _fma(r, s, Float32(4.99119423e-3))
    r = _fma(r, s, Float32(-2.67681349e-2))
    r = _fma(r, s, Float32(1.12819925e-1))
    r = _fma(r, s, Float32(-3.76125336e-1))
    r = _fma(r, s, Float32(1.28379166e-1))
    return _fma(r, a, a)


@always_inline
def mps_erfc(xf: Float32) -> Float32:
    """`c10::metal::erfc` (the fast::divide quotients as plain divisions)."""
    if is_nan_f(xf):
        return xf
    var a = _abs(xf)
    if a <= Float32(0.927734375):
        return Float32(1.0) - _mps_erf(xf)
    var t = min(a, Float32(10.5))
    var q = (t - Float32(4.0)) / (t + Float32(4.0))
    var p = Float32(5.271875e-03)
    p = _fma(p, q, Float32(-1.6534764e-02))
    p = _fma(p, q, Float32(3.702093e-02))
    p = _fma(p, q, Float32(-6.6275224e-02))
    p = _fma(p, q, Float32(9.375815e-02))
    p = _fma(p, q, Float32(-1.01042934e-01))
    p = _fma(p, q, Float32(6.809548e-02))
    p = _fma(p, q, Float32(1.5379757e-02))
    p = _fma(p, q, Float32(-1.396211e-01))
    p = _fma(p, q, Float32(2.3299512e-01))
    var s = t * t
    var e = air_exp(-s)
    e = _fma(-e, _fma(t, t, -s), e)
    var den = _fma(Float32(2.0), t, Float32(1.0))
    var r = ((Float32(1.0) + p) / den) * e
    return (Float32(2.0) - r) if xf < Float32(0.0) else r


@always_inline
def mps_erfinv[dtype: DType](y: Float32) -> Float32:
    """`c10::metal::erfinv` (`y * y` in the tensor type)."""
    if is_nan_f(y):
        return y
    var y_abs = _abs(y)
    if y_abs >= Float32(1.0):
        return _NAN if y_abs > Float32(1.0) else _copysign(_INF, y)
    if y_abs <= Float32(0.7):
        var z = _rt[dtype](y * y)
        var num = (
            (Float32(-0.140543331) * z + Float32(0.914624893)) * z
            + Float32(-1.645349621)
        ) * z + Float32(0.886226899)
        var dem = (
            (
                (Float32(0.012229801) * z + Float32(-0.329097515)) * z
                + Float32(1.442710462)
            )
            * z
            + Float32(-2.118377725)
        ) * z + Float32(1.0)
        return y * num / dem
    var z = air_sqrt(
        Float32(-1.0) * air_log((Float32(1.0) - y_abs) / Float32(2.0))
    )
    var num = (
        (Float32(1.641345311) * z + Float32(3.429567803)) * z
        + Float32(-1.624906493)
    ) * z + Float32(-1.970840454)
    var dem = (Float32(1.637067800) * z + Float32(3.543889200)) * z + Float32(
        1.0
    )
    return _copysign(num, y) / dem


@always_inline
def _chbevl_f[n: Int, //, c: Array[Float32, n]](x: Float32) -> Float32:
    """`c10::metal::chbevl` (unfused, as the Metal source spells it)."""
    comptime c0 = c[0]
    var b0 = c0
    var b1 = Float32(0.0)
    var b2 = Float32(0.0)
    comptime for i in range(1, n):
        b2 = b1
        b1 = b0
        comptime ci = c[i]
        b0 = x * b1 - b2 + ci
    return Float32(0.5) * (b0 - b2)


comptime _I0_A: Array[Float32, 30] = [
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
]

comptime _I0_B: Array[Float32, 25] = [
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]


@always_inline
def mps_i0[dtype: DType](a: Float32) -> Float32:
    """`c10::metal::i0` (fabs, exp and sqrt of x in the tensor type)."""
    if is_nan_f(a):
        return a
    var x = _abs(a)
    if is_inf_f(x):
        return _INF
    if x <= Float32(8.0):
        var y = (x / Float32(2.0)) - Float32(2.0)
        return _rt[dtype](air_exp(x)) * _chbevl_f[_I0_A](y)
    return (
        _rt[dtype](air_exp(x))
        * _chbevl_f[_I0_B](Float32(32.0) / x - Float32(2.0))
    ) / _rt[dtype](air_sqrt(x))


@always_inline
def mps_sinc(a: Float32) -> Float32:
    """`c10::metal::sinc`: precise::sin(pi a) / (pi a), 1 at 0."""
    if a == Float32(0.0):
        return Float32(1.0)
    var product = M_PI_F * a
    return air_sin(product) / product


comptime _GAMMA_NUM: Array[Float32, 8] = [
    -1.71618513886549492533811e0,
    2.47656508055759199108314e1,
    -3.79804256470945635097577e2,
    6.29331155312818442661052e2,
    8.66966202790413211295064e2,
    -3.14512729688483675254357e4,
    -3.61444134186911729807069e4,
    6.64561438202405440627855e4,
]
comptime _GAMMA_DEN: Array[Float32, 8] = [
    -3.08402300119738975254353e1,
    3.15350626979604161529144e2,
    -1.01515636749021914166146e3,
    -3.10777167157231109440444e3,
    2.25381184209801510330112e4,
    4.75584627752788110767815e3,
    -1.34659959864969306392456e5,
    -1.15132259675553483497211e5,
]
comptime _LGAMMA_EXP: Array[Float32, 8] = [
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
]


@always_inline
def _log_gamma_asymptotic(abs_x: Float32) -> Float32:
    """`log_gamma`'s branch for |x| >= 12 (Abramowitz and Stegun 6.1.41)."""
    comptime HALF_LOG_TWO_PI = Float32(0.91893853320467274178032973640562)
    var z = Float32(1.0) / (abs_x * abs_x)
    comptime c7 = _LGAMMA_EXP[7]
    var sum = c7
    comptime for k in range(7):
        comptime ck = _LGAMMA_EXP[6 - k]
        sum *= z
        sum += ck
    var series = sum / abs_x
    return (
        (abs_x - Float32(0.5)) * air_log(abs_x)
        - abs_x
        + HALF_LOG_TWO_PI
        + series
    )


@always_inline
def mps_gamma(x: Float32) -> Float32:
    """`c10::metal::gamma` (John D Cook's approximation)."""
    if x < Float32(0.001):
        comptime EULER_MASCHERONI = Float32(0.577215664901532860606512090)
        return Float32(1.0) / (x * (Float32(1.0) + EULER_MASCHERONI * x))
    if x >= Float32(12.0):
        return air_exp(_log_gamma_asymptotic(x))
    var y = Float32(1.0) + _fract(x)
    var num = Float32(0.0)
    var den = Float32(1.0)
    var z = y - Float32(1.0)
    comptime for i in range(8):
        comptime ni = _GAMMA_NUM[i]
        comptime di = _GAMMA_DEN[i]
        num = (num + ni) * z
        den = den * z + di
    var result = num / den + Float32(1.0)
    if x < Float32(1.0):
        result /= y - Float32(1.0)
    else:
        var n = Int(_floor(x))
        for _ in range(1, n):
            result *= y
            y += Float32(1.0)
    return result


@always_inline
def mps_log_gamma(x: Float32) -> Float32:
    """`c10::metal::log_gamma`."""
    comptime LOG_PI = Float32(1.14472988584940017414342735135305)
    if is_nan_f(x):
        return x
    var abs_x = _abs(x)
    if abs_x == Float32(0.0):
        return _INF
    if is_inf_f(x):
        # (inf - 0.5) * log(inf) - inf: NaN in IEEE arithmetic, which the
        # fast-math build would not reproduce.
        return _NAN
    var rc: Float32
    if abs_x < Float32(12.0):
        rc = air_log(_abs(mps_gamma(abs_x)))
    else:
        rc = _log_gamma_asymptotic(abs_x)
    if x >= Float32(0.0):
        return rc
    var log_arg = abs_x * _abs(air_sinpi(abs_x))
    return LOG_PI - rc - air_log(log_arg)


@always_inline
def _mps_zeta(x: Float32, q: Float32) -> Float32:
    """`c10::metal::zeta` (Hurwitz zeta)."""
    comptime MACHEP = Float32(1.11022302462515654042e-16)
    comptime A: Array[Float32, 12] = [
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9,
        7.47242496e10,
        -2.950130727918164224e12,
        1.1646782814350067249e14,
        -4.5979787224074726105e15,
        1.8152105401943546773e17,
        -7.1661652561756670113e18,
    ]
    if x == Float32(1.0):
        return _INF
    if x < Float32(1.0):
        return _NAN
    if q <= Float32(0.0):
        if q == _trunc(q):
            return _INF
        if x != _trunc(x):
            return _NAN
    var s = air_pow(q, -x)
    var a = q
    var i = 0
    var b = Float32(0.0)
    while (i < 9) or (a <= Float32(9.0)):
        i += 1
        a += Float32(1.0)
        b = air_pow(a, -x)
        s += b
        if (-MACHEP * s < b) and (b < MACHEP * s):
            return s
    var w = a
    s += b * w / (x - Float32(1.0))
    s -= Float32(0.5) * b
    a = Float32(1.0)
    var k = Float32(0.0)
    comptime for j in range(12):
        a *= x + k
        b /= w
        comptime aj = A[j]
        var t = a * b / aj
        s += t
        t = _abs(t / s)
        if t < MACHEP:
            return s
        k += Float32(1.0)
        a *= x + k
        b /= w
        k += Float32(1.0)
    return s


@always_inline
def _mps_digamma_positive(x_in: Float32) -> Float32:
    """`c10::metal::calc_digamma_positive_domain`."""
    comptime C: Array[Float32, 7] = [
        8.33333333333333333333e-2,
        -2.10927960927960927961e-2,
        7.57575757575757575758e-3,
        -4.16666666666666666667e-3,
        3.96825396825396825397e-3,
        -8.33333333333333333333e-3,
        8.33333333333333333333e-2,
    ]
    var x = x_in
    var result = Float32(0.0)
    while x < Float32(10.0):
        result -= Float32(1.0) / x
        x += Float32(1.0)
    if x == Float32(10.0):
        return result + Float32(2.25175258906672110764)
    var y = Float32(0.0)
    if x < Float32(1.0e17):
        var z = Float32(1.0) / (x * x)
        comptime for i in range(7):
            comptime ci = C[i]
            y += air_pow(z, Float32(i)) * ci
        y *= z
    return result + air_log(x) - (Float32(0.5) / x) - y


@always_inline
def mps_digamma(x: Float32) -> Float32:
    """`c10::metal::digamma`."""
    if is_nan_f(x):
        return x
    if is_inf_f(x):
        # +inf: the asymptotic branch's log(inf) = inf; -inf is an integer.
        return _INF if x > 0 else _NAN
    if x < Float32(0.0):
        if x == _trunc(x):
            return _NAN
        var r = _fract(x)
        return _mps_digamma_positive(Float32(1.0) - x) - M_PI_F / air_tan(
            M_PI_F * r
        )
    if x == Float32(0.0):
        return _copysign(_INF, -x)
    return _mps_digamma_positive(x)


@always_inline
def mps_trigamma(x_in: Float32) -> Float32:
    """`c10::metal::trigamma`."""
    if is_nan_f(x_in):
        return x_in
    var x = x_in
    var sign = Float32(1.0)
    var result = Float32(0.0)
    if x < Float32(0.0):
        if is_inf_f(x):
            return _NAN
        sign = Float32(-1.0)
        var sin_pi_x = air_sin(M_PI_F * x)
        result -= (M_PI_F * M_PI_F) / (sin_pi_x * sin_pi_x)
        x = Float32(1.0) - x
    elif x == Float32(0.0):
        return _INF
    elif x < Float32(1.0):
        result += Float32(1.0) / (x * x)
        x += Float32(1.0)
    if is_inf_f(x):
        return Float32(0.0)
    for _ in range(6):
        result += Float32(1.0) / (x * x)
        x += Float32(1.0)
    var ixx = Float32(1.0) / (x * x)
    result += (
        Float32(1.0)
        + Float32(1.0) / (Float32(2.0) * x)
        + ixx
        * (
            (Float32(1.0) / Float32(6.0))
            - ixx
            * (
                (Float32(1.0) / Float32(30.0))
                - ixx * (Float32(1.0) / Float32(42.0))
            )
        )
    ) / x
    return sign * result


@always_inline
def mps_polygamma(x: Float32, n: Int) -> Float32:
    """`polygamma_kernel` (UnaryKernel.mm): digamma for n = 0, trigamma for
    n = 1, else `c10::metal::polygamma`: (-1)^(n+1) gamma(n + 1) zeta(n + 1, x).
    """
    if n == 0:
        return mps_digamma(x)
    if n == 1:
        return mps_trigamma(x)
    if is_nan_f(x):
        return x
    var nf = Float32(n)
    var sgn = Float32(1.0) if n % 2 == 1 else Float32(-1.0)
    return sgn * mps_gamma(nf + Float32(1.0)) * _mps_zeta(nf + Float32(1.0), x)


@always_inline
def mps_round_decimals(x: Float32, decimals: Float32) -> Float32:
    """`round_decimals_functor`: rint(exp10(n) x) exp10(-n), in float."""
    return _rint(air_exp10(decimals) * x) * air_exp10(-decimals)


@always_inline
def mps_logit[dtype: DType](x: Float32, eps: Float32) -> Float32:
    """`logit_mps_impl`: an MPSGraph in the tensor type -- clamp to
    [eps, 1 - eps] when eps is given (lo applied last), then log(z / (1 - z)),
    every step rounded to the tensor type. A negative eps is the schema's
    None here, as in the CUDA kernel."""
    var z = x
    if not (eps < Float32(0.0)):
        var lo = _rt[dtype](eps)
        var hi = _rt[dtype](Float32(1.0) - lo)
        z = hi if x > hi else x
        z = lo if x < lo else z
    var q = _rt[dtype](z / _rt[dtype](Float32(1.0) - z))
    return air_log(q)
