"""Floating pow(base, exponent) with C's special cases, for the tensor-tensor
and tensor-scalar pow kernels (logic `BOP_POW`, elementwise `SOP_POW`).

Stock torch runs `::pow` / `::powf` (cuda/Pow.cuh `pow_`), which follow C99
Annex F: pow(x, +-0) = 1 and pow(1, y) = 1 even for NaN, NaN otherwise
propagates, an infinite exponent gives 0, 1 or inf by |x| against 1, signed
zeros and infinities keep the sign of an odd integral exponent, and a negative
finite base with a non-integral exponent is NaN. std.math.pow's GPU lowering
gets several of these wrong: an integral-looking exponent goes through an
int32 repeated-squaring loop (pow(2, -inf) came out 1, and |y| >= 2^31
overflows), and a NaN exponent came out inf.

So the finite, positive-base core is computed on |x| and every special case
is selected explicitly (bit-based `isnan` / `isinf`, which survive the
fast-math flags of the GPU build). The core for float32 and the half types is
exp(y * log|x|) in float64 through the libdevice ports: the product's
rounding costs |y log x| * 2^-53 relative, far below a float32 ulp over the
whole float32 range (AMD). On NVIDIA and Apple the float32 core is instead
the CUDA math library's float powf (`cuda_math.cuda_powf_core`), what stock
torch runs on CUDA, in float32 arithmetic alone (Apple GPUs have no float64;
MPS computes a float pow there too). float64 is `_pow_f64_core`: a
double-double log and a corrected exp, within about an ulp like CUDA's
double pow (std.math.pow's float64 GPU lowering is 2^-12 relative off).
"""

from std.math import floor, fma
from std.memory import bitcast
from std.sys._assembly import inlined_assembly
from std.sys.info import is_amd_gpu, is_apple_gpu, is_nvidia_gpu
from std.utils.numerics import inf, isinf, isnan, nan

from tmb.kernels.common.cuda_math import cuda_powf_core
from tmb.kernels.common.libdevice_port import nv_exp, nv_log


@always_inline
def _add_rn(a: Float64, b: Float64) -> Float64:
    """`a + b`, rounded, and opaque to the optimizer: the GPU build's
    fast-math flags let LLVM reassociate `(a - (s - bb))` to zero, which
    silently turns every double-double sum into a plain double one."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rn.f64 $0, $1, $2;",
            Float64,
            constraints="=d,d,d",
            has_side_effect=False,
        ](a, b)
    elif is_amd_gpu():
        return inlined_assembly[
            "v_add_f64 $0, $1, $2",
            Float64,
            constraints="=v,v,v",
            has_side_effect=False,
        ](a, b)
    else:
        return a + b


@always_inline
def _two_sum(a: Float64, b: Float64) -> Tuple[Float64, Float64]:
    """Knuth's TwoSum: s + e == a + b exactly."""
    var s = _add_rn(a, b)
    var bb = _add_rn(s, -a)
    var e = _add_rn(_add_rn(a, -_add_rn(s, -bb)), _add_rn(b, -bb))
    return (s, e)


@always_inline
def _pow_f64_core(ax: Float64, y: Float64) -> Float64:
    """ax ** y for a finite positive `ax` and a finite nonzero `y`, within
    about an ulp, like CUDA's and ROCm's double `pow`: log(ax) in
    double-double (ax = 2^k m, m in [sqrt(1/2), sqrt(2)), log m = 2 atanh(s),
    s = (m - 1) / (m + 1), its linear and cubic terms in double-double),
    times y with the product's error kept, then exp of the head corrected
    by the tail. The previous core, std.math.pow, was exp(y log x) in the
    GPU's approximate float64 exp/log (2^-12 relative error), or repeated
    squaring for an integral y."""
    comptime SQRT2 = 1.4142135623730951
    comptime LN2_HI = 6.93147180369123816490e-01  # 21 trailing zero bits
    comptime LN2_LO = 1.90821492927058770002e-10
    var x = ax
    var k = 0
    if x < 2.2250738585072014e-308:
        x = x * 18014398509481984.0  # 2**54
        k = -54
    var bits = bitcast[DType.uint64](x)
    k += Int((bits >> UInt64(52)) & UInt64(0x7FF)) - 1023
    var m = bitcast[DType.float64](
        (bits & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FF0000000000000)
    )
    if m > SQRT2:
        m = m * 0.5
        k += 1
    var f = m - 1.0  # exact (Sterbenz)
    var d = _two_sum(m, 1.0)
    var s_hi = f / d[0]
    var s_lo = (fma(-s_hi, d[0], f) - s_hi * d[1]) / d[0]
    # s^2, s^3 in double-double; the tail from s^5 on in double.
    var p = s_hi * s_hi
    var pe = fma(s_hi, s_hi, -p) + 2.0 * s_hi * s_lo
    var q = p * s_hi
    var qe = fma(p, s_hi, -q) + pe * s_hi + p * s_lo
    comptime TH = 0.66666666666666663  # 2/3 rounded
    comptime TL = 3.700743415417188e-17  # 2/3 - TH
    var c = TH * q
    var ce = fma(TH, q, -c) + TL * q + TH * qe
    var t2 = p
    var tail = 2.0 / 23.0
    tail = fma(tail, t2, 2.0 / 21.0)
    tail = fma(tail, t2, 2.0 / 19.0)
    tail = fma(tail, t2, 2.0 / 17.0)
    tail = fma(tail, t2, 2.0 / 15.0)
    tail = fma(tail, t2, 2.0 / 13.0)
    tail = fma(tail, t2, 2.0 / 11.0)
    tail = fma(tail, t2, 2.0 / 9.0)
    tail = fma(tail, t2, 2.0 / 7.0)
    tail = fma(tail, t2, 2.0 / 5.0)
    tail = tail * (t2 * q)
    # log m = 2 s + c + tail
    var h = _two_sum(2.0 * s_hi, c)
    var lm_lo = h[1] + (2.0 * s_lo + ce + tail)
    var lm = _two_sum(h[0], lm_lo)
    # + k ln 2 (k * LN2_HI is exact)
    var kd = Float64(k)
    var g = _two_sum(kd * LN2_HI, lm[0])
    var l_lo = g[1] + (lm[1] + kd * LN2_LO)
    var l = _two_sum(g[0], l_lo)
    # t = y * log(ax) in double-double.
    var t_hi = y * l[0]
    # Saturate on the head, before the correction: once y * log(ax)
    # overflows (pow(10, 1e308)), fma(y, l, -t_hi) is inf - inf = NaN. The
    # head is within about an ulp of the double-double sum, far inside the
    # thresholds' margin (exp overflows past 709.783 and is 0 below -745.134).
    if t_hi > 709.8:
        return inf[DType.float64]()
    if t_hi < -745.2:
        return 0.0
    var t_lo = fma(y, l[0], -t_hi) + y * l[1]
    var th = _add_rn(t_hi, t_lo)
    var tl = _add_rn(t_lo, -_add_rn(th, -t_hi))
    var r = nv_exp(th)
    # exp overflows from 709.7827 on, below the 709.8 cutoff above: then
    # fma(inf, tl, inf) with a negative tail would be inf - inf = NaN
    # (pow(2, 1024.003)). The underflow side needs no guard: fma(0, tl, 0) = 0.
    if r == inf[DType.float64]():
        return r
    return fma(r, tl, r)


@always_inline
def _core[w: DType](ax: Scalar[w], y: Scalar[w]) -> Scalar[w]:
    """|x| ** y for a finite positive |x| and a finite y."""
    comptime if w == DType.float64:
        return _pow_f64_core(
            ax.cast[DType.float64](), y.cast[DType.float64]()
        ).cast[w]()
    elif is_nvidia_gpu() or is_apple_gpu():
        # float32 (and the half types, widened by the caller): the CUDA
        # powf that Pow.cuh's `::pow` runs. Apple GPUs too: torch MPS runs
        # Metal's float `precise::pow` (UnaryKernel.metal `pow_scalar`,
        # MPSGraph `power`), and float64 does not exist there.
        return cuda_powf_core(
            ax.cast[DType.float32](), y.cast[DType.float32]()
        ).cast[w]()
    else:
        var yd = y.cast[DType.float64]()
        var ld = nv_log(ax.cast[DType.float64]())
        return nv_exp(yd * ld).cast[w]()


@always_inline
def _pow_scalar[w: DType](x: Scalar[w], y: Scalar[w]) -> Scalar[w]:
    comptime if w == DType.float32 and is_nvidia_gpu():
        # The common case first, on the bits: x positive, finite and not 1,
        # y finite and nonzero need none of the selects below (three integer
        # compares instead of every special-case test per element).
        var xb = x.to_bits[DType.uint32]()
        var yb = y.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)
        if (
            xb - UInt32(1) < UInt32(0x7F7FFFFF)
            and xb != UInt32(0x3F800000)
            and yb - UInt32(1) < UInt32(0x7F7FFFFF)
        ):
            return cuda_powf_core(
                x.cast[DType.float32](), y.cast[DType.float32]()
            ).cast[w]()
    if y == 0:
        return 1
    if x == 1:
        return 1
    if isnan(x) or isnan(y):
        return nan[w]()
    var ax = abs(x)
    if isinf(y):
        if ax == 1:
            return 1
        var big = ax > 1
        if (y > 0) == big:
            return inf[w]()
        return 0
    var integral = y == floor(y)
    var half = y * 0.5
    # Every |y| >= 2^53 is an even integer (and so is every float32 one
    # past 2^24, which the same test sees through `half`).
    var odd = integral and half != floor(half) and abs(y) < 9007199254740992.0
    # The sign bit: -0 and -inf included (NaN is already out).
    comptime ibits = DType.int64 if w == DType.float64 else DType.int32
    var negative = bitcast[ibits](x) < 0
    var mag: Scalar[w]
    if ax == 0:
        mag = inf[w]() if y < 0 else Scalar[w](0)
    elif isinf(x):
        mag = Scalar[w](0) if y < 0 else inf[w]()
    else:
        if negative and not integral:
            return nan[w]()
        mag = _core(ax, y)
    return -mag if negative and odd else mag


@always_inline
def torch_pow[
    dtype: DType, n: Int
](a: SIMD[dtype, n], b: SIMD[dtype, n]) -> SIMD[
    dtype, n
] where dtype.is_floating_point():
    """pow(a, b) per lane: float16 / bfloat16 widened to float32 (Pow.cuh's
    `pow_` computes the half types in float) and rounded once."""
    comptime w = DType.float64 if dtype == DType.float64 else DType.float32
    var r = SIMD[dtype, n]()
    comptime for i in range(n):
        r[i] = _pow_scalar[w](a[i].cast[w](), b[i].cast[w]()).cast[dtype]()
    return r


@always_inline
def powf_c99(a: Float32, b: Float32) -> Float32:
    """Scalar float `pow(a, b)`: the same special cases and core as
    `torch_pow` (on NVIDIA the CUDA math library's powf, bit for bit)."""
    return _pow_scalar[DType.float32](a, b)
