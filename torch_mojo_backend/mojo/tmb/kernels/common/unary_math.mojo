"""Unary SIMD expressions shared by eager and torch.compile execution."""

from std.math import (
    acos,
    asinh,
    atanh,
    ceil,
    cos,
    cosh,
    copysign,
    erf,
    exp,
    floor,
    log,
    log1p,
    log2,
    sin,
    sinh,
    sqrt,
    tanh,
)
from std.collections import Array
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys.info import (
    bit_width_of,
    is_amd_gpu,
    is_apple_gpu,
    is_gpu,
    is_nvidia_gpu,
)
from std.utils.numerics import isnan, max_or_inf, nan
from tmb.kernels.common.cuda_math import (
    nv_asinf,
    nv_atanf,
    nv_erf,
    nv_erfcf,
    nv_erff,
    nv_erfinvf,
    nv_exp2f,
    nv_expm1f,
    nv_lgammaf,
    nv_log10f,
)
from tmb.kernels.common.libdevice_port import (
    nv_asinh,
    nv_asinhf,
    nv_erfc,
    nv_erfinv,
    nv_lgamma,
    nv_log2f,
    nv_logf,
)
from tmb.kernels.common.math_utils import custom_tan, ieee_sqrt
from tmb.kernels.common.ocml_math import (
    oc_asinf,
    oc_asinh,
    oc_asinhf,
    oc_atanf,
    oc_erfinvf,
    oc_exp2f,
    oc_expm1f,
    oc_log10f,
    oc_log2f,
    oc_logf,
)
from tmb.kernels.common.mps_math import (
    air_asin,
    air_asinh,
    air_atan,
    air_log10,
    mps_digamma,
    mps_erfc,
    mps_erfinv,
    mps_exp2,
    mps_expm1,
    mps_i0,
    mps_log_gamma,
    mps_logit,
    mps_polygamma,
    mps_polygamma_series,
    mps_round_decimals,
    mps_sinc,
    rint_even,
)
from tmb.kernels.common.special_math import (
    airy_ai,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    digamma,
    entr,
    erfcx,
    i0,
    i0e,
    i1,
    i1e,
    log_ndtr,
    logit,
    modified_bessel_i0,
    modified_bessel_i1,
    modified_bessel_k0,
    modified_bessel_k1,
    ndtri,
    polygamma,
    polygamma_series,
    sinc,
    spherical_bessel_j0,
    trigamma,
)


@always_inline
def _log1p_nonneg[
    dtype: DType, width: SIMDLength
](t: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """log1p for finite t >= 0 (special values are the caller's to select).

    float32 is CUDA's log1pf (libdevice): 1 + t = 2^i (1 + m) by exponent
    arithmetic on the bits, m in about [-1/4, 1/2], then a degree-9
    polynomial -- no division and no hardware lg2 approximation, whose
    absolute error near one the compensated log1p kind has to repair.
    """
    comptime if dtype == DType.float32:
        var x = rebind[SIMD[DType.float32, width]](t)
        var e = (bitcast[DType.int32](x + 1) - 0x3F400000) & -8388608
        var m = bitcast[DType.float32](bitcast[DType.int32](x) - e)
        var s = bitcast[DType.float32](0x40800000 - e)
        m += SIMD[DType.float32, width](0.25).fma(s, -1)
        var i = e.cast[DType.float32]() * 1.1920928955078125e-7
        var p = SIMD[DType.float32, width](-0.04534861445426941)
        p = p.fma(m, 0.10546888411045074)
        p = p.fma(m, -0.13229703903198242)
        p = p.fma(m, 0.14491446316242218)
        p = p.fma(m, -0.16641564667224884)
        p = p.fma(m, 0.199888676404953)
        p = p.fma(m, -0.2500019669532776)
        p = p.fma(m, 0.33333510160446167)
        p = p.fma(m, -0.5)
        var r = (p * m).fma(m, m)
        return rebind[SIMD[dtype, width]](i.fma(0.6931471824645996, r))
    else:
        return log1p(t)


@always_inline
def _acosh[
    dtype: DType, width: SIMDLength, //, exact_sqrt: Bool
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """acosh as CUDA's acoshf computes it (libdevice __nv_acoshf, what stock
    torch runs): log1p(d + sqrt(d (x + 1))) with d = x - 1, exact near one.

    Past d = 2^23 acosh(x) = log(2x) = log1p(d) + ln 2 instead, where
    d (x + 1) would overflow float32. `exact_sqrt=False` takes the hardware
    approximation (~2^-22 relative on NVIDIA), for float16/bfloat16 inputs
    whose result is rounded to 11 or 8 bits anyway: the correctly rounded
    root cost ~10 instructions per element. std.math.acosh is libm/CPU-only.
    """
    var d = a - 1
    var large = ~(d.ge(0) & d.le(8388608.0))
    var prod = a.fma(d, d)
    comptime if exact_sqrt:
        prod = ieee_sqrt(prod)
    else:
        prod = sqrt(prod)
    var res = _log1p_nonneg(large.select(d, d + prod)) + large.select(
        SIMD[dtype, width](0.69314718055994530942), 0
    )
    # One mask for every special value: NaN below one and for NaN, +inf
    # at +inf; only finite x >= 1 keeps the computed value.
    var ge1 = a.ge(1)
    return (ge1 & a.lt(max_or_inf[dtype]())).select(
        res, ge1.select(a, SIMD[dtype, width](nan[dtype]()))
    )


@always_inline
def is_scalar_special[kind: StaticString]() -> Bool:
    """The kinds computed one lane at a time by a scalar port of the CUDA
    routine stock torch runs (`cuda_math.mojo`, `special_math.mojo`):
    branchy Cephes/libdevice code, evaluated in float for float16/bfloat16
    inputs as the jiterator does, and in double for float64 where
    `has_f64_port` (in float otherwise).
    """
    return (
        kind == "asin"
        or kind == "atan"
        or kind == "exp2"
        or kind == "expm1"
        or kind == "log10"
        or has_f64_port[kind]()
    )


@always_inline
def has_f64_port[kind: StaticString]() -> Bool:
    """The scalar-special kinds with a double port (the special functions:
    their jiterator strings / kernels instantiated for T = double)."""
    return (
        kind == "airy_ai"
        or kind == "bessel_j0"
        or kind == "bessel_j1"
        or kind == "bessel_y0"
        or kind == "bessel_y1"
        or kind == "digamma"
        or kind == "entr"
        or kind == "erfc"
        or kind == "erfcx"
        or kind == "erfinv"
        or kind == "i0"
        or kind == "i0e"
        or kind == "i1"
        or kind == "i1e"
        or kind == "lgamma"
        or kind == "log_ndtr"
        or kind == "logit"
        or kind == "modified_bessel_i0"
        or kind == "modified_bessel_i1"
        or kind == "modified_bessel_k0"
        or kind == "modified_bessel_k1"
        or kind == "ndtri"
        or kind == "scaled_modified_bessel_k0"
        or kind == "scaled_modified_bessel_k1"
        or kind == "sinc"
        or kind == "spherical_bessel_j0"
        or kind == "trigamma"
    )


@always_inline
def _scalar_special[kind: StaticString, ct: DType](a: Scalar[ct]) -> Scalar[ct]:
    """`kind` in the compute type `ct`: float32, or float64 for the
    `has_f64_port` kinds."""
    comptime if kind == "airy_ai":
        return airy_ai(a)
    elif kind == "bessel_j0":
        return bessel_j0(a)
    elif kind == "bessel_j1":
        return bessel_j1(a)
    elif kind == "bessel_y0":
        return bessel_y0(a)
    elif kind == "bessel_y1":
        return bessel_y1(a)
    elif kind == "digamma":
        return digamma(a)
    elif kind == "entr":
        return entr(a)
    elif kind == "erfcx":
        return erfcx(a)
    elif kind == "i0":
        return i0(a)
    elif kind == "i0e":
        return i0e(a)
    elif kind == "i1":
        return i1(a)
    elif kind == "i1e":
        return i1e(a)
    elif kind == "log_ndtr":
        return log_ndtr(a)
    elif kind == "logit":
        return logit(a, Scalar[ct](-1.0))
    elif kind == "modified_bessel_i0":
        return modified_bessel_i0(a)
    elif kind == "modified_bessel_i1":
        return modified_bessel_i1(a)
    elif kind == "modified_bessel_k0":
        return modified_bessel_k0[ct, False](a)
    elif kind == "modified_bessel_k1":
        return modified_bessel_k1[ct, False, fb=ct == DType.float32](a)
    elif kind == "ndtri":
        return ndtri(a)
    elif kind == "scaled_modified_bessel_k0":
        return modified_bessel_k0[ct, True, fb=True](a)
    elif kind == "scaled_modified_bessel_k1":
        return modified_bessel_k1[ct, True](a)
    elif kind == "sinc":
        return sinc(a)
    elif kind == "spherical_bessel_j0":
        return spherical_bessel_j0(a)
    elif kind == "trigamma":
        return trigamma(a)
    elif ct == DType.float64:
        var d = rebind[Float64](a)
        comptime if kind == "erfc":
            return rebind[Scalar[ct]](nv_erfc(d))
        elif kind == "erfinv":
            return rebind[Scalar[ct]](nv_erfinv(d))
        else:
            comptime assert kind == "lgamma", "no float64 port of this kind"
            return rebind[Scalar[ct]](nv_lgamma(d))
    else:
        return rebind[Scalar[ct]](_scalar_special_f32[kind](rebind[Float32](a)))


@always_inline
def _scalar_special_f32[kind: StaticString](a: Float32) -> Float32:
    """The libm kinds in float (CUDA's, or ocml's on AMD GPUs)."""
    comptime if kind == "asin":
        comptime if is_amd_gpu():
            return oc_asinf(a)
        return nv_asinf(a)
    elif kind == "atan":
        comptime if is_amd_gpu():
            return oc_atanf(a)
        return nv_atanf(a)
    elif kind == "erfc":
        return nv_erfcf(a)
    elif kind == "erfinv":
        comptime if is_amd_gpu():
            return oc_erfinvf(a)
        return nv_erfinvf(a)
    elif kind == "exp2":
        comptime if is_amd_gpu():
            return oc_exp2f(a)
        return nv_exp2f(a)
    elif kind == "expm1":
        comptime if is_amd_gpu():
            return oc_expm1f(a)
        return nv_expm1f(a)
    elif kind == "lgamma":
        return nv_lgammaf(a)
    else:
        comptime assert kind == "log10", "unsupported scalar special kind"
        comptime if is_amd_gpu():
            return oc_log10f(a)
        return nv_log10f(a)


@always_inline
def _is_mps_special[kind: StaticString]() -> Bool:
    """The scalar-special kinds whose Apple GPU kernel is the MPS routine
    (`mps_math.mojo`) instead of the CUDA port."""
    return (
        kind == "asin"
        or kind == "atan"
        or kind == "digamma"
        or kind == "erfc"
        or kind == "erfinv"
        or kind == "exp2"
        or kind == "expm1"
        or kind == "i0"
        or kind == "lgamma"
        or kind == "log10"
        or kind == "sinc"
    )


@always_inline
def _mps_special[kind: StaticString, dtype: DType](a: Float32) -> Float32:
    """UnaryKernel.metal's functor of `kind`, in float (the precise Metal
    library or the c10::metal port)."""
    comptime if kind == "asin":
        return air_asin(a)
    elif kind == "atan":
        return air_atan(a)
    elif kind == "digamma":
        return mps_digamma(a)
    elif kind == "erfc":
        return mps_erfc(a)
    elif kind == "erfinv":
        return mps_erfinv[dtype](a)
    elif kind == "exp2":
        return mps_exp2(a)
    elif kind == "expm1":
        return mps_expm1(a)
    elif kind == "i0":
        return mps_i0[dtype](a)
    elif kind == "lgamma":
        return mps_log_gamma(a)
    elif kind == "log10":
        return air_log10(a)
    else:
        comptime assert kind == "sinc", "unsupported MPS special kind"
        return mps_sinc(a)


@always_inline
def _rounding[
    kind: StaticString, dtype: DType, width: SIMDLength
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """trunc / round (half to even: ATen's nearbyint) / frac / angle of a
    real input: exact in any float dtype, so float64 computes natively."""
    comptime if kind == "trunc":
        return llvm_intrinsic["llvm.trunc", type_of(a), has_side_effect=False](
            a
        )
    elif kind == "round":
        comptime if is_apple_gpu():
            # `llvm.roundeven` crashes Apple's Metal shader compiler.
            return rint_even(a)
        else:
            return llvm_intrinsic[
                "llvm.roundeven", type_of(a), has_side_effect=False
            ](a)
    elif kind == "frac":
        # `a - trunc(a)`: exact, and NaN / +-inf -> NaN as in ATen.
        return a - llvm_intrinsic[
            "llvm.trunc", type_of(a), has_side_effect=False
        ](a)
    else:
        comptime assert kind == "angle", "unsupported rounding kind"
        # UnaryComplexKernels.cu `angle_wrapper`: NaN stays, pi below zero.
        return isnan(a).select(
            a,
            a.lt(0).select(
                SIMD[dtype, width](3.14159265358979323846),
                SIMD[dtype, width](0),
            ),
        )


@always_inline
def is_rounding[kind: StaticString]() -> Bool:
    return (
        kind == "trunc" or kind == "round" or kind == "frac" or kind == "angle"
    )


@always_inline
def _gpu_log_f32[
    kind: StaticString, width: SIMDLength
](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """float32 log/log2 with the libm stock torch calls: libdevice's
    `__nv_logf` / `__nv_log2f` on NVIDIA, ocml's on AMD. std.math's GPU
    versions are approximations that mishandle subnormal inputs.
    """
    var r = SIMD[DType.float32, width]()
    comptime for i in range(width):
        comptime if is_amd_gpu():
            comptime if kind == "log":
                r[i] = oc_logf(a[i])
            else:
                r[i] = oc_log2f(a[i])
        else:
            comptime if kind == "log":
                r[i] = nv_logf(a[i])
            else:
                r[i] = nv_log2f(a[i])
    return r


@always_inline
def _asinh[
    dtype: DType, width: SIMDLength
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """asinh with the libm stock torch calls: libdevice's `__nv_asinhf` /
    `__nv_asinh` on NVIDIA, ocml's on AMD, Metal's on Apple, libm on the
    host. (`log(x + sqrt(x^2 + 1))` cancels for negative x.)
    """
    comptime if not is_gpu():
        return asinh(a)
    var r = SIMD[dtype, width]()
    comptime for i in range(width):
        comptime if dtype == DType.float32:
            var x = rebind[Float32](a[i])
            comptime if is_nvidia_gpu():
                r[i] = rebind[Scalar[dtype]](nv_asinhf(x))
            elif is_amd_gpu():
                r[i] = rebind[Scalar[dtype]](oc_asinhf(x))
            else:
                r[i] = rebind[Scalar[dtype]](air_asinh(x))
        else:
            comptime assert dtype == DType.float64, "float32 or float64 only"
            var x = rebind[Float64](a[i])
            comptime if is_nvidia_gpu():
                r[i] = rebind[Scalar[dtype]](nv_asinh(x))
            elif is_amd_gpu():
                r[i] = rebind[Scalar[dtype]](oc_asinh(x))
            else:  # Apple GPUs have no float64
                r[i] = air_asinh(x.cast[DType.float32]()).cast[dtype]()
    return r


@always_inline
def _float_unary[
    kind: StaticString, dtype: DType, width: SIMDLength
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """The float-only unary math, evaluated in `dtype` (float32 or float64).

    Only instantiated for float32/float64 (half inputs are promoted before
    the call), so every std.math call below sees a supported dtype.
    """
    comptime assert (
        kind == "exp"
        or kind == "tanh"
        or kind == "ceil"
        or kind == "floor"
        or kind == "acos"
        or kind == "acosh"
        or kind == "asinh"
        or kind == "atanh"
        or kind == "cos"
        or kind == "cosh"
        or kind == "erf"
        or kind == "log"
        or kind == "log1p"
        or kind == "log2"
        or kind == "reciprocal"
        or kind == "rsqrt"
        or kind == "sigmoid"
        or kind == "silu"
        or kind == "sin"
        or kind == "sinh"
        or kind == "sqrt"
        or kind == "tan"
        or kind == "gelu_none"
        or kind == "gelu_tanh"
    ), "unsupported elementwise kind"
    var res = a
    comptime if kind == "exp":
        res = exp(a)
    comptime if kind == "tanh":
        res = tanh(a)
    comptime if kind == "ceil":
        res = ceil(a)
    comptime if kind == "floor":
        res = floor(a)
    comptime if kind == "acos":
        res = acos(a)
        # std.math.acos clamps outside [-1, 1]; ATen requires NaN.
        comptime if is_nvidia_gpu() and dtype == DType.float32:
            # Integer masking keeps NVIDIA's packed float32-to-half conversion;
            # a floating select lets LLVM split it into scalar conversions.
            var magnitude = bitcast[DType.uint32](a) & 0x7FFFFFFF
            var invalid = magnitude.gt(0x3F800000)
            var nan_bits = invalid.cast[DType.uint32]() * 0x7FC00000
            res = bitcast[dtype](bitcast[DType.uint32](res) | nan_bits)
        else:
            res = (abs(a).gt(1) | isnan(a)).select(
                SIMD[dtype, width](nan[dtype]()), res
            )
    comptime if kind == "acosh":
        res = _acosh[exact_sqrt=True](a)
    comptime if kind == "asinh":
        res = _asinh(a)
    comptime if kind == "atanh":
        res = atanh(a)
    comptime if kind == "cos":
        res = cos(a)
    comptime if kind == "cosh":
        res = cosh(a)
    comptime if kind == "erf":
        comptime if is_nvidia_gpu():
            # erf_kernel_cuda's `::erf`: CUDA's erff / erf.
            comptime for i in range(width):
                comptime if dtype == DType.float32:
                    res[i] = rebind[Scalar[dtype]](
                        nv_erff(rebind[Float32](a[i]))
                    )
                else:
                    res[i] = rebind[Scalar[dtype]](
                        nv_erf(rebind[Float64](a[i]))
                    )
        else:
            res = erf(a)
    comptime if (
        kind == "log" or kind == "log2"
    ) and dtype == DType.float32 and (is_nvidia_gpu() or is_amd_gpu()):
        res = _gpu_log_f32[kind](a.cast[DType.float32]()).cast[dtype]()
    elif kind == "log":
        # Preserve +inf, which the stdlib CPU approximation treats as finite.
        res = a.eq(max_or_inf[dtype]()).select(a, log(a))
    elif kind == "log2":
        res = log2(a)
        comptime if dtype == DType.float64 or is_apple_gpu():
            # The double approximation and Metal's float log2 omit +inf.
            res = a.eq(max_or_inf[dtype]()).select(a, res)
    comptime if kind == "log1p":
        comptime if is_apple_gpu() or (
            is_nvidia_gpu() and dtype == DType.float32
        ):
            # std.log1p promotes to float64: unsupported on Metal and costly
            # on NVIDIA. Reuse the compensated float32 Metal algorithm;
            # the NVIDIA near-zero polynomial below handles lg2.approx error.
            var xp1 = 1 + a
            var rc = log(xp1)
            var corrected = rc * (a / (xp1 - 1))
            rc = (a.gt(-0.5) & a.lt(0.5)).select(corrected, rc)
            res = xp1.eq(1).select(a, rc)
            comptime if is_nvidia_gpu() and dtype == DType.float32:
                # Avoid lg2.approx's absolute error near one. The degree-eight
                # Taylor remainder is below 7e-9 relative for |x| < 1/8.
                var p = SIMD[dtype, width](-1 / 8)
                p = p.fma(a, 1 / 7)
                p = p.fma(a, -1 / 6)
                p = p.fma(a, 1 / 5)
                p = p.fma(a, -1 / 4)
                p = p.fma(a, 1 / 3)
                p = p.fma(a, -1 / 2)
                var small = a.fma(a * p, a)
                res = abs(a).lt(0.125).select(small, res)
        else:
            res = log1p(a)
        res = a.eq(max_or_inf[dtype]()).select(a, res)
    comptime if kind == "reciprocal":
        res = 1 / a
    comptime if kind == "rsqrt":
        res = 1 / ieee_sqrt(a)
    comptime if kind == "sigmoid":
        res = 1 / (1 + exp(-a))
    comptime if kind == "silu":
        res = a / (1 + exp(-a))
    comptime if kind == "sin":
        res = sin(a)
    comptime if kind == "sinh":
        res = sinh(a)
        # std.sinh's expm1 approximation saturates large CPU inputs and
        # computes inf/inf at infinity. exp(|x|/2)^2/2 keeps the valid
        # finite range before overflow and restores the signed infinities.
        var half_exp = exp(abs(a) * 0.5)
        var large = copysign((0.5 * half_exp) * half_exp, a)
        res = abs(a).gt(20).select(large, res)
    comptime if kind == "sqrt":
        res = ieee_sqrt(a)
    comptime if kind == "tan":
        # `custom_tan` picks libm, the argument-reduced polynomial or
        # `sin / cos` from the compilation target and `dtype` on its own.
        res = custom_tan(a)
    comptime if kind == "gelu_none":
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        comptime inv_sqrt2 = 0.70710678118654752440
        res = 0.5 * a * (1 + erf(a * inv_sqrt2))
    comptime if kind == "gelu_tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        comptime sqrt_2_over_pi = 0.79788456080286535588
        var inner = sqrt_2_over_pi * (a + 0.044715 * a * a * a)
        res = 0.5 * a * (1 + tanh(inner))
    # CPU stdlib approximations omit NaNs for exp/tanh/cosh. NVIDIA's float32
    # math preserves them already, so avoid an extra select on every lane.
    # Retain the existing correction on other backends and float64, except
    # for the vendor asinh ports, which quiet a signaling NaN as CUDA does.
    comptime if (
        not (is_nvidia_gpu() and dtype == DType.float32)
        and not (kind == "asinh" and (is_nvidia_gpu() or is_amd_gpu()))
        and not (kind == "erf" and is_nvidia_gpu())
    ):
        return isnan(a).select(a, res)
    else:
        return res


@always_inline
def elementwise_unary[
    kind: StaticString, dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if kind == "relu":
        # Return the input for NaN and signed zero, as ATen relu does.
        return x.lt(0).select(SIMD[dtype, width](0), x)
    elif kind == "abs":
        return abs(x)
    elif kind == "neg":
        return -x
    elif kind == "sign":
        comptime if dtype == DType.bool:
            return x
        else:
            # Both comparisons are false for NaN, matching ATen's zero result.
            return x.gt(0).cast[dtype]() - x.lt(0).cast[dtype]()
    elif (
        kind == "ceil" or kind == "floor" or kind == "trunc" or kind == "round"
    ) and dtype.is_integral():
        return x
    elif is_rounding[kind]():
        comptime assert (
            dtype.is_floating_point()
        ), "floating point input required"
        comptime if dtype == DType.float16 or dtype == DType.bfloat16:
            return _rounding[kind](x.cast[DType.float32]()).cast[dtype]()
        else:
            return _rounding[kind](x)
    elif is_apple_gpu() and _is_mps_special[kind]():
        var xf = x.cast[DType.float32]()
        var r = SIMD[DType.float32, width]()
        comptime for i in range(width):
            r[i] = _mps_special[kind, dtype](xf[i])
        return r.cast[dtype]()
    elif is_scalar_special[kind]():
        comptime if dtype == DType.float64 and has_f64_port[kind]():
            var r = SIMD[dtype, width]()
            comptime for i in range(width):
                r[i] = _scalar_special[kind, dtype](x[i])
            return r
        var xf = x.cast[DType.float32]()
        var r = SIMD[DType.float32, width]()
        comptime for i in range(width):
            r[i] = _scalar_special[kind, DType.float32](xf[i])
        return r.cast[dtype]()
    elif kind == "acosh" and (
        dtype == DType.float16 or dtype == DType.bfloat16
    ):
        return _acosh[exact_sqrt=False](x.cast[DType.float32]()).cast[dtype]()
    elif dtype == DType.float16 or dtype == DType.bfloat16:
        # Compute once in float32 and round only the final result.
        return _float_unary[kind](x.cast[DType.float32]()).cast[dtype]()
    else:
        comptime assert (
            dtype.is_floating_point()
        ), "floating point input required"
        return _float_unary[kind](x)


@always_inline
def elementwise_predicate[
    kind: StaticString, dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    comptime if kind == "isnan":
        # Bit-based fpclass survives fast-math and is false for integers.
        return isnan(x)
    elif kind == "logical_not":
        return x.eq(SIMD[dtype, width](0))
    elif (
        kind == "isinf"
        or kind == "isfinite"
        or kind == "isposinf"
        or kind == "isneginf"
    ):
        # On the bit pattern: the GPU build's fast-math flags let LLVM assume
        # no infinity, which may fold `x == inf` to false. An integer (or
        # bool) is always finite and never infinite, as in ATen.
        comptime if dtype.is_floating_point():
            comptime bits = DType.uint64 if dtype == DType.float64 else (
                DType.uint32 if dtype == DType.float32 else DType.uint16
            )
            comptime top = bit_width_of[dtype]() - 1
            comptime mant = (
                52 if dtype
                == DType.float64 else (
                    23 if dtype
                    == DType.float32 else (10 if dtype == DType.float16 else 7)
                )
            )
            comptime sign = Scalar[bits](1) << Scalar[bits](top)
            comptime inf_bits = (sign - 1) ^ (
                (Scalar[bits](1) << Scalar[bits](mant)) - 1
            )
            var u = bitcast[bits](x)
            comptime if kind == "isposinf":
                return u.eq(SIMD[bits, width](inf_bits))
            elif kind == "isneginf":
                return u.eq(SIMD[bits, width](inf_bits | sign))
            elif kind == "isinf":
                return (u & SIMD[bits, width](sign - 1)).eq(inf_bits)
            else:
                return (u & SIMD[bits, width](sign - 1)).lt(inf_bits)
        else:
            return SIMD[DType.bool, width](fill=kind == "isfinite")
    elif kind == "signbit":
        # signbit_kernel_cuda: the sign bit of a float (-0.0 and -NaN too),
        # `x < 0` for an integer (false for every unsigned value).
        comptime if dtype.is_floating_point():
            comptime bits = DType.uint64 if dtype == DType.float64 else (
                DType.uint32 if dtype == DType.float32 else DType.uint16
            )
            comptime top = bit_width_of[dtype]() - 1
            return (bitcast[bits](x) >> SIMD[bits, width](top)).ne(0)
        else:
            return x.lt(0)
    else:
        comptime assert False, "unsupported elementwise predicate"


@always_inline
def _to_dtype_rounded[
    dtype: DType, width: SIMDLength
](v: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Round a float32 intermediate to `dtype` and back: a step the C++
    kernel performs in scalar_t (c10::Half / c10::BFloat16 arithmetic)."""
    return v.cast[dtype]().cast[DType.float32]()


@always_inline
def param_compute_dtype[dtype: DType]() -> DType:
    """The type the parameterized route's scalars cross the launch in: float64
    for a float64 tensor, float32 (opmath) otherwise. The host converts them
    before launch, as torch does (`eps_scalar.to<T_ACC>()`, `static_cast<
    scalar_t>(pow(10, decimals))`, `NanToNumParams<float>`), so no float32 /
    half / bfloat16 kernel takes a double argument -- which Apple GPUs could
    not even build."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@always_inline
def elementwise_unary_param[
    kind: StaticString, dtype: DType, width: SIMDLength
](
    x: SIMD[dtype, width],
    p0: Scalar[param_compute_dtype[dtype]()],
    p1: Scalar[param_compute_dtype[dtype]()],
    p2: Scalar[param_compute_dtype[dtype]()],
) -> SIMD[dtype, width] where dtype.is_floating_point():
    """The unary ops with runtime scalar arguments (the elementwise family's
    parameterized route; `p0..p2` already in `param_compute_dtype`):

    * round_decimals: p0 = 10^|decimals| (rounded to float on the host, then
      to the tensor type here), p1 = 1 when decimals < 0, p2 = decimals.
      round_decimals_kernel_cuda computes in scalar_t: each step rounds. On
      Apple GPUs `round_decimals_functor`: rint(exp10(n) x) exp10(-n).
    * logit: p0 = eps, negative for None (logit_kernel_cuda, in float; on
      Apple the MPSGraph of logit_mps_impl, whose clamp bounds lo = eps and
      hi = 1 - lo, rounded to the tensor type, the host passes in p1 / p2).
    * polygamma: p0 = n (polygamma_kernel_cuda / UnaryKernel.mm, in float).
    * mvlgamma: p0 = p, p1 = p (p - 1) log(pi) / 4 (computed in double on the
      host, rounded to float). ATen composes it (UnaryOps.cpp `mvlgamma`):
      sum_j lgamma(x + ((1 - p)/2 + j/2)) + p1, the offsets (an arange), the
      arguments, the terms and the sum in the tensor dtype, the sum accumulated in float in the order of CUDA's
      reduce kernel.
    * nan_to_num: p0 / p1 / p2 replace NaN / +inf / -inf, each cast to the
      tensor dtype (nan_to_num_kernel_cuda; the MPS kernel is the same).
    """
    comptime ct = param_compute_dtype[dtype]()
    comptime if kind == "nan_to_num":
        var r_nan = SIMD[dtype, width](p0.cast[dtype]())
        var r_pos = SIMD[dtype, width](p1.cast[dtype]())
        var r_neg = SIMD[dtype, width](p2.cast[dtype]())
        return isnan(x).select(
            r_nan,
            x.eq(max_or_inf[dtype]()).select(
                r_pos, x.eq(-max_or_inf[dtype]()).select(r_neg, x)
            ),
        )
    elif kind == "round_decimals":
        comptime if dtype == DType.float64:
            var tp = SIMD[dtype, width](rebind[Float64](p0))
            if p1 != 0:
                return _rounding["round"](x / tp) * tp
            return _rounding["round"](x * tp) / tp
        elif is_apple_gpu():
            var xf = x.cast[DType.float32]()
            var r = SIMD[DType.float32, width]()
            var d = rebind[Float32](p2)
            comptime for i in range(width):
                r[i] = mps_round_decimals(xf[i], d)
            return r.cast[dtype]()
        else:
            var tp = SIMD[DType.float32, width](
                rebind[Float32](p0).cast[dtype]().cast[DType.float32]()
            )
            var xf = x.cast[DType.float32]()
            if p1 != 0:
                var q = _to_dtype_rounded[dtype](xf / tp)
                return (_rounding["round"](q) * tp).cast[dtype]()
            var q = _to_dtype_rounded[dtype](xf * tp)
            return (_rounding["round"](q) / tp).cast[dtype]()
    elif ct == DType.float64 and kind == "mvlgamma":
        # The float32 sum below in double: same offsets, accumulator order
        # and tree (CUDA's reduce over the p-long dim), nv_lgamma per term.
        # The double tensor's offsets are exact, so nothing rounds per step.
        var p = Int(rebind[Float64](p0))
        var c = rebind[Float64](p1)
        var start = Float64(1 - p) * Float64(0.5)
        var r = SIMD[dtype, width]()
        comptime for i in range(width):
            var xi = rebind[Float64](x[i])
            var acc = Array[Float64, 32](fill=Float64(0.0))
            for j in range(p):
                acc[j % 32] += nv_lgamma(
                    xi + (start + Float64(j) * Float64(0.5))
                )
            comptime for level in range(5):
                comptime half = 16 >> level
                comptime for k in range(half):
                    acc[k] += acc[k + half]
            r[i] = rebind[Scalar[dtype]](acc[0] + c)
        return r
    elif ct == DType.float64 and (kind == "logit" or kind == "polygamma"):
        var r = SIMD[dtype, width]()
        comptime if kind == "logit":
            var eps = rebind[Scalar[dtype]](p0)
            comptime for i in range(width):
                r[i] = logit(x[i], eps)
        else:
            var n = Int32(Int(rebind[Float64](p0))) * Int32(65536) + Int32(
                Int(rebind[Float64](p1))
            )
            var nf = rebind[Float64](p2)
            comptime for i in range(width):
                r[i] = rebind[Scalar[dtype]](
                    polygamma_lane(
                        rebind[Float64](x[i]),
                        n,
                        abs(nf),
                        nf > 0.0,
                        nf != 0.0,
                    )
                )
        return r
    else:
        comptime assert ct == DType.float32, "float64 has no special port"
        var xf = x.cast[DType.float32]()
        var r = SIMD[DType.float32, width]()
        comptime if kind == "logit":
            var eps = rebind[Float32](p0)
            comptime if is_apple_gpu():
                var lo = rebind[Float32](p1)
                var hi = rebind[Float32](p2)
                comptime for i in range(width):
                    r[i] = mps_logit(xf[i], eps, lo, hi)
            else:
                comptime for i in range(width):
                    r[i] = logit(xf[i], eps)
        elif kind == "polygamma":
            # The order's int32 in two exact float halves, then 0 for
            # digamma / trigamma, else the order as a float, negated when
            # even (see `polygamma_lane`).
            var n = Int32(Int(rebind[Float32](p0))) * Int32(65536) + Int32(
                Int(rebind[Float32](p1))
            )
            var nf = rebind[Float32](p2)
            comptime for i in range(width):
                r[i] = polygamma_lane(
                    xf[i], n, abs(nf), nf > Float32(0.0), nf != Float32(0.0)
                )
        else:
            comptime assert kind == "mvlgamma", "unsupported param kind"
            var p = Int(rebind[Float32](p0))
            var c = rebind[Float32](p1)
            var start = Float32(1 - p) * Float32(0.5)
            comptime for i in range(width):
                # The sum(-1) as CUDA's reduce kernel orders it (measured on
                # H100, exact for p <= 100): term j into accumulator j % 32,
                # then a halving tree over the 32 accumulators.
                var acc = Array[Float32, 32](fill=Float32(0.0))
                for j in range(p):
                    # `arange(-p/2 + 1/2, 1/2, 1/2)` in the tensor dtype
                    # (each offset rounds: bfloat16 -129.5 is -130), then
                    # `add(self)`, rounding again.
                    var off = _to_dtype_rounded[dtype](
                        SIMD[DType.float32, 1](
                            start + Float32(j) * Float32(0.5)
                        )
                    )[0]
                    var arg = _to_dtype_rounded[dtype](
                        SIMD[DType.float32, 1](xf[i] + off)
                    )[0]
                    var lg: Float32
                    comptime if is_apple_gpu():
                        lg = mps_log_gamma(arg)
                    else:
                        lg = nv_lgammaf(arg)
                    acc[j % 32] += _to_dtype_rounded[dtype](
                        SIMD[DType.float32, 1](lg)
                    )[0]
                comptime for level in range(5):
                    comptime half = 16 >> level
                    comptime for k in range(half):
                        acc[k] += acc[k + half]
                r[i] = (
                    _to_dtype_rounded[dtype](SIMD[DType.float32, 1](acc[0]))[0]
                    + c
                )
        return r.cast[dtype]()


@always_inline
def polygamma_lane[
    dt: DType
](x: Scalar[dt], n: Int32, nf: Scalar[dt], odd: Bool, series: Bool) -> Scalar[
    dt
]:
    """polygamma(n, x) as each backend's kernel computes it. ATen picks
    digamma / trigamma on the int64 order; `series` says it was 2 or more.
    CUDA's series then takes an `int` (`n`, the order truncated to 32 bits;
    0 or 1 itself when `series` is False), Apple's the int64 order itself
    (`nf` = its float, `odd` its parity)."""
    comptime if is_apple_gpu():
        comptime assert dt == DType.float32, "no float64 on Apple GPUs"
        var xf = rebind[Float32](x)
        if series:
            return rebind[Scalar[dt]](
                mps_polygamma_series(xf, rebind[Float32](nf), odd)
            )
        return rebind[Scalar[dt]](mps_polygamma(xf, Int(n)))
    else:
        if series:
            return polygamma_series(x, n)
        return polygamma(x, Int(n))


@always_inline
def elementwise_polygamma[
    n: Int, series: Bool, dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """The polygamma function lane by lane for the torch.compile graph, the
    order `n` (the int64 order) a custom-op parameter: exact, where a float
    operand of the input dtype would round it."""
    comptime ct = DType.float64 if dtype == DType.float64 else DType.float32
    var xf = x.cast[ct]()
    var r = SIMD[ct, width]()
    comptime for i in range(width):
        r[i] = polygamma_lane(
            xf[i], Int32(n), Scalar[ct](n), n % 2 != 0, series
        )
    return r.cast[dtype]()
