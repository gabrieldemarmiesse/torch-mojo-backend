"""Float8 <-> float32 conversions, bit for bit the ones c10 performs.

Every torch float8 dtype converts through float32 (`c10::Float8_*(float)`
and `operator float()`), and CUDA's copy kernels run exactly those
functions (`aten/src/ATen/native/cuda/Copy.cu`, `float8_copy_kernel_cuda`).
They are ported here as integer bit manipulation (plus the one float add
c10 itself uses to round into the subnormal range) rather than left to the
hardware: PTX's `cvt.*.e4m3x2/e5m2x2` only exists `.satfinite`, which
clamps an overflow to the largest finite value where torch gives NaN
(e4m3fn) or inf (e5m2), and the Apple GPU has no float8 conversion at all.
The sources are torch/headeronly/util/Float8_*.h (torch 2.14):

| dtype | to float32 | from float32 |
|---|---|---|
| float8_e4m3fn | `fp8e4m3fn_to_fp32_value` | `fp8e4m3fn_from_fp32_value` |
| float8_e5m2 | `fp8e5m2_to_fp32_value` | `fp8e5m2_from_fp32_value` |
| float8_e4m3fnuz | `fp8_fnuz_to_fp32_value<4, 3>` | `fp8e4m3fnuz_from_fp32_value` |
| float8_e5m2fnuz | `fp8_fnuz_to_fp32_value<5, 2>` | `fp8e5m2fnuz_from_fp32_value` |
| float8_e8m0fnu | `Float8_e8m0fnu::operator float` | `fp8e8m0fnu_from_fp32_value` |

Plain SIMD math with no accelerator choice: the host (`.item()`) and every
GPU run the same code.
"""

from std.bit import count_leading_zeros
from std.memory import bitcast


@always_inline
def is_float8(dt: DType) -> Bool:
    """One of the five torch float8 dtypes."""
    return (
        dt == DType.float8_e4m3fn
        or dt == DType.float8_e5m2
        or dt == DType.float8_e4m3fnuz
        or dt == DType.float8_e5m2fnuz
        or dt == DType.float8_e8m0fnu
    )


comptime U32 = DType.uint32


@always_inline
def _u32[w: Int](v: Int) -> SIMD[U32, w]:
    """A uint32 constant from an Int that may be negative (wraps mod 2**32,
    as c10's `(uint32_t)(7 - 127) << 23` does)."""
    return SIMD[U32, w](UInt32(v & 0xFFFFFFFF))


@always_inline
def _f32_bits[w: Int](v: SIMD[U32, w]) -> SIMD[DType.float32, w]:
    return bitcast[DType.float32, w](v)


@always_inline
def _ieee_fp8_to_f32[
    E: Int, M: Int, BIAS: Int, w: Int
](x: SIMD[U32, w]) -> SIMD[U32, w]:
    """Finite, non-NaN-encoded float8 bits (sign, E exponent bits, M
    mantissa bits, bias BIAS) to float32 bits, subnormals normalized. The
    callers patch in their own special encodings afterwards."""
    var sign = (x >> 7) << 31
    var e = (x >> _u32[w](M)) & _u32[w]((1 << E) - 1)
    var m = x & _u32[w]((1 << M) - 1)
    var normal = (
        sign | ((e + _u32[w](127 - BIAS)) << 23) | (m << _u32[w](23 - M))
    )
    # Subnormal: m * 2**(1 - BIAS - M) with its top set bit at position p
    # (0 <= p < M) becomes 2**(p + 1 - BIAS - M) * 1.f -- always a normal
    # float32, so no float arithmetic (and no subnormal flushing) is needed.
    var p = _u32[w](31) - count_leading_zeros(m | 1)
    var sub = (
        sign
        | ((p + _u32[w](1 - BIAS - M + 127)) << 23)
        | ((m - (_u32[w](1) << p)) << (_u32[w](23) - p))
    )
    var zero = sign
    return e.ne(0).select(normal, m.ne(0).select(sub, zero))


@always_inline
def fp8_to_f32[
    dt: DType, w: Int
](bits: SIMD[DType.uint8, w]) -> SIMD[DType.float32, w]:
    """float8 `dt` (as its raw bits) to float32, as c10's `operator float()`."""
    var x = bits.cast[U32]()
    comptime if dt == DType.float8_e4m3fn:
        var v = _ieee_fp8_to_f32[4, 3, 7, w](x)
        # S.1111.111 is the only NaN; c10 returns sign | 0x7FF00000.
        var nan = (x & 0x7F).eq(0x7F)
        return _f32_bits(nan.select((x >> 7) << 31 | 0x7FF00000, v))
    elif dt == DType.float8_e5m2:
        var v = _ieee_fp8_to_f32[5, 2, 15, w](x)
        # Exponent 11111: inf (m == 0) or NaN, payload kept and quieted as
        # c10's fp16 -> fp32 (a float multiply) leaves it.
        var special = (x & 0x7C).eq(0x7C)
        var m = x & 3
        var quiet = m.ne(0).select(_u32[w](0x400000), _u32[w](0))
        var s = (x >> 7) << 31 | 0x7F800000 | (m << 21) | quiet
        return _f32_bits(special.select(s, v))
    elif dt == DType.float8_e4m3fnuz or dt == DType.float8_e5m2fnuz:
        comptime E = 4 if dt == DType.float8_e4m3fnuz else 5
        comptime M = 7 - E
        # fp8_fnuz_to_fp32_value: exponent bias 2**(E-1), 0x80 is the NaN,
        # there is no -0 and no inf.
        var v = _ieee_fp8_to_f32[E, M, 1 << (E - 1), w](x)
        return _f32_bits(x.eq(0x80).select(_u32[w](0x7F800001), v))
    elif dt == DType.float8_e8m0fnu:
        # A bare exponent: 2**(x - 127); 0 is 2**-127 (a float32 subnormal),
        # 0xFF is the NaN.
        var v = x << 23
        v = x.eq(0).select(_u32[w](0x00400000), v)
        return _f32_bits(x.eq(0xFF).select(_u32[w](0x7F800001), v))
    else:
        comptime assert False, "fp8_to_f32: not a float8 dtype"


@always_inline
def _round_ieee[
    M: Int, BIAS: Int, DENORM_EXP: Int, MIN_NORMAL_EXP: Int, w: Int
](f: SIMD[U32, w]) -> SIMD[U32, w]:
    """The finite, non-negative branch every IEEE-like c10 encoder shares:
    round-to-nearest-even into the subnormal range through one float add
    (`denorm_mask`), else in the normal range on the bits. `f` is the
    float32 magnitude bits; the result is the unsigned float8 code."""
    var denorm_mask = _u32[w](DENORM_EXP << 23)
    var den = (
        bitcast[U32, w](_f32_bits(f) + _f32_bits(denorm_mask)) - denorm_mask
    )
    var mant_odd = (f >> _u32[w](23 - M)) & 1
    var norm = (
        f + _u32[w](((BIAS - 127) << 23) + (1 << (22 - M)) - 1) + mant_odd
    ) >> _u32[w](23 - M)
    return f.lt(_u32[w](MIN_NORMAL_EXP << 23)).select(den, norm) & 0xFF


@always_inline
def f32_to_fp8[
    dt: DType, w: Int
](v: SIMD[DType.float32, w]) -> SIMD[DType.uint8, w]:
    """float32 to float8 `dt` raw bits, as c10's `Float8_*(float)`: round to
    nearest even; an overflow is NaN for e4m3fn and the fnuz types and inf
    for e5m2 (no saturation); e8m0fnu rounds the exponent half-up as c10
    does and drops the sign."""
    var bits = bitcast[U32, w](v)
    var sign = bits & 0x80000000
    var f = bits ^ sign
    var sign8 = sign >> 24
    comptime if dt == DType.float8_e4m3fn:
        # fp8_max = 480.0f, the first value past the range: NaN (0x7F).
        var r = _round_ieee[3, 7, 141, 121, w](f)
        r = f.ge(_u32[w](1087 << 20)).select(_u32[w](0x7F), r)
        return (r | sign8).cast[DType.uint8]()
    elif dt == DType.float8_e5m2:
        # 65536.0f and up: inf (0x7C), NaN input stays NaN (0x7F).
        var r = _round_ieee[2, 15, 134, 113, w](f)
        var over = f.gt(_u32[w](255 << 23)).select(_u32[w](0x7F), _u32[w](0x7C))
        r = f.ge(_u32[w](143 << 23)).select(over, r)
        return (r | sign8).cast[DType.uint8]()
    elif dt == DType.float8_e4m3fnuz or dt == DType.float8_e5m2fnuz:
        comptime E4 = dt == DType.float8_e4m3fnuz
        comptime M = 3 if E4 else 2
        comptime BIAS = 8 if E4 else 16
        comptime DENORM = 0x8C if E4 else 0x85
        comptime MIN_NORMAL = 0x78 if E4 else 0x70
        comptime FNUZ_MAX = 0x87 if E4 else 0x8F
        var r = _round_ieee[M, BIAS, DENORM, MIN_NORMAL, w](f)
        # A result of zero is +0 (fnuz has no -0); overflow and NaN are 0x80.
        r = r.eq(0).select(r, r | sign8)
        r = f.ge(_u32[w](FNUZ_MAX << 23)).select(_u32[w](0x80), r)
        return r.cast[DType.uint8]()
    elif dt == DType.float8_e8m0fnu:
        var e = (bits >> 23) & 0xFF
        var g = (bits & 0x400000).ne(0)
        var rest = (bits & 0x3FFFFF).ne(0)  # r or s
        var up = g & (rest | e.gt(0))
        var r = up.select(e + 1, e)
        return e.eq(0xFF).select(e, r).cast[DType.uint8]()
    else:
        comptime assert False, "f32_to_fp8: not a float8 dtype"


def fp8_bits_to_f64(dt: DType, bits: UInt8) -> Float64:
    """One float8 value, read on the host (`.item()`)."""
    var b = SIMD[DType.uint8, 1](bits)
    if dt == DType.float8_e4m3fn:
        return Float64(fp8_to_f32[DType.float8_e4m3fn, 1](b))
    if dt == DType.float8_e5m2:
        return Float64(fp8_to_f32[DType.float8_e5m2, 1](b))
    if dt == DType.float8_e4m3fnuz:
        return Float64(fp8_to_f32[DType.float8_e4m3fnuz, 1](b))
    if dt == DType.float8_e5m2fnuz:
        return Float64(fp8_to_f32[DType.float8_e5m2fnuz, 1](b))
    return Float64(fp8_to_f32[DType.float8_e8m0fnu, 1](b))


def f64_to_fp8_bits(dt: DType, v: Float64) -> UInt8:
    """One float8 value from a host double: through float32 first, as c10
    (whose constructors only take float) does."""
    var f = SIMD[DType.float32, 1](Float32(v))
    if dt == DType.float8_e4m3fn:
        return f32_to_fp8[DType.float8_e4m3fn, 1](f)[0]
    if dt == DType.float8_e5m2:
        return f32_to_fp8[DType.float8_e5m2, 1](f)[0]
    if dt == DType.float8_e4m3fnuz:
        return f32_to_fp8[DType.float8_e4m3fnuz, 1](f)[0]
    if dt == DType.float8_e5m2fnuz:
        return f32_to_fp8[DType.float8_e5m2fnuz, 1](f)[0]
    return f32_to_fp8[DType.float8_e8m0fnu, 1](f)[0]
