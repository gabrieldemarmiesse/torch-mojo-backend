# ===----------------------------------------------------------------------=== #
# Host-side float conversions in integer arithmetic: the host's own float
# instructions run under the process's FTZ / DAZ flags and may flush a
# subnormal float result, where c10's conversions (what stock CUDA applies to
# a CPU scalar or a Scalar parameter) keep it.
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast


@always_inline
def rne_shift(x: UInt64, s: Int) -> UInt64:
    """x / 2^s rounded to nearest even, in integers."""
    if s <= 0:
        return x
    if s > 63:
        return 0
    var q = x >> UInt64(s)
    var rem = x & ((UInt64(1) << UInt64(s)) - 1)
    var half = UInt64(1) << UInt64(s - 1)
    if rem > half or (rem == half and (q & 1) == 1):
        q += 1
    return q


@always_inline
def f32_bits_of(v: Float64) -> UInt32:
    """The bits of `static_cast<float>(v)`, rounded to nearest even in
    integers (the host's conversion may flush a subnormal result); a NaN
    keeps its sign and top payload bits, quieted."""
    var d = bitcast[DType.uint64](v)
    var sign = (d >> 63).cast[DType.uint32]() << 31
    var e = Int((d >> 52) & 0x7FF)
    var m = d & 0xFFFFFFFFFFFFF
    if e == 0x7FF:
        if m == 0:
            return sign | 0x7F800000
        return sign | 0x7FC00000 | (m >> 29).cast[DType.uint32]()
    if e == 0:
        return sign
    var full = m | (UInt64(1) << 52)
    var fe = e - 1023 + 127
    if fe >= 1:
        var bits = (UInt64(fe) << 23) + rne_shift(full, 29) - (UInt64(1) << 23)
        return sign | min(bits, UInt64(0x7F800000)).cast[DType.uint32]()
    return sign | rne_shift(full, 29 + 1 - fe).cast[DType.uint32]()


@always_inline
def c10_half[dt: DType](v: Float64) -> Scalar[dt]:
    """c10's Half(float) / BFloat16(float) of `float(v)`, as stock CUDA
    converts a CPU scalar (`iter.scalar_value<scalar_t>`): the double
    rounded to float first, then to nearest even (subnormals kept); a NaN
    becomes 0x7E00 with its sign (Half) or 0x7FC0 (BFloat16). Integer
    arithmetic only: the host's float conversions may flush subnormals."""
    var d = bitcast[DType.uint64](v)
    var sign = UInt32((d >> 63).cast[DType.uint32]())
    var nan = ((d >> 52) & 0x7FF) == 0x7FF and (d & 0xFFFFFFFFFFFFF) != 0
    var u = f32_bits_of(v) & 0x7FFFFFFF  # float(v)'s magnitude
    comptime if dt == DType.bfloat16:
        if nan:
            return bitcast[dt](UInt16(0x7FC0))
        var w = u | (sign << 31)
        var r = (w + 0x7FFF + ((w >> 16) & 1)) >> 16
        return bitcast[dt](r.cast[DType.uint16]())
    else:
        var h: UInt32
        if nan:
            h = 0x7E00
        elif u >= 0x47800000:  # 2^16 and up (inf too): inf
            h = 0x7C00
        elif u < 0x38800000:  # below half's smallest normal: 2^-24 units
            var fe = Int(u >> 23)
            var m = UInt64(u & 0x7FFFFF)
            if fe != 0:
                m |= UInt64(1) << 23
            else:
                fe = 1
            h = UInt32(rne_shift(m, 126 - fe).cast[DType.uint32]())
        else:
            var m = u - 0x38000000  # rebias 127 -> 15, kept in place
            h = (m + 0xFFF + ((m >> 13) & 1)) >> 13
        return bitcast[dt]((h | (sign << 15)).cast[DType.uint16]())
