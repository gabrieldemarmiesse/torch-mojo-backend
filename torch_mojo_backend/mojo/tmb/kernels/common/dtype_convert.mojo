"""Element conversion between copy dtypes, the way ATen's copy_ does it."""

from std.memory import bitcast
from std.sys import is_apple_gpu


@always_inline
def storage_dtype[dt: DType]() -> DType:
    """What a kernel loads and stores for `dt`: bool travels as uint8."""
    return DType.uint8 if dt == DType.bool else dt


@always_inline
def f32_to_bf16[w: Int](v: SIMD[DType.float32, w]) -> SIMD[DType.bfloat16, w]:
    """`v.cast[DType.bfloat16]()`, rounded to nearest even like c10::BFloat16.

    Apple GPUs' hardware conversion flushes subnormal float32 inputs to zero
    (measured on an M4: 1e-39 -> 0.0, where torch gives 1.0102e-39), so on
    Metal the rounding is done on the bits: add 0x7FFF plus the kept LSB and
    shift (overflow carries into inf, as it should). NaN keeps its sign and
    is quieted. Every other target keeps its native conversion.
    """
    comptime if is_apple_gpu():
        var bits = bitcast[DType.uint32, w](v)
        var rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        var nan = (bits & 0x7FFFFFFF).gt(0x7F800000)
        var picked = nan.select((bits >> 16) | 0x40, rounded)
        return bitcast[DType.bfloat16, w](picked.cast[DType.uint16]())
    else:
        return v.cast[DType.bfloat16]()


@always_inline
def convert[
    src: DType, dst: DType, S: DType, D: DType, w: Int
](v: SIMD[S, w]) -> SIMD[D, w]:
    """`src` -> `dst` on their storage dtypes `S` / `D` (see storage_dtype)."""
    comptime if dst == DType.bool:
        return rebind[SIMD[D, w]](v.ne(0).cast[DType.uint8]())
    elif S == D:
        return rebind[SIMD[D, w]](v)
    elif (dst == DType.float16 or dst == DType.bfloat16) and (
        src != DType.float32
    ):
        # c10::Half and c10::BFloat16 are built from float: round through it.
        return v.cast[DType.float32]().cast[D]()
    elif src == DType.float32 and dst == DType.bfloat16:
        return rebind[SIMD[D, w]](
            f32_to_bf16(rebind[SIMD[DType.float32, w]](v))
        )
    else:
        return v.cast[D]()
