"""Element conversion between copy dtypes, the way ATen's copy_ does it."""


@always_inline
def storage_dtype[dt: DType]() -> DType:
    """What a kernel loads and stores for `dt`: bool travels as uint8."""
    return DType.uint8 if dt == DType.bool else dt


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
    else:
        return v.cast[D]()
