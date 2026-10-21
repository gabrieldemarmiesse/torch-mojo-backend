"""Apple 8x8 simdgroup-matrix primitives, vendored from
linalg.matmul.gpu.apple.matmul_8x8, whose helpers are not exported."""

from std.sys import llvm_intrinsic

comptime MMA8_DIM = 8
comptime FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane


@always_inline
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout: lane owns
    (row, col_base) and (row, col_base + 1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _mma8x8(
    a: SIMD[DType.float32, FRAG8],
    b: SIMD[DType.float32, FRAG8],
    c: SIMD[DType.float32, FRAG8],
) -> SIMD[DType.float32, FRAG8]:
    """One 8x8x8 simdgroup-matrix multiply-accumulate: D = A @ B + C."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, FRAG8],
    ](a, b, c)
