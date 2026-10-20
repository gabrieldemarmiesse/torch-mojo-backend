# ===----------------------------------------------------------------------=== #
# Which NaN a float64 result carries on NVIDIA, made explicit.
#
# A double add / mul / fma on an NVIDIA GPU returns one of its NaN operands,
# quieted, picked by the operand's slot in the SASS instruction (DADD / DMUL:
# B, then A; DFMA: B, then C, then A), and ptxas chooses the slots. A uniform
# operand -- a CPU scalar or an alpha in stock CUDA torch's functors -- always
# lands in B, so its NaN wins deterministically; our kernels put it in place
# with these helpers rather than hang on the slots the Mojo compiler and
# ptxas pick for our own instructions. Between two tensor operands the slots
# change with the launch (mul's vectorized body is DMUL a, b, its tail
# DMUL b, a), so no kernel can reproduce that choice.
#
# float32 needs none of this: a float arithmetic NaN is always the canonical
# 0x7FFFFFFF on NVIDIA, whatever the operands.
#
# Other targets keep their native NaN propagation (`nan_first*` return `r`).
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast
from std.sys import is_nvidia_gpu
from std.sys.info import size_of
from std.utils.numerics import nan


@always_inline
def _bits_dtype[dtype: DType]() -> DType:
    comptime if size_of[dtype]() == 8:
        return DType.uint64
    elif size_of[dtype]() == 4:
        return DType.uint32
    else:
        return DType.uint16


@always_inline
def is_nan_bits[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    """isnan from the bits (the GPU builds' fast-math flags may fold a float
    NaN test away)."""
    comptime bt = _bits_dtype[dtype]()
    comptime nbits = size_of[dtype]() * 8
    comptime sign = Scalar[bt](1) << Scalar[bt](nbits - 1)
    var m = bitcast[bt, w](x) & ~sign
    comptime if dtype == DType.float64:
        return m.gt(Scalar[bt](0x7FF0000000000000))
    elif dtype == DType.float32:
        return m.gt(Scalar[bt](0x7F800000))
    elif dtype == DType.bfloat16:
        return m.gt(Scalar[bt](0x7F80))
    else:
        comptime assert dtype == DType.float16, "a float dtype"
        return m.gt(Scalar[bt](0x7C00))


@always_inline
def quiet_nan[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`x` (a NaN) with its quiet bit set, sign and payload kept."""
    comptime bt = _bits_dtype[dtype]()
    comptime if dtype == DType.float64:
        comptime q = Scalar[bt](0x0008000000000000)
        return bitcast[dtype, w](bitcast[bt, w](x) | q)
    elif dtype == DType.float32:
        comptime q = Scalar[bt](0x00400000)
        return bitcast[dtype, w](bitcast[bt, w](x) | q)
    elif dtype == DType.bfloat16:
        comptime q = Scalar[bt](0x0040)
        return bitcast[dtype, w](bitcast[bt, w](x) | q)
    else:
        comptime q = Scalar[bt](0x0200)
        return bitcast[dtype, w](bitcast[bt, w](x) | q)


@always_inline
def nan_first2[
    dtype: DType, w: Int
](r: SIMD[dtype, w], p: SIMD[dtype, w], q: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`r`, unless `p` or `q` is a NaN: then the first of them that is, quieted.
    """
    comptime if is_nvidia_gpu() and dtype == DType.float64:
        var np = is_nan_bits(p)
        var nq = is_nan_bits(q)
        return np.select(quiet_nan(p), nq.select(quiet_nan(q), r))
    else:
        return r


@always_inline
def nan_first3[
    dtype: DType, w: Int
](
    r: SIMD[dtype, w], p: SIMD[dtype, w], q: SIMD[dtype, w], s: SIMD[dtype, w]
) -> SIMD[dtype, w]:
    """`nan_first2` over three operands, in priority order p, q, s."""
    comptime if is_nvidia_gpu() and dtype == DType.float64:
        return nan_first2(nan_first2(r, q, s), p, p)
    else:
        return r


@always_inline
def arith_nan[dtype: DType, w: Int]() -> SIMD[dtype, w]:
    """The NaN a float32 arithmetic instruction returns on NVIDIA, whatever
    its operands: 0x7FFFFFFF, which rounds to 0x7FFF in half and bfloat16.
    The default NaN elsewhere."""
    comptime if is_nvidia_gpu() and dtype != DType.float64:
        comptime bt = _bits_dtype[dtype]()
        comptime if dtype == DType.float32:
            return bitcast[dtype, w](SIMD[bt, w](Scalar[bt](0x7FFFFFFF)))
        else:
            return bitcast[dtype, w](SIMD[bt, w](Scalar[bt](0x7FFF)))
    else:
        return SIMD[dtype, w](nan[dtype]())


@always_inline
def nan_result2[
    dtype: DType, w: Int
](r: SIMD[dtype, w], p: SIMD[dtype, w], q: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`r`, or the NaN an arithmetic chain over `p` and `q` ends in when
    either is NaN: on NVIDIA the first NaN of p, q, quieted, for float64
    (`nan_first2`), the canonical `arith_nan` otherwise; the default NaN
    off NVIDIA."""
    var any_nan = is_nan_bits(p) | is_nan_bits(q)
    comptime if is_nvidia_gpu() and dtype == DType.float64:
        return nan_first2(r, p, q)
    else:
        return any_nan.select(arith_nan[dtype, w](), r)
