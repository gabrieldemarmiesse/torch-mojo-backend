"""ATen's CUDA distribution kernel and transforms, reproduced bit for bit.

`distribution_elementwise_grid_stride_kernel` (aten/src/ATen/native/cuda/
DistributionTemplates.h): thread `idx` owns curand subsequence `idx`; each
loop iteration is one `curand4` (UNROLL = 4 float / u32 values, 2 double /
u64 values) and value `ii` lands on element `linear + T * ii`, `T = 256 *
grid`. The transforms are ATen/core/TransformationHelper.h with the same
accumulation type (float, or double for a float64 tensor), the same fma
contractions and the same math: ATen's `at::log/exp/tan/log1p` are the CUDA
fast intrinsics for float and the precise libdevice routines for double
(ATen/NumericUtils.h). The grid, the counter reservation and the TensorIterator
element order are the caller's (tmb/ops/random.mojo).
"""
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceil, fma
from std.sys.info import (
    bit_width_of,
    has_accelerator,
    has_apple_gpu_accelerator,
)

from tmb.kernels.random.philox import (
    U32x2,
    U32x4,
    curand4,
    curand_ctr,
    curand_key,
    curand_normal2_double,
    curand_normal4,
    curand_uniform2_double,
    curand_uniform4,
)
from tmb.kernels.common.libdevice_port import (
    nv_exp,
    nv_fast_expf,
    nv_fast_logf,
    nv_fast_tanf,
    nv_log,
    nv_log1p,
    nv_tan,
)
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.random.rng_metadata import I64x8

comptime DIST_UNIFORM = 0
comptime DIST_NORMAL = 1
comptime DIST_LOG_NORMAL = 2
comptime DIST_CAUCHY = 3
comptime DIST_EXPONENTIAL = 4
comptime DIST_GEOMETRIC = 5
comptime DIST_BERNOULLI = 6
comptime DIST_RANDOM_FROM_TO_32 = 7
comptime DIST_RANDOM_FROM_TO_64 = 8
comptime DIST_RANDOM_FULL_64 = 9
comptime DIST_RANDOM_32 = 10
comptime DIST_RANDOM_64 = 11

comptime BLOCK = 256

comptime _EPS_F32 = Float32(1.1920929e-07)
comptime _EPS_F64 = Float64(2.220446049250313e-16)
comptime _PI_F32 = Float32(3.14159265358979323846)
comptime _PI_F64 = Float64(3.14159265358979323846)


@always_inline
def acc_dtype[dtype: DType]() -> DType:
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def dist_unroll[DIST: Int, dtype: DType]() -> Int:
    comptime if (
        DIST == DIST_RANDOM_FROM_TO_64
        or DIST == DIST_RANDOM_FULL_64
        or DIST == DIST_RANDOM_64
    ):
        return 2
    elif DIST == DIST_RANDOM_FROM_TO_32 or DIST == DIST_RANDOM_32:
        return 4
    elif dtype == DType.float64:
        return 2
    else:
        return 4


@always_inline
def _dist_name[DIST: Int]() -> StaticString:
    comptime if DIST == DIST_UNIFORM:
        return "uniform"
    elif DIST == DIST_NORMAL:
        return "normal"
    elif DIST == DIST_LOG_NORMAL:
        return "log_normal"
    elif DIST == DIST_CAUCHY:
        return "cauchy"
    elif DIST == DIST_EXPONENTIAL:
        return "exponential"
    elif DIST == DIST_GEOMETRIC:
        return "geometric"
    elif DIST == DIST_BERNOULLI:
        return "bernoulli"
    elif DIST == DIST_RANDOM_FROM_TO_32:
        return "random_from_to32"
    elif DIST == DIST_RANDOM_FROM_TO_64:
        return "random_from_to64"
    elif DIST == DIST_RANDOM_FULL_64:
        return "random_full64"
    elif DIST == DIST_RANDOM_32:
        return "random32"
    else:
        return "random64"


@always_inline
def _uniform_acc[ACC: DType, N: Int](w: U32x4) -> SIMD[ACC, N]:
    comptime if ACC == DType.float64:
        return rebind[SIMD[ACC, N]](curand_uniform2_double(w))
    else:
        return rebind[SIMD[ACC, N]](curand_uniform4(w))


@always_inline
def _normal_acc[ACC: DType, N: Int](w: U32x4) -> SIMD[ACC, N]:
    comptime if ACC == DType.float64:
        return rebind[SIMD[ACC, N]](curand_normal2_double(w))
    else:
        return rebind[SIMD[ACC, N]](curand_normal4(w))


@always_inline
def _u64_pairs(w: U32x4) -> SIMD[DType.uint64, 2]:
    var w64 = w.cast[DType.uint64]()
    return SIMD[DType.uint64, 2](
        (w64[0] << 32) | w64[1], (w64[2] << 32) | w64[3]
    )


@always_inline
def _int64_to[dtype: DType, N: Int](v: SIMD[DType.int64, N]) -> SIMD[dtype, N]:
    """`static_cast<scalar_t>(int64_t)`: the 16-bit floats go through float."""
    comptime if dtype == DType.bool:
        return rebind[SIMD[dtype, N]](v.ne(0))
    elif dtype == DType.float16 or dtype == DType.bfloat16:
        return v.cast[DType.float32]().cast[dtype]()
    else:
        return v.cast[dtype]()


@always_inline
def _float_to[
    dtype: DType, ACC: DType, N: Int
](v: SIMD[ACC, N]) -> SIMD[dtype, N]:
    comptime if dtype == DType.bool:
        return rebind[SIMD[dtype, N]](v.ne(0))
    else:
        return v.cast[dtype]()


@always_inline
def _log_acc[ACC: DType](v: Scalar[ACC]) -> Scalar[ACC]:
    comptime if ACC == DType.float64:
        return rebind[Scalar[ACC]](nv_log(rebind[Float64](v)))
    else:
        return rebind[Scalar[ACC]](nv_fast_logf(rebind[Float32](v)))


@always_inline
def _log1p_acc[ACC: DType](v: Scalar[ACC]) -> Scalar[ACC]:
    comptime if ACC == DType.float64:
        return rebind[Scalar[ACC]](nv_log1p(rebind[Float64](v)))
    else:
        # at::log1p on device is `__logf(1.0f + x)`.
        return rebind[Scalar[ACC]](
            nv_fast_logf(Float32(1.0) + rebind[Float32](v))
        )


@always_inline
def _exp_acc[ACC: DType](v: Scalar[ACC]) -> Scalar[ACC]:
    comptime if ACC == DType.float64:
        return rebind[Scalar[ACC]](nv_exp(rebind[Float64](v)))
    else:
        return rebind[Scalar[ACC]](nv_fast_expf(rebind[Float32](v)))


@always_inline
def _tan_acc[ACC: DType](v: Scalar[ACC]) -> Scalar[ACC]:
    comptime if ACC == DType.float64:
        return rebind[Scalar[ACC]](nv_tan(rebind[Float64](v)))
    else:
        return rebind[Scalar[ACC]](nv_fast_tanf(rebind[Float32](v)))


# ---------------------------------------------------------------------------
# One transform per distribution: `w` is one curand4 result, the return is
# the UNROLL output values (ATen/core/TransformationHelper.h).
# ---------------------------------------------------------------------------


@always_inline
def _draw_uniform[
    dtype: DType, N: Int
](
    w: U32x4,
    from_out: Scalar[dtype],
    to_out: Scalar[dtype],
    from_acc: Scalar[acc_dtype[dtype]()],
    range_acc: Scalar[acc_dtype[dtype]()],
) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var u = _uniform_acc[ACC, N](w)
    var value = fma(u, SIMD[ACC, N](range_acc), SIMD[ACC, N](from_acc)).cast[
        dtype
    ]()
    # curand's (0, 1] folded onto [from, to): `value == to ? from : value`.
    return value.eq(SIMD[dtype, N](to_out)).select(
        SIMD[dtype, N](from_out), value
    )


@always_inline
def _draw_normal[
    dtype: DType, N: Int
](
    w: U32x4, mean: Scalar[acc_dtype[dtype]()], std: Scalar[acc_dtype[dtype]()]
) -> SIMD[acc_dtype[dtype](), N]:
    comptime ACC = acc_dtype[dtype]()
    return fma(_normal_acc[ACC, N](w), SIMD[ACC, N](std), SIMD[ACC, N](mean))


@always_inline
def _draw_log_normal[
    dtype: DType, N: Int
](
    w: U32x4, mean: Scalar[acc_dtype[dtype]()], std: Scalar[acc_dtype[dtype]()]
) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var n = _draw_normal[dtype, N](w, mean, std)
    var out = SIMD[ACC, N]()

    comptime for i in range(N):
        out[i] = _exp_acc[ACC](n[i])
    return out.cast[dtype]()


@always_inline
def _cauchy_one[
    ACC: DType
](u: Scalar[ACC], median: Scalar[ACC], sigma: Scalar[ACC]) -> Scalar[ACC]:
    comptime if ACC == DType.float32:
        # __tanf overflows at the ends of (0, 1); float32 only.
        comptime ONE_MINUS = Float32(1.0) - _EPS_F32
        var v = rebind[Float32](u)
        if v > ONE_MINUS:
            v = ONE_MINUS
        if v < _EPS_F32:
            v = _EPS_F32
        var t = nv_fast_tanf(_PI_F32 * (v - Float32(0.5)))
        return fma(sigma, rebind[Scalar[ACC]](t), median)
    else:
        var t = nv_tan(_PI_F64 * (rebind[Float64](u) - Float64(0.5)))
        return fma(sigma, rebind[Scalar[ACC]](t), median)


@always_inline
def _draw_cauchy[
    dtype: DType, N: Int
](
    w: U32x4,
    median: Scalar[acc_dtype[dtype]()],
    sigma: Scalar[acc_dtype[dtype]()],
) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var u = _uniform_acc[ACC, N](w)
    var out = SIMD[ACC, N]()

    comptime for i in range(N):
        out[i] = _cauchy_one[ACC](u[i], median, sigma)
    return out.cast[dtype]()


@always_inline
def _exponential_one[
    ACC: DType
](u: Scalar[ACC], lambd: Scalar[ACC]) -> Scalar[ACC]:
    # curand's range is (0, 1]: log(1) == 0 is excluded by hand.
    var lg: Scalar[ACC]
    comptime if ACC == DType.float32:
        var v = rebind[Float32](u)
        if v >= Float32(1.0) - _EPS_F32 / 2:
            lg = rebind[Scalar[ACC]](-_EPS_F32 / 2)
        else:
            lg = rebind[Scalar[ACC]](nv_fast_logf(v))
    else:
        var v = rebind[Float64](u)
        if v >= Float64(1.0) - _EPS_F64 / 2:
            lg = rebind[Scalar[ACC]](-_EPS_F64 / 2)
        else:
            lg = rebind[Scalar[ACC]](nv_log(v))
    return (Scalar[ACC](-1.0) / lambd) * lg


@always_inline
def _draw_exponential[
    dtype: DType, N: Int
](w: U32x4, lambd: Scalar[acc_dtype[dtype]()]) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var u = _uniform_acc[ACC, N](w)
    var out = SIMD[ACC, N]()

    comptime for i in range(N):
        out[i] = _exponential_one[ACC](u[i], lambd)
    return out.cast[dtype]()


@always_inline
def _draw_geometric[
    dtype: DType, N: Int
](w: U32x4, p: Scalar[acc_dtype[dtype]()]) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var u = _uniform_acc[ACC, N](w)
    var out = SIMD[ACC, N]()

    comptime for i in range(N):
        out[i] = ceil(_log_acc[ACC](u[i]) / _log1p_acc[ACC](-p))
    return _float_to[dtype, ACC, N](out)


@always_inline
def _draw_bernoulli[
    dtype: DType, N: Int
](w: U32x4, p: Scalar[acc_dtype[dtype]()]) -> SIMD[dtype, N]:
    comptime ACC = acc_dtype[dtype]()
    var keep = _uniform_acc[ACC, N](w).lt(SIMD[ACC, N](p))
    comptime if dtype == DType.bool:
        return rebind[SIMD[dtype, N]](keep)
    else:
        return keep.cast[dtype]()


@always_inline
def _random_words[N: Int](w: U32x4) -> SIMD[DType.uint64, N]:
    """The integer draw: four u32 words, or two u64 (hi << 32 | lo) pairs."""
    comptime if N == 4:
        return rebind[SIMD[DType.uint64, N]](w.cast[DType.uint64]())
    else:
        return rebind[SIMD[DType.uint64, N]](_u64_pairs(w))


@always_inline
def _draw_random_from_to[
    dtype: DType, N: Int
](w: U32x4, range_bits: Int64, base: Int64) -> SIMD[dtype, N]:
    """`(val % range) + base` in uint64, then `static_cast<int64_t>`."""
    var range_ = UInt64(range_bits.cast[DType.uint64]())
    var b = SIMD[DType.uint64, N](base.cast[DType.uint64]())
    var val = _random_words[N](w)
    return _int64_to[dtype, N](((val % range_) + b).cast[DType.int64]())


@always_inline
def _draw_random_full_64[dtype: DType, N: Int](w: U32x4) -> SIMD[dtype, N]:
    return _int64_to[dtype, N](_random_words[N](w).cast[DType.int64]())


@always_inline
def _draw_random[dtype: DType, N: Int](w: U32x4) -> SIMD[dtype, N]:
    """`transformation::uniform_int`."""
    var val = _random_words[N](w)
    comptime if dtype == DType.bool:
        return rebind[SIMD[dtype, N]]((val & 1).ne(0))
    else:
        comptime M: UInt64 = _uniform_int_modulus[dtype]()
        return _int64_to[dtype, N]((val % M).cast[DType.int64]())


@always_inline
def dist_draw[
    dtype: DType, DIST: Int
](
    w: U32x4,
    p_out0: Scalar[dtype],
    p_out1: Scalar[dtype],
    p_acc0: Scalar[acc_dtype[dtype]()],
    p_acc1: Scalar[acc_dtype[dtype]()],
    i0: Int64,
    i1: Int64,
) -> SIMD[dtype, dist_unroll[DIST, dtype]()]:
    """One `curand4` result -> UNROLL output values.

    Parameter slots by DIST: uniform (from_out, to_out, from_acc, range_acc);
    normal / log_normal (-, -, mean, std); cauchy (-, -, median, sigma);
    exponential (-, -, lambda, -); geometric / bernoulli (-, -, p, -);
    random_from_to (range bits in i0, base in i1).
    """
    comptime N = dist_unroll[DIST, dtype]()
    comptime if DIST == DIST_UNIFORM:
        return _draw_uniform[dtype, N](w, p_out0, p_out1, p_acc0, p_acc1)
    elif DIST == DIST_NORMAL:
        return _draw_normal[dtype, N](w, p_acc0, p_acc1).cast[dtype]()
    elif DIST == DIST_LOG_NORMAL:
        return _draw_log_normal[dtype, N](w, p_acc0, p_acc1)
    elif DIST == DIST_CAUCHY:
        return _draw_cauchy[dtype, N](w, p_acc0, p_acc1)
    elif DIST == DIST_EXPONENTIAL:
        return _draw_exponential[dtype, N](w, p_acc0)
    elif DIST == DIST_GEOMETRIC:
        return _draw_geometric[dtype, N](w, p_acc0)
    elif DIST == DIST_BERNOULLI:
        return _draw_bernoulli[dtype, N](w, p_acc0)
    elif DIST == DIST_RANDOM_FROM_TO_32 or DIST == DIST_RANDOM_FROM_TO_64:
        return _draw_random_from_to[dtype, N](w, i0, i1)
    elif DIST == DIST_RANDOM_FULL_64:
        return _draw_random_full_64[dtype, N](w)
    else:  # DIST_RANDOM_32 / DIST_RANDOM_64
        return _draw_random[dtype, N](w)


@always_inline
def _uniform_int_modulus[dtype: DType]() -> UInt64:
    """`transformation::uniform_int`: `2^digits + 1` for floats, `max + 1`
    for integers."""
    comptime if dtype.is_floating_point():
        return (UInt64(1) << UInt64(DType.mantissa_width[dtype]() + 1)) + 1
    elif dtype.is_unsigned():
        return UInt64(1) << UInt64(bit_width_of[dtype]())
    else:
        return UInt64(1) << UInt64(bit_width_of[dtype]() - 1)


@always_inline
def _offset_of[
    TRIVIAL: Bool
](li: Int, sizes: I64x8, strides: I64x8, ndim: Int, stride0: Int = 0) -> Int:
    """Element offset of iterator index `li` (dims fastest-first).

    `stride0` is the contiguous case's whole answer, and the caller passes it
    already loaded: see `_dist_kernel` for why it cannot be read here.
    """
    comptime if TRIVIAL:
        return li * stride0
    else:
        var rem = li
        var off = 0
        for d in range(ndim):
            var s = Int(sizes[d])
            off += (rem % s) * Int(strides[d])
            rem //= s
        return off


@__name(t"philox_grid_stride_{_dist_name[DIST]()}_{dtype}")
def _dist_kernel[
    dtype: DType, DIST: Int, TRIVIAL: Bool
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    numel_arg: Int64,
    ndim_arg: Int64,
    sizes: I64x8,
    strides: I64x8,
    p_out0: Scalar[dtype],
    p_out1: Scalar[dtype],
    p_acc0: Scalar[acc_dtype[dtype]()],
    p_acc1: Scalar[acc_dtype[dtype]()],
    i0: Int64,
    i1: Int64,
    seed: UInt64,
    offset: UInt64,
):
    comptime N = dist_unroll[DIST, dtype]()
    var numel = Int(numel_arg)
    var ndim = Int(ndim_arg)
    var total = Int(grid_dim.x) * Int(block_dim.x)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var ctr = curand_ctr(offset, UInt64(idx))
    var key = curand_key(seed)
    var rounded = ((numel - 1) // (total * N) + 1) * total * N
    var linear = idx
    var k: UInt64 = 0
    # `strides[0]` is loop-invariant, but it lives in parameter space and the
    # backend does not hoist that load out of the grid-stride loop: it reloaded
    # it once per unrolled lane, N times an iteration, which cost ~10% on the
    # large contiguous cases (H100 sm_90a). A SIMD argument was hoisted for
    # free, being a value rather than an aggregate; this reads it once instead.
    var stride0 = Int(strides[0])
    while linear < rounded:
        var vals = dist_draw[dtype, DIST](
            curand4(ctr, key, k), p_out0, p_out1, p_acc0, p_acc1, i0, i1
        )

        comptime for ii in range(N):
            var li = linear + total * ii
            if li < numel:
                dst[
                    unsafe_offset=_offset_of[TRIVIAL](
                        li, sizes, strides, ndim, stride0
                    )
                ] = vals[ii]
        k += 1
        linear += total * N


def enqueue_distribution[
    dtype: DType, DIST: Int
](
    ctx: DeviceContext,
    dst_addr: Int,
    numel: Int,
    ndim: Int,
    sizes: I64x8,
    strides: I64x8,
    grid: Int,
    p_out0: Scalar[dtype],
    p_out1: Scalar[dtype],
    p_acc0: Scalar[acc_dtype[dtype]()],
    p_acc1: Scalar[acc_dtype[dtype]()],
    i0: Int64,
    i1: Int64,
    seed: UInt64,
    offset: UInt64,
) raises:
    """Fill `numel` elements of `dst` in iterator order; `grid` is the launch
    grid the caller derived the counter reservation from."""
    comptime N = dist_unroll[DIST, dtype]()
    if numel <= 0:
        return
    var dst = _make_ptr[dtype](dst_addr)
    var trivial = ndim == 1

    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            if trivial:
                _enqueue_cached[_dist_kernel[dtype, DIST, True]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst.as_unsafe_any_origin(),
                    Int64(numel),
                    Int64(ndim),
                    sizes,
                    strides,
                    p_out0,
                    p_out1,
                    p_acc0,
                    p_acc1,
                    i0,
                    i1,
                    seed,
                    offset,
                )
            else:
                _enqueue_cached[_dist_kernel[dtype, DIST, False]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst.as_unsafe_any_origin(),
                    Int64(numel),
                    Int64(ndim),
                    sizes,
                    strides,
                    p_out0,
                    p_out1,
                    p_acc0,
                    p_acc1,
                    i0,
                    i1,
                    seed,
                    offset,
                )


# ---------------------------------------------------------------------------
# bernoulli_.Tensor: CUDA_tensor_apply2<scalar, prob, /*step=*/4> geometry
# (ATen/cuda/CUDAApplyUtils.cuh): 512 threads, grid = ceil(numel / 2048),
# thread t owns elements 4t..4t+3, and the functor re-initialises its state
# on every call, so a grid-stride revisit redraws the same four words.
# Dims arrive rearranged (`rearrangeDims`) and fastest-first.
# ---------------------------------------------------------------------------

comptime APPLY_BLOCK = 512


@__name(t"philox_apply2_bernoulli_{dtype}_{PDT}")
def _bernoulli_tensor_kernel[
    dtype: DType, PDT: DType
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    p: Pointer[Scalar[PDT], MutAnyOrigin],
    numel_arg: Int64,
    ndim_arg: Int64,
    sizes: I64x8,
    dst_strides: I64x8,
    p_strides: I64x8,
    seed: UInt64,
    offset: UInt64,
):
    var numel = Int(numel_arg)
    var ndim = Int(ndim_arg)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var key = curand_key(seed)
    var stride_all = Int(grid_dim.x) * Int(block_dim.x) * 4
    var linear = t * 4
    while linear < numel:
        var r = curand_uniform4(curand4(curand_ctr(offset, UInt64(t)), key, 0))

        comptime for j in range(4):
            var li = linear + j
            if li < numel:
                var keep = (
                    r[j].cast[PDT]()
                    <= p[
                        unsafe_offset=_offset_of[False](
                            li, sizes, p_strides, ndim
                        )
                    ]
                )
                dst[
                    unsafe_offset=_offset_of[False](
                        li, sizes, dst_strides, ndim
                    )
                ] = _bool_to[dtype](keep)
        linear += stride_all


@always_inline
def _bool_to[dtype: DType](b: Bool) -> Scalar[dtype]:
    comptime if dtype == DType.bool:
        return rebind[Scalar[dtype]](Scalar[DType.bool](b))
    else:
        return Scalar[dtype](1) if b else Scalar[dtype](0)


def enqueue_bernoulli_tensor[
    dtype: DType, PDT: DType
](
    ctx: DeviceContext,
    dst_addr: Int,
    p_addr: Int,
    numel: Int,
    ndim: Int,
    sizes: I64x8,
    dst_strides: I64x8,
    p_strides: I64x8,
    grid: Int,
    seed: UInt64,
    offset: UInt64,
) raises:
    if numel <= 0:
        return
    var dst = _make_ptr[dtype](dst_addr)
    var p = _make_ptr[PDT](p_addr)

    comptime if (
        dtype == DType.float64 or PDT == DType.float64
    ) and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            _enqueue_cached[_bernoulli_tensor_kernel[dtype, PDT]](
                ctx,
                grid,
                1,
                1,
                APPLY_BLOCK,
                dst.as_unsafe_any_origin(),
                p.as_unsafe_any_origin(),
                Int64(numel),
                Int64(ndim),
                sizes,
                dst_strides,
                p_strides,
                seed,
                offset,
            )
