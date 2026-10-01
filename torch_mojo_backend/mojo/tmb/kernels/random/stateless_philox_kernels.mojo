"""ATen's stateless Philox ops, bit for bit with their CUDA kernels
(aten/src/ATen/native/cuda/PhiloxKeySplit.cu, PhiloxDistribution.cu and
ATen/cuda/StatelessPhilox4x32.cuh at v2.14.0).

A key is a `(seed, offset)` pair of uint64. `philox_4x32(seed, offset)` is the
Philox4x32-10 block cipher with key = seed and counter = (offset, 0): no
generator state, no subsequence, so values never depend on the launch
geometry. Splitting / folding a key draws one block and repacks its four
words as a new key; a distribution draws one block per chunk of 4 outputs
(2 for float64), chunk `c` of key `(s, o)` reading block `(s, o + c)`.
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import fma
from std.sys.info import has_accelerator, has_apple_gpu_accelerator

from tmb.kernels.common.libdevice_port import (
    nv_fast_logf,
    nv_fast_sincosf,
    nv_log,
    nv_sincos,
)
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr, ieee_sqrt
from tmb.kernels.random.philox import U32x2, U32x4, philox4x32_10

comptime BLOCK = 256
comptime _MAX_GRID = 65535

comptime PHILOX_UNIFORM = 0
comptime PHILOX_NORMAL = 1

# 1/2^32 and 2*pi as PhiloxDistribution.cu spells them.
comptime _M_F32 = Float32(2.3283064365386963e-10)
comptime _TWO_PI_F32 = Float32(6.2831853071795864)
comptime _M_F64 = Float64(2.3283064365386963e-10)
comptime _TWO_PI_F64 = Float64(6.2831853071795864)


@always_inline
def stateless_philox(seed: UInt64, offset: UInt64) -> U32x4:
    """`at::cuda::philox_4x32(seed, offset)`: counter (offset, 0, 0)."""
    return philox4x32_10(
        U32x4(
            offset.cast[DType.uint32](),
            (offset >> 32).cast[DType.uint32](),
            0,
            0,
        ),
        U32x2(seed.cast[DType.uint32](), (seed >> 32).cast[DType.uint32]()),
    )


@always_inline
def _derive_key(r: U32x4, dst: Pointer[UInt64, MutAnyOrigin], at: Int):
    """`philox_derive_key`: words (x, y) -> seed, (z, w) -> offset."""
    var r64 = r.cast[DType.uint64]()
    dst[unsafe_offset=at] = r64[0] | (r64[1] << 32)
    dst[unsafe_offset=at + 1] = r64[2] | (r64[3] << 32)


@__name("philox_key_split_u64")
def _key_split_kernel(
    keys: Pointer[UInt64, MutAnyOrigin],
    dst: Pointer[UInt64, MutAnyOrigin],
    num_keys: Int64,
    num_splits: Int64,
):
    var total = Int(num_keys) * Int(num_splits)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nk = Int(num_keys)
    while tid < total:
        var split = tid // nk
        var key = tid % nk
        var r = stateless_philox(
            keys[unsafe_offset=key * 2],
            keys[unsafe_offset=key * 2 + 1] + UInt64(split),
        )
        _derive_key(r, dst, (split * nk + key) * 2)
        tid += stride


@__name("philox_key_fold_in_u64")
def _key_fold_in_kernel(
    keys: Pointer[UInt64, MutAnyOrigin],
    dst: Pointer[UInt64, MutAnyOrigin],
    num_keys: Int64,
    data: UInt64,
    data_ptr: Pointer[UInt64, MutAnyOrigin],
    data_on_device: Int32,
):
    # The Tensor overload reads `data` on the device (no host sync).
    var fold = data_ptr[unsafe_offset=0] if data_on_device != 0 else data
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while tid < Int(num_keys):
        var r = stateless_philox(
            keys[unsafe_offset=tid * 2], keys[unsafe_offset=tid * 2 + 1] + fold
        )
        _derive_key(r, dst, tid * 2)
        tid += stride


@always_inline
def philox_acc[dtype: DType]() -> DType:
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def philox_epc[dtype: DType]() -> Int:
    """Elements per Philox call: 2 for float64, 4 otherwise."""
    comptime if dtype == DType.float64:
        return 2
    else:
        return 4


@always_inline
def _digits[dtype: DType]() -> Int:
    """`std::numeric_limits<T>::digits`."""
    comptime if dtype == DType.float64:
        return 53
    elif dtype == DType.float32:
        return 24
    elif dtype == DType.float16:
        return 11
    else:
        return 8


@always_inline
def _box_muller_float(r: U32x4) -> SIMD[DType.float32, 4]:
    var u = fma(
        r.cast[DType.float32](),
        SIMD[DType.float32, 4](_M_F32),
        SIMD[DType.float32, 4](_M_F32 * Float32(0.5)),
    )
    var radius1 = ieee_sqrt(Float32(-2.0) * nv_fast_logf(u[0]))
    var radius2 = ieee_sqrt(Float32(-2.0) * nv_fast_logf(u[2]))
    var sc1 = nv_fast_sincosf(_TWO_PI_F32 * u[1])
    var sc2 = nv_fast_sincosf(_TWO_PI_F32 * u[3])
    return SIMD[DType.float32, 4](
        radius1 * sc1[1], radius1 * sc1[0], radius2 * sc2[1], radius2 * sc2[0]
    )


@always_inline
def _box_muller_double(r: U32x4) -> SIMD[DType.float64, 2]:
    comptime HALF_MM = _M_F64 * _M_F64 * Float64(0.5)
    var d = r.cast[DType.float64]()
    # `x * M * M + M * M * 0.5`: nvcc rounds x * M, then fuses the second
    # product with the constant.
    var u1 = fma(d[0], _M_F64, fma(d[1] * _M_F64, _M_F64, HALF_MM))
    var u2 = fma(d[2], _M_F64, fma(d[3] * _M_F64, _M_F64, HALF_MM))
    var radius = ieee_sqrt(Float64(-2.0) * nv_log(u1))
    var sc = nv_sincos(_TWO_PI_F64 * u2)
    return SIMD[DType.float64, 2](radius * sc[1], radius * sc[0])


@always_inline
def philox_draw[
    dtype: DType, KIND: Int
](
    r: U32x4, p0: Scalar[philox_acc[dtype]()], p1: Scalar[philox_acc[dtype]()]
) -> SIMD[dtype, philox_epc[dtype]()]:
    """One Philox block -> EPC outputs. Uniform: p0 = from, p1 = to - from
    (both already rounded through the output dtype, as ATen's scalar_t
    arithmetic does). Normal: p0 = mean, p1 = std."""
    comptime ACC = philox_acc[dtype]()
    comptime N = philox_epc[dtype]()
    comptime if KIND == PHILOX_UNIFORM:
        # transformation::uniform_real: (val & MASK) * 2^-digits, then
        # x * (to - from) + from (an fma once nvcc contracts it).
        comptime if dtype == DType.float64:
            var r64 = r.cast[DType.uint64]()
            comptime MASK = (UInt64(1) << 53) - 1
            var v = SIMD[DType.uint64, 2](
                ((r64[0] << 32) | r64[1]) & MASK,
                ((r64[2] << 32) | r64[3]) & MASK,
            )
            var x = v.cast[DType.float64]() * SIMD[DType.float64, 2](
                Float64(1.0) / Float64(UInt64(1) << 53)
            )
            return rebind[SIMD[dtype, N]](
                fma(
                    x,
                    SIMD[DType.float64, 2](rebind[Float64](p1)),
                    SIMD[DType.float64, 2](rebind[Float64](p0)),
                )
            )
        else:
            comptime D = _digits[dtype]()
            comptime MASK = (UInt32(1) << UInt32(D)) - 1
            var x = (r & U32x4(MASK)).cast[DType.float32]() * SIMD[
                DType.float32, 4
            ](Float32(1.0) / Float32(1 << D))
            return rebind[SIMD[dtype, N]](
                fma(
                    x,
                    SIMD[DType.float32, 4](rebind[Float32](p1)),
                    SIMD[DType.float32, 4](rebind[Float32](p0)),
                ).cast[dtype]()
            )
    else:
        comptime if dtype == DType.float64:
            return rebind[SIMD[dtype, N]](
                fma(
                    _box_muller_double(r),
                    SIMD[DType.float64, 2](rebind[Float64](p1)),
                    SIMD[DType.float64, 2](rebind[Float64](p0)),
                )
            )
        else:
            return rebind[SIMD[dtype, N]](
                fma(
                    _box_muller_float(r),
                    SIMD[DType.float32, 4](rebind[Float32](p1)),
                    SIMD[DType.float32, 4](rebind[Float32](p0)),
                ).cast[dtype]()
            )


@__name(t"philox_stateless_dist{KIND}_{dtype}")
def _dist_kernel[
    dtype: DType, KIND: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    keys: Pointer[UInt64, MutAnyOrigin],
    num_keys: Int64,
    elems_per_key: Int64,
    p0: Scalar[philox_acc[dtype]()],
    p1: Scalar[philox_acc[dtype]()],
):
    """One thread per (key, chunk); `dst` is contiguous, key `k` owns
    elements [k * elems_per_key, (k + 1) * elems_per_key)."""
    comptime N = philox_epc[dtype]()
    var epk = Int(elems_per_key)
    var chunks = (epk + N - 1) // N
    var total = Int(num_keys) * chunks
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while tid < total:
        var key = tid // chunks
        var chunk = tid % chunks
        var vals = philox_draw[dtype, KIND](
            stateless_philox(
                keys[unsafe_offset=key * 2],
                keys[unsafe_offset=key * 2 + 1] + UInt64(chunk),
            ),
            p0,
            p1,
        )
        var base = key * epk + chunk * N
        comptime for j in range(N):
            if chunk * N + j < epk:
                dst[unsafe_offset=base + j] = vals[j]
        tid += stride


@always_inline
def _grid(n: Int) -> Int:
    return max(1, min((n + BLOCK - 1) // BLOCK, _MAX_GRID))


def enqueue_key_split(
    ctx: DeviceContext,
    keys_addr: Int,
    out_addr: Int,
    num_keys: Int,
    num_splits: Int,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_key_split_kernel](
            ctx,
            _grid(num_keys * num_splits),
            1,
            1,
            BLOCK,
            _make_ptr[DType.uint64](keys_addr).as_unsafe_any_origin(),
            _make_ptr[DType.uint64](out_addr).as_unsafe_any_origin(),
            Int64(num_keys),
            Int64(num_splits),
        )


def enqueue_key_fold_in(
    ctx: DeviceContext,
    keys_addr: Int,
    out_addr: Int,
    num_keys: Int,
    data: UInt64,
    data_addr: Int,
) raises:
    """`data_addr` != 0: read the fold value there, on the device."""
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_key_fold_in_kernel](
            ctx,
            _grid(num_keys),
            1,
            1,
            BLOCK,
            _make_ptr[DType.uint64](keys_addr).as_unsafe_any_origin(),
            _make_ptr[DType.uint64](out_addr).as_unsafe_any_origin(),
            Int64(num_keys),
            data,
            _make_ptr[DType.uint64](
                data_addr if data_addr != 0 else keys_addr
            ).as_unsafe_any_origin(),
            Int32(1) if data_addr != 0 else Int32(0),
        )


def enqueue_stateless_dist[
    dtype: DType, KIND: Int
](
    ctx: DeviceContext,
    out_addr: Int,
    keys_addr: Int,
    num_keys: Int,
    elems_per_key: Int,
    p0: Float64,
    p1: Float64,
) raises:
    """`p0`/`p1` arrive in the accumulation type's range already rounded the
    way ATen rounds them (the caller's job); they are narrowed here."""
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            comptime N = philox_epc[dtype]()
            comptime ACC = philox_acc[dtype]()
            var chunks = (elems_per_key + N - 1) // N
            _enqueue_cached[_dist_kernel[dtype, KIND]](
                ctx,
                _grid(num_keys * chunks),
                1,
                1,
                BLOCK,
                _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint64](keys_addr).as_unsafe_any_origin(),
                Int64(num_keys),
                Int64(elems_per_key),
                p0.cast[ACC](),
                p1.cast[ACC](),
            )
