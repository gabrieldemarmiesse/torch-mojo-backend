"""`aten::native_dropout` / `native_dropout_backward`, bit-identical to
aten/src/ATen/native/cuda/Dropout.cu.

Forward is `fused_dropout_kernel_vec<VEC>` when the tensors are dense and the
input pointer allows a VEC-wide access (flat memory index, one thread per VEC
consecutive elements, RAND_SIZE = ceil(VEC / 4) fresh `curand_uniform4` per
iteration), else `fused_dropout_kernel` (UNROLL = 4 grid-stride over the
logical index, offsets through each tensor's strides). Keep probability
`p = float(1 - p_drop)`, `scale = float(1.0 / double(p))`, element
`(x * float(rand < p)) * scale`, all in the accumulation type (float, or double
for a float64 tensor). Geometry and the counter reservation come from the
caller (tmb/ops/random.mojo).
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    is_apple_gpu,
)

from tmb.kernels.random.philox import (
    U32x4,
    curand4,
    curand_ctr,
    curand_key,
    curand_uniform4,
    philox4x32_10,
)
from tmb.kernels.common.op_utils import (
    _enqueue_cached,
    _fill_blocks,
    _make_ptr,
    FILL_THREADS,
)
from tmb.kernels.random.rng_metadata import I64x8

comptime BLOCK = 256


@always_inline
def _philox4x32_10(counter: UInt64, seed: UInt64) -> U32x4:
    """Philox of the 128-bit counter `(counter, 0)`: nn' fused
    softmax-dropout keys its stream this way."""
    return philox4x32_10(
        U32x4(
            counter.cast[DType.uint32](),
            (counter >> 32).cast[DType.uint32](),
            0,
            0,
        ),
        curand_key(seed),
    )


@always_inline
def _acc[dtype: DType]() -> DType:
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def _scale_of[ACC: DType](p: Scalar[ACC]) -> Scalar[ACC]:
    """`accscalar_t scale = 1.0 / p`: a double division, then narrowed."""
    comptime if is_apple_gpu():
        # Metal has no float64 arithmetic. Its supported dropout dtypes all
        # accumulate in float32; keep the reciprocal in that type here.
        return Scalar[ACC](1.0) / p
    else:
        return (Float64(1.0) / p.cast[DType.float64]()).cast[ACC]()


@always_inline
def _offset_of(li: Int, sizes: I64x8, strides: I64x8, ndim: Int) -> Int:
    var rem = li
    var off = 0
    for d in range(ndim):
        var s = Int(sizes[d])
        off += (rem % s) * Int(strides[d])
        rem //= s
    return off


@__name(t"dropout_philox_vec{VEC}_{dtype}")
def _dropout_vec_kernel[
    dtype: DType, VEC: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    mask: Pointer[Scalar[DType.bool], MutAnyOrigin],
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    numel_arg: Int64,
    p: Scalar[_acc[dtype]()],
    seed: UInt64,
    offset: UInt64,
):
    comptime ACC = _acc[dtype]()
    comptime RAND_SIZE = (VEC + 3) // 4
    var numel = Int(numel_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var ctr = curand_ctr(offset, UInt64(idx))
    var key = curand_key(seed)
    var scale = _scale_of[ACC](p)
    var stride = Int(grid_dim.x) * Int(block_dim.x) * VEC
    var linear = idx * VEC
    var k: UInt64 = 0
    while linear < numel:
        var keep = SIMD[DType.bool, 4 * RAND_SIZE]()

        comptime for jj in range(RAND_SIZE):
            var r = curand_uniform4(curand4(ctr, key, k))
            k += 1
            var m = r.cast[ACC]().lt(SIMD[ACC, 4](p))

            comptime for ii in range(4):
                keep[jj * 4 + ii] = m[ii]

        comptime for e in range(VEC):
            var m = keep[e].cast[ACC]()
            var x = inp[unsafe_offset=linear + e].cast[ACC]()
            dst[unsafe_offset=linear + e] = ((x * m) * scale).cast[dtype]()
            mask[unsafe_offset=linear + e] = keep[e]
        linear += stride


@__name(t"dropout_philox_strided_{dtype}")
def _dropout_strided_kernel[
    dtype: DType
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    mask: Pointer[Scalar[DType.bool], MutAnyOrigin],
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    numel_arg: Int64,
    ndim_arg: Int64,
    sizes: I64x8,
    in_strides: I64x8,
    out_strides: I64x8,
    p: Scalar[_acc[dtype]()],
    seed: UInt64,
    offset: UInt64,
):
    comptime ACC = _acc[dtype]()
    var numel = Int(numel_arg)
    var ndim = Int(ndim_arg)
    var total = Int(grid_dim.x) * Int(block_dim.x)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var ctr = curand_ctr(offset, UInt64(idx))
    var key = curand_key(seed)
    var scale = _scale_of[ACC](p)
    var rounded = ((numel - 1) // (total * 4) + 1) * total * 4
    var linear = idx
    var k: UInt64 = 0
    while linear < rounded:
        var r = curand_uniform4(curand4(ctr, key, k))
        k += 1
        var keep = r.cast[ACC]().lt(SIMD[ACC, 4](p))

        comptime for ii in range(4):
            var li = linear + total * ii
            if li < numel:
                var ai = _offset_of(li, sizes, in_strides, ndim)
                var bi = _offset_of(li, sizes, out_strides, ndim)
                var m = keep[ii].cast[ACC]()
                var x = inp[unsafe_offset=ai].cast[ACC]()
                dst[unsafe_offset=bi] = ((x * m) * scale).cast[dtype]()
                mask[unsafe_offset=bi] = keep[ii]
        linear += total * 4


@__name(t"dropout_backward_masked_scale_{dtype}")
def _dropout_backward_kernel[
    dtype: DType
](
    grad_input: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], MutAnyOrigin],
    mask: Pointer[Scalar[DType.bool], MutAnyOrigin],
    numel_arg: Int64,
    scale: Scalar[_acc[dtype]()],
):
    comptime ACC = _acc[dtype]()
    var numel = Int(numel_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < numel:
        var m = mask[unsafe_offset=i].cast[ACC]()
        var g = grad[unsafe_offset=i].cast[ACC]()
        grad_input[unsafe_offset=i] = ((m * g) * scale).cast[dtype]()
        i += stride


def enqueue_native_dropout[
    dtype: DType
](
    ctx: DeviceContext,
    out_addr: Int,
    mask_addr: Int,
    in_addr: Int,
    numel: Int,
    ndim: Int,
    sizes: I64x8,
    in_strides: I64x8,
    out_strides: I64x8,
    vec: Int,
    grid: Int,
    keep_p: Float64,
    seed: UInt64,
    offset: UInt64,
) raises:
    comptime ACC = _acc[dtype]()
    if numel <= 0:
        return
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            var dst = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
            var mask = _make_ptr[DType.bool](mask_addr).as_unsafe_any_origin()
            var inp = _make_ptr[dtype](in_addr).as_unsafe_any_origin()
            var p = keep_p.cast[ACC]()
            if vec == 8:
                _enqueue_cached[_dropout_vec_kernel[dtype, 8]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst,
                    mask,
                    inp,
                    Int64(numel),
                    p,
                    seed,
                    offset,
                )
            elif vec == 4:
                _enqueue_cached[_dropout_vec_kernel[dtype, 4]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst,
                    mask,
                    inp,
                    Int64(numel),
                    p,
                    seed,
                    offset,
                )
            elif vec == 2:
                _enqueue_cached[_dropout_vec_kernel[dtype, 2]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst,
                    mask,
                    inp,
                    Int64(numel),
                    p,
                    seed,
                    offset,
                )
            else:
                _enqueue_cached[_dropout_strided_kernel[dtype]](
                    ctx,
                    grid,
                    1,
                    1,
                    BLOCK,
                    dst,
                    mask,
                    inp,
                    Int64(numel),
                    Int64(ndim),
                    sizes,
                    in_strides,
                    out_strides,
                    p,
                    seed,
                    offset,
                )


def enqueue_native_dropout_backward[
    dtype: DType
](
    ctx: DeviceContext,
    grad_input_addr: Int,
    grad_addr: Int,
    mask_addr: Int,
    numel: Int,
    scale: Float64,
) raises:
    comptime ACC = _acc[dtype]()
    if numel <= 0:
        return
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            _enqueue_cached[_dropout_backward_kernel[dtype]](
                ctx,
                _fill_blocks(numel),
                1,
                1,
                FILL_THREADS,
                _make_ptr[dtype](grad_input_addr).as_unsafe_any_origin(),
                _make_ptr[dtype](grad_addr).as_unsafe_any_origin(),
                _make_ptr[DType.bool](mask_addr).as_unsafe_any_origin(),
                Int64(numel),
                scale.cast[ACC](),
            )
