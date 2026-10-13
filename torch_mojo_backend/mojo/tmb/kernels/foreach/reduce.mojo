"""The foreach reductions: `_foreach_norm` (ord 0, 1, 2, inf),
`_foreach_powsum` (ord 1, 2) and `_foreach_max`, one result per tensor of
the list (ForeachReduceOp.cu).

Two launches per descriptor batch, CUDA's shape: `_reduce_chunks_kernel`
reduces one FE_REDUCE_CHUNK-element chunk per block into a scratch partial
(the chunks of the whole list concatenated, a tensor's partials contiguous),
then `_reduce_finalize_kernel` reduces each tensor's partials in one block
and writes its 0-d result (the root of an L2 norm, the result dtype's
rounding). Accumulation is in the result's opmath type for the norms (float32
for the half types and float32, double for float64), in the tensor's own
dtype for max, with NaN propagating through every max as in
`max_propagate_nan` (SharedReduceOps.h).
"""

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import min
from std.memory import stack_allocation
from std.utils.numerics import isnan, min_or_neg_inf

from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.foreach.apply import FE_DESC_CAP, FE_THREADS, ForeachDesc
from tmb.kernels.foreach.functors import foreach_opmath


# Elements per chunk: CUDA's kChunkSize. Fixed, so the host can size the
# partials scratch from the numels alone.
comptime FE_REDUCE_CHUNK = 65_536

comptime FR_L0 = 0
comptime FR_L1 = 1
comptime FR_L2 = 2
comptime FR_LINF = 3
comptime FR_MAX = 4


@always_inline
def fr_is_max[norm: Int]() -> Bool:
    return norm == FR_LINF or norm == FR_MAX


@always_inline
def fr_label[norm: Int]() -> StaticString:
    comptime if norm == FR_L0:
        return "l0"
    elif norm == FR_L1:
        return "l1"
    elif norm == FR_L2:
        return "l2"
    elif norm == FR_LINF:
        return "linf"
    else:
        return "max"


@always_inline
def _max_nan[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """`max_propagate_nan`: a NaN on either side wins."""
    comptime if dt.is_floating_point():
        if isnan(b):
            return b
        if isnan(a):
            return a
    return max(a, b)


@always_inline
def _identity[acc: DType, norm: Int]() -> Scalar[acc]:
    comptime if norm == FR_MAX:
        return min_or_neg_inf[acc]()
    else:
        return 0


@always_inline
def _combine[
    acc: DType, norm: Int
](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
    comptime if fr_is_max[norm]():
        return _max_nan(a, b)
    else:
        return a + b


@always_inline
def _block_combine[acc: DType, norm: Int](v: Scalar[acc]) -> Scalar[acc]:
    """`v` combined over the block (FE_THREADS, a power of two) through a
    shared-memory tree: float64 and the integer maxima included, which the
    warp-shuffle primitives do not all lower. The total is in thread 0."""
    var sh = stack_allocation[
        FE_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    sh[unsafe_offset=tid] = v
    barrier()
    var stride = FE_THREADS // 2
    while stride > 0:
        if tid < stride:
            sh[unsafe_offset=tid] = _combine[acc, norm](
                sh[unsafe_offset=tid], sh[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    return sh[unsafe_offset=0]


@always_inline
def _element[
    dtype: DType, acc: DType, norm: Int
](x: Scalar[dtype]) -> Scalar[acc]:
    """One element's contribution, in the accumulator type."""
    comptime if norm == FR_MAX:
        return x.cast[acc]()
    else:
        var v = x.cast[acc]()
        comptime if norm == FR_L0:
            return Scalar[acc](1) if v != 0 else Scalar[acc](0)
        elif norm == FR_L2:
            return v * v
        else:
            return abs(v)


@__name(
    t"multi_tensor_reduce_{fr_label[norm]()}_chunks_{dtype}_{acc}_t{FE_THREADS}"
)
def _reduce_chunks_kernel[
    dtype: DType, acc: DType, norm: Int
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    desc_count_arg: Int64,
    partials_addr_arg: Int64,
):
    """One block per chunk of the batch: its partial, at the chunk's index in
    the batch's slice of the scratch."""
    var desc_count = Int(desc_count_arg)
    var chunk = Int(block_idx.x)
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1
    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * FE_REDUCE_CHUNK
    var end = min(begin + FE_REDUCE_CHUNK, desc.numel)
    var values = _make_ptr[dtype](desc.a)
    var val = _identity[acc, norm]()
    var index = begin + Int(thread_idx.x)
    while index < end:
        val = _combine[acc, norm](
            val, _element[dtype, acc, norm](values[unsafe_offset=index])
        )
        index += FE_THREADS
    var total = _block_combine[acc, norm](val)
    if thread_idx.x == 0:
        _make_ptr[acc](Int(partials_addr_arg))[unsafe_offset=chunk] = total


@__name(
    t"multi_tensor_reduce_{fr_label[norm]()}_finalize_{acc}_{out_dtype}_t{FE_THREADS}"
)
def _reduce_finalize_kernel[
    acc: DType, out_dtype: DType, norm: Int, root: Bool
](descs: Array[ForeachDesc, FE_DESC_CAP], partials_addr_arg: Int64,):
    """One block per tensor of the batch: combine its partials, write the
    0-d result (`desc.dst`). A tensor with no chunk gets the identity."""
    var desc_index = Int(block_idx.x)
    var desc = descs[desc_index]
    var begin = 0
    if desc_index != 0:
        begin = descs[desc_index - 1].chunk_end
    var partials = _make_ptr[acc](Int(partials_addr_arg))
    var val = _identity[acc, norm]()
    var index = begin + Int(thread_idx.x)
    while index < desc.chunk_end:
        val = _combine[acc, norm](val, partials[unsafe_offset=index])
        index += FE_THREADS
    var total = _block_combine[acc, norm](val)
    if thread_idx.x == 0:
        comptime if root and norm == FR_L2:
            total = ieee_sqrt(total)
        _make_ptr[out_dtype](desc.dst)[unsafe_offset=0] = total.cast[
            out_dtype
        ]()


def foreach_reduce_enqueue[
    dtype: DType, acc: DType, out_dtype: DType, norm: Int, root: Bool
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    """One batch: its chunk partials (`partials_addr` is the batch's slice),
    then its results."""
    if desc_count <= 0:
        return
    if total_chunks > 0:
        _enqueue_cached[_reduce_chunks_kernel[dtype, acc, norm]](
            ctx,
            total_chunks,
            1,
            1,
            FE_THREADS,
            descs,
            Int64(desc_count),
            Int64(partials_addr),
        )
    _enqueue_cached[_reduce_finalize_kernel[acc, out_dtype, norm, root]](
        ctx,
        desc_count,
        1,
        1,
        FE_THREADS,
        descs,
        Int64(partials_addr),
    )


@always_inline
def foreach_reduce_acc[dtype: DType, out_dtype: DType, norm: Int]() -> DType:
    """The accumulator: the result's opmath for a norm, the dtype for max."""
    comptime if norm == FR_MAX:
        return dtype
    else:
        return foreach_opmath[out_dtype]()
