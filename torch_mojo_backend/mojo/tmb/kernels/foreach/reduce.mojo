"""The foreach reductions: `_foreach_norm`, `_foreach_powsum` and
`_foreach_max`, one result per tensor of the list -- ForeachReduceOp.cu's
functors for a list on CUDA's fast route (ord 0, 1, 2, inf), and for the
rest the reductions its slow route runs per tensor (ReduceNormKernel.cu's
NormOps / AbsMinOps for any other order, -inf included). Every element is
first converted to the result dtype (the TensorIterator cast of
`make_reduction`), then to its opmath; for the fast route, whose result
dtype never narrows the input, that first conversion is exact.

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
from std.utils.numerics import isnan, max_or_inf, min_or_neg_inf

from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.foreach.apply import FE_DESC_CAP, FE_THREADS, ForeachDesc
from tmb.kernels.foreach.contract import (
    FE_REDUCE_CHUNK,
    FR_L0,
    FR_L1,
    FR_L2,
    FR_LINF,
    FR_LNEGINF,
    FR_LP,
    FR_MAX,
)
from tmb.kernels.common.pow_math import pow_c99
from tmb.kernels.foreach.functors import foreach_opmath


@always_inline
def fr_is_max[norm: Int]() -> Bool:
    return norm == FR_LINF or norm == FR_MAX


@always_inline
def fr_is_min[norm: Int]() -> Bool:
    return norm == FR_LNEGINF


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
    elif norm == FR_LNEGINF:
        return "lneginf"
    elif norm == FR_LP:
        return "lp"
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
def _min_nan[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """`min_propagate_nan`: a NaN on either side wins."""
    comptime if dt.is_floating_point():
        if isnan(b):
            return b
        if isnan(a):
            return a
    return min(a, b)


@always_inline
def _identity[acc: DType, norm: Int]() -> Scalar[acc]:
    comptime if norm == FR_MAX:
        # -inf, the identity of max: _foreach_max([-inf]) is -inf, as on CPU
        # and in the per-tensor max. Stock CUDA seeds with
        # `std::numeric_limits<T>::lowest()` instead (ForeachReduceOp.cu:77,
        # :105, :127 and the scratch fill at :192) and returns -3.4028e38
        # for a float32 list of -inf.
        return min_or_neg_inf[acc]()
    elif norm == FR_LNEGINF:
        # AbsMinOps' seed, `numeric_limits<acc_t>::infinity()`.
        return max_or_inf[acc]()
    else:
        return 0


@always_inline
def _combine[
    acc: DType, norm: Int
](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
    comptime if fr_is_max[norm]():
        return _max_nan(a, b)
    elif fr_is_min[norm]():
        return _min_nan(a, b)
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
def _pow[acc: DType](a: Scalar[acc], p: Scalar[acc]) -> Scalar[acc]:
    """`compat_pow`: C's pow in the accumulator type."""
    comptime if acc == DType.float64:
        return rebind[Scalar[acc]](
            pow_c99[DType.float64](rebind[Float64](a), rebind[Float64](p))
        )
    elif acc == DType.float32:
        return rebind[Scalar[acc]](
            pow_c99[DType.float32](rebind[Float32](a), rebind[Float32](p))
        )
    else:
        comptime assert False, "an Lp reduction accumulates in a float"


@always_inline
def _element[
    dtype: DType, out_dtype: DType, acc: DType, norm: Int
](x: Scalar[dtype], p: Scalar[acc]) -> Scalar[acc]:
    """One element's contribution, in the accumulator type, after the cast
    to the result dtype."""
    comptime if norm == FR_MAX:
        return x.cast[acc]()
    else:
        var v = x.cast[out_dtype]().cast[acc]()
        comptime if norm == FR_L0:
            return Scalar[acc](1) if v != 0 else Scalar[acc](0)
        elif norm == FR_L2:
            return v * v
        elif norm == FR_LP:
            return _pow(abs(v), p)
        else:
            return abs(v)


@__name(
    t"multi_tensor_reduce_{fr_label[norm]()}_chunks_{dtype}_{acc}_t{FE_THREADS}"
)
def _reduce_chunks_kernel[
    dtype: DType, out_dtype: DType, acc: DType, norm: Int
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    desc_count_arg: Int64,
    partials_addr_arg: Int64,
    p_arg: Float64,
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
    var p = p_arg.cast[acc]()
    var val = _identity[acc, norm]()
    var index = begin + Int(thread_idx.x)
    while index < end:
        val = _combine[acc, norm](
            val,
            _element[dtype, out_dtype, acc, norm](
                values[unsafe_offset=index], p
            ),
        )
        index += FE_THREADS
    var total = _block_combine[acc, norm](val)
    if thread_idx.x == 0:
        _make_ptr[acc](Int(partials_addr_arg))[unsafe_offset=chunk] = total


@__name(
    t"multi_tensor_reduce_{fr_label[norm]()}_finalize_{acc}_{out_dtype}_t{FE_THREADS}"
)
def _reduce_finalize_kernel[
    dtype: DType, acc: DType, out_dtype: DType, norm: Int, root: Bool
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    partials_addr_arg: Int64,
    p_arg: Float64,
    one_element_abs_arg: Int64,
):
    """One block per tensor of the batch: combine its partials, write the
    0-d result (`desc.dst`). A tensor with no chunk gets the identity.

    `one_element_abs`: linalg_vector_norm's one-element shortcut
    (`is_reduce_over_1D_vector`): |x| (x != 0 for ord 0) in the result
    dtype, not the reduction's root of a power."""
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
        elif root and norm == FR_LP:
            var p = p_arg.cast[acc]()
            total = _pow(total, Scalar[acc](1) / p)
        var result = total.cast[out_dtype]()
        comptime if norm != FR_MAX:
            # An empty reduction runs no kernel on CUDA: TensorIterator
            # fills the result with the identity, 0, with no root applied
            # (so ord nan or 1e100 give 0, not pow(0, 1/p)).
            if desc.numel == 0:
                result = 0
            if one_element_abs_arg != 0 and desc.numel == 1:
                var x = _make_ptr[dtype](desc.a)[unsafe_offset=0].cast[
                    out_dtype
                ]()
                comptime if norm == FR_L0:
                    result = Scalar[out_dtype](1) if x != 0 else Scalar[
                        out_dtype
                    ](0)
                else:
                    result = abs(x)
        _make_ptr[out_dtype](desc.dst)[unsafe_offset=0] = result


def foreach_reduce_enqueue[
    dtype: DType, acc: DType, out_dtype: DType, norm: Int, root: Bool
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    p: Float64,
    one_element_abs: Bool,
    ctx: DeviceContext,
) raises:
    """One batch: its chunk partials (`partials_addr` is the batch's slice),
    then its results."""
    if desc_count <= 0:
        return
    if total_chunks > 0:
        _enqueue_cached[_reduce_chunks_kernel[dtype, out_dtype, acc, norm]](
            ctx,
            total_chunks,
            1,
            1,
            FE_THREADS,
            descs,
            Int64(desc_count),
            Int64(partials_addr),
            p,
        )
    _enqueue_cached[_reduce_finalize_kernel[dtype, acc, out_dtype, norm, root]](
        ctx,
        desc_count,
        1,
        1,
        FE_THREADS,
        descs,
        Int64(partials_addr),
        p,
        Int64(1 if one_element_abs else 0),
    )


@always_inline
def foreach_reduce_acc[dtype: DType, out_dtype: DType, norm: Int]() -> DType:
    """The accumulator: the result's opmath for a norm, the dtype for max."""
    comptime if norm == FR_MAX:
        return dtype
    else:
        return foreach_opmath[out_dtype]()
