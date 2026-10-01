"""Multi-class and multi-label margin loss kernels, forward and backward.

Ported from torch's CUDA kernels (aten/src/ATen/native/cuda/
MultiMarginLoss.cu and MultiLabelMarginCriterion.cu at v2.14.0), one block
of 128 threads per sample (row of a contiguous `[nframe, dim]` input), with
their rounding points:

  * margin terms `margin - x[target] + x[i]` are formed in the input dtype
    (two half roundings for a half input), squared / weighted in it too, and
    accumulated in float32 for half inputs (`acc_type`), float64 for float64;
  * the multi-margin per-thread partials are summed by thread 0 in thread
    order, the multi-label ones by a block reduction;
  * every per-sample value is rounded once to the input dtype; the batch
    reduction of the forward is the op's `sum` over those (the CUDA code
    calls `at::sum_out` too).

Targets outside `[0, dim)` are a device-side assert (multi-margin) or an
unchecked write (multi-label) in CUDA; here a multi-margin sample with such
a target yields NaN and leaves its gradient row NaN, and a multi-label target
past `dim` is skipped.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.memory import stack_allocation
from std.utils.numerics import nan

from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.common.block_reduce import block_sum

comptime MARGIN_THREADS = 128


@always_inline
def _acc[dtype: DType]() -> DType:
    """torch's `acc_type<scalar_t, /*is_cuda=*/true>`."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@__name(t"multi_margin_forward_{dtype}")
def _multi_margin_forward_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    w_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    has_w_arg: Int64,
    nframe_arg: Int64,
    dim_arg: Int64,
    p_arg: Int64,
    size_average_arg: Int64,
    margin: Scalar[dtype],
):
    """MultiMarginLoss_forward_kernel: one block per sample."""
    comptime acc_t = _acc[dtype]()
    var buffer = stack_allocation[
        MARGIN_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var dim = Int(dim_arg)
    var k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = k * dim
    var target_k = Int(t_ptr[unsafe_offset=k])
    var in_range = target_k >= 0 and target_k < dim
    var acc = Scalar[acc_t](0)
    if in_range:
        var x_t = in_ptr[unsafe_offset=row + target_k]
        var i = tid
        while i < dim:
            if i != target_k:
                var z = margin - x_t + in_ptr[unsafe_offset=row + i]
                if z > 0:
                    var h = z if Int(p_arg) == 1 else z * z
                    if Int(has_w_arg) != 0:
                        h *= w_ptr[unsafe_offset=target_k]
                    acc += h.cast[acc_t]()
            i += MARGIN_THREADS
    buffer[unsafe_offset=tid] = acc
    barrier()
    if tid == 0:
        var total = Scalar[acc_t](0)
        for j in range(MARGIN_THREADS):
            total += buffer[unsafe_offset=j]
        var denom = Int(nframe_arg) * dim if Int(size_average_arg) != 0 else dim
        if in_range:
            out_ptr[unsafe_offset=k] = (total / Scalar[acc_t](denom)).cast[
                dtype
            ]()
        else:
            out_ptr[unsafe_offset=k] = nan[dtype]()


@__name(t"multi_margin_backward_{dtype}")
def _multi_margin_backward_kernel[
    dtype: DType
](
    gi_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    w_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    has_w_arg: Int64,
    nframe_arg: Int64,
    dim_arg: Int64,
    p_arg: Int64,
    size_average_arg: Int64,
    reduce_arg: Int64,
    margin: Scalar[dtype],
):
    """MultiMarginLoss_backward_kernel: one block per sample."""
    comptime acc_t = _acc[dtype]()
    var buffer = stack_allocation[
        MARGIN_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var dim = Int(dim_arg)
    var k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = k * dim
    var reduce = Int(reduce_arg) != 0
    var target_k = Int(t_ptr[unsafe_offset=k])
    var in_range = target_k >= 0 and target_k < dim
    var go = go_ptr[unsafe_offset=0 if reduce else k]
    var denom = (
        Int(nframe_arg) * dim if Int(size_average_arg) != 0 and reduce else dim
    )
    var g = Scalar[acc_t](1) / Scalar[acc_t](denom)
    var acc = Scalar[acc_t](0)
    if in_range:
        var x_t = in_ptr[unsafe_offset=row + target_k]
        var i = tid
        while i < dim:
            if i != target_k:
                var z = margin - x_t + in_ptr[unsafe_offset=row + i]
                if z > 0:
                    var h = g if Int(p_arg) == 1 else 2 * g * z.cast[acc_t]()
                    if Int(has_w_arg) != 0:
                        h *= w_ptr[unsafe_offset=target_k].cast[acc_t]()
                    var hr = h.cast[dtype]()
                    acc -= hr.cast[acc_t]()
                    gi_ptr[unsafe_offset=row + i] = hr
                else:
                    gi_ptr[unsafe_offset=row + i] = 0
            i += MARGIN_THREADS
    buffer[unsafe_offset=tid] = acc
    barrier()
    if tid == 0:
        var total = Scalar[acc_t](0)
        for j in range(MARGIN_THREADS):
            total += buffer[unsafe_offset=j]
        if in_range:
            gi_ptr[unsafe_offset=row + target_k] = total.cast[dtype]()
    barrier()
    var i = tid
    while i < dim:
        if in_range:
            gi_ptr[unsafe_offset=row + i] = gi_ptr[unsafe_offset=row + i] * go
        else:
            gi_ptr[unsafe_offset=row + i] = nan[dtype]()
        i += MARGIN_THREADS


def multi_margin_forward[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    nframe: Int,
    dim: Int,
    p: Int,
    size_average: Bool,
    margin: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_multi_margin_forward_kernel[dtype]](
        ctx,
        nframe,
        1,
        1,
        MARGIN_THREADS,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](weight_addr).as_unsafe_any_origin(),
        Int64(1 if weight_addr != 0 else 0),
        Int64(nframe),
        Int64(dim),
        Int64(p),
        Int64(1 if size_average else 0),
        margin,
    )


def multi_margin_backward[
    dtype: DType
](
    gi_addr: Int,
    go_addr: Int,
    in_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    nframe: Int,
    dim: Int,
    p: Int,
    size_average: Bool,
    reduce: Bool,
    margin: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_multi_margin_backward_kernel[dtype]](
        ctx,
        nframe,
        1,
        1,
        MARGIN_THREADS,
        _make_ptr[dtype](gi_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](weight_addr).as_unsafe_any_origin(),
        Int64(1 if weight_addr != 0 else 0),
        Int64(nframe),
        Int64(dim),
        Int64(p),
        Int64(1 if size_average else 0),
        Int64(1 if reduce else 0),
        margin,
    )


@__name(t"multilabel_margin_forward_{dtype}")
def _multilabel_forward_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    is_target_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    nframe_arg: Int64,
    dim_arg: Int64,
    size_average_arg: Int64,
):
    """multilabel_margin_loss_forward_kernel: one block per sample, also
    writing the sample's 0/1 `is_target` row."""
    comptime acc_t = _acc[dtype]()
    var dim = Int(dim_arg)
    var k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = k * dim
    var d = tid
    while d < dim:
        is_target_ptr[unsafe_offset=row + d] = 0
        d += MARGIN_THREADS
    barrier()
    if tid == 0:
        for dt in range(dim):
            var target_idx = Int(t_ptr[unsafe_offset=row + dt])
            if target_idx < 0:
                break
            if target_idx < dim:
                is_target_ptr[unsafe_offset=row + target_idx] = 1
    barrier()
    var sum = Scalar[acc_t](0)
    for dt in range(dim):
        var target_idx = Int(t_ptr[unsafe_offset=row + dt])
        if target_idx < 0:
            break
        if target_idx >= dim:
            continue
        var x_t = in_ptr[unsafe_offset=row + target_idx]
        var j = tid
        while j < dim:
            if is_target_ptr[unsafe_offset=row + j] == 0:
                var z = Scalar[dtype](1) - x_t + in_ptr[unsafe_offset=row + j]
                if z > 0:
                    sum += z.cast[acc_t]()
            j += MARGIN_THREADS
    var total = block_sum[acc_t, MARGIN_THREADS](sum)
    if tid == 0:
        var per_dim = total / Scalar[acc_t](dim)
        if Int(size_average_arg) != 0:
            per_dim = per_dim / Scalar[acc_t](Int(nframe_arg))
        out_ptr[unsafe_offset=k] = per_dim.cast[dtype]()


@__name(t"multilabel_margin_backward_{dtype}")
def _multilabel_backward_kernel[
    dtype: DType
](
    gi_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    is_target_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    dim_arg: Int64,
    reduce_arg: Int64,
    g: Scalar[dtype],
):
    """multilabel_margin_loss_backward_kernel: one block per sample; `g` is
    the gain `1 / (nframe * dim)` or `1 / dim`, rounded to the dtype on the
    host as CUDA's `static_cast<scalar_t>(1. / ...)` is."""
    comptime acc_t = _acc[dtype]()
    var dim = Int(dim_arg)
    var k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = k * dim
    var go = go_ptr[unsafe_offset=k if Int(reduce_arg) == 0 else 0]
    var d = tid
    while d < dim:
        gi_ptr[unsafe_offset=row + d] = 0
        d += MARGIN_THREADS
    barrier()
    for dt in range(dim):
        var target_idx = Int(t_ptr[unsafe_offset=row + dt])
        if target_idx < 0:
            break
        if target_idx >= dim:
            continue
        var x_t = in_ptr[unsafe_offset=row + target_idx]
        var sum = Scalar[acc_t](0)
        var j = tid
        while j < dim:
            if is_target_ptr[unsafe_offset=row + j] == 0:
                var z = Scalar[dtype](1) - x_t + in_ptr[unsafe_offset=row + j]
                if z > 0:
                    sum -= g.cast[acc_t]()
                    gi_ptr[unsafe_offset=row + j] = (
                        gi_ptr[unsafe_offset=row + j] + g
                    )
            j += MARGIN_THREADS
        barrier()
        var total = block_sum[acc_t, MARGIN_THREADS](sum)
        if tid == 0:
            gi_ptr[unsafe_offset=row + target_idx] = (
                gi_ptr[unsafe_offset=row + target_idx] + total.cast[dtype]()
            )
        barrier()
    d = tid
    while d < dim:
        gi_ptr[unsafe_offset=row + d] = gi_ptr[unsafe_offset=row + d] * go
        d += MARGIN_THREADS


def multilabel_margin_forward[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    target_addr: Int,
    is_target_addr: Int,
    nframe: Int,
    dim: Int,
    size_average: Bool,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_multilabel_forward_kernel[dtype]](
        ctx,
        nframe,
        1,
        1,
        MARGIN_THREADS,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](is_target_addr).as_unsafe_any_origin(),
        Int64(nframe),
        Int64(dim),
        Int64(1 if size_average else 0),
    )


def multilabel_margin_backward[
    dtype: DType
](
    gi_addr: Int,
    go_addr: Int,
    in_addr: Int,
    target_addr: Int,
    is_target_addr: Int,
    nframe: Int,
    dim: Int,
    reduce: Bool,
    g: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_multilabel_backward_kernel[dtype]](
        ctx,
        nframe,
        1,
        1,
        MARGIN_THREADS,
        _make_ptr[dtype](gi_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](is_target_addr).as_unsafe_any_origin(),
        Int64(dim),
        Int64(1 if reduce else 0),
        g,
    )
