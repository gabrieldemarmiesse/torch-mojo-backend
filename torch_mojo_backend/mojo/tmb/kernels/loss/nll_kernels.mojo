"""NLL loss kernels for every float dtype, with class weights, byte or long
targets, the 1-D (no batch) form and the spatial (`nll_loss2d`) form.

One addressing for all of them: the input is a contiguous
`[batch, classes, map]` block (`map` = 1 for `nll_loss`, H*W for
`nll_loss2d`), the target a contiguous `[batch, map]` one, so the logit of
sample `b`, position `s` and class `t` is `input[(b * classes + t) * map + s]`.

Ported from torch's CUDA kernels (aten/src/ATen/native/cuda/Loss.cu and
NLLLoss2d.cu at v2.14.0), including their rounding points:

  * the per-element loss `-w[t] * x` is formed in the input dtype (a half
    product rounds to half) and only then accumulated, in float32 for half
    inputs (`acc_type`), float64 for float64;
  * `nll_loss`'s reduced form is ONE block of `nll_loss_threads(N)` threads,
    each summing a strided slice of the rows, then a shared-memory tree, then
    one rounding to the input dtype (`sum / total_weight` for the mean, so a
    zero total weight gives NaN);
  * `nll_loss2d`'s reduced form is per-block partials (`blocks_per_sample`
    blocks of 128 threads per sample) rounded to the input dtype, which CUDA
    then adds with `atomicAdd` IN THE INPUT DTYPE, followed by a separate
    `output /= total_weight`. The partials here are added by one thread in
    block order instead: the same rounding points, deterministically;
  * the 1-D input: mean does NOT divide (`-x[t]`, NaN when `w[t] == 0`).

Targets outside `[0, classes)` (other than `ignore_index`) are a device-side
assert in CUDA. There is no asynchronous device assert here that would not
poison the shared context, so such a row contributes nothing (0 loss, 0
weight, untouched gradient), as the f32 fast path in entry.mojo does.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import log2
from std.memory import stack_allocation
from std.utils.numerics import nan
from std.utils.coord import Coord

from tmb.kernels.common.block_reduce import block_sum
from tmb.kernels.common.op_utils import (
    _enqueue_cached,
    _make_ptr,
    _parallel_for_dt,
)

# Shared-memory capacity of the one-block reduction: `nll_loss_threads` is
# clamped to [32, 1024].
comptime NLL_MAX_THREADS = 1024
# ATen's CUDA_NUM_THREADS (ATen/cuda/detail/KernelUtils.h).
comptime NLL2D_THREADS = 128


@always_inline
def _acc[dtype: DType]() -> DType:
    """torch's `acc_type<scalar_t, /*is_cuda=*/true>`."""
    return DType.float64 if dtype == DType.float64 else DType.float32


# Params tuple of the NLL ops (tmb/ops/loss.mojo fills it in this order).
comptime P_BATCH = 0
comptime P_CLASSES = 1
comptime P_MAP = 2
comptime P_REDUCTION = 3  # 0 none, 1 mean, 2 sum
comptime P_IGNORE = 4
comptime P_ONE_D = 5  # 1: a 1-D input (one sample, no batch dim)
comptime P_SPATIAL = 6  # 1: nll_loss2d's reduction scheme
comptime P_LEN = 7


@always_inline
def _valid(t: Int, ignore_index: Int, classes: Int) -> Bool:
    return t != ignore_index and t >= 0 and t < classes


def nll_forward_none[
    dtype: DType, tdtype: DType
](
    out_addr: Int,
    in_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    batch: Int,
    classes: Int,
    map: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    """`reduction='none'`: `out[b, s] = -w[t] * input[b, t, s]`, 0 when
    ignored (Loss.cu nll_loss_forward_no_reduce_cuda_kernel /
    NLLLoss2d.cu nll_loss2d_forward_no_reduce_kernel)."""
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var t_ptr = _make_ptr[tdtype](target_addr)
    var w_ptr = _make_ptr[dtype](weight_addr)
    var has_w = weight_addr != 0

    @always_inline
    @__parameter
    @__copy_capture(
        out_ptr, in_ptr, t_ptr, w_ptr, has_w, classes, map, ignore_index
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var b = i // map
        var s = i - b * map
        var t = Int(t_ptr[unsafe_offset=i])
        var loss = Scalar[dtype](0)
        if _valid(t, ignore_index, classes):
            var w = w_ptr[unsafe_offset=t] if has_w else Scalar[dtype](1)
            loss = -w * in_ptr[unsafe_offset=(b * classes + t) * map + s]
        out_ptr[unsafe_offset=i] = loss

    _parallel_for_dt[dtype, func](batch * map, ctx)


@__name(t"nll_forward_reduce_{dtype}_{tdtype}")
def _nll_reduce_kernel[
    dtype: DType, tdtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    tw_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    w_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    has_w_arg: Int64,
    nframe_arg: Int64,
    classes_arg: Int64,
    mean_arg: Int64,
    ignore_arg: Int64,
    one_d_arg: Int64,
):
    """Loss.cu nll_loss_forward_reduce_cuda_kernel_{1d,2d}: one block."""
    comptime acc_t = _acc[dtype]()
    var has_w = Int(has_w_arg) != 0
    var nframe = Int(nframe_arg)
    var classes = Int(classes_arg)
    var mean = Int(mean_arg) != 0
    var ignore_index = Int(ignore_arg)
    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)
    if Int(one_d_arg) != 0:
        if tid == 0:
            var t = Int(t_ptr[unsafe_offset=0])
            if _valid(t, ignore_index, classes):
                var w = w_ptr[unsafe_offset=t] if has_w else Scalar[dtype](1)
                tw_ptr[unsafe_offset=0] = w
                if mean:
                    # Normalizing a zero weight gives NaN.
                    if w == 0:
                        out_ptr[unsafe_offset=0] = nan[dtype]()
                    else:
                        out_ptr[unsafe_offset=0] = -in_ptr[unsafe_offset=t]
                else:
                    out_ptr[unsafe_offset=0] = -w * in_ptr[unsafe_offset=t]
            else:
                out_ptr[unsafe_offset=0] = 0
                tw_ptr[unsafe_offset=0] = 0
        return
    var sh_in = stack_allocation[
        NLL_MAX_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var sh_w = stack_allocation[
        NLL_MAX_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var acc_in = Scalar[acc_t](0)
    var acc_w = Scalar[acc_t](0)
    var i = tid
    while i < nframe:
        var t = Int(t_ptr[unsafe_offset=i])
        if _valid(t, ignore_index, classes):
            var w = w_ptr[unsafe_offset=t] if has_w else Scalar[dtype](1)
            acc_in -= (in_ptr[unsafe_offset=i * classes + t] * w).cast[acc_t]()
            acc_w += w.cast[acc_t]()
        i += nthreads
    sh_in[unsafe_offset=tid] = acc_in
    sh_w[unsafe_offset=tid] = acc_w
    barrier()
    var stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            sh_in[unsafe_offset=tid] += sh_in[unsafe_offset=tid + stride]
            sh_w[unsafe_offset=tid] += sh_w[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        var total_in = sh_in[unsafe_offset=0]
        var total_w = sh_w[unsafe_offset=0]
        tw_ptr[unsafe_offset=0] = total_w.cast[dtype]()
        if mean:
            out_ptr[unsafe_offset=0] = (total_in / total_w).cast[dtype]()
        else:
            out_ptr[unsafe_offset=0] = total_in.cast[dtype]()


def nll_threads(nframe: Int) -> Int:
    """Loss.cu `nll_loss_threads`: clamp(1 << round(log2(nframe / 16)), 32,
    1024), `nframe / 16` an integer division (0 rows -> 32)."""
    var q = nframe // 16
    if q <= 0:
        return 32
    var k = Int(round(log2(Float64(q))))
    return max(32, min(1 << min(k, 30), NLL_MAX_THREADS))


def nll_forward_reduce[
    dtype: DType, tdtype: DType
](
    out_addr: Int,
    tw_addr: Int,
    in_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    nframe: Int,
    classes: Int,
    mean: Bool,
    ignore_index: Int,
    one_d: Bool,
    ctx: DeviceContext,
) raises:
    var threads = 1 if one_d else nll_threads(nframe)
    _enqueue_cached[_nll_reduce_kernel[dtype, tdtype]](
        ctx,
        1,
        1,
        1,
        threads,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](tw_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[tdtype](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](weight_addr).as_unsafe_any_origin(),
        Int64(1 if weight_addr != 0 else 0),
        Int64(nframe),
        Int64(classes),
        Int64(1 if mean else 0),
        Int64(ignore_index),
        Int64(1 if one_d else 0),
    )


@__name(t"nll2d_forward_partial_{dtype}")
def _nll2d_partial_kernel[
    dtype: DType, tdtype: DType
](
    scratch_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    t_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    w_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    has_w_arg: Int64,
    classes_arg: Int64,
    map_arg: Int64,
    bps_arg: Int64,
    ignore_arg: Int64,
):
    """NLLLoss2d.cu nll_loss2d_forward_kernel up to its block reduction; the
    two block sums go to `scratch[2 * block]` rounded to the input dtype
    instead of into CUDA's atomics."""
    comptime acc_t = _acc[dtype]()
    var has_w = Int(has_w_arg) != 0
    var classes = Int(classes_arg)
    var map = Int(map_arg)
    var bps = Int(bps_arg)
    var ignore_index = Int(ignore_arg)
    var blk = Int(block_idx.x)
    var sample = blk // bps
    var toffset = sample * map
    var ioffset = sample * map * classes
    var step = NLL2D_THREADS * bps
    var input_sum = Scalar[acc_t](0)
    var acc_weight = Scalar[acc_t](0)
    var i = (blk % bps) * NLL2D_THREADS + Int(thread_idx.x)
    while i < map:
        var t = Int(t_ptr[unsafe_offset=toffset + i])
        if _valid(t, ignore_index, classes):
            var w = w_ptr[unsafe_offset=t] if has_w else Scalar[dtype](1)
            input_sum -= (in_ptr[unsafe_offset=ioffset + i + map * t] * w).cast[
                acc_t
            ]()
            acc_weight += w.cast[acc_t]()
        i += step
    var tw = block_sum[acc_t, NLL2D_THREADS](acc_weight)
    var total = block_sum[acc_t, NLL2D_THREADS](input_sum)
    if thread_idx.x == 0:
        scratch_ptr[unsafe_offset=2 * blk] = total.cast[dtype]()
        scratch_ptr[unsafe_offset=2 * blk + 1] = tw.cast[dtype]()


@__name(t"nll2d_forward_final_{dtype}")
def _nll2d_final_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    tw_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    scratch_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    blocks_arg: Int64,
    mean_arg: Int64,
):
    """The atomic adds of the partials (in the input dtype, from zero), then
    nll_loss2d_forward_size_average_kernel's `output /= total_weight`."""
    if thread_idx.x != 0:
        return
    var out = Scalar[dtype](0)
    var tw = Scalar[dtype](0)
    for b in range(Int(blocks_arg)):
        tw = tw + scratch_ptr[unsafe_offset=2 * b + 1]
        out = out + scratch_ptr[unsafe_offset=2 * b]
    if Int(mean_arg) != 0:
        out = out / tw
    out_ptr[unsafe_offset=0] = out
    tw_ptr[unsafe_offset=0] = tw


def nll2d_blocks_per_sample(map: Int) -> Int:
    """NLLLoss2d.cu: `GET_BLOCKS(map_nelem) / 128`, at least 1."""
    var bps = ((map + NLL2D_THREADS - 1) // NLL2D_THREADS) // 128
    return 1 if bps == 0 else bps


def nll2d_forward_reduce[
    dtype: DType, tdtype: DType
](
    out_addr: Int,
    tw_addr: Int,
    in_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    scratch_addr: Int,
    batch: Int,
    classes: Int,
    map: Int,
    mean: Bool,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    """`scratch` holds `2 * batch * nll2d_blocks_per_sample(map)` elements
    of the input dtype."""
    var bps = nll2d_blocks_per_sample(map)
    var blocks = bps * batch
    var scratch = _make_ptr[dtype](scratch_addr).as_unsafe_any_origin()
    _enqueue_cached[_nll2d_partial_kernel[dtype, tdtype]](
        ctx,
        blocks,
        1,
        1,
        NLL2D_THREADS,
        scratch,
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[tdtype](target_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](weight_addr).as_unsafe_any_origin(),
        Int64(1 if weight_addr != 0 else 0),
        Int64(classes),
        Int64(map),
        Int64(bps),
        Int64(ignore_index),
    )
    _enqueue_cached[_nll2d_final_kernel[dtype]](
        ctx,
        1,
        1,
        1,
        32,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](tw_addr).as_unsafe_any_origin(),
        scratch,
        Int64(blocks),
        Int64(1 if mean else 0),
    )


def nll_backward[
    dtype: DType, tdtype: DType
](
    gi_addr: Int,
    go_addr: Int,
    target_addr: Int,
    weight_addr: Int,
    tw_addr: Int,
    batch: Int,
    classes: Int,
    map: Int,
    reduction: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    """Every backward form into a ZEROED `[batch, classes, map]` grad_input:
    `-w[t] * grad_output[b, s]` for 'none' (Loss.cu / NLLLoss2d.cu
    *_backward_no_reduce_kernel), else `w[t] * g` with the scalar
    `g = -(mean ? grad_output / total_weight : grad_output)` formed in the
    input dtype (*_backward_reduce_* / nll_loss2d_backward_kernel). The
    1-D input goes through the reduced form whatever its reduction."""
    var gi_ptr = _make_ptr[dtype](gi_addr)
    var go_ptr = _make_ptr[dtype](go_addr)
    var t_ptr = _make_ptr[tdtype](target_addr)
    var w_ptr = _make_ptr[dtype](weight_addr)
    var tw_ptr = _make_ptr[dtype](tw_addr)
    var has_w = weight_addr != 0

    @always_inline
    @__parameter
    @__copy_capture(
        gi_ptr,
        go_ptr,
        t_ptr,
        w_ptr,
        tw_ptr,
        has_w,
        classes,
        map,
        reduction,
        ignore_index,
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var t = Int(t_ptr[unsafe_offset=i])
        if not _valid(t, ignore_index, classes):
            return
        var b = i // map
        var s = i - b * map
        var at = (b * classes + t) * map + s
        if reduction == 0:
            var w = w_ptr[unsafe_offset=t] if has_w else Scalar[dtype](1)
            gi_ptr[unsafe_offset=at] = -w * go_ptr[unsafe_offset=i]
        else:
            var g = go_ptr[unsafe_offset=0]
            if reduction == 1:
                g = g / tw_ptr[unsafe_offset=0]
            g = -g
            gi_ptr[unsafe_offset=at] = (
                w_ptr[unsafe_offset=t] * g if has_w else g
            )

    _parallel_for_dt[dtype, func](batch * map, ctx)
