# ===----------------------------------------------------------------------=== #
# Fast eager-mode NLL loss kernels for mojo_device (float32 log-probs, int64
# targets, no class weights), ported from the validated Fable candidate.
#
# Same architecture as tmb/kernels/nn/entry.mojo: Python-visible functions get raw integer
# pointers (tensor `._ptr`, offset pre-applied) plus shape/mode ints and the
# device's DeviceContext pointer, and enqueue work on the device queue (fire
# and forget, no sync, no host reads, no host allocation).
#
# Forward:
#   - reduction=0 (none): grid-stride kernel, one loss per row, total_weight=0.
#   - reduction=1 (mean): a single 1024-thread block reduces the gathered
#     losses and valid-row count for small row counts; larger row counts use a
#     two-kernel scheme (per-block partials into a stream-ordered scratch
#     buffer, then a one-block finalize) to spread the gathers across SMs
#     without an extra zero-initialization launch.
#   - reduction=2 (sum): the reference sum is a serial fp32 accumulation whose
#     absolute tolerance is below one ulp of the result, so the kernel loads
#     row losses cooperatively into shared memory and one thread accumulates
#     them in row order, reproducing the reference rounding bit-for-bit.
#
# Backward writes the whole dense gradient, zero except target slots. For
# `classes % 4 == 0`, classes >= 1024, and a 16-byte-aligned grad_input base
# pointer a 2D strip-mapped kernel issues 16-byte vector stores (block_idx.y
# is the row, so no integer division and no wave tail); otherwise a
# warp-per-row kernel issues coalesced scalar stores. The raw ATen out
# pointer only guarantees element alignment (a contiguous tensor may start at
# a storage offset), so the vector regime is gated on the runtime pointer and
# the Int64 target loads use element alignment.
#
# Supported contract: non-ignore targets must lie in [0, classes). Out-of-
# range targets are skipped like ignore_index rather than trapped: Mojo has
# no asynchronous device-side assert that would not poison the shared device
# context, and these kernels never synchronize to check on the host.
# ===----------------------------------------------------------------------=== #

from max.gpu.sync import barrier
from max.gpu import WARP_SIZE, block_idx, grid_dim, lane_id, thread_idx, warp_id
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.math import ceildiv
from std.memory import stack_allocation
from std.os import abort
from std.sys.info import has_accelerator

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
)

from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)
from tmb.kernels.loss.nll_kernels import (
    P_BATCH,
    P_CLASSES,
    P_IGNORE,
    P_LEN,
    P_MAP,
    P_ONE_D,
    P_REDUCTION,
    P_SPATIAL,
    nll2d_forward_reduce,
    nll_backward,
    nll_forward_none,
    nll_forward_reduce,
)

from tmb.kernels.loss.margin_kernels import (
    multi_margin_backward,
    multi_margin_forward,
    multilabel_margin_backward,
    multilabel_margin_forward,
)

# Every float dtype the generic loss kernels take (CUDA's
# AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16)).
comptime LOSS_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]
# nll_loss targets: Long or Byte (AT_DISPATCH_NLL_LOSS_INDEX_TYPES).
comptime NLL_TARGET_DTYPES = [DType.int64, DType.uint8]


comptime _NONE_BLOCK = 256
comptime _MEAN_BLOCK = 1024
comptime _MEAN_ILP = 8
comptime _MEAN_CHUNK = _MEAN_BLOCK * _MEAN_ILP
comptime _MEAN_SINGLE_MAX_ROWS = 4096
comptime _PARTIAL_BLOCK = 256
comptime _SUM_BLOCK = 256
comptime _BWD_BLOCK = 256
comptime _VEC_UNROLL = 8
comptime _VEC_MIN_CLASSES = 4 * _BWD_BLOCK


@__name("nll_forward_none")
def _nll_forward_none(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    log_probs: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var ignore_index = Int(ignore_index_arg)
    var row = Int(block_idx.x) * _NONE_BLOCK + Int(thread_idx.x)
    if row == 0:
        total_weight[unsafe_offset=0] = 0.0
    var stride = Int(grid_dim.x) * _NONE_BLOCK
    while row < rows:
        var t = Int(target[unsafe_offset=row])
        var loss = Float32(0.0)
        if t != ignore_index and t >= 0 and t < classes:
            loss = -log_probs[unsafe_offset=row * classes + t]
        output[unsafe_offset=row] = loss
        row += stride


@__name("nll_forward_mean")
def _nll_forward_mean(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    log_probs: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var ignore_index = Int(ignore_index_arg)
    # Single block on one SM: use wide target loads and independent
    # accumulator lanes so enough gathers stay in flight to hide latency.
    # The wide load only claims element alignment: the target base pointer
    # may sit at any 8-byte boundary (storage offset).
    var tid = Int(thread_idx.x)
    var acc = SIMD[DType.float32, _MEAN_ILP](0.0)
    var count = SIMD[DType.float32, _MEAN_ILP](0.0)
    var base = 0
    while base + _MEAN_CHUNK <= rows:
        var r = base + tid * _MEAN_ILP
        var tv = target.unsafe_load[width=_MEAN_ILP, alignment=8](r)
        comptime for lane in range(_MEAN_ILP):
            var t = Int(tv[lane])
            if t != ignore_index and t >= 0 and t < classes:
                acc[lane] += log_probs[unsafe_offset=(r + lane) * classes + t]
                count[lane] += 1.0
        base += _MEAN_CHUNK
    var row = base + tid
    while row < rows:
        var t = Int(target[unsafe_offset=row])
        if t != ignore_index and t >= 0 and t < classes:
            acc[0] += log_probs[unsafe_offset=row * classes + t]
            count[0] += 1.0
        row += _MEAN_BLOCK
    var total = block.sum[block_size=_MEAN_BLOCK, broadcast=False](
        acc.reduce_add()
    )
    var valid = block.sum[block_size=_MEAN_BLOCK, broadcast=False](
        count.reduce_add()
    )
    if tid == 0:
        output[unsafe_offset=0] = -total / valid
        total_weight[unsafe_offset=0] = valid


@__name("nll_forward_mean_partial")
def _nll_forward_mean_partial(
    scratch: Pointer[Scalar[DType.float32], MutAnyOrigin],
    log_probs: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var ignore_index = Int(ignore_index_arg)
    var acc = Float32(0.0)
    var count = Float32(0.0)
    var row = Int(block_idx.x) * _PARTIAL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _PARTIAL_BLOCK
    while row < rows:
        var t = Int(target[unsafe_offset=row])
        if t != ignore_index and t >= 0 and t < classes:
            acc += log_probs[unsafe_offset=row * classes + t]
            count += 1.0
        row += stride
    var total = block.sum[block_size=_PARTIAL_BLOCK, broadcast=False](acc)
    var valid = block.sum[block_size=_PARTIAL_BLOCK, broadcast=False](count)
    if thread_idx.x == 0:
        var b = Int(block_idx.x)
        scratch[unsafe_offset=2 * b] = total
        scratch[unsafe_offset=2 * b + 1] = valid


@__name("nll_forward_mean_final")
def _nll_forward_mean_final(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    scratch: Pointer[Scalar[DType.float32], MutAnyOrigin],
    partials_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var partials = Int(partials_arg)
    var acc = Float32(0.0)
    var count = Float32(0.0)
    var i = Int(thread_idx.x)
    while i < partials:
        acc += scratch[unsafe_offset=2 * i]
        count += scratch[unsafe_offset=2 * i + 1]
        i += _PARTIAL_BLOCK
    var total = block.sum[block_size=_PARTIAL_BLOCK, broadcast=False](acc)
    var valid = block.sum[block_size=_PARTIAL_BLOCK, broadcast=False](count)
    if thread_idx.x == 0:
        output[unsafe_offset=0] = -total / valid
        total_weight[unsafe_offset=0] = valid


@__name("nll_forward_sum")
def _nll_forward_sum(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    log_probs: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var ignore_index = Int(ignore_index_arg)
    var losses = stack_allocation[
        _SUM_BLOCK, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var count = Float32(0.0)
    var base = 0
    while base < rows:
        var row = base + tid
        var loss = Float32(0.0)
        if row < rows:
            var t = Int(target[unsafe_offset=row])
            if t != ignore_index and t >= 0 and t < classes:
                loss = -log_probs[unsafe_offset=row * classes + t]
                count += 1.0
        losses[unsafe_offset=tid] = loss
        barrier()
        if tid == 0:
            # Serial fp32 accumulation in row order; ignored rows contribute
            # an exact 0.0 so the rounding sequence matches a reference that
            # skips them.
            var limit = min(_SUM_BLOCK, rows - base)
            for i in range(limit):
                acc += losses[unsafe_offset=i]
        barrier()
        base += _SUM_BLOCK
    var valid = block.sum[block_size=_SUM_BLOCK, broadcast=False](count)
    if tid == 0:
        output[unsafe_offset=0] = acc
        total_weight[unsafe_offset=0] = valid


@__name("nll_backward_vec4")
def _nll_backward_vec4(
    grad_input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    reduction_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var reduction = Int(reduction_arg)
    var ignore_index = Int(ignore_index_arg)
    # 2D strip mapping: block_idx.y selects the row (grid-stride for very
    # large row counts) and block_idx.x selects a strip of _VEC_UNROLL
    # 16-byte chunks per thread, so no integer division and no wave tail.
    # The host only routes here when the grad_input base pointer is 16-byte
    # aligned (classes % 4 == 0 keeps every row base on that boundary).
    var vec_cols = classes // 4
    var tid = Int(thread_idx.x)
    var strip = Int(block_idx.x) * (_BWD_BLOCK * _VEC_UNROLL)
    var scale = Float32(0.0)
    if reduction == 1:
        scale = grad_output[unsafe_offset=0] / total_weight[unsafe_offset=0]
    elif reduction == 2:
        scale = grad_output[unsafe_offset=0]
    var row = Int(block_idx.y)
    while row < rows:
        var t = Int(target[unsafe_offset=row])
        var valid = t != ignore_index and t >= 0 and t < classes
        var grad = Float32(0.0)
        if valid:
            grad = -(
                grad_output[unsafe_offset=row] if reduction == 0 else scale
            )
        var base = row * classes
        comptime for u in range(_VEC_UNROLL):
            var v = strip + u * _BWD_BLOCK + tid
            if v < vec_cols:
                var col = v * 4
                var chunk = SIMD[DType.float32, 4](0.0)
                # Comptime lane indices keep the vector in registers; a
                # runtime lane index would demote it to local memory.
                comptime for lane in range(4):
                    if valid and t == col + lane:
                        chunk[lane] = grad
                grad_input.unsafe_store[width=4, alignment=16](
                    base + col, chunk
                )
        row += Int(grid_dim.y)


@__name("nll_backward_scalar")
def _nll_backward_scalar(
    grad_input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    rows_arg: Int64,
    classes_arg: Int64,
    reduction_arg: Int64,
    ignore_index_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var classes = Int(classes_arg)
    var reduction = Int(reduction_arg)
    var ignore_index = Int(ignore_index_arg)
    var lane = Int(lane_id())
    var warp_stride = Int(grid_dim.x) * (_BWD_BLOCK // WARP_SIZE)
    var scale = Float32(0.0)
    if reduction == 1:
        scale = grad_output[unsafe_offset=0] / total_weight[unsafe_offset=0]
    elif reduction == 2:
        scale = grad_output[unsafe_offset=0]
    var row = Int(block_idx.x) * (_BWD_BLOCK // WARP_SIZE) + Int(warp_id())
    while row < rows:
        var t = Int(target[unsafe_offset=row])
        var valid = t != ignore_index and t >= 0 and t < classes
        var grad = Float32(0.0)
        if valid:
            grad = -(
                grad_output[unsafe_offset=row] if reduction == 0 else scale
            )
        var base = row * classes
        var col = lane
        while col < classes:
            grad_input[unsafe_offset=base + col] = grad if (
                valid and col == t
            ) else 0.0
            col += WARP_SIZE
        row += warp_stride


def enqueue_nll_forward_f32(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    log_probs: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    rows: Int,
    classes: Int,
    reduction: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        if reduction == 0:
            var grid = min(ceildiv(rows, _NONE_BLOCK), 4096)
            _enqueue_cached[_nll_forward_none](
                ctx,
                grid,
                1,
                1,
                _NONE_BLOCK,
                output,
                total_weight,
                log_probs,
                target,
                Int64(rows),
                Int64(classes),
                Int64(ignore_index),
            )
        elif reduction == 1:
            if rows <= _MEAN_SINGLE_MAX_ROWS:
                _enqueue_cached[_nll_forward_mean](
                    ctx,
                    1,
                    1,
                    1,
                    _MEAN_BLOCK,
                    output,
                    total_weight,
                    log_probs,
                    target,
                    Int64(rows),
                    Int64(classes),
                    Int64(ignore_index),
                )
            else:
                var grid = min(ceildiv(rows, _PARTIAL_BLOCK), 1024)
                var scratch = ctx.enqueue_create_buffer[DType.float32](2 * grid)
                var scratch_ptr = scratch.unsafe_ptr().as_unsafe_any_origin()
                _enqueue_cached[_nll_forward_mean_partial](
                    ctx,
                    grid,
                    1,
                    1,
                    _PARTIAL_BLOCK,
                    scratch_ptr,
                    log_probs,
                    target,
                    Int64(rows),
                    Int64(classes),
                    Int64(ignore_index),
                )
                _enqueue_cached[_nll_forward_mean_final](
                    ctx,
                    1,
                    1,
                    1,
                    _PARTIAL_BLOCK,
                    output,
                    total_weight,
                    scratch_ptr,
                    Int64(grid),
                )
                # Dropping `scratch` schedules a stream-ordered free after
                # the enqueued kernels complete.
                _ = scratch^
        elif reduction == 2:
            _enqueue_cached[_nll_forward_sum](
                ctx,
                1,
                1,
                1,
                _SUM_BLOCK,
                output,
                total_weight,
                log_probs,
                target,
                Int64(rows),
                Int64(classes),
                Int64(ignore_index),
            )
        else:
            raise Error("reduction must be 0, 1, or 2")


def enqueue_nll_backward_f32(
    grad_input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    target: Pointer[Scalar[DType.int64], MutAnyOrigin],
    total_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    classes: Int,
    reduction: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        if reduction < 0 or reduction > 2:
            raise Error("reduction must be 0, 1, or 2")
        if (
            classes % 4 == 0
            and classes >= _VEC_MIN_CLASSES
            and Int(grad_input) % 16 == 0
        ):
            var strips = ceildiv(classes // 4, _BWD_BLOCK * _VEC_UNROLL)
            _enqueue_cached[_nll_backward_vec4](
                ctx,
                strips,
                min(rows, 65535),
                1,
                _BWD_BLOCK,
                grad_input,
                grad_output,
                target,
                total_weight,
                Int64(rows),
                Int64(classes),
                Int64(reduction),
                Int64(ignore_index),
            )
        else:
            var grid = min(ceildiv(rows, _BWD_BLOCK // WARP_SIZE), 8192)
            _enqueue_cached[_nll_backward_scalar](
                ctx,
                grid,
                1,
                1,
                _BWD_BLOCK,
                grad_input,
                grad_output,
                target,
                total_weight,
                Int64(rows),
                Int64(classes),
                Int64(reduction),
                Int64(ignore_index),
            )


# ---------------------------------------------------------------------------
# Raw CPython entry points
# ---------------------------------------------------------------------------


def _nll_forward_go(
    output_ptr_obj: Arg,
    total_weight_ptr_obj: Arg,
    log_probs_ptr_obj: Arg,
    target_ptr_obj: Arg,
    rows_obj: Arg,
    classes_obj: Arg,
    reduction_obj: Arg,
    ignore_index_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[DType.float32](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var total_weight = _make_ptr[DType.float32](
        _raw_int(total_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var log_probs = _make_ptr[DType.float32](
        _raw_int(log_probs_ptr_obj)
    ).as_unsafe_any_origin()
    var target = _make_ptr[DType.int64](
        _raw_int(target_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_nll_forward_f32(
        output,
        total_weight,
        log_probs,
        target,
        _raw_int(rows_obj),
        _raw_int(classes_obj),
        _raw_int(reduction_obj),
        _raw_int(ignore_index_obj),
        ctx,
    )


def _nll_backward_go(
    grad_input_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    target_ptr_obj: Arg,
    total_weight_ptr_obj: Arg,
    rows_obj: Arg,
    classes_obj: Arg,
    reduction_obj: Arg,
    ignore_index_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var grad_input = _make_ptr[DType.float32](
        _raw_int(grad_input_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var target = _make_ptr[DType.int64](
        _raw_int(target_ptr_obj)
    ).as_unsafe_any_origin()
    var total_weight = _make_ptr[DType.float32](
        _raw_int(total_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_nll_backward_f32(
        grad_input,
        grad_output,
        target,
        total_weight,
        _raw_int(rows_obj),
        _raw_int(classes_obj),
        _raw_int(reduction_obj),
        _raw_int(ignore_index_obj),
        ctx,
    )


def _nll_forward_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _nll_forward_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
    )


def _nll_backward_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _nll_backward_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
    )


# ---------------------------------------------------------------------------
# Generic NLL (every float dtype, weights, 1-D / 2-D / spatial)
#
#   Nll:          out, total_weight, input, target, weight (0 = none),
#                 scratch (nll_loss2d's reduced form), params, ctx
#   NllBackward:  grad_input (zeroed), grad_output, target, weight,
#                 total_weight, params, ctx
# with params the P_* tuple of nll_kernels.mojo.
# ---------------------------------------------------------------------------


def _nll_go[dtype: DType, tdtype: DType](argv: Argv) raises:
    var p = argv[unsafe_offset=6]
    if _raw_tuple_len(p) != P_LEN:
        raise Error("nll: expected ", P_LEN, " params")
    var batch = _raw_tuple_int(p, P_BATCH)
    var classes = _raw_tuple_int(p, P_CLASSES)
    var map = _raw_tuple_int(p, P_MAP)
    var reduction = _raw_tuple_int(p, P_REDUCTION)
    var ignore_index = _raw_tuple_int(p, P_IGNORE)
    var one_d = _raw_tuple_int(p, P_ONE_D) != 0
    var spatial = _raw_tuple_int(p, P_SPATIAL) != 0
    var ctx = _raw_ctx(argv[unsafe_offset=7])
    if reduction == 0 and not one_d:
        nll_forward_none[dtype, tdtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            batch,
            classes,
            map,
            ignore_index,
            ctx,
        )
    elif spatial:
        nll2d_forward_reduce[dtype, tdtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            _raw_int(argv[unsafe_offset=5]),
            batch,
            classes,
            map,
            reduction == 1,
            ignore_index,
            ctx,
        )
    else:
        nll_forward_reduce[dtype, tdtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            batch,
            classes,
            reduction == 1,
            ignore_index,
            one_d,
            ctx,
        )


def _nll_backward_go[dtype: DType, tdtype: DType](argv: Argv) raises:
    var p = argv[unsafe_offset=5]
    if _raw_tuple_len(p) != P_LEN:
        raise Error("nll: expected ", P_LEN, " params")
    var reduction = _raw_tuple_int(p, P_REDUCTION)
    if _raw_tuple_int(p, P_ONE_D) != 0 and reduction == 0:
        reduction = 2  # the 1-D input takes the reduced kernel
    nll_backward[dtype, tdtype](
        _raw_int(argv[unsafe_offset=0]),
        _raw_int(argv[unsafe_offset=1]),
        _raw_int(argv[unsafe_offset=2]),
        _raw_int(argv[unsafe_offset=3]),
        _raw_int(argv[unsafe_offset=4]),
        _raw_tuple_int(p, P_BATCH),
        _raw_tuple_int(p, P_CLASSES),
        _raw_tuple_int(p, P_MAP),
        reduction,
        _raw_tuple_int(p, P_IGNORE),
        _raw_ctx(argv[unsafe_offset=6]),
    )


def _nll_dispatch[backward: Bool](argv: Argv) raises:
    comptime for dt in LOSS_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            comptime for tdt in NLL_TARGET_DTYPES:
                comptime if _dtype_arg_on[1, tdt]():
                    comptime if backward:
                        _nll_backward_go[dt, tdt](argv)
                    else:
                        _nll_go[dt, tdt](argv)
                    return
    raise Error("nll: no (input, target) dtype pair compiled into this build")


# ---------------------------------------------------------------------------
# Margin losses (DTYPE_ARG_0 = the input dtype)
#
#   MultiMargin:               out, input, target, weight, params
#                              (nframe, dim, p, size_average), margin, ctx
#   MultiMarginBackward:       grad_input, grad_output, input, target, weight,
#                              params (nframe, dim, p, size_average, reduce),
#                              margin, ctx
#   MultilabelMargin:          out, input, target, is_target, params
#                              (nframe, dim, size_average), ctx
#   MultilabelMarginBackward:  grad_input, grad_output, input, target,
#                              is_target, params (nframe, dim, reduce), gain,
#                              ctx
# `margin` / `gain` travel as float64 and are rounded to the input dtype here
# on the host, through float32 for the narrow dtypes like `c10::Half(double)`.
# ---------------------------------------------------------------------------


def _to_dtype[dtype: DType](v: Float64) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        return v.cast[dtype]()
    else:
        return v.cast[DType.float32]().cast[dtype]()


def _margin_go[dtype: DType](argv: Argv) raises:
    comptime if _op_on["MultiMargin"]():
        var p = argv[unsafe_offset=4]
        multi_margin_forward[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3) != 0,
            _to_dtype[dtype](_raw_f64(argv[unsafe_offset=5])),
            _raw_ctx(argv[unsafe_offset=6]),
        )
    elif _op_on["MultiMarginBackward"]():
        var p = argv[unsafe_offset=5]
        multi_margin_backward[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3) != 0,
            _raw_tuple_int(p, 4) != 0,
            _to_dtype[dtype](_raw_f64(argv[unsafe_offset=6])),
            _raw_ctx(argv[unsafe_offset=7]),
        )
    elif _op_on["MultilabelMargin"]():
        var p = argv[unsafe_offset=4]
        multilabel_margin_forward[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2) != 0,
            _raw_ctx(argv[unsafe_offset=5]),
        )
    elif _op_on["MultilabelMarginBackward"]():
        var p = argv[unsafe_offset=5]
        multilabel_margin_backward[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2) != 0,
            _to_dtype[dtype](_raw_f64(argv[unsafe_offset=6])),
            _raw_ctx(argv[unsafe_offset=7]),
        )
    else:
        raise Error(NO_OP_COMPILED)


def _margin_dispatch(argv: Argv) raises:
    comptime for dt in LOSS_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _margin_go[dt](argv)
            return
    raise Error("margin loss: no input dtype compiled into this build")


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["NllLossForwardF32"]():
            _nll_forward_dispatcher(argv, argc)
            return 0
        comptime if _op_on["NllLossBackwardF32"]():
            _nll_backward_dispatcher(argv, argc)
            return 0
        comptime if _op_on["Nll"]():
            _nll_dispatch[False](argv)
            return 0
        comptime if _op_on["NllBackward"]():
            _nll_dispatch[True](argv)
            return 0
        comptime if (
            _op_on["MultiMargin"]()
            or _op_on["MultiMarginBackward"]()
            or _op_on["MultilabelMargin"]()
            or _op_on["MultilabelMarginBackward"]()
        ):
            _margin_dispatch(argv)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
