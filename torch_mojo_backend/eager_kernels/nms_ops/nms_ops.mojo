"""Stable device merge sort, 64-box IoU masks, and kept-index gather."""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import AddressSpace, stack_allocation
from max.gpu.sync import barrier
from op_utils import (
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)


@always_inline
def _before[dt: DType](a: Scalar[dt], ai: Int, b: Scalar[dt], bi: Int) -> Bool:
    # Stable descending order, including torch's NaNs-before-numbers rule.
    var an = a != a
    var bn = b != b
    return (an and not bn) or (
        an == bn and (a > b or ((a == b or an) and ai < bi))
    )


@__name("nms_order_init_i64")
def _init(order: Pointer[Int64, MutAnyOrigin], n64: Int64):
    var n = Int(n64)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    while i < n:
        order[unsafe_offset=i] = Int64(i)
        i += Int(grid_dim.x * block_dim.x)


@__name("nms_merge_sort_" + String(dt))
def _merge[
    dt: DType
](
    scores: Pointer[Scalar[dt], MutAnyOrigin],
    src: Pointer[Int64, MutAnyOrigin],
    dst: Pointer[Int64, MutAnyOrigin],
    n64: Int64,
    width64: Int64,
):
    var n = Int(n64)
    var width = Int(width64)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    while i < n:
        var base = i // (2 * width) * (2 * width)
        var middle = min(base + width, n)
        var end = min(base + 2 * width, n)
        var left = base if i >= middle else middle
        var right = middle if i >= middle else end
        var other_start = left
        var original = Int(src[unsafe_offset=i])
        var score = scores[unsafe_offset=original]
        while left < right:
            var mid = (left + right) // 2
            var other = Int(src[unsafe_offset=mid])
            if _before[dt](scores[unsafe_offset=other], other, score, original):
                left = mid + 1
            else:
                right = mid
        var local = i - (middle if i >= middle else base)
        dst[unsafe_offset=base + local + left - other_start] = Int64(original)
        i += Int(grid_dim.x * block_dim.x)


@__name("nms_iou_mask_" + String(dt) + "_b64")
def _mask[
    dt: DType, acc: DType
](
    boxes: Pointer[Scalar[dt], MutAnyOrigin],
    order: Pointer[Int64, MutAnyOrigin],
    mask: Pointer[UInt64, MutAnyOrigin],
    n64: Int64,
    columns64: Int64,
    threshold: Scalar[acc],
):
    var n = Int(n64)
    var columns = Int(columns64)
    var tile = stack_allocation[256, dt, address_space=AddressSpace.SHARED]()
    var lane = Int(thread_idx.x)
    var task = Int(block_idx.x)
    while task < columns * columns:
        var row = task // columns
        var col = task % columns
        if col >= row:
            var col_count = min(64, n - col * 64)
            if lane < col_count:
                var bi = Int(order[unsafe_offset=col * 64 + lane]) * 4
                for j in range(4):
                    tile[unsafe_offset=lane * 4 + j] = boxes[
                        unsafe_offset=bi + j
                    ]
            barrier()
            var i = row * 64 + lane
            if i < n:
                var bi = Int(order[unsafe_offset=i]) * 4
                var ax1 = boxes[unsafe_offset=bi].cast[acc]()
                var ay1 = boxes[unsafe_offset=bi + 1].cast[acc]()
                var ax2 = boxes[unsafe_offset=bi + 2].cast[acc]()
                var ay2 = boxes[unsafe_offset=bi + 3].cast[acc]()
                var area = (ax2 - ax1) * (ay2 - ay1).cast[dt]().cast[acc]()
                var bits = UInt64(0)
                var start = lane + 1 if row == col else 0
                for j in range(start, col_count):
                    var bx1 = tile[unsafe_offset=j * 4].cast[acc]()
                    var by1 = tile[unsafe_offset=j * 4 + 1].cast[acc]()
                    var bx2 = tile[unsafe_offset=j * 4 + 2].cast[acc]()
                    var by2 = tile[unsafe_offset=j * 4 + 3].cast[acc]()
                    var width = max(
                        Scalar[acc](0),
                        (min(ax2, bx2) - max(ax1, bx1)).cast[dt]().cast[acc](),
                    )
                    var height = max(
                        Scalar[acc](0),
                        (min(ay2, by2) - max(ay1, by1)).cast[dt]().cast[acc](),
                    )
                    var inter = width * height
                    var union = (
                        area
                        + (bx2 - bx1) * (by2 - by1).cast[dt]().cast[acc]()
                        - inter
                    )
                    var overlap = inter / union
                    if overlap > threshold:
                        bits |= UInt64(1) << UInt64(j)
                mask[unsafe_offset=i * columns + col] = bits
            barrier()
        task += Int(grid_dim.x)


@__name("nms_keep_gather_i64")
def _gather(
    order: Pointer[Int64, MutAnyOrigin],
    keep: Pointer[Int64, MutAnyOrigin],
    output: Pointer[Int64, MutAnyOrigin],
    n64: Int64,
):
    var n = Int(n64)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    while i < n:
        output[unsafe_offset=i] = order[
            unsafe_offset=Int(keep[unsafe_offset=i])
        ]
        i += Int(grid_dim.x * block_dim.x)


def _run_mask[dt: DType](argv: Argv) raises:
    var boxes = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var scores = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var order = _make_ptr[DType.int64](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var temporary = _make_ptr[DType.int64](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    var mask = _make_ptr[DType.uint64](
        _raw_int(argv[unsafe_offset=4])
    ).as_unsafe_any_origin()
    var n = _raw_int(argv[unsafe_offset=5])
    var threshold = _raw_f64(argv[unsafe_offset=6])
    var ctx = _raw_ctx(argv[unsafe_offset=7])
    var sms = max(1, _device_sm_count(ctx))
    # Portable initial occupancy choice; these constants are not hardware-fitted.
    var blocks = max(1, min((n + 255) // 256, sms * 4))
    _enqueue_cached[_init](ctx, blocks, 1, 1, 256, order, Int64(n))
    var src = order
    var dst = temporary
    var width = 1
    while width < n:
        _enqueue_cached[_merge[dt]](
            ctx,
            blocks,
            1,
            1,
            256,
            scores,
            src,
            dst,
            Int64(n),
            Int64(width),
        )
        var swap = src
        src = dst
        dst = swap
        width *= 2
    var columns = (n + 63) // 64
    comptime acc = DType.float64 if dt == DType.float64 else DType.float32
    # CUDA devIoU takes float even for double boxes.
    var cutoff = Float32(threshold).cast[acc]()
    _enqueue_cached[_mask[dt, acc]](
        ctx,
        min(columns * columns, sms * 8),
        1,
        1,
        64,
        boxes,
        src,
        mask,
        Int64(n),
        Int64(columns),
        cutoff,
    )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["NmsMask"]():
            if argc != 8:
                raise Error("NmsMask expects eight slots")
            comptime for dt in [DType.float16, DType.float32, DType.float64]:
                comptime if _dtype_arg_on[0, dt]():
                    _run_mask[dt](argv)
                    return 0
        elif _op_on["NmsGather"]():
            if argc != 5:
                raise Error("NmsGather expects five slots")
            var order = _make_ptr[DType.int64](
                _raw_int(argv[unsafe_offset=0])
            ).as_unsafe_any_origin()
            var keep = _make_ptr[DType.int64](
                _raw_int(argv[unsafe_offset=1])
            ).as_unsafe_any_origin()
            var out = _make_ptr[DType.int64](
                _raw_int(argv[unsafe_offset=2])
            ).as_unsafe_any_origin()
            var n = _raw_int(argv[unsafe_offset=3])
            var ctx = _raw_ctx(argv[unsafe_offset=4])
            var blocks = max(
                1, min((n + 255) // 256, max(1, _device_sm_count(ctx)) * 4)
            )
            _enqueue_cached[_gather](
                ctx,
                blocks,
                1,
                1,
                256,
                order,
                keep,
                out,
                Int64(n),
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
