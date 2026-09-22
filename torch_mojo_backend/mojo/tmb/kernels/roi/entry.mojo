"""Dynamic ROI sampling with shared geometry and atomic backward scatter."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import mulhi
from std.sys import (
    is_amd_gpu,
    is_nvidia_gpu,
    inlined_assembly,
    has_apple_gpu_accelerator,
)
from std.utils.fast_div import FastDiv
from std.math import ceil, ceildiv, floor
from tmb.kernels.common.dtype_arithmetic import _product
from std.memory import bitcast
from tmb.kernels.common.op_utils import (
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

# Vector kernel arguments lack Metal buffer metadata; use an aggregate.
comptime DivisorArgs = InlineArray[UInt32, 4]

comptime BLOCK = 256
# Measured on H100: 32 forward blocks/SM, 8 scatter blocks/SM.
comptime FORWARD_BLOCKS_PER_SM = 32
comptime BACKWARD_BLOCKS_PER_SM = 8


@always_inline
def _roi_scale[dt: DType](value: Float64) -> Scalar[dt]:
    # Preserve c10::Half(float)'s float32 rounding against cast folding.
    comptime if dt == DType.float16:
        var rounded = value.cast[DType.float32]()
        var ptr = Pointer(to=rounded)
        ptr.unsafe_store[volatile=True](0, rounded)
        return ptr.unsafe_load[volatile=True](0).cast[dt]()
    else:
        return value.cast[dt]()


@always_inline
def _valid_roi[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    roi: Int,
    n: Int,
    scale: Scalar[acc],
) -> Bool:
    comptime coord = DType.float64 if dt == DType.float64 else DType.float32
    var batch = rois[unsafe_offset=roi * 5].cast[coord]()
    if not (batch >= 0 and batch < Scalar[coord](9223372036854775808.0)):
        return False
    if Int(batch) >= n:
        return False
    for axis in range(1, 5):
        var value = _product(
            rois[unsafe_offset=roi * 5 + axis].cast[acc](), scale
        ).cast[coord]()
        # Invalid coordinates must never become pointer offsets.
        if not (abs(value) < Scalar[coord](1 << 60)):
            return False
    return True


@always_inline
def _axis[
    dt: DType
](var pos: Scalar[dt], size: Int) -> Tuple[Int, Int, Scalar[dt]]:
    pos = max(pos, Scalar[dt](0))
    var low = Int(pos)
    if low >= size - 1:
        return (size - 1, size - 1, Scalar[dt](0))
    return (low, low + 1, pos - Scalar[dt](low))


@always_inline
def _sample[
    dt: DType, acc: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    base: Int,
    y: Scalar[acc],
    x: Scalar[acc],
    h: Int,
    w: Int,
) -> Scalar[acc]:
    comptime coord = DType.float64 if acc == DType.float64 else DType.float32
    if (
        y < -1
        or y.cast[coord]() > Scalar[coord](h)
        or x < -1
        or x.cast[coord]() > Scalar[coord](w)
    ):
        return 0
    var (yl, yh, ly) = _axis(y, h)
    var (xl, xh, lx) = _axis(x, w)
    var hy = 1 - ly
    var hx = 1 - lx
    return (
        _product(
            _product(hy, hx),
            input[unsafe_offset=base + yl * w + xl].cast[acc](),
        )
        + _product(
            _product(hy, lx),
            input[unsafe_offset=base + yl * w + xh].cast[acc](),
        )
        + _product(
            _product(ly, hx),
            input[unsafe_offset=base + yh * w + xl].cast[acc](),
        )
        + _product(
            _product(ly, lx),
            input[unsafe_offset=base + yh * w + xh].cast[acc](),
        )
    )


@always_inline
def _round_away[dt: DType](value: Scalar[dt]) -> Int:
    var magnitude = abs(value)
    var whole = Int(magnitude)
    var rounded = whole + Int(magnitude - Scalar[dt](whole) >= 0.5)
    return rounded if value >= 0 else -rounded


@always_inline
def _divisor(divisor: Int) -> DivisorArgs:
    # FastDiv is not DevicePassable; pack its pinned multiplier/shift fields.
    var d = FastDiv[DType.uint32](divisor)
    var result = DivisorArgs(fill=UInt32(0))
    result[0] = UInt32(d._mprime)
    result[1] = UInt32(d._sh1)
    result[2] = UInt32(d._log2_shift) if d._is_pow2 else UInt32(d._sh2)
    return result^


@always_inline
def _divide(value: UInt32, divisor: DivisorArgs) -> UInt32:
    var high = mulhi(divisor[0], value)
    return (high + ((value - high) >> divisor[1])) >> divisor[2]


@always_inline
def _coordinates[
    fast: Bool
](
    index: Int,
    c: Int,
    ph: Int,
    pw: Int,
    div_pw: DivisorArgs,
    div_ph: DivisorArgs,
    div_c: DivisorArgs,
) -> Tuple[Int, Int, Int, Int]:
    comptime if fast:
        var q0 = _divide(UInt32(index), div_pw)
        var q1 = _divide(q0, div_ph)
        var q2 = _divide(q1, div_c)
        return (
            Int(UInt32(index) - q0 * UInt32(pw)),
            Int(q0 - q1 * UInt32(ph)),
            Int(q1 - q2 * UInt32(c)),
            Int(q2),
        )
    else:
        return (
            index % pw,
            index // pw % ph,
            index // (pw * ph) % c,
            index // (pw * ph * c),
        )


@always_inline
def _pool_bounds[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    roi: Int,
    n: Int,
    h: Int,
    w: Int,
    ph: Int,
    pw: Int,
    by: Int,
    bx: Int,
    scale: Scalar[acc],
) -> Tuple[Int, Int, Int, Int, Int]:
    if not _valid_roi(rois, roi, n, scale):
        return (-1, 0, 0, 0, 0)
    var batch = Int(rois[unsafe_offset=roi * 5])
    var x0 = _round_away(
        _product(rois[unsafe_offset=roi * 5 + 1].cast[acc](), scale)
    )
    var y0 = _round_away(
        _product(rois[unsafe_offset=roi * 5 + 2].cast[acc](), scale)
    )
    var x1 = _round_away(
        _product(rois[unsafe_offset=roi * 5 + 3].cast[acc](), scale)
    )
    var y1 = _round_away(
        _product(rois[unsafe_offset=roi * 5 + 4].cast[acc](), scale)
    )
    var bh = Scalar[acc](max(y1 - y0 + 1, 1)) / Scalar[acc](ph)
    var bw = Scalar[acc](max(x1 - x0 + 1, 1)) / Scalar[acc](pw)
    var ys = min(max(y0 + Int(floor(_product(Scalar[acc](by), bh))), 0), h)
    var ye = min(max(y0 + Int(ceil(Scalar[acc](by + 1) * bh)), 0), h)
    var xs = min(max(x0 + Int(floor(_product(Scalar[acc](bx), bw))), 0), w)
    var xe = min(max(x0 + Int(ceil(Scalar[acc](bx + 1) * bw)), 0), w)
    return (batch, ys, ye, xs, xe)


@__name("roi_pool_bin_geometry_" + String(dt))
def _pool_geometry[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    geometry: Pointer[Int64, MutAnyOrigin],
    n: Int64,
    h: Int64,
    w: Int64,
    k: Int64,
    ph: Int64,
    pw: Int64,
    scale: Scalar[acc],
):
    var index = Int32(block_idx.x) * BLOCK + Int32(thread_idx.x)
    var count = Int32(k * ph * pw)
    while index < count:
        var roi = Int(index // Int32(ph * pw))
        var by = Int(index // Int32(pw) % Int32(ph))
        var bx = Int(index % Int32(pw))
        var bounds = _pool_bounds(
            rois, roi, Int(n), Int(h), Int(w), Int(ph), Int(pw), by, bx, scale
        )
        comptime for field in range(5):
            geometry[unsafe_offset=Int(index) * 5 + field] = Int64(
                bounds[field]
            )
        index += Int32(grid_dim.x) * BLOCK


@__name("roi_align_fwd_" + String(dt) + ("_fastdiv" if fast else "_i64"))
def _align_forward[
    dt: DType, acc: DType, fast: Bool
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    sampling64: Int64,
    aligned64: Int64,
    div_pw: DivisorArgs,
    div_ph: DivisorArgs,
    div_c: DivisorArgs,
):
    var aligned = aligned64 != 0
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    comptime idt = DType.int32 if fast else DType.int64
    var count = Scalar[idt](Int(k64) * c * ph * pw)
    var index = Scalar[idt](block_idx.x) * BLOCK + Scalar[idt](thread_idx.x)
    while index < count:
        var (bx, by, channel, roi) = _coordinates[fast](
            Int(index), c, ph, pw, div_pw, div_ph, div_c
        )
        if not _valid_roi(rois, roi, n, scale):
            output[unsafe_offset=Int(index)] = 0
            index += Scalar[idt](grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var offset = Scalar[acc](0.5) if aligned else Scalar[acc](0)
        var x0 = (
            _product(rois[unsafe_offset=roi * 5 + 1].cast[acc](), scale)
            - offset
        )
        var y0 = (
            _product(rois[unsafe_offset=roi * 5 + 2].cast[acc](), scale)
            - offset
        )
        var rw = (
            _product(rois[unsafe_offset=roi * 5 + 3].cast[acc](), scale)
            - offset
            - x0
        )
        var rh = (
            _product(rois[unsafe_offset=roi * 5 + 4].cast[acc](), scale)
            - offset
            - y0
        )
        if not aligned:
            rw = max(rw, Scalar[acc](1))
            rh = max(rh, Scalar[acc](1))
        var bh = rh / Scalar[acc](ph)
        var bw = rw / Scalar[acc](pw)
        var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
        var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
        var value = Scalar[acc](0)
        if batch >= 0 and batch < n:
            for iy in range(gh):
                var y = (
                    y0
                    + _product(Scalar[acc](by), bh)
                    + _product((Scalar[acc](iy) + 0.5), bh) / Scalar[acc](gh)
                )
                for ix in range(gw):
                    var x = (
                        x0
                        + _product(Scalar[acc](bx), bw)
                        + _product((Scalar[acc](ix) + 0.5), bw)
                        / Scalar[acc](gw)
                    )
                    value += _sample(
                        input, (batch * c + channel) * h * w, y, x, h, w
                    )
        output[unsafe_offset=Int(index)] = (
            value / Scalar[acc](max(gh * gw, 1))
        ).cast[dt]()
        index += Scalar[idt](grid_dim.x) * BLOCK


@__name(
    "roi_pool_fwd_"
    + String(dt)
    + ("_geometry" if precomputed else "_fastdiv" if fast else "_i64")
)
def _pool_forward[
    dt: DType, acc: DType, fast: Bool, precomputed: Bool
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    geometry: Pointer[Int64, MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    div_pw: DivisorArgs,
    div_ph: DivisorArgs,
    div_c: DivisorArgs,
):
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    comptime idt = DType.int32 if fast else DType.int64
    var count = Scalar[idt](Int(k64) * c * ph * pw)
    var index = Scalar[idt](block_idx.x) * BLOCK + Scalar[idt](thread_idx.x)
    while index < count:
        var (bx, by, channel, roi) = _coordinates[fast](
            Int(index), c, ph, pw, div_pw, div_ph, div_c
        )
        var bounds = (Int(0), Int(0), Int(0), Int(0), Int(0))
        comptime if precomputed:
            var bin = (roi * ph + by) * pw + bx
            bounds = (
                Int(geometry[unsafe_offset=bin * 5]),
                Int(geometry[unsafe_offset=bin * 5 + 1]),
                Int(geometry[unsafe_offset=bin * 5 + 2]),
                Int(geometry[unsafe_offset=bin * 5 + 3]),
                Int(geometry[unsafe_offset=bin * 5 + 4]),
            )
        else:
            bounds = _pool_bounds(rois, roi, n, h, w, ph, pw, by, bx, scale)
        var (batch, ys, ye, xs, xe) = bounds
        var value = Scalar[dt](-3.4028234663852886e38)
        var winner = -1
        if ye <= ys or xe <= xs or batch < 0 or batch >= n:
            value = 0
        else:
            for y in range(ys, ye):
                for x in range(xs, xe):
                    var v = input[
                        unsafe_offset=(batch * c + channel) * h * w + y * w + x
                    ]
                    if v > value:
                        value = v
                        winner = y * w + x
        output[unsafe_offset=Int(index)] = value
        argmax[unsafe_offset=Int(index)] = Int32(winner)
        index += Scalar[idt](grid_dim.x) * BLOCK


@always_inline
def _scope() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _add[dt: DType](ptr: Pointer[Scalar[dt], MutAnyOrigin], value: Scalar[dt]):
    _ = Atomic[dt, scope=_scope()].fetch_add[ordering=Ordering.RELAXED](
        ptr, value
    )


# Metal has no 16-bit atomic add/CAS. Store each half accumulator in its
# own float32 word, rounding EVERY successful addition back to half. This
# preserves half atomic semantics without touching adjacent tensor storage.
def _scatter_storage_dtype[dt: DType]() -> DType:
    return (
        DType.float32 if dt == DType.float16
        and has_apple_gpu_accelerator() else dt
    )


@__name("scatter_accumulator_cast_" + String(src) + "_" + String(dst))
def _scatter_cast[
    src: DType, dst: DType
](
    input: Pointer[Scalar[src], ImmutAnyOrigin],
    output: Pointer[Scalar[dst], MutAnyOrigin],
    count: Int64,
):
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i < Int(count):
        output[unsafe_offset=i] = input[unsafe_offset=i].cast[dst]()


def _scatter_buffer[
    dt: DType
](
    ctx: DeviceContext,
    output: Pointer[Scalar[dt], MutAnyOrigin],
    count: Int,
    initialize: Bool = True,
) raises -> DeviceBuffer[_scatter_storage_dtype[dt]()]:
    comptime storage = _scatter_storage_dtype[dt]()
    comptime if storage != dt:
        var buffer = ctx.enqueue_create_buffer[storage](count)
        if initialize:
            ctx.enqueue_memset(buffer, Scalar[storage](0))
        else:
            _enqueue_cached[_scatter_cast[dt, storage]](
                ctx,
                ceildiv(count, BLOCK),
                1,
                1,
                BLOCK,
                output.as_imm(),
                buffer.unsafe_ptr().as_unsafe_any_origin(),
                Int64(count),
            )
        return buffer^
    else:
        var buffer = DeviceBuffer[storage](
            ctx,
            output.unsafe_bitcast[Scalar[storage]]().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            count,
            owning=False,
        )
        if initialize:
            ctx.enqueue_memset(buffer, Scalar[storage](0))
        return buffer^


def _finish_scatter[
    dt: DType
](
    ctx: DeviceContext,
    buffer: DeviceBuffer[_scatter_storage_dtype[dt]()],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    count: Int,
) raises:
    comptime storage = _scatter_storage_dtype[dt]()
    comptime if storage != dt:
        _enqueue_cached[_scatter_cast[storage, dt]](
            ctx,
            ceildiv(count, BLOCK),
            1,
            1,
            BLOCK,
            buffer.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            output,
            Int64(count),
        )


@__name("roi_align_bwd_scatter_" + String(dt))
def _align_scatter[
    dt: DType, acc: DType, storage: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[storage], MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    sampling64: Int64,
    aligned64: Int64,
):
    comptime coord = DType.float64 if acc == DType.float64 else DType.float32
    var aligned = aligned64 != 0
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    var count = Int(k64) * c * ph * pw
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var bx = index % pw
        var by = index // pw % ph
        var channel = index // (pw * ph) % c
        var roi = index // (pw * ph * c)
        if not _valid_roi(rois, roi, n, scale):
            index += Int(grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var offset = Scalar[acc](0.5) if aligned else Scalar[acc](0)
        var x0 = (
            _product(rois[unsafe_offset=roi * 5 + 1].cast[acc](), scale)
            - offset
        )
        var y0 = (
            _product(rois[unsafe_offset=roi * 5 + 2].cast[acc](), scale)
            - offset
        )
        var rw = (
            _product(rois[unsafe_offset=roi * 5 + 3].cast[acc](), scale)
            - offset
            - x0
        )
        var rh = (
            _product(rois[unsafe_offset=roi * 5 + 4].cast[acc](), scale)
            - offset
            - y0
        )
        if not aligned:
            rw = max(rw, Scalar[acc](1))
            rh = max(rh, Scalar[acc](1))
        var bh = rh / Scalar[acc](ph)
        var bw = rw / Scalar[acc](pw)
        var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
        var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
        var grad = input[unsafe_offset=index].cast[acc]()
        var samples = Scalar[acc](gh * gw)
        if batch >= 0 and batch < n:
            for iy in range(gh):
                var y = (
                    y0
                    + _product(Scalar[acc](by), bh)
                    + _product((Scalar[acc](iy) + 0.5), bh) / Scalar[acc](gh)
                )
                for ix in range(gw):
                    var x = (
                        x0
                        + _product(Scalar[acc](bx), bw)
                        + _product((Scalar[acc](ix) + 0.5), bw)
                        / Scalar[acc](gw)
                    )
                    if (
                        y < -1
                        or y.cast[coord]() > Scalar[coord](h)
                        or x < -1
                        or x.cast[coord]() > Scalar[coord](w)
                    ):
                        continue
                    var (yl, yh, ly) = _axis(y, h)
                    var (xl, xh, lx) = _axis(x, w)
                    var base = (batch * c + channel) * h * w
                    _ps_add(
                        output,
                        base + yl * w + xl,
                        (grad * ((1 - ly) * (1 - lx)) / samples).cast[acc](),
                        n * c * h * w,
                    )
                    _ps_add(
                        output,
                        base + yl * w + xh,
                        (grad * ((1 - ly) * lx) / samples).cast[acc](),
                        n * c * h * w,
                    )
                    _ps_add(
                        output,
                        base + yh * w + xl,
                        (grad * (ly * (1 - lx)) / samples).cast[acc](),
                        n * c * h * w,
                    )
                    _ps_add(
                        output,
                        base + yh * w + xh,
                        (grad * (ly * lx) / samples).cast[acc](),
                        n * c * h * w,
                    )
        index += Int(grid_dim.x) * BLOCK


@always_inline
def _pool_add_half[
    dt: DType
](
    ptr: Pointer[Scalar[dt], MutAnyOrigin],
    value: Scalar[dt],
    offset: Int,
    count: Int,
):
    comptime if is_nvidia_gpu():
        var address = UInt64(ptr)
        var lane = Int((address >> 1) & 1)
        # The zero neighbor must also be inside this allocation.
        if (lane == 0 and offset + 1 < count) or (lane == 1 and offset > 0):
            var bits = UInt32(bitcast[DType.uint16](value)) << UInt32(lane * 16)
            _ = inlined_assembly[
                "atom.relaxed.gpu.global.add.noftz.f16x2 $0, [$1], $2;",
                UInt32,
                constraints="=r,l,r,~{memory}",
            ](address & ~UInt64(3), bits)
        else:
            _add(ptr, value)
    else:
        _add(ptr, value)


@__name("roi_pool_bwd_scatter_" + String(dt))
def _pool_scatter[
    dt: DType, acc: DType, storage: DType
](
    grad: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[storage], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    n: Int64,
    c: Int64,
    hw: Int64,
    bins: Int64,
    count: Int64,
):
    comptime coord = DType.float64 if dt == DType.float64 else DType.float32
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    while i < Int(count):
        var plane = i // Int(bins)
        var roi = plane // Int(c)
        var channel = plane % Int(c)
        var batch_value = rois[unsafe_offset=roi * 5].cast[coord]()
        var pixel = Int(argmax[unsafe_offset=i])
        if (
            batch_value >= 0
            and batch_value < Scalar[coord](9223372036854775808.0)
            and Int(batch_value) < Int(n)
            and pixel >= 0
            and pixel < Int(hw)
        ):
            var offset = (Int(batch_value) * Int(c) + channel) * Int(hw) + pixel
            _ps_add(
                output,
                offset,
                grad[unsafe_offset=i].cast[acc](),
                Int(n * c * hw),
            )
        i += Int(grid_dim.x) * 256


def _launch_backward[dt: DType, pool: Bool](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("ROI kernel expects 15 argument slots")
    comptime acc = dt
    comptime storage = _scatter_storage_dtype[dt]()
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var rois = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var output = _make_ptr[acc](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var n = Int64(_raw_int(argv[unsafe_offset=4]))
    var c = Int64(_raw_int(argv[unsafe_offset=5]))
    var h = Int64(_raw_int(argv[unsafe_offset=6]))
    var w = Int64(_raw_int(argv[unsafe_offset=7]))
    var k = Int64(_raw_int(argv[unsafe_offset=8]))
    var ph = Int64(_raw_int(argv[unsafe_offset=9]))
    var pw = Int64(_raw_int(argv[unsafe_offset=10]))
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var nin = Int(n * c * h * w)
    if nin == 0:
        return
    var buffer = _scatter_buffer(ctx, output, nin)
    var accumulator = buffer.unsafe_ptr().as_unsafe_any_origin()
    var nout = Int(k * c * ph * pw)
    if nout == 0:
        _finish_scatter(ctx, buffer, output, nin)
        return
    var blocks = min(
        ceildiv(nout, BLOCK), _device_sm_count(ctx) * BACKWARD_BLOCKS_PER_SM
    )
    comptime if pool:
        var indices = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=_raw_int(argv[unsafe_offset=3])
        )
        _enqueue_cached[_pool_scatter[dt, acc, storage]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            accumulator,
            indices,
            n,
            c,
            h * w,
            ph * pw,
            Int64(nout),
        )
    else:
        var scale = _roi_scale[acc](_raw_f64(argv[unsafe_offset=11]))
        var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
        var aligned = Int64(_raw_int(argv[unsafe_offset=13]))
        _enqueue_cached[_align_scatter[dt, acc, storage]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            accumulator,
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            sampling,
            aligned,
        )
    _finish_scatter(ctx, buffer, output, nin)
    _ = buffer
    _ = ctx


def _enqueue_forward[
    dt: DType, pool: Bool, fast: Bool
](argv: Argv, blocks: Int, sm: Int) raises:
    comptime acc = dt
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var rois = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var output = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var n = Int64(_raw_int(argv[unsafe_offset=4]))
    var c = Int64(_raw_int(argv[unsafe_offset=5]))
    var h = Int64(_raw_int(argv[unsafe_offset=6]))
    var w = Int64(_raw_int(argv[unsafe_offset=7]))
    var k = Int64(_raw_int(argv[unsafe_offset=8]))
    var ph = Int64(_raw_int(argv[unsafe_offset=9]))
    var pw = Int64(_raw_int(argv[unsafe_offset=10]))
    var scale = _roi_scale[acc](_raw_f64(argv[unsafe_offset=11]))
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var div_pw = DivisorArgs(fill=UInt32(0))
    var div_ph = DivisorArgs(fill=UInt32(0))
    var div_c = DivisorArgs(fill=UInt32(0))
    comptime if fast:
        div_pw = _divisor(Int(pw))
        div_ph = _divisor(Int(ph))
        div_c = _divisor(Int(c))
    comptime if pool:
        var indices = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=_raw_int(argv[unsafe_offset=3])
        )
        # H100 measurements favor shared bin geometry beyond 32 channels and one 32-block/SM wave.
        comptime if fast:
            if c >= 32 and k * c * ph * pw >= Int64(
                sm * BLOCK * FORWARD_BLOCKS_PER_SM
            ):
                var geometry = ctx.enqueue_create_buffer[DType.int64](
                    Int(k * ph * pw * 5)
                )
                var geom = geometry.unsafe_ptr().as_unsafe_any_origin()
                _enqueue_cached[_pool_geometry[dt, acc]](
                    ctx,
                    min(ceildiv(Int(k * ph * pw), BLOCK), sm * 8),
                    1,
                    1,
                    BLOCK,
                    rois,
                    geom,
                    n,
                    h,
                    w,
                    k,
                    ph,
                    pw,
                    scale,
                )
                _enqueue_cached[_pool_forward[dt, acc, fast, True]](
                    ctx,
                    blocks,
                    1,
                    1,
                    BLOCK,
                    input,
                    rois,
                    output,
                    indices,
                    geom,
                    n,
                    c,
                    h,
                    w,
                    k,
                    ph,
                    pw,
                    scale,
                    div_pw,
                    div_ph,
                    div_c,
                )
                _ = geometry
                _ = ctx
                return
        _enqueue_cached[_pool_forward[dt, acc, fast, False]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            indices,
            indices.unsafe_bitcast[Int64](),
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            div_pw,
            div_ph,
            div_c,
        )
    else:
        var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
        var aligned = Int64(_raw_int(argv[unsafe_offset=13]))
        _enqueue_cached[_align_forward[dt, acc, fast]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            sampling,
            aligned,
            div_pw,
            div_ph,
            div_c,
        )
    _ = ctx


def _launch[
    dt: DType, pool: Bool, backward: Bool
](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("ROI kernel expects 15 argument slots")
    comptime if backward:
        _launch_backward[dt, pool](argv, argc)
    else:
        var count = (
            _raw_int(argv[unsafe_offset=8])
            * _raw_int(argv[unsafe_offset=5])
            * _raw_int(argv[unsafe_offset=9])
            * _raw_int(argv[unsafe_offset=10])
        )
        if count == 0:
            return
        var ctx = _raw_ctx(argv[unsafe_offset=14])
        var sm = _device_sm_count(ctx)
        var blocks = min(ceildiv(count, BLOCK), sm * FORWARD_BLOCKS_PER_SM)
        # Reserve the final grid-stride increment as well as every valid index.
        if count <= 2147483647 - blocks * BLOCK:
            _enqueue_forward[dt, pool, True](argv, blocks, sm)
        else:
            _enqueue_forward[dt, pool, False](argv, blocks, sm)
        _ = ctx


@always_inline
def _ps_pool_bounds[
    dt: DType, acc: DType, backward: Bool
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    roi: Int,
    n: Int,
    h: Int,
    w: Int,
    ph: Int,
    pw: Int,
    by: Int,
    bx: Int,
    scale: Scalar[acc],
) -> Tuple[Int, Int, Int, Int, Int]:
    if not _valid_roi(rois, roi, n, scale):
        return (-1, 0, 0, 0, 0)
    var batch = Int(rois[unsafe_offset=roi * 5])
    var x0 = _round_away(
        (_product(rois[unsafe_offset=roi * 5 + 1].cast[acc](), scale)).cast[
            DType.float32
        ]()
    )
    var y0 = _round_away(
        (_product(rois[unsafe_offset=roi * 5 + 2].cast[acc](), scale)).cast[
            DType.float32
        ]()
    )
    var x1 = _round_away(
        (_product(rois[unsafe_offset=roi * 5 + 3].cast[acc](), scale)).cast[
            DType.float32
        ]()
    )
    var y1 = _round_away(
        (_product(rois[unsafe_offset=roi * 5 + 4].cast[acc](), scale)).cast[
            DType.float32
        ]()
    )
    var bh = Scalar[acc](max(y1 - y0, 1)) / Scalar[acc](ph)
    var bw = Scalar[acc](max(x1 - x0, 1)) / Scalar[acc](pw)
    # Upstream PS pooling clips forward to size-1, backward to size.
    var ymax = h if backward else h - 1
    var xmax = w if backward else w - 1
    var ys = min(max(y0 + Int(floor(_product(Scalar[acc](by), bh))), 0), ymax)
    var ye = min(max(y0 + Int(ceil(Scalar[acc](by + 1) * bh)), 0), ymax)
    var xs = min(max(x0 + Int(floor(_product(Scalar[acc](bx), bw))), 0), xmax)
    var xe = min(max(x0 + Int(ceil(Scalar[acc](bx + 1) * bw)), 0), xmax)
    return (batch, ys, ye, xs, xe)


@always_inline
def _ps_add[
    dt: DType, storage: DType
](
    output: Pointer[Scalar[storage], MutAnyOrigin],
    offset: Int,
    value: Scalar[dt],
    count: Int,
):
    comptime if dt == DType.float16 and storage == DType.float32:
        var ptr = output.unsafe_offset(offset)
        var expected = Scalar[storage](0)
        while True:
            var desired = (
                (expected + value.cast[storage]()).cast[dt]().cast[storage]()
            )
            if Atomic[storage].compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
                weak=True,
            ](ptr, expected, desired):
                break
    elif dt == DType.float16 and is_nvidia_gpu():
        _pool_add_half(
            output.unsafe_offset(offset), value.cast[storage](), offset, count
        )
    else:
        _add(output.unsafe_offset(offset), value.cast[storage]())


@__name(
    "ps_roi_"
    + ("pool" if pool else "align")
    + ("_bwd_scatter_" if backward else "_fwd_")
    + String(dt)
    + ("_fastdiv" if fast else "_i64")
)
def _ps_roi[
    dt: DType, acc: DType, out_dt: DType, pool: Bool, backward: Bool, fast: Bool
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[out_dt], MutAnyOrigin],
    mapping: Pointer[Int32, MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    sampling64: Int64,
    div_pw: DivisorArgs,
    div_ph: DivisorArgs,
    div_c: DivisorArgs,
):
    comptime coord = DType.float64 if acc == DType.float64 else DType.float32
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    var co = c // (ph * pw)
    comptime idt = DType.int32 if fast else DType.int64
    var count = Scalar[idt](Int(k64) * c)
    var index = Scalar[idt](block_idx.x) * BLOCK + Scalar[idt](thread_idx.x)
    while index < count:
        var i = Int(index)
        var (bx, by, channel, roi) = _coordinates[fast](
            i, co, ph, pw, div_pw, div_ph, div_c
        )
        var ci = (channel * ph + by) * pw + bx
        comptime if backward:
            ci = Int(mapping[unsafe_offset=i])
        else:
            mapping[unsafe_offset=i] = Int32(ci)
            output[unsafe_offset=i] = 0
        if ci < 0 or ci >= c or not _valid_roi(rois, roi, n, scale):
            index += Scalar[idt](grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var base = (batch * c + ci) * h * w
        comptime if pool:
            var (_, ys, ye, xs, xe) = _ps_pool_bounds[dt, acc, backward](
                rois, roi, n, h, w, ph, pw, by, bx, scale
            )
            if ye > ys and xe > xs:
                var area = Scalar[acc]((ye - ys) * (xe - xs))
                comptime if backward:
                    var value = (
                        input[unsafe_offset=i].cast[acc]() / area
                    ).cast[acc]()
                    for y in range(ys, ye):
                        for x in range(xs, xe):
                            _ps_add(
                                output, base + y * w + x, value, n * c * h * w
                            )
                else:
                    var value = Scalar[acc](0)
                    for y in range(ys, ye):
                        for x in range(xs, xe):
                            value += input[unsafe_offset=base + y * w + x].cast[
                                acc
                            ]()
                    output[unsafe_offset=i] = (value / area).cast[out_dt]()
        else:
            var x0 = (
                _product(rois[unsafe_offset=roi * 5 + 1].cast[acc](), scale)
                - 0.5
            )
            var y0 = (
                _product(rois[unsafe_offset=roi * 5 + 2].cast[acc](), scale)
                - 0.5
            )
            var rw = (
                _product(rois[unsafe_offset=roi * 5 + 3].cast[acc](), scale)
                - 0.5
                - x0
            )
            var rh = (
                _product(rois[unsafe_offset=roi * 5 + 4].cast[acc](), scale)
                - 0.5
                - y0
            )
            var bh = rh / Scalar[acc](ph)
            var bw = rw / Scalar[acc](pw)
            var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
            var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
            var samples = Scalar[acc](gh * gw)
            var value = Scalar[acc](0)
            for iy in range(gh):
                var y = (
                    y0
                    + _product(Scalar[acc](by), bh)
                    + _product((Scalar[acc](iy) + 0.5), bh) / Scalar[acc](gh)
                )
                for ix in range(gw):
                    var x = (
                        x0
                        + _product(Scalar[acc](bx), bw)
                        + _product((Scalar[acc](ix) + 0.5), bw)
                        / Scalar[acc](gw)
                    )
                    comptime if backward:
                        if (
                            y < -1
                            or y.cast[coord]() > Scalar[coord](h)
                            or x < -1
                            or x.cast[coord]() > Scalar[coord](w)
                        ):
                            continue
                        var (yl, yh, ly) = _axis(y, h)
                        var (xl, xh, lx) = _axis(x, w)
                        var grad = input[unsafe_offset=i].cast[acc]()
                        _ps_add(
                            output,
                            base + yl * w + xl,
                            (grad * ((1 - ly) * (1 - lx)) / samples).cast[
                                acc
                            ](),
                            n * c * h * w,
                        )
                        _ps_add(
                            output,
                            base + yl * w + xh,
                            (grad * ((1 - ly) * lx) / samples).cast[acc](),
                            n * c * h * w,
                        )
                        _ps_add(
                            output,
                            base + yh * w + xl,
                            (grad * (ly * (1 - lx)) / samples).cast[acc](),
                            n * c * h * w,
                        )
                        _ps_add(
                            output,
                            base + yh * w + xh,
                            (grad * (ly * lx) / samples).cast[acc](),
                            n * c * h * w,
                        )
                    else:
                        value += _sample(input, base, y, x, h, w)
            comptime if not backward:
                output[unsafe_offset=i] = (value / samples).cast[out_dt]()
        index += Scalar[idt](grid_dim.x) * BLOCK


def _enqueue_ps[
    dt: DType, pool: Bool, backward: Bool, fast: Bool
](argv: Argv, blocks: Int, output_address: Int) raises:
    comptime acc = dt
    comptime out_dt = _scatter_storage_dtype[dt]() if backward else dt
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var rois = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var output = _make_ptr[out_dt](output_address).as_unsafe_any_origin()
    var mapping = Pointer[Int32, MutAnyOrigin](
        unsafe_from_address=_raw_int(argv[unsafe_offset=3])
    )
    var n = Int64(_raw_int(argv[unsafe_offset=4]))
    var c = Int64(_raw_int(argv[unsafe_offset=5]))
    var h = Int64(_raw_int(argv[unsafe_offset=6]))
    var w = Int64(_raw_int(argv[unsafe_offset=7]))
    var k = Int64(_raw_int(argv[unsafe_offset=8]))
    var ph = Int64(_raw_int(argv[unsafe_offset=9]))
    var pw = Int64(_raw_int(argv[unsafe_offset=10]))
    var scale = _roi_scale[acc](_raw_f64(argv[unsafe_offset=11]))
    var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var div_pw = DivisorArgs(fill=UInt32(0))
    var div_ph = DivisorArgs(fill=UInt32(0))
    var div_c = DivisorArgs(fill=UInt32(0))
    comptime if fast:
        div_pw = _divisor(Int(pw))
        div_ph = _divisor(Int(ph))
        div_c = _divisor(Int(c // (ph * pw)))
    _enqueue_cached[_ps_roi[dt, acc, out_dt, pool, backward, fast]](
        ctx,
        blocks,
        1,
        1,
        BLOCK,
        input,
        rois,
        output,
        mapping,
        n,
        c,
        h,
        w,
        k,
        ph,
        pw,
        scale,
        sampling,
        div_pw,
        div_ph,
        div_c,
    )
    _ = ctx


def _launch_ps[
    dt: DType, pool: Bool, backward: Bool
](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("PS ROI kernel expects 15 argument slots")
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var output_address = _raw_int(argv[unsafe_offset=2])
    comptime if backward:
        var output_count = (
            _raw_int(argv[unsafe_offset=4])
            * _raw_int(argv[unsafe_offset=5])
            * _raw_int(argv[unsafe_offset=6])
            * _raw_int(argv[unsafe_offset=7])
        )
        if output_count == 0:
            return
        var output = _make_ptr[dt](output_address).as_unsafe_any_origin()
        var buffer = _scatter_buffer(ctx, output, output_count)
        output_address = Int(buffer.unsafe_ptr())
        _dispatch_ps[dt, pool, backward](argv, output_address)
        _finish_scatter(ctx, buffer, output, output_count)
        _ = buffer
    else:
        _dispatch_ps[dt, pool, backward](argv, output_address)
    _ = ctx


def _dispatch_ps[
    dt: DType, pool: Bool, backward: Bool
](argv: Argv, output_address: Int) raises:
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var count = _raw_int(argv[unsafe_offset=8]) * _raw_int(
        argv[unsafe_offset=5]
    )
    if count == 0:
        return
    var cap = BACKWARD_BLOCKS_PER_SM if backward else FORWARD_BLOCKS_PER_SM
    var blocks = min(ceildiv(count, BLOCK), _device_sm_count(ctx) * cap)
    if count <= 2147483647 - blocks * BLOCK:
        _enqueue_ps[dt, pool, backward, True](argv, blocks, output_address)
    else:
        _enqueue_ps[dt, pool, backward, False](argv, blocks, output_address)
    _ = ctx


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime for dt in [DType.float16, DType.float32, DType.float64]:
            comptime if _dtype_arg_on[0, dt]():
                comptime if _op_on["RoiAlignForward"]():
                    _launch[dt, False, False](argv, argc)
                    return 0
                elif _op_on["RoiAlignBackward"]():
                    _launch[dt, False, True](argv, argc)
                    return 0
                elif _op_on["RoiPoolForward"]():
                    _launch[dt, True, False](argv, argc)
                    return 0
                elif _op_on["RoiPoolBackward"]():
                    _launch[dt, True, True](argv, argc)
                    return 0
                elif _op_on["PsRoiAlignForward"]():
                    _launch_ps[dt, False, False](argv, argc)
                    return 0
                elif _op_on["PsRoiAlignBackward"]():
                    _launch_ps[dt, False, True](argv, argc)
                    return 0
                elif _op_on["PsRoiPoolForward"]():
                    _launch_ps[dt, True, False](argv, argc)
                    return 0
                elif _op_on["PsRoiPoolBackward"]():
                    _launch_ps[dt, True, True](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
