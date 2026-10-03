# ===----------------------------------------------------------------------=== #
# Pooling, unpooling and im2col / col2im for the mojo device.
#
# One generic kernel per family, written for three spatial dims (D, H, W):
# the 2-D ops pass D = 1 with a unit window (kernel 1, stride 1, padding 0,
# dilation 1), and ATen's 1-D ops are composites over the 2-D ones. Every
# kernel is a parallel-for over independent elements with fully dynamic
# shapes, over contiguous (N*C, D, H, W) planes.
#
# Window and index math follow torch's CUDA kernels (aten/src/ATen/native/
# cuda at v2.14.0): DilatedMaxPool{2,3}d.cu, FractionalMaxPool{2,3}d.cu,
# AveragePool{2,3}d.cu,
# AdaptiveAveragePooling{,3d}.cu, AdaptiveMaxPooling{2,3}d.cu,
# MaxUnpooling.cu and im2col.cuh. Max pooling lets NaN win (the last NaN of
# the window, as CUDA's `val > max || isnan(val)`), and its index starts at
# the window's first in-bounds element. Half inputs accumulate in float32,
# float64 in float64.
#
# The average backwards and the 2-D max-pool backward are deterministic
# gathers: one thread owns one input element and sums the outputs whose
# window can hold it (the inverse window range per dim). For max pool 2-D
# that is exactly CUDA's own backward (max_pool_backward_nchw), which also
# ignores a saved index outside its output's window. The 3-D and adaptive
# max-pool backwards scatter every grad_output element to its saved index,
# as CUDA does, so arbitrary indices accumulate like torch's, with atomics
# in the tensor's own dtype (CUDA's atomicAdd: a half accumulator rounds on
# every add; 16-bit adds are a compare-and-swap loop on the 32-bit word
# that holds them, or on Metal, where that loop never completes, on a
# float32 word per element that the op casts back). The average backwards reproduce CUDA's per-kernel rounding (see
# `_avg_pool_backward`).
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.atomic import Atomic, Ordering
from std.memory import bitcast
from std.sys import is_amd_gpu, is_nvidia_gpu, size_of
from std.sys.info import has_apple_gpu_accelerator
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.numerics import min_or_neg_inf

from max.gpu import block_idx, grid_dim, thread_idx

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _parallel_for_dt,
    _raw_ctx,
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

comptime POOL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]
# max_unpool moves every non-bool type (CUDA's AT_DISPATCH_ALL_TYPES_AND2).
comptime UNPOOL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.uint8,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
]
# im2col / col2im also move bool (CUDA's Im2Col.cu / Col2Im.cu dispatch it).
comptime FOLD_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.bool,
]

# Geometry slots of the pooling kernels (the op's params tuple, in order).
comptime G_PLANES = 0  # N * C
comptime G_IN = 1  # iD, iH, iW
comptime G_OUT = 4  # oD, oH, oW
comptime G_K = 7  # kernel (unused by the adaptive kernels)
comptime G_S = 10  # stride
comptime G_P = 13  # padding
comptime G_DIL = 16  # dilation
comptime G_CIP = 19  # avg: count_include_pad
comptime G_DIV = 20  # avg: divisor_override (0 = none)
comptime G_3D = 21  # 1 for the 3-D ops (the adaptive-avg divisor order)
comptime G_LEN = 22
comptime Geom = IndexList[G_LEN]


@always_inline
def _acc[dtype: DType]() -> DType:
    """torch's `acc_type<scalar_t, /*is_cuda=*/true>`."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@always_inline
def _start_index(a: Int, b: Int, c: Int) -> Int:
    """AdaptivePooling.h `start_index`: floor(a * c / b)."""
    return (a // b) * c + ((a % b) * c) // b


@always_inline
def _end_index(a: Int, b: Int, c: Int) -> Int:
    """AdaptivePooling.h `end_index`: ceil((a + 1) * c / b)."""
    return 1 + ((a + 1) * c - 1) // b


@always_inline
def _max_window(g: Geom, dim: Int, o: Int) -> Tuple[Int, Int]:
    """Dilated max-pool window of output `o` along `dim`: the first
    in-bounds tap and the (exclusive) end, stepped by the dilation."""
    var s = o * g[G_S + dim] - g[G_P + dim]
    var e = min(s + (g[G_K + dim] - 1) * g[G_DIL + dim] + 1, g[G_IN + dim])
    while s < 0:
        s += g[G_DIL + dim]
    return (s, e)


@always_inline
def _max_out_range(g: Geom, dim: Int, x: Int) -> Tuple[Int, Int]:
    """Outputs whose dilated window can hold input `x` (CUDA's p_start /
    p_end); the backward still matches the saved index."""
    var extent = (g[G_K + dim] - 1) * g[G_DIL + dim] + 1
    var xp = x + g[G_P + dim]
    var s = 0 if xp < extent else (xp - extent) // g[G_S + dim] + 1
    var e = min(xp // g[G_S + dim] + 1, g[G_OUT + dim])
    return (s, e)


@always_inline
def _avg_window(g: Geom, dim: Int, o: Int) -> Tuple[Int, Int, Int]:
    """Average-pool window of output `o` along `dim`: (start, end, padded
    extent). The extent counts padding (count_include_pad's divisor), the
    bounds are clamped to the input."""
    var s = o * g[G_S + dim] - g[G_P + dim]
    var e = min(s + g[G_K + dim], g[G_IN + dim] + g[G_P + dim])
    var extent = e - s
    return (max(s, 0), min(e, g[G_IN + dim]), extent)


@always_inline
def _avg_out_range(g: Geom, dim: Int, x: Int) -> Tuple[Int, Int]:
    """Outputs whose average window holds input `x` (AveragePool2d.cu's
    backward phstart / phend)."""
    var xp = x + g[G_P + dim]
    var k = g[G_K + dim]
    var s = 0 if xp < k else (xp - k) // g[G_S + dim] + 1
    var e = min(xp // g[G_S + dim] + 1, g[G_OUT + dim])
    return (s, e)


@always_inline
def _adaptive_window(g: Geom, dim: Int, o: Int) -> Tuple[Int, Int]:
    var isz = g[G_IN + dim]
    var osz = g[G_OUT + dim]
    return (_start_index(o, osz, isz), _end_index(o, osz, isz))


@always_inline
def _adaptive_out_range(g: Geom, dim: Int, x: Int) -> Tuple[Int, Int]:
    """Exactly the outputs whose adaptive window holds input `x`."""
    var isz = g[G_IN + dim]
    var osz = g[G_OUT + dim]
    return (_start_index(x, isz, osz), _end_index(x, isz, osz))


@always_inline
def _geom(params: Arg) raises -> Geom:
    if _raw_tuple_len(params) != G_LEN:
        raise Error("pool: expected ", G_LEN, " geometry slots")
    var g = Geom(0)
    for i in range(G_LEN):
        g[i] = _raw_tuple_int(params, i)
    return g


@always_inline
def _split_out(g: Geom, i: Int) -> Tuple[Int, Int, Int, Int]:
    """(plane, od, oh, ow) of flat output element `i`."""
    var ow = i % g[G_OUT + 2]
    var oh = (i // g[G_OUT + 2]) % g[G_OUT + 1]
    var od = (i // (g[G_OUT + 2] * g[G_OUT + 1])) % g[G_OUT]
    var plane = i // (g[G_OUT + 2] * g[G_OUT + 1] * g[G_OUT])
    return (plane, od, oh, ow)


@always_inline
def _split_in(g: Geom, i: Int) -> Tuple[Int, Int, Int, Int]:
    """(plane, id, ih, iw) of flat input element `i`."""
    var iw = i % g[G_IN + 2]
    var ih = (i // g[G_IN + 2]) % g[G_IN + 1]
    var id = (i // (g[G_IN + 2] * g[G_IN + 1])) % g[G_IN]
    var plane = i // (g[G_IN + 2] * g[G_IN + 1] * g[G_IN])
    return (plane, id, ih, iw)


@always_inline
def _in_plane(g: Geom) -> Int:
    return g[G_IN] * g[G_IN + 1] * g[G_IN + 2]


@always_inline
def _out_plane(g: Geom) -> Int:
    return g[G_OUT] * g[G_OUT + 1] * g[G_OUT + 2]


# ---------------------------------------------------------------------------
# Max pooling (dilated or adaptive), with int64 indices into the flattened
# (D, H, W) input plane.
# ---------------------------------------------------------------------------


def _max_pool[
    dtype: DType, adaptive: Bool
](
    out_addr: Int, idx_addr: Int, in_addr: Int, g: Geom, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, idx_ptr, in_ptr, g)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = _split_out(g, i)
        var wd: Tuple[Int, Int]
        var wh: Tuple[Int, Int]
        var ww: Tuple[Int, Int]
        var step_d = 1
        var step_h = 1
        var step_w = 1
        comptime if adaptive:
            wd = _adaptive_window(g, 0, o[1])
            wh = _adaptive_window(g, 1, o[2])
            ww = _adaptive_window(g, 2, o[3])
        else:
            wd = _max_window(g, 0, o[1])
            wh = _max_window(g, 1, o[2])
            ww = _max_window(g, 2, o[3])
            step_d = g[G_DIL]
            step_h = g[G_DIL + 1]
            step_w = g[G_DIL + 2]
        var in_h = g[G_IN + 1]
        var in_w = g[G_IN + 2]
        var base = o[0] * _in_plane(g)
        var best = min_or_neg_inf[dtype]()
        var best_idx = (wd[0] * in_h + wh[0]) * in_w + ww[0]
        var d = wd[0]
        while d < wd[1]:
            var h = wh[0]
            while h < wh[1]:
                var row = (d * in_h + h) * in_w
                var w = ww[0]
                while w < ww[1]:
                    var v = in_ptr[unsafe_offset=base + row + w]
                    if v > best or v != v:
                        best = v
                        best_idx = row + w
                    w += step_w
                h += step_h
            d += step_d
        out_ptr[unsafe_offset=i] = best
        idx_ptr[unsafe_offset=i] = Int64(best_idx)

    _parallel_for_dt[dtype, func](g[G_PLANES] * _out_plane(g), ctx)


def _max_pool_backward[
    dtype: DType
](
    gin_addr: Int, gout_addr: Int, idx_addr: Int, g: Geom, ctx: DeviceContext
) raises:
    """grad_input[x] = sum of grad_output[o] over the outputs whose saved
    index is x, gathered per input element."""
    var gin_ptr = _make_ptr[dtype](gin_addr)
    var gout_ptr = _make_ptr[dtype](gout_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    comptime acc_t = _acc[dtype]()

    @always_inline
    @__parameter
    @__copy_capture(gin_ptr, gout_ptr, idx_ptr, g)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var x = _split_in(g, i)
        var rd = _max_out_range(g, 0, x[1])
        var rh = _max_out_range(g, 1, x[2])
        var rw = _max_out_range(g, 2, x[3])
        var out_h = g[G_OUT + 1]
        var out_w = g[G_OUT + 2]
        var me = Int64((x[1] * g[G_IN + 1] + x[2]) * g[G_IN + 2] + x[3])
        var base = x[0] * _out_plane(g)
        var total = Scalar[acc_t](0)
        for od in range(rd[0], rd[1]):
            for oh in range(rh[0], rh[1]):
                var row = base + (od * out_h + oh) * out_w
                for ow in range(rw[0], rw[1]):
                    if idx_ptr[unsafe_offset=row + ow] == me:
                        total += gout_ptr[unsafe_offset=row + ow].cast[acc_t]()
        gin_ptr[unsafe_offset=i] = total.cast[dtype]()

    _parallel_for_dt[dtype, func](g[G_PLANES] * _in_plane(g), ctx)


# ---------------------------------------------------------------------------
# Fractional max pooling (FractionalMaxPool{2,3}d.cu): pseudo-random window
# starts from one sample in [0, 1) per (plane, axis), int64 indices into the
# flattened (D, H, W) input plane. The backward is the scatter below.
# ---------------------------------------------------------------------------


@always_inline
def _frac_interval[
    acc_t: DType
](
    sample: Scalar[acc_t], index: Int, in_size: Int, out_size: Int, pool: Int
) -> Int:
    """get_interval(s): the window start of output `index`, the last one
    flush with the input's end."""
    if index == out_size - 1:
        return in_size - pool
    var alpha = Scalar[acc_t](in_size - pool) / Scalar[acc_t](out_size - 1)
    return Int((Scalar[acc_t](index) + sample) * alpha) - Int(sample * alpha)


def _fractional_max_pool[
    dtype: DType
](
    out_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    samples_addr: Int,
    g: Geom,
    ctx: DeviceContext,
) raises:
    """`samples` is the contiguous (N, C, 2 | 3) `_random_samples`: plane p
    reads row p, (W, H) for the 2-D op, (T, H, W) for the 3-D one."""
    var out_ptr = _make_ptr[dtype](out_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var smp_ptr = _make_ptr[dtype](samples_addr)
    comptime acc_t = _acc[dtype]()

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, idx_ptr, in_ptr, smp_ptr, g)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = _split_out(g, i)
        var three = g[G_3D] != 0
        var row = o[0] * (3 if three else 2)
        var sd = Scalar[acc_t](0)
        var sh: Scalar[acc_t]
        var sw: Scalar[acc_t]
        if three:
            sd = smp_ptr[unsafe_offset=row].cast[acc_t]()
            sh = smp_ptr[unsafe_offset=row + 1].cast[acc_t]()
            sw = smp_ptr[unsafe_offset=row + 2].cast[acc_t]()
        else:
            sw = smp_ptr[unsafe_offset=row].cast[acc_t]()
            sh = smp_ptr[unsafe_offset=row + 1].cast[acc_t]()
        var in_d = g[G_IN]
        var in_h = g[G_IN + 1]
        var in_w = g[G_IN + 2]
        var pd = _frac_interval[acc_t](sd, o[1], in_d, g[G_OUT], g[G_K])
        var ph = _frac_interval[acc_t](sh, o[2], in_h, g[G_OUT + 1], g[G_K + 1])
        var pw = _frac_interval[acc_t](sw, o[3], in_w, g[G_OUT + 2], g[G_K + 2])
        var base = o[0] * _in_plane(g)
        var best = min_or_neg_inf[dtype]()
        var best_idx = (pd * in_h + ph) * in_w + pw
        for d in range(pd, pd + g[G_K]):
            for h in range(ph, ph + g[G_K + 1]):
                for w in range(pw, pw + g[G_K + 2]):
                    # A sample outside [0, 1) can push a window off the
                    # input (CUDA reads out of bounds): skip those taps.
                    if (
                        d < 0
                        or d >= in_d
                        or h < 0
                        or h >= in_h
                        or w < 0
                        or w >= in_w
                    ):
                        continue
                    var at = (d * in_h + h) * in_w + w
                    var v = in_ptr[unsafe_offset=base + at]
                    if v > best or v != v:
                        best = v
                        best_idx = at
        out_ptr[unsafe_offset=i] = best
        idx_ptr[unsafe_offset=i] = Int64(best_idx)

    _parallel_for_dt[dtype, func](g[G_PLANES] * _out_plane(g), ctx)


@always_inline
def _scope() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _atomic_add[
    dtype: DType
](ptr: Pointer[Scalar[dtype], MutAnyOrigin], value: Scalar[dtype]):
    """Relaxed `*ptr += value` rounded in `dtype`, as CUDA's atomicAdd.

    A 16-bit dtype (NVIDIA / AMD; Metal takes `_metal_half`'s float32 words
    instead) swaps the aligned 32-bit word that holds it: the other half of
    the word is carried over unchanged, and a concurrent update of it only
    makes the swap retry."""
    comptime if size_of[Scalar[dtype]]() == 2:
        var addr = Int(ptr)
        var word = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=addr & ~3)
        var shift = UInt32((addr & 2) * 8)
        var expected = word[]
        while True:
            var bits = UInt16((expected >> shift) & 0xFFFF)
            var sum = (
                bitcast[dtype](bits).cast[DType.float32]()
                + value.cast[DType.float32]()
            ).cast[dtype]()
            var desired = (expected & ~(UInt32(0xFFFF) << shift)) | (
                UInt32(bitcast[DType.uint16](sum)) << shift
            )
            if Atomic[UInt32, scope=_scope()].compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
                weak=True,
            ](word, expected, desired):
                return
    else:
        _ = Atomic[Scalar[dtype], scope=_scope()].fetch_add[
            ordering=Ordering.RELAXED
        ](ptr, value)


@always_inline
def _metal_half[dtype: DType]() -> Bool:
    """Whether the scatter goes through a float32 word per element: a 16-bit
    dtype on an Apple GPU (Metal has no 16-bit atomics, and the 32-bit-word
    swap of `_atomic_add` never completes there)."""
    return has_apple_gpu_accelerator() and size_of[Scalar[dtype]]() == 2


comptime SCATTER_BLOCK = 256


@__name("max_pool_bwd_scatter_" + String(dtype))
def _max_pool_scatter_kernel[
    dtype: DType, storage: DType
](
    gin: Pointer[Scalar[storage], MutAnyOrigin],
    gout: Pointer[Scalar[dtype], MutAnyOrigin],
    indices: Pointer[Int64, MutAnyOrigin],
    count: Int64,
    out_plane: Int64,
    in_plane: Int64,
):
    """A grid-stride kernel, not an elementwise closure: Metal's compiler
    rejects atomics in the closure form (the ROI scatters launch the same
    way)."""
    var i = Int(block_idx.x) * SCATTER_BLOCK + Int(thread_idx.x)
    var step = Int(grid_dim.x) * SCATTER_BLOCK
    while i < Int(count):
        var target = indices[unsafe_offset=i]
        if target >= 0 and target < in_plane:
            var at = (i // Int(out_plane)) * Int(in_plane) + Int(target)
            var value = gout[unsafe_offset=i]
            comptime if storage != dtype:
                # One float32 word per half value, rounded to the half dtype
                # on every add.
                var word = gin.unsafe_offset(at)
                var add = value.cast[storage]()
                var expected = Scalar[storage](0)
                while True:
                    var desired = (expected + add).cast[dtype]().cast[storage]()
                    if Atomic[Scalar[storage]].compare_exchange[
                        success_ordering=Ordering.RELAXED,
                        failure_ordering=Ordering.RELAXED,
                        weak=True,
                    ](word, expected, desired):
                        break
            else:
                _atomic_add[dtype](
                    rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                        gin.unsafe_offset(at)
                    ),
                    value,
                )
        i += step


def _max_pool_scatter[
    dtype: DType
](
    gin_addr: Int,
    gout_addr: Int,
    idx_addr: Int,
    count: Int,
    out_plane: Int,
    in_plane: Int,
    ctx: DeviceContext,
) raises:
    """gin[plane][indices[i]] += grad_output[i] over every grad_output
    element into the zeroed `gin`, atomically in `dtype` (CUDA's atomic
    max-pool backwards). An index outside the input plane is skipped rather
    than written. Under `_metal_half`, `gin` is a zeroed float32 buffer, each
    word holding one `dtype` value, rounded to `dtype` on every add (the
    ROI kernels' Metal scatter)."""
    comptime storage = DType.float32 if _metal_half[dtype]() else dtype
    var blocks = min((count + SCATTER_BLOCK - 1) // SCATTER_BLOCK, 65535)
    _enqueue_cached[_max_pool_scatter_kernel[dtype, storage]](
        ctx,
        blocks,
        1,
        1,
        SCATTER_BLOCK,
        Pointer[Scalar[storage], MutAnyOrigin](unsafe_from_address=gin_addr),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=gout_addr),
        Pointer[Int64, MutAnyOrigin](unsafe_from_address=idx_addr),
        Int64(count),
        Int64(out_plane),
        Int64(in_plane),
    )


# ---------------------------------------------------------------------------
# Average pooling (padded / ceil-mode windows, or adaptive).
# ---------------------------------------------------------------------------


@always_inline
def _avg_divisor(
    g: Geom,
    wd: Tuple[Int, Int, Int],
    wh: Tuple[Int, Int, Int],
    ww: Tuple[Int, Int, Int],
) -> Int:
    if g[G_DIV] != 0:
        return g[G_DIV]
    if g[G_CIP] != 0:
        return wd[2] * wh[2] * ww[2]
    return (wd[1] - wd[0]) * (wh[1] - wh[0]) * (ww[1] - ww[0])


@always_inline
def _adaptive_scale[
    acc_t: DType
](v: Scalar[acc_t], g: Geom, kd: Int, kh: Int, kw: Int) -> Scalar[acc_t]:
    """The adaptive average's division: 2-D divides by kH then kW
    (AdaptiveAveragePooling.cu), 3-D by the window volume
    (AdaptiveAveragePooling3d.cu)."""
    if g[G_3D] != 0:
        return v / Scalar[acc_t](kd * kh * kw)
    return v / Scalar[acc_t](kh) / Scalar[acc_t](kw)


def _avg_pool[
    dtype: DType, adaptive: Bool
](out_addr: Int, in_addr: Int, g: Geom, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    comptime acc_t = _acc[dtype]()

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, g)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = _split_out(g, i)
        var wd: Tuple[Int, Int, Int]
        var wh: Tuple[Int, Int, Int]
        var ww: Tuple[Int, Int, Int]
        comptime if adaptive:
            var ad = _adaptive_window(g, 0, o[1])
            var ah = _adaptive_window(g, 1, o[2])
            var aw = _adaptive_window(g, 2, o[3])
            wd = (ad[0], ad[1], 0)
            wh = (ah[0], ah[1], 0)
            ww = (aw[0], aw[1], 0)
        else:
            wd = _avg_window(g, 0, o[1])
            wh = _avg_window(g, 1, o[2])
            ww = _avg_window(g, 2, o[3])
            if wd[0] >= wd[1] or wh[0] >= wh[1] or ww[0] >= ww[1]:
                # Window entirely in padding: CUDA writes 0.
                out_ptr[unsafe_offset=i] = Scalar[dtype](0)
                return
        var in_h = g[G_IN + 1]
        var in_w = g[G_IN + 2]
        var base = o[0] * _in_plane(g)
        var total = Scalar[acc_t](0)
        for d in range(wd[0], wd[1]):
            for h in range(wh[0], wh[1]):
                var row = base + (d * in_h + h) * in_w
                for w in range(ww[0], ww[1]):
                    total += in_ptr[unsafe_offset=row + w].cast[acc_t]()
        comptime if adaptive:
            out_ptr[unsafe_offset=i] = _adaptive_scale(
                total, g, wd[1] - wd[0], wh[1] - wh[0], ww[1] - ww[0]
            ).cast[dtype]()
        else:
            var div = _avg_divisor(g, wd, wh, ww)
            out_ptr[unsafe_offset=i] = (total / Scalar[acc_t](div)).cast[
                dtype
            ]()

    _parallel_for_dt[dtype, func](g[G_PLANES] * _out_plane(g), ctx)


@always_inline
def _div_t[dtype: DType](a: Scalar[dtype], b: Int) -> Scalar[dtype]:
    """`scalar_t / int` in c10: the int converts to `scalar_t`, and a half
    quotient rounds to half."""
    comptime acc_t = _acc[dtype]()
    return (a.cast[acc_t]() / Scalar[dtype](b).cast[acc_t]()).cast[dtype]()


@always_inline
def _add_t[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """`scalar_t += scalar_t` (a half sum rounds to half)."""
    comptime acc_t = _acc[dtype]()
    return (a.cast[acc_t]() + b.cast[acc_t]()).cast[dtype]()


def _avg_pool_backward[
    dtype: DType, adaptive: Bool
](gin_addr: Int, gout_addr: Int, g: Geom, ctx: DeviceContext) raises:
    """Gather per input element, rounding the way CUDA's kernel for the
    same case does (a half accumulator in place of its atomicAdd, summed in
    one fixed order):

    * avg_pool2d (AveragePool2d.cu): float sum of `grad / divisor`, each
      quotient a `scalar_t / int`, rounded once.
    * avg_pool3d (AveragePool3d.cu): stride 1 and no padding, a float sum
      scaled by `1 / divisor`, rounded once; otherwise each
      `scalar_t(float(grad) / divisor)` added in `scalar_t`.
    * adaptive 2-D (AdaptiveAveragePooling.cu, always atomic): each
      `grad / kW / kH` in `scalar_t`, added in `scalar_t`.
    * adaptive 3-D (AdaptiveAveragePooling3d.cu): when a size does not
      divide, each `grad / kT / kH / kW` in `scalar_t`; otherwise each
      `scalar_t(float(grad) / (kT * kH * kW))`; added in `scalar_t`.
    """
    var gin_ptr = _make_ptr[dtype](gin_addr)
    var gout_ptr = _make_ptr[dtype](gout_addr)
    comptime acc_t = _acc[dtype]()

    @always_inline
    @__parameter
    @__copy_capture(gin_ptr, gout_ptr, g)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var x = _split_in(g, i)
        var rd: Tuple[Int, Int]
        var rh: Tuple[Int, Int]
        var rw: Tuple[Int, Int]
        comptime if adaptive:
            rd = _adaptive_out_range(g, 0, x[1])
            rh = _adaptive_out_range(g, 1, x[2])
            rw = _adaptive_out_range(g, 2, x[3])
        else:
            rd = _avg_out_range(g, 0, x[1])
            rh = _avg_out_range(g, 1, x[2])
            rw = _avg_out_range(g, 2, x[3])
        var three = g[G_3D] != 0
        var stride1 = (
            g[G_S] == 1
            and g[G_S + 1] == 1
            and g[G_S + 2] == 1
            and g[G_P] == 0
            and g[G_P + 1] == 0
            and g[G_P + 2] == 0
        )
        var divisible = (
            g[G_IN] % g[G_OUT] == 0
            and g[G_IN + 1] % g[G_OUT + 1] == 0
            and g[G_IN + 2] % g[G_OUT + 2] == 0
        )
        var out_h = g[G_OUT + 1]
        var out_w = g[G_OUT + 2]
        var base = x[0] * _out_plane(g)
        var wide = Scalar[acc_t](0)  # float accumulator
        var narrow = Scalar[dtype](0)  # scalar_t accumulator
        for od in range(rd[0], rd[1]):
            for oh in range(rh[0], rh[1]):
                var row = base + (od * out_h + oh) * out_w
                for ow in range(rw[0], rw[1]):
                    var v = gout_ptr[unsafe_offset=row + ow]
                    comptime if adaptive:
                        var ad = _adaptive_window(g, 0, od)
                        var ah = _adaptive_window(g, 1, oh)
                        var aw = _adaptive_window(g, 2, ow)
                        var kt = ad[1] - ad[0]
                        var kh = ah[1] - ah[0]
                        var kw = aw[1] - aw[0]
                        var delta: Scalar[dtype]
                        if not three:
                            delta = _div_t(_div_t(v, kw), kh)
                        elif not divisible:
                            delta = _div_t(_div_t(_div_t(v, kt), kh), kw)
                        else:
                            delta = (
                                v.cast[acc_t]() / Scalar[acc_t](kt * kh * kw)
                            ).cast[dtype]()
                        narrow = _add_t(narrow, delta)
                    else:
                        var wd = _avg_window(g, 0, od)
                        var wh = _avg_window(g, 1, oh)
                        var ww = _avg_window(g, 2, ow)
                        if wd[0] >= wd[1] or wh[0] >= wh[1] or ww[0] >= ww[1]:
                            continue
                        var div = _avg_divisor(g, wd, wh, ww)
                        if not three:
                            wide += _div_t(v, div).cast[acc_t]()
                        elif stride1:
                            wide += v.cast[acc_t]()
                        else:
                            narrow = _add_t(
                                narrow,
                                (v.cast[acc_t]() / Scalar[acc_t](div)).cast[
                                    dtype
                                ](),
                            )
        var result: Scalar[dtype]
        comptime if adaptive:
            result = narrow
        else:
            if not three:
                result = wide.cast[dtype]()
            elif stride1:
                # Every window is whole: one divisor for all of them.
                var div = g[G_DIV] if g[G_DIV] != 0 else (
                    g[G_K] * g[G_K + 1] * g[G_K + 2]
                )
                result = (wide * (Scalar[acc_t](1) / Scalar[acc_t](div))).cast[
                    dtype
                ]()
            else:
                result = narrow
        gin_ptr[unsafe_offset=i] = result

    _parallel_for_dt[dtype, func](g[G_PLANES] * _in_plane(g), ctx)


# ---------------------------------------------------------------------------
# Max unpooling: out (zero-filled by the op) [plane][indices[i]] = in[i].
# An index outside the output plane is not written: the kernel records it in
# `flag` ([1, index]) and the op raises the CPU's "Found an invalid max
# index" (CUDA device-asserts). Duplicate indices race exactly as on CUDA.
# ---------------------------------------------------------------------------


def _max_unpool[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    idx_addr: Int,
    count: Int,
    in_plane: Int,
    out_plane: Int,
    flag_addr: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var flag_ptr = _make_ptr[DType.int64](flag_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, idx_ptr, flag_ptr, in_plane, out_plane)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var target = idx_ptr[unsafe_offset=i]
        if target >= 0 and Int(target) < out_plane:
            out_ptr[
                unsafe_offset=(i // in_plane) * out_plane + Int(target)
            ] = in_ptr[unsafe_offset=i]
        else:
            flag_ptr[unsafe_offset=1] = target
            flag_ptr[unsafe_offset=0] = 1

    _parallel_for_dt[dtype, func](count, ctx)


# ---------------------------------------------------------------------------
# im2col / col2im, batch-major (N, C*KH*KW, L) columns as aten::im2col
# returns them (nn.Unfold / nn.Fold). im2col.cuh's index math; col2im is
# its gather form and accumulates in float (bool: logical or, CUDA's bool
# accumulate type).
# Slots: (channels, in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh,
# dw, batch); in/out are the image's and the sliding-block grid's sizes.
# ---------------------------------------------------------------------------

comptime F_LEN = 14
comptime Fold = IndexList[F_LEN]


@always_inline
def _fold(params: Arg) raises -> Fold:
    if _raw_tuple_len(params) != F_LEN:
        raise Error("im2col: expected ", F_LEN, " geometry slots")
    var f = Fold(0)
    for i in range(F_LEN):
        f[i] = _raw_tuple_int(params, i)
    return f


def _im2col[
    dtype: DType
](out_addr: Int, in_addr: Int, f: Fold, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, f)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var channels = f[0]
        var in_h = f[1]
        var in_w = f[2]
        var out_w = f[4]
        var kh = f[5]
        var kw = f[6]
        var cols = f[3] * out_w
        var j = i % cols
        var r = (i // cols) % (channels * kh * kw)
        var s = i // (cols * channels * kh * kw)
        var fw = r % kw
        var fh = (r // kw) % kh
        var c = r // (kw * kh)
        var ih = (j // out_w) * f[7] - f[9] + fh * f[11]
        var iw = (j % out_w) * f[8] - f[10] + fw * f[12]
        if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
            out_ptr[unsafe_offset=i] = Scalar[dtype](0)
        else:
            out_ptr[unsafe_offset=i] = in_ptr[
                unsafe_offset=((s * channels + c) * in_h + ih) * in_w + iw
            ]

    _parallel_for_dt[dtype, func](f[13] * f[0] * f[5] * f[6] * f[3] * f[4], ctx)


def _col2im[
    dtype: DType
](out_addr: Int, in_addr: Int, f: Fold, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    # bool sums its 0/1 taps in float and stores "any": CUDA's bool
    # accumulate type ORs.
    comptime acc_t = _acc[dtype]()

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, f)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var in_h = f[1]
        var in_w = f[2]
        var out_h = f[3]
        var out_w = f[4]
        var kh = f[5]
        var kw = f[6]
        var sh = f[7]
        var sw = f[8]
        var dh = f[11]
        var dw = f[12]
        var w_im = i % in_w + f[10]
        var h_im = (i // in_w) % in_h + f[9]
        var c_im = i // (in_w * in_h)  # sample * channels + channel
        var ext_w = (kw - 1) * dw + 1
        var ext_h = (kh - 1) * dh + 1
        var w0 = 0 if w_im < ext_w else (w_im - ext_w) // sw + 1
        var w1 = min(w_im // sw + 1, out_w)
        var h0 = 0 if h_im < ext_h else (h_im - ext_h) // sh + 1
        var h1 = min(h_im // sh + 1, out_h)
        var total = Scalar[acc_t](0)
        for h_col in range(h0, h1):
            var h_k = h_im - h_col * sh
            if h_k % dh != 0:
                continue
            h_k //= dh
            for w_col in range(w0, w1):
                var w_k = w_im - w_col * sw
                if w_k % dw != 0:
                    continue
                w_k //= dw
                var at = (
                    ((c_im * kh + h_k) * kw + w_k) * out_h + h_col
                ) * out_w + w_col
                total += in_ptr[unsafe_offset=at].cast[acc_t]()
        out_ptr[unsafe_offset=i] = total.cast[dtype]()  # bool: nonzero

    _parallel_for_dt[dtype, func](f[13] * f[0] * f[1] * f[2], ctx)


# ---------------------------------------------------------------------------
# C entry. Slots per op (pointers are data addresses, offset applied):
#   MaxPool / AdaptiveMaxPool:          out, indices, input, geom, ctx
#   FractionalMaxPool:                  out, indices, input, samples, geom,
#                                       ctx
#   MaxPoolBackward (2-D gather):       grad_in, grad_out, indices, geom, ctx
#   MaxPoolScatter:                     workspace, grad_out, indices, count,
#                                       out_plane, in_plane, ctx
#   AvgPool / AdaptiveAvgPool:          out, input, geom, ctx
#   AvgPoolBackward / Adaptive...:      grad_in, grad_out, geom, ctx
#   MaxUnpool:                          out, input, indices, count,
#                                       in_plane, out_plane, flag, ctx
#   Im2col / Col2im:                    out, input, fold, ctx
# ---------------------------------------------------------------------------


def _launch[dtype: DType](argv: Argv, argc: Int) raises:
    comptime if _op_on["MaxPool"]() or _op_on["AdaptiveMaxPool"]():
        _max_pool[dtype, _op_on["AdaptiveMaxPool"]()](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _geom(argv[unsafe_offset=3]),
            _raw_ctx(argv[unsafe_offset=4]),
        )
    elif _op_on["FractionalMaxPool"]():
        _fractional_max_pool[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _geom(argv[unsafe_offset=4]),
            _raw_ctx(argv[unsafe_offset=5]),
        )
    elif _op_on["MaxPoolBackward"]():
        _max_pool_backward[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _geom(argv[unsafe_offset=3]),
            _raw_ctx(argv[unsafe_offset=4]),
        )
    elif _op_on["MaxPoolScatter"]():
        _max_pool_scatter[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            _raw_int(argv[unsafe_offset=5]),
            _raw_ctx(argv[unsafe_offset=6]),
        )
    elif _op_on["AvgPool"]() or _op_on["AdaptiveAvgPool"]():
        _avg_pool[dtype, _op_on["AdaptiveAvgPool"]()](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _geom(argv[unsafe_offset=2]),
            _raw_ctx(argv[unsafe_offset=3]),
        )
    elif _op_on["AvgPoolBackward"]() or _op_on["AdaptiveAvgPoolBackward"]():
        _avg_pool_backward[dtype, _op_on["AdaptiveAvgPoolBackward"]()](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _geom(argv[unsafe_offset=2]),
            _raw_ctx(argv[unsafe_offset=3]),
        )
    elif _op_on["MaxUnpool"]():
        _max_unpool[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _raw_int(argv[unsafe_offset=2]),
            _raw_int(argv[unsafe_offset=3]),
            _raw_int(argv[unsafe_offset=4]),
            _raw_int(argv[unsafe_offset=5]),
            _raw_int(argv[unsafe_offset=6]),
            _raw_ctx(argv[unsafe_offset=7]),
        )
    else:
        raise Error(NO_OP_COMPILED)


def _launch_fold[dtype: DType](argv: Argv, argc: Int) raises:
    comptime if _op_on["Im2col"]():
        _im2col[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _fold(argv[unsafe_offset=2]),
            _raw_ctx(argv[unsafe_offset=3]),
        )
    elif _op_on["Col2im"]():
        _col2im[dtype](
            _raw_int(argv[unsafe_offset=0]),
            _raw_int(argv[unsafe_offset=1]),
            _fold(argv[unsafe_offset=2]),
            _raw_ctx(argv[unsafe_offset=3]),
        )
    else:
        raise Error(NO_OP_COMPILED)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["Im2col"]() or _op_on["Col2im"]():
            comptime for dt in FOLD_DTYPES:
                comptime if _dtype_arg_on[0, dt]():
                    _launch_fold[dt](argv, argc)
                    return 0
        elif _op_on["MaxUnpool"]():
            comptime for dt in UNPOOL_DTYPES:
                comptime if _dtype_arg_on[0, dt]():
                    _launch[dt](argv, argc)
                    return 0
        else:
            comptime for dt in POOL_DTYPES:
                comptime if _dtype_arg_on[0, dt]():
                    _launch[dt](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
