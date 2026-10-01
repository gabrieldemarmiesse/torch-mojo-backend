# ===----------------------------------------------------------------------=== #
# grid_sampler_2d / grid_sampler_3d, forward and backward (F.grid_sample).
#
# A port of ATen's CUDA kernels (aten/src/ATen/native/cuda/GridSampler.cu and
# GridSampler.cuh at v2.14.0): one thread per output location (n, [d,] h, w),
# looping over the channels; bilinear / nearest / bicubic (2-d only)
# interpolation, zeros / border / reflection padding, align_corners, every
# operand read through its own strides. The interpolation and padding modes
# are runtime arguments, as on CUDA, so one build per (op, dtype) covers them.
#
# Arithmetic follows CUDA's types:
#
# * The forward computes in `opmath` (float32 for half / bfloat16, the dtype
#   otherwise), rounding once into the output -- except bicubic's
#   `get_value_bounded<scalar_t>`, whose coordinates arrive as `scalar_t`
#   and are clipped / reflected in it.
# * The backward computes in `scalar_t` itself: for half and bfloat16 every
#   c10::Half / c10::BFloat16 operator rounds its float result back to the
#   16-bit type, which `_r` reproduces after each operation (a product that
#   feeds a rounding is never contracted into an fma by the compiler).
# * clip_coordinates uses CUDA's fminf/fmaxf, so a NaN coordinate clips to 0
#   in the forward; the backward's clip_coordinates_set_grad compares, so NaN
#   passes through to safe_downgrade_to_int_range (-100: out of bounds).
#
# The input gradient is an atomic scatter, as on CUDA (fastAtomicAdd): float32
# and float64 add in place; half and bfloat16 accumulate in a float32
# workspace word per element with a compare-and-swap that rounds every
# addition to the 16-bit type (what CUDA's 16-bit atomics compute, and the
# only form Metal has), then a final kernel casts the workspace into
# grad_input. The op layer raises torch's nondeterminism alert.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import block_idx, grid_dim, thread_idx
from std.atomic import Atomic, Ordering
from std.math import ceildiv, floor, fma
from std.sys import (
    is_amd_gpu,
    is_apple_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
)

from tmb.kernels.common.mps_math import rint_even
from tmb.kernels.common.op_utils import (
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _fmod_f64_exact_scalar,
    _fmod_float32,
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _raw_tuple_int,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime BLOCK = 256
# Grid-stride cap; the per-thread work is a whole channel loop, so a few
# waves of resident blocks are enough (not tuned).
comptime BLOCKS_PER_SM = 32

# GridSamplerInterpolation / GridSamplerPadding (GridSamplerUtils.h).
comptime BILINEAR = 0
comptime NEAREST = 1
comptime BICUBIC = 2
comptime ZEROS = 0
comptime BORDER = 1
comptime REFLECTION = 2

# Geometry slots (the op layer's tuple, in this order). 2-d ops set the D
# extents to 1 and the D strides to 0.
comptime G_N = 0
comptime G_C = 1
comptime G_ID = 2
comptime G_IH = 3
comptime G_IW = 4
comptime G_OD = 5
comptime G_OH = 6
comptime G_OW = 7
comptime G_IN_SN = 8  # input strides: N, C, D, H, W
comptime G_GR_SN = 13  # grid strides: N, D, H, W, coordinate
comptime G_OUT_SN = 18  # output (forward) / grad_output strides: N, C, D, H, W
comptime G_INTERP = 23
comptime G_PAD = 24
comptime G_ALIGN = 25
comptime G_INPUT_GRAD = 26
comptime G_COUNT = 27
comptime NG = 28

comptime Geometry = Array[Int64, NG]


@always_inline
def _acc[dt: DType]() -> DType:
    """at::opmath_type: float32 for the 16-bit floats."""
    return DType.float64 if dt == DType.float64 else DType.float32


@always_inline
def _r[dt: DType](x: Scalar[_acc[dt]()]) -> Scalar[_acc[dt]()]:
    """The rounding a `scalar_t` operator applies to its float result:
    c10::Half / c10::BFloat16 compute in float and store the 16-bit type."""
    comptime if dt == DType.bfloat16:
        # Round to nearest even on the float32 bits rather than through a
        # bfloat16 cast: Metal's lowering folds an int -> float32 -> bfloat16
        # chain into a wrong conversion (8 came back as 65536).
        var bits = rebind[Float32](x).to_bits[DType.uint32]()
        if (bits & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
            return x  # NaN
        bits = (bits + UInt32(0x7FFF) + ((bits >> 16) & UInt32(1))) & UInt32(
            0xFFFF0000
        )
        return rebind[Scalar[_acc[dt]()]](Float32(from_bits=bits))
    elif dt == DType.float16:
        return x.cast[dt]().cast[_acc[dt]()]()
    else:
        return x


@always_inline
def _rs[dt: DType, s: Bool](x: Scalar[_acc[dt]()]) -> Scalar[_acc[dt]()]:
    """`_r` when the arithmetic is `scalar_t`'s (s), else opmath's (exact)."""
    comptime if s:
        return _r[dt](x)
    else:
        return x


@always_inline
def _fmod[dt: DType](x: Scalar[dt], y: Scalar[dt]) -> Scalar[dt]:
    """C `fmod` (exact) for x >= 0, y > 0."""
    comptime if dt == DType.float64:
        return rebind[Scalar[dt]](
            _fmod_f64_exact_scalar(rebind[Float64](x), rebind[Float64](y))
        )
    else:
        return rebind[Scalar[dt]](
            _fmod_float32(rebind[Float32](x), rebind[Float32](y))
        )


@always_inline
def _floor[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return llvm_intrinsic["llvm.floor", Scalar[dt], has_side_effect=False](x)


@always_inline
def _nearbyint[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """std::nearbyint: round half to even (dt is float32 or float64)."""
    comptime if is_apple_gpu():
        # `llvm.roundeven` crashes Apple's Metal shader compiler.
        comptime if dt == DType.float64:
            return rebind[Scalar[dt]](rint_even(rebind[Float64](x)))
        else:
            return rebind[Scalar[dt]](rint_even(rebind[Float32](x)))
    else:
        return llvm_intrinsic[
            "llvm.roundeven", Scalar[dt], has_side_effect=False
        ](x)


@always_inline
def _to_int[dt: DType](x: Scalar[dt]) -> Int:
    """static_cast<int> of a value safe_downgrade_to_int_range bounded to
    [INT_MIN, 2**31] (finite)."""
    return Int(x)


@always_inline
def _within(d: Int, h: Int, w: Int, D: Int, H: Int, W: Int) -> Bool:
    return d >= 0 and d < D and h >= 0 and h < H and w >= 0 and w < W


# ---------------------------------------------------------------------------
# Coordinate math (GridSampler.cuh). `s` selects scalar_t arithmetic.
# ---------------------------------------------------------------------------


@always_inline
def _unnormalize[
    dt: DType, s: Bool
](coord: Scalar[_acc[dt]()], size: Int, align: Bool) -> Scalar[_acc[dt]()]:
    """grid_sampler_unnormalize: float arithmetic (`coord + 1.f`), the
    result stored as the caller's type."""
    comptime A = _acc[dt]()
    if align:
        return _rs[dt, s](((coord + 1) / 2) * Scalar[A](size - 1))
    return _rs[dt, s](fma(coord + 1, Scalar[A](size), Scalar[A](-1)) / 2)


@always_inline
def _clip[
    dt: DType, s: Bool
](x: Scalar[_acc[dt]()], limit: Int) -> Scalar[_acc[dt]()]:
    """clip_coordinates: fminf(limit - 1, fmaxf(x, 0)) -- NaN clips to 0."""
    comptime A = _acc[dt]()
    var hi = _rs[dt, s](Scalar[A](limit - 1))
    var v = x if x > 0 else Scalar[A](0)
    return hi if v > hi else v


@always_inline
def _flips_odd[A: DType](q: Scalar[A]) -> Bool:
    """`static_cast<int>(floor(q)) % 2 != 0` for q >= 0 (or NaN), with the
    device conversion's saturation: NaN -> 0, >= 2**31 -> INT_MAX (odd)."""
    if not (q >= 0):
        return False
    if q >= Scalar[A](2147483647.0):
        return True
    return (Int(q) & 1) == 1


@always_inline
def _reflect[
    dt: DType, s: Bool
](x: Scalar[_acc[dt]()], twice_low: Int, twice_high: Int) -> Scalar[_acc[dt]()]:
    """reflect_coordinates."""
    comptime A = _acc[dt]()
    if twice_low == twice_high:
        return 0
    var mn = _rs[dt, s](_rs[dt, s](Scalar[A](twice_low)) / 2)
    var span = _rs[dt, s](_rs[dt, s](Scalar[A](twice_high - twice_low)) / 2)
    var v = abs(_rs[dt, s](x - mn))
    var extra = _rs[dt, s](_fmod[A](v, span))
    if _flips_odd[A](_floor[A](_rs[dt, s](v / span))):
        return _rs[dt, s](_rs[dt, s](span - extra) + mn)
    return _rs[dt, s](extra + mn)


@always_inline
def _safe_downgrade[A: DType](x: Scalar[A]) -> Scalar[A]:
    """safe_downgrade_to_int_range: -100 (out of bounds) for anything an int
    cannot hold. `INT_MAX - 1` converts to the comparison's float type.
    Finiteness is read off the exponent bits: the GPU builds' fast-math
    flags may fold a float test against infinity or NaN (seen on Metal,
    where every coordinate then read as out of range)."""
    var finite: Bool
    comptime if A == DType.float64:
        finite = (x.to_bits[DType.uint64]() & UInt64(0x7FF0000000000000)) != (
            UInt64(0x7FF0000000000000)
        )
    else:
        finite = (x.to_bits[DType.uint32]() & UInt32(0x7F800000)) != UInt32(
            0x7F800000
        )
    if not finite or x > Scalar[A](2147483646) or x < Scalar[A](-2147483648.0):
        return Scalar[A](-100.0)
    return x


@always_inline
def _compute_coordinates[
    dt: DType, s: Bool
](coord: Scalar[_acc[dt]()], size: Int, pad: Int, align: Bool) -> Scalar[
    _acc[dt]()
]:
    var c = coord
    if pad == BORDER:
        c = _clip[dt, s](c, size)
    elif pad == REFLECTION:
        if align:
            c = _reflect[dt, s](c, 0, 2 * (size - 1))
        else:
            c = _reflect[dt, s](c, -1, 2 * size - 1)
        c = _clip[dt, s](c, size)
    return _safe_downgrade(c)


@always_inline
def _source_index[
    dt: DType
](coord: Scalar[_acc[dt]()], size: Int, pad: Int, align: Bool) -> Scalar[
    _acc[dt]()
]:
    """grid_sampler_compute_source_index, in opmath (the forward)."""
    return _compute_coordinates[dt, False](
        _unnormalize[dt, False](coord, size, align), size, pad, align
    )


@always_inline
def _clip_set_grad[
    dt: DType
](x: Scalar[_acc[dt]()], limit: Int) -> Tuple[
    Scalar[_acc[dt]()], Scalar[_acc[dt]()]
]:
    """clip_coordinates_set_grad: the borders count as out of bounds."""
    comptime A = _acc[dt]()
    if x <= 0:
        return (Scalar[A](0), Scalar[A](0))
    var hi = _r[dt](Scalar[A](limit - 1))
    if x >= hi:
        return (hi, Scalar[A](0))
    return (x, Scalar[A](1))


@always_inline
def _reflect_set_grad[
    dt: DType
](x: Scalar[_acc[dt]()], twice_low: Int, twice_high: Int) -> Tuple[
    Scalar[_acc[dt]()], Scalar[_acc[dt]()]
]:
    """reflect_coordinates_set_grad."""
    comptime A = _acc[dt]()
    if twice_low == twice_high:
        return (Scalar[A](0), Scalar[A](0))
    var mn = _r[dt](_r[dt](Scalar[A](twice_low)) / 2)
    var span = _r[dt](_r[dt](Scalar[A](twice_high - twice_low)) / 2)
    var v = _r[dt](x - mn)
    var mult = Scalar[A](1)
    if v < 0:
        mult = -1
        v = -v
    var extra = _r[dt](_fmod[A](v, span))
    if _flips_odd[A](_floor[A](_r[dt](v / span))):
        return (_r[dt](_r[dt](span - extra) + mn), -mult)
    return (_r[dt](extra + mn), mult)


@always_inline
def _source_index_set_grad[
    dt: DType
](coord: Scalar[_acc[dt]()], size: Int, pad: Int, align: Bool) -> Tuple[
    Scalar[_acc[dt]()], Scalar[_acc[dt]()]
]:
    """grid_sampler_compute_source_index_set_grad, in scalar_t: the index
    and d index / d coord."""
    comptime A = _acc[dt]()
    var g = _r[dt](_r[dt](Scalar[A](size - 1 if align else size)) / 2)
    var c = _unnormalize[dt, True](coord, size, align)
    if pad == BORDER:
        var cl = _clip_set_grad[dt](c, size)
        c = cl[0]
        g = _r[dt](g * cl[1])
    elif pad == REFLECTION:
        var rf: Tuple[Scalar[A], Scalar[A]]
        if align:
            rf = _reflect_set_grad[dt](c, 0, 2 * (size - 1))
        else:
            rf = _reflect_set_grad[dt](c, -1, 2 * size - 1)
        var cl = _clip_set_grad[dt](rf[0], size)
        c = cl[0]
        g = _r[dt](_r[dt](g * rf[1]) * cl[1])
    return (_safe_downgrade(c), g)


# ---------------------------------------------------------------------------
# Bicubic helpers
# ---------------------------------------------------------------------------


@always_inline
def _cubic1[A: DType](x: Scalar[A]) -> Scalar[A]:
    """cubic_convolution1 with A = -0.75: ((A + 2) x - (A + 3)) x x + 1, as
    nvcc contracts it (see tmb/kernels/resample/entry.mojo)."""
    return fma(x, x * fma(x, Scalar[A](1.25), Scalar[A](-2.25)), 1)


@always_inline
def _cubic2[A: DType](x: Scalar[A]) -> Scalar[A]:
    """cubic_convolution2 with A = -0.75: ((A x - 5A) x + 8A) x - 4A."""
    var r = fma(x, Scalar[A](-0.75), Scalar[A](3.75))
    r = fma(x, r, Scalar[A](-6))
    return fma(x, r, Scalar[A](3))


@always_inline
def _cubic_interp[
    A: DType
](
    x0: Scalar[A],
    x1: Scalar[A],
    x2: Scalar[A],
    x3: Scalar[A],
    t: Scalar[A],
) -> Scalar[A]:
    """cubic_interp1d in opmath."""
    var u = Scalar[A](1) - t
    var c0 = _cubic2[A](t + 1)
    var c1 = _cubic1[A](t)
    var c2 = _cubic1[A](u)
    var c3 = _cubic2[A](u + 1)
    return fma(x3, c3, fma(x2, c2, fma(x1, c1, x0 * c0)))


@always_inline
def _s_cubic1[dt: DType](x: Scalar[_acc[dt]()]) -> Scalar[_acc[dt]()]:
    """cubic_convolution1<scalar_t>, rounded per operator."""
    comptime A = _acc[dt]()
    var r = _r[dt](Scalar[A](1.25) * x)
    r = _r[dt](r - Scalar[A](2.25))
    r = _r[dt](r * x)
    r = _r[dt](r * x)
    return _r[dt](r + 1)


@always_inline
def _s_cubic2[dt: DType](x: Scalar[_acc[dt]()]) -> Scalar[_acc[dt]()]:
    """cubic_convolution2<scalar_t>, rounded per operator."""
    comptime A = _acc[dt]()
    var r = _r[dt](Scalar[A](-0.75) * x)
    r = _r[dt](r + Scalar[A](3.75))
    r = _r[dt](r * x)
    r = _r[dt](r + Scalar[A](-6))
    r = _r[dt](r * x)
    return _r[dt](r + Scalar[A](3))


@always_inline
def _s_coeff[dt: DType](t: Scalar[_acc[dt]()], k: Int) -> Scalar[_acc[dt]()]:
    """get_cubic_upsampling_coefficients<scalar_t>(t)[k] (`t + 1.0` is a
    double sum stored back as scalar_t)."""
    comptime A = _acc[dt]()
    if k == 0:
        return _s_cubic2[dt](_r[dt](t + 1))
    if k == 1:
        return _s_cubic1[dt](t)
    var x2 = _r[dt](Scalar[A](1) - t)
    if k == 2:
        return _s_cubic1[dt](x2)
    return _s_cubic2[dt](_r[dt](x2 + 1))


@always_inline
def _s_coeff_grad[
    dt: DType
](t: Scalar[_acc[dt]()], k: Int) -> Scalar[_acc[dt]()]:
    """get_cubic_coefficients_grad<scalar_t>(t)[k], rounded per operator."""
    comptime A = _acc[dt]()
    if k == 0:
        var x = _r[dt](Scalar[A](-1) - t)
        var r = _r[dt](Scalar[A](2.25) * x)
        r = _r[dt](r + Scalar[A](7.5))
        r = _r[dt](r * x)
        return _r[dt](r + Scalar[A](6))
    if k == 1:
        var x = -t
        var r = _r[dt](Scalar[A](-3.75) * x)
        r = _r[dt](r - Scalar[A](4.5))
        return _r[dt](r * x)
    if k == 2:
        var x = _r[dt](Scalar[A](1) - t)
        var r = _r[dt](Scalar[A](3.75) * x)
        r = _r[dt](r - Scalar[A](4.5))
        return _r[dt](r * x)
    var x = _r[dt](Scalar[A](2) - t)
    var r = _r[dt](Scalar[A](-2.25) * x)
    r = _r[dt](r + Scalar[A](7.5))
    r = _r[dt](r * x)
    return _r[dt](r + Scalar[A](-6))


@always_inline
def _bounded_offset[
    dt: DType
](
    x: Scalar[_acc[dt]()],
    y: Scalar[_acc[dt]()],
    W: Int,
    H: Int,
    sW: Int,
    sH: Int,
    pad: Int,
    align: Bool,
) -> Int:
    """get_value_bounded / add_value_bounded's address (scalar_t
    coordinates): the element offset, or -1 out of bounds."""
    var cx = _compute_coordinates[dt, True](x, W, pad, align)
    var cy = _compute_coordinates[dt, True](y, H, pad, align)
    var ix = _to_int(cx)
    var iy = _to_int(cy)
    if _within(0, iy, ix, 1, H, W):
        return iy * sH + ix * sW
    return -1


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@__name("grid_sampler_2d_fwd_" + String(dt))
def _forward_2d[
    dt: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    grid: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    g: Geometry,
):
    comptime A = _acc[dt]()
    var C = Int(g[G_C])
    var H = Int(g[G_IH])
    var W = Int(g[G_IW])
    var oH = Int(g[G_OH])
    var oW = Int(g[G_OW])
    var sN = Int(g[G_IN_SN])
    var sC = Int(g[G_IN_SN + 1])
    var sH = Int(g[G_IN_SN + 3])
    var sW = Int(g[G_IN_SN + 4])
    var gN = Int(g[G_GR_SN])
    var gH = Int(g[G_GR_SN + 2])
    var gW = Int(g[G_GR_SN + 3])
    var gC = Int(g[G_GR_SN + 4])
    var oN = Int(g[G_OUT_SN])
    var oC = Int(g[G_OUT_SN + 1])
    var ooH = Int(g[G_OUT_SN + 3])
    var ooW = Int(g[G_OUT_SN + 4])
    var interp = Int(g[G_INTERP])
    var pad = Int(g[G_PAD])
    var align = g[G_ALIGN] != 0
    var count = Int(g[G_COUNT])
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var w = index % oW
        var h = (index // oW) % oH
        var n = index // (oH * oW)
        var go = n * gN + h * gH + w * gW
        var x = grid[unsafe_offset=go].cast[A]()
        var y = grid[unsafe_offset=go + gC].cast[A]()
        var inp = n * sN
        var out = n * oN + h * ooH + w * ooW
        if interp == BILINEAR:
            var ix = _source_index[dt](x, W, pad, align)
            var iy = _source_index[dt](y, H, pad, align)
            var x0 = _to_int(_floor[A](ix))
            var y0 = _to_int(_floor[A](iy))
            var x1 = x0 + 1
            var y1 = y0 + 1
            var nw = (Scalar[A](x1) - ix) * (Scalar[A](y1) - iy)
            var ne = (ix - Scalar[A](x0)) * (Scalar[A](y1) - iy)
            var sw = (Scalar[A](x1) - ix) * (iy - Scalar[A](y0))
            var se = (ix - Scalar[A](x0)) * (iy - Scalar[A](y0))
            var in_nw = _within(0, y0, x0, 1, H, W)
            var in_ne = _within(0, y0, x1, 1, H, W)
            var in_sw = _within(0, y1, x0, 1, H, W)
            var in_se = _within(0, y1, x1, 1, H, W)
            for c in range(C):
                var base = inp + c * sC
                var acc = Scalar[A](0)
                if in_nw:
                    acc = fma(
                        input[unsafe_offset=base + y0 * sH + x0 * sW].cast[A](),
                        nw,
                        acc,
                    )
                if in_ne:
                    acc = fma(
                        input[unsafe_offset=base + y0 * sH + x1 * sW].cast[A](),
                        ne,
                        acc,
                    )
                if in_sw:
                    acc = fma(
                        input[unsafe_offset=base + y1 * sH + x0 * sW].cast[A](),
                        sw,
                        acc,
                    )
                if in_se:
                    acc = fma(
                        input[unsafe_offset=base + y1 * sH + x1 * sW].cast[A](),
                        se,
                        acc,
                    )
                output[unsafe_offset=out + c * oC] = acc.cast[dt]()
        elif interp == NEAREST:
            var ix = _source_index[dt](x, W, pad, align)
            var iy = _source_index[dt](y, H, pad, align)
            var xn = _to_int(_nearbyint[A](ix))
            var yn = _to_int(_nearbyint[A](iy))
            var inside = _within(0, yn, xn, 1, H, W)
            for c in range(C):
                var v = Scalar[dt](0)
                if inside:
                    v = input[unsafe_offset=inp + c * sC + yn * sH + xn * sW]
                output[unsafe_offset=out + c * oC] = v
        else:
            var ix = _unnormalize[dt, False](x, W, align)
            var iy = _unnormalize[dt, False](y, H, align)
            var fx = _floor[A](ix)
            var fy = _floor[A](iy)
            var tx = ix - fx
            var ty = iy - fy
            # get_value_bounded<scalar_t>: the coordinates become scalar_t.
            var offs = Array[Int, 16](fill=-1)
            comptime for i in range(4):
                var yy = _r[dt](fy - 1 + Scalar[A](i))
                comptime for j in range(4):
                    var xx = _r[dt](fx - 1 + Scalar[A](j))
                    offs[i * 4 + j] = _bounded_offset[dt](
                        xx, yy, W, H, sW, sH, pad, align
                    )
            for c in range(C):
                var base = inp + c * sC
                var rows = Array[Scalar[A], 4](fill=0)
                comptime for i in range(4):
                    var v = Array[Scalar[A], 4](fill=0)
                    comptime for j in range(4):
                        var o = offs[i * 4 + j]
                        if o >= 0:
                            v[j] = input[unsafe_offset=base + o].cast[A]()
                    rows[i] = _cubic_interp[A](v[0], v[1], v[2], v[3], tx)
                output[unsafe_offset=out + c * oC] = _cubic_interp[A](
                    rows[0], rows[1], rows[2], rows[3], ty
                ).cast[dt]()
        index += Int(grid_dim.x) * BLOCK


@__name("grid_sampler_3d_fwd_" + String(dt))
def _forward_3d[
    dt: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    grid: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    g: Geometry,
):
    comptime A = _acc[dt]()
    var C = Int(g[G_C])
    var D = Int(g[G_ID])
    var H = Int(g[G_IH])
    var W = Int(g[G_IW])
    var oD = Int(g[G_OD])
    var oH = Int(g[G_OH])
    var oW = Int(g[G_OW])
    var sN = Int(g[G_IN_SN])
    var sC = Int(g[G_IN_SN + 1])
    var sD = Int(g[G_IN_SN + 2])
    var sH = Int(g[G_IN_SN + 3])
    var sW = Int(g[G_IN_SN + 4])
    var gN = Int(g[G_GR_SN])
    var gD = Int(g[G_GR_SN + 1])
    var gH = Int(g[G_GR_SN + 2])
    var gW = Int(g[G_GR_SN + 3])
    var gC = Int(g[G_GR_SN + 4])
    var oN = Int(g[G_OUT_SN])
    var oC = Int(g[G_OUT_SN + 1])
    var ooD = Int(g[G_OUT_SN + 2])
    var ooH = Int(g[G_OUT_SN + 3])
    var ooW = Int(g[G_OUT_SN + 4])
    var interp = Int(g[G_INTERP])
    var pad = Int(g[G_PAD])
    var align = g[G_ALIGN] != 0
    var count = Int(g[G_COUNT])
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var w = index % oW
        var h = (index // oW) % oH
        var d = (index // (oH * oW)) % oD
        var n = index // (oD * oH * oW)
        var go = n * gN + d * gD + h * gH + w * gW
        var ix = _source_index[dt](
            grid[unsafe_offset=go].cast[A](), W, pad, align
        )
        var iy = _source_index[dt](
            grid[unsafe_offset=go + gC].cast[A](), H, pad, align
        )
        var iz = _source_index[dt](
            grid[unsafe_offset=go + 2 * gC].cast[A](), D, pad, align
        )
        var inp = n * sN
        var out = n * oN + d * ooD + h * ooH + w * ooW
        if interp == BILINEAR:
            var x0 = _to_int(_floor[A](ix))
            var y0 = _to_int(_floor[A](iy))
            var z0 = _to_int(_floor[A](iz))
            var x1 = x0 + 1
            var y1 = y0 + 1
            var z1 = z0 + 1
            var ax0 = Scalar[A](x1) - ix  # weight of x0
            var ax1 = ix - Scalar[A](x0)
            var ay0 = Scalar[A](y1) - iy
            var ay1 = iy - Scalar[A](y0)
            var az0 = Scalar[A](z1) - iz
            var az1 = iz - Scalar[A](z0)
            # tnw, tne, tsw, tse, bnw, bne, bsw, bse (CUDA's order)
            var wt = Array[Scalar[A], 8](fill=0)
            var off = Array[Int, 8](fill=-1)
            comptime for k in range(8):
                var xb = k & 1
                var yb = (k >> 1) & 1
                var zb = (k >> 2) & 1
                wt[k] = (
                    (ax1 if xb else ax0)
                    * (ay1 if yb else ay0)
                    * (az1 if zb else az0)
                )
                var xx = x1 if xb else x0
                var yy = y1 if yb else y0
                var zz = z1 if zb else z0
                if _within(zz, yy, xx, D, H, W):
                    off[k] = zz * sD + yy * sH + xx * sW
            for c in range(C):
                var base = inp + c * sC
                var acc = Scalar[A](0)
                comptime for k in range(8):
                    if off[k] >= 0:
                        acc = fma(
                            input[unsafe_offset=base + off[k]].cast[A](),
                            wt[k],
                            acc,
                        )
                output[unsafe_offset=out + c * oC] = acc.cast[dt]()
        else:
            var xn = _to_int(_nearbyint[A](ix))
            var yn = _to_int(_nearbyint[A](iy))
            var zn = _to_int(_nearbyint[A](iz))
            var inside = _within(zn, yn, xn, D, H, W)
            for c in range(C):
                var v = Scalar[dt](0)
                if inside:
                    v = input[
                        unsafe_offset=inp + c * sC + zn * sD + yn * sH + xn * sW
                    ]
                output[unsafe_offset=out + c * oC] = v
        index += Int(grid_dim.x) * BLOCK


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _storage_dtype[dt: DType]() -> DType:
    """The accumulator of the input-gradient scatter: a float32 word per
    16-bit element (see the header), the dtype itself otherwise."""
    return DType.float32 if dt == DType.float16 or dt == DType.bfloat16 else dt


@always_inline
def _scope() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _scatter[
    dt: DType
](
    acc: Pointer[Scalar[_storage_dtype[dt]()], MutAnyOrigin],
    offset: Int,
    value: Scalar[_acc[dt]()],
):
    """fastAtomicAdd of a scalar_t value."""
    comptime S = _storage_dtype[dt]()
    var ptr = acc.unsafe_offset(offset)
    comptime if S != dt:
        var expected = Scalar[S](0)
        while True:
            var desired = (
                (expected + rebind[Scalar[S]](value)).cast[dt]().cast[S]()
            )
            if Atomic[Scalar[S], scope=_scope()].compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
                weak=True,
            ](ptr, expected, desired):
                break
    else:
        _ = Atomic[Scalar[S], scope=_scope()].fetch_add[
            ordering=Ordering.RELAXED
        ](ptr, rebind[Scalar[S]](value))


@__name("grid_sampler_2d_bwd_" + String(dt))
def _backward_2d[
    dt: DType
](
    grad_out: Pointer[Scalar[dt], MutAnyOrigin],
    input: Pointer[Scalar[dt], MutAnyOrigin],
    grid: Pointer[Scalar[dt], MutAnyOrigin],
    grad_in: Pointer[Scalar[_storage_dtype[dt]()], MutAnyOrigin],
    grad_grid: Pointer[Scalar[dt], MutAnyOrigin],
    g: Geometry,
):
    comptime A = _acc[dt]()
    var C = Int(g[G_C])
    var H = Int(g[G_IH])
    var W = Int(g[G_IW])
    var oH = Int(g[G_OH])
    var oW = Int(g[G_OW])
    var sN = Int(g[G_IN_SN])
    var sC = Int(g[G_IN_SN + 1])
    var sH = Int(g[G_IN_SN + 3])
    var sW = Int(g[G_IN_SN + 4])
    var gN = Int(g[G_GR_SN])
    var gH = Int(g[G_GR_SN + 2])
    var gW = Int(g[G_GR_SN + 3])
    var gC = Int(g[G_GR_SN + 4])
    var oN = Int(g[G_OUT_SN])
    var oC = Int(g[G_OUT_SN + 1])
    var ooH = Int(g[G_OUT_SN + 3])
    var ooW = Int(g[G_OUT_SN + 4])
    var interp = Int(g[G_INTERP])
    var pad = Int(g[G_PAD])
    var align = g[G_ALIGN] != 0
    var need_in = g[G_INPUT_GRAD] != 0
    var count = Int(g[G_COUNT])
    # grad_input is contiguous (zeros_like(input, LEGACY_CONTIGUOUS)).
    var iC = H * W
    var iN = C * iC
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var w = index % oW
        var h = (index // oW) % oH
        var n = index // (oH * oW)
        var go = n * gN + h * gH + w * gW
        var x = grid[unsafe_offset=go].cast[A]()
        var y = grid[unsafe_offset=go + gC].cast[A]()
        var gout = n * oN + h * ooH + w * ooW
        var inp = n * sN
        var gin = n * iN
        var gx = Scalar[A](0)
        var gy = Scalar[A](0)
        var mx: Scalar[A]
        var my: Scalar[A]
        if interp == BILINEAR or interp == NEAREST:
            var sx = _source_index_set_grad[dt](x, W, pad, align)
            var sy = _source_index_set_grad[dt](y, H, pad, align)
            var ix = sx[0]
            var iy = sy[0]
            mx = sx[1]
            my = sy[1]
            if interp == BILINEAR:
                var x0 = _to_int(_floor[A](ix))
                var y0 = _to_int(_floor[A](iy))
                var x1 = x0 + 1
                var y1 = y0 + 1
                # int - scalar_t: the int converts to scalar_t first.
                var fx0 = _r[dt](Scalar[A](x0))
                var fx1 = _r[dt](Scalar[A](x1))
                var fy0 = _r[dt](Scalar[A](y0))
                var fy1 = _r[dt](Scalar[A](y1))
                var dx1 = _r[dt](fx1 - ix)  # ix_se - ix
                var dx0 = _r[dt](ix - fx0)  # ix - ix_nw
                var dy1 = _r[dt](fy1 - iy)
                var dy0 = _r[dt](iy - fy0)
                var nw = _r[dt](dx1 * dy1)
                var ne = _r[dt](dx0 * dy1)
                var sw = _r[dt](dx1 * dy0)
                var se = _r[dt](dx0 * dy0)
                var in_nw = _within(0, y0, x0, 1, H, W)
                var in_ne = _within(0, y0, x1, 1, H, W)
                var in_sw = _within(0, y1, x0, 1, H, W)
                var in_se = _within(0, y1, x1, 1, H, W)
                for c in range(C):
                    var go_v = grad_out[unsafe_offset=gout + c * oC].cast[A]()
                    var gbase = gin + c * iC
                    if need_in:
                        if in_nw:
                            _scatter[dt](
                                grad_in, gbase + y0 * W + x0, _r[dt](nw * go_v)
                            )
                        if in_ne:
                            _scatter[dt](
                                grad_in, gbase + y0 * W + x1, _r[dt](ne * go_v)
                            )
                        if in_sw:
                            _scatter[dt](
                                grad_in, gbase + y1 * W + x0, _r[dt](sw * go_v)
                            )
                        if in_se:
                            _scatter[dt](
                                grad_in, gbase + y1 * W + x1, _r[dt](se * go_v)
                            )
                    var base = inp + c * sC
                    if in_nw:
                        var v = input[
                            unsafe_offset=base + y0 * sH + x0 * sW
                        ].cast[A]()
                        gx = _r[dt](gx - _r[dt](_r[dt](v * dy1) * go_v))
                        gy = _r[dt](gy - _r[dt](_r[dt](v * dx1) * go_v))
                    if in_ne:
                        var v = input[
                            unsafe_offset=base + y0 * sH + x1 * sW
                        ].cast[A]()
                        gx = _r[dt](gx + _r[dt](_r[dt](v * dy1) * go_v))
                        gy = _r[dt](gy - _r[dt](_r[dt](v * dx0) * go_v))
                    if in_sw:
                        var v = input[
                            unsafe_offset=base + y1 * sH + x0 * sW
                        ].cast[A]()
                        gx = _r[dt](gx - _r[dt](_r[dt](v * dy0) * go_v))
                        gy = _r[dt](gy + _r[dt](_r[dt](v * dx1) * go_v))
                    if in_se:
                        var v = input[
                            unsafe_offset=base + y1 * sH + x1 * sW
                        ].cast[A]()
                        gx = _r[dt](gx + _r[dt](_r[dt](v * dy0) * go_v))
                        gy = _r[dt](gy + _r[dt](_r[dt](v * dx0) * go_v))
            else:
                if need_in:
                    var xn = _to_int(_nearbyint[A](ix))
                    var yn = _to_int(_nearbyint[A](iy))
                    if _within(0, yn, xn, 1, H, W):
                        for c in range(C):
                            _scatter[dt](
                                grad_in,
                                gin + c * iC + yn * W + xn,
                                grad_out[unsafe_offset=gout + c * oC].cast[A](),
                            )
                mx = 0
                my = 0
        else:
            var ux = _unnormalize[dt, True](x, W, align)
            var uy = _unnormalize[dt, True](y, H, align)
            mx = _r[dt](_r[dt](Scalar[A](W - 1 if align else W)) / 2)
            my = _r[dt](_r[dt](Scalar[A](H - 1 if align else H)) / 2)
            var fx = _floor[A](ux)
            var fy = _floor[A](uy)
            var tx = _r[dt](ux - fx)
            var ty = _r[dt](uy - fy)
            var xc = Array[Scalar[A], 4](fill=0)
            var yc = Array[Scalar[A], 4](fill=0)
            var xg = Array[Scalar[A], 4](fill=0)
            var yg = Array[Scalar[A], 4](fill=0)
            comptime for k in range(4):
                xc[k] = _s_coeff[dt](tx, k)
                yc[k] = _s_coeff[dt](ty, k)
                xg[k] = _s_coeff_grad[dt](tx, k)
                yg[k] = _s_coeff_grad[dt](ty, k)
            # Offsets of the 16 taps: (ix_nw - 1 + i, iy_nw - 1 + j), the
            # scalar_t sums `ix_nw - 1 + i` rounded per operator. Input and
            # grad_input are addressed with their own strides.
            var in_off = Array[Int, 16](fill=-1)
            var gi_off = Array[Int, 16](fill=-1)
            comptime for i in range(4):
                var xx = _r[dt](_r[dt](fx - 1) + Scalar[A](i))
                comptime for j in range(4):
                    var yy = _r[dt](_r[dt](fy - 1) + Scalar[A](j))
                    in_off[i * 4 + j] = _bounded_offset[dt](
                        xx, yy, W, H, sW, sH, pad, align
                    )
                    gi_off[i * 4 + j] = _bounded_offset[dt](
                        xx, yy, W, H, 1, W, pad, align
                    )
            for c in range(C):
                var go_v = grad_out[unsafe_offset=gout + c * oC].cast[A]()
                var base = inp + c * sC
                var gbase = gin + c * iC
                comptime for i in range(4):
                    comptime for j in range(4):
                        if need_in and gi_off[i * 4 + j] >= 0:
                            _scatter[dt](
                                grad_in,
                                gbase + gi_off[i * 4 + j],
                                _r[dt](_r[dt](go_v * xc[i]) * yc[j]),
                            )
                        var v = Scalar[A](0)
                        if in_off[i * 4 + j] >= 0:
                            v = input[
                                unsafe_offset=base + in_off[i * 4 + j]
                            ].cast[A]()
                        gx = _r[dt](
                            gx
                            - _r[dt](_r[dt](_r[dt](v * xg[i]) * yc[j]) * go_v)
                        )
                        gy = _r[dt](
                            gy
                            - _r[dt](_r[dt](_r[dt](v * yg[j]) * xc[i]) * go_v)
                        )
        grad_grid[unsafe_offset=index * 2] = _r[dt](mx * gx).cast[dt]()
        grad_grid[unsafe_offset=index * 2 + 1] = _r[dt](my * gy).cast[dt]()
        index += Int(grid_dim.x) * BLOCK


@__name("grid_sampler_3d_bwd_" + String(dt))
def _backward_3d[
    dt: DType
](
    grad_out: Pointer[Scalar[dt], MutAnyOrigin],
    input: Pointer[Scalar[dt], MutAnyOrigin],
    grid: Pointer[Scalar[dt], MutAnyOrigin],
    grad_in: Pointer[Scalar[_storage_dtype[dt]()], MutAnyOrigin],
    grad_grid: Pointer[Scalar[dt], MutAnyOrigin],
    g: Geometry,
):
    comptime A = _acc[dt]()
    var C = Int(g[G_C])
    var D = Int(g[G_ID])
    var H = Int(g[G_IH])
    var W = Int(g[G_IW])
    var oD = Int(g[G_OD])
    var oH = Int(g[G_OH])
    var oW = Int(g[G_OW])
    var sN = Int(g[G_IN_SN])
    var sC = Int(g[G_IN_SN + 1])
    var sD = Int(g[G_IN_SN + 2])
    var sH = Int(g[G_IN_SN + 3])
    var sW = Int(g[G_IN_SN + 4])
    var gN = Int(g[G_GR_SN])
    var gD = Int(g[G_GR_SN + 1])
    var gH = Int(g[G_GR_SN + 2])
    var gW = Int(g[G_GR_SN + 3])
    var gC = Int(g[G_GR_SN + 4])
    var oN = Int(g[G_OUT_SN])
    var oC = Int(g[G_OUT_SN + 1])
    var ooD = Int(g[G_OUT_SN + 2])
    var ooH = Int(g[G_OUT_SN + 3])
    var ooW = Int(g[G_OUT_SN + 4])
    var interp = Int(g[G_INTERP])
    var pad = Int(g[G_PAD])
    var align = g[G_ALIGN] != 0
    var need_in = g[G_INPUT_GRAD] != 0
    var count = Int(g[G_COUNT])
    var iH = W
    var iD = H * W
    var iC = D * H * W
    var iN = C * iC
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var w = index % oW
        var h = (index // oW) % oH
        var d = (index // (oH * oW)) % oD
        var n = index // (oD * oH * oW)
        var go = n * gN + d * gD + h * gH + w * gW
        var sx = _source_index_set_grad[dt](
            grid[unsafe_offset=go].cast[A](), W, pad, align
        )
        var sy = _source_index_set_grad[dt](
            grid[unsafe_offset=go + gC].cast[A](), H, pad, align
        )
        var sz = _source_index_set_grad[dt](
            grid[unsafe_offset=go + 2 * gC].cast[A](), D, pad, align
        )
        var ix = sx[0]
        var iy = sy[0]
        var iz = sz[0]
        var gout = n * oN + d * ooD + h * ooH + w * ooW
        var inp = n * sN
        var gin = n * iN
        var gx = Scalar[A](0)
        var gy = Scalar[A](0)
        var gz = Scalar[A](0)
        var mx = sx[1]
        var my = sy[1]
        var mz = sz[1]
        if interp == BILINEAR:
            var x0 = _to_int(_floor[A](ix))
            var y0 = _to_int(_floor[A](iy))
            var z0 = _to_int(_floor[A](iz))
            var x1 = x0 + 1
            var y1 = y0 + 1
            var z1 = z0 + 1
            var dx1 = _r[dt](_r[dt](Scalar[A](x1)) - ix)  # ix_bse - ix
            var dx0 = _r[dt](ix - _r[dt](Scalar[A](x0)))  # ix - ix_tnw
            var dy1 = _r[dt](_r[dt](Scalar[A](y1)) - iy)
            var dy0 = _r[dt](iy - _r[dt](Scalar[A](y0)))
            var dz1 = _r[dt](_r[dt](Scalar[A](z1)) - iz)
            var dz0 = _r[dt](iz - _r[dt](Scalar[A](z0)))
            # Corner k: x = x0 + (k & 1), y = y0 + (k >> 1 & 1),
            # z = z0 + (k >> 2): tnw, tne, tsw, tse, bnw, bne, bsw, bse.
            var wt = Array[Scalar[A], 8](fill=0)
            var in_off = Array[Int, 8](fill=-1)
            var gi_off = Array[Int, 8](fill=-1)
            comptime for k in range(8):
                var xb = k & 1
                var yb = (k >> 1) & 1
                var zb = (k >> 2) & 1
                wt[k] = _r[dt](
                    _r[dt]((dx0 if xb else dx1) * (dy0 if yb else dy1))
                    * (dz0 if zb else dz1)
                )
                var xx = x1 if xb else x0
                var yy = y1 if yb else y0
                var zz = z1 if zb else z0
                if _within(zz, yy, xx, D, H, W):
                    in_off[k] = zz * sD + yy * sH + xx * sW
                    gi_off[k] = zz * iD + yy * iH + xx
            for c in range(C):
                var go_v = grad_out[unsafe_offset=gout + c * oC].cast[A]()
                if need_in:
                    comptime for k in range(8):
                        if gi_off[k] >= 0:
                            _scatter[dt](
                                grad_in,
                                gin + c * iC + gi_off[k],
                                _r[dt](wt[k] * go_v),
                            )
                var base = inp + c * sC
                comptime for k in range(8):
                    if in_off[k] >= 0:
                        var v = input[unsafe_offset=base + in_off[k]].cast[A]()
                        var xb = k & 1
                        var yb = (k >> 1) & 1
                        var zb = (k >> 2) & 1
                        var wx = dx0 if xb else dx1
                        var wy = dy0 if yb else dy1
                        var wz = dz0 if zb else dz1
                        # d/dix of the corner weight: +-(y weight)(z weight).
                        var tx = _r[dt](_r[dt](_r[dt](v * wy) * wz) * go_v)
                        var ty = _r[dt](_r[dt](_r[dt](v * wx) * wz) * go_v)
                        var tz = _r[dt](_r[dt](_r[dt](v * wx) * wy) * go_v)
                        gx = _r[dt](gx + tx) if xb else _r[dt](gx - tx)
                        gy = _r[dt](gy + ty) if yb else _r[dt](gy - ty)
                        gz = _r[dt](gz + tz) if zb else _r[dt](gz - tz)
        else:
            if need_in:
                var xn = _to_int(_nearbyint[A](ix))
                var yn = _to_int(_nearbyint[A](iy))
                var zn = _to_int(_nearbyint[A](iz))
                if _within(zn, yn, xn, D, H, W):
                    for c in range(C):
                        _scatter[dt](
                            grad_in,
                            gin + c * iC + zn * iD + yn * iH + xn,
                            grad_out[unsafe_offset=gout + c * oC].cast[A](),
                        )
            mx = 0
            my = 0
            mz = 0
        grad_grid[unsafe_offset=index * 3] = _r[dt](mx * gx).cast[dt]()
        grad_grid[unsafe_offset=index * 3 + 1] = _r[dt](my * gy).cast[dt]()
        grad_grid[unsafe_offset=index * 3 + 2] = _r[dt](mz * gz).cast[dt]()
        index += Int(grid_dim.x) * BLOCK


@__name("grid_sampler_scatter_cast_" + String(dt))
def _cast_out[
    dt: DType
](
    src: Pointer[Scalar[DType.float32], MutAnyOrigin],
    dst: Pointer[Scalar[dt], MutAnyOrigin],
    count: Int64,
):
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i < Int(count):
        dst[unsafe_offset=i] = src[unsafe_offset=i].cast[dt]()
        i += Int(grid_dim.x) * BLOCK


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _geometry(argv: Argv, slot: Int) -> Geometry:
    var g = Geometry(fill=Int64(0))
    for k in range(NG):
        g[k] = Int64(_raw_tuple_int(argv[unsafe_offset=slot], k))
    return g^


def _blocks(ctx: DeviceContext, count: Int) raises -> Int:
    return max(
        min(ceildiv(count, BLOCK), _device_sm_count(ctx) * BLOCKS_PER_SM), 1
    )


def _launch_forward[dt: DType, rank: Int](argv: Argv, argc: Int) raises:
    """Slots: out, input, grid, geometry, ctx."""
    if argc != 5:
        raise Error("grid sampler forward expects 5 argument slots")
    var g = _geometry(argv, 3)
    var count = Int(g[G_COUNT])
    if count == 0:
        return
    var out = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var grid = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(argv[unsafe_offset=4])
    var blocks = _blocks(ctx, count)
    comptime if rank == 2:
        _enqueue_cached[_forward_2d[dt]](
            ctx, blocks, 1, 1, BLOCK, input, grid, out, g
        )
    else:
        _enqueue_cached[_forward_3d[dt]](
            ctx, blocks, 1, 1, BLOCK, input, grid, out, g
        )
    _ = ctx


def _launch_backward[dt: DType, rank: Int](argv: Argv, argc: Int) raises:
    """Slots: grad_input (0 when not requested; contiguous, written whole),
    grad_grid (contiguous), grad_output, input, grid, geometry, ctx."""
    if argc != 7:
        raise Error("grid sampler backward expects 7 argument slots")
    comptime S = _storage_dtype[dt]()
    var g = _geometry(argv, 5)
    var count = Int(g[G_COUNT])
    var need_in = g[G_INPUT_GRAD] != 0
    var gi_count = Int(g[G_N] * g[G_C] * g[G_ID] * g[G_IH] * g[G_IW])
    var grad_in = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var grad_grid = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var grad_out = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    var grid = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=4])
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(argv[unsafe_offset=6])
    if need_in and gi_count > 0:
        comptime if S == dt:
            var buf = DeviceBuffer[dt](
                ctx,
                grad_in.unsafe_origin_cast[MutUntrackedOrigin](),
                gi_count,
                owning=False,
            )
            ctx.enqueue_memset(buf, Scalar[dt](0))
            _ = buf
    if count == 0:
        comptime if S != dt:
            if need_in and gi_count > 0:
                # No sample: grad_input is all zeros (a memset of 16 bits
                # through a float32 word would not be).
                var ws = ctx.enqueue_create_buffer[S](gi_count)
                ctx.enqueue_memset(ws, Scalar[S](0))
                _enqueue_cached[_cast_out[dt]](
                    ctx,
                    _blocks(ctx, gi_count),
                    1,
                    1,
                    BLOCK,
                    rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
                        ws.unsafe_ptr().as_unsafe_any_origin()
                    ),
                    grad_in,
                    Int64(gi_count),
                )
                _ = ws
        _ = ctx
        return
    var blocks = _blocks(ctx, count)
    comptime if S != dt:
        # A float32 word per element; one element is enough when unused.
        var ws = ctx.enqueue_create_buffer[S](
            gi_count if need_in and gi_count > 0 else 1
        )
        ctx.enqueue_memset(ws, Scalar[S](0))
        var acc = ws.unsafe_ptr().as_unsafe_any_origin()
        comptime if rank == 2:
            _enqueue_cached[_backward_2d[dt]](
                ctx,
                blocks,
                1,
                1,
                BLOCK,
                grad_out,
                input,
                grid,
                acc,
                grad_grid,
                g,
            )
        else:
            _enqueue_cached[_backward_3d[dt]](
                ctx,
                blocks,
                1,
                1,
                BLOCK,
                grad_out,
                input,
                grid,
                acc,
                grad_grid,
                g,
            )
        if need_in and gi_count > 0:
            _enqueue_cached[_cast_out[dt]](
                ctx,
                _blocks(ctx, gi_count),
                1,
                1,
                BLOCK,
                rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](acc),
                grad_in,
                Int64(gi_count),
            )
        _ = ws
    else:
        var acc = rebind[Pointer[Scalar[S], MutAnyOrigin]](grad_in)
        comptime if rank == 2:
            _enqueue_cached[_backward_2d[dt]](
                ctx,
                blocks,
                1,
                1,
                BLOCK,
                grad_out,
                input,
                grid,
                acc,
                grad_grid,
                g,
            )
        else:
            _enqueue_cached[_backward_3d[dt]](
                ctx,
                blocks,
                1,
                1,
                BLOCK,
                grad_out,
                input,
                grid,
                acc,
                grad_grid,
                g,
            )
    _ = ctx


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one (op, dtype) per build."""
    try:
        comptime for dt in [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.float64,
        ]:
            comptime if _dtype_arg_on[0, dt]():
                comptime if _op_on["GridSampler2d"]():
                    _launch_forward[dt, 2](argv, argc)
                    return 0
                elif _op_on["GridSampler3d"]():
                    _launch_forward[dt, 3](argv, argc)
                    return 0
                elif _op_on["GridSampler2dBackward"]():
                    _launch_backward[dt, 2](argv, argc)
                    return 0
                elif _op_on["GridSampler3dBackward"]():
                    _launch_backward[dt, 3](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
