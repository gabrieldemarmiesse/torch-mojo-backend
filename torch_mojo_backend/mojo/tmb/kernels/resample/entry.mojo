# ===----------------------------------------------------------------------=== #
# Resampling kernels for mojo_device: reflection / replication padding and
# nearest / nearest-exact / linear / cubic / antialiased bilinear upsampling,
# forward and backward,
# over 1 to 3 spatial dims.
#
# Every kernel sees a contiguous (planes, D, H, W) tensor; a 1-d or 2-d op is
# the same kernel with the missing leading spatial extents set to 1, and the
# RANK define (1..3) selects, at compile time, which axes the interpolation
# formula actually has -- so a bilinear output is computed with CUDA's
# two-level blend, not a trilinear blend whose third level multiplies by 1.
#
# The math is ATen's CUDA kernels' (aten/src/ATen/native/cuda/UpSample.cuh,
# UpSample{Nearest,Linear1d,Bilinear2d,Bicubic2d,Trilinear3d}.cu,
# ReflectionPad.cu, ReplicationPadding.cu at v2.14.0), including where nvcc
# contracts a product into an fma (checked against the sm_90 SASS of
# upsample_bilinear2d_out_frame and upsample_bicubic2d_out_frame): source
# index `fma(dst + 0.5, scale, -0.5)`, blends `fma(w0, a, w1 * b)`, the cubic
# coefficients as fma chains. Scales arrive already rounded to the kernel's
# accumulation type (float32 for half/bfloat16/float32, float64 for float64;
# nearest always float32, as on CUDA).
#
# Backward passes are gathers, one task per grad_input element summing every
# grad_output element that reads it, instead of CUDA's atomic scatters: the
# result is deterministic and needs no atomics (Metal has none for 16-bit
# types). Each axis's contributing outputs form a contiguous range found by
# binary search over the (monotone) forward source index.
# ===----------------------------------------------------------------------=== #

from std.math import ceil, floor, fma
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceContext
from std.utils.coord import Coord
from std.utils.index import IndexList

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _parallel_for_dt,
    _raw_ctx,
    _raw_int,
    _raw_tuple_f64,
    _raw_tuple_int,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_arg_width_on,
    _op_on,
    _tmb_entry_error,
)

# Interpolation mode of the Upsample* ops: 0 nearest, 1 nearest-exact,
# 2 linear, 3 cubic, 4 antialiased bilinear. RANK is the number of spatial dims (1..3). REFLECT picks
# reflection (1) or replication (0) padding for the Pad* ops.
comptime MODE = get_defined_int["MODE", 0]()
comptime RANK = get_defined_int["RANK", 1]()
comptime REFLECT = get_defined_int["REFLECT", 0]()

comptime _FLOATS = [DType.float32, DType.float16, DType.bfloat16, DType.float64]
comptime _NEAREST_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.uint8,
]


@always_inline
def _acc_dtype[dtype: DType]() -> DType:
    """ATen's `acc_type<scalar_t, /*is_cuda=*/true>`: float64 for double,
    int64 for uint8 (nearest backward), float32 otherwise."""
    comptime if dtype == DType.float64:
        return DType.float64
    elif dtype == DType.uint8:
        return DType.int64
    else:
        return DType.float32


# ---------------------------------------------------------------------------
# Per-axis source index math (UpSample.cuh)
# ---------------------------------------------------------------------------


@always_inline
def _nearest_src[exact: Bool](scale: Float32, dst: Int, in_size: Int) -> Int:
    """nearest_neighbor(_exact)_compute_source_index."""
    var real: Float32
    comptime if exact:
        real = (Float32(dst) + 0.5) * scale
    else:
        real = Float32(dst) * scale
    return min(Int(floor(real)), in_size - 1)


@always_inline
def _nearest_bw_src[
    exact: Bool
](scale: Float32, dst: Int, out_size: Int) -> Int:
    """nearest_neighbor(_exact)_bw_compute_source_index: the first output
    index that reads input `dst` (not clamped below `out_size`)."""
    var real: Float32
    comptime if exact:
        real = fma(Float32(dst), scale, Float32(-0.5))
    else:
        real = Float32(dst) * scale
    return min(Int(ceil(real)), out_size)


@always_inline
def _area_src[
    acc: DType, cubic: Bool
](scale: Scalar[acc], dst: Int, align: Bool) -> Scalar[acc]:
    """area_pixel_compute_source_index."""
    if align:
        return scale * Scalar[acc](dst)
    var real = fma(Scalar[acc](dst) + 0.5, scale, Scalar[acc](-0.5))
    comptime if not cubic:
        if real < 0:
            return 0
    return real


@always_inline
def _src_base[
    acc: DType, cubic: Bool
](scale: Scalar[acc], dst: Int, align: Bool) -> Int:
    """The first input tap of output `dst`: truncation for linear
    (`const int h1 = h1r`), floorf for cubic. Non-decreasing in `dst`."""
    var real = _area_src[acc, cubic](scale, dst, align)
    comptime if cubic:
        return Int(floor(real))
    else:
        return Int(real)


@always_inline
def _cubic1[acc: DType](x: Scalar[acc]) -> Scalar[acc]:
    """cubic_convolution1 with A = -0.75: ((A + 2) x - (A + 3)) x x + 1."""
    return fma(x, x * fma(x, Scalar[acc](1.25), Scalar[acc](-2.25)), 1)


@always_inline
def _cubic2[acc: DType](x: Scalar[acc]) -> Scalar[acc]:
    """cubic_convolution2 with A = -0.75: ((A x - 5A) x + 8A) x - 4A."""
    var r = fma(x, Scalar[acc](-0.75), Scalar[acc](3.75))
    r = fma(x, r, Scalar[acc](-6))
    return fma(x, r, Scalar[acc](3))


@always_inline
def _cubic_coeff[acc: DType](t: Scalar[acc], k: Int) -> Scalar[acc]:
    """get_cubic_upsampling_coefficients(t)[k]."""
    if k == 0:
        return _cubic2[acc](t + 1)
    if k == 1:
        return _cubic1[acc](t)
    var x2 = Scalar[acc](1) - t
    if k == 2:
        return _cubic1[acc](x2)
    return _cubic2[acc](x2 + 1)


@always_inline
def _cubic_interp[
    acc: DType
](
    x0: Scalar[acc],
    x1: Scalar[acc],
    x2: Scalar[acc],
    x3: Scalar[acc],
    t: Scalar[acc],
) -> Scalar[acc]:
    """cubic_interp1d: x0 c0 + x1 c1 + x2 c2 + x3 c3, left to right, the
    first sum contracted as fma(x0, c0, x1 c1) like nvcc does."""
    var r = fma(x0, _cubic_coeff[acc](t, 0), x1 * _cubic_coeff[acc](t, 1))
    r = fma(x2, _cubic_coeff[acc](t, 2), r)
    return fma(x3, _cubic_coeff[acc](t, 3), r)


@always_inline
def _first_base_ge[
    acc: DType, cubic: Bool
](target: Int, scale: Scalar[acc], out_size: Int, align: Bool) -> Int:
    """The first output index whose first tap is >= `target` (out_size when
    none): binary search over the monotone `_src_base`."""
    var lo = 0
    var hi = out_size
    while lo < hi:
        var mid = (lo + hi) // 2
        if _src_base[acc, cubic](scale, mid, align) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


@always_inline
def _interp_range[
    acc: DType, cubic: Bool
](
    i: Int, scale: Scalar[acc], in_size: Int, out_size: Int, align: Bool
) -> Tuple[Int, Int]:
    """[first, last) outputs whose taps can include input `i`."""
    comptime if cubic:
        # Taps are clamp(base - 1 + k), k < 4: base in [i - 2, i + 1], and
        # every output past an edge folds onto that edge.
        var lo = 0 if i == 0 else _first_base_ge[acc, True](
            i - 2, scale, out_size, align
        )
        var hi = out_size if i == in_size - 1 else _first_base_ge[acc, True](
            i + 2, scale, out_size, align
        )
        return (lo, hi)
    else:
        # Taps are base and base + (base < in - 1): base in [i - 1, i].
        return (
            _first_base_ge[acc, False](i - 1, scale, out_size, align),
            _first_base_ge[acc, False](i + 1, scale, out_size, align),
        )


@always_inline
def _interp_weight[
    acc: DType, cubic: Bool
](o: Int, i: Int, scale: Scalar[acc], in_size: Int, align: Bool) -> Tuple[
    Scalar[acc], Bool
]:
    """The total weight output `o` gives input `i` along one axis (taps that
    clamp onto the same input add up, as CUDA's scattered adds do), and
    whether one of those taps has weight exactly 0: CUDA still adds
    `0 * grad` for it, which is NaN for an infinite grad."""
    var real = _area_src[acc, cubic](scale, o, align)
    var w = Scalar[acc](0)
    var zero_tap = False
    comptime if cubic:
        var base = Int(floor(real))
        var t = real - Scalar[acc](base)
        for k in range(4):
            if max(min(base - 1 + k, in_size - 1), 0) == i:
                var c = _cubic_coeff[acc](t, k)
                w += c
                zero_tap = zero_tap or c == 0
    else:
        var base = Int(real)
        var p = 1 if base < in_size - 1 else 0
        var l1 = real - Scalar[acc](base)
        var l0 = Scalar[acc](1) - l1
        if base == i:
            w += l0
            zero_tap = l0 == 0
        if base + p == i:
            w += l1
            zero_tap = zero_tap or l1 == 0
    return (w, zero_tap)


# ---------------------------------------------------------------------------
# Antialiased bilinear (UpSample.cuh's upsample_antialias, bilinear filter):
# output i averages the inputs within `support` of its center, with weights
# rounded to the tensor dtype and normalized, as the CUDA kernel keeps them
# in shared memory.
# ---------------------------------------------------------------------------


@always_inline
def _aa_filter[acc: DType](x: Scalar[acc]) -> Scalar[acc]:
    """BilinearFilterFunctor: the tent 1 - |x| on (-1, 1)."""
    var a = -x if x < 0 else x
    return Scalar[acc](1) - a if a < 1 else Scalar[acc](0)


@always_inline
def _aa_span[
    acc: DType
](i: Int, in_size: Int, scale: Scalar[acc]) -> Tuple[Int, Int, Scalar[acc]]:
    """_compute_weights_span: (xmin, xsize, xmin - center) of output `i`,
    with nvcc's contractions: `center -/+ support` and `xmin - center` are
    each one fma over the unrounded center `scale * (i + 0.5)`."""
    var support = scale if scale >= 1 else Scalar[acc](1)
    var half_i = Scalar[acc](i) + 0.5
    var xmin = max(Int(fma(half_i, scale, -support) + 0.5), 0)
    var xsize = min(Int(fma(half_i, scale, support) + 0.5), in_size) - xmin
    return (xmin, xsize, fma(-half_i, scale, Scalar[acc](xmin)))


@always_inline
def _aa_invscale[acc: DType](scale: Scalar[acc]) -> Scalar[acc]:
    return Scalar[acc](1) / scale if scale >= 1 else Scalar[acc](1)


@always_inline
def _aa_total[
    acc: DType
](span: Tuple[Int, Int, Scalar[acc]], scale: Scalar[acc]) -> Scalar[acc]:
    """The unrounded weight sum _compute_weights normalizes by."""
    var invscale = _aa_invscale[acc](scale)
    var xmc = span[2]
    var total = Scalar[acc](0)
    for j in range(span[1]):
        total += _aa_filter[acc]((Scalar[acc](j) + xmc + 0.5) * invscale)
    return total


@always_inline
def _aa_weight[
    dtype: DType, acc: DType
](
    j: Int,
    span: Tuple[Int, Int, Scalar[acc]],
    scale: Scalar[acc],
    total: Scalar[acc],
) -> Scalar[acc]:
    """Weight j of a span, rounded to `dtype` before and after the
    normalization (`wt_ptr[j] = scalar_t(w)`, then `wt_ptr[j] /= total_w`)."""
    var xmc = span[2]
    var w = _aa_filter[acc](
        (Scalar[acc](j) + xmc + 0.5) * _aa_invscale[acc](scale)
    )
    w = w.cast[dtype]().cast[acc]()
    if total != 0:
        # `wt_ptr[j] /= total_w` on a scalar_t: the sum is rounded to the
        # dtype first (c10::Half's only compound division takes a Half).
        w = (w / total.cast[dtype]().cast[acc]()).cast[dtype]().cast[acc]()
    return w


@always_inline
def _aa_first_past[
    acc: DType, use_max: Bool
](target: Int, scale: Scalar[acc], in_size: Int, out_size: Int) -> Int:
    """The first output whose span start (or, with `use_max`, span end) is
    past `target`: both are non-decreasing in the output index."""
    var lo = 0
    var hi = out_size
    while lo < hi:
        var mid = (lo + hi) // 2
        var sp = _aa_span[acc](mid, in_size, scale)
        var edge = sp[0] + sp[1] if use_max else sp[0]
        if edge > target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Upsample forward: out[p, od, oh, ow] over a contiguous (p, id, ih, iw)
# ---------------------------------------------------------------------------


@always_inline
def _upsample_fwd[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    planes: Int,
    g: IndexList[6],
    scales: SIMD[DType.float64, 4],
    align_i: Int,
    ctx: DeviceContext,
) raises:
    comptime acc = _acc_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    # Rounded to the kernel's types on the host: a float64 value captured
    # into the kernel would put doubles in the Metal IR even for float32
    # (Metal has none; its compiler rejected -- or spun on -- it).
    var sd = Scalar[acc](scales[0])
    var sh = Scalar[acc](scales[1])
    var sw = Scalar[acc](scales[2])
    var nd = Float32(scales[0])
    var nh = Float32(scales[1])
    var nw = Float32(scales[2])
    var count = planes * g[3] * g[4] * g[5]

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, sd, sh, sw, nd, nh, nw)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var in_d = g[0]
        var in_h = g[1]
        var in_w = g[2]
        var out_d = g[3]
        var out_h = g[4]
        var out_w = g[5]
        var align = align_i != 0
        var i = Int(idx[0].value())
        var ow = i % out_w
        var r = i // out_w
        var oh = r % out_h
        r = r // out_h
        var od = r % out_d
        var plane = r // out_d
        var base = plane * in_d * in_h * in_w

        @always_inline
        @__parameter
        def at(d: Int, h: Int, w: Int) -> Scalar[acc]:
            return in_ptr[unsafe_offset=base + (d * in_h + h) * in_w + w].cast[
                acc
            ]()

        comptime if MODE <= 1:
            comptime exact = MODE == 1
            var d = _nearest_src[exact](nd, od, in_d)
            var h = _nearest_src[exact](nh, oh, in_h)
            var w = _nearest_src[exact](nw, ow, in_w)
            comptime if RANK == 2:
                # upsample_nearest2d_out_frame's per-axis shortcut: an axis
                # that keeps its size is copied whatever the scale says (the
                # 1-d / 3-d kernels and every backward have none).
                if in_h == out_h:
                    h = oh
                if in_w == out_w:
                    w = ow
            out_ptr[unsafe_offset=i] = in_ptr[
                unsafe_offset=base + (d * in_h + h) * in_w + w
            ]
        elif MODE == 2:
            var wr = _area_src[acc, False](sw, ow, align)
            var w0 = Int(wr)
            var wp = 1 if w0 < in_w - 1 else 0
            var wl1 = wr - Scalar[acc](w0)
            var wl0 = Scalar[acc](1) - wl1

            @always_inline
            @__parameter
            def row(d: Int, h: Int) -> Scalar[acc]:
                return fma(wl0, at(d, h, w0), wl1 * at(d, h, w0 + wp))

            var val: Scalar[acc]
            comptime if RANK == 1:
                val = row(0, 0)
            else:
                var hr = _area_src[acc, False](sh, oh, align)
                var h0 = Int(hr)
                var hp = 1 if h0 < in_h - 1 else 0
                var hl1 = hr - Scalar[acc](h0)
                var hl0 = Scalar[acc](1) - hl1

                @always_inline
                @__parameter
                def plane2(d: Int) -> Scalar[acc]:
                    return fma(hl0, row(d, h0), hl1 * row(d, h0 + hp))

                comptime if RANK == 2:
                    val = plane2(0)
                else:
                    var dr = _area_src[acc, False](sd, od, align)
                    var d0 = Int(dr)
                    var dp = 1 if d0 < in_d - 1 else 0
                    var dl1 = dr - Scalar[acc](d0)
                    var dl0 = Scalar[acc](1) - dl1
                    val = fma(dl0, plane2(d0), dl1 * plane2(d0 + dp))
            out_ptr[unsafe_offset=i] = val.cast[dtype]()
        elif MODE == 4:
            # Antialiased bilinear (RANK 2): a weighted row sum along W per
            # input row of the span, each rounded to the dtype (CUDA's
            # scalar_t buffer), then the weighted sum of those along H.
            var sh_a = sh
            var sw_a = sw
            var xs = _aa_span[acc](ow, in_w, sw_a)
            var ys = _aa_span[acc](oh, in_h, sh_a)
            var xt = _aa_total[acc](xs, sw_a)
            var yt = _aa_total[acc](ys, sh_a)
            var val = Scalar[acc](0)
            for y in range(ys[1]):
                var rowv = Scalar[acc](0)
                for x in range(xs[1]):
                    rowv = fma(
                        at(0, ys[0] + y, xs[0] + x),
                        _aa_weight[dtype, acc](x, xs, sw_a, xt),
                        rowv,
                    )
                rowv = rowv.cast[dtype]().cast[acc]()
                val = fma(rowv, _aa_weight[dtype, acc](y, ys, sh_a, yt), val)
            out_ptr[unsafe_offset=i] = val.cast[dtype]()
        else:
            # Bicubic (RANK 2): four cubic row interpolations along W, then
            # one along H, over edge-clamped taps.
            var xr = _area_src[acc, True](sw, ow, align)
            var x0 = Int(floor(xr))
            var tx = xr - Scalar[acc](x0)
            var yr = _area_src[acc, True](sh, oh, align)
            var y0 = Int(floor(yr))
            var ty = yr - Scalar[acc](y0)

            @always_inline
            @__parameter
            def tap(y: Int, x: Int) -> Scalar[acc]:
                return at(
                    0,
                    max(min(y, in_h - 1), 0),
                    max(min(x, in_w - 1), 0),
                )

            @always_inline
            @__parameter
            def crow(k: Int) -> Scalar[acc]:
                var y = y0 - 1 + k
                return _cubic_interp[acc](
                    tap(y, x0 - 1),
                    tap(y, x0),
                    tap(y, x0 + 1),
                    tap(y, x0 + 2),
                    tx,
                )

            var val = _cubic_interp[acc](crow(0), crow(1), crow(2), crow(3), ty)
            out_ptr[unsafe_offset=i] = val.cast[dtype]()

    _parallel_for_dt[dtype, func](count, ctx)


# ---------------------------------------------------------------------------
# Upsample backward: grad_in[p, id, ih, iw] = sum over the outputs reading it
# ---------------------------------------------------------------------------


@always_inline
def _upsample_bwd[
    dtype: DType
](
    gin_addr: Int,
    gout_addr: Int,
    planes: Int,
    g: IndexList[6],
    scales: SIMD[DType.float64, 4],
    align_i: Int,
    ctx: DeviceContext,
) raises:
    comptime acc = _acc_dtype[dtype]()
    var gin = _make_ptr[dtype](gin_addr)
    var gout = _make_ptr[dtype](gout_addr)
    # Rounded to the kernel's types on the host: a float64 value captured
    # into the kernel would put doubles in the Metal IR even for float32
    # (Metal has none; its compiler rejected -- or spun on -- it).
    var sd = Scalar[acc](scales[0])
    var sh = Scalar[acc](scales[1])
    var sw = Scalar[acc](scales[2])
    var nd = Float32(scales[0])
    var nh = Float32(scales[1])
    var nw = Float32(scales[2])
    var count = planes * g[0] * g[1] * g[2]

    @always_inline
    @__parameter
    @__copy_capture(gin, gout, sd, sh, sw, nd, nh, nw)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var in_d = g[0]
        var in_h = g[1]
        var in_w = g[2]
        var out_d = g[3]
        var out_h = g[4]
        var out_w = g[5]
        var align = align_i != 0
        var i = Int(idx[0].value())
        var iw = i % in_w
        var r = i // in_w
        var ih = r % in_h
        r = r // in_h
        var id = r % in_d
        var plane = r // in_d
        var obase = plane * out_d * out_h * out_w
        var total = Scalar[acc](0)

        comptime if MODE <= 1:
            # Nearest: the outputs reading input i are [bw(i), bw(i + 1)).
            comptime exact = MODE == 1
            var d_lo = _nearest_bw_src[exact](nd, id, out_d)
            var d_hi = _nearest_bw_src[exact](nd, id + 1, out_d)
            var h_lo = _nearest_bw_src[exact](nh, ih, out_h)
            var h_hi = _nearest_bw_src[exact](nh, ih + 1, out_h)
            var w_lo = _nearest_bw_src[exact](nw, iw, out_w)
            var w_hi = _nearest_bw_src[exact](nw, iw + 1, out_w)
            for d in range(d_lo, d_hi):
                for h in range(h_lo, h_hi):
                    var row = obase + (d * out_h + h) * out_w
                    for w in range(w_lo, w_hi):
                        total += gout[unsafe_offset=row + w].cast[acc]()
        elif MODE == 4:
            var sh_a = sh
            var sw_a = sw
            var h_lo = _aa_first_past[acc, True](ih, sh_a, in_h, out_h)
            var h_hi = _aa_first_past[acc, False](ih, sh_a, in_h, out_h)
            var w_lo = _aa_first_past[acc, True](iw, sw_a, in_w, out_w)
            var w_hi = _aa_first_past[acc, False](iw, sw_a, in_w, out_w)
            for h in range(h_lo, h_hi):
                var ys = _aa_span[acc](h, in_h, sh_a)
                var wy = _aa_weight[dtype, acc](
                    ih - ys[0], ys, sh_a, _aa_total[acc](ys, sh_a)
                )
                var row = obase + h * out_w
                for w in range(w_lo, w_hi):
                    var xs = _aa_span[acc](w, in_w, sw_a)
                    var wx = _aa_weight[dtype, acc](
                        iw - xs[0], xs, sw_a, _aa_total[acc](xs, sw_a)
                    )
                    total = fma(
                        wx * wy, gout[unsafe_offset=row + w].cast[acc](), total
                    )
        else:
            comptime cubic = MODE == 3
            var sd_a = sd
            var sh_a = sh
            var sw_a = sw
            var dr = (0, 1)
            comptime if RANK == 3:
                dr = _interp_range[acc, cubic](id, sd_a, in_d, out_d, align)
            var hr = (0, 1)
            comptime if RANK >= 2:
                hr = _interp_range[acc, cubic](ih, sh_a, in_h, out_h, align)
            var wr = _interp_range[acc, cubic](iw, sw_a, in_w, out_w, align)
            # Every output of the ranges reads input i; a zero-weight tap
            # still contributes `0 * grad` on CUDA (NaN for an inf grad).
            for d in range(dr[0], dr[1]):
                var wd = (Scalar[acc](1), False)
                comptime if RANK == 3:
                    wd = _interp_weight[acc, cubic](d, id, sd_a, in_d, align)
                for h in range(hr[0], hr[1]):
                    var wh = wd
                    comptime if RANK >= 2:
                        var e = _interp_weight[acc, cubic](
                            h, ih, sh_a, in_h, align
                        )
                        wh = (wd[0] * e[0], wd[1] or e[1])
                    var row = obase + (d * out_h + h) * out_w
                    for w in range(wr[0], wr[1]):
                        var ww = _interp_weight[acc, cubic](
                            w, iw, sw_a, in_w, align
                        )
                        var g = gout[unsafe_offset=row + w].cast[acc]()
                        total = fma(wh[0] * ww[0], g, total)
                        if wh[1] or ww[1]:
                            total += Scalar[acc](0) * g
        gin[unsafe_offset=i] = total.cast[dtype]()

    _parallel_for_dt[dtype, func](count, ctx)


# ---------------------------------------------------------------------------
# Padding: out[b, od, oh, ow] = in[b, map(od - pf), map(oh - pt), map(ow - pl)]
# ---------------------------------------------------------------------------


@always_inline
def _reflect(u: Int, n: Int) -> Int:
    """ReflectionPad.cu's get_index_mapping: one mirror at each edge, the edge
    element not repeated (pads are < n, so one fold is enough)."""
    return abs(u) - abs(u - (n - 1)) - u + n - 1


@always_inline
def _replicate(u: Int, n: Int) -> Int:
    return max(min(u, n - 1), 0)


@always_inline
def _pad_fwd[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    batch: Int,
    g: IndexList[9],
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var count = batch * g[6] * g[7] * g[8]

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var in_d = g[0]
        var in_h = g[1]
        var in_w = g[2]
        var out_d = g[6]
        var out_h = g[7]
        var out_w = g[8]
        var i = Int(idx[0].value())
        var ow = i % out_w
        var r = i // out_w
        var oh = r % out_h
        r = r // out_h
        var od = r % out_d
        var b = r // out_d
        var d: Int
        var h: Int
        var w: Int
        comptime if REFLECT != 0:
            d = _reflect(od - g[3], in_d)
            h = _reflect(oh - g[4], in_h)
            w = _reflect(ow - g[5], in_w)
        else:
            d = _replicate(od - g[3], in_d)
            h = _replicate(oh - g[4], in_h)
            w = _replicate(ow - g[5], in_w)
        out_ptr[unsafe_offset=i] = in_ptr[
            unsafe_offset=((b * in_d + d) * in_h + h) * in_w + w
        ]

    _parallel_for_dt[dtype, func](count, ctx)


@always_inline
def _pad_sources(i: Int, n: Int, pad_lo: Int, out_n: Int) -> IndexList[4]:
    """The output indices along one axis that read input `i`: for
    replication the inclusive range [e0, e1] (e3 = 0 marks a range), for
    reflection up to three points e0..e2 (-1 = none, e3 = 1)."""
    comptime if REFLECT != 0:
        var r = IndexList[4](-1, -1, -1, 1)
        var o = i + pad_lo
        if o >= 0 and o < out_n:
            r[0] = o
        if i != 0:
            o = pad_lo - i
            if o >= 0 and o < out_n:
                r[1] = o
        if i != n - 1:
            o = pad_lo + 2 * (n - 1) - i
            if o >= 0 and o < out_n:
                r[2] = o
        return r
    else:
        var lo = 0 if i == 0 else i + pad_lo
        var hi = out_n - 1 if i == n - 1 else i + pad_lo
        return IndexList[4](max(lo, 0), min(hi, out_n - 1), 0, 0)


@always_inline
def _pad_bwd[
    dtype: DType
](
    gin_addr: Int,
    gout_addr: Int,
    batch: Int,
    g: IndexList[9],
    ctx: DeviceContext,
) raises:
    comptime acc = _acc_dtype[dtype]()
    var gin = _make_ptr[dtype](gin_addr)
    var gout = _make_ptr[dtype](gout_addr)
    var count = batch * g[0] * g[1] * g[2]

    @always_inline
    @__parameter
    @__copy_capture(gin, gout)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var in_d = g[0]
        var in_h = g[1]
        var in_w = g[2]
        var out_d = g[6]
        var out_h = g[7]
        var out_w = g[8]
        var i = Int(idx[0].value())
        var iw = i % in_w
        var r = i // in_w
        var ih = r % in_h
        r = r // in_h
        var id = r % in_d
        var b = r // in_d
        var sd = _pad_sources(id, in_d, g[3], out_d)
        var sh = _pad_sources(ih, in_h, g[4], out_h)
        var sw = _pad_sources(iw, in_w, g[5], out_w)
        var obase = b * out_d * out_h * out_w
        var total = Scalar[acc](0)
        comptime if REFLECT != 0:
            for a in range(3):
                var d = sd[a]
                if d < 0:
                    continue
                for c in range(3):
                    var h = sh[c]
                    if h < 0:
                        continue
                    var row = obase + (d * out_h + h) * out_w
                    for e in range(3):
                        var w = sw[e]
                        if w >= 0:
                            total += gout[unsafe_offset=row + w].cast[acc]()
        else:
            for d in range(sd[0], sd[1] + 1):
                for h in range(sh[0], sh[1] + 1):
                    var row = obase + (d * out_h + h) * out_w
                    for w in range(sw[0], sw[1] + 1):
                        total += gout[unsafe_offset=row + w].cast[acc]()
        gin[unsafe_offset=i] = total.cast[dtype]()

    _parallel_for_dt[dtype, func](count, ctx)


# ---------------------------------------------------------------------------
# Argument unpacking. Slots: (dst, src, planes, geometry tuple[, scales
# tuple, align_corners], ctx).
# ---------------------------------------------------------------------------


def _geom[n: Int](t: Arg) -> IndexList[n]:
    var g = IndexList[n](0)
    comptime for k in range(n):
        g[k] = _raw_tuple_int(t, k)
    return g


def _scales(t: Arg) -> SIMD[DType.float64, 4]:
    var s = SIMD[DType.float64, 4](0)
    comptime for k in range(3):
        s[k] = _raw_tuple_f64(t, k)
    return s


def _upsample_dispatcher[backward: Bool](argv: Argv, argc: Int) raises:
    if argc != 7:
        raise Error("Upsample: expected 7 arguments, got ", argc)
    var dst = _raw_int(argv[unsafe_offset=0])
    var src = _raw_int(argv[unsafe_offset=1])
    var planes = _raw_int(argv[unsafe_offset=2])
    var g = _geom[6](argv[unsafe_offset=3])
    var s = _scales(argv[unsafe_offset=4])
    var align = _raw_int(argv[unsafe_offset=5])
    var ctx = _raw_ctx(argv[unsafe_offset=6])
    # uint8 only reaches here for nearest (the op declines the rest).
    comptime for dt in _NEAREST_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            comptime if backward:
                _upsample_bwd[dt](dst, src, planes, g, s, align, ctx)
            else:
                _upsample_fwd[dt](dst, src, planes, g, s, align, ctx)
            return
    raise Error("Upsample: unsupported dtype")


def _pad_dispatcher[backward: Bool](argv: Argv, argc: Int) raises:
    if argc != 5:
        raise Error("Pad: expected 5 arguments, got ", argc)
    var dst = _raw_int(argv[unsafe_offset=0])
    var src = _raw_int(argv[unsafe_offset=1])
    var batch = _raw_int(argv[unsafe_offset=2])
    var g = _geom[9](argv[unsafe_offset=3])
    var ctx = _raw_ctx(argv[unsafe_offset=4])
    comptime if backward:
        comptime for dt in _FLOATS:
            comptime if _dtype_arg_on[0, dt]():
                _pad_bwd[dt](dst, src, batch, g, ctx)
                return
        raise Error("PadBackward: unsupported dtype")
    else:
        # A copy: only the element width matters.
        comptime if _dtype_arg_width_on[0, 8]():
            _pad_fwd[DType.uint8](dst, src, batch, g, ctx)
        elif _dtype_arg_width_on[0, 16]():
            _pad_fwd[DType.uint16](dst, src, batch, g, ctx)
        elif _dtype_arg_width_on[0, 32]():
            _pad_fwd[DType.uint32](dst, src, batch, g, ctx)
        elif _dtype_arg_width_on[0, 64]():
            _pad_fwd[DType.uint64](dst, src, batch, g, ctx)
        else:
            raise Error("Pad: unsupported element size")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Upsample"]():
            _upsample_dispatcher[False](argv, argc)
            return 0
        comptime if _op_on["UpsampleBackward"]():
            _upsample_dispatcher[True](argv, argc)
            return 0
        comptime if _op_on["Pad"]():
            _pad_dispatcher[False](argv, argc)
            return 0
        comptime if _op_on["PadBackward"]():
            _pad_dispatcher[True](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
