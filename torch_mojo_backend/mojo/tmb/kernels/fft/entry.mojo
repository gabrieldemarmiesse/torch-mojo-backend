# ===----------------------------------------------------------------------=== #
# Discrete Fourier transforms of the mojo device (tmb/ops/spectral.mojo):
# aten::_fft_c2c / _fft_r2c / _fft_c2r, without cuFFT.
#
# One 1-D transform of length n along one dim of a batch of rows. A complex
# buffer is interleaved (re, im) of the component dtype (float32 / float64).
# A tensor is addressed as [outer, n, inner] (contiguous), a row being one
# (outer, inner) pair:
#     element (row, j) = (row // inner) * stride_outer + row % inner
#                        + j * stride_j
# in complex elements (real elements for a real buffer); the work buffers
# between two stages are [rows, n] (inner 1, stride_j 1, stride_outer n).
#
# n = p1 * p2 * ... with every radix in {2, 3, 4, 5, 7, 8, 11, 13} (fixed
# butterflies) or a prime up to 127 (a direct-DFT pass, one thread per
# output): the
# Stockham autosort FFT (Govindaraju et al., "High Performance Discrete
# Fourier Transforms on Graphics Processors", 2008), one launch per radix
# over every row, out of place, results in natural order after the last
# stage:
#     for j in [0, n / p):  k = j mod Ns
#         v[r] = in[j + r n/p] * w^(r k n / (Ns p))      r < p
#         out[(j / Ns) Ns p + k + q Ns] = sum_r v[r] W_p^(q r)
# with w = e^(-2 pi i / n) (its conjugate for the inverse), Ns the product
# of the radices already applied. An n with a prime factor above 127 is
# Bluestein's chirp-z
# transform, a circular convolution of length M = 2^m >= 2n - 1 done with
# the power-of-two Stockham FFT:
#     X[k] = c[k] * IFFT_M(FFT_M(x . c) * FFT_M(conj c'))[k],
#     c[m] = e^(-i pi m^2 / n)
# so every length is O(n log n).
#
# The twiddle table w^t, t < n, and the chirp are computed on the host in
# float64 (tmb/ops/spectral.mojo) and rounded once to the component dtype.
#
# Flags of the first read and the last write: REAL_IN (a real input, the
# imaginary parts 0), HERM_IN (a one-sided Hermitian input of `in_len`
# entries: entry j >= in_len is conj(entry n - j)), REAL_OUT (only the real
# parts are written, to a real buffer), INVERSE (conjugate twiddles). The
# last write keeps the entries j < out_len (the one-sided half of r2c) and
# multiplies by `scale` (the normalization).
#
# Ops (DTYPE_ARG_0 = the component dtype):
#   Stage(in, out, tw, n, p, ns, rows, in_layout3, out_layout3, flags,
#         in_len, out_len, scale, ctx)
#   ChirpPre(in, buf, chirp, n, m, rows, in_layout3, flags, in_len, ctx)
#   RowMul(buf, b, m, rows, ctx)
#   ChirpPost(buf, out, chirp, m, rows, out_layout3, flags, out_len, scale,
#             ctx)
# A `layout3` slot is a tuple (inner, stride_j, stride_outer).
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
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

comptime COMPONENT_DTYPES = [DType.float32, DType.float64]
comptime RADICES = [1, 2, 3, 4, 5, 7, 8, 11, 13]

comptime REAL_IN = 1
comptime HERM_IN = 2
comptime REAL_OUT = 4
comptime INVERSE = 8


@fieldwise_init
struct _Layout(Copyable, Movable):
    """[outer, n, inner] addressing of one buffer (see the header). Built
    in the kernel from three scalar arguments."""

    var inner: Int64
    var stride_j: Int64
    var stride_outer: Int64

    @always_inline
    def at(self, row: Int, j: Int) -> Int:
        var inner = Int(self.inner)
        return (
            (row // inner) * Int(self.stride_outer)
            + row % inner
            + j * Int(self.stride_j)
        )


comptime _Ptr[dt: DType] = Pointer[Scalar[dt], MutAnyOrigin]


@always_inline
def _load[
    dt: DType
](
    ptr: _Ptr[dt],
    lay: _Layout,
    row: Int,
    j: Int,
    n: Int,
    flags: Int,
    in_len: Int,
) -> Tuple[Scalar[dt], Scalar[dt]]:
    """Entry j of a row of the transform's input, as the flags read it."""
    if flags & REAL_IN:
        return (ptr[unsafe_offset=lay.at(row, j)], Scalar[dt](0))
    if flags & HERM_IN:
        if j < in_len:
            var e = lay.at(row, j)
            # the DC entry and, for an even n, the Nyquist one are their own
            # mirrors, so real: their imaginary parts are dropped, as
            # CPU/pocketfft does; cuFFT drops it only up to rounding (a huge
            # one would otherwise leak through our Bluestein path)
            if j == 0 or 2 * j == n:
                return (ptr[unsafe_offset=2 * e], Scalar[dt](0))
            return (ptr[unsafe_offset=2 * e], ptr[unsafe_offset=2 * e + 1])
        var m = n - j
        if m < in_len:
            var e = lay.at(row, m)
            return (ptr[unsafe_offset=2 * e], -ptr[unsafe_offset=2 * e + 1])
        return (Scalar[dt](0), Scalar[dt](0))
    var e = lay.at(row, j)
    return (ptr[unsafe_offset=2 * e], ptr[unsafe_offset=2 * e + 1])


@always_inline
def _store[
    dt: DType
](
    ptr: _Ptr[dt],
    lay: _Layout,
    row: Int,
    j: Int,
    flags: Int,
    re: Scalar[dt],
    im: Scalar[dt],
):
    var e = lay.at(row, j)
    if flags & REAL_OUT:
        ptr[unsafe_offset=e] = re
    else:
        ptr[unsafe_offset=2 * e] = re
        ptr[unsafe_offset=2 * e + 1] = im


@always_inline
def _twiddle_mul[
    dt: DType
](
    xr: Scalar[dt], xi: Scalar[dt], tw: _Ptr[dt], ti: Int, n: Int, inv: Bool
) -> Tuple[Scalar[dt], Scalar[dt]]:
    """(xr + i xi) * w^ti, w = e^(-+2 pi i / n). A quarter turn's multiple
    (-i, -1, +i) is a sign and a swap, as cuFFT applies it, rather than a
    multiply by a table entry with a zero part: an infinite input then stays
    infinite instead of picking up a NaN (inf * 0)."""
    if (4 * ti) % n == 0:
        var quarter = (4 * ti) // n
        if quarter == 2:
            return (-xr, -xi)
        # forward w^(n/4) = -i, w^(3n/4) = +i; the inverse flips them
        if (quarter == 1) != inv:
            return (xi, -xr)
        return (-xi, xr)
    var wr = tw[unsafe_offset=2 * ti]
    var wi = tw[unsafe_offset=2 * ti + 1]
    if inv:
        wi = -wi
    return (xr * wr - xi * wi, xr * wi + xi * wr)


@__name(t"fft_stockham_radix{p}_{dt}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _stage_kernel[
    dt: DType, p: Int
](
    src: _Ptr[dt],
    dst: _Ptr[dt],
    tw: _Ptr[dt],
    n_arg: Int64,
    ns_arg: Int64,
    rows_arg: Int64,
    in_inner: Int64,
    in_sj: Int64,
    in_so: Int64,
    out_inner: Int64,
    out_sj: Int64,
    out_so: Int64,
    flags_arg: Int64,
    in_len_arg: Int64,
    out_len_arg: Int64,
    scale: Scalar[dt],
):
    var lin = _Layout(in_inner, in_sj, in_so)
    var lout = _Layout(out_inner, out_sj, out_so)
    var n = Int(n_arg)
    var ns = Int(ns_arg)
    var flags = Int(flags_arg)
    var in_len = Int(in_len_arg)
    var out_len = Int(out_len_arg)
    var m = n // p  # butterflies per row
    var total = Int(rows_arg) * m
    var tstride = n // (ns * p)  # twiddle-table step of one r * k
    var inv = (flags & INVERSE) != 0
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while t < total:
        var row = t // m
        var j = t - row * m
        var k = j % ns
        var vr = StaticTuple[Scalar[dt], p](fill=Scalar[dt](0))
        var vi = StaticTuple[Scalar[dt], p](fill=Scalar[dt](0))
        comptime for r in range(p):
            var x = _load[dt](src, lin, row, j + r * m, n, flags, in_len)
            var xr = x[0]
            var xi = x[1]
            comptime if r > 0:
                # k == 0: the twiddle is 1, and multiplying by it would turn
                # an infinite input into NaNs (inf * 0) where cuFFT keeps it
                if k != 0:
                    var y = _twiddle_mul[dt](
                        xr, xi, tw, r * k * tstride, n, inv
                    )
                    xr = y[0]
                    xi = y[1]
            vr[r] = xr
            vi[r] = xi
        var base = (j // ns) * ns * p + k
        comptime for q in range(p):
            var sr = vr[0]
            var si = vi[0]
            comptime for r in range(1, p):
                comptime e = (q * r) % p
                comptime if e == 0:
                    sr += vr[r]
                    si += vi[r]
                elif (4 * e) % p == 0:
                    # a quarter turn's multiple: -1 or -+i exactly, a sign
                    # and a swap rather than a multiply (inf stays inf)
                    comptime quarter = (4 * e) // p
                    comptime if quarter == 2:
                        sr -= vr[r]
                        si -= vi[r]
                    else:
                        # forward w^(p/4) = -i, w^(3p/4) = +i; inverse flips
                        var minus_i = (quarter == 1) != inv
                        if minus_i:
                            sr += vi[r]
                            si -= vr[r]
                        else:
                            sr -= vi[r]
                            si += vr[r]
                else:
                    var ti = e * m
                    var wr = tw[unsafe_offset=2 * ti]
                    var wi = tw[unsafe_offset=2 * ti + 1]
                    if inv:
                        wi = -wi
                    sr += vr[r] * wr - vi[r] * wi
                    si += vr[r] * wi + vi[r] * wr
            var idx = base + q * ns
            if idx < out_len:
                _store[dt](dst, lout, row, idx, flags, sr * scale, si * scale)
        t += step


@__name(t"fft_stockham_radix_any_{dt}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _stage_any_kernel[
    dt: DType
](
    src: _Ptr[dt],
    dst: _Ptr[dt],
    tw: _Ptr[dt],
    n_arg: Int64,
    p_arg: Int64,
    ns_arg: Int64,
    rows_arg: Int64,
    in_inner: Int64,
    in_sj: Int64,
    in_so: Int64,
    out_inner: Int64,
    out_sj: Int64,
    out_so: Int64,
    flags_arg: Int64,
    in_len_arg: Int64,
    out_len_arg: Int64,
    scale: Scalar[dt],
):
    """One Stockham pass of a runtime radix p (a prime from 17 up, below
    cuFFT's Bluestein threshold): one thread per OUTPUT, a direct length-p
    DFT of its butterfly's inputs, O(p) per output. The twiddle and root
    multiplies are skipped where the factor is exactly 1."""
    var lin = _Layout(in_inner, in_sj, in_so)
    var lout = _Layout(out_inner, out_sj, out_so)
    var n = Int(n_arg)
    var p = Int(p_arg)
    var ns = Int(ns_arg)
    var flags = Int(flags_arg)
    var in_len = Int(in_len_arg)
    var out_len = Int(out_len_arg)
    var m = n // p
    var total = Int(rows_arg) * n
    var tstride = n // (ns * p)
    var inv = (flags & INVERSE) != 0
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while t < total:
        var row = t // n
        var o = t - row * n
        var blk = o // (ns * p)
        var rem = o - blk * ns * p
        var q = rem // ns
        var k = rem - q * ns
        var j = blk * ns + k
        if o < out_len:
            var sr = Scalar[dt](0)
            var si = Scalar[dt](0)
            for r in range(p):
                var x = _load[dt](src, lin, row, j + r * m, n, flags, in_len)
                var xr = x[0]
                var xi = x[1]
                if r * k != 0:
                    var y = _twiddle_mul[dt](
                        xr, xi, tw, r * k * tstride, n, inv
                    )
                    xr = y[0]
                    xi = y[1]
                var e = (q * r) % p
                if e == 0:
                    sr += xr
                    si += xi
                else:
                    var ri = e * m
                    var wr = tw[unsafe_offset=2 * ri]
                    var wi = tw[unsafe_offset=2 * ri + 1]
                    if inv:
                        wi = -wi
                    sr += xr * wr - xi * wi
                    si += xr * wi + xi * wr
            _store[dt](dst, lout, row, o, flags, sr * scale, si * scale)
        t += step


@__name(t"fft_bluestein_pre_{dt}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _chirp_pre_kernel[
    dt: DType
](
    src: _Ptr[dt],
    buf: _Ptr[dt],
    chirp: _Ptr[dt],
    n_arg: Int64,
    m_arg: Int64,
    rows_arg: Int64,
    in_inner: Int64,
    in_sj: Int64,
    in_so: Int64,
    flags_arg: Int64,
    in_len_arg: Int64,
):
    """buf[row, k] = x[row, k] * chirp[k] for k < n, 0 up to m."""
    var lin = _Layout(in_inner, in_sj, in_so)
    var n = Int(n_arg)
    var m = Int(m_arg)
    var flags = Int(flags_arg)
    var in_len = Int(in_len_arg)
    var total = Int(rows_arg) * m
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while t < total:
        var row = t // m
        var k = t - row * m
        var re = Scalar[dt](0)
        var im = Scalar[dt](0)
        if k < n:
            var x = _load[dt](src, lin, row, k, n, flags, in_len)
            var cr = chirp[unsafe_offset=2 * k]
            var ci = chirp[unsafe_offset=2 * k + 1]
            re = x[0] * cr - x[1] * ci
            im = x[0] * ci + x[1] * cr
        buf[unsafe_offset=2 * t] = re
        buf[unsafe_offset=2 * t + 1] = im
        t += step


@__name(t"fft_bluestein_rowmul_{dt}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _row_mul_kernel[
    dt: DType
](buf: _Ptr[dt], b: _Ptr[dt], m_arg: Int64, rows_arg: Int64):
    """buf[row, k] *= b[k]."""
    var m = Int(m_arg)
    var total = Int(rows_arg) * m
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while t < total:
        var k = t % m
        var xr = buf[unsafe_offset=2 * t]
        var xi = buf[unsafe_offset=2 * t + 1]
        var br = b[unsafe_offset=2 * k]
        var bi = b[unsafe_offset=2 * k + 1]
        buf[unsafe_offset=2 * t] = xr * br - xi * bi
        buf[unsafe_offset=2 * t + 1] = xr * bi + xi * br
        t += step


@__name(t"fft_bluestein_post_{dt}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _chirp_post_kernel[
    dt: DType
](
    buf: _Ptr[dt],
    dst: _Ptr[dt],
    chirp: _Ptr[dt],
    m_arg: Int64,
    rows_arg: Int64,
    out_inner: Int64,
    out_sj: Int64,
    out_so: Int64,
    flags_arg: Int64,
    out_len_arg: Int64,
    scale: Scalar[dt],
):
    """out[row, k] = chirp[k] * buf[row, k] * scale, k < out_len."""
    var lout = _Layout(out_inner, out_sj, out_so)
    var m = Int(m_arg)
    var flags = Int(flags_arg)
    var out_len = Int(out_len_arg)
    var total = Int(rows_arg) * out_len
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while t < total:
        var row = t // out_len
        var k = t - row * out_len
        var e = row * m + k
        var xr = buf[unsafe_offset=2 * e]
        var xi = buf[unsafe_offset=2 * e + 1]
        var cr = chirp[unsafe_offset=2 * k]
        var ci = chirp[unsafe_offset=2 * k + 1]
        _store[dt](
            dst,
            lout,
            row,
            k,
            flags,
            (xr * cr - xi * ci) * scale,
            (xr * ci + xi * cr) * scale,
        )
        t += step


def _layout(a: Arg) -> _Layout:
    return _Layout(
        Int64(_raw_tuple_int(a, 0)),
        Int64(_raw_tuple_int(a, 1)),
        Int64(_raw_tuple_int(a, 2)),
    )


@always_inline
def _ptr[dt: DType](addr: Int) -> _Ptr[dt]:
    return _make_ptr[dt](addr).as_unsafe_any_origin()


def _stage_go[dt: DType](argv: Argv, argc: Int) raises:
    if argc != 14:
        raise Error("fft Stage expects 14 arguments, got ", argc)
    var src = _ptr[dt](_raw_int(argv[unsafe_offset=0]))
    var dst = _ptr[dt](_raw_int(argv[unsafe_offset=1]))
    var tw = _ptr[dt](_raw_int(argv[unsafe_offset=2]))
    var n = _raw_int(argv[unsafe_offset=3])
    var p = _raw_int(argv[unsafe_offset=4])
    var ns = _raw_int(argv[unsafe_offset=5])
    var rows = _raw_int(argv[unsafe_offset=6])
    var lin = _layout(argv[unsafe_offset=7])
    var lout = _layout(argv[unsafe_offset=8])
    var flags = _raw_int(argv[unsafe_offset=9])
    var in_len = _raw_int(argv[unsafe_offset=10])
    var out_len = _raw_int(argv[unsafe_offset=11])
    var scale = _raw_f64(argv[unsafe_offset=12]).cast[dt]()
    var ctx = _raw_ctx(argv[unsafe_offset=13])
    var total = rows * (n // p)
    if total <= 0:
        return
    var handled = False
    comptime for radix in RADICES:
        if p == radix:
            _enqueue_cached[_stage_kernel[dt, radix]](
                ctx,
                _gs_blocks(total),
                1,
                1,
                GS_THREADS,
                src,
                dst,
                tw,
                Int64(n),
                Int64(ns),
                Int64(rows),
                lin.inner,
                lin.stride_j,
                lin.stride_outer,
                lout.inner,
                lout.stride_j,
                lout.stride_outer,
                Int64(flags),
                Int64(in_len),
                Int64(out_len),
                scale,
            )
            handled = True
    if not handled:
        # a prime radix past the fixed butterflies: the direct-DFT pass
        var all_total = rows * n
        _enqueue_cached[_stage_any_kernel[dt]](
            ctx,
            _gs_blocks(all_total),
            1,
            1,
            GS_THREADS,
            src,
            dst,
            tw,
            Int64(n),
            Int64(p),
            Int64(ns),
            Int64(rows),
            lin.inner,
            lin.stride_j,
            lin.stride_outer,
            lout.inner,
            lout.stride_j,
            lout.stride_outer,
            Int64(flags),
            Int64(in_len),
            Int64(out_len),
            scale,
        )


def _chirp_pre_go[dt: DType](argv: Argv, argc: Int) raises:
    if argc != 10:
        raise Error("fft ChirpPre expects 10 arguments, got ", argc)
    var n = _raw_int(argv[unsafe_offset=3])
    var m = _raw_int(argv[unsafe_offset=4])
    var rows = _raw_int(argv[unsafe_offset=5])
    var ctx = _raw_ctx(argv[unsafe_offset=9])
    if rows * m <= 0:
        return
    var lin = _layout(argv[unsafe_offset=6])
    _enqueue_cached[_chirp_pre_kernel[dt]](
        ctx,
        _gs_blocks(rows * m),
        1,
        1,
        GS_THREADS,
        _ptr[dt](_raw_int(argv[unsafe_offset=0])),
        _ptr[dt](_raw_int(argv[unsafe_offset=1])),
        _ptr[dt](_raw_int(argv[unsafe_offset=2])),
        Int64(n),
        Int64(m),
        Int64(rows),
        lin.inner,
        lin.stride_j,
        lin.stride_outer,
        Int64(_raw_int(argv[unsafe_offset=7])),
        Int64(_raw_int(argv[unsafe_offset=8])),
    )


def _row_mul_go[dt: DType](argv: Argv, argc: Int) raises:
    if argc != 5:
        raise Error("fft RowMul expects 5 arguments, got ", argc)
    var m = _raw_int(argv[unsafe_offset=2])
    var rows = _raw_int(argv[unsafe_offset=3])
    var ctx = _raw_ctx(argv[unsafe_offset=4])
    if rows * m <= 0:
        return
    _enqueue_cached[_row_mul_kernel[dt]](
        ctx,
        _gs_blocks(rows * m),
        1,
        1,
        GS_THREADS,
        _ptr[dt](_raw_int(argv[unsafe_offset=0])),
        _ptr[dt](_raw_int(argv[unsafe_offset=1])),
        Int64(m),
        Int64(rows),
    )


def _chirp_post_go[dt: DType](argv: Argv, argc: Int) raises:
    if argc != 10:
        raise Error("fft ChirpPost expects 10 arguments, got ", argc)
    var m = _raw_int(argv[unsafe_offset=3])
    var rows = _raw_int(argv[unsafe_offset=4])
    var out_len = _raw_int(argv[unsafe_offset=7])
    var ctx = _raw_ctx(argv[unsafe_offset=9])
    if rows * out_len <= 0:
        return
    var lout = _layout(argv[unsafe_offset=5])
    _enqueue_cached[_chirp_post_kernel[dt]](
        ctx,
        _gs_blocks(rows * out_len),
        1,
        1,
        GS_THREADS,
        _ptr[dt](_raw_int(argv[unsafe_offset=0])),
        _ptr[dt](_raw_int(argv[unsafe_offset=1])),
        _ptr[dt](_raw_int(argv[unsafe_offset=2])),
        Int64(m),
        Int64(rows),
        lout.inner,
        lout.stride_j,
        lout.stride_outer,
        Int64(_raw_int(argv[unsafe_offset=6])),
        Int64(out_len),
        _raw_f64(argv[unsafe_offset=8]).cast[dt](),
    )


def _go[op: StaticString](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in COMPONENT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            comptime if dt == DType.float64 and has_apple_gpu_accelerator():
                raise Error("float64 is not supported on Apple GPU")
            elif not has_accelerator():
                raise Error("no GPU accelerator available")
            else:
                comptime if op == "Stage":
                    _stage_go[dt](argv, argc)
                elif op == "ChirpPre":
                    _chirp_pre_go[dt](argv, argc)
                elif op == "RowMul":
                    _row_mul_go[dt](argv, argc)
                else:
                    _chirp_post_go[dt](argv, argc)
                handled = True
    if not handled:
        raise Error("fft ", op, ": dtype not compiled in")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one op and one component dtype per build.
    Slots are described in the header; errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Stage"]():
            _go["Stage"](argv, argc)
            return 0
        comptime if _op_on["ChirpPre"]():
            _go["ChirpPre"](argv, argc)
            return 0
        comptime if _op_on["RowMul"]():
            _go["RowMul"](argv, argc)
            return 0
        comptime if _op_on["ChirpPost"]():
            _go["ChirpPost"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
