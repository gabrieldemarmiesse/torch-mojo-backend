"""Discrete Fourier transforms on the mojo device: aten::_fft_c2c, _fft_r2c
and _fft_c2r (and their `out=` overloads), which every `torch.fft` function
(fft, ifft, rfft, irfft, hfft, ihfft, fft2, fftn, rfftn, irfftn, ...)
lowers to. No cuFFT (AGENTS.md eager rule 1): the transforms are the
kernels of tmb/kernels/fft (a mixed-radix Stockham FFT, Bluestein's chirp-z
for lengths with a prime factor above 127, so O(n log n) for every n).

Semantics follow ATen's CPU kernels (aten/src/ATen/native/mkl/
SpectralOps.cpp), the reference the conformance suite compares against:

* a multi-dim transform is a sequence of 1-D ones, one per `dim`;
* r2c transforms the LAST of `dim` first, from the real input, keeping the
  `n // 2 + 1` one-sided half when `onesided` (else the full spectrum), then
  the rest of `dim` as c2c;
* c2r runs the c2c inverse over every dim but the last, then the last one as
  a Hermitian-to-real inverse of length `last_dim_size`: entry j >= n/2 + 1
  of the input is conj(entry n - j), and only real parts are kept, so the
  imaginary parts of the zero and Nyquist entries are ignored, as a c2r
  transform does;
* `normalization` is fft_norm_mode: 0 none, 1 by 1/sqrt(n), 2 by 1/n, n the
  signal length of each transformed dim, applied in that dim's last pass.

Inputs are made contiguous first; every result is contiguous. float32 /
float64 (complex64 / complex128) only: ATen's CPU kernels have no half FFT
and CUDA's covers powers of two only, so half inputs are declined. The
twiddle and chirp tables are computed here on the host in float64 and
uploaded rounded to the component dtype.
"""
from std.collections import Dict
from std.ffi import _get_global_or_null, external_call
from std.math import cos, sin, sqrt
from std.memory.alloc import unsafe_alloc
from std.utils import IndexList

from tmb.backend.abi import (
    ST_COMPLEX128,
    ST_COMPLEX64,
    ST_FLOAT32,
    ST_FLOAT64,
    Owned,
    T,
    Value,
    Values,
    IntList,
    call_op,
    complex_of,
    cpu_empty,
    dtype_name,
    is_complex_stype,
    new_tensor,
    own,
    own_if_new,
    real_of,
    ret_owned,
    ret_ref,
    tensor_arg,
    bool_arg,
    retain,
    unsupported,
    v_bool,
    v_int,
    v_tensor_c,
)
from tmb.backend.device import copy_from_host, ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    contiguous,
    device_str,
    is_float_stype,
    resize_out,
)
from tmb.ops.complex import _st_name

comptime _REAL_IN = 1
comptime _HERM_IN = 2
comptime _REAL_OUT = 4
comptime _INVERSE = 8

# Radices the Stockham stages have butterflies for (tmb/kernels/fft's
# RADICES), largest first; any other prime factor goes through Bluestein.
comptime _RADICES = [8, 4, 2, 3, 5, 7, 11, 13]

comptime _PI = 3.141592653589793238462643383279502884


@fieldwise_init
struct _Lay(Copyable, Movable):
    """[outer, n, inner] addressing of a contiguous tensor along one dim."""

    var inner: Int
    var stride_j: Int
    var stride_outer: Int

    def tup(self) -> List[Int]:
        var l = List[Int](capacity=3)
        l.append(self.inner)
        l.append(self.stride_j)
        l.append(self.stride_outer)
        return l^


def _rows_lay(n: Int) -> _Lay:
    """A [rows, n] work buffer."""
    return _Lay(1, 1, n)


def _lay(t: T, d: Int) -> _Lay:
    var inner = 1
    for k in range(d + 1, t.rank):
        inner *= t.dim(k)
    return _Lay(inner, inner, t.dim(d) * inner)


def _rows(t: T, d: Int) -> Int:
    var rows = 1
    for k in range(t.rank):
        if k != d:
            rows *= t.dim(k)
    return rows


def _component(st: Int32) -> DType:
    return DType.float64 if st == ST_FLOAT64 or st == ST_COMPLEX128 else (
        DType.float32
    )


# Prime factors up to this are direct DFT stages (tmb/kernels/fft's generic
# radix, O(p) per output); a larger one makes the length Bluestein's. 127 is
# cuFFT's own boundary between its direct kernels and Bluestein, so a
# non-finite input spreads (inf * 0 = NaN through the chirp convolution)
# exactly for the lengths where cuFFT's Bluestein spreads it too.
comptime _DIRECT_PRIME_MAX = 127


def _radices(n: Int) -> List[Int]:
    """n as a product of the Stockham radices (the fixed butterflies first,
    then any prime factor up to _DIRECT_PRIME_MAX), or [] if a larger prime
    factor remains (Bluestein's case)."""
    var out = List[Int]()
    var m = n
    comptime for p in _RADICES:
        while m % p == 0 and m > 1:
            out.append(p)
            m //= p
    var q = 17
    while m > 1 and q <= _DIRECT_PRIME_MAX:
        while m % q == 0:
            out.append(q)
            m //= q
        q += 2
    if m != 1:
        return List[Int]()
    if len(out) == 0:
        out.append(1)  # n == 1: one copy pass
    return out^


def _upload(values: List[Float64], real_st: Int32, device: Int) raises -> Owned:
    """A device tensor of `len(values)` reals of `real_st`, from float64 host
    values rounded once."""
    var n = len(values)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    var host = own(cpu_empty(shape, 1, real_st))
    if real_st == ST_FLOAT64:
        var p = Pointer[Float64, MutUntrackedOrigin](
            unsafe_from_address=host.t.ptr
        )
        for i in range(n):
            p[unsafe_offset=i] = values[i]
    else:
        var p = Pointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=host.t.ptr
        )
        for i in range(n):
            p[unsafe_offset=i] = Float32(values[i])
    var dev_t = own(new_tensor(shape, 1, real_st, device))
    var ctx = ctx_for(device)
    copy_from_host(device, ctx, dev_t.t.ptr, host.t.ptr, n * dev_t.t.itemsize)
    # a pageable source is staged and copied in the current stream's order:
    # finish the copy here, before the table is cached, so a later call on
    # ANY stream reads a complete table
    ctx.synchronize()
    _ = host^
    return dev_t^


# Uploaded tables (twiddles, chirps) are kept for the process, keyed by
# (kind, length, dtype, device): they are pure functions of those, and an
# upload is a blocking host-to-device copy. Bounded, so a sweep over many
# lengths cannot grow it without limit (past the bound a table is uploaded
# per call). Only host uploads are cached, each synchronized before it is
# (`_upload`), so a later call on any stream may read them. No lock of its own:
# every op body runs under the shim's recursive `tmb_mutex` (the boxed
# adapter in native/csrc/shim_dispatch.cpp holds it across the Mojo call),
# so two Python threads' FFTs reach this table one at a time.
comptime _TABLES_GLOBAL = "TMB_FFT_TABLES"
comptime _TABLES_MAX = 256
# and at most this many bytes of device memory (a few Bluestein chirps of
# long lengths would otherwise hold hundreds of MB): past either bound a
# table is uploaded per call instead
comptime _TABLES_MAX_BYTES = 64 * 1024 * 1024
comptime _BYTES_KEY = "#bytes"


def _tables() -> Pointer[Dict[String, Int], MutUntrackedOrigin]:
    var g = _get_global_or_null(_TABLES_GLOBAL)
    if g:
        return g.value().unsafe_bitcast[Dict[String, Int]]()
    var box = unsafe_alloc[Dict[String, Int]](1)
    box.unsafe_write(Dict[String, Int]())
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(_TABLES_GLOBAL), box.unsafe_bitcast[NoneType]()
    )
    return (
        _get_global_or_null(_TABLES_GLOBAL)
        .value()
        .unsafe_bitcast[Dict[String, Int]]()
    )


def _table_key(kind: String, n: Int, st: Int32, device: Int) -> String:
    return String(kind, ":", n, ":", st, ":", device)


def _cached(key: String) raises -> Optional[Owned]:
    """The cached table under `key`, as a borrowed (non-releasing) Owned."""
    var tables = _tables()
    var h = tables[].get(key)
    if not h:
        return None
    var held = own(T(h.value()))
    held.live = False
    return held^


def _keep(key: String, t: T):
    """Cache `t` under `key` (a reference of the cache's own)."""
    var tables = _tables()
    var size = t.numel * t.itemsize
    var held = tables[].get(_BYTES_KEY).or_else(0)
    if key in tables[]:
        # already cached (a partial hit re-uploaded both tables): keep the
        # cached one; overwriting it would leak its reference
        return
    if len(tables[]) <= _TABLES_MAX and held + size <= _TABLES_MAX_BYTES:
        tables[][key] = retain(t)
        tables[][_BYTES_KEY] = held + size


def _unit(num: Int, den: Int) -> Tuple[Float64, Float64]:
    """e^(-2 pi i num / den) in float64, exact at the multiples of a quarter
    turn and symmetric about the octants (so w^t and w^(n - t) are exact
    conjugates)."""
    var t = num % den
    if t < 0:
        t += den
    if t == 0:
        return (1.0, 0.0)
    if 4 * t == den:
        return (0.0, -1.0)
    if 2 * t == den:
        return (-1.0, 0.0)
    if 4 * t == 3 * den:
        return (0.0, 1.0)
    # Reduce to [0, pi/4] by the octant symmetries.
    var neg = False
    if 2 * t > den:
        t = den - t
        neg = True  # angle -> -angle: conjugate
    # now 0 < t < den / 2: angle a = 2 pi t / den in (0, pi)
    var c: Float64
    var s: Float64
    if 4 * t < den:
        var a = 2.0 * _PI * Float64(t) / Float64(den)
        c = cos(a)
        s = sin(a)
    else:
        # a = pi/2 + b, b = 2 pi (t - den/4) / den = 2 pi (4t - den) / (4 den)
        var b = 2.0 * _PI * Float64(4 * t - den) / Float64(4 * den)
        c = -sin(b)
        s = cos(b)
    # e^(-i a) = (cos a, -sin a)
    if neg:
        return (c, s)
    return (c, -s)


def _twiddles(n: Int, real_st: Int32, device: Int) raises -> Owned:
    """w^t = e^(-2 pi i t / n), t < n, interleaved."""
    var key = _table_key("tw", n, real_st, device)
    var hit = _cached(key)
    if hit:
        return hit.take()
    var v = List[Float64](capacity=2 * n)
    for t in range(n):
        var w = _unit(t, n)
        v.append(w[0])
        v.append(w[1])
    var table = _upload(v, real_st, device)
    _keep(key, table.t)
    return table^


def _next_pow2(x: Int) -> Int:
    var m = 1
    while m < x:
        m *= 2
    return m


def _stage(
    src_ptr: Int,
    src: _Lay,
    dst_ptr: Int,
    dst: _Lay,
    tw_ptr: Int,
    n: Int,
    p: Int,
    ns: Int,
    rows: Int,
    flags: Int,
    in_len: Int,
    out_len: Int,
    scale: Float64,
    comp: DType,
    device: Int,
) raises:
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("fft", "Stage")
    call.arg_dtype(0, comp)
    call.int(src_ptr)
    call.int(dst_ptr)
    call.int(tw_ptr)
    call.int(n)
    call.int(p)
    call.int(ns)
    call.int(rows)
    call.tuple(src.tup())
    call.tuple(dst.tup())
    call.int(flags)
    call.int(in_len)
    call.int(out_len)
    call.f64(scale)
    call.int(cp)
    call.run()
    _ = ctx


def _stockham(
    src_ptr: Int,
    src: _Lay,
    dst_ptr: Int,
    dst: _Lay,
    n: Int,
    rows: Int,
    radices: List[Int],
    in_flags: Int,
    in_len: Int,
    out_flags: Int,
    out_len: Int,
    inverse: Bool,
    scale: Float64,
    complex_st: Int32,
    device: Int,
) raises:
    """One length-n transform of every row, src -> dst, one launch per
    radix (ping-ponging through [rows, n] work buffers between the first
    read and the last write)."""
    var comp = _component(complex_st)
    var tw = _twiddles(n, real_of(complex_st), device)
    var inv = _INVERSE if inverse else 0
    var stages = len(radices)
    var shape_a = IndexList[MAX_RANK](1)
    shape_a[MAX_RANK - 1] = rows * n if stages > 1 else 0
    var shape_b = IndexList[MAX_RANK](1)
    shape_b[MAX_RANK - 1] = rows * n if stages > 2 else 0
    var work_a = own(new_tensor(shape_a, 1, complex_st, device))
    var work_b = own(new_tensor(shape_b, 1, complex_st, device))
    var ns = 1
    var cur_ptr = src_ptr
    var cur_lay = src.copy()
    for s in range(stages):
        var p = radices[s]
        var last = s == stages - 1
        var flags = inv
        if s == 0:
            flags |= in_flags
        var out_ptr = dst_ptr
        var out_lay = dst.copy()
        var limit = out_len
        var sc = scale
        if last:
            flags |= out_flags
        else:
            out_ptr = work_a.t.ptr if s % 2 == 0 else work_b.t.ptr
            out_lay = _rows_lay(n)
            limit = n
            sc = 1.0
        _stage(
            cur_ptr,
            cur_lay,
            out_ptr,
            out_lay,
            tw.t.ptr,
            n,
            p,
            ns,
            rows,
            flags,
            in_len if s == 0 else n,
            limit,
            sc,
            comp,
            device,
        )
        ns *= p
        cur_ptr = out_ptr
        cur_lay = out_lay.copy()
    _ = tw^
    _ = work_a^
    _ = work_b^


def _bluestein(
    src_ptr: Int,
    src: _Lay,
    dst_ptr: Int,
    dst: _Lay,
    n: Int,
    rows: Int,
    in_flags: Int,
    in_len: Int,
    out_flags: Int,
    out_len: Int,
    inverse: Bool,
    scale: Float64,
    complex_st: Int32,
    device: Int,
) raises:
    """Bluestein's chirp-z transform of length n (see tmb/kernels/fft)."""
    var comp = _component(complex_st)
    var real_st = real_of(complex_st)
    var m = _next_pow2(2 * n - 1)
    var radices = _radices(m)
    # chirp c[k] = e^(-+ i pi k^2 / n) = w_{2n}^(k^2 mod 2n), sign by
    # direction, and b = conj(c) at k and at m - k
    var ckey = _table_key(
        "chirp-inv" if inverse else "chirp", n, real_st, device
    )
    var bkey = _table_key("b-inv" if inverse else "b", n, real_st, device)
    var chirp_hit = _cached(ckey)
    var b_hit = _cached(bkey)
    var chirp: Owned
    var b: Owned
    if Bool(chirp_hit) and Bool(b_hit):
        chirp = chirp_hit.take()
        b = b_hit.take()
    else:
        var chirp_v = List[Float64](capacity=2 * n)
        var b_v = List[Float64](capacity=2 * m)
        for _ in range(2 * m):
            b_v.append(0.0)
        for k in range(n):
            var w = _unit((k * k) % (2 * n), 2 * n)
            var cr = w[0]
            var ci = -w[1] if inverse else w[1]
            chirp_v.append(cr)
            chirp_v.append(ci)
            b_v[2 * k] = cr
            b_v[2 * k + 1] = -ci
            if k > 0:
                b_v[2 * (m - k)] = cr
                b_v[2 * (m - k) + 1] = -ci
        chirp = _upload(chirp_v, real_st, device)
        b = _upload(b_v, real_st, device)
        _keep(ckey, chirp.t)
        _keep(bkey, b.t)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = m
    var big_b = own(new_tensor(shape, 1, complex_st, device))
    _stockham(
        b.t.ptr,
        _rows_lay(m),
        big_b.t.ptr,
        _rows_lay(m),
        m,
        1,
        radices,
        0,
        m,
        0,
        m,
        False,
        # 1/m (a power of two: exact) folded into the chirp's spectrum, not
        # applied after the inverse transform: the convolution's values
        # then stay the size of the result, so an input near the dtype's
        # max does not overflow mid-convolution (CUDA keeps it finite)
        1.0 / Float64(m),
        complex_st,
        device,
    )
    _ = b^
    var bshape = IndexList[MAX_RANK](1)
    bshape[MAX_RANK - 1] = rows * m
    var buf = own(new_tensor(bshape, 1, complex_st, device))
    var buf2 = own(new_tensor(bshape, 1, complex_st, device))
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var pre = KernelCall("fft", "ChirpPre")
    pre.arg_dtype(0, comp)
    pre.int(src_ptr)
    pre.int(buf.t.ptr)
    pre.int(chirp.t.ptr)
    pre.int(n)
    pre.int(m)
    pre.int(rows)
    pre.tuple(src.tup())
    pre.int(in_flags)
    pre.int(in_len)
    pre.int(cp)
    pre.run()
    _stockham(
        buf.t.ptr,
        _rows_lay(m),
        buf2.t.ptr,
        _rows_lay(m),
        m,
        rows,
        radices,
        0,
        m,
        0,
        m,
        False,
        1.0,
        complex_st,
        device,
    )
    var mul = KernelCall("fft", "RowMul")
    mul.arg_dtype(0, comp)
    mul.int(buf2.t.ptr)
    mul.int(big_b.t.ptr)
    mul.int(m)
    mul.int(rows)
    mul.int(cp)
    mul.run()
    _stockham(
        buf2.t.ptr,
        _rows_lay(m),
        buf.t.ptr,
        _rows_lay(m),
        m,
        rows,
        radices,
        0,
        m,
        0,
        m,
        True,
        1.0,
        complex_st,
        device,
    )
    var post = KernelCall("fft", "ChirpPost")
    post.arg_dtype(0, comp)
    post.int(buf.t.ptr)
    post.int(dst_ptr)
    post.int(chirp.t.ptr)
    post.int(m)
    post.int(rows)
    post.tuple(dst.tup())
    post.int(out_flags)
    post.int(out_len)
    post.f64(scale)
    post.int(cp)
    post.run()
    _ = ctx
    _ = chirp^
    _ = big_b^
    _ = buf^
    _ = buf2^


def _norm_scale(normalization: Int, n: Int) raises -> Float64:
    if normalization == 0:
        return 1.0
    if normalization == 1:
        return 1.0 / sqrt(Float64(n))
    if normalization == 2:
        return 1.0 / Float64(n)
    raise Error("Invalid normalization mode: ", normalization)


def _transform(
    src: T,
    dst: T,
    d: Int,
    n: Int,
    in_flags: Int,
    out_flags: Int,
    inverse: Bool,
    normalization: Int,
    complex_st: Int32,
    in_len_cap: Int = -1,
) raises:
    """One 1-D transform of length n along dim d, src -> dst (both
    contiguous; their extents along d are the input and output lengths, or
    `in_len_cap` input entries of src's)."""
    var rows = _rows(dst, d)
    if rows == 0 or dst.numel == 0:
        # an empty batch is an empty result, as CPU torch computes it;
        # CUDA's CUFFT_INVALID_SIZE there is a cuFFT plan limitation, not
        # torch semantics
        return
    var in_len = src.dim(d) if in_len_cap < 0 else in_len_cap
    var out_len = dst.dim(d)
    var scale = _norm_scale(normalization, n)
    var radices = _radices(n)
    if len(radices) > 0:
        _stockham(
            src.ptr,
            _lay(src, d),
            dst.ptr,
            _lay(dst, d),
            n,
            rows,
            radices,
            in_flags,
            in_len,
            out_flags,
            out_len,
            inverse,
            scale,
            complex_st,
            dst.device,
        )
    else:
        _bluestein(
            src.ptr,
            _lay(src, d),
            dst.ptr,
            _lay(dst, d),
            n,
            rows,
            in_flags,
            in_len,
            out_flags,
            out_len,
            inverse,
            scale,
            complex_st,
            dst.device,
        )


def _dims(v: Value, rank: Int) raises -> List[Int]:
    var dims = IntList(v)
    var out = List[Int]()
    for i in range(len(dims)):
        var d = dims[i]
        var lim = max(rank, 1)
        if d < -lim or d >= lim:
            raise Error(
                "Dimension out of range (expected to be in range of [",
                -lim,
                ", ",
                lim - 1,
                "], but got ",
                d,
                ")",
            )
        out.append(d + lim if d < 0 else d)
    return out^


def _check_fft_input(t: T, want_complex: Bool, name: StaticString) raises:
    if not t.on_mojo():
        unsupported(
            String(name) + " of a tensor that is not on the mojo device"
        )
    if want_complex:
        if not t.is_complex():
            raise Error(
                name,
                " expects a complex input tensor, but got ",
                dtype_name(t.stype),
            )
    elif t.stype != ST_FLOAT32 and t.stype != ST_FLOAT64:
        unsupported(
            String(name)
            + " of "
            + dtype_name(t.stype)
            + " (float32 / float64 only)"
        )


def _c2c_dims(
    var cur: Owned,
    dims: List[Int],
    normalization: Int,
    forward: Bool,
) raises -> Owned:
    """c2c over `dims` in order, each pass into a fresh tensor."""
    for d in dims:
        var n = cur.t.dim(d)
        var nxt = own(
            new_tensor(cur.t.shape, cur.t.rank, cur.t.stype, cur.t.device)
        )
        _transform(
            cur.t,
            nxt.t,
            d,
            n,
            0,
            0,
            not forward,
            normalization,
            cur.t.stype,
        )
        cur = nxt^
    return cur^


def _fresh(t: T) raises -> Owned:
    """A contiguous tensor holding t (a copy when t already is one, so the
    caller always owns a fresh result)."""
    var dense = own_if_new(contiguous(t), t)
    if dense.live:
        return dense^
    var out = own(new_tensor(t.shape, t.rank, t.stype, t.device))
    var c = call_op(
        "aten::copy_",
        "",
        [tensor_arg(out.t), tensor_arg(t), bool_arg(False)],
        1,
    )
    _ = c^
    return out^


def _c2c(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    _check_fft_input(x, True, "_fft_c2c")
    var dims = _dims(args[unsafe_offset=1], x.rank)
    var normalization = v_int(args[unsafe_offset=2])
    var forward = v_bool(args[unsafe_offset=3])
    var cur = own_if_new(contiguous(x), x)
    if len(dims) == 0:
        return _fresh(x)
    return _c2c_dims(cur^, dims, normalization, forward)


def _r2c(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    _check_fft_input(x, False, "_fft_r2c")
    var dims = _dims(args[unsafe_offset=1], x.rank)
    var normalization = v_int(args[unsafe_offset=2])
    var onesided = v_bool(args[unsafe_offset=3])
    var cst = complex_of(x.stype)
    var src = own_if_new(contiguous(x), x)
    if len(dims) == 0:
        raise Error("_fft_r2c: no dims to transform")
    var last = dims[len(dims) - 1]
    var n = x.dim(last)
    var shape = x.shape
    shape[MAX_RANK - x.rank + last] = n // 2 + 1 if onesided else n
    var out = own(new_tensor(shape, x.rank, cst, x.device))
    _transform(src.t, out.t, last, n, _REAL_IN, 0, False, normalization, cst)
    _ = src^
    var rest = List[Int]()
    for i in range(len(dims) - 1):
        rest.append(dims[i])
    return _c2c_dims(out^, rest, normalization, True)


def _c2r(args: Values) raises -> Owned:
    var x = v_tensor_c(args[unsafe_offset=0])
    _check_fft_input(x, True, "_fft_c2r")
    var dims = _dims(args[unsafe_offset=1], x.rank)
    var normalization = v_int(args[unsafe_offset=2])
    var n = v_int(args[unsafe_offset=3])
    if len(dims) == 0:
        raise Error("_fft_c2r: no dims to transform")
    if n < 1:
        raise Error("Invalid number of data points (", n, ") specified")
    var last = dims[len(dims) - 1]
    var lead = List[Int]()
    for i in range(len(dims) - 1):
        lead.append(dims[i])
    var cur = _c2c_dims(
        own_if_new(contiguous(x), x), lead, normalization, False
    )
    var shape = cur.t.shape
    shape[MAX_RANK - x.rank + last] = n
    var out = own(new_tensor(shape, x.rank, real_of(x.stype), x.device))
    # The input's one-sided half: n // 2 + 1 entries are read (fewer are
    # zero-padded, more ignored), in place through the full tensor's layout,
    # so no input size can make the kernel read out of bounds. (CUDA accepts
    # e.g. 4 entries for last_dim_size 4; every torch.fft caller passes
    # exactly n // 2 + 1.)
    var half = min(x.dim(last), n // 2 + 1)
    _transform(
        cur.t,
        out.t,
        last,
        n,
        _HERM_IN,
        _REAL_OUT,
        True,
        normalization,
        x.stype,
        half,
    )
    _ = cur^
    return out^


def _check_out(x: T, dest: T) raises:
    """The device check CUDA's generated `out=` wrapper makes before any
    work: the out is seen first, so the message names self as the odd one
    out (a host self reaches this device's kernel through a mojo out)."""
    if not x.on_mojo() or not dest.on_mojo() or x.device != dest.device:
        raise Error(
            "Expected all tensors to be on the same device, but got self is",
            " on ",
            device_str(x),
            ", different from other tensors on ",
            device_str(dest),
        )


def _store_any(
    mut dest: T, var res: Owned, rets: Values, resize_first: Bool = False
) raises:
    """The `_out` kernels of cuda/SpectralOps.cpp: the unnormalized result,
    then `at::mul_out(out, result, scale)` -- a TensorIterator, so the out's
    internal overlap is checked as given, before it is resized (the r2c
    two-sided path `resize_output`s first and checks a slice of the
    resized out: `resize_first`), and the result type must be castable to
    the out's (a complex result into a real out, a real one into an
    integer out, are refused; a wider out is fine)."""
    if resize_first:
        resize_out(dest, res.t.shape, res.t.rank)
    assert_no_internal_overlap(dest)
    var cplx = is_complex_stype(res.t.stype)
    if (cplx and not is_complex_stype(dest.stype)) or (
        not cplx
        and not is_float_stype(dest.stype)
        and not is_complex_stype(dest.stype)
    ):
        raise Error(
            "result type ",
            _st_name(res.t.stype),
            " can't be cast to the desired output type ",
            _st_name(dest.stype),
        )
    resize_out(dest, res.t.shape, res.t.rank)
    var c = call_op(
        "aten::copy_",
        "",
        [tensor_arg(dest), tensor_arg(res.t), bool_arg(False)],
        1,
    )
    _ = c^
    _ = res^
    ret_ref(rets, 0, dest)


# aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
def op_fft_c2c(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var r = _c2c(args)
    ret_owned(rets, 0, r)


# aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization,
#   bool forward, *, Tensor(a!) out) -> Tensor(a!)
def op_fft_c2c_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor_c(args[unsafe_offset=4])
    _check_out(v_tensor_c(args[unsafe_offset=0]), dest)
    _store_any(dest, _c2c(args), rets)


# aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
def op_fft_r2c(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var r = _r2c(args)
    ret_owned(rets, 0, r)


# aten::_fft_r2c.out(Tensor self, int[] dim, int normalization,
#   bool onesided, *, Tensor(a!) out) -> Tensor(a!)
def op_fft_r2c_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor_c(args[unsafe_offset=4])
    _check_out(v_tensor_c(args[unsafe_offset=0]), dest)
    var onesided = v_bool(args[unsafe_offset=3])
    _store_any(dest, _r2c(args), rets, not onesided)


# aten::_fft_c2r(Tensor self, int[] dim, int normalization,
#   SymInt last_dim_size) -> Tensor
def op_fft_c2r(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var r = _c2r(args)
    ret_owned(rets, 0, r)


# aten::_fft_c2r.out(Tensor self, int[] dim, int normalization,
#   SymInt last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
def op_fft_c2r_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor_c(args[unsafe_offset=4])
    _check_out(v_tensor_c(args[unsafe_offset=0]), dest)
    _store_any(dest, _c2r(args), rets)


def register_spectral(site: Site) raises:
    impl[op_fft_c2c, "_fft_c2c"](site)
    impl[op_fft_c2c_out, "_fft_c2c.out"](site)
    impl[op_fft_r2c, "_fft_r2c"](site)
    impl[op_fft_r2c_out, "_fft_r2c.out"](site)
    impl[op_fft_c2r, "_fft_c2r"](site)
    impl[op_fft_c2r_out, "_fft_c2r.out"](site)
