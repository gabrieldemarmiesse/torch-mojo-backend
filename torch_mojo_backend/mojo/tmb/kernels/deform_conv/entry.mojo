"""Deformable im2col, its derivatives, and convolution layout transforms."""
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu import block_idx, grid_dim, thread_idx
from std.math import ceildiv, floor
from max.gpu.primitives import block
from tmb.kernels.roi.entry import (
    DivisorArgs,
    _divisor,
    _divide,
    _ps_add,
    _scatter_storage_dtype,
    _scatter_buffer,
    _finish_scatter,
)
from tmb.kernels.common.dtype_arithmetic import _product
from std.sys import has_nvidia_gpu_accelerator
from tmb.kernels.common.op_utils import (
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _raw_tuple_int,
    _device_sm_count,
    _enqueue_cached,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime BLOCK = 256


struct Geometry(DevicePassable, TrivialRegisterPassable):
    comptime device_type = Self

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: Pointer[mut=True, NoneType, _],
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Geometry"

    var n: Int64
    var c: Int64
    var h: Int64
    var w: Int64
    var oc: Int64
    var kh: Int64
    var kw: Int64
    var oh: Int64
    var ow: Int64
    var sh: Int64
    var sw: Int64
    var ph: Int64
    var pw: Int64
    var dh: Int64
    var dw: Int64
    var og: Int64
    var mask: Int64

    def __init__(
        out self,
        n: Int,
        c: Int,
        h: Int,
        w: Int,
        oc: Int,
        kh: Int,
        kw: Int,
        oh: Int,
        ow: Int,
        sh: Int,
        sw: Int,
        ph: Int,
        pw: Int,
        dh: Int,
        dw: Int,
        og: Int,
        mask: Int,
    ):
        self.n = Int64(n)
        self.c = Int64(c)
        self.h = Int64(h)
        self.w = Int64(w)
        self.oc = Int64(oc)
        self.kh = Int64(kh)
        self.kw = Int64(kw)
        self.oh = Int64(oh)
        self.ow = Int64(ow)
        self.sh = Int64(sh)
        self.sw = Int64(sw)
        self.ph = Int64(ph)
        self.pw = Int64(pw)
        self.dh = Int64(dh)
        self.dw = Int64(dw)
        self.og = Int64(og)
        self.mask = Int64(mask)


@always_inline
def _pixel[
    dt: DType, acc: DType, idx: DType = DType.int64
](
    x: Pointer[Scalar[dt], MutAnyOrigin],
    base: Scalar[idx],
    y: Scalar[idx],
    z: Scalar[idx],
    p: Geometry,
) -> Scalar[acc]:
    if y >= 0 and y < Scalar[idx](p.h) and z >= 0 and z < Scalar[idx](p.w):
        return x[unsafe_offset=Int(base + y * Scalar[idx](p.w) + z)].cast[acc]()
    return 0


@always_inline
def _sample[
    dt: DType, acc: DType, idx: DType = DType.int64
](
    x: Pointer[Scalar[dt], MutAnyOrigin],
    base: Int,
    y: Scalar[acc],
    z: Scalar[acc],
    p: Geometry,
) -> Tuple[Scalar[acc], Scalar[acc], Scalar[acc]]:
    comptime coord = DType.float64 if acc == DType.float64 else DType.float32
    if not (
        y >= -1
        and y.cast[coord]() < Scalar[coord](p.h)
        and z >= -1
        and z.cast[coord]() < Scalar[coord](p.w)
    ):
        return (0, 0, 0)
    var yl = Scalar[idx](floor(y))
    var zl = Scalar[idx](floor(z))
    var ly = y - Scalar[acc](yl)
    var lz = z - Scalar[acc](zl)
    var a = _pixel[dt, acc, idx](x, Scalar[idx](base), yl, zl, p)
    var b = _pixel[dt, acc, idx](x, Scalar[idx](base), yl, zl + 1, p)
    var c = _pixel[dt, acc, idx](x, Scalar[idx](base), yl + 1, zl, p)
    var d = _pixel[dt, acc, idx](x, Scalar[idx](base), yl + 1, zl + 1, p)
    var value = Scalar[acc](0)
    if y > -1 and z > -1:
        value = (
            _product(_product(1 - ly, 1 - lz), a)
            + _product(_product(1 - ly, lz), b)
            + _product(_product(ly, 1 - lz), c)
            + _product(_product(ly, lz), d)
        )
    return (
        value,
        _product(lz, d - b) + _product(1 - lz, c - a),
        _product(ly, d - c) + _product(1 - ly, b - a),
    )


@always_inline
def _column_coordinates[
    fast: Bool
](
    i: Int,
    p: Geometry,
    ds: DivisorArgs,
    dn: DivisorArgs,
    dk: DivisorArgs,
    dc: DivisorArgs,
    dw: DivisorArgs,
    dkw: DivisorArgs,
) -> Tuple[Int, Int, Int, Int, Int, Int, Int, Int, Int]:
    var s = Int(p.oh) * Int(p.ow)
    var ks = Int(p.kh) * Int(p.kw)
    comptime if fast:
        var q = _divide(UInt32(i), ds)
        var pos = i - Int(q) * s
        var row = _divide(q, dn)
        var batch = Int(q) - Int(row) * Int(p.n)
        var ch = Int(_divide(row, dk))
        var k = Int(row) - ch * ks
        var og = Int(_divide(UInt32(ch), dc))
        var py = Int(_divide(UInt32(pos), dw))
        var px = pos - py * Int(p.ow)
        var ky = Int(_divide(UInt32(k), dkw))
        var kx = k - ky * Int(p.kw)
        return (pos, batch, k, ch, og, py, px, ky, kx)
    else:
        var pos = i % s
        var batch = i // s % Int(p.n)
        var k = i // (s * Int(p.n)) % ks
        var ch = i // (s * Int(p.n) * ks)
        var og = ch // (Int(p.c) // Int(p.og))
        return (
            pos,
            batch,
            k,
            ch,
            og,
            pos // Int(p.ow),
            pos % Int(p.ow),
            k // Int(p.kw),
            k % Int(p.kw),
        )


@__name("deformable_im2col_" + String(dt) + "_fast" + String(fast))
def _im2col[
    dt: DType, fast: Bool
](
    x: Pointer[Scalar[dt], MutAnyOrigin],
    off: Pointer[Scalar[dt], MutAnyOrigin],
    mask: Pointer[Scalar[dt], MutAnyOrigin],
    col: Pointer[Scalar[dt], MutAnyOrigin],
    p: Geometry,
    ds: DivisorArgs,
    dn: DivisorArgs,
    dk: DivisorArgs,
    dc: DivisorArgs,
    dw: DivisorArgs,
    dkw: DivisorArgs,
):
    comptime acc = dt
    var s = Int(p.oh) * Int(p.ow)
    var ks = Int(p.kh) * Int(p.kw)
    var count = Int(p.c) * ks * Int(p.n) * s
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i < count:
        var (pos, batch, k, ch, og, py, px, ky, kx) = _column_coordinates[fast](
            i, p, ds, dn, dk, dc, dw, dkw
        )
        var mi = ((batch * Int(p.og) + og) * ks + k) * s + pos
        var oi = ((batch * Int(p.og) + og) * 2 * ks + 2 * k) * s + pos
        var y = (
            Scalar[acc](py * Int(p.sh) - Int(p.ph) + ky * Int(p.dh))
            + off[unsafe_offset=oi].cast[acc]()
        )
        var z = (
            Scalar[acc](px * Int(p.sw) - Int(p.pw) + kx * Int(p.dw))
            + off[unsafe_offset=oi + s].cast[acc]()
        )
        var m = mask[unsafe_offset=mi].cast[acc]() if Int(p.mask) else Scalar[
            acc
        ](1)
        var v = _sample(
            x, (batch * Int(p.c) + ch) * Int(p.h) * Int(p.w), y, z, p
        )
        col[unsafe_offset=i] = (v[0] * m).cast[dt]()
        i += Int(grid_dim.x) * BLOCK


@__name("deformable_im2col_pixel_loop_i32_" + String(dt))
def _im2col_pixel_loop[
    dt: DType
](
    x: Pointer[Scalar[dt], MutAnyOrigin],
    off: Pointer[Scalar[dt], MutAnyOrigin],
    mask: Pointer[Scalar[dt], MutAnyOrigin],
    col: Pointer[Scalar[dt], MutAnyOrigin],
    p: Geometry,
    ds: DivisorArgs,
    dn: DivisorArgs,
    dc: DivisorArgs,
    dw: DivisorArgs,
):
    var n = Int32(p.n)
    var channels = Int32(p.c)
    var h = Int32(p.h)
    var w = Int32(p.w)
    var oh = Int32(p.oh)
    var ow = Int32(p.ow)
    var kh = Int32(p.kh)
    var kw = Int32(p.kw)
    var sh = Int32(p.sh)
    var sw = Int32(p.sw)
    var ph = Int32(p.ph)
    var pw = Int32(p.pw)
    var dh = Int32(p.dh)
    var dilw = Int32(p.dw)
    var groups = Int32(p.og)
    var s = oh * ow
    var ks = kh * kw
    var count = channels * n * s
    var i = Int32(block_idx.x) * BLOCK + Int32(thread_idx.x)
    while i < count:
        var q = _divide(UInt32(i), ds)
        var pos = i - Int32(q) * s
        var ch = Int32(_divide(q, dn))
        var batch = Int32(q) - ch * n
        var og = Int32(_divide(UInt32(ch), dc))
        var py = Int32(_divide(UInt32(pos), dw))
        var px = pos - py * ow
        var y0 = py * sh - ph
        var x0 = px * sw - pw
        var base = (batch * channels + ch) * h * w
        var oi0 = (batch * groups + og) * 2 * ks * s + pos
        var mi0 = (batch * groups + og) * ks * s + pos
        var ci0 = (ch * ks * n + batch) * s + pos
        var ky = Int32(0)
        while ky < kh:
            var kx = Int32(0)
            while kx < kw:
                var k = ky * kw + kx
                var oi = oi0 + 2 * k * s
                var y = (
                    Scalar[dt](y0 + ky * dh)
                    + off[unsafe_offset=Int(oi)].cast[dt]()
                )
                var z = (
                    Scalar[dt](x0 + kx * dilw)
                    + off[unsafe_offset=Int(oi + s)].cast[dt]()
                )
                var m = mask[unsafe_offset=Int(mi0 + k * s)].cast[
                    dt
                ]() if p.mask else Scalar[dt](1)
                var v = _sample[dt, dt, DType.int32](x, Int(base), y, z, p)[0]
                col[unsafe_offset=Int(ci0 + k * n * s)] = (v * m).cast[dt]()
                kx += 1
            ky += 1
        i += Int32(grid_dim.x) * BLOCK


@always_inline
def _scatter_neighbors[
    dt: DType, span: Int, storage: DType
](
    output: Pointer[Scalar[storage], MutAnyOrigin],
    y: Scalar[dt],
    z: Scalar[dt],
    m: Scalar[dt],
    v: Scalar[dt],
    base: Int,
    h: Int,
    w: Int,
    count: Int,
):
    comptime coord = DType.float64 if dt == DType.float64 else DType.float32
    var yl = Int(floor(y))
    var zl = Int(floor(z))
    comptime start = -1 if span == 3 else 0
    comptime for dy in range(span):
        comptime for dx in range(span):
            var yy = yl + dy + start
            var xx = zl + dx + start
            if yy >= 0 and yy < h and xx >= 0 and xx < w:
                # std::abs promotes Half to float before the weight product.
                var a = 1 - abs((y - Scalar[dt](yy)).cast[coord]())
                var b = 1 - abs((z - Scalar[dt](xx)).cast[coord]())
                if a > 0 and b > 0:
                    var weight = (a * b).cast[dt]()
                    _ps_add(output, base + yy * w + xx, m * weight * v, count)


@__name("deformable_col2im_scatter_" + String(dt) + "_fast" + String(fast))
def _scatter[
    dt: DType, acc: DType, fast: Bool, storage: DType
](
    col: Pointer[Scalar[dt], MutAnyOrigin],
    off: Pointer[Scalar[dt], MutAnyOrigin],
    mask: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[storage], MutAnyOrigin],
    p: Geometry,
    ds: DivisorArgs,
    dn: DivisorArgs,
    dk: DivisorArgs,
    dc: DivisorArgs,
    dw: DivisorArgs,
    dkw: DivisorArgs,
):
    comptime coord = DType.float64 if acc == DType.float64 else DType.float32
    var s = Int(p.oh) * Int(p.ow)
    var ks = Int(p.kh) * Int(p.kw)
    var count = Int(p.c) * ks * Int(p.n) * s
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i < count:
        var (pos, batch, k, ch, og, py, px, ky, kx) = _column_coordinates[fast](
            i, p, ds, dn, dk, dc, dw, dkw
        )
        var mi = ((batch * Int(p.og) + og) * ks + k) * s + pos
        var oi = ((batch * Int(p.og) + og) * 2 * ks + 2 * k) * s + pos
        var y = (
            Scalar[acc](py * Int(p.sh) - Int(p.ph) + ky * Int(p.dh))
            + off[unsafe_offset=oi].cast[acc]()
        )
        var z = (
            Scalar[acc](px * Int(p.sw) - Int(p.pw) + kx * Int(p.dw))
            + off[unsafe_offset=oi + s].cast[acc]()
        )
        var in_bounds = (
            y > -1
            and y.cast[coord]() < Scalar[coord](p.h)
            and z > -1
            and z.cast[coord]() < Scalar[coord](p.w)
        )
        comptime if dt == DType.float16:
            if abs(y) >= 2048 or abs(z) >= 2048:
                # CUDA bounds-checks neighbors, even when Half(size-1) == size.
                in_bounds = (
                    abs(y).cast[coord]() < 65536
                    and abs(z).cast[coord]() < 65536
                )
        if in_bounds:
            var v = col[unsafe_offset=i].cast[acc]()
            var m = mask[unsafe_offset=mi].cast[acc]() if Int(
                p.mask
            ) else Scalar[acc](1)
            var base = (batch * Int(p.c) + ch) * Int(p.h) * Int(p.w)
            var count = Int(p.n * p.c * p.h * p.w)
            comptime if dt == DType.float16:
                # Half(index +/- 1) can equal Half(index) beyond 2048.
                if abs(y) >= 2048 or abs(z) >= 2048:
                    _scatter_neighbors[acc, 3, storage](
                        output, y, z, m, v, base, Int(p.h), Int(p.w), count
                    )
                else:
                    _scatter_neighbors[acc, 2, storage](
                        output, y, z, m, v, base, Int(p.h), Int(p.w), count
                    )
            else:
                _scatter_neighbors[acc, 2, storage](
                    output, y, z, m, v, base, Int(p.h), Int(p.w), count
                )

        i += Int(grid_dim.x) * BLOCK


@__name("deformable_offset_mask_grad_" + String(dt))
def _offset_grad[
    dt: DType
](
    x: Pointer[Scalar[dt], MutAnyOrigin],
    off: Pointer[Scalar[dt], MutAnyOrigin],
    mask: Pointer[Scalar[dt], MutAnyOrigin],
    col: Pointer[Scalar[dt], MutAnyOrigin],
    go: Pointer[Scalar[dt], MutAnyOrigin],
    gm: Pointer[Scalar[dt], MutAnyOrigin],
    p: Geometry,
):
    comptime acc = dt
    var s = Int(p.oh) * Int(p.ow)
    var ks = Int(p.kh) * Int(p.kw)
    var count = Int(p.n) * Int(p.og) * ks * s
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i < count:
        var pos = i % s
        var k = i // s % ks
        var og = i // (s * ks) % Int(p.og)
        var batch = i // (s * ks * Int(p.og))
        var oi = ((batch * Int(p.og) + og) * 2 * ks + 2 * k) * s + pos
        var y = (
            Scalar[acc](
                pos // Int(p.ow) * Int(p.sh)
                - Int(p.ph)
                + k // Int(p.kw) * Int(p.dh)
            )
            + off[unsafe_offset=oi].cast[acc]()
        )
        var z = (
            Scalar[acc](
                pos % Int(p.ow) * Int(p.sw)
                - Int(p.pw)
                + k % Int(p.kw) * Int(p.dw)
            )
            + off[unsafe_offset=oi + s].cast[acc]()
        )
        var m = mask[unsafe_offset=i].cast[acc]() if Int(p.mask) else Scalar[
            acc
        ](1)
        var gy = Scalar[acc](0)
        var gx = Scalar[acc](0)
        var gmask = Scalar[acc](0)
        for ch in range(
            og * (Int(p.c) // Int(p.og)), (og + 1) * (Int(p.c) // Int(p.og))
        ):
            var v = _sample(
                x, (batch * Int(p.c) + ch) * Int(p.h) * Int(p.w), y, z, p
            )
            var g = col[
                unsafe_offset=((ch * ks + k) * Int(p.n) + batch) * s + pos
            ].cast[acc]()
            gy += _product(_product(m, v[1]), g)
            gx += _product(_product(m, v[2]), g)
            gmask += _product(v[0], g)
        go[unsafe_offset=oi] = gy.cast[dt]()
        go[unsafe_offset=oi + s] = gx.cast[dt]()
        if Int(p.mask):
            gm[unsafe_offset=i] = gmask.cast[dt]()
        i += Int(grid_dim.x) * BLOCK


@__name(
    "deform_conv_layout_"
    + String(dt)
    + "_"
    + String(op)
    + "_fast"
    + String(fast)
)
def _layout[
    dt: DType, op: Int, fast: Bool
](
    a: Pointer[Scalar[dt], MutAnyOrigin],
    b: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    p: Geometry,
    ds: DivisorArgs,
    dd: DivisorArgs,
    add: Int64,
):
    var s = Int(p.oh) * Int(p.ow)
    var count = Int(p.n) * Int(p.oc) * s
    comptime if op == 5:
        count = Int(p.oc) * Int(p.c) * Int(p.kh) * Int(p.kw)
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i < count:
        comptime if op == 3 or op == 4:
            var plane = Int(_divide(UInt32(i), ds)) if fast else i // s
            var pos = i - plane * s
            var divisor = Int(p.oc) if op == 3 else Int(p.n)
            var major = (
                Int(_divide(UInt32(plane), dd)) if fast else plane // divisor
            )
            var minor = plane - major * divisor
            comptime if op == 3:
                output[unsafe_offset=i] = (
                    a[unsafe_offset=(minor * Int(p.n) + major) * s + pos]
                    + b[unsafe_offset=minor]
                )
            else:
                output[unsafe_offset=i] = a[
                    unsafe_offset=(minor * Int(p.oc) + major) * s + pos
                ]
        else:
            if add != 0:
                output[unsafe_offset=i] += a[unsafe_offset=i]
            else:
                output[unsafe_offset=i] = a[unsafe_offset=i]
        i += Int(grid_dim.x) * BLOCK


@__name("deform_conv_bias_reduce_" + String(dt))
def _bias_reduce[
    dt: DType
](
    a: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    p: Geometry,
    ignored_mask: Pointer[Scalar[dt], MutAnyOrigin],
    ignored_count: Int64,
):
    var zero_i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while zero_i < Int(ignored_count):
        ignored_mask[unsafe_offset=zero_i] = 0
        zero_i += Int(grid_dim.x) * BLOCK
    comptime acc = DType.float64 if dt == DType.float64 else DType.float32
    comptime if dt == DType.float64:
        # MAX block.sum has no float64 warp shuffle.
        if thread_idx.x != 0:
            return
    comptime step = 1 if dt == DType.float64 else 256
    var ch = Int(block_idx.x)
    var s = Int(p.oh) * Int(p.ow)
    var i = Int(thread_idx.x)
    var total = Scalar[acc](0)
    while i < Int(p.n) * s:
        total += a[unsafe_offset=(i // s * Int(p.oc) + ch) * s + i % s].cast[
            acc
        ]()
        i += step
    comptime if dt == DType.float64:
        output[unsafe_offset=ch] = total.cast[dt]()
    else:
        var result = block.sum[block_size=256](total)
        if thread_idx.x == 0:
            output[unsafe_offset=ch] = result.cast[dt]()


def _enqueue_columns[
    dt: DType, scatter: Bool, fast: Bool
](argv: Argv, p: Geometry, blocks: Int) raises:
    var ds = DivisorArgs(fill=UInt32(0))
    var dn = ds.copy()
    var dk = ds.copy()
    var dc = ds.copy()
    var dw = ds.copy()
    var dkw = ds.copy()
    comptime if fast:
        ds = _divisor(Int(p.oh) * Int(p.ow))
        dn = _divisor(Int(p.n))
        dk = _divisor(Int(p.kh) * Int(p.kw))
        dc = _divisor(Int(p.c) // Int(p.og))
        dw = _divisor(Int(p.ow))
        dkw = _divisor(Int(p.kw))
    var a = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var b = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var c = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(argv[unsafe_offset=8])
    comptime if scatter:
        comptime acc = dt
        var output = _make_ptr[acc](
            _raw_int(argv[unsafe_offset=3])
        ).as_unsafe_any_origin()
        comptime storage = _scatter_storage_dtype[dt]()
        var count = Int(p.n * p.c * p.h * p.w)
        if count == 0:
            return
        # col2im adds to its caller's accumulator, which can already contain
        # contributions from previous column tiles.
        var buffer = _scatter_buffer(ctx, output, count, initialize=False)
        _enqueue_cached[_scatter[dt, acc, fast, storage]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            a,
            b,
            c,
            buffer.unsafe_ptr().as_unsafe_any_origin(),
            p,
            ds,
            dn,
            dk,
            dc,
            dw,
            dkw,
        )
        _finish_scatter(ctx, buffer, output, count)
        _ = buffer
    else:
        var output = _make_ptr[dt](
            _raw_int(argv[unsafe_offset=3])
        ).as_unsafe_any_origin()
        _enqueue_cached[_im2col[dt, fast]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            a,
            b,
            c,
            output,
            p,
            ds,
            dn,
            dk,
            dc,
            dw,
            dkw,
        )
    _ = ctx


def _loop32_allowed(p: Geometry, columns: Int, blocks: Int) -> Bool:
    # The large-input threshold and 32 blocks/SM were measured on H100.
    var pixels = Int(p.n) * Int(p.c) * Int(p.oh) * Int(p.ow)
    if pixels < 1048576 or p.kh * p.kw <= 1:
        return False
    for size in [
        p.n,
        p.c,
        p.h,
        p.w,
        p.oh,
        p.ow,
        p.kh,
        p.kw,
        p.sh,
        p.sw,
        p.ph,
        p.pw,
        p.dh,
        p.dw,
        p.og,
    ]:
        if size < 0 or size > 2147483647:
            return False
    if columns > 2147483647 or pixels > 2147483647 - blocks * BLOCK:
        return False
    if (
        p.n * p.c * p.h * p.w > 2147483647
        or p.n * p.og * 2 * p.kh * p.kw * p.oh * p.ow > 2147483647
    ):
        return False
    if (
        p.oh * p.sh + p.kh * p.dh + p.ph > 2147483647
        or p.ow * p.sw + p.kw * p.dw + p.pw > 2147483647
    ):
        return False
    return True


def _enqueue_loop[dt: DType](argv: Argv, p: Geometry, blocks: Int) raises:
    var ctx = _raw_ctx(argv[unsafe_offset=8])
    var x = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var off = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var mask = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var col = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    _enqueue_cached[_im2col_pixel_loop[dt]](
        ctx,
        blocks,
        1,
        1,
        BLOCK,
        x,
        off,
        mask,
        col,
        p,
        _divisor(Int(p.oh) * Int(p.ow)),
        _divisor(Int(p.n)),
        _divisor(Int(p.c) // Int(p.og)),
        _divisor(Int(p.ow)),
    )
    _ = ctx


def _enqueue_layout[
    dt: DType, op: Int, fast: Bool
](argv: Argv, p: Geometry, blocks: Int) raises:
    var ctx = _raw_ctx(argv[unsafe_offset=8])
    var a = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var b = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var out = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    var ds = DivisorArgs(fill=UInt32(0))
    var dd = ds.copy()
    comptime if fast:
        ds = _divisor(Int(p.oh) * Int(p.ow))
        dd = _divisor(Int(p.oc) if op == 3 else Int(p.n))
    _enqueue_cached[_layout[dt, op, fast]](
        ctx,
        blocks,
        1,
        1,
        BLOCK,
        a,
        b,
        out,
        p,
        ds,
        dd,
        Int64(_raw_int(argv[unsafe_offset=9])),
    )
    _ = ctx


def launch[dt: DType, op: Int](argv: Argv, argc: Int) raises:
    if argc != 10:
        raise Error("deform convolution expects 10 argument slots")
    var pp = argv[unsafe_offset=7]
    var p = Geometry(
        _raw_tuple_int(pp, 0),
        _raw_tuple_int(pp, 1),
        _raw_tuple_int(pp, 2),
        _raw_tuple_int(pp, 3),
        _raw_tuple_int(pp, 4),
        _raw_tuple_int(pp, 5),
        _raw_tuple_int(pp, 6),
        _raw_tuple_int(pp, 7),
        _raw_tuple_int(pp, 8),
        _raw_tuple_int(pp, 9),
        _raw_tuple_int(pp, 10),
        _raw_tuple_int(pp, 11),
        _raw_tuple_int(pp, 12),
        _raw_tuple_int(pp, 13),
        _raw_tuple_int(pp, 14),
        _raw_tuple_int(pp, 15),
        _raw_tuple_int(pp, 16),
    )
    var a = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var b = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var c = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var d = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    var e = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=4])
    ).as_unsafe_any_origin()
    var f = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=5])
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(argv[unsafe_offset=8])
    var count = (
        Int(p.n) * Int(p.c) * Int(p.kh) * Int(p.kw) * Int(p.oh) * Int(p.ow)
    )
    comptime if op == 2:
        count = (
            Int(p.n) * Int(p.og) * Int(p.kh) * Int(p.kw) * Int(p.oh) * Int(p.ow)
        )
    elif op == 3 or op == 4:
        count = Int(p.n) * Int(p.oc) * Int(p.oh) * Int(p.ow)
    elif op == 5:
        count = Int(p.oc) * Int(p.c) * Int(p.kh) * Int(p.kw)
    elif op == 6:
        count = Int(p.oc)
    if count == 0:
        return
    var blocks = min(ceildiv(Int(count), BLOCK), _device_sm_count(ctx) * 8)
    comptime if op == 0 or op == 1:
        comptime if op == 0 and has_nvidia_gpu_accelerator() and (
            dt == DType.float16 or dt == DType.float32
        ):
            var pixels = Int(p.n) * Int(p.c) * Int(p.oh) * Int(p.ow)
            var loop_blocks = min(
                ceildiv(pixels, BLOCK), _device_sm_count(ctx) * 32
            )
            if _loop32_allowed(p, Int(count), loop_blocks):
                _enqueue_loop[dt](argv, p, loop_blocks)
                return
        # Every positive divisor is bounded by the number of column elements.
        if count <= 4294967295:
            _enqueue_columns[dt, op == 1, True](argv, p, blocks)
        else:
            _enqueue_columns[dt, op == 1, False](argv, p, blocks)
    elif op == 2:
        _enqueue_cached[_offset_grad[dt]](
            ctx,
            blocks,
            1,
            1,
            BLOCK,
            a,
            b,
            c,
            d,
            e,
            f,
            p,
        )
    elif op == 6:
        _enqueue_cached[_bias_reduce[dt]](
            ctx,
            Int(p.oc),
            1,
            1,
            BLOCK,
            a,
            d,
            p,
            b,
            Int64(_raw_int(argv[unsafe_offset=2])),
        )
    else:
        comptime if op == 3 or op == 4:
            if count <= 4294967295:
                _enqueue_layout[dt, op, True](argv, p, blocks)
            else:
                _enqueue_layout[dt, op, False](argv, p, blocks)
        else:
            _enqueue_layout[dt, op, False](argv, p, blocks)
    _ = ctx


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime for dt in [DType.float16, DType.float32, DType.float64]:
            comptime if _dtype_arg_on[0, dt]():
                comptime if _op_on["Im2col"]():
                    launch[dt, 0](argv, argc)
                    return 0
                elif _op_on["Col2im"]():
                    launch[dt, 1](argv, argc)
                    return 0
                elif _op_on["OffsetGrad"]():
                    launch[dt, 2](argv, argc)
                    return 0
                elif _op_on["Output"]():
                    launch[dt, 3](argv, argc)
                    return 0
                elif _op_on["GradOutput"]():
                    launch[dt, 4](argv, argc)
                    return 0
                elif _op_on["Accumulate"]():
                    launch[dt, 5](argv, argc)
                    return 0
                elif _op_on["BiasGrad"]():
                    launch[dt, 6](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
