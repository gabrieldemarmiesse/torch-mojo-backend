"""Deformable convolution through dynamic im2col and existing Mojo GEMM."""
from abi import (
    T,
    Values,
    new_like,
    own,
    ret_tensor,
    unsupported,
    v_tensor,
    v_int,
    v_bool,
    view_strided,
    contiguous_strides,
    alert_not_deterministic,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from ops_common import fill_value, copy_strided_into
from ops_matmul import (
    Tmp,
    _new,
    _index_list,
    _mm_route,
    _transpose_2d,
    _sm90_cuda,
    _tf32_enabled,
)
from registry import Site, impl


def _matrix(base: T, rows: Int, cols: Int, offset: Int = 0) raises -> T:
    var shape = _index_list([rows, cols])
    return view_strided(
        base, shape, contiguous_strides(shape, 2), 2, base.offset + offset
    )


def _mm(a: T, b: T) raises -> T:
    var result = _mm_route(a, b)
    if not result:
        unsupported("deform_conv2d has no GEMM route for these operands")
    return result.value().copy()


def _forward_mm(a: T, b: T) raises -> T:
    # Shape thresholds measured on H100; reuse the existing strict-f32 TN route.
    if (
        a.dtype == DType.float32
        and a.dim(0) >= 128
        and b.dim(1) >= 8192
        and a.dim(1) >= 512
        and not _tf32_enabled()
        and _sm90_cuda(a.device)
    ):
        var transposed = own(_transpose_2d(a))
        var packed = Tmp(transposed.t)
        var logical = own(_transpose_2d(packed.t))
        var result = _mm(logical.t, b)
        _ = logical^
        _ = packed^
        _ = transposed^
        return result^
    return _mm(a, b)


def _run(
    op: String, x: T, ptrs: List[Int], p: List[Int], add: Bool = False
) raises:
    var ctx = ctx_for(x.device)
    var call = KernelCall("deform_conv", op)
    call.arg_dtype(0, x.dtype)
    for i in range(7):
        call.int(ptrs[i] if i < len(ptrs) else 0)
    call.tuple(p)
    call.int(ctx_ptr(ctx))
    call.int(Int(add))
    call.run()
    _ = ctx


def _parameters(
    args: Values, start: Int, x: T, w: T, off: T, mask: T, bias: T
) raises -> List[Int]:
    if not x.on_mojo() or dev(x.device)[].is_cpu:
        unsupported("deform_conv2d requires mojo GPU tensors")
    if (
        x.dtype != DType.float16
        and x.dtype != DType.float32
        and x.dtype != DType.float64
    ):
        unsupported("deform_conv2d supports float16, float32, and float64")
    if x.dtype == DType.float64 and dev(x.device)[].api == "metal":
        unsupported("deform_conv2d float64 is unavailable on Apple GPUs")
    for t in [w.copy(), off.copy(), mask.copy(), bias.copy()]:
        if not t.on_mojo() or t.device != x.device or t.dtype != x.dtype:
            unsupported("deform_conv2d operands must share device and dtype")
    if x.rank != 4 or w.rank != 4 or off.rank != 4:
        unsupported("deform_conv2d expects rank-four input, weight, and offset")
    var sh = v_int(args[unsafe_offset=start])
    var sw = v_int(args[unsafe_offset=start + 1])
    var ph = v_int(args[unsafe_offset=start + 2])
    var pw = v_int(args[unsafe_offset=start + 3])
    var dh = v_int(args[unsafe_offset=start + 4])
    var dw = v_int(args[unsafe_offset=start + 5])
    var groups = v_int(args[unsafe_offset=start + 6])
    var og = v_int(args[unsafe_offset=start + 7])
    var use_mask = v_bool(args[unsafe_offset=start + 8])
    if (
        sh <= 0
        or sw <= 0
        or dh <= 0
        or dw <= 0
        or ph < 0
        or pw < 0
        or groups <= 0
        or og <= 0
    ):
        unsupported("deform_conv2d has invalid convolution parameters")
    var n = x.dim(0)
    var c = x.dim(1)
    var h = x.dim(2)
    var wi = x.dim(3)
    var oc = w.dim(0)
    var kh = w.dim(2)
    var kw = w.dim(3)
    if c <= 0 or h <= 0 or wi <= 0 or oc <= 0 or kh <= 0 or kw <= 0:
        unsupported(
            "deform_conv2d channel and spatial dimensions must be positive"
        )
    var oh = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    var ow = (wi + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    if oh <= 0 or ow <= 0 or w.dim(1) * groups != c or oc % groups or c % og:
        unsupported(
            "deform_conv2d shapes are incompatible with convolution groups"
        )
    if (
        off.dim(0) != n
        or off.dim(1) != 2 * og * kh * kw
        or off.dim(2) != oh
        or off.dim(3) != ow
    ):
        unsupported("deform_conv2d offset shape is invalid")
    if use_mask:
        if (
            mask.rank != 4
            or mask.dim(0) != n
            or mask.dim(1) != og * kh * kw
            or mask.dim(2) != oh
            or mask.dim(3) != ow
        ):
            unsupported("deform_conv2d mask shape is invalid")
    if bias.rank != 1 or bias.dim(0) != oc:
        unsupported("deform_conv2d bias shape is invalid")
    return [
        n,
        c,
        h,
        wi,
        oc,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        og,
        Int(use_mask),
    ]


def _deform[backward: Bool](args: Values, rets: Values) raises:
    comptime shift = 1 if backward else 0
    var input = v_tensor(args[unsafe_offset=shift])
    var weight = v_tensor(args[unsafe_offset=shift + 1])
    var offset = v_tensor(args[unsafe_offset=shift + 2])
    var mask = v_tensor(args[unsafe_offset=shift + 3])
    var bias = v_tensor(args[unsafe_offset=shift + 4])
    var p = _parameters(args, shift + 5, input, weight, offset, mask, bias)
    var groups = v_int(args[unsafe_offset=shift + 11])
    var n = p[0]
    var c = p[1]
    var h = p[2]
    var wi = p[3]
    var oc = p[4]
    var ks = p[5] * p[6]
    var s = p[7] * p[8]
    var og = p[15]
    var x = Tmp(input)
    var w = Tmp(weight)
    var off = Tmp(offset)
    var m = Tmp(mask)
    var bi = Tmp(bias)
    var st = input.stype
    var device = input.device
    var bytes = input.itemsize
    var chunk = min(n, 32)
    while chunk > 1 and n % chunk:
        chunk -= 1
    var cols = own(_new([c * ks, chunk * s], st, device))
    comptime if backward:
        var grad = v_tensor(args[unsafe_offset=0])
        if (
            not grad.on_mojo()
            or grad.device != device
            or grad.dtype != input.dtype
            or grad.rank != 4
        ):
            unsupported(
                "deform_conv2d gradient must match input device and dtype"
            )
        if (
            grad.dim(0) != n
            or grad.dim(1) != oc
            or grad.dim(2) != p[7]
            or grad.dim(3) != p[8]
        ):
            unsupported("deform_conv2d gradient shape is invalid")
        var g = Tmp(grad)
        var gx = own(
            _new(
                input.logical_shape(),
                st,
                device,
            )
        )
        var gw = own(new_like(weight))
        var go = own(new_like(offset))
        var gm = own(new_like(mask))
        var gb = own(new_like(bias))
        fill_value(gx.t, 0.0)
        if n == 0:
            fill_value(gw.t, 0.0)
        var gc = own(
            _new([0], st, device) if groups
            == 1 else _new([c * ks, chunk * s], st, device)
        )
        var gy = own(_new([oc, chunk * s], st, device))
        for begin in range(0, n, max(chunk, 1)):
            p[0] = chunk
            var xp = x.t.ptr + begin * c * h * wi * bytes
            var op = off.t.ptr + begin * og * 2 * ks * s * bytes
            var mp = m.t.ptr + begin * og * ks * s * bytes if p[16] else m.t.ptr
            _run(
                "GradOutput",
                input,
                [g.t.ptr + begin * oc * s * bytes, 0, 0, gy.t.ptr],
                p,
            )
            _run("Im2col", input, [xp, op, mp, cols.t.ptr], p)
            for group in range(groups):
                var wg = own(
                    _matrix(
                        w.t,
                        oc // groups,
                        c * ks // groups,
                        group * (oc // groups) * (c * ks // groups),
                    )
                )
                var wt = own(_transpose_2d(wg.t))
                var yg = own(
                    _matrix(
                        gy.t,
                        oc // groups,
                        chunk * s,
                        group * (oc // groups) * chunk * s,
                    )
                )
                var dc = own(_mm(wt.t, yg.t))
                if groups == 1:
                    gc = own(dc.take())
                else:
                    var dc_dst = own(
                        _matrix(
                            gc.t,
                            c * ks // groups,
                            chunk * s,
                            group * (c * ks // groups) * chunk * s,
                        )
                    )
                    copy_strided_into(dc_dst.t, dc.t)
                    _ = dc_dst^
                var cg = own(
                    _matrix(
                        cols.t,
                        c * ks // groups,
                        chunk * s,
                        group * (c * ks // groups) * chunk * s,
                    )
                )
                var ct = own(_transpose_2d(cg.t))
                var dw = own(_mm(yg.t, ct.t))
                var ap = p.copy()
                ap[1] = c // groups
                ap[4] = oc // groups
                _run(
                    "Accumulate",
                    input,
                    [
                        dw.t.ptr,
                        0,
                        0,
                        gw.t.ptr
                        + group * (oc // groups) * (c * ks // groups) * bytes,
                    ],
                    ap,
                    add=begin != 0,
                )
                _ = wg^
                _ = wt^
                _ = yg^
                _ = dc^
                _ = cg^
                _ = ct^
                _ = dw^
            _run(
                "OffsetGrad",
                input,
                [
                    xp,
                    op,
                    mp,
                    gc.t.ptr,
                    go.t.ptr + begin * og * 2 * ks * s * bytes,
                    gm.t.ptr
                    + begin * og * ks * s * bytes if p[16] else gm.t.ptr,
                ],
                p,
            )
            alert_not_deterministic("compute_grad_input")
            _run(
                "Col2im",
                input,
                [
                    gc.t.ptr,
                    op,
                    mp,
                    gx.t.ptr + begin * c * h * wi * gx.t.itemsize,
                ],
                p,
            )
        p[0] = n
        _run(
            "BiasGrad",
            input,
            [g.t.ptr, gm.t.ptr, gm.t.numel if not p[16] else 0, gb.t.ptr],
            p,
        )
        ret_tensor(rets, 0, gx.take())
        ret_tensor(rets, 1, gw.take())
        ret_tensor(rets, 2, go.take())
        ret_tensor(rets, 3, gm.take())
        ret_tensor(rets, 4, gb.take())
        _ = gx^
        _ = gc^
        _ = gy^
        _ = g^
    else:
        var output = own(_new([n, oc, p[7], p[8]], st, device))
        var buf = own(
            _new([0], st, device) if groups
            == 1 else _new([oc, chunk * s], st, device)
        )
        for begin in range(0, n, max(chunk, 1)):
            p[0] = chunk
            _run(
                "Im2col",
                input,
                [
                    x.t.ptr + begin * c * h * wi * bytes,
                    off.t.ptr + begin * og * 2 * ks * s * bytes,
                    m.t.ptr + begin * og * ks * s * bytes if p[16] else m.t.ptr,
                    cols.t.ptr,
                ],
                p,
            )
            for group in range(groups):
                var wg = own(
                    _matrix(
                        w.t,
                        oc // groups,
                        c * ks // groups,
                        group * (oc // groups) * (c * ks // groups),
                    )
                )
                var cg = own(
                    _matrix(
                        cols.t,
                        c * ks // groups,
                        chunk * s,
                        group * (c * ks // groups) * chunk * s,
                    )
                )
                var result = own(_forward_mm(wg.t, cg.t))
                if groups == 1:
                    buf = own(result.take())
                else:
                    var dest = own(
                        _matrix(
                            buf.t,
                            oc // groups,
                            chunk * s,
                            group * (oc // groups) * chunk * s,
                        )
                    )
                    copy_strided_into(dest.t, result.t)
                    _ = dest^
                _ = wg^
                _ = cg^
                _ = result^
            _run(
                "Output",
                input,
                [buf.t.ptr, bi.t.ptr, 0, output.t.ptr + begin * oc * s * bytes],
                p,
            )
        ret_tensor(rets, 0, output.take())
        _ = buf^
    _ = cols^
    _ = x^
    _ = w^
    _ = off^
    _ = m^
    _ = bi^


def op_deform_conv2d(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _deform[False](args, rets)


def op_deform_conv2d_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _deform[True](args, rets)


def register_deform_conv(site: Site) raises:
    impl[op_deform_conv2d, "torchvision::deform_conv2d"](site)
    impl[op_deform_conv2d_backward, "torchvision::_deform_conv2d_backward"](
        site
    )
