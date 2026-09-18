"""Native region-of-interest sampling and pooling."""

from std.utils import IndexList

from abi import (
    ST_INT32,
    T,
    Values,
    alert_not_deterministic,
    new_tensor,
    own,
    own_if_new,
    ret_tensor,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_tensor,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import contiguous
from registry import Site, impl


def _check_pair(input: T, rois: T) raises:
    if not input.on_mojo() or not rois.on_mojo() or input.device != rois.device:
        unsupported("ROI operands must be on the same mojo device")
    if dev(input.device)[].is_cpu:
        unsupported("ROI operations require a mojo GPU")
    if (
        input.dtype != DType.float16
        and input.dtype != DType.float32
        and input.dtype != DType.float64
    ):
        unsupported("ROI operations support float16, float32, and float64")
    if input.dtype == DType.float64 and dev(input.device)[].api == "metal":
        unsupported("ROI float64 is unavailable on Apple GPUs")
    if rois.dtype != input.dtype:
        unsupported("ROI coordinates and input must have the same dtype")
    if input.rank != 4 or rois.rank != 2 or rois.dim(1) != 5:
        unsupported("ROI operations require NCHW input and K x 5 coordinates")


def _shape(n: Int, c: Int, h: Int, w: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 4] = n
    shape[MAX_RANK - 3] = c
    shape[MAX_RANK - 2] = h
    shape[MAX_RANK - 1] = w
    return shape


def _run(
    op: String,
    input: T,
    rois: T,
    output: T,
    argmax_ptr: Int,
    n: Int,
    c: Int,
    h: Int,
    w: Int,
    ph: Int,
    pw: Int,
    scale: Float64,
    sampling: Int,
    aligned: Bool,
) raises:
    var ctx = ctx_for(input.device)
    var call = KernelCall("roi_ops", op)
    call.arg_dtype(0, input.dtype)
    call.int(input.ptr)
    call.int(rois.ptr)
    call.int(output.ptr)
    call.int(argmax_ptr)
    call.int(n)
    call.int(c)
    call.int(h)
    call.int(w)
    call.int(rois.dim(0))
    call.int(ph)
    call.int(pw)
    call.f64(scale)
    call.int(sampling)
    call.int(Int(aligned))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _forward[pool: Bool](args: Values, rets: Values) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var rois = v_tensor(args[unsafe_offset=1])
    _check_pair(input, rois)
    var ph = v_int(args[unsafe_offset=3])
    var pw = v_int(args[unsafe_offset=4])
    if ph <= 0 or pw <= 0 or input.dim(2) <= 0 or input.dim(3) <= 0:
        unsupported("ROI pooled and input spatial dimensions must be positive")
    var x = own_if_new(contiguous(input), input)
    var r = own_if_new(contiguous(rois), rois)
    var shape = _shape(rois.dim(0), input.dim(1), ph, pw)
    var out = own(new_tensor(shape, 4, input.stype, input.device))
    comptime if pool:
        var indices = own(new_tensor(shape, 4, ST_INT32, input.device))
        if out.t.numel:
            _run(
                "RoiPoolForward",
                x.t,
                r.t,
                out.t,
                indices.t.ptr,
                input.dim(0),
                input.dim(1),
                input.dim(2),
                input.dim(3),
                ph,
                pw,
                v_f64(args[unsafe_offset=2]),
                0,
                False,
            )
        ret_tensor(rets, 1, indices.take())
    else:
        if out.t.numel:
            _run(
                "RoiAlignForward",
                x.t,
                r.t,
                out.t,
                0,
                input.dim(0),
                input.dim(1),
                input.dim(2),
                input.dim(3),
                ph,
                pw,
                v_f64(args[unsafe_offset=2]),
                v_int(args[unsafe_offset=5]),
                v_bool(args[unsafe_offset=6]),
            )
    ret_tensor(rets, 0, out.take())
    _ = x^
    _ = r^


def _backward[pool: Bool](args: Values, rets: Values) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var rois = v_tensor(args[unsafe_offset=1])
    _check_pair(grad, rois)
    comptime s = 3 if pool else 2
    var ph = v_int(args[unsafe_offset=s + 1])
    var pw = v_int(args[unsafe_offset=s + 2])
    var n = v_int(args[unsafe_offset=s + 3])
    var c = v_int(args[unsafe_offset=s + 4])
    var h = v_int(args[unsafe_offset=s + 5])
    var w = v_int(args[unsafe_offset=s + 6])
    if ph <= 0 or pw <= 0 or h <= 0 or w <= 0 or n < 0 or c < 0:
        unsupported("ROI backward dimensions are invalid")
    if (
        grad.dim(0) != rois.dim(0)
        or grad.dim(1) != c
        or grad.dim(2) != ph
        or grad.dim(3) != pw
    ):
        unsupported(
            "ROI backward gradient shape does not match pooled dimensions"
        )
    if grad.numel:
        alert_not_deterministic(
            "roi_pool_backward_kernel" if pool else "roi_align_backward_kernel"
        )
    var g = own_if_new(contiguous(grad), grad)
    var r = own_if_new(contiguous(rois), rois)
    var accumulation_type = grad.stype
    var out = own(
        new_tensor(_shape(n, c, h, w), 4, accumulation_type, grad.device)
    )
    comptime if pool:
        var indices = v_tensor(args[unsafe_offset=2])
        if (
            not indices.on_mojo()
            or indices.device != grad.device
            or indices.stype != ST_INT32
            or indices.rank != 4
        ):
            unsupported(
                "ROI pool argmax must be an int32 tensor on the same mojo"
                " device"
            )
        for dim in range(4):
            if indices.dim(dim) != grad.dim(dim):
                unsupported("ROI pool argmax shape must match gradient")
        var idx = own_if_new(contiguous(indices), indices)
        if out.t.numel:
            _run(
                "RoiPoolBackward",
                g.t,
                r.t,
                out.t,
                idx.t.ptr,
                n,
                c,
                h,
                w,
                ph,
                pw,
                v_f64(args[unsafe_offset=s]),
                0,
                False,
            )
        _ = idx^
    else:
        if out.t.numel:
            _run(
                "RoiAlignBackward",
                g.t,
                r.t,
                out.t,
                0,
                n,
                c,
                h,
                w,
                ph,
                pw,
                v_f64(args[unsafe_offset=s]),
                v_int(args[unsafe_offset=9]),
                v_bool(args[unsafe_offset=10]),
            )
    ret_tensor(rets, 0, out.take())
    _ = out^
    _ = g^
    _ = r^


def op_roi_align(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _forward[False](args, rets)


def op_roi_align_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _backward[False](args, rets)


def op_roi_pool(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _forward[True](args, rets)


def op_roi_pool_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _backward[True](args, rets)


def _ps_forward[pool: Bool](args: Values, rets: Values) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var rois = v_tensor(args[unsafe_offset=1])
    _check_pair(input, rois)
    var ph = v_int(args[unsafe_offset=3])
    var pw = v_int(args[unsafe_offset=4])
    if ph <= 0 or pw <= 0 or input.dim(2) <= 0 or input.dim(3) <= 0:
        unsupported(
            "PS ROI pooled and input spatial dimensions must be positive"
        )
    if input.dim(1) % (ph * pw):
        unsupported(
            "PS ROI input channels must be divisible by pooled height * width"
        )
    var shape = _shape(rois.dim(0), input.dim(1) // (ph * pw), ph, pw)
    var out = own(new_tensor(shape, 4, input.stype, input.device))
    var mapping = own(new_tensor(shape, 4, ST_INT32, input.device))
    var x = own_if_new(contiguous(input), input)
    var r = own_if_new(contiguous(rois), rois)
    var sampling = 0
    comptime if not pool:
        sampling = v_int(args[unsafe_offset=5])
    if out.t.numel:
        _run(
            "PsRoiPoolForward" if pool else "PsRoiAlignForward",
            x.t,
            r.t,
            out.t,
            mapping.t.ptr,
            input.dim(0),
            input.dim(1),
            input.dim(2),
            input.dim(3),
            ph,
            pw,
            v_f64(args[unsafe_offset=2]),
            sampling,
            True,
        )
    ret_tensor(rets, 0, out.take())
    ret_tensor(rets, 1, mapping.take())
    _ = x^
    _ = r^


def _ps_backward[pool: Bool](args: Values, rets: Values) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var rois = v_tensor(args[unsafe_offset=1])
    var mapping = v_tensor(args[unsafe_offset=2])
    _check_pair(grad, rois)
    var ph = v_int(args[unsafe_offset=4])
    var pw = v_int(args[unsafe_offset=5])
    comptime s = 6 if pool else 7
    var n = v_int(args[unsafe_offset=s])
    var c = v_int(args[unsafe_offset=s + 1])
    var h = v_int(args[unsafe_offset=s + 2])
    var w = v_int(args[unsafe_offset=s + 3])
    if ph <= 0 or pw <= 0 or h <= 0 or w <= 0 or n < 0 or c < 0:
        unsupported("PS ROI backward dimensions are invalid")
    if c % (ph * pw):
        unsupported(
            "PS ROI input channels must be divisible by pooled height * width"
        )
    if (
        grad.dim(0) != rois.dim(0)
        or grad.dim(1) != c // (ph * pw)
        or grad.dim(2) != ph
        or grad.dim(3) != pw
    ):
        unsupported(
            "PS ROI backward gradient shape does not match pooled dimensions"
        )
    if (
        not mapping.on_mojo()
        or mapping.device != grad.device
        or mapping.stype != ST_INT32
        or mapping.rank != 4
    ):
        unsupported(
            "PS ROI channel mapping must be an int32 tensor on the same mojo"
            " device"
        )
    for dim in range(4):
        if mapping.dim(dim) != grad.dim(dim):
            unsupported("PS ROI channel mapping shape must match gradient")
    if grad.numel:
        alert_not_deterministic(
            "ps_roi_pool_backward_kernel" if pool else "ps_roi_align_backward_kernel"
        )
    var g = own_if_new(contiguous(grad), grad)
    var r = own_if_new(contiguous(rois), rois)
    var m = own_if_new(contiguous(mapping), mapping)
    var ctx = ctx_for(grad.device)
    var out = own(
        new_tensor(
            _shape(n, c, h, w),
            4,
            grad.stype,
            grad.device,
        )
    )
    var sampling = 0
    comptime if not pool:
        sampling = v_int(args[unsafe_offset=6])
    if out.t.numel:
        _run(
            "PsRoiPoolBackward" if pool else "PsRoiAlignBackward",
            g.t,
            r.t,
            out.t,
            m.t.ptr,
            n,
            c,
            h,
            w,
            ph,
            pw,
            v_f64(args[unsafe_offset=3]),
            sampling,
            True,
        )
    ret_tensor(rets, 0, out.take())
    _ = out^
    _ = g^
    _ = r^
    _ = m^
    _ = ctx


def op_ps_roi_align(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _ps_forward[False](args, rets)


def op_ps_roi_align_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _ps_backward[False](args, rets)


def op_ps_roi_pool(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _ps_forward[True](args, rets)


def op_ps_roi_pool_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _ps_backward[True](args, rets)


def register_roi(site: Site) raises:
    impl[op_ps_roi_align, "torchvision::ps_roi_align"](site)
    impl[op_ps_roi_align_backward, "torchvision::_ps_roi_align_backward"](site)
    impl[op_ps_roi_pool, "torchvision::ps_roi_pool"](site)
    impl[op_ps_roi_pool_backward, "torchvision::_ps_roi_pool_backward"](site)
    impl[op_roi_align, "torchvision::roi_align"](site)
    impl[op_roi_align_backward, "torchvision::_roi_align_backward"](site)
    impl[op_roi_pool, "torchvision::roi_pool"](site)
    impl[op_roi_pool_backward, "torchvision::_roi_pool_backward"](site)
