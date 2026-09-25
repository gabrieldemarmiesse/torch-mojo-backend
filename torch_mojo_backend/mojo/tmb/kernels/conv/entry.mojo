# ===----------------------------------------------------------------------=== #
# Fast eager-mode conv2d support kernels for mojo_device — pure Mojo, no
# cuDNN. Convolution is lowered to (batched) im2col + the pure-Mojo GEMM in
# `matmul`: the torch (K, C, R, S) weight is used as-is (the im2col row
# order matches its reduction order) and the matmul output is already NCHW.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from max.gpu.host import DeviceContext
from std.utils.coord import Coord as StdCoord

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
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


# ---------------------------------------------------------------------------
# Batched im2col for NCHW input: builds the (N, C*KH*KW, OH*OW) patch
# matrix so that conv = weight.view(K, C*KH*KW) @ col[s]. Row order matches
# the reduction order of torch's (K, C, KH, KW) filter, so the weight can
# be used as-is (zero copy) and the matmul output is already NCHW. Rows are
# channel-major, so grouped convolution can slice the row range of each
# group with a plain element offset.
#
# `patch_major` lays the same elements out as (C*KH*KW, N, OH*OW) instead:
# one row per filter tap holding every sample's output pixels back to back.
# That is the layout the convolution backward wants, because it folds the
# batch into the GEMM's reduction: grad_weight = grad_out(K, N*OH*OW) @
# col^T is then ONE GEMM with no per-sample partials to sum afterwards.
# ---------------------------------------------------------------------------


@always_inline
def _im2col[
    dtype: DType, patch_major: Bool = False
](
    out_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    dil_h: Int,
    dil_w: Int,
    channels: Int,
    batch: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        var cols = out_h * out_w
        var crs = channels * kh * kw
        var s: Int
        var r: Int
        comptime if patch_major:
            r = i // (batch * cols)
            s = (i // cols) % batch
        else:
            s = i // (crs * cols)
            r = (i // cols) % crs
        var j = i % cols
        var fw = r % kw
        var fh = (r // kw) % kh
        var c = r // (kw * kh)
        var oh = j // out_w
        var ow = j % out_w
        var ih = oh * stride_h - pad_h + fh * dil_h
        var iw = ow * stride_w - pad_w + fw * dil_w
        if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
            out_ptr[unsafe_offset=i] = Scalar[dtype](0)
        else:
            out_ptr[unsafe_offset=i] = in_ptr[
                unsafe_offset=((s * channels + c) * in_h + ih) * in_w + iw
            ]

    _parallel_for[func](batch * channels * kh * kw * out_h * out_w, ctx)


# ---------------------------------------------------------------------------
# col2im: the exact adjoint of the patch-major im2col above, for the conv
# backward's data gradient. im2col GATHERS one input pixel per (tap, sample,
# output pixel); its adjoint sums every column element back onto the pixel
# it was read from.
#
# Written as a gather over the columns -- one thread owns one input pixel
# and loops over the taps that could have read it -- rather than as a
# scatter with atomics: no atomics at all, and bit-for-bit deterministic
# (an atomic scatter sums a pixel's taps in launch order, so two runs of the
# same backward could differ).
#
# The tap test inverts the forward index map ih = oh * stride_h - pad_h +
# r * dil_h for a fixed ih: oh exists iff th = ih + pad_h - r * dil_h is
# non-negative, divisible by stride_h and inside [0, out_h). th >= 0 is
# tested before the modulo, so `%` never sees a negative dividend.
#
# Accumulation is float32 for every dtype, as torch's own col2im does.
# ---------------------------------------------------------------------------


@always_inline
def _col2im[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    dil_h: Int,
    dil_w: Int,
    channels: Int,
    batch: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        var cols = out_h * out_w
        # One patch-major row is (batch, out_h * out_w) long.
        var row_len = batch * cols
        var iw = i % in_w
        var ih = (i // in_w) % in_h
        var c = (i // (in_w * in_h)) % channels
        var s = i // (in_w * in_h * channels)
        var sample_base = s * cols
        var acc = Scalar[DType.float32](0)
        for fh in range(kh):
            var th = ih + pad_h - fh * dil_h
            if th < 0 or th % stride_h != 0:
                continue
            var oh = th // stride_h
            if oh >= out_h:
                continue
            var tap_row = (c * kh + fh) * kw
            for fw in range(kw):
                var tw = iw + pad_w - fw * dil_w
                if tw < 0 or tw % stride_w != 0:
                    continue
                var ow = tw // stride_w
                if ow >= out_w:
                    continue
                acc += in_ptr[
                    unsafe_offset=(tap_row + fw) * row_len
                    + sample_base
                    + oh * out_w
                    + ow
                ].cast[DType.float32]()
        out_ptr[unsafe_offset=i] = acc.cast[dtype]()

    _parallel_for[func](batch * channels * in_h * in_w, ctx)


comptime _IM2COL = 0
comptime _IM2COL_PATCH_MAJOR = 1
comptime _COL2IM = 2


def _im2col_go[
    kind: Int
](
    col_ptr: Arg,
    in_ptr: Arg,
    # (in_h, in_w, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w,
    #  dil_h, dil_w, channels, batch); batch defaults to 1 when omitted.
    # For _COL2IM the two pointers keep their im2col roles: `col_ptr` is the
    # IMAGE written and `in_ptr` the patch-major columns read.
    params: Arg,
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(col_ptr)
    var in_addr = _raw_int(in_ptr)
    var in_h = _raw_tuple_int(params, 0)
    var in_w = _raw_tuple_int(params, 1)
    var out_h = _raw_tuple_int(params, 2)
    var out_w = _raw_tuple_int(params, 3)
    var kh = _raw_tuple_int(params, 4)
    var kw = _raw_tuple_int(params, 5)
    var stride_h = _raw_tuple_int(params, 6)
    var stride_w = _raw_tuple_int(params, 7)
    var pad_h = _raw_tuple_int(params, 8)
    var pad_w = _raw_tuple_int(params, 9)
    var dil_h = _raw_tuple_int(params, 10)
    var dil_w = _raw_tuple_int(params, 11)
    var channels = _raw_tuple_int(params, 12)
    var batch = _raw_tuple_int(params, 13) if _raw_tuple_len(params) > 13 else 1
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                comptime if kind == _COL2IM:
                    _col2im[dt](
                        out_addr,
                        in_addr,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        kh,
                        kw,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        dil_h,
                        dil_w,
                        channels,
                        batch,
                        ctx,
                    )
                else:
                    _im2col[dt, kind == _IM2COL_PATCH_MAJOR](
                        out_addr,
                        in_addr,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        kh,
                        kw,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        dil_h,
                        dil_w,
                        channels,
                        batch,
                        ctx,
                    )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast im2col: " + String(dtype))


def _im2col_dispatcher[kind: Int](argv: Argv, argc: Int) raises:
    var args = argv
    _im2col_go[kind](
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
    )


# ---------------------------------------------------------------------------
# In-place per-channel bias add on a (batch, channels, plane) tensor:
# out[i] += bias[(i // plane) % channels].
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_chan[
    dtype: DType
](
    out_addr: Int,
    bias_addr: Int,
    total: Int,
    plane: Int,
    channels: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[unsafe_offset=i] = (
            out_ptr[unsafe_offset=i]
            + bias_ptr[unsafe_offset=(i // plane) % channels]
        )

    _parallel_for[func](total, ctx)


def _bias_add_chan_go(
    out_ptr: Arg,
    bias_ptr: Arg,
    params: Arg,  # (plane, channels, total_elements)
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr)
    var bias_addr = _raw_int(bias_ptr)
    var plane_val = _raw_tuple_int(params, 0)
    var channels = _raw_tuple_int(params, 1)
    var total = _raw_tuple_int(params, 2)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _bias_add_chan[dt](
                    out_addr, bias_addr, total, plane_val, channels, ctx
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


def _bias_add_chan_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _bias_add_chan_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
    )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Im2col"]():
            _im2col_dispatcher[_IM2COL](argv, argc)
            return 0
        comptime if _op_on["Im2colPatchMajor"]():
            _im2col_dispatcher[_IM2COL_PATCH_MAJOR](argv, argc)
            return 0
        comptime if _op_on["Col2im"]():
            _im2col_dispatcher[_COL2IM](argv, argc)
            return 0
        comptime if _op_on["BiasAddChan"]():
            _bias_add_chan_dispatcher(argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
