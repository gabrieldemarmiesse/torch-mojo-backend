# ===----------------------------------------------------------------------=== #
# Fast eager-mode conv2d support kernels for mojo_device — pure Mojo, no
# cuDNN. Convolution is lowered to (batched) im2col + the pure-Mojo GEMM in
# `matmul_ops`: the torch (K, C, R, S) weight is used as-is (the im2col row
# order matches its reduction order) and the matmul output is already NCHW.
#
# Both GPU and CPU MAX devices are supported: `_parallel_for` runs the
# `elementwise` framework on the host when `ctx.api() == "cpu"`.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.primitives import warp
from std.os import abort
from max.gpu.host import DeviceContext
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord as StdCoord

from op_utils import (
    _spec_unsupported,
    FLOAT_DTYPES,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _raw_tuple_len,
)

from variant_gates import _dtype_arg_on, _op_on, _register_call


# ---------------------------------------------------------------------------
# Batched im2col for NCHW input: builds the (N, C*KH*KW, OH*OW) patch
# matrix so that conv = weight.view(K, C*KH*KW) @ col[s]. Row order matches
# the reduction order of torch's (K, C, KH, KW) filter, so the weight can
# be used as-is (zero copy) and the matmul output is already NCHW. Rows are
# channel-major, so grouped convolution can slice the row range of each
# group with a plain element offset.
# ---------------------------------------------------------------------------


@always_inline
def _im2col_scalar[
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
    """One element per thread, on the generic `elementwise` framework.

    Kept for the CPU device only (`_parallel_for` also runs `elementwise` on
    host there). On GPU, `_im2col_gpu` below replaces this: profiling a
    32x64x56x56 k3s1 bf16 conv (benchmarks/test_vision.py's body shape)
    showed this scalar form at 86.7% of the whole aten::convolution's device
    time -- far more than the GEMM itself (9.4%) -- because every one of
    `batch*channels*kh*kw*out_h*out_w` threads redoes the full
    div/mod decomposition and moves one scalar element.
    """
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        var cols = out_h * out_w
        var crs = channels * kh * kw
        var s = i // (crs * cols)
        var r = (i // cols) % crs
        var j = i % cols
        var fw = r % kw
        var fh = (r // kw) % kh
        var c = r // (kw * kh)
        var oh = j // out_w
        var ow = j % out_w
        var ih = oh * stride_h - pad_h + fh * dil_h
        var iw = ow * stride_w - pad_w + fw * dil_w
        if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
            out_ptr[i] = Scalar[dtype](0)
        else:
            out_ptr[i] = in_ptr[((s * channels + c) * in_h + ih) * in_w + iw]

    _parallel_for[func](batch * channels * kh * kw * out_h * out_w, ctx)


# ---------------------------------------------------------------------------
# GPU im2col: one WARP (32 lanes, fixed) per (sample, channel, kh-tap,
# kw-tap, out-row), one lane per output column `ow` (striding by 32 when
# out_w > 32). `ncu` on the body benchmark shape (benchmarks/test_vision.py's
# N32xC64x56x56 k3s1, bf16) drove three iterations of this design, in order:
#
#   1. One thread per ROW (out_w-fold fewer threads than one-per-element,
#      each still repeating the full row decomposition once): 1.02ms,
#      barely faster than the scalar baseline's 1.21ms. `sm__throughput`
#      was 70.85% against `dram__throughput` at 2.64% -- compute-bound on
#      the decomposition's integer division, not memory -- but this design
#      also loses warp coalescing (consecutive threads then own different
#      rows `out_w` elements apart).
#   2. One warp-or-more (32-256 lanes, rounded from out_w) per row, every
#      lane repeating the decomposition: 1.28ms, SLOWER than the scalar
#      baseline, because col_threads >= out_w in the common case means
#      MORE total repeats (total_rows * col_threads) than the scalar form's
#      one-per-element (total_rows * out_w, i.e. once per (row, column)).
#      Coalescing alone does not pay for redoing the decomposition more.
#   3. The same split, but lane 0 alone computes the decomposition and
#      broadcasts (ih, the input row base, fw) through SHARED memory after
#      a block-wide `barrier()`: 1.28ms, no better than #2.
#      `smsp__average_warps_issue_stalled_barrier` jumped to 8.2 (from ~0)
#      -- the barrier, paid twice per row over hundreds of rows per block,
#      cost as much as the divisions it removed.
#
# This (both fixes at once, cheaply) is what actually wins: fixing
# `col_threads` at exactly one warp removes the barrier entirely -- lane 0
# computes, and `warp.shuffle_idx` broadcasts to the other 31 lanes with no
# shared memory and no block-wide sync, just a couple of register-to-register
# shuffle instructions every participating lane issues together. The
# decomposition then runs exactly once per row (`total_rows` times, against
# the scalar form's `total_rows * out_w`), and consecutive lanes still own
# consecutive `ow` -> consecutive addresses, so the store (always
# contiguous) and the stride_w == 1 read (the row is then a contiguous slice
# of the input row) stay warp-coalesced. stride_w != 1 (e.g. the k7s2 stem
# shape) keeps a per-lane bounds check -- reads aren't contiguous there --
# but the write still coalesces.
# ---------------------------------------------------------------------------

comptime _IM2COL_WARP = 32


@__name(t"im2col_row_{dtype}")
def _im2col_row_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_h_arg: Int64,
    in_w_arg: Int64,
    out_h_arg: Int64,
    out_w_arg: Int64,
    kh_arg: Int64,
    kw_arg: Int64,
    stride_h_arg: Int64,
    stride_w_arg: Int64,
    pad_h_arg: Int64,
    pad_w_arg: Int64,
    dil_h_arg: Int64,
    dil_w_arg: Int64,
    channels_arg: Int64,
    total_rows_arg: Int64,
):
    var in_h = Int(in_h_arg)
    var in_w = Int(in_w_arg)
    var out_h = Int(out_h_arg)
    var out_w = Int(out_w_arg)
    var kh = Int(kh_arg)
    var kw = Int(kw_arg)
    var stride_h = Int(stride_h_arg)
    var stride_w = Int(stride_w_arg)
    var pad_h = Int(pad_h_arg)
    var pad_w = Int(pad_w_arg)
    var dil_h = Int(dil_h_arg)
    var dil_w = Int(dil_w_arg)
    var channels = Int(channels_arg)
    var total_rows = Int(total_rows_arg)

    var crs = channels * kh * kw
    var row_stride = Int(grid_dim.x)
    var lane = Int(thread_idx.x)

    var row = Int(block_idx.x)
    while row < total_rows:
        # Only lane 0's values matter: `shuffle_idx(_, 0)` reads lane 0's
        # copy for every participating lane regardless of what its own
        # local variable holds, so the other lanes' zeros here are inert --
        # only lane 0 pays for the division.
        var ih_local = Int32(0)
        var fw_local = Int32(0)
        var in_row_local = Int32(0)
        if lane == 0:
            var s = row // (crs * out_h)
            var rem = row - s * (crs * out_h)
            var r = rem // out_h
            var oh = rem - r * out_h
            var fw = r % kw
            var fh = (r // kw) % kh
            var c = r // (kw * kh)
            var ih = oh * stride_h - pad_h + fh * dil_h
            ih_local = Int32(ih)
            fw_local = Int32(fw)
            if ih >= 0 and ih < in_h:
                in_row_local = Int32(((s * channels + c) * in_h + ih) * in_w)
        var ih = Int(warp.shuffle_idx(SIMD[DType.int32, 1](ih_local), 0)[0])
        var fw = Int(warp.shuffle_idx(SIMD[DType.int32, 1](fw_local), 0)[0])
        var in_row = Int(
            warp.shuffle_idx(SIMD[DType.int32, 1](in_row_local), 0)[0]
        )
        var out_row = row * out_w

        if ih < 0 or ih >= in_h:
            var ow = lane
            while ow < out_w:
                out_ptr[out_row + ow] = Scalar[dtype](0)
                ow += _IM2COL_WARP
        elif stride_w == 1:
            # iw(ow) = ow + iw0: a contiguous slice of the input row.
            var iw0 = fw * dil_w - pad_w
            var lo = max(0, -iw0)
            var hi = min(out_w, in_w - iw0)
            var ow = lane
            while ow < out_w:
                if ow < lo or ow >= hi:
                    out_ptr[out_row + ow] = Scalar[dtype](0)
                else:
                    out_ptr[out_row + ow] = in_ptr[in_row + iw0 + ow]
                ow += _IM2COL_WARP
        else:
            var ow = lane
            while ow < out_w:
                var iw = ow * stride_w - pad_w + fw * dil_w
                if iw >= 0 and iw < in_w:
                    out_ptr[out_row + ow] = in_ptr[in_row + iw]
                else:
                    out_ptr[out_row + ow] = Scalar[dtype](0)
                ow += _IM2COL_WARP

        row += row_stride


# One warp (32 lanes) per block is tiny next to a full SM, so one block per
# row, uncapped, oversubscribes the scheduler by orders of magnitude on real
# shapes -- the same failure a much bigger 64-lane block already showed
# (see the design note above: uncapped one-block-per-row measured 1.51ms,
# against the scalar baseline's 1.21ms, from over a million tiny blocks).
# `_gs_blocks` (op_utils) picks 4096 for its GS_THREADS=256-wide grid-stride
# launches; blocks here are 8x smaller (32 vs 256 lanes), so this scales the
# cap up by the same factor to keep a comparable number of resident lanes.
comptime _IM2COL_MAX_ROW_BLOCKS = 32768


@always_inline
def _im2col_gpu[
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
    var total_rows = batch * channels * kh * kw * out_h
    var blocks = min(max(total_rows, 1), _IM2COL_MAX_ROW_BLOCKS)
    _enqueue_cached[_im2col_row_kernel[dtype]](
        ctx,
        String(t"im2col_row_{dtype}"),
        blocks,
        1,
        1,
        _IM2COL_WARP,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        Int64(in_h),
        Int64(in_w),
        Int64(out_h),
        Int64(out_w),
        Int64(kh),
        Int64(kw),
        Int64(stride_h),
        Int64(stride_w),
        Int64(pad_h),
        Int64(pad_w),
        Int64(dil_h),
        Int64(dil_w),
        Int64(channels),
        Int64(total_rows),
    )


@always_inline
def _im2col[
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
    if ctx.api() == "cpu":
        _im2col_scalar[dtype](
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
        return
    # `_im2col_gpu` pulls in `_enqueue_cached`, which needs a target GPU
    # architecture at compile time (`gpu/host/info.mojo`). A runtime
    # `if`/`else` does not stop Mojo from instantiating the `else` branch
    # too (same hazard matmul_ops.mojo's MFMA gate documents), so on a host
    # with no accelerator at all this has to be a comptime gate, or the
    # whole module fails to build -- and with it every conv test, including
    # the CPU-device ones (AGENTS.md eager rule 1: a torch-cpu-only install
    # must still build this file). `ctx.api() == "cpu"` already returned
    # above, so this is unreachable when `has_accelerator()` is False; the
    # gate below exists only to keep it out of that host's build.
    comptime if has_accelerator():
        _im2col_gpu[dtype](
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


def _im2col_go(
    col_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    # (in_h, in_w, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w,
    #  dil_h, dil_w, channels, batch); batch defaults to 1 when omitted.
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
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
                _im2col[dt](
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


def _im2col_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _im2col_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# In-place per-channel bias add on a (batch, channels, plane) tensor:
# out[i] += bias[(i // plane) % channels].
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_chan_scalar[
    dtype: DType
](
    out_addr: Int,
    bias_addr: Int,
    total: Int,
    plane: Int,
    channels: Int,
    ctx: DeviceContext,
) raises:
    """One element per thread, on the generic `elementwise` framework.

    Kept for the CPU device only; `_bias_add_chan_gpu` below replaces this
    on GPU with the same one-warp-per-row restructuring `_im2col_gpu` uses,
    for the same reason: profiling the body benchmark shape
    (benchmarks/test_vision.py's N32xC64x56x56 k3s1, bf16) showed this
    scalar form was 26.9% of the whole aten::convolution's device time,
    bigger than the GEMM itself (24.7%), from recomputing `(i // plane) %
    channels` once per element instead of once per `plane`-long run.
    """
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[i] = out_ptr[i] + bias_ptr[(i // plane) % channels]

    _parallel_for[func](total, ctx)


# One warp (32 lanes) per (sample, channel) row of `plane` contiguous
# elements, one lane per element (striding by 32 when plane > 32) --
# `plane`-fold fewer redundant `(i // plane) % channels` computations than
# the scalar form, one per row instead of one per element. Unlike im2col's
# row-invariant metadata, `row % channels` is one mod plus one small,
# warp-uniform load (every lane reads the SAME `bias_ptr` element, which the
# memory system broadcasts on its own), so every lane just recomputes it --
# not worth a `warp.shuffle_idx` broadcast the way im2col's much heavier
# multi-division decomposition was. Consecutive lanes still own consecutive
# elements, so both the read and the write stay warp-coalesced.
@__name(t"bias_add_chan_row_{dtype}")
def _bias_add_chan_row_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    plane_arg: Int64,
    channels_arg: Int64,
    total_rows_arg: Int64,
):
    var plane = Int(plane_arg)
    var channels = Int(channels_arg)
    var total_rows = Int(total_rows_arg)
    var row_stride = Int(grid_dim.x)
    var lane = Int(thread_idx.x)

    var row = Int(block_idx.x)
    while row < total_rows:
        var bias_val = bias_ptr[row % channels]
        var base = row * plane
        var i = lane
        while i < plane:
            out_ptr[base + i] = out_ptr[base + i] + bias_val
            i += _IM2COL_WARP
        row += row_stride


@always_inline
def _bias_add_chan_gpu[
    dtype: DType
](
    out_addr: Int,
    bias_addr: Int,
    total: Int,
    plane: Int,
    channels: Int,
    ctx: DeviceContext,
) raises:
    var total_rows = total // plane if plane > 0 else 0
    var blocks = min(max(total_rows, 1), _IM2COL_MAX_ROW_BLOCKS)
    _enqueue_cached[_bias_add_chan_row_kernel[dtype]](
        ctx,
        String(t"bias_add_chan_row_{dtype}"),
        blocks,
        1,
        1,
        _IM2COL_WARP,
        _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](bias_addr).as_unsafe_any_origin(),
        Int64(plane),
        Int64(channels),
        Int64(total_rows),
    )


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
    if ctx.api() == "cpu" or plane <= 0 or total % plane != 0:
        _bias_add_chan_scalar[dtype](
            out_addr, bias_addr, total, plane, channels, ctx
        )
        return
    # Same comptime-gate requirement as `_im2col` above: `_bias_add_chan_gpu`
    # pulls in `_enqueue_cached`, which fails to build with no target GPU
    # architecture at all. `ctx.api() == "cpu"` already returned above, so
    # this is unreachable when `has_accelerator()` is False.
    comptime if has_accelerator():
        _bias_add_chan_gpu[dtype](
            out_addr, bias_addr, total, plane, channels, ctx
        )


def _bias_add_chan_go(
    out_ptr: PyObjectPtr,
    bias_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (plane, channels, total_elements)
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
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


def _bias_add_chan_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _bias_add_chan_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_conv_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("conv_ops")
        comptime if _op_on["Im2col"]():
            _register_call(
                b,
                _im2col_dispatcher,
                docstring=(
                    "batched NCHW im2col -> (N, C*KH*KW, OH*OW) patch matrix"
                ),
            )
        comptime if _op_on["BiasAddChan"]():
            _register_call(
                b,
                _bias_add_chan_dispatcher,
                docstring=(
                    "in-place out[i] += bias[(i // plane) % channels] on a"
                    " (batch, channels, plane) tensor"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create conv_ops python module: {e}")
