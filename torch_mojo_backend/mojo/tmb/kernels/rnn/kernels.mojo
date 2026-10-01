# ===----------------------------------------------------------------------=== #
# Fused LSTM / GRU cell gate math (the pointwise half of a recurrent cell; the
# two GEMMs before it are ordinary mm/addmm calls).
#
# A port of aten/src/ATen/native/cuda/RNN.cu (v2.14.0): one thread per
# (batch row, hidden unit), every gate read in the tensor dtype, the math in
# float32 (CUDA's `accscalar_t`), each stored value rounded once. Operands are
# dense row-major; the op layer (tmb/ops/rnn.mojo) makes them so.
#
# Layouts (H = hidden size, N = batch):
#   LSTM gates     (N, 4H): i | f | c | o
#   LSTM workspace (N, 4H): the activated gates, same order
#   GRU gates      (N, 3H): r | i | n
#   GRU workspace  (N, 5H): r | i | n | hx | hn + b_hn
# ===----------------------------------------------------------------------=== #

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.sys.info import has_accelerator

from tmb.kernels.common.op_utils import (
    FILL_THREADS,
    _enqueue_cached,
    _fill_blocks,
    _make_ptr,
)
from tmb.kernels.common.unary_math import _float_unary

comptime F32 = DType.float32


@always_inline
def _sigmoid(x: Float32) -> Float32:
    return _float_unary["sigmoid", F32, 1](x)


@always_inline
def _tanh(x: Float32) -> Float32:
    return _float_unary["tanh", F32, 1](x)


@always_inline
def _ld[
    dtype: DType
](p: Pointer[Scalar[dtype], MutAnyOrigin], i: Int) -> Float32:
    return p[unsafe_offset=i].cast[F32]()


@__name(t"lstm_cell_fwd_{dtype}")
def _lstm_cell_fwd_kernel[
    dtype: DType
](
    input_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    hidden_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    bias1: Pointer[Scalar[dtype], MutAnyOrigin],
    bias2: Pointer[Scalar[dtype], MutAnyOrigin],
    cx: Pointer[Scalar[dtype], MutAnyOrigin],
    hy: Pointer[Scalar[dtype], MutAnyOrigin],
    cy: Pointer[Scalar[dtype], MutAnyOrigin],
    workspace: Pointer[Scalar[dtype], MutAnyOrigin],
    hsz_arg: Int64,
    total_arg: Int64,
    has_bias_arg: Int64,
):
    var hsz = Int(hsz_arg)
    var total = Int(total_arg)
    var has_bias = has_bias_arg != 0
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var h = i % hsz
        var off = (i // hsz) * 4 * hsz + h
        # bias1 + bias2 are added one at a time in float, as CUDA does.
        var gi = _ld(input_gates, off) + _ld(hidden_gates, off)
        var gf = _ld(input_gates, off + hsz) + _ld(hidden_gates, off + hsz)
        var gc = _ld(input_gates, off + 2 * hsz) + _ld(
            hidden_gates, off + 2 * hsz
        )
        var go = _ld(input_gates, off + 3 * hsz) + _ld(
            hidden_gates, off + 3 * hsz
        )
        if has_bias:
            gi = gi + _ld(bias1, h) + _ld(bias2, h)
            gf = gf + _ld(bias1, h + hsz) + _ld(bias2, h + hsz)
            gc = gc + _ld(bias1, h + 2 * hsz) + _ld(bias2, h + 2 * hsz)
            go = go + _ld(bias1, h + 3 * hsz) + _ld(bias2, h + 3 * hsz)
        var ig = _sigmoid(gi)
        var fg = _sigmoid(gf)
        var cg = _tanh(gc)
        var og = _sigmoid(go)
        var f_cy = (fg * _ld(cx, i)) + (ig * cg)
        var f_hy = og * _tanh(f_cy)
        hy[unsafe_offset=i] = f_hy.cast[dtype]()
        cy[unsafe_offset=i] = f_cy.cast[dtype]()
        workspace[unsafe_offset=off] = ig.cast[dtype]()
        workspace[unsafe_offset=off + hsz] = fg.cast[dtype]()
        workspace[unsafe_offset=off + 2 * hsz] = cg.cast[dtype]()
        workspace[unsafe_offset=off + 3 * hsz] = og.cast[dtype]()
        i += stride


@__name(t"lstm_cell_bwd_{dtype}")
def _lstm_cell_bwd_kernel[
    dtype: DType
](
    workspace: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    cx: Pointer[Scalar[dtype], MutAnyOrigin],
    cy: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_hy: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_cy: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_cx: Pointer[Scalar[dtype], MutAnyOrigin],
    hsz_arg: Int64,
    total_arg: Int64,
    has_grad_hy_arg: Int64,
    has_grad_cy_arg: Int64,
):
    var hsz = Int(hsz_arg)
    var total = Int(total_arg)
    var has_ghy = has_grad_hy_arg != 0
    var has_gcy = has_grad_cy_arg != 0
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var off = (i // hsz) * 4 * hsz + i % hsz
        var ig = _ld(workspace, off)
        var fg = _ld(workspace, off + hsz)
        var cg = _ld(workspace, off + 2 * hsz)
        var og = _ld(workspace, off + 3 * hsz)
        var go = _ld(grad_hy, i) if has_ghy else Float32(0)
        var goc = _ld(grad_cy, i) if has_gcy else Float32(0)
        var gcx = _tanh(_ld(cy, i))
        var gog = go * gcx
        gcx = go * og * (1 - gcx * gcx) + goc
        var gig = gcx * cg
        var gfg = gcx * _ld(cx, i)
        var gcg = gcx * ig
        gcx = gcx * fg
        gig = gig * (1 - ig) * ig
        gfg = gfg * (1 - fg) * fg
        gcg = gcg * (1 - cg * cg)
        gog = gog * (1 - og) * og
        grad_gates[unsafe_offset=off] = gig.cast[dtype]()
        grad_gates[unsafe_offset=off + hsz] = gfg.cast[dtype]()
        grad_gates[unsafe_offset=off + 2 * hsz] = gcg.cast[dtype]()
        grad_gates[unsafe_offset=off + 3 * hsz] = gog.cast[dtype]()
        grad_cx[unsafe_offset=i] = gcx.cast[dtype]()
        i += stride


@__name(t"gru_cell_fwd_{dtype}")
def _gru_cell_fwd_kernel[
    dtype: DType
](
    input_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    hidden_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    bias1: Pointer[Scalar[dtype], MutAnyOrigin],
    bias2: Pointer[Scalar[dtype], MutAnyOrigin],
    hx: Pointer[Scalar[dtype], MutAnyOrigin],
    hy: Pointer[Scalar[dtype], MutAnyOrigin],
    workspace: Pointer[Scalar[dtype], MutAnyOrigin],
    hsz_arg: Int64,
    total_arg: Int64,
    has_bias_arg: Int64,
):
    var hsz = Int(hsz_arg)
    var total = Int(total_arg)
    var has_bias = has_bias_arg != 0
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var h = i % hsz
        var row = i // hsz
        var off = row * 3 * hsz + h
        var b1r = Float32(0)
        var b1i = Float32(0)
        var b1n = Float32(0)
        var b2r = Float32(0)
        var b2i = Float32(0)
        var b2n = Float32(0)
        if has_bias:
            b1r = _ld(bias1, h)
            b1i = _ld(bias1, h + hsz)
            b1n = _ld(bias1, h + 2 * hsz)
            b2r = _ld(bias2, h)
            b2i = _ld(bias2, h + hsz)
            b2n = _ld(bias2, h + 2 * hsz)
        var hn = _ld(hidden_gates, off + 2 * hsz)
        var rg = _sigmoid(
            _ld(input_gates, off) + _ld(hidden_gates, off) + b1r + b2r
        )
        var ig = _sigmoid(
            _ld(input_gates, off + hsz)
            + _ld(hidden_gates, off + hsz)
            + b1i
            + b2i
        )
        var ng = _tanh(_ld(input_gates, off + 2 * hsz) + b1n + rg * (hn + b2n))
        var hxv = hx[unsafe_offset=i]
        hy[unsafe_offset=i] = (ng + ig * (hxv.cast[F32]() - ng)).cast[dtype]()
        var w = row * 5 * hsz + h
        workspace[unsafe_offset=w] = rg.cast[dtype]()
        workspace[unsafe_offset=w + hsz] = ig.cast[dtype]()
        workspace[unsafe_offset=w + 2 * hsz] = ng.cast[dtype]()
        workspace[unsafe_offset=w + 3 * hsz] = hxv
        workspace[unsafe_offset=w + 4 * hsz] = (hn + b2n).cast[dtype]()
        i += stride


@__name(t"gru_cell_bwd_{dtype}")
def _gru_cell_bwd_kernel[
    dtype: DType
](
    grad_input_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_hidden_gates: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_hy: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_hx: Pointer[Scalar[dtype], MutAnyOrigin],
    workspace: Pointer[Scalar[dtype], MutAnyOrigin],
    hsz_arg: Int64,
    total_arg: Int64,
):
    var hsz = Int(hsz_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var h = i % hsz
        var row = i // hsz
        var w = row * 5 * hsz + h
        var rg = _ld(workspace, w)
        var ig = _ld(workspace, w + hsz)
        var ng = _ld(workspace, w + 2 * hsz)
        var hxv = _ld(workspace, w + 3 * hsz)
        var hn = _ld(workspace, w + 4 * hsz)
        var go = _ld(grad_hy, i)
        var gig = go * (hxv - ng) * (1 - ig) * ig
        var ghx = go * ig
        var gin = go * (1 - ig) * (1 - ng * ng)
        var ghn = gin * rg
        var grg = gin * hn * (1 - rg) * rg
        var off = row * 3 * hsz + h
        grad_input_gates[unsafe_offset=off] = grg.cast[dtype]()
        grad_input_gates[unsafe_offset=off + hsz] = gig.cast[dtype]()
        grad_input_gates[unsafe_offset=off + 2 * hsz] = gin.cast[dtype]()
        grad_hidden_gates[unsafe_offset=off] = grg.cast[dtype]()
        grad_hidden_gates[unsafe_offset=off + hsz] = gig.cast[dtype]()
        grad_hidden_gates[unsafe_offset=off + 2 * hsz] = ghn.cast[dtype]()
        grad_hx[unsafe_offset=i] = ghx.cast[dtype]()
        i += stride


def enqueue_lstm_cell_fwd[
    dtype: DType
](
    ctx: DeviceContext,
    input_gates: Int,
    hidden_gates: Int,
    bias1: Int,
    bias2: Int,
    cx: Int,
    hy: Int,
    cy: Int,
    workspace: Int,
    hsz: Int,
    total: Int,
    has_bias: Int,
) raises:
    if total <= 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_lstm_cell_fwd_kernel[dtype]](
            ctx,
            _fill_blocks(total),
            1,
            1,
            FILL_THREADS,
            _make_ptr[dtype](input_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](hidden_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](bias1).as_unsafe_any_origin(),
            _make_ptr[dtype](bias2).as_unsafe_any_origin(),
            _make_ptr[dtype](cx).as_unsafe_any_origin(),
            _make_ptr[dtype](hy).as_unsafe_any_origin(),
            _make_ptr[dtype](cy).as_unsafe_any_origin(),
            _make_ptr[dtype](workspace).as_unsafe_any_origin(),
            Int64(hsz),
            Int64(total),
            Int64(has_bias),
        )


def enqueue_lstm_cell_bwd[
    dtype: DType
](
    ctx: DeviceContext,
    workspace: Int,
    grad_gates: Int,
    cx: Int,
    cy: Int,
    grad_hy: Int,
    grad_cy: Int,
    grad_cx: Int,
    hsz: Int,
    total: Int,
    has_grad_hy: Int,
    has_grad_cy: Int,
) raises:
    if total <= 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_lstm_cell_bwd_kernel[dtype]](
            ctx,
            _fill_blocks(total),
            1,
            1,
            FILL_THREADS,
            _make_ptr[dtype](workspace).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](cx).as_unsafe_any_origin(),
            _make_ptr[dtype](cy).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_hy).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_cy).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_cx).as_unsafe_any_origin(),
            Int64(hsz),
            Int64(total),
            Int64(has_grad_hy),
            Int64(has_grad_cy),
        )


def enqueue_gru_cell_fwd[
    dtype: DType
](
    ctx: DeviceContext,
    input_gates: Int,
    hidden_gates: Int,
    bias1: Int,
    bias2: Int,
    hx: Int,
    hy: Int,
    workspace: Int,
    hsz: Int,
    total: Int,
    has_bias: Int,
) raises:
    if total <= 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_gru_cell_fwd_kernel[dtype]](
            ctx,
            _fill_blocks(total),
            1,
            1,
            FILL_THREADS,
            _make_ptr[dtype](input_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](hidden_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](bias1).as_unsafe_any_origin(),
            _make_ptr[dtype](bias2).as_unsafe_any_origin(),
            _make_ptr[dtype](hx).as_unsafe_any_origin(),
            _make_ptr[dtype](hy).as_unsafe_any_origin(),
            _make_ptr[dtype](workspace).as_unsafe_any_origin(),
            Int64(hsz),
            Int64(total),
            Int64(has_bias),
        )


def enqueue_gru_cell_bwd[
    dtype: DType
](
    ctx: DeviceContext,
    grad_input_gates: Int,
    grad_hidden_gates: Int,
    grad_hy: Int,
    grad_hx: Int,
    workspace: Int,
    hsz: Int,
    total: Int,
) raises:
    if total <= 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_gru_cell_bwd_kernel[dtype]](
            ctx,
            _fill_blocks(total),
            1,
            1,
            FILL_THREADS,
            _make_ptr[dtype](grad_input_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_hidden_gates).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_hy).as_unsafe_any_origin(),
            _make_ptr[dtype](grad_hx).as_unsafe_any_origin(),
            _make_ptr[dtype](workspace).as_unsafe_any_origin(),
            Int64(hsz),
            Int64(total),
        )
