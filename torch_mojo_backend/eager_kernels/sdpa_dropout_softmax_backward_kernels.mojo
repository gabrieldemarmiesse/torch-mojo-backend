"""Pure-Mojo fused dropout and softmax backward kernels for eager SDPA.

For every contiguous FP32 row, these kernels compute::

    masked_dP = dP_drop * Float32(mask) * Float32(dropout_scale)
    row_sum = sum(P * masked_dP)
    dScores = P * (masked_dP - row_sum) * Float32(score_scale)

The no-mask path omits the mask and dropout-scale operations completely.  A
block owns one row at a time and grid-strides over rows, so all shapes and
scalar values remain runtime-dynamic.  The implementation uses no scratch
allocation, host access, vendor library, or synchronization; launches are
enqueued asynchronously on the caller's ``DeviceContext``.

The causal variants exploit SDPA's causal structure: row ``r`` (query index
``i = r % q_len``) reads and reduces only columns ``j <= i`` — halving the
kernel's traffic at ``q_len == kv_len`` and never touching ``dP_drop`` above
the boundary, which the Apple causal score GEMM leaves unwritten.  Outputs
above the boundary are zero-filled up to the next multiple of ``_ZPAD`` (64)
so tile-granular causal GEMM consumers (which read at most to the next
multiple of their 32-row subtile) always see exact zeros there; columns past
that band are left unwritten and must not be read.
"""

from std.gpu import WARP_SIZE, block_idx, grid_dim, thread_idx, warp_id
from std.gpu.host import DeviceContext
from std.gpu.primitives import block, warp
from std.gpu import lane_id
from std.math import ceildiv
from std.sys.info import has_accelerator
from std.utils.static_tuple import StaticTuple

from op_utils import _enqueue_cached


comptime _BLOCK = 256
comptime _MAX_GRID = 65535
# Causal outputs are zero-filled from the boundary up to the next multiple
# of this, so causal consumers cutting reductions at 32-row (or coarser,
# up to 64) tile granularity never read uninitialized memory.
comptime _ZPAD = 64
# Warp-per-row causal kernel geometry (mirrors the Apple forward softmax
# warp kernel: whole allowed prefix in registers, one read, warp shuffles).
comptime _WARPS_PER_BLOCK = 8
comptime _MAX_VPT = 8  # per-lane register slots
comptime _CVEC = 4  # float4 loads; requires 16B-aligned rows


@__name("nanogpt_sdpa_dropout_softmax_backward_masked_f32")
def _masked_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < cols:
            var index = base + col
            # Preserve the pinned IEEE operation order. In particular, a
            # false mask is multiplied rather than used as a selection, so
            # non-finite gradients still propagate as specified.
            var masked_d_p = (
                grad_after_dropout[index]
                * mask[index].cast[DType.float32]()
                * dropout_scale
            )
            partial_sum += probabilities[index] * masked_d_p
            col += _BLOCK

        # Every thread in the block reaches this row reduction. Broadcasting
        # keeps the result available for the independent output columns.
        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < cols:
            var index = base + col
            var masked_d_p = (
                grad_after_dropout[index]
                * mask[index].cast[DType.float32]()
                * dropout_scale
            )
            output[index] = (
                probabilities[index] * (masked_d_p - row_sum) * score_scale
            )
            col += _BLOCK
        row += row_stride


@__name("nanogpt_sdpa_dropout_softmax_backward_unmasked_f32")
def _unmasked_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < cols:
            var index = base + col
            partial_sum += probabilities[index] * grad_after_dropout[index]
            col += _BLOCK

        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < cols:
            var index = base + col
            output[index] = (
                probabilities[index]
                * (grad_after_dropout[index] - row_sum)
                * score_scale
            )
            col += _BLOCK
        row += row_stride


# Causal variants: the row reduction and the dScores writes cover only the
# causal-visible prefix [0, i + 1); the band [i + 1, min(cols, next multiple
# of _ZPAD)) is zero-filled and the rest of the row left unwritten.  dP_drop
# is never read at or past the boundary (the causal score-gradient GEMM
# leaves those positions unwritten).
#
# The warp kernels below are the fast path (one simdgroup per row, the
# allowed prefix held in registers so each operand is read exactly once,
# float4 loads, shuffle reduction — the same structure as the Apple forward
# softmax warp kernel).  The block kernels that follow are the generic
# fallback for longer or unaligned rows.


@__name(t"sdpa_dsb_masked_causal_warp_f32_{V}_{VPT}")
def _masked_causal_warp_f32[
    V: Int, VPT: Int
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    # One warp per row; requires cols % V == 0 and
    # cols <= WARP_SIZE * VPT * V (host-checked).
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK
    while row < rows:
        var base = row * cols
        var allowed = min(cols, row % q_len + 1)
        var padded = min(cols, ceildiv(allowed, _ZPAD) * _ZPAD)

        # Read pass: the allowed prefix into registers, masked-dP applied,
        # per-lane partial of sum(P * masked_dP).
        var p_vals = StaticTuple[SIMD[DType.float32, V], VPT]()
        var d_vals = StaticTuple[SIMD[DType.float32, V], VPT]()
        var partial = SIMD[DType.float32, V](0)
        comptime for slot in range(VPT):
            var j0 = (lane + slot * WARP_SIZE) * V
            var p = SIMD[DType.float32, V](0)
            var masked_d_p = SIMD[DType.float32, V](0)
            if j0 < allowed:
                p = probabilities.load[width=V, alignment=V * 4](base + j0)
                # Preserve the pinned IEEE operation order (`_masked_f32`).
                masked_d_p = (
                    grad_after_dropout.load[width=V, alignment=V * 4](base + j0)
                    * mask.load[width=V](base + j0).cast[DType.float32]()
                    * dropout_scale
                )
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            p[li] = 0
                            masked_d_p[li] = 0
                partial += p * masked_d_p
            p_vals[slot] = p
            d_vals[slot] = masked_d_p
        var row_sum = warp.sum(partial.reduce_add())

        # Write pass from registers: dScores on the allowed prefix, zeros on
        # the band up to the _ZPAD boundary (registers there are zero).
        comptime for slot in range(VPT):
            var j0 = (lane + slot * WARP_SIZE) * V
            if j0 < padded:
                var value = (
                    p_vals[slot] * (d_vals[slot] - row_sum) * score_scale
                )
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            value[li] = 0
                output.store[width=V, alignment=V * 4](base + j0, value)
        row += row_stride


@__name(t"sdpa_dsb_unmasked_causal_warp_f32_{V}_{VPT}")
def _unmasked_causal_warp_f32[
    V: Int, VPT: Int
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    score_scale: Float32,
):
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK
    while row < rows:
        var base = row * cols
        var allowed = min(cols, row % q_len + 1)
        var padded = min(cols, ceildiv(allowed, _ZPAD) * _ZPAD)

        var p_vals = StaticTuple[SIMD[DType.float32, V], VPT]()
        var d_vals = StaticTuple[SIMD[DType.float32, V], VPT]()
        var partial = SIMD[DType.float32, V](0)
        comptime for slot in range(VPT):
            var j0 = (lane + slot * WARP_SIZE) * V
            var p = SIMD[DType.float32, V](0)
            var d = SIMD[DType.float32, V](0)
            if j0 < allowed:
                p = probabilities.load[width=V, alignment=V * 4](base + j0)
                d = grad_after_dropout.load[width=V, alignment=V * 4](base + j0)
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            p[li] = 0
                            d[li] = 0
                partial += p * d
            p_vals[slot] = p
            d_vals[slot] = d
        var row_sum = warp.sum(partial.reduce_add())

        comptime for slot in range(VPT):
            var j0 = (lane + slot * WARP_SIZE) * V
            if j0 < padded:
                var value = (
                    p_vals[slot] * (d_vals[slot] - row_sum) * score_scale
                )
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            value[li] = 0
                output.store[width=V, alignment=V * 4](base + j0, value)
        row += row_stride


@__name("nanogpt_sdpa_dropout_softmax_backward_masked_causal_f32")
def _masked_causal_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var boundary = min(cols, row % q_len + 1)
        var padded = min(cols, ceildiv(boundary, _ZPAD) * _ZPAD)
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < boundary:
            var index = base + col
            # Preserve the pinned IEEE operation order (see `_masked_f32`).
            var masked_d_p = (
                grad_after_dropout[index]
                * mask[index].cast[DType.float32]()
                * dropout_scale
            )
            partial_sum += probabilities[index] * masked_d_p
            col += _BLOCK

        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < padded:
            var index = base + col
            var value = Float32(0.0)
            if col < boundary:
                var masked_d_p = (
                    grad_after_dropout[index]
                    * mask[index].cast[DType.float32]()
                    * dropout_scale
                )
                value = (
                    probabilities[index] * (masked_d_p - row_sum) * score_scale
                )
            output[index] = value
            col += _BLOCK
        row += row_stride


@__name("nanogpt_sdpa_dropout_softmax_backward_unmasked_causal_f32")
def _unmasked_causal_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var boundary = min(cols, row % q_len + 1)
        var padded = min(cols, ceildiv(boundary, _ZPAD) * _ZPAD)
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < boundary:
            var index = base + col
            partial_sum += probabilities[index] * grad_after_dropout[index]
            col += _BLOCK

        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < padded:
            var index = base + col
            var value = Float32(0.0)
            if col < boundary:
                value = (
                    probabilities[index]
                    * (grad_after_dropout[index] - row_sum)
                    * score_scale
                )
            output[index] = value
            col += _BLOCK
        row += row_stride


def enqueue_sdpa_dropout_softmax_backward_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: Optional[UnsafePointer[Scalar[DType.bool], MutAnyOrigin]],
    rows: Int,
    cols: Int,
    has_mask: Bool,
    dropout_scale: Float64,
    score_scale: Float64,
    ctx: DeviceContext,
    causal: Bool = False,
    q_len: Int = 0,
) raises:
    # Check the nullable-mask ABI even for empty shapes, before any possible
    # launch. This catches contradictory host metadata deterministically.
    if has_mask:
        if not mask:
            raise Error(
                "sdpa_dropout_softmax_backward_kernels: has_mask requires"
                " a non-null mask"
            )
    elif mask:
        raise Error(
            "sdpa_dropout_softmax_backward_kernels: a mask requires"
            " has_mask=true"
        )
    if causal and q_len <= 0:
        raise Error(
            "sdpa_dropout_softmax_backward_kernels: causal requires a"
            " positive q_len"
        )

    comptime if has_accelerator():
        if rows <= 0 or cols <= 0:
            return

        var grid = min(rows, _MAX_GRID)
        var score_scale_f32 = Float32(score_scale)
        # Warp-per-row causal fast path: every operand row 16B-aligned
        # (float4 loads) and short enough to live in registers. VPT (the
        # per-lane register slot count) is the smallest power of two that
        # covers the row, so short rows keep registers — and occupancy —
        # for latency hiding.
        var warp_ok = (
            causal
            and cols % _CVEC == 0
            and cols <= WARP_SIZE * _MAX_VPT * _CVEC
            and Int(output) % 16 == 0
            and Int(probabilities) % 16 == 0
            and Int(grad_after_dropout) % 16 == 0
        )
        var warp_grid = min(ceildiv(rows, _WARPS_PER_BLOCK), 32768)

        @always_inline
        @parameter
        def _launch_warp[VPT: Int, HAS_MASK: Bool]() raises:
            comptime if HAS_MASK:
                _enqueue_cached[_masked_causal_warp_f32[_CVEC, VPT]](
                    ctx,
                    String(t"sdpa_dsb_masked_causal_warp_f32_{_CVEC}_{VPT}"),
                    warp_grid,
                    1,
                    1,
                    _WARPS_PER_BLOCK * WARP_SIZE,
                    output,
                    probabilities,
                    grad_after_dropout,
                    mask.value(),
                    rows,
                    cols,
                    q_len,
                    Float32(dropout_scale),
                    score_scale_f32,
                )
            else:
                _enqueue_cached[_unmasked_causal_warp_f32[_CVEC, VPT]](
                    ctx,
                    String(t"sdpa_dsb_unmasked_causal_warp_f32_{_CVEC}_{VPT}"),
                    warp_grid,
                    1,
                    1,
                    _WARPS_PER_BLOCK * WARP_SIZE,
                    output,
                    probabilities,
                    grad_after_dropout,
                    rows,
                    cols,
                    q_len,
                    score_scale_f32,
                )

        @always_inline
        @parameter
        def _launch_warp_vpt[HAS_MASK: Bool]() raises:
            if cols <= WARP_SIZE * _CVEC:
                _launch_warp[1, HAS_MASK]()
            elif cols <= WARP_SIZE * 2 * _CVEC:
                _launch_warp[2, HAS_MASK]()
            elif cols <= WARP_SIZE * 4 * _CVEC:
                _launch_warp[4, HAS_MASK]()
            else:
                _launch_warp[_MAX_VPT, HAS_MASK]()

        if warp_ok and has_mask and Int(mask.value()) % _CVEC == 0:
            _launch_warp_vpt[True]()
        elif warp_ok and not has_mask:
            _launch_warp_vpt[False]()
        elif causal and has_mask:
            _enqueue_cached[_masked_causal_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_masked_causal_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                mask.value(),
                rows,
                cols,
                q_len,
                Float32(dropout_scale),
                score_scale_f32,
            )
        elif causal:
            _enqueue_cached[_unmasked_causal_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_unmasked_causal_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                rows,
                cols,
                q_len,
                score_scale_f32,
            )
        elif has_mask:
            var mask_ptr = mask.value()
            _enqueue_cached[_masked_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_masked_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                mask_ptr,
                rows,
                cols,
                Float32(dropout_scale),
                score_scale_f32,
            )
        else:
            _enqueue_cached[_unmasked_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_unmasked_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                rows,
                cols,
                score_scale_f32,
            )
    else:
        raise Error("no GPU accelerator available at compile time")
