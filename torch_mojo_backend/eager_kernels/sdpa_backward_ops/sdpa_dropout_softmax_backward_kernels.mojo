"""Pure-Mojo fused dropout and softmax backward kernels for eager SDPA.

For every contiguous row, these kernels compute::

    masked_dP = dP_drop * Float32(mask) * Float32(dropout_scale)
    row_sum = sum(P * masked_dP)
    dScores = P * (masked_dP - row_sum) * Float32(score_scale)

The no-mask path omits the mask and dropout-scale operations completely.  All
arithmetic is performed in float32 whatever the operand dtype, because a row
reduction over `cols` products cannot be carried in bfloat16 or float16 without
losing the small terms; only the loads, the stores and the row extent depend on
the dtype.

With ``causal != 0`` the row is treated as query index ``row % q_len`` of a
top-left-aligned causal mask, so only columns ``< min(cols, row % q_len + 1)``
carry a nonzero probability.  Softmax makes ``P`` exactly zero beyond that, so
``dScores`` there is exactly zero: the kernel skips reading those columns and
writes the zeros directly.  That is an exact restatement of the same arithmetic,
not an approximation, and it roughly halves the traffic at attention shapes.

One warp owns a row; four (64-wide) or eight (32-wide) warps share a block, and
blocks grid-stride over rows.  Two regimes are selected from runtime metadata
only: 16-byte vector accesses when the row length and every operand address
admit them, and scalar accesses otherwise.  All extents and scalars stay
runtime-dynamic.  The implementation uses no scratch allocation, host access,
vendor library, or synchronization beyond warp shuffles; launches are enqueued
asynchronously on the caller's ``DeviceContext``.
"""

from std.gpu import WARP_SIZE, block_idx, grid_dim, lane_id, thread_idx, warp_id
from std.gpu.host import DeviceContext
from std.gpu.primitives import block, warp
from std.math import ceildiv
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.static_tuple import StaticTuple

from op_utils import _enqueue_cached


comptime _WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4
comptime _BLOCK = _WARPS_PER_BLOCK * WARP_SIZE
# gridDim.x is not the 65535-limited axis.  One block per group of rows (rather
# than a small grid that strides many rows) lets the hardware balance the very
# uneven per-row work a causal mask creates.
comptime _MAX_GRID = 1 << 20
comptime _VECTOR_BYTES = 16


@always_inline
def _fused_rows[
    dtype: DType, has_mask: Bool, causal: Bool, VEC: Int
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    comptime F32 = DType.float32
    # The wide regime is only selected when every row base and every operand
    # address is a multiple of VEC elements, so these accesses are aligned.
    comptime ALIGN = VEC * size_of[dtype]()
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK

    @always_inline
    @parameter
    def _grad_vec(index: Int) -> SIMD[F32, VEC]:
        var d_p = grad_after_dropout.load[width=VEC, alignment=ALIGN](
            index
        ).cast[F32]()
        comptime if has_mask:
            # Preserve the pinned IEEE operation order.  A false mask is
            # multiplied rather than used as a selection, so non-finite
            # gradients still propagate as specified.
            d_p = d_p * mask.load[width=VEC](index).cast[F32]() * dropout_scale
        return d_p

    @always_inline
    @parameter
    def _grad_one(index: Int) -> Float32:
        var d_p = grad_after_dropout[index].cast[F32]()
        comptime if has_mask:
            d_p = d_p * mask[index].cast[F32]() * dropout_scale
        return d_p

    while row < rows:
        var limit = cols
        comptime if causal:
            limit = min(cols, row % q_len + 1)
        var base = row * cols
        var vec_limit = (limit // VEC) * VEC

        # Pass 1: the row reduction, in float32.
        var acc = SIMD[F32, VEC](0.0)
        var col = lane * VEC
        while col < vec_limit:
            var index = base + col
            acc = (
                probabilities.load[width=VEC, alignment=ALIGN](index)
                .cast[F32]()
                .fma(_grad_vec(index), acc)
            )
            col += WARP_SIZE * VEC
        # Ragged remainder: fewer than VEC columns, so at most one per lane.
        var tail = vec_limit + lane
        var tail_sum = Float32(0.0)
        if tail < limit:
            tail_sum = probabilities[base + tail].cast[F32]() * _grad_one(
                base + tail
            )

        # The row bound is warp-uniform, so every lane reaches the shuffles.
        var row_sum = warp.sum(acc.reduce_add() + tail_sum)

        # Pass 2: the outputs.  The row is L1/L2-resident from pass 1.
        col = lane * VEC
        while col < vec_limit:
            var index = base + col
            var value = (
                probabilities.load[width=VEC, alignment=ALIGN](index).cast[
                    F32
                ]()
                * (_grad_vec(index) - row_sum)
                * score_scale
            )
            output.store[width=VEC, alignment=ALIGN](index, value.cast[dtype]())
            col += WARP_SIZE * VEC
        if tail < limit:
            output[base + tail] = (
                probabilities[base + tail].cast[F32]()
                * (_grad_one(base + tail) - row_sum)
                * score_scale
            ).cast[dtype]()

        # Fully masked columns: P is exactly zero there, so dScores is exactly
        # zero.  Write it rather than leave the fresh allocation undefined.
        comptime if causal:
            var zero_head = min(cols, ceildiv(limit, VEC) * VEC)
            if limit + lane < zero_head:
                output[base + limit + lane] = Scalar[dtype](0)
            col = zero_head + lane * VEC
            while col + VEC <= cols:
                output.store[width=VEC, alignment=ALIGN](
                    base + col, SIMD[dtype, VEC](Scalar[dtype](0))
                )
                col += WARP_SIZE * VEC
            # `cols` and `zero_head` are both multiples of VEC in the wide
            # regime, so the loop above leaves nothing behind; VEC == 1 makes
            # `zero_head == limit` and the loop covers everything.
        row += row_stride


@__name(t"sdpa_dropout_softmax_bwd_{dtype}_m{has_mask}_c{causal}_v{VEC}")
def _fused_kernel[
    dtype: DType, has_mask: Bool, causal: Bool, VEC: Int
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    _fused_rows[dtype, has_mask, causal, VEC](
        output,
        probabilities,
        grad_after_dropout,
        mask,
        rows,
        cols,
        q_len,
        dropout_scale,
        score_scale,
    )


@always_inline
def _enqueue_one[
    dtype: DType, has_mask: Bool, causal: Bool, VEC: Int
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    dropout_scale: Float32,
    score_scale: Float32,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_fused_kernel[dtype, has_mask, causal, VEC]](
        ctx,
        String(
            t"sdpa_dropout_softmax_bwd_{dtype}_m{has_mask}_c{causal}_v{VEC}"
        ),
        min(ceildiv(rows, _WARPS_PER_BLOCK), _MAX_GRID),
        1,
        1,
        _BLOCK,
        output,
        probabilities,
        grad_after_dropout,
        mask,
        rows,
        cols,
        q_len,
        dropout_scale,
        score_scale,
    )


@always_inline
def _enqueue_regime[
    dtype: DType, causal: Bool
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    has_mask: Bool,
    dropout_scale: Float32,
    score_scale: Float32,
    ctx: DeviceContext,
) raises:
    comptime WIDE = _VECTOR_BYTES // size_of[dtype]()
    # 16-byte accesses need the row stride and every operand address to be a
    # whole number of vectors; both are runtime facts about the operands, not
    # about the model.
    var wide_ok = (
        cols % WIDE == 0
        and Int(output) % _VECTOR_BYTES == 0
        and Int(probabilities) % _VECTOR_BYTES == 0
        and Int(grad_after_dropout) % _VECTOR_BYTES == 0
        and (not has_mask or Int(mask) % WIDE == 0)
    )
    if has_mask:
        if wide_ok:
            _enqueue_one[dtype, True, causal, WIDE](
                output,
                probabilities,
                grad_after_dropout,
                mask,
                rows,
                cols,
                q_len,
                dropout_scale,
                score_scale,
                ctx,
            )
        else:
            _enqueue_one[dtype, True, causal, 1](
                output,
                probabilities,
                grad_after_dropout,
                mask,
                rows,
                cols,
                q_len,
                dropout_scale,
                score_scale,
                ctx,
            )
    elif wide_ok:
        _enqueue_one[dtype, False, causal, WIDE](
            output,
            probabilities,
            grad_after_dropout,
            mask,
            rows,
            cols,
            q_len,
            dropout_scale,
            score_scale,
            ctx,
        )
    else:
        _enqueue_one[dtype, False, causal, 1](
            output,
            probabilities,
            grad_after_dropout,
            mask,
            rows,
            cols,
            q_len,
            dropout_scale,
            score_scale,
            ctx,
        )


def enqueue_sdpa_dropout_softmax_backward[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: Optional[UnsafePointer[Scalar[DType.bool], MutAnyOrigin]],
    rows: Int,
    cols: Int,
    q_len: Int,
    has_mask: Bool,
    causal: Bool,
    dropout_scale: Float64,
    score_scale: Float64,
    ctx: DeviceContext,
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
            "sdpa_dropout_softmax_backward_kernels: causal requires q_len > 0"
        )

    comptime if has_accelerator():
        if rows <= 0 or cols <= 0:
            return
        # `UnsafePointer` is non-nullable, and the kernel's `has_mask` is a
        # compile-time parameter, so the no-mask instantiation never emits a
        # single mask access.  Pass an aliased operand pointer as the unused
        # argument rather than model nullability all the way into the kernel.
        var mask_ptr = mask.value() if has_mask else probabilities.bitcast[
            Scalar[DType.bool]
        ]()
        if causal:
            _enqueue_regime[dtype, True](
                output,
                probabilities,
                grad_after_dropout,
                mask_ptr,
                rows,
                cols,
                q_len,
                has_mask,
                Float32(dropout_scale),
                Float32(score_scale),
                ctx,
            )
        else:
            _enqueue_regime[dtype, False](
                output,
                probabilities,
                grad_after_dropout,
                mask_ptr,
                rows,
                cols,
                q_len,
                has_mask,
                Float32(dropout_scale),
                Float32(score_scale),
                ctx,
            )
    else:
        raise Error("no GPU accelerator available at compile time")


# ===------------------------------------------------------------------=== #
# Apple-tuned FP32 variants (enqueue_sdpa_dropout_softmax_backward_f32).
# Kept alongside the generic dtype-parametric implementation above: these
# were measured on M-series hardware (warp-per-row register-resident
# causal kernels, VPT-dispatched) and back the Apple route in
# aten_fast.fast_sdpa_backward via SDPADropoutSoftmaxBackwardF32.
# ===------------------------------------------------------------------=== #

comptime _APPLE_BLOCK = 256
comptime _APPLE_MAX_GRID = 65535
# Causal outputs are zero-filled from the boundary up to the next multiple
# of this, so causal consumers cutting reductions at 32-row (or coarser,
# up to 64) tile granularity never read uninitialized memory.
comptime _ZPAD = 64
# Warp-per-row causal kernel geometry (mirrors the Apple forward softmax
# warp kernel: whole allowed prefix in registers, one read, warp shuffles).
comptime _APPLE_WARPS_PER_BLOCK = 8
# Metal demotes the per-lane staging tuples to threadgroup memory rather than
# keeping them in registers: at 8 warps per block each VPT slot costs 6 KiB of
# threadgroup memory (49152 B observed at VPT=8 on macOS 26.3), over Apple's
# 32 KiB cap, so pipeline creation fails outright.  VPT stops at 4 there
# (24 KiB); longer rows take the block-per-row kernels below.  Other targets
# keep true register staging and the measured VPT=8 ladder.
comptime _MAX_VPT = 4 if has_apple_gpu_accelerator() else 8  # per-lane slots
comptime _CVEC = 4  # float4 loads; requires 16B-aligned rows


@__name("sdpa_dropout_softmax_backward_masked_f32")
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
            col += _APPLE_BLOCK

        # Every thread in the block reaches this row reduction. Broadcasting
        # keeps the result available for the independent output columns.
        var row_sum = block.sum[block_size=_APPLE_BLOCK, broadcast=True](
            partial_sum
        )
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
            col += _APPLE_BLOCK
        row += row_stride


@__name("sdpa_dropout_softmax_backward_unmasked_f32")
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
            col += _APPLE_BLOCK

        var row_sum = block.sum[block_size=_APPLE_BLOCK, broadcast=True](
            partial_sum
        )
        col = tid
        while col < cols:
            var index = base + col
            output[index] = (
                probabilities[index]
                * (grad_after_dropout[index] - row_sum)
                * score_scale
            )
            col += _APPLE_BLOCK
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
    var row = Int(block_idx.x) * _APPLE_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_WARPS_PER_BLOCK
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
    var row = Int(block_idx.x) * _APPLE_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_WARPS_PER_BLOCK
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


@__name("sdpa_dropout_softmax_backward_masked_causal_f32")
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
            col += _APPLE_BLOCK

        var row_sum = block.sum[block_size=_APPLE_BLOCK, broadcast=True](
            partial_sum
        )
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
            col += _APPLE_BLOCK
        row += row_stride


@__name("sdpa_dropout_softmax_backward_unmasked_causal_f32")
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
            col += _APPLE_BLOCK

        var row_sum = block.sum[block_size=_APPLE_BLOCK, broadcast=True](
            partial_sum
        )
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
            col += _APPLE_BLOCK
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

        var grid = min(rows, _APPLE_MAX_GRID)
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
        var warp_grid = min(ceildiv(rows, _APPLE_WARPS_PER_BLOCK), 32768)

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
                    _APPLE_WARPS_PER_BLOCK * WARP_SIZE,
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
                    _APPLE_WARPS_PER_BLOCK * WARP_SIZE,
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
                "sdpa_dropout_softmax_backward_masked_causal_f32",
                grid,
                1,
                1,
                _APPLE_BLOCK,
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
                "sdpa_dropout_softmax_backward_unmasked_causal_f32",
                grid,
                1,
                1,
                _APPLE_BLOCK,
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
                "sdpa_dropout_softmax_backward_masked_f32",
                grid,
                1,
                1,
                _APPLE_BLOCK,
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
                "sdpa_dropout_softmax_backward_unmasked_f32",
                grid,
                1,
                1,
                _APPLE_BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                rows,
                cols,
                score_scale_f32,
            )
    else:
        raise Error("no GPU accelerator available at compile time")
