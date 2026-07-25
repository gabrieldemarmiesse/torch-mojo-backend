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

from std.gpu import WARP_SIZE, block_idx, grid_dim, lane_id, warp_id
from std.gpu.host import DeviceContext
from std.gpu.primitives import warp
from std.math import ceildiv
from std.sys.info import has_accelerator, size_of

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


@__name(
    t"nanogpt_sdpa_dropout_softmax_bwd_{dtype}_m{has_mask}_c{causal}_v{VEC}"
)
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
            t"nanogpt_sdpa_dropout_softmax_bwd_{dtype}_m{has_mask}_c{causal}_v{VEC}"
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
