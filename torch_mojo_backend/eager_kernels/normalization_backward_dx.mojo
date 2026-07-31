"""Fable-owned LayerNorm backward grad-input candidate API.

Pure-Mojo H100 kernels for the f32 LayerNorm dx with runtime shapes.

For each row:
    xhat = (input - mean[row]) * rstd[row]
    q    = grad_output * (weight if has_weight else 1)
    dx   = rstd[row]/cols * (cols*q - sum(q) - xhat*sum(q*xhat))

The op needs two row reductions before any element can be written, so the
fast path reads each row from global memory exactly once and parks q (and
xhat while it fits) in per-warp shared-memory scratch between the reduction
and the store pass.  Registers were the previous home for that state, but the
footprint capped occupancy at 37.5% and the kernel is latency-bound on its
global loads; shared staging keeps the kernel at 48-54 registers with a
fixed-latency store pass.  The fast path assigns one warp per row: each lane
owns up to `chunks` 16-byte vector columns, accumulates both sums locally,
resolves them with two warp shuffles, and replays dx from shared.  `chunks`
is a comptime parameter instantiated for a few broad column regimes
(cols <= 128/256/384/512/768/1024) so small rows do not pay the shared/
register footprint of the largest regime; every instantiation still serves an
open range of runtime shapes.  The narrow regimes (chunks <= 4) also mark the
grad/input reads as streaming (evict-first): measured on H100, giving L2 to
dx writeback wins while the concurrent-row working set is small and reverses
for wider rows.

The fast path requires cols % 4 == 0 and 16-byte-aligned dx/grad/input (plus
weight when used); with cols a multiple of 4, every row start inherits the
base alignment.  Anything else -- odd columns, four-byte storage offsets, or
cols > 1024 -- takes the generic block-per-row kernel: scalar coalesced loads,
two `block.sum` reductions, and a second pass that re-reads the row (L1/L2
resident) to write dx.  Both kernels handle arbitrary positive rows/cols and
grid-stride over rows.
"""

from std.gpu import (
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import CacheOperation
from std.gpu.memory import load as global_load
from std.gpu.primitives import block, warp
from std.math import ceildiv
from std.memory import AddressSpace, stack_allocation
from std.sys.info import has_accelerator, has_apple_gpu_accelerator

from op_utils import _enqueue_cached

# One warp per row.  Warps per block is sized off WARP_SIZE so the largest
# staged instantiation (c6, q + xhat) stays within a portable per-workgroup
# shared-memory budget: 2 * warps * WARP_SIZE * 16 B * 6 chunks = 48 KiB with
# eight 32-wide warps, and the same 48 KiB with four 64-wide warps (eight
# would need 96 KiB).  Block size stays 256 threads either way.
comptime _WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4
comptime _GEN_BLOCK = 256
comptime _VEC = 4
comptime _MAX_GRID = 65535


@always_inline
def _dx_warp_rows[
    chunks: Int
](
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK

    # Per-warp staging for q (and xh when the budget allows) between the
    # reduction and the store pass: keeping them in registers costs enough
    # occupancy to starve the latency-bound loads, and re-reading grad/weight
    # through L1/L2 puts the store pass back on the global-load scoreboard.
    # Shared memory has fixed low latency and each lane only ever touches its
    # own slots, so no barrier is needed.  Staging xh as well is only worth it
    # while the combined footprint stays small enough not to gate occupancy
    # below the register limit (2 * _WARPS_PER_BLOCK * WARP_SIZE * 16 B per
    # chunk per block, i.e. 48 KiB at c6 for both the 8-warp/32-wide and the
    # 4-warp/64-wide configurations).
    comptime stage_xh = chunks <= 6
    # Streaming (evict-first) reads help while the per-SM working set of
    # concurrent rows is small enough that L2 capacity is better spent
    # buffering dx writebacks; measured on H100, that holds for the small-row
    # regimes and reverses once rows get wide (chunks > 4).  Apple has no
    # cache-policy loads: the STREAMING lowering crashes the Metal compiler.
    comptime stream_loads = chunks <= 4 and not has_apple_gpu_accelerator()
    comptime q_slots = _WARPS_PER_BLOCK * chunks * WARP_SIZE * _VEC
    comptime xh_slots = q_slots if stage_xh else _VEC
    var q_shared = stack_allocation[
        q_slots, DType.float32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var xh_shared = stack_allocation[
        xh_slots,
        DType.float32,
        alignment=16,
        address_space=AddressSpace.SHARED,
    ]()
    var warp_slot = (Int(warp_id()) * chunks * WARP_SIZE + lane) * _VEC
    var inv_cols = 1.0 / Float32(cols)

    while row < rows:
        var m = mean[row]
        var r = rstd[row]
        var scale = r * inv_cols
        var base = row * cols
        var acc_q = SIMD[DType.float32, _VEC](0.0)
        var acc_qx = SIMD[DType.float32, _VEC](0.0)
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var offset = base + c * _VEC
                var q: SIMD[DType.float32, _VEC]
                var x: SIMD[DType.float32, _VEC]
                comptime if stream_loads:
                    q = global_load[
                        width=_VEC,
                        cache_policy=CacheOperation.STREAMING,
                        alignment=16,
                    ](grad_output, offset)
                    x = global_load[
                        width=_VEC,
                        cache_policy=CacheOperation.STREAMING,
                        alignment=16,
                    ](input, offset)
                else:
                    q = grad_output.load[width=_VEC, alignment=16](offset)
                    x = input.load[width=_VEC, alignment=16](offset)
                if has_weight != 0:
                    q *= weight.load[width=_VEC, alignment=16](c * _VEC)
                var xh = (x - m) * r
                q_shared.store[width=_VEC, alignment=16](
                    warp_slot + u * WARP_SIZE * _VEC, q
                )
                comptime if stage_xh:
                    xh_shared.store[width=_VEC, alignment=16](
                        warp_slot + u * WARP_SIZE * _VEC, xh
                    )
                acc_q += q
                acc_qx = q.fma(xh, acc_qx)
        # Row condition is warp-uniform, so every lane reaches the shuffles.
        var kb = warp.sum(acc_q) * scale
        var kc = warp.sum(acc_qx) * scale
        # Store pass: q (and xh when staged) comes from shared; otherwise
        # xhat is recomputed from the L1/L2-resident input row.
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var offset = base + c * _VEC
                var q = q_shared.load[width=_VEC, alignment=16](
                    warp_slot + u * WARP_SIZE * _VEC
                )
                var xh: SIMD[DType.float32, _VEC]
                comptime if stage_xh:
                    xh = xh_shared.load[width=_VEC, alignment=16](
                        warp_slot + u * WARP_SIZE * _VEC
                    )
                else:
                    xh = (input.load[width=_VEC, alignment=16](offset) - m) * r
                var out = q.fma(r, xh.fma(-kc, -kb))
                dx.store[width=_VEC, alignment=16](offset, out)
        row += row_stride


@__name("nanogpt_layer_norm_backward_dx_f32_c1")
def _dx_c1(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[1](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_c2")
def _dx_c2(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[2](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_c3")
def _dx_c3(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[3](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_c4")
def _dx_c4(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[4](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_c6")
def _dx_c6(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[6](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_c8")
def _dx_c8(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    has_weight: Int,
):
    _dx_warp_rows[8](
        dx,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        rows,
        cols,
        vec_cols,
        has_weight,
    )


@__name("nanogpt_layer_norm_backward_dx_f32_generic")
def _dx_generic(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    has_weight: Int,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var m = mean[row]
        var r = rstd[row]
        var base = row * cols
        var acc_q = Float32(0.0)
        var acc_qx = Float32(0.0)
        var col = tid
        while col < cols:
            var q = grad_output[base + col]
            if has_weight != 0:
                q *= weight[col]
            var xh = (input[base + col] - m) * r
            acc_q += q
            acc_qx += q * xh
            col += _GEN_BLOCK
        # Row condition is block-uniform, so every thread reaches both
        # reductions.
        var kb = (
            block.sum[block_size=_GEN_BLOCK, broadcast=True](acc_q)
            * r
            / Float32(cols)
        )
        var kc = (
            block.sum[block_size=_GEN_BLOCK, broadcast=True](acc_qx)
            * r
            / Float32(cols)
        )
        col = tid
        while col < cols:
            var q = grad_output[base + col]
            if has_weight != 0:
                q *= weight[col]
            var xh = (input[base + col] - m) * r
            dx[base + col] = q * r - kb - xh * kc
            col += _GEN_BLOCK
        row += row_stride


def enqueue_layer_norm_backward_dx_f32(
    dx: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    has_weight: Bool,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        if rows <= 0 or cols <= 0:
            return
        var hw = 1 if has_weight else 0
        # 16-byte vector paths need proven alignment: the harness may offset
        # every pointer by one f32, so check the participating bases at
        # runtime.  With cols % 4 == 0, row starts inherit the base alignment.
        var aligned = (Int(dx) | Int(grad_output) | Int(input)) % 16 == 0
        if has_weight:
            aligned = aligned and Int(weight) % 16 == 0
        if aligned and cols % _VEC == 0:
            var vec_cols = cols // _VEC
            var needed = ceildiv(vec_cols, WARP_SIZE)
            if needed <= 8:
                var grid = min(ceildiv(rows, _WARPS_PER_BLOCK), _MAX_GRID)
                var block_dim = _WARPS_PER_BLOCK * WARP_SIZE
                if needed <= 1:
                    _enqueue_cached[_dx_c1](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c1",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                elif needed <= 2:
                    _enqueue_cached[_dx_c2](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c2",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                elif needed <= 3:
                    _enqueue_cached[_dx_c3](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c3",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                elif needed <= 4:
                    _enqueue_cached[_dx_c4](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c4",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                elif needed <= 6:
                    _enqueue_cached[_dx_c6](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c6",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                else:
                    _enqueue_cached[_dx_c8](
                        ctx,
                        "nanogpt_layer_norm_backward_dx_f32_c8",
                        grid,
                        1,
                        1,
                        block_dim,
                        dx,
                        grad_output,
                        input,
                        mean,
                        rstd,
                        weight,
                        rows,
                        cols,
                        vec_cols,
                        hw,
                    )
                return
        var grid = min(rows, _MAX_GRID)
        _enqueue_cached[_dx_generic](
            ctx,
            "nanogpt_layer_norm_backward_dx_f32_generic",
            grid,
            1,
            1,
            _GEN_BLOCK,
            dx,
            grad_output,
            input,
            mean,
            rstd,
            weight,
            rows,
            cols,
            hw,
        )
    else:
        raise Error("no GPU accelerator available at compile time")
