"""Correct baseline for the frozen native LayerNorm-forward harness.

This deliberately mirrors the production kernel's block-per-row structure so
the optimization agent starts from a working, runtime-dynamic implementation.
The frozen contract and harness, not this editable file, define acceptance.
"""

from std.ffi import _get_global_or_null, external_call
from std.gpu import (
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.primitives import block, warp
from std.math import ceildiv, min, sqrt
from std.memory import alloc
from std.sys.info import has_apple_gpu_accelerator

from op_utils import _enqueue_cached


comptime _BLOCK = 256
comptime _VEC = 4
comptime _VEC_BLOCK = 128
comptime _MAX_GRID = 65535
# Warp-per-row fast path (Apple): rows are short enough that a whole block
# per row leaves most lanes idle and pays two block-wide barriers; one warp
# per row with `chunks` vectors per lane keeps the row in registers and
# reduces with two warp shuffles.
comptime _WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4


@always_inline
def _ln_fwd_warp_rows[
    chunks: Int
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK
    var inv_cols = 1.0 / Float32(cols)
    while row < rows:
        var base = row * cols
        # The whole row stays in registers between the two reductions and
        # the store pass; `chunks` is comptime so the array indexes are
        # static.
        var x = InlineArray[SIMD[DType.float32, _VEC], chunks](
            fill=SIMD[DType.float32, _VEC](0.0)
        )
        var acc = SIMD[DType.float32, _VEC](0.0)
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                x[u] = input.load[width=_VEC, alignment=16](base + c * _VEC)
                acc += x[u]
        # Row condition is warp-uniform: every lane reaches the shuffles.
        var row_mean = warp.sum(acc.reduce_add()) * inv_cols
        var vacc = SIMD[DType.float32, _VEC](0.0)
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var centered = x[u] - row_mean
                vacc = centered.fma(centered, vacc)
        var variance = warp.sum(vacc.reduce_add()) * inv_cols
        var row_rstd = 1.0 / sqrt(variance + epsilon)
        # Match ATen: NaN/inf rows report NaN for both statistics (the
        # centered-square reduction is nonnegative for finite rows).
        if variance != variance:
            row_mean = variance
        if lane == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var result = (x[u] - row_mean) * row_rstd
                if has_weight != 0:
                    result *= weight.load[width=_VEC, alignment=16](c * _VEC)
                if has_bias != 0:
                    result += bias.load[width=_VEC, alignment=16](c * _VEC)
                output.store[width=_VEC, alignment=16](base + c * _VEC, result)
        row += row_stride


@__name("layer_norm_forward_f32_warp_c1")
def _ln_fwd_warp_c1(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[1](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_warp_c2")
def _ln_fwd_warp_c2(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[2](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_warp_c3")
def _ln_fwd_warp_c3(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[3](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_warp_c4")
def _ln_fwd_warp_c4(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[4](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_warp_c6")
def _ln_fwd_warp_c6(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[6](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_warp_c8")
def _ln_fwd_warp_c8(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    _ln_fwd_warp_rows[8](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        vec_cols,
        epsilon,
        has_weight,
        has_bias,
    )


@__name("layer_norm_forward_f32_vec4")
def _layer_norm_forward_f32_vec4[
    has_weight: Bool, has_bias: Bool
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
):
    """Retain up to two aligned vectors per lane across both reductions."""
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var col0 = tid * _VEC
        var col1 = (tid + _VEC_BLOCK) * _VEC
        var values0 = SIMD[DType.float32, _VEC](0.0)
        var values1 = SIMD[DType.float32, _VEC](0.0)
        if col0 < cols:
            values0 = input.load[width=_VEC, alignment=16](base + col0)
        if col1 < cols:
            values1 = input.load[width=_VEC, alignment=16](base + col1)

        var row_mean = block.sum[block_size=_VEC_BLOCK, broadcast=True](
            values0.reduce_add() + values1.reduce_add()
        ) / Float32(cols)
        var centered0 = values0 - row_mean
        var centered1 = values1 - row_mean
        var thread_variance = Float32(0.0)
        if col0 < cols:
            thread_variance += (centered0 * centered0).reduce_add()
        if col1 < cols:
            thread_variance += (centered1 * centered1).reduce_add()
        var variance = block.sum[block_size=_VEC_BLOCK, broadcast=True](
            thread_variance
        ) / Float32(cols)
        var row_rstd = 1.0 / sqrt(variance + epsilon)
        # A centered-square reduction is nonnegative for every finite row, so
        # no clamp is needed.  Preserve a non-finite reduction as NaN and use
        # it for both statistics: ATen reports NaN mean/rstd for rows that
        # contain NaN or infinity (including the all-infinity case).
        if variance != variance:
            row_mean = variance

        if tid == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd
        if col0 < cols:
            var result0 = centered0 * row_rstd
            comptime if has_weight:
                result0 *= weight.load[width=_VEC, alignment=16](col0)
            comptime if has_bias:
                result0 += bias.load[width=_VEC, alignment=16](col0)
            output.store[width=_VEC, alignment=16](base + col0, result0)
        if col1 < cols:
            var result1 = centered1 * row_rstd
            comptime if has_weight:
                result1 *= weight.load[width=_VEC, alignment=16](col1)
            comptime if has_bias:
                result1 += bias.load[width=_VEC, alignment=16](col1)
            output.store[width=_VEC, alignment=16](base + col1, result1)
        row += row_stride


@__name("layer_norm_forward_f32_baseline")
def _layer_norm_forward_f32_baseline(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var thread_sum = Float32(0.0)
        var col = tid
        while col < cols:
            thread_sum += input[base + col]
            col += _BLOCK
        var row_mean = block.sum[block_size=_BLOCK, broadcast=True](
            thread_sum
        ) / Float32(cols)

        var thread_variance = Float32(0.0)
        col = tid
        while col < cols:
            var centered = input[base + col] - row_mean
            thread_variance += centered * centered
            col += _BLOCK
        var variance = block.sum[block_size=_BLOCK, broadcast=True](
            thread_variance
        ) / Float32(cols)
        var row_rstd = 1.0 / sqrt(variance + epsilon)
        if variance != variance:
            row_mean = variance
        if tid == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd

        col = tid
        while col < cols:
            var value = (input[base + col] - row_mean) * row_rstd
            if has_weight != 0:
                value *= weight[col]
            if has_bias != 0:
                value += bias[col]
            output[base + col] = value
            col += _BLOCK
        row += row_stride


def _enqueue_vec4_cached[
    has_weight: Bool, has_bias: Bool
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    var cache_name = String(
        t"LAYER_NORM_FORWARD_F32_VEC4_V1_{has_weight}_{has_bias}_{ctx.id()}"
    )
    comptime FuncT = type_of(
        ctx.compile_function[
            _layer_norm_forward_f32_vec4[has_weight, has_bias]
        ]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            mean,
            rstd,
            input,
            weight,
            bias,
            rows,
            cols,
            epsilon,
            grid_dim=(min(rows, _MAX_GRID),),
            block_dim=(_VEC_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[
        _layer_norm_forward_f32_vec4[has_weight, has_bias]
    ]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        grid_dim=(min(rows, _MAX_GRID),),
        block_dim=(_VEC_BLOCK,),
    )


def _enqueue_baseline_cached(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"LAYER_NORM_FORWARD_F32_BASE_V1_{ctx.id()}")
    comptime FuncT = type_of(
        ctx.compile_function[_layer_norm_forward_f32_baseline]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            mean,
            rstd,
            input,
            weight,
            bias,
            rows,
            cols,
            epsilon,
            has_weight,
            has_bias,
            grid_dim=(min(rows, _MAX_GRID),),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_layer_norm_forward_f32_baseline]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        has_weight,
        has_bias,
        grid_dim=(min(rows, _MAX_GRID),),
        block_dim=(_BLOCK,),
    )


def enqueue_layer_norm_forward_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    if rows <= 0 or cols <= 0:
        return
    var aligned = (Int(output) | Int(input)) % 16 == 0
    if has_weight:
        aligned = aligned and Int(weight) % 16 == 0
    if has_bias:
        aligned = aligned and Int(bias) % 16 == 0
    comptime if has_apple_gpu_accelerator():
        if aligned and cols % _VEC == 0 and cols <= 8 * WARP_SIZE * _VEC:
            var vec_cols = cols // _VEC
            var chunks = ceildiv(vec_cols, WARP_SIZE)
            var grid = max(1, min(ceildiv(rows, _WARPS_PER_BLOCK), _MAX_GRID))
            var hw = 1 if has_weight else 0
            var hb = 1 if has_bias else 0
            # Smallest ladder entry that covers `chunks`.
            if chunks <= 1:
                _enqueue_cached[_ln_fwd_warp_c1](
                    ctx,
                    "ln_fwd_warp_c1",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            elif chunks <= 2:
                _enqueue_cached[_ln_fwd_warp_c2](
                    ctx,
                    "ln_fwd_warp_c2",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            elif chunks <= 3:
                _enqueue_cached[_ln_fwd_warp_c3](
                    ctx,
                    "ln_fwd_warp_c3",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            elif chunks <= 4:
                _enqueue_cached[_ln_fwd_warp_c4](
                    ctx,
                    "ln_fwd_warp_c4",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            elif chunks <= 6:
                _enqueue_cached[_ln_fwd_warp_c6](
                    ctx,
                    "ln_fwd_warp_c6",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            else:
                _enqueue_cached[_ln_fwd_warp_c8](
                    ctx,
                    "ln_fwd_warp_c8",
                    grid,
                    1,
                    1,
                    WARP_SIZE * _WARPS_PER_BLOCK,
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    vec_cols,
                    epsilon,
                    hw,
                    hb,
                )
            return
    if aligned and cols % _VEC == 0 and cols <= 2 * _VEC_BLOCK * _VEC:
        if has_weight:
            if has_bias:
                _enqueue_vec4_cached[True, True](
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    epsilon,
                    ctx,
                )
            else:
                _enqueue_vec4_cached[True, False](
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    epsilon,
                    ctx,
                )
        elif has_bias:
            _enqueue_vec4_cached[False, True](
                output,
                mean,
                rstd,
                input,
                weight,
                bias,
                rows,
                cols,
                epsilon,
                ctx,
            )
        else:
            _enqueue_vec4_cached[False, False](
                output,
                mean,
                rstd,
                input,
                weight,
                bias,
                rows,
                cols,
                epsilon,
                ctx,
            )
        return
    _enqueue_baseline_cached(
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        1 if has_weight else 0,
        1 if has_bias else 0,
        ctx,
    )
