"""Pure-Mojo H100 candidate for LayerNorm f32 affine parameter gradients.

Two runtime regimes, both with deterministic fixed-order f32 reductions and no
global atomics:

  * direct: rows <= _DIRECT_MAX_ROWS, or a degenerate single-chunk geometry.
    One launch; each thread owns one column and accumulates every row
    sequentially.
  * two-stage: a partial kernel where block_idx.y selects a fixed contiguous
    row chunk (bounded chunk count) and each thread owns one column inside the
    chunk, then a final kernel where _FINAL_TY chunk-lanes per column each sum
    a fixed strided chunk subset and are combined in fixed lane order through
    shared memory.  Chunk partials live in ordinary context-owned scratch
    allocated and released inside the enqueue call.

Column ownership makes every global load warp-coalesced; the chunk geometry
depends only on (rows, cols), so repeated invocations produce identical f32
bit patterns.
"""

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import has_accelerator

from op_utils import _enqueue_cached, _enqueue_cached_2d

comptime _BLOCK = 128
comptime _DIRECT_MAX_ROWS = 128
comptime _MIN_CHUNK_ROWS = 32
comptime _MAX_PARTIALS = 512
comptime _TARGET_BLOCKS = 1280
comptime _UNROLL = 8
comptime _FINAL_TX = 32
comptime _FINAL_TY = 32


@__name("nanogpt_layer_norm_backward_params_direct")
def _direct_kernel(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    want_weight: Int,
    want_bias: Int,
):
    var col = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if col >= cols:
        return
    var weight_acc = Float32(0.0)
    var bias_acc = Float32(0.0)
    var index = col
    if want_weight != 0:
        for row in range(rows):
            var dy = grad_output[index]
            var xhat = (input[index] - mean[row]) * rstd[row]
            weight_acc += dy * xhat
            bias_acc += dy
            index += cols
        grad_weight[col] = weight_acc
    else:
        for _ in range(rows):
            bias_acc += grad_output[index]
            index += cols
    if want_bias != 0:
        grad_bias[col] = bias_acc


@__name("nanogpt_layer_norm_backward_params_partial")
def _partial_kernel(
    partial_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    partial_bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    chunk_rows: Int,
    want_weight: Int,
    want_bias: Int,
):
    var col = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if col >= cols:
        return
    var chunk = Int(block_idx.y)
    var row_start = chunk * chunk_rows
    var row_end = min(row_start + chunk_rows, rows)
    var weight_acc = Float32(0.0)
    var bias_acc = Float32(0.0)
    var row = row_start
    var index = row_start * cols + col
    if want_weight != 0:
        while row + _UNROLL <= row_end:
            comptime for k in range(_UNROLL):
                var dy = grad_output[index + k * cols]
                var x = input[index + k * cols]
                weight_acc += dy * ((x - mean[row + k]) * rstd[row + k])
                bias_acc += dy
            row += _UNROLL
            index += _UNROLL * cols
        while row < row_end:
            var dy = grad_output[index]
            weight_acc += dy * ((input[index] - mean[row]) * rstd[row])
            bias_acc += dy
            row += 1
            index += cols
        partial_weight[chunk * cols + col] = weight_acc
    else:
        while row + _UNROLL <= row_end:
            comptime for k in range(_UNROLL):
                bias_acc += grad_output[index + k * cols]
            row += _UNROLL
            index += _UNROLL * cols
        while row < row_end:
            bias_acc += grad_output[index]
            row += 1
            index += cols
    if want_bias != 0:
        partial_bias[chunk * cols + col] = bias_acc


@__name("nanogpt_layer_norm_backward_params_final")
def _final_kernel(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    partial_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    partial_bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    cols: Int,
    num_chunks: Int,
    want_weight: Int,
    want_bias: Int,
):
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var col = Int(block_idx.x) * _FINAL_TX + tx
    var shared_weight = stack_allocation[
        _FINAL_TX * _FINAL_TY,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_bias = stack_allocation[
        _FINAL_TX * _FINAL_TY,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var weight_acc = Float32(0.0)
    var bias_acc = Float32(0.0)
    if col < cols:
        # Lane ty owns chunks ty, ty+_FINAL_TY, ... in ascending order.
        var chunk = ty
        while chunk < num_chunks:
            var index = chunk * cols + col
            if want_weight != 0:
                weight_acc += partial_weight[index]
            if want_bias != 0:
                bias_acc += partial_bias[index]
            chunk += _FINAL_TY
    shared_weight[ty * _FINAL_TX + tx] = weight_acc
    shared_bias[ty * _FINAL_TX + tx] = bias_acc
    barrier()
    if ty == 0 and col < cols:
        if want_weight != 0:
            var total = Float32(0.0)
            comptime for lane in range(_FINAL_TY):
                total += shared_weight[lane * _FINAL_TX + tx]
            grad_weight[col] = total
        if want_bias != 0:
            var total = Float32(0.0)
            comptime for lane in range(_FINAL_TY):
                total += shared_bias[lane * _FINAL_TX + tx]
            grad_bias[col] = total


def enqueue_layer_norm_backward_params_f32(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    want_weight: Bool,
    want_bias: Bool,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        if not want_weight and not want_bias:
            return
        var weight_flag = 1 if want_weight else 0
        var bias_flag = 1 if want_bias else 0
        var col_blocks = ceildiv(cols, _BLOCK)

        if rows <= _DIRECT_MAX_ROWS:
            _enqueue_cached[_direct_kernel](
                ctx,
                "nanogpt_layer_norm_backward_params_direct",
                col_blocks,
                1,
                1,
                _BLOCK,
                grad_weight,
                grad_bias,
                grad_output,
                input,
                mean,
                rstd,
                rows,
                cols,
                weight_flag,
                bias_flag,
            )
            return

        # Bounded chunk count: fill a fixed block budget across the column
        # blocks, never exceed the hard partial cap, and keep chunks at least
        # _MIN_CHUNK_ROWS rows.  Rounding chunk_rows to the unroll width keeps
        # the unrolled loop tail-free for interior chunks.  The geometry is a
        # pure function of (rows, cols), so replays are bit-identical.
        var num_chunks = max(1, _TARGET_BLOCKS // col_blocks)
        num_chunks = min(num_chunks, _MAX_PARTIALS)
        num_chunks = min(num_chunks, ceildiv(rows, _MIN_CHUNK_ROWS))
        var chunk_rows = ceildiv(ceildiv(rows, num_chunks), _UNROLL) * _UNROLL
        num_chunks = ceildiv(rows, chunk_rows)
        if num_chunks == 1:
            # A single chunk degenerates to the direct regime: no scratch needed.
            _enqueue_cached[_direct_kernel](
                ctx,
                "nanogpt_layer_norm_backward_params_direct",
                col_blocks,
                1,
                1,
                _BLOCK,
                grad_weight,
                grad_bias,
                grad_output,
                input,
                mean,
                rstd,
                rows,
                cols,
                weight_flag,
                bias_flag,
            )
            return
        var lane = num_chunks * cols
        var lanes = weight_flag + bias_flag
        var scratch = ctx.enqueue_create_buffer[DType.float32](lanes * lane)
        var scratch_base = scratch.unsafe_ptr().as_unsafe_any_origin()
        var partial_weight = scratch_base
        var partial_bias = scratch_base + (lane if want_weight else 0)

        _enqueue_cached[_partial_kernel](
            ctx,
            "nanogpt_layer_norm_backward_params_partial",
            col_blocks,
            num_chunks,
            1,
            _BLOCK,
            partial_weight,
            partial_bias,
            grad_output,
            input,
            mean,
            rstd,
            rows,
            cols,
            chunk_rows,
            weight_flag,
            bias_flag,
        )
        _enqueue_cached_2d[_final_kernel](
            ctx,
            "nanogpt_layer_norm_backward_params_final",
            ceildiv(cols, _FINAL_TX),
            1,
            1,
            _FINAL_TX,
            _FINAL_TY,
            grad_weight,
            grad_bias,
            partial_weight,
            partial_bias,
            cols,
            num_chunks,
            weight_flag,
            bias_flag,
        )
        # Normal release after both queued consumers are enqueued in stream order.
        _ = scratch^
    else:
        raise Error("no GPU accelerator available at compile time")
