"""Pure-Mojo benchmark for the two row-wise passes of the eager SDPA decomposition.

Same two-second build/iterate loop as the other harnesses here, and it calls the
same production entry points the Python path reaches:

* `nn_ops._softmax_rows` -- the causal, scaled row softmax of the forward.
* `sdpa_dropout_softmax_backward_kernels.enqueue_sdpa_dropout_softmax_backward`
  -- the fused dropout/softmax backward of the backward.

Both are timed at the nanoGPT attention shape (rows = batch * heads * q_len =
576 * 1024, cols = 1024) and at shapes that are not multiples of the vector
width, so the ragged and scalar regimes are covered.

Correctness is checked against a deliberately naive one-thread-per-row scalar
reference written in this file, which shares no code with the kernels: it walks
each row sequentially in float32 (max, then sum, then write for softmax; and
row_sum then write for the backward).  The gate is the maximum absolute
difference over *every* element, reduced on the device, against the largest
error a single BF16 rounding can produce for the value range involved.

Build and run:
    uv run --no-sync mojo build harness/nanogpt_train/bench_attention_softmax.mojo \
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_attention_softmax
    /tmp/bench_attention_softmax
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, exp
from std.time import perf_counter_ns

from internal_utils import arg_parse
from nn_ops import _softmax_rows
from sdpa_dropout_softmax_backward_kernels import (
    enqueue_sdpa_dropout_softmax_backward,
)

comptime DT = DType.bfloat16
comptime F32 = DType.float32
comptime BLOCK = 256
comptime REDUCE_BLOCKS = 1024


@__name("bench_softmax_fill")
def _fill(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    count: Int,
    seed: Int,
):
    """Deterministic values in [-4, 4), varying along both row and column."""
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    while index < count:
        var h = (index * 2654435761 + seed * 40503) & 0xFFFF
        ptr[index] = (Float32(h) * (8.0 / 65536.0) - 4.0).cast[DT]()
        index += stride


@__name("bench_softmax_reference")
def _softmax_reference(
    out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
):
    var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)
        var m = Float32.MIN
        for j in range(allowed):
            var x = in_ptr[base + j].cast[F32]() * scale
            if x > m:
                m = x
        var denom = Float32(0)
        for j in range(allowed):
            denom += exp(in_ptr[base + j].cast[F32]() * scale - m)
        for j in range(cols):
            if j < allowed:
                out_ptr[base + j] = (
                    exp(in_ptr[base + j].cast[F32]() * scale - m) / denom
                ).cast[DT]()
            else:
                out_ptr[base + j] = Scalar[DT](0)
        row += stride


@__name("bench_sdpa_backward_reference")
def _backward_reference(
    out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    probs: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    q_len: Int,
    causal: Int,
    score_scale: Float32,
):
    var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)
        var row_sum = Float32(0)
        for j in range(allowed):
            row_sum += probs[base + j].cast[F32]() * grad[base + j].cast[F32]()
        for j in range(cols):
            if j < allowed:
                out_ptr[base + j] = (
                    probs[base + j].cast[F32]()
                    * (grad[base + j].cast[F32]() - row_sum)
                    * score_scale
                ).cast[DT]()
            else:
                out_ptr[base + j] = Scalar[DT](0)
        row += stride


@__name("bench_softmax_max_abs_diff")
def _max_abs_diff(
    diffs: UnsafePointer[Scalar[F32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    count: Int,
):
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var worst = Float32(0)
    while index < count:
        var d = abs(a[index].cast[F32]() - b[index].cast[F32]())
        if d > worst:
            worst = d
        index += stride
    diffs[Int(block_idx.x) * BLOCK + Int(thread_idx.x)] = worst


def _reduce_max(
    diffs: DeviceBuffer[F32], slots: Int, ctx: DeviceContext
) raises -> Float64:
    var host = ctx.enqueue_create_host_buffer[F32](slots)
    ctx.enqueue_copy(host, diffs)
    ctx.synchronize()
    var worst = Float32(0)
    for index in range(slots):
        if host[index] > worst:
            worst = host[index]
    return Float64(worst)


def _percentile(sorted_samples: List[Float64], numerator: Int) -> Float64:
    var scaled = numerator * (len(sorted_samples) - 1)
    var lower = scaled // 100
    var remainder = scaled % 100
    if remainder == 0:
        return sorted_samples[lower]
    return (
        sorted_samples[lower]
        + (sorted_samples[lower + 1] - sorted_samples[lower])
        * Float64(remainder)
        / 100.0
    )


def _fixed(value: Float64, decimals: Int) -> String:
    var scale = 1.0
    for _ in range(decimals):
        scale *= 10.0
    return String(Float64(Int(value * scale + 0.5)) / scale)


def _pad(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out += " "
    return out


def run_case(
    label: String,
    rows: Int,
    cols: Int,
    q_len: Int,
    causal: Int,
    calls_per_step: Float64,
    warmup: Int,
    iterations: Int,
    ctx: DeviceContext,
) raises -> Int:
    var count = rows * cols
    var scale = Float32(0.125)
    var input = ctx.enqueue_create_buffer[DT](count)
    var mine = ctx.enqueue_create_buffer[DT](count)
    var reference = ctx.enqueue_create_buffer[DT](count)
    var grad = ctx.enqueue_create_buffer[DT](count)
    var slots = REDUCE_BLOCKS * BLOCK
    var diffs = ctx.enqueue_create_buffer[F32](slots)

    var fill_grid = max(1, min(ceildiv(count, BLOCK), 4096))
    ctx.enqueue_function[_fill](
        input.unsafe_ptr().as_unsafe_any_origin(),
        count,
        1,
        grid_dim=(fill_grid,),
        block_dim=(BLOCK,),
    )
    ctx.enqueue_function[_fill](
        grad.unsafe_ptr().as_unsafe_any_origin(),
        count,
        7,
        grid_dim=(fill_grid,),
        block_dim=(BLOCK,),
    )

    var failures = 0
    var row_grid = max(1, min(ceildiv(rows, BLOCK), 8192))

    # --- forward softmax -------------------------------------------------
    @always_inline
    @parameter
    def _forward() raises:
        _softmax_rows[DT](
            Int(mine.unsafe_ptr()),
            Int(input.unsafe_ptr()),
            rows,
            cols,
            scale,
            causal,
            q_len,
            ctx,
        )

    for _ in range(warmup):
        _forward()
    ctx.synchronize()
    var fwd = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _forward()
        ctx.synchronize()
        fwd.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(fwd)

    ctx.enqueue_function[_softmax_reference](
        reference.unsafe_ptr().as_unsafe_any_origin(),
        input.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        rows,
        cols,
        scale,
        causal,
        q_len,
        grid_dim=(row_grid,),
        block_dim=(BLOCK,),
    )
    ctx.enqueue_function[_max_abs_diff](
        diffs.unsafe_ptr().as_unsafe_any_origin(),
        mine.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        reference.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        count,
        grid_dim=(REDUCE_BLOCKS,),
        block_dim=(BLOCK,),
    )
    var fwd_err = _reduce_max(diffs, slots, ctx)
    # Probabilities are in [0, 1]; one BF16 rounding of a value below 1 is at
    # most 2^-9, and the two implementations reassociate the row sum, so allow
    # two roundings.
    var fwd_ok = fwd_err <= 2.0 / 512.0

    # --- fused dropout/softmax backward ---------------------------------
    # Feed it the reference probabilities, so the zero-masked structure the
    # kernel relies on is exactly the one softmax produces.
    @always_inline
    @parameter
    def _backward() raises:
        enqueue_sdpa_dropout_softmax_backward[DT](
            mine.unsafe_ptr().as_unsafe_any_origin(),
            reference.unsafe_ptr().as_unsafe_any_origin(),
            grad.unsafe_ptr().as_unsafe_any_origin(),
            None,
            rows,
            cols,
            q_len,
            False,
            causal != 0,
            1.0,
            Float64(scale),
            ctx,
        )

    for _ in range(warmup):
        _backward()
    ctx.synchronize()
    var bwd = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _backward()
        ctx.synchronize()
        bwd.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(bwd)

    var bwd_reference = ctx.enqueue_create_buffer[DT](count)
    ctx.enqueue_function[_backward_reference](
        bwd_reference.unsafe_ptr().as_unsafe_any_origin(),
        reference.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        grad.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        rows,
        cols,
        q_len,
        causal,
        scale,
        grid_dim=(row_grid,),
        block_dim=(BLOCK,),
    )
    ctx.enqueue_function[_max_abs_diff](
        diffs.unsafe_ptr().as_unsafe_any_origin(),
        mine.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        bwd_reference.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        count,
        grid_dim=(REDUCE_BLOCKS,),
        block_dim=(BLOCK,),
    )
    var bwd_err = _reduce_max(diffs, slots, ctx)
    # dScores = P * (dP - rowsum) * scale with P <= 1, |dP| <= 4 and
    # |rowsum| <= 4, so |dScores| <= 1; same two-rounding budget.
    var bwd_ok = bwd_err <= 2.0 / 512.0

    print(
        _pad(label, 22),
        _pad(String(rows), 8),
        _pad(String(cols), 6),
        causal,
        _pad(_fixed(_percentile(fwd, 50), 2), 10),
        _pad(_fixed(_percentile(bwd, 50), 2), 10),
        _pad(_fixed(_percentile(fwd, 50) * calls_per_step / 1000.0, 3), 8),
        _pad(_fixed(_percentile(bwd, 50) * calls_per_step / 1000.0, 3), 8),
        _pad(_fixed(fwd_err, 6), 10),
        _pad(_fixed(bwd_err, 6), 10),
        "pass" if fwd_ok and bwd_ok else "FAIL",
    )
    if not fwd_ok or not bwd_ok:
        failures += 1
    _ = input^
    _ = mine^
    _ = reference^
    _ = grad^
    _ = bwd_reference^
    _ = diffs^
    return failures


def main() raises:
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var only = String(arg_parse("case", "all"))
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    print(
        "case                     rows   cols c   fwd_us     bwd_us"
        "     fwd_ms/step bwd_ms/step fwd_err   bwd_err   correct"
    )
    var failures = 0
    with DeviceContext() as ctx:
        # The nanoGPT training shape: 12 layers, one call each per direction.
        if only == "all" or only == "nanogpt_1024":
            failures += run_case(
                "nanogpt_1024",
                576 * 1024,
                1024,
                1024,
                1,
                12.0,
                warmup,
                iterations,
                ctx,
            )
        # Ragged sequence lengths: 1000 is a multiple of the BF16 vector width
        # (8), 1025 is not, so both the vector and the scalar regimes run.
        if only == "all" or only == "ragged_1000":
            failures += run_case(
                "ragged_1000",
                48 * 1000,
                1000,
                1000,
                1,
                12.0,
                warmup,
                iterations,
                ctx,
            )
        if only == "all" or only == "ragged_1025":
            failures += run_case(
                "ragged_1025",
                48 * 1025,
                1025,
                1025,
                1,
                12.0,
                warmup,
                iterations,
                ctx,
            )
        if only == "all" or only == "single_head_512":
            failures += run_case(
                "single_head_512",
                512,
                512,
                512,
                1,
                12.0,
                warmup,
                iterations,
                ctx,
            )
    if failures != 0:
        raise Error(failures, " case(s) FAILED - their timings mean nothing")
    print("correctness: all cases pass")
