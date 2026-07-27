"""Standalone Apple-GPU FP32 linear-backward grad-weight benchmark.

dW = dY^T @ X is the TN layout: dY is stored (rows, n_out) and X (rows,
k_in), both row-major, so the reduction runs down the stored rows. This
harness times the three generations of that path against a naive on-GPU
A^T @ B reference:

    variant 0: permute copy of dY + dense GEMM (what the strided-operand
               fallback in `_matmul_spec_operands_launch` still does for
               layouts the TN route rejects)
    variant 1: the shared `_apple8_fat_kernel[TRANSPOSE_A]` (matmul_ops)
    variant 2: `apple_tn_gemm_enqueue` (apple_gemm_tn_kernels), the
               production route, which picks between the two geometries
    variants 3/4: those two geometries pinned, so the router's choice can be
               checked per shape

    uv run --no-sync mojo build -I torch_mojo_backend/eager_kernels \
        bench_apple_linear_bwd.mojo -o /tmp/bench_linear_bwd
    MODULAR_DEBUG=device-sync-mode uv run --no-sync /tmp/bench_linear_bwd
    /tmp/bench_linear_bwd --reps=20            # floor-free

A synchronize costs ~0.24-0.30 ms on this stack — more than several of these
GEMMs take — so `--reps=N` enqueues N launches per sync and divides, which is
the only way to see the kernel rather than the sync. The default (reps=1)
keeps the historical per-launch medians for comparison. Variants are timed
round-robin inside one loop so thermal drift hits them equally.

Pin clocks for A/B: uv run python bench_gpu_locked.py Maximum -- env ...
Queue-death guard included — on "implausible" errors just rerun.

torch MPS reference: dW-equivalent GEMMs run 0.22-0.77 ms at these shapes
(see bench_apple_gemm.mojo suite, k=2048 rows); MPS pays no separate
permute because its GEMM consumes the transposed view directly.
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.time import perf_counter_ns

from internal_utils import arg_parse

from apple_gemm_tn_kernels import apple_tn_enqueue, apple_tn_gemm_enqueue
from data_movement_ops import _permute_copy
from matmul_ops import _apple8_fat_enqueue, _gemm_enqueue

comptime _BLOCK = 256
comptime _VARIANTS = 5


@__name("bench_linbwd_fill_hash")
def _fill_hash_kernel(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    count: Int,
    salt: Int,
):
    var i = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    while i < count:
        var h = ((i + salt * 1_000_003) * 2654435761) & 0xFFFFFF
        ptr[i] = Float32(h) / Float32(0x1000000) - 0.5
        i += stride


# Reference dW[n_out, k_in] = sum_r dY[r, n_out] * X[r, k_in]; one thread
# per output element.
@__name("bench_linbwd_reference")
def _reference_kernel(
    dw: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dy: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rows: Int,
    n_out: Int,
    k_in: Int,
):
    var idx = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if idx >= n_out * k_in:
        return
    var no = idx // k_in
    var ki = idx - no * k_in
    var acc = Float32(0)
    for r in range(rows):
        acc += dy[r * n_out + no] * x[r * k_in + ki]
    dw[idx] = acc


@always_inline
def _percentile(sorted_samples: List[Float64], numerator: Int) -> Float64:
    var scaled = numerator * (len(sorted_samples) - 1)
    var lower = scaled // 100
    var remainder = scaled % 100
    if remainder == 0:
        return sorted_samples[lower]
    var fraction = Float64(remainder) / 100.0
    return (
        sorted_samples[lower] * (1.0 - fraction)
        + sorted_samples[lower + 1] * fraction
    )


def _run_shape(
    rows: Int,
    n_out: Int,
    k_in: Int,
    warmup: Int,
    iterations: Int,
    reps: Int,
    mps_ms: Float64,
    ctx: DeviceContext,
) raises:
    ctx.synchronize()
    var dy_buf = ctx.enqueue_create_buffer[DType.float32](rows * n_out)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](rows * k_in)
    var dyt_buf = ctx.enqueue_create_buffer[DType.float32](rows * n_out)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](n_out * k_in)
    var ref_buf = ctx.enqueue_create_buffer[DType.float32](n_out * k_in)

    ctx.enqueue_function[_fill_hash_kernel](
        dy_buf.unsafe_ptr().as_unsafe_any_origin(),
        rows * n_out,
        1,
        grid_dim=(1024,),
        block_dim=(_BLOCK,),
    )
    ctx.enqueue_function[_fill_hash_kernel](
        x_buf.unsafe_ptr().as_unsafe_any_origin(),
        rows * k_in,
        2,
        grid_dim=(1024,),
        block_dim=(_BLOCK,),
    )

    # The GEMM is (m, n, k) = (n_out, k_in, rows) with A = dY stored
    # (rows, n_out) — i.e. (k, m) row-major — and B = X stored (k, n).
    @always_inline
    @parameter
    def _launch(variant: Int) raises:
        var c = Int(dw_buf.unsafe_ptr())
        var a = Int(dy_buf.unsafe_ptr())
        var b = Int(x_buf.unsafe_ptr())
        if variant == 0:
            _permute_copy[DType.float32](
                Int(dyt_buf.unsafe_ptr()),
                a,
                1,
                1,
                n_out,
                rows,
                0,
                0,
                1,
                n_out,
                ctx,
            )
            _gemm_enqueue[DType.float32, False](
                c,
                Int(dyt_buf.unsafe_ptr()),
                b,
                1,
                n_out,
                k_in,
                rows,
                n_out * rows,
                ctx,
            )
        elif variant == 1:
            _apple8_fat_enqueue[False, 64, 64, 2, 2, False, 0, True](
                c, a, b, 1, n_out, k_in, rows, n_out * rows, ctx
            )
        elif variant == 2:
            apple_tn_gemm_enqueue(
                c, a, b, 1, n_out, k_in, rows, n_out * rows, ctx
            )
        elif variant == 3:
            apple_tn_enqueue[32, 64, 1, 2](
                c, a, b, 1, n_out, k_in, rows, n_out * rows, ctx
            )
        else:
            apple_tn_enqueue[64, 64, 2, 2](
                c, a, b, 1, n_out, k_in, rows, n_out * rows, ctx
            )

    ctx.enqueue_function[_reference_kernel](
        ref_buf.unsafe_ptr().as_unsafe_any_origin(),
        dy_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        x_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        rows,
        n_out,
        k_in,
        grid_dim=(ceildiv(n_out * k_in, _BLOCK),),
        block_dim=(_BLOCK,),
    )
    var count = n_out * k_in
    var host_got = ctx.enqueue_create_host_buffer[DType.float32](count)
    var host_want = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(host_want, ref_buf)
    ctx.synchronize()

    # ---- correctness (dw is re-poisoned first so a stale result from a
    # previous variant can't mask a kernel that never wrote) ---------------
    for variant in range(_VARIANTS):
        ctx.enqueue_function[_fill_hash_kernel](
            dw_buf.unsafe_ptr().as_unsafe_any_origin(),
            count,
            99,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        _launch(variant)
        ctx.enqueue_copy(host_got, dw_buf)
        ctx.synchronize()
        var max_diff = Float32(0)
        var magnitude = Float32(0)
        for i in range(count):
            var diff = abs(host_got[i] - host_want[i])
            if diff > max_diff:
                max_diff = diff
            var mag = abs(host_want[i])
            if mag > magnitude:
                magnitude = mag
        if magnitude == 0.0:
            raise Error("reference is all zeros: kernels not running")
        if max_diff > 4e-3:
            raise Error(
                t"WRONG RESULT variant={variant} rows={rows} n={n_out}"
                t" k={k_in}: max diff {max_diff}"
            )
        for _ in range(warmup):
            _launch(variant)
    ctx.synchronize()

    # ---- timing (round-robin over variants) -------------------------------
    var samples = List[List[Float64]]()
    for _ in range(_VARIANTS):
        samples.append(List[Float64](capacity=iterations))
    for _ in range(iterations):
        for variant in range(_VARIANTS):
            ctx.synchronize()
            var start = perf_counter_ns()
            for _ in range(reps):
                _launch(variant)
            ctx.synchronize()
            var stop = perf_counter_ns()
            samples[variant].append(
                Float64(stop - start) / 1.0e6 / Float64(reps)
            )
    for variant in range(_VARIANTS):
        sort(samples[variant])
        var median = _percentile(samples[variant], 50)
        var gflops = (
            2.0
            * Float64(rows)
            * Float64(n_out)
            * Float64(k_in)
            / (median * 1.0e6)
        )
        if gflops > 20000.0:
            raise Error(
                t"implausible timing rows={rows} n={n_out} k={k_in}"
                t" ({gflops} GFLOP/s): the GPU queue died — rerun"
            )
        print(
            t"variant={variant} rows={rows} n_out={n_out} k_in={k_in}"
            t"  median_ms={median}  gflops={gflops}  mps_ms={mps_ms}"
            t"  slowdown_vs_mps={median / mps_ms}"
        )
    ctx.synchronize()


def main() raises:
    var rows = Int(arg_parse("rows", 0))
    var n_out = Int(arg_parse("n", 0))
    var k_in = Int(arg_parse("k", 0))
    var warmup = Int(arg_parse("warmup", 10))
    var iterations = Int(arg_parse("iterations", 21))
    var reps = Int(arg_parse("reps", 1))

    with DeviceContext() as ctx:
        if rows > 0 and n_out > 0 and k_in > 0:
            _run_shape(rows, n_out, k_in, warmup, iterations, reps, 1.0, ctx)
            return
        # nanoGPT dW shapes: dY (2048, n_out), X (2048, k_in); MPS reference
        # is its plain dense GEMM at the same shape (it needs no permute).
        _run_shape(2048, 1152, 384, warmup, iterations, reps, 0.571, ctx)
        _run_shape(2048, 384, 384, warmup, iterations, reps, 0.224, ctx)
        _run_shape(2048, 1536, 384, warmup, iterations, reps, 0.768, ctx)
        _run_shape(2048, 384, 1536, warmup, iterations, reps, 0.750, ctx)
        _run_shape(2048, 65, 384, warmup, iterations, reps, 0.066, ctx)
        # Odd-shape guards for the ragged M/N edges and the K tail, and
        # deep-k / wide-grid shapes that take the 64-row branch (mps_ms is a
        # 1.0 placeholder on these).
        _run_shape(1000, 100, 100, warmup, iterations, reps, 1.0, ctx)
        _run_shape(333, 17, 51, warmup, iterations, reps, 1.0, ctx)
        _run_shape(8192, 768, 768, warmup, iterations, reps, 1.0, ctx)
        _run_shape(2048, 2048, 2048, warmup, iterations, reps, 1.0, ctx)
