"""Standalone Apple-GPU FP32 GEMM benchmark for the eager mojo_device path.

Benchmarks the production dispatch (`_gemm_enqueue` in
`torch_mojo_backend/eager_kernels/matmul_ops.mojo`) on the GEMM shapes one
nanoGPT training step issues (batch 8, block 256, n_embd 384: forward
`x @ w^T`, grad-input `dy @ w`, grad-weight `dy^T @ x`), validating every
run against a naive on-GPU reference. No Python, no pytest: edit the kernels,
rebuild this file, rerun — the whole loop is one `mojo build`.

    uv run --no-sync mojo build -I torch_mojo_backend/eager_kernels \
        bench_apple_gemm.mojo -o /tmp/bench_apple_gemm
    MODULAR_DEBUG=device-sync-mode uv run --no-sync /tmp/bench_apple_gemm

Always run under MODULAR_DEBUG=device-sync-mode: plain async submission on
this Metal driver intermittently kills the queue mid-run, after which every
launch silently no-ops (the implausible-throughput guard below catches it).
Single shape: append `--m=2048 --n=1536 --k=384 --tb=1` (equals form only).
Single-shape runs also take `--geo=N` to pin one `_apple8_fat_enqueue` tile
geometry instead of the production dispatch (see `_launch_geo` below).

Reference numbers measured on this machine (torch MPS, f32, 50-iter mean):
the suite table below carries the per-shape MPS milliseconds; MPS sustains
roughly 3 TFLOP/s on the fat shapes.  The end-to-end gap this suite explains:
mojo linear fwd+bwd ~339 ms per training step vs ~57 ms on MPS.
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext, HostBuffer
from std.math import ceildiv
from std.time import perf_counter_ns

from internal_utils import arg_parse

from matmul_ops import _apple8_fat_enqueue, _gemm_enqueue

comptime _FILL_BLOCK = 256


# Deterministic pseudo-random fill in [-0.5, 0.5): keeps accumulation error
# bounded and every run reproducible without host RNG.
@__name("bench_apple_gemm_fill_hash")
def _fill_hash_kernel(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    count: Int,
    salt: Int,
):
    var i = Int(block_idx.x) * _FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _FILL_BLOCK
    while i < count:
        var h = ((i + salt * 1_000_003) * 2654435761) & 0xFFFFFF
        ptr[i] = Float32(h) / Float32(0x1000000) - 0.5
        i += stride


# One thread per output element; sequential fp32 k-loop.
@__name("bench_apple_gemm_reference")
def _reference_kernel(
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
):
    var idx = Int(block_idx.x) * _FILL_BLOCK + Int(thread_idx.x)
    if idx >= m * n:
        return
    var row = idx // n
    var col = idx - row * n
    var acc = Float32(0)
    for kk in range(k):
        var bv = b[col * k + kk] if transpose_b != 0 else b[kk * n + col]
        acc += a[row * k + kk] * bv
    c[idx] = acc


@always_inline
def _enqueue_fill(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    count: Int,
    salt: Int,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_fill_hash_kernel](
        ptr,
        count,
        salt,
        grid_dim=(max(1, min(ceildiv(count, _FILL_BLOCK), 1024)),),
        block_dim=(_FILL_BLOCK,),
    )


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
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    warmup: Int,
    iterations: Int,
    mps_ms: Float64,
    host_c: HostBuffer[DType.float32],
    host_ref: HostBuffer[DType.float32],
    ctx: DeviceContext,
    geo: Int = 0,
) raises:
    ctx.synchronize()
    var a_buf = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](k * n)
    var c_buf = ctx.enqueue_create_buffer[DType.float32](m * n)
    var ref_buf = ctx.enqueue_create_buffer[DType.float32](m * n)

    var a_addr = Int(a_buf.unsafe_ptr())
    var b_addr = Int(b_buf.unsafe_ptr())
    var c_addr = Int(c_buf.unsafe_ptr())

    _enqueue_fill(a_buf.unsafe_ptr().as_unsafe_any_origin(), m * k, 1, ctx)
    _enqueue_fill(b_buf.unsafe_ptr().as_unsafe_any_origin(), k * n, 2, ctx)

    # Tile-geometry experiments for the Apple fat kernel (`--geo=N`, single
    # shape only): 0 = production dispatch, others force one
    # `_apple8_fat_enqueue` instantiation [BLOCK_M, BLOCK_N, SG_ROWS, SG_COLS].
    @always_inline
    @parameter
    def _launch_geo[tb: Bool]() raises:
        if geo == 1:
            _apple8_fat_enqueue[tb, 64, 64, 2, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 2:
            _apple8_fat_enqueue[tb, 128, 64, 2, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 3:
            _apple8_fat_enqueue[tb, 64, 128, 2, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 4:
            _apple8_fat_enqueue[tb, 128, 64, 4, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 5:
            _apple8_fat_enqueue[tb, 64, 128, 2, 4](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 6:
            _apple8_fat_enqueue[tb, 128, 128, 4, 4](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 7:
            _apple8_fat_enqueue[tb, 96, 64, 2, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 8:
            _apple8_fat_enqueue[tb, 32, 64, 2, 2](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        elif geo == 9:
            # Threadgroup-staged double-buffered variant.
            _apple8_fat_enqueue[tb, 64, 64, 2, 2, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )
        else:
            _gemm_enqueue[DType.float32, tb](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
            )

    @always_inline
    @parameter
    def _launch() raises:
        if transpose_b != 0:
            _launch_geo[True]()
        else:
            _launch_geo[False]()

    # ---- correctness (against the naive on-GPU reference) -----------------
    ctx.synchronize()
    _launch()
    ctx.synchronize()
    ctx.enqueue_function[_reference_kernel](
        ref_buf.unsafe_ptr().as_unsafe_any_origin(),
        a_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        b_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        m,
        n,
        k,
        transpose_b,
        grid_dim=(ceildiv(m * n, _FILL_BLOCK),),
        block_dim=(_FILL_BLOCK,),
    )
    ctx.enqueue_copy(host_c.create_sub_buffer[DType.float32](0, m * n), c_buf)
    ctx.enqueue_copy(
        host_ref.create_sub_buffer[DType.float32](0, m * n), ref_buf
    )
    ctx.synchronize()
    var max_diff = Float32(0)
    var ref_magnitude = Float32(0)
    for i in range(m * n):
        var diff = abs(host_c[i] - host_ref[i])
        if diff > max_diff:
            max_diff = diff
        var mag = abs(host_ref[i])
        if mag > ref_magnitude:
            ref_magnitude = mag
    if ref_magnitude == 0.0:
        raise Error(
            t"reference output is all zeros (m={m} n={n} k={k}): kernels are"
            t" not actually running"
        )
    if max_diff > 2e-3:
        raise Error(
            t"WRONG RESULT m={m} n={n} k={k} tb={transpose_b}:"
            t" max |mojo - ref| = {max_diff}"
        )

    # ---- timing -----------------------------------------------------------
    for _ in range(warmup):
        _launch()
    ctx.synchronize()
    var samples = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _launch()
        ctx.synchronize()
        var stop = perf_counter_ns()
        samples.append(Float64(stop - start) / 1.0e6)
    sort(samples)

    var median = _percentile(samples, 50)
    var p90 = _percentile(samples, 90)
    var gflops = 2.0 * Float64(m) * Float64(n) * Float64(k) / (median * 1.0e6)
    # The Metal queue occasionally dies mid-run and later launches silently
    # no-op; that shows up as physically impossible throughput. Fail loudly
    # instead of reporting garbage (M-series f32 peak is ~10 TFLOP/s).
    if gflops > 20000.0:
        # Distinguish "queue dead" from "synchronize stopped blocking": scrub
        # C, relaunch once, and re-verify against the reference on the host.
        var zero_probe = host_c.create_sub_buffer[DType.float32](0, m * n)
        ctx.enqueue_memset(c_buf, 0)
        _launch()
        ctx.enqueue_copy(zero_probe, c_buf)
        ctx.synchronize()
        var still_correct = True
        for i in range(m * n):
            if abs(host_c[i] - host_ref[i]) > 2e-3:
                still_correct = False
                break
        raise Error(
            t"implausible timing for m={m} n={n} k={k} tb={transpose_b}"
            t" ({gflops} GFLOP/s); relaunch-after-timing still correct:"
            t" {still_correct} (True = synchronize stopped blocking,"
            t" False = the GPU queue died)"
        )
    var vs_mps = median / mps_ms if mps_ms > 0 else 0.0
    print(
        t"m={m} n={n} k={k} tb={transpose_b}  median_ms={median}"
        t"  p90_ms={p90}  gflops={gflops}  mps_ms={mps_ms}"
        t"  slowdown_vs_mps={vs_mps}  maxdiff={max_diff}"
    )
    ctx.synchronize()


def main() raises:
    var m = Int(arg_parse("m", 0))
    var n = Int(arg_parse("n", 0))
    var k = Int(arg_parse("k", 0))
    var transpose_b = Int(arg_parse("tb", 0))
    var warmup = Int(arg_parse("warmup", 10))
    var iterations = Int(arg_parse("iterations", 30))
    var geo = Int(arg_parse("geo", 0))

    with DeviceContext() as ctx:
        comptime MAX_OUT = 2048 * 1536
        var host_c = ctx.enqueue_create_host_buffer[DType.float32](MAX_OUT)
        var host_ref = ctx.enqueue_create_host_buffer[DType.float32](MAX_OUT)
        ctx.synchronize()
        if m > 0 and n > 0 and k > 0:
            if m * n > MAX_OUT:
                raise Error("shape exceeds preallocated host readback buffer")
            _run_shape(
                m,
                n,
                k,
                transpose_b,
                warmup,
                iterations,
                0.0,
                host_c,
                host_ref,
                ctx,
                geo,
            )
            return

        # nanoGPT training-step suite (batch 8, block 256 -> rows 2048).
        # Columns: m, n, k, transpose_b, torch-MPS f32 reference ms.
        # fwd x @ w^T
        _run_shape(
            2048, 1152, 384, 1, warmup, iterations, 0.616, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 384, 384, 1, warmup, iterations, 0.216, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 1536, 384, 1, warmup, iterations, 0.781, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 384, 1536, 1, warmup, iterations, 0.822, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 65, 384, 1, warmup, iterations, 0.085, host_c, host_ref, ctx
        )
        # bwd grad-input dy @ w
        _run_shape(
            2048, 384, 1152, 0, warmup, iterations, 0.576, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 384, 384, 0, warmup, iterations, 0.200, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 1536, 384, 0, warmup, iterations, 0.717, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 384, 1536, 0, warmup, iterations, 0.762, host_c, host_ref, ctx
        )
        _run_shape(
            2048, 384, 65, 0, warmup, iterations, 0.053, host_c, host_ref, ctx
        )
        # bwd grad-weight dy^T @ x (materialized-contiguous A)
        _run_shape(
            1152, 384, 2048, 0, warmup, iterations, 0.571, host_c, host_ref, ctx
        )
        _run_shape(
            384, 384, 2048, 0, warmup, iterations, 0.224, host_c, host_ref, ctx
        )
        _run_shape(
            1536, 384, 2048, 0, warmup, iterations, 0.768, host_c, host_ref, ctx
        )
        _run_shape(
            384, 1536, 2048, 0, warmup, iterations, 0.750, host_c, host_ref, ctx
        )
        _run_shape(
            65, 384, 2048, 0, warmup, iterations, 0.066, host_c, host_ref, ctx
        )
