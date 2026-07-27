"""Standalone Apple-GPU FP32 SDPA *forward* benchmark for mojo_device.

Benchmarks the exact kernel sequence `aten_fast._sdpa_math_forward_with_dropout`
launches today on Metal for the nanoGPT training shape (batch 8, heads 6,
seq 256, head_dim 64, causal, f32):

    stage 1   scores = Q @ K^T   causal batched GEMM, tb=1, upper-triangle
              simdgroup tiles skipped/unwritten      (matmul_ops CAUSAL=1)
    stage 2   probs = causal softmax(scale * scores) (nn_ops SoftmaxRows,
              Apple warp kernel) — dropout == 0 only
    stage 2+3 probs, p_drop, mask in one fused launch when 0 < p < 1
              (nn_ops SoftmaxRowsDropoutF32; same Philox stream as the
              standalone dropout kernel, byte-identical outputs)
    stage 4   out = p_drop @ V   causal batched GEMM, tb=0, reduction cut at
              the causal boundary                    (matmul_ops CAUSAL=2)

The old standalone dropout kernel is still timed (dropout_ms) for comparison
but is no longer part of the composed total. copy_ms/copy_gbs report a pure
vec4 device copy of the probs matrix — the bandwidth roofline a softmax-like
stage can hope for on this machine (~47 GB/s under sync-mode).

It validates the composed result against a naive on-GPU attention reference
(dropout applied through the *produced* mask, so the check is exact and
deterministic), reports the fused total plus a per-stage breakdown, and
carries the torch-MPS reference times measured on this machine.

    uv run --no-sync mojo build -I torch_mojo_backend/eager_kernels \
        bench_apple_sdpa_fwd.mojo -o /tmp/bench_sdpa_fwd
    MODULAR_DEBUG=device-sync-mode uv run --no-sync /tmp/bench_sdpa_fwd

For stable clocks run it under the pin wrapper:

    uv run python bench_gpu_locked.py Maximum -- env \
        MODULAR_DEBUG=device-sync-mode /tmp/bench_sdpa_fwd

Always use MODULAR_DEBUG=device-sync-mode (see bench_apple_gemm.mojo: the
async Metal queue intermittently dies; the implausible-time guard here
catches it — just rerun). Shape overrides: --bh=48 --q=256 --s=256 --d=64
(equals form only; bh = batch * heads).

torch MPS references at this shape (training decomposed path, this machine):
    fwd dropout=0.0: 1.93 ms      fwd dropout=0.2: 2.94 ms
The current mojo sequence is the "before"; parity with MPS is the target,
beating it is plausible since MPS also runs unfused here.
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext, HostBuffer
from std.math import ceildiv, exp, sqrt
from std.time import perf_counter_ns

from internal_utils import arg_parse

from matmul_ops import _apple8_fat_enqueue, _gemm_enqueue
from native_dropout_kernels import enqueue_native_dropout_f32
from nn_ops import _softmax_rows, enqueue_softmax_rows_dropout_f32

comptime _BLOCK = 256
comptime _SEED = UInt64(0x5EED_5EED)


@__name("bench_sdpa_copy_vec4")
def _copy_vec4_kernel(
    dst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    count4: Int,
):
    var i = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    while i < count4:
        dst.store[width=4, alignment=16](
            i * 4, src.load[width=4, alignment=16](i * 4)
        )
        i += stride


@__name("bench_sdpa_fill_hash")
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


# Naive reference: one thread per (bh, qi) row. Recomputes scores + causal
# softmax in f32, then applies the mask/scale the benchmarked pipeline
# actually produced, so dropout runs are checked exactly, not statistically.
@__name("bench_sdpa_reference")
def _reference_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ref_probs: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    q: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    bh: Int,
    q_len: Int,
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
    keep_scale: Float32,
    use_mask: Int,
):
    var row = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if row >= bh * q_len:
        return
    var b = row // q_len
    var qi = row - b * q_len
    var q_base = (b * q_len + qi) * head_dim
    var kv_base = b * kv_len * head_dim
    var allowed = qi + 1  # causal

    var m = Float32(-3.4e38)
    for j in range(allowed):
        var s = Float32(0)
        for dd in range(head_dim):
            s += q[q_base + dd] * k[kv_base + j * head_dim + dd]
        s *= scale
        if s > m:
            m = s
    var denom = Float32(0)
    for j in range(allowed):
        var s = Float32(0)
        for dd in range(head_dim):
            s += q[q_base + dd] * k[kv_base + j * head_dim + dd]
        denom += exp(s * scale - m)

    for dd in range(head_dim):
        var acc = Float32(0)
        for j in range(allowed):
            var s = Float32(0)
            for d2 in range(head_dim):
                s += q[q_base + d2] * k[kv_base + j * head_dim + d2]
            var p = exp(s * scale - m) / denom
            if use_mask != 0:
                var kept = mask[(b * q_len + qi) * kv_len + j]
                p = p * keep_scale if kept else Float32(0)
            acc += p * v[kv_base + j * head_dim + dd]
        out_ptr[q_base + dd] = acc

    # Pre-dropout probabilities for validating the probs the pipeline saves
    # for backward (row written once per thread; padded region must be 0).
    for j in range(kv_len):
        var p = Float32(0)
        if j < allowed:
            var s = Float32(0)
            for d2 in range(head_dim):
                s += q[q_base + d2] * k[kv_base + j * head_dim + d2]
            p = exp(s * scale - m) / denom
        ref_probs[(b * q_len + qi) * kv_len + j] = p


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


def _time_stage[
    launch: def() raises capturing -> None
](warmup: Int, iterations: Int, ctx: DeviceContext) raises -> Float64:
    for _ in range(warmup):
        launch()
    ctx.synchronize()
    var samples = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        launch()
        ctx.synchronize()
        var stop = perf_counter_ns()
        samples.append(Float64(stop - start) / 1.0e6)
    sort(samples)
    return _percentile(samples, 50)


def main() raises:
    var bh = Int(arg_parse("bh", 48))
    var q_len = Int(arg_parse("q", 256))
    var kv_len = Int(arg_parse("s", 256))
    var head_dim = Int(arg_parse("d", 64))
    var warmup = Int(arg_parse("warmup", 10))
    var iterations = Int(arg_parse("iterations", 30))
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    with DeviceContext() as ctx:
        var q_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * head_dim
        )
        var k_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * kv_len * head_dim
        )
        var v_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * kv_len * head_dim
        )
        var scores_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * kv_len
        )
        var probs_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * kv_len
        )
        var pdrop_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * kv_len
        )
        var mask_buf = ctx.enqueue_create_buffer[DType.bool](
            bh * q_len * kv_len
        )
        var out_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * head_dim
        )
        var ref_out_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * head_dim
        )
        var ref_probs_buf = ctx.enqueue_create_buffer[DType.float32](
            bh * q_len * kv_len
        )

        ctx.enqueue_function[_fill_hash_kernel](
            q_buf.unsafe_ptr().as_unsafe_any_origin(),
            bh * q_len * head_dim,
            1,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_fill_hash_kernel](
            k_buf.unsafe_ptr().as_unsafe_any_origin(),
            bh * kv_len * head_dim,
            2,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_fill_hash_kernel](
            v_buf.unsafe_ptr().as_unsafe_any_origin(),
            bh * kv_len * head_dim,
            3,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.synchronize()

        var q_addr = Int(q_buf.unsafe_ptr())
        var k_addr = Int(k_buf.unsafe_ptr())
        var v_addr = Int(v_buf.unsafe_ptr())
        var scores_addr = Int(scores_buf.unsafe_ptr())
        var probs_addr = Int(probs_buf.unsafe_ptr())
        var out_addr = Int(out_buf.unsafe_ptr())
        var pdrop_ptr = pdrop_buf.unsafe_ptr().as_unsafe_any_origin()
        var mask_ptr = mask_buf.unsafe_ptr().as_unsafe_any_origin()
        var probs_mut = probs_buf.unsafe_ptr().as_unsafe_any_origin()

        for dropout_case in [0, 1]:
            var p = Float64(0.2) if dropout_case == 1 else Float64(0.0)
            var keep_scale = Float32(1.0 / (1.0 - p))
            var elements = bh * q_len * kv_len

            # Production launches on Apple (aten_fast metal path): causal QK
            # (upper-triangle simdgroup tiles skipped, left unwritten),
            # fused softmax+dropout when 0 < p < 1 (else plain SoftmaxRows),
            # and the causal-K-cut PV GEMM.
            @always_inline
            @parameter
            def _stage1() raises:  # scores = Q @ K^T (causal tiles skipped)
                _apple8_fat_enqueue[True, 64, 64, 2, 2, False, 1](
                    scores_addr,
                    q_addr,
                    k_addr,
                    bh,
                    q_len,
                    kv_len,
                    head_dim,
                    q_len * head_dim,
                    ctx,
                )

            @always_inline
            @parameter
            def _stage2() raises:  # probs = causal softmax(scale * scores)
                _softmax_rows[DType.float32](
                    probs_addr,
                    scores_addr,
                    bh * q_len,
                    kv_len,
                    scale,
                    1,
                    q_len,
                    ctx,
                )

            @always_inline
            @parameter
            def _stage23() raises:  # probs, p_drop, mask in one launch
                enqueue_softmax_rows_dropout_f32(
                    probs_addr,
                    Int(pdrop_buf.unsafe_ptr()),
                    Int(mask_buf.unsafe_ptr()),
                    scores_addr,
                    bh * q_len,
                    kv_len,
                    scale,
                    1,
                    q_len,
                    p,
                    _SEED,
                    UInt64(0),
                    ctx,
                )

            @always_inline
            @parameter
            def _stage3() raises:  # unfused dropout (kept as comparison)
                enqueue_native_dropout_f32(
                    pdrop_ptr,
                    mask_ptr,
                    probs_mut,
                    elements,
                    p,
                    _SEED,
                    UInt64(0),
                    ctx,
                )

            @always_inline
            @parameter
            def _stage4() raises:  # out = p_drop @ V (K cut at the boundary)
                var src = (
                    Int(pdrop_buf.unsafe_ptr()) if dropout_case
                    == 1 else probs_addr
                )
                _apple8_fat_enqueue[False, 64, 64, 2, 2, False, 2](
                    out_addr,
                    src,
                    v_addr,
                    bh,
                    q_len,
                    head_dim,
                    kv_len,
                    q_len * kv_len,
                    ctx,
                )

            @always_inline
            @parameter
            def _full() raises:
                _stage1()
                if dropout_case == 1:
                    _stage23()
                else:
                    _stage2()
                _stage4()

            # ---- correctness ------------------------------------------------
            ctx.synchronize()
            _full()
            ctx.synchronize()
            ctx.enqueue_function[_reference_kernel](
                ref_out_buf.unsafe_ptr().as_unsafe_any_origin(),
                ref_probs_buf.unsafe_ptr().as_unsafe_any_origin(),
                q_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                k_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                v_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                mask_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                bh,
                q_len,
                kv_len,
                head_dim,
                scale,
                keep_scale,
                dropout_case,
                grid_dim=(ceildiv(bh * q_len, _BLOCK),),
                block_dim=(_BLOCK,),
            )
            var n_out = bh * q_len * head_dim
            var n_probs = bh * q_len * kv_len
            var host_a = ctx.enqueue_create_host_buffer[DType.float32](n_out)
            var host_b = ctx.enqueue_create_host_buffer[DType.float32](n_out)
            var host_p = ctx.enqueue_create_host_buffer[DType.float32](n_probs)
            var host_rp = ctx.enqueue_create_host_buffer[DType.float32](n_probs)
            ctx.enqueue_copy(host_a, out_buf)
            ctx.enqueue_copy(host_b, ref_out_buf)
            ctx.enqueue_copy(host_p, probs_buf)
            ctx.enqueue_copy(host_rp, ref_probs_buf)
            ctx.synchronize()
            var max_out_diff = Float32(0)
            var out_magnitude = Float32(0)
            for i in range(n_out):
                var diff = abs(host_a[i] - host_b[i])
                if diff > max_out_diff:
                    max_out_diff = diff
                var mag = abs(host_b[i])
                if mag > out_magnitude:
                    out_magnitude = mag
            var max_probs_diff = Float32(0)
            for i in range(n_probs):
                var diff = abs(host_p[i] - host_rp[i])
                if diff > max_probs_diff:
                    max_probs_diff = diff
            if out_magnitude == 0.0:
                raise Error(
                    "reference output is all zeros: kernels not running"
                )
            if max_out_diff > 2e-3 or max_probs_diff > 2e-4:
                raise Error(
                    t"WRONG RESULT dropout={p}: out diff {max_out_diff},"
                    t" probs diff {max_probs_diff}"
                )

            # ---- timing -----------------------------------------------------
            var total = _time_stage[_full](warmup, iterations, ctx)
            var s1 = _time_stage[_stage1](warmup, iterations, ctx)
            # s2 is the fused softmax+dropout launch in the dropout case; s3
            # then reports the old standalone dropout kernel for comparison
            # (it is NOT part of _full anymore).
            var s2: Float64
            if dropout_case == 1:
                s2 = _time_stage[_stage23](warmup, iterations, ctx)
            else:
                s2 = _time_stage[_stage2](warmup, iterations, ctx)
            var s3 = Float64(0)
            if dropout_case == 1:
                s3 = _time_stage[_stage3](warmup, iterations, ctx)
            var s4 = _time_stage[_stage4](warmup, iterations, ctx)

            # Bandwidth probe: pure vec4 copy of the probs matrix (read +
            # write = 2 * elements * 4 bytes) to expose the achievable
            # roofline for the softmax stage on this machine.
            @always_inline
            @parameter
            def _copy_probe() raises:
                ctx.enqueue_function[_copy_vec4_kernel](
                    pdrop_ptr,
                    probs_buf.unsafe_ptr()
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    elements // 4,
                    grid_dim=(2048,),
                    block_dim=(_BLOCK,),
                )

            var scopy = _time_stage[_copy_probe](warmup, iterations, ctx)
            var copy_gbs = Float64(2 * elements * 4) / (scopy * 1.0e6)
            var mps = 2.942 if dropout_case == 1 else 1.931
            # Queue-death guard (see module docstring): the composed pipeline
            # moves ~50 MB and >1.5 GFLOP; sub-0.15 ms totals are impossible.
            if total < 0.15:
                raise Error(
                    t"implausible total {total} ms: the GPU queue died —"
                    t" rerun the benchmark"
                )
            print(
                t"bh={bh} q={q_len} s={kv_len} d={head_dim} dropout={p}"
                t"  total_ms={total}  qk_ms={s1}  softmax_ms={s2}"
                t"  dropout_ms={s3}  pv_ms={s4}  mps_ms={mps}"
                t"  slowdown_vs_mps={total / mps}"
                t"  copy_ms={scopy}  copy_gbs={copy_gbs}"
                t"  out_diff={max_out_diff}  probs_diff={max_probs_diff}"
            )
