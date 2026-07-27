"""Standalone Apple-GPU FP32 SDPA *backward* benchmark for mojo_device.

Replicates the launch sequence `_ScaledDotProductAttentionAutograd.backward`
(torch_mojo_backend/mojo_device/mojo_device_autograd.py, via
`aten_fast.fast_sdpa_backward`) issues for the nanoGPT training shape
(batch 8, heads 6, seq 256, head_dim 64, causal, dropout 0.2, all three
input grads):

    dP = dO @ V^T                     (matmul_ops, causal mode 1: tiles above
                                       the diagonal skipped and unwritten)
    dS = fused causal dropout+softmax+scale backward
                                      (sdpa_dropout_softmax_backward_kernels;
                                       zero band to the 64 boundary)
    dQ = dS @ K                       (matmul_ops, causal mode 2: reduction
                                       cut at the boundary)
    dV = (P * mask * ds)^T @ dO       (sdpa_backward_gemm_kernels: transposed
                                       -A GEMM, dropout fused in the A load,
                                       causal reduction start)
    dK = dS^T @ Q                     (same transposed-A GEMM, unmasked)

No permute copies and no dropout-backward pass remain — the old 8-launch
composition is now only the non-Apple fallback.

Inputs P/mask are produced by the real forward sequence during setup, so the
backward is validated against a naive on-GPU reference with the exact
production probabilities and dropout mask (deterministic check, no
statistics).

    uv run --no-sync mojo build -I torch_mojo_backend/eager_kernels \
        bench_apple_sdpa_bwd.mojo -o /tmp/bench_sdpa_bwd
    MODULAR_DEBUG=device-sync-mode uv run --no-sync /tmp/bench_sdpa_bwd

Pin GPU clocks for A/B decisions:

    uv run python bench_gpu_locked.py Maximum -- env \
        MODULAR_DEBUG=device-sync-mode /tmp/bench_sdpa_bwd

Always use MODULAR_DEBUG=device-sync-mode (flaky async Metal queue; the
implausible-time guard catches dead-queue runs — rerun). Shape overrides:
--bh=48 --q=256 --s=256 --d=64 (equals form only).

torch MPS reference at this shape (training decomposed path, this machine,
fwd+bwd minus fwd): backward ~1.52 ms with dropout 0.2, ~1.14 ms without.
"""

from std.builtin.sort import sort
from std.collections import List, Optional
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext, HostBuffer
from std.math import ceildiv, exp, sqrt
from std.time import perf_counter_ns

from internal_utils import arg_parse

from matmul_ops import _apple8_fat_enqueue, _gemm_enqueue
from native_dropout_kernels import enqueue_native_dropout_f32
from nn_ops import _softmax_rows
from sdpa_backward_gemm_kernels import enqueue_sdpa_ta_gemm_f32
from sdpa_dropout_softmax_backward_kernels import (
    enqueue_sdpa_dropout_softmax_backward_f32,
)

comptime _BLOCK = 256
comptime _SEED = UInt64(0x5EED_5EED)


@__name("bench_sdpa_bwd_fill_hash")
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


# Reference dS: one thread per (bh, qi) row.
#   dPd = dO @ V^T ;  dP = dPd * mask * ds ;  dS = P*(dP - rowsum(dP*P))*scale
@__name("bench_sdpa_bwd_ref_dscores")
def _ref_dscores_kernel(
    ds_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    grad_out: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    bh: Int,
    q_len: Int,
    kv_len: Int,
    head_dim: Int,
    drop_scale: Float32,
    score_scale: Float32,
    use_mask: Int,
):
    var row = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if row >= bh * q_len:
        return
    var b = row // q_len
    var qi = row - b * q_len
    var go_base = (b * q_len + qi) * head_dim
    var kv_base = b * kv_len * head_dim
    var p_base = (b * q_len + qi) * kv_len

    var row_sum = Float32(0)
    for j in range(kv_len):
        var dpd = Float32(0)
        for dd in range(head_dim):
            dpd += grad_out[go_base + dd] * v[kv_base + j * head_dim + dd]
        var dp = dpd
        if use_mask != 0:
            dp = dpd * drop_scale if mask[p_base + j] else Float32(0)
        row_sum += dp * probs[p_base + j]
    for j in range(kv_len):
        var dpd = Float32(0)
        for dd in range(head_dim):
            dpd += grad_out[go_base + dd] * v[kv_base + j * head_dim + dd]
        var dp = dpd
        if use_mask != 0:
            dp = dpd * drop_scale if mask[p_base + j] else Float32(0)
        ds_out[p_base + j] = probs[p_base + j] * (dp - row_sum) * score_scale


# Naive batched C[b] = A[b] @ B[b] with optional logical transposes of either
# operand: one thread per output element (reference only).
@__name("bench_sdpa_bwd_ref_bmm")
def _ref_bmm_kernel(
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    ta: Int,
    tb: Int,
):
    var idx = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if idx >= batch * m * n:
        return
    var bz = idx // (m * n)
    var rest = idx - bz * m * n
    var row = rest // n
    var col = rest - row * n
    var a_base = bz * m * k
    var b_base = bz * k * n
    var acc = Float32(0)
    for kk in range(k):
        var av = (
            a[a_base + kk * m + row] if ta != 0 else a[a_base + row * k + kk]
        )
        var bv = (
            b[b_base + col * k + kk] if tb != 0 else b[b_base + kk * n + col]
        )
        acc += av * bv
    c[idx] = acc


# Element-wise dropout backward reference: Pd = P * mask * ds.
@__name("bench_sdpa_bwd_ref_pdrop")
def _ref_pdrop_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    count: Int,
    drop_scale: Float32,
    use_mask: Int,
):
    var i = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if i >= count:
        return
    var p = probs[i]
    if use_mask != 0:
        p = p * drop_scale if mask[i] else Float32(0)
    out_ptr[i] = p


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


def _check(
    name: StringSlice,
    got: HostBuffer[DType.float32],
    want: HostBuffer[DType.float32],
    count: Int,
    tolerance: Float32,
) raises:
    var max_diff = Float32(0)
    var magnitude = Float32(0)
    for i in range(count):
        var diff = abs(got[i] - want[i])
        if diff > max_diff:
            max_diff = diff
        var mag = abs(want[i])
        if mag > magnitude:
            magnitude = mag
    if magnitude == 0.0:
        raise Error(t"reference {name} is all zeros: kernels not running")
    if max_diff > tolerance:
        raise Error(t"WRONG RESULT for {name}: max diff {max_diff}")
    print(t"  {name}: max diff {max_diff} (ok)")


def _check_causal_band(
    name: StringSlice,
    got: HostBuffer[DType.float32],
    want: HostBuffer[DType.float32],
    bh: Int,
    q_len: Int,
    kv_len: Int,
    tolerance: Float32,
) raises:
    """Compare only the causal-defined region of a (bh, q_len, kv_len)
    matrix: row i is written for columns [0, next 64 multiple of i + 1) and
    left unwritten (never read) beyond."""
    var max_diff = Float32(0)
    var magnitude = Float32(0)
    for b in range(bh):
        for qi in range(q_len):
            var boundary = min(kv_len, qi + 1)
            var padded = min(kv_len, ceildiv(boundary, 64) * 64)
            var base = (b * q_len + qi) * kv_len
            for j in range(padded):
                var diff = abs(got[base + j] - want[base + j])
                if diff > max_diff:
                    max_diff = diff
                var mag = abs(want[base + j])
                if mag > magnitude:
                    magnitude = mag
    if magnitude == 0.0:
        raise Error(t"reference {name} is all zeros: kernels not running")
    if max_diff > tolerance:
        raise Error(t"WRONG RESULT for {name}: max diff {max_diff}")
    print(t"  {name}: max diff {max_diff} (ok, causal region)")


def main() raises:
    var bh = Int(arg_parse("bh", 48))
    var q_len = Int(arg_parse("q", 256))
    var kv_len = Int(arg_parse("s", 256))
    var head_dim = Int(arg_parse("d", 64))
    var warmup = Int(arg_parse("warmup", 10))
    var iterations = Int(arg_parse("iterations", 30))
    var score_scale = Float32(1.0) / sqrt(Float32(head_dim))
    var p = Float64(0.2)
    var drop_scale = Float32(1.0 / (1.0 - p))

    var n_qd = bh * q_len * head_dim
    var n_kvd = bh * kv_len * head_dim
    var n_ps = bh * q_len * kv_len

    with DeviceContext() as ctx:
        var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qd)
        var k_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)
        var v_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)
        var go_buf = ctx.enqueue_create_buffer[DType.float32](n_qd)
        var scores_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var mask_buf = ctx.enqueue_create_buffer[DType.bool](n_ps)
        var scratch_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        # Production-path work buffers.
        var dv_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)
        var dp_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var ds_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var dq_buf = ctx.enqueue_create_buffer[DType.float32](n_qd)
        var dk_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)
        # References.
        var ref_ds_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var ref_pd_buf = ctx.enqueue_create_buffer[DType.float32](n_ps)
        var ref_dq_buf = ctx.enqueue_create_buffer[DType.float32](n_qd)
        var ref_dk_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)
        var ref_dv_buf = ctx.enqueue_create_buffer[DType.float32](n_kvd)

        ctx.enqueue_function[_fill_hash_kernel](
            q_buf.unsafe_ptr().as_unsafe_any_origin(),
            n_qd,
            1,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_fill_hash_kernel](
            k_buf.unsafe_ptr().as_unsafe_any_origin(),
            n_kvd,
            2,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_fill_hash_kernel](
            v_buf.unsafe_ptr().as_unsafe_any_origin(),
            n_kvd,
            3,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_fill_hash_kernel](
            go_buf.unsafe_ptr().as_unsafe_any_origin(),
            n_qd,
            4,
            grid_dim=(1024,),
            block_dim=(_BLOCK,),
        )

        # Produce real P and mask with the production forward stages.
        _gemm_enqueue[DType.float32, True](
            Int(scores_buf.unsafe_ptr()),
            Int(q_buf.unsafe_ptr()),
            Int(k_buf.unsafe_ptr()),
            bh,
            q_len,
            kv_len,
            head_dim,
            q_len * head_dim,
            ctx,
        )
        _softmax_rows[DType.float32](
            Int(probs_buf.unsafe_ptr()),
            Int(scores_buf.unsafe_ptr()),
            bh * q_len,
            kv_len,
            score_scale,
            1,
            q_len,
            ctx,
        )
        enqueue_native_dropout_f32(
            scratch_buf.unsafe_ptr().as_unsafe_any_origin(),
            mask_buf.unsafe_ptr().as_unsafe_any_origin(),
            probs_buf.unsafe_ptr().as_unsafe_any_origin(),
            n_ps,
            p,
            _SEED,
            UInt64(0),
            ctx,
        )
        ctx.synchronize()

        var probs_mut = probs_buf.unsafe_ptr().as_unsafe_any_origin()
        var mask_mut = mask_buf.unsafe_ptr().as_unsafe_any_origin()
        var dp_mut = dp_buf.unsafe_ptr().as_unsafe_any_origin()
        var ds_mut = ds_buf.unsafe_ptr().as_unsafe_any_origin()
        var mask_imm = mask_mut.as_immutable()

        # ---- production backward stages ------------------------------------

        @always_inline
        @parameter
        def _s1_dp() raises:  # dP_drop = dO @ V^T (causal mode 1)
            _apple8_fat_enqueue[True, 64, 64, 2, 2, False, 1](
                Int(dp_buf.unsafe_ptr()),
                Int(go_buf.unsafe_ptr()),
                Int(v_buf.unsafe_ptr()),
                bh,
                q_len,
                kv_len,
                head_dim,
                q_len * head_dim,
                ctx,
            )

        @always_inline
        @parameter
        def _s2_ds() raises:  # dS = fused causal dropout+softmax+scale bwd
            enqueue_sdpa_dropout_softmax_backward_f32(
                ds_mut,
                probs_mut,
                dp_mut,
                Optional[UnsafePointer[Scalar[DType.bool], MutAnyOrigin]](
                    mask_mut
                ),
                bh * q_len,
                kv_len,
                True,
                Float64(drop_scale),
                Float64(score_scale),
                ctx,
                causal=True,
                q_len=q_len,
            )

        @always_inline
        @parameter
        def _s3_dq() raises:  # dQ = dS @ K (causal mode 2)
            _apple8_fat_enqueue[False, 64, 64, 2, 2, False, 2](
                Int(dq_buf.unsafe_ptr()),
                Int(ds_buf.unsafe_ptr()),
                Int(k_buf.unsafe_ptr()),
                bh,
                q_len,
                head_dim,
                kv_len,
                q_len * kv_len,
                ctx,
            )

        @always_inline
        @parameter
        def _s4_dv() raises:  # dV = (P * mask * ds)^T @ dO
            enqueue_sdpa_ta_gemm_f32(
                dv_buf.unsafe_ptr().as_unsafe_any_origin(),
                probs_mut.as_immutable(),
                go_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                Optional[UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin]](
                    mask_imm
                ),
                bh,
                kv_len,
                head_dim,
                q_len,
                True,
                Float64(drop_scale),
                ctx,
            )

        @always_inline
        @parameter
        def _s5_dk() raises:  # dK = dS^T @ Q
            enqueue_sdpa_ta_gemm_f32(
                dk_buf.unsafe_ptr().as_unsafe_any_origin(),
                ds_mut.as_immutable(),
                q_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
                Optional[UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin]](
                    None
                ),
                bh,
                kv_len,
                head_dim,
                q_len,
                True,
                Float64(1.0),
                ctx,
            )

        @always_inline
        @parameter
        def _full() raises:
            _s1_dp()
            _s2_ds()
            _s3_dq()
            _s4_dv()
            _s5_dk()

        # ---- correctness ----------------------------------------------------
        ctx.synchronize()
        _full()
        ctx.synchronize()

        ctx.enqueue_function[_ref_dscores_kernel](
            ref_ds_buf.unsafe_ptr().as_unsafe_any_origin(),
            probs_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            go_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            mask_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            bh,
            q_len,
            kv_len,
            head_dim,
            drop_scale,
            score_scale,
            1,
            grid_dim=(ceildiv(bh * q_len, _BLOCK),),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_ref_pdrop_kernel](
            ref_pd_buf.unsafe_ptr().as_unsafe_any_origin(),
            probs_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            mask_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            n_ps,
            drop_scale,
            1,
            grid_dim=(ceildiv(n_ps, _BLOCK),),
            block_dim=(_BLOCK,),
        )
        # ref dQ = ref_dS @ K ; ref dK = ref_dS^T @ Q ; ref dV = Pd^T @ dO
        ctx.enqueue_function[_ref_bmm_kernel](
            ref_dq_buf.unsafe_ptr().as_unsafe_any_origin(),
            ref_ds_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            bh,
            q_len,
            head_dim,
            kv_len,
            0,
            0,
            grid_dim=(ceildiv(n_qd, _BLOCK),),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_ref_bmm_kernel](
            ref_dk_buf.unsafe_ptr().as_unsafe_any_origin(),
            ref_ds_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            bh,
            kv_len,
            head_dim,
            q_len,
            1,
            0,
            grid_dim=(ceildiv(n_kvd, _BLOCK),),
            block_dim=(_BLOCK,),
        )
        ctx.enqueue_function[_ref_bmm_kernel](
            ref_dv_buf.unsafe_ptr().as_unsafe_any_origin(),
            ref_pd_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            go_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            bh,
            kv_len,
            head_dim,
            q_len,
            1,
            0,
            grid_dim=(ceildiv(n_kvd, _BLOCK),),
            block_dim=(_BLOCK,),
        )

        var host_got = ctx.enqueue_create_host_buffer[DType.float32](n_ps)
        var host_want = ctx.enqueue_create_host_buffer[DType.float32](n_ps)
        ctx.enqueue_copy(
            host_got.create_sub_buffer[DType.float32](0, n_ps), ds_buf
        )
        ctx.enqueue_copy(
            host_want.create_sub_buffer[DType.float32](0, n_ps), ref_ds_buf
        )
        ctx.synchronize()
        _check_causal_band("dS", host_got, host_want, bh, q_len, kv_len, 2e-4)

        ctx.enqueue_copy(
            host_got.create_sub_buffer[DType.float32](0, n_qd), dq_buf
        )
        ctx.enqueue_copy(
            host_want.create_sub_buffer[DType.float32](0, n_qd), ref_dq_buf
        )
        ctx.synchronize()
        _check("dQ", host_got, host_want, n_qd, 2e-3)

        ctx.enqueue_copy(
            host_got.create_sub_buffer[DType.float32](0, n_kvd), dk_buf
        )
        ctx.enqueue_copy(
            host_want.create_sub_buffer[DType.float32](0, n_kvd), ref_dk_buf
        )
        ctx.synchronize()
        _check("dK", host_got, host_want, n_kvd, 2e-3)

        ctx.enqueue_copy(
            host_got.create_sub_buffer[DType.float32](0, n_kvd), dv_buf
        )
        ctx.enqueue_copy(
            host_want.create_sub_buffer[DType.float32](0, n_kvd), ref_dv_buf
        )
        ctx.synchronize()
        _check("dV", host_got, host_want, n_kvd, 2e-3)

        # ---- timing ---------------------------------------------------------
        # Burn-in: hold the GPU busy long enough for the clock governor to
        # settle at its sustained state before any measurement (per-stage
        # medians otherwise drift 2x between cold and hot runs).
        for _ in range(200):
            _full()
        ctx.synchronize()
        var total = _time_stage[_full](warmup, iterations, ctx)
        var t1 = _time_stage[_s1_dp](warmup, iterations, ctx)
        var t2 = _time_stage[_s2_ds](warmup, iterations, ctx)
        var t3 = _time_stage[_s3_dq](warmup, iterations, ctx)
        var t4 = _time_stage[_s4_dv](warmup, iterations, ctx)
        var t5 = _time_stage[_s5_dk](warmup, iterations, ctx)
        var totalb = _time_stage[_full](warmup, iterations, ctx)
        print(t"  recheck: total={totalb}")
        if total < 0.15:
            raise Error(
                t"implausible total {total} ms: the GPU queue died — rerun"
            )
        print(
            t"bh={bh} q={q_len} s={kv_len} d={head_dim} dropout=0.2"
            t"  total_ms={total}  mps_bwd_ms=1.52"
            t"  slowdown_vs_mps={total / 1.52}"
        )
        print(
            t"  stages: dP={t1}  dS_fused={t2}  dQ={t3}  dV_ta={t4}  dK_ta={t5}"
        )
