"""Pure-Mojo harness for the fused flash-attention forward. THE ACCEPTANCE GATE.

Times `flash_attention_fwd_kernels.enqueue_flash_attention_fwd` against
PyTorch-ROCm's own fused kernel and checks it against a reference that shares no
code with it. Builds in about two seconds, so the edit/measure loop never pays
the ~10 minute rebuild of the eager CPython extensions.

    uv run --no-sync mojo build harness/nanogpt_train/bench_flash_attention.mojo \
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_fa
    /tmp/bench_fa

    /tmp/bench_fa --case=nanogpt --iterations=200     # one case, longer run
    /tmp/bench_fa --check=0                           # timing only

Flags parse ONLY as `--name=value`. A space-separated flag is silently ignored
and you get the default, so read the values echoed in the header.

THE REFERENCE. `_reference_kernel` is one thread per query row doing a textbook
three-pass softmax -- find the max, sum the exponentials, then weight the values
-- in FP32, with no tiling, no online rescaling and no shared memory. It is
deliberately nothing like a flash-attention kernel: it shares no tile geometry,
no reduction and no sentinel with the code under test, so a defect common to both
is implausible rather than merely unlikely. It is slow on purpose.

THE GATE, per case:
  * every output element compared, on the device, against that reference;
  * NaN and Inf counted separately, because a poisoned row is the failure mode
    this kernel family actually has (a -inf softmax sentinel makes
    `exp(-inf - -inf)` = nan, which has shipped twice in this repository);
  * max absolute error against `4 * eps(dtype)`. The output is a convex
    combination of V rows, so its error does not grow with `seq_kv` and no
    sqrt factor belongs in the bound.

The thresholds are mutation-tested rather than guessed. With the correct kernel
the error is one bf16 ULP of the output, 0.00195; five deliberate defects give:

    ignore the causal mask            FAIL
    causal off by one (no diagonal)   FAIL, and 64 NaN
    drop the last KV tile             FAIL, and 16384 NaN
    skip the online rescale           0.0721, FAIL (2.3x over the bound)
    seed the running max with -inf    inert in THIS structure, see below

That fourth one is why the Q fill has a gain: with unit-scale operands the
softmax is nearly flat, the output degenerates into an average over all keys,
and skipping the rescale moved the result by only 0.0136 -- inside a tolerance
that a correct kernel also needs. A gain of 8 concentrates the distribution, and
the same defect then moves it by 0.0721, a 37x separation from the correct error.

The -inf sentinel is inert here only because `while kv_start < limit` guarantees
the first tile of every row has at least one live key, so a fully masked tile
never reaches the reduction. A rewrite that iterates a fixed tile count, or
splits KV across warps, can reintroduce it -- that defect has shipped twice in
this repository -- so the NaN and Inf counters stay.

A case that fails any check has no meaningful time.
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import barrier, block_idx, block_dim, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, exp, sqrt
from std.time import perf_counter_ns

from internal_utils import arg_parse
from flash_attention_fwd_kernels import (
    RowStrides,
    dense_strides,
    enqueue_flash_attention_fwd,
)

comptime FILL_BLOCK = 256
comptime CHECK_BLOCKS = 512
comptime REF_THREADS = 128


struct FaCase(ImplicitlyCopyable, Movable):
    """One attention shape, with the PyTorch-ROCm time it is measured against."""

    var label: String
    var batch: Int
    var heads: Int
    var seq_q: Int
    var seq_kv: Int
    var head_dim: Int
    var causal: Bool
    var calls_per_step: Int
    var rocm_us: Float64

    def __init__(
        out self,
        var label: String,
        batch: Int,
        heads: Int,
        seq_q: Int,
        seq_kv: Int,
        head_dim: Int,
        causal: Bool,
        calls_per_step: Int,
        rocm_us: Float64,
    ):
        self.label = label^
        self.batch = batch
        self.heads = heads
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.head_dim = head_dim
        self.causal = causal
        self.calls_per_step = calls_per_step
        self.rocm_us = rocm_us


def cases() -> List[FaCase]:
    """The frozen case list.

    `nanogpt` is the acceptance case: nanoGPT's GPT-2 124M attention at batch 48,
    block 1024, twelve layers per step, measured on PyTorch-ROCm at 679.80 us per
    layer (`scripts/rocm_attention_reference.py`, 25 warmups / 100 synchronized
    iterations). The rest exist so a kernel cannot pass by being right only at
    tile-aligned extents: ragged sequence lengths, a single batch, a single head,
    a head_dim that is not 64, and a non-causal case.
    """
    var out = List[FaCase]()
    out.append(FaCase("nanogpt", 48, 12, 1024, 1024, 64, True, 12, 679.80))
    out.append(FaCase("seq1000", 8, 12, 1000, 1000, 64, True, 1, 0.0))
    out.append(FaCase("seq1025", 8, 12, 1025, 1025, 64, True, 1, 0.0))
    out.append(FaCase("batch1", 1, 12, 1024, 1024, 64, True, 1, 0.0))
    out.append(FaCase("head1", 1, 1, 512, 512, 64, True, 1, 0.0))
    out.append(FaCase("hd128", 4, 8, 512, 512, 128, True, 1, 0.0))
    out.append(FaCase("hd96_ragged", 2, 5, 300, 300, 96, True, 1, 0.0))
    out.append(FaCase("noncausal", 4, 8, 512, 512, 64, False, 1, 0.0))
    out.append(FaCase("cross_kv_longer", 2, 4, 256, 1024, 64, True, 1, 0.0))
    out.append(FaCase("tiny", 1, 1, 3, 3, 8, True, 1, 0.0))
    return out^


@__name("fa_fill_deterministic")
def _fill_deterministic(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    count: Int,
    seed: Int,
    gain: Float32,
):
    """A bounded, non-repeating fill, scaled by `gain`.

    Values vary along every axis, so a swapped index or a stale tile changes the
    result. `gain` on Q widens the score distribution: with unit-scale operands
    and head_dim 64 the scores have a standard deviation near 0.6, the softmax is
    almost flat, and the output degenerates into an average over all keys -- which
    a wrong causal mask barely perturbs, because averages concentrate. A gain of 8
    puts the score spread near 4.6, so the distribution has an effective support
    of tens of keys and the output records WHICH keys were attended. Both the
    kernel and the reference subtract a row max before exponentiating, so the
    wider range costs no accuracy on either side.
    """
    var i = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    while i < count:
        var h = (i * 2654435761 + seed * 40503) % 65521
        ptr[i] = (Float32(h) * (2.0 / 65521.0) - 1.0) * gain
        i += stride


@__name(t"fa_cast_from_f32_{dtype}")
def _cast_from_f32[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    count: Int,
):
    var i = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    while i < count:
        dst[i] = src[i].cast[dtype]()
        i += stride


@__name(t"fa_reference_{dtype}")
def _reference_kernel[
    dtype: DType
](
    ref_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """One thread per query row, three passes, FP32, no tiling and no rescaling.

    Textbook softmax: max, then sum of exponentials, then the weighted sum. This
    is the independent oracle; it must not be optimized, and it must not borrow
    anything from the kernel under test.
    """
    var row = Int(block_idx.x) * REF_THREADS + Int(thread_idx.x)
    var rows = batch * heads * seq_q
    if row >= rows:
        return

    var qi = row % seq_q
    var bh = row // seq_q
    var q_base = bh * seq_q * head_dim + qi * head_dim
    var kv_base = bh * seq_kv * head_dim

    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + 1)

    # Pass 1: the maximum score.
    var m = Float32.MIN_FINITE
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_base + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        var s = dot * scale
        if s > m:
            m = s

    # Pass 2: the normalizer.
    var denom = Float32(0.0)
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_base + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        denom += exp(dot * scale - m)

    # Pass 3: the weighted values.
    for e in range(head_dim):
        ref_out[q_base + e] = 0.0
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_base + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        var p = exp(dot * scale - m) / denom
        for e in range(head_dim):
            ref_out[q_base + e] += (
                p * value[kv_base + j * head_dim + e].cast[DType.float32]()
            )


@__name(t"fa_compare_{dtype}")
def _compare_kernel[
    dtype: DType
](
    worst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    nan_count: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    inf_count: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    got: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    want: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    count: Int,
):
    """Per-thread worst absolute error plus NaN and Inf tallies.

    NaN is counted through a self-inequality test rather than a comparison, so a
    NaN cannot slip past a `>` the way it does through `if d > worst`. Inf is
    counted separately because a saturated exponential and a poisoned reduction
    are different defects.
    """
    var slot = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var i = slot
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var local = Float32(0.0)
    var nans = Int32(0)
    var infs = Int32(0)
    while i < count:
        var g = got[i].cast[DType.float32]()
        if g != g:
            nans += 1
        elif (g > Float32.MAX_FINITE) or (g < Float32.MIN_FINITE):
            infs += 1
        else:
            var d = g - want[i]
            if d < 0.0:
                d = -d
            if d > local:
                local = d
        i += stride
    worst[slot] = local
    nan_count[slot] = nans
    inf_count[slot] = infs


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


def _pad(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out += " "
    return out


def _fixed(value: Float64, decimals: Int) -> String:
    var scale = 1.0
    for _ in range(decimals):
        scale *= 10.0
    return String(Float64(Int(value * scale + 0.5)) / scale)


struct FaResult(ImplicitlyCopyable, Movable):
    var median_us: Float64
    var p10_us: Float64
    var p90_us: Float64
    var max_err: Float64
    var tolerance: Float64
    var nans: Int
    var infs: Int
    var checked: Bool

    def __init__(
        out self,
        median_us: Float64,
        p10_us: Float64,
        p90_us: Float64,
        max_err: Float64,
        tolerance: Float64,
        nans: Int,
        infs: Int,
        checked: Bool,
    ):
        self.median_us = median_us
        self.p10_us = p10_us
        self.p90_us = p90_us
        self.max_err = max_err
        self.tolerance = tolerance
        self.nans = nans
        self.infs = infs
        self.checked = checked

    def passed(self) -> Bool:
        if not self.checked:
            return True
        return (
            self.nans == 0 and self.infs == 0 and self.max_err <= self.tolerance
        )


def run_case[
    dtype: DType
](
    target: FaCase,
    warmup: Int,
    iterations: Int,
    check: Bool,
    ctx: DeviceContext,
) raises -> FaResult:
    var batch = target.batch
    var heads = target.heads
    var seq_q = target.seq_q
    var seq_kv = target.seq_kv
    var head_dim = target.head_dim
    var q_count = batch * heads * seq_q * head_dim
    var kv_count = batch * heads * seq_kv * head_dim
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var q_st = dense_strides(heads, seq_q, head_dim)
    var k_st = dense_strides(heads, seq_kv, head_dim)
    var v_st = dense_strides(heads, seq_kv, head_dim)

    var q_f32 = ctx.enqueue_create_buffer[DType.float32](q_count)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](kv_count)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](kv_count)
    var q = ctx.enqueue_create_buffer[dtype](q_count)
    var k = ctx.enqueue_create_buffer[dtype](kv_count)
    var v = ctx.enqueue_create_buffer[dtype](kv_count)
    var out_buf = ctx.enqueue_create_buffer[dtype](q_count)
    var ref_buf = ctx.enqueue_create_buffer[DType.float32](
        q_count if check else 1
    )

    # Fill in FP32 and cast down, so the kernel and the reference read bitwise
    # identical operands and the comparison isolates the algorithm.
    ctx.enqueue_function[_fill_deterministic](
        q_f32.unsafe_ptr().as_unsafe_any_origin(),
        q_count,
        13,
        Float32(8.0),
        grid_dim=(max(1, min(ceildiv(q_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_fill_deterministic](
        k_f32.unsafe_ptr().as_unsafe_any_origin(),
        kv_count,
        7932,
        Float32(1.0),
        grid_dim=(max(1, min(ceildiv(kv_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_fill_deterministic](
        v_f32.unsafe_ptr().as_unsafe_any_origin(),
        kv_count,
        15851,
        Float32(1.0),
        grid_dim=(max(1, min(ceildiv(kv_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast_from_f32[dtype]](
        q.unsafe_ptr().as_unsafe_any_origin(),
        q_f32.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        q_count,
        grid_dim=(max(1, min(ceildiv(q_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast_from_f32[dtype]](
        k.unsafe_ptr().as_unsafe_any_origin(),
        k_f32.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        kv_count,
        grid_dim=(max(1, min(ceildiv(kv_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast_from_f32[dtype]](
        v.unsafe_ptr().as_unsafe_any_origin(),
        v_f32.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        kv_count,
        grid_dim=(max(1, min(ceildiv(kv_count, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )

    # The forward also emits the per-row log-sum-exp the backward consumes;
    # this harness does not check it (bench_flash_attention_bwd.mojo does, by
    # feeding its own L to the backward), it just supplies the buffer.
    var lse_buf = ctx.enqueue_create_buffer[DType.float32](
        batch * heads * seq_q
    )

    @always_inline
    @parameter
    def _launch() raises:
        enqueue_flash_attention_fwd[dtype](
            out_buf.unsafe_ptr().as_unsafe_any_origin(),
            lse_buf.unsafe_ptr().as_unsafe_any_origin(),
            q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q_st,
            k_st,
            v_st,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            scale,
            target.causal,
            ctx,
        )

    for _ in range(warmup):
        _launch()
    ctx.synchronize()

    var samples = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _launch()
        ctx.synchronize()
        samples.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(samples)

    var max_err = Float64(0.0)
    var nans = 0
    var infs = 0
    var tolerance = Float64(0.0)
    if check:
        var rows = batch * heads * seq_q
        ctx.enqueue_function[_reference_kernel[dtype]](
            ref_buf.unsafe_ptr().as_unsafe_any_origin(),
            q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            scale,
            1 if target.causal else 0,
            grid_dim=(ceildiv(rows, REF_THREADS),),
            block_dim=(REF_THREADS,),
        )
        var slots = CHECK_BLOCKS * FILL_BLOCK
        var worst = ctx.enqueue_create_buffer[DType.float32](slots)
        var nan_buf = ctx.enqueue_create_buffer[DType.int32](slots)
        var inf_buf = ctx.enqueue_create_buffer[DType.int32](slots)
        ctx.enqueue_function[_compare_kernel[dtype]](
            worst.unsafe_ptr().as_unsafe_any_origin(),
            nan_buf.unsafe_ptr().as_unsafe_any_origin(),
            inf_buf.unsafe_ptr().as_unsafe_any_origin(),
            out_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            ref_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q_count,
            grid_dim=(CHECK_BLOCKS,),
            block_dim=(FILL_BLOCK,),
        )
        var h_worst = ctx.enqueue_create_host_buffer[DType.float32](slots)
        var h_nan = ctx.enqueue_create_host_buffer[DType.int32](slots)
        var h_inf = ctx.enqueue_create_host_buffer[DType.int32](slots)
        ctx.enqueue_copy(h_worst, worst)
        ctx.enqueue_copy(h_nan, nan_buf)
        ctx.enqueue_copy(h_inf, inf_buf)
        ctx.synchronize()
        for i in range(slots):
            var value = Float64(h_worst[i])
            if value > max_err:
                max_err = value
            nans += Int(h_nan[i])
            infs += Int(h_inf[i])
        _ = worst^
        _ = nan_buf^
        _ = inf_buf^

        # The output is a convex combination of V rows, so |output| <= max|V|
        # and the error does not grow with seq_kv. A few eps covers a different
        # summation order; anything structural -- a wrong mask, a dropped KV
        # tile, a stale rescale -- moves the result far further, because the
        # score distribution is deliberately concentrated (see `_fill`).
        comptime eps = 0.0078125 if dtype == DType.bfloat16 else 1.1920929e-7
        tolerance = 4.0 * Float64(eps)

    var result = FaResult(
        _percentile(samples, 50),
        _percentile(samples, 10),
        _percentile(samples, 90),
        max_err,
        tolerance,
        nans,
        infs,
        check,
    )
    _ = q_f32^
    _ = k_f32^
    _ = v_f32^
    _ = q^
    _ = k^
    _ = v^
    _ = out_buf^
    _ = ref_buf^
    return result


def main() raises:
    var only = String(arg_parse("case", "all"))
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var check = Int(arg_parse("check", 1)) != 0
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    print(
        "flash-attention forward, bfloat16 | warmup=",
        warmup,
        " iterations=",
        iterations,
        " check=",
        Int(check),
    )
    print(
        "case             b   h   sq   skv  hd c   mojo_us    rocm_us  ratio"
        "     max_err  tol        nan  inf  status"
    )

    var mojo_step_us = Float64(0.0)
    var rocm_step_us = Float64(0.0)
    var failed = 0
    var matched = 0
    with DeviceContext() as ctx:
        for target in cases():
            if only != "all" and target.label != only:
                continue
            matched += 1
            var r = run_case[DType.bfloat16](
                target, warmup, iterations, check, ctx
            )
            if target.rocm_us > 0.0:
                mojo_step_us += r.median_us * Float64(target.calls_per_step)
                rocm_step_us += target.rocm_us * Float64(target.calls_per_step)
            var ratio = "-"
            if target.rocm_us > 0.0:
                ratio = _fixed(r.median_us / target.rocm_us, 3)
            print(
                _pad(target.label, 16),
                _pad(String(target.batch), 3),
                _pad(String(target.heads), 3),
                _pad(String(target.seq_q), 4),
                _pad(String(target.seq_kv), 5),
                _pad(String(target.head_dim), 3),
                Int(target.causal),
                _pad(_fixed(r.median_us, 2), 10),
                _pad(_fixed(target.rocm_us, 2), 10),
                _pad(ratio, 7),
                _pad(_fixed(r.max_err, 6), 11),
                _pad(_fixed(r.tolerance, 6), 10),
                _pad(String(r.nans), 4),
                _pad(String(r.infs), 4),
                "pass" if r.passed() else "FAIL",
            )
            if not r.passed():
                failed += 1
    if matched == 0:
        raise Error("no case matched --case=", only)

    if rocm_step_us > 0.0:
        print()
        print(
            "nanoGPT per-step attention forward: this kernel",
            _fixed(mojo_step_us / 1000.0, 3),
            "ms, PyTorch-ROCm",
            _fixed(rocm_step_us / 1000.0, 3),
            "ms, ratio",
            _fixed(mojo_step_us / rocm_step_us, 3),
        )
        # Two bars, both real. The eager device already computes this forward by
        # a causal decomposition (batched GEMM, row softmax, batched GEMM) that
        # measures 29.854 ms/step in the production profile, 3.29x ROCm. A fused
        # kernel is only worth wiring in once it beats that; ROCm's 8.158 is the
        # target beyond it.
        print(
            "  bar to beat (current decomposition): 29.854 ms/step -> this"
            " kernel is",
            _fixed(mojo_step_us / 1000.0 / 29.854, 3),
            "x that",
        )
        print(
            "  target (PyTorch-ROCm fused):          8.158 ms/step -> ratio",
            _fixed(mojo_step_us / rocm_step_us, 3),
        )
    if failed == 0:
        print("correctness: all cases pass")
        return
    print(
        "correctness: ", failed, "case(s) FAILED - their timings mean nothing"
    )
    raise Error("flash attention correctness failure in ", failed, " case(s)")
