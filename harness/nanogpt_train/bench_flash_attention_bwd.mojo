"""Pure-Mojo harness for the fused flash-attention backward. THE ACCEPTANCE GATE.

Times `flash_attention_bwd_kernels.enqueue_flash_attention_bwd` against
PyTorch-ROCm's own fused backward and checks all three gradients against a
reference that shares no structure with it. Builds in about two seconds.

    uv run --no-sync mojo build harness/nanogpt_train/bench_flash_attention_bwd.mojo \
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_fa_bwd
    /tmp/bench_fa_bwd

    /tmp/bench_fa_bwd --case=head1 --iterations=200
    /tmp/bench_fa_bwd --check=0

Flags parse ONLY as `--name=value`; a space-separated flag is silently ignored
and you get the default, so read the values echoed in the header.

## Inputs

Q, K, V and dO come from the same deterministic fill the forward harness uses,
with a gain on Q so the score distribution is concentrated rather than nearly
flat -- a flat softmax makes every gradient an average, which hides a wrong mask.
O and the row log-sum-exp L are then produced by `_setup_forward`, a plain
three-pass FP32 kernel, so the backward under test receives exactly the
quantities the real forward would hand it.

## The reference

`_ref_dq` and `_ref_dkv` are one thread per output row, fully sequential, FP32,
no tiling and no shared memory. They recompute S and P from Q, K and L directly.
The baseline kernel is block-per-row with block reductions, so the two differ in
decomposition, in reduction, and in which axis is parallel -- a defect common to
both is implausible rather than merely unlikely.

## The gate, per case, per gradient

Every element of dQ, dK and dV is compared on the device. NaN and Inf are
counted separately, because a poisoned row is this kernel family's real failure
mode. The tolerance is per tensor and relative to that tensor's own magnitude,
`8 * eps * max|ref|`, because dQ, dK and dV do not share a scale: dV is a convex
combination of dO rows while dQ carries the `scale` factor and a difference of
two dot products.

A case that fails any of the nine checks has no meaningful time.
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, exp, log, sqrt
from std.time import perf_counter_ns

from internal_utils import arg_parse
from flash_attention_bwd_kernels import enqueue_flash_attention_bwd

comptime FILL_BLOCK = 256
comptime CHECK_BLOCKS = 512
comptime REF_THREADS = 128


struct BwdCase(ImplicitlyCopyable, Movable):
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


def cases() -> List[BwdCase]:
    """The frozen case list.

    `nanogpt` is the acceptance case: nanoGPT's GPT-2 124M attention backward at
    batch 48, block 1024, twelve layers per step. PyTorch-ROCm's fused
    `bwd_kernel_fuse` measures 2413.83 us per layer, 28.966 ms/step
    (`scripts/rocm_attention_reference.py`, 25 warmups / 100 iterations). The
    rest exist so a kernel cannot pass by being right only at tile-aligned
    extents.
    """
    var out = List[BwdCase]()
    out.append(BwdCase("nanogpt", 48, 12, 1024, 1024, 64, True, 12, 2413.83))
    out.append(BwdCase("seq1000", 8, 12, 1000, 1000, 64, True, 1, 0.0))
    out.append(BwdCase("seq1025", 8, 12, 1025, 1025, 64, True, 1, 0.0))
    out.append(BwdCase("batch1", 1, 12, 1024, 1024, 64, True, 1, 0.0))
    out.append(BwdCase("head1", 1, 1, 512, 512, 64, True, 1, 0.0))
    out.append(BwdCase("hd128", 4, 8, 512, 512, 128, True, 1, 0.0))
    out.append(BwdCase("hd96_ragged", 2, 5, 300, 300, 96, True, 1, 0.0))
    out.append(BwdCase("noncausal", 4, 8, 512, 512, 64, False, 1, 0.0))
    out.append(BwdCase("cross_kv_longer", 2, 4, 256, 1024, 64, True, 1, 0.0))
    out.append(BwdCase("tiny", 1, 1, 3, 3, 8, True, 1, 0.0))
    return out^


@__name("fab_fill")
def _fill(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    count: Int,
    seed: Int,
    gain: Float32,
):
    var i = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    while i < count:
        var h = (i * 2654435761 + seed * 40503) % 65521
        ptr[i] = (Float32(h) * (2.0 / 65521.0) - 1.0) * gain
        i += stride


@__name(t"fab_cast_{dtype}")
def _cast[
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


@__name(t"fab_setup_forward_{dtype}")
def _setup_forward[
    dtype: DType
](
    out_fwd: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    """Produce the O and L a real forward would hand the backward.

    One thread per query row, three passes, FP32. This is setup, not the thing
    under test, so it is written for obviousness.
    """
    var row = Int(block_idx.x) * REF_THREADS + Int(thread_idx.x)
    if row >= batch * heads * seq_q:
        return
    var qi = row % seq_q
    var bh = row // seq_q
    var q_row = bh * seq_q * head_dim + qi * head_dim
    var kv_base = bh * seq_kv * head_dim
    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + 1)

    var m = Float32.MIN_FINITE
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_row + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        var s = dot * scale
        if s > m:
            m = s
    var denom = Float32(0.0)
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_row + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        denom += exp(dot * scale - m)
    lse[bh * seq_q + qi] = m + log(denom)
    for e in range(head_dim):
        out_fwd[q_row + e] = Scalar[dtype](0)
    for j in range(limit):
        var dot = Float32(0.0)
        for e in range(head_dim):
            dot += (
                query[q_row + e].cast[DType.float32]()
                * key[kv_base + j * head_dim + e].cast[DType.float32]()
            )
        var p = exp(dot * scale - m) / denom
        for e in range(head_dim):
            out_fwd[q_row + e] = (
                out_fwd[q_row + e].cast[DType.float32]()
                + p * value[kv_base + j * head_dim + e].cast[DType.float32]()
            ).cast[dtype]()


@__name(t"fab_ref_dq_{dtype}")
def _ref_dq[
    dtype: DType
](
    ref_dq: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """dQ, one thread per query row, sequential, FP32. The oracle."""
    var row = Int(block_idx.x) * REF_THREADS + Int(thread_idx.x)
    if row >= batch * heads * seq_q:
        return
    var qi = row % seq_q
    var bh = row // seq_q
    var q_row = bh * seq_q * head_dim + qi * head_dim
    var kv_base = bh * seq_kv * head_dim
    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + 1)

    var row_d = Float32(0.0)
    for e in range(head_dim):
        row_d += (
            grad_output[q_row + e].cast[DType.float32]()
            * out_fwd[q_row + e].cast[DType.float32]()
        )
    for e in range(head_dim):
        ref_dq[q_row + e] = 0.0
    var l = lse[bh * seq_q + qi]
    for j in range(limit):
        var krow = kv_base + j * head_dim
        var s = Float32(0.0)
        var dp = Float32(0.0)
        for e in range(head_dim):
            s += (
                query[q_row + e].cast[DType.float32]()
                * key[krow + e].cast[DType.float32]()
            )
            dp += (
                grad_output[q_row + e].cast[DType.float32]()
                * value[krow + e].cast[DType.float32]()
            )
        var p = exp(s * scale - l)
        var ds = p * (dp - row_d)
        for e in range(head_dim):
            ref_dq[q_row + e] += (
                ds * key[krow + e].cast[DType.float32]() * scale
            )


@__name(t"fab_ref_dkv_{dtype}")
def _ref_dkv[
    dtype: DType
](
    ref_dk: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ref_dv: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """dK and dV, one thread per key row, sequential, FP32. The oracle."""
    var row = Int(block_idx.x) * REF_THREADS + Int(thread_idx.x)
    if row >= batch * heads * seq_kv:
        return
    var j = row % seq_kv
    var bh = row // seq_kv
    var kv_row = bh * seq_kv * head_dim + j * head_dim
    var q_base = bh * seq_q * head_dim

    for e in range(head_dim):
        ref_dk[kv_row + e] = 0.0
        ref_dv[kv_row + e] = 0.0
    var first_q = 0
    if causal != 0:
        first_q = j

    for qi in range(first_q, seq_q):
        var q_row = q_base + qi * head_dim
        var s = Float32(0.0)
        var dp = Float32(0.0)
        var row_d = Float32(0.0)
        for e in range(head_dim):
            var dov = grad_output[q_row + e].cast[DType.float32]()
            s += (
                query[q_row + e].cast[DType.float32]()
                * key[kv_row + e].cast[DType.float32]()
            )
            dp += dov * value[kv_row + e].cast[DType.float32]()
            row_d += dov * out_fwd[q_row + e].cast[DType.float32]()
        var p = exp(s * scale - lse[bh * seq_q + qi])
        var ds = p * (dp - row_d)
        for e in range(head_dim):
            ref_dv[kv_row + e] += (
                p * grad_output[q_row + e].cast[DType.float32]()
            )
            ref_dk[kv_row + e] += (
                ds * query[q_row + e].cast[DType.float32]() * scale
            )


@__name(t"fab_compare_{dtype}")
def _compare[
    dtype: DType
](
    worst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    refmax: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    nan_count: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    inf_count: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    got: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    want: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    count: Int,
):
    """Worst absolute error, plus the reference's own magnitude for scaling.

    NaN is detected by self-inequality rather than a comparison, so it cannot
    slip through a `>` the way it does in `if d > worst`.
    """
    var slot = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var i = slot
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var local = Float32(0.0)
    var localmax = Float32(0.0)
    var nans = Int32(0)
    var infs = Int32(0)
    while i < count:
        var g = got[i].cast[DType.float32]()
        var w = want[i]
        var aw = w if w >= 0.0 else -w
        if aw > localmax:
            localmax = aw
        if g != g:
            nans += 1
        elif (g > Float32.MAX_FINITE) or (g < Float32.MIN_FINITE):
            infs += 1
        else:
            var d = g - w
            if d < 0.0:
                d = -d
            if d > local:
                local = d
        i += stride
    worst[slot] = local
    refmax[slot] = localmax
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


struct GradCheck(ImplicitlyCopyable, Movable):
    var max_err: Float64
    var tolerance: Float64
    var nans: Int
    var infs: Int

    def __init__(
        out self, max_err: Float64, tolerance: Float64, nans: Int, infs: Int
    ):
        self.max_err = max_err
        self.tolerance = tolerance
        self.nans = nans
        self.infs = infs

    def passed(self) -> Bool:
        return (
            self.nans == 0 and self.infs == 0 and self.max_err <= self.tolerance
        )


def _check_one[
    dtype: DType
](
    got: DeviceBuffer[dtype],
    want: DeviceBuffer[DType.float32],
    count: Int,
    ctx: DeviceContext,
) raises -> GradCheck:
    var slots = CHECK_BLOCKS * FILL_BLOCK
    var worst = ctx.enqueue_create_buffer[DType.float32](slots)
    var refmax = ctx.enqueue_create_buffer[DType.float32](slots)
    var nan_buf = ctx.enqueue_create_buffer[DType.int32](slots)
    var inf_buf = ctx.enqueue_create_buffer[DType.int32](slots)
    ctx.enqueue_function[_compare[dtype]](
        worst.unsafe_ptr().as_unsafe_any_origin(),
        refmax.unsafe_ptr().as_unsafe_any_origin(),
        nan_buf.unsafe_ptr().as_unsafe_any_origin(),
        inf_buf.unsafe_ptr().as_unsafe_any_origin(),
        got.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        want.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        count,
        grid_dim=(CHECK_BLOCKS,),
        block_dim=(FILL_BLOCK,),
    )
    var h_worst = ctx.enqueue_create_host_buffer[DType.float32](slots)
    var h_refmax = ctx.enqueue_create_host_buffer[DType.float32](slots)
    var h_nan = ctx.enqueue_create_host_buffer[DType.int32](slots)
    var h_inf = ctx.enqueue_create_host_buffer[DType.int32](slots)
    ctx.enqueue_copy(h_worst, worst)
    ctx.enqueue_copy(h_refmax, refmax)
    ctx.enqueue_copy(h_nan, nan_buf)
    ctx.enqueue_copy(h_inf, inf_buf)
    ctx.synchronize()
    var err = Float64(0.0)
    var scale_ref = Float64(0.0)
    var nans = 0
    var infs = 0
    for i in range(slots):
        if Float64(h_worst[i]) > err:
            err = Float64(h_worst[i])
        if Float64(h_refmax[i]) > scale_ref:
            scale_ref = Float64(h_refmax[i])
        nans += Int(h_nan[i])
        infs += Int(h_inf[i])
    comptime eps = 0.0078125 if dtype == DType.bfloat16 else 1.1920929e-7
    _ = worst^
    _ = refmax^
    _ = nan_buf^
    _ = inf_buf^
    # Relative to this gradient's own magnitude: dQ, dK and dV do not share a
    # scale, so one absolute bound cannot serve all three.
    return GradCheck(err, 8.0 * Float64(eps) * scale_ref, nans, infs)


struct BwdResult(ImplicitlyCopyable, Movable):
    var median_us: Float64
    var dq: GradCheck
    var dk: GradCheck
    var dv: GradCheck
    var checked: Bool

    def __init__(
        out self,
        median_us: Float64,
        var dq: GradCheck,
        var dk: GradCheck,
        var dv: GradCheck,
        checked: Bool,
    ):
        self.median_us = median_us
        self.dq = dq^
        self.dk = dk^
        self.dv = dv^
        self.checked = checked

    def passed(self) -> Bool:
        if not self.checked:
            return True
        return self.dq.passed() and self.dk.passed() and self.dv.passed()


def run_case[
    dtype: DType
](
    target: BwdCase,
    warmup: Int,
    iterations: Int,
    check: Bool,
    ctx: DeviceContext,
) raises -> BwdResult:
    var b = target.batch
    var h = target.heads
    var sq = target.seq_q
    var skv = target.seq_kv
    var hd = target.head_dim
    var qn = b * h * sq * hd
    var kn = b * h * skv * hd
    var rows = b * h * sq
    var scale = Float32(1.0) / sqrt(Float32(hd))
    var causal = 1 if target.causal else 0

    var qf = ctx.enqueue_create_buffer[DType.float32](qn)
    var kf = ctx.enqueue_create_buffer[DType.float32](kn)
    var vf = ctx.enqueue_create_buffer[DType.float32](kn)
    var gf = ctx.enqueue_create_buffer[DType.float32](qn)
    var q = ctx.enqueue_create_buffer[dtype](qn)
    var k = ctx.enqueue_create_buffer[dtype](kn)
    var v = ctx.enqueue_create_buffer[dtype](kn)
    var go = ctx.enqueue_create_buffer[dtype](qn)
    var ofwd = ctx.enqueue_create_buffer[dtype](qn)
    var lse = ctx.enqueue_create_buffer[DType.float32](rows)
    var dq = ctx.enqueue_create_buffer[dtype](qn)
    var dk = ctx.enqueue_create_buffer[dtype](kn)
    var dv = ctx.enqueue_create_buffer[dtype](kn)

    ctx.enqueue_function[_fill](
        qf.unsafe_ptr().as_unsafe_any_origin(),
        qn,
        13,
        Float32(8.0),
        grid_dim=(max(1, min(ceildiv(qn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_fill](
        kf.unsafe_ptr().as_unsafe_any_origin(),
        kn,
        7932,
        Float32(1.0),
        grid_dim=(max(1, min(ceildiv(kn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_fill](
        vf.unsafe_ptr().as_unsafe_any_origin(),
        kn,
        15851,
        Float32(1.0),
        grid_dim=(max(1, min(ceildiv(kn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_fill](
        gf.unsafe_ptr().as_unsafe_any_origin(),
        qn,
        24601,
        Float32(1.0),
        grid_dim=(max(1, min(ceildiv(qn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast[dtype]](
        q.unsafe_ptr().as_unsafe_any_origin(),
        qf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        qn,
        grid_dim=(max(1, min(ceildiv(qn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast[dtype]](
        k.unsafe_ptr().as_unsafe_any_origin(),
        kf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        kn,
        grid_dim=(max(1, min(ceildiv(kn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast[dtype]](
        v.unsafe_ptr().as_unsafe_any_origin(),
        vf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        kn,
        grid_dim=(max(1, min(ceildiv(kn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_cast[dtype]](
        go.unsafe_ptr().as_unsafe_any_origin(),
        gf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        qn,
        grid_dim=(max(1, min(ceildiv(qn, FILL_BLOCK), 1024)),),
        block_dim=(FILL_BLOCK,),
    )
    ctx.enqueue_function[_setup_forward[dtype]](
        ofwd.unsafe_ptr().as_unsafe_any_origin(),
        lse.unsafe_ptr().as_unsafe_any_origin(),
        q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        b,
        h,
        sq,
        skv,
        hd,
        scale,
        causal,
        grid_dim=(ceildiv(rows, REF_THREADS),),
        block_dim=(REF_THREADS,),
    )

    @always_inline
    @parameter
    def _launch() raises:
        # The contract says the outputs arrive zeroed, so zero them each time.
        ctx.enqueue_memset(dq, Scalar[dtype](0))
        ctx.enqueue_memset(dk, Scalar[dtype](0))
        ctx.enqueue_memset(dv, Scalar[dtype](0))
        enqueue_flash_attention_bwd[dtype](
            dq.unsafe_ptr().as_unsafe_any_origin(),
            dk.unsafe_ptr().as_unsafe_any_origin(),
            dv.unsafe_ptr().as_unsafe_any_origin(),
            go.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            ofwd.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            lse.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            b,
            h,
            sq,
            skv,
            hd,
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

    var zero = GradCheck(0.0, 1.0, 0, 0)
    var c_dq = zero
    var c_dk = zero
    var c_dv = zero
    if check:
        var rdq = ctx.enqueue_create_buffer[DType.float32](qn)
        var rdk = ctx.enqueue_create_buffer[DType.float32](kn)
        var rdv = ctx.enqueue_create_buffer[DType.float32](kn)
        ctx.enqueue_function[_ref_dq[dtype]](
            rdq.unsafe_ptr().as_unsafe_any_origin(),
            go.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            ofwd.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            lse.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            b,
            h,
            sq,
            skv,
            hd,
            scale,
            causal,
            grid_dim=(ceildiv(rows, REF_THREADS),),
            block_dim=(REF_THREADS,),
        )
        ctx.enqueue_function[_ref_dkv[dtype]](
            rdk.unsafe_ptr().as_unsafe_any_origin(),
            rdv.unsafe_ptr().as_unsafe_any_origin(),
            go.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            q.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            k.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            v.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            ofwd.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            lse.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            b,
            h,
            sq,
            skv,
            hd,
            scale,
            causal,
            grid_dim=(ceildiv(b * h * skv, REF_THREADS),),
            block_dim=(REF_THREADS,),
        )
        c_dq = _check_one[dtype](dq, rdq, qn, ctx)
        c_dk = _check_one[dtype](dk, rdk, kn, ctx)
        c_dv = _check_one[dtype](dv, rdv, kn, ctx)
        _ = rdq^
        _ = rdk^
        _ = rdv^

    var result = BwdResult(_percentile(samples, 50), c_dq, c_dk, c_dv, check)
    _ = qf^
    _ = kf^
    _ = vf^
    _ = gf^
    _ = q^
    _ = k^
    _ = v^
    _ = go^
    _ = ofwd^
    _ = lse^
    _ = dq^
    _ = dk^
    _ = dv^
    return result


def main() raises:
    var only = String(arg_parse("case", "all"))
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var check = Int(arg_parse("check", 1)) != 0
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    print(
        "flash-attention BACKWARD, bfloat16 | warmup=",
        warmup,
        " iterations=",
        iterations,
        " check=",
        Int(check),
    )
    print(
        "case             b   h   sq   skv  hd c   mojo_us    rocm_us  ratio  "
        "  dq_err     dk_err     dv_err     nan  status"
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
                _pad(_fixed(r.dq.max_err, 6), 10),
                _pad(_fixed(r.dk.max_err, 6), 10),
                _pad(_fixed(r.dv.max_err, 6), 10),
                _pad(String(r.dq.nans + r.dk.nans + r.dv.nans), 4),
                "pass" if r.passed() else "FAIL",
            )
            if not r.passed():
                failed += 1
                if not r.dq.passed():
                    print(
                        "      dQ over bound:",
                        _fixed(r.dq.max_err, 8),
                        "vs",
                        _fixed(r.dq.tolerance, 8),
                        " nan",
                        r.dq.nans,
                        " inf",
                        r.dq.infs,
                    )
                if not r.dk.passed():
                    print(
                        "      dK over bound:",
                        _fixed(r.dk.max_err, 8),
                        "vs",
                        _fixed(r.dk.tolerance, 8),
                        " nan",
                        r.dk.nans,
                        " inf",
                        r.dk.infs,
                    )
                if not r.dv.passed():
                    print(
                        "      dV over bound:",
                        _fixed(r.dv.max_err, 8),
                        "vs",
                        _fixed(r.dv.tolerance, 8),
                        " nan",
                        r.dv.nans,
                        " inf",
                        r.dv.infs,
                    )
    if matched == 0:
        raise Error("no case matched --case=", only)

    if rocm_step_us > 0.0:
        print()
        print(
            "nanoGPT per-step attention backward: this kernel",
            _fixed(mojo_step_us / 1000.0, 3),
            "ms, PyTorch-ROCm",
            _fixed(rocm_step_us / 1000.0, 3),
            "ms, ratio",
            _fixed(mojo_step_us / rocm_step_us, 3),
        )
        # Two bars. The eager device currently computes this backward by a
        # decomposition measured at 46.153 ms/step in the production profile,
        # 1.36x ROCm; a fused kernel is worth wiring in once it beats that.
        print(
            "  bar to beat (current decomposition): 46.153 ms/step -> this"
            " kernel is",
            _fixed(mojo_step_us / 1000.0 / 46.153, 3),
            "x that",
        )
        print(
            "  target (PyTorch-ROCm fused):         28.966 ms/step -> ratio",
            _fixed(mojo_step_us / rocm_step_us, 3),
        )
    if failed == 0:
        print("correctness: all cases pass on dQ, dK and dV")
        return
    print(
        "correctness: ", failed, "case(s) FAILED - their timings mean nothing"
    )
    raise Error("flash attention backward failure in ", failed, " case(s)")
