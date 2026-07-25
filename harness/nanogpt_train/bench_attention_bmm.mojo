"""Pure-Mojo benchmark for the batched GEMMs of the eager SDPA decomposition.

Same two-second build/iterate loop as `bench_linear_gemm.mojo`, and it calls the
same production entry point the Python path reaches:
`matmul_ops._gemm_dtype_dispatch`, which internally tries
`_amd_dynamic_mfma_dispatch` and falls back to the portable `pure_gemm_tiled`
kernel.  That fallback is the point: the AMD MFMA dispatch currently returns
`False` for `batch != 1`, so every one of these GEMMs runs on a scalar-FFMA
kernel and leaves the matrix cores idle.

The reference column is `torch.bmm` on PyTorch-ROCm for the identical batched
shape, from `scripts/rocm_attention_reference.py`.  Read it as "how fast a
batched BF16 GEMM of this shape can run on this hardware", NOT as an acceptance
target: PyTorch-ROCm's attention runs one fused flash-attention kernel and never
issues these GEMMs at all.  The acceptance target is the whole op, in
`rocm_attention_targets.csv`.

Build and run:
    uv run --no-sync mojo build harness/nanogpt_train/bench_attention_bmm.mojo \
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_attention_bmm
    /tmp/bench_attention_bmm \
        --targets=harness/nanogpt_train/rocm_attention_bmm_targets.csv

    /tmp/bench_attention_bmm --case=fwd_scores_qkT --pattern-check=1
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.time import perf_counter_ns

from internal_utils import arg_parse
from matmul_ops import _amd_batched_mfma_gemm, _gemm_dtype_dispatch

comptime FILL_BLOCK = 256
comptime FILL_VEC = 4
comptime CHECK_BLOCKS = 1024
# Nonzero only on the first and last EDGE contraction indices, so every output
# stays a small exact integer while still varying along m, n and k.
comptime EDGE = 4


struct BmmCase(ImplicitlyCopyable, Movable):
    var label: String
    var role: String
    var batch: Int
    var m: Int
    var n: Int
    var k: Int
    var transpose_b: Bool
    var calls_per_step: Int
    var rocm_us: Float64

    def __init__(
        out self,
        var label: String,
        var role: String,
        batch: Int,
        m: Int,
        n: Int,
        k: Int,
        transpose_b: Bool,
        calls_per_step: Int,
        rocm_us: Float64,
    ):
        self.label = label^
        self.role = role^
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.transpose_b = transpose_b
        self.calls_per_step = calls_per_step
        self.rocm_us = rocm_us


@__name("bench_bmm_fill_const_bf16")
def _fill_const(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
    value: Scalar[DType.bfloat16],
):
    var i = (Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)) * FILL_VEC
    var stride = Int(grid_dim.x) * FILL_BLOCK * FILL_VEC
    while i < count:
        if i + FILL_VEC <= count:
            ptr.store[width=FILL_VEC](i, SIMD[DType.bfloat16, FILL_VEC](value))
        else:
            for lane in range(FILL_VEC):
                if i + lane < count:
                    ptr[i + lane] = value
        i += stride


@always_inline
def _pattern_left(row: Int, index: Int) -> Int:
    return ((row + index) % 3) - 1


@always_inline
def _pattern_right(index: Int, col: Int) -> Int:
    return ((2 * index + col) % 3) - 1


@__name("bench_bmm_fill_pattern_bf16")
def _fill_pattern(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    batch: Int,
    rows: Int,
    cols: Int,
    k: Int,
    k_is_column: Int,
    is_left_operand: Int,
):
    """Fill every matrix of the batch with the sparse edge pattern.

    Each batch element gets the same pattern, so the closed-form reference is
    batch-independent; a kernel that mixes up batch strides still shows up,
    because it would read a differently-offset row and land on a different
    pattern value.
    """
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var per_matrix = rows * cols
    var count = batch * per_matrix
    while index < count:
        var within = index % per_matrix
        var row = within // cols
        var col = within % cols
        var contraction = col if k_is_column != 0 else row
        var other = row if k_is_column != 0 else col
        var value = 0
        if contraction < EDGE or contraction >= k - EDGE:
            if is_left_operand != 0:
                value = _pattern_left(other, contraction)
            else:
                value = _pattern_right(contraction, other)
        ptr[index] = Float32(value).cast[DType.bfloat16]()
        index += stride


@__name("bench_bmm_count_ne_bf16")
def _count_not_equal(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    values: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    count: Int,
    expected: Float32,
):
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var local = Int32(0)
    while index < count:
        if values[index].cast[DType.float32]() != expected:
            local += 1
        index += stride
    counts[Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)] = local


@__name("bench_bmm_count_pattern_ne_bf16")
def _count_pattern_not_equal(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    values: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
):
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var per_matrix = m * n
    var count = batch * per_matrix
    var local = Int32(0)
    while index < count:
        var within = index % per_matrix
        var row = within // n
        var col = within % n
        var expected = 0
        for edge in range(EDGE):
            expected += _pattern_left(row, edge) * _pattern_right(edge, col)
            var tail = k - 1 - edge
            if tail >= EDGE:
                expected += _pattern_left(row, tail) * _pattern_right(tail, col)
        if values[index].cast[DType.float32]() != Float32(expected):
            local += 1
        index += stride
    counts[Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)] = local


@always_inline
def _enqueue_fill_const(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
    value: Float32,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_fill_const](
        ptr,
        count,
        value.cast[DType.bfloat16](),
        grid_dim=(max(1, min(ceildiv(count, FILL_BLOCK * FILL_VEC), 512)),),
        block_dim=(FILL_BLOCK,),
    )


@always_inline
def _enqueue_fill_pattern(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    batch: Int,
    rows: Int,
    cols: Int,
    k: Int,
    k_is_column: Int,
    is_left_operand: Int,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_fill_pattern](
        ptr,
        batch,
        rows,
        cols,
        k,
        k_is_column,
        is_left_operand,
        grid_dim=(max(1, min(ceildiv(batch * rows * cols, FILL_BLOCK), 512)),),
        block_dim=(FILL_BLOCK,),
    )


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


def _split_line(line: String) -> List[String]:
    var fields = List[String]()
    var current = String("")
    for character in line.codepoints():
        if character == Codepoint.ord(","):
            fields.append(current)
            current = String("")
        else:
            current += String(character)
    fields.append(current)
    return fields^


def read_targets(path: String) raises -> List[BmmCase]:
    var text: String
    with open(path, "r") as file:
        text = file.read()
    var cases = List[BmmCase]()
    var first = True
    for line in text.split("\n"):
        var trimmed = String(line.strip())
        if trimmed.byte_length() == 0:
            continue
        if first:
            first = False  # header
            continue
        var fields = _split_line(trimmed)
        if len(fields) < 9:
            raise Error("malformed target row: ", trimmed)
        cases.append(
            BmmCase(
                String(fields[0].strip()),
                String(fields[1].strip()),
                Int(atol(fields[2].strip())),
                Int(atol(fields[3].strip())),
                Int(atol(fields[4].strip())),
                Int(atol(fields[5].strip())),
                atol(fields[6].strip()) != 0,
                Int(atol(fields[7].strip())),
                Float64(atof(fields[8].strip())),
            )
        )
    return cases^


struct CaseResult(ImplicitlyCopyable, Movable):
    var median_us: Float64
    var p10_us: Float64
    var p90_us: Float64
    var tflops: Float64
    var mismatches: Int

    def __init__(
        out self,
        median_us: Float64,
        p10_us: Float64,
        p90_us: Float64,
        tflops: Float64,
        mismatches: Int,
    ):
        self.median_us = median_us
        self.p10_us = p10_us
        self.p90_us = p90_us
        self.tflops = tflops
        self.mismatches = mismatches


def _sum_counts(
    counts: DeviceBuffer[DType.int32], slots: Int, ctx: DeviceContext
) raises -> Int:
    var host = ctx.enqueue_create_host_buffer[DType.int32](slots)
    ctx.enqueue_copy(host, counts)
    ctx.synchronize()
    var total = 0
    for index in range(slots):
        total += Int(host[index])
    return total


def _bf16_round(value: Int) -> Float32:
    """The BF16 value an exact FP32 integer becomes when stored as BF16.

    BF16 keeps 8 significand bits, so the all-ones expected value K is not
    always representable (K = 50304 is not; K = 1024 and 64 are).  Rounding the
    reference the same way the store does keeps this an exact equality test.
    """
    var magnitude = Float32(value)
    var scale = Float32(1.0)
    while magnitude >= 256.0:
        magnitude /= 2.0
        scale *= 2.0
    # magnitude is now in [128, 256) where every integer is BF16-exact.
    return Float32(Int(magnitude + 0.5)) * scale


@always_inline
def _explicit_batched[
    transpose_b: Bool
](
    config: Int,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Candidate batched-MFMA geometries, selected by `--config`.

    Benchmark-only: production dispatch (`--config=0`) picks a geometry from the
    runtime shape.  Every extent stays runtime here too; only the tile is
    compile-time.  Transposed-B geometries one MMA wide in a dimension are
    miscompiled on gfx942 (journal Change 10), so they are only instantiated for
    `transpose_b=False`.
    """
    comptime DT = DType.bfloat16
    var bs = m * k
    if config == 1:
        _amd_batched_mfma_gemm[DT, 128, 128, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 2:
        _amd_batched_mfma_gemm[DT, 64, 128, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 3:
        _amd_batched_mfma_gemm[DT, 128, 64, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 4:
        _amd_batched_mfma_gemm[DT, 64, 64, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 5:
        _amd_batched_mfma_gemm[DT, 64, 256, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 6:
        _amd_batched_mfma_gemm[DT, 128, 128, 64, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 7:
        _amd_batched_mfma_gemm[DT, 64, 128, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 8:
        _amd_batched_mfma_gemm[DT, 32, 128, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 9:
        _amd_batched_mfma_gemm[DT, 128, 128, 32, 64, transpose_b, 32, 1, 3](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 10:
        _amd_batched_mfma_gemm[DT, 64, 64, 32, 32, transpose_b, 32, 1, 3](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 11:
        _amd_batched_mfma_gemm[DT, 128, 128, 32, 64, transpose_b, 64](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 12:
        _amd_batched_mfma_gemm[DT, 64, 128, 32, 64, transpose_b, 64](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 13:
        _amd_batched_mfma_gemm[DT, 256, 128, 64, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 14:
        _amd_batched_mfma_gemm[DT, 128, 256, 64, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 15:
        _amd_batched_mfma_gemm[DT, 32, 64, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 16:
        _amd_batched_mfma_gemm[DT, 64, 32, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 17:
        _amd_batched_mfma_gemm[DT, 32, 32, 32, 32, transpose_b, 32, 2](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 18:
        _amd_batched_mfma_gemm[DT, 256, 64, 64, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 23:
        _amd_batched_mfma_gemm[DT, 128, 256, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 24:
        _amd_batched_mfma_gemm[DT, 256, 128, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 25:
        _amd_batched_mfma_gemm[DT, 128, 128, 32, 32, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 26:
        _amd_batched_mfma_gemm[DT, 64, 64, 32, 32, transpose_b, 64](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    elif config == 27:
        _amd_batched_mfma_gemm[DT, 128, 64, 32, 64, transpose_b](
            c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
        )
    else:
        comptime if transpose_b:
            raise Error("config not available for transpose_b=True")
        else:
            if config == 19:
                _amd_batched_mfma_gemm[DT, 128, 64, 32, 16, transpose_b](
                    c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
                )
            elif config == 20:
                _amd_batched_mfma_gemm[DT, 64, 64, 16, 32, transpose_b](
                    c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
                )
            elif config == 21:
                _amd_batched_mfma_gemm[DT, 32, 64, 16, 32, transpose_b](
                    c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
                )
            elif config == 22:
                _amd_batched_mfma_gemm[DT, 64, 128, 16, 64, transpose_b](
                    c_addr, a_addr, b_addr, batch, m, n, k, bs, ctx
                )
            else:
                raise Error("unknown --config")


def run_case(
    target: BmmCase,
    warmup: Int,
    iterations: Int,
    pattern_check: Bool,
    config: Int,
    ctx: DeviceContext,
) raises -> CaseResult:
    var batch = target.batch
    var m = target.m
    var n = target.n
    var k = target.k
    var a_elements = batch * m * k
    var b_elements = batch * k * n
    var c_elements = batch * m * n

    var a_buf = ctx.enqueue_create_buffer[DType.bfloat16](a_elements)
    var b_buf = ctx.enqueue_create_buffer[DType.bfloat16](b_elements)
    var c_buf = ctx.enqueue_create_buffer[DType.bfloat16](c_elements)
    var a_ptr = a_buf.unsafe_ptr().as_unsafe_any_origin()
    var b_ptr = b_buf.unsafe_ptr().as_unsafe_any_origin()
    var c_ptr = c_buf.unsafe_ptr().as_unsafe_any_origin()

    if pattern_check:
        _enqueue_fill_pattern(a_ptr, batch, m, k, k, 1, 1, ctx)
        if target.transpose_b:
            _enqueue_fill_pattern(b_ptr, batch, n, k, k, 1, 0, ctx)
        else:
            _enqueue_fill_pattern(b_ptr, batch, k, n, k, 0, 0, ctx)
    else:
        _enqueue_fill_const(a_ptr, a_elements, 1.0, ctx)
        _enqueue_fill_const(b_ptr, b_elements, 1.0, ctx)
    _enqueue_fill_const(c_ptr, c_elements, 0.0, ctx)

    @always_inline
    @parameter
    def _launch() raises:
        if config == 0:
            # The exact production route: _bmm_go builds these arguments from
            # the Python-side pointers and calls this function.
            _gemm_dtype_dispatch(
                DType.bfloat16,
                Int(c_buf.unsafe_ptr()),
                Int(a_buf.unsafe_ptr()),
                Int(b_buf.unsafe_ptr()),
                batch,
                m,
                n,
                k,
                m * k,
                1 if target.transpose_b else 0,
                0,
                0,
                0,
                ctx,
            )
        elif target.transpose_b:
            _explicit_batched[True](
                config,
                Int(c_buf.unsafe_ptr()),
                Int(a_buf.unsafe_ptr()),
                Int(b_buf.unsafe_ptr()),
                batch,
                m,
                n,
                k,
                ctx,
            )
        else:
            _explicit_batched[False](
                config,
                Int(c_buf.unsafe_ptr()),
                Int(a_buf.unsafe_ptr()),
                Int(b_buf.unsafe_ptr()),
                batch,
                m,
                n,
                k,
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

    var slots = CHECK_BLOCKS * FILL_BLOCK
    var counts = ctx.enqueue_create_buffer[DType.int32](slots)
    if pattern_check:
        ctx.enqueue_function[_count_pattern_not_equal](
            counts.unsafe_ptr().as_unsafe_any_origin(),
            c_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            batch,
            m,
            n,
            k,
            grid_dim=(CHECK_BLOCKS,),
            block_dim=(FILL_BLOCK,),
        )
    else:
        ctx.enqueue_function[_count_not_equal](
            counts.unsafe_ptr().as_unsafe_any_origin(),
            c_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            c_elements,
            _bf16_round(k),
            grid_dim=(CHECK_BLOCKS,),
            block_dim=(FILL_BLOCK,),
        )
    var mismatches = _sum_counts(counts, slots, ctx)

    var median = _percentile(samples, 50)
    var flops = 2.0 * Float64(batch) * Float64(m) * Float64(n) * Float64(k)
    var result = CaseResult(
        median,
        _percentile(samples, 10),
        _percentile(samples, 90),
        flops / (median * 1.0e6),
        mismatches,
    )
    _ = a_buf^
    _ = b_buf^
    _ = c_buf^
    _ = counts^
    return result


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


def main() raises:
    var targets_path = String(
        arg_parse(
            "targets", "harness/nanogpt_train/rocm_attention_bmm_targets.csv"
        )
    )
    var only = String(arg_parse("case", "all"))
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var pattern_check = Int(arg_parse("pattern-check", 0)) != 0
    var config = Int(arg_parse("config", 0))
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    var cases = read_targets(targets_path)
    print(
        "case              role  batch      m      n      k tb   mojo_us"
        "    rocm_us  ratio  TFLOP/s  correct"
    )
    var mojo_step_us = Float64(0.0)
    var rocm_step_us = Float64(0.0)
    var failed = 0
    with DeviceContext() as ctx:
        for target in cases:
            if only != "all" and target.label != only:
                continue
            var result = run_case(
                target, warmup, iterations, pattern_check, config, ctx
            )
            mojo_step_us += result.median_us * Float64(target.calls_per_step)
            rocm_step_us += target.rocm_us * Float64(target.calls_per_step)
            print(
                _pad(target.label, 17),
                _pad(target.role, 5),
                _pad(String(target.batch), 5),
                _pad(String(target.m), 6),
                _pad(String(target.n), 6),
                _pad(String(target.k), 6),
                Int(target.transpose_b),
                _pad(_fixed(result.median_us, 2), 10),
                _pad(_fixed(target.rocm_us, 2), 10),
                _pad(_fixed(result.median_us / target.rocm_us, 3), 6),
                _pad(_fixed(result.tflops, 1), 7),
                "pass" if result.mismatches
                == 0 else String(
                    "FAIL:",
                    result.mismatches,
                    "/",
                    target.batch * target.m * target.n,
                ),
            )
            if result.mismatches != 0:
                failed += 1
    if rocm_step_us == 0.0:
        raise Error("no case matched --case=", only)
    print()
    print(
        "per-step batched-GEMM total: mojo",
        _fixed(mojo_step_us / 1000.0, 3),
        "ms, rocm torch.bmm",
        _fixed(rocm_step_us / 1000.0, 3),
        "ms, ratio",
        _fixed(mojo_step_us / rocm_step_us, 3),
    )
    print(
        "reminder: PyTorch-ROCm attention issues none of these GEMMs. Its whole"
        " fused attention is 37.1 ms/step for both directions, so matching"
        " torch.bmm here is necessary but not sufficient."
    )
    if failed == 0:
        print(
            "correctness: all cases pass (pattern_check=",
            Int(pattern_check),
            ")",
        )
        return
    print(
        "correctness: ", failed, "case(s) FAILED - their timings mean nothing"
    )
    raise Error("batched GEMM correctness failure in ", failed, " case(s)")
