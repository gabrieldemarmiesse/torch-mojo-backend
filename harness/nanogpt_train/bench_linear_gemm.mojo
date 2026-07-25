"""Pure-Mojo benchmark for the nanoGPT training step's Linear GEMMs.

Builds in about two seconds and calls the production dispatch
(`matmul_ops._amd_dynamic_mfma_dispatch`) directly, so an optimization agent
iterates on `torch_mojo_backend/eager_kernels/matmul_ops.mojo` without paying
the ~10 minute cost of rebuilding every eager CPython extension.  Editing any
`.mojo` file under `eager_kernels/` invalidates the whole `__mojocache__`; this
binary does not use that cache at all.

Cases come from `rocm_gemm_targets.csv`, produced once by
`scripts/rocm_gemm_reference.py`, so each line prints the Mojo time, the
PyTorch-ROCm time for the identical GEMM, and their ratio.  Acceptance for this
workload is ratio <= 1.02 on the per-step weighted total.

`transpose_a` rows are the weight-gradient GEMMs.  PyTorch-ROCm consumes the
transposed operand as a strided view; the Mojo route has no transposed-A
kernel, so `_matmul_spec_operands_launch` materializes a contiguous copy first.
The harness reproduces that exactly and reports the copy separately, because
removing it is a legitimate optimization and its cost belongs to the op.

Build and run:
    uv run --no-sync mojo build harness/nanogpt_train/bench_linear_gemm.mojo \
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_linear_gemm
    /tmp/bench_linear_gemm --targets harness/nanogpt_train/rocm_gemm_targets.csv

    # one case, more iterations, with the layout-sensitive pattern check
    /tmp/bench_linear_gemm --case=mlp_c_fc_wgrad --iterations=200 --pattern-check=1

Profile a single case per kernel (durations, VGPR/LDS, counters):
    uv run --no-sync python scripts/rocprof_kernels.py -- \
        /tmp/bench_linear_gemm --case=mlp_c_fc_fwd --warmup=25 --iterations=100
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.time import perf_counter_ns
from std.utils import IndexList

from internal_utils import arg_parse
from matmul_ops import _amd_dynamic_mfma_dispatch
from op_utils import MAX_RANK, _copy_strided

comptime FILL_BLOCK = 256
comptime FILL_VEC = 4
comptime CHECK_BLOCKS = 1024


struct GemmCase(ImplicitlyCopyable, Movable):
    """One GEMM the training step issues, plus its PyTorch-ROCm target time."""

    var label: String
    var role: String
    var m: Int
    var n: Int
    var k: Int
    var transpose_a: Bool
    var transpose_b: Bool
    var calls_per_step: Int
    var rocm_us: Float64

    def __init__(
        out self,
        var label: String,
        var role: String,
        m: Int,
        n: Int,
        k: Int,
        transpose_a: Bool,
        transpose_b: Bool,
        calls_per_step: Int,
        rocm_us: Float64,
    ):
        self.label = label^
        self.role = role^
        self.m = m
        self.n = n
        self.k = k
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.calls_per_step = calls_per_step
        self.rocm_us = rocm_us


@__name("bench_gemm_fill_const_bf16")
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


# The layout-sensitive pattern is nonzero only on the first and last few K
# indices.  Values in {-1, 0, 1} are exact in BF16 and at most 2 * EDGE terms
# contribute, so every output is a small integer the FP32 accumulator carries
# exactly and BF16 stores exactly: the check is an equality, not a tolerance.
# Loading both ends means a kernel that truncates either its first or its last
# K tile is caught, while the all-ones check separately proves that every K
# index is summed exactly once.
comptime EDGE = 4


@always_inline
def _pattern_left(row: Int, index: Int) -> Int:
    """A[row, index], zero away from the two K edges."""
    return ((row + index) % 3) - 1


@always_inline
def _pattern_right(index: Int, col: Int) -> Int:
    """B[index, col], zero away from the two K edges."""
    return ((2 * index + col) % 3) - 1


@always_inline
def _is_edge(index: Int, k: Int) -> Bool:
    return index < EDGE or index >= k - EDGE


@__name("bench_gemm_fill_pattern_bf16")
def _fill_pattern(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
    cols: Int,
    k: Int,
    k_is_column: Int,
    is_left_operand: Int,
):
    """Write the sparse edge pattern into a rows x cols row-major buffer.

    ``k_is_column`` says which axis of this physical buffer is the contraction
    axis, which is what makes the same kernel serve normal and transposed
    operands.
    """
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var count = rows * cols
    while index < count:
        var row = index // cols
        var col = index % cols
        var contraction = col if k_is_column != 0 else row
        var other = row if k_is_column != 0 else col
        var value = 0
        if _is_edge(contraction, k):
            if is_left_operand != 0:
                value = _pattern_left(other, contraction)
            else:
                value = _pattern_right(contraction, other)
        ptr[index] = Float32(value).cast[DType.bfloat16]()
        index += stride


@__name("bench_gemm_count_pattern_ne_bf16")
def _count_pattern_not_equal(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    values: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """Count outputs that differ from the closed-form product of the pattern.

    The reference is recomputed here from the pattern's definition rather than
    from another GEMM, so it shares no code with what it is checking.
    """
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var count = m * n
    var local = Int32(0)
    while index < count:
        var row = index // n
        var col = index % n
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


@__name("bench_gemm_count_ne_bf16")
def _count_not_equal(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    values: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    count: Int,
    expected: Float32,
):
    """Per-block count of elements differing from `expected`.

    Reducing on the device keeps the check O(1) in host transfer, so every
    output element of even the 49152 x 50304 logits GEMM is inspected.
    """
    var index = Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * FILL_BLOCK
    var local = Int32(0)
    while index < count:
        if values[index].cast[DType.float32]() != expected:
            local += 1
        index += stride
    # One slot per thread avoids atomics; the host sums the small array.
    counts[Int(block_idx.x) * FILL_BLOCK + Int(thread_idx.x)] = local


@always_inline
def _fill_blocks(count: Int) -> Int:
    return max(1, min(ceildiv(count, FILL_BLOCK * FILL_VEC), 512))


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
        grid_dim=(_fill_blocks(count),),
        block_dim=(FILL_BLOCK,),
    )


@always_inline
def _enqueue_fill_pattern(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
    cols: Int,
    k: Int,
    k_is_column: Int,
    is_left_operand: Int,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_fill_pattern](
        ptr,
        rows,
        cols,
        k,
        k_is_column,
        is_left_operand,
        grid_dim=(max(1, min(ceildiv(rows * cols, FILL_BLOCK), 512)),),
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


def _row_major_2d(rows: Int, cols: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = rows
    shape[MAX_RANK - 1] = cols
    return shape


def _strides_2d(row_stride: Int, col_stride: Int) -> IndexList[MAX_RANK]:
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 2] = row_stride
    strides[MAX_RANK - 1] = col_stride
    return strides


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


def read_targets(path: String) raises -> List[GemmCase]:
    """Parse the ROCm reference CSV written by scripts/rocm_gemm_reference.py."""
    var text: String
    with open(path, "r") as file:
        text = file.read()
    var cases = List[GemmCase]()
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
            GemmCase(
                String(fields[0].strip()),
                String(fields[1].strip()),
                Int(atol(fields[2].strip())),
                Int(atol(fields[3].strip())),
                Int(atol(fields[4].strip())),
                atol(fields[5].strip()) != 0,
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
    var copy_us: Float64
    var tflops: Float64
    var mismatches: Int

    def __init__(
        out self,
        median_us: Float64,
        p10_us: Float64,
        p90_us: Float64,
        copy_us: Float64,
        tflops: Float64,
        mismatches: Int,
    ):
        self.median_us = median_us
        self.p10_us = p10_us
        self.p90_us = p90_us
        self.copy_us = copy_us
        self.tflops = tflops
        self.mismatches = mismatches


def run_case(
    target: GemmCase,
    warmup: Int,
    iterations: Int,
    pattern_check: Bool,
    ctx: DeviceContext,
) raises -> CaseResult:
    var m = target.m
    var n = target.n
    var k = target.k

    # Physical buffers hold what the layout flags say: a transposed operand is
    # stored in the un-transposed extents and read through strides.
    var a_buf = ctx.enqueue_create_buffer[DType.bfloat16](m * k)
    var b_buf = ctx.enqueue_create_buffer[DType.bfloat16](k * n)
    var c_buf = ctx.enqueue_create_buffer[DType.bfloat16](m * n)
    # Materialization target for the transposed-A route.
    var a_contig = ctx.enqueue_create_buffer[DType.bfloat16](
        m * k if target.transpose_a else 1
    )

    var a_ptr = a_buf.unsafe_ptr().as_unsafe_any_origin()
    var b_ptr = b_buf.unsafe_ptr().as_unsafe_any_origin()
    var c_ptr = c_buf.unsafe_ptr().as_unsafe_any_origin()

    if pattern_check:
        # A is logically (m, k); when transposed its buffer is (k, m).
        if target.transpose_a:
            _enqueue_fill_pattern(a_ptr, k, m, k, 0, 1, ctx)
        else:
            _enqueue_fill_pattern(a_ptr, m, k, k, 1, 1, ctx)
        if target.transpose_b:
            _enqueue_fill_pattern(b_ptr, n, k, k, 1, 0, ctx)
        else:
            _enqueue_fill_pattern(b_ptr, k, n, k, 0, 0, ctx)
    else:
        _enqueue_fill_const(a_ptr, m * k, 1.0, ctx)
        _enqueue_fill_const(b_ptr, k * n, 1.0, ctx)
    _enqueue_fill_const(c_ptr, m * n, 0.0, ctx)

    var gemm_a_addr = Int(a_contig.unsafe_ptr()) if target.transpose_a else Int(
        a_buf.unsafe_ptr()
    )

    @always_inline
    @parameter
    def _materialize() raises:
        # Exactly what _matmul_spec_operands_launch does for a strided operand.
        _copy_strided[DType.uint16](
            Int(a_contig.unsafe_ptr()),
            Int(a_buf.unsafe_ptr()),
            _row_major_2d(m, k),
            _strides_2d(k, 1),
            _strides_2d(1, m),
            ctx,
        )

    @always_inline
    @parameter
    def _gemm() raises:
        var handled: Bool
        if target.transpose_b:
            handled = _amd_dynamic_mfma_dispatch[DType.bfloat16, True, False](
                Int(c_buf.unsafe_ptr()),
                gemm_a_addr,
                Int(b_buf.unsafe_ptr()),
                1,
                m,
                n,
                k,
                m * k,
                0,
                ctx,
            )
        else:
            handled = _amd_dynamic_mfma_dispatch[DType.bfloat16, False, False](
                Int(c_buf.unsafe_ptr()),
                gemm_a_addr,
                Int(b_buf.unsafe_ptr()),
                1,
                m,
                n,
                k,
                m * k,
                0,
                ctx,
            )
        if not handled:
            raise Error(
                "the AMD MFMA dispatch declined ",
                target.label,
                ": m=",
                m,
                " n=",
                n,
                " k=",
                k,
                ". A declined shape falls back to the portable scalar kernel"
                " in production, so it must be handled here too.",
            )

    @always_inline
    @parameter
    def _full() raises:
        if target.transpose_a:
            _materialize()
        _gemm()

    for _ in range(warmup):
        _full()
    ctx.synchronize()

    var samples = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _full()
        ctx.synchronize()
        samples.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(samples)

    # Time the materialization alone so the report can separate it from the
    # GEMM: it is pure overhead that PyTorch-ROCm never pays.
    var copy_samples = List[Float64](capacity=iterations)
    if target.transpose_a:
        for _ in range(warmup):
            _materialize()
        ctx.synchronize()
        for _ in range(iterations):
            ctx.synchronize()
            var start = perf_counter_ns()
            _materialize()
            ctx.synchronize()
            copy_samples.append(Float64(perf_counter_ns() - start) / 1000.0)
        sort(copy_samples)

    var mismatches: Int
    if pattern_check:
        mismatches = _count_pattern_mismatches(target, c_buf, ctx)
    else:
        mismatches = _count_constant_mismatches(target, c_buf, ctx)

    var median = _percentile(samples, 50)
    var result = CaseResult(
        median,
        _percentile(samples, 10),
        _percentile(samples, 90),
        _percentile(copy_samples, 50) if target.transpose_a else 0.0,
        2.0 * Float64(m) * Float64(n) * Float64(k) / (median * 1.0e6),
        mismatches,
    )
    _ = a_buf^
    _ = b_buf^
    _ = c_buf^
    _ = a_contig^
    return result


def _count_constant_mismatches(
    target: GemmCase, c_buf: DeviceBuffer[DType.bfloat16], ctx: DeviceContext
) raises -> Int:
    """Every output must equal K rounded to BF16, for the all-ones operands.

    The FP32 accumulator holds K exactly for every K here, but the BF16 *store*
    has only 8 significand bits, so K is reproduced exactly only when it needs
    at most 8: 768, 2304, 3072 and 49152 do, and K = 50304 does not
    (50304 = 2^15+2^14+2^10+2^7 needs 9, and round-to-nearest-even gives
    50176).  Comparing against the rounded value keeps this an exact equality
    test over the whole output — which is what catches tiles that were never
    written, written twice, or written with a truncated K loop — without
    reporting the unavoidable BF16 rounding of the reference itself as a
    kernel defect.
    """
    var slots = CHECK_BLOCKS * FILL_BLOCK
    var counts = ctx.enqueue_create_buffer[DType.int32](slots)
    ctx.enqueue_function[_count_not_equal](
        counts.unsafe_ptr().as_unsafe_any_origin(),
        c_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        target.m * target.n,
        Float32(target.k).cast[DType.bfloat16]().cast[DType.float32](),
        grid_dim=(CHECK_BLOCKS,),
        block_dim=(FILL_BLOCK,),
    )
    var mismatches = _sum_counts(counts, slots, ctx)
    _ = counts^
    return mismatches


def _count_pattern_mismatches(
    target: GemmCase, c_buf: DeviceBuffer[DType.bfloat16], ctx: DeviceContext
) raises -> Int:
    """Compare every output against the pattern's closed form on the device.

    Unlike the constant check this varies along m, n and k, so it detects
    swapped indices, wrong strides and a transposed operand read with the wrong
    layout.  It needs no host transfer of the output, so it applies at the real
    shapes rather than a shrunken stand-in.
    """
    var slots = CHECK_BLOCKS * FILL_BLOCK
    var counts = ctx.enqueue_create_buffer[DType.int32](slots)
    ctx.enqueue_function[_count_pattern_not_equal](
        counts.unsafe_ptr().as_unsafe_any_origin(),
        c_buf.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        target.m,
        target.n,
        target.k,
        grid_dim=(CHECK_BLOCKS,),
        block_dim=(FILL_BLOCK,),
    )
    var mismatches = _sum_counts(counts, slots, ctx)
    _ = counts^
    return mismatches


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


def _pad(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out += " "
    return out


def _fixed(value: Float64, decimals: Int) -> String:
    var scale = 1.0
    for _ in range(decimals):
        scale *= 10.0
    var rounded = Float64(Int(value * scale + 0.5)) / scale
    return String(rounded)


def main() raises:
    var targets_path = String(
        arg_parse("targets", "harness/nanogpt_train/rocm_gemm_targets.csv")
    )
    var only = String(arg_parse("case", "all"))
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var pattern_check = Int(arg_parse("pattern-check", 0)) != 0
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    var cases = read_targets(targets_path)
    print(
        "case                      role   ",
        "     m      n      k ta tb   mojo_us    rocm_us  ratio  TFLOP/s"
        "   copy_us  correct",
    )
    var mojo_step_us = Float64(0.0)
    var rocm_step_us = Float64(0.0)
    var copy_step_us = Float64(0.0)
    var failed = 0
    var failed_us = Float64(0.0)
    with DeviceContext() as ctx:
        for target in cases:
            if only != "all" and target.label != only:
                continue
            var result = run_case(
                target, warmup, iterations, pattern_check, ctx
            )
            mojo_step_us += result.median_us * Float64(target.calls_per_step)
            rocm_step_us += target.rocm_us * Float64(target.calls_per_step)
            copy_step_us += result.copy_us * Float64(target.calls_per_step)
            print(
                _pad(target.label, 25),
                _pad(target.role, 6),
                _pad(String(target.m), 6),
                _pad(String(target.n), 6),
                _pad(String(target.k), 6),
                Int(target.transpose_a),
                Int(target.transpose_b),
                _pad(_fixed(result.median_us, 2), 10),
                _pad(_fixed(target.rocm_us, 2), 10),
                _pad(_fixed(result.median_us / target.rocm_us, 3), 6),
                _pad(_fixed(result.tflops, 1), 7),
                _pad(_fixed(result.copy_us, 2), 9),
                "pass" if result.mismatches
                == 0 else String(
                    "FAIL:", result.mismatches, "/", target.m * target.n
                ),
            )
            if result.mismatches != 0:
                failed += 1
                failed_us += result.median_us * Float64(target.calls_per_step)
    if rocm_step_us == 0.0:
        raise Error("no case matched --case ", only)
    print()
    print(
        "per-step weighted total: mojo",
        _fixed(mojo_step_us / 1000.0, 3),
        "ms, rocm",
        _fixed(rocm_step_us / 1000.0, 3),
        "ms, ratio",
        _fixed(mojo_step_us / rocm_step_us, 3),
    )
    print(
        "of which transpose materialization:",
        _fixed(copy_step_us / 1000.0, 3),
        "ms (PyTorch-ROCm pays none of this)",
    )
    if failed == 0:
        print(
            "correctness: all cases pass (pattern_check=",
            Int(pattern_check),
            ")",
        )
        return
    # A wrong GEMM has no meaningful speed, so say so loudly and exit nonzero.
    print(
        "correctness: ",
        failed,
        "case(s) FAILED, covering",
        _fixed(failed_us / 1000.0, 3),
        "ms of the per-step total. Their timings are not acceptable results.",
    )
    raise Error("GEMM correctness failure in ", failed, " case(s)")
