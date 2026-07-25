"""Pure-Mojo benchmark for every permute the nanoGPT training step materializes.

Same two-second build/iterate loop as the other harnesses here, and it calls the
production entry point the Python path reaches:
`data_movement_ops._permute_copy`, which is what `TorchMojoTensor.
_materialize_contiguous` invokes for any rank-4-or-less strided source.

The five cases are the five distinct (shape, stride) pairs a step actually asks
for, obtained by logging `_materialize_contiguous`:

    72x  out (48,12,1024,64)  src strides (2359296,64,2304,1)   q/k/v
    48x  out (48,1024,12,64)  src strides (786432,64,65536,1)    y and dq/dk/dv
    12x  out (48,12,1024,64)  src strides (786432,64,768,1)      dy
    24x  out (576,1024,64)    src strides (65536,1,1024)         dO^T, Q^T
    24x  out (576,64,1024)    src strides (65536,1,64)           dV^T, dK^T

The first three have `s3 == 1`: their innermost extent is contiguous in *both*
operands, so they are gathers of 64-element runs rather than transposes. The last
two are genuine batched transposes and take the tiled LDS path.

Three timings per case, so a change is visible rather than asserted:

* `generic` -- the rank-4 element-at-a-time gather, launched directly here. This
  is what every case used before the tiled transpose and the run gather existed.
* `prod` -- whatever `_permute_copy`'s dispatch selects today.
* `rocm` -- PyTorch-ROCm's own `.contiguous()` on the identical shape and
  strides, measured separately with `scripts/rocm_permute_reference.py`.

Correctness is an exact gate over every output element, in two independent
parts. The source is filled with a value that is a function of its own linear
index and never equals `SENTINEL`; the destination is pre-filled with
`SENTINEL`. After the production copy, (a) no output element may still hold the
sentinel -- that catches a tile, run or tail never written -- and (b) every
output element must be bit-identical to the generic kernel's answer for the same
strides, which is the semantic reference. A wrong stride, a swapped axis, a
dropped run or a misaligned vector all move at least one element.

Build and run:
    uv run --no-sync mojo build harness/nanogpt_train/bench_permute_copy.mojo \\
        -I torch_mojo_backend/eager_kernels -o /tmp/bench_permute_copy
    /tmp/bench_permute_copy
    /tmp/bench_permute_copy --sweep=1     # raw run-gather at 8/16/32-byte widths
"""

from std.builtin.sort import sort
from std.collections import List
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.sys.info import size_of
from std.time import perf_counter_ns

from internal_utils import arg_parse

from data_movement_ops import (
    _permute_copy,
    _permute_copy_kernel,
    _run_gather_kernel,
)
from op_utils import GS_THREADS, _enqueue_cached, _gs_blocks, _make_ptr

comptime DT = DType.uint16
comptime BLOCK = 256
comptime REDUCE_BLOCKS = 1024
# The fill never produces this, so it doubles as a "never written" marker.
comptime SENTINEL = 0xFFFF


@__name("bench_permute_fill")
def _fill(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    count: Int,
):
    """`ptr[i] = hash(i) & 0x7FFF`, so no element can equal SENTINEL."""
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while index < count:
        ptr[index] = Scalar[DT]((index * 2654435761) >> 7 & 0x7FFF)
        index += stride


@__name("bench_permute_set")
def _set(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    count: Int,
    value: Int,
):
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while index < count:
        ptr[index] = Scalar[DT](value)
        index += stride


@__name("bench_permute_compare")
def _compare(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    mine: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    reference: UnsafePointer[Scalar[DT], ImmutAnyOrigin],
    count: Int,
):
    """Per-thread tally of (mismatch, still-sentinel) over the whole output."""
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var wrong = 0
    var unwritten = 0
    while index < count:
        var value = mine[index]
        if value != reference[index]:
            wrong += 1
        if Int(value) == SENTINEL:
            unwritten += 1
        index += stride
    var slot = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    counts[2 * slot] = Scalar[DType.int32](wrong)
    counts[2 * slot + 1] = Scalar[DType.int32](unwritten)


def _sum_counts(
    counts: DeviceBuffer[DType.int32], slots: Int, ctx: DeviceContext
) raises -> Tuple[Int, Int]:
    var host = ctx.enqueue_create_host_buffer[DType.int32](2 * slots)
    ctx.enqueue_copy(host, counts)
    ctx.synchronize()
    var wrong = 0
    var unwritten = 0
    for index in range(slots):
        wrong += Int(host[2 * index])
        unwritten += Int(host[2 * index + 1])
    return (wrong, unwritten)


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


def _rpad(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out = " " + out
    return out


def run_case(
    label: String,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    s0: Int,
    s1: Int,
    s2: Int,
    s3: Int,
    src_shift: Int,
    calls_per_step: Float64,
    rocm_us: Float64,
    warmup: Int,
    iterations: Int,
    sweep: Int,
    ctx: DeviceContext,
) raises -> Int:
    var total = d0 * d1 * d2 * d3
    # Widest span the strides can reach, plus room for a deliberate element
    # shift that breaks vector alignment.
    var span = (
        (d0 - 1) * s0 + (d1 - 1) * s1 + (d2 - 1) * s2 + (d3 - 1) * s3 + 1
    )
    var source = ctx.enqueue_create_buffer[DT](span + src_shift)
    var mine = ctx.enqueue_create_buffer[DT](total)
    var reference = ctx.enqueue_create_buffer[DT](total)
    var slots = REDUCE_BLOCKS * BLOCK
    var counts = ctx.enqueue_create_buffer[DType.int32](2 * slots)

    ctx.enqueue_function[_fill](
        source.unsafe_ptr().as_unsafe_any_origin(),
        span + src_shift,
        grid_dim=(max(1, min(ceildiv(span + src_shift, BLOCK), 4096)),),
        block_dim=(BLOCK,),
    )
    var in_addr = Int(source.unsafe_ptr()) + src_shift * size_of[DT]()
    var out_addr = Int(mine.unsafe_ptr())

    # --- the generic rank-4 gather, i.e. the state before the fast paths ----
    @always_inline
    @parameter
    def _generic(dst: Int) raises:
        _enqueue_cached[_permute_copy_kernel[DT]](
            ctx,
            String("bench_permute_generic"),
            _gs_blocks(total),
            1,
            1,
            GS_THREADS,
            _make_out(dst),
            _make_in(in_addr),
            d1,
            d2,
            d3,
            s0,
            s1,
            s2,
            s3,
            total,
        )

    for _ in range(warmup):
        _generic(Int(reference.unsafe_ptr()))
    ctx.synchronize()
    var generic = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _generic(Int(reference.unsafe_ptr()))
        ctx.synchronize()
        generic.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(generic)

    # --- the production dispatch -------------------------------------------
    @always_inline
    @parameter
    def _production() raises:
        _permute_copy[DT](
            out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
        )

    for _ in range(warmup):
        _production()
    ctx.synchronize()
    var prod = List[Float64](capacity=iterations)
    for _ in range(iterations):
        ctx.synchronize()
        var start = perf_counter_ns()
        _production()
        ctx.synchronize()
        prod.append(Float64(perf_counter_ns() - start) / 1000.0)
    sort(prod)

    # Re-run the production copy into a sentinel-filled destination, so an
    # element the dispatch never touches is visible as such.
    ctx.enqueue_function[_set](
        mine.unsafe_ptr().as_unsafe_any_origin(),
        total,
        SENTINEL,
        grid_dim=(max(1, min(ceildiv(total, BLOCK), 4096)),),
        block_dim=(BLOCK,),
    )
    _production()
    ctx.enqueue_function[_compare](
        counts.unsafe_ptr().as_unsafe_any_origin(),
        mine.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        reference.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        total,
        grid_dim=(REDUCE_BLOCKS,),
        block_dim=(BLOCK,),
    )
    var tallies = _sum_counts(counts, slots, ctx)
    var ok = tallies[0] == 0 and tallies[1] == 0

    var prod_us = _percentile(prod, 50)
    var bytes = Float64(2 * total * size_of[DT]())
    print(
        _pad(label, 12),
        _rpad(_fixed(_percentile(generic, 50), 2), 10),
        _rpad(_fixed(prod_us, 2), 10),
        _rpad(_fixed(rocm_us, 2), 9),
        _rpad(_fixed(bytes / prod_us / 1000.0, 1), 9),
        _rpad(_fixed(_percentile(generic, 50) * calls_per_step / 1000.0, 2), 9),
        _rpad(_fixed(prod_us * calls_per_step / 1000.0, 2), 9),
        _rpad(_fixed(rocm_us * calls_per_step / 1000.0, 2), 9),
        _rpad(String(tallies[0]), 8),
        _rpad(String(tallies[1]), 8),
        " pass" if ok else " FAIL",
    )

    if sweep != 0 and s3 == 1:
        _sweep_widths(
            label,
            out_addr,
            in_addr,
            d0,
            d1,
            d2,
            d3,
            s0,
            s1,
            s2,
            total,
            warmup,
            iterations,
            ctx,
        )

    _ = source^
    _ = mine^
    _ = reference^
    _ = counts^
    return 0 if ok else 1


@always_inline
def _make_out(addr: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return _make_ptr[DT](addr).as_unsafe_any_origin()


@always_inline
def _make_in(addr: Int) -> UnsafePointer[Scalar[DT], ImmutAnyOrigin]:
    return _make_ptr[DT](addr).as_unsafe_any_origin().as_immutable()


def _sweep_widths(
    label: String,
    out_addr: Int,
    in_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    s0: Int,
    s1: Int,
    s2: Int,
    total: Int,
    warmup: Int,
    iterations: Int,
    ctx: DeviceContext,
) raises:
    """Time the run gather at every vector width the case admits."""

    @always_inline
    @parameter
    def _one[VEC: Int]() raises:
        if (
            d3 % VEC != 0
            or s0 % VEC != 0
            or s1 % VEC != 0
            or s2 % VEC != 0
            or out_addr % min(16, VEC * size_of[DT]()) != 0
            or in_addr % min(16, VEC * size_of[DT]()) != 0
        ):
            print("   width", VEC, "n/a")
            return

        @always_inline
        @parameter
        def _launch() raises:
            _enqueue_cached[_run_gather_kernel[DT, VEC]](
                ctx,
                String(t"bench_rungather_v{VEC}"),
                _gs_blocks(total // VEC),
                1,
                1,
                GS_THREADS,
                _make_out(out_addr),
                _make_in(in_addr),
                d1,
                d2,
                d3 // VEC,
                s0,
                s1,
                s2,
                total // VEC,
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
        var us = _percentile(samples, 50)
        print(
            "   width",
            _rpad(String(VEC), 3),
            "elems",
            _rpad(_fixed(us, 2), 9),
            "us",
            _rpad(_fixed(Float64(2 * total * size_of[DT]()) / us / 1000.0, 1), 9),
            "GB/s",
        )

    _one[32 // size_of[DT]()]()
    _one[16 // size_of[DT]()]()
    _one[8 // size_of[DT]()]()
    _one[4 // size_of[DT]()]()


def main() raises:
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    var only = String(arg_parse("case", "all"))
    var sweep = Int(arg_parse("sweep", 0))
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")

    print(
        "case          generic_us    prod_us   rocm_us    prod_GB/s"
        " gen_ms/st prod_ms/s rocm_ms/s    wrong unwritten  correct"
    )
    var failures = 0
    with DeviceContext() as ctx:
        # q/k/v: a (48,1024,2304) c_attn output sliced to 768 columns, viewed as
        # (48,1024,12,64) and transposed to (48,12,1024,64).
        if only == "all" or only == "qkv":
            failures += run_case(
                "qkv",
                48,
                12,
                1024,
                64,
                2359296,
                64,
                2304,
                1,
                0,
                72.0,
                55.21,
                warmup,
                iterations,
                sweep,
                ctx,
            )
        # y.transpose(1,2).contiguous(), and the same shape for dq/dk/dv.
        if only == "all" or only == "heads_out":
            failures += run_case(
                "heads_out",
                48,
                1024,
                12,
                64,
                786432,
                64,
                65536,
                1,
                0,
                48.0,
                55.77,
                warmup,
                iterations,
                sweep,
                ctx,
            )
        # dy: a contiguous (48,1024,768) grad viewed and transposed.
        if only == "all" or only == "dy":
            failures += run_case(
                "dy",
                48,
                12,
                1024,
                64,
                786432,
                64,
                768,
                1,
                0,
                12.0,
                55.05,
                warmup,
                iterations,
                sweep,
                ctx,
            )
        # The two genuine batched transposes of the SDPA backward.
        if only == "all" or only == "bmm_seq":
            failures += run_case(
                "bmm_seq",
                1,
                576,
                1024,
                64,
                0,
                65536,
                1,
                1024,
                0,
                24.0,
                161.93,
                warmup,
                iterations,
                sweep,
                ctx,
            )
        if only == "all" or only == "bmm_head":
            failures += run_case(
                "bmm_head",
                1,
                576,
                64,
                1024,
                0,
                65536,
                1,
                64,
                0,
                24.0,
                256.14,
                warmup,
                iterations,
                sweep,
                ctx,
            )
        # Edge regimes for the gate, not part of the step.
        #  - run_12: d3 = 12 is a multiple of 4 but not of 8 or 16, so the
        #    widest admissible vector is 8 bytes.
        #  - run_63: d3 = 63 admits no vector at all; the copy must decline to
        #    the general kernel and still be exact.
        #  - shifted: the same shape as qkv with the source displaced by one
        #    element, so every vector width is misaligned.
        if only == "all" or only == "run_12":
            failures += run_case(
                "run_12", 5, 7, 129, 12, 10836, 12, 84, 1, 0, 0.0, 0.0,
                warmup, iterations, sweep, ctx,
            )
        if only == "all" or only == "run_63":
            failures += run_case(
                "run_63", 3, 11, 101, 63, 69993, 63, 693, 1, 0, 0.0, 0.0,
                warmup, iterations, sweep, ctx,
            )
        if only == "all" or only == "shifted":
            failures += run_case(
                "shifted", 8, 12, 1024, 64, 2359296, 64, 2304, 1, 1, 0.0, 0.0,
                warmup, iterations, sweep, ctx,
            )
    if failures != 0:
        raise Error(failures, " case(s) FAILED - their timings mean nothing")
    print("correctness: all cases pass")
