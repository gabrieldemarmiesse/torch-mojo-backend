# ===----------------------------------------------------------------------=== #
# Per-segment and per-row statistics that are neither a scalar reduction nor
# a scan:
#
#   ModeRuns           aten::mode over rows the sort family already sorted:
#                      the longest run of equal values (the smallest value on
#                      a tie) and the largest original index holding it.
#   SegmentReduce      aten::segment_reduce (lengths or offsets) and its
#   SegmentReduceBwd   backward, one thread per (outer, segment, inner)
#                      output, as SegmentReduce.cu's general kernels.
#   LinearCombination  aten::_compute_linear_combination: out[i, ...] +=
#                      sum_j coefficients[i, j] * input[j, ...].
#
# Arithmetic follows the CUDA kernels named above, including where they
# compute in the element dtype: segment reduce and the linear combination
# keep their running value in `scalar_t` and so round every step (a half
# operand computes each step in float32 and rounds, as c10::Half's operators
# do), and `x / length` rounds the length to the element dtype first (c10's
# `Half / int64_t`). float32 and float64 products are fused into the running
# sum (nvcc contracts `out += a * b` into an FMA).
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_down
from max.gpu.sync import barrier
from std.math import ceildiv, fma, sqrt
from std.memory import bitcast, stack_allocation
from std.bit import count_trailing_zeros
from std.sys.info import has_accelerator, size_of
from std.utils.numerics import isnan, nan
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _spec_dispatcher7,
    _spec_dispatcher9,
    _spec_dispatcher11,
    _spec_dispatcher13,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime STATS_THREADS = 256
comptime STATS_MAX_BLOCKS = 65535

# bool rides uint8 storage.
comptime MODE_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
]

comptime SEGMENT_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]

comptime LINCOMB_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
]

# segment_reduce's reductions (ATen's ReductionType order).
comptime RED_MAX = 0
comptime RED_MEAN = 1
comptime RED_MIN = 2
comptime RED_SUM = 3
comptime RED_PROD = 4


@always_inline
def _grid(total: Int) -> Int:
    return max(1, min(ceildiv(total, STATS_THREADS), STATS_MAX_BLOCKS))


# ---------------------------------------------------------------------------
# element-dtype arithmetic (c10::Half / c10::BFloat16 operators: compute in
# float32, round once)
# ---------------------------------------------------------------------------


@always_inline
def _add[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float16 or dt == DType.bfloat16:
        return (a.cast[DType.float32]() + b.cast[DType.float32]()).cast[dt]()
    else:
        return a + b


@always_inline
def _mul[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float16 or dt == DType.bfloat16:
        return (a.cast[DType.float32]() * b.cast[DType.float32]()).cast[dt]()
    else:
        return a * b


@always_inline
def _div[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float16 or dt == DType.bfloat16:
        return (a.cast[DType.float32]() / b.cast[DType.float32]()).cast[dt]()
    else:
        return a / b


@always_inline
def _mul_add[
    dt: DType
](acc: Scalar[dt], a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """`acc += a * b` as CUDA compiles it for scalar_t."""
    comptime if dt == DType.float32 or dt == DType.float64:
        return fma(a, b, acc)
    elif dt == DType.float16 or dt == DType.bfloat16:
        return _add[dt](acc, _mul[dt](a, b))
    else:
        return acc + a * b


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(STATS_THREADS))
)
@__name(t"mode_runs_{dtype}")
def _mode_runs_kernel[
    dtype: DType
](
    out_v: Pointer[Scalar[dtype], MutAnyOrigin],
    out_i: Pointer[Scalar[DType.int64], MutAnyOrigin],
    sorted_v: Pointer[Scalar[dtype], ImmutAnyOrigin],
    sorted_i: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
    first_arg: Int64,
):
    """One thread per sorted row. A run ends where the next value differs
    (`!=`, so every NaN is a run of its own, as CUDA's segment flags are);
    the first longest run wins (the smallest value). The stable sort orders
    a run by original index, so its last element is the LARGEST index of the
    mode (CUDA's fused kernel, rows up to 2048) and its first the SMALLEST
    (`first_arg`: CUDA's thrust fallback for longer rows, `thrust::find` on
    the stably sorted row)."""
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var first = Int(first_arg) != 0
    var r = Int(block_idx.x) * STATS_THREADS + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * STATS_THREADS
    while r < rows:
        var base = r * n
        var best_len = 0
        var best_end = 0
        var run = 0
        for j in range(n):
            run += 1
            var ends = j == n - 1
            if not ends:
                ends = (
                    sorted_v[unsafe_offset=base + j]
                    != sorted_v[unsafe_offset=base + j + 1]
                )
            if ends:
                if run > best_len:
                    best_len = run
                    best_end = j
                run = 0
        var pick = best_end - best_len + 1 if first else best_end
        out_v[unsafe_offset=r] = sorted_v[unsafe_offset=base + best_end]
        out_i[unsafe_offset=r] = sorted_i[unsafe_offset=base + pick]
        r += stride


# ---------------------------------------------------------------------------
# segment_reduce
# ---------------------------------------------------------------------------


@always_inline
def _len_as[dt: DType](length: Int64) -> Scalar[dt]:
    """`x / length` converts the int64 length to scalar_t first."""
    comptime if dt == DType.float16 or dt == DType.bfloat16:
        return Float32(length).cast[dt]()
    else:
        return Scalar[dt](length)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(STATS_THREADS))
)
@__name(t"segment_reduce_{dtype}")
def _segment_reduce_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    data: Pointer[Scalar[dtype], ImmutAnyOrigin],
    lengths: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    offsets: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    reduction_arg: Int64,
    outer_arg: Int64,
    segments_arg: Int64,
    inner_arg: Int64,
    axis_size_arg: Int64,
    initial_set_arg: Int64,
    initial: Scalar[dtype],
):
    """`segment_reduce_forward_kernel` (SegmentReduce.cu): the running value
    starts at `initial` (or the reduction's identity) and is kept in
    scalar_t; mean divides by the length unless the sum is NaN, and an empty
    segment without an initial value is NaN."""
    var reduction = Int(reduction_arg)
    var outer = Int(outer_arg)
    var segments = Int(segments_arg)
    var inner = Int(inner_arg)
    var axis_size = Int(axis_size_arg)
    var initial_set = Int(initial_set_arg) != 0
    var total = outer * segments * inner
    var idx = Int(block_idx.x) * STATS_THREADS + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * STATS_THREADS
    while idx < total:
        var row = idx // inner
        var lane = idx - row * inner
        var o = row // segments
        var s = row - o * segments
        var start = Int(offsets[unsafe_offset=o * (segments + 1) + s])
        var end = Int(offsets[unsafe_offset=o * (segments + 1) + s + 1])
        var v = initial
        for j in range(start, end):
            var x = data[unsafe_offset=(o * axis_size + j) * inner + lane]
            # std::max / std::min after the NaN test: a NaN running value
            # sticks. Tested explicitly, since the build may assume no NaN.
            if reduction == RED_MAX:
                if isnan(x):
                    v = x
                elif not isnan(v):
                    v = x if v < x else v
            elif reduction == RED_MIN:
                if isnan(x):
                    v = x
                elif not isnan(v):
                    v = x if x < v else v
            elif reduction == RED_PROD:
                v = _mul[dtype](v, x)
            else:
                v = _add[dtype](v, x)
        var length = lengths[unsafe_offset=o * segments + s]
        if reduction == RED_MEAN:
            if length == 0 and not initial_set:
                v = nan[dtype]()
            elif length > 0 and not isnan(v):
                v = _div[dtype](v, _len_as[dtype](length))
        out_ptr[unsafe_offset=idx] = v
        idx += stride


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(STATS_THREADS))
)
@__name(t"segment_reduce_backward_{dtype}")
def _segment_reduce_backward_kernel[
    dtype: DType
](
    grad_in: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    output: Pointer[Scalar[dtype], ImmutAnyOrigin],
    data: Pointer[Scalar[dtype], ImmutAnyOrigin],
    lengths: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    offsets: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    reduction_arg: Int64,
    outer_arg: Int64,
    segments_arg: Int64,
    inner_arg: Int64,
    axis_size_arg: Int64,
    initial_prod: Scalar[dtype],
):
    """`segment_reduce_backward_kernel` (SegmentReduce.cu) into a zeroed
    `grad_in`, including its quirks: max/min split the gradient only over
    the entries whose stored gradient is positive, and prod recomputes the
    exclusive product (seeded with `initial`) for a zero or NaN entry."""
    var reduction = Int(reduction_arg)
    var outer = Int(outer_arg)
    var segments = Int(segments_arg)
    var inner = Int(inner_arg)
    var axis_size = Int(axis_size_arg)
    var total = outer * segments * inner
    var idx = Int(block_idx.x) * STATS_THREADS + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * STATS_THREADS
    while idx < total:
        var row = idx // inner
        var lane = idx - row * inner
        var o = row // segments
        var s = row - o * segments
        var length = lengths[unsafe_offset=o * segments + s]
        if length == 0:
            idx += stride
            continue
        var start = Int(offsets[unsafe_offset=o * (segments + 1) + s])
        var end = Int(offsets[unsafe_offset=o * (segments + 1) + s + 1])
        var g = grad[unsafe_offset=idx]
        var y = output[unsafe_offset=idx]
        if reduction == RED_MAX or reduction == RED_MIN:
            var counter = 0
            for j in range(start, end):
                var di = (o * axis_size + j) * inner + lane
                var x = data[unsafe_offset=di]
                if isnan(x) or x == y:
                    grad_in[unsafe_offset=di] = g
                    counter += 1
            if counter >= 2:
                var cnt = _len_as[dtype](Int64(counter))
                for j in range(start, end):
                    var di = (o * axis_size + j) * inner + lane
                    var gi = grad_in[unsafe_offset=di]
                    if gi > Scalar[dtype](0):
                        grad_in[unsafe_offset=di] = _div[dtype](gi, cnt)
        elif reduction == RED_MEAN:
            var gv = _div[dtype](g, _len_as[dtype](length))
            for j in range(start, end):
                grad_in[unsafe_offset=(o * axis_size + j) * inner + lane] = gv
        elif reduction == RED_SUM:
            for j in range(start, end):
                grad_in[unsafe_offset=(o * axis_size + j) * inner + lane] = g
        else:
            var gy = _mul[dtype](g, y)
            for j in range(start, end):
                var di = (o * axis_size + j) * inner + lane
                var x = data[unsafe_offset=di]
                if isnan(x) or x == Scalar[dtype](0):
                    var excl = initial_prod
                    for k in range(start, end):
                        if k != j:
                            excl = _mul[dtype](
                                excl,
                                data[
                                    unsafe_offset=(o * axis_size + k) * inner
                                    + lane
                                ],
                            )
                    grad_in[unsafe_offset=di] = _mul[dtype](g, excl)
                else:
                    grad_in[unsafe_offset=di] = _div[dtype](gy, x)
        idx += stride


# ---------------------------------------------------------------------------
# _compute_linear_combination
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(STATS_THREADS))
)
@__name(t"linear_combination_{dtype}")
def _linear_combination_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    inp: Pointer[Scalar[dtype], ImmutAnyOrigin],
    coeff: Pointer[Scalar[dtype], ImmutAnyOrigin],
    m_arg: Int64,
    n_arg: Int64,
    rest_arg: Int64,
):
    """One thread per output element (i, k): `out[i, k] += in[j, k] *
    coeff[i, j]` for j in order, accumulated in the output itself."""
    var m = Int(m_arg)
    var n = Int(n_arg)
    var rest = Int(rest_arg)
    var total = m * rest
    var idx = Int(block_idx.x) * STATS_THREADS + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * STATS_THREADS
    while idx < total:
        var i = idx // rest
        var k = idx - i * rest
        var acc = out_ptr[unsafe_offset=idx]
        for j in range(n):
            acc = _mul_add[dtype](
                acc,
                inp[unsafe_offset=j * rest + k],
                coeff[unsafe_offset=i * n + j],
            )
        out_ptr[unsafe_offset=idx] = acc
        idx += stride


# ---------------------------------------------------------------------------
# Welford moments (var_mean / std_mean, and float64 var / std):
# SharedReduceOps.h `WelfordOps`, element updates and Chan merges, accumulated
# in float32 (float64 for float64) with the count as a float, as CUDA's
# `WelfordData<acc_scalar_t, index_t>` keeps it. Neither the mean nor M2 is
# ever a sum that can overflow where CUDA's do not.
# ---------------------------------------------------------------------------

comptime WELFORD_THREADS = 256

# Splitting a row across blocks pays off only once each thread of each split
# has a few elements to walk; below this many elements per split block the
# merge launch costs more than it saves.
comptime WELFORD_MIN_PER_SPLIT = 4096
# ...and for the strided (column) kernel, rows of the axis per split lane.
comptime WELFORD_MIN_ROWS_PER_SPLIT = 16
# Widest column tile of the strided kernel; the rest of the block's threads
# split the reduce axis (row lanes).
comptime WELFORD_COLS_TILE = 32
# Splits up to this many are merged by one thread per output.
comptime WELFORD_SERIAL_MERGE_MAX = 256
# A row gets the fewest threads (a power of two, at least WELFORD_MIN_TPR)
# leaving each WELFORD_MIN_UNITS_PER_THREAD vector loads; 16 measured best
# on an H100 (8 left 4096-wide rows 15-20% slower).
comptime WELFORD_MIN_TPR = 8
comptime WELFORD_MIN_UNITS_PER_THREAD = 16
# ...or scalar loads, when the rows are not aligned to the vector (357 x 789
# float32: 8 per thread 6.0 us, 16 per thread 8.8 us).
comptime WELFORD_MIN_SCALARS_PER_THREAD = 8

# Stage-1 blocks targeted when there are too few rows or columns to fill
# the device, per SM. Rows: the 75-register vector kernel keeps 3 blocks of
# 256 resident on an H100, so 3 is one full wave (8 left a partial third
# wave: 16M-element var_mean 54 -> 47 us). Columns (36 registers): 8 (3 left
# (8, 1024, 1024) over dim 1 at 1.21x CUDA, 8 at 1.05x). Not swept on AMD
# or Apple.
comptime WELFORD_BLOCKS_PER_SM = 3
comptime WELFORD_COLS_BLOCKS_PER_SM = 8

comptime WELFORD_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]


@always_inline
def _wacc[dt: DType]() -> DType:
    comptime if dt == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def _welford_combine[
    acc: DType
](
    mut mean: Scalar[acc],
    mut m2: Scalar[acc],
    mut nf: Scalar[acc],
    b_mean: Scalar[acc],
    b_m2: Scalar[acc],
    b_nf: Scalar[acc],
):
    """`WelfordOps::combine` (Chan et al.)."""
    if nf == 0:
        mean = b_mean
        m2 = b_m2
        nf = b_nf
        return
    if b_nf == 0:
        return
    var delta = b_mean - mean
    var new_count = nf + b_nf
    var nb_over_n = b_nf / new_count
    m2 = m2 + b_m2 + delta * delta * nf * nb_over_n
    mean = mean + delta * nb_over_n
    nf = new_count


@always_inline
def _shfl_down[acc: DType](v: Scalar[acc], off: UInt32) -> Scalar[acc]:
    """`shuffle_down`, float64 moved as its uint64 bits (no f64 shuffle)."""
    comptime if acc == DType.float64:
        return bitcast[acc, 1](shuffle_down(bitcast[DType.uint64, 1](v), off))
    else:
        return shuffle_down(v, off)


@always_inline
def _welford_warp[
    acc: DType
](width: Int, mut mean: Scalar[acc], mut m2: Scalar[acc], mut nf: Scalar[acc],):
    """Merge each aligned run of `width` lanes (a power of two; a whole warp
    when >= WARP_SIZE) into its first lane: shuffles with offsets below
    `width` only, so a run never reads its neighbour."""
    comptime for k in range(count_trailing_zeros(WARP_SIZE)):
        comptime off = UInt32(WARP_SIZE >> (k + 1))
        if Int(off) >= width:
            continue
        var b_mean = _shfl_down[acc](mean, off)
        var b_m2 = _shfl_down[acc](m2, off)
        var b_nf = _shfl_down[acc](nf, off)
        _welford_combine[acc](mean, m2, nf, b_mean, b_m2, b_nf)


@always_inline
def _welford_project[
    dtype: DType, acc: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    r: Int,
    mean: Scalar[acc],
    m2: Scalar[acc],
    nf: Scalar[acc],
    correction: Scalar[acc],
    take_sqrt: Bool,
):
    """`WelfordOps::project`: var = m2 / max(n - correction, 0), its root for
    std, rounded once to the output dtype."""
    # `nf > correction ? nf - correction : 0`: a NaN correction (every
    # comparison false) divides by 0, as on CUDA.
    var divisor = nf - correction if nf > correction else Scalar[acc](0)
    var v = m2 / divisor
    if take_sqrt:
        v = sqrt(v)
    out_ptr[unsafe_offset=r] = v.cast[dtype]()
    if Int(mean_ptr) != 0:
        mean_ptr[unsafe_offset=r] = mean.cast[dtype]()


# Independent Welford states per thread: the element update is a dependent
# chain through a division, so one state per thread is latency-bound; CUDA's
# reduction keeps several accumulators per thread for the same reason.
comptime WELFORD_ILP = 4
# The vectorized rows path: VEC_ILP loads of `_welford_vec` lanes in flight,
# each lane its own state. More states than 8 per thread spilled the H100
# register budget (169 registers, 12% occupancy at 4 x 4 for float32).
comptime WELFORD_VEC_ILP = 2


def _welford_vec[dtype: DType]() -> Int:
    """Lanes per load of the rows kernel: 16 bytes, at most 4 lanes."""
    return min(16 // size_of[dtype](), 4)


@always_inline
def _welford_update[
    acc: DType
](
    x: Scalar[acc],
    mut mean: Scalar[acc],
    mut m2: Scalar[acc],
    mut nf: Scalar[acc],
):
    """`WelfordOps::reduce`."""
    nf += 1
    var delta = x - mean
    mean = mean + delta / nf
    m2 = m2 + delta * (x - mean)


@always_inline
def _welford_vec_update[
    acc: DType, V: Int
](
    x: SIMD[acc, V],
    mut mean: SIMD[acc, V],
    mut m2: SIMD[acc, V],
    mut nf: Scalar[acc],
):
    """`WelfordOps::reduce` on V independent states sharing one count: one
    division for V > 1 lanes (delta * (1 / n), within an ulp of CUDA's
    delta / n), where V IEEE divisions cost registers and issue slots."""
    nf += 1
    var delta = x - mean
    comptime if V == 1:
        mean = mean + delta / nf
    else:
        mean = mean + delta * SIMD[acc, V](1 / nf)
    m2 = m2 + delta * (x - mean)


@always_inline
def _lanes_of[
    acc: DType, W: Int, //, V: Int, offset: Int
](v: SIMD[acc, W]) -> SIMD[acc, V]:
    """`v.slice[V, offset=offset]()` element by element (Metal has no
    llvm.vector.extract / insert)."""
    var r = SIMD[acc, V](0)
    comptime for e in range(V):
        r[e] = v[offset + e]
    return r


@always_inline
def _set_lanes[
    acc: DType, W: Int, V: Int, //, offset: Int
](mut v: SIMD[acc, W], x: SIMD[acc, V]):
    """`v = v.insert[offset=offset](x)` element by element."""
    comptime for e in range(V):
        v[offset + e] = x[e]


@always_inline
def _welford_lanes[
    acc: DType, V: Int
](mean: SIMD[acc, V], m2: SIMD[acc, V], nf: Scalar[acc]) -> Tuple[
    Scalar[acc], Scalar[acc], Scalar[acc]
]:
    """Chan-merge V states of equal count `nf` by halving: with equal counts
    nb / n is exactly 0.5, so the tree needs no division."""
    comptime if V == 1:
        return (mean[0], m2[0], nf)
    else:
        comptime H = V // 2
        var a_mean = _lanes_of[H, 0](mean)
        var b_mean = _lanes_of[H, H](mean)
        var delta = b_mean - a_mean
        var m = a_mean + delta * 0.5
        var q = (
            _lanes_of[H, 0](m2)
            + _lanes_of[H, H](m2)
            + delta * delta * SIMD[acc, H](nf) * 0.5
        )
        return _welford_lanes[acc, H](m, q, nf + nf)


@always_inline
def _welford_group[
    acc: DType
](
    tid: Int,
    tpr: Int,
    mut mean: Scalar[acc],
    mut m2: Scalar[acc],
    mut nf: Scalar[acc],
):
    """Merge each group of `tpr` threads (a power of two) into its first
    thread: shuffles, then (a group wider than a warp) that thread merges
    its group's warps in order. Every thread must call it; safe to call
    again."""
    comptime NW = WELFORD_THREADS // WARP_SIZE
    var sm = stack_allocation[NW, acc, address_space=AddressSpace.SHARED]()
    var sq = stack_allocation[NW, acc, address_space=AddressSpace.SHARED]()
    var sn = stack_allocation[NW, acc, address_space=AddressSpace.SHARED]()
    _welford_warp[acc](tpr, mean, m2, nf)
    var wpg = tpr // WARP_SIZE
    if wpg <= 1:
        return
    var w = tid // WARP_SIZE
    barrier()  # the previous round's partials are read
    if tid % WARP_SIZE == 0:
        sm[unsafe_offset=w] = mean
        sq[unsafe_offset=w] = m2
        sn[unsafe_offset=w] = nf
    barrier()
    if tid % tpr == 0:
        for k in range(1, wpg):
            _welford_combine[acc](
                mean,
                m2,
                nf,
                sm[unsafe_offset=w + k],
                sq[unsafe_offset=w + k],
                sn[unsafe_offset=w + k],
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WELFORD_THREADS))
)
@__name(t"welford_rows_{dtype}_v{vec}_s{segmented}")
def _welford_rows_kernel[
    dtype: DType, vec: Bool, segmented: Bool
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: Pointer[Scalar[_wacc[dtype]()], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
    seg_arg: Int64,
    splits_arg: Int64,
    tpr_arg: Int64,
    correction: Scalar[_wacc[dtype]()],
    take_sqrt_arg: Int64,
):
    """Row r's n elements are n / seg contiguous segments of seg elements,
    segment s at (s * rows + r) * seg (seg == n: plain contiguous rows; seg
    < n: a kept dim between two reduced groups, NCHW over (0, 2, 3)).

    A group of `tpr` threads (WELFORD_THREADS / tpr groups per block) owns
    one (row, split) shard at a time: with `vec` (every segment aligned to
    a V-lane load) WELFORD_VEC_ILP states of V lanes per thread, lanes merged
    division-free, else WELFORD_ILP scalar states; then the group merges.
    `vec` and `segmented` are parameters, not branches: the segment
    arithmetic alone took the plain kernel from 64 to 106 registers. One split writes the result;
    several write (mean, m2, n) partials for `_welford_merge_kernel`."""
    comptime acc = _wacc[dtype]()
    comptime V = _welford_vec[dtype]()
    comptime VB = V * size_of[dtype]()  # bytes per load
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var seg = Int(seg_arg)
    var splits = Int(splits_arg)
    var tpr = Int(tpr_arg)
    var tid = Int(thread_idx.x)
    var gt = tid % tpr  # thread within the group
    var groups = WELFORD_THREADS // tpr
    var split = Int(block_idx.y)
    comptime W = V if vec else 1  # elements per unit
    var nu = n // W
    var segu = seg // W
    var chunk = ceildiv(nu, splits)
    var u0 = split * chunk
    var u1 = min(nu, u0 + chunk)
    var step = tpr * WELFORD_ILP
    # Every thread runs every round (the group merge has barriers); a group
    # past the last row reduces nothing.
    var r0 = Int(block_idx.x) * groups
    while r0 < rows:
        var r = r0 + tid // tpr
        var live = r < rows
        var tm = Scalar[acc](0)
        var tq = Scalar[acc](0)
        var tn = Scalar[acc](0)
        comptime if vec:
            # WELFORD_VEC_ILP states of V lanes, state k at [k*V, k*V+V).
            comptime ILP = WELFORD_VEC_ILP
            var means = SIMD[acc, V * ILP](0)
            var m2s = SIMD[acc, V * ILP](0)
            var nfs = SIMD[acc, ILP](0)
            var vstep = tpr * ILP
            var j = u0 + gt
            while live and j < u1:
                comptime for k in range(ILP):
                    var jj = j + k * tpr
                    if jj < u1:
                        var off = (r * nu + jj) * V
                        comptime if segmented:  # the host keeps nu < 2**31
                            var q = Int(UInt32(jj) // UInt32(segu))
                            off = ((q * rows + r) * segu + jj - q * segu) * V
                        var x = in_ptr.unsafe_load[width=V, alignment=VB](
                            off
                        ).cast[acc]()
                        var mk = _lanes_of[V, k * V](means)
                        var qk = _lanes_of[V, k * V](m2s)
                        var ck = nfs[k]
                        _welford_vec_update[acc, V](x, mk, qk, ck)
                        _set_lanes[k * V](means, mk)
                        _set_lanes[k * V](m2s, qk)
                        nfs[k] = ck
                j += vstep
            comptime for k in range(ILP):
                var t = _welford_lanes[acc, V](
                    _lanes_of[V, k * V](means),
                    _lanes_of[V, k * V](m2s),
                    nfs[k],
                )
                _welford_combine[acc](tm, tq, tn, t[0], t[1], t[2])
        else:
            var means = SIMD[acc, WELFORD_ILP](0)
            var m2s = SIMD[acc, WELFORD_ILP](0)
            var nfs = SIMD[acc, WELFORD_ILP](0)
            var j = u0 + gt
            while live and j < u1:
                comptime for k in range(WELFORD_ILP):
                    var jj = j + k * tpr
                    if jj < u1:
                        var off = r * n + jj
                        comptime if segmented:
                            var q = Int(UInt32(jj) // UInt32(seg))
                            off = (q * rows + r) * seg + jj - q * seg
                        var mu = means[k]
                        var qu = m2s[k]
                        var cu = nfs[k]
                        _welford_update[acc](
                            in_ptr[unsafe_offset=off].cast[acc](), mu, qu, cu
                        )
                        means[k] = mu
                        m2s[k] = qu
                        nfs[k] = cu
                j += step
            comptime for k in range(WELFORD_ILP):
                _welford_combine[acc](tm, tq, tn, means[k], m2s[k], nfs[k])
        _welford_group[acc](tid, tpr, tm, tq, tn)
        if live and gt == 0:
            if splits == 1:
                _welford_project[dtype, acc](
                    out_ptr,
                    mean_ptr,
                    r,
                    tm,
                    tq,
                    tn,
                    correction,
                    Int(take_sqrt_arg) != 0,
                )
            else:
                var w = (r * splits + split) * 3
                ws_ptr[unsafe_offset=w] = tm
                ws_ptr[unsafe_offset=w + 1] = tq
                ws_ptr[unsafe_offset=w + 2] = tn
        r0 += Int(grid_dim.x) * groups


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WELFORD_THREADS))
)
@__name(t"welford_cols_{dtype}")
def _welford_cols_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: Pointer[Scalar[_wacc[dtype]()], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    n_arg: Int64,
    inner_arg: Int64,
    splits_arg: Int64,
    cw_arg: Int64,
    correction: Scalar[_wacc[dtype]()],
    take_sqrt_arg: Int64,
):
    """Strided reduce axis, (outer, n, inner) with inner > 1. A block is a
    tile of `cw` adjacent columns (consecutive threads read consecutive
    addresses) by WELFORD_THREADS / cw row lanes: lane l walks rows l,
    l + lanes, ... of its split's shard with ILP states (WELFORD_ILP, doubled for 2-byte dtypes), then the
    first lane merges the column's lanes in order."""
    comptime acc = _wacc[dtype]()
    # 2-byte elements need twice the loads in flight per thread.
    comptime ILP = WELFORD_ILP * (2 if size_of[dtype]() < 4 else 1)
    var n = Int(n_arg)
    var inner = Int(inner_arg)
    var splits = Int(splits_arg)
    var cw = Int(cw_arg)
    var lanes = WELFORD_THREADS // cw
    var tid = Int(thread_idx.x)
    var c = tid % cw
    var lane = tid // cw
    var tiles = ceildiv(inner, cw)
    var blk = Int(block_idx.x)
    var o = blk // tiles
    var i = (blk % tiles) * cw + c
    var live = i < inner
    var split = Int(block_idx.y)
    var chunk = ceildiv(n, splits)
    var r0 = split * chunk
    var r1 = min(n, r0 + chunk)
    var base = o * n * inner + i
    var mean = SIMD[acc, ILP](0)
    var m2 = SIMD[acc, ILP](0)
    var nf = SIMD[acc, ILP](0)
    var r = r0 + lane
    var step = lanes * ILP
    # Unconditional ILP bodies (the loads issue together), then the tail.
    while live and r + (ILP - 1) * lanes < r1:
        comptime for u in range(ILP):
            var mu = mean[u]
            var qu = m2[u]
            var cu = nf[u]
            _welford_update[acc](
                in_ptr[unsafe_offset=base + (r + u * lanes) * inner].cast[
                    acc
                ](),
                mu,
                qu,
                cu,
            )
            mean[u] = mu
            m2[u] = qu
            nf[u] = cu
        r += step
    var tm = mean[0]
    var tq = m2[0]
    var tn = nf[0]
    comptime for u in range(1, ILP):
        _welford_combine[acc](tm, tq, tn, mean[u], m2[u], nf[u])
    while live and r < r1:
        # `WelfordOps::reduce`, not a merge of a seeded singleton: its m2
        # comes from x - mean, which makes an inf or NaN element's m2 NaN.
        _welford_update[acc](
            in_ptr[unsafe_offset=base + r * inner].cast[acc](), tm, tq, tn
        )
        r += lanes
    var sm = stack_allocation[
        WELFORD_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var sq = stack_allocation[
        WELFORD_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var sn = stack_allocation[
        WELFORD_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    if lanes > 1:
        sm[unsafe_offset=tid] = tm
        sq[unsafe_offset=tid] = tq
        sn[unsafe_offset=tid] = tn
        barrier()
    if not live or lane != 0:
        return
    for l in range(1, lanes):
        var t = l * cw + c
        _welford_combine[acc](
            tm,
            tq,
            tn,
            sm[unsafe_offset=t],
            sq[unsafe_offset=t],
            sn[unsafe_offset=t],
        )
    var out_index = o * inner + i
    if splits == 1:
        _welford_project[dtype, acc](
            out_ptr,
            mean_ptr,
            out_index,
            tm,
            tq,
            tn,
            correction,
            Int(take_sqrt_arg) != 0,
        )
    else:
        var w = (out_index * splits + split) * 3
        ws_ptr[unsafe_offset=w] = tm
        ws_ptr[unsafe_offset=w + 1] = tq
        ws_ptr[unsafe_offset=w + 2] = tn


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WELFORD_THREADS))
)
@__name(t"welford_merge_{dtype}")
def _welford_merge_kernel[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: Pointer[Scalar[_wacc[dtype]()], MutAnyOrigin],
    rows_arg: Int64,
    splits_arg: Int64,
    tpr_arg: Int64,
    correction: Scalar[_wacc[dtype]()],
    take_sqrt_arg: Int64,
):
    """A group of `tpr` threads per output (1: the thread merges its
    splits' partials in order; else the group's threads merge strided
    partials, then the group merges)."""
    comptime acc = _wacc[dtype]()
    var rows = Int(rows_arg)
    var splits = Int(splits_arg)
    var tpr = Int(tpr_arg)
    var tid = Int(thread_idx.x)
    var g = tid % tpr
    var groups = WELFORD_THREADS // tpr
    var r0 = Int(block_idx.x) * groups
    while r0 < rows:
        var r = r0 + tid // tpr
        var mean = Scalar[acc](0)
        var m2 = Scalar[acc](0)
        var nf = Scalar[acc](0)
        var k = g
        while r < rows and k < splits:
            var w = (r * splits + k) * 3
            _welford_combine[acc](
                mean,
                m2,
                nf,
                ws_ptr[unsafe_offset=w],
                ws_ptr[unsafe_offset=w + 1],
                ws_ptr[unsafe_offset=w + 2],
            )
            k += tpr
        if tpr > 1:
            _welford_group[acc](tid, tpr, mean, m2, nf)
        if r < rows and g == 0:
            _welford_project[dtype, acc](
                out_ptr,
                mean_ptr,
                r,
                mean,
                m2,
                nf,
                correction,
                Int(take_sqrt_arg) != 0,
            )
        r0 += Int(grid_dim.x) * groups


def _welford_go(
    out_o: Arg,
    mean_o: Arg,
    in_o: Arg,
    outer_o: Arg,
    n_o: Arg,
    inner_o: Arg,
    seg_o: Arg,
    correction_o: Arg,
    take_sqrt_o: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    """Slots: output (var or std), mean output (0 for none), contiguous
    input viewed as (outer, n, inner), outer, n, inner, seg (inner == 1: a
    row is n / seg segments `outer * seg` apart, see `_welford_rows_kernel`),
    correction (float64 bits), take_sqrt, dtype code, context."""
    var dtype = _raw_dtype_int(dtype_o)
    var outer = _raw_int(outer_o)
    var n = _raw_int(n_o)
    var inner = _raw_int(inner_o)
    var seg = _raw_int(seg_o)
    var outputs = outer * inner
    if outputs == 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        var ctx = _raw_ctx(ctx_o)
        comptime for dt in WELFORD_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    comptime acc = _wacc[dt]()
                    var target = WELFORD_BLOCKS_PER_SM * _device_sm_count(ctx)
                    # Threads per row: fewer than a block when the row is
                    # short, so each walks a few vectors (CUDA likewise
                    # sizes block.x to the row).
                    var vec = (
                        seg % _welford_vec[dt]() == 0
                        and _raw_int(in_o)
                        % (_welford_vec[dt]() * size_of[dt]())
                        == 0
                    )
                    var units = n // _welford_vec[dt]() if vec else n
                    var per_thread = (
                        WELFORD_MIN_UNITS_PER_THREAD if vec else WELFORD_MIN_SCALARS_PER_THREAD
                    )
                    var tpr = WELFORD_THREADS
                    while tpr > WELFORD_MIN_TPR and tpr * per_thread > units:
                        tpr //= 2
                    var base_blocks = ceildiv(outputs, WELFORD_THREADS // tpr)
                    var min_per_split = WELFORD_MIN_PER_SPLIT
                    # Column tile of the strided kernel: the power of two
                    # covering `inner`, within [16, WELFORD_COLS_TILE].
                    var cw = 16
                    while cw < inner and cw < WELFORD_COLS_TILE:
                        cw *= 2
                    if inner > 1:
                        target = WELFORD_COLS_BLOCKS_PER_SM * _device_sm_count(
                            ctx
                        )
                        base_blocks = outer * ceildiv(inner, cw)
                        min_per_split = WELFORD_MIN_ROWS_PER_SPLIT * (
                            WELFORD_THREADS // cw
                        )
                    var splits = 1
                    if base_blocks < target:
                        splits = max(
                            1,
                            min(
                                ceildiv(target, base_blocks),
                                n // min_per_split,
                            ),
                        )
                    var outp = _make_ptr[dt](
                        _raw_int(out_o)
                    ).as_unsafe_any_origin()
                    var meanp = _make_ptr[dt](
                        _raw_int(mean_o)
                    ).as_unsafe_any_origin()
                    var inp = (
                        _make_ptr[dt](_raw_int(in_o))
                        .as_unsafe_any_origin()
                        .as_imm()
                    )
                    var corr = _raw_f64(correction_o).cast[acc]()
                    var ts = Int64(_raw_int(take_sqrt_o))
                    var ws = ctx.enqueue_create_buffer[acc](
                        3 * outputs * splits if splits > 1 else 1
                    )
                    var wsp = ws.unsafe_ptr().as_unsafe_any_origin()
                    if inner == 1:
                        comptime for variant in range(4):
                            comptime v = variant // 2 == 1
                            comptime sg = variant % 2 == 1
                            if vec == v and (seg != n) == sg:
                                _enqueue_cached[
                                    _welford_rows_kernel[dt, v, sg]
                                ](
                                    ctx,
                                    min(base_blocks, STATS_MAX_BLOCKS),
                                    splits,
                                    1,
                                    WELFORD_THREADS,
                                    outp,
                                    meanp,
                                    wsp,
                                    inp,
                                    Int64(outputs),
                                    Int64(n),
                                    Int64(seg),
                                    Int64(splits),
                                    Int64(tpr),
                                    corr,
                                    ts,
                                )
                    else:
                        _enqueue_cached[_welford_cols_kernel[dt]](
                            ctx,
                            base_blocks,
                            splits,
                            1,
                            WELFORD_THREADS,
                            outp,
                            meanp,
                            wsp,
                            inp,
                            Int64(n),
                            Int64(inner),
                            Int64(splits),
                            Int64(cw),
                            corr,
                            ts,
                        )
                    if splits > 1:
                        # Few partials per output: a thread each; many: a
                        # whole block each.
                        var mtpr = (
                            1 if splits
                            <= WELFORD_SERIAL_MERGE_MAX else WELFORD_THREADS
                        )
                        _enqueue_cached[_welford_merge_kernel[dt]](
                            ctx,
                            min(
                                ceildiv(outputs, WELFORD_THREADS // mtpr),
                                STATS_MAX_BLOCKS,
                            ),
                            1,
                            1,
                            WELFORD_THREADS,
                            outp,
                            meanp,
                            wsp,
                            Int64(outputs),
                            Int64(splits),
                            Int64(mtpr),
                            corr,
                            ts,
                        )
                    _ = ws^  # a stream-ordered free after the kernels
                    return
        raise Error("mojo Welford: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# bridges
# ---------------------------------------------------------------------------


def _mode_dispatch(
    out_v_o: Arg,
    out_i_o: Arg,
    sv_o: Arg,
    si_o: Arg,
    rows_o: Arg,
    n_o: Arg,
    first_o: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    """Slots: values out, indices out, sorted values, sorted indices, rows,
    n, whether to report the first index of the mode, dtype code, context."""
    var dtype = _raw_dtype_int(dtype_o)
    var rows = _raw_int(rows_o)
    var n = _raw_int(n_o)
    if rows == 0 or n == 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime for dt in MODE_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    _enqueue_cached[_mode_runs_kernel[dt]](
                        _raw_ctx(ctx_o),
                        _grid(rows),
                        1,
                        1,
                        STATS_THREADS,
                        _make_ptr[dt](_raw_int(out_v_o)).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(out_i_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[dt](_raw_int(sv_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[DType.int64](_raw_int(si_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        Int64(rows),
                        Int64(n),
                        Int64(_raw_int(first_o)),
                    )
                    return
        raise Error("mojo mode: unsupported dtype ", dtype)


def _segment_go(
    out_o: Arg,
    data_o: Arg,
    lengths_o: Arg,
    offsets_o: Arg,
    reduction_o: Arg,
    outer_o: Arg,
    segments_o: Arg,
    inner_o: Arg,
    axis_size_o: Arg,
    initial_set_o: Arg,
    initial_o: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    """Slots: output, data, lengths (int64), offsets (int64, segments + 1
    per outer row), reduction, outer, segments, inner, data's axis size,
    whether `initial` was given, the start value (float64 bits; rounded to
    the element dtype here, on the host), dtype code, context."""
    var dtype = _raw_dtype_int(dtype_o)
    var total = _raw_int(outer_o) * _raw_int(segments_o) * _raw_int(inner_o)
    if total == 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime for dt in SEGMENT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    _enqueue_cached[_segment_reduce_kernel[dt]](
                        _raw_ctx(ctx_o),
                        _grid(total),
                        1,
                        1,
                        STATS_THREADS,
                        _make_ptr[dt](_raw_int(out_o)).as_unsafe_any_origin(),
                        _make_ptr[dt](_raw_int(data_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[DType.int64](_raw_int(lengths_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[DType.int64](_raw_int(offsets_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        Int64(_raw_int(reduction_o)),
                        Int64(_raw_int(outer_o)),
                        Int64(_raw_int(segments_o)),
                        Int64(_raw_int(inner_o)),
                        Int64(_raw_int(axis_size_o)),
                        Int64(_raw_int(initial_set_o)),
                        _raw_f64(initial_o).cast[dt](),
                    )
                    return
        raise Error("mojo segment_reduce: unsupported dtype ", dtype)


def _segment_bwd_go(
    grad_in_o: Arg,
    grad_o: Arg,
    output_o: Arg,
    data_o: Arg,
    lengths_o: Arg,
    offsets_o: Arg,
    reduction_o: Arg,
    outer_o: Arg,
    segments_o: Arg,
    inner_o: Arg,
    axis_size_o: Arg,
    initial_o: Arg,
    dtype_ctx_o: Arg,
) raises:
    """Slots: grad_input (zeroed), grad, output, data, lengths, offsets,
    reduction, outer, segments, inner, axis size, prod's initial value
    (float64 bits), and a tuple (dtype code, context)."""
    var dtype = _raw_dtype_int(
        Argv(unsafe_from_address=dtype_ctx_o)[unsafe_offset=1]
    )
    var ctx_addr = Argv(unsafe_from_address=dtype_ctx_o)[unsafe_offset=2]
    var total = _raw_int(outer_o) * _raw_int(segments_o) * _raw_int(inner_o)
    if total == 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime for dt in SEGMENT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    _enqueue_cached[_segment_reduce_backward_kernel[dt]](
                        _raw_ctx(ctx_addr),
                        _grid(total),
                        1,
                        1,
                        STATS_THREADS,
                        _make_ptr[dt](
                            _raw_int(grad_in_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[dt](_raw_int(grad_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[dt](_raw_int(output_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[dt](_raw_int(data_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[DType.int64](_raw_int(lengths_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[DType.int64](_raw_int(offsets_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        Int64(_raw_int(reduction_o)),
                        Int64(_raw_int(outer_o)),
                        Int64(_raw_int(segments_o)),
                        Int64(_raw_int(inner_o)),
                        Int64(_raw_int(axis_size_o)),
                        _raw_f64(initial_o).cast[dt](),
                    )
                    return
        raise Error("mojo segment_reduce backward: unsupported dtype ", dtype)


def _lincomb_go(
    out_o: Arg,
    in_o: Arg,
    coeff_o: Arg,
    m_o: Arg,
    n_o: Arg,
    rest_o: Arg,
    dtype_ctx_o: Arg,
) raises:
    """Slots: output (accumulated into), input (n, rest), coefficients
    (m, n), m, n, rest, and a tuple (dtype code, context)."""
    var dtype = _raw_dtype_int(
        Argv(unsafe_from_address=dtype_ctx_o)[unsafe_offset=1]
    )
    var ctx_addr = Argv(unsafe_from_address=dtype_ctx_o)[unsafe_offset=2]
    var total = _raw_int(m_o) * _raw_int(rest_o)
    if total == 0:
        return
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime for dt in LINCOMB_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    _enqueue_cached[_linear_combination_kernel[dt]](
                        _raw_ctx(ctx_addr),
                        _grid(total),
                        1,
                        1,
                        STATS_THREADS,
                        _make_ptr[dt](_raw_int(out_o)).as_unsafe_any_origin(),
                        _make_ptr[dt](_raw_int(in_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        _make_ptr[dt](_raw_int(coeff_o))
                        .as_unsafe_any_origin()
                        .as_imm(),
                        Int64(_raw_int(m_o)),
                        Int64(_raw_int(n_o)),
                        Int64(_raw_int(rest_o)),
                    )
                    return
        raise Error("mojo linear combination: unsupported dtype ", dtype)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["ModeRuns"]():
            _spec_dispatcher9[_mode_dispatch, "ModeRuns"](argv, argc)
            return 0
        comptime if _op_on["SegmentReduce"]():
            _spec_dispatcher13[_segment_go, "SegmentReduce"](argv, argc)
            return 0
        comptime if _op_on["SegmentReduceBwd"]():
            _spec_dispatcher13[_segment_bwd_go, "SegmentReduceBwd"](argv, argc)
            return 0
        comptime if _op_on["Welford"]():
            _spec_dispatcher11[_welford_go, "Welford"](argv, argc)
            return 0
        comptime if _op_on["LinearCombination"]():
            _spec_dispatcher7[_lincomb_go, "LinearCombination"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
