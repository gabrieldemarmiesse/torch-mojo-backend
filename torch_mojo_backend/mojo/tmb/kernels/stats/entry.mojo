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
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.math import ceildiv, fma
from std.sys.info import has_accelerator
from std.utils.numerics import isnan, nan
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _spec_dispatcher7,
    _spec_dispatcher9,
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
        comptime if _op_on["LinearCombination"]():
            _spec_dispatcher7[_lincomb_go, "LinearCombination"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
