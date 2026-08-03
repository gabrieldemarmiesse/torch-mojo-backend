# ===----------------------------------------------------------------------=== #
# Fast eager-mode reduction kernels for mojo_device: row-wise sum / max /
# min / prod / argmin / min-with-index / variance / log-softmax / any / all
# over the trailing dimension of a contiguous tensor. Any other layout is
# permuted and materialized in PYTHON before the call (through the queued
# strided copy, so the transient is budget-metered and covered by the
# allocation retry); the bridges here refuse non-trailing dims or strided
# operands rather than scratch-copying, and the kernels only ever see a
# contiguous (rows, cols) buffer reducing each row to one output element.
#
# Raw-pointer calling convention (see elementwise_ops.mojo / nn_ops.mojo):
# tensor operands arrive as element-aligned int addresses, sizes and dtypes as
# ints, ctx_ptr last. Every kernel has a CPU branch (one sequential task per
# row) and a GPU branch (one thread block per row, shared-memory tree reduce
# via `_enqueue_cached`) — the same split nn_ops uses for layer norm / softmax
# / argmax, because full reductions and vocab-dim reductions are called with
# rows == 1 and cols in the tens of thousands, where a thread-per-row launch
# would leave the GPU idle.
#
# Floating-point rows accumulate in float32 (matching torch); integer rows
# accumulate in their own dtype.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import exp, log
from std.memory import stack_allocation
from std.memory.unsafe import bitcast
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys._assembly import inlined_assembly
from std.sys.info import (
    has_apple_gpu_accelerator,
    has_accelerator,
    is_nvidia_gpu,
    size_of,
)
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.numerics import min_or_neg_inf, max_or_inf
from std.utils.static_tuple import StaticTuple

from std.algorithm.reduction import product, sum
from std.algorithm.reduction import max as reduce_max
from std.algorithm.reduction import min as reduce_min

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    FLOAT_DTYPES,
    MAX_RANK,
    TensorSpec,
    _check_into,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _raw_f64,
    _raw_tuple_int,
    _raw_tuple_len,
    _reduce_spec_geom,
    _spec_dispatcher2,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_ptr,
    _raw_ret_none,
    _spec_unsupported,
)

from variant_gates import (
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _register_call,
)


comptime ROWRED_THREADS = 256
# log2(ROWRED_THREADS): number of halving steps in the reduction trees.
comptime ROWRED_STAGES = 8

# Reduction opcodes shared by the generic value-reduction kernel.
comptime RED_SUM = 0
comptime RED_MAX = 1
comptime RED_MIN = 2
comptime RED_PROD = 3


@always_inline
def _accum_dtype[dtype: DType]() -> DType:
    """float rows accumulate in float32; int rows in their own dtype."""
    comptime if dtype.is_floating_point():
        return DType.float32
    else:
        return dtype


# ---------------------------------------------------------------------------
# Library-vs-block routing gate (GPU only).
#
# Benchmarking against the pinned nightly showed the stdlib reduction library is
# NOT strictly better than a 256-thread block-per-row kernel: its two-phase tier
# allocates two device buffers + a memset per call (bad for the small full
# reductions the decode loop issues every step, e.g. max(1,256)/all(1,~3000)),
# and its block-saturated tier uses 128 threads/block (half ours), losing ~2x on
# few-row, huge-col shapes like (256, vocab). The library only wins where there
# are too few rows to saturate the device yet each row is huge (its two-phase
# multi-block fan-out) — full reductions of multi-million-element tensors. Route
# only that regime to the library; everything else uses the block kernel below.
# ---------------------------------------------------------------------------

comptime LIB_MIN_COLS = 1 << 20  # 1,048,576
comptime LIB_MAX_ROWS = 128


@always_inline
def _use_library_reduce(rows: Int, cols: Int) -> Bool:
    return rows <= LIB_MAX_ROWS and cols >= LIB_MIN_COLS


@always_inline
def _red_init[acc_dtype: DType, op_code: Int]() -> Scalar[acc_dtype]:
    comptime if op_code == RED_SUM:
        return Scalar[acc_dtype](0)
    comptime if op_code == RED_PROD:
        return Scalar[acc_dtype](1)
    comptime if op_code == RED_MAX:
        return min_or_neg_inf[acc_dtype]()
    comptime if op_code == RED_MIN:
        return max_or_inf[acc_dtype]()
    return Scalar[acc_dtype](0)


@always_inline
def _red_combine[
    acc_dtype: DType, op_code: Int
](mut acc: Scalar[acc_dtype], v: Scalar[acc_dtype]):
    comptime if op_code == RED_SUM:
        acc += v
    comptime if op_code == RED_PROD:
        acc *= v
    comptime if op_code == RED_MAX:
        if v > acc:
            acc = v
    comptime if op_code == RED_MIN:
        if v < acc:
            acc = v


# ---------------------------------------------------------------------------
# Generic row reduction (sum / max / min / prod): input viewed as
# (rows, cols) contiguous, out has `rows` elements of the same dtype.
# rows == 1 covers full reductions (x.sum(), x.max(), ...).
#
# The under-saturated, huge-col case is handed to the stdlib reduction library
# (`std.algorithm.reduction`) via per-element input_fn/output_fn closures; its
# two-phase tier fans the reduction across the device (rows == 1 over millions
# of elements: 24 ms -> 0.2 ms). Every other GPU case, and all CPU cases, use a
# 256-thread block-per-row kernel (no per-call allocation). Floats reduce in
# float32 (`_accum_dtype`), matching torch.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"reduce_rows_block_{dtype}_{op_code}")
def _reduce_block_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    """One block per row (grid.x = rows); lanes stride over the row and
    tree-reduce their partials in shared memory."""
    comptime acc_dtype = _accum_dtype[dtype]()
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var acc = _red_init[acc_dtype, op_code]()
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j].cast[acc_dtype]()
        _red_combine[acc_dtype, op_code](acc, v)

    var red = stack_allocation[
        ROWRED_THREADS, acc_dtype, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var cur = red[tid]
            _red_combine[acc_dtype, op_code](cur, red[tid + stride])
            red[tid] = cur
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = red[0].cast[dtype]()


@always_inline
def _reduce_rows[
    dtype: DType, op_code: Int
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    comptime acc_dtype = _accum_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[acc_dtype, width]:
        var flat = coords[0] * cols + coords[1]
        return in_ptr.load[width=width](flat).cast[acc_dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], val: SIMD[acc_dtype, width]):
        out_ptr[coords[0]] = val[0].cast[dtype]()

    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        comptime if op_code == RED_SUM:
            sum[acc_dtype, input_fn, output_fn, target=target, reduce_dim=1](
                Coord(shape), ctx
            )
        elif op_code == RED_PROD:
            product[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        elif op_code == RED_MAX:
            reduce_max[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        else:  # RED_MIN
            reduce_min[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(rows, cols):
                run["gpu"]()
            else:
                _enqueue_cached[_reduce_block_kernel[dtype, op_code]](
                    ctx,
                    String(t"reduce_rows_{dtype}_{op_code}"),
                    rows,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    cols,
                )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Direct contiguous adjacent-dimension FP32 sum. Presenting a row-major input
# as (outer, reduce, inner) lets the pinned stdlib's saturated reduction path
# SIMD-pack adjacent inner values and read the source directly. This avoids the
# full-tensor permutation scratch required by the generic non-trailing path.
# All dimensions remain runtime values; unsupported layouts keep the existing
# materialize-and-reduce fallback.
# ---------------------------------------------------------------------------


@always_inline
# ---------------------------------------------------------------------------
# Direct contiguous adjacent-dimension FP32 sum. Presenting a row-major input
# as (outer, reduce, inner) lets the pinned stdlib's saturated reduction path
# SIMD-pack adjacent inner values and read the source directly. This avoids the
# full-tensor permutation scratch required by the generic non-trailing path.
# All dimensions remain runtime values; unsupported layouts keep the existing
# materialize-and-reduce fallback.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name("sum_contiguous_middle_f32_few_outputs")
def _sum_middle_few_outputs_kernel(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    reduce_elements: Int,
    inner_elements: Int,
):
    """One cooperative block per output for the stdlib's scratch regime."""
    var output_index = Int(block_idx.x)
    var outer_index = output_index // inner_elements
    var inner_index = output_index % inner_elements
    var partial = Float32(0.0)
    for reduction_index in range(
        Int(thread_idx.x), reduce_elements, ROWRED_THREADS
    ):
        partial += input[
            (outer_index * reduce_elements + reduction_index) * inner_elements
            + inner_index
        ]
    var reduced = block.sum[block_size=ROWRED_THREADS](partial)
    if thread_idx.x == 0:
        output[output_index] = reduced


@always_inline
def _sum_contiguous_middle_f32(
    out_addr: Int,
    in_addr: Int,
    outer_elements: Int,
    reduce_elements: Int,
    inner_elements: Int,
    ctx: DeviceContext,
) raises:
    var output = _make_ptr[DType.float32](out_addr)
    var input = _make_ptr[DType.float32](in_addr)
    var output_elements = outer_elements * inner_elements
    comptime sm_count = ctx.default_device_info.sm_count
    if output_elements < sm_count and reduce_elements > 128:
        _enqueue_cached[_sum_middle_few_outputs_kernel](
            ctx,
            "sum_contiguous_middle_f32_few_outputs",
            output_elements,
            1,
            1,
            ROWRED_THREADS,
            output.as_unsafe_any_origin(),
            input.as_unsafe_any_origin().as_immutable(),
            reduce_elements,
            inner_elements,
        )
        return

    @always_inline
    @parameter
    @__copy_capture(input, reduce_elements, inner_elements)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var flat = (
            coords[0] * reduce_elements + coords[1]
        ) * inner_elements + coords[2]
        return input.load[width=width](flat)

    @always_inline
    @parameter
    @__copy_capture(output, inner_elements)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], value: SIMD[DType.float32, width]):
        var flat = coords[0] * inner_elements + coords[2]
        output.store[width=width](flat, value)

    var shape = IndexList[3](outer_elements, reduce_elements, inner_elements)
    sum[
        DType.float32,
        input_fn,
        output_fn,
        target="gpu",
        reduce_dim=1,
    ](Coord(shape), ctx)


@always_inline
def _adjacent_reduce_geom(
    a: TensorSpec,
    rdims_t: PyObjectPtr,
    mut outer_elements: Int,
    mut reduce_elements: Int,
    mut inner_elements: Int,
) raises -> Bool:
    """Collapse a contiguous adjacent reduction interval into runtime
    (outer, reduce, inner) dimensions. Empty/non-adjacent intervals reject."""
    if not a.contig:
        return False
    var n = _raw_tuple_len(rdims_t)
    if n == 0:
        return False
    var first = _raw_tuple_int(rdims_t, 0)
    if first < 0 or first + n > a.rank:
        return False
    for k in range(n):
        if _raw_tuple_int(rdims_t, k) != first + k:
            return False

    outer_elements = 1
    reduce_elements = 1
    inner_elements = 1
    for d in range(first):
        outer_elements *= a.dim(d)
    for d in range(first, first + n):
        reduce_elements *= a.dim(d)
    for d in range(first + n, a.rank):
        inner_elements *= a.dim(d)
    return True


# ---------------------------------------------------------------------------
# Row-wise argmin: input viewed as (rows, cols), out is `rows` int64 indices
# (first occurrence wins, matching torch). Mirror of nn_ops' ArgmaxRows with
# the comparison flipped.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"argmin_rows_block_{dtype}")
def _argmin_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    """One block per row; lanes pick their own first-min (value, index) with
    strict `<`, then a shared-memory tree reduction combines lanes with a
    lower-index tiebreak on equal values — torch's first-occurrence-wins."""
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = max_or_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j]
        if v < best_val:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ROWRED_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ROWRED_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            if other_val < cur_val or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[r] = idx_smem[0]


@always_inline
def _argmin_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                if v < best:
                    best = v
                    best_idx = j
            out_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_argmin_rows_block_kernel[dtype]](
                ctx,
                String(t"argmin_rows_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Row-wise min/max with indices: values AND int64 indices (first occurrence
# wins). `is_min` selects the direction. Covers aten.min.dim / max.dim.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"minmax_idx_rows_block_{dtype}_{is_min}")
def _minmax_idx_block_kernel[
    dtype: DType, is_min: Bool
](
    val_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = max_or_inf[dtype]() if is_min else min_or_neg_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j]
        var take = (v < best_val) if is_min else (v > best_val)
        if take:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ROWRED_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ROWRED_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            var strictly_better = (other_val < cur_val) if is_min else (
                other_val > cur_val
            )
            if strictly_better or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        val_ptr[r] = val_smem[0]
        idx_ptr[r] = idx_smem[0]


@always_inline
def _minmax_idx_rows[
    dtype: DType, is_min: Bool
](
    val_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var val_ptr = _make_ptr[dtype](val_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(val_ptr, idx_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                var take = (v < best) if is_min else (v > best)
                if take:
                    best = v
                    best_idx = j
            val_ptr[r] = best
            idx_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_minmax_idx_block_kernel[dtype, is_min]](
                ctx,
                String(t"minmax_idx_rows_{dtype}_{is_min}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                val_ptr.as_unsafe_any_origin(),
                idx_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
# ---------------------------------------------------------------------------
# Row-wise variance (two-pass mean then squared deviations, float32 accum),
# divided by (cols - correction). Covers aten.var.correction.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"var_rows_block_{dtype}")
def _var_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
    correction: Float32,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        1, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += in_ptr[base + j].cast[DType.float32]()
    red[tid] = s
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0] / Float32(cols)
    barrier()
    var mean = bcast[0]

    var vs = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        var d = in_ptr[base + j].cast[DType.float32]() - mean
        vs += d * d
    red[tid] = vs
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = (red[0] / (Float32(cols) - correction)).cast[dtype]()


@always_inline
def _var_rows[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    correction: Float32,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var total = Float32(0)
            for j in range(cols):
                total += in_ptr[base + j].cast[DType.float32]()
            var mean = total / Float32(cols)
            var var_sum = Float32(0)
            for j in range(cols):
                var d = in_ptr[base + j].cast[DType.float32]() - mean
                var_sum += d * d
            out_ptr[r] = (var_sum / (Float32(cols) - correction)).cast[dtype]()

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_var_rows_block_kernel[dtype]](
                ctx,
                String(t"var_rows_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
                correction,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Row-wise log-softmax over the trailing dim: out = x - max - log(sum(exp(x -
# max))), float32 accumulation. Covers aten._log_softmax.
# ---------------------------------------------------------------------------


# Keep the concurrent input footprint under L2 (H100: 50 MB) with headroom, so
# a row survives from its pass-1 read to its pass-2 re-read.
comptime LSM_L2_BUDGET = 23_000_000
# Rows whose bytes exceed this need the grid capped below full 256-thread
# occupancy to fit L2; recover the lost parallelism with 1024-thread blocks.
# Below it, 256-thread blocks keep every thread busy and reductions cheap.
comptime LSM_BIG_ROW_BYTES = 25_000

# Floor under the long-row grid, in blocks per CU. The L2 budget above is an
# H100 number and it starves a wide part: at the nanoGPT logits row (50304 bf16
# columns) it admits 228 blocks, and a gfx942 MI300X has 304 CUs. Measured
# there, 20 launches of the kernel below at [49152, 50304] bf16, us:
#
#   228 blocks 4460/4459/4455   608 3536   912 3570   1216 3538/3561
#   1824 3517   2432 3509   4096 3515/3543   8192 3494   16384 3498
#   49152 (one block per row) 3545
#
# i.e. the residency the cap buys back is worth far less than the parallelism it
# costs, and everything from 2 blocks per CU upward is within 2% of flat. 4 sits
# inside that plateau. The cap still applies above this floor, so short rows --
# where it admits a large grid anyway -- are unaffected.
comptime LSM_BLOCKS_PER_CU = 4


@always_inline
def _lsm_store_out_16B[
    dtype: DType, width: Int, //, vec_align: Int
](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], val: SIMD[dtype, width]):
    """128-bit output store. On NVIDIA use a streaming (evict-first)
    `st.global.cs` store so the writes do not evict the input rows we re-read in
    pass 2; elsewhere fall back to a normal vectorized store."""
    comptime if is_nvidia_gpu():
        var u = bitcast[DType.uint32, 4](val)
        inlined_assembly[
            "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};",
            NoneType,
            constraints="l,r,r,r,r",
            has_side_effect=True,
        ](ptr, u[0], u[1], u[2], u[3])
    else:
        ptr.store[width=width, alignment=vec_align](val)


@always_inline
def _lsm_vec_phase[dtype: DType](addr: Int) -> Int:
    """Element offset of `addr` inside its own 16-byte block, or -1 when `addr`
    is not even element-aligned.

    A 128-bit access is legal only at an address that is a multiple of 16, so
    this — and not the column index — is what decides where a row's vectorized
    body may start: element index `j` of a buffer sits at a 16-byte boundary iff
    `(phase + j) % V == 0`, which collapses to `j % V == 0` only for a buffer
    whose own base is 16-byte aligned.
    """
    comptime esize = size_of[dtype]()
    if addr % esize != 0:
        return -1
    return (addr % 16) // esize


# Fused online (single-read) log-softmax. The reduction reads each row ONCE
# (vectorized 16-byte loads; per-lane running max m + running sum s rescaled on
# a new max), then the output pass re-reads that row — kept resident in L2 by
# the grid cap in `_log_softmax_rows` — and writes with a streaming store.
# Unaligned rows: a per-row scalar head walks each buffer up to a genuinely
# 16-byte-aligned ADDRESS, plus a scalar tail. Input and output are independent
# allocations with independent 16-byte phases (a sliced input can sit at
# `ptr % 16 == 4` while its freshly allocated output sits at 0), so each buffer
# gets its own head/body/tail split. Correct for any rows/cols >= 1,
# bf16/f16/f32.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"log_softmax_rows_block_{dtype}_{threads}")
def _log_softmax_rows_block_kernel[
    dtype: DType, threads: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
    rows: Int,
):
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var tid = Int(thread_idx.x)

    # 16-byte phase of each buffer, folded into the per-row head length below so
    # every vectorized access lands on a real 16-byte boundary. Loop-invariant.
    var in_phase = _lsm_vec_phase[dtype](Int(in_ptr))
    var out_phase = _lsm_vec_phase[dtype](Int(out_ptr))
    # Pass 2 loads and stores the SAME row-local index, so it can keep the
    # 16-byte load only when both buffers share a phase (the aligned hot path
    # has both at 0). Otherwise it keeps the aligned streaming store and gathers
    # the input element-wise.
    var pass2_vec_load = in_phase >= 0 and in_phase == out_phase

    var row = Int(block_idx.x)
    while row < rows:
        var base = row * cols

        # Input split — pass 1 reads only the input.
        var head = cols
        var n_vec = 0
        if in_phase >= 0:
            head = (V - (in_phase + base) % V) % V
            if head > cols:
                head = cols
            n_vec = (cols - head) // V
        var vec_start = base + head  # 16-byte-aligned address in `in_ptr`
        var tail_start = head + n_vec * V  # first row-local index of the tail

        # Output split — pass 2 stores here; its phase is independent.
        var head_o = cols
        var n_vec_o = 0
        if out_phase >= 0:
            head_o = (V - (out_phase + base) % V) % V
            if head_o > cols:
                head_o = cols
            n_vec_o = (cols - head_o) // V
        var vec_start_o = base + head_o  # 16-byte-aligned address in `out_ptr`
        var tail_start_o = head_o + n_vec_o * V

        # ---- Pass 1: online max + sum over the row, one global read. ----
        # The running max starts at the lowest FINITE float, not at
        # `Float32.MIN`, which is -inf. A lane or thread that never runs its loop
        # body -- any thread with `tid >= n_vec`, and there are many whenever the
        # row has fewer vectors than the block has threads -- keeps the sentinel
        # in both `m` and a zero `s`. The collapse below then evaluates
        # `s * exp(m - m)`: with -inf that is `0 * exp(nan) = nan`, and the block
        # sum turns one idle thread's nan into a nan `log_denom`, so the whole
        # row comes out nan. With a finite sentinel the same expression is
        # `0 * exp(0) = 0`, the identity this monoid needs. A genuine -inf input
        # still behaves: `exp(-inf - MIN_FINITE)` is 0, not nan.
        #
        # Do NOT replace this with an `-inf` seed plus an `a == b` select on every
        # `exp(a - b)`, which fixes the same nan and is what `main` carries as of
        # #319. Measured here, that form costs 12-16% on bf16 rows (bf16
        # 12288x50304: 1034 us against 890), because the two extra vector
        # compare-selects per trip do not hide behind bf16's halved byte count.
        # It is also less faithful to torch: the `x == new_m` guard turns
        # `exp(inf - inf)` into 1.0, so a row holding `+inf` returns -inf for its
        # finite entries where torch returns nan. The finite seed needs no guard,
        # since it is never an operand of a subtraction that can reach inf - inf.
        var m_vec = SIMD[DType.float32, V](Float32.MIN_FINITE)
        var s_vec = SIMD[DType.float32, V](0.0)
        var v = tid
        while v < n_vec:
            var x = in_ptr.load[width=V, alignment=vec_align](
                vec_start + v * V
            ).cast[DType.float32]()
            var new_m = max(m_vec, x)
            s_vec = s_vec * exp(m_vec - new_m) + exp(x - new_m)
            m_vec = new_m
            v += threads

        # Collapse the per-lane accumulator to a thread-local (m, s).
        var m_t = m_vec.reduce_max()
        var s_t = (s_vec * exp(m_vec - m_t)).reduce_add()

        # Fold in the unaligned scalar head/tail (each < V elements, one thread
        # per element; no-ops when head == tail == 0).
        var jh = tid
        while jh < head:
            var x = in_ptr[base + jh].cast[DType.float32]()
            var nm = max(m_t, x)
            s_t = s_t * exp(m_t - nm) + exp(x - nm)
            m_t = nm
            jh += threads
        var jt = tail_start + tid
        while jt < cols:
            var x = in_ptr[base + jt].cast[DType.float32]()
            var nm = max(m_t, x)
            s_t = s_t * exp(m_t - nm) + exp(x - nm)
            m_t = nm
            jt += threads

        # ---- Block combine: global max, then rescale + global sum. ----
        var block_m = block.max[block_size=threads](m_t)
        var s_scaled = s_t * exp(m_t - block_m)
        var block_s = block.sum[block_size=threads](s_scaled)
        var log_denom = log(block_s)

        # ---- Pass 2: output = x - max - log_denom. Input read hits L2; the
        # streaming store keeps the writes from evicting it. ----
        if pass2_vec_load:
            # Aligned hot path: both buffers share a phase, so `vec_start_o`
            # names a 16-byte boundary in each.
            var vo = tid
            while vo < n_vec_o:
                var x = in_ptr.load[width=V, alignment=vec_align](
                    vec_start_o + vo * V
                ).cast[DType.float32]()
                var y = (x - block_m - log_denom).cast[dtype]()
                _lsm_store_out_16B[vec_align=vec_align](
                    out_ptr + (vec_start_o + vo * V), y
                )
                vo += threads
        else:
            # Phases disagree: the store stays 16-byte aligned and vectorized,
            # the input is gathered one element at a time (still fully
            # coalesced across the lanes of a warp).
            var vo = tid
            while vo < n_vec_o:
                var off = vec_start_o + vo * V
                var x = SIMD[DType.float32, V](0.0)

                @parameter
                for k in range(V):
                    x[k] = in_ptr[off + k].cast[DType.float32]()
                var y = (x - block_m - log_denom).cast[dtype]()
                _lsm_store_out_16B[vec_align=vec_align](out_ptr + off, y)
                vo += threads
        var jho = tid
        while jho < head_o:
            var x = in_ptr[base + jho].cast[DType.float32]()
            out_ptr[base + jho] = (x - block_m - log_denom).cast[dtype]()
            jho += threads
        var jto = tail_start_o + tid
        while jto < cols:
            var x = in_ptr[base + jto].cast[DType.float32]()
            out_ptr[base + jto] = (x - block_m - log_denom).cast[dtype]()
            jto += threads

        row += Int(grid_dim.x)


@always_inline
def _log_softmax_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var m = Float32.MIN
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                if x > m:
                    m = x
            var denom = Float32(0)
            for j in range(cols):
                denom += exp(in_ptr[base + j].cast[DType.float32]() - m)
            var log_denom = log(denom)
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                out_ptr[base + j] = (x - m - log_denom).cast[dtype]()

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            # Cap concurrent rows so their input bytes stay resident in L2 for
            # the pass-2 re-read; grid-stride over the rest.
            comptime FILL = LSM_BLOCKS_PER_CU * ctx.default_device_info.sm_count
            var esize = size_of[dtype]()
            var blocks = min(rows, max(1, LSM_L2_BUDGET // (cols * esize)))
            var mout = out_ptr.as_unsafe_any_origin()
            var min_ = in_ptr.as_unsafe_any_origin().as_immutable()
            # Big rows: 1024-thread blocks so the small (L2-capped) grid still
            # saturates memory. Small rows: 256 threads keep every thread busy.
            if cols * esize > LSM_BIG_ROW_BYTES:
                # ...but the budget alone cannot be allowed to leave the device
                # idle. See LSM_BLOCKS_PER_CU: at the nanoGPT logits row it
                # admits 228 blocks on a 304-CU part and that costs 27%.
                blocks = min(rows, max(blocks, FILL))
                _enqueue_cached[_log_softmax_rows_block_kernel[dtype, 1024]](
                    ctx,
                    String(t"log_softmax_rows_{dtype}_1024"),
                    blocks,
                    1,
                    1,
                    1024,
                    mout,
                    min_,
                    cols,
                    rows,
                )
            else:
                _enqueue_cached[_log_softmax_rows_block_kernel[dtype, 256]](
                    ctx,
                    String(t"log_softmax_rows_{dtype}_256"),
                    blocks,
                    1,
                    1,
                    256,
                    mout,
                    min_,
                    cols,
                    rows,
                )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Row-wise any / all: input viewed as (rows, cols) of any dtype, out is
# `rows` bool elements (nonzero test). Covers aten.any.dim / all.dim(s).
# Same library-vs-block routing as the value reductions: the library's max/min
# accept DType.bool natively (any = max init False -> OR; all = min init True ->
# AND), but its two-phase tier allocates per call, so only the under-saturated,
# huge-col regime uses it; everything else uses the block kernel.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"anyall_rows_block_{dtype}_{is_all}")
def _anyall_rows_block_kernel[
    dtype: DType, is_all: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols
    var zero = Scalar[dtype](0)

    var acc: Bool = is_all
    for j in range(tid, cols, ROWRED_THREADS):
        var nz = Bool(in_ptr[base + j] != zero)
        comptime if is_all:
            acc = acc and nz
        else:
            acc = acc or nz

    var red = stack_allocation[
        ROWRED_THREADS, DType.bool, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            comptime if is_all:
                red[tid] = red[tid] and red[tid + stride]
            else:
                red[tid] = red[tid] or red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = red[0]


@always_inline
def _anyall_rows[
    dtype: DType, is_all: Bool
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """any/all over the trailing axis. The nonzero test maps each row to bools,
    then any = max / all = min over DType.bool. Routes the under-saturated,
    huge-col regime to the stdlib library and everything else to a block kernel
    (see `_use_library_reduce`)."""
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    # cols == 0 always arrives with rows == 0 (an empty reduce dim means an
    # empty tensor; the guard lives upstream in aten_fast._reduce_to_rows), so
    # both the library (zero-size shape check) and the block kernel (grid 0)
    # write nothing — same pre-existing contract as the old kernels.

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[DType.bool, width]:
        var flat = coords[0] * cols + coords[1]
        # Nonzero test with the block kernel's `!=` semantics: `.eq` is an
        # ORDERED equal (NaN == 0 -> False), so ~eq gives NaN -> True, matching
        # torch's NaN-is-truthy any/all. (`.cast[DType.bool]()` lowers to an
        # ordered ne, which would wrongly map NaN -> False.) For bool input,
        # eq-with-False then invert is the identity.
        var v = in_ptr.load[width=width](flat)
        return ~v.eq(SIMD[dtype, width]())

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], val: SIMD[DType.bool, width]):
        out_ptr[coords[0]] = val[0]

    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        comptime if is_all:
            reduce_min[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        else:
            reduce_max[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(rows, cols):
                run["gpu"]()
            else:
                _enqueue_cached[_anyall_rows_block_kernel[dtype, is_all]](
                    ctx,
                    String(t"anyall_rows_{dtype}_{is_all}"),
                    rows,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    cols,
                )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): trailing-dims reductions
# over a contiguous input — dim checks, rows/cols/keepdim geometry, output
# alloc and launch in one boundary call, reusing the row kernels above.
# Python still parses the dim spec (`_norm_reduce_dims`) and does dtype
# promotion; non-trailing/strided layouts raise so the classic
# permute+materialize path keeps handling them. Failed checks raise a real
# NotImplementedError into Python ("take the classic path").
# ---------------------------------------------------------------------------

comptime SPEC_ROWRED_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
]

comptime SPEC_ANYALL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
]


def _rowred_spec_into_go[
    op_code: Int
](
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_ROWRED_DTYPES](a.dtype):
        raise Error("mojo spec reduce: unsupported dtype ", a.dtype)
    if a.numel == 0:
        # sum-of-empty is a Python-side fill; amax/amin reject empty dims.
        raise Error("mojo spec reduce: empty input")
    var ctx0 = a.ctx()
    # Zero-copy direct path, kept deliberately: a contiguous fp32 sum over
    # one adjacent NON-trailing dim interval allocates nothing, so it stays
    # in Mojo — only layouts that would scratch-COPY are refused by the
    # geometry below (Python pre-materializes those). nanoGPT's positional-
    # embedding gradient (sum over dim 0) takes this every step; the
    # permute + row-reduce replacement measured ~3 ms/step slower there
    # (rows huge, cols tiny).
    comptime if op_code == RED_SUM:
        if a.dtype == DType.float32 and ctx0.api() != "cpu":
            var n_red = _raw_tuple_len(rdims_t)
            var trailing = True
            for k in range(n_red):
                if _raw_tuple_int(rdims_t, k) != a.rank - n_red + k:
                    trailing = False
            if not trailing:
                var outer_elements = 0
                var reduce_elements = 0
                var inner_elements = 0
                if _adjacent_reduce_geom(
                    a,
                    rdims_t,
                    outer_elements,
                    reduce_elements,
                    inner_elements,
                ):
                    var direct_rows = outer_elements * inner_elements
                    if (
                        out.numel != direct_rows
                        or not out.contig
                        or out.ctx_ptr != a.ctx_ptr
                    ):
                        raise Error("mojo spec into: output buffer mismatch")
                    if out.dtype != a.dtype:
                        raise Error("mojo spec into: output dtype mismatch")
                    if direct_rows > 0:
                        comptime if has_accelerator():
                            _sum_contiguous_middle_f32(
                                out.ptr,
                                a.ptr,
                                outer_elements,
                                reduce_elements,
                                inner_elements,
                                ctx0,
                            )
                        else:
                            raise Error(
                                "no GPU accelerator available at compile time"
                            )
                    return

    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)

    var ctx = a.ctx()
    var nbytes = rows * a.itemsize
    _ = nbytes
    if out.numel != rows or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if rows > 0:
        comptime for dt in SPEC_ROWRED_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _reduce_rows[dt, op_code](addr, a.ptr, rows, cols, ctx)


def _argmin_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_ROWRED_DTYPES](a.dtype):
        raise Error("mojo spec argmin: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec argmin: empty input")
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)

    var ctx = a.ctx()
    var nbytes = rows * 8  # int64 output
    _ = nbytes
    if out.numel != rows or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != DType.int64:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if rows > 0:
        comptime for dt in SPEC_ROWRED_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _argmin_rows[dt](addr, a.ptr, rows, cols, ctx)


def _min_dim_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_v_o: PyObjectPtr,
    out_i_o: PyObjectPtr,
) raises:
    """aten::min.dim values+indices in one call — the multi-output protocol:
    Python allocates both outputs and passes their specs as the last two
    arguments, so one boundary call still fills both."""
    ref a = _spec_ptr(a_o)[]
    ref out_v = _spec_ptr(out_v_o)[]
    ref out_i = _spec_ptr(out_i_o)[]
    if not _dtype_supported[SPEC_ROWRED_DTYPES](a.dtype):
        raise Error("mojo spec min.dim: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec min.dim: empty input")
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)

    var ctx = a.ctx()
    if (
        out_v.numel != rows
        or out_i.numel != rows
        or not out_v.contig
        or not out_i.contig
        or out_v.ctx_ptr != a.ctx_ptr
        or out_i.ctx_ptr != a.ctx_ptr
    ):
        raise Error("mojo spec min.dim into: output buffer mismatch")
    if out_v.dtype != a.dtype or out_i.dtype != DType.int64:
        raise Error("mojo spec min.dim into: output dtype mismatch")
    var addr_v = out_v.ptr
    var addr_i = out_i.ptr
    if rows > 0:
        comptime for dt in SPEC_ROWRED_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _minmax_idx_rows[dt, True](
                        addr_v, addr_i, a.ptr, rows, cols, ctx
                    )


def _var_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    corr_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[FLOAT_DTYPES](a.dtype):
        raise Error("mojo spec var: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec var: empty input")
    var correction = Float32(_raw_f64(corr_o))
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)

    var ctx = a.ctx()
    var nbytes = rows * a.itemsize
    _ = nbytes
    if out.numel != rows or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if rows > 0:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _var_rows[dt](addr, a.ptr, rows, cols, correction, ctx)


def _anyall_spec_into_go[
    is_all: Bool
](
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_ANYALL_DTYPES](a.dtype):
        raise Error("mojo spec any/all: unsupported dtype ", a.dtype)
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)
    var ctx = a.ctx()
    var nbytes = rows  # bool output
    _ = nbytes
    if out.numel != rows or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != DType.bool:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if rows > 0:
        comptime for dt in SPEC_ANYALL_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _anyall_rows[dt, is_all](addr, a.ptr, rows, cols, ctx)


def _log_softmax_spec_into_go(a_o: PyObjectPtr, out_o: PyObjectPtr) raises:
    """log_softmax over the trailing dim; full-shape output. The non-trailing
    dim transpose recursion stays in Python (view ops)."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[FLOAT_DTYPES](a.dtype):
        raise Error("mojo spec log_softmax: unsupported dtype ", a.dtype)
    if a.rank < 1 or a.numel == 0:
        raise Error("mojo spec log_softmax: empty or rank-0 input")

    var cols = a.shape[MAX_RANK - 1]
    var rows = a.numel // cols
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.contig:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _log_softmax_rows[dt](addr, a.ptr, rows, cols, ctx)
    else:
        raise Error(
            "mojo spec log_softmax: input must be contiguous"
            " (Python pre-materializes)"
        )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_reduction_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("reduction_ops")
        comptime if _op_on["SumSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[RED_SUM], "a row-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["AmaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[RED_MAX], "a row-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["AminSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[RED_MIN], "a row-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["ArgminSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_argmin_spec_into_go, "ArgminSpec"],
                docstring="(a_spec, rdims, keepdim, out_spec); int64 indices",
            )
        comptime if _op_on["MinDimSpec"]():
            _register_call(
                b,
                _spec_dispatcher5[_min_dim_spec_into_go, "MinDimSpec"],
                docstring="(a_spec, rdims, keepdim, values_spec, indices_spec)",
            )
        comptime if _op_on["VarSpec"]():
            _register_call(
                b,
                _spec_dispatcher5[_var_spec_into_go, "VarSpec"],
                docstring="(a_spec, rdims, keepdim, correction, out_spec)",
            )
        comptime if _op_on["AnySpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _anyall_spec_into_go[False], "an any/all spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec); bool",
            )
        comptime if _op_on["AllSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _anyall_spec_into_go[True], "an any/all spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec); bool",
            )
        comptime if _op_on["LogSoftmaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher2[_log_softmax_spec_into_go, "LogSoftmaxSpec"],
                docstring="(a_spec, out_spec); trailing dim",
            )
        return b.finalize()
    except e:
        abort(t"failed to create reduction_ops python module: {e}")
