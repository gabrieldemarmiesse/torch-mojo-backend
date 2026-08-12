# ===----------------------------------------------------------------------=== #
# Reductions with a (value, index) payload: argmin / argmax, which keep the
# index of the first extremum along one contiguous interval of dimensions, and
# min.dim / max.dim, which keep both. `with_values` is the only difference
# between them -- the scan, the tie-break, the split policy and the merge are
# one mechanism, so a fix to first-occurrence or NaN handling lands on all
# four ops at once.
#
# One mechanism, shared by both bridges -- reduction_ops owns ArgminSpec and
# nn_ops owns ArgmaxSpec, so the direction is a comptime parameter here and
# neither module carries its own copy of the algorithm. Two entry points, one
# per memory regime:
#
#   `_argreduce_rows`  reduce axis CONTIGUOUS: input viewed as (rows, cols),
#                      one block per (row, chunk), lanes walk the row.
#   `_argreduce_cols`  reduce axis STRIDED: input viewed as
#                      (outer, reduce, inner) with inner > 1, one thread per
#                      output column, lanes walk the contiguous inner axis so
#                      the loads stay coalesced. This is what lets a
#                      non-trailing reduce dim skip the transposed copy the
#                      Python side would otherwise materialize.
#
# Both split the reduce axis across blocks when the output count alone cannot
# fill the device -- a full reduction is `rows == 1`, where one block per row
# leaves every SM but one idle -- writing (value, index) partials into a
# workspace that a second kernel merges. Atomics are deliberately not used:
# a workspace plus a merge kernel is deterministic and is the house rule
# (AGENTS.md) for split reductions. That workspace holds one (value, index)
# pair per BLOCK, not per element -- tens of KB for a 16M-element reduction --
# so it is an ordinary stream-ordered allocation here, unlike the whole-tensor
# transposed copies these bridges refuse and Python materializes instead.
#
# Tie-breaking is torch's and it survives the split: the FIRST occurrence of
# the extremum wins, and NaN beats every number (torch propagates NaN through
# argmin/argmax and answers with the index of the first one). `_ar_better`
# makes no assumption about the order in which candidates arrive, so
# intra-lane, intra-block and cross-workspace merges all give the same answer
# as a sequential scan.
#
# Why not modular's `nn.argmaxmin_gpu` (same two-stage shape, and admissible
# under the eager rules -- pure Mojo, no vendor BLAS)? Measured against it on
# this H100, f32 device time of the whole op:
#
#   case                        here   nn.argmaxmin_gpu   stock torch
#   16.7M full reduce          42.2               39.6          46.3
#   1M full reduce              7.6                6.8          11.9
#   4096x4096 reduce dim 1     39.6              107.6          46.0
#   1x50257 reduce dim 1        6.3                6.5          16.8
#   4096x4096 reduce dim 0     63.0    (not supported)         106.0
#
# It is 7-9% faster on a single long row and 2.7x slower once there are many
# rows (its split count collapses to 1, it still launches the combine kernel,
# and its 1024-thread blocks then hold one vector each), it only reduces the
# inner-most dimension, its running index is int32 (it raises past 2^31
# elements), and -- decisively -- it ignores NaN where torch propagates it:
# for [1, nan, 7, nan] torch answers 1 for BOTH argmax and argmin, it answers
# 2 and 0. That is a wrong answer, not a tuning difference, and it is inside
# its per-lane update and `TopK_2.insert`, not something a caller can fix.
# Its 4-way staged load was tried here (AR_UNROLL = 4 in `_ar_scan_row`) and
# measured SLOWER on this card: 43.4us against 42.2 on the full reduce, 7.7
# against 7.6 on 1M, no change elsewhere -- so the remaining 7% is not the
# unroll. The untried candidates are its fatter blocks (64 blocks of 1024
# threads instead of 456 of 256) and its PDL launch attributes.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import has_accelerator, size_of
from std.utils.coord import Coord
from std.utils.static_tuple import StaticTuple

from op_utils import (
    TensorSpec,
    _adjacent_reduce_geom,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
)
from std.python._cpython import PyObjectPtr

from variant_gates import _dtype_arg_on


comptime AR_THREADS = 256
# log2(AR_THREADS): halving steps of the shared-memory reduction tree.
comptime AR_STAGES = 8

# Split-path launch geometry. Both were fitted on an NVIDIA H100 PCIe
# (114 SMs, f32); they are expressed in SMs and in elements per block, so they
# travel to other cards as a shape, not as a constant, but the numbers
# themselves were only measured here.
#
# AR_SPLIT_WAVES: blocks per SM the split path aims for. Swept on this card
# (f32 device time of the whole op; stock torch 46.3us and 106us):
#   waves                       2      4      8     16
#   16.7M full reduce, us    43.9   42.4   44.1   46.0
#   4096x4096 dim 0,   us    71.5   62.7   65.6   73.5
# Shallow optimum, and in the same place for both kernels.
comptime AR_SPLIT_WAVES = 4
# AR_MIN_CHUNK: the smallest slice of the reduce axis worth one block, so a
# medium-sized reduction does not launch thousands of blocks that each read a
# handful of elements. The two regimes divide the slice differently and so
# have different floors: on the contiguous axis a block's 256 lanes share the
# slice (4096 = 16 elements per lane), while on the strided axis every lane
# walks the WHOLE slice (32 elements per lane). Swept here, f32 device time:
#   contiguous       1024   4096  16384      strided        8     32    128
#   1M full, us       8.2    7.5    7.7      357x789 d0    9.1    9.5   18.7
#   16.7M full, us   42.2   42.3   42.2      4096^2 d0    63.0   62.1   62.3
# Both are plateaus over an order of magnitude and only bite at the small end,
# where the floor keeps a split from shrinking below a launch's worth of work.
comptime AR_MIN_CHUNK = 4096
comptime AR_COLS_MIN_CHUNK = 32


@always_inline
def _ar_take_next[
    dtype: DType, is_min: Bool
](v: Scalar[dtype], best: Scalar[dtype]) -> Bool:
    """Does `v`, read at a HIGHER index than `best`, replace it?

    Strict comparison keeps the first occurrence on ties. NaN replaces any
    number and nothing replaces a NaN, so the first NaN wins -- torch's
    answer. Integer dtypes compile the NaN arm away."""
    comptime if dtype.is_floating_point():
        if best != best:
            return False
        if v != v:
            return True
    comptime if is_min:
        return v < best
    else:
        return v > best


@always_inline
def _ar_better[
    dtype: DType, is_min: Bool
](
    cand_v: Scalar[dtype], cand_i: Int64, cur_v: Scalar[dtype], cur_i: Int64
) -> Bool:
    """Does the candidate beat the incumbent, with NO ordering assumption?

    The order-free counterpart of `_ar_take_next`, used wherever partials
    meet out of order (block tree, workspace merge). A negative index marks
    an empty partial and always loses; equal values -- including two NaNs --
    go to the LOWER index, which is what makes an arbitrary split of the axis
    reproduce a sequential first-occurrence scan."""
    if cand_i < 0:
        return False
    if cur_i < 0:
        return True
    comptime if dtype.is_floating_point():
        if cur_v != cur_v:
            return cand_v != cand_v and cand_i < cur_i
        if cand_v != cand_v:
            return True
    comptime if is_min:
        if cand_v < cur_v:
            return True
    else:
        if cand_v > cur_v:
            return True
    return cand_v == cur_v and cand_i < cur_i


@always_inline
def _ar_block_merge[
    dtype: DType, is_min: Bool
](tid: Int, mut best_val: Scalar[dtype], mut best_idx: Int64):
    """Reduce one (value, index) per lane to the block's winner in shared
    memory; every lane leaves with the winner."""
    var val_smem = stack_allocation[
        AR_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        AR_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()
    var stride = AR_THREADS // 2
    for _ in range(AR_STAGES):
        if tid < stride:
            if _ar_better[dtype, is_min](
                val_smem[tid + stride],
                idx_smem[tid + stride],
                val_smem[tid],
                idx_smem[tid],
            ):
                val_smem[tid] = val_smem[tid + stride]
                idx_smem[tid] = idx_smem[tid + stride]
        barrier()
        stride //= 2
    best_val = val_smem[0]
    best_idx = idx_smem[0]


@always_inline
def _ar_scan_row[
    dtype: DType, is_min: Bool, VEC: Int
](
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    base: Int,
    start: Int,
    end: Int,
    tid: Int,
    mut best_val: Scalar[dtype],
    mut best_idx: Int64,
):
    """Scan `[start, end)` of the row starting at element `base` with
    AR_THREADS lanes, VEC contiguous elements per access.

    The caller guarantees `base` and `start` are multiples of VEC and the base
    pointer is 16-byte aligned, so every vector access is naturally aligned;
    the at-most-VEC-1 leftover elements are scanned one at a time. A lane's
    own indices only ever increase, which is what `_ar_take_next` needs."""
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    var nslots = (end - start) // VEC
    for k in range(tid, nslots, AR_THREADS):
        var off = start + k * VEC
        var v = in_ptr.load[width=VEC, alignment=ALIGN](base + off)

        @parameter
        for lane in range(VEC):
            if best_idx < 0 or _ar_take_next[dtype, is_min](v[lane], best_val):
                best_val = v[lane]
                best_idx = Int64(off + lane)
    for j in range(start + nslots * VEC + tid, end, AR_THREADS):
        var v = in_ptr[base + j]
        if best_idx < 0 or _ar_take_next[dtype, is_min](v, best_val):
            best_val = v
            best_idx = Int64(j)


# ---------------------------------------------------------------------------
# Contiguous reduce axis: (rows, cols) -> `rows` int64 indices
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"argreduce_rows_block_{dtype}_min{is_min}_v{VEC}")
def _argreduce_rows_kernel[
    dtype: DType, is_min: Bool, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
):
    """One block per row (grid.x = rows), for the saturated regime."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var tid = Int(thread_idx.x)
    var r = Int(block_idx.x)
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    _ar_scan_row[dtype, is_min, VEC](
        in_ptr, r * cols, 0, cols, tid, best_val, best_idx
    )
    _ar_block_merge[dtype, is_min](tid, best_val, best_idx)
    if tid == 0:
        out_ptr[r] = best_idx


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"argreduce_rows_split_{dtype}_min{is_min}_v{VEC}")
def _argreduce_rows_split_kernel[
    dtype: DType, is_min: Bool, VEC: Int
](
    ws_val: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_idx: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
    chunk_arg: Int64,
):
    """grid = (rows, splits): block (r, s) owns chunk s of row r and writes
    its partial to `ws[r * splits + s]` (split-minor, so the merge block for
    one row reads a contiguous run)."""
    var cols = Int(cols_arg)
    var chunk = Int(chunk_arg)
    var tid = Int(thread_idx.x)
    var r = Int(block_idx.x)
    var s = Int(block_idx.y)
    var splits = Int(grid_dim.y)
    var start = min(s * chunk, cols)
    var end = min(start + chunk, cols)
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    _ar_scan_row[dtype, is_min, VEC](
        in_ptr, r * cols, start, end, tid, best_val, best_idx
    )
    _ar_block_merge[dtype, is_min](tid, best_val, best_idx)
    if tid == 0:
        ws_val[r * splits + s] = best_val
        ws_idx[r * splits + s] = best_idx


# ---------------------------------------------------------------------------
# Strided reduce axis: (outer, reduce, inner) -> outer * inner int64 indices
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"argreduce_cols_{dtype}_min{is_min}")
def _argreduce_cols_kernel[
    dtype: DType, is_min: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    reduce_arg: Int64,
    inner_arg: Int64,
    lanes_arg: Int64,
):
    """One thread per output column; grid.x = `lanes` * outer, where `lanes`
    is ceil(inner / AR_THREADS). Neighbouring lanes read neighbouring elements
    of the contiguous inner axis, so each step of the strided walk is one
    coalesced burst.

    `outer` is folded into grid.x rather than given its own grid dimension
    because grid.y/z are capped at 65535 on CUDA and `outer` is a tensor
    extent, which is not."""
    var reduce_n = Int(reduce_arg)
    var inner = Int(inner_arg)
    var lanes = Int(lanes_arg)
    var o = Int(block_idx.x) // lanes
    var i = (Int(block_idx.x) % lanes) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= inner:
        return
    var base = o * reduce_n * inner + i
    var best_val = in_ptr[base]
    var best_idx = Int64(0)
    for r in range(1, reduce_n):
        var v = in_ptr[base + r * inner]
        if _ar_take_next[dtype, is_min](v, best_val):
            best_val = v
            best_idx = Int64(r)
    out_ptr[o * inner + i] = best_idx


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"argreduce_cols_split_{dtype}_min{is_min}")
def _argreduce_cols_split_kernel[
    dtype: DType, is_min: Bool
](
    ws_val: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_idx: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    reduce_arg: Int64,
    inner_arg: Int64,
    lanes_arg: Int64,
    chunk_arg: Int64,
):
    """grid = (`lanes` * outer, splits): the column walk of
    `_argreduce_cols_kernel` restricted to chunk s of the reduce axis."""
    var reduce_n = Int(reduce_arg)
    var inner = Int(inner_arg)
    var chunk = Int(chunk_arg)
    var lanes = Int(lanes_arg)
    var o = Int(block_idx.x) // lanes
    var i = (Int(block_idx.x) % lanes) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= inner:
        return
    var s = Int(block_idx.y)
    var splits = Int(grid_dim.y)
    var start = min(s * chunk, reduce_n)
    var end = min(start + chunk, reduce_n)
    var base = o * reduce_n * inner + i
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    if start < end:
        best_val = in_ptr[base + start * inner]
        best_idx = Int64(start)
        for r in range(start + 1, end):
            var v = in_ptr[base + r * inner]
            if _ar_take_next[dtype, is_min](v, best_val):
                best_val = v
                best_idx = Int64(r)
    var p = o * inner + i
    ws_val[p * splits + s] = best_val
    ws_idx[p * splits + s] = best_idx


# ---------------------------------------------------------------------------
# (value, index) stage 1: the same two scans, keeping the value as well.
# Only the UNSPLIT kernels need their own entry: the split kernels above
# already write (value, index) pairs into the workspace, so min.dim reuses
# them verbatim and differs only in the merge.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"minmax_rows_block_{dtype}_min{is_min}_v{VEC}")
def _minmax_rows_kernel[
    dtype: DType, is_min: Bool, VEC: Int
](
    val_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
):
    """One block per row (grid.x = rows), for the saturated regime."""
    var cols = Int(cols_arg)
    var tid = Int(thread_idx.x)
    var r = Int(block_idx.x)
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    _ar_scan_row[dtype, is_min, VEC](
        in_ptr, r * cols, 0, cols, tid, best_val, best_idx
    )
    _ar_block_merge[dtype, is_min](tid, best_val, best_idx)
    if tid == 0:
        val_ptr[r] = best_val
        idx_ptr[r] = best_idx


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"minmax_cols_{dtype}_min{is_min}")
def _minmax_cols_kernel[
    dtype: DType, is_min: Bool
](
    val_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    reduce_arg: Int64,
    inner_arg: Int64,
    lanes_arg: Int64,
):
    """One thread per output column; see `_argreduce_cols_kernel`."""
    var reduce_n = Int(reduce_arg)
    var inner = Int(inner_arg)
    var lanes = Int(lanes_arg)
    var o = Int(block_idx.x) // lanes
    var i = (Int(block_idx.x) % lanes) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= inner:
        return
    var base = o * reduce_n * inner + i
    var best_val = in_ptr[base]
    var best_idx = Int64(0)
    for r in range(1, reduce_n):
        var v = in_ptr[base + r * inner]
        if _ar_take_next[dtype, is_min](v, best_val):
            best_val = v
            best_idx = Int64(r)
    val_ptr[o * inner + i] = best_val
    idx_ptr[o * inner + i] = best_idx


# ---------------------------------------------------------------------------
# Workspace merge (shared by both split paths)
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"argreduce_merge_{dtype}_min{is_min}")
def _argreduce_merge_kernel[
    dtype: DType, is_min: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    ws_val: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    ws_idx: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    splits_arg: Int64,
):
    """One block per output element, merging its `splits` partials. The
    partial indices are already global along the reduce axis, so the ordinary
    lower-index tiebreak in `_ar_better` is all first-occurrence needs."""
    var splits = Int(splits_arg)
    var tid = Int(thread_idx.x)
    var p = Int(block_idx.x)
    var base = p * splits
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    for s in range(tid, splits, AR_THREADS):
        if _ar_better[dtype, is_min](
            ws_val[base + s], ws_idx[base + s], best_val, best_idx
        ):
            best_val = ws_val[base + s]
            best_idx = ws_idx[base + s]
    _ar_block_merge[dtype, is_min](tid, best_val, best_idx)
    if tid == 0:
        out_ptr[p] = best_idx


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(AR_THREADS))
)
@__name(t"minmax_merge_{dtype}_min{is_min}")
def _minmax_merge_kernel[
    dtype: DType, is_min: Bool
](
    val_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    ws_val: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    ws_idx: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    splits_arg: Int64,
):
    """`_argreduce_merge_kernel` keeping the winning value as well."""
    var splits = Int(splits_arg)
    var tid = Int(thread_idx.x)
    var p = Int(block_idx.x)
    var base = p * splits
    var best_val = Scalar[dtype](0)
    var best_idx = Int64(-1)
    for s in range(tid, splits, AR_THREADS):
        if _ar_better[dtype, is_min](
            ws_val[base + s], ws_idx[base + s], best_val, best_idx
        ):
            best_val = ws_val[base + s]
            best_idx = ws_idx[base + s]
    _ar_block_merge[dtype, is_min](tid, best_val, best_idx)
    if tid == 0:
        val_ptr[p] = best_val
        idx_ptr[p] = best_idx


# ---------------------------------------------------------------------------
# Host-side dispatch
# ---------------------------------------------------------------------------


@always_inline
def _ar_cpu_scan[
    dtype: DType, is_min: Bool
](
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    base: Int,
    stride: Int,
    n: Int,
) -> Tuple[Scalar[dtype], Int]:
    """One slice, sequentially: the CPU branch of both entry points.

    Factored out so the `with_values` and index-only closures below can be
    written separately -- which they must be, because a `@__copy_capture` list
    is not comptime-conditional, and capturing an unused values pointer would
    change the parallel-for kernel argmin/argmax emit for no reason -- without
    the scan itself existing twice.
    """
    var best = in_ptr[base]
    var best_idx = 0
    for r in range(1, n):
        var v = in_ptr[base + r * stride]
        if _ar_take_next[dtype, is_min](v, best):
            best = v
            best_idx = r
    return (best, best_idx)


@always_inline
def _ar_splits(
    blocks: Int, reduce_n: Int, min_chunk: Int, ctx: DeviceContext
) -> Int:
    """How many ways to cut the reduce axis so the grid fills the device.

    `blocks` is what the unsplit launch would have — one per row on the
    contiguous axis, one per AR_THREADS-wide column tile on the strided one,
    which is NOT the output count and was worth 574us against 45us on a
    4096x4096 dim-0 reduction when it was confused for it. 1 means the grid
    already fills the device; the split path exists for the other end (a full
    reduction is one block)."""
    var target = AR_SPLIT_WAVES * _device_sm_count(ctx)
    if blocks >= target or blocks <= 0:
        return 1
    var by_work = reduce_n // min_chunk
    if by_work < 2:
        return 1
    return min(ceildiv(target, blocks), by_work)


@always_inline
def _ar_merge_launch[
    dtype: DType, is_min: Bool, with_values: Bool
](
    out_addr: Int,
    val_addr: Int,
    ws_val: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_idx: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    outputs: Int,
    splits: Int,
    ctx: DeviceContext,
) raises:
    comptime if with_values:
        _enqueue_cached[_minmax_merge_kernel[dtype, is_min]](
            ctx,
            String(t"minmax_merge_{dtype}_{is_min}"),
            outputs,
            1,
            1,
            AR_THREADS,
            _make_ptr[dtype](val_addr).as_unsafe_any_origin(),
            _make_ptr[DType.int64](out_addr).as_unsafe_any_origin(),
            ws_val.as_immutable(),
            ws_idx.as_immutable(),
            Int64(splits),
        )
    else:
        _enqueue_cached[_argreduce_merge_kernel[dtype, is_min]](
            ctx,
            String(t"argreduce_merge_{dtype}_{is_min}"),
            outputs,
            1,
            1,
            AR_THREADS,
            _make_ptr[DType.int64](out_addr).as_unsafe_any_origin(),
            ws_val.as_immutable(),
            ws_idx.as_immutable(),
            Int64(splits),
        )


@always_inline
def _argreduce_rows[
    dtype: DType, is_min: Bool, with_values: Bool
](
    out_addr: Int,
    val_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """argmin/argmax (and min.dim/max.dim, when `with_values`) over the
    trailing (contiguous) axis of a (rows, cols) buffer; `rows == 1` is the
    full reduction."""
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var val_ptr = _make_ptr[dtype](val_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":
        comptime if with_values:

            @always_inline
            @parameter
            @__copy_capture(out_ptr, val_ptr, in_ptr)
            def func_v[width: Int, alignment: Int = 1](idx: Coord):
                var r = Int(idx[0].value())
                var best, best_idx = _ar_cpu_scan[dtype, is_min](
                    in_ptr, r * cols, 1, cols
                )
                out_ptr[r] = Int64(best_idx)
                val_ptr[r] = best

            _parallel_for[func_v](rows, ctx)
        else:

            @always_inline
            @parameter
            @__copy_capture(out_ptr, in_ptr)
            def func[width: Int, alignment: Int = 1](idx: Coord):
                var r = Int(idx[0].value())
                var _best, best_idx = _ar_cpu_scan[dtype, is_min](
                    in_ptr, r * cols, 1, cols
                )
                out_ptr[r] = Int64(best_idx)

            _parallel_for[func](rows, ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        var splits = _ar_splits(rows, cols, AR_MIN_CHUNK, ctx)
        # Chunk boundaries stay VEC-aligned so every block's vector accesses
        # are, whatever the widest usable VEC turns out to be below.
        comptime WIDEST = max(1, 16 // size_of[dtype]())
        var chunk = 0
        if splits > 1:
            chunk = ceildiv(ceildiv(cols, splits), WIDEST) * WIDEST
            splits = ceildiv(cols, chunk)

        @always_inline
        @parameter
        def _launch[VEC: Int]() raises -> Bool:
            comptime ALIGN = min(16, VEC * size_of[dtype]())
            # Every row base must land on the same alignment, not just the
            # first: row r starts at element r * cols.
            if in_addr % ALIGN != 0 or (
                rows > 1 and (cols * size_of[dtype]()) % ALIGN != 0
            ):
                return False
            if splits <= 1:
                comptime if with_values:
                    _enqueue_cached[_minmax_rows_kernel[dtype, is_min, VEC]](
                        ctx,
                        String(t"minmax_rows_{dtype}_{is_min}_v{VEC}"),
                        rows,
                        1,
                        1,
                        AR_THREADS,
                        val_ptr.as_unsafe_any_origin(),
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        Int64(cols),
                    )
                else:
                    _enqueue_cached[_argreduce_rows_kernel[dtype, is_min, VEC]](
                        ctx,
                        String(t"argreduce_rows_{dtype}_{is_min}_v{VEC}"),
                        rows,
                        1,
                        1,
                        AR_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        Int64(cols),
                    )
                return True
            var ws_val = ctx.enqueue_create_buffer[dtype](rows * splits)
            var ws_idx = ctx.enqueue_create_buffer[DType.int64](rows * splits)
            var ws_val_ptr = ws_val.unsafe_ptr().as_unsafe_any_origin()
            var ws_idx_ptr = ws_idx.unsafe_ptr().as_unsafe_any_origin()
            _enqueue_cached[_argreduce_rows_split_kernel[dtype, is_min, VEC]](
                ctx,
                String(t"argreduce_rows_split_{dtype}_{is_min}_v{VEC}"),
                rows,
                splits,
                1,
                AR_THREADS,
                ws_val_ptr,
                ws_idx_ptr,
                in_ptr.as_unsafe_any_origin().as_immutable(),
                Int64(cols),
                Int64(chunk),
            )
            _ar_merge_launch[dtype, is_min, with_values](
                out_addr, val_addr, ws_val_ptr, ws_idx_ptr, rows, splits, ctx
            )
            return True

        # Widest naturally aligned access first, down to one element: the
        # regime is a compile-time parameter, the choice a runtime one.
        comptime if WIDEST >= 8:
            if _launch[8]():
                return
        comptime if WIDEST >= 4:
            if _launch[4]():
                return
        comptime if WIDEST >= 2:
            if _launch[2]():
                return
        _ = _launch[1]()


@always_inline
def _argreduce_cols[
    dtype: DType, is_min: Bool, with_values: Bool
](
    out_addr: Int,
    val_addr: Int,
    in_addr: Int,
    outer: Int,
    reduce_n: Int,
    inner: Int,
    ctx: DeviceContext,
) raises:
    """argmin/argmax (and min.dim/max.dim, when `with_values`) over a STRIDED
    axis: the input is (outer, reduce, inner) row-major and the index runs over
    the middle axis."""
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var val_ptr = _make_ptr[dtype](val_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var outputs = outer * inner

    if ctx.api() == "cpu":
        comptime if with_values:

            @always_inline
            @parameter
            @__copy_capture(out_ptr, val_ptr, in_ptr)
            def func_v[width: Int, alignment: Int = 1](idx: Coord):
                var p = Int(idx[0].value())
                var base = (p // inner) * reduce_n * inner + (p % inner)
                var best, best_idx = _ar_cpu_scan[dtype, is_min](
                    in_ptr, base, inner, reduce_n
                )
                out_ptr[p] = Int64(best_idx)
                val_ptr[p] = best

            _parallel_for[func_v](outputs, ctx)
        else:

            @always_inline
            @parameter
            @__copy_capture(out_ptr, in_ptr)
            def func[width: Int, alignment: Int = 1](idx: Coord):
                var p = Int(idx[0].value())
                var base = (p // inner) * reduce_n * inner + (p % inner)
                var _best, best_idx = _ar_cpu_scan[dtype, is_min](
                    in_ptr, base, inner, reduce_n
                )
                out_ptr[p] = Int64(best_idx)

            _parallel_for[func](outputs, ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        var lanes = ceildiv(inner, AR_THREADS)
        var splits = _ar_splits(lanes * outer, reduce_n, AR_COLS_MIN_CHUNK, ctx)
        if splits <= 1:
            comptime if with_values:
                _enqueue_cached[_minmax_cols_kernel[dtype, is_min]](
                    ctx,
                    String(t"minmax_cols_{dtype}_{is_min}"),
                    lanes * outer,
                    1,
                    1,
                    AR_THREADS,
                    val_ptr.as_unsafe_any_origin(),
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(lanes),
                )
            else:
                _enqueue_cached[_argreduce_cols_kernel[dtype, is_min]](
                    ctx,
                    String(t"argreduce_cols_{dtype}_{is_min}"),
                    lanes * outer,
                    1,
                    1,
                    AR_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(lanes),
                )
            return
        var chunk = ceildiv(reduce_n, splits)
        splits = ceildiv(reduce_n, chunk)
        var ws_val = ctx.enqueue_create_buffer[dtype](outputs * splits)
        var ws_idx = ctx.enqueue_create_buffer[DType.int64](outputs * splits)
        var ws_val_ptr = ws_val.unsafe_ptr().as_unsafe_any_origin()
        var ws_idx_ptr = ws_idx.unsafe_ptr().as_unsafe_any_origin()
        _enqueue_cached[_argreduce_cols_split_kernel[dtype, is_min]](
            ctx,
            String(t"argreduce_cols_split_{dtype}_{is_min}"),
            lanes * outer,
            splits,
            1,
            AR_THREADS,
            ws_val_ptr,
            ws_idx_ptr,
            in_ptr.as_unsafe_any_origin().as_immutable(),
            Int64(reduce_n),
            Int64(inner),
            Int64(lanes),
            Int64(chunk),
        )
        _ar_merge_launch[dtype, is_min, with_values](
            out_addr, val_addr, ws_val_ptr, ws_idx_ptr, outputs, splits, ctx
        )


@always_inline
def _argreduce_spec_into[
    dtypes: List[DType], is_min: Bool, with_values: Bool = False
](
    a: TensorSpec,
    dst: TensorSpec,
    rdims_t: PyObjectPtr,
    ctx: DeviceContext,
    val_addr: Int = 0,
    val_numel: Int = -1,
) raises:
    """The whole body of ArgminSpec / ArgmaxSpec / MinDimSpec below the dtype
    guard: geometry, output validation and dtype dispatch. The bridges differ
    only in `is_min` and `with_values`, so they cannot drift apart.

    `keepdim` never reaches here: Python allocates the outputs and therefore
    owns their shapes; this side only ever fills contiguous runs. (`dst`, not
    `out`: that name is Mojo's result-argument keyword.) `val_addr` /
    `val_numel` are the values buffer of the (value, index) ops and are unused
    when `with_values` is off.
    """
    var outer = 0
    var reduce_n = 0
    var inner = 0
    if not _adjacent_reduce_geom(a, rdims_t, outer, reduce_n, inner):
        raise Error(
            "mojo spec argreduce: reduce dims must be an adjacent ascending"
            " interval of a contiguous operand (Python pre-materializes)"
        )
    var outputs = outer * inner
    if dst.numel != outputs or not dst.contig or dst.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if dst.dtype != DType.int64:
        raise Error("mojo spec into: output dtype mismatch")
    comptime if with_values:
        if val_numel != outputs:
            raise Error("mojo spec into: output buffer mismatch")
    if outputs <= 0 or reduce_n <= 0:
        return
    comptime for dt in dtypes:
        comptime if _dtype_arg_on[0, dt]():
            if a.dtype == dt:
                if inner == 1:
                    # Trailing reduce dims: the contiguous-axis kernels.
                    _argreduce_rows[dt, is_min, with_values](
                        dst.ptr, val_addr, a.ptr, outer, reduce_n, ctx
                    )
                else:
                    _argreduce_cols[dt, is_min, with_values](
                        dst.ptr, val_addr, a.ptr, outer, reduce_n, inner, ctx
                    )
