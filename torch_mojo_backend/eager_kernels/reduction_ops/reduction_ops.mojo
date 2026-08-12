# ===----------------------------------------------------------------------=== #
# Fast eager-mode reduction kernels for mojo_device.
#
# The scalar reductions -- sum / mean / amax / amin / max / min / L2 norm /
# any / all -- are NOT written here: they are one accumulator apiece against
# the generic skeleton in `reduce_skeleton.mojo`, which owns the geometry, the
# split-the-reduce-axis launch policy and the workspace merge. This file keeps
# what is not a scalar reduction: the variance moments (a two-slot payload with
# its own cancellation re-pass) and the fused log-softmax row pass, plus the
# TensorSpec registrations that name which accumulator each aten op gets.
# Arg-reductions and min.dim live in `argreduce_kernels.mojo` (a (value, index)
# payload).
#
# Every one of those mechanisms reads an ADJACENT reduce interval of a
# contiguous operand where it lies, as (outer, reduce, inner) -- so a
# non-trailing reduce dim costs one pass and not a materialized transposed
# copy. Non-adjacent intervals and strided operands are permuted and
# materialized in PYTHON before the call (through the queued strided copy, so
# the transient is budget-metered and covered by the allocation retry); the
# bridges here refuse them rather than scratch-copying. Python mirrors the
# accepted regimes in `aten_fast._reduce_middle_direct_ok` and
# `_arg_strided_direct_ok`.
#
# Raw-pointer calling convention (see elementwise_ops.mojo / nn_ops.mojo):
# tensor operands arrive as element-aligned int addresses, sizes and dtypes as
# ints, ctx_ptr last. Every kernel has a CPU branch (one sequential task per
# output) and a GPU branch.
#
# Floating-point reductions accumulate in float32 (matching torch); integer
# ones accumulate in their own dtype.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.math import ceildiv, exp, log
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

from std.python._cpython import PyObjectPtr, Py_ssize_t

from argreduce_kernels import _argreduce_spec_into
from reduce_skeleton import (
    AllOp,
    AnyOp,
    MaxOp,
    MinOp,
    NormL2Op,
    SumOp,
    _rowred_spec_into_go,
)

from op_utils import (
    FLOAT_DTYPES,
    MAX_RANK,
    TensorSpec,
    _adjacent_reduce_geom,
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
    _vec16_phase,
)

from variant_gates import (
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _register_call,
)


# Row-wise argmin lives in `argreduce_kernels.mojo` (`_argreduce_rows`, with
# `_argreduce_cols` for a strided reduce axis): one comptime-parametrized
# mechanism shared with nn_ops' argmax, so the two ops cannot drift apart in
# either semantics or launch geometry.


# ---------------------------------------------------------------------------
# Variance: a single generic (outer, reduce, inner) moment reduction.
# Covers aten.var.correction for every layout and regime with ONE mechanism
# rather than a kernel per benchmark shape.
#
# The reduced axis is an arbitrary interval of a contiguous tensor, so element
# (o, r, i) sits at `(o * reduce + r) * inner + i` and output (o, i) at
# `o * inner + i`. `inner == 1` is a reduction over the trailing dims (rows of
# a (rows, cols) buffer, full reductions being rows == 1); `inner > 1` is a
# reduction over an interior/leading interval, which used to cost a full
# permuted materialization in Python before the kernel even started.
#
# Three axes of genericity, all comptime or runtime parameters — no shape is
# ever baked in:
#
#   * dtype       — bf16 / f16 / f32 (comptime), accumulating in float32.
#   * layout      — `_moments_contig_kernel` when the reduced axis is the
#                   contiguous one (16-byte vector loads, one block per
#                   (output, split)); `_moments_strided_kernel` when it is
#                   not (one thread per output column, so the loads coalesce
#                   along `inner` and no transpose is materialized).
#   * split count — chosen at launch from the RUNTIME sm count so a reduction
#                   with too few outputs to fill the device (the extreme being
#                   a full reduction: one output) still runs on the whole GPU.
#                   splits == 1 finalizes inside stage 1 and launches nothing
#                   else; splits > 1 writes partials to a workspace that a
#                   merge kernel reduces. A workspace + a separate reduce is
#                   preferred to `Atomic.fetch_add` (sequentially consistent by
#                   default) and is deterministic.
#
# Both stage-1 kernels read the input exactly ONCE, accumulating the moment
# pair (sum of deviations, sum of squared deviations) about an assumed mean K
# — the first element of the reduced slice, which every split of the same
# output re-reads so partials merge by plain addition:
#
#   s = sum(x - K)   q = sum((x - K)^2)   M2 = q - s^2/n   var = M2/(n - corr)
#
# The shift is what makes one pass safe: with K = 0 the two terms of `M2`
# cancel catastrophically on data with a large mean (x ~ 1e6 + noise loses
# every significant digit in float32 — measured 3.5e4 relative error for the
# two-pass predecessor, which suffered the same thing in its float32 mean),
# while shifting by any value near the mean leaves `q` the same order as the
# answer. It costs one broadcast load and one subtract per element, and unlike
# Welford it needs no per-element division and no count bookkeeping in the
# merge (an empty split contributes the additive identity, 0). Accumulation
# stays float32, matching torch's accumulator; the finalize deliberately does
# NOT use float64, which Apple GPUs cannot execute.
#
# Choosing K from the data is what makes it cheap and also what makes it
# fallible: an element is *usually* near the mean, but if the slice's first
# element happens to be a wild outlier then K is the worst shift available and
# `M2` cancels just as badly as K = 0 would (a slice of N(0,1) whose x[0] is
# 1e6 measured 37% error). So the moments are **adaptive**: the point where
# the merged (s, q) for an output first exists is also the point where the
# accurate mean `K + s/n` first exists, and `_moment_cancels` tests right
# there whether the subtraction kept enough bits. If it did not, that output —
# and only that output — is read a second time about the accurate mean, which
# leaves no cancellation at all. Well-conditioned data never pays it (the
# check is a compare on values already in registers), the pathological case
# pays exactly 2x bandwidth, and the decision is made entirely on the device:
# for a single-split reduction the block already holds its whole slice and
# simply scans again, and for a split reduction the merge records a per-output
# flag that a second scan+merge pair consumes.
# ---------------------------------------------------------------------------

comptime MOMENT_THREADS = 256

# Stage-1 blocks in flight targeted when the reduction has too few outputs to
# fill the device on its own. FITTED ON AN H100 PCIe (132 SMs) and nowhere
# else; scaling by the runtime `sm_count` keeps the shape of the grid — not
# its absolute size — portable, but the position of the optimum below is this
# card's. Device us over blocks/SM of 1/2/4/8/16:
#
#   16.7M full reduction   f32  73.8 48.9 44.5 46.8 46.2
#                          bf16 15.6 14.5 14.1 18.4 25.3
#   4096x4096 var(dim=0)   f32    -  50.6 45.6 51.3   -
#                          bf16   -  25.1 22.7 30.5   -
#
# i.e. both layouts and both dtypes bottom out at 4: below it the device is
# starved, above it the workspace the extra shards write (and stage 2 reads
# back) costs more than the parallelism buys.
comptime MOMENT_BLOCKS_PER_SM = 4

# Floor on the reduce extent handed to one split of the strided kernel. Below
# a few rows the workspace slot costs more than the shard computes.
comptime MOMENT_MIN_ROWS_PER_SPLIT = 8

# Cancellation budget for `M2 = q - s^2/n`: re-read a slice about its accurate
# mean once `q / M2` exceeds this, i.e. once more than 4 of float32's 24
# significand bits have been eaten by the subtraction. Not fitted to any card
# — it is a property of the float32 format, and the two ends of the trade are:
#
#   * how often the second read is paid. `q/M2 = 1 + (K - mean)^2 / var`, so
#     tripping 16 needs the slice's FIRST element to sit 3.9 standard
#     deviations from its mean. Uniform data (every benchmarked case) tops out
#     at q/M2 = 4 and never re-passes; Gaussian data re-passes one slice in
#     ~1e4.
#   * the error left when it does not trip. Bounded by 16x the accumulator's
#     own relative error (~4e-7 measured at n = 2^22), i.e. ~6e-6 — still an
#     order of magnitude better than torch's own worst measured case (2.2e-4,
#     a 2048x2048 dim=0 reduction of N(1e4,1)).
comptime MOMENT_CANCEL_RATIO = Float32(16)


@always_inline
def _moment_finish[
    dtype: DType
](s: Float32, q: Float32, n: Int, correction: Float32) -> Scalar[dtype]:
    """(sum of deviations, sum of squared deviations) about an assumed mean,
    over `n` elements -> the corrected variance.

    `max(n - correction, 0)` and the resulting inf/nan for a divisor of zero
    are torch's own rule (`WelfordOps::project` in SharedReduceOps.h): a
    1-element sample with correction=1 is nan there and nan here. M2 is
    clamped at 0 because the subtraction can land a few ulps below zero on a
    constant slice, and a negative variance would poison a downstream sqrt.
    """
    var nf = Float32(n)
    var m2 = q - s * s / nf
    if m2 < 0:  # a few ulps below zero on a constant slice; nan passes through
        m2 = Float32(0)
    var divisor = nf - correction
    if divisor < 0:
        divisor = Float32(0)
    return (m2 / divisor).cast[dtype]()


@always_inline
def _moment_cancels(s: Float32, q: Float32, n: Int) -> Bool:
    """Did `M2 = q - s^2/n` lose too much of the significand to be trusted?

    `q >= s^2/n` always (Cauchy-Schwarz), so the subtraction is pure
    cancellation and destroys about `log2(q / M2)` bits. `q / M2` equals
    `1 + (K - mean)^2 / var`, so it is a direct measure of how bad the assumed
    mean K was: it stays near 1 for any K within a standard deviation of the
    mean and explodes when K is an outlier — which is exactly the case a
    single-pass shifted formulation cannot survive on its own (a slice whose
    FIRST element is 1e6 while the rest are N(0,1) reaches q/M2 = 4e6, i.e.
    22 of float32's 24 bits gone and a 37% error). The caller answers a true
    here by re-reading the slice about the now-known accurate mean, which
    leaves no cancellation at all.

    A `q` of exactly 0 (constant slice) reports false and stays exact; a nan
    reports true and re-passes to the same nan.
    """
    return not ((q - s * s / Float32(n)) * MOMENT_CANCEL_RATIO >= q)


@always_inline
def _moment_flag_repass[
    dtype: DType
](
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: Int,
    outputs: Int,
    meta: Int,
    reduce_n: Int,
    inner: Int,
    s: Float32,
    q: Float32,
):
    """Record, for output `o`, whether its merged moments need a second read
    and what shift that read should use.

    The two meta slots live past the partials: `ws[meta + o]` is the accurate
    mean (the old shift plus `s / n`) and `ws[meta + outputs + o]` is the
    flag. The old shift is not carried through the workspace — it is the first
    element of the slice, one broadcast load away.
    """
    if _moment_cancels(s, q, reduce_n):
        var base = _moment_slice_base(o, reduce_n, inner)
        ws_ptr[meta + o] = in_ptr[base].cast[DType.float32]() + s / Float32(
            reduce_n
        )
        ws_ptr[meta + outputs + o] = Float32(1)
    else:
        ws_ptr[meta + outputs + o] = Float32(0)


@always_inline
def _moment_slice_base(o: Int, reduce_n: Int, inner: Int) -> Int:
    """Flat index of the first element of output `o`'s reduced slice.

    Collapses to `o * reduce_n` when the reduced axis is the contiguous one.
    """
    return (o // inner) * reduce_n * inner + (o % inner)


@always_inline
def _moments_scan_contig[
    dtype: DType, //, V: Int, vec_align: Int
](
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    start: Int,
    n: Int,
    head: Int,
    n_vec: Int,
    vec_start: Int,
    tail_start: Int,
    tid: Int,
    shift: Float32,
    mut s_t: Float32,
    mut q_t: Float32,
):
    """One thread's share of a contiguous shard, as moments about `shift`.

    16-byte vector body plus the scalar head and tail the buffer's own
    alignment phase leaves over. The partition arguments are shift-independent
    so a caller that has to re-scan about a corrected shift passes the same
    ones back in.
    """
    s_t = Float32(0)
    q_t = Float32(0)
    if n <= 0:
        return
    var s_vec = SIMD[DType.float32, V](0)
    var q_vec = SIMD[DType.float32, V](0)
    var v = tid
    while v < n_vec:
        var d = (
            in_ptr.load[width=V, alignment=vec_align](vec_start + v * V).cast[
                DType.float32
            ]()
            - shift
        )
        s_vec += d
        q_vec = d.fma(d, q_vec)
        v += MOMENT_THREADS
    s_t = s_vec.reduce_add()
    q_t = q_vec.reduce_add()

    var jh = tid
    while jh < head:
        var d = in_ptr[start + jh].cast[DType.float32]() - shift
        s_t += d
        q_t += d * d
        jh += MOMENT_THREADS
    var jt = tail_start + tid
    while jt < n:
        var d = in_ptr[start + jt].cast[DType.float32]() - shift
        s_t += d
        q_t += d * d
        jt += MOMENT_THREADS


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MOMENT_THREADS))
)
@__name(t"shifted_moments_contig_{dtype}")
def _moments_contig_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
    outputs_arg: Int64,
    splits_arg: Int64,
    repass_arg: Int64,
    correction: Float32,
):
    """Reduced axis contiguous: block (x=output, y=split) walks its shard of
    one row with 16-byte vector loads and block-reduces the moment pair.

    `ws_ptr` is only touched when splits > 1; the launcher passes a null
    pointer for the fused single-split case, which writes the finished
    variance straight to `out_ptr` and needs no second launch. That fused case
    also handles its own cancellation re-pass in-block, for free — the block
    owns the whole slice, so it just scans it again about the corrected shift.

    `repass_arg != 0` is the split case's second read: the shift comes from
    the meta slots the merge kernel filled in, and a block whose output was
    not flagged exits without reading anything else.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    var repass = Int(repass_arg) != 0
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var split = Int(block_idx.y)
    var base = row * cols

    # Assumed mean: the row's first element, identical in every split of this
    # row so the partial moment pairs merge by addition. On a re-pass it is
    # the accurate mean instead, which leaves no cancellation to correct.
    var shift = in_ptr[base].cast[DType.float32]()
    if repass:
        # Uniform across the block: an early exit here cannot desynchronize
        # the barriers inside `block.sum` below.
        if ws_ptr[2 * splits * outputs + outputs + row] == 0:
            return
        shift = ws_ptr[2 * splits * outputs + row]

    # This split's shard of the row. The last shard may be empty (splits does
    # not divide cols); an empty shard contributes (0, 0), the identity.
    var chunk = ceildiv(cols, splits)
    var j0 = split * chunk
    var j1 = min(cols, j0 + chunk)
    var fused = splits == 1 and not repass

    # Head/body/tail split on the ADDRESS, not on the column index: the shard
    # start has whatever 16-byte phase `in_ptr` and `j0` give it. None of it
    # depends on the shift, so it is computed once and the re-pass below
    # re-walks the same partition.
    var start = base + j0
    var n = j1 - j0
    var phase = _vec16_phase[dtype](Int(in_ptr))
    var head = n
    var n_vec = 0
    if phase >= 0:
        head = (V - (phase + start) % V) % V
        if head > n:
            head = n
        n_vec = (n - head) // V
    var vec_start = start + head
    var tail_start = head + n_vec * V

    var s_t = Float32(0)
    var q_t = Float32(0)
    _moments_scan_contig[V=V, vec_align=vec_align](
        in_ptr,
        start,
        n,
        head,
        n_vec,
        vec_start,
        tail_start,
        tid,
        shift,
        s_t,
        q_t,
    )
    # The block reductions are outside the shard guard inside the scan: every
    # thread of the block must reach them (they barrier internally).
    var bs = block.sum[block_size=MOMENT_THREADS](s_t)
    var bq = block.sum[block_size=MOMENT_THREADS](q_t)

    # Cold path, kept off the straight line above: `block.sum` broadcasts, so
    # the whole block agrees, and a second read about the now-known accurate
    # mean removes the cancellation entirely.
    if fused and _moment_cancels(bs, bq, cols):
        _moments_scan_contig[V=V, vec_align=vec_align](
            in_ptr,
            start,
            n,
            head,
            n_vec,
            vec_start,
            tail_start,
            tid,
            shift + bs / Float32(cols),
            s_t,
            q_t,
        )
        bs = block.sum[block_size=MOMENT_THREADS](s_t)
        bq = block.sum[block_size=MOMENT_THREADS](q_t)

    if tid == 0:
        if fused:
            out_ptr[row] = _moment_finish[dtype](bs, bq, cols, correction)
        else:
            ws_ptr[split * outputs + row] = bs
            ws_ptr[(splits + split) * outputs + row] = bq


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MOMENT_THREADS))
)
@__name(t"shifted_moments_strided_{dtype}")
def _moments_strided_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    reduce_arg: Int64,
    inner_arg: Int64,
    outputs_arg: Int64,
    splits_arg: Int64,
    repass_arg: Int64,
    correction: Float32,
):
    """Reduced axis strided: one thread per output column, walking down the
    reduced axis with stride `inner`.

    Consecutive threads read consecutive addresses, so every step of the walk
    is a fully coalesced load — which is the whole point: reducing a leading
    dimension in place costs one pass over the input instead of a permuted
    copy plus a trailing-dim reduction. Block (x = outer * column-tile,
    y = split); no block reduction is needed because a thread owns its output
    outright, so the cancellation re-pass of the fused single-split case is a
    per-thread decision with no barrier to respect.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var reduce_n = Int(reduce_arg)
    var inner = Int(inner_arg)
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    var repass = Int(repass_arg) != 0

    var tiles = ceildiv(inner, MOMENT_THREADS)
    var blk = Int(block_idx.x)
    var outer_index = blk // tiles
    var i = (blk % tiles) * MOMENT_THREADS + Int(thread_idx.x)
    if i >= inner:
        return

    var chunk = ceildiv(reduce_n, splits)
    var split = Int(block_idx.y)
    var r0 = split * chunk
    var r1 = min(reduce_n, r0 + chunk)

    var base = outer_index * reduce_n * inner + i
    var out_index = outer_index * inner + i
    var shift = in_ptr[base].cast[DType.float32]()
    if repass:
        if ws_ptr[2 * splits * outputs + outputs + out_index] == 0:
            return
        shift = ws_ptr[2 * splits * outputs + out_index]
    var fused = splits == 1 and not repass

    var s = Float32(0)
    var q = Float32(0)
    for r in range(r0, r1):
        var d = in_ptr[base + r * inner].cast[DType.float32]() - shift
        s += d
        q += d * d

    # Cold path: a second read about the now-known accurate mean, for the rare
    # column whose first element was a poor stand-in for its own mean.
    if fused and _moment_cancels(s, q, reduce_n):
        shift += s / Float32(reduce_n)
        s = Float32(0)
        q = Float32(0)
        for r in range(r0, r1):
            var d = in_ptr[base + r * inner].cast[DType.float32]() - shift
            s += d
            q += d * d

    if fused:
        out_ptr[out_index] = _moment_finish[dtype](s, q, reduce_n, correction)
    else:
        ws_ptr[split * outputs + out_index] = s
        ws_ptr[(splits + split) * outputs + out_index] = q


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MOMENT_THREADS))
)
@__name(t"shifted_moments_merge_thread_{dtype}")
def _moments_merge_thread_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    outputs_arg: Int64,
    splits_arg: Int64,
    reduce_arg: Int64,
    inner_arg: Int64,
    repass_arg: Int64,
    correction: Float32,
):
    """Stage 2 for many outputs: one thread per output sums its `splits`
    partials. The workspace is laid out split-major, so the threads of a warp
    read consecutive addresses at every step.

    On the first merge this is also where the split path decides whether its
    assumed mean was good enough: an output whose moments cancelled gets its
    flag set and the accurate mean written beside it, and the re-pass launch
    that follows re-reads only those slices. On the second merge
    (`repass_arg != 0`) an unflagged output is left exactly as the first merge
    wrote it."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    var reduce_n = Int(reduce_arg)
    var meta = 2 * splits * outputs
    var o = Int(block_idx.x) * MOMENT_THREADS + Int(thread_idx.x)
    if o >= outputs:
        return
    if Int(repass_arg) != 0 and ws_ptr[meta + outputs + o] == 0:
        return
    var s = Float32(0)
    var q = Float32(0)
    for k in range(splits):
        s += ws_ptr[k * outputs + o]
        q += ws_ptr[(splits + k) * outputs + o]
    out_ptr[o] = _moment_finish[dtype](s, q, reduce_n, correction)
    if Int(repass_arg) == 0:
        _moment_flag_repass[dtype](
            ws_ptr, in_ptr, o, outputs, meta, reduce_n, Int(inner_arg), s, q
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MOMENT_THREADS))
)
@__name(t"shifted_moments_merge_block_{dtype}")
def _moments_merge_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    outputs_arg: Int64,
    splits_arg: Int64,
    reduce_arg: Int64,
    inner_arg: Int64,
    repass_arg: Int64,
    correction: Float32,
):
    """Stage 2 for few outputs (the full-reduction end of the range): one
    block per output tree-reduces the split partials, because a single thread
    walking hundreds of them would serialize the tail of the launch. Flags the
    cancellation re-pass exactly like the thread-per-output merge above."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    var reduce_n = Int(reduce_arg)
    var meta = 2 * splits * outputs
    var o = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    # Uniform across the block (every thread reads the same flag), so this
    # early exit cannot desynchronize the barriers inside `block.sum`.
    if Int(repass_arg) != 0 and ws_ptr[meta + outputs + o] == 0:
        return
    var s = Float32(0)
    var q = Float32(0)
    for k in range(tid, splits, MOMENT_THREADS):
        s += ws_ptr[k * outputs + o]
        q += ws_ptr[(splits + k) * outputs + o]
    var bs = block.sum[block_size=MOMENT_THREADS](s)
    var bq = block.sum[block_size=MOMENT_THREADS](q)
    if tid == 0:
        out_ptr[o] = _moment_finish[dtype](bs, bq, reduce_n, correction)
        if Int(repass_arg) == 0:
            _moment_flag_repass[dtype](
                ws_ptr,
                in_ptr,
                o,
                outputs,
                meta,
                reduce_n,
                Int(inner_arg),
                bs,
                bq,
            )


@always_inline
def _moment_splits(
    base_blocks: Int, reduce_n: Int, min_per_split: Int, target: Int
) -> Int:
    """How many ways to split the reduced axis so the grid fills the device.

    `base_blocks` is the grid the layout produces with no splitting at all
    (one block per output for the contiguous kernel, one per column tile for
    the strided one). Splitting is pure overhead once that already fills the
    device, so it only kicks in below `target` — derived by the caller from
    the runtime sm count — and never shards the reduced axis finer than
    `min_per_split` elements.
    """
    if base_blocks >= target or reduce_n < 2 * min_per_split:
        return 1
    var by_fill = ceildiv(target, base_blocks)
    var by_work = reduce_n // min_per_split
    var splits = min(by_fill, by_work)
    return max(splits, 1)


@always_inline
def _var_moments[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    outer: Int,
    reduce_n: Int,
    inner: Int,
    correction: Float32,
    ctx: DeviceContext,
) raises:
    """Variance of `in` viewed as (outer, reduce_n, inner) into `outer*inner`
    contiguous outputs."""
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var outputs = outer * inner

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var o = Int(idx[0].value())
            var base = _moment_slice_base(o, reduce_n, inner)
            var shift = in_ptr[base].cast[DType.float32]()
            var s = Float32(0)
            var q = Float32(0)
            # Same adaptive re-pass as the GPU path: a second read only when
            # the first element turned out to be a poor stand-in for the mean.
            for _ in range(2):
                s = Float32(0)
                q = Float32(0)
                for r in range(reduce_n):
                    var d = (
                        in_ptr[base + r * inner].cast[DType.float32]() - shift
                    )
                    s += d
                    q += d * d
                if not _moment_cancels(s, q, reduce_n):
                    break
                shift += s / Float32(reduce_n)
            out_ptr[o] = _moment_finish[dtype](s, q, reduce_n, correction)

        _parallel_for[func](outputs, ctx)
        return

    comptime if has_accelerator():
        comptime sm_count = ctx.default_device_info.sm_count
        var target = MOMENT_BLOCKS_PER_SM * sm_count
        var mout = out_ptr.as_unsafe_any_origin()
        var min_ = in_ptr.as_unsafe_any_origin().as_immutable()

        var base_blocks = outputs
        # The contiguous kernel wants a whole 16-byte vector per thread before
        # a shard is worth its own block; the strided kernel wants a few rows.
        var min_per_split = MOMENT_THREADS * (16 // size_of[dtype]())
        if inner != 1:
            base_blocks = outer * ceildiv(inner, MOMENT_THREADS)
            min_per_split = MOMENT_MIN_ROWS_PER_SPLIT
        var splits = _moment_splits(
            base_blocks, reduce_n, min_per_split, target
        )

        if splits == 1:
            # Fused: stage 1 writes the finished variance -- including its own
            # cancellation re-pass, which the block can do in place because it
            # owns the whole slice. `ws` is never touched.
            var no_ws = _make_ptr[DType.float32](0).as_unsafe_any_origin()
            if inner == 1:
                _enqueue_cached[_moments_contig_kernel[dtype]](
                    ctx,
                    String(t"moments_contig_{dtype}"),
                    base_blocks,
                    1,
                    1,
                    MOMENT_THREADS,
                    mout,
                    no_ws,
                    min_,
                    Int64(reduce_n),
                    Int64(outputs),
                    Int64(1),
                    Int64(0),
                    correction,
                )
            else:
                _enqueue_cached[_moments_strided_kernel[dtype]](
                    ctx,
                    String(t"moments_strided_{dtype}"),
                    base_blocks,
                    1,
                    1,
                    MOMENT_THREADS,
                    mout,
                    no_ws,
                    min_,
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(outputs),
                    Int64(1),
                    Int64(0),
                    correction,
                )
            return

        # Split: partials to a stream-ordered workspace, merged by stage 2.
        # Two meta slots per output sit past the partials: the accurate mean
        # and the cancellation flag the merge writes.
        var ws = ctx.enqueue_create_buffer[DType.float32](
            2 * outputs * splits + 2 * outputs
        )
        var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
        # Pass 0 assumes each slice's first element is a usable mean; its merge
        # flags the outputs where `q - s^2/n` cancelled and records the mean it
        # now knows exactly, and pass 1 re-reads ONLY those slices. Both passes
        # are enqueued unconditionally because the decision is made on the
        # device — testing it on the host would need a sync. With nothing
        # flagged (every benchmarked case, and any data whose first element is
        # within 3.9 sigma of its mean) pass 1 is two launches whose blocks
        # read one float and exit.
        for repass in range(2):
            if inner == 1:
                _enqueue_cached[_moments_contig_kernel[dtype]](
                    ctx,
                    String(t"moments_contig_{dtype}"),
                    base_blocks,
                    splits,
                    1,
                    MOMENT_THREADS,
                    mout,
                    ws_ptr,
                    min_,
                    Int64(reduce_n),
                    Int64(outputs),
                    Int64(splits),
                    Int64(repass),
                    correction,
                )
            else:
                _enqueue_cached[_moments_strided_kernel[dtype]](
                    ctx,
                    String(t"moments_strided_{dtype}"),
                    base_blocks,
                    splits,
                    1,
                    MOMENT_THREADS,
                    mout,
                    ws_ptr,
                    min_,
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(outputs),
                    Int64(splits),
                    Int64(repass),
                    correction,
                )

            # Few outputs cannot keep the device busy one thread each, so they
            # get a block apiece; many outputs would waste 255 of every 256
            # threads that way and get a thread apiece instead.
            if outputs <= sm_count:
                _enqueue_cached[_moments_merge_block_kernel[dtype]](
                    ctx,
                    String(t"moments_merge_block_{dtype}"),
                    outputs,
                    1,
                    1,
                    MOMENT_THREADS,
                    mout,
                    ws_ptr,
                    min_,
                    Int64(outputs),
                    Int64(splits),
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(repass),
                    correction,
                )
            else:
                _enqueue_cached[_moments_merge_thread_kernel[dtype]](
                    ctx,
                    String(t"moments_merge_thread_{dtype}"),
                    ceildiv(outputs, MOMENT_THREADS),
                    1,
                    1,
                    MOMENT_THREADS,
                    mout,
                    ws_ptr,
                    min_,
                    Int64(outputs),
                    Int64(splits),
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(repass),
                    correction,
                )
        # Dropping `ws` schedules a stream-ordered free after the kernels.
        _ = ws^
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
    cols_arg: Int64,
    rows_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var rows = Int(rows_arg)
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var tid = Int(thread_idx.x)

    # 16-byte phase of each buffer, folded into the per-row head length below so
    # every vectorized access lands on a real 16-byte boundary. Loop-invariant.
    var in_phase = _vec16_phase[dtype](Int(in_ptr))
    var out_phase = _vec16_phase[dtype](Int(out_ptr))
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
                    Int64(cols),
                    Int64(rows),
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
                    Int64(cols),
                    Int64(rows),
                )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): dim checks, geometry,
# preallocated-output validation and the launch in one boundary call. Python
# parses the dim spec (`_norm_reduce_dims`) and does dtype promotion; layouts
# outside the adjacent-interval regime raise so the classic
# permute+materialize path keeps handling them. Failed checks raise a real
# NotImplementedError into Python ("take the classic path").
#
# The scalar reductions have no entry point of their own here: they are
# `_rowred_spec_into_go[Op]` from `reduce_skeleton.mojo` with the accumulator
# named at the registration site below, so sum, amax, amin, any, all and the
# vector norm are one function and cannot drift apart.
# ---------------------------------------------------------------------------

# Annotated List[DType]: rc1 infers bare `[...]` literals as Array, which no
# longer binds to variant_gates._dtype_supported's `List[DType]` parameter.
comptime SPEC_ROWRED_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
]


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
    _argreduce_spec_into[SPEC_ROWRED_DTYPES, True](a, out, rdims_t, a.ctx())


def _min_dim_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_v_o: PyObjectPtr,
    out_i_o: PyObjectPtr,
) raises:
    """aten::min.dim values+indices in one call — the multi-output protocol:
    Python allocates both outputs and passes their specs as the last two
    arguments, so one boundary call still fills both.

    The payload is a (value, index) pair, which is the arg-reduction's payload
    with the value kept instead of dropped, so this is `_argreduce_spec_into`
    with `with_values` on rather than a fifth hand-written row kernel — and it
    inherits that mechanism's split path (a full-length row no longer runs in
    one block), its strided-axis kernel (a `dim=0` min.dim no longer
    materializes a transposed copy) and its NaN rule (torch propagates NaN
    through min.dim; the old kernel's plain `<` silently did not)."""
    ref a = _spec_ptr(a_o)[]
    ref out_v = _spec_ptr(out_v_o)[]
    ref out_i = _spec_ptr(out_i_o)[]
    if not _dtype_supported[SPEC_ROWRED_DTYPES](a.dtype):
        raise Error("mojo spec min.dim: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec min.dim: empty input")
    if out_v.dtype != a.dtype:
        raise Error("mojo spec min.dim into: output dtype mismatch")
    _argreduce_spec_into[SPEC_ROWRED_DTYPES, True, with_values=True](
        a, out_i, rdims_t, a.ctx(), out_v.ptr, out_v.numel
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
    # List[DType](...) wrap: op_utils.FLOAT_DTYPES is an rc1 Array literal and
    # _dtype_supported takes List[DType] (scanner still reads index 0 here).
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
        raise Error("mojo spec var: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec var: empty input")
    var correction = Float32(_raw_f64(corr_o))
    # One geometry for every layout: an adjacent ascending reduce interval of
    # a contiguous operand collapses to (outer, reduce, inner), with inner == 1
    # for the trailing-dims case. A non-adjacent interval never reaches here
    # with its original dims — Python permutes and materializes first — so the
    # geometry helper below only has to reproduce the rejection message.
    var outer = 0
    var reduce_n = 0
    var inner = 0
    if not _adjacent_reduce_geom(a, rdims_t, outer, reduce_n, inner):
        var rows = 0
        var cols = 0
        var out_rank = 0
        var oshape = IndexList[MAX_RANK](1)
        _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)
        outer = rows
        reduce_n = cols
        inner = 1

    var ctx = a.ctx()
    var outputs = outer * inner
    var nbytes = outputs * a.itemsize
    _ = nbytes
    if out.numel != outputs or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if outputs > 0:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _var_moments[dt](
                        addr, a.ptr, outer, reduce_n, inner, correction, ctx
                    )


def _log_softmax_spec_into_go(a_o: PyObjectPtr, out_o: PyObjectPtr) raises:
    """log_softmax over the trailing dim; full-shape output. The non-trailing
    dim transpose recursion stays in Python (view ops)."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    # List[DType](...) wrap: see _var_spec_into_go above.
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
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
                    _rowred_spec_into_go[SumOp], "a scalar-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["AmaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[MaxOp], "a scalar-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["AminSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[MinOp], "a scalar-reduction spec op"
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
                    _rowred_spec_into_go[AnyOp], "a scalar-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec); bool",
            )
        comptime if _op_on["AllSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[AllOp], "a scalar-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec); bool",
            )
        comptime if _op_on["NormSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[
                    _rowred_spec_into_go[NormL2Op], "a scalar-reduction spec op"
                ],
                docstring="(a_spec, rdims, keepdim, out_spec); ord=2",
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
