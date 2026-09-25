# ===----------------------------------------------------------------------=== #
# GPU kernels + host-side enqueue wrappers for aten::cumsum's fast eager path.
#
# Ported from a standalone kernel-optimization engagement (see the PR that
# added this file for the harness measurements and the negative-result
# writeup on the long-line regime). Two orthogonal regimes, chosen by the
# STRIDE of the scan dim:
#
#  * INNER  (stride 1, e.g. dim=-1 on a contiguous tensor): the scan is
#    along the fastest-varying axis, so parallelizing it needs cooperation
#    across threads. One block owns one independent "line" (or, via
#    grid-stride, several); within a line, `threads`-sized tiles are scanned
#    with `block.prefix_sum` + `block.broadcast` (both from the mojo
#    stdlib -- no hand-rolled shared memory), threaded together by a
#    register-resident serial `carry`.
#
#  * OUTER  (stride == line_len, e.g. dim=0 on a contiguous 2D tensor): the
#    scan is along the SLOW axis and the fast axis is the independent-lines
#    axis, so coalescing falls out for free from adjacent threads owning
#    adjacent lines. No cooperation needed at all: each thread walks its own
#    line with a plain serial accumulate-and-store loop.
#
# A single very long INNER line (few lines, e.g. one row of 10M elements)
# cannot be parallelized by "one block per line" -- one block would own the
# whole GPU's worth of work alone. That regime uses a 3-pass workspace scan
# (reduce chunk totals -> exclusive-scan the small workspace -> re-scan each
# chunk seeded with its workspace prefix) instead of decoupled lookback:
# per AGENTS.md, atomics with sequential ordering are out, and decoupled
# lookback needs acquire/release visibility across SMs that is not something
# this kernel leans on -- the workspace passes are correct by construction
# (kernel-launch ordering on one stream already gives the needed
# happens-before, no fences to get right). This 3-pass design measures
# ~1.6x stock on the extreme "one 10M-element line" shape (vs. today's
# ~18700x SLOWER one-thread-per-row kernel) -- a real regression on an
# absolute roofline, but still a large win, and a decoupled-lookback
# single-pass scan to close the remaining gap is a separate, queued
# follow-up (see the PR description).
#
# This file only contains the device kernels and pointer-taking host
# wrappers (no TensorSpec/CPU-fallback plumbing, no dtype gating loops --
# see `_cumsum_spec_into_go` in tmb/kernels/nn/entry.mojo for that), mirroring how
# `tn_f32_gemm_core.mojo`/`gemm_splitk_common.mojo` sit next to
# `tmb/kernels/matmul/entry.mojo` in the sibling family.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.math import ceildiv
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import _device_sm_count, _enqueue_cached

# Annotated List[DType]: rc1 infers bare `[...]` literals as Array, which no
# longer binds to variant_gates._dtype_supported's `List[DType]` parameter
# (see reduction.mojo's SPEC_ROWRED_DTYPES for the same wrap).
comptime CUMSUM_DTYPES: List[DType] = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.int32,
    DType.int64,
]

# Threads-per-block choices for the INNER family. 1024 covers the primary
# 4096-column benchmark shape in a single tile (zero carry-loop rounds); 256
# is kept for short rows where a 1024-thread block would leave most lanes
# idle on every tile and cost occupancy for no benefit.
comptime INNER_THREADS_BIG = 1024
comptime INNER_THREADS_SMALL = 256
# Row byte threshold between them -- mirrors reduction.mojo's
# LSM_BIG_ROW_BYTES (same reasoning: below this, more, smaller blocks beat
# fewer, bigger ones because there is nothing to hide behind fewer tiles).
comptime INNER_BIG_ROW_BYTES = 8192

comptime OUTER_THREADS = WARP_SIZE

# Threads-per-block and elements-per-thread ("tiles") for the long-line
# workspace path's phase 1/3 kernels -- deliberately a SEPARATE knob from
# INNER_THREADS_{BIG,SMALL} above: that pair was fitted to the "many lines,
# one block owns a whole line" regime, and the workspace regime's optimum
# turned out to sit at a different point entirely (fewer threads/block, not
# more). Fitted on H100 PCIe (114 SMs) -- see the PR description for the
# full swept table (threads x tiles vs. phase1/phase3 device time):
#   * threads=1024, tiles=1 (one block = one 4KB tile, the naive starting
#     point) is LATENCY-bound by the block-wide combine's own round-trip
#     latency, not bandwidth -- far too many small blocks per wave.
#   * Growing `tiles` while reusing `_inner_tile` per round does not help
#     past tiles~8-16: total ROUNDS over the whole line is invariant to
#     tiles under that scheme.
#   * Fix: `_cumsum_chunk_finish_kernel` below does ONE block.prefix_sum per
#     chunk regardless of `tiles` (each thread serially scans its own
#     CONTIGUOUS `tiles`-run in registers first). A big `tiles` on a
#     1024-thread block then blows the register budget and collapses
#     occupancy -- measured: threads=1024 CRASHES outright at tiles=32 with
#     CUDA LAUNCH_OUT_OF_RESOURCES, the exact trap AGENTS.md's "Fully
#     comptime-unrolled inner loops" note warns about. The
#     MAX_THREADS_PER_BLOCK_METADATA launch-bounds hint below is what turns
#     that class of failure into a compile-time-checked launch instead of a
#     silent runtime crash on a future retune.
#   * The real optimum sits at SMALLER blocks: threads=256, tiles=4 (chunk
#     = 1024 elements/block) was the fastest of the swept configurations,
#     with 128/192 threads at tiles=4 statistically indistinguishable; 256
#     is kept for being a round, warp-aligned, already-used-elsewhere
#     (INNER_THREADS_SMALL) number.
comptime WS_THREADS = 256
comptime WS_CHUNK_TILES = 4

# Below this many independent lines (relative to the device's own SM count),
# a regime cannot fill the GPU by line parallelism alone. FILL_WAVES=2 is a
# launch-geometry choice (not device code -- would not show up in a
# PTX/asm diff, see AGENTS.md's cross-compile note), fitted once and
# expected to transfer across NVIDIA parts of different SM counts because
# the actual SM count is read at runtime via `_device_sm_count` below
# (an H100 PCIe has 114 SMs, an H100 SXM 132 -- the compile-time
# `default_device_info` table reports 132 for both, which is why this uses
# the runtime count, not the table).
comptime FILL_WAVES = 2


@always_inline
def _acc_dtype[dtype: DType]() -> DType:
    """Accumulation dtype: f32 for the 16-bit floats (stock torch's cumsum
    accumulates half/bf16 in f32 and casts back down), native dtype
    otherwise -- matches stock behavior and avoids a lossy running sum."""
    comptime if dtype in (DType.float16, DType.bfloat16):
        return DType.float32
    else:
        return dtype


# ===========================================================================
# INNER family
# ===========================================================================


@always_inline
def _inner_tile[
    dtype: DType, acc: DType, threads: Int, exclusive: Bool = False
](
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    base: Int,
    n_valid: Int,
    carry: Scalar[acc],
) -> Scalar[acc]:
    """Scan one tile of up to `threads` contiguous elements starting at
    `base` (element index in `in_ptr`/`out_ptr`), add `carry`, store to
    `out_ptr`. Returns the new carry (running total through this tile,
    always the INCLUSIVE tile sum regardless of `exclusive`, so the caller
    can chain tiles regardless of which flavor it asked for).

    `n_valid <= threads` on a ragged last tile: lanes >= n_valid contribute
    the identity (0) to the scan and do not store (there is nothing at
    `base + tid` for them to write).
    """
    var tid = Int(thread_idx.x)
    var val = Scalar[acc](0)
    if tid < n_valid:
        val = in_ptr[unsafe_offset=base + tid].cast[acc]()
    # Always scan exclusive internally: the inclusive value (needed for the
    # tile-total broadcast either way) is one add away, and the caller's
    # requested flavor is a comptime select, not a second scan.
    var excl = block.prefix_sum[block_size=threads, exclusive=True](val)
    var incl = excl + val

    comptime if exclusive:
        if tid < n_valid:
            out_ptr[unsafe_offset=base + tid] = (carry + excl).cast[dtype]()
    else:
        if tid < n_valid:
            out_ptr[unsafe_offset=base + tid] = (carry + incl).cast[dtype]()

    var last = n_valid - 1
    var tile_total = block.broadcast[block_size=threads](incl, src_thread=last)
    return carry + tile_total


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"cumsum_inner_lines_{dtype}_{threads}_{exclusive}")
def _cumsum_inner_lines_kernel[
    dtype: DType, threads: Int, exclusive: Bool = False
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    line_len_arg: Int64,
    num_lines_arg: Int64,
):
    """One block owns a whole line (grid-stride over lines); within a line,
    `threads`-sized tiles are chained by a register `carry`. Used both as
    the main "many lines" INNER kernel and, with dtype=acc/exclusive=True,
    as phase 2 (the small in-place exclusive scan of a workspace) of the
    long-line path below.
    """
    comptime acc = _acc_dtype[dtype]()
    var line_len = Int(line_len_arg)
    var num_lines = Int(num_lines_arg)
    var line = Int(block_idx.x)
    while line < num_lines:
        var row_base = line * line_len
        var carry = Scalar[acc](0)
        var chunk_start = 0
        while chunk_start < line_len:
            var n_valid = min(threads, line_len - chunk_start)
            carry = _inner_tile[dtype, acc, threads, exclusive](
                in_ptr, out_ptr, row_base + chunk_start, n_valid, carry
            )
            chunk_start += threads
        line += Int(grid_dim.x)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"cumsum_chunk_reduce_{dtype}_{threads}_{tiles}")
def _cumsum_chunk_reduce_kernel[
    dtype: DType, threads: Int, tiles: Int
](
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    workspace_ptr: Pointer[Scalar[_acc_dtype[dtype]()], MutAnyOrigin],
    line_len_arg: Int64,
    chunks_per_line_arg: Int64,
):
    """Phase 1 of the long-line path: one block per (line, chunk) of up to
    `tiles * threads` elements, each thread summing its up-to-`tiles`
    strided elements serially before a single `block.sum` combine, into
    `workspace[line*chunks_per_line+chunk]`.

    `tiles` amortizes fixed per-block cost (CTA launch/teardown, the
    block-wide combine's own latency) over more bytes moved per block --
    at tiles=1 (one element/thread/block) this kernel is latency-bound by
    the SHEER NUMBER of blocks/waves, not bandwidth, on a line long enough
    to need the workspace path at all.
    """
    comptime acc = _acc_dtype[dtype]()
    var line_len = Int(line_len_arg)
    var chunks_per_line = Int(chunks_per_line_arg)
    var flat = Int(block_idx.x)
    var line = flat // chunks_per_line
    var chunk = flat % chunks_per_line
    var chunk_start = chunk * threads * tiles
    var row_base = line * line_len
    var tid = Int(thread_idx.x)
    var partial = Scalar[acc](0)

    comptime for t in range(tiles):
        var local = t * threads + tid
        var global_idx = chunk_start + local
        if global_idx < line_len:
            partial += in_ptr[unsafe_offset=row_base + global_idx].cast[acc]()
    var total = block.sum[block_size=threads](partial)
    if tid == 0:
        workspace_ptr[unsafe_offset=flat] = total


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"cumsum_chunk_finish_{dtype}_{threads}_{tiles}")
def _cumsum_chunk_finish_kernel[
    dtype: DType, threads: Int, tiles: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    workspace_ptr: Pointer[Scalar[_acc_dtype[dtype]()], ImmutAnyOrigin],
    line_len_arg: Int64,
    chunks_per_line_arg: Int64,
):
    """Phase 3: one block per (line, chunk) again, re-scans its own up to
    `tiles * threads` elements (cheap ALU vs. the alternative of storing and
    re-reading the phase-1 local scan -- this is what keeps total traffic at
    2 reads + 1 write instead of 2 reads + 2 writes) seeded with the
    workspace's exclusive prefix. No cross-block carry: every chunk already
    knows its true global starting offset from the workspace.

    Coarsened (blocked, not striped) assignment: thread `tid` owns the
    CONTIGUOUS run `[tid*tiles, tid*tiles+tiles)` of the chunk, scans it
    serially in registers (no synchronization), and only ONE
    `block.prefix_sum` (over the `threads` per-thread local totals) is
    needed to place every thread's run, regardless of `tiles`. Without the
    launch-bounds metadata below, a large `tiles` on a large `threads` block
    (`Array[Scalar[acc], tiles]` fully live across the
    `block.prefix_sum` call) can ask the register allocator for more than
    the hardware has and crash at launch (CUDA LAUNCH_OUT_OF_RESOURCES,
    measured at threads=1024/tiles=32) instead of just running slower --
    the metadata turns that into a compile-time-checked bound.
    """
    comptime acc = _acc_dtype[dtype]()
    var line_len = Int(line_len_arg)
    var chunks_per_line = Int(chunks_per_line_arg)
    var flat = Int(block_idx.x)
    var line = flat // chunks_per_line
    var chunk = flat % chunks_per_line
    var chunk_start = chunk * threads * tiles
    var row_base = line * line_len
    var seed = workspace_ptr[unsafe_offset=flat]
    var tid = Int(thread_idx.x)
    var local_base = chunk_start + tid * tiles

    var local_scan = Array[Scalar[acc], tiles](uninitialized=True)
    var running = Scalar[acc](0)

    comptime for k in range(tiles):
        var idx = local_base + k
        if idx < line_len:
            running += in_ptr[unsafe_offset=row_base + idx].cast[acc]()
        local_scan[k] = running

    var excl = block.prefix_sum[block_size=threads, exclusive=True](running)
    var base = seed + excl

    comptime for k in range(tiles):
        var idx = local_base + k
        if idx < line_len:
            out_ptr[unsafe_offset=row_base + idx] = (base + local_scan[k]).cast[
                dtype
            ]()


# ===========================================================================
# OUTER family: scan dim stride == line_len (e.g. dim=0 on a contiguous 2D
# tensor). `outer`/`inner` generalize past rank 2: elements before the scan
# dim fold into `outer`, elements after it (the contiguous, independent-line
# axis) fold into `inner`. dim=0 on a plain (R, C) tensor is outer=1,
# inner=C, scan_len=R. Only the rank-2 case is wired up on the Python side
# today (see tmb/kernels/nn/entry.mojo); the kernel itself is already generic.
# ===========================================================================


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"cumsum_outer_lines_{dtype}_{threads}")
def _cumsum_outer_kernel[
    dtype: DType, threads: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    scan_len_arg: Int64,
    inner_arg: Int64,
    num_lines_arg: Int64,
):
    comptime acc = _acc_dtype[dtype]()
    var scan_len = Int(scan_len_arg)
    var inner = Int(inner_arg)
    var num_lines = Int(num_lines_arg)
    var gtid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var line = gtid
    while line < num_lines:
        var outer_idx = line // inner
        var inner_idx = line % inner
        var base = outer_idx * scan_len * inner + inner_idx
        var acc_v = Scalar[acc](0)
        var r = 0
        while r < scan_len:
            var addr = base + r * inner
            acc_v += in_ptr[unsafe_offset=addr].cast[acc]()
            out_ptr[unsafe_offset=addr] = acc_v.cast[dtype]()
            r += 1
        line += gstride


# ===========================================================================
# Host-side enqueue wrappers. Pointers are already typed/origin-cast by the
# caller (see tmb/kernels/nn/entry.mojo's `_cumsum_inner_into`/`_cumsum_outer_into`, which
# also own the CPU-fallback branch and the TensorSpec address plumbing).
# ===========================================================================


@always_inline
def _inner_threads(cols: Int, itemsize: Int) -> Int:
    if cols * itemsize > INNER_BIG_ROW_BYTES:
        return INNER_THREADS_BIG
    return INNER_THREADS_SMALL


@always_inline
def enqueue_cumsum_rows[
    dtype: DType
](
    ctx: DeviceContext,
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
) raises:
    """INNER family, 'many lines' regime: no workspace, one block per line
    (grid-stride)."""
    comptime esize = size_of[dtype]()
    var threads = _inner_threads(cols, esize)
    var blocks = min(rows, _device_sm_count(ctx) * 8)
    if threads == INNER_THREADS_BIG:
        _enqueue_cached[_cumsum_inner_lines_kernel[dtype, INNER_THREADS_BIG]](
            ctx,
            blocks,
            1,
            1,
            INNER_THREADS_BIG,
            out_ptr,
            in_ptr,
            Int64(cols),
            Int64(rows),
        )
    else:
        _enqueue_cached[_cumsum_inner_lines_kernel[dtype, INNER_THREADS_SMALL]](
            ctx,
            blocks,
            1,
            1,
            INNER_THREADS_SMALL,
            out_ptr,
            in_ptr,
            Int64(cols),
            Int64(rows),
        )


@always_inline
def cumsum_workspace_lines[dtype: DType](rows: Int, cols: Int) -> Int:
    """Element count (in acc dtype) the caller must allocate for
    `enqueue_cumsum_rows_workspace`'s workspace buffer."""
    return rows * ceildiv(cols, WS_THREADS * WS_CHUNK_TILES)


@always_inline
def enqueue_cumsum_rows_workspace[
    dtype: DType
](
    ctx: DeviceContext,
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    workspace_ptr: Pointer[Scalar[_acc_dtype[dtype]()], MutAnyOrigin],
) raises:
    """INNER family, 'few very long lines' regime: 3-pass workspace scan.
    `workspace_ptr` must hold >= cumsum_workspace_lines[dtype](rows, cols)
    elements of the accumulation dtype. Phase 1/3 threads/tiles are the
    WS_THREADS/WS_CHUNK_TILES fitted separately from the 'many lines'
    regime -- see their definition for the measurements.
    """
    comptime acc = _acc_dtype[dtype]()
    comptime esize = size_of[dtype]()
    var chunks_per_line = ceildiv(cols, WS_THREADS * WS_CHUNK_TILES)
    var total_chunks = rows * chunks_per_line
    var ws_immut = workspace_ptr.as_imm()
    var sm_blocks = min(rows, _device_sm_count(ctx) * 8)

    _enqueue_cached[
        _cumsum_chunk_reduce_kernel[dtype, WS_THREADS, WS_CHUNK_TILES]
    ](
        ctx,
        total_chunks,
        1,
        1,
        WS_THREADS,
        in_ptr,
        workspace_ptr,
        Int64(cols),
        Int64(chunks_per_line),
    )

    # Phase 2: exclusive-scan the (rows, chunks_per_line) workspace in place.
    # Reuses the INNER 'many lines' kernel with dtype=acc: each of the `rows`
    # lines resets its own carry, so per-line independence is automatic.
    var ws_threads = _inner_threads(chunks_per_line, esize)
    if ws_threads == INNER_THREADS_BIG:
        _enqueue_cached[
            _cumsum_inner_lines_kernel[acc, INNER_THREADS_BIG, True]
        ](
            ctx,
            sm_blocks,
            1,
            1,
            INNER_THREADS_BIG,
            workspace_ptr,
            ws_immut,
            Int64(chunks_per_line),
            Int64(rows),
        )
    else:
        _enqueue_cached[
            _cumsum_inner_lines_kernel[acc, INNER_THREADS_SMALL, True]
        ](
            ctx,
            sm_blocks,
            1,
            1,
            INNER_THREADS_SMALL,
            workspace_ptr,
            ws_immut,
            Int64(chunks_per_line),
            Int64(rows),
        )

    _enqueue_cached[
        _cumsum_chunk_finish_kernel[dtype, WS_THREADS, WS_CHUNK_TILES]
    ](
        ctx,
        total_chunks,
        1,
        1,
        WS_THREADS,
        out_ptr,
        in_ptr,
        ws_immut,
        Int64(cols),
        Int64(chunks_per_line),
    )


@always_inline
def enqueue_cumsum_cols[
    dtype: DType
](
    ctx: DeviceContext,
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
) raises:
    """OUTER family (dim=0 on a plain contiguous 2D tensor): outer=1,
    inner=cols, scan_len=rows."""
    var blocks = ceildiv(cols, OUTER_THREADS)
    _enqueue_cached[_cumsum_outer_kernel[dtype, OUTER_THREADS]](
        ctx,
        blocks,
        1,
        1,
        OUTER_THREADS,
        out_ptr,
        in_ptr,
        Int64(rows),
        Int64(cols),
        Int64(cols),
    )
