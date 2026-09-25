# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/all_reduce.cuh
#
# Intra-node collectives (allreduce / broadcast / allgather) over IPC-mapped
# peer regions -- a Mojo replacement for the NCCL/RCCL calls a DDP
# ProcessGroup makes inside one node.
#
# Why the data path looks like it does
# ------------------------------------
# The user's tensors live in MAX-allocated memory, which cannot be exported
# with legacy IPC (measured: `cuIpcGetMemHandle` -> CUDA_ERROR_INVALID_VALUE),
# so peers can only ever read the region.  The naive way to bridge that -- copy
# the input into stage_in, run a direct reduce-scatter + all-gather over the
# regions, copy the result back out -- costs two extra full HBM round trips
# (+45 us on the 27 MiB GPT-2 bucket, measured in the feasibility study, 210 us
# vs 165 us direct).  Both copies are avoidable:
#
#   phase 1  PUSH   every rank reads its own input straight out of user memory
#                   and writes shard s into peer s's slot `rank` of the arena.
#                   That write IS the copy-in, and it travels over NVLink.
#   phase 2  REDUCE each rank sums the `world` contributions to its own shard
#                   (its own straight from user memory, the peers' from the
#                   arena), scales, and writes the result to the arena and to
#                   its slice of the user output.
#   phase 3  PULL   each rank reads the other ranks' reduced shards out of
#                   their arenas directly into its user output.
#
# NVLink traffic is 2*(world-1)/world * bytes per GPU -- the unicast minimum,
# the same as a direct reduce-scatter + all-gather -- and no byte is copied
# locally that the direct kernel would not also copy.  The staging is free.
#
# Link direction: AMD (gfx942 / MI300A) takes a different phase 3
# ------------------------------------------------------------------
# On an xGMI mesh a GPU-initiated remote *read* does not scale across links
# while a remote *write* does.  Measured on a 4x MI300A node with the copy
# loop below (perf-work/linkbw.mojo, 168 MiB, per-GPU GB/s): one link 91
# either way; three peers written at once 233; three peers read at once 93,
# and a ring of simultaneous readers 56.  RCCL reaches the same 236 GB/s at
# 512 MiB and gets there the same way -- its P2P transport hard-wires
# `read = 0` on AMD (rccl:src/graph/paths.cc:441 only lets compCap 80 read),
# so the sender stores into the receiver's buffer and no rank ever loads
# across a link.
#
# So on AMD phase 3 is a second push instead of a pull:
#
#   phase 2' each rank writes its reduced shard into EVERY peer's gather slot
#            (and into its own slice of the user output),
#   phase 3' each rank copies the `world-1` gather slots of its OWN region
#            into the user output -- a local HBM copy, because the user's
#            output cannot be IPC-mapped and so a peer cannot write it
#            directly.
#
# The cross-link traffic is bit for bit the same 2*(world-1)/world * bytes; it
# has only changed direction.  The price is that local copy, and it is small:
# the region is `hipDeviceMallocUncached` but reads out of it at full HBM rate
# (1434 GB/s measured against 1453 for a normal buffer), so 0.75 * message
# costs ~90 us at 168 MiB against the ~1130 us the wire needs.
#
# NVIDIA keeps the pull: behind a switch every direction is equivalent, the
# pull needs no gather slots and no local copy, and the H100 numbers in
# agents_docs/mojo_collectives_kernel_results.md were measured with it.  The split is
# a `comptime if has_amd_gpu_accelerator()` in the kernel and in the one host
# line that sizes the arena, so NVIDIA device code is unchanged.
#
# The hierarchical (multi-node) allreduce splits that into
# `reduce_scatter_stage` (phases 1-2) and `allgather_finish` (phase 3), with
# the vendor library's inter-node allreduce of one shard in between; the split
# spends two generations and one extra launch. See the block comment above
# `_rs_stage_kernel`.
#
# Every host function takes the DeviceStream to enqueue on (production wraps
# the caller's foreign cudaStream_t with `DeviceContext.create_external_stream`);
# `ctx` is only the handle used to compile and cache the DeviceFunction.
#
# Builds for both targets:
#   uv run --no-sync mojo build device/symmetric/all_reduce.mojo --target-accelerator sm_90a
#   uv run --no-sync mojo build device/symmetric/all_reduce.mojo --target-accelerator gfx942

from std.collections import Array
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.utils import StaticTuple
from std.sys import size_of
from max.gpu.host import DeviceContext, DeviceStream

from tmb.ccl.device.common import (
    _enqueue_cached,
    device_now_ns,
    spin_timeout_ns,
)
from tmb.ccl.device.symmetric.data_ops import (
    _copy_scalar_tail,
    _copy_span,
    _copy_span_scaled,
    _copy_vec,
    _share,
)
from tmb.ccl.device.symmetric.primitives import (
    _check_common,
    _gather_slot,
    _peer_step,
    _peer_step0,
    _region_ptrs,
    _shard_cnt,
    _shard_off,
    _shard_per,
    _vcount,
    _vstart,
)
from tmb.ccl.include.device import (
    BLOCK,
    ERR_AG_FINISH_SYNC,
    ERR_ALLREDUCE_SYNC,
    ERR_RS_STAGE_SYNC,
    MAX_WORLD,
    _AMD,
    _AR_BIG_BLOCKS,
    _AR_BIG_BYTES,
    _AR_MAX_BLOCKS,
    _ONESHOT_MAX_BYTES,
    _SIGNAL_BYTES,
    _UNROLL,
    _align_up,
)
from tmb.ccl.include.nccl_device.lsa_barrier import _flag_target, _sync


# ===-------------------------------------------------------------------=== #
# Allreduce -- two-shot (push / reduce / pull)
# ===-------------------------------------------------------------------=== #
#
# Buffer-reuse invariant. Every collective in this file opens with a start
# barrier, so the whole arena obeys one rule:
#
#     no rank writes an arena byte for generation g until every rank has
#     finished reading arena bytes for generation g-1
#
# (a rank reaches generation g's start barrier only after its own generation
# g-1 work has retired, and nobody passes that barrier until all have arrived).
# That is what lets collectives of different kinds and sizes share the arena
# and interleave freely -- which they do: DDP issues a 4-byte one-shot
# allreduce and an 8-byte allgather in between 27 MiB two-shot allreduces, and
# their staging layouts overlap. Dropping the start barrier saves ~2.5 us and
# is only sound for a run of identically shaped collectives; it was measured
# (166.4 us vs 169 us at 27 MiB) and rejected as a correctness trap.
#
# Within a call the two data syncs order the three phases:
#     start(g) < push(g) < A(g) < reduce(g) < B(g) < pull(g) < start(g+1)
# (on AMD: reduce-and-push-back instead of reduce, local gather instead of
# pull -- same three phases, same two data syncs) and a sync is a full N-way
# rendezvous of matching block indices. In-place (in_ptr == out_ptr) is safe
# because the phases are block-matched: block b writes exactly the elements
# block b read.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allreduce_push_reduce_pull_{dtype}_w{NW}")
def _ar_twoshot_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    shard_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(numel)
    var nvec = n // W
    var q = nvec // world
    var rem = nvec % world
    var tail = n - nvec * W
    var slot_stride = Int(slot_stride_b)
    var push_off = Int(push_off_b)
    var shard_off = Int(shard_off_b)

    # --- phase 0: start barrier -- nobody writes the arena for generation g
    # until every rank has finished reading it for generation g-1 ------------
    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base, t0, timeout_ns
    ):
        return

    # --- phase 1: push shard s of my input into peer s's slot `rank` --------
    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var src = in_ptr.unsafe_offset(_vstart(s, q, rem) * W)
        var vc = _vcount(s, q, rem)
        var dst = (
            regions[s]
            .unsafe_offset(push_off + slot_stride * rank)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
        if tail > 0 and s == world - 1:
            _copy_scalar_tail(dst, src, vc * W, tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    # --- phase 2: reduce my shard, to the arena and to the user output ------
    var my_vs = _vstart(rank, q, rem)
    var my_vc = _vcount(rank, q, rem)
    var my_tail = tail if rank == world - 1 else 0
    var uin = in_ptr.unsafe_offset(my_vs * W)
    var uout = out_ptr.unsafe_offset(my_vs * W)
    # NVIDIA publishes the reduced shard in its own region for the peers to
    # pull.  AMD pushes it into theirs instead (phase 2b below), so there is
    # nothing to publish locally; `shard` then aliases `uout` and the store to
    # it is elided at compile time.
    var shard = (
        uout if _AMD else regions[rank]
        .unsafe_offset(shard_off)
        .unsafe_bitcast[Scalar[dtype]]()
    )

    # Slot pointers are formed by arithmetic inside the unrolled loop, never
    # held in an array: a stack array of `world` pointers is demoted to local
    # memory (MOCO-1431) and turns every payload load into a generic-address
    # `ld.v4.b32` plus an `ld.local.b64` of the pointer itself.
    var slots = regions[rank].unsafe_offset(push_off)

    # AMD only: the peers' gather slots this rank pushes its reduced shard
    # into, hoisted out of the element loop.  A `comptime for` writes and
    # reads this array at constant indices only, so SROA keeps the `world-1`
    # pointers in registers -- the MOCO-1431 demotion above bites when the
    # index is a runtime value, which is why the reduce's own source pointers
    # are still formed by arithmetic.  `NW == 0` (world 3, 5, 6, 7) has no
    # comptime bound and falls back to a second pass over the shard.
    var gout = Array[Pointer[Scalar[dtype], MutAnyOrigin], MAX_WORLD](
        uninitialized=True
    )
    comptime if _AMD and NW > 0:
        comptime for j in range(1, NW):
            var pj = rank + _peer_step(j, NW)
            if pj >= NW:
                pj -= NW
            gout[j] = (
                regions[pj]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(rank, pj))
                .unsafe_bitcast[Scalar[dtype]]()
            )

    for v in range(tid, my_vc, stride):
        var acc = _share(
            uin.unsafe_load[width=W, alignment=16](v * W).cast[accum](), scale
        )
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += _share(
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += _share(
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        var res = acc.cast[dtype]()
        comptime if not _AMD:
            shard.unsafe_store[width=W, alignment=16](v * W, res)
        uout.unsafe_store[width=W, alignment=16](v * W, res)
        # One reduce, `world` stores: my output slice and every peer's gather
        # slot.  This is NCCL's MULTIDSTS shape (rccl:src/device/
        # common_kernel.h reduceCopyPacks stores to all destinations from one
        # accumulator) and it keeps the local reduce traffic inside the wire
        # transfer instead of adding a pass in front of it.
        comptime if _AMD and NW > 0:
            comptime for j in range(1, NW):
                gout[j].unsafe_store[width=W, alignment=16](v * W, res)

    for i in range(tid, my_tail, stride):
        var k = my_vc * W + i
        var a = _share[accum, 1](uin[unsafe_offset=k].cast[accum](), scale)
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += _share[accum, 1](
                slots.unsafe_offset(slot_stride * p)
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum](),
                scale,
            )
        comptime if not _AMD:
            shard[unsafe_offset=k] = a.cast[dtype]()
        uout[unsafe_offset=k] = a.cast[dtype]()
        comptime if _AMD and NW > 0:
            comptime for j in range(1, NW):
                gout[j][unsafe_offset=k] = a.cast[dtype]()

    comptime if _AMD and NW == 0:
        # --- phase 2b (AMD, generic world): push my reduced shard into every
        # peer's gather slot in a second pass.  Only worlds 3, 5, 6 and 7 come
        # here; 2, 4 and 8 fuse the stores into the reduce above.  Thread
        # `tid` reads back only the elements thread `tid` just wrote (both
        # loops walk `{tid, tid+stride, ...}`), so no fence is involved.
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            var dst = (
                regions[p]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(rank, p))
                .unsafe_bitcast[Scalar[dtype]]()
            )
            _copy_vec[dtype, W, U](dst, uout, my_vc, tid, stride)
            if my_tail > 0:
                _copy_scalar_tail(dst, uout, my_vc * W, my_tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 2, t0, timeout_ns
    ):
        return

    # --- phase 3: the peers' reduced shards into the user output ------------
    # NVIDIA reads them across the fabric; AMD reads them out of its own
    # region, where phase 2b's pushes left them (module header, "Link
    # direction").
    for i in range(1, world):
        var p = rank + _peer_step(i, world)
        if p >= world:
            p -= world
        var vs = _vstart(p, q, rem)
        var vc = _vcount(p, q, rem)
        var src: Pointer[Scalar[dtype], MutAnyOrigin]
        comptime if _AMD:
            src = (
                regions[rank]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(p, rank))
                .unsafe_bitcast[Scalar[dtype]]()
            )
        else:
            src = (
                regions[p]
                .unsafe_offset(shard_off)
                .unsafe_bitcast[Scalar[dtype]]()
            )
        var dst = out_ptr.unsafe_offset(vs * W)
        _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
        if tail > 0 and p == world - 1:
            _copy_scalar_tail(dst, src, vc * W, tail, tid, stride)


# ===-------------------------------------------------------------------=== #
# Allreduce -- one-shot (push whole input / reduce), for small messages
# ===-------------------------------------------------------------------=== #
#
# (world-1)x the NVLink bytes of the two-shot path but one data sync instead of
# two, which wins while latency dominates: measured 9.3 us against 19.0 us at
# 128 KiB, with the crossover just above 512 KiB.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allreduce_oneshot_{dtype}_w{NW}")
def _ar_oneshot_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(numel)
    var nvec = n // W
    var tail = n - nvec * W
    var slot_stride = Int(slot_stride_b)
    var push_off = Int(push_off_b)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base, t0, timeout_ns
    ):
        return

    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var dst = (
            regions[s]
            .unsafe_offset(push_off + slot_stride * rank)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_vec[dtype, W, U](dst, in_ptr, nvec, tid, stride)
        if tail > 0:
            _copy_scalar_tail(dst, in_ptr, nvec * W, tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    var slots = regions[rank].unsafe_offset(push_off)

    for v in range(tid, nvec, stride):
        var acc = _share(
            in_ptr.unsafe_load[width=W, alignment=16](v * W).cast[accum](),
            scale,
        )
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += _share(
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += _share(
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        out_ptr.unsafe_store[width=W, alignment=16](v * W, acc.cast[dtype]())

    for i in range(tid, tail, stride):
        var k = nvec * W + i
        var a = _share[accum, 1](in_ptr[unsafe_offset=k].cast[accum](), scale)
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += _share[accum, 1](
                slots.unsafe_offset(slot_stride * p)
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum](),
                scale,
            )
        out_ptr[unsafe_offset=k] = a.cast[dtype]()


# ===-------------------------------------------------------------------=== #
# Split allreduce -- reduce-scatter stage, then all-gather finish
# ===-------------------------------------------------------------------=== #
#
# The hierarchical (multi-node) allreduce is
#
#   reduce_scatter_stage(g)   my shard, summed over the node, into my stage_out
#   <inter-node step>         NCCL/RCCL allreduces that shard across nodes,
#                             in place, on the same stream, among the ranks
#                             that share my local index
#   allgather_finish(g+1)     pull every rank's now-global shard into my output
#
# Layout. Shards follow `shard_range` (equal, 16-byte aligned, last one short).
#   push slots  stage_in base, `world-1` slots of `per * elem_bytes` bytes.
#               Slot indices are *compacted*: writer r stores into destination
#               rank s's slot `r if r < s else r-1`, because s never writes its
#               own slot (it reads its own contribution from user memory). The
#               compaction is not cosmetic: `world` uncompacted slots can be
#               up to 16*world bytes larger than stage_in when
#               numel*elem_bytes == cap (the per-shard round-up to the vector
#               width, times world), which would spill onto rank 0's shard in
#               stage_out. `world-1` of them never can -- checked exhaustively
#               for every dtype width, world and cap.
#   shard       stage_out + offset(rank) * elem_bytes, i.e. stage_out is an
#               image of the whole buffer of which only my shard is live. That
#               placement is what makes the ABI layer's job one line -- the
#               inter-node collective gets `stage_out + offset*elem_bytes` and
#               `count` -- and it makes the pull address the same on both
#               sides. It always fits: offset+count <= numel and
#               numel*elem_bytes <= cap_bytes.
#
# Ordering. Three things happen between the two kernels that the fused
# allreduce never has to think about, and all three are covered without a new
# protocol:
#
#  1. A foreign library writes my stage_out shard in place. Nobody may read it
#     before that write lands. `allgather_finish`'s start barrier is the fence:
#     a rank publishes its generation g+1 flags only from inside that kernel,
#     which its stream starts only after its inter-node op has completed, so
#     seeing peer p's flag implies p's shard is final.
#  2. Peer p's shard was written by a *previous kernel* (p's reduce_scatter_
#     stage), not by the thread that publishes the flag. Stream order makes
#     that kernel's writes happen-before the flag's release store, and the
#     release/acquire pair is system-scoped and cumulative, so they are visible
#     to the acquiring reader. (On gfx942 `_sync`'s AMD-only release fence adds
#     the `buffer_wbl2 sc0 sc1` writeback that the workgroup barrier omits.)
#     Note the consequence: block-index matching, which the fused allreduce
#     relies on *within* a kernel, is not needed across this boundary -- the
#     two kernels may be launched with different grids, and they are.
#  3. Arena reuse after the pulls. Nothing extra is needed: the next collective
#     of any kind opens with a start barrier, and a rank reaches it only after
#     its own `allgather_finish` retired, so no generation g+2 write can race a
#     generation g+1 pull. That is the same one rule as everywhere else in this
#     file -- the split pair just spends two generations instead of one.
#
# Each call consumes one generation. `reduce_scatter_stage` uses flag phases
# 0 and 1, `allgather_finish` phase 0.


@always_inline
def _rs_stage_body[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    arena_off: Int,
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    per: Int,
    slot_stride: Int,
    push_off: Int,
    out_off: Int,
    world: Int,
    rank: Int,
    tid: Int,
    stride: Int,
    flag_base: UInt64,
    scale: Float32,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """The reduce-scatter half, as a body two kernels share.

    `_rs_stage_kernel` is one launch of it; the pipelined inter-node kernel
    (all_reduce_gin.mojo) runs it once per chunk with `arena_off` naming the
    arena and `tid`/`stride` the caller's grid-stride slice. Everything the
    launcher used to compute from `Int64` arguments arrives here as `Int`.

    False means a barrier gave up (deadline or abort); it has already
    recorded why, and the caller must return without touching the arena.
    """
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    comptime esize = size_of[dtype]()

    # --- phase 0: start barrier (the arena-reuse invariant) -----------------
    if not _sync(
        regions,
        world,
        rank,
        ERR_RS_STAGE_SYNC,
        flag_base,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # --- phase 1: push shard s of my input into peer s's slot for me --------
    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var off = _shard_off(n, per, s)
        var cnt = _shard_cnt(n, per, s)
        if cnt <= 0:
            continue
        var dst = (
            regions[s]
            .unsafe_offset(
                arena_off
                + push_off
                + slot_stride * (rank if rank < s else rank - 1)
            )
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_span[dtype, W, U](
            dst, in_ptr.unsafe_offset(off), cnt, tid, stride
        )

    if not _sync(
        regions,
        world,
        rank,
        ERR_RS_STAGE_SYNC,
        flag_base + 1,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # --- phase 2: sum the `world` contributions to my shard into stage_out --
    # Every contribution is scaled as it enters the fp32 accumulator
    # (`_share`), so the store to the wire dtype carries an average and never
    # a sum: the inter-node step sums these node partials again in that dtype,
    # and an unscaled fp16 sum of 8 x 10000 is already inf.
    # `allgather_finish` then runs with scale 1.
    var my_off = _shard_off(n, per, rank)
    var my_cnt = _shard_cnt(n, per, rank)
    if my_cnt <= 0:
        return True
    var uin = in_ptr.unsafe_offset(my_off)
    var shard = (
        regions[rank]
        .unsafe_offset(arena_off + out_off + my_off * esize)
        .unsafe_bitcast[Scalar[dtype]]()
    )
    # Slot pointers are formed arithmetically, never held in a stack array:
    # such an array is demoted to local memory (MOCO-1431) and every payload
    # load becomes a generic-address `ld.v4.b32` plus an `ld.local.b64`.
    var slots = regions[rank].unsafe_offset(arena_off + push_off)
    var my_vc = my_cnt // W

    for v in range(tid, my_vc, stride):
        var acc = _share(
            uin.unsafe_load[width=W, alignment=16](v * W).cast[accum](), scale
        )
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += _share(
                    slots.unsafe_offset(
                        slot_stride * (p if p < rank else p - 1)
                    )
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += _share(
                    slots.unsafe_offset(
                        slot_stride * (p if p < rank else p - 1)
                    )
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum](),
                    scale,
                )
        shard.unsafe_store[width=W, alignment=16](v * W, acc.cast[dtype]())

    for i in range(tid, my_cnt - my_vc * W, stride):
        var k = my_vc * W + i
        var a = _share[accum, 1](uin[unsafe_offset=k].cast[accum](), scale)
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += _share[accum, 1](
                slots.unsafe_offset(slot_stride * (p if p < rank else p - 1))
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum](),
                scale,
            )
        shard[unsafe_offset=k] = a.cast[dtype]()
    return True


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_reduce_scatter_stage_{dtype}_w{NW}")
def _rs_stage_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    per_e: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    out_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    _ = _rs_stage_body[dtype, W, U, NW](
        regions,
        0,
        in_ptr,
        Int(numel),
        Int(per_e),
        Int(slot_stride_b),
        Int(push_off_b),
        Int(out_off_b),
        NW if NW > 0 else Int(world_i),
        Int(rank_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
        flag_base,
        scale,
        device_now_ns(),
        timeout_ns,
    )


@always_inline
def _ag_finish_body[
    dtype: DType, W: Int, U: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    arena_off: Int,
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    per: Int,
    out_off: Int,
    world: Int,
    rank: Int,
    tid: Int,
    stride: Int,
    flag_base: UInt64,
    scale: Float32,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """The all-gather half, as a body two kernels share (see
    `_rs_stage_body`)."""
    comptime esize = size_of[dtype]()

    # Start barrier. Doubles as the wait for every peer's inter-node step: a
    # peer publishes this flag from inside this kernel, which its stream runs
    # after that step.
    if not _sync(
        regions,
        world,
        rank,
        ERR_AG_FINISH_SYNC,
        flag_base,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # My own shard is pulled out of my own stage_out like everyone else's: the
    # inter-node step rewrote it, so the reduce-scatter's result in the user
    # buffer would be stale even if it had been written there.
    for i in range(world):
        var p = rank + _peer_step0(i, world)
        if p >= world:
            p -= world
        var off = _shard_off(n, per, p)
        var cnt = _shard_cnt(n, per, p)
        if cnt <= 0:
            continue
        var src = (
            regions[p]
            .unsafe_offset(arena_off + out_off + off * esize)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_span_scaled[dtype, W, U](
            out_ptr.unsafe_offset(off), src, cnt, tid, stride, scale
        )
    return True


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allgather_finish_{dtype}_w{NW}")
def _ag_finish_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    per_e: Int64,
    out_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    _ = _ag_finish_body[dtype, W, U](
        regions,
        0,
        out_ptr,
        Int(numel),
        Int(per_e),
        Int(out_off_b),
        NW if NW > 0 else Int(world_i),
        Int(rank_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
        flag_base,
        scale,
        device_now_ns(),
        timeout_ns,
    )


@always_inline
def _launch_allreduce[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
    one_shot: Bool,
) raises:
    var arena = _SIGNAL_BYTES
    var arena_end = _SIGNAL_BYTES + 2 * cap_bytes
    var esize = size_of[dtype]()
    var nvec = numel // W
    var tail = numel - nvec * W
    var ip = Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr)
    var op = Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr)

    if one_shot:
        # `world` slots, each a whole message, at the base of the arena. They
        # overlap the two-shot staging on purpose: the start barrier orders
        # every generation's writes after the previous generation's reads, so
        # collectives of different shapes and sizes may interleave freely.
        var slot = _align_up(numel * esize, 16)
        if world * slot > 2 * cap_bytes:
            raise Error("collectives: one-shot slots exceed the region")
        var push_off = arena
        var blocks = min(_AR_MAX_BLOCKS, max(1, (nvec + BLOCK - 1) // BLOCK))
        _enqueue_cached[_ar_oneshot_kernel[dtype, W, _UNROLL, NW]](
            ctx,
            stream,
            String(t"ar1_{dtype}_{NW}"),
            blocks,
            regions,
            ip,
            op,
            Int64(numel),
            Int64(slot),
            Int64(push_off),
            Int32(world),
            Int32(rank),
            _flag_target(generation, 0),
            scale,
            spin_timeout_ns(),
        )
        return

    var q = nvec // world
    var rem = nvec % world
    var max_shard_elems = max((q + (1 if rem > 0 else 0)) * W, q * W + tail)
    var slot = _align_up(max_shard_elems * esize, 16)
    var push_off = arena
    var shard_off = arena + world * slot
    # NVIDIA parks one reduced shard at `shard_off` for the peers to pull;
    # AMD parks `world-1` gather slots there for the peers to push into
    # (module header, "Link direction").  `world*slot` is already about
    # `numel*esize` <= cap, so `world` more slots would not fit an arena of
    # 2*cap when the message is exactly cap bytes -- hence the compacted
    # `world-1`, whose worst case is cap*(2 - 1/world) plus alignment.
    comptime tail_slots = 1
    var end = shard_off + (max(world - 1, 1) if _AMD else tail_slots) * slot
    if end > arena_end:
        raise Error("collectives: allreduce staging exceeds the region")
    var cap_blocks = (
        _AR_BIG_BLOCKS if numel * esize >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    var blocks = min(cap_blocks, max(1, (q + 1 + BLOCK - 1) // BLOCK))
    _enqueue_cached[_ar_twoshot_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"ar2_{dtype}_{NW}"),
        blocks,
        regions,
        ip,
        op,
        Int64(numel),
        Int64(slot),
        Int64(push_off),
        Int64(shard_off),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


def allreduce[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    """Elementwise `out[i] = scale * sum over ranks of in_r[i]`, on `stream`.

    `in_ptr` may equal `out_ptr`. `numel * size_of[dtype]()` must be <=
    `cap_bytes`. `scale` is applied on the final write and ignored for integer
    dtypes (the caller passes 1.0 there).
    """
    _check_common(rank, world, cap_bytes, generation)
    if numel == 0:
        return
    if numel < 0:
        raise Error("collectives: numel must be >= 0")
    comptime W = 16 // size_of[dtype]()
    if numel * size_of[dtype]() > cap_bytes:
        raise Error("collectives: allreduce message exceeds cap_bytes")
    if (in_ptr | out_ptr) % 16 != 0:
        # The payload loops use 16-byte vector loads/stores, which fault (or
        # silently misbehave) on a misaligned address. Every allocator-returned
        # pointer satisfies this; a mid-tensor view may not, and the caller
        # must stage such a tensor into an aligned buffer rather than have this
        # kernel guess. Byte collectives have a scalar fallback and need no
        # such rule.
        raise Error(
            "collectives: allreduce needs 16-byte aligned in_ptr and out_ptr"
        )
    var rp = _region_ptrs(regions, rank, world)
    # world == 1 needs no special case: the push and pull loops are empty, the
    # sync is a self-rendezvous, and the reduce degenerates to out = scale*in.
    var bytes = numel * size_of[dtype]()
    var one_shot = bytes <= _ONESHOT_MAX_BYTES and (
        world * _align_up(bytes, 16) <= 2 * cap_bytes
    )

    if world == 8:
        _launch_allreduce[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    elif world == 4:
        _launch_allreduce[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    elif world == 2:
        _launch_allreduce[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    else:
        _launch_allreduce[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )


@always_inline
def _split_blocks[dtype: DType, W: Int](numel: Int, per: Int) -> Int:
    """Grid for both halves of the split allreduce: sized by the shard, capped
    by the same one-wave rule the fused kernel uses (RESULTS.md block sweep).
    The two halves need not agree -- see ordering note 2 above the kernels."""
    var cap_blocks = (
        _AR_BIG_BLOCKS if numel * size_of[dtype]()
        >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    return min(cap_blocks, max(1, (per // W + 1 + BLOCK - 1) // BLOCK))


def _check_split[
    dtype: DType, W: Int
](
    rank: Int,
    world: Int,
    ptr: Int,
    numel: Int,
    cap_bytes: Int,
    generation: Int,
    what: String,
) raises -> Int:
    """Shared preconditions of the two split entry points; returns `per`."""
    _check_common(rank, world, cap_bytes, generation)
    if numel < 0:
        raise Error("collectives: numel must be >= 0")
    if numel * size_of[dtype]() > cap_bytes:
        raise Error("collectives: " + what + " message exceeds cap_bytes")
    if ptr % 16 != 0:
        # Same rule as `allreduce`: the payload loops use 16-byte vectors.
        raise Error("collectives: " + what + " needs a 16-byte aligned buffer")
    var per = _shard_per(numel, world, W)
    if (world - 1) * per * size_of[dtype]() > cap_bytes:
        # Cannot happen -- the compacted slot table is ~(world-1)/world of the
        # message -- but the arena has no guard page, so check it anyway.
        raise Error("collectives: split push slots exceed the region")
    return per


@always_inline
def _launch_rs_stage[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    numel: Int,
    per: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    generation: Int,
    scale: Float32,
) raises:
    _enqueue_cached[_rs_stage_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"rs_{dtype}_{NW}"),
        _split_blocks[dtype, W](numel, per),
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr),
        Int64(numel),
        Int64(per),
        Int64(per * size_of[dtype]()),
        Int64(_SIGNAL_BYTES),
        Int64(_SIGNAL_BYTES + cap_bytes),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


@always_inline
def _launch_ag_finish[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    out_ptr: Int,
    numel: Int,
    per: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    _enqueue_cached[_ag_finish_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"agf_{dtype}_{NW}"),
        _split_blocks[dtype, W](numel, per),
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(numel),
        Int64(per),
        Int64(_SIGNAL_BYTES + cap_bytes),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


def reduce_scatter_stage[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    generation: Int,
    scale: Float32 = Float32(1.0),
) raises:
    """First half of a hierarchical allreduce: push + local reduce, on `stream`.

    When the enqueued work completes, this rank's shard --
    `shard_range(numel, world, rank, size_of[dtype]())` -- summed over the
    `world` node-local ranks and multiplied by `scale` (ignored for integer
    dtypes), sits in this rank's own stage_out, at element offset `offset`
    from `region + signal_bytes() + cap_bytes`. The caller then runs the
    inter-node collective in place on exactly that range, on this same
    stream, and calls `allgather_finish` with `generation + 1` and scale 1.

    `scale` goes here rather than into `allgather_finish` so that an AVG never
    stores an unscaled sum in a narrow wire dtype (NCCL's PreMulSum); pass
    the communicator-wide 1/world, and the inter-node SUM of the node
    partials is the average.

    `rank` / `world` / `regions` are the node-local group. Preconditions are
    `allreduce`'s: 16-byte aligned `in_ptr`, `numel * size_of[dtype]() <=
    cap_bytes`, strictly increasing `generation`.
    """
    comptime W = 16 // size_of[dtype]()
    var per = _check_split[dtype, W](
        rank, world, in_ptr, numel, cap_bytes, generation, "reduce_scatter"
    )
    if numel == 0:
        return
    var rp = _region_ptrs(regions, rank, world)
    # world == 1 needs no special case: the push loop is empty, the sync is a
    # self-rendezvous and the reduce copies the input into stage_out.
    if world == 8:
        _launch_rs_stage[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    elif world == 4:
        _launch_rs_stage[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    elif world == 2:
        _launch_rs_stage[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    else:
        _launch_rs_stage[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )


def allgather_finish[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    out_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    """Second half: start barrier, then pull every rank's shard, on `stream`.

    The start barrier is also the wait for the peers' inter-node steps -- a
    peer publishes this generation's flags from inside this kernel, which its
    stream runs only after that step. Then rank p's shard is read from p's
    stage_out (mine included, since the inter-node step rewrote it) into
    `out_ptr` at the same element offset, times `scale` (ignored for integer
    dtypes). Same `numel`, `world` and preconditions as the matching
    `reduce_scatter_stage`; `generation` is that call's plus one.

    In place is safe: `out_ptr` may be the `in_ptr` the matching
    `reduce_scatter_stage` read, because the two are separate launches on one
    stream and no rank ever touches another rank's user memory.

    No exit protocol is needed: the next collective's start barrier already
    orders any arena reuse after every peer's pulls.
    """
    comptime W = 16 // size_of[dtype]()
    var per = _check_split[dtype, W](
        rank, world, out_ptr, numel, cap_bytes, generation, "allgather_finish"
    )
    if numel == 0:
        return
    var rp = _region_ptrs(regions, rank, world)
    if world == 8:
        _launch_ag_finish[dtype, W, 8](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    elif world == 4:
        _launch_ag_finish[dtype, W, 4](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    elif world == 2:
        _launch_ag_finish[dtype, W, 2](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    else:
        _launch_ag_finish[dtype, W, 0](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
