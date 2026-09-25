# Rewrite of: none (mojoccl-only: NCCL ships reduce_scatter_gin/all_gather_gin but no all-reduce one). Closest: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/reduce_scatter_gin.cuh
#
# The pipelined multi-node allreduce as ONE kernel launch.
#
# Same schedule as the split path agents_docs/distributed.md describes -- K chunks,
# chunk k's intra-node reduce-scatter, an RDMA exchange of its shard, and its
# all-gather `depth-1` chunks later so the proxy exchanges chunk k while the
# GPU reduce-scatters later chunks and all-gathers earlier ones -- but the
# loop runs inside a persistent kernel instead of on the host issuing five
# kernels per chunk. GPT-2 XL's backward issues 146 allreduces from the
# autograd thread, the thread that also issues its GEMMs; at ~24 us of driver
# time per launch the split schedule cost that thread ~32 ms per step, 17 ms
# of it exposed as idle gaps on the compute stream (job 250680). One launch
# per collective is NCCL's shape (ncclDevKernel_AllReduce per bucket).
#
# What the launch boundaries used to provide and this kernel says itself:
#   * "every block of this rank finished phase P": `grid_barrier` -- after
#     the reduce-scatter (before the shard is handed to the NIC), after the
#     exchange wait (before any block reads the inbox), after the inbox add,
#     and at the top of every chunk. The grid must therefore be co-resident:
#     the launcher never asks for more blocks than multiprocessors.
#   * the proxy request and the completion wait: block 0 / thread 0 stores
#     into and spins on the same pinned mailbox the two one-thread kernels
#     used, so `transport/net.mojo`'s progress thread is untouched.
#   * the inbox credit: `MB_CONSUMED` is stored after the add has RUN, which
#     is the fact the credit asserts (the split path could only say "the add
#     is enqueued" and rely on stream order).

from std.collections import Array
from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from std.utils import StaticTuple
from std.sys import size_of

from tmb.ccl.device.common import (
    _cached_occupancy,
    _enqueue_cached_dim,
    abort_raised,
    device_now_ns,
    latch_arena_error,
    publish_fault,
    status_page,
)
from tmb.ccl.device.symmetric.all_reduce import _ag_finish_body, _rs_stage_body
from tmb.ccl.device.symmetric.gin_scratch import _inbox_add_body
from tmb.ccl.device.symmetric.primitives import (
    _region_ptrs,
    _shard_cnt,
    _shard_off,
    _shard_per,
)
from tmb.ccl.include.device import (
    ERR_FUSED_GRID,
    ERR_PROXY_WAIT,
    FAULT_NO_PEER,
    MAX_BLOCKS,
    MAX_WORLD,
    PHASES_PER_GEN,
    _GFX942,
    _SIGNAL_BYTES,
    _align_up,
    poison_offset,
)
from tmb.ccl.include.nccl_device.lsa_barrier import grid_barrier
from tmb.ccl.transport.net import EMPTY_SHARD_BYTES


# Geometry of the fused kernel. Compile-time because the register budget
# follows from it; the block caps below select the measured hardware defaults.
#
# The whole design is per-SM throughput: a GEMM block of this backend needs
# the entire register file, so every SM holding a block of this kernel is
# lost to the compute stream for the collective's whole life -- measured on
# GPT-2 XL 2x8 H100 (job 250753, 256 threads, 32 B in flight per thread):
# 132 blocks 448k tok/s, 64 482k, 32 481k, 16 462k, 1 450k. Fewer SMs are
# free only if each SM moves more bytes, and NVLink wants bytes IN FLIGHT:
# the all-gather's pulls are remote loads at ~2 us of latency. NCCL's Simple
# protocol keeps 512 worker threads x 8 x 16 B = 64 KiB in flight per block
# (nccl:src/include/device.h ncclCollUnroll = 8 on sm_80+, NCCL_SIMPLE_MAX_NTHREADS
# 512); the split kernels got there with 216 blocks x 256 threads x 4 vectors.
comptime FUSED_THREADS = 512
"""Threads per block. Part of the wire layout: the barriers of
`_rs_stage_body`/`_ag_finish_body` are matched by block index and every
block's grid-stride slice follows from (blocks, threads), so every rank of a
node must be built with the same value -- one `.so` per build, so it is."""

comptime FUSED_UNROLL = 4
"""16-byte vectors in flight per thread in the all-gather's remote loads.

Fitted on GPT-2 XL 2x8 H100 (job 250904, mean tok/s of two passes, mojo+NCCL
495.5k): 512 threads x 4 vectors x 16 blocks 487.6k; x 8 vectors 484.3k;
1024 x 8 x 16 484.6k; 512 x 8 x 32 485.3k; 512 x 8 x 8 465.2k; 256 threads
(2 CTAs/SM) x 8 x 32 474.3k, x 2 x 32 476.9k; the split schedule 449.1k.
Doubling the bytes in flight past this point does not shorten the kernel's
life in the step, so what it costs the GEMMs is what decides."""

comptime FUSED_PUSH_UNROLL = 4
"""Same, for the reduce-scatter's remote stores."""

comptime FUSED_CTAS_PER_SM = 1
"""`nvvm.minctasm`: caps registers at 65536 / (FUSED_THREADS x this). 512 x 1
is 128 registers, enough for the 8-way reduce with 8 vectors in flight."""

comptime _GRID_GRACE_NS = UInt64(1_000_000_000)
"""Added to the grid barriers' deadline. Every spin in the kernel shares
`t0`, so when a phase barrier or the exchange wait gives up, the blocks
waiting in the next grid barrier would hit their own deadline at the same
instant and race the informative fault for the status page (seen: a rank
reporting ERR_FUSED_GRID where its exchange wait was the one that timed
out). With the grace they read the poison word instead."""

comptime _MB_ABORT_CHECK = 256
"""Mailbox polls between abort-word probes in `_await_exchange` -- both are
PCIe reads, so this is what bounds how long an abort holds the stream (a
few hundred microseconds), as `_proxy_wait_kernel` had it."""

# Fitted on 2x4 MI300A, GPT-2 XL, Adastra job 5417296 (2026-09-15):
# full-model ABBA 16/64 -> 8/16 raised 125104.5 -> 128363.6 tokens/s
# (+2.61%, -13.30 ms/step); 4/16 lost 1.16%. This is an architecture fit,
# not a CU-count rule: 228 CUs alone does not explain the nonmonotonic sweep.
# NVIDIA keeps the H100 fit, 16/64, byte-for-byte (jobs 250904/250995).
comptime DEFAULT_FUSED_BLOCKS = 8 if _GFX942 else 16
"""Block cap, checked equal on every rank at init because the barriers are
block-matched. See FUSED_THREADS for why the count is small."""

comptime DEFAULT_FUSED_BIG_BLOCKS = 16 if _GFX942 else 64
comptime DEFAULT_FUSED_BIG_MB = 128
"""Block cap for messages of at least `DEFAULT_FUSED_BIG_MB`. A message that large
is DDP's last bucket (GPT-2 XL: 313 MiB, the tied embedding), which nothing
overlaps: the SMs the small cap saves for the GEMMs are idle, and at 16
blocks it ran 5.9 ms against NCCL's 3.0 (job 250904 traces). Shape-based,
like the split path's `_AR_BIG_BYTES`."""


def fused_block_cap() -> Int:
    """Measured block cap for this accelerator."""
    return DEFAULT_FUSED_BLOCKS


def fused_big_block_cap() -> Int:
    """Measured block cap for large messages on this accelerator."""
    return DEFAULT_FUSED_BIG_BLOCKS


def fused_big_bytes() -> Int:
    """Bytes from which `fused_big_block_cap` applies."""
    return DEFAULT_FUSED_BIG_MB * (1024 * 1024)


def fused_blocks(cap: Int, resident: Int, per: Int, W: Int) -> Int:
    """Grid of the fused kernel: the cap, the co-resident bound (`resident`,
    the driver's occupancy for this kernel times the multiprocessor count --
    every block must be resident for `grid_barrier`), and what the shard can
    keep busy. Every rank of a node derives the same number from the same
    shape and the same GPU, which the block-matched barriers need."""
    var lim = min(cap, MAX_BLOCKS)
    if resident > 0:
        lim = min(lim, resident)
    return min(lim, max(1, (per // W + 1 + FUSED_THREADS - 1) // FUSED_THREADS))


def _fused_key[dtype: DType, NW: Int]() -> String:
    return String(t"fused_ar_{dtype}_{NW}")


def _resident_blocks[
    dtype: DType, NW: Int
](ctx: DeviceContext, sm_count: Int) raises -> Int:
    """How many blocks of this instantiation the device can hold at once."""
    comptime W = 16 // size_of[dtype]()
    return (
        _cached_occupancy[_fused_ar_kernel[dtype, W, NW]](
            ctx, _fused_key[dtype, NW](), FUSED_THREADS
        )
        * sm_count
    )


def _resident_all_dtypes[
    NW: Int
](ctx: DeviceContext, sm_count: Int) raises -> Int:
    var r = _resident_blocks[DType.float32, NW](ctx, sm_count)
    r = min(r, _resident_blocks[DType.float16, NW](ctx, sm_count))
    r = min(r, _resident_blocks[DType.bfloat16, NW](ctx, sm_count))
    r = min(r, _resident_blocks[DType.int32, NW](ctx, sm_count))
    return min(r, _resident_blocks[DType.int64, NW](ctx, sm_count))


def fused_resident_blocks(
    ctx: DeviceContext, world: Int, sm_count: Int
) raises -> Int:
    """Co-resident bound of the fused kernel for a node of `world` ranks: the
    minimum over every dtype it can be launched with.

    Every instantiation is compiled here, at init -- so no compilation, no
    occupancy query and no per-rank clamp is left for launch time, and the
    grid a rank launches is the node-agreed number (this bound is exchanged
    and checked across the node's ranks) rather than anything derived
    locally. Zero means some instantiation cannot run on this device, and the
    communicator takes the split schedule."""
    if world == 8:
        return _resident_all_dtypes[8](ctx, sm_count)
    if world == 4:
        return _resident_all_dtypes[4](ctx, sm_count)
    if world == 2:
        return _resident_all_dtypes[2](ctx, sm_count)
    return _resident_all_dtypes[0](ctx, sm_count)


@always_inline
def _give_up(poison: Pointer[UInt64, MutAnyOrigin]):
    """Every exit that is not the end of the collective goes through here:
    the blocks still in `grid_barrier` have no other way to learn that the
    block they wait for has left."""
    if thread_idx.x == 0:
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            poison, UInt64(1)
        )


@always_inline
def _await_exchange(
    mb_done: Pointer[UInt64, MutAnyOrigin],
    poison: Pointer[UInt64, MutAnyOrigin],
    region: Pointer[UInt8, MutAnyOrigin],
    page: Int,
    seq: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
):
    """`_proxy_wait_kernel`'s spin, inside the collective: one thread on
    pinned host memory, the abort word probed every `_MB_ABORT_CHECK` polls
    (both are PCIe reads). On failure it poisons the kernel rather than
    return alone: this thread still has to reach the next grid barrier, or
    every other block burns its own full deadline there."""
    var spins = 0
    while (
        Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](mb_done)
        < seq
    ):
        spins += 1
        if spins < _MB_ABORT_CHECK:
            continue
        spins = 0
        var gave_up = abort_raised(page)
        if gave_up:
            # An abort, or a fault raised elsewhere, already has the better
            # explanation in the arena word.
            _ = latch_arena_error(
                region.unsafe_bitcast[UInt64](), ERR_PROXY_WAIT, 0
            )
        elif device_now_ns() - t0 > timeout_ns:
            gave_up = True
            if latch_arena_error(
                region.unsafe_bitcast[UInt64](), ERR_PROXY_WAIT, 0
            ):
                # The peer of this wait is my own progress thread: `seen` is
                # how far the engine got, `target` the exchange asked for.
                publish_fault(
                    page,
                    ERR_PROXY_WAIT,
                    0,
                    Int(block_idx.x),
                    FAULT_NO_PEER,
                    Atomic[Scalar[DType.uint64]].load[
                        ordering=Ordering.ACQUIRE
                    ](mb_done),
                    seq,
                    Int(region),
                )
        if gave_up:
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                poison, UInt64(1)
            )
            return


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FUSED_THREADS)),
    `nvvm.minctasm`=SIMDLength(FUSED_CTAS_PER_SM),
)
@__name(t"ccl_internode_allreduce_pipelined_{dtype}_w{NW}")
def _fused_ar_kernel[
    dtype: DType, W: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    mb_done: Pointer[UInt64, MutAnyOrigin],
    mb_consumed: Pointer[UInt64, MutAnyOrigin],
    count: Int64,
    chunk_elems: Int64,
    seq0: Int64,
    arena_stride: Int64,
    arena_cap: Int64,
    inbox_origin: Int64,
    inbox_stride: Int64,
    shape: StaticTuple[Int32, 5],
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    """One multi-node allreduce, chunks and all. `shape` is `(nchunks, depth,
    narenas, nslots, npeers)`."""
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var nblocks = Int(grid_dim.x)
    var stride = nblocks * FUSED_THREADS
    var total = Int(count)
    var ce = Int(chunk_elems)
    var nchunks = Int(shape[0])
    var depth = Int(shape[1])
    var narenas = Int(shape[2])
    var nslots = Int(shape[3])
    var npeers = Int(shape[4])
    var astride = Int(arena_stride)
    var me = regions[rank]
    var page = status_page(me)
    var poison = me.unsafe_offset(poison_offset()).unsafe_bitcast[UInt64]()
    comptime esize = size_of[dtype]()
    var push_off = _SIGNAL_BYTES
    var out_off = _SIGNAL_BYTES + Int(arena_cap)

    # Cleared before the first barrier, so every block sees the zero.
    if block_idx.x == 0 and thread_idx.x == 0:
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELAXED](
            poison, UInt64(0)
        )

    for k in range(nchunks + depth - 1):
        # The deadline is per phase, as it was per kernel on the split
        # schedule: `t0` restarts here, before the exchange wait and before
        # the all-gather, so MOJOCCL_IB_TIMEOUT_S bounds each wait rather
        # than the whole collective.
        var t0 = device_now_ns()
        # The launch boundary, restored: nothing of chunk k starts until every
        # block of this rank is done with chunk k-1; peers inherit it because
        # chunk k's flags are published after it. A False here is a deadline
        # or another block's poison, either way for the whole block.
        if not grid_barrier(
            me, poison, nblocks, ERR_FUSED_GRID, t0, timeout_ns + _GRID_GRACE_NS
        ):
            _give_up(poison)
            return

        if k < nchunks:
            var off = k * ce
            var cnt = min(ce, total - off)
            var per = _shard_per(cnt, world, W)
            if not _rs_stage_body[dtype, W, FUSED_PUSH_UNROLL, NW](
                regions,
                (k % narenas) * astride,
                in_ptr.unsafe_offset(off),
                cnt,
                per,
                per * esize,
                push_off,
                out_off,
                world,
                rank,
                tid,
                stride,
                flag_base + UInt64(2 * k * PHASES_PER_GEN),
                scale,
                t0,
                timeout_ns,
            ):
                _give_up(poison)
                return
            # The shard is whole only after this barrier, whose release chain
            # also makes every block's stores visible to the NIC before the
            # request naming them goes out.
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return
            if block_idx.x == 0 and thread_idx.x == 0:
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                    mb_req, UInt64(Int(seq0) + k)
                )

        var j = k - (depth - 1)
        if j >= 0:
            var off = j * ce
            var cnt = min(ce, total - off)
            var arena_off = (j % narenas) * astride
            var per = _shard_per(cnt, world, W)
            var my_off = _shard_off(cnt, per, rank)
            var my_cnt = _shard_cnt(cnt, per, rank)
            var seq = Int(seq0) + j

            t0 = device_now_ns()
            if block_idx.x == 0 and thread_idx.x == 0:
                _await_exchange(
                    mb_done, poison, me, page, UInt64(seq), t0, timeout_ns
                )
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return

            if my_cnt > 0:
                _inbox_add_body[dtype, W](
                    me.unsafe_offset(
                        arena_off + out_off + my_off * esize
                    ).unsafe_bitcast[Scalar[dtype]](),
                    me.unsafe_offset(
                        Int(inbox_origin) + (seq % nslots) * Int(inbox_stride)
                    ),
                    my_cnt,
                    _align_up(max(my_cnt * esize, EMPTY_SHARD_BYTES), 16),
                    npeers,
                    tid,
                    stride,
                )
            # The add has run on every block: the inbox slot group is free,
            # which is what the credit asserts. The engine reads this word.
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return
            if block_idx.x == 0 and thread_idx.x == 0:
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                    mb_consumed, UInt64(seq)
                )

            t0 = device_now_ns()
            if not _ag_finish_body[dtype, W, FUSED_UNROLL](
                regions,
                arena_off,
                out_ptr.unsafe_offset(off),
                cnt,
                per,
                out_off,
                world,
                rank,
                tid,
                stride,
                flag_base + UInt64((2 * j + 1) * PHASES_PER_GEN),
                Float32(1.0),
                t0,
                timeout_ns,
            ):
                _give_up(poison)
                return


@always_inline
def _launch_fused[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    blocks: Int,
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    mailbox: StaticTuple[Int, 3],
    count: Int,
    chunk_elems: Int,
    seq0: Int,
    arena_stride: Int,
    arena_cap: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    shape: StaticTuple[Int32, 5],
    world: Int,
    rank: Int,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
) raises:
    # `blocks` is the node-agreed grid (`fused_blocks` over the bound every
    # rank checked at init); nothing here re-derives it, or the block-matched
    # barriers would disagree. The kernel was compiled at init; the launch is
    # cooperative so a grid the device cannot hold is refused, not hung.
    _enqueue_cached_dim[_fused_ar_kernel[dtype, W, NW]](
        ctx,
        stream,
        _fused_key[dtype, NW](),
        blocks,
        FUSED_THREADS,
        True,
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[0]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[1]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[2]),
        Int64(count),
        Int64(chunk_elems),
        Int64(seq0),
        Int64(arena_stride),
        Int64(arena_cap),
        Int64(inbox_origin),
        Int64(inbox_stride),
        shape,
        Int32(world),
        Int32(rank),
        flag_base,
        scale,
        timeout_ns,
    )


def check_fused_call[
    dtype: DType
](
    in_ptr: Int,
    out_ptr: Int,
    chunk_elems: Int,
    arena_cap: Int,
    nchunks: Int,
) raises:
    """Everything `internode_allreduce_fused` would refuse, checked BEFORE
    the caller reserves exchange counters and fills work items: a call
    rejected after that would strand ring slots the engine never sees."""
    if nchunks <= 0 or chunk_elems <= 0:
        raise Error("mojoccl: fused allreduce with no chunks")
    if chunk_elems * size_of[dtype]() > arena_cap:
        raise Error("mojoccl: fused allreduce chunk exceeds an arena")
    if in_ptr % 16 != 0 or out_ptr % 16 != 0:
        raise Error("mojoccl: fused allreduce needs 16-byte aligned buffers")


def internode_allreduce_fused[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    mailbox: StaticTuple[Int, 3],
    count: Int,
    chunk_elems: Int,
    nchunks: Int,
    depth: Int,
    narenas: Int,
    arena_stride: Int,
    arena_cap: Int,
    seq0: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    nslots: Int,
    npeers: Int,
    generation: Int,
    scale: Float32,
    blocks: Int,
    timeout_ns: UInt64,
) raises:
    """Enqueue the whole pipelined multi-node allreduce as one kernel.

    `generation` is the first of the `2 * nchunks` this call consumes: chunk k
    reduce-scatters at `generation + 2k` and all-gathers at `generation +
    2k + 1`. `seq0` is the first of `nchunks` consecutive exchange counters
    the caller reserved and filled work items for; the kernel only publishes
    them.
    """
    comptime W = 16 // size_of[dtype]()
    if nchunks <= 0 or chunk_elems <= 0 or count <= 0:
        return
    if chunk_elems * size_of[dtype]() > arena_cap:
        raise Error("mojoccl: fused allreduce chunk exceeds an arena")
    if in_ptr % 16 != 0 or out_ptr % 16 != 0:
        raise Error("mojoccl: fused allreduce needs 16-byte aligned buffers")
    if blocks < 1 or blocks > MAX_BLOCKS:
        raise Error("mojoccl: fused allreduce grid out of range")
    var rp = _region_ptrs(regions, rank, world)
    var shape = StaticTuple[Int32, 5](
        Int32(nchunks),
        Int32(depth),
        Int32(narenas),
        Int32(nslots),
        Int32(npeers),
    )
    var flag_base = UInt64(generation) * UInt64(PHASES_PER_GEN)
    if world == 8:
        _launch_fused[dtype, W, 8](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            arena_cap,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            flag_base,
            scale,
            timeout_ns,
        )
    elif world == 4:
        _launch_fused[dtype, W, 4](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            arena_cap,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            flag_base,
            scale,
            timeout_ns,
        )
    elif world == 2:
        _launch_fused[dtype, W, 2](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            arena_cap,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            flag_base,
            scale,
            timeout_ns,
        )
    else:
        _launch_fused[dtype, W, 0](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            arena_cap,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            flag_base,
            scale,
            timeout_ns,
        )
