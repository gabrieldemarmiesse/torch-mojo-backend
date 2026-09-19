# Streaming hierarchical fp32 reduce-scatter: reduce as the pushes land.
#
# `rs_fused.mojo` runs a chunk as push -> 8-way barrier -> reduce: no rank
# starts reducing before every rank finished pushing the whole chunk, so the
# NVLink push (fabric bound) and the 8-way reduce (HBM bound) never overlap,
# the barrier exposes the slowest rank's whole push, and the chunk's RDMA
# exchange only starts when all of it is done. NCCL's ring has none of that:
# its unit of "has data arrived" is a 1 MiB slice, checked by four threads of
# the block against the neighbour's step counter, with an eight-deep credit
# pipeline and no grid barrier anywhere (nccl:src/device/prims_simple.h
# waitPeer/postPeer, src/device/reduce_scatter.h).
#
# This kernel keeps the hierarchical schedule -- the bytes are the same as
# NCCL's ring here: 7*nnodes shard pushes on NVLink and one shard on the wire,
# against a ring's 14/15 NVLink and 1/15 network hops -- and borrows the
# handoff. Each block owns a contiguous slice of the chunk and walks it in
# `slice_vecs` pieces:
#
#   push piece j to all (node, peer) staging slots
#   publish DATA[peer][block][me] = ordinal(chunk, j)          (release)
#   wait   DATA[me][block][peer] >= ordinal(chunk, j-DEPTH)    for every peer
#   reduce piece j-DEPTH out of the slots into the staging output
#
# so the reduce of piece j runs under the push of piece j+DEPTH, a rank waits
# for one piece of a slow peer instead of its whole chunk, and nothing in the
# kernel is a grid-wide rendezvous. The two counters are generation-tagged
# monotone words compared with `>=`, never reset, exactly like the block
# barrier's flags.
#
# Two more one-directional handoffs replace the fused kernel's grid barriers:
#
#   * FREE[peer][block][me], published after this block reduced chunk k, is
#     the credit that lets peers overwrite my arena at chunk k+narenas. In
#     steady state it is already there: it is narenas chunks of slack.
#   * a per-chunk arrival counter (rank-local, one word per chunk, reset by
#     the last arriver) releases the chunk's RDMA exchange and, on the way
#     back, publishes the inbox credit. Blocks arrive and keep going; only
#     the last one stores into the mailbox.
#
# What is left of the fused kernel's stop-the-world points is the wait for
# the exchange, which every block does for itself on the pinned mailbox.

from std.atomic import Atomic, Ordering, fence
from std.collections import InlineArray
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.memory import AddressSpace, stack_allocation
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.sync import barrier

from netutil import MAX_NODES
from internode import WORK_SLOTS
from internode_fused import FUSED_THREADS, _MB_ABORT_CHECK
from collectives_kernels import (
    ERR_PROXY_WAIT,
    ERR_REDUCE_SCATTER_SYNC,
    FAULT_NO_PEER,
    MAX_BLOCKS,
    MAX_WORLD,
    PHASES_PER_GEN,
    _AMD,
    _FLAG_BYTE_OFFSET,
    _SIGNAL_BYTES,
    _SPIN_CHECK,
    _abort_raised,
    _align_up,
    _cached_occupancy,
    _copy_span_flex,
    _enqueue_cached_dim,
    _peer_step,
    _region_ptrs,
    _rs_slot,
    _share,
    abort_raised,
    device_now_ns,
    latch_arena_error,
    publish_fault,
    status_page,
)

# ===-------------------------------------------------------------------=== #
# Measured geometry
# ===-------------------------------------------------------------------=== #

comptime RS_STREAM_ENABLED = True
"""Whether the multi-node fp32 reduce-scatter streams. `False` sends it back
to `rs_fused.mojo`'s push-barrier-reduce kernel, which stays the fallback for
every geometry this one declines; it is also how the two were measured against
each other, one build each."""

comptime RS_STREAM_THREADS = FUSED_THREADS
"""Threads per block, the fused kernel's. Part of the wire layout: a piece is
`slice_vecs` 16-byte vectors of one block, so every rank of a node must be
built with the same value -- one `.so` per build, so it is."""

comptime RS_STREAM_UNROLL = 8
"""16-byte vectors in flight per thread, per destination, in the push. NCCL's
Ring/Simple worker keeps 8 (`ncclCollUnroll` on sm_90) but against one
neighbour; here 14 destinations are walked back to back and their stores do
not depend on one another, so the in-flight count is this times the
destinations the scheduler has already reached."""

comptime RS_STREAM_SLICE_UNROLLS = 1
"""Pieces are `RS_STREAM_UNROLL * RS_STREAM_SLICE_UNROLLS` vectors per thread,
i.e. this many full unrolled passes per thread per destination per piece. 1 is
16 KiB per block per destination (512 threads x 4 x 16 B) and, at 32 blocks,
a 2 MiB piece of the chunk."""

comptime RS_STREAM_DEPTH = 1
"""Pieces of lookahead between the push and the reduce: the reduce of piece j
runs while piece j+DEPTH is being pushed."""

comptime RS_STREAM_TARGET_CHUNKS = 4
"""Chunks a reduce-scatter of at least `PIPE_SPLIT_UNIT` bytes per rank is cut
into (geometry may force more). Only the last chunk's RDMA exchange is
exposed, and streaming made a chunk boundary cheap -- one credit and one
arrival, no barrier -- which is why this is twice `rs_fused.mojo`'s."""

comptime RS_STREAM_BIG_BLOCKS = 32
"""Grid cap at `PIPE_SPLIT_UNIT` bytes per rank and above. Every block holds
an SM's whole register file for the call, so this is also how many SMs the
backward's GEMMs lose while a reduce-scatter runs; `rs_fused.mojo`'s
RS_FUSED_BIG_BLOCKS records the end-to-end sweep that chose 32.

Unlike that kernel, this one wants the same number in isolation: the handoff
is per block, so a larger grid multiplies the 7 remote flag stores and the
8-way rendezvous it costs per piece while giving each block less to push
between them. Isolated block / root fp32 on 2x8 H100, 16 ranks, us:
32 CTAs 515/1277, 128 CTAs 602-628/1370-1383 (NCCL 465-479/998-1000)."""

comptime RS_STREAM_ROWS = 128
"""Flag rows (blocks) the streaming tables hold. The grids here are capped at
`RS_STREAM_BIG_BLOCKS`; `reduce_scatter_stream_blocks` refuses more."""

# ===-------------------------------------------------------------------=== #
# Streaming sub-area of arena 0's signal page
# ===-------------------------------------------------------------------=== #
#
# Everything here lives in arena 0 only, so a counter is one monotone sequence
# for the whole call however the chunks rotate over the arenas.

comptime _RS_STREAM_DATA = _FLAG_BYTE_OFFSET + MAX_BLOCKS * MAX_WORLD * 8
"""DATA[block][writer]: pieces writer has delivered into this region."""

comptime _RS_STREAM_FREE = _RS_STREAM_DATA + RS_STREAM_ROWS * MAX_WORLD * 8
"""FREE[block][reader]: chunks reader has consumed of what I pushed to it."""

comptime _RS_STREAM_REQ = _RS_STREAM_FREE + RS_STREAM_ROWS * MAX_WORLD * 8
"""One arrival counter per chunk for the exchange release; rank-local."""

comptime _RS_STREAM_CONS = _RS_STREAM_REQ + WORK_SLOTS * 8
"""One arrival counter per chunk for the inbox credit; rank-local."""

comptime _RS_STREAM_POISON = _RS_STREAM_CONS + WORK_SLOTS * 8
"""Give-up word, rank-local. Generation-tagged rather than 0/1 so a call never
has to clear it: a block compares it against its own `flag_base` and reads a
previous call's value as no poison at all."""

comptime _RS_STREAM_DONE = _RS_STREAM_POISON + 8
"""Highest exchange block 0 saw retire, in device memory.

Dropping the grid barrier left every block polling the pinned mailbox for
itself, and 32 threads reading host memory over the link the NIC is moving the
shard on cost far more than the barrier did: see the measured A/B in
docs/distributed.md's "Streaming reduce-scatter". One PCIe reader, and the
rest spin on an L2 line."""

comptime _RS_STREAM_END = _RS_STREAM_DONE + 8


def stream_area_bytes() -> Int:
    """End of the streaming sub-area; the check is the point."""
    comptime assert (
        _RS_STREAM_END <= _SIGNAL_BYTES
    ), "the streaming tables must fit the signal area"
    comptime assert (
        RS_STREAM_THREADS >= MAX_WORLD
    ), "the handoff needs one thread per peer"
    return _RS_STREAM_END


@always_inline
def _row(
    region: Pointer[UInt8, MutAnyOrigin], table: Int, block: Int
) -> Pointer[UInt64, MutAnyOrigin]:
    return region.unsafe_offset(table + block * MAX_WORLD * 8).unsafe_bitcast[
        UInt64
    ]()


@always_inline
def _poison_word(
    region: Pointer[UInt8, MutAnyOrigin],
) -> Pointer[UInt64, MutAnyOrigin]:
    return region.unsafe_offset(_RS_STREAM_POISON).unsafe_bitcast[UInt64]()


@always_inline
def _poisoned(region: Pointer[UInt8, MutAnyOrigin], tag: UInt64) -> Bool:
    return (
        Atomic[DType.uint64].load[ordering=Ordering.RELAXED](
            _poison_word(region)
        )
        >= tag
    )


@always_inline
def _poison(region: Pointer[UInt8, MutAnyOrigin], tag: UInt64):
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
        _poison_word(region), tag
    )


@always_inline
def _publish(
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    table: Int,
    world: Int,
    rank: Int,
    block: Int,
    target: UInt64,
):
    """Tell every peer this block reached `target`, after its stores landed.

    Same ordering argument as `_sync`: on NVIDIA `bar.sync` is a CTA-scope
    fence and the release store below is cumulative over it; AMD needs the
    writeback in every thread (see `_sync`'s note)."""
    comptime if _AMD:
        fence[ordering=Ordering.RELEASE]()
    barrier()
    var t = Int(thread_idx.x)
    if t < world and t != rank:
        Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
            _row(regions[t], table, block).unsafe_offset(rank), target
        )


@always_inline
def _await(
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    table: Int,
    world: Int,
    rank: Int,
    block: Int,
    target: UInt64,
    tag: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """Wait for every peer's `table` counter for this block to reach `target`.

    One thread per peer, as `_sync` does, and the whole block learns the
    outcome through shared memory so nobody is left inside a `barrier()`.
    A block that gives up poisons the region: the peers of a rank that
    stopped answering are not the only blocks stuck, and every other block
    of this grid would otherwise burn its own full deadline."""
    var me = regions[rank]
    var failed = stack_allocation[
        1, DType.uint32, address_space = AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    barrier()
    var t = Int(thread_idx.x)
    if t < world and t != rank:
        var mine = _row(me, table, block).unsafe_offset(t)
        var spins = 0
        while (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](mine) < target
        ):
            spins += 1
            if spins < _SPIN_CHECK:
                continue
            spins = 0
            if _poisoned(me, tag):
                failed[unsafe_offset=0] = 1
                break
            if _abort_raised(me):
                _ = latch_arena_error(
                    me.unsafe_bitcast[UInt64](), ERR_REDUCE_SCATTER_SYNC, 0
                )
                failed[unsafe_offset=0] = 1
                break
            # Same-GPU timer difference only; never compared across GPUs.
            if device_now_ns() - t0 > timeout_ns:
                if latch_arena_error(
                    me.unsafe_bitcast[UInt64](), ERR_REDUCE_SCATTER_SYNC, 0
                ):
                    publish_fault(
                        status_page(me),
                        ERR_REDUCE_SCATTER_SYNC,
                        0,
                        block,
                        UInt64(t),
                        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                            mine
                        ),
                        target,
                        Int(me),
                    )
                failed[unsafe_offset=0] = 1
                break
    barrier()
    if failed[unsafe_offset=0] != 0:
        if thread_idx.x == 0:
            _poison(me, tag)
        return False
    return True


@always_inline
def _await_exchange(
    me: Pointer[UInt8, MutAnyOrigin],
    mb_done: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
    tag: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """`internode_fused.mojo`'s mailbox spin without its grid barrier.

    Block 0 is the only PCIe reader: it polls the pinned mailbox, probing the
    abort word every `_MB_ABORT_CHECK` polls (both are PCIe reads), and
    republishes what it saw into `_RS_STREAM_DONE`, a device word the other
    blocks spin on out of L2. A block that reads that word has, transitively,
    acquired what the NIC delivered -- which is what the grid barrier used to
    give it -- and `seq` only grows, so a block 0 already further down the
    pipeline releases the blocks behind it."""
    var failed = stack_allocation[
        1, DType.uint32, address_space = AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    barrier()
    if thread_idx.x == 0:
        var lead = block_idx.x == 0
        var word = (
            mb_done if lead else me.unsafe_offset(
                _RS_STREAM_DONE
            ).unsafe_bitcast[UInt64]()
        )
        var page = status_page(me)
        var spins = 0
        while (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](word) < seq
        ):
            spins += 1
            if spins < _MB_ABORT_CHECK:
                continue
            spins = 0
            if _poisoned(me, tag):
                failed[unsafe_offset=0] = 1
                break
            if abort_raised(page):
                _ = latch_arena_error(
                    me.unsafe_bitcast[UInt64](), ERR_PROXY_WAIT, 0
                )
                failed[unsafe_offset=0] = 1
                break
            if device_now_ns() - t0 > timeout_ns:
                if latch_arena_error(
                    me.unsafe_bitcast[UInt64](), ERR_PROXY_WAIT, 0
                ):
                    # The peer of this wait is my own progress thread.
                    publish_fault(
                        page,
                        ERR_PROXY_WAIT,
                        0,
                        Int(block_idx.x),
                        FAULT_NO_PEER,
                        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                            word
                        ),
                        seq,
                        Int(me),
                    )
                failed[unsafe_offset=0] = 1
                break
        if failed[unsafe_offset=0] != 0:
            _poison(me, tag)
        elif lead:
            Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                me.unsafe_offset(_RS_STREAM_DONE).unsafe_bitcast[UInt64](), seq
            )
    barrier()
    return failed[unsafe_offset=0] == 0


@always_inline
def _arrive(
    me: Pointer[UInt8, MutAnyOrigin], table: Int, chunk: Int, nblocks: Int
) -> Bool:
    """Count this block into `chunk`'s arrival; True in the last block.

    One counter per chunk, so a block that has already run ahead to a later
    chunk cannot be mistaken for a straggler arriving at this one, and the
    last arriver clears it -- nothing carries across launches."""
    var arrive = me.unsafe_offset(table + chunk * 8).unsafe_bitcast[UInt64]()
    var was = Atomic[DType.uint64].fetch_add[
        ordering = Ordering.ACQUIRE_RELEASE
    ](arrive, UInt64(1))
    if Int(was) != nblocks - 1:
        return False
    Atomic[DType.uint64].store[ordering=Ordering.RELAXED](arrive, UInt64(0))
    return True


# ===-------------------------------------------------------------------=== #
# Per-piece payload
# ===-------------------------------------------------------------------=== #


@always_inline
def _push_piece[
    NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Float32, MutAnyOrigin],
    e0: Int,
    n: Int,
    world: Int,
    rank: Int,
    nnodes: Int,
    push_off: Int,
    slot_stride: Int,
    in_stride: Int,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    vec: Bool,
):
    """Elements `[e0, e0+n)` of my contribution to every destination shard."""
    var t = Int(thread_idx.x)
    for d in range(nnodes):
        for i in range(1, world):
            var s = rank + _peer_step(i, world)
            if s >= world:
                s -= world
            var dst = (
                regions[s]
                .unsafe_offset(
                    push_off
                    + slot_stride
                    * (d * (world - 1) + (rank if rank < s else rank - 1))
                )
                .unsafe_bitcast[Float32]()
            )
            _copy_span_flex[DType.float32, 4, RS_STREAM_UNROLL](
                dst.unsafe_offset(e0),
                in_ptr.unsafe_offset(
                    Int(rank_ids[d * world + s]) * in_stride + e0
                ),
                n,
                t,
                RS_STREAM_THREADS,
                vec,
            )


@always_inline
def _one(
    uin: Pointer[Float32, MutAnyOrigin],
    slots: Pointer[UInt8, MutAnyOrigin],
    slot_stride: Int,
    world: Int,
    rank: Int,
    k: Int,
    scale: Float32,
) -> Float32:
    var a = _share[DType.float32, 1](uin[unsafe_offset=k], scale)
    for j in range(1, world):
        var p = rank + j
        if p >= world:
            p -= world
        a += _share[DType.float32, 1](
            _rs_slot[DType.float32](slots, slot_stride, p, rank)[
                unsafe_offset=k
            ],
            scale,
        )
    return a


@always_inline
def _reduce_piece[
    NW: Int
](
    me: Pointer[UInt8, MutAnyOrigin],
    in_ptr: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
    e0: Int,
    n: Int,
    world: Int,
    rank: Int,
    nnodes: Int,
    push_off: Int,
    slot_stride: Int,
    out_stride_e: Int,
    in_stride: Int,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    scale: Float32,
    vec: Bool,
):
    """Sum the `world` contributions to elements `[e0, e0+n)` of every
    destination shard this rank reduces. Each contribution is scaled as it
    enters the accumulator (`_share`), not the finished sum."""
    var t = Int(thread_idx.x)
    for d in range(nnodes):
        var dst = out_ptr.unsafe_offset(d * out_stride_e + e0)
        var uin = in_ptr.unsafe_offset(
            Int(rank_ids[d * world + rank]) * in_stride + e0
        )
        var slots = me.unsafe_offset(
            push_off + d * (world - 1) * slot_stride + e0 * 4
        )
        var vc = n // 4
        if vec:
            for v in range(t, vc, RS_STREAM_THREADS):
                var acc = _share[DType.float32, 4](
                    uin.unsafe_load[width=4, alignment=16](v * 4), scale
                )
                # Slot pointers are formed by arithmetic, never held in a stack
                # array: such an array is demoted to local memory (MOCO-1431).
                comptime if NW > 0:
                    comptime for j in range(1, NW):
                        var p = rank + j
                        if p >= NW:
                            p -= NW
                        acc += _share[DType.float32, 4](
                            _rs_slot[DType.float32](
                                slots, slot_stride, p, rank
                            ).unsafe_load[width=4, alignment=16](v * 4),
                            scale,
                        )
                else:
                    for j in range(1, world):
                        var p = rank + j
                        if p >= world:
                            p -= world
                        acc += _share[DType.float32, 4](
                            _rs_slot[DType.float32](
                                slots, slot_stride, p, rank
                            ).unsafe_load[width=4, alignment=16](v * 4),
                            scale,
                        )
                dst.unsafe_store[width=4, alignment=16](v * 4, acc)
        else:
            for v in range(t, vc, RS_STREAM_THREADS):
                comptime for e in range(4):
                    dst[unsafe_offset = v * 4 + e] = _one(
                        uin, slots, slot_stride, world, rank, v * 4 + e, scale
                    )
        for i in range(t, n - vc * 4, RS_STREAM_THREADS):
            dst[unsafe_offset = vc * 4 + i] = _one(
                uin, slots, slot_stride, world, rank, vc * 4 + i, scale
            )


@always_inline
def _sum_out(
    output: Pointer[Float32, MutAnyOrigin],
    partial: Pointer[Float32, MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    n: Int,
    slot_bytes: Int,
    npeers: Int,
):
    """This block's range of `out = my node's partial + every peer node's`.
    Every pointer is already offset to the range's first element."""
    var t = Int(thread_idx.x)
    var vc = n // 4
    var v = t
    while v < vc:
        # Four rows per iteration so 4 * (1 + npeers) loads are in flight.
        var acc = InlineArray[SIMD[DType.float32, 4], 4](uninitialized=True)
        comptime for u in range(4):
            if v + u * RS_STREAM_THREADS < vc:
                acc[u] = partial.unsafe_load[width=4, alignment=16](
                    (v + u * RS_STREAM_THREADS) * 4
                )
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * slot_bytes).unsafe_bitcast[
                Float32
            ]()
            comptime for u in range(4):
                if v + u * RS_STREAM_THREADS < vc:
                    acc[u] += src.unsafe_load[width=4, alignment=16](
                        (v + u * RS_STREAM_THREADS) * 4
                    )
        comptime for u in range(4):
            if v + u * RS_STREAM_THREADS < vc:
                output.unsafe_store[width=4](
                    (v + u * RS_STREAM_THREADS) * 4, acc[u]
                )
        v += 4 * RS_STREAM_THREADS
    for i in range(vc * 4 + t, n, RS_STREAM_THREADS):
        var acc = partial[unsafe_offset=i]
        for j in range(npeers):
            acc += inbox.unsafe_offset(j * slot_bytes).unsafe_bitcast[Float32]()[
                unsafe_offset=i
            ]
        output[unsafe_offset=i] = acc


# ===-------------------------------------------------------------------=== #
# The kernel
# ===-------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(RS_STREAM_THREADS)
    ),
    `nvvm.minctasm`=SIMDLength(1),
)
@__name(t"ccl_reduce_scatter_nodes_streamed_float32_w{NW}")
def _stream_rs_kernel[
    NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    mb_done: Pointer[UInt64, MutAnyOrigin],
    mb_consumed: Pointer[UInt64, MutAnyOrigin],
    count: Int64,
    chunk_elems: Int64,
    seq0: Int64,
    arena_stride: Int64,
    inbox_origin: Int64,
    inbox_stride: Int64,
    piece_vecs: Int64,
    pieces_per_chunk: Int64,
    shape: StaticTuple[Int32, 7],
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    vector_ok: Int32,
):
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var b = Int(block_idx.x)
    var nblocks = Int(grid_dim.x)
    var total = Int(count)
    var ce = Int(chunk_elems)
    var nchunks = Int(shape[0])
    var depth = Int(shape[1])
    var narenas = Int(shape[2])
    var nslots = Int(shape[3])
    var npeers = Int(shape[4])
    var nnodes = Int(shape[5])
    var my_node = Int(shape[6])
    var me = regions[rank]
    var pv = Int(piece_vecs)
    var ppc = Int(pieces_per_chunk)
    var vec = vector_ok != 0
    var t0 = device_now_ns()
    # Every spin of this call compares the poison word against `flag_base`,
    # which strictly increases per call, so a previous call's value is not one.
    var tag = flag_base
    # THE ARENA LAYOUT IS THE SAME FOR EVERY CHUNK, AND THAT IS LOAD-BEARING.
    # The FREE credit is per block index: before writing the arena at chunk
    # k a block waits only for the SAME block index on every peer to have
    # released chunk k-narenas. That is sound exactly while block b owns the
    # same bytes in both, so the slot stride and the partition come from the
    # full `chunk_elems`, never from a short last chunk's `cnt`, and a short
    # chunk simply leaves the tail of the layout unwritten.
    var slot = _align_up(ce * 4, 16)
    var vc_full = ce // 4
    var vpb = max(1, (vc_full + nblocks - 1) // nblocks)

    for k in range(nchunks + depth - 1):
        if k < nchunks:
            var off = k * ce
            var cnt = min(ce, total - off)
            var arena_off = (k % narenas) * Int(arena_stride)
            var push_off = arena_off + _SIGNAL_BYTES
            var out_off = push_off + (world - 1) * nnodes * slot
            # The peers may not overwrite the arena I read `narenas` chunks ago.
            if k >= narenas:
                if not _await(
                    regions,
                    _RS_STREAM_FREE,
                    world,
                    rank,
                    b,
                    flag_base + UInt64(1 + k - narenas),
                    tag,
                    t0,
                    timeout_ns,
                ):
                    return
            # This block's slice of the fixed partition, clipped to what this
            # chunk actually carries; the `cnt % 4` tail rides on whichever
            # block owns the vector it starts at.
            var vc = cnt // 4
            var tailn = cnt - vc * 4
            var vlo = min(vc, b * vpb)
            var vhi = min(vc, (b + 1) * vpb)
            # Only a `count` that is not a multiple of four has a tail, so
            # the division stays off the hot path.
            var tail_owner = min(nblocks - 1, vc // vpb) if tailn > 0 else 0
            var mine_tail = tailn if b == tail_owner else 0
            var npieces = (vhi - vlo + pv - 1) // pv
            if mine_tail > 0 and npieces == 0:
                npieces = 1
            var ord = flag_base + UInt64(1 + k * ppc)
            for j in range(npieces + RS_STREAM_DEPTH):
                if j < npieces:
                    var v0 = vlo + j * pv
                    var n = min(vhi - v0, pv) * 4
                    if j == npieces - 1:
                        n += mine_tail
                    _push_piece[NW](
                        regions,
                        in_ptr.unsafe_offset(off),
                        v0 * 4,
                        n,
                        world,
                        rank,
                        nnodes,
                        push_off,
                        slot,
                        Int(count),
                        rank_ids,
                        vec,
                    )
                    _publish(
                        regions,
                        _RS_STREAM_DATA,
                        world,
                        rank,
                        b,
                        ord + UInt64(j),
                    )
                var r = j - RS_STREAM_DEPTH
                if r >= 0:
                    if not _await(
                        regions,
                        _RS_STREAM_DATA,
                        world,
                        rank,
                        b,
                        ord + UInt64(r),
                        tag,
                        t0,
                        timeout_ns,
                    ):
                        return
                    var v0 = vlo + r * pv
                    var n = min(vhi - v0, pv) * 4
                    if r == npieces - 1:
                        n += mine_tail
                    _reduce_piece[NW](
                        me,
                        in_ptr.unsafe_offset(off),
                        me.unsafe_offset(out_off).unsafe_bitcast[Float32](),
                        v0 * 4,
                        n,
                        world,
                        rank,
                        nnodes,
                        push_off,
                        slot,
                        slot // 4,
                        Int(count),
                        rank_ids,
                        scale,
                        vec,
                    )
            # The arena I just read is free for the peers again, and once every
            # block's reduce is in, the chunk's shard can go on the wire.
            _publish(
                regions,
                _RS_STREAM_FREE,
                world,
                rank,
                b,
                flag_base + UInt64(1 + k),
            )
            if nnodes > 1 and thread_idx.x == 0:
                if _arrive(me, _RS_STREAM_REQ, k, nblocks):
                    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                        mb_req, UInt64(Int(seq0) + k)
                    )
        var j = k - (depth - 1)
        if j >= 0 and nnodes > 1:
            var off = j * ce
            var cnt = min(ce, total - off)
            var arena_off = (j % narenas) * Int(arena_stride)
            var out_off = (
                arena_off + _SIGNAL_BYTES + (world - 1) * nnodes * slot
            )
            var seq = Int(seq0) + j
            if not _await_exchange(
                me, mb_done, UInt64(seq), tag, t0, timeout_ns
            ):
                return
            var vc = cnt // 4
            var vlo = min(vc, b * vpb)
            var vhi = min(vc, (b + 1) * vpb)
            var tailn = cnt - vc * 4
            var tail_owner = min(nblocks - 1, vc // vpb) if tailn > 0 else 0
            var n = (vhi - vlo) * 4 + (tailn if b == tail_owner else 0)
            _sum_out(
                out_ptr.unsafe_offset(off + vlo * 4),
                me.unsafe_offset(
                    out_off + my_node * slot + vlo * 16
                ).unsafe_bitcast[Float32](),
                me.unsafe_offset(
                    Int(inbox_origin)
                    + (seq % nslots) * Int(inbox_stride)
                    + vlo * 16
                ),
                n,
                slot,
                npeers,
            )
            barrier()
            if thread_idx.x == 0:
                if _arrive(me, _RS_STREAM_CONS, j, nblocks):
                    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                        mb_consumed, UInt64(seq)
                    )


# ===-------------------------------------------------------------------=== #
# Host side
# ===-------------------------------------------------------------------=== #


def _resident[NW: Int](ctx: DeviceContext, sm_count: Int) raises -> Int:
    return sm_count * _cached_occupancy[_stream_rs_kernel[NW]](
        ctx,
        String(t"stream_rs_float32_{NW}"),
        RS_STREAM_THREADS,
    )


def reduce_scatter_stream_wanted(count: Int, split_unit_bytes: Int) -> Bool:
    """Whether this call has anything to stream.

    Below `PIPE_SPLIT_UNIT` a call is one chunk and one piece per block -- the
    fused kernel's schedule exactly, minus its grid barriers and plus the
    exchange's extra device-word hop -- and it measured 128 us against 124 for
    a 512 KiB fp32 reduce-scatter at 16 ranks on 2x8 H100 (NCCL 76). So the
    small end keeps `rs_fused.mojo`."""
    return count * 4 >= split_unit_bytes


def reduce_scatter_stream_plan(
    count: Int,
    chunk_cap: Int,
    narenas: Int,
    split_unit_bytes: Int,
) raises -> Tuple[Int, Int, Int]:
    """(chunk elements, chunks, pipeline depth) for `count` elements."""
    if count <= 0 or chunk_cap <= 0 or narenas <= 0:
        raise Error("mojoccl: invalid streaming reduce-scatter geometry")
    var chunk = min(count, chunk_cap)
    if count * 4 >= split_unit_bytes:
        chunk = min(
            chunk,
            _align_up(
                (count + RS_STREAM_TARGET_CHUNKS - 1)
                // RS_STREAM_TARGET_CHUNKS
                * 4,
                16,
            )
            // 4,
        )
    var nchunks = (count + chunk - 1) // chunk
    return Tuple(chunk, nchunks, min(narenas, nchunks))


def reduce_scatter_stream_pieces(
    chunk_elems: Int, blocks: Int
) -> Tuple[Int, Int]:
    """(vectors per piece, pieces per chunk) -- host and device agree, and so
    does every rank, because both follow from the agreed (chunk, grid)."""
    var pv = RS_STREAM_UNROLL * RS_STREAM_SLICE_UNROLLS * RS_STREAM_THREADS
    var vc = chunk_elems // 4
    var vpb = max(1, (vc + blocks - 1) // blocks)
    return Tuple(pv, max(1, (vpb + pv - 1) // pv))


def reduce_scatter_stream_generations(nchunks: Int, pieces: Int) -> Int:
    """Generations one call consumes: its flag values run from `base+1` to
    `base + nchunks*pieces`, and they are never reset."""
    return (nchunks * pieces + PHASES_PER_GEN) // PHASES_PER_GEN


def reduce_scatter_stream_blocks(
    ctx: DeviceContext,
    world: Int,
    sm_count: Int,
    chunk_elems: Int,
    block_cap: Int,
) raises -> Int:
    var resident: Int
    if world == 8:
        resident = _resident[8](ctx, sm_count)
    elif world == 4:
        resident = _resident[4](ctx, sm_count)
    elif world == 2:
        resident = _resident[2](ctx, sm_count)
    else:
        resident = _resident[0](ctx, sm_count)
    # All grid inputs are agreed at bootstrap; local occupancy only validates.
    var blocks = min(
        sm_count,
        min(
            RS_STREAM_ROWS,
            min(
                block_cap,
                max(
                    1,
                    (chunk_elems // 4 + RS_STREAM_THREADS - 1)
                    // RS_STREAM_THREADS,
                ),
            ),
        ),
    )
    if blocks <= 0 or resident < blocks:
        raise Error(
            "mojoccl: streaming reduce-scatter cannot hold the agreed grid"
        )
    return blocks


def _launch[
    NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    blocks: Int,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    mailbox: StaticTuple[Int, 3],
    count: Int,
    chunk_elems: Int,
    seq0: Int,
    arena_stride: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    piece_vecs: Int,
    pieces: Int,
    shape: StaticTuple[Int32, 7],
    world: Int,
    rank: Int,
    generation: Int,
    scale: Float32,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
) raises:
    _enqueue_cached_dim[_stream_rs_kernel[NW]](
        ctx,
        stream,
        String(t"stream_rs_float32_{NW}"),
        blocks,
        RS_STREAM_THREADS,
        True,
        regions,
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=out_ptr),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[0]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[1]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[2]),
        Int64(count),
        Int64(chunk_elems),
        Int64(seq0),
        Int64(arena_stride),
        Int64(inbox_origin),
        Int64(inbox_stride),
        Int64(piece_vecs),
        Int64(pieces),
        shape,
        Int32(world),
        Int32(rank),
        UInt64(generation) * UInt64(PHASES_PER_GEN),
        scale,
        timeout_ns,
        rank_ids,
        Int32(1) if (in_ptr | (count * 4) | (chunk_elems * 4)) % 16
        == 0 else Int32(0),
    )


def reduce_scatter_stream(
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
    seq0: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    nslots: Int,
    npeers: Int,
    nnodes: Int,
    my_node: Int,
    generation: Int,
    scale: Float32,
    blocks: Int,
    piece_vecs: Int,
    pieces: Int,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
) raises:
    if count <= 0:
        return
    if nchunks <= 0 or nchunks > WORK_SLOTS or chunk_elems <= 0:
        raise Error("mojoccl: invalid streaming reduce-scatter chunks")
    if blocks <= 0 or blocks > RS_STREAM_ROWS:
        raise Error("mojoccl: invalid streaming reduce-scatter grid")
    var rp = _region_ptrs(regions, rank, world)
    var shape = StaticTuple[Int32, 7](
        Int32(nchunks),
        Int32(depth),
        Int32(narenas),
        Int32(nslots),
        Int32(npeers),
        Int32(nnodes),
        Int32(my_node),
    )
    if world == 8:
        _launch[8](
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
            inbox_origin,
            inbox_stride,
            piece_vecs,
            pieces,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    elif world == 4:
        _launch[4](
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
            inbox_origin,
            inbox_stride,
            piece_vecs,
            pieces,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    elif world == 2:
        _launch[2](
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
            inbox_origin,
            inbox_stride,
            piece_vecs,
            pieces,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    else:
        _launch[0](
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
            inbox_origin,
            inbox_stride,
            piece_vecs,
            pieces,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
