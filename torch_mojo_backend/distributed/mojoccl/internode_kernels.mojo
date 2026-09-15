# The three device kernels the inter-node hop needs on top of
# collectives_kernels.mojo, which is untouched.
#
# None of them synchronizes with anything: stream order does it. Each runs
# after the host callback that put the network data in place (internode.mojo)
# and before the intra-node collective that consumes the result, so there is
# no flag protocol here and no peer pointer -- every address is inside this
# rank's own region or its own user buffers.

from std.atomic import Atomic, Ordering
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from collectives_kernels import device_now_ns
from std.sys import size_of
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream

from collectives_kernels import (
    BLOCK,
    ERR_PROXY_WAIT,
    FAULT_NO_PEER,
    MAX_WORLD,
    _copy_bytes,
    _enqueue_cached,
    abort_raised,
    latch_arena_error,
    publish_fault,
)

comptime _UNROLL = 4
comptime _MAX_BLOCKS = 432
comptime _ABORT_CHECK = 256
"""Mailbox reads between two probes of the abort word. Both are host memory
across PCIe, so probing every iteration would double the wait kernel's traffic
for no gain: 256 iterations is well under a millisecond."""

# REVERTED: polling the mailbox with a relaxed load and one acquire fence at
# the end.
#
# The measurement that motivated it is real. On gfx942 an acquire load at
# system scope lowers to `global_load ... sc0 sc1` followed by `buffer_inv sc0
# sc1`, a whole L1 AND L2 invalidate, and this kernel spins for as long as an
# exchange takes -- so every other kernel resident on the GPU loses its L2,
# millions of times a second. The relaxed spelling removed every invalidate
# from the loop (verified in the assembly: 0 in the loop, 1 for the fence
# after it) and left the sm_90a PTX byte-identical.
#
# It is reverted anyway, for the reason the same change was reverted in
# `collectives_kernels.mojo`'s barrier (see that file's history): there is no
# argument for why the cheap version is sound on this hardware, only a
# symmetry that looks right -- the spin load still carries `sc0 sc1`, so it
# cannot read a stale flag, and the payload ordering is provided once by the
# fence. A one-element allreduce at 2 ranks flaked 2 runs in 13 with the
# barrier's version and 0 in 12 without. And the benefit here was never
# measured on a workload: nanoGPT's throughput was identical with and without
# it, because what actually cost 26x was MAX's VMM allocator, not this.
#
# Unmeasured benefit plus an unexplained multi-node stall in the same
# neighbourhood is not a trade worth making. If it comes back it should come
# back with a soundness argument and a workload that shows the win.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_internode_inbox_add_{dtype}")
def _inbox_add_kernel[
    dtype: DType, W: Int
](
    shard: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int64,
    slot_bytes: Int64,
    npeers_i: Int32,
):
    """`shard += sum of the npeers inbox slots`, elementwise.

    The node-local sum of this rank's shard is already in `shard` (that is
    what `reduce_scatter_stage` left there); each remote node's node-local
    sum of the SAME shard has landed in one inbox slot. Adding them makes
    `shard` the global sum, which `allgather_finish` then spreads.

    Accumulated in the wire dtype, like the rest of the hierarchical path:
    the remote sums arrived rounded already, so a wider accumulator here
    would buy nothing.
    """
    _inbox_add_body[dtype, W](
        shard,
        inbox,
        Int(count),
        Int(slot_bytes),
        Int(npeers_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
    )


@always_inline
def _inbox_add_body[
    dtype: DType, W: Int
](
    shard: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    n: Int,
    sb: Int,
    npeers: Int,
    tid: Int,
    stride: Int,
):
    """`_inbox_add_kernel`'s body, shared with the fused inter-node kernel
    (internode_fused.mojo), which runs it once per chunk on its own
    grid-stride slice."""
    var nv = n // W

    for v in range(tid, nv, stride):
        var acc = shard.unsafe_load[width=W, alignment=16](v * W)
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src.unsafe_load[width=W, alignment=16](v * W)
        shard.unsafe_store[width=W, alignment=16](v * W, acc)

    for i in range(nv * W + tid, n, stride):
        var acc = shard[unsafe_offset=i]
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src[unsafe_offset=i]
        shard[unsafe_offset=i] = acc


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_proxy_request")
def _proxy_request_kernel(mailbox: Pointer[UInt64, MutAnyOrigin], seq: UInt64):
    """Hand exchange `seq` to the progress thread.

    A release store into pinned host memory, so everything the stream did
    before this kernel -- the reduce-scatter that produced the shard the
    thread is about to send -- is visible to the CPU that acquires it.

    One store and nothing else: the stopped-communicator guard lives on the
    thread that acquires this store (`internode._proxy_main`, which says why
    that is the stronger place), not in front of it, where each load of the
    pinned status page cost a PCIe round trip per exchange.
    """
    if global_idx.x == 0:
        Atomic[DType.uint64].store[ordering=Ordering.RELEASE](mailbox, seq)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_proxy_wait")
def _proxy_wait_kernel(
    mailbox: Pointer[UInt64, MutAnyOrigin],
    error_word: Pointer[UInt64, MutAnyOrigin],
    status: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
    timeout_ns: UInt64,
):
    """Hold the stream until the progress thread reports exchange `seq` done.

    One thread spinning on pinned host memory. This is the whole reason the
    proxy exists: the same rendezvous through `cuLaunchHostFunc` cost about
    480 us per exchange on this cluster (measured, job 234035 -- 1 MiB
    allreduce 496 us against 24 us on one node), because the driver has to
    stop the stream, wake a thread and restart it. A spin kernel and a
    spinning CPU thread cost a launch each.

    On the deadline -- or as soon as `ncclCommAbort` raises the abort word,
    which is why abort does not cost a full deadline -- it records the failure
    and gives up rather than hanging the stream forever. A deadline also
    latches the communicator's fault (`publish_fault`), which is what stops
    the queued `inbox_add` behind this kernel from being treated as a
    successful exchange: the host's next collective returns
    NCCL_REMOTE_ERROR instead of the sum of an inbox nobody filled.

    An already-latched fault leaves the same way an abort does -- it raises
    the same word (`publish_fault`) -- and for the same reason: after the
    first failure the progress thread stops honouring the mailbox
    (`internode._proxy_main`), so every wait still queued behind it is
    waiting for an exchange that will never be asked for, and waiting a full
    deadline each would turn one 60 s stall into as many, one per chunk.

    Nothing is checked BEFORE the spin: a load of the pinned status page is
    a PCIe round trip in a kernel that runs once per exchange, and the spin's
    own check every `_ABORT_CHECK` iterations already bounds how long a
    stopped communicator holds the stream (a few hundred microseconds
    against a 60 s deadline). A wait that finds the mailbox already at `seq`
    never touches the page, as before.
    """
    if global_idx.x == 0:
        var page = Int(status)
        var t0 = device_now_ns()
        var spins = 0
        while (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](mailbox) < seq
        ):
            spins += 1
            if spins >= _ABORT_CHECK:
                spins = 0
                if abort_raised(page):
                    # First writer wins, as everywhere else: an abort or a
                    # fault that arrived from elsewhere already has a better
                    # explanation in this word than "the exchange wait gave
                    # up because of it".
                    _ = latch_arena_error(error_word, ERR_PROXY_WAIT, 0)
                    return
            if device_now_ns() - t0 > timeout_ns:
                # The peer of this wait is my own progress thread, not another
                # rank, so there is no flag and no peer to name: `seen` is how
                # far the engine had got, `target` the exchange asked for.
                if latch_arena_error(error_word, ERR_PROXY_WAIT, 0):
                    publish_fault(
                        page,
                        ERR_PROXY_WAIT,
                        0,
                        0,
                        FAULT_NO_PEER,
                        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                            mailbox
                        ),
                        seq,
                        Int(error_word),
                    )
                return


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_copy_bytes")
def _copy_kernel(
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
):
    """Staging copy: a user buffer is not in the registered region, so
    broadcast and allgather have to move their payload in and out of it."""
    _copy_bytes[_UNROLL](
        dst,
        src,
        Int(nbytes),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_place_blocks")
def _place_blocks_kernel(
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    dst_offsets: StaticTuple[Int64, MAX_WORLD],
    block_bytes: Int64,
    nblocks_i: Int32,
):
    """Scatter one node's allgather block into the output by global rank.

    A node's block holds its `local_world` contributions in local-rank order;
    the output wants them at their global ranks, which torchrun's numbering
    makes contiguous but which this library reads out of the bootstrap table
    instead of assuming. One launch per node beats `local_world` launches of
    a few bytes each.
    """
    var nb = Int(block_bytes)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    for l in range(Int(nblocks_i)):
        _copy_bytes[_UNROLL](
            dst.unsafe_offset(Int(dst_offsets[l])),
            src.unsafe_offset(l * nb),
            nb,
            tid,
            stride,
        )


# ===-------------------------------------------------------------------=== #
# Launchers
# ===-------------------------------------------------------------------=== #


def _blocks_for(nbytes: Int) -> Int:
    return min(_MAX_BLOCKS, max(1, (nbytes // 16 + BLOCK - 1) // BLOCK))


def inbox_add[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    shard_ptr: Int,
    inbox_ptr: Int,
    count: Int,
    slot_bytes: Int,
    npeers: Int,
) raises:
    """Enqueue `shard += sum(inbox slots)` on `stream`."""
    if count <= 0 or npeers <= 0:
        return
    comptime W = 16 // size_of[dtype]()
    _enqueue_cached[_inbox_add_kernel[dtype, W]](
        ctx,
        stream,
        String(t"ib_add_{dtype}"),
        _blocks_for(count * size_of[dtype]()),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=shard_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=inbox_ptr),
        Int64(count),
        Int64(slot_bytes),
        Int32(npeers),
    )


def copy_bytes(
    ctx: DeviceContext, stream: DeviceStream, dst: Int, src: Int, nbytes: Int
) raises:
    if nbytes <= 0:
        return
    _enqueue_cached[_copy_kernel](
        ctx,
        stream,
        "ib_copy",
        _blocks_for(nbytes),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        Int64(nbytes),
    )


def place_blocks(
    ctx: DeviceContext,
    stream: DeviceStream,
    dst: Int,
    src: Int,
    dst_offsets: StaticTuple[Int64, MAX_WORLD],
    block_bytes: Int,
    nblocks: Int,
) raises:
    if block_bytes <= 0 or nblocks <= 0:
        return
    _enqueue_cached[_place_blocks_kernel](
        ctx,
        stream,
        "ib_place",
        _blocks_for(block_bytes),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        dst_offsets,
        Int64(block_bytes),
        Int32(nblocks),
    )


def proxy_request(
    ctx: DeviceContext, stream: DeviceStream, mailbox: Int, seq: Int
) raises:
    _enqueue_cached[_proxy_request_kernel](
        ctx,
        stream,
        "ib_req",
        1,
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox),
        UInt64(seq),
    )


def proxy_wait(
    ctx: DeviceContext,
    stream: DeviceStream,
    mailbox: Int,
    error_word: Int,
    status: Int,
    seq: Int,
    timeout_ns: Int,
) raises:
    _enqueue_cached[_proxy_wait_kernel](
        ctx,
        stream,
        "ib_wait",
        1,
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=error_word),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=status),
        UInt64(seq),
        UInt64(timeout_ns),
    )
