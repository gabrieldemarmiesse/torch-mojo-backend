# The three device kernels the inter-node hop needs on top of
# collectives_kernels.mojo, which is untouched.
#
# None of them synchronizes with anything: stream order does it. Each runs
# after the host callback that put the network data in place (internode.mojo)
# and before the intra-node collective that consumes the result, so there is
# no flag protocol here and no peer pointer -- every address is inside this
# rank's own region or its own user buffers.

from std.atomic import Atomic, Ordering
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from tmb.ccl.collectives_kernels import device_now_ns
from std.sys import size_of
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream

from tmb.ccl.collectives_kernels import (
    BLOCK,
    ERR_PROXY_WAIT,
    FAULT_NO_PEER,
    MAX_WORLD,
    _POLL_ORDER,
    _copy_bytes,
    _enqueue_cached,
    abort_raised,
    latch_arena_error,
    poll_acquire,
    poll_pause,
    publish_fault,
)

comptime _UNROLL = 4
comptime _MAX_BLOCKS = 432
comptime _ABORT_CHECK = 256
"""Mailbox reads between two probes of the abort word. Both are host memory
across PCIe, so probing every iteration would double the wait kernel's traffic
for no gain: 256 iterations is well under a millisecond."""

# gfx942 polls MB_DONE relaxed and acquires once (`poll_acquire`); the
# progress thread flushes the NIC writes before release-storing it. A
# transport error can release it without data, as before; the fault stays
# latched.


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
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            mailbox, seq
        )


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
            Atomic[Scalar[DType.uint64]].load[ordering=_POLL_ORDER](mailbox)
            < seq
        ):
            poll_pause()
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
                        Atomic[Scalar[DType.uint64]].load[
                            ordering=Ordering.ACQUIRE
                        ](mailbox),
                        seq,
                        Int(error_word),
                    )
                return

        poll_acquire()


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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_internode_inbox_sum_out_{dtype}_v{W}")
def _inbox_sum_out_kernel[
    dtype: DType, W: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    partial: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int64,
    slot_bytes: Int64,
    npeers_i: Int32,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(count)
    for v in range(tid, n // W, stride):
        var acc = partial.unsafe_load[width=W](v * W).cast[accum]()
        for j in range(Int(npeers_i)):
            var src = inbox.unsafe_offset(j * Int(slot_bytes)).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src.unsafe_load[width=W](v * W).cast[accum]()
        dst.unsafe_store[width=W](v * W, acc.cast[dtype]())
    for i in range(n // W * W + tid, n, stride):
        var acc = partial[unsafe_offset=i].cast[accum]()
        for j in range(Int(npeers_i)):
            var src = inbox.unsafe_offset(j * Int(slot_bytes)).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src[unsafe_offset=i].cast[accum]()
        dst[unsafe_offset=i] = acc.cast[dtype]()


def inbox_sum_out[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    out_ptr: Int,
    partial_ptr: Int,
    inbox_ptr: Int,
    count: Int,
    slot_bytes: Int,
    npeers: Int,
) raises:
    """Sum node partials straight into user memory, including offset views."""
    comptime W = 16 // size_of[dtype]()
    _enqueue_cached[_inbox_sum_out_kernel[dtype, W]](
        ctx,
        stream,
        String(t"ib_sum_out_{dtype}"),
        _blocks_for(count * size_of[dtype]()),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=partial_ptr),
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
