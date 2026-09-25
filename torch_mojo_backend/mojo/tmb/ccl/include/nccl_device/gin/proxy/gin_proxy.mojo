# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_device/gin/proxy/gin_proxy.h

from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.utils import StaticTuple

from tmb.ccl.device.common import (
    _POLL_ORDER,
    _enqueue_cached,
    abort_raised,
    device_now_ns,
    latch_arena_error,
    poll_acquire,
    poll_pause,
    publish_fault,
)
from tmb.ccl.include.device import BLOCK, ERR_PROXY_WAIT, FAULT_NO_PEER


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
@__name("ccl_internode_proxy_request")
def _proxy_request_kernel(mailbox: Pointer[UInt64, MutAnyOrigin], seq: UInt64):
    """Hand exchange `seq` to the progress thread.

    A release store into pinned host memory, so everything the stream did
    before this kernel -- the reduce-scatter that produced the shard the
    thread is about to send -- is visible to the CPU that acquires it.

    One store and nothing else: the stopped-communicator guard lives on the
    thread that acquires this store (`proxy._proxy_main`, which says why
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
    (`proxy._proxy_main`), so every wait still queued behind it is
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
