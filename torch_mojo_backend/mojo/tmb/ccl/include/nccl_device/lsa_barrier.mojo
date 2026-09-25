# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_device/lsa_barrier.h
#
# Synchronisation
# ---------------
# The signal area holds one UInt64 flag per (block, writer rank).  Rank r's
# block b publishes `generation * PHASES_PER_GEN + phase` into every peer's
# flags[b][r] with a system-scope release store and then waits for its own
# flags[b][p] to reach that value for every peer p.  Flag values are derived
# from `generation`, never reset, and compared with `>=`, so a peer that is a
# whole call ahead can never deadlock a peer that is behind and nothing has to
# be cleared between calls.  Every collective opens with a start barrier, so
# one rule covers the whole arena: no generation writes it until every rank has
# finished reading the previous generation.  That is what lets collectives of
# different kinds and sizes share the staging area -- see the invariant above
# `_ar_twoshot_kernel`.
#
# Portability: NVIDIA and AMD share every line of the device code except the
# two places the header calls out (phase 3 of the allreduce, and the fence
# discipline).  On NVIDIA ordering is `Atomic[...].store[RELEASE]` /
# `load[ACQUIRE]` at default (system) scope, which lowers to
# `st.release.sys.global` / `ld.acquire.sys.global` on sm_90a.  On gfx942 the
# region is uncached and the flags are published with `s_waitcnt lgkmcnt(0)
# vmcnt(0)` plus a relaxed store -- RCCL's "cheap post-send fence", see
# `_sync`.  Blocks are 256 threads (RCCL's gfx942 maximum) and every layout is
# wave-64 safe.

from std.memory import AddressSpace, stack_allocation
from std.collections import Array
from std.atomic import Atomic, Ordering, fence
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx
from std.sys import llvm_intrinsic

from tmb.ccl.device.common import (
    _POLL_ORDER,
    _abort_raised,
    abort_raised,
    device_now_ns,
    latch_arena_error,
    poll_acquire,
    poll_pause,
    publish_fault,
    status_page,
)
from tmb.ccl.include.device import (
    FAULT_NO_PEER,
    MAX_WORLD,
    PHASES_PER_GEN,
    _AMD,
    _FLAG_BYTE_OFFSET,
    _GRIDBAR_ARRIVE_OFFSET,
    _GRIDBAR_RELEASE_OFFSET,
    _SPIN_CHECK,
)


@always_inline
def _flag_target(generation: Int, phase: Int) -> UInt64:
    """The flag value that marks `phase` of `generation`.

    Strictly increasing in (generation, phase), never reset, so waiters compare
    with `>=` and a peer running ahead is harmless.
    """
    return UInt64(generation) * UInt64(PHASES_PER_GEN) + UInt64(phase)


# ===-------------------------------------------------------------------=== #
# Device-side primitives
# ===-------------------------------------------------------------------=== #


@always_inline
def _flags(
    region: Pointer[UInt8, MutAnyOrigin],
) -> Pointer[UInt64, MutAnyOrigin]:
    """flags[block][writer_rank], row-major, MAX_WORLD columns."""
    return region.unsafe_offset(_FLAG_BYTE_OFFSET).unsafe_bitcast[UInt64]()


@always_inline
def _record_deadline(
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    rank: Int,
    code: Int,
    target: UInt64,
    peer: UInt64,
    seen: UInt64,
    arena_off: Int = 0,
):
    """A barrier gave up: record it in the arena and latch it for the host.

    Called by the thread that hit the deadline, which is the only one that
    knows which peer it was waiting for and what that peer's flag held. The
    arena word is what it always was (`ncclCommGetAsyncError` reads it); the
    status page is what makes the next collective on this communicator fail
    loudly instead of returning garbage.
    """
    var region = regions[rank].unsafe_offset(arena_off)
    var phase = Int(target % UInt64(PHASES_PER_GEN))
    # `error_offset()` is 0: the error word is the first word of the region.
    if not latch_arena_error(region.unsafe_bitcast[UInt64](), code, phase):
        return
    publish_fault(
        status_page(region),
        code,
        phase,
        Int(block_idx.x),
        peer,
        seen,
        target,
        Int(region),
    )


@always_inline
def _sync(
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    world: Int,
    rank: Int,
    code: Int,
    target: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
    arena_off: Int = 0,
    row: Int = -1,
) -> Bool:
    """Block-scoped barrier across the same block index on every rank.

    `row` >= 0 uses that flag row instead of the block index: the gate
    (`_rs_gate_kernel`) barriers a whole rank from one block on `GATE_ROW`.

    Thread `p` (p < world) publishes `target` into peer p's flags[bid][rank]
    -- after every payload write this block made into peer memory has landed,
    which is what the fence below is for -- then waits for peer p's flag in my
    own region to reach `target`.

    Blocks are matched by index: every collective in this file gives block b of
    every rank exactly the same grid-stride slice of the index space, so block b
    only ever consumes bytes block b of a peer produced.

    Returns False if any participating thread hit the deadline; the whole block
    learns that through shared memory so no thread is left inside a `barrier()`.

    `code` is the caller's `ERR_*` constant, and recording the failure is this
    function's job rather than the caller's: only the thread that gave up knows
    which peer it was waiting for and what that peer's flag held, and that is
    most of what a reader of the message needs. Every caller used to follow a
    False with the same two lines; folding them in is also how a new caller
    stops being able to forget them.
    """
    var failed = stack_allocation[
        1, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    comptime if _AMD:
        # gfx942's `s_barrier` is emitted with `s_waitcnt lgkmcnt(0)` only, so
        # another wave's payload stores can still be in flight when one thread
        # publishes the flag; RCCL puts `vmcnt(0)` inside its block barrier for
        # exactly this reason (rccl:src/device/prims_simple.h:193-210). A
        # release fence in every thread is the portable spelling and lowers to
        # `s_waitcnt vmcnt(0)` + `buffer_wbl2 sc0 sc1`, i.e. the writeback that
        # makes this block's payload stores visible to the peer that is about
        # to be told they are there.
        #
        # Both cheaper spellings were tried and both are wrong on this box, in
        # the same way and only for small payloads:
        #   * no writeback at all (RCCL's `skip_fence` for cudaArch 940, which
        #     is sound for RCCL because its P2P buffers are uncached) --
        #     broke every broadcast;
        #   * the writeback moved after the barrier into the `world` threads
        #     that publish flags -- fixed the broadcast at 4 ranks but still
        #     failed a 1-element allreduce and a broadcast at 2 ranks.
        # Our region is `hipDeviceMallocUncached`, but the mapping a peer
        # writes *through* comes from `hipIpcOpenMemHandle` and does not carry
        # that memory type, so a few bytes can still be sitting in the writer's
        # cache. Megabyte payloads drain on their own, which is why only the
        # small collectives ever failed. NVIDIA needs nothing: `bar.sync` is a
        # CTA-scope fence and the release store below is cumulative over it,
        # which is what NCCL's postPeer relies on.
        fence[ordering=Ordering.RELEASE]()
    barrier()

    if Int(thread_idx.x) < world:
        var peer = Int(thread_idx.x)
        var bid = Int(block_idx.x) if row < 0 else row
        # Only the acquire side is relaxed on gfx942 (`poll_acquire`); both
        # releases stay, for the small-payload reason above.
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            _flags(regions[peer].unsafe_offset(arena_off)).unsafe_offset(
                bid * MAX_WORLD + rank
            ),
            target,
        )
        var mine = _flags(regions[rank].unsafe_offset(arena_off)).unsafe_offset(
            bid * MAX_WORLD + peer
        )
        var spins = 0
        while (
            Atomic[Scalar[DType.uint64]].load[ordering=_POLL_ORDER](mine)
            < target
        ):
            poll_pause()
            spins += 1
            if spins >= _SPIN_CHECK:
                spins = 0
                if _abort_raised(regions[rank].unsafe_offset(arena_off)):
                    # Abort is a request, not a failure: record it in the
                    # arena word the way this file always has, but do not
                    # latch a fault. `ncclCommAbort` already told the host
                    # what happened, and an aborted communicator has to keep
                    # answering NCCL_INVALID_USAGE rather than start
                    # answering NCCL_REMOTE_ERROR.
                    _ = latch_arena_error(
                        regions[rank]
                        .unsafe_offset(arena_off)
                        .unsafe_bitcast[UInt64](),
                        code,
                        Int(target % UInt64(PHASES_PER_GEN)),
                    )
                    failed[unsafe_offset=0] = 1
                    break
                # Same-GPU timer difference only; never compared across GPUs.
                if device_now_ns() - t0 > timeout_ns:
                    _record_deadline(
                        regions,
                        rank,
                        code,
                        target,
                        UInt64(peer),
                        Atomic[Scalar[DType.uint64]].load[
                            ordering=Ordering.ACQUIRE
                        ](mine),
                        arena_off,
                    )
                    failed[unsafe_offset=0] = 1
                    break
    poll_acquire()  # outside the polling branch on purpose
    barrier()
    return failed[unsafe_offset=0] == 0


@always_inline
def grid_barrier(
    region: Pointer[UInt8, MutAnyOrigin],
    poison: Pointer[UInt64, MutAnyOrigin],
    nblocks: Int,
    code: Int,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """Rendezvous of every block of ONE grid on ONE GPU; True if it completed.

    What a kernel boundary used to provide. The split inter-node schedule got
    "every block of this rank has finished phase P" from launching P+1 as a
    separate kernel; the fused kernel (all_reduce_gin.mojo) runs the whole
    pipeline in one launch and has to say it itself -- before it hands a shard
    to the NIC, before it reads what the NIC delivered, and at the end of each
    chunk, where it restores exactly the invariant the launch boundary had.

    THE CALLER MUST GUARANTEE THAT EVERY BLOCK IS RESIDENT. Blocks that have
    arrived spin, so a grid larger than the device can hold at once deadlocks:
    the launcher sizes the grid at one block per SM/CU for that reason.

    Sense reversal, not a target count: the last block to arrive resets the
    counter and bumps a release word every other block is waiting to see
    change. Nothing is carried across launches and nothing has to agree with
    the host, which a "wait for arrival number N" barrier would need -- and
    would hang on for ever if the host's arithmetic and the kernel's loop ever
    disagreed by one.

    `poison` is how a block that has already given up -- in a spin of its own,
    or on a peer that stopped answering -- releases the blocks waiting here
    instead of leaving each of them to burn its own full deadline. It is a
    device word, read only from the slow path of the spin.

    Ordering: each block's arrival is a release RMW after its payload writes,
    the last arriver's RMW acquires the release sequence of all of them, and
    its release store of the sense word is what the waiters acquire. So a
    block that leaves this barrier sees every write every block made before
    entering it -- including, for block 0, the shard the NIC is about to read.
    """
    var failed = stack_allocation[
        1, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    comptime if _AMD:
        # Same reason as `_sync`: gfx942's `s_barrier` does not wait on
        # outstanding vector stores, so the block's payload writes need an
        # explicit release before one thread announces them.
        fence[ordering=Ordering.RELEASE]()
    barrier()

    if thread_idx.x == 0:
        var arrive = region.unsafe_offset(
            _GRIDBAR_ARRIVE_OFFSET
        ).unsafe_bitcast[UInt64]()
        var release = region.unsafe_offset(
            _GRIDBAR_RELEASE_OFFSET
        ).unsafe_bitcast[UInt64]()
        var seen = Atomic[Scalar[DType.uint64]].load[ordering=Ordering.RELAXED](
            release
        )
        var was = Atomic[Scalar[DType.uint64]].fetch_add[
            ordering=Ordering.ACQUIRE_RELEASE
        ](arrive, UInt64(1))
        if Int(was) == nblocks - 1:
            # Last in. Reset the counter first: no block can arrive at the
            # next barrier before it has seen the sense word below change,
            # and the release store orders this plain one ahead of it.
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELAXED](
                arrive, UInt64(0)
            )
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                release, seen + 1
            )
        else:
            var page = status_page(region)
            var spins = 0
            while (
                Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                    release
                )
                == seen
            ):
                comptime if _AMD:
                    # On gfx942 the acquire load above is a `buffer_inv sc0
                    # sc1` -- an L1+L2 invalidate -- per iteration, and
                    # nblocks-1 threads spin here for a whole network round
                    # trip (gin_proxy.mojo has the measurement for
                    # one such thread). Sleep between polls; the ordering
                    # stays as it is.
                    # MI300A sweep, Adastra job 5417296 (2026-09-15):
                    # sleep 0/1/2/4/8 passed payload/deadline probes; the
                    # best 168 MiB gain was 0.89%, below reproduction
                    # noise (~1%). Keep the measured production sleep 2.
                    llvm_intrinsic[
                        "llvm.amdgcn.s.sleep", NoneType, has_side_effect=True
                    ](Int32(2))
                spins += 1
                if spins >= _SPIN_CHECK:
                    spins = 0
                    if (
                        Atomic[Scalar[DType.uint64]].load[
                            ordering=Ordering.ACQUIRE
                        ](poison)
                        != 0
                    ):
                        failed[unsafe_offset=0] = 1
                        break
                    if abort_raised(page):
                        _ = latch_arena_error(
                            region.unsafe_bitcast[UInt64](), code, 0
                        )
                        failed[unsafe_offset=0] = 1
                        break
                    if device_now_ns() - t0 > timeout_ns:
                        if latch_arena_error(
                            region.unsafe_bitcast[UInt64](), code, 0
                        ):
                            publish_fault(
                                page,
                                code,
                                0,
                                Int(block_idx.x),
                                FAULT_NO_PEER,
                                seen,
                                seen + 1,
                                Int(region),
                            )
                        failed[unsafe_offset=0] = 1
                        break
        # Read once, by one thread, for the whole block: a block that has
        # been released may still be one another block just poisoned, and
        # the answer has to be the same for every thread of this CTA or part
        # of it would leave while the rest reaches the next `barrier()`.
        if (
            failed[unsafe_offset=0] == 0
            and Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                poison
            )
            != 0
        ):
            failed[unsafe_offset=0] = 1
    barrier()
    return failed[unsafe_offset=0] == 0
