# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/comm.h

from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream
from std.collections import Dict
from std.ffi import OwnedDLHandle, external_call
from std.utils import StaticTuple
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc
from max.gpu import global_idx

from tmb.ccl.device.all_reduce import ERR_NVLS_SYNC
from tmb.ccl.device.common import _enqueue_cached, spin_timeout_ns
from tmb.ccl.device.symmetric.all_reduce_gin import fused_big_bytes
from tmb.ccl.include.device import (
    ERR_AG_FINISH_SYNC,
    ERR_ALLGATHER_SYNC,
    ERR_ALLREDUCE_SYNC,
    ERR_BROADCAST_SYNC,
    ERR_FUSED_GRID,
    ERR_HOST_LAUNCH,
    ERR_PROXY_WAIT,
    ERR_REDUCE_SCATTER_SYNC,
    ERR_RS_STAGE_SYNC,
    FAULT_ARENA,
    FAULT_BLOCK,
    FAULT_CODE,
    FAULT_NO_PEER,
    FAULT_PEER,
    FAULT_PHASE,
    FAULT_SEEN,
    FAULT_TARGET,
    MAX_WORLD,
    PHASES_PER_GEN,
    STATUS_FAULT_WORD,
    STATUS_HOST_FAULT_WORD,
    error_offset,
    signal_bytes,
)
from tmb.ccl.include.transport import NvlsRegion
from tmb.ccl.misc.strongstream import CompletionEvent
from tmb.ccl.nccl import NCCL_INTERNAL_ERROR, NCCL_REMOTE_ERROR, NCCL_SUCCESS
from tmb.ccl.transport.net import CREDIT_AREA_BYTES, ib_error
from tmb.ccl.transport.nvls import sm_count


# Staging arenas the multi-node region is carved into, and therefore chunks
# the pipeline may keep alive (`_arena_regions`, `_do_allreduce`).
#
# 4, not 2: the pipeline is GPU-bound at every size that gets chunked, so
# depth 2 overlaps in principle, but its window is one reduce-scatter and
# the progress thread's idle backoff alone (20 us)
# can eat that. Depth 4 gives a window of two whole chunks, and costs region
# layout rather than memory -- each arena is 1/4 of the staging.
comptime PIPE_ARENAS = 4

# Fixed inbox slot groups (transport/net.mojo owns what they are for). One more
# than the arenas: at PIPE_ARENAS the credit a rank needs arrives exactly
# when it asks for it, putting a round trip on the critical path; the extra
# group means it went out a chunk earlier.
comptime INBOX_SLOTS = PIPE_ARENAS + 1

# Largest number of pipeline chunks one collective is cut into.
comptime PIPE_MAX_CHUNKS = 16

# Chunking constant, in bytes: a `B`-byte multi-node allreduce is cut into
# `K = sqrt(B / (local_world * PIPE_SPLIT_UNIT))` chunks.
#
# An extra chunk costs one more reduce-scatter and one more all-gather
# launch, each a launch plus an 8-way start barrier, ~8 us apiece on this
# cluster (agents_docs/mojo_collectives_kernel_results.md 10.3 reads the pair at
# 15.9 us for a 4-byte message, which is all fixed cost). Pipelining hides
# all but ~1/K of the network, and the network of a B-byte allreduce is
# B/(local_world * 40 GB/s), the RDMA measured at 40-45 GB/s per rank (one
# HCA each, job 234072). Minimising 16K + B/(L*40000) us gives
# K = sqrt(B / (L * 640000)).
#
# It also keeps a chunk's shard above 1 MiB, where the RDMA is still at line
# rate, without a second clause: at K > 1 the shard is sqrt(B * 640000 / L)
# bytes, 1.5 MB at the 27 MiB bucket and 3.8 MB at 168 MiB.
#
# Measured, not assumed: halving this to 320_000 (K of 3/8/14 instead of
# 2/5/10 at 27/168/512 MiB) is WORSE -- 27 MiB unchanged, 168 MiB 1291 vs
# 1192 us and 512 MiB 3648 vs 3541, 16 ranks on 2x8 H100, ABBA against the
# same base commit in one job (234242 against 234237). The per-exchange
# latency an extra chunk adds is not fully hidden, so splitting past the
# point where the network is covered only buys launches.
comptime PIPE_SPLIT_UNIT = 640_000


# ---------------------------------------------------------------------------
# Communicator state, behind the opaque `ncclComm_t` (void*) every exported
# function after ncclCommInitRank receives. Heap-allocated once per
# communicator and never freed on destroy (the struct itself is a few
# hundred bytes; only the GPU region, the peer IPC mappings and the IB
# resources -- the ones that matter -- are released there).
#
# `regions` is indexed by LOCAL rank and padded to MAX_WORLD with zeros:
# only same-node peers are IPC-mapped, and a node contributes at most
# MAX_WORLD ranks however large the communicator is.
# ---------------------------------------------------------------------------


struct CommState(Movable):
    var rank: Int
    var world: Int
    var ordinal: Int
    var ctx: DeviceContext
    var driver: OwnedDLHandle
    var cap_bytes: Int
    var regions: StaticTuple[Int, MAX_WORLD]
    var owned_base: Int
    var generation: Int
    var last_stream: Int64
    var aborted: Bool
    # Set once every resource below has been released (by destroy, or by an
    # abort that reached quiescence); makes the other one a no-op.
    var released: Bool
    # The status page: pinned host memory the device spins poll through
    # `install_status_page`'s slot in each arena header. `abort_host` is the
    # page's host address -- word 0 is what `ncclCommAbort` stores into, the
    # second cache line is the fault record a timed-out kernel latches --
    # and `abort_dev` is the same page as a kernel addresses it.
    var abort_host: Int
    var abort_dev: Int
    # Whether the latched fault has already been printed. The record is
    # permanent, and every later collective on the communicator reads it; the
    # message is worth one line, not one per call.
    var fault_said: Bool
    var stream_cache: Dict[Int64, DeviceStream]
    # Recorded on the stream after every collective; a collective issued on
    # another stream waits for it first (`_order_before` / `_order_after`).
    var order_event: CompletionEvent
    var order_recorded: Bool
    # Protected by the submission lock, including while a call is in progress.
    var order_incomplete: Bool
    # Atomic terminal flag: watchdogs may read it without the submission lock.
    var submission_failed: Int64
    var local_rank: Int
    var local_world: Int
    var my_node: Int
    var nnodes: Int
    var rank_at: List[Int]
    var ib: Int
    var net_off: Int
    var narenas: Int
    var arena_cap: Int
    var arena_stride: Int
    var nslots: Int
    var nvls: NvlsRegion
    var nvls_on: Bool
    var nvls_grid: Int
    var nvls_min: Int
    var nvls_bars: Int
    # SM/CU count of this rank's GPU: the fused inter-node kernel's grid has
    # to be co-resident, so it is one block per multiprocessor
    # (`fused_blocks`), and every rank of a node derives the same number.
    var sm_count: Int
    # Grid caps of the fused kernel (`fused_blocks`, and
    # `fused_big_blocks` for messages of at least `fused_big_bytes`),
    # checked equal on every rank at init: its barriers are matched by block
    # index.
    var fused_cap: Int
    var fused_big_cap: Int
    var fused_big_bytes: Int
    # Co-resident bound of the fused kernel on this GPU (occupancy times SMs,
    # `fused_resident_blocks`); the grid never exceeds it.
    var fused_resident: Int
    # `PIPE_SPLIT_UNIT`; checked equal on every rank at init.
    var split_unit: Int
    # Grid caps of a multi-node all-gather's gathers and reduce-scatter's
    # node reduce (`_node_grids`), the same on every rank.
    var ag_node_blocks: Int
    var rs_node_blocks: Int
    # Whether multi-node allreduces go through the one-launch fused kernel.
    # Requires the progress thread. A message exceeding the fused work-ring
    # capacity still takes the split schedule at launch time.
    var fused: Bool
    # `ncclCommGetAsyncError`'s scratch, built once instead of per poll: the
    # device word the copy kernel writes and the host word it lands in.
    # Reusing them avoids an allocation (and formerly a leak) per poll.
    var err_buf: DeviceBuffer[DType.uint64]
    var err_host: Int
    # Submission lock (`_lock`/`_unlock`): 0 free, 1 held.
    var lock: Int64

    def __init__(
        out self,
        rank: Int,
        world: Int,
        ordinal: Int,
        ctx: DeviceContext,
        var driver: OwnedDLHandle,
        cap_bytes: Int,
        regions: StaticTuple[Int, MAX_WORLD],
        owned_base: Int,
        local_rank: Int,
        local_world: Int,
        my_node: Int,
        nnodes: Int,
        var rank_at: List[Int],
        ib: Int,
        net_off: Int,
        narenas: Int,
        arena_cap: Int,
        arena_stride: Int,
        nslots: Int,
        var nvls: NvlsRegion,
        nvls_on: Bool,
        nvls_grid: Int,
        nvls_min: Int,
        sm_count: Int,
        fused_cap: Int,
        fused_big_cap: Int,
        fused_big_bytes: Int,
        fused_resident: Int,
        split_unit: Int,
        ag_node_blocks: Int,
        rs_node_blocks: Int,
        fused: Bool,
        abort_host: Int,
        abort_dev: Int,
    ) raises:
        self.rank = rank
        self.world = world
        self.ordinal = ordinal
        self.ctx = ctx
        self.driver = driver^
        self.cap_bytes = cap_bytes
        self.regions = regions
        self.owned_base = owned_base
        self.generation = 0
        self.last_stream = 0
        self.order_event = CompletionEvent(ctx)
        self.order_recorded = False
        self.order_incomplete = False
        self.submission_failed = 0
        self.aborted = False
        self.released = False
        self.abort_host = abort_host
        self.abort_dev = abort_dev
        self.fault_said = False
        self.stream_cache = Dict[Int64, DeviceStream]()
        self.local_rank = local_rank
        self.local_world = local_world
        self.my_node = my_node
        self.nnodes = nnodes
        self.rank_at = rank_at^
        self.ib = ib
        self.net_off = net_off
        self.narenas = narenas
        self.arena_cap = arena_cap
        self.arena_stride = arena_stride
        self.nslots = nslots
        self.nvls = nvls^
        self.nvls_on = nvls_on
        self.nvls_grid = nvls_grid
        self.nvls_min = nvls_min
        self.sm_count = sm_count
        self.fused_cap = fused_cap
        self.fused_big_cap = fused_big_cap
        self.fused_big_bytes = fused_big_bytes
        self.fused_resident = fused_resident
        self.split_unit = split_unit
        self.ag_node_blocks = ag_node_blocks
        self.rs_node_blocks = rs_node_blocks
        self.fused = fused
        # Barriers the NVLS kernel has completed on this region. Its flag is a
        # single UInt64 counter that every GPU adds 1 to per barrier, so the
        # value to wait for is `(nvls_bars + 1) * local_world`; it is never
        # reset, and it is independent of `generation` (different address,
        # different protocol) so the two paths can alternate freely.
        self.nvls_bars = 0
        self.err_buf = self.ctx.enqueue_create_buffer[DType.uint64](1)
        self.err_host = Int(unsafe_alloc[UInt64](1))
        self.lock = 0


def _lock(mut state: CommState):
    """Serialize whole-collective submission on one communicator.

    A collective is several launches plus host bookkeeping (generations,
    exchange numbers, work descriptors, arena choice). One stream orders the
    launches, not the sequence: torch can issue collectives from more than
    one host thread (DDP's reducer runs on the autograd thread while the main
    thread may broadcast) and ctypes releases the GIL around the call, so two
    submissions could interleave -- thread A's reduce-scatter, then thread
    B's on the same arena before A released its exchange. NCCL declares
    concurrent calls on one communicator unsupported; a spin word costs ~20 ns
    uncontended and turns that into a serialization. Watchdogs use bounded
    or nonblocking attempts: a submitter may be stuck behind a full queue.
    """
    var p = Pointer(to=state.lock).unsafe_origin_cast[MutAnyOrigin]()
    while True:
        var expected: Int64 = 0
        if Atomic[Scalar[DType.int64]].compare_exchange[
            success_ordering=Ordering.ACQUIRE,
            failure_ordering=Ordering.RELAXED,
        ](p, expected, 1):
            return
        _ = external_call["sched_yield", Int32]()


def _try_lock(mut state: CommState, deadline_ns: Int) -> Bool:
    """`_lock` with a deadline, for `ncclCommAbort`.

    Abort must never block behind a submitter, but it does have to own the
    communicator before it frees anything under one. The submitter it can be
    waiting for is running host code only -- a few launches -- and the abort
    word has already released whatever the GPU was spinning on, so this
    normally takes microseconds; the deadline is what keeps the promise if it
    does not.
    """
    var p = Pointer(to=state.lock).unsafe_origin_cast[MutAnyOrigin]()
    while True:
        var expected: Int64 = 0
        if Atomic[Scalar[DType.int64]].compare_exchange[
            success_ordering=Ordering.ACQUIRE,
            failure_ordering=Ordering.RELAXED,
        ](p, expected, 1):
            return True
        if perf_counter_ns() > deadline_ns:
            return False
        _ = external_call["sched_yield", Int32]()


def _unlock(mut state: CommState):
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        Pointer(to=state.lock).unsafe_origin_cast[MutAnyOrigin](), 0
    )


@always_inline
def _raise_abort_word(state: CommState):
    """Store 1 into the pinned abort word: every device spin leaves at its
    next check, and no driver call is needed to do it."""
    if state.abort_host == 0:
        return
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=state.abort_host),
        UInt64(1),
    )


@always_inline
def _align_up(x: Int, a: Int) -> Int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Region geometry -- pure integer arithmetic, no communicator state, so
# tests/multinode/selftest/geometry_test.mojo can sweep it over every region
# size, local_world and node count this library claims to support. Every rank
# derives the same numbers from `(cap_bytes, nnodes)` alone, which is what
# makes a sender's address arithmetic agree with its receiver's.
# ---------------------------------------------------------------------------


def region_layout(cap_bytes: Int, nnodes: Int) -> Tuple[Int, Int, Int, Int]:
    """`(narenas, arena_cap, arena_stride, region_bytes)`.

    Single node: one arena of `cap_bytes` halves and no network area -- the
    layout this library had before it learned about nodes, byte for byte.
    Multi-node: `PIPE_ARENAS` arenas of `cap/PIPE_ARENAS` halves (so the
    staging total is unchanged) plus one cap-sized network area.
    """
    var narenas = PIPE_ARENAS if nnodes > 1 else 1
    var arena_cap = cap_bytes // narenas // 4096 * 4096
    var arena_stride = signal_bytes() + 2 * arena_cap
    var net_bytes = cap_bytes if nnodes > 1 else 0
    return Tuple(
        narenas, arena_cap, arena_stride, narenas * arena_stride + net_bytes
    )


def net_stage_bytes(cap_bytes: Int) -> Int:
    """Staging bytes broadcast and allgather may use in the network area."""
    return cap_bytes // 2 - CREDIT_AREA_BYTES


def inbox_group_bytes(cap_bytes: Int, nslots: Int) -> Int:
    """Bytes of one of the `nslots` fixed inbox slot groups."""
    return (cap_bytes // 2) // nslots // 4096 * 4096


def max_chunk_bytes(
    cap_bytes: Int, arena_cap: Int, nslots: Int, local_world: Int, nnodes: Int
) -> Int:
    """Largest allreduce chunk whose inbox slot group fits.

    A chunk of `B` bytes puts `B/L` bytes in each of the `N-1` slots of one
    group, so `B <= group * L / (N-1)`, less a page of alignment slop (the
    shard size is rounded up to a 16-byte vector per rank) -- and never more
    than one arena's `arena_cap`, which the intra-node kernels require. With
    the 256 MiB default region and 4 arenas that is 64 MiB at 2 nodes and
    36 MiB at 8; the pipeline split usually asks for less.
    """
    var b = (
        inbox_group_bytes(cap_bytes, nslots) * local_world // (nnodes - 1)
    ) - 2 * 4096
    b = min(b, arena_cap)
    if b < 4096:
        return 4096
    return b // 4096 * 4096


def pipeline_chunk_bytes(
    max_chunk: Int, local_world: Int, total_bytes: Int, split_unit: Int
) -> Int:
    """Chunk size of a pipelined multi-node allreduce; see PIPE_SPLIT_UNIT
    for where the square root comes from."""
    var unit = local_world * split_unit
    var k = 1
    while k < PIPE_MAX_CHUNKS and (k + 1) * (k + 1) * unit <= total_bytes:
        k += 1
    var chunk = _align_up((total_bytes + k - 1) // k, 4096)
    chunk = min(chunk, max_chunk)
    return max(chunk, 4096)


@always_inline
def _comm_ptr(comm: Int64) -> Pointer[CommState, MutAnyOrigin]:
    return Pointer[CommState, MutAnyOrigin](unsafe_from_address=Int(comm))


def _fail_submission(mut state: CommState):
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        Pointer(to=state.submission_failed).unsafe_origin_cast[MutAnyOrigin](),
        1,
    )


def _submission_exception_code(mut state: CommState) -> Int32:
    """Preserve a transport failure raised while the host enqueued work.

    A split collective can fill the work ring before returning to Python.
    Its ring wait then raises on a peer deadline. Classify that transport
    error before marking the submission failed: the marker makes *later*
    calls remote errors, but an unrelated launch exception must still be
    INTERNAL on its first reporting call.
    """
    var rc = NCCL_INTERNAL_ERROR
    if state.ib != 0 and ib_error(state.ib) != 0:
        rc = NCCL_REMOTE_ERROR
    _fail_submission(state)
    return rc


def _submission_failed(state: CommState) -> Bool:
    return (
        Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](
            Pointer(to=state.submission_failed).unsafe_origin_cast[
                MutAnyOrigin
            ]()
        )
        != 0
    )


def _copy_error_word(
    dst: Pointer[UInt64, MutAnyOrigin], src: Pointer[UInt64, MutAnyOrigin]
):
    """One-thread device kernel: copies the error word out of a region's
    signal area -- raw driver memory (`cuMemAlloc`/`hipExtMallocWithFlags`),
    not host-accessible -- into a proper `DeviceBuffer` `enqueue_copy` can
    D2H-copy from. Mirrors the production kernel harness's `_copy_u64`
    (kernel/harness.mojo, `check_error`): the error word cannot be read by
    dereferencing a host pointer at the region address, that faults or reads
    garbage.
    """
    if global_idx.x == 0:
        dst[unsafe_offset=0] = src[unsafe_offset=0]


def _read_error_word(mut state: CommState, region: Int) raises -> UInt64:
    """The UInt64 error word at `region + error_offset()`, fetched to the
    host via a real device-to-host copy (see `_copy_error_word`).

    Kernel and D2H copy both run on the context's own stream, so the copy is
    ordered after the read that fills the buffer. Putting the kernel on the
    caller's stream and the copy on the context's -- which is what
    `enqueue_copy` uses -- left them on two streams with nothing between
    them, and the copy could win. Nothing is lost by not touching the
    caller's stream: the caller has already synchronized it, which is what
    makes the word final (`ncclCommGetAsyncError` does exactly that).
    Kernel, device word and host word are all cached on the communicator.
    """
    var src = Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=region + error_offset()
    )
    _enqueue_cached[_copy_error_word](
        state.ctx,
        state.ctx.stream(),
        "errword",
        1,
        state.err_buf.unsafe_ptr(),
        src,
    )
    var host_word = Pointer[UInt64, MutUntrackedOrigin](
        unsafe_from_address=state.err_host
    )
    state.ctx.enqueue_copy(host_word, state.err_buf)
    state.ctx.synchronize()
    return host_word[unsafe_offset=0]


# ---------------------------------------------------------------------------
# The latched device fault. A kernel that gives up on a peer records it in the
# communicator's pinned status page (`publish_fault`,
# device/common.mojo); this is the host half -- read it for the price of
# a load, say what happened once, and fail every later collective.
# ---------------------------------------------------------------------------


@always_inline
def _fault_field(state: CommState, index: Int) -> UInt64:
    """One word of the fault record, by its `FAULT_*` index."""
    return Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=state.abort_host
            + (STATUS_FAULT_WORD + index) * 8
        )
    )


@always_inline
def _fault_code(state: CommState) -> UInt64:
    """The code of this communicator's latched device deadline, 0 if none.

    One acquire load of pinned host memory: no stream synchronized, no device
    memory copied, no lock taken -- which is the whole reason the record lives
    there. `ncclCommGetAsyncError`'s D2H read of the arena error word blocks
    behind the very kernels it is asking about (`_read_error_word`), so no
    collective could afford to call it, so nothing ever read the only record a
    timed-out barrier used to leave, and the run continued on garbage.
    """
    if state.abort_host == 0:
        return UInt64(0)
    return _fault_field(state, FAULT_CODE)


@always_inline
def _host_fault_word(state: CommState) -> UInt64:
    """`STATUS_HOST_FAULT_WORD`: the host's own latched failure, 0 if none."""
    if state.abort_host == 0:
        return UInt64(0)
    return Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=state.abort_host + STATUS_HOST_FAULT_WORD * 8
        )
    )


def _latch_host_fault_record(page: Int, code: Int, detail: Int):
    """Latch a separate host record under the communicator's submission lock."""
    if page == 0:
        return
    var host = Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=page + STATUS_HOST_FAULT_WORD * 8
    )
    if Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](host) != 0:
        return
    # First fully published fault observed here wins. A device record still
    # being published loses to this host fault; their detail words are disjoint.
    var device = Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=page + (STATUS_FAULT_WORD + FAULT_CODE) * 8
        )
    )
    var device_first = UInt64(device != 0) << 63
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        host,
        device_first
        | (UInt64(code) << 32)
        | (UInt64(detail) & UInt64(0xFFFF_FFFF)),
    )


def _latch_host_fault(state: CommState, code: Int, detail: Int):
    """Publish only to the host record, then release this rank's device spins.
    """
    _latch_host_fault_record(state.abort_host, code, detail)
    _raise_abort_word(state)


def _fault_kind(code: UInt64) -> String:
    """The collective a fault code names, for the message."""
    var c = Int(code)
    if c == ERR_ALLREDUCE_SYNC:
        return String("the allreduce")
    if c == ERR_BROADCAST_SYNC:
        return String("the broadcast")
    if c == ERR_ALLGATHER_SYNC:
        return String("the allgather")
    if c == ERR_REDUCE_SCATTER_SYNC:
        return String("the reduce-scatter")
    if c == ERR_RS_STAGE_SYNC:
        return String("the multi-node allreduce's reduce-scatter stage")
    if c == ERR_AG_FINISH_SYNC:
        return String("the multi-node allreduce's allgather stage")
    if c == ERR_NVLS_SYNC:
        return String("the NVLS allreduce")
    if c == ERR_PROXY_WAIT:
        return String("the inter-node exchange wait")
    if c == ERR_FUSED_GRID:
        return String("the multi-node allreduce's grid barrier")
    if c == ERR_HOST_LAUNCH:
        return String("the host's launch of the multi-node allreduce")
    return String("an unknown collective (code " + String(c) + ")")


def _report_fault(mut state: CommState):
    """Print the latched deadline, once per communicator.

    Once, because the record is permanent and every later collective reads it:
    the reader wants one line saying which barrier gave up and what it was
    waiting for, not one per call for the rest of the run. Everything printed
    comes from the record itself except the rank, which the process knows, and
    the deadline, which is this process's own `MOJOCCL_IB_TIMEOUT_S`.
    """
    if state.fault_said:
        return
    var code = _fault_code(state)
    var host = _host_fault_word(state)
    # Bit 63 snapshots precedence at host publication; a later device
    # fault must not replace the host fault before the first report.
    if host != 0 and host >> 63 == 0:
        state.fault_said = True
        print(
            "mojoccl: rank",
            state.rank,
            ": HOST FAULT in",
            _fault_kind(host >> 32),
            "at exchange",
            host & UInt64(0xFFFF_FFFF),
            (
                "-- the kernel launch failed after its exchanges were reserved,"
                " so this rank's peers will report a deadline waiting for it;"
                " every later collective on this communicator fails with"
                " ncclRemoteError."
            ),
        )
        return
    if code == 0:
        return
    state.fault_said = True
    var phase = _fault_field(state, FAULT_PHASE)
    var block = _fault_field(state, FAULT_BLOCK)
    var peer = _fault_field(state, FAULT_PEER)
    var seen = _fault_field(state, FAULT_SEEN)
    var target = _fault_field(state, FAULT_TARGET)
    var arena_base = Int(_fault_field(state, FAULT_ARENA))
    var mine = state.regions[state.local_rank]
    var arena = 0
    if state.arena_stride > 0 and arena_base >= mine:
        arena = (arena_base - mine) // state.arena_stride
    var secs = Float64(spin_timeout_ns()) / 1.0e9
    var what = String("mojoccl: rank ") + String(state.rank)
    what += String(": DEVICE DEADLINE in ") + _fault_kind(code)
    if Int(code) == ERR_PROXY_WAIT:
        what += String(" (arena ") + String(arena) + String("): waited ")
        what += String(secs) + String(" s for this rank's progress thread to")
        what += String(" retire exchange ") + String(target)
        what += String(", and it had retired ") + String(seen)
    elif Int(code) == ERR_FUSED_GRID:
        # Rank-local: the other blocks of the same kernel never arrived,
        # which only happens after another spin in it gave up.
        what += String(": block ") + String(block) + String(" waited ")
        what += String(secs)
        what += String(" s for the other blocks of its own kernel")
    elif peer == FAULT_NO_PEER:
        # The NVLS barrier is a multicast counter, not a per-peer flag.
        what += String(" (arena ") + String(arena) + String(", phase ")
        what += String(phase) + String("): block ") + String(block)
        what += String(" waited ") + String(secs)
        what += String(" s for the multicast barrier to reach ")
        what += String(target)
    else:
        what += String(" (arena ") + String(arena) + String(", generation ")
        what += String(target // UInt64(PHASES_PER_GEN))
        what += String(", phase ") + String(phase) + String("): block ")
        what += String(block) + String(" waited ") + String(secs)
        what += String(" s for rank ") + String(peer) + String("'s flag")
        what += String(", and saw ") + String(seen) + String(" wanting ")
        what += String(target)
    what += String(
        " -- the result of that collective is undefined, so every later"
        " collective on this communicator fails with ncclRemoteError. The peer"
        " it waited for either died, timed out itself, or never reached the"
        " same collective."
    )
    print(what)


def _latched_error(mut state: CommState) -> Int32:
    """NCCL_SUCCESS, or the error this communicator has already suffered.

    Called at the head of every collective. `ib_error` is the transport's own
    give-up (host state, no read at all); the fault word is the device's, and
    checking it is what turned a timed-out barrier from a silently wrong
    result into a failed call. Neither check touches a stream.
    """
    if _submission_failed(state):
        return NCCL_REMOTE_ERROR
    if state.ib != 0 and ib_error(state.ib) != 0:
        return NCCL_REMOTE_ERROR
    if _fault_code(state) != 0 or _host_fault_word(state) != 0:
        _report_fault(state)
        return NCCL_REMOTE_ERROR
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------


def _arena_regions(state: CommState, arena: Int) -> StaticTuple[Int, MAX_WORLD]:
    """Peer bases of one pipeline arena.

    Each arena is a complete region as far as device/symmetric/ is
    concerned -- its own signal area at offset 0, its own [stage_in |
    stage_out] of `arena_cap` bytes each -- so concurrent pipeline chunks are
    kept apart by handing the split kernels a shifted base and a smaller cap,
    with no change to their device code and no new parameter. Only the
    `local_world` entries a node actually has are shifted; the rest stay 0 and
    are never read (`_region_ptrs` fills them with this rank's own base).
    """
    var out = StaticTuple[Int, MAX_WORLD](fill=0)
    for r in range(state.local_world):
        out[r] = state.regions[r] + arena * state.arena_stride
    return out


def _arena_shard(state: CommState, arena: Int, byte_off: Int) -> Int:
    """Address of this rank's shard inside one arena's stage_out."""
    return (
        state.owned_base
        + arena * state.arena_stride
        + signal_bytes()
        + state.arena_cap
        + byte_off
    )


def _net_stage_off(state: CommState) -> Int:
    """Region offset of the staging half broadcast and allgather copy
    through (user buffers are not registered, so the NIC cannot read them)."""
    return state.net_off + CREDIT_AREA_BYTES


def _net_stage_bytes(state: CommState) -> Int:
    return net_stage_bytes(state.cap_bytes)


def _inbox_group_bytes(state: CommState) -> Int:
    """Bytes of one inbox slot group. See `_inbox_base`."""
    return inbox_group_bytes(state.cap_bytes, state.nslots)


def _inbox_base(state: CommState, seq: Int) -> Int:
    """Region offset of exchange `seq`'s inbox slot group.

    FIXED, not derived from the message: the second half of the network area
    is carved once into `nslots` equal groups and `seq % nslots` picks one.

    That the groups do not move is what makes reuse a matter of one credit
    rather than of two messages happening to be the same shape. Sizing a
    group from the current message -- which an early version did, with two
    parity halves -- kept e and e+nslots apart but let e and e+1 OVERLAP
    whenever consecutive exchanges had different geometry, and DDP produces
    exactly that: a 4-byte AVG allreduce (slot 16 B) next to a 27 MiB bucket
    put the small exchange's second half at `net_off+16` and the big one's
    first at `net_off+0`. Nothing orders a peer's e+1 write against my e
    consumer -- the peer's e+1 send is gated on its own credits, not on my
    consumption -- so the overlap is a silent data race on the inbox, and
    mixing collectives (an allreduce's group against a broadcast's staged
    chunk) is the same bug once more.
    """
    return (
        state.net_off
        + state.cap_bytes // 2
        + (seq % state.nslots) * _inbox_group_bytes(state)
    )


def _max_chunk_bytes(state: CommState) -> Int:
    if state.nnodes <= 1:
        return state.cap_bytes
    return max_chunk_bytes(
        state.cap_bytes,
        state.arena_cap,
        state.nslots,
        state.local_world,
        state.nnodes,
    )


def _pipeline_chunk_bytes(state: CommState, total_bytes: Int) -> Int:
    return pipeline_chunk_bytes(
        _max_chunk_bytes(state),
        state.local_world,
        total_bytes,
        state.split_unit,
    )
