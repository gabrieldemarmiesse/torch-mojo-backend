# mojoccl: a Mojo shared library exporting NCCL's C ABI (nccl.h), so
# torch_mojo_backend/distributed/nccl.py can dlopen it exactly like
# libnccl.so.2/librccl.so.1 (TORCH_MOJO_BACKEND_CCL=mojo), and so external
# tools (nccl-tests) can link it as a libnccl.so drop-in.
#
# Signatures, enum values and ncclResult_t codes are pinned to
# /home/gabriel/projects/nccl/src/nccl.h.in (2.31.2) -- the source of truth
# is nccl.py's `_declare()`, which this library's exports were written
# against. AllReduce/Broadcast/AllGather/ReduceScatter are real; Reduce/
# Send/Recv return ncclInvalidUsage (GPT-2 DDP needs only the first three;
# tests/ddp_worker.py skips the checks that need them when
# TORCH_MOJO_BACKEND_CCL=mojo is set).
#
# One process per GPU. Within a node, one region per rank of raw
# driver-owned memory -- MAX's own allocator memory cannot be shared across
# processes at all (see agents_docs/mojo_collectives_feasibility.md, section 5.6) --
# and the collectives of collectives_kernels.mojo run over the peer mappings.
# Across nodes, GPUDirect RDMA written here over libibverbs (ibverbs.mojo,
# internode.mojo): no vendor collective library takes part at any level.
#
# That region comes in two shapes, chosen once per communicator in round 1 of
# the bootstrap and identical on every rank:
#
#   default   `cuMemAlloc` / `hipExtMallocWithFlags`, shared with legacy IPC
#             (driver.mojo). Every node count, every vendor.
#   NVLS      single node, every local device reporting
#             CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED: VMM memory bound to a
#             per-node multicast object and mapped twice (vmm.mojo), so that
#             allreduces of NVLS_MIN_BYTES or more can go through the
#             NVSwitch's own reduction engine (nvls_kernels.mojo) instead of
#             over unicast NVLink. Peers are imported through the same
#             file-descriptor exchange rather than with cuIpcOpenMemHandle,
#             which cannot open VMM memory. The unicast kernels see no
#             difference: same layout, same offsets, the plain mapping.
#
# A multi-node communicator runs each collective hierarchically:
#
#   allreduce  reduce_scatter_stage (node-local)  ->  one RDMA exchange of
#              this rank's 1/local_world shard with the SAME local_rank on
#              every other node, summed by inbox_add  ->  allgather_finish
#   broadcast  root stages and RDMA-writes to its same-local_rank peers,
#              then every node runs the node-local broadcast from its own
#              local root
#   allgather  node-local allgather, RDMA exchange of each rank's contribution,
#              then node-local dissemination of the remote contributions.
#              Place by GLOBAL rank from the bootstrap topology table.
#              NVIDIA gathers directly into the mapped output slots.
#
# Reduce-scatter reduces all node destinations locally, exchanges each
# remote destination with its owner, and sums into the user output. It uses
# the same pipeline arenas and inbox credits as allreduce. Allgather also
# runs during FSDP2 training; NVIDIA pipelines large gathers over two arenas.
# Allreduce is PIPELINED: the bucket is cut into
# K chunks and issued on the one comm stream as
#
#   RS(0) release(0) RS(1) release(1) ... wait(0) add(0) AG(0) RS(3) ...
#
# with at most `PIPE_ARENAS` chunks alive, so the proxy exchanges chunk k
# while the GPU reduce-scatters later chunks and all-gathers earlier ones.
# Two things make that safe and neither is stream order across ranks: the
# staging arena is replicated (`_arena_regions`), and the inbox is reused
# only against an explicit credit (internode.mojo's header).
#
# Single-node communicators never touch libibverbs at all -- same fused
# kernels, same numbers as before this file learned about nodes.

from std.atomic import Atomic, Ordering
from std.collections import Dict
from std.ffi import OwnedDLHandle, external_call
from std.gpu import global_idx
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.sys import has_nvidia_gpu_accelerator, size_of
from std.time import perf_counter_ns, sleep
from std.utils import StaticTuple
from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    DeviceStream,
)

from tmb.ccl.env_vars import (
    MOJOCCL_BOOTSTRAP_TIMEOUT_S,
    MOJOCCL_NVLS,
    MOJOCCL_REGION_MB,
)
from tmb.ccl.driver import (
    HANDLE_BYTES,
    CompletionEvent,
    alloc_host,
    alloc_region,
    close_handle,
    current_device_ordinal,
    free_host,
    free_region,
    get_handle,
    host_device_ptr,
    open_driver,
    open_handle,
    stream_done,
)
from tmb.ccl.bootstrap import (
    UID_BYTES,
    BootstrapConn,
    bootstrap_allgather,
    bootstrap_barrier,
    bootstrap_connect,
    decode_id,
    derive_topology,
    host_hash,
    make_unique_id,
)
from tmb.ccl.collectives_kernels import (
    ERR_ALLGATHER_SYNC,
    ERR_ALLREDUCE_SYNC,
    ERR_AG_FINISH_SYNC,
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
    STATUS_PAGE_BYTES,
    _shard_per,
    allgather,
    allgather_max_bytes,
    allgather_mapped,
    allgather_finish,
    allreduce,
    broadcast,
    _enqueue_cached,
    error_offset,
    install_status_page,
    region_init,
    reduce_scatter,
    rank_gate,
    reduce_scatter_max_count,
    reduce_scatter_stage,
    shard_range,
    signal_bytes,
    spin_timeout_ns,
)
from tmb.ccl.internode import (
    CREDIT_AREA_BYTES,
    EMPTY_SHARD_BYTES,
    IB_BLOB_BYTES,
    MAX_NODES,
    WORK_SLOTS,
    OP_ALLGATHER,
    OP_ALLREDUCE,
    OP_REDUCE_SCATTER,
    OP_BROADCAST,
    PIPE_MAX_SLOTS,
    ib_connect,
    ib_enqueue_request,
    ib_enqueue_wait,
    ib_error,
    ib_local_info,
    ib_mailbox_dev,
    ib_next_seq,
    ib_note_consumed,
    ib_npeers,
    ib_prepare_request,
    ib_reserve_seqs,
    ib_timeout_ns,
    ib_uses_proxy,
    ib_set_abort_word,
    ib_setup,
    ib_signal_abort,
    ib_teardown,
)
from tmb.ccl.internode_fused import (
    _MI300A,
    FUSED_THREADS,
    check_fused_call,
    fused_big_block_cap,
    fused_big_bytes,
    fused_block_cap,
    fused_blocks,
    fused_resident_blocks,
    internode_allreduce_fused,
)
from tmb.ccl.internode_kernels import (
    copy_bytes,
    inbox_add,
    inbox_sum_out,
    place_blocks,
)
from tmb.ccl.reduce_scatter.fused import (
    RS_FUSED_BIG_BLOCKS,
    reduce_scatter_fused,
    reduce_scatter_fused_blocks,
    reduce_scatter_fused_plan,
)
from tmb.ccl.reduce_scatter.multinode import (
    reduce_scatter_nodes,
    reduce_scatter_nodes_max_count,
    reduce_scatter_rank_ids,
)
from tmb.ccl.reduce_scatter.stream import (
    RS_STREAM_BIG_BLOCKS,
    RS_STREAM_ENABLED,
    reduce_scatter_stream,
    reduce_scatter_stream_blocks,
    reduce_scatter_stream_generations,
    reduce_scatter_stream_pieces,
    reduce_scatter_stream_plan,
    reduce_scatter_stream_wanted,
)
from tmb.ccl.nvls_kernels import (
    ERR_NVLS_SYNC,
    nvls_allreduce,
    nvls_available,
    nvls_barriers_per_call,
    nvls_blocks,
    nvls_chunk_bytes,
    nvls_chunk_vecs,
    nvls_min_bytes,
)
from tmb.ccl.vmm import (
    NvlsRegion,
    multicast_capable,
    multicast_granularity,
    nvls_bind_and_map,
    nvls_create_and_share,
    nvls_teardown,
    scm_bind,
    scm_unbind,
    sm_count,
    socket_path,
)


# ncclResult_t (nccl.h.in:44-53)
comptime NCCL_SUCCESS: Int32 = 0
comptime NCCL_UNHANDLED_CUDA_ERROR: Int32 = 1
comptime NCCL_SYSTEM_ERROR: Int32 = 2
comptime NCCL_INTERNAL_ERROR: Int32 = 3
comptime NCCL_INVALID_ARGUMENT: Int32 = 4
comptime NCCL_INVALID_USAGE: Int32 = 5
comptime NCCL_REMOTE_ERROR: Int32 = 6
comptime NCCL_IN_PROGRESS: Int32 = 7

# ncclDataType_t (nccl.h.in:466-479). AllReduce runs a real kernel and only
# instantiates the five below; Broadcast/AllGather move raw bytes (item size
# x count -> nbytes) and accept every type in the enum.
comptime NCCL_INT8: Int32 = 0
comptime NCCL_UINT8: Int32 = 1
comptime NCCL_INT32: Int32 = 2
comptime NCCL_UINT32: Int32 = 3
comptime NCCL_INT64: Int32 = 4
comptime NCCL_UINT64: Int32 = 5
comptime NCCL_FLOAT16: Int32 = 6
comptime NCCL_FLOAT32: Int32 = 7
comptime NCCL_FLOAT64: Int32 = 8
comptime NCCL_BFLOAT16: Int32 = 9

# ncclRedOp_t (nccl.h.in:448-463) -- values this library implements.
comptime NCCL_SUM: Int32 = 0
comptime NCCL_AVG: Int32 = 4

# Staging arenas the multi-node region is carved into, and therefore chunks
# the pipeline may keep alive (`_arena_regions`, `_do_allreduce`).
#
# 4, not 2: the pipeline is GPU-bound at every size that gets chunked, so
# depth 2 overlaps in principle, but its window is one reduce-scatter and
# the progress thread's idle backoff alone (20 us)
# can eat that. Depth 4 gives a window of two whole chunks, and costs region
# layout rather than memory -- each arena is 1/4 of the staging.
comptime PIPE_ARENAS = 4

# Fixed inbox slot groups (internode.mojo owns what they are for). One more
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

# Part of the wire layout in the same
# way `MOJOCCL_REGION_MB` is -- K decides how many exchange counters a
# collective consumes, so two ranks that disagree about it stop agreeing about
# which exchange is which -- so `ncclCommInitRank` checks it matches.
#
# Re-measured for the fused kernel (2x8 H100, GPT-2 XL, job 250995, mean
# tok/s of steps 10-20 over two passes, vs mojo+NCCL in the same job), where
# an extra chunk costs two grid barriers and two 8-way start barriers rather
# than launches: 640_000 (K=2 at the 39 MiB bucket) 0.982, 320_000 (K=3)
# 0.975, 160_000 (K=4) 0.971. The rule stands.


def _pipe_split_unit() -> Int:
    return PIPE_SPLIT_UNIT


comptime AG_NODE_BLOCKS = 96
"""Grid cap of the node-local gathers of a multi-node all-gather, in place
of the single-node copy cap (432). The gathers run under the forward's and
backward's GEMMs, so the cap is fitted end to end, not on the isolated
collective: GPT-2 XL FSDP2 on 2x8 H100, mojo+mojoccl tok/s at 32
reduce-scatter CTAs (CUDA+NCCL 70.2k): 32 -> 66.5k, 64 -> 66.5-67.0k,
96 -> 67.3-67.6k, 128 -> 65.3-66.7k; 432 with 128 reduce-scatter CTAs
62.4k. Isolated, 64 blocks still beat NCCL (block bf16 0.92x, root fp32
0.94x). NCCL's 16 CTAs are not the answer for a pull: 16 blocks x 16
vectors in flight measures the same isolated time as 96 x 4 on the XL sizes
(block 360 vs 348 us, root 1031 vs 1027) yet 63.1-64.8k tok/s end to end
against 66.1-69.1k -- a latency-bound pull under the compute stream's HBM
traffic loses far more from 6x fewer CTAs than the GEMMs gain from the
freed SMs, and the compute stream waits on this gather."""
comptime AG_NODE_UNROLL = 4
"""16-byte vectors in flight per thread in those gathers; 8 measured
66.1k tok/s against 67.3-67.6k at 96 blocks."""


# MI300A: 64 MiB supports four ranks/node without the large shared-memory
# reservation conflict (Adastra 124M measurements in agents_docs/distributed.md),
# and the GPT-2 XL five-round series at 0.9873x stock used it, job 5417296,
# 2026-09-15. NVIDIA retains its H100 staging fit of 256 MiB.
comptime DEFAULT_REGION_MB = 64 if _MI300A else 256
comptime DEFAULT_BOOTSTRAP_TIMEOUT_S: Float64 = 120.0

comptime DEFAULT_SOCKET_DIR = "/tmp"
"""Where the node-local AF_UNIX sockets that carry the VMM/multicast file
descriptors are bound. Node-local by definition -- a
shared filesystem would work too but buys nothing, since the ranks that talk
over it are on one host. NCCL puts its own at /tmp as well
(nccl:src/os/linux_ipcsocket.cc)."""

comptime ABORT_QUIESCE_TIMEOUT_S: Float64 = 5.0
"""How long `ncclCommAbort` polls for local work to stop before it gives up on
reclaiming the region and the IB state.

Deliberately not `MOJOCCL_IB_TIMEOUT_S`: raising the abort word releases every
device spin within about a millisecond, so this bounds only the tail of what
was already running -- a large copy kernel, a launch queue draining -- and
abort's contract is not to wait. Past it the resources are left allocated,
which is a leak until the process exits and is the safe half of the trade:
freeing a region a kernel may still be reading is a fault.
"""

comptime NVLS_FD_TIMEOUT_S: Float64 = 30.0
"""Bound on one fd hand-off. The whole bring-up is 150-230 ms when it works,
so 30 s only ever fires when a local peer died mid-init -- and it has to fire,
or the surviving ranks block in `recvmsg` forever."""


def _region_cap_bytes() -> Int:
    var s = getenv(MOJOCCL_REGION_MB, String(DEFAULT_REGION_MB))
    try:
        return Int(s) * 1024 * 1024
    except:
        return DEFAULT_REGION_MB * 1024 * 1024


def _bootstrap_timeout_s() -> Float64:
    var s = getenv(
        MOJOCCL_BOOTSTRAP_TIMEOUT_S, String(DEFAULT_BOOTSTRAP_TIMEOUT_S)
    )
    try:
        return Float64(s)
    except:
        return DEFAULT_BOOTSTRAP_TIMEOUT_S


def _fused_enabled() -> Bool:
    """Prefer fused allreduce whenever the region can hold its pipeline.

    Fitted on H100 (job 250904: 487.6k vs split 449.1k tokens/s) and
    MI300A (job 5417296: 128400.0 vs split 99759.1, 2026-09-15).
    The effective schedule is exchanged and checked at init because its
    intra-node barriers are matched by block index.
    """
    return True


def _nvls_enabled() -> Bool:
    """`MOJOCCL_NVLS=0` turns the multicast path off, region and all.

    Folded into the round-1 capability word, so one rank setting it takes the
    whole communicator back to the unicast kernels -- a communicator where
    some ranks bound a multicast region and others did not is not a
    configuration, it is a crash.
    """
    return getenv(MOJOCCL_NVLS, String("1")) != String("0")


def _nvls_min_bytes() -> Int:
    """Message size at or above which a single-node allreduce goes through the
    switch (48 MiB -- the measured crossover,
    see `NVLS_MIN_BYTES` in nvls_kernels.mojo)."""
    return nvls_min_bytes()


def _nvls_recommended_granularity() -> Bool:
    """Use MINIMUM multicast granularity: it allocates what the region asks
    for. MINIMUM and RECOMMENDED measured the same on H100; see
    agents_docs/distributed.md's NVLS measurements."""
    return False


def _socket_dir() -> String:
    return String(DEFAULT_SOCKET_DIR)


def _dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for the five dtypes AllReduce's kernel is instantiated for."""
    if nccl_dtype == NCCL_INT32 or nccl_dtype == NCCL_FLOAT32:
        return 4
    elif nccl_dtype == NCCL_INT64:
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0


def _any_dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for every ncclDataType_t -- Broadcast/AllGather move raw
    bytes and never look at the dtype beyond this."""
    if nccl_dtype == NCCL_INT8 or nccl_dtype == NCCL_UINT8:
        return 1
    elif (
        nccl_dtype == NCCL_INT32
        or nccl_dtype == NCCL_UINT32
        or nccl_dtype == NCCL_FLOAT32
    ):
        return 4
    elif (
        nccl_dtype == NCCL_INT64
        or nccl_dtype == NCCL_UINT64
        or nccl_dtype == NCCL_FLOAT64
    ):
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0


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
        if Atomic[DType.int64].compare_exchange[
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
        if Atomic[DType.int64].compare_exchange[
            success_ordering=Ordering.ACQUIRE,
            failure_ordering=Ordering.RELAXED,
        ](p, expected, 1):
            return True
        if perf_counter_ns() > deadline_ns:
            return False
        _ = external_call["sched_yield", Int32]()


def _unlock(mut state: CommState):
    Atomic[DType.int64].store[ordering=Ordering.RELEASE](
        Pointer(to=state.lock).unsafe_origin_cast[MutAnyOrigin](), 0
    )


@always_inline
def _raise_abort_word(state: CommState):
    """Store 1 into the pinned abort word: every device spin leaves at its
    next check, and no driver call is needed to do it."""
    if state.abort_host == 0:
        return
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
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


@always_inline
def _any(p: Pointer[UInt8, MutUntrackedOrigin]) -> Pointer[UInt8, MutAnyOrigin]:
    """Rebinds an `unsafe_alloc`-returned pointer's origin to `MutAnyOrigin`,
    matching what driver.mojo/bootstrap.mojo (and the exported ABI functions
    they share signatures with) declare."""
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p))


def _order_before(mut state: CommState, handle: Int64) raises:
    """Wait for the previous collective, including one on default stream 0.

    Record before returning to the caller: a later call cannot touch the
    previous stream, which the caller may have destroyed in the meantime.
    The event orders reuse of the communicator's shared barrier words.
    """
    _ensure_stream_cached(state, handle)
    if state.order_recorded and state.last_stream != handle:
        state.order_event.wait_on(handle)
    state.last_stream = handle
    state.order_incomplete = True


def _order_after(mut state: CommState, handle: Int64) raises:
    """Record completion before returning; one driver event record per call."""
    state.order_event.record(handle)
    state.order_recorded = True
    state.order_incomplete = False


def _fail_submission(mut state: CommState):
    Atomic[DType.int64].store[ordering=Ordering.RELEASE](
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
        Atomic[DType.int64].load[ordering=Ordering.ACQUIRE](
            Pointer(to=state.submission_failed).unsafe_origin_cast[
                MutAnyOrigin
            ]()
        )
        != 0
    )


def _ensure_stream_cached(mut state: CommState, handle: Int64) raises:
    """`create_external_stream` wraps a raw `cudaStream_t`/`hipStream_t` in a
    `DeviceStream`; every collective call was re-wrapping the SAME handle
    (torch hands this library one stable stream per torch.Stream for the
    whole communicator's life -- it is not reused across streams), so cache
    the wrapper in `state.stream_cache` keyed by the raw handle instead of
    re-wrapping on every call. A handle not seen before is wrapped and
    inserted; callers then read `state.stream_cache[handle]` directly (a
    `ref`, no copy). If a handle were ever reused for a different stream this
    would keep returning the stale wrapper -- not something a torch process
    does, but worth knowing if that assumption ever breaks.
    """
    if handle not in state.stream_cache:
        var wrapped = state.ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(handle))
        )
        _ = state.stream_cache.insert(handle, wrapped^)


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
# collectives_kernels.mojo); this is the host half -- read it for the price of
# a load, say what happened once, and fail every later collective.
# ---------------------------------------------------------------------------


@always_inline
def _fault_field(state: CommState, index: Int) -> UInt64:
    """One word of the fault record, by its `FAULT_*` index."""
    return Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
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
    return Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
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
    if Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](host) != 0:
        return
    # First fully published fault observed here wins. A device record still
    # being published loses to this host fault; their detail words are disjoint.
    var device = Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=page + (STATUS_FAULT_WORD + FAULT_CODE) * 8
        )
    )
    var device_first = UInt64(device != 0) << 63
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
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
# Version / error string / unique id
# ---------------------------------------------------------------------------


@export
def ncclGetVersion(version: Pointer[Int32, MutAnyOrigin]) abi("C") -> Int32:
    # 2.31.2, encoded per nccl.h.in's NCCL_VERSION macro: X*10000+Y*100+Z for
    # Y>8 (true from 2.9 on) -- 2*10000 + 31*100 + 2 = 23102, matching the
    # pinned header this ABI was written against. (A prior value here, 22031,
    # did not decode to 2.31.2 under that formula.)
    version[] = 23102
    return NCCL_SUCCESS


comptime _ERR_SUCCESS: StaticString = "no error"
comptime _ERR_UNHANDLED_CUDA: StaticString = "unhandled cuda/hip error"
comptime _ERR_SYSTEM: StaticString = "system error"
comptime _ERR_INTERNAL: StaticString = "internal error"
comptime _ERR_INVALID_ARGUMENT: StaticString = "invalid argument"
comptime _ERR_INVALID_USAGE: StaticString = (
    "invalid usage (not implemented by mojoccl)"
)
comptime _ERR_REMOTE: StaticString = "remote error (a peer's barrier timed out)"
comptime _ERR_IN_PROGRESS: StaticString = "operation in progress"
comptime _ERR_UNKNOWN: StaticString = "unknown result code"


@export
def ncclGetErrorString(
    result: Int32,
) abi("C") -> Pointer[UInt8, ImmStaticOrigin]:
    if result == NCCL_SUCCESS:
        return _ERR_SUCCESS.unsafe_ptr()
    elif result == NCCL_UNHANDLED_CUDA_ERROR:
        return _ERR_UNHANDLED_CUDA.unsafe_ptr()
    elif result == NCCL_SYSTEM_ERROR:
        return _ERR_SYSTEM.unsafe_ptr()
    elif result == NCCL_INTERNAL_ERROR:
        return _ERR_INTERNAL.unsafe_ptr()
    elif result == NCCL_INVALID_ARGUMENT:
        return _ERR_INVALID_ARGUMENT.unsafe_ptr()
    elif result == NCCL_INVALID_USAGE:
        return _ERR_INVALID_USAGE.unsafe_ptr()
    elif result == NCCL_REMOTE_ERROR:
        return _ERR_REMOTE.unsafe_ptr()
    elif result == NCCL_IN_PROGRESS:
        return _ERR_IN_PROGRESS.unsafe_ptr()
    else:
        return _ERR_UNKNOWN.unsafe_ptr()


@export
def ncclGetUniqueId(uid_out: Pointer[UInt8, MutAnyOrigin]) abi("C") -> Int32:
    try:
        make_unique_id(uid_out)
        return NCCL_SUCCESS
    except:
        return NCCL_SYSTEM_ERROR


# ---------------------------------------------------------------------------
# ncclCommInitRank -- the one struct-by-value export. `ncclUniqueId commId`
# is 128 bytes, SysV MEMORY class: it consumes no integer register and is
# pushed on the stack by a real C caller, so `rank` (declared after it) gets
# the next FREE register (rdx), not the one following `nranks` textually.
# Mirrored here by 3 real leading params (comm/nranks/rank -> rdi/esi/edx),
# 3 Int64 dummies exhausting rcx/r8/r9, then 16 UInt64 stack params that ARE
# the struct -- the callee-side twin of the caller-side shim
# proto/ipc_probe.mojo uses for cuIpcOpenMemHandle. Verified host-only with
# ctypes against this exact parameter shape (register/stack layout, no GPU
# needed): a probe function of this shape decoded nranks, rank and a
# checksum of the 16 id words correctly when called exactly as
# ncclCommInitRank(comm, nranks, ncclUniqueId, rank) would be.
# ---------------------------------------------------------------------------


@export
def ncclCommInitRank(
    comm_out: Pointer[Int64, MutAnyOrigin],
    nranks: Int32,
    rank: Int32,
    _r1: Int64,
    _r2: Int64,
    _r3: Int64,
    id0: UInt64,
    id1: UInt64,
    id2: UInt64,
    id3: UInt64,
    id4: UInt64,
    id5: UInt64,
    id6: UInt64,
    id7: UInt64,
    id8: UInt64,
    id9: UInt64,
    id10: UInt64,
    id11: UInt64,
    id12: UInt64,
    id13: UInt64,
    id14: UInt64,
    id15: UInt64,
) abi("C") -> Int32:
    # `regions` (CommState and every collective kernel's argument list) is a
    # fixed StaticTuple[.., MAX_WORLD]: a larger world would index past its
    # end. Guard explicitly rather than rely on StaticTuple's own bounds
    # check, whose behavior under a release (non-debug) build is not this
    # library's contract to depend on.
    if Int(nranks) < 1 or Int(rank) < 0 or Int(rank) >= Int(nranks):
        return NCCL_INVALID_ARGUMENT
    try:
        var idbuf = unsafe_alloc[UInt8](UID_BYTES)
        var idbuf64 = idbuf.unsafe_bitcast[UInt64]()
        idbuf64[unsafe_offset=0] = id0
        idbuf64[unsafe_offset=1] = id1
        idbuf64[unsafe_offset=2] = id2
        idbuf64[unsafe_offset=3] = id3
        idbuf64[unsafe_offset=4] = id4
        idbuf64[unsafe_offset=5] = id5
        idbuf64[unsafe_offset=6] = id6
        idbuf64[unsafe_offset=7] = id7
        idbuf64[unsafe_offset=8] = id8
        idbuf64[unsafe_offset=9] = id9
        idbuf64[unsafe_offset=10] = id10
        idbuf64[unsafe_offset=11] = id11
        idbuf64[unsafe_offset=12] = id12
        idbuf64[unsafe_offset=13] = id13
        idbuf64[unsafe_offset=14] = id14
        idbuf64[unsafe_offset=15] = id15
        return _init_rank(_any(idbuf), Int(rank), Int(nranks), comm_out)
    except e:
        print("mojoccl: ncclCommInitRank failed:", e)
        return NCCL_INTERNAL_ERROR


def _init_rank(
    uid: Pointer[UInt8, MutAnyOrigin],
    rank: Int,
    nranks: Int,
    comm_out: Pointer[Int64, MutAnyOrigin],
) raises -> Int32:
    """Bring the rendezvous up, run init under it, and always hand it back.

    `BootstrapConn` has no destructor, so every path out of `_bootstrap`
    has to close its sockets explicitly: a leaked root listener keeps the
    unique id's port bound for the life of the process, and a leaked
    per-rank socket leaves the peers blocked in `_recv_all` until their own
    deadline instead of failing fast on a closed connection.
    """
    var timeout_s = _bootstrap_timeout_s()
    # The id's magic names this communicator's node-local fd sockets, so the
    # rendezvous carries no extra round for them (vmm.mojo's `socket_path`).
    var magic = decode_id(uid)[2]
    var conn = bootstrap_connect(uid, rank, nranks, timeout_s)
    try:
        var rc = _bootstrap(conn, rank, nranks, magic, comm_out, timeout_s)
        conn.close()
        return rc
    except e:
        conn.close()
        raise e


def _unwind_init(
    lib: OwnedDLHandle,
    ordinal: Int,
    ib: Int,
    use_nvls: Bool,
    mut nvls: NvlsRegion,
    base: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    local_world: Int,
    local_rank: Int,
    abort_host: Int,
):
    """Release what `_bootstrap` acquired before it failed: the IB state
    (progress thread, QPs, MRs, pinned mailbox), the peer mappings opened so
    far, the pinned abort word, and the region itself. Best effort -- the
    error that got us here is the one worth reporting."""
    ib_teardown(ib)
    try:
        if abort_host != 0:
            free_host(lib, abort_host)
    except e:
        # Best effort: the error that got us here is the one worth reporting.
        print(
            "mojoccl: init unwind: freeing the status page failed (ignored):", e
        )
    try:
        if use_nvls:
            nvls_teardown(nvls, lib, ordinal)
        else:
            for r in range(local_world):
                if r != local_rank and regions[r] != 0:
                    close_handle(lib, regions[r])
            free_region(lib, base)
    except e:
        # Best effort: the error that got us here is the one worth reporting.
        print("mojoccl: init unwind: releasing the region failed (ignored):", e)


def _bootstrap(
    mut conn: BootstrapConn,
    rank: Int,
    nranks: Int,
    magic: UInt64,
    comm_out: Pointer[Int64, MutAnyOrigin],
    timeout_s: Float64,
) raises -> Int32:
    """Three bootstrap rounds and everything they gate.

    Round 1 gathers host identity, from which every rank derives the same
    node/local_rank table. Round 2 gathers the 64-byte IPC handle plus, on a
    multi-node communicator, this rank's IB connection data. Round 3 is a
    barrier: past it every peer's region is zeroed, its IPC handles are open
    and its queue pairs are in RTS, so the first collective may write into
    it.
    """
    var lib = open_driver()
    var ordinal = current_device_ordinal(lib)
    var ctx = DeviceContext(device_id=ordinal)

    # Round 1: host identity, and whether this rank can take the NVLS path.
    #
    # The capability travels here, before anything is allocated, because the
    # answer decides what KIND of memory the region is: VMM bound to a
    # multicast object, or a plain cuMemAlloc. Every rank ANDs the whole
    # column, so one rank without multicast (or with MOJOCCL_NVLS=0) takes the
    # communicator back to the unicast kernels with no half-built state to
    # unwind. Rank 0 additionally does a real `cuMulticastCreate` of a
    # granularity-sized object and releases it: that is where a broken
    # fabric-manager setup shows up, it costs ~88 us, and finding out now is
    # what makes "fall back" a decision rather than a recovery.
    comptime BLOB1 = 24
    var b1 = unsafe_alloc[UInt8](BLOB1)
    var b1w = b1.unsafe_bitcast[UInt64]()
    b1w[unsafe_offset=0] = host_hash()
    b1w[unsafe_offset=1] = UInt64(rank)
    var my_caps: UInt64 = 0
    if (
        _nvls_enabled()
        and nvls_available()
        and nranks >= 2
        and multicast_capable(lib, ordinal, nranks, rank == 0)
    ):
        my_caps = 1
    b1w[unsafe_offset=2] = my_caps
    var t1 = unsafe_alloc[UInt8](BLOB1 * nranks)
    bootstrap_allgather(conn, _any(b1), BLOB1, _any(t1), timeout_s)
    var t1w = t1.unsafe_bitcast[UInt64]()
    var hashes = List[UInt64]()
    var all_caps = True
    for r in range(nranks):
        hashes.append(t1w[unsafe_offset=3 * r])
        if t1w[unsafe_offset=3 * r + 2] & 1 == 0:
            all_caps = False
    var topo = derive_topology(hashes, rank)
    if topo.local_world > MAX_WORLD:
        raise Error(
            "mojoccl: "
            + String(topo.local_world)
            + " ranks on one node, but the intra-node kernels are built for at"
            " most "
            + String(MAX_WORLD)
        )
    if topo.nnodes > MAX_NODES:
        raise Error(
            "mojoccl: "
            + String(topo.nnodes)
            + " nodes exceeds the "
            + String(MAX_NODES)
            + "-node limit of the per-node QPN table in the bootstrap blob"
        )

    var cap_bytes = _region_cap_bytes()
    var fused_cap = fused_block_cap()
    var fused_big_cap = fused_big_block_cap()
    var fused_big = fused_big_bytes()
    var split_unit = _pipe_split_unit()
    var fused = False
    var fused_resident = 0
    # The NVLS grid and the fused grid must be entirely resident, so both are
    # functions of this device's SM count, not constants. MAX's attribute
    # query answers on both vendors (the driver-binding one, `vmm.sm_count`,
    # is NVIDIA-only and reads 0 on AMD, which would leave AMD on the split
    # schedule for no reason); the binding is the fallback.
    var device_sms: Int
    try:
        device_sms = Int(
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        )
    except:
        device_sms = 0
    if device_sms <= 0:
        device_sms = sm_count(lib, ordinal)
    # A positive multiple of 4096 (the kernels' own precondition,
    # RESULTS.md section 9) is what keeps every per-chunk offset the
    # collectives form 16-byte aligned for every supported dtype.
    if cap_bytes <= 0 or cap_bytes % 4096 != 0:
        raise Error(
            "mojoccl: MOJOCCL_REGION_MB must be a positive 4 KiB multiple"
        )
    # Region layout.
    #
    # Single node, unchanged: one arena, [signal | stage_in cap | stage_out
    # cap], and nothing else -- same bytes, same offsets, same numbers as
    # before this file learned about nodes.
    #
    # Multi-node: PIPE_ARENAS of those, each a complete region in its own
    # right with `arena_cap = cap/PIPE_ARENAS` halves, followed by one
    # cap-sized network area
    #
    #     [credits 4 KiB | staging cap/2 - 4 KiB | inbox: INBOX_SLOTS groups]
    #
    # The arenas are what let pipeline chunks run concurrently without a new
    # kernel parameter: `_arena_regions` hands the split kernels a shifted
    # base and `arena_cap`, and each arena keeps its own flag matrix, so
    # concurrent chunks cannot collide in stage_in's push slots or in
    # stage_out's shard. Total staging is 2*cap either way, so the region is
    # the size it always was, plus PIPE_ARENAS-1 extra 128 KiB signal areas.
    #
    # Everything the network touches lives in the network area and nowhere
    # else. Aliasing it onto stage_in would have been free in memory and
    # wrong in fact: a peer node writes my inbox as soon as ITS
    # reduce-scatter is done, which is not ordered against MY reduce-scatter
    # still using stage_in, nor against a local peer still reading the
    # previous generation's staging.
    # NVLS: on a single-node communicator whose devices all support NVSwitch
    # multicast, the region is VMM memory bound to a per-node multicast object
    # instead of a `cuMemAlloc` block, and peers are imported with
    # `cuMemImportFromShareableHandle` instead of `cuIpcOpenMemHandle` (legacy
    # IPC cannot open VMM memory). The unicast kernels do not notice: they get
    # the plain mapping of exactly the same layout, and `nvls.mc` is a second
    # mapping only nvls_kernels.mojo ever touches.
    #
    # Single node only, deliberately. The inter-node path RDMA-writes straight
    # out of an arena's stage_out, so the arenas have to be inside the
    # registered MR -- and `ibv_reg_mr` on a cuMemMap'd VA did not work on this
    # cluster first try, with no dmabuf fallback (CU_DEVICE_ATTRIBUTE_DMA_BUF_
    # SUPPORTED is 0 there), see the prototype's RESULTS.md section 4. Since
    # the multi-node collective stays on the unicast split kernels anyway
    # (their pipeline chunks are below the NVLS crossover; see
    # agents_docs/distributed.md), a multicast region there would cost 150-230 ms of
    # bring-up and a granularity rounding for nothing.
    var use_nvls = all_caps and topo.nnodes == 1 and topo.local_world >= 2
    var granularity = 0
    if use_nvls:
        # The region is rounded up to this. `MOJOCCL_REGION_MB` still means
        # what it says because vmm.mojo asks for the MINIMUM granularity
        # (2 MiB on H100) rather than NCCL's RECOMMENDED (512 MiB); every rank
        # derives the same number from the same device property, and round 2
        # checks that they agree.
        granularity = multicast_granularity(
            lib,
            topo.local_world,
            ordinal,
            signal_bytes() + 2 * cap_bytes,
            _nvls_recommended_granularity(),
        )

    var layout = region_layout(cap_bytes, topo.nnodes)
    var narenas = layout[0]
    var arena_cap = layout[1]
    var arena_stride = layout[2]
    var region_bytes = layout[3]
    if arena_cap < 4096:
        raise Error(
            "mojoccl: MOJOCCL_REGION_MB is too small to carve "
            + String(narenas)
            + " pipeline arenas out of"
        )
    var net_off = narenas * arena_stride

    var nvls = NvlsRegion()
    var base: Int
    if use_nvls:
        var libc = OwnedDLHandle("libc.so.6")
        var spath = socket_path(_socket_dir(), magic, topo.my_local_rank)
        var sock = scm_bind(libc, spath, NVLS_FD_TIMEOUT_S)
        try:
            # Three rendezvous, and the two inner ones are the ordering rules
            # of the multicast API: every device must be in the team before
            # any memory is bound, and every rank must have mapped before any
            # rank issues a multimem instruction. On a single-node
            # communicator the bootstrap barrier IS the node barrier.
            bootstrap_barrier(conn, timeout_s)
            nvls_create_and_share(
                lib,
                libc,
                _socket_dir(),
                magic,
                topo.my_local_rank,
                topo.local_world,
                ordinal,
                _align_up(region_bytes, granularity),
                granularity,
                sock,
                NVLS_FD_TIMEOUT_S,
                nvls,
            )
            bootstrap_barrier(conn, timeout_s)
            nvls_bind_and_map(
                lib,
                libc,
                _socket_dir(),
                magic,
                topo.my_local_rank,
                topo.local_world,
                ordinal,
                sock,
                NVLS_FD_TIMEOUT_S,
                nvls,
            )
            bootstrap_barrier(conn, timeout_s)
        except e:
            try:
                nvls_teardown(nvls, lib, ordinal)
                scm_unbind(libc, sock, spath)
            except e2:
                # Best effort: the bring-up error below is the one to report.
                print("mojoccl: NVLS unwind failed (ignored):", e2)
            raise Error(
                "mojoccl: the NVSwitch-multicast region failed to come up ("
                + String(e)
                + "); every device reported CU_DEVICE_ATTRIBUTE_MULTICAST_"
                "SUPPORTED and rank 0's create probe passed, so this is a"
                " real fault rather than an unsupported machine. Set"
                " MOJOCCL_NVLS=0 to run on the unicast kernels."
            )
        scm_unbind(libc, sock, spath)
        base = nvls.uc
    else:
        base = alloc_region(lib, region_bytes)
    # From here on real resources exist (the region, and on several nodes
    # the IB state with its progress thread); any later failure -- a geometry
    # mismatch, an IPC import, ib_connect, the closing barrier -- has to
    # release them, because no communicator handle is returned through
    # which the caller could.
    var ib = 0
    var regions = StaticTuple[Int, MAX_WORLD](fill=0)
    var nvls_min = _nvls_min_bytes()
    var abort_host = 0
    var abort_dev = 0
    try:
        # The status page, before any spin can start: pinned host memory, so
        # `ncclCommAbort` raises the abort word with one store and no driver
        # call, and device-mapped, so a spinning kernel can read it -- and so
        # a kernel that gives up can latch its fault where the host reads it
        # without synchronizing a stream. Its device address goes in every
        # arena's header -- the six launcher signatures in
        # collectives_kernels.mojo are fixed, so that slot is how the kernels
        # find it.
        abort_host = alloc_host(lib, STATUS_PAGE_BYTES)
        for i in range(STATUS_PAGE_BYTES // 8):
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=abort_host)[
                unsafe_offset=i
            ] = 0
        abort_dev = host_device_ptr(lib, abort_host)
        for a in range(narenas):
            region_init(ctx, base + a * arena_stride)
            install_status_page(ctx, base + a * arena_stride, abort_dev)

        if topo.nnodes > 1:
            ib = ib_setup(
                lib,
                ordinal,
                topo.my_local_rank,
                topo.local_world,
                topo.my_node,
                topo.nnodes,
                base,
                region_bytes,
                INBOX_SLOTS,
                net_off,
            )
            ib_set_abort_word(ib, abort_dev, abort_host)

        # The schedule this rank will run and the fused kernel's residency,
        # decided before round 2 so the peers can check them. The probe
        # compiles the fp32 kernel now rather than at the first collective.
        if (
            _fused_enabled()
            and ib != 0
            and ib_uses_proxy(ib)
            and device_sms > 0
        ):
            fused_resident = fused_resident_blocks(
                ctx, topo.local_world, device_sms
            )
            fused = fused_resident > 0

        # Round 2: IPC handle + IB connection data + the geometry every rank has
        # to agree on. `MOJOCCL_REGION_MB` reaching one rank and not another
        # (a per-node environment, a stale export) silently gives the peers
        # different arena and inbox offsets, which is a data race, not an error;
        # a per-rank `NVLS_MIN_BYTES` sends one rank into the multicast
        # counter barrier and another into the flag barrier, which is a hang.
        # Checking three integers here turns both into a message.
        comptime CFG_BYTES = 88
        comptime BLOB2 = HANDLE_BYTES + IB_BLOB_BYTES + CFG_BYTES
        var b2 = unsafe_alloc[UInt8](BLOB2)
        for i in range(BLOB2):
            b2[unsafe_offset=i] = 0
        if not use_nvls:
            # `cuIpcGetMemHandle` cannot export VMM memory; under NVLS the peers
            # already have this region through the fd exchange above, and these
            # 64 bytes stay zero.
            get_handle(lib, base, _any(b2))
        if ib != 0:
            ib_local_info(
                ib,
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(b2) + HANDLE_BYTES
                ),
            )
        var cfg = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(b2) + HANDLE_BYTES + IB_BLOB_BYTES
        )
        cfg[unsafe_offset=0] = Int64(cap_bytes)
        cfg[unsafe_offset=1] = Int64(
            narenas * 1000 + INBOX_SLOTS + (1_000_000 if use_nvls else 0)
        )
        cfg[unsafe_offset=2] = Int64(nvls_min if use_nvls else 0)
        cfg[unsafe_offset=3] = Int64(fused_cap)
        cfg[unsafe_offset=4] = Int64(split_unit)
        cfg[unsafe_offset=5] = Int64(fused_big_cap)
        cfg[unsafe_offset=6] = Int64(fused_big)
        # The effective schedule and the fused kernel's geometry. The split
        # and fused schedules launch different grids and the barriers are
        # block-matched, so a rank on the other schedule, built with another
        # thread count, or on a GPU that holds a different number of blocks
        # would sync a different slice of the data: a hang or a race, caught
        # here as a message. The two device numbers are compared within a
        # node only -- the barriers never cross nodes.
        cfg[unsafe_offset=7] = Int64(1 if fused else 0)
        cfg[unsafe_offset=8] = Int64(FUSED_THREADS)
        cfg[unsafe_offset=9] = Int64(fused_resident)
        cfg[unsafe_offset=10] = Int64(device_sms)
        var t2 = unsafe_alloc[UInt8](BLOB2 * nranks)
        bootstrap_allgather(conn, _any(b2), BLOB2, _any(t2), timeout_s)
        for r in range(nranks):
            var rcfg = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=Int(t2)
                + r * BLOB2
                + HANDLE_BYTES
                + IB_BLOB_BYTES
            )
            if (
                rcfg[unsafe_offset=0] != cfg[unsafe_offset=0]
                or rcfg[unsafe_offset=1] != cfg[unsafe_offset=1]
                or rcfg[unsafe_offset=2] != cfg[unsafe_offset=2]
                or rcfg[unsafe_offset=3] != cfg[unsafe_offset=3]
                or rcfg[unsafe_offset=4] != cfg[unsafe_offset=4]
                or rcfg[unsafe_offset=5] != cfg[unsafe_offset=5]
                or rcfg[unsafe_offset=6] != cfg[unsafe_offset=6]
                or rcfg[unsafe_offset=7] != cfg[unsafe_offset=7]
                or rcfg[unsafe_offset=8] != cfg[unsafe_offset=8]
            ):
                raise Error(
                    "mojoccl: rank "
                    + String(r)
                    + " built a region of "
                    + String(Int(rcfg[unsafe_offset=0]) // (1024 * 1024))
                    + " MiB with layout code "
                    + String(Int(rcfg[unsafe_offset=1]))
                    + " and NVLS floor "
                    + String(Int(rcfg[unsafe_offset=2]) // (1024 * 1024))
                    + " MiB, this rank "
                    + String(cap_bytes // (1024 * 1024))
                    + " MiB / "
                    + String(Int(cfg[unsafe_offset=1]))
                    + " / "
                    + String(Int(cfg[unsafe_offset=2]) // (1024 * 1024))
                    + " MiB and a fused grid of "
                    + String(Int(cfg[unsafe_offset=3]))
                    + "/"
                    + String(Int(cfg[unsafe_offset=5]))
                    + " from "
                    + String(Int(cfg[unsafe_offset=6]) // (1024 * 1024))
                    + " MiB against that rank's "
                    + String(Int(rcfg[unsafe_offset=3]))
                    + "/"
                    + String(Int(rcfg[unsafe_offset=5]))
                    + " from "
                    + String(Int(rcfg[unsafe_offset=6]) // (1024 * 1024))
                    + " MiB, and a split unit of "
                    + String(Int(cfg[unsafe_offset=4]))
                    + " against that rank's "
                    + String(Int(rcfg[unsafe_offset=4]))
                    + ", schedule "
                    + ("fused" if cfg[unsafe_offset=7] != 0 else "split")
                    + " x "
                    + String(Int(cfg[unsafe_offset=8]))
                    + " threads against that rank's "
                    + ("fused" if rcfg[unsafe_offset=7] != 0 else "split")
                    + " x "
                    + String(Int(rcfg[unsafe_offset=8]))
                    + "; MOJOCCL_REGION_MB and the build's collective geometry,"
                    " thresholds and schedule must match on every rank"
                )
            if topo.node_of[r] == topo.my_node and (
                rcfg[unsafe_offset=9] != cfg[unsafe_offset=9]
                or rcfg[unsafe_offset=10] != cfg[unsafe_offset=10]
            ):
                raise Error(
                    "mojoccl: rank "
                    + String(r)
                    + " on this node can hold "
                    + String(Int(rcfg[unsafe_offset=9]))
                    + " blocks of the fused allreduce kernel on "
                    + String(Int(rcfg[unsafe_offset=10]))
                    + " multiprocessors, this rank "
                    + String(fused_resident)
                    + " on "
                    + String(device_sms)
                    + "; the ranks of a node must run identical GPUs"
                )

        # Same-node peers only: an IPC handle from another host is meaningless.
        regions[topo.my_local_rank] = base
        if use_nvls:
            for r in range(topo.local_world):
                regions[r] = nvls.peer_va[r]
        else:
            for r in range(nranks):
                if topo.node_of[r] != topo.my_node or r == rank:
                    continue
                var lr = topo.local_rank_of[r]
                regions[lr] = open_handle(
                    lib,
                    Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=Int(t2) + r * BLOB2
                    ),
                )

        if ib != 0:
            var peer_rank_of_node = List[Int]()
            for j in range(topo.nnodes):
                peer_rank_of_node.append(
                    topo.rank_at[j * topo.local_world + topo.my_local_rank]
                )
            ib_connect(
                ib,
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(t2) + HANDLE_BYTES
                ),
                BLOB2,
                peer_rank_of_node,
            )

        bootstrap_barrier(conn, timeout_s)
    except e:
        _unwind_init(
            lib,
            ordinal,
            ib,
            use_nvls,
            nvls,
            base,
            regions,
            topo.local_world,
            topo.my_local_rank,
            abort_host,
        )
        raise e

    var rank_at = List[Int]()
    for i in range(len(topo.rank_at)):
        rank_at.append(topo.rank_at[i])
    var nvls_grid = nvls_blocks(device_sms)
    var state = CommState(
        rank=rank,
        world=nranks,
        ordinal=ordinal,
        ctx=ctx,
        driver=lib^,
        cap_bytes=cap_bytes,
        regions=regions,
        owned_base=base,
        local_rank=topo.my_local_rank,
        local_world=topo.local_world,
        my_node=topo.my_node,
        nnodes=topo.nnodes,
        rank_at=rank_at^,
        ib=ib,
        net_off=net_off,
        narenas=narenas,
        arena_cap=arena_cap,
        arena_stride=arena_stride,
        nslots=INBOX_SLOTS,
        nvls=nvls^,
        nvls_on=use_nvls,
        nvls_grid=nvls_grid,
        nvls_min=nvls_min,
        sm_count=device_sms,
        fused_cap=fused_cap,
        fused_big_cap=fused_big_cap,
        fused_big_bytes=fused_big,
        fused_resident=fused_resident,
        split_unit=split_unit,
        fused=fused,
        abort_host=abort_host,
        abort_dev=abort_dev,
    )
    var handle_ptr = unsafe_alloc[CommState](1)
    handle_ptr.unsafe_write(state^)
    comm_out[] = Int64(Int(handle_ptr))
    return NCCL_SUCCESS


def _cached_stream_handles(state: CommState) -> List[Int64]:
    """Copy cached stream handles for abort's bounded polling."""
    var handles = List[Int64]()
    for h in state.stream_cache.keys():
        handles.append(h)
    return handles^


def _drain_all_streams(state: CommState) raises:
    """The last completion event covers the total order across all streams.

    Caller-owned streams may already be destroyed; only the event is ours.
    """
    if state.order_incomplete:
        raise Error("collective completion was not recorded; abort required")
    if state.order_recorded:
        state.order_event.synchronize()


# ---------------------------------------------------------------------------
# Communicator lifecycle
# ---------------------------------------------------------------------------


@export
def ncclCommDestroy(comm: Int64) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    _lock(state)
    var rc = _destroy_locked(comm)
    _unlock(state)
    return rc


def _release_resources(mut state: CommState) raises:
    """Give every resource this communicator owns back, once nothing can be
    using it. Shared by `ncclCommDestroy` and by an abort that reached
    quiescence, and it is the same unwind `_unwind_init` runs on a failed
    bring-up.
    """
    ib_teardown(state.ib)
    state.ib = 0
    if state.nvls_on:
        # One call releases the peer mappings, both of my own, the multicast
        # binding and every handle; there is no cuIpc mapping and no
        # cuMemAlloc block to free.
        nvls_teardown(state.nvls, state.driver, state.ordinal)
    else:
        for r in range(state.local_world):
            if r != state.local_rank:
                close_handle(state.driver, state.regions[r])
        free_region(state.driver, state.owned_base)
    if state.abort_host != 0:
        free_host(state.driver, state.abort_host)
        state.abort_host = 0
        state.abort_dev = 0
    state.order_event.release()
    state.released = True


def _destroy_locked(comm: Int64) -> Int32:
    try:
        ref state = _comm_ptr(comm)[]
        # Destroy after abort is legal (nccl.h.in) and has nothing left to do:
        # abort either ran this same unwind or decided it could not safely.
        if state.released or state.aborted:
            return NCCL_SUCCESS
        # Completion includes all inter-node callbacks that use IbState.
        _drain_all_streams(state)
        state.ctx.synchronize()
        _release_resources(state)
        return NCCL_SUCCESS
    except:
        return NCCL_INTERNAL_ERROR


def _abort_quiesced(state: CommState, deadline_ns: Int) -> Bool:
    """Poll owned completion; failed submissions need conservative stream checks.
    """
    var handles = List[Int64]()
    if state.order_incomplete:
        handles = _cached_stream_handles(state)
    while True:
        var pending = False
        if not state.order_incomplete:
            pending = state.order_recorded and not state.order_event.done()
        else:
            for i in range(len(handles)):
                if not stream_done(state.driver, Int(handles[i])):
                    pending = True
                    break
        if not pending:
            return True
        if perf_counter_ns() > deadline_ns:
            return False
        sleep(0.001)


@export
def ncclCommAbort(comm: Int64) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    if state.released:
        return NCCL_SUCCESS
    # nccl.h.in's contract: stop submissions, abort the device operations, and
    # free the communicator's resources -- without waiting on peers, which may
    # be dead. All three, in that order.
    #
    #   1. `aborted` makes every later collective return ncclInvalidUsage and
    #      `ncclCommGetAsyncError` report ncclSystemError.
    #   2. the abort word releases the device spins (the intra-node barriers,
    #      the NVLS barrier, the proxy wait kernel) within a check interval,
    #      each with its region's error word set, instead of holding the
    #      stream to a 60 s deadline; the progress thread stops too.
    #   3. once nothing local is running any more -- polled, with a bound, and
    #      never a wait on a peer -- the region, the peer mappings, the IB
    #      state and the pinned word go back.
    #
    # Step 3 is best effort by construction: a local kernel that will not
    # quiesce inside `ABORT_QUIESCE_TIMEOUT_S` keeps its memory, because
    # freeing a region a kernel is still reading faults the process.
    state.aborted = True
    _raise_abort_word(state)
    var proxy_stopped = ib_signal_abort(state.ib)
    var deadline = perf_counter_ns() + Int(ABORT_QUIESCE_TIMEOUT_S * 1.0e9)
    if not _try_lock(state, deadline):
        print(
            "mojoccl: ncclCommAbort left the region allocated -- another"
            " thread still holds the submission lock"
        )
        return NCCL_SUCCESS
    var quiesced = _abort_quiesced(state, deadline)
    if quiesced and proxy_stopped:
        try:
            _release_resources(state)
        except e:
            print("mojoccl: ncclCommAbort could not release cleanly:", e)
    elif not proxy_stopped:
        # `ib_teardown` joins the progress thread unconditionally, so
        # reclaiming here would be both an unbounded wait and a free under a
        # thread still reading the state.
        print(
            "mojoccl: ncclCommAbort left the resources allocated -- the"
            " progress thread did not stop"
        )
    else:
        print(
            (
                "mojoccl: ncclCommAbort left the region allocated -- the device"
                " was still busy after"
            ),
            ABORT_QUIESCE_TIMEOUT_S,
            "s",
        )
    _unlock(state)
    return NCCL_SUCCESS


@export
def ncclCommGetAsyncError(
    comm: Int64, err_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    try:
        ref state = _comm_ptr(comm)[]
        if state.aborted:
            err_out[] = NCCL_SYSTEM_ERROR
            return NCCL_SUCCESS
        if state.ib != 0 and ib_error(state.ib) != 0:
            # A host callback gave up (post failed, a completion came back
            # with a bad status, or nothing arrived inside the deadline).
            # Host-side state, so it needs no device read.
            err_out[] = NCCL_REMOTE_ERROR
            return NCCL_SUCCESS
        if _fault_code(state) != 0 or _host_fault_word(state) != 0:
            # A device deadline, latched in the status page. Also host memory,
            # so a watchdog polling this function pays nothing for it and --
            # unlike the arena read below -- does not block behind the kernel
            # it is asking about.
            _report_fault(state)
            err_out[] = NCCL_REMOTE_ERROR
            return NCCL_SUCCESS
        if _submission_failed(state):
            err_out[] = NCCL_REMOTE_ERROR
            return NCCL_SUCCESS
        # Never read ordering fields or share poll scratch without the lock.
        # A busy submitter is healthy unless it publishes a terminal failure.
        if not _try_lock(state, perf_counter_ns()):
            err_out[] = NCCL_REMOTE_ERROR if _submission_failed(
                state
            ) else NCCL_SUCCESS
            return NCCL_SUCCESS
        try:
            err_out[] = _async_error_locked(state)
        except e:
            _unlock(state)
            raise e
        _unlock(state)
        return NCCL_SUCCESS
    except:
        return NCCL_INTERNAL_ERROR


def _async_error_locked(mut state: CommState) raises -> Int32:
    if state.aborted or state.released:
        return NCCL_SYSTEM_ERROR
    if _submission_failed(state):
        return NCCL_REMOTE_ERROR
    # Do not hold up submissions behind unfinished GPU work.
    if state.order_recorded and not state.order_event.query():
        return NCCL_SUCCESS
    _drain_all_streams(state)
    if state.ib != 0 and ib_error(state.ib) != 0:
        return NCCL_REMOTE_ERROR
    for a in range(state.narenas):
        var word = _read_error_word(
            state,
            state.regions[state.local_rank] + a * state.arena_stride,
        )
        if Int(word) != 0:
            return NCCL_REMOTE_ERROR
    return NCCL_SUCCESS


@export
def ncclCommCount(
    comm: Int64, count_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    count_out[] = Int32(state.world)
    return NCCL_SUCCESS


@export
def ncclCommUserRank(
    comm: Int64, rank_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    rank_out[] = Int32(state.rank)
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Group semantics: every op here already executes on the same stream, in
# issue order, so aggregating a group buys nothing -- immediate execution is
# correct, per the design brief.
# ---------------------------------------------------------------------------


@export
def ncclGroupStart() abi("C") -> Int32:
    return NCCL_SUCCESS


@export
def ncclGroupEnd() abi("C") -> Int32:
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------


def _arena_regions(state: CommState, arena: Int) -> StaticTuple[Int, MAX_WORLD]:
    """Peer bases of one pipeline arena.

    Each arena is a complete region as far as collectives_kernels.mojo is
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


def _exchange_release[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    arena: Int,
    numel: Int,
    chunk_index: Int,
    nchunks: Int,
) raises -> Int:
    """Release one chunk's shard to the network and return its exchange
    counter.

    `chunk_index` / `nchunks` are carried into the work item for the stall
    messages only (`internode._ring_state`): they cost nothing and turn "the
    GPU has not released exchange 78" into a sentence naming the collective
    and the chunk.

    On entry this rank's node-local sum of the chunk's shard sits in arena
    `arena`'s stage_out (that is `reduce_scatter_stage`'s contract). This
    enqueues the release only; the stream runs straight on into the next
    chunk's reduce-scatter, and `_exchange_consume` is what eventually waits.

    Slot geometry is derived from `(numel, local_world, item)` alone so a
    sender computes the same addresses as its receiver: a group holds
    `(nnodes-1)` slots of one shard each, at the fixed base `_inbox_base`
    picks by `seq % nslots`. `_max_chunk_bytes` keeps a group inside its
    share of the network area.
    """
    comptime item = size_of[dtype]()
    var sr = shard_range(numel, state.local_world, state.local_rank, item)
    var off_e = sr[0]
    var cnt_e = sr[1]
    var seq = ib_next_seq(state.ib)
    var npeers = ib_npeers(state.ib)
    var nbytes = cnt_e * item
    var slot_bytes = _align_up(max(nbytes, EMPTY_SHARD_BYTES), 16)
    if npeers * slot_bytes > _inbox_group_bytes(state):
        raise Error(
            "mojoccl: the inter-node inbox overflows the network area; this"
            " is a chunking bug"
        )
    var inbox_base = _inbox_base(state, seq)
    var shard = _arena_shard(state, arena, off_e * item)
    # A rank with an empty shard (numel below local_world's rounded-up shard
    # size -- DDP's 4-byte AVG allreduce does exactly this on 7 of 8 local
    # ranks) still exchanges, with EMPTY_SHARD_BYTES of ignored payload. See
    # `EMPTY_SHARD_BYTES` for why every exchange has to be all-to-all.
    ib_enqueue_request(
        state.ib,
        state.driver,
        state.ctx,
        stream,
        Int(raw_stream),
        shard if cnt_e > 0 else state.owned_base,
        nbytes if cnt_e > 0 else EMPTY_SHARD_BYTES,
        inbox_base,
        slot_bytes,
        True,
        npeers,
        state.owned_base + inbox_base if cnt_e > 0 else 0,
        seq,
        OP_ALLREDUCE,
        chunk_index,
        nchunks,
        numel,
    )
    return seq


def _exchange_consume[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    arena: Int,
    seq: Int,
    numel: Int,
) raises:
    """Wait for exchange `seq`, add what arrived, and release its inbox slot.

    `ib_note_consumed` is the credit: it records that the add kernel is
    enqueued, and the number rides out to the peers on this rank's next
    request, by which point that kernel has run (stream order). On exit the
    shard holds the sum over the WHOLE communicator, ready for
    `allgather_finish`.
    """
    comptime item = size_of[dtype]()
    var sr = shard_range(numel, state.local_world, state.local_rank, item)
    var cnt_e = sr[1]
    var npeers = ib_npeers(state.ib)
    var slot_bytes = _align_up(max(cnt_e * item, EMPTY_SHARD_BYTES), 16)
    ib_enqueue_wait(state.ib, state.ctx, stream, seq)
    if cnt_e > 0:
        inbox_add[dtype](
            state.ctx,
            stream,
            _arena_shard(state, arena, sr[0] * item),
            state.owned_base + _inbox_base(state, seq),
            cnt_e,
            slot_bytes,
            npeers,
        )
    ib_note_consumed(state.ib, seq)


def _do_allreduce_nvls[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
) raises:
    """The single-node allreduce through the NVSwitch.

    Chunked by the region, then pipelined inside the kernel: `chunk` here is
    "what fits the staging half", `nvls_chunk_bytes` is the much smaller unit
    the kernel overlaps copy-in / reduce / copy-out over.

    The flag counter is advanced by exactly what the kernel will consume, on
    the host, before the launch -- the kernel gets the value its FIRST barrier
    waits for and walks up from there, so back-to-back calls on one stream
    never reuse a target.
    """
    comptime item = size_of[dtype]()
    var world = state.local_world
    # The whole `2 * cap` arena, not one half: this kernel stages one buffer
    # and no other collective is live while it runs (its start barrier is what
    # guarantees that), so a 512 MiB allreduce is one launch on the default
    # 256 MiB region. Rounded down to whole per-rank 16-byte slices, which is
    # what the padding rule needs.
    var payload = 2 * state.cap_bytes
    var max_elems = (payload // 16) // world * world * (16 // item)
    var done = 0
    while done < count:
        var chunk = min(max_elems, count - done)
        var cv = nvls_chunk_vecs(nvls_chunk_bytes(chunk * item), world)
        var target = (state.nvls_bars + 1) * world
        state.nvls_bars += nvls_barriers_per_call(chunk, world, item, cv)
        nvls_allreduce[dtype](
            state.ctx,
            stream,
            state.local_rank,
            world,
            state.nvls.mc,
            state.owned_base,
            sendbuff + done * item,
            recvbuff + done * item,
            chunk,
            payload,
            scale,
            target,
            state.nvls_grid,
            cv,
        )
        done += chunk


def _do_allreduce[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
) raises:
    comptime item = size_of[dtype]()
    if state.nnodes == 1:
        comptime if not dtype.is_integral():
            # NVLS: above the measured crossover the switch's own reduction
            # engine moves 1.75x fewer NVLink bytes than the unicast
            # push/reduce/pull, and wins by 4% at 48 MiB growing to 23% at
            # 512 MiB. Below it the extra HBM staging dominates and the
            # unicast kernel keeps the traffic. int32/int64 stay unicast --
            # `multimem.ld_reduce` has no integer add this schedule can use
            # and integer allreduces are never the bucket that matters.
            if state.nvls_on and count * item >= state.nvls_min:
                _do_allreduce_nvls[dtype](
                    state, stream, sendbuff, recvbuff, count, scale
                )
                return
        var max_elems = max(1, state.cap_bytes // item)
        var done = 0
        while done < count:
            var chunk = min(max_elems, count - done)
            state.generation += 1
            allreduce[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                state.regions,
                sendbuff + done * item,
                recvbuff + done * item,
                chunk,
                state.cap_bytes,
                scale,
                state.generation,
            )
            done += chunk
        return

    # The fused kernel files every chunk's work item before it launches, so a
    # collective of more chunks than the ring has slots would wait on itself
    # (`_await_ring_slot`); geometry can cut a large message that finely
    # (129 MiB at MOJOCCL_REGION_MB=1 is over 500 chunks). The split schedule
    # releases chunks as it goes and has no such bound.
    var plan = _pipeline_plan(state, count, item)
    if state.fused and plan[1] <= WORK_SLOTS:
        _do_allreduce_fused[dtype](
            state, stream, sendbuff, recvbuff, count, scale, plan
        )
        return
    _do_allreduce_split[dtype](
        state, stream, raw_stream, sendbuff, recvbuff, count, scale, plan
    )


def _pipeline_plan(
    mut state: CommState, count: Int, item: Int
) -> Tuple[Int, Int, Int]:
    """`(chunk_elems, nchunks, depth)` of a pipelined multi-node allreduce.

    Shared by the fused and the split schedules so they cut a bucket the same
    way: the only difference between them is who runs the loop.
    """
    var chunk_elems = max(1, _pipeline_chunk_bytes(state, count * item) // item)
    var nchunks = (count + chunk_elems - 1) // chunk_elems
    return Tuple(chunk_elems, nchunks, min(state.narenas, nchunks))


def _do_allreduce_fused[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
    plan: Tuple[Int, Int, Int],
) raises:
    """The pipelined multi-node allreduce as one launch (internode_fused.mojo).

    Everything below the launch is what `_do_allreduce_split` does per chunk,
    hoisted: the exchange counters are reserved as one run, every chunk's work
    item is filled before the kernel starts, and the kernel publishes the
    counters, waits for them and releases the inbox slots itself. The host
    touches the driver exactly once.
    """
    comptime item = size_of[dtype]()
    comptime W = 16 // item
    var chunk_elems = plan[0]
    var nchunks = plan[1]
    var npeers = ib_npeers(state.ib)
    var group = _inbox_group_bytes(state)
    # Two generations per chunk, reserved up front, so each chunk's
    # all-gather is its own reduce-scatter's plus one (the split kernels'
    # documented pairing).
    # Every reason to refuse the call comes before anything is reserved:
    # the largest chunk (the first) sets the largest inbox slot.
    check_fused_call[dtype](
        sendbuff, recvbuff, chunk_elems, state.arena_cap, nchunks
    )
    var sr0 = shard_range(
        min(chunk_elems, count), state.local_world, state.local_rank, item
    )
    if npeers * _align_up(max(sr0[1] * item, EMPTY_SHARD_BYTES), 16) > group:
        raise Error(
            "mojoccl: the inter-node inbox overflows the network area; this"
            " is a chunking bug"
        )
    var g0 = state.generation + 1
    state.generation += 2 * nchunks
    var seq0 = ib_reserve_seqs(state.ib, nchunks)
    for k in range(nchunks):
        var off = k * chunk_elems
        var cnt = min(chunk_elems, count - off)
        var sr = shard_range(cnt, state.local_world, state.local_rank, item)
        var nbytes = sr[1] * item
        var slot_bytes = _align_up(max(nbytes, EMPTY_SHARD_BYTES), 16)
        var seq = seq0 + k
        var inbox_base = _inbox_base(state, seq)
        var shard = _arena_shard(state, k % state.narenas, sr[0] * item)
        # A rank with an empty shard still exchanges, with EMPTY_SHARD_BYTES
        # of ignored payload -- every exchange is all-to-all because an
        # arrival tally of `npeers` is what completes one.
        ib_prepare_request(
            state.ib,
            shard if sr[1] > 0 else state.owned_base,
            nbytes if sr[1] > 0 else EMPTY_SHARD_BYTES,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            state.owned_base + inbox_base if sr[1] > 0 else 0,
            seq,
            OP_ALLREDUCE,
            k,
            nchunks,
            cnt,
        )
    try:
        internode_allreduce_fused[dtype](
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            state.regions,
            sendbuff,
            recvbuff,
            ib_mailbox_dev(state.ib),
            count,
            chunk_elems,
            nchunks,
            plan[2],
            state.narenas,
            state.arena_stride,
            state.arena_cap,
            seq0,
            state.net_off + state.cap_bytes // 2,
            group,
            state.nslots,
            npeers,
            g0,
            scale,
            fused_blocks(
                state.fused_big_cap if count * item
                >= state.fused_big_bytes else state.fused_cap,
                state.fused_resident,
                _shard_per(chunk_elems, state.local_world, W),
                W,
            ),
            spin_timeout_ns(),
        )
    except e:
        # The exchanges are reserved and the peers will wait for this rank's
        # flags: fail the communicator now, so every later call here returns
        # ncclRemoteError and the peers' own deadlines say what happened,
        # instead of a hang.
        _latch_host_fault(state, ERR_HOST_LAUNCH, seq0)
        raise e


def _do_allreduce_split[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
    plan: Tuple[Int, Int, Int],
) raises:
    """The pipelined multi-node allreduce as five kernels per chunk.

    Releases work-ring slots incrementally, so it handles messages whose
    chunk count exceeds the fused kernel's pre-enqueued work capacity.
    """
    comptime item = size_of[dtype]()
    # Chunk k is issued as reduce-scatter, release; its wait / add /
    # all-gather come `depth-1` chunks later, so between them the GPU runs
    # whole chunks of other work while the proxy exchanges this one.
    # `depth <= narenas` is what keeps chunk k+narenas's reduce-scatter
    # enqueued AFTER chunk k's all-gather, which is what the arena's start
    # barrier needs to order the reuse.
    var chunk_elems = plan[0]
    var nchunks = plan[1]
    var depth = plan[2]
    # AVG's 1/world is applied by the reduce-scatter to each input (NCCL's
    # PreMulSum): the node partials that cross the network and the inbox add
    # are then already scaled, so no fp16 sum ever exceeds the average, and
    # the all-gather copies with scale 1.
    # Two generations per chunk, reserved up front, so each chunk's
    # all-gather is its own reduce-scatter's plus one (the split kernels'
    # documented pairing) even though the enqueue order interleaves chunks.
    # Per arena the values are still strictly increasing, which is the rule
    # that matters.
    var g0 = state.generation + 1
    state.generation += 2 * nchunks
    var seqs = List[Int]()
    for _ in range(nchunks):
        seqs.append(0)
    for k in range(nchunks + depth - 1):
        if k < nchunks:
            var off = k * chunk_elems
            var cnt = min(chunk_elems, count - off)
            reduce_scatter_stage[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                _arena_regions(state, k % state.narenas),
                sendbuff + off * item,
                cnt,
                state.arena_cap,
                g0 + 2 * k,
                scale,
            )
            seqs[k] = _exchange_release[dtype](
                state, stream, raw_stream, k % state.narenas, cnt, k, nchunks
            )
        var j = k - (depth - 1)
        if j >= 0:
            var off = j * chunk_elems
            var cnt = min(chunk_elems, count - off)
            _exchange_consume[dtype](
                state, stream, j % state.narenas, seqs[j], cnt
            )
            allgather_finish[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                _arena_regions(state, j % state.narenas),
                recvbuff + off * item,
                cnt,
                state.arena_cap,
                Float32(1.0),
                g0 + 2 * j + 1,
            )


@export
def ncclAllReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        if op != NCCL_SUM and op != NCCL_AVG:
            return NCCL_INVALID_USAGE
        # The payload loops use 16-byte vector loads/stores on
        # in_ptr/out_ptr and fault on a misaligned address (RESULTS.md
        # section 9). Every allocator-returned pointer and every chunk
        # offset this function forms satisfy that (the chunk size is a
        # multiple of 4096 bytes) -- only a mid-tensor view the caller
        # passes directly can violate it, so reject that case here with a
        # clear error instead of letting the kernel raise.
        if Int(sendbuff) % 16 != 0 or Int(recvbuff) % 16 != 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _allreduce_locked(
                comm, sendbuff, recvbuff, count, datatype, op, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclAllReduce failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclAllReduce failed:", e)
        return NCCL_INTERNAL_ERROR


def _allreduce_locked(
    comm: Int64,
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    stream: Int64,
) raises -> Int32:
    ref state = _comm_ptr(comm)[]
    if state.aborted:
        return NCCL_INVALID_USAGE
    var latched = _latched_error(state)
    if latched != NCCL_SUCCESS:
        return latched
    if state.order_incomplete:
        return NCCL_REMOTE_ERROR
    _order_before(state, stream)
    ref s = state.stream_cache[stream]
    _enqueue_allreduce(
        state, s, sendbuff, recvbuff, count, datatype, op, stream
    )
    return NCCL_SUCCESS


def _enqueue_allreduce(
    mut state: CommState,
    s: DeviceStream,
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    stream: Int64,
) raises:
    """Enqueue one reduction within an already-open collective order scope.

    Reduce-scatter invokes this repeatedly while its outer operation owns
    the communicator lock and completion event. It must not reopen the
    submission scope between chunks.
    """
    var scale = Float32(1.0)
    if op == NCCL_AVG:
        scale = Float32(1.0) / Float32(state.world)
    if datatype == NCCL_INT32:
        _do_allreduce[DType.int32](
            state,
            s,
            stream,
            Int(sendbuff),
            Int(recvbuff),
            Int(count),
            scale,
        )
    elif datatype == NCCL_INT64:
        _do_allreduce[DType.int64](
            state,
            s,
            stream,
            Int(sendbuff),
            Int(recvbuff),
            Int(count),
            scale,
        )
    elif datatype == NCCL_FLOAT16:
        _do_allreduce[DType.float16](
            state,
            s,
            stream,
            Int(sendbuff),
            Int(recvbuff),
            Int(count),
            scale,
        )
    elif datatype == NCCL_FLOAT32:
        _do_allreduce[DType.float32](
            state,
            s,
            stream,
            Int(sendbuff),
            Int(recvbuff),
            Int(count),
            scale,
        )
    else:  # NCCL_BFLOAT16, ruled in by _dtype_item_bytes above
        _do_allreduce[DType.bfloat16](
            state,
            s,
            stream,
            Int(sendbuff),
            Int(recvbuff),
            Int(count),
            scale,
        )


@export
def ncclBroadcast(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _broadcast_locked(
                comm, sendbuff, recvbuff, Int(count) * item, root, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclBroadcast failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclBroadcast failed:", e)
        return NCCL_INTERNAL_ERROR


def _broadcast_locked(
    comm: Int64,
    sendbuff: Int64,
    recvbuff: Int64,
    total_bytes: Int,
    root: Int32,
    stream: Int64,
) raises -> Int32:
    ref state = _comm_ptr(comm)[]
    if state.aborted:
        return NCCL_INVALID_USAGE
    if Int(root) < 0 or Int(root) >= state.world:
        return NCCL_INVALID_ARGUMENT
    var latched = _latched_error(state)
    if latched != NCCL_SUCCESS:
        return latched
    if state.order_incomplete:
        return NCCL_REMOTE_ERROR
    _order_before(state, stream)
    ref s = state.stream_cache[stream]
    if state.nnodes == 1:
        var max_bytes = max(1, state.cap_bytes)
        var done = 0
        while done < total_bytes:
            var chunk = min(max_bytes, total_bytes - done)
            state.generation += 1
            broadcast(
                state.ctx,
                s,
                state.local_rank,
                Int(root),
                state.local_world,
                state.regions,
                Int(sendbuff) + done,
                Int(recvbuff) + done,
                chunk,
                state.cap_bytes,
                state.generation,
            )
            done += chunk
        return NCCL_SUCCESS
    _broadcast_multinode(
        state,
        s,
        stream,
        Int(sendbuff),
        Int(recvbuff),
        total_bytes,
        Int(root),
    )
    return NCCL_SUCCESS


def _broadcast_multinode(
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    total_bytes: Int,
    root: Int,
) raises:
    """Root -> one rank per node over IB, then a node-local broadcast.

    The rank that receives on node j is the one sharing the root's
    local_rank, because that is the only rank the root has a queue pair to.
    Every OTHER rank still runs a credit-only exchange with its own
    same-local_rank peers (EMPTY_SHARD_BYTES). Correct and simple beats fast
    here: broadcast runs at DDP init, on parameters, not in the step.
    """
    var lw = state.local_world
    var root_node = 0
    var root_lr = 0
    for j in range(state.nnodes):
        for l in range(lw):
            if state.rank_at[j * lw + l] == root:
                root_node = j
                root_lr = l
    var i_am_root = state.rank == root
    var i_am_local_root = state.local_rank == root_lr
    var receives = i_am_local_root and state.my_node != root_node
    # The root's staged chunk goes in the network area's staging half; a
    # peer slot has to fit inside one inbox slot group, one slot PER PEER
    # since the exchange is all-to-all. The node-local half runs in arena 0,
    # so the chunk is bounded by `arena_cap` too.
    var npeers = ib_npeers(state.ib)
    var max_bytes = max(
        4096,
        (
            min(
                min(_net_stage_bytes(state), state.arena_cap),
                _inbox_group_bytes(state) // npeers,
            )
            - 2 * 4096
        )
        // 4096
        * 4096,
    )
    var done = 0
    var chunk_index = 0
    var nchunks = (total_bytes + max_bytes - 1) // max_bytes
    while done < total_bytes:
        var chunk = min(max_bytes, total_bytes - done)
        var seq = ib_next_seq(state.ib)
        var slot_bytes = _align_up(chunk, 16)
        if npeers * slot_bytes > _inbox_group_bytes(state):
            raise Error("mojoccl: broadcast inbox does not fit; chunking bug")
        var inbox_base = _inbox_base(state, seq)
        var recv_slot = 0
        if receives:
            recv_slot = (
                root_node if root_node < state.my_node else root_node - 1
            )
        var send_ptr = recvbuff + done
        # Everyone in this local_rank group exchanges; only the root's
        # payload is real (see EMPTY_SHARD_BYTES).
        var send_addr = state.owned_base
        var send_bytes = EMPTY_SHARD_BYTES
        var flush_addr = 0
        if i_am_root:
            # The user buffer is not registered, so the root's payload has to
            # be copied into the region before the NIC can read it.
            copy_bytes(
                state.ctx,
                stream,
                state.owned_base + _net_stage_off(state),
                sendbuff + done,
                chunk,
            )
            send_addr = state.owned_base + _net_stage_off(state)
            send_bytes = chunk
            send_ptr = sendbuff + done
        elif receives:
            flush_addr = state.owned_base + inbox_base + recv_slot * slot_bytes
            send_ptr = flush_addr
        ib_enqueue_request(
            state.ib,
            state.driver,
            state.ctx,
            stream,
            Int(raw_stream),
            send_addr,
            send_bytes,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            flush_addr,
            seq,
            OP_BROADCAST,
            chunk_index,
            nchunks,
            chunk,
        )
        # Unpipelined on purpose: one exchange at a time, waited for right
        # where it is released. Broadcast's fan-out is a different shape from
        # the allreduce's (only one rank per node has real data) and it runs
        # at DDP init, not in the step.
        ib_enqueue_wait(state.ib, state.ctx, stream, seq)
        state.generation += 1
        broadcast(
            state.ctx,
            stream,
            state.local_rank,
            root_lr,
            lw,
            _arena_regions(state, 0),
            send_ptr,
            recvbuff + done,
            chunk,
            state.arena_cap,
            state.generation,
        )
        # The node-local broadcast is what reads the inbox slot, so the
        # credit is only due now.
        ib_note_consumed(state.ib, seq)
        done += chunk
        chunk_index += 1


@export
def ncclAllGather(
    sendbuff: Int64,
    recvbuff: Int64,
    sendcount: Int64,
    datatype: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _allgather_locked(
                comm, sendbuff, recvbuff, Int(sendcount) * item, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclAllGather failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclAllGather failed:", e)
        return NCCL_INTERNAL_ERROR


def _allgather_locked(
    comm: Int64,
    sendbuff: Int64,
    recvbuff: Int64,
    per_rank_bytes: Int,
    stream: Int64,
) raises -> Int32:
    ref state = _comm_ptr(comm)[]
    if state.aborted:
        return NCCL_INVALID_USAGE
    var latched = _latched_error(state)
    if latched != NCCL_SUCCESS:
        return latched
    if state.order_incomplete:
        return NCCL_REMOTE_ERROR
    _order_before(state, stream)
    ref s = state.stream_cache[stream]
    if state.nnodes == 1:
        # AMD's push all-gather stages `world-1` slots, so its chunk is
        # smaller than the region cap; `allgather_max_bytes` is the identity
        # on NVIDIA.
        var max_bytes = max(
            1, allgather_max_bytes(state.cap_bytes, state.local_world)
        )
        var done = 0
        while done < per_rank_bytes:
            var chunk = min(max_bytes, per_rank_bytes - done)
            state.generation += 1
            allgather(
                state.ctx,
                s,
                state.local_rank,
                state.local_world,
                state.regions,
                Int(sendbuff) + done,
                Int(recvbuff) + done,
                chunk,
                state.cap_bytes,
                state.generation,
                stride_bytes=per_rank_bytes,
            )
            done += chunk
        return NCCL_SUCCESS
    _allgather_multinode(
        state, s, stream, Int(sendbuff), Int(recvbuff), per_rank_bytes
    )
    return NCCL_SUCCESS


def allgather_mapped_max_bytes(
    arena_cap: Int, inbox_group: Int, npeers: Int
) -> Int:
    return min(arena_cap, inbox_group // npeers) // 16 * 16


def _allgather_node_mapped(
    mut state: CommState,
    stream: DeviceStream,
    arena: Int,
    node: Int,
    src: Int,
    dst: Int,
    count: Int,
    stride: Int,
    seq: Int = 0,
    gated: Bool = False,
) raises:
    var ranks = StaticTuple[Int32, MAX_WORLD](fill=0)
    for l in range(state.local_world):
        ranks[l] = Int32(state.rank_at[node * state.local_world + l])
    state.generation += 1
    var mb_req = ib_mailbox_dev(state.ib)[0] if seq != 0 else 0
    if gated:
        allgather_mapped[AG_NODE_UNROLL, True](
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            _arena_regions(state, arena),
            src,
            dst,
            count,
            state.arena_cap,
            state.generation,
            stride,
            ranks,
            AG_NODE_BLOCKS,
            mb_req,
            seq,
        )
        return
    allgather_mapped[AG_NODE_UNROLL](
        state.ctx,
        stream,
        state.local_rank,
        state.local_world,
        _arena_regions(state, arena),
        src,
        dst,
        count,
        state.arena_cap,
        state.generation,
        stride,
        ranks,
        AG_NODE_BLOCKS,
        mb_req,
        seq,
    )


def allgather_mapped_pipeline_plan(
    count: Int,
    chunk_cap: Int,
    narenas: Int,
    nslots: Int,
    split_threshold: Int,
) raises -> Tuple[Int, Int, Int]:
    if count <= 0 or chunk_cap < 16 or narenas <= 0 or nslots < 2:
        raise Error("mojoccl: invalid mapped allgather pipeline geometry")
    var chunk = chunk_cap
    if count >= split_threshold:
        chunk = min(chunk, _align_up((count + 1) // 2, 16))
    var nchunks = (count + chunk - 1) // chunk
    return Tuple(chunk, nchunks, min(2, min(narenas, min(nslots - 1, nchunks))))


def _allgather_multinode_mapped(
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    per_rank_bytes: Int,
) raises:
    if per_rank_bytes == 0:
        return
    var npeers = ib_npeers(state.ib)
    var max_bytes = allgather_mapped_max_bytes(
        state.arena_cap, _inbox_group_bytes(state), npeers
    )
    var plan = allgather_mapped_pipeline_plan(
        per_rank_bytes,
        max_bytes,
        state.narenas,
        state.nslots,
        # Measured on 2x8 H100: overlap large gathers, retain the passing
        # single-chunk route below this node-scaled threshold.
        PIPE_SPLIT_UNIT * state.local_world,
    )
    var chunk_bytes = plan[0]
    var nchunks = plan[1]
    var depth = plan[2]
    var seqs = List[Int](length=depth, fill=0)
    for k in range(nchunks + depth - 1):
        if k < nchunks:
            var off = k * chunk_bytes
            var count = min(chunk_bytes, per_rank_bytes - off)
            var arena = k % depth
            # The work item is filled before the gather launches; the gather
            # itself releases the exchange once the contribution is staged.
            var seq = ib_next_seq(state.ib)
            var slot_bytes = _align_up(count, 16)
            var inbox_base = _inbox_base(state, seq)
            ib_prepare_request(
                state.ib,
                state.owned_base + arena * state.arena_stride + signal_bytes(),
                count,
                inbox_base,
                slot_bytes,
                True,
                npeers,
                state.owned_base + inbox_base,
                seq,
                OP_ALLGATHER,
                k,
                nchunks,
                count,
            )
            _allgather_node_mapped(
                state,
                stream,
                arena,
                state.my_node,
                sendbuff + off,
                recvbuff + off,
                count,
                per_rank_bytes,
                seq,
            )
            seqs[k % depth] = seq
        var j = k - (depth - 1)
        if j >= 0:
            var off = j * chunk_bytes
            var count = min(chunk_bytes, per_rank_bytes - off)
            var slot_bytes = _align_up(count, 16)
            var arena = j % depth
            var seq = seqs[j % depth]
            # Send completion protects this arena before the remote gathers
            # reuse it; the next chunk reuses it only after those consumers.
            ib_enqueue_wait(state.ib, state.ctx, stream, seq)
            # The peers' exchanges retire at different times: wait for them
            # on one SM (`rank_gate`), not in the first remote gather's
            # start barrier with the whole grid resident.
            state.generation += 1
            rank_gate(
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                state.regions,
                ERR_ALLGATHER_SYNC,
                state.generation,
            )
            var slot = 0
            for node in range(state.nnodes):
                if node == state.my_node:
                    continue
                _allgather_node_mapped(
                    state,
                    stream,
                    arena,
                    node,
                    state.owned_base
                    + _inbox_base(state, seq)
                    + slot * slot_bytes,
                    recvbuff + off,
                    count,
                    per_rank_bytes,
                    0,
                    slot == 0,
                )
                slot += 1
            ib_note_consumed(state.ib, seq)


def _allgather_multinode(
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    per_rank_bytes: Int,
) raises:
    """Exchange one contribution per NIC, disseminate on the receiving node."""
    comptime if has_nvidia_gpu_accelerator():
        _allgather_multinode_mapped(
            state, stream, raw_stream, sendbuff, recvbuff, per_rank_bytes
        )
        return
    var lw = state.local_world
    var npeers = ib_npeers(state.ib)
    var block_stage = state.owned_base + _net_stage_off(state)
    var max_bytes = (
        min(
            _net_stage_bytes(state) // lw,
            min(
                allgather_max_bytes(state.arena_cap, lw),
                _inbox_group_bytes(state) // npeers,
            ),
        )
        // 16
        * 16
    )
    if max_bytes < 16:
        raise Error("mojoccl: allgather staging cannot hold one vector")
    var done = 0
    var chunk_index = 0
    var nchunks = (per_rank_bytes + max_bytes - 1) // max_bytes
    while done < per_rank_bytes:
        var chunk = min(max_bytes, per_rank_bytes - done)
        state.generation += 1
        allgather(
            state.ctx,
            stream,
            state.local_rank,
            lw,
            _arena_regions(state, 0),
            sendbuff + done,
            block_stage,
            chunk,
            state.arena_cap,
            state.generation,
            stride_bytes=chunk,
        )
        var seq = ib_next_seq(state.ib)
        var slot_bytes = _align_up(chunk, 16)
        var inbox_base = _inbox_base(state, seq)
        if npeers * slot_bytes > _inbox_group_bytes(state):
            raise Error("mojoccl: allgather inbox does not fit")
        ib_enqueue_request(
            state.ib,
            state.driver,
            state.ctx,
            stream,
            Int(raw_stream),
            block_stage + state.local_rank * chunk,
            chunk,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            state.owned_base + inbox_base,
            seq,
            OP_ALLGATHER,
            chunk_index,
            nchunks,
            chunk,
        )
        _place_node_block(
            state,
            stream,
            recvbuff,
            block_stage,
            state.my_node,
            chunk,
            done,
            per_rank_bytes,
        )
        # Completion includes sends: staging is now safe to overwrite.
        ib_enqueue_wait(state.ib, state.ctx, stream, seq)
        var slot = 0
        for node in range(state.nnodes):
            if node == state.my_node:
                continue
            state.generation += 1
            allgather(
                state.ctx,
                stream,
                state.local_rank,
                lw,
                _arena_regions(state, 0),
                state.owned_base + inbox_base + slot * slot_bytes,
                block_stage,
                chunk,
                state.arena_cap,
                state.generation,
                stride_bytes=chunk,
            )
            _place_node_block(
                state,
                stream,
                recvbuff,
                block_stage,
                node,
                chunk,
                done,
                per_rank_bytes,
            )
            slot += 1
        ib_note_consumed(state.ib, seq)
        done += chunk
        chunk_index += 1


def _place_node_block(
    mut state: CommState,
    stream: DeviceStream,
    recvbuff: Int,
    src: Int,
    node: Int,
    chunk: Int,
    done: Int,
    per_rank_bytes: Int,
) raises:
    var offs = StaticTuple[Int64, MAX_WORLD](fill=0)
    for l in range(state.local_world):
        offs[l] = Int64(
            state.rank_at[node * state.local_world + l] * per_rank_bytes + done
        )
    place_blocks(
        state.ctx, stream, recvbuff, src, offs, chunk, state.local_world
    )


# ---------------------------------------------------------------------------
# Not implemented: Reduce and point-to-point operations (+
# barrier, which routes to gloo -- see process_group.py). Returning
# ncclInvalidUsage rather than silently mis-computing is the point.
# ---------------------------------------------------------------------------


@export
def ncclReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE


@export
def ncclReduceScatter(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _dtype_item_bytes(datatype)
        if item == 0 or count < 0:
            return NCCL_INVALID_ARGUMENT
        if op != NCCL_SUM and op != NCCL_AVG:
            return NCCL_INVALID_USAGE
        if op == NCCL_AVG and (
            datatype == NCCL_INT32 or datatype == NCCL_INT64
        ):
            return NCCL_INVALID_USAGE
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _reduce_scatter_locked(
                comm, sendbuff, recvbuff, Int(count), datatype, op, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclReduceScatter failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclReduceScatter failed:", e)
        return NCCL_INTERNAL_ERROR


def _reduce_scatter_locked(
    comm: Int64,
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int,
    datatype: Int32,
    op: Int32,
    stream: Int64,
) raises -> Int32:
    """Push/reduce locally, then exchange destination shards across nodes."""
    ref state = _comm_ptr(comm)[]
    if state.aborted:
        return NCCL_INVALID_USAGE
    var latched = _latched_error(state)
    if latched != NCCL_SUCCESS:
        return latched
    if state.order_incomplete:
        return NCCL_REMOTE_ERROR
    # Order before the first launch: the previous collective may have produced
    # sendbuff on another stream. The exported wrapper records completion
    # after every chunk, even when count is zero.
    _order_before(state, stream)
    ref s = state.stream_cache[stream]
    if count == 0:
        return NCCL_SUCCESS
    if state.nnodes == 1:
        var scale = Float32(1.0)
        if op == NCCL_AVG:
            scale = Float32(1.0) / Float32(state.world)
        if datatype == NCCL_INT32:
            _do_reduce_scatter[DType.int32](
                state, s, Int(sendbuff), Int(recvbuff), count, scale
            )
        elif datatype == NCCL_INT64:
            _do_reduce_scatter[DType.int64](
                state, s, Int(sendbuff), Int(recvbuff), count, scale
            )
        elif datatype == NCCL_FLOAT16:
            _do_reduce_scatter[DType.float16](
                state, s, Int(sendbuff), Int(recvbuff), count, scale
            )
        elif datatype == NCCL_FLOAT32:
            _do_reduce_scatter[DType.float32](
                state, s, Int(sendbuff), Int(recvbuff), count, scale
            )
        else:  # NCCL_BFLOAT16, ruled in by _dtype_item_bytes above
            _do_reduce_scatter[DType.bfloat16](
                state, s, Int(sendbuff), Int(recvbuff), count, scale
            )
        return NCCL_SUCCESS
    return _reduce_scatter_multinode(
        state, s, sendbuff, recvbuff, count, datatype, op, stream
    )


def _do_reduce_scatter[
    dtype: DType
](
    mut state: CommState,
    s: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
) raises:
    """One node's reduce-scatter, chunked only if the arena cannot hold it."""
    comptime item = size_of[dtype]()
    var max_count = reduce_scatter_max_count(
        state.cap_bytes, state.local_world, item
    )
    var done = 0
    while done < count:
        var chunk = min(max_count, count - done)
        state.generation += 1
        reduce_scatter[dtype](
            state.ctx,
            s,
            state.local_rank,
            state.local_world,
            state.regions,
            sendbuff + done * item,
            recvbuff + done * item,
            chunk,
            state.cap_bytes,
            scale,
            state.generation,
            in_stride=count,
        )
        done += chunk


def _reduce_scatter_multinode(
    mut state: CommState,
    s: DeviceStream,
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int,
    datatype: Int32,
    op: Int32,
    stream: Int64,
) raises -> Int32:
    var scale = Float32(1.0)
    if op == NCCL_AVG:
        scale /= Float32(state.world)
    if datatype == NCCL_INT32:
        _do_reduce_scatter_nodes[DType.int32](
            state, s, stream, Int(sendbuff), Int(recvbuff), count, scale
        )
    elif datatype == NCCL_INT64:
        _do_reduce_scatter_nodes[DType.int64](
            state, s, stream, Int(sendbuff), Int(recvbuff), count, scale
        )
    elif datatype == NCCL_FLOAT16:
        _do_reduce_scatter_nodes[DType.float16](
            state, s, stream, Int(sendbuff), Int(recvbuff), count, scale
        )
    elif datatype == NCCL_FLOAT32:
        _do_reduce_scatter_nodes[DType.float32](
            state, s, stream, Int(sendbuff), Int(recvbuff), count, scale
        )
    else:
        _do_reduce_scatter_nodes[DType.bfloat16](
            state, s, stream, Int(sendbuff), Int(recvbuff), count, scale
        )
    return NCCL_SUCCESS


def _reduce_scatter_partial(
    state: CommState, arena: Int, slot_bytes: Int
) -> Int:
    return (
        state.owned_base
        + arena * state.arena_stride
        + signal_bytes()
        + (state.local_world - 1) * state.nnodes * slot_bytes
    )


def _do_reduce_scatter_fused(
    mut state: CommState,
    stream: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
    chunk_elems: Int,
    nchunks: Int,
    depth: Int,
) raises:
    # Called only for fp32, after packed arena and inbox geometry validation.
    if count <= 0 or chunk_elems <= 0 or nchunks <= 0 or nchunks > WORK_SLOTS:
        raise Error("mojoccl: invalid fused reduce-scatter plan")
    var npeers = ib_npeers(state.ib)
    var group = _inbox_group_bytes(state)
    if npeers * _align_up(min(chunk_elems, count) * 4, 16) > group:
        raise Error("mojoccl: fused reduce-scatter inbox overflow")
    var cap = (
        RS_FUSED_BIG_BLOCKS if count * 4 >= PIPE_SPLIT_UNIT else state.fused_cap
    )
    var blocks = reduce_scatter_fused_blocks(
        state.ctx,
        state.local_world,
        state.sm_count,
        chunk_elems,
        cap,
    )
    var rank_ids = reduce_scatter_rank_ids(state.rank_at)
    var g0 = state.generation + 1
    state.generation += nchunks
    var seq0 = ib_reserve_seqs(state.ib, nchunks)
    for k in range(nchunks):
        var cnt = min(chunk_elems, count - k * chunk_elems)
        var slot = _align_up(cnt * 4, 16)
        var seq = seq0 + k
        var inbox_base = _inbox_base(state, seq)
        ib_prepare_request(
            state.ib,
            _reduce_scatter_partial(state, k % state.narenas, slot),
            cnt * 4,
            inbox_base,
            slot,
            True,
            npeers,
            state.owned_base + inbox_base,
            seq,
            OP_REDUCE_SCATTER,
            k,
            nchunks,
            cnt,
            send_node_stride=slot,
        )
    try:
        rank_gate(
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            state.regions,
            ERR_REDUCE_SCATTER_SYNC,
            g0,
        )
        reduce_scatter_fused(
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            state.regions,
            sendbuff,
            recvbuff,
            ib_mailbox_dev(state.ib),
            count,
            chunk_elems,
            nchunks,
            depth,
            state.narenas,
            state.arena_stride,
            seq0,
            state.net_off + state.cap_bytes // 2,
            group,
            state.nslots,
            npeers,
            state.nnodes,
            state.my_node,
            g0,
            scale,
            blocks,
            spin_timeout_ns(),
            rank_ids,
        )
    except e:
        _latch_host_fault(state, ERR_HOST_LAUNCH, seq0)
        raise e


def _do_reduce_scatter_stream(
    mut state: CommState,
    stream: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
    chunk_elems: Int,
    nchunks: Int,
    depth: Int,
) raises:
    # Called only for fp32, after packed arena and inbox geometry validation.
    if count <= 0 or chunk_elems <= 0 or nchunks <= 0 or nchunks > WORK_SLOTS:
        raise Error("mojoccl: invalid streaming reduce-scatter plan")
    var npeers = ib_npeers(state.ib)
    var group = _inbox_group_bytes(state)
    if npeers * _align_up(min(chunk_elems, count) * 4, 16) > group:
        raise Error("mojoccl: streaming reduce-scatter inbox overflow")
    var cap = (
        RS_STREAM_BIG_BLOCKS if count * 4
        >= PIPE_SPLIT_UNIT else state.fused_cap
    )
    var blocks = reduce_scatter_stream_blocks(
        state.ctx, state.local_world, state.sm_count, chunk_elems, cap
    )
    var pieces = reduce_scatter_stream_pieces(chunk_elems, blocks)
    var rank_ids = reduce_scatter_rank_ids(state.rank_at)
    var g0 = state.generation + 1
    # The handoff counters run from `g0*PHASES_PER_GEN + 1` to
    # `+ nchunks*pieces` and are never reset, so the call reserves that many.
    state.generation += reduce_scatter_stream_generations(nchunks, pieces[1])
    var seq0 = ib_reserve_seqs(state.ib, nchunks)
    # One layout for every chunk of the call: the kernel's FREE credit is per
    # block index and only covers the peers' same-indexed block, which is
    # sound exactly while a short last chunk cannot re-cut the arena. The
    # payload it sends is still its own `cnt`.
    var slot = _align_up(chunk_elems * 4, 16)
    for k in range(nchunks):
        var cnt = min(chunk_elems, count - k * chunk_elems)
        var seq = seq0 + k
        var inbox_base = _inbox_base(state, seq)
        ib_prepare_request(
            state.ib,
            _reduce_scatter_partial(state, k % state.narenas, slot),
            cnt * 4,
            inbox_base,
            slot,
            True,
            npeers,
            state.owned_base + inbox_base,
            seq,
            OP_REDUCE_SCATTER,
            k,
            nchunks,
            cnt,
            send_node_stride=slot,
        )
    try:
        rank_gate(
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            state.regions,
            ERR_REDUCE_SCATTER_SYNC,
            g0,
        )
        reduce_scatter_stream(
            state.ctx,
            stream,
            state.local_rank,
            state.local_world,
            state.regions,
            sendbuff,
            recvbuff,
            ib_mailbox_dev(state.ib),
            count,
            chunk_elems,
            nchunks,
            depth,
            state.narenas,
            state.arena_stride,
            seq0,
            state.net_off + state.cap_bytes // 2,
            group,
            state.nslots,
            npeers,
            state.nnodes,
            state.my_node,
            g0,
            scale,
            blocks,
            pieces[0],
            pieces[1],
            spin_timeout_ns(),
            rank_ids,
        )
    except e:
        _latch_host_fault(state, ERR_HOST_LAUNCH, seq0)
        raise e


def _do_reduce_scatter_nodes[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
) raises:
    comptime item = size_of[dtype]()
    var npeers = ib_npeers(state.ib)
    # Reduced destinations follow the push slots in the combined arena.
    # Inbox groups are disjoint from both and credit protected.
    var slot_cap = (
        min(
            reduce_scatter_nodes_max_count(
                state.arena_cap, state.local_world, item, state.nnodes
            )
            * item,
            _inbox_group_bytes(state) // npeers,
        )
        // 16
        * 16
    )
    if slot_cap < 16:
        raise Error("mojoccl: reduce-scatter staging cannot hold one vector")
    var chunk_elems = min(count, slot_cap // item)
    var nchunks = (count + chunk_elems - 1) // chunk_elems
    var depth = min(state.narenas, nchunks)
    comptime if dtype == DType.float32 and has_nvidia_gpu_accelerator():
        if state.fused:
            comptime if RS_STREAM_ENABLED:
                if reduce_scatter_stream_wanted(count, PIPE_SPLIT_UNIT):
                    var plan = reduce_scatter_stream_plan(
                        count,
                        chunk_elems,
                        state.narenas,
                        PIPE_SPLIT_UNIT,
                    )
                    if plan[1] <= WORK_SLOTS:
                        _do_reduce_scatter_stream(
                            state,
                            stream,
                            sendbuff,
                            recvbuff,
                            count,
                            scale,
                            plan[0],
                            plan[1],
                            plan[2],
                        )
                        return
            var plan = reduce_scatter_fused_plan(
                count,
                chunk_elems,
                state.narenas,
                PIPE_SPLIT_UNIT,
            )
            if plan[1] <= WORK_SLOTS:
                _do_reduce_scatter_fused(
                    state,
                    stream,
                    sendbuff,
                    recvbuff,
                    count,
                    scale,
                    plan[0],
                    plan[1],
                    plan[2],
                )
                return
    var rank_ids = reduce_scatter_rank_ids(state.rank_at)
    var g0 = state.generation + 1
    state.generation += nchunks
    var seqs = List[Int](length=depth, fill=0)
    for k in range(nchunks + depth - 1):
        if k < nchunks:
            var off = k * chunk_elems
            var cnt = min(chunk_elems, count - off)
            var slot = _align_up(cnt * item, 16)
            var partials = _reduce_scatter_partial(
                state, k % state.narenas, slot
            )
            reduce_scatter_nodes[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                _arena_regions(state, k % state.narenas),
                sendbuff + off * item,
                partials,
                cnt,
                state.arena_cap,
                scale,
                g0 + k,
                count,
                rank_ids,
                state.nnodes,
            )
            var seq = ib_next_seq(state.ib)
            var inbox_base = _inbox_base(state, seq)
            ib_enqueue_request(
                state.ib,
                state.driver,
                state.ctx,
                stream,
                Int(raw_stream),
                partials,
                cnt * item,
                inbox_base,
                slot,
                True,
                npeers,
                state.owned_base + inbox_base,
                seq,
                OP_REDUCE_SCATTER,
                k,
                nchunks,
                cnt,
                send_node_stride=slot,
            )
            seqs[k % depth] = seq
        var j = k - (depth - 1)
        if j >= 0:
            var off = j * chunk_elems
            var cnt = min(chunk_elems, count - off)
            var slot = _align_up(cnt * item, 16)
            var seq = seqs[j % depth]
            ib_enqueue_wait(state.ib, state.ctx, stream, seq)
            inbox_sum_out[dtype](
                state.ctx,
                stream,
                recvbuff + off * item,
                _reduce_scatter_partial(state, j % state.narenas, slot)
                + state.my_node * slot,
                state.owned_base + _inbox_base(state, seq),
                cnt,
                slot,
                npeers,
            )
            ib_note_consumed(state.ib, seq)


@export
def ncclSend(
    sendbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE


@export
def ncclRecv(
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE
