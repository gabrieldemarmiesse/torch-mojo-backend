# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/init.cc
#
# The region comes in two shapes, chosen once per communicator in round 1 of
# the bootstrap and identical on every rank:
#
#   default   `cuMemAlloc` / `hipExtMallocWithFlags`, shared with legacy IPC
#             (transport/p2p.mojo). Every node count, every vendor.
#   NVLS      single node, every local device reporting
#             CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED: VMM memory bound to a
#             per-node multicast object and mapped twice (transport/nvls.mojo), so that
#             allreduces of NVLS_MIN_BYTES or more can go through the
#             NVSwitch's own reduction engine (device/all_reduce.mojo) instead of
#             over unicast NVLink. Peers are imported through the same
#             file-descriptor exchange rather than with cuIpcOpenMemHandle,
#             which cannot open VMM memory. The unicast kernels see no

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceAttribute
from std.os import getenv
from std.memory.alloc import unsafe_alloc
from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns, sleep

from tmb.ccl.bootstrap import (
    BootstrapConn,
    bootstrap_allgather,
    bootstrap_barrier,
    bootstrap_connect,
    decode_id,
    make_unique_id,
)
from tmb.ccl.device.all_reduce import nvls_available, nvls_blocks
from tmb.ccl.device.common import _enqueue_cached
from tmb.ccl.device.symmetric.all_reduce_gin import (
    FUSED_THREADS,
    fused_big_block_cap,
    fused_big_bytes,
    fused_block_cap,
    fused_resident_blocks,
)
from tmb.ccl.env_vars import MOJOCCL_BOOTSTRAP_TIMEOUT_S, MOJOCCL_REGION_MB
from tmb.ccl.include.comm import (
    CommState,
    INBOX_SLOTS,
    PIPE_SPLIT_UNIT,
    _align_up,
    _comm_ptr,
    _fault_code,
    _host_fault_word,
    _lock,
    _raise_abort_word,
    _read_error_word,
    _report_fault,
    _submission_failed,
    _try_lock,
    _unlock,
    region_layout,
)
from tmb.ccl.include.device import (
    BLOCK,
    MAX_WORLD,
    STATUS_PAGE_BYTES,
    _AMD,
    _COPY_MAX_BLOCKS,
    _GFX942,
    _SIGNAL_BYTES,
    _STATUS_PTR_OFFSET,
    signal_bytes,
)
from tmb.ccl.include.plugin.nccl_net import MAX_NODES
from tmb.ccl.include.transport import NvlsRegion
from tmb.ccl.misc.cudawrap import (
    alloc_host,
    current_device_ordinal,
    direct_managed_mem_access,
    free_host,
    host_device_ptr,
    open_driver,
)
from tmb.ccl.misc.strongstream import stream_done
from tmb.ccl.misc.utils import _any, host_hash
from tmb.ccl.nccl import (
    NCCL_INTERNAL_ERROR,
    NCCL_INVALID_ARGUMENT,
    NCCL_INVALID_USAGE,
    NCCL_IN_PROGRESS,
    NCCL_REMOTE_ERROR,
    NCCL_SUCCESS,
    NCCL_SYSTEM_ERROR,
    NCCL_UNHANDLED_CUDA_ERROR,
    UID_BYTES,
)
from tmb.ccl.os.linux_ipcsocket import (
    _socket_dir,
    scm_bind,
    scm_unbind,
    socket_path,
)
from tmb.ccl.proxy import ib_setup, ib_signal_abort, ib_teardown
from tmb.ccl.transport.multicast import nvls_create_and_share
from tmb.ccl.transport.net import (
    IB_BLOB_BYTES,
    ib_connect,
    ib_error,
    ib_local_info,
    ib_set_abort_word,
    ib_uses_proxy,
)
from tmb.ccl.transport.nvls import (
    NVLS_FD_TIMEOUT_S,
    _nvls_enabled,
    _nvls_min_bytes,
    _nvls_recommended_granularity,
    multicast_capable,
    multicast_granularity,
    nvls_bind_and_map,
    nvls_teardown,
    sm_count,
)
from tmb.ccl.transport.p2p import (
    HANDLE_BYTES,
    alloc_region,
    close_handle,
    free_region,
    get_handle,
    open_handle,
)


# ===-------------------------------------------------------------------=== #
# Topology, derived identically on every rank from the round-1 table
# ===-------------------------------------------------------------------=== #


struct Topology(Movable):
    var nnodes: Int
    var local_world: Int
    var my_node: Int
    var my_local_rank: Int
    var node_of: List[Int]  # global rank -> node index
    var local_rank_of: List[Int]  # global rank -> local rank
    var rank_at: List[Int]  # node * local_world + local_rank -> global rank

    def __init__(out self, nranks: Int):
        self.nnodes = 1
        self.local_world = nranks
        self.my_node = 0
        self.my_local_rank = 0
        self.node_of = List[Int](length=nranks, fill=0)
        self.local_rank_of = List[Int](length=nranks, fill=0)
        self.rank_at = List[Int](length=nranks, fill=0)


def derive_topology(host_hashes: List[UInt64], rank: Int) raises -> Topology:
    """Node index = order of first appearance scanning ranks 0..n-1; local
    rank = how many earlier ranks share the host. Pure function of the
    gathered table, so every rank computes the same answer with no round
    trip -- and the DDP invariant (equal ranks per node) is checked here,
    where the error message can name the offending node."""
    var n = len(host_hashes)
    var topo = Topology(n)
    var node_hash = List[UInt64]()
    topo.nnodes = 0
    for r in range(n):
        var h = host_hashes[r]
        var idx = -1
        for j in range(len(node_hash)):
            if node_hash[j] == h:
                idx = j
                break
        if idx < 0:
            idx = len(node_hash)
            node_hash.append(h)
        topo.node_of[r] = idx
    topo.nnodes = len(node_hash)
    var counts = List[Int](length=topo.nnodes, fill=0)
    for r in range(n):
        var nd = topo.node_of[r]
        topo.local_rank_of[r] = counts[nd]
        counts[nd] += 1
    topo.local_world = counts[0]
    for j in range(topo.nnodes):
        if counts[j] != topo.local_world:
            raise Error(
                "mojoccl: every node must contribute the same number of ranks"
                " (node 0 has "
                + String(topo.local_world)
                + ", node "
                + String(j)
                + " has "
                + String(counts[j])
                + "); launch with a uniform --nproc-per-node"
            )
    topo.my_node = topo.node_of[rank]
    topo.my_local_rank = topo.local_rank_of[rank]
    topo.rank_at = List[Int](length=topo.nnodes * topo.local_world, fill=0)
    for r in range(n):
        topo.rank_at[
            topo.node_of[r] * topo.local_world + topo.local_rank_of[r]
        ] = r
    return topo^


# ===-------------------------------------------------------------------=== #
# region_init
# ===-------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_zero_signal_area")
def _zero_kernel(ptr: Pointer[UInt32, MutAnyOrigin], nwords: Int32):
    var i = Int(global_idx.x)
    if i < Int(nwords):
        ptr[unsafe_offset=i] = 0


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_store_u64")
def _store_u64_kernel(dst: Pointer[UInt64, MutAnyOrigin], value: UInt64):
    if global_idx.x == 0:
        dst[unsafe_offset=0] = value


# ===-------------------------------------------------------------------=== #
# Public API
# ===-------------------------------------------------------------------=== #


def region_init(ctx: DeviceContext, region: Int) raises:
    """Zero the signal area of this rank's region and block until it is done.

    Called once per rank at communicator creation, on `ctx`'s own queue -- no
    `stream` parameter, unlike the collectives: this must be complete before
    any peer writes a flag into the area, and the caller only has to make the
    ranks meet (its own rendezvous) after calling it.
    """
    var words = _SIGNAL_BYTES // 4
    _enqueue_cached[_zero_kernel](
        ctx,
        ctx.stream(),
        "zero",
        (words + BLOCK - 1) // BLOCK,
        Pointer[UInt32, MutAnyOrigin](unsafe_from_address=region),
        Int32(words),
    )
    ctx.synchronize()


def install_status_page(ctx: DeviceContext, region: Int, dev_addr: Int) raises:
    """Publish the device mapping of the communicator's pinned status page in
    this region's header, and block until it has landed.

    The spins read it out of the region rather than take it as a kernel
    argument because the launcher signatures below are fixed; the region is
    raw driver memory, so the host cannot poke the slot directly. Called once
    per arena, right after `region_init` zeroed it and before any peer can
    reach the region.
    """
    _enqueue_cached[_store_u64_kernel](
        ctx,
        ctx.stream(),
        "statusptr",
        1,
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=region + _STATUS_PTR_OFFSET
        ),
        UInt64(dev_addr),
    )
    ctx.synchronize()


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


comptime RCCL_APU_NODE_CTAS = 24
"""Grid of a multi-node collective's node-local kernels on an APU (MI300A):
the all-gather's gathers and the reduce-scatter's node reduce. RCCL 2.22.3
forces 24 channels (4 rings x 6, one 256-thread CTA each) on a gfx942 whose
host accesses its managed memory directly (an APU) when there is more than
one node (rccl `src/init.cc:1339-1346`, `src/device/device.h:74`).
Measured on 2x4 MI300A, Adastra job 5447705, GPT-2 XL FSDP2 bf16, ABBA legs,
tok/s: gathers at 432/96/24 blocks 20.9k/22.0k/23.0k (mojo+RCCL 23.1k), node
reduce at 128 -> 24 blocks 22.0k -> 23.2k. In isolation 96 blocks is the
faster gather (XL bf16 block 599 vs 650 us) and the reduce is network-bound
(950 vs 951 us), so 24 wins end to end only: every CU a collective holds
beside the compute stream is one the GEMMs lose. A discrete gfx942 (MI300X,
MI325X) fails RCCL's test and keeps its caps; nothing was measured on one."""


def _node_grids(apu: Bool) -> Tuple[Int, Int]:
    """Grid caps of a multi-node all-gather's gathers and reduce-scatter's
    node reduce (0: the allreduce caps), chosen once at init. `apu`: RCCL's
    test, ANDed over the ranks in bootstrap round 2, since ranks on
    different grids would block-match different slices of a collective.
    Only gfx942 asks the question, as RCCL does."""
    if apu:
        return (RCCL_APU_NODE_CTAS, RCCL_APU_NODE_CTAS)
    comptime if _AMD:
        return (_COPY_MAX_BLOCKS, 0)  # the single-node copy cap
    return (AG_NODE_BLOCKS, 0)


# MI300A: 64 MiB supports four ranks/node without the large shared-memory
# reservation conflict (Adastra 124M measurements in agents_docs/distributed.md),
# and the GPT-2 XL five-round series at 0.9873x stock used it, job 5417296,
# 2026-09-15. NVIDIA retains its H100 staging fit of 256 MiB.
comptime DEFAULT_REGION_MB = 64 if _GFX942 else 256
comptime DEFAULT_BOOTSTRAP_TIMEOUT_S: Float64 = 120.0


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


# ---------------------------------------------------------------------------
# Version / error string / unique id
# ---------------------------------------------------------------------------


def ncclGetVersion(version: Pointer[Int32, MutAnyOrigin]) -> Int32:
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


def ncclGetErrorString(
    result: Int32,
) -> Pointer[UInt8, ImmStaticOrigin]:
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


def ncclGetUniqueId(uid_out: Pointer[UInt8, MutAnyOrigin]) -> Int32:
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
) -> Int32:
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
    # rendezvous carries no extra round for them (os/linux_ipcsocket.mojo's `socket_path`).
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
    # query answers on both vendors (the driver-binding one,
    # `transport.nvls.sm_count`, is NVIDIA-only and reads 0 on AMD, which would
    # leave AMD on the split schedule for no reason); the binding is the
    # fallback.
    var device_sms: Int
    try:
        device_sms = Int(
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        )
    except:
        device_sms = 0
    if device_sms <= 0:
        device_sms = sm_count(lib, ordinal)
    var apu = False
    comptime if _GFX942:  # RCCL's multi-node APU rule is gfx942's
        apu = direct_managed_mem_access(lib, ordinal)
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
    # mapping only device/all_reduce.mojo ever touches.
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
        # what it says because transport/nvls.mojo asks for the MINIMUM granularity
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
        # device/symmetric/ are fixed, so that slot is how the kernels
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
        comptime CFG_BYTES = 96
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
        # RCCL's APU test, which picks the multi-node grids (`_node_grids`).
        # The query is rank-local and a rank whose driver call fails answers
        # "discrete" alone, while ranks on different grids block-match
        # different slices of a collective. So, like the NVLS column of round
        # 1, every rank ANDs the column: one "discrete" answer puts the whole
        # communicator on the discrete-GPU grids.
        cfg[unsafe_offset=11] = Int64(1 if apu else 0)
        var t2 = unsafe_alloc[UInt8](BLOB2 * nranks)
        bootstrap_allgather(conn, _any(b2), BLOB2, _any(t2), timeout_s)
        var not_apu_rank = -1
        for r in range(nranks):
            var rcfg = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=Int(t2)
                + r * BLOB2
                + HANDLE_BYTES
                + IB_BLOB_BYTES
            )
            if rcfg[unsafe_offset=11] == 0 and not_apu_rank < 0:
                not_apu_rank = r
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
        if apu and not_apu_rank >= 0:
            apu = False
            print(
                "mojoccl: rank",
                rank,
                "is on an APU but rank",
                not_apu_rank,
                (
                    "reported a discrete GPU (or could not query it); every"
                    " rank uses the discrete-GPU collective grids"
                ),
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
    var node_grids = _node_grids(apu)
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
        ag_node_blocks=node_grids[0],
        rs_node_blocks=node_grids[1],
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


def ncclCommDestroy(comm: Int64) -> Int32:
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


def ncclCommAbort(comm: Int64) -> Int32:
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


def ncclCommGetAsyncError(
    comm: Int64, err_out: Pointer[Int32, MutAnyOrigin]
) -> Int32:
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


def ncclCommCount(
    comm: Int64, count_out: Pointer[Int32, MutAnyOrigin]
) -> Int32:
    ref state = _comm_ptr(comm)[]
    count_out[] = Int32(state.world)
    return NCCL_SUCCESS


def ncclCommUserRank(
    comm: Int64, rank_out: Pointer[Int32, MutAnyOrigin]
) -> Int32:
    ref state = _comm_ptr(comm)[]
    rank_out[] = Int32(state.rank)
    return NCCL_SUCCESS
