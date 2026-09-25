# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/all_gather.cuh

from std.collections import Array
from std.atomic import Atomic, Ordering, fence
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
    thread_idx,
)
from std.utils import StaticTuple
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceStream

from tmb.ccl.device.common import (
    _enqueue_cached,
    device_now_ns,
    spin_timeout_ns,
)
from tmb.ccl.device.symmetric.data_ops import _copy_bytes, _copy_bytes2
from tmb.ccl.device.symmetric.primitives import (
    _check_common,
    _gather_slot,
    _peer_step,
    _region_ptrs,
)
from tmb.ccl.include.device import (
    BLOCK,
    ERR_ALLGATHER_SYNC,
    MAX_BLOCKS,
    MAX_WORLD,
    _AG_ARRIVE_OFFSET,
    _AMD,
    _COPY_MAX_BLOCKS,
    _SIGNAL_BYTES,
    _UNROLL,
    _align_up,
)
from tmb.ccl.include.nccl_device.lsa_barrier import _flag_target, _sync


@always_inline
def _allgather_rank[
    MAPPED: Bool
](rank_at: Array[Int32, MAX_WORLD], rank: Int) -> Int:
    comptime if MAPPED:
        return Int(rank_at[rank])
    return rank


@always_inline
def _ag_release_to_nic(
    region: Pointer[UInt8, MutAnyOrigin],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
):
    """The last block to finish staging releases RDMA exchange `seq` to the
    proxy (acq_rel arrivals, release mailbox store: `_allgather_body`)."""
    comptime if _AMD:
        # Like _sync, flush every wave before the block rendezvous;
        # s_barrier alone does not drain AMD vector-memory stores.
        fence[ordering=Ordering.RELEASE]()
    barrier()
    if thread_idx.x == 0:
        var arrive = region.unsafe_offset(_AG_ARRIVE_OFFSET).unsafe_bitcast[
            UInt64
        ]()
        var was = Atomic[Scalar[DType.uint64]].fetch_add[
            ordering=Ordering.ACQUIRE_RELEASE
        ](arrive, UInt64(1))
        if Int(was) == Int(grid_dim.x) - 1:
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELAXED](
                arrive, UInt64(0)
            )
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                mb_req, seq
            )


@always_inline
def _allgather_body[
    U: Int, MAPPED: Bool, GATED: Bool
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[UInt8, MutAnyOrigin],
    out_ptr: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stride_b: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
    rank_at: Array[Int32, MAX_WORLD],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
):
    """Local stage + peer gather -- already the unicast minimum: `nbytes` of
    local copy and `(world-1)*nbytes` of cross-link traffic per GPU (peer
    reads on NVIDIA, peer writes on AMD -- see the branch below).

    Rank r's contribution lands at `out_ptr + r*stride_b`; `stride_b` is the
    output layout's true per-rank size, which differs from `nbytes` when the
    caller splits one rank's contribution across several calls.

    `seq != 0` (mapped): the staged contribution is also this chunk's RDMA
    payload, and the last block to finish staging stores `seq` into the
    proxy mailbox `mb_req`, so the NIC reads it while the local peer copies
    run instead of after them. Every arrival RMW is acquire-release, so the
    last arriver's mailbox store is ordered after every block's stage
    stores; the counter is reset by the last arriver and the next launch on
    this arena is stream-ordered behind this kernel. On NVIDIA the stage is
    the slot the peers pull from; on AMD, where the peers push into compact
    slots of this region, it is a separate slot after them
    (`allgather_nic_stage_off`).
    """
    var t0 = device_now_ns()
    var world = Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(nbytes)
    var out_stride = Int(stride_b)
    var stage_off = Int(stage_off_b)

    # A GATED caller ran `rank_gate` ahead of the grid instead: every peer
    # finished its previous stream work, which the AMD pushes rely on too.
    comptime if not GATED:
        if not _sync(
            regions, world, rank, ERR_ALLGATHER_SYNC, flag_base, t0, timeout_ns
        ):
            return

    comptime if _AMD:
        # Push instead of pull (module header, "Link direction"): my
        # contribution goes into every peer's slot for me and straight into my
        # own output slice, and the gather is then a local copy out of my own
        # region.  `world-1` compacted slots, addressed by `_gather_slot`, so
        # the staging is `(world-1) * nbytes` -- which is why
        # `allgather_max_bytes` chunks smaller here than on NVIDIA.
        var slot = (n + 15) // 16 * 16
        var own_output = out_ptr.unsafe_offset(
            _allgather_rank[MAPPED](rank_at, rank) * out_stride
        )
        var nic = False
        comptime if MAPPED:
            nic = seq != 0
        if nic:
            _copy_bytes2[U](
                regions[rank].unsafe_offset(
                    stage_off + allgather_nic_stage_off(world, n)
                ),
                own_output,
                in_ptr,
                n,
                tid,
                stride,
            )
            _ag_release_to_nic(regions[rank], mb_req, seq)
        else:
            _copy_bytes[U](own_output, in_ptr, n, tid, stride)
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            _copy_bytes[U](
                regions[p].unsafe_offset(
                    stage_off + slot * _gather_slot(rank, p)
                ),
                in_ptr,
                n,
                tid,
                stride,
            )
        if not _sync(
            regions,
            world,
            rank,
            ERR_ALLGATHER_SYNC,
            flag_base + 1,
            t0,
            timeout_ns,
        ):
            return
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            _copy_bytes[U](
                out_ptr.unsafe_offset(
                    _allgather_rank[MAPPED](rank_at, p) * out_stride
                ),
                regions[rank].unsafe_offset(
                    stage_off + slot * _gather_slot(p, rank)
                ),
                n,
                tid,
                stride,
            )
        return

    # One read of my contribution, two stores: my region (what the peers
    # read) and my own slice of the output.
    _copy_bytes2[U](
        regions[rank].unsafe_offset(stage_off),
        out_ptr.unsafe_offset(
            _allgather_rank[MAPPED](rank_at, rank) * out_stride
        ),
        in_ptr,
        n,
        tid,
        stride,
    )
    comptime if MAPPED:
        if seq != 0:
            _ag_release_to_nic(regions[rank], mb_req, seq)

    if not _sync(
        regions, world, rank, ERR_ALLGATHER_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    for i in range(1, world):
        var p = rank + _peer_step(i, world)
        if p >= world:
            p -= world
        _copy_bytes[U](
            out_ptr.unsafe_offset(
                _allgather_rank[MAPPED](rank_at, p) * out_stride
            ),
            regions[p].unsafe_offset(stage_off),
            n,
            tid,
            stride,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_allgather_bytes")
def _allgather_kernel[
    U: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[UInt8, MutAnyOrigin],
    out_ptr: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stride_b: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
):
    _allgather_body[U, False, False](
        regions,
        in_ptr,
        out_ptr,
        nbytes,
        stride_b,
        stage_off_b,
        world_i,
        rank_i,
        flag_base,
        timeout_ns,
        Array[Int32, MAX_WORLD](fill=0),
        regions[0].unsafe_bitcast[UInt64](),
        UInt64(0),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_allgather_mapped_bytes")
def _allgather_mapped_kernel[
    U: Int, GATED: Bool
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[UInt8, MutAnyOrigin],
    out_ptr: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stride_b: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
    rank_at: Array[Int32, MAX_WORLD],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
):
    _allgather_body[U, True, GATED](
        regions,
        in_ptr,
        out_ptr,
        nbytes,
        stride_b,
        stage_off_b,
        world_i,
        rank_i,
        flag_base,
        timeout_ns,
        rank_at,
        mb_req,
        seq,
    )


def allgather_max_bytes(
    cap_bytes: Int, world: Int, nic_stage: Bool = False
) -> Int:
    """Largest per-rank contribution one `allgather` call may carry.

    NVIDIA stages one message-sized buffer per rank in its own region and
    reads the peers', so `cap_bytes` is the bound and this is the identity.
    AMD pushes instead, which needs `world-1` message-sized slots inside the
    `2*cap_bytes` arena; the caller chunks to that. `nic_stage`: the call
    also stages a multi-node exchange's RDMA source, one more slot on AMD
    (`allgather_nic_stage_off`).
    """
    comptime if _AMD:
        var slots = world if nic_stage else world - 1
        if slots <= 1:
            return cap_bytes
        return min(cap_bytes, (2 * cap_bytes // slots) // 16 * 16)
    return cap_bytes


@always_inline
def allgather_nic_stage_off(world: Int, nbytes: Int) -> Int:
    """Offset of a mapped all-gather's RDMA source from its stage. NVIDIA: 0,
    the stage the peers pull from. AMD: past the `world-1` compact slots the
    peers push into, which they may be writing while the NIC reads. The
    kernel stages it and the host posts the read from it, so both use this.
    """
    comptime if _AMD:
        return (world - 1) * _align_up(nbytes, 16)
    return 0


def allgather(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    nbytes_per_rank: Int,
    cap_bytes: Int,
    generation: Int,
    stride_bytes: Int = -1,
) raises:
    """Gather: `out[r*stride_bytes ...]` <- rank r's input, on `stream`.

    `stride_bytes` defaults to `nbytes_per_rank` and is the output layout's
    true per-rank size, which differs when the caller splits one rank's
    contribution across several calls. `nbytes_per_rank` must be <=
    `cap_bytes`.
    """
    _check_common(rank, world, cap_bytes, generation)
    if nbytes_per_rank == 0:
        return
    if nbytes_per_rank < 0:
        raise Error("collectives: nbytes_per_rank must be >= 0")
    if nbytes_per_rank > allgather_max_bytes(cap_bytes, world):
        raise Error("collectives: allgather message exceeds cap_bytes")
    var stride = stride_bytes if stride_bytes >= 0 else nbytes_per_rank
    if stride < nbytes_per_rank:
        raise Error("collectives: stride_bytes < nbytes_per_rank")
    var rp = _region_ptrs(regions, rank, world)
    var blocks = min(
        _COPY_MAX_BLOCKS,
        max(1, (nbytes_per_rank // 16 + BLOCK - 1) // BLOCK),
    )
    _enqueue_cached[_allgather_kernel[_UNROLL]](
        ctx,
        stream,
        "allgather",
        blocks,
        rp,
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(nbytes_per_rank),
        Int64(stride),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        spin_timeout_ns(),
    )


def allgather_mapped[
    U: Int = _UNROLL, GATED: Bool = False
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    nbytes_per_rank: Int,
    cap_bytes: Int,
    generation: Int,
    stride_bytes: Int,
    rank_at: StaticTuple[Int32, MAX_WORLD],
    max_blocks: Int = _COPY_MAX_BLOCKS,
    mb_req: Int = 0,
    seq: Int = 0,
) raises:
    """Gather local ranks directly into their mapped global output slots.

    `seq != 0`: release RDMA exchange `seq` through the mailbox at `mb_req`
    once the contribution is staged (see `_allgather_body`). `GATED`: the
    caller ran `rank_gate` just before, so the kernel skips its start
    barrier."""
    _check_common(rank, world, cap_bytes, generation)
    if nbytes_per_rank == 0:
        return
    if nbytes_per_rank < 0:
        raise Error("collectives: nbytes_per_rank must be >= 0")
    if nbytes_per_rank > allgather_max_bytes(
        cap_bytes, world, nic_stage=seq != 0
    ):
        raise Error("collectives: allgather message exceeds cap_bytes")
    var stride = stride_bytes if stride_bytes >= 0 else nbytes_per_rank
    if stride < nbytes_per_rank:
        raise Error("collectives: stride_bytes < nbytes_per_rank")
    if max_blocks < 1 or max_blocks > MAX_BLOCKS:
        raise Error("collectives: allgather grid cap out of range")
    var rp = _region_ptrs(regions, rank, world)
    var blocks = min(
        max_blocks,
        max(1, (nbytes_per_rank // 16 + BLOCK - 1) // BLOCK),
    )
    var ranks = Array[Int32, MAX_WORLD](fill=0)
    for i in range(world):
        ranks[i] = rank_at[i]
    _enqueue_cached[_allgather_mapped_kernel[U, GATED]](
        ctx,
        stream,
        String(t"allgather_mapped_u{U}_g{GATED}"),
        blocks,
        rp,
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(nbytes_per_rank),
        Int64(stride),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        spin_timeout_ns(),
        ranks,
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=mb_req if seq != 0 else regions[rank]
        ),
        UInt64(seq),
    )
