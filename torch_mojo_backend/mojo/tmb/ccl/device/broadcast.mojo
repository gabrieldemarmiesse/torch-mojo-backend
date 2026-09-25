# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/broadcast.h
#
# Broadcast is scatter + all-gather, not "root stages, everyone reads": the
# latter puts (world-1) x nbytes on the root's one outbound link and measured
# 3.6x slower.  Allgather is a local stage + a peer gather, which is already
# the unicast minimum.

from std.collections import Array
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream

from tmb.ccl.device.common import (
    _enqueue_cached,
    device_now_ns,
    spin_timeout_ns,
)
from tmb.ccl.device.symmetric.data_ops import _copy_bytes
from tmb.ccl.device.symmetric.primitives import (
    _check_common,
    _gather_slot,
    _peer_step,
    _peer_step0,
    _region_ptrs,
    _vcount,
    _vstart,
)
from tmb.ccl.include.device import (
    BLOCK,
    ERR_BROADCAST_SYNC,
    MAX_WORLD,
    _AMD,
    _COPY_MAX_BLOCKS,
    _SIGNAL_BYTES,
    _UNROLL,
)
from tmb.ccl.include.nccl_device.lsa_barrier import _flag_target, _sync


# ===-------------------------------------------------------------------=== #
# Broadcast and allgather -- one peer-copy kernel each, chunked to `cap`
# ===-------------------------------------------------------------------=== #
#
# Both walk the buffer in chunks of `cap` bytes and alternate between the two
# cap-sized halves of the arena, so the write of chunk c races only against
# reads of chunk c-1 (the other half); reads of chunk c-2 are ordered before
# it by the sync of chunk c-1. A leading sync separates the first two chunks
# from the previous call's reads.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_broadcast_scatter_gather_bytes")
def _bcast_kernel[
    U: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    send: Pointer[UInt8, MutAnyOrigin],
    recv: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    root_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
):
    """Broadcast as scatter + all-gather.

    The obvious design -- root stages the whole message, everybody reads it --
    puts `(world-1) * nbytes` on the root's single outbound link and measured
    552 us on 27 MiB where the fabric could do 150. Instead the root scatters
    shard p into rank p's stage (nbytes out of the root, spread over the peers)
    and every other rank gathers the `world` shards, so the read side is a
    permutation too and no link carries more than nbytes.

    Out-of-place capable (ncclBroadcast): the root reads `send`, everyone
    writes `recv`, and the root copies `send` to `recv` locally when they
    differ (cheaper than gathering its own message back over NVLink).
    """
    var t0 = device_now_ns()
    var world = Int(world_i)
    var rank = Int(rank_i)
    var root = Int(root_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(nbytes)
    var stage_off = Int(stage_off_b)
    var nv = n // 16
    var q = nv // world
    var rem = nv % world
    var mtail = n - nv * 16

    if not _sync(
        regions, world, rank, ERR_BROADCAST_SYNC, flag_base, t0, timeout_ns
    ):
        return

    if rank == root:
        for i in range(world):
            var p = root + i
            if p >= world:
                p -= world
            var vc = _vcount(p, q, rem)
            _copy_bytes[U](
                regions[p].unsafe_offset(stage_off),
                send.unsafe_offset(_vstart(p, q, rem) * 16),
                vc * 16 + (mtail if p == world - 1 else 0),
                tid,
                stride,
            )
        if Int(send) != Int(recv):
            _copy_bytes[U](recv, send, n, tid, stride)

    if not _sync(
        regions, world, rank, ERR_BROADCAST_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    comptime if _AMD:
        # The gather half is a push too (module header, "Link direction").
        # After the scatter every rank holds its own shard in the scatter area
        # at `stage_off`; each then writes that shard into every OTHER
        # non-root rank's gather slot, and everyone assembles locally.  The
        # root needs nothing back -- it copied `send` to `recv` above -- so it
        # is skipped as a destination, which is also why `world-1` compacted
        # slots are enough.  The whole staging is `world` shard slots, i.e.
        # about `nbytes`, so the caller's chunking is unchanged.
        var sslot = (_vcount(0, q, rem) * 16 + mtail + 15) // 16 * 16
        var gbase = stage_off + sslot
        var my_vc = _vcount(rank, q, rem)
        var my_bytes = my_vc * 16 + (mtail if rank == world - 1 else 0)
        var mine = regions[rank].unsafe_offset(stage_off)
        if rank != root:
            _copy_bytes[U](
                recv.unsafe_offset(_vstart(rank, q, rem) * 16),
                mine,
                my_bytes,
                tid,
                stride,
            )
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            if p == root:
                continue
            _copy_bytes[U](
                regions[p].unsafe_offset(gbase + sslot * _gather_slot(rank, p)),
                mine,
                my_bytes,
                tid,
                stride,
            )
        if not _sync(
            regions,
            world,
            rank,
            ERR_BROADCAST_SYNC,
            flag_base + 2,
            t0,
            timeout_ns,
        ):
            return
        if rank != root:
            for i in range(1, world):
                var p = rank + _peer_step(i, world)
                if p >= world:
                    p -= world
                var vc = _vcount(p, q, rem)
                _copy_bytes[U](
                    recv.unsafe_offset(_vstart(p, q, rem) * 16),
                    regions[rank].unsafe_offset(
                        gbase + sslot * _gather_slot(p, rank)
                    ),
                    vc * 16 + (mtail if p == world - 1 else 0),
                    tid,
                    stride,
                )
        return

    if rank != root:
        for i in range(world):
            var p = rank + _peer_step0(i, world)
            if p >= world:
                p -= world
            var vc = _vcount(p, q, rem)
            _copy_bytes[U](
                recv.unsafe_offset(_vstart(p, q, rem) * 16),
                regions[p].unsafe_offset(stage_off),
                vc * 16 + (mtail if p == world - 1 else 0),
                tid,
                stride,
            )


def broadcast(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    root: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    send_ptr: Int,
    recv_ptr: Int,
    nbytes: Int,
    cap_bytes: Int,
    generation: Int,
) raises:
    """Every rank's `recv` <- `root`'s `send`, dtype-agnostic, on `stream`.

    Out-of-place capable: `send_ptr` and `recv_ptr` may differ (only the root
    reads `send_ptr`). `nbytes` must be <= `cap_bytes`; the caller chunks
    anything larger.
    """
    _check_common(rank, world, cap_bytes, generation)
    if root < 0 or root >= world:
        raise Error("collectives: root out of range")
    if nbytes == 0:
        return
    if nbytes < 0:
        raise Error("collectives: nbytes must be >= 0")
    if nbytes > cap_bytes:
        raise Error("collectives: broadcast message exceeds cap_bytes")
    var rp = _region_ptrs(regions, rank, world)
    var blocks = min(
        _COPY_MAX_BLOCKS,
        max(1, (nbytes // 16 // world + BLOCK - 1) // BLOCK),
    )
    _enqueue_cached[_bcast_kernel[_UNROLL]](
        ctx,
        stream,
        "bcast",
        blocks,
        rp,
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=send_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=recv_ptr),
        Int64(nbytes),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        Int32(root),
        _flag_target(generation, 0),
        spin_timeout_ns(),
    )
