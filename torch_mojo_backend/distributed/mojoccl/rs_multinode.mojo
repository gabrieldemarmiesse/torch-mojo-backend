# Node-local partials for hierarchical reduce-scatter.
# Compacted push slots precede the node outputs in the 2*cap staging arena.
from std.collections import InlineArray
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from max.gpu.host import DeviceContext, DeviceStream
from std.sys import size_of
from std.utils import StaticTuple
from netutil import MAX_NODES
from collectives_kernels import (
    BLOCK,
    MAX_WORLD,
    ERR_REDUCE_SCATTER_SYNC,
    _SIGNAL_BYTES,
    _UNROLL,
    _AR_BIG_BLOCKS,
    _AR_BIG_BYTES,
    _AR_MAX_BLOCKS,
    _sync,
    _peer_step,
    _copy_span_flex,
    _share,
    _rs_slot,
    _rs_one,
    _enqueue_cached,
    _align_up,
    _flag_target,
    spin_timeout_ns,
    device_now_ns,
    _check_common,
    _region_ptrs,
)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_reduce_scatter_nodes_push_reduce_{dtype}_w{NW}")
def _rs_nodes_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int64,
    in_stride_e: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
    vector_ok: Int32,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    node_count: Int32,
):
    _ = _rs_nodes_body[dtype, W, U, NW, BLOCK, False](
        regions,
        in_ptr,
        out_ptr,
        count,
        in_stride_e,
        slot_stride_b,
        push_off_b,
        world_i,
        rank_i,
        flag_base,
        scale,
        timeout_ns,
        vector_ok,
        rank_ids,
        node_count,
        0,
    )


@always_inline
def _rs_nodes_body[
    dtype: DType, W: Int, U: Int, NW: Int, THREADS: Int, GATED: Bool
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int64,
    in_stride_e: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
    vector_ok: Int32,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    node_count: Int32,
    arena_off: Int,
) -> Bool:
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * THREADS
    var n = Int(count)
    var in_stride = Int(in_stride_e)
    var slot_stride = Int(slot_stride_b)
    var push_off = arena_off + Int(push_off_b)
    var vec = vector_ok != 0

    # --- phase 0: start barrier (the arena-reuse invariant) -----------------
    # A GATED caller ran `_gate_kernel` ahead of the whole grid instead.
    comptime if not GATED:
        if not _sync(
            regions,
            world,
            rank,
            ERR_REDUCE_SCATTER_SYNC,
            flag_base,
            t0,
            timeout_ns,
            arena_off,
        ):
            return False

    # Each node contributes one destination chunk per local rank.
    for d in range(Int(node_count)):
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
                .unsafe_bitcast[Scalar[dtype]]()
            )
            _copy_span_flex[dtype, W, U](
                dst,
                in_ptr.unsafe_offset(Int(rank_ids[d * world + s]) * in_stride),
                n,
                tid,
                stride,
                vec,
            )

    if not _sync(
        regions,
        world,
        rank,
        ERR_REDUCE_SCATTER_SYNC,
        flag_base + 1,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    for d in range(Int(node_count)):
        var dst = out_ptr.unsafe_offset(d * slot_stride // size_of[dtype]())
        var uin = in_ptr.unsafe_offset(
            Int(rank_ids[d * world + rank]) * in_stride
        )
        var slots = regions[rank].unsafe_offset(
            push_off + d * (world - 1) * slot_stride
        )
        var vc = n // W
        if vec:
            for v in range(tid, vc, stride):
                var acc = _share(
                    uin.unsafe_load[width=W, alignment=16](v * W).cast[accum](),
                    scale,
                )
                # Slot pointers are formed by arithmetic, never held in a stack
                # array: such an array is demoted to local memory (MOCO-1431).
                comptime if NW > 0:
                    comptime for j in range(1, NW):
                        var p = rank + j
                        if p >= NW:
                            p -= NW
                        acc += _share(
                            _rs_slot[dtype](slots, slot_stride, p, rank)
                            .unsafe_load[width=W, alignment=16](v * W)
                            .cast[accum](),
                            scale,
                        )
                else:
                    for j in range(1, world):
                        var p = rank + j
                        if p >= world:
                            p -= world
                        acc += _share(
                            _rs_slot[dtype](slots, slot_stride, p, rank)
                            .unsafe_load[width=W, alignment=16](v * W)
                            .cast[accum](),
                            scale,
                        )
                dst.unsafe_store[width=W, alignment=16](
                    v * W, acc.cast[dtype]()
                )
        else:
            for v in range(tid, vc, stride):
                comptime for e in range(W):
                    var k = v * W + e
                    dst[unsafe_offset=k] = _rs_one[dtype, accum](
                        uin, slots, slot_stride, world, rank, k, scale
                    )

        for i in range(tid, n - vc * W, stride):
            var k = vc * W + i
            dst[unsafe_offset=k] = _rs_one[dtype, accum](
                uin, slots, slot_stride, world, rank, k, scale
            )

    return True


def reduce_scatter_rank_ids(
    rank_at: List[Int],
) raises -> InlineArray[Int32, MAX_WORLD * MAX_NODES]:
    # Dynamic writes to large StaticTuple silently vanished in Mojo 1.0.
    if len(rank_at) > MAX_WORLD * MAX_NODES:
        raise Error("collectives: hierarchical rank map exceeds capacity")
    var ids = InlineArray[Int32, MAX_WORLD * MAX_NODES](fill=0)
    for i in range(len(rank_at)):
        ids[i] = Int32(rank_at[i])
    return ids^


def reduce_scatter_nodes_max_count(
    cap_bytes: Int, world: Int, elem_bytes: Int, node_count: Int
) -> Int:
    var slots = node_count * world
    return (2 * cap_bytes // slots // 16 * 16) // elem_bytes


def _launch_rs_nodes[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    count: Int,
    in_stride: Int,
    world: Int,
    rank: Int,
    blocks: Int,
    vec: Bool,
    scale: Float32,
    generation: Int,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    node_count: Int,
) raises:
    _enqueue_cached[_rs_nodes_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"rs_nodes_{dtype}_{NW}"),
        blocks,
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(count),
        Int64(in_stride),
        Int64(_align_up(count * size_of[dtype](), 16)),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
        Int32(1) if vec else Int32(0),
        rank_ids,
        Int32(node_count),
    )


def reduce_scatter_nodes[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    count: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
    in_stride: Int,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    node_count: Int,
) raises:
    """Reduce node-local contributions into node_count aligned output chunks."""
    _check_common(rank, world, cap_bytes, generation)
    if count == 0:
        return
    if count < 0 or in_stride < count:
        raise Error("collectives: invalid hierarchical reduce-scatter count")
    if node_count < 1 or node_count > MAX_NODES:
        raise Error("collectives: invalid hierarchical reduce-scatter nodes")
    comptime W = 16 // size_of[dtype]()
    comptime esize = size_of[dtype]()
    if count > reduce_scatter_nodes_max_count(
        cap_bytes, world, esize, node_count
    ):
        raise Error("collectives: hierarchical reduce-scatter exceeds arena")
    var rp = _region_ptrs(regions, rank, world)
    var vec = (in_ptr | out_ptr | (in_stride * esize)) % 16 == 0
    var cap_blocks = (
        _AR_BIG_BLOCKS if world * node_count * count * esize
        >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    var blocks = min(cap_blocks, max(1, (count // W + 1 + BLOCK - 1) // BLOCK))
    if world == 8:
        _launch_rs_nodes[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            in_stride,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
            rank_ids,
            node_count,
        )
    elif world == 4:
        _launch_rs_nodes[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            in_stride,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
            rank_ids,
            node_count,
        )
    elif world == 2:
        _launch_rs_nodes[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            in_stride,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
            rank_ids,
            node_count,
        )
    else:
        _launch_rs_nodes[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            in_stride,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
            rank_ids,
            node_count,
        )
