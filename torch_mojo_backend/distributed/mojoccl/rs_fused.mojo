# Hierarchical fp32 reduce-scatter, including every pipelined chunk, in one launch.
from std.atomic import Atomic, Ordering
from std.collections import InlineArray
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream
from netutil import MAX_NODES
from rs_multinode import _rs_nodes_body
from internode import WORK_SLOTS
from internode_fused import (
    _await_exchange,
    _give_up,
    _GRID_GRACE_NS,
    FUSED_THREADS,
)
from collectives_kernels import (
    MAX_WORLD,
    MAX_BLOCKS,
    PHASES_PER_GEN,
    ERR_FUSED_GRID,
    _SIGNAL_BYTES,
    _align_up,
    _region_ptrs,
    _cached_occupancy,
    _enqueue_cached_dim,
    device_now_ns,
    grid_barrier,
    status_page,
    poison_offset,
)

comptime RS_FUSED_THREADS = FUSED_THREADS
"""Threads per block, the allreduce's. 256 was measured and not taken: at
512 the kernel is capped at 128 registers and spills (ptxas: 468 B of spill
stores on sm_90a), at 256 it takes 252 and spills nothing, and the isolated
block fp32 reduce-scatter on 2x8 H100 at 32 CTAs improves 618 -> 576 us
(NCCL 522) -- but the block still owns the SM's whole register file, and
GPT-2 XL FSDP2 measured 65.8k tok/s against 66.1-69.1k at 512 (one leg
each, noise +-2k), so the isolated gain did not survive the step."""

comptime RS_FUSED_UNROLL = 8
"""16-byte vectors in flight per thread in the node-local push. At 32 CTAs
(below) 4 -> 8 took the isolated block/root fp32 reduce-scatter from
682/1579 us to 668/1567 on 2x8 H100."""

comptime RS_FUSED_TARGET_CHUNKS = 2
"""Chunks a reduce-scatter of at least `PIPE_SPLIT_UNIT` bytes per rank is
cut into (geometry may force more). Every chunk costs two 8-way start
barriers and four grid barriers, and only the last chunk's exchange is
exposed. Fitted on 2x8 H100, 128 CTAs, isolated root/block fp32 us:
2 -> 1181/534, 4 -> 1331/535, 8 -> 1403/655 (NCCL 1051/522); 3 chunks
measured no different end to end (GPT-2 XL FSDP2, 66.5k vs 66.8k tok/s)."""

comptime RS_FUSED_BIG_BLOCKS = 32
"""Grid cap of the fused reduce-scatter at `PIPE_SPLIT_UNIT` bytes per rank
and above; smaller calls keep the allreduce's `fused_cap`. Every block holds
an SM's whole register file for the call, so this is also how many SMs the
backward's GEMMs lose while a reduce-scatter runs, and the end-to-end fit
is the opposite of the isolated one. Isolated root fp32 on 2x8 H100 at 512
threads: 128 CTAs 1263 us, 64 -> 1383, 32 -> 1567 (NCCL 1051). GPT-2 XL
FSDP2 on the same nodes, mojo+mojoccl tok/s (CUDA+NCCL 70.2k): 128 ->
62.4k, 64 -> 66.3k, 48 -> 64.5k, 32 -> 67.0-67.6k, 24 -> 66.1k, 16 -> 62.4k.
At 256 threads the isolated block fp32 reads 32 -> 576 us, 28 -> 599,
16 -> 716 (NCCL 522): the push runs at the fabric's rate from 16 CTAs up
(140-170 us per 53.8 MB chunk at every grid), but fewer CTAs leave the
8-way phase-1 barrier waiting 40-50 us on the slowest rank's push and
slow the HBM-bound reduce (41 -> 76 us)."""


@always_inline
def _sum_out(
    output: Pointer[Float32, MutAnyOrigin],
    partial: Pointer[Float32, MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int,
    slot_bytes: Int,
    npeers: Int,
    tid: Int,
    stride: Int,
):
    var vc = count // 4
    var v = tid
    while v < vc:
        # Four rows per iteration so 4 * (1 + npeers) loads are in flight.
        var acc = InlineArray[SIMD[DType.float32, 4], 4](uninitialized=True)
        comptime for u in range(4):
            if v + u * stride < vc:
                acc[u] = partial.unsafe_load[width=4, alignment=16](
                    (v + u * stride) * 4
                )
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * slot_bytes).unsafe_bitcast[
                Float32
            ]()
            comptime for u in range(4):
                if v + u * stride < vc:
                    acc[u] += src.unsafe_load[width=4, alignment=16](
                        (v + u * stride) * 4
                    )
        comptime for u in range(4):
            if v + u * stride < vc:
                output.unsafe_store[width=4]((v + u * stride) * 4, acc[u])
        v += 4 * stride
    for i in range(count // 4 * 4 + tid, count, stride):
        var acc = partial[unsafe_offset=i]
        for j in range(npeers):
            acc += inbox.unsafe_offset(j * slot_bytes).unsafe_bitcast[
                Float32
            ]()[unsafe_offset=i]
        output[unsafe_offset=i] = acc


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(RS_FUSED_THREADS)
    ),
    `nvvm.minctasm`=SIMDLength(1),
)
@__name(t"ccl_reduce_scatter_nodes_pipelined_float32_w{NW}")
def _fused_rs_kernel[
    NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
    mb_req: Pointer[UInt64, MutAnyOrigin],
    mb_done: Pointer[UInt64, MutAnyOrigin],
    mb_consumed: Pointer[UInt64, MutAnyOrigin],
    count: Int64,
    chunk_elems: Int64,
    seq0: Int64,
    arena_stride: Int64,
    inbox_origin: Int64,
    inbox_stride: Int64,
    shape: StaticTuple[Int32, 7],
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
    vector_ok: Int32,
):
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var nblocks = Int(grid_dim.x)
    var stride = nblocks * RS_FUSED_THREADS
    var total = Int(count)
    var ce = Int(chunk_elems)
    var nchunks = Int(shape[0])
    var depth = Int(shape[1])
    var narenas = Int(shape[2])
    var nslots = Int(shape[3])
    var npeers = Int(shape[4])
    var nnodes = Int(shape[5])
    var my_node = Int(shape[6])
    var me = regions[rank]
    var page = status_page(me)
    var poison = me.unsafe_offset(poison_offset()).unsafe_bitcast[UInt64]()
    if block_idx.x == 0 and thread_idx.x == 0:
        Atomic[DType.uint64].store[ordering=Ordering.RELAXED](poison, UInt64(0))
    for k in range(nchunks + depth - 1):
        var t0 = device_now_ns()
        if not grid_barrier(
            me, poison, nblocks, ERR_FUSED_GRID, t0, timeout_ns + _GRID_GRACE_NS
        ):
            _give_up(poison)
            return
        if k < nchunks:
            var off = k * ce
            var cnt = min(ce, total - off)
            var slot = _align_up(cnt * 4, 16)
            var arena_off = (k % narenas) * Int(arena_stride)
            var out_off = _SIGNAL_BYTES + (world - 1) * nnodes * slot
            if not _rs_nodes_body[
                DType.float32, 4, RS_FUSED_UNROLL, NW, RS_FUSED_THREADS, True
            ](
                regions,
                in_ptr.unsafe_offset(off),
                me.unsafe_offset(arena_off + out_off).unsafe_bitcast[Float32](),
                Int64(cnt),
                count,
                Int64(slot),
                Int64(_SIGNAL_BYTES),
                world_i,
                rank_i,
                flag_base + UInt64(k * PHASES_PER_GEN),
                scale,
                timeout_ns,
                vector_ok,
                rank_ids,
                shape[5],
                arena_off,
            ):
                _give_up(poison)
                return
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return
            if block_idx.x == 0 and thread_idx.x == 0:
                Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                    mb_req, UInt64(Int(seq0) + k)
                )
        var j = k - (depth - 1)
        if j >= 0:
            var off = j * ce
            var cnt = min(ce, total - off)
            var slot = _align_up(cnt * 4, 16)
            var arena_off = (j % narenas) * Int(arena_stride)
            var out_off = _SIGNAL_BYTES + (world - 1) * nnodes * slot
            var seq = Int(seq0) + j
            t0 = device_now_ns()
            if block_idx.x == 0 and thread_idx.x == 0:
                _await_exchange(
                    mb_done, poison, me, page, UInt64(seq), t0, timeout_ns
                )
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return
            _sum_out(
                out_ptr.unsafe_offset(off),
                me.unsafe_offset(
                    arena_off + out_off + my_node * slot
                ).unsafe_bitcast[Float32](),
                me.unsafe_offset(
                    Int(inbox_origin) + (seq % nslots) * Int(inbox_stride)
                ),
                cnt,
                slot,
                npeers,
                tid,
                stride,
            )
            if not grid_barrier(
                me,
                poison,
                nblocks,
                ERR_FUSED_GRID,
                t0,
                timeout_ns + _GRID_GRACE_NS,
            ):
                _give_up(poison)
                return
            if block_idx.x == 0 and thread_idx.x == 0:
                Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                    mb_consumed, UInt64(seq)
                )


def _resident[NW: Int](ctx: DeviceContext, sm_count: Int) raises -> Int:
    return sm_count * _cached_occupancy[_fused_rs_kernel[NW]](
        ctx,
        String(t"fused_rs_float32_{NW}"),
        RS_FUSED_THREADS,
    )


def reduce_scatter_fused_plan(
    count: Int,
    chunk_cap: Int,
    narenas: Int,
    split_unit_bytes: Int,
) raises -> Tuple[Int, Int, Int]:
    if count <= 0 or chunk_cap <= 0 or narenas <= 0:
        raise Error("mojoccl: invalid fused reduce-scatter geometry")
    var chunk = min(count, chunk_cap)
    if count * 4 >= split_unit_bytes:
        var target = min(narenas, RS_FUSED_TARGET_CHUNKS)
        chunk = min(
            chunk, _align_up((count + target - 1) // target * 4, 16) // 4
        )
    var nchunks = (count + chunk - 1) // chunk
    return Tuple(chunk, nchunks, min(narenas, nchunks))


def reduce_scatter_fused_blocks(
    ctx: DeviceContext,
    world: Int,
    sm_count: Int,
    chunk_elems: Int,
    block_cap: Int,
) raises -> Int:
    var resident: Int
    if world == 8:
        resident = _resident[8](ctx, sm_count)
    elif world == 4:
        resident = _resident[4](ctx, sm_count)
    elif world == 2:
        resident = _resident[2](ctx, sm_count)
    else:
        resident = _resident[0](ctx, sm_count)
    # All grid inputs are agreed at bootstrap; local occupancy only validates.
    var blocks = min(
        sm_count,
        min(
            MAX_BLOCKS,
            min(
                block_cap,
                max(
                    1,
                    (chunk_elems // 4 + RS_FUSED_THREADS - 1)
                    // RS_FUSED_THREADS,
                ),
            ),
        ),
    )
    if blocks <= 0 or resident < blocks:
        raise Error("mojoccl: fused reduce-scatter cannot hold the agreed grid")
    return blocks


def _launch[
    NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    blocks: Int,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    mailbox: StaticTuple[Int, 3],
    count: Int,
    chunk_elems: Int,
    seq0: Int,
    arena_stride: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    shape: StaticTuple[Int32, 7],
    world: Int,
    rank: Int,
    generation: Int,
    scale: Float32,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
) raises:
    _enqueue_cached_dim[_fused_rs_kernel[NW]](
        ctx,
        stream,
        String(t"fused_rs_float32_{NW}"),
        blocks,
        RS_FUSED_THREADS,
        True,
        regions,
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=out_ptr),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[0]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[1]),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox[2]),
        Int64(count),
        Int64(chunk_elems),
        Int64(seq0),
        Int64(arena_stride),
        Int64(inbox_origin),
        Int64(inbox_stride),
        shape,
        Int32(world),
        Int32(rank),
        UInt64(generation) * UInt64(PHASES_PER_GEN),
        scale,
        timeout_ns,
        rank_ids,
        Int32(1) if (in_ptr | (count * 4) | (chunk_elems * 4)) % 16
        == 0 else Int32(0),
    )


def reduce_scatter_fused(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    mailbox: StaticTuple[Int, 3],
    count: Int,
    chunk_elems: Int,
    nchunks: Int,
    depth: Int,
    narenas: Int,
    arena_stride: Int,
    seq0: Int,
    inbox_origin: Int,
    inbox_stride: Int,
    nslots: Int,
    npeers: Int,
    nnodes: Int,
    my_node: Int,
    generation: Int,
    scale: Float32,
    blocks: Int,
    timeout_ns: UInt64,
    rank_ids: InlineArray[Int32, MAX_WORLD * MAX_NODES],
) raises:
    if count <= 0:
        return
    if nchunks <= 0 or nchunks > WORK_SLOTS or chunk_elems <= 0:
        raise Error("mojoccl: invalid fused reduce-scatter chunks")
    if blocks <= 0 or blocks > MAX_BLOCKS:
        raise Error("mojoccl: invalid fused reduce-scatter grid")
    var rp = _region_ptrs(regions, rank, world)
    var shape = StaticTuple[Int32, 7](
        Int32(nchunks),
        Int32(depth),
        Int32(narenas),
        Int32(nslots),
        Int32(npeers),
        Int32(nnodes),
        Int32(my_node),
    )
    if world == 8:
        _launch[8](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    elif world == 4:
        _launch[4](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    elif world == 2:
        _launch[2](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
    else:
        _launch[0](
            ctx,
            stream,
            blocks,
            rp,
            in_ptr,
            out_ptr,
            mailbox,
            count,
            chunk_elems,
            seq0,
            arena_stride,
            inbox_origin,
            inbox_stride,
            shape,
            world,
            rank,
            generation,
            scale,
            timeout_ns,
            rank_ids,
        )
