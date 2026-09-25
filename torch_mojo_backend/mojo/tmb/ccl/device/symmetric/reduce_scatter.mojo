# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/reduce_scatter.cuh

from std.collections import Array
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream
from std.sys import size_of

from tmb.ccl.device.common import (
    _enqueue_cached,
    _enqueue_cached_dim,
    device_now_ns,
    spin_timeout_ns,
)
from tmb.ccl.device.symmetric.data_ops import _copy_span_flex, _share
from tmb.ccl.device.symmetric.primitives import (
    _check_common,
    _peer_step,
    _region_ptrs,
)
from tmb.ccl.include.device import (
    BLOCK,
    ERR_REDUCE_SCATTER_SYNC,
    MAX_BLOCKS,
    MAX_WORLD,
    _AR_BIG_BLOCKS,
    _AR_BIG_BYTES,
    _AR_MAX_BLOCKS,
    _SIGNAL_BYTES,
    _UNROLL,
    _align_up,
)
from tmb.ccl.include.nccl_device.lsa_barrier import _flag_target, _sync


# ===-------------------------------------------------------------------=== #
# Reduce-scatter -- push + reduce, straight into the user's output
# ===-------------------------------------------------------------------=== #
#
# `out[i] = scale * sum over ranks r of in_r[rank*in_stride + i]`, which is
# what ncclReduceScatter means. Two phases, two barriers, no pull and no
# staging of the result:
#
#   phase 1  PUSH   rank r reads chunk s of its own input and writes it into
#                   peer s's compacted slot r. That write IS the wire
#                   transfer, exactly as in the allreduce's phase 1.
#   phase 2  REDUCE every rank sums its own chunk (straight from user memory)
#                   and the `world-1` pushed slots, scales, and stores the
#                   result into `out_ptr`.
#
# Cross-link traffic is `(world-1)/world * bytes` per GPU, the unicast
# minimum, and all of it is in the WRITE direction -- so unlike the allreduce
# and the all-gather this schedule needs no separate AMD variant (module
# header, "Link direction"): no rank ever loads across a link.
#
# Slots are compacted like the split allreduce's (`world-1` of them, writer r
# into destination s's slot `r if r < s else r-1`), and they may use the whole
# `2*cap` arena: the start barrier is what orders a generation's writes after
# every peer's previous reads, so one call owning the arena needs no
# reservation -- the same reason the NVLS allreduce stages one buffer across
# both halves. `reduce_scatter_max_count` is that bound, and at the default
# region every size FSDP2 asks for is one launch.
#
# In place is safe in NCCL's sense (`recvbuff == sendbuff + rank*count`): the
# push loop never reads chunk `rank`, and in phase 2 each thread writes only
# the elements it just read.
#
# `vector_ok` is per rank and the host computes it: 16-byte vectors when
# in_ptr, out_ptr and the input stride all allow them, W-element scalar
# groups when one does not. Both walk the same groups in the same order, so
# two ranks may disagree about it (see `_copy_span_flex`).


@always_inline
def _rs_slot[
    dtype: DType
](
    slots: Pointer[UInt8, MutAnyOrigin], slot_stride: Int, p: Int, rank: Int
) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    """Peer `p`'s compacted push slot inside my own arena."""
    return slots.unsafe_offset(
        slot_stride * (p if p < rank else p - 1)
    ).unsafe_bitcast[Scalar[dtype]]()


@always_inline
def _rs_one[
    dtype: DType, accum: DType
](
    uin: Pointer[Scalar[dtype], MutAnyOrigin],
    slots: Pointer[UInt8, MutAnyOrigin],
    slot_stride: Int,
    world: Int,
    rank: Int,
    k: Int,
    scale: Float32,
) -> Scalar[dtype]:
    """Element `k` of my reduced chunk: my own input plus the pushed slots."""
    var a = _share[accum, 1](uin[unsafe_offset=k].cast[accum](), scale)
    for j in range(1, world):
        var p = rank + j
        if p >= world:
            p -= world
        a += _share[accum, 1](
            _rs_slot[dtype](slots, slot_stride, p, rank)[unsafe_offset=k].cast[
                accum
            ](),
            scale,
        )
    return a.cast[dtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_reduce_scatter_push_reduce_{dtype}_w{NW}")
def _rs_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
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
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(count)
    var in_stride = Int(in_stride_e)
    var slot_stride = Int(slot_stride_b)
    var push_off = Int(push_off_b)
    var vec = vector_ok != 0

    # --- phase 0: start barrier (the arena-reuse invariant) -----------------
    if not _sync(
        regions,
        world,
        rank,
        ERR_REDUCE_SCATTER_SYNC,
        flag_base,
        t0,
        timeout_ns,
    ):
        return

    # --- phase 1: push chunk s of my input into peer s's slot for me --------
    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var dst = (
            regions[s]
            .unsafe_offset(
                push_off + slot_stride * (rank if rank < s else rank - 1)
            )
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_span_flex[dtype, W, U](
            dst, in_ptr.unsafe_offset(s * in_stride), n, tid, stride, vec
        )

    if not _sync(
        regions,
        world,
        rank,
        ERR_REDUCE_SCATTER_SYNC,
        flag_base + 1,
        t0,
        timeout_ns,
    ):
        return

    # --- phase 2: sum the `world` contributions to my chunk into the output -
    # Each contribution is scaled as it enters the fp32 accumulator
    # (`_share`), not the finished sum.
    var uin = in_ptr.unsafe_offset(rank * in_stride)
    var slots = regions[rank].unsafe_offset(push_off)
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
            out_ptr.unsafe_store[width=W, alignment=16](
                v * W, acc.cast[dtype]()
            )
    else:
        for v in range(tid, vc, stride):
            comptime for e in range(W):
                var k = v * W + e
                out_ptr[unsafe_offset=k] = _rs_one[dtype, accum](
                    uin, slots, slot_stride, world, rank, k, scale
                )

    for i in range(tid, n - vc * W, stride):
        var k = vc * W + i
        out_ptr[unsafe_offset=k] = _rs_one[dtype, accum](
            uin, slots, slot_stride, world, rank, k, scale
        )


comptime GATE_ROW = MAX_BLOCKS - 1
"""Flag row of the reduce-scatter gate. A multi-node communicator's grids
are at most the SM count (fused kernels) or `_COPY_MAX_BLOCKS` (432), so
the last row is never a block's own; single-node communicators, whose
grids may reach MAX_BLOCKS, never gate."""


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_rank_gate")
def _gate_kernel(
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    world_i: Int32,
    rank_i: Int32,
    code: Int32,
    target: UInt64,
    timeout_ns: UInt64,
):
    """A collective's start barrier, by one block ahead of the grid.

    In a training step the collectives are where the ranks' skew shows: the
    kernel of an early rank sat in its start barrier for most of its
    in-situ life (GPT-2 XL FSDP2 on 2x8 H100: the fused reduce-scatter
    1.8 ms resident for 0.6 ms of work, the remote all-gather up to 2 ms)
    holding all of its SMs' register files, and the compute stream's GEMMs
    could not use them. This block waits instead, on one SM; the worker
    grid is stream-ordered behind it and skips its phase-0 barrier
    (`GATED=True`). The guarantee is the same one those barriers gave: my
    flag is published after every collective before this one on my stream
    has completed, and I wait for every peer's, so when the worker starts
    every rank has finished reading and pushing the arenas of the previous
    collective -- including every peer's pulls from my staging, since those
    peers have completed too. `_sync` records a deadline or an abort the
    way it does for any other barrier, under the caller's `code`.
    """
    var t0 = device_now_ns()
    _ = _sync(
        regions,
        Int(world_i),
        Int(rank_i),
        Int(code),
        target,
        t0,
        timeout_ns,
        0,
        GATE_ROW,
    )


def rank_gate(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    code: Int,
    generation: Int,
) raises:
    _enqueue_cached_dim[_gate_kernel](
        ctx,
        stream,
        "rank_gate",
        1,
        32,
        False,
        _region_ptrs(regions, rank, world),
        Int32(world),
        Int32(rank),
        Int32(code),
        _flag_target(generation, 0),
        spin_timeout_ns(),
    )


def reduce_scatter_max_count(
    cap_bytes: Int, world: Int, elem_bytes: Int
) -> Int:
    """Largest per-rank element count one `reduce_scatter` call may carry.

    The only staging is `world-1` compacted push slots of one per-rank chunk,
    and they may use the whole `2*cap_bytes` arena (see the block comment
    above `_rs_kernel`). At the 256 MiB default region that is 512 MiB of
    slots at world 2 and 73 MiB per chunk at world 8, so FSDP2's largest
    reduce-scatter is a single launch.
    """
    if elem_bytes <= 0 or cap_bytes <= 0:
        return 1
    var per = (2 * cap_bytes // max(world - 1, 1)) // 16 * 16
    return max(1, per // elem_bytes)


def _launch_rs[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
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
) raises:
    _enqueue_cached[_rs_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"rsd_{dtype}_{NW}"),
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
    )


def reduce_scatter[
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
    in_stride: Int = -1,
) raises:
    """`out[i] = scale * sum over ranks of in_r[rank*in_stride + i]`, on
    `stream`.

    `count` is the PER-RANK element count (the output's); the input holds
    `world` chunks of `in_stride` elements, of which this call reduces the
    first `count` of each. `in_stride` defaults to `count` and is larger only
    when the caller cuts one collective into several calls. `count` must be
    <= `reduce_scatter_max_count(cap_bytes, world, size_of[dtype]())`.
    `scale` is ignored for integer dtypes. Any alignment is accepted:
    misaligned pointers or an odd stride take the scalar path.
    """
    _check_common(rank, world, cap_bytes, generation)
    if count == 0:
        return
    if count < 0:
        raise Error("collectives: count must be >= 0")
    comptime W = 16 // size_of[dtype]()
    comptime esize = size_of[dtype]()
    var stride_e = in_stride if in_stride >= 0 else count
    if stride_e < count:
        raise Error("collectives: reduce_scatter in_stride < count")
    if count > reduce_scatter_max_count(cap_bytes, world, esize):
        raise Error("collectives: reduce_scatter chunk exceeds the region")
    var rp = _region_ptrs(regions, rank, world)
    # world == 1 needs no special case: the push loop is empty, the sync is a
    # self-rendezvous and the reduce copies the input, scaled.
    var vec = (in_ptr | out_ptr | (stride_e * esize)) % 16 == 0
    # One wave of blocks past `_AR_BIG_BYTES` of traffic, like the allreduce:
    # these barriers are matched by block index, so a second wave would run
    # the whole collective after the first.
    var cap_blocks = (
        _AR_BIG_BLOCKS if world * count * esize
        >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    var blocks = min(cap_blocks, max(1, (count // W + 1 + BLOCK - 1) // BLOCK))
    if world == 8:
        _launch_rs[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            stride_e,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
        )
    elif world == 4:
        _launch_rs[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            stride_e,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
        )
    elif world == 2:
        _launch_rs[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            stride_e,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
        )
    else:
        _launch_rs[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            count,
            stride_e,
            world,
            rank,
            blocks,
            vec,
            scale,
            generation,
        )
