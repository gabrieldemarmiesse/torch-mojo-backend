# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc
#
# (Flattened: a module named after its own directory would be shadowed by
# the package, so src/enqueue/enqueue.cc is tmb/ccl/enqueue.mojo.)
#
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
# only against an explicit credit (transport/net.mojo's header).
#
# Single-node communicators never touch libibverbs at all -- same fused
# kernels, same numbers as before this file learned about nodes.

from max.gpu.host import DeviceStream
from std.sys import size_of, has_nvidia_gpu_accelerator
from std.utils import StaticTuple

from tmb.ccl.device.all_reduce import (
    nvls_allreduce,
    nvls_barriers_per_call,
    nvls_chunk_bytes,
    nvls_chunk_vecs,
)
from tmb.ccl.device.broadcast import broadcast
from tmb.ccl.device.common import spin_timeout_ns
from tmb.ccl.device.symmetric.all_gather import (
    allgather,
    allgather_mapped,
    allgather_max_bytes,
    allgather_nic_stage_off,
)
from tmb.ccl.device.symmetric.all_reduce import (
    allgather_finish,
    allreduce,
    reduce_scatter_stage,
)
from tmb.ccl.device.symmetric.all_reduce_gin import (
    check_fused_call,
    fused_blocks,
    internode_allreduce_fused,
)
from tmb.ccl.device.symmetric.gin_scratch import (
    copy_bytes,
    inbox_add,
    inbox_sum_out,
)
from tmb.ccl.device.symmetric.primitives import _shard_per, shard_range
from tmb.ccl.device.symmetric.reduce_scatter import (
    rank_gate,
    reduce_scatter,
    reduce_scatter_max_count,
)
from tmb.ccl.device.symmetric.reduce_scatter_gin import (
    reduce_scatter_nodes,
    reduce_scatter_nodes_max_count,
    reduce_scatter_rank_ids,
)
from tmb.ccl.device.symmetric.reduce_scatter_gin_fused import (
    RS_FUSED_BIG_BLOCKS,
    reduce_scatter_fused,
    reduce_scatter_fused_blocks,
    reduce_scatter_fused_plan,
)
from tmb.ccl.device.symmetric.reduce_scatter_gin_stream import (
    RS_STREAM_BIG_BLOCKS,
    RS_STREAM_ENABLED,
    reduce_scatter_stream,
    reduce_scatter_stream_blocks,
    reduce_scatter_stream_generations,
    reduce_scatter_stream_pieces,
    reduce_scatter_stream_plan,
    reduce_scatter_stream_wanted,
)
from tmb.ccl.include.comm import (
    CommState,
    PIPE_SPLIT_UNIT,
    _align_up,
    _arena_regions,
    _arena_shard,
    _comm_ptr,
    _inbox_base,
    _inbox_group_bytes,
    _latch_host_fault,
    _latched_error,
    _net_stage_bytes,
    _net_stage_off,
    _pipeline_chunk_bytes,
)
from tmb.ccl.include.device import (
    ERR_ALLGATHER_SYNC,
    ERR_HOST_LAUNCH,
    ERR_REDUCE_SCATTER_SYNC,
    MAX_WORLD,
    _GFX942,
    signal_bytes,
)
from tmb.ccl.nccl import (
    NCCL_AVG,
    NCCL_FLOAT16,
    NCCL_FLOAT32,
    NCCL_INT32,
    NCCL_INT64,
    NCCL_INVALID_ARGUMENT,
    NCCL_INVALID_USAGE,
    NCCL_REMOTE_ERROR,
    NCCL_SUCCESS,
)
from tmb.ccl.transport.net import (
    EMPTY_SHARD_BYTES,
    OP_ALLGATHER,
    OP_ALLREDUCE,
    OP_BROADCAST,
    OP_REDUCE_SCATTER,
    WORK_SLOTS,
    ib_enqueue_request,
    ib_enqueue_wait,
    ib_mailbox_dev,
    ib_next_seq,
    ib_note_consumed,
    ib_npeers,
    ib_prepare_request,
    ib_reserve_seqs,
)


comptime AG_NODE_UNROLL = 2 if _GFX942 else 4
"""16-byte vectors in flight per thread in those gathers. gfx942: RCCL's
unroll for gfx94 parts with more than 80 CUs (rccl `src/init.cc:101-105`,
NCCL_UNROLL_2; the generic kernel it launches there is the unroll-2 one).
That rule covers MI300A and MI300X alike; a gfx942 part with 80 CUs or
fewer would get RCCL's unroll 4 and gets 2 here, unmeasured. Unroll 2 was
only measured together with the 24-block grid, never on its own.
H100: 8 measured 66.1k tok/s against 67.3-67.6k at 96 blocks."""


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
    messages only (`_ring_state` in transport/net.mojo): they cost nothing
    and turn "the GPU has not released exchange 78" into a sentence naming
    the collective and the chunk.

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
    """The pipelined multi-node allreduce as one launch (all_reduce_gin.mojo).

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
    _allgather_multinode(state, s, Int(sendbuff), Int(recvbuff), per_rank_bytes)
    return NCCL_SUCCESS


def allgather_mapped_max_bytes(
    arena_cap: Int, inbox_group: Int, npeers: Int, local_world: Int
) -> Int:
    return (
        min(
            allgather_max_bytes(arena_cap, local_world, nic_stage=True),
            inbox_group // npeers,
        )
        // 16
        * 16
    )


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
            state.ag_node_blocks,
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
        state.ag_node_blocks,
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


def _allgather_multinode(
    mut state: CommState,
    stream: DeviceStream,
    sendbuff: Int,
    recvbuff: Int,
    per_rank_bytes: Int,
) raises:
    """Exchange one contribution per NIC, disseminate on the receiving node.
    The node-local gathers write straight into the mapped output slots and
    stage the RDMA source; chunks pipeline through two arenas."""
    if per_rank_bytes == 0:
        return
    var npeers = ib_npeers(state.ib)
    var max_bytes = allgather_mapped_max_bytes(
        state.arena_cap, _inbox_group_bytes(state), npeers, state.local_world
    )
    var plan = allgather_mapped_pipeline_plan(
        per_rank_bytes,
        max_bytes,
        state.narenas,
        state.nslots,
        # Measured on 2x8 H100: overlap large gathers, retain the passing
        # single-chunk route below this node-scaled threshold. 2x4 MI300A:
        # agents_docs/distributed.md.
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
            var send_stage = (
                state.owned_base
                + arena * state.arena_stride
                + signal_bytes()
                + allgather_nic_stage_off(state.local_world, count)
            )
            ib_prepare_request(
                state.ib,
                send_stage,
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
                state.rs_node_blocks,
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
