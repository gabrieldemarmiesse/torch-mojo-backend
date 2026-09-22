"""Persistent bf16 GEMM with rolling TMA ring stage and phase counters.

The producer and consumer advance explicit counters across all output
work, avoiding division/modulo by the non-power-of-two stage count; both
counters span output-work boundaries, so the barrier sequence is the
parent's with the repeated stage-index arithmetic removed.

Derived from upstream gemm16_nn_v4_kernels.mojo.  Only the scalar-pair
shared-store loop changes: four 8x8 matrices are packed per st.matrix,
using the original BM-high, 64-column swizzled TMA boxes.  The existing
pipeline, layouts, consumer barriers, C descriptor, and TMA-store launches
remain unchanged.  The upstream col_a/kmaj_b/ragged_n parameters remain
available, so the candidate serves NN/TN/NT/TT with runtime M/N/K.

The raster group (macro-rows per rasterization band) is an explicit kernel
parameter, `group`, rather than a build define: every instantiation's value
travels in its `@__name` and its `_enqueue_cached` key, so two builds that
disagree on it never collide on one cached DeviceFunction.  NN (dgrad) uses
group=4, the upstream H100-tuned value; the NT+bias 192x192 route below
uses group=8, independently measured for that regime (worst ratio over
group in {1,2,4,6,8,12}: g=8 -> 1.037).

The NT+bias 192x192 route (`_nt_bias_rolling_ws`, reached from
gemm16_candidate_dispatch.mojo's `_try_enqueue_nt_bias_rolling_192`) fuses
`bias[n]` into the epilogue's fp32 accumulator before the single bf16
round; it shares the mainloop, pipeline and barrier logic with
`_rolling_persistent_ws` through one inlined body, `_rolling_persistent_
body` (has_bias=True there, fixed to the NT layout and the TMA-store
epilogue), rather than a `has_bias` parameter bolted onto
`_rolling_persistent_ws` itself: that kernel's compiled ABI must never
change for its existing has_bias=False (NN/TN) callers, so the bias-fused
route is its own `@__name`d entry point with its own `bias` argument. See
`_store_accum_bm_boxes_stmatrix`'s docstring for why the bias column map
does not generalize past NT, and the comptime assert in
`_rolling_persistent_body` that enforces it.

Output tiles are assigned DYNAMICALLY: every cluster takes its next work
index from a global ticket counter (gemm16_sched_pool.mojo) instead of the
static `w = cluster_id; w += num_clusters` this body used to walk.  That is
what keeps these kernels from losing nearly half their throughput whenever
another kernel -- NCCL on a DDP job's comm stream -- holds some of the SMs a
persistent grid assumed it owned.  The `@__name`s are deliberately unchanged
(a profile of this kernel names the same algorithm it always did); the
`_enqueue_cached` keys are not, because the kernel ABI gained the `sched`
counter pointer.

Tuning provenance: BK=64 and the parent's intended BM=128/BN=256/stages=3/
cluster_m=2/consumers=2 regime are the upstream H100-tuned configuration
for the has_bias=False (NN) route.  Problem dimensions are never
compile-time constants.  The exported enqueuer has the upstream generic
signature; its caller is responsible for the existing SM90/TMA regime
guards.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.mma import (
    st_matrix,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from std.memory import AddressSpace
from max.gpu.sync import named_barrier
from max.gpu.primitives import block_rank_in_cluster, cluster_sync
from std.memory import bitcast, stack_allocation
from std.sys import size_of
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    _convert_cfrags_to_simd,
    _convert_cfrags_to_tuple,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from std.sys import get_defined_bool
from tmb.kernels.gemm16_matmul.gemm16_dtype import _GEMM16_DT, _GEMM16_TAG

from tmb.kernels.common.op_utils import _enqueue_cached

from tmb.kernels.gemm16_matmul.gemm16_sched_pool import (
    SCHED_PTR,
    SCHED_RING,
    sched_advance,
    sched_fetch_add,
    sched_finish,
    sched_init_ring,
    sched_poll_local,
    sched_publish_first,
    sched_publish_round,
    sched_read_local,
    sched_slot_ptr,
    sched_supported,
)

from tmb.kernels.gemm16_matmul.gemm16_nn_v4_kernels import (
    _V4_DT,
    _V4_F32,
    _V4_PTR,
    _V4_BK,
    _V4_SWIZZLE,
    _v4_bias_epilogue_quad,
    _v4_dyn_smem_tile,
    _v4_persistent_smem_bytes,
    _v4_mma_tile,
    _v4_persistent_layout_tag,
    _v4_persistent_ragged_tag,
)

# The one place this define is read. It travels from here as the `pair_cast`
# comptime parameter of the kernels below, never read again inside them: the
# selected instruction sequence differs between the two values, so it has to
# be part of the launch cache key, and only a parameter of the kernel is (the
# key is the kernel's linkage name, and a define read in its body is invisible
# there -- two builds of this family would share one cached DeviceFunction).
comptime _ROLL_PAIR_CAST = get_defined_bool["PAIR_CAST", False]()


@always_inline
def _pack_accum_pair[pair_cast: Bool](x: Float32, y: Float32) -> Float32:
    # Optional instruction-selection experiment, retaining identical rounding.
    comptime if pair_cast:
        return bitcast[DType.float32, 1](
            SIMD[DType.float32, 2](x, y).cast[_V4_DT]()
        )
    else:
        return bitcast[DType.float32, 1](
            SIMD[_V4_DT, 2](x.cast[_V4_DT](), y.cast[_V4_DT]())
        )


@always_inline
def _store_accum_bm_boxes_stmatrix[
    bm: Int, bn: Int, has_bias: Bool, pair_cast: Bool
](
    c_smem: Pointer[
        Scalar[_V4_DT], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    accum: LayoutTensor[
        _V4_F32,
        Layout.row_major(1, 64 * bn // 128),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
    warp: Int,
    lane: Int,
    warp_group_idx: Int,
    bias: _V4_PTR,
    n0: Int,
    n: Int,
):
    """Store one warp group's WGMMA accumulator fragments into the 128B-
    swizzled TMA staging tile using `st.matrix.x4` (bn // 16 instructions
    per thread instead of bn // 4 scalar pair stores).

    For instruction t, matrix j holds fragment pair q = 4t + j: row half
    j % 2, column block 2t + j // 2.  Lane group l // 8 supplies the
    address of matrix (l // 8), row l % 8, which this function maps through
    the canonical SWIZZLE_128B 64x64 box layout.

    has_bias fuses bias[n] into the fp32 accumulator before the single bf16
    round (the NT+bias 192x192 rolling kernel, `bias`/`n0`/`n` used only
    then); comptime-eliminated to the original bias-free expression
    otherwise, so the NN/TN routes' codegen is unchanged.  NT-specific
    addressing: every consumer warp group owns 64 rows in the SAME BM-high
    C box (successive column boxes stride by bm*64) -- do not reuse for a
    layout with per-warp-group C boxes without re-deriving `row`.
    """
    comptime CFRAG = 64 * bn // 128
    var mi = lane // 8
    # Every consumer owns 64 rows in the SAME BM-high C box.  In contrast
    # to NT's separate consumer slices, successive column boxes stride by
    # bm*64 and each consumer's rows begin at (warp_group_idx-1)*64.
    var row = (warp_group_idx - 1) * 64 + warp * 16 + (lane % 8) + 8 * (mi % 2)
    var row_base = row * 64
    var row_mod = row % 8
    var c0 = mi // 2
    comptime for t in range(CFRAG // 8):
        var col = 16 * t + 8 * c0
        var off = (
            (col // 64) * (bm * 64)
            + row_base
            + (((col % 64) // 8) ^ row_mod) * 8
        )
        var data: SIMD[DType.float32, 4]
        comptime if has_bias:
            data = _v4_bias_epilogue_quad[bn](accum, t, lane, bias, n0, n)
        else:
            data = SIMD[DType.float32, 4](
                _pack_accum_pair[pair_cast](
                    accum.ptr[unsafe_offset=8 * t],
                    accum.ptr[unsafe_offset=8 * t + 1],
                ),
                _pack_accum_pair[pair_cast](
                    accum.ptr[unsafe_offset=8 * t + 2],
                    accum.ptr[unsafe_offset=8 * t + 3],
                ),
                _pack_accum_pair[pair_cast](
                    accum.ptr[unsafe_offset=8 * t + 4],
                    accum.ptr[unsafe_offset=8 * t + 5],
                ),
                _pack_accum_pair[pair_cast](
                    accum.ptr[unsafe_offset=8 * t + 6],
                    accum.ptr[unsafe_offset=8 * t + 7],
                ),
            )
        st_matrix[simd_width=4](c_smem.unsafe_offset(off), data)


@always_inline
def _rolling_persistent_body[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool,
    col_a: Bool,
    kmaj_b: Bool,
    ragged_n: Bool,
    group: Int,
    # Fuses bias[n] into the epilogue (the NT+bias 192x192 route); requires
    # tma_store and the NT layout (kmaj_b, not col_a) -- see the comptime
    # assert below and _store_accum_bm_boxes_stmatrix's docstring. `bias` is
    # a live pointer only when has_bias -- callers with has_bias=False pass
    # any valid pointer (never read) rather than a constructed null one.
    has_bias: Bool,
    # The epilogue's PAIR_CAST build identity, threaded in rather than read
    # here, so that it reaches the launch cache key (see `_ROLL_PAIR_CAST`).
    pair_cast: Bool,
    a_tile_shape: IndexList[2],
    a_desc_shape: IndexList[2],
    b_tile_shape: IndexList[2],
    b_desc_shape: IndexList[2],
](
    a_tma: TMATensorTile[_V4_DT, 2, a_tile_shape, a_desc_shape],
    b_tma: TMATensorTile[_V4_DT, 2, b_tile_shape, b_desc_shape],
    c_tma: TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)],
    output: _V4_PTR,
    bias: _V4_PTR,
    sched: SCHED_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    """Shared mainloop/epilogue body of every persistent-rolling device
    kernel in this file (`_rolling_persistent_ws` has_bias=False and
    `_nt_bias_rolling_ws` has_bias=True call it, inlined, from their own
    `@__name`d entry points): a change here reaches both, and there is no
    second copy of the ring/barrier/pipeline logic to keep in sync. Whether
    a particular instantiation's compiled kernel takes `bias` as a real ABI
    argument, and its exact device-side behavior, is decided entirely by
    each caller's own signature and has_bias -- this body has no `@__name`
    of its own and is never launched directly.

    `sched` is the launch's ticket counter (gemm16_sched_pool.mojo): rank 0's
    producer thread dispenses work indices from it, the peer rank reads them
    over DSMEM and every consumer reads its own CTA's ring.  Per-tile K order
    is untouched, so results are bit-identical to the static assignment this
    replaced -- only WHICH cluster computes a tile changes.
    """
    comptime assert not has_bias or (
        tma_store and kmaj_b and not col_a
    ), "the fused bias epilogue is NT-only (kmaj_b) and needs the TMA store"
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_mn_major[
            _V4_DT, bm, _V4_BK, _V4_SWIZZLE
        ]() if col_a else tile_layout_k_major[_V4_DT, bm, _V4_BK, _V4_SWIZZLE]()
        comptime B_LAYOUT = tile_layout_k_major[
            _V4_DT, bn, _V4_BK, _V4_SWIZZLE
        ]() if kmaj_b else tile_layout_mn_major[
            _V4_DT, bn, _V4_BK, _V4_SWIZZLE
        ]()
        # For both majornesses a 64-row chunk of the bn-row tile is one
        # contiguous 64 * BK block at offset chunk * 64 * BK (BK = 64 bf16 is
        # exactly one 128B swizzle atom row, so the K-major layout is a plain
        # stack of 8-row atoms; the NT kernel's half-tile multicast relies on
        # the same decomposition).
        comptime B_CHUNK_LAYOUT = tile_layout_k_major[
            _V4_DT, 64, _V4_BK, _V4_SWIZZLE
        ]() if kmaj_b else tile_layout_mn_major[
            _V4_DT, 64, _V4_BK, _V4_SWIZZLE
        ]()
        comptime A_PIPE_LAYOUT = Layout.row_major(stages, bm * _V4_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(stages, bn * _V4_BK)
        # Three carvings of one extern slab -- see `_v4_dyn_smem_tile`.
        var a_pipeline = _v4_dyn_smem_tile[A_PIPE_LAYOUT, 128, 0]()
        var b_pipeline = _v4_dyn_smem_tile[
            B_PIPE_LAYOUT, 128, stages * bm * _V4_BK
        ]()
        # C staging tile for the TMA-store epilogue (swizzled 128B rows of
        # 64 elements, bn // 64 chunks).  A dummy allocation when disabled.
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        comptime C_SMEM_OFFSET = stages * (bm + bn) * _V4_BK
        var c_smem = _v4_dyn_smem_tile[
            Layout.row_major(1, C_SMEM_ELEMS), 1024, C_SMEM_OFFSET
        ]()
        # The C tile is the last carving, so its end IS the slab size the
        # launch must ask for; keeping the two in step is not left to a
        # comment.
        comptime assert (
            _v4_persistent_smem_bytes[stages, bm, bn, tma_store]()
            == (C_SMEM_OFFSET + C_SMEM_ELEMS) * 2
        ), "persistent-body smem carve and launch size disagree"
        var full_barriers = stack_allocation[
            stages,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            stages,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        # Published work tickets, one 32-bit word per round (see
        # gemm16_sched_pool.mojo).  A slot is rewritten SCHED_RING rounds
        # later and publication runs at most `stages + 1` rounds ahead of the
        # slowest reader in the cluster, so this bound is what keeps a live
        # ticket from being overwritten under a reader.
        comptime assert (
            SCHED_RING >= stages + 4
        ), "work-ticket ring too shallow for this pipeline depth"
        var work_ring = stack_allocation[
            SCHED_RING,
            Scalar[DType.uint32],
            address_space=AddressSpace.SHARED,
            alignment=16,
        ]()
        # Round 0's ticket is fetched HERE, before the barrier inits, the TMA
        # descriptor prefetches and the cluster barrier: every cluster's rank
        # 0 hits the same counter word at the same instant, and same-address
        # L2 atomics serialise.  Issuing it first lets that queue drain behind
        # the prologue instead of standing between kernel entry and the first
        # TMA.
        var ticket0 = UInt32(0)
        if thread_idx.x == 0:
            if Int(block_rank_in_cluster()) == 0:
                ticket0 = UInt32(sched_fetch_add(sched, 1))
            sched_init_ring(work_ring)
            comptime for stage in range(stages):
                full_barriers[unsafe_offset=stage].init()
                # Released by every consumer warp group of every CTA in the
                # cluster: the multicast source must not overwrite a peer's
                # tile while that peer is still reading it.
                empty_barriers[unsafe_offset=stage].init(
                    Int32(consumers * cluster_m)
                )
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
            fence_mbarrier_init()
        # All barriers must be initialized cluster-wide before any arrival
        # (the consumers below arrive at peer CTAs' empty barriers), and the
        # zeroed ticket ring must be VISIBLE to the peer before it can poll
        # it.  `cluster_sync_relaxed` orders neither -- it is an arrive/wait
        # with no memory ordering -- and `fence_mbarrier_init` covers only
        # the mbarrier state, so a peer could read whatever was in that
        # shared word before the kernel started; one arbitrary word in 1024
        # carries round 0's tag and would be accepted as a published ticket
        # (wrong macro-row, or a hang).  `cluster_sync` is the same barrier
        # with the fence, once per launch (a Codex review finding).
        cluster_sync()

        comptime CFRAG = 64 * bn // 128
        comptime MACRO_BM = bm * cluster_m
        comptime TMA_BYTES = (bm + bn) * _V4_BK * 2
        comptime MCAST_MASK = UInt16((1 << cluster_m) - 1)
        comptime B_CHUNKS = bn // 64
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var rank = Int(block_rank_in_cluster())
        # No cluster id: the work index is a ticket from the global counter,
        # not `cluster_id + j * num_clusters` (gemm16_sched_pool.mojo).
        var num_clusters = Int(grid_dim.x) // cluster_m
        # m may be ragged: TMA A reads clamp out-of-bounds rows and the
        # epilogue stores are row-predicated.  With ragged_n, n may be too
        # (see the parameter comment above).
        var macro_rows = (m + MACRO_BM - 1) // MACRO_BM
        var blocks_n = n // bn
        comptime if ragged_n:
            blocks_n = (n + bn - 1) // bn
        var total_works = macro_rows * blocks_n
        var num_tiles = k // _V4_BK
        var group_span = group * blocks_n

        # Release every pipeline slot to the producers (cluster-wide).
        if warp_group_idx > 0 and warp_group_thread_idx < cluster_m:
            comptime for stage in range(stages):
                empty_barriers[unsafe_offset=stage].arrive_cluster(
                    UInt32(warp_group_thread_idx)
                )

        if warp_group_idx == 0:
            # 32, not 24: the scheduler adds live state to this warp
            # group and 24 spills it to local memory (ptxas -v: 32 bytes of
            # spill stores at 24, zero at 32).  32 still fits the SM's 65536
            # registers beside three consumer warp groups at 160
            # (3 * 128 * 160 + 128 * 32 = 65536 exactly) and two at 232.
            # Measured worth up to 6% under contention (tn_c_attn at a 16-SM
            # hog: 252.9 -> 239.1 us).
            warpgroup_reg_dealloc[32]()
            if warp_group_thread_idx == 0:
                var ring_stage = 0
                var ring_phase = UInt32(0)
                var rm = UInt32(0)
                sched_publish_first(work_ring, rank, rm, ticket0)
                var w = sched_read_local(work_ring, rm)
                while w < total_works:
                    # Publish round j+1 BEFORE issuing round j, so the peer
                    # rank's DSMEM poll and both CTAs' consumer polls are
                    # already satisfied when they look.
                    var rm_next = sched_advance(rm)
                    sched_publish_round(sched, work_ring, rank, rm_next)
                    var g = w // group_span
                    var rem = w % group_span
                    var rows_in_group = min(group, macro_rows - g * group)
                    var macro_row = g * group + rem % rows_in_group
                    var n0 = (rem // rows_in_group) * bn
                    var m0 = macro_row * MACRO_BM + rank * bm
                    var t = 0
                    while t < num_tiles:
                        var stage = ring_stage
                        var phase = ring_phase
                        empty_barriers[unsafe_offset=stage].wait(phase)
                        full_barriers[unsafe_offset=stage].expect_bytes(
                            Int32(TMA_BYTES)
                        )
                        var a_tile = LayoutTensor[
                            _V4_DT,
                            A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr.unsafe_offset(stage * bm * _V4_BK))
                        var k0 = t * _V4_BK
                        # TMA coordinates are (fastest dim, slower dim) of
                        # the global tensor the descriptor was built over:
                        # (m, k) for the col-major (K, M) wgrad operand.
                        comptime if col_a:
                            a_tma.async_copy(
                                a_tile,
                                full_barriers[unsafe_offset=stage],
                                (m0, k0),
                            )
                        else:
                            a_tma.async_copy(
                                a_tile,
                                full_barriers[unsafe_offset=stage],
                                (k0, m0),
                            )
                        # Cooperative B load: each cluster rank reads its
                        # share of the 64-column chunks once from L2 and
                        # multicasts it to every peer, so the per-SM TMA
                        # engines split the shared-tile traffic instead of
                        # rank 0 funneling all of B (nvjet's "coopB").
                        var cc = rank * B_CHUNKS // cluster_m
                        var cend = (rank + 1) * B_CHUNKS // cluster_m
                        while cc < cend:
                            var b_chunk = LayoutTensor[
                                _V4_DT,
                                B_CHUNK_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr.unsafe_offset(
                                    stage * bn * _V4_BK + cc * 64 * _V4_BK
                                )
                            )
                            # B TMA coordinates follow the descriptor's
                            # global tensor: (k, n) for the K-major (N, K)
                            # kmaj_b operand, (n, k) for the row-major
                            # (K, N) one.
                            comptime if cluster_m > 1:
                                comptime if kmaj_b:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (k0, n0 + cc * 64),
                                        MCAST_MASK,
                                    )
                                else:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (n0 + cc * 64, k0),
                                        MCAST_MASK,
                                    )
                            else:
                                comptime if kmaj_b:
                                    b_tma.async_copy(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (k0, n0 + cc * 64),
                                    )
                                else:
                                    b_tma.async_copy(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (n0 + cc * 64, k0),
                                    )
                            cc += 1
                        t += 1
                        ring_stage += 1
                        if ring_stage == stages:
                            ring_stage = 0
                            ring_phase = ring_phase ^ UInt32(1)
                    rm = rm_next
                    w = sched_read_local(work_ring, rm)
                if rank == 0:
                    # `w` is this cluster's past-the-end ticket; see
                    # gemm16_sched_pool.mojo for why the highest one resets
                    # the counter.
                    sched_finish(sched, w, total_works, num_clusters)
        else:
            # Consumer registers: three warp groups fit 65536 regs/SM only
            # at 160 regs/thread (96 accumulator + addressing); two fit 232.
            comptime if consumers >= 3:
                warpgroup_reg_alloc[160]()
            else:
                warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V4_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            comptime wgmma = TensorCoreAsync[
                _V4_F32,
                _V4_DT,
                _V4_DT,
                Index(64, bn, 16),
                a_swizzle=_V4_SWIZZLE,
                b_swizzle=_V4_SWIZZLE,
                transpose_b=False,
            ]()

            var ring_stage = 0
            var ring_phase = UInt32(0)
            var rm = UInt32(0)
            var w = sched_poll_local(work_ring, rm)
            while w < total_works:
                var g = w // group_span
                var rem = w % group_span
                var rows_in_group = min(group, macro_rows - g * group)
                var macro_row = g * group + rem % rows_in_group
                var n0 = (rem // rows_in_group) * bn
                var m0 = macro_row * MACRO_BM + rank * bm
                _ = accum.fill(0.0)
                var t = 0
                while t < num_tiles:
                    var stage = ring_stage
                    var phase = ring_phase
                    full_barriers[unsafe_offset=stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _V4_DT,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr.unsafe_offset(stage * bm * _V4_BK))
                    var b_tile = LayoutTensor[
                        _V4_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr.unsafe_offset(stage * bn * _V4_BK))
                    comptime if col_a or kmaj_b:
                        # Raw descriptor path: TensorCoreAsync has no
                        # col-major A mode (and the TT instantiation's
                        # K-major B rides the same majorness-generic
                        # helper).
                        _v4_mma_tile[bn, col_a, kmaj_b, A_LAYOUT, B_LAYOUT](
                            a_tile.ptr, b_tile.ptr, accum, warp_group_idx
                        )
                    else:
                        warpgroup_fence(accum)
                        wgmma.arrive()
                        wgmma.wgmma[consumers](
                            a_tile, b_tile, accum, warp_group_idx - 1
                        )
                        wgmma.commit_group()
                        warpgroup_fence(accum)
                        wgmma.wait_group()
                    if warp_group_thread_idx < cluster_m:
                        empty_barriers[unsafe_offset=stage].arrive_cluster(
                            UInt32(warp_group_thread_idx)
                        )
                    t += 1
                    ring_stage += 1
                    if ring_stage == stages:
                        ring_stage = 0
                        ring_phase = ring_phase ^ UInt32(1)

                var tid = warp_group_thread_idx
                var warp = tid // 32
                var lane = tid % 32
                var base_row = warp * 16 + lane // 4
                var base_col = (lane % 4) * 2
                comptime if tma_store:
                    # Stage the tile in shared memory and hand it to TMA;
                    # the store drains in the background of the next work's
                    # mainloop, and TMA clips rows past a ragged m edge.
                    comptime NCONS = Int32(consumers * 128)
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        # Previous work's store must fully drain before the
                        # staging tile is overwritten.
                        c_tma.wait_group[0]()
                    named_barrier[NCONS](1)
                    _store_accum_bm_boxes_stmatrix[bm, bn, has_bias, pair_cast](
                        c_smem.ptr,
                        accum,
                        warp,
                        lane,
                        warp_group_idx,
                        bias,
                        n0,
                        n,
                    )
                    fence_async_view_proxy()
                    named_barrier[NCONS](1)
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        comptime for chunk in range(bn // 64):
                            var c_chunk = LayoutTensor[
                                _V4_DT,
                                Layout.row_major(bm, 64),
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](c_smem.ptr.unsafe_offset(chunk * bm * 64))
                            c_tma.async_store(c_chunk, (n0 + chunk * 64, m0))
                        c_tma.commit_group()
                else:
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        var pair = SIMD[_V4_DT, 2](
                            accum.ptr[unsafe_offset=e].cast[_V4_DT](),
                            accum.ptr[unsafe_offset=e + 1].cast[_V4_DT](),
                        )
                        if m0 + row < m and n0 + col + 1 < n:
                            output.unsafe_store[alignment=4](
                                (m0 + row) * n + n0 + col, pair
                            )
                rm = sched_advance(rm)
                w = sched_poll_local(work_ring, rm)
            comptime if tma_store:
                # Outstanding bulk stores must complete before kernel exit.
                if warp_group_idx == 1 and warp_group_thread_idx == 0:
                    c_tma.wait_group[0]()

        # Peer CTAs receive multicast writes into this CTA's shared memory;
        # do not tear the block down while any cluster member is running.
        cluster_sync()


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(128 * (consumers + 1))
    ),
    `nvvm.cluster_dim`=StaticTuple[Int32, 3](
        Int32(cluster_m), Int32(1), Int32(1)
    ),
)
# One kernel symbol per layout: the TN and NN instantiations of this body
# would otherwise share one base name (differing only by mangling hash), so
# GPU profiles could not tell them apart and scripts/compare_kernel_asm.py --
# which pairs kernels by hash-stripped name -- would collide them.  The
# ragged tag does the same for the n-clip TT instantiation while keeping
# every exact-n symbol byte-identical to its pre-existing name.  This entry
# point's own runtime ABI (no `bias` argument) is exactly what it was before
# has_bias existed: the bias-fused route is a SEPARATE kernel below
# (`_nt_bias_rolling_ws`), not a `bias` parameter bolted onto this one, so
# adding it could not add a dead pointer argument to this compiled kernel --
# scripts/compare_kernel_asm.py caught exactly that mistake in an earlier
# revision of this change.
@__name(
    t"{_GEMM16_TAG}_gemm_{_v4_persistent_layout_tag[col_a, kmaj_b]()}_v4_persistent_stmatrix_rolling_m{bm}n{bn}_s{stages}c{cluster_m}wg{consumers}g{group}{_v4_persistent_ragged_tag[ragged_n]()}"
)
def _rolling_persistent_ws[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool,
    # col_a extends the persistent body to the TN (wgrad) layout: A is
    # physically (K, M), TMA-loaded into an MN-major shared tile and
    # consumed through WGMMA's col-major A mode via _v4_mma_tile.  kmaj_b
    # does the same for B: physically (N, K), TMA-loaded into a K-major
    # shared tile for WGMMA's col-major B mode; col_a + kmaj_b is the TT
    # instantiation.  The trailing shape parameters exist because the TMA
    # boxes follow each operand's majorness; their defaults keep every
    # pre-existing NN and TN instantiation (and its generated code)
    # unchanged.
    col_a: Bool = False,
    kmaj_b: Bool = False,
    # ragged_n admits n % bn != 0 (still n % 64 == 0): blocks_n becomes a
    # ceil-div, the B TMA reads clamp past the n edge (zero-fill, zero
    # contributions) and the C TMA store's partial last column box clips
    # against the (m, n) descriptor -- the same machinery the ragged-m path
    # uses, on the other axis.  The NN, TN and TT routes all instantiate
    # it.
    ragged_n: Bool = False,
    # Macro-rows per rasterization group before advancing one BN column
    # (keeps the in-flight A slab and current B column resident in L2). A
    # kernel parameter rather than a build define -- see try_enqueue_
    # candidate_nn in gemm16_candidate_dispatch.mojo for the measured value.
    group: Int = 4,
    # The epilogue's PAIR_CAST build identity (`_ROLL_PAIR_CAST`), a parameter
    # so the launch cache distinguishes the two builds. Positioned after
    # `group` so every existing positional instantiation is unchanged.
    pair_cast: Bool = _ROLL_PAIR_CAST,
    a_tile_shape: IndexList[2] = Index(_V4_BK, bm) if col_a else Index(
        bm, _V4_BK
    ),
    a_desc_shape: IndexList[2] = Index(_V4_BK, 64) if col_a else Index(
        bm, _V4_BK
    ),
    b_tile_shape: IndexList[2] = Index(64, _V4_BK) if kmaj_b else Index(
        _V4_BK, 64
    ),
    b_desc_shape: IndexList[2] = Index(64, _V4_BK) if kmaj_b else Index(
        _V4_BK, 64
    ),
](
    a_tma: TMATensorTile[_V4_DT, 2, a_tile_shape, a_desc_shape],
    b_tma: TMATensorTile[_V4_DT, 2, b_tile_shape, b_desc_shape],
    c_tma: TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)],
    output: _V4_PTR,
    sched: SCHED_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # `output` fills the body's unread `bias` slot: has_bias=False comptime-
    # eliminates every bias read, so this never dereferences it.
    _rolling_persistent_body[
        stages,
        cluster_m,
        bm,
        bn,
        consumers,
        tma_store,
        col_a,
        kmaj_b,
        ragged_n,
        group,
        False,
        pair_cast,
        a_tile_shape,
        a_desc_shape,
        b_tile_shape,
        b_desc_shape,
    ](a_tma, b_tma, c_tma, output, output, sched, m_arg, n_arg, k_arg)


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(128 * (consumers + 1))
    ),
    `nvvm.cluster_dim`=StaticTuple[Int32, 3](
        Int32(cluster_m), Int32(1), Int32(1)
    ),
)
# The 192x192 NT+bias rolling kernel: its own entry point (own `@__name`,
# own `bias` ABI slot) rather than a parameter on `_rolling_persistent_ws`,
# so that kernel's compiled signature never changes for has_bias=False
# callers (NN/TN). Fixed to the NT layout (kmaj_b, not col_a), the TMA-store
# epilogue and ragged_n=True -- the only configuration this route needs;
# `group` stays a parameter, per-instantiation-measured like the sibling
# kernel's. Name matches the standalone engagement that measured it
# (gemm16_candidate_dispatch.mojo's `_try_enqueue_nt_bias_rolling_192`).
@__name(
    t"{_GEMM16_TAG}_gemm_nt_bias_rolling_ws_m{bm}n{bn}_s{stages}c{cluster_m}wg{consumers}g{group}"
)
def _nt_bias_rolling_ws[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    group: Int,
    pair_cast: Bool = _ROLL_PAIR_CAST,
](
    a_tma: TMATensorTile[_V4_DT, 2, Index(bm, _V4_BK), Index(bm, _V4_BK)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(64, _V4_BK), Index(64, _V4_BK)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)],
    output: _V4_PTR,
    bias: _V4_PTR,
    sched: SCHED_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    _rolling_persistent_body[
        stages,
        cluster_m,
        bm,
        bn,
        consumers,
        True,
        False,
        True,
        True,
        group,
        True,
        pair_cast,
        Index(bm, _V4_BK),
        Index(bm, _V4_BK),
        Index(64, _V4_BK),
        Index(64, _V4_BK),
    ](a_tma, b_tma, c_tma, output, bias, sched, m_arg, n_arg, k_arg)


def enqueue_rolling_persistent[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool = False,
    col_a: Bool = False,
    kmaj_b: Bool = False,
    ragged_n: Bool = False,
    group: Int = 4,
    has_bias: Bool = False,
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    # Unused (never read) unless has_bias: the NN/TN callers pass `output`
    # as a valid-but-ignored filler rather than constructing a null pointer.
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    sm_count: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Launch one of this file's persistent-rolling kernels, or decline.

    Returns False WITHOUT launching -- and without building a single TMA
    descriptor -- when the dynamic tile scheduler cannot serve the shape: a
    work census past the ring word's 22-bit payload, or a counter table with
    no free entry (see gemm16_sched_pool.mojo).  Every caller treats that as
    "this rung declines" and falls through to the next one.
    """
    # The work census decides both the grid and whether the scheduler can
    # serve the shape at all, so it is computed before anything is allocated.
    var macro_rows = (m + bm * cluster_m - 1) // (bm * cluster_m)
    var blocks_n = n // bn
    comptime if ragged_n:
        blocks_n = (n + bn - 1) // bn
    var total_works = macro_rows * blocks_n
    if not sched_supported(total_works):
        return False
    # One ticket counter per (device, stream); allocated once and self-reset
    # by the kernel, so nothing is allocated or memset per launch.
    var slot = sched_slot_ptr(ctx)
    if not slot:
        return False
    var sched = slot.value()
    # Each descriptor follows its operand's physical layout: (M, K) row-major
    # with a whole-tile box, or -- for the TN/wgrad and TT col_a routes --
    # (K, M) row-major with a (BK, 64) box feeding the MN-major shared tile;
    # likewise (K, N) row-major for B, or -- for the TT kmaj_b route --
    # (N, K) row-major with a (64, BK) box feeding the K-major shared tile.
    comptime A_TILE = Index(_V4_BK, bm) if col_a else Index(bm, _V4_BK)
    comptime A_DESC = Index(_V4_BK, 64) if col_a else Index(bm, _V4_BK)
    comptime B_TILE = Index(64, _V4_BK) if kmaj_b else Index(_V4_BK, 64)
    var a_dim0 = k if col_a else m
    var a_dim1 = m if col_a else k
    var a_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](a_dim0, a_dim1),
        IndexList[2](a_dim1, 1),
        IndexList[2](A_DESC[0], A_DESC[1]),
    )
    var b_dim0 = n if kmaj_b else k
    var b_dim1 = k if kmaj_b else n
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](b_dim0, b_dim1),
        IndexList[2](b_dim1, 1),
        IndexList[2](B_TILE[0], B_TILE[1]),
    )
    var c_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            output.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, n),
        IndexList[2](n, 1),
        IndexList[2](bm, 64),
    )
    var a_tma = TMATensorTile[_V4_DT, 2, A_TILE, A_DESC](a_desc)
    var b_tma = TMATensorTile[_V4_DT, 2, B_TILE, B_TILE](b_desc)
    var c_tma = TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)](c_desc)
    var num_clusters = min(sm_count // cluster_m, total_works)
    var grid_x = num_clusters * cluster_m
    comptime DYN_SMEM = _v4_persistent_smem_bytes[stages, bm, bn, tma_store]()
    # Compiled once per process and context: `ctx.enqueue_function[kernel]`
    # re-runs compile_function on every launch. The key is the kernel's
    # linkage name, so everything that selects the code -- dtype, geometry,
    # stages, layout, epilogue, raster group, `pair_cast` -- is one of its
    # comptime parameters, and nothing about this call's pointers or its
    # m/n/k, which travel as arguments. The cluster shape
    # rides on the kernel's own `nvvm.cluster_dim` metadata, as it did
    # before.  has_bias picks which DEVICE KERNEL is launched (see their
    # docstrings): _rolling_persistent_ws's own ABI never gains a `bias`
    # argument just because this host function grew one.
    comptime if has_bias:
        comptime assert (
            tma_store and kmaj_b and not col_a
        ), "the fused bias epilogue is NT-only (kmaj_b) and needs the TMA store"
        _enqueue_cached[
            _nt_bias_rolling_ws[stages, cluster_m, bm, bn, consumers, group],
            dyn_smem=DYN_SMEM,
        ](
            ctx,
            grid_x,
            1,
            1,
            128 * (consumers + 1),
            a_tma,
            b_tma,
            c_tma,
            output,
            bias,
            sched,
            Int64(m),
            Int64(n),
            Int64(k),
        )
    else:
        _enqueue_cached[
            _rolling_persistent_ws[
                stages,
                cluster_m,
                bm,
                bn,
                consumers,
                tma_store,
                col_a,
                kmaj_b,
                ragged_n,
                group,
            ],
            dyn_smem=DYN_SMEM,
        ](
            ctx,
            grid_x,
            1,
            1,
            128 * (consumers + 1),
            a_tma,
            b_tma,
            c_tma,
            output,
            sched,
            Int64(m),
            Int64(n),
            Int64(k),
        )
    return True
