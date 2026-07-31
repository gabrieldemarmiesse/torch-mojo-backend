"""Persistent warp-specialized H100 BF16 NT GEMM (v4).

Targets the forward-linear layout C[m,n] = A[m,k] @ B[n,k]^T on SM90.

Improvements over the v3 NT kernel (`_v3_nt_ws_m128n256_tma_s3`):

1. Persistent tile scheduler: a fixed grid of CTAs (one per SM) loops over
   output tiles.  The producer warp group keeps the shared-memory pipeline
   full ACROSS output tiles, so the per-tile pipeline fill/drain stalls of
   the non-persistent kernel disappear.  This matters most for short-K
   problems (few mainloop iterations per tile) where fill/drain dominated.
2. TMA store epilogue: accumulators are staged through a 128B-swizzled
   shared-memory tile via `st.matrix` and written with
   `cp.async.bulk.tensor` (TMA).  The old direct epilogue issued scattered
   4-byte global stores at 50% sector efficiency; TMA stores are fully
   coalesced.  The store is committed asynchronously and only waited on at
   the START of the next epilogue, so it drains while the next output
   tile's mainloop runs.
3. CTA clusters (size 2) with TMA multicast of B: the rasterization pairs
   CTAs on the same column block, so each B tile is read once from L2 and
   broadcast to both CTAs.
4. Per-regime tile width: BN = 256 for wide/deep problems, BN = 192 (with
   a deeper 4-stage pipeline) when it reduces persistent-wave imbalance
   (e.g. narrow n where ceil(tiles / grid) quantization costs several
   percent).  Selection is by a shape-generic cost model, not hardcoded
   model dimensions.

The kernel is fully dynamic in m, n, k (runtime dispatch, no hardcoded
model dimensions).  Partial edge tiles are handled by TMA out-of-bounds
clipping on both loads (zero-fill) and stores (write suppression).
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.compute.mma import st_matrix
from std.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.gpu.sync import named_barrier
from std.memory import bitcast, stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

comptime _V4_BF16 = DType.bfloat16
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_BF16], MutAnyOrigin]
comptime _V4_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B

comptime _V4_BM = 128
comptime _V4_BK = 64
comptime _V4_THREADS = 384
comptime _V4_CONSUMERS = 2
# Consumer warp group owns a 64-row half of the 128-row output tile.
comptime _V4_WG_ROWS = _V4_BM // _V4_CONSUMERS
# TMA store box: 64 rows x 64 bf16 columns (128B swizzle span).
comptime _V4_C_BOX_N = 64
comptime _V4_C_BOX_ELEMS = _V4_WG_ROWS * _V4_C_BOX_N
comptime _V4_CLUSTER = 2
comptime _V4_CLUSTER_SHAPE = StaticTuple[Int32, 3](Int32(_V4_CLUSTER), 1, 1)

comptime _V4_A_LAYOUT = tile_layout_k_major[
    _V4_BF16, _V4_BM, _V4_BK, _V4_SWIZZLE
]()
comptime _V4_A_TMA = TMATensorTile[
    _V4_BF16, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
]


def _v4_b_layout[bn: Int]() -> Layout:
    return tile_layout_k_major[_V4_BF16, bn, _V4_BK, _V4_SWIZZLE]()


def _v4_b_half_layout[bn: Int]() -> Layout:
    return tile_layout_k_major[
        _V4_BF16, bn // _V4_CLUSTER, _V4_BK, _V4_SWIZZLE
    ]()


@always_inline
def _v4_store_accum_stmatrix[
    bn: Int
](
    wg_half: UnsafePointer[
        Scalar[_V4_BF16], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    accum: LayoutTensor[
        _V4_F32,
        Layout.row_major(1, 64 * bn // 128),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
    warp: Int,
    lane: Int,
):
    """Store one warp group's WGMMA accumulator fragments into the 128B-
    swizzled TMA staging tile using `st.matrix.x4` (bn // 16 instructions
    per thread instead of bn // 4 scalar pair stores).

    For instruction t, matrix j holds fragment pair q = 4t + j: row half
    j % 2, column block 2t + j // 2.  Lane group l // 8 supplies the
    address of matrix (l // 8), row l % 8, which this function maps through
    the canonical SWIZZLE_128B 64x64 box layout.
    """
    comptime CFRAG = 64 * bn // 128
    var mi = lane // 8
    var row = warp * 16 + (lane % 8) + 8 * (mi % 2)
    var row_base = row * _V4_C_BOX_N
    var row_mod = row % 8
    var c0 = mi // 2
    comptime for t in range(CFRAG // 8):
        var col = 16 * t + 8 * c0
        var off = (
            (col // _V4_C_BOX_N) * _V4_C_BOX_ELEMS
            + row_base
            + (((col % _V4_C_BOX_N) // 8) ^ row_mod) * 8
        )
        var data = SIMD[DType.float32, 4](
            bitcast[DType.float32, 1](
                SIMD[_V4_BF16, 2](
                    accum.ptr[8 * t].cast[_V4_BF16](),
                    accum.ptr[8 * t + 1].cast[_V4_BF16](),
                )
            ),
            bitcast[DType.float32, 1](
                SIMD[_V4_BF16, 2](
                    accum.ptr[8 * t + 2].cast[_V4_BF16](),
                    accum.ptr[8 * t + 3].cast[_V4_BF16](),
                )
            ),
            bitcast[DType.float32, 1](
                SIMD[_V4_BF16, 2](
                    accum.ptr[8 * t + 4].cast[_V4_BF16](),
                    accum.ptr[8 * t + 5].cast[_V4_BF16](),
                )
            ),
            bitcast[DType.float32, 1](
                SIMD[_V4_BF16, 2](
                    accum.ptr[8 * t + 6].cast[_V4_BF16](),
                    accum.ptr[8 * t + 7].cast[_V4_BF16](),
                )
            ),
        )
        st_matrix[simd_width=4](wg_half + off, data)


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS)),
    `nvvm.cluster_dim`=_V4_CLUSTER_SHAPE,
)
def _v4c_nt_persistent[
    bn: Int, stages: Int, raster_h: Int, defer_release: Bool = False
](
    a_tma: _V4_A_TMA,
    b_tma: TMATensorTile[
        _V4_BF16,
        2,
        Index(bn // _V4_CLUSTER, _V4_BK),
        Index(bn // _V4_CLUSTER, _V4_BK),
    ],
    c_tma: TMATensorTile[
        _V4_BF16, 2, Index(_V4_WG_ROWS, bn), Index(_V4_WG_ROWS, _V4_C_BOX_N)
    ],
    m: Int,
    n: Int,
    k: Int,
):
    """Clustered persistent NT kernel with TMA multicast of B.

    CTA pairs (cluster dim x = 2) process work tiles (2p, 2p + rank), which
    the rasterization places at the same n0 with adjacent m0.  Each rank
    TMA-loads its own A tile plus HALF of the shared B tile, multicast into
    both CTAs' shared memory, halving B traffic out of L2.  When the pair's
    work ids do not share n0 (partial raster groups) or fall off the end of
    the work list, ranks clamp to a common valid tile / load B privately, so
    both CTAs always execute identical barrier trip counts.
    """
    comptime B_HALF = bn // _V4_CLUSTER
    comptime B_LAYOUT = _v4_b_layout[bn]()
    comptime B_HALF_LAYOUT = _v4_b_half_layout[bn]()
    comptime CFRAG = 64 * bn // 128
    comptime TMA_BYTES = (_V4_BM + bn) * _V4_BK * 2
    comptime if _is_sm_9x():
        var a_pipeline = LayoutTensor[
            _V4_BF16,
            Layout.row_major(stages, _V4_BM * _V4_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V4_BF16,
            Layout.row_major(stages, bn * _V4_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        # One 64 x bn bf16 staging slice per consumer warp group, arranged
        # as consecutive 64x64 boxes in the canonical 128B-swizzled TMA
        # layout expected by the C descriptor.
        var c_staging = LayoutTensor[
            _V4_BF16,
            Layout.row_major(_V4_CONSUMERS, _V4_WG_ROWS * bn),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
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
        if thread_idx.x == 0:
            comptime for stage in range(stages):
                full_barriers[stage].init()
                # Each of the two consumer warp groups in BOTH cluster CTAs
                # signals every slot release (arrive_cluster below).
                empty_barriers[stage].init(Int32(_V4_CONSUMERS * _V4_CLUSTER))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            c_tma.prefetch_descriptor()
        barrier()
        # Peer CTA barriers must be initialized before any cross-CTA arrival.
        cluster_sync()

        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var rank = Int(block_rank_in_cluster())
        var blocks_m = (m + _V4_BM - 1) // _V4_BM
        var blocks_n = (n + bn - 1) // bn
        var total_work = blocks_m * blocks_n
        var num_k_tiles = k // _V4_BK
        var num_pairs = Int(grid_dim.x) // _V4_CLUSTER
        var pair0 = Int(block_idx.x) // _V4_CLUSTER
        var group_span = raster_h * blocks_n

        if warp_group_idx > 0 and warp_group_thread_idx < _V4_CLUSTER:
            comptime for stage in range(stages):
                empty_barriers[stage].arrive_cluster(
                    UInt32(warp_group_thread_idx)
                )
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                # Producer: stream A/B tiles for every assigned output tile
                # through one continuously-cycling pipeline.  The slot/phase
                # arithmetic uses a single global tile counter so consumers
                # stay in lockstep across output-tile boundaries.
                var gkt = 0
                var pair = pair0
                while pair * _V4_CLUSTER < total_work:
                    var my_work = min(pair * _V4_CLUSTER + rank, total_work - 1)
                    var peer_work = min(
                        pair * _V4_CLUSTER + (1 - rank), total_work - 1
                    )
                    # Rasterization: groups of raster_h m-blocks, n-major
                    # inside a group; keeps a wave of CTAs inside a band of
                    # A rows and limits per-pass B re-reads from DRAM.
                    var group = my_work // group_span
                    var rem = my_work % group_span
                    var rows_in_group = min(
                        raster_h, blocks_m - group * raster_h
                    )
                    var m0 = (group * raster_h + rem % rows_in_group) * _V4_BM
                    var n0 = (rem // rows_in_group) * bn
                    var peer_group = peer_work // group_span
                    var peer_rem = peer_work % group_span
                    var peer_rows = min(
                        raster_h, blocks_m - peer_group * raster_h
                    )
                    var peer_n0 = (peer_rem // peer_rows) * bn
                    var can_multicast = peer_n0 == n0
                    var kt = 0
                    while kt < num_k_tiles:
                        var stage = gkt % stages
                        var phase = UInt32((gkt // stages) % 2)
                        empty_barriers[stage].wait(phase)
                        full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                        var a_tile = LayoutTensor[
                            _V4_BF16,
                            _V4_A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr + stage * _V4_BM * _V4_BK)
                        var k0 = kt * _V4_BK
                        a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                        if can_multicast:
                            # Load our half of B, multicast to both CTAs.
                            var b_half = LayoutTensor[
                                _V4_BF16,
                                B_HALF_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr
                                + stage * bn * _V4_BK
                                + rank * B_HALF * _V4_BK
                            )
                            b_tma.async_multicast_load(
                                b_half,
                                full_barriers[stage],
                                (k0, n0 + rank * B_HALF),
                                UInt16(0b11),
                            )
                        else:
                            # Divergent pair: load the full B tile privately.
                            comptime for half in range(_V4_CLUSTER):
                                var b_half = LayoutTensor[
                                    _V4_BF16,
                                    B_HALF_LAYOUT,
                                    MutAnyOrigin,
                                    address_space=AddressSpace.SHARED,
                                    alignment=128,
                                ](
                                    b_pipeline.ptr
                                    + stage * bn * _V4_BK
                                    + half * B_HALF * _V4_BK
                                )
                                b_tma.async_copy(
                                    b_half,
                                    full_barriers[stage],
                                    (k0, n0 + half * B_HALF),
                                )
                        kt += 1
                        gkt += 1
                    pair += num_pairs
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
                _V4_BF16,
                _V4_BF16,
                Index(64, bn, 16),
                a_swizzle=_V4_SWIZZLE,
                b_swizzle=_V4_SWIZZLE,
                transpose_b=True,
            ]()

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var wg_half = c_staging.ptr + (
                (warp_group_idx - 1) * _V4_WG_ROWS * bn
            )

            var gkt = 0
            var pair = pair0
            while pair * _V4_CLUSTER < total_work:
                var my_work = min(pair * _V4_CLUSTER + rank, total_work - 1)
                var group = my_work // group_span
                var rem = my_work % group_span
                var rows_in_group = min(raster_h, blocks_m - group * raster_h)
                var m0 = (group * raster_h + rem % rows_in_group) * _V4_BM
                var n0 = (rem // rows_in_group) * bn

                var kt = 0
                var prev_stage = -1
                while kt < num_k_tiles:
                    var stage = gkt % stages
                    var phase = UInt32((gkt // stages) % 2)
                    full_barriers[stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _V4_BF16,
                        _V4_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * _V4_BM * _V4_BK)
                    var b_tile = LayoutTensor[
                        _V4_BF16,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * bn * _V4_BK)
                    warpgroup_fence(accum)
                    wgmma.arrive()
                    if kt == 0:
                        # scale_c = 0: first k-tile overwrites the
                        # accumulator, so no zero-fill pass is needed.
                        wgmma.wgmma[_V4_CONSUMERS, scale_c=0](
                            a_tile, b_tile, accum, warp_group_idx - 1
                        )
                    else:
                        wgmma.wgmma[_V4_CONSUMERS](
                            a_tile, b_tile, accum, warp_group_idx - 1
                        )
                    wgmma.commit_group()
                    warpgroup_fence(accum)
                    comptime if defer_release:
                        # Keep one WGMMA group in flight so the tensor pipe
                        # never drains between k-tiles; release the PREVIOUS
                        # slot, whose group is provably complete.  Needs a
                        # deeper pipeline (stages >= 4) to avoid starving
                        # the producer.
                        wgmma.wait_group[1]()
                        if (
                            prev_stage >= 0
                            and warp_group_thread_idx < _V4_CLUSTER
                        ):
                            empty_barriers[prev_stage].arrive_cluster(
                                UInt32(warp_group_thread_idx)
                            )
                        prev_stage = stage
                    else:
                        wgmma.wait_group()
                        if warp_group_thread_idx < _V4_CLUSTER:
                            empty_barriers[stage].arrive_cluster(
                                UInt32(warp_group_thread_idx)
                            )
                    kt += 1
                    gkt += 1
                comptime if defer_release:
                    wgmma.wait_group[0]()
                    if prev_stage >= 0 and warp_group_thread_idx < _V4_CLUSTER:
                        empty_barriers[prev_stage].arrive_cluster(
                            UInt32(warp_group_thread_idx)
                        )

                # ---- Epilogue: registers -> swizzled smem -> TMA store ----
                # The previous output tile's TMA store must have drained
                # before this warp group's staging slice is overwritten.  The
                # wait is executed by the issuing thread; the named barrier
                # (one per consumer warp group) releases the rest.
                if warp_group_thread_idx == 0:
                    c_tma.wait_group[0]()
                named_barrier[Int32(128)](Int32(warp_group_idx))

                _v4_store_accum_stmatrix[bn](wg_half, accum, warp, lane)

                # Make generic-proxy smem writes visible to the async proxy,
                # then let one thread issue the (clipped) TMA store.  It is
                # only waited on at the next epilogue, so the store drains
                # while the next output tile's mainloop runs.
                fence_async_view_proxy()
                named_barrier[Int32(128)](Int32(warp_group_idx))
                if warp_group_thread_idx == 0:
                    var c_store = LayoutTensor[
                        _V4_BF16,
                        Layout.row_major(_V4_WG_ROWS, bn),
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](wg_half)
                    c_tma.async_store(
                        c_store,
                        (n0, m0 + (warp_group_idx - 1) * _V4_WG_ROWS),
                    )
                    c_tma.commit_group()
                pair += num_pairs

            if warp_group_thread_idx == 0:
                c_tma.wait_group[0]()

        # Keep the cluster resident until every CTA is done: peer shared
        # memory (barriers) must stay valid for cross-CTA arrivals.
        cluster_sync()


def _v4c_enqueue_nt_persistent[
    bn: Int, stages: Int, raster_h: Int, defer_release: Bool = False
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V4_BF16, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V4_BM, _V4_BK),
    )
    var b_desc = create_tma_descriptor[_V4_BF16, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](n, k),
        IndexList[2](k, 1),
        IndexList[2](bn // _V4_CLUSTER, _V4_BK),
    )
    var c_desc = create_tma_descriptor[_V4_BF16, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            output.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, n),
        IndexList[2](n, 1),
        IndexList[2](_V4_WG_ROWS, _V4_C_BOX_N),
    )
    var a_tma = _V4_A_TMA(a_desc)
    var b_tma = TMATensorTile[
        _V4_BF16,
        2,
        Index(bn // _V4_CLUSTER, _V4_BK),
        Index(bn // _V4_CLUSTER, _V4_BK),
    ](b_desc)
    var c_tma = TMATensorTile[
        _V4_BF16, 2, Index(_V4_WG_ROWS, bn), Index(_V4_WG_ROWS, _V4_C_BOX_N)
    ](c_desc)
    ctx.enqueue_function[
        _v4c_nt_persistent[bn, stages, raster_h, defer_release]
    ](
        a_tma,
        b_tma,
        c_tma,
        m,
        n,
        k,
        grid_dim=(grid_x,),
        block_dim=(_V4_THREADS,),
    )


def maybe_enqueue_bf16_gemm_nt_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Route an NT GEMM to the persistent v4 kernel if the regime allows.

    Returns True when the work was enqueued.  Callers must fall back to the
    v3 dispatcher when False is returned.  Alignment requirements: 16B base
    pointers (TMA), n % 8 for 16B-aligned gmem rows, k % BK for the
    pipeline, and TMA descriptor dimension limits.  m and n need NOT be
    tile-aligned: TMA clips partial edge tiles on load and store.
    """
    comptime if _has_sm_9x():
        if ctx.api() == "cuda":
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if (
                cc_major == 9
                and cc_minor == 0
                and m >= 1
                and n >= _V4_C_BOX_N
                and k >= _V4_BK
                and n % 8 == 0
                and k % _V4_BK == 0
                and Int(output) % 16 == 0
                and Int(a) % 16 == 0
                and Int(b) % 16 == 0
                and m <= 2_147_483_647
                and n <= 2_147_483_647
                and k <= 2_147_483_647
                and k <= 9_223_372_036_854_775_807 // m
                and k <= 9_223_372_036_854_775_807 // n
                and n <= 9_223_372_036_854_775_807 // m
            ):
                var blocks_m = (m + _V4_BM - 1) // _V4_BM
                var sm_count = ctx.get_attribute(
                    DeviceAttribute.MULTIPROCESSOR_COUNT
                )
                var max_grid_x = ctx.get_attribute(
                    DeviceAttribute.MAX_GRID_DIM_X
                )
                var grid_pairs = sm_count // _V4_CLUSTER
                var blocks_n_256 = (n + 255) // 256
                var blocks_n_192 = (n + 191) // 192
                if (
                    blocks_m > 0
                    and grid_pairs > 0
                    and max_grid_x > 0
                    and blocks_m <= max_grid_x // blocks_n_256
                    and blocks_m <= max_grid_x // blocks_n_192
                ):
                    # Persistent-wave cost model: per-cluster time is
                    # (waves) x (per-tile cost ~ BN + fixed per-tile
                    # overhead).  Measured on H100: the fixed overhead makes
                    # BN = 256 win whenever both widths tile n comparably;
                    # BN = 192 only pays off for genuinely narrow n where
                    # the wide tile would compute mostly-clipped columns.
                    var pairs_256 = (
                        blocks_m * blocks_n_256 + _V4_CLUSTER - 1
                    ) // _V4_CLUSTER
                    var pairs_192 = (
                        blocks_m * blocks_n_192 + _V4_CLUSTER - 1
                    ) // _V4_CLUSTER
                    var cost_256 = (
                        (pairs_256 + grid_pairs - 1) // grid_pairs
                    ) * (256 + 32)
                    var cost_192 = (
                        (pairs_192 + grid_pairs - 1) // grid_pairs
                    ) * (192 + 32)
                    if cost_192 * 102 < cost_256 * 100:
                        var grid_x = min(pairs_192, grid_pairs) * _V4_CLUSTER
                        _v4c_enqueue_nt_persistent[192, 4, 16](
                            output, a, b, m, n, k, grid_x, ctx
                        )
                        return True
                    var grid_x = min(pairs_256, grid_pairs) * _V4_CLUSTER
                    _v4c_enqueue_nt_persistent[256, 3, 16](
                        output, a, b, m, n, k, grid_x, ctx
                    )
                    return True
    return False
