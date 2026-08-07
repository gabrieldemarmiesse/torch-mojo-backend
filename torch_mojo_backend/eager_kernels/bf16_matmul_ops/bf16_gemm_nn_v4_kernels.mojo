"""Persistent clustered H100 BF16 NN GEMM (dgrad) kernels — v4.

C[m, n] = A[m, k] @ B[k, n] with both operands row-major.

Design, motivated by ncu on the nanogpt dgrad shapes (m=32768, short k):
  - The v3 NN kernel is operand-delivery bound, not DRAM bound: DRAM sits at
    ~18%, L2 at ~66%, the tensor pipes at ~49%, and the dominant warp stall
    is consumers waiting on TMA full barriers.  Every k-tile moves
    (BM + BN) * BK * 2 bytes from L2 into each CTA.
  - CTA clusters of CLUSTER_M x 1 multicast each B tile to the CLUSTER_M
    CTAs that share it (same n0, adjacent m0), removing a third of the
    per-CTA L2 traffic at BM=128 / BN=256.
  - A persistent grid (one CTA per SM, grid == SM count) removes the
    per-wave pipeline refill and epilogue serialization of the former
    multi-wave launch: the producer warp group prefetches the next work
    tile's operands while the consumers are still storing the previous
    accumulators.
  - A TMA-store epilogue (the single largest win, ~20%): accumulators are
    staged in a 128B-swizzled shared-memory tile and handed to TMA, which
    drains the store in the background of the next work tile's mainloop.
    The former scalar epilogue both serialized ~4-5us per work tile and
    inflated L2 write traffic ~3x through partial-sector stores.
  - A 192x192 / 3-consumer tile (nvjet's pick) was also implemented and
    benched; 128x256 with 2 consumers won on every nanogpt dgrad shape.

Dynamic shapes: any problem in the tall-m NN regime with n % BN == 0 and
k % BK == 0 is handled (m may be ragged: TMA clamps loads and clips
stores); everything else must be routed to the existing v3 dispatcher by
the caller (`maybe_enqueue_...` returns False in that case).
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import (
    AddressSpace,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from std.gpu.sync import named_barrier
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

comptime _V4_BF16 = DType.bfloat16
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_BF16], MutAnyOrigin]
comptime _V4_BK = 64
# Macro-rows per rasterization group: consecutive work indices cover
# _V4_GROUP macro rows before advancing one BN column, keeping the in-flight
# A slab and the current B column resident in L2.
comptime _V4_GROUP = 4
comptime _V4_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B

# Production configuration (best of the kernel_bench sweeps: "s3c2ts").
comptime _V4_PROD_STAGES = 3
comptime _V4_PROD_CLUSTER_M = 2
comptime _V4_PROD_BM = 128
comptime _V4_PROD_BN = 256
comptime _V4_PROD_CONSUMERS = 2
comptime _V4_PROD_TMA_STORE = True


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
def _v4_nn_persistent_ws[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool,
](
    a_tma: TMATensorTile[_V4_BF16, 2, Index(bm, _V4_BK), Index(bm, _V4_BK)],
    b_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, 64), Index(_V4_BK, 64)],
    c_tma: TMATensorTile[_V4_BF16, 2, Index(bm, 64), Index(bm, 64)],
    output: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_k_major[
            _V4_BF16, bm, _V4_BK, _V4_SWIZZLE
        ]()
        comptime B_LAYOUT = tile_layout_mn_major[
            _V4_BF16, bn, _V4_BK, _V4_SWIZZLE
        ]()
        comptime B_CHUNK_LAYOUT = tile_layout_mn_major[
            _V4_BF16, 64, _V4_BK, _V4_SWIZZLE
        ]()
        comptime A_PIPE_LAYOUT = Layout.row_major(stages, bm * _V4_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(stages, bn * _V4_BK)
        var a_pipeline = LayoutTensor[
            _V4_BF16,
            A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V4_BF16,
            B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        # C staging tile for the TMA-store epilogue (swizzled 128B rows of
        # 64 elements, bn // 64 chunks).  A dummy allocation when disabled.
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        var c_smem = LayoutTensor[
            _V4_BF16,
            Layout.row_major(1, C_SMEM_ELEMS),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=1024,
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
                # Released by every consumer warp group of every CTA in the
                # cluster: the multicast source must not overwrite a peer's
                # tile while that peer is still reading it.
                empty_barriers[stage].init(Int32(consumers * cluster_m))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
            fence_mbarrier_init()
        # All barriers must be initialized cluster-wide before any arrival
        # (the consumers below arrive at peer CTAs' empty barriers).
        cluster_sync_relaxed()

        comptime CFRAG = 64 * bn // 128
        comptime MACRO_BM = bm * cluster_m
        comptime TMA_BYTES = (bm + bn) * _V4_BK * 2
        comptime MCAST_MASK = UInt16((1 << cluster_m) - 1)
        comptime B_CHUNKS = bn // 64
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var rank = Int(block_rank_in_cluster())
        var cluster_id = Int(block_idx.x) // cluster_m
        var num_clusters = Int(grid_dim.x) // cluster_m
        # m may be ragged: TMA A reads clamp out-of-bounds rows and the
        # epilogue stores are row-predicated.
        var macro_rows = (m + MACRO_BM - 1) // MACRO_BM
        var blocks_n = n // bn
        var total_works = macro_rows * blocks_n
        var num_tiles = k // _V4_BK
        var group_span = _V4_GROUP * blocks_n

        # Release every pipeline slot to the producers (cluster-wide).
        if warp_group_idx > 0 and warp_group_thread_idx < cluster_m:
            comptime for stage in range(stages):
                empty_barriers[stage].arrive_cluster(
                    UInt32(warp_group_thread_idx)
                )

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if warp_group_thread_idx == 0:
                var gt = 0
                var w = cluster_id
                while w < total_works:
                    var group = w // group_span
                    var rem = w % group_span
                    var rows_in_group = min(
                        _V4_GROUP, macro_rows - group * _V4_GROUP
                    )
                    var macro_row = group * _V4_GROUP + rem % rows_in_group
                    var n0 = (rem // rows_in_group) * bn
                    var m0 = macro_row * MACRO_BM + rank * bm
                    var t = 0
                    while t < num_tiles:
                        var stage = gt % stages
                        var phase = UInt32((gt // stages) % 2)
                        empty_barriers[stage].wait(phase)
                        full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                        var a_tile = LayoutTensor[
                            _V4_BF16,
                            A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr + stage * bm * _V4_BK)
                        var k0 = t * _V4_BK
                        a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                        # Cooperative B load: each cluster rank reads its
                        # share of the 64-column chunks once from L2 and
                        # multicasts it to every peer, so the per-SM TMA
                        # engines split the shared-tile traffic instead of
                        # rank 0 funneling all of B (nvjet's "coopB").
                        var cc = rank * B_CHUNKS // cluster_m
                        var cend = (rank + 1) * B_CHUNKS // cluster_m
                        while cc < cend:
                            var b_chunk = LayoutTensor[
                                _V4_BF16,
                                B_CHUNK_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr
                                + stage * bn * _V4_BK
                                + cc * 64 * _V4_BK
                            )
                            comptime if cluster_m > 1:
                                b_tma.async_multicast_load(
                                    b_chunk,
                                    full_barriers[stage],
                                    (n0 + cc * 64, k0),
                                    MCAST_MASK,
                                )
                            else:
                                b_tma.async_copy(
                                    b_chunk,
                                    full_barriers[stage],
                                    (n0 + cc * 64, k0),
                                )
                            cc += 1
                        t += 1
                        gt += 1
                    w += num_clusters
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
                _V4_BF16,
                _V4_BF16,
                Index(64, bn, 16),
                a_swizzle=_V4_SWIZZLE,
                b_swizzle=_V4_SWIZZLE,
                transpose_b=False,
            ]()

            var gt = 0
            var w = cluster_id
            while w < total_works:
                var group = w // group_span
                var rem = w % group_span
                var rows_in_group = min(
                    _V4_GROUP, macro_rows - group * _V4_GROUP
                )
                var macro_row = group * _V4_GROUP + rem % rows_in_group
                var n0 = (rem // rows_in_group) * bn
                var m0 = macro_row * MACRO_BM + rank * bm
                _ = accum.fill(0.0)
                var t = 0
                while t < num_tiles:
                    var stage = gt % stages
                    var phase = UInt32((gt // stages) % 2)
                    full_barriers[stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _V4_BF16,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * bm * _V4_BK)
                    var b_tile = LayoutTensor[
                        _V4_BF16,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * bn * _V4_BK)
                    warpgroup_fence(accum)
                    wgmma.arrive()
                    wgmma.wgmma[consumers](
                        a_tile, b_tile, accum, warp_group_idx - 1
                    )
                    wgmma.commit_group()
                    warpgroup_fence(accum)
                    wgmma.wait_group()
                    if warp_group_thread_idx < cluster_m:
                        empty_barriers[stage].arrive_cluster(
                            UInt32(warp_group_thread_idx)
                        )
                    t += 1
                    gt += 1

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
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        var pair = SIMD[_V4_BF16, 2](
                            accum.ptr[e].cast[_V4_BF16](),
                            accum.ptr[e + 1].cast[_V4_BF16](),
                        )
                        # 128B-swizzled staging layout: 16B units within
                        # each 64-element row are XORed with (row % 8).
                        var lcol = col % 64
                        var elem = (
                            (col // 64) * (bm * 64)
                            + row * 64
                            + ((lcol // 8) ^ (row % 8)) * 8
                            + lcol % 8
                        )
                        c_smem.ptr.store[alignment=4](elem, pair)
                    fence_async_view_proxy()
                    named_barrier[NCONS](1)
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        comptime for chunk in range(bn // 64):
                            var c_chunk = LayoutTensor[
                                _V4_BF16,
                                Layout.row_major(bm, 64),
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](c_smem.ptr + chunk * bm * 64)
                            c_tma.async_store(c_chunk, (n0 + chunk * 64, m0))
                        c_tma.commit_group()
                else:
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        var pair = SIMD[_V4_BF16, 2](
                            accum.ptr[e].cast[_V4_BF16](),
                            accum.ptr[e + 1].cast[_V4_BF16](),
                        )
                        if m0 + row < m and n0 + col + 1 < n:
                            output.store[alignment=4](
                                (m0 + row) * n + n0 + col, pair
                            )
                w += num_clusters
            comptime if tma_store:
                # Outstanding bulk stores must complete before kernel exit.
                if warp_group_idx == 1 and warp_group_thread_idx == 0:
                    c_tma.wait_group[0]()

        # Peer CTAs receive multicast writes into this CTA's shared memory;
        # do not tear the block down while any cluster member is running.
        cluster_sync()


def _v4_enqueue_nn_persistent[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool = False,
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    sm_count: Int,
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
        IndexList[2](bm, _V4_BK),
    )
    var b_desc = create_tma_descriptor[_V4_BF16, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V4_BK, 64),
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
        IndexList[2](bm, 64),
    )
    var a_tma = TMATensorTile[
        _V4_BF16, 2, Index(bm, _V4_BK), Index(bm, _V4_BK)
    ](a_desc)
    var b_tma = TMATensorTile[
        _V4_BF16, 2, Index(_V4_BK, 64), Index(_V4_BK, 64)
    ](b_desc)
    var c_tma = TMATensorTile[_V4_BF16, 2, Index(bm, 64), Index(bm, 64)](c_desc)
    var macro_rows = (m + bm * cluster_m - 1) // (bm * cluster_m)
    var total_works = macro_rows * (n // bn)
    var num_clusters = min(sm_count // cluster_m, total_works)
    var grid_x = num_clusters * cluster_m
    ctx.enqueue_function[
        _v4_nn_persistent_ws[stages, cluster_m, bm, bn, consumers, tma_store]
    ](
        a_tma,
        b_tma,
        c_tma,
        output,
        m,
        n,
        k,
        grid_dim=(grid_x,),
        block_dim=(128 * (consumers + 1),),
    )


def maybe_enqueue_bf16_gemm_nn_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    """Route an NN GEMM to the persistent clustered v4 kernel if it fits the
    tall-m aligned regime.  Returns False when the caller must fall back."""
    comptime if _has_sm_9x():
        if ctx.api() == "cuda":
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if cc_major == 9 and cc_minor == 0:
                # Same tall-m NN regime as the v3 dispatcher.  m may be
                # ragged (TMA clamps reads, stores are predicated); n and k
                # must tile exactly.  The ordered bounds make all descriptor
                # and address products machine-width safe.
                if (
                    not transpose_a
                    and not transpose_b
                    and not has_bias
                    and m >= _V4_PROD_BM * _V4_PROD_CLUSTER_M
                    and n >= _V4_PROD_BN
                    and k >= _V4_BK
                    and m // n >= 8
                    and n % _V4_PROD_BN == 0
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
                    var sm_count = ctx.get_attribute(
                        DeviceAttribute.MULTIPROCESSOR_COUNT
                    )
                    if sm_count >= _V4_PROD_CLUSTER_M:
                        _v4_enqueue_nn_persistent[
                            _V4_PROD_STAGES,
                            _V4_PROD_CLUSTER_M,
                            _V4_PROD_BM,
                            _V4_PROD_BN,
                            _V4_PROD_CONSUMERS,
                            _V4_PROD_TMA_STORE,
                        ](output, a, b, m, n, k, sm_count, ctx)
                        return True
    return False
