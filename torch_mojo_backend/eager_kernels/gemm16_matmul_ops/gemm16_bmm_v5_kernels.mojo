"""Batched persistent H100 16-bit NN BMM — v5.

C[b, m, n] = A[b, m, k] @ B[b, k, n], all row-major per batch item, arbitrary
(element) batch strides including 0 (broadcast operand, e.g. conv's im2col
GEMM where one A weight matrix serves every batch item).

There was previously no batched (grid.z) wgmma NN kernel: BMM NN reached
only the pre-wgmma accepted kernel (gemm16_v3_kernels.mojo's
`_enqueue_accepted_bf16_bmm`), and looping the mm-level `_v4_nn_persistent_ws`
ladder once per batch item (the way #389 serves TN) collapses on the small
per-item grids attention/conv-shaped BMM produce: batch items independently
underfill the GPU, and the loop leaves (SMs - per-item-tiles) idle on every
launch. This kernel instead folds the batch dimension into a single launch's
work list AND its TMA descriptors, so the whole batch fills the machine
together.

Derived from the mm-level `_v4_nn_persistent_ws` body in
gemm16_nn_v4_kernels.mojo (persistent grid, CLUSTER_M x 1 B multicast,
3-stage TMA pipeline, TMA-store epilogue) with the batch dimension folded
into BOTH the work decomposition and the TMA descriptors:

  - Rank-3 TMA descriptors (batch, rows, cols) with a (1, rows, cols) box:
    per-item ragged m and ragged n are clipped by TMA against the item's own
    extents (loads zero-fill, stores clip), so no tile ever reads or writes a
    neighbouring batch item, and non-contiguous batch strides (any multiple
    of 8 elements, or 0 via a coordinate multiplier) ride the descriptor.
  - The persistent work list spans batch_count * macro_rows * blocks_n work
    items, batch-outermost, so small per-item grids (attention/conv-shaped
    BMM) still fill every SM instead of leaving (SMs - per-item-tiles) idle
    per launch the way looping the mm kernel would.
  - A `tiny` sibling body (`_b5_bmm_nn_tiny`) serves the short-k regime:
    one 128-thread CTA per work item, self-issued 2-stage pipeline, 3-4
    CTAs resident per SM (see the dispatch comment in `_b5_dispatch`).

No K-split is implemented: batching alone fills the machine on the deep-K
shapes measured (S4 8x1024x1024x8192 reaches 0.91x of cuBLAS).  If a
batched deep-K underfilled case appears, the work-list decode extends
naturally with a ksplit factor plus an f32 workspace + separate reduce
kernel (never atomics).

The operand dtype is bfloat16 or float16 via _GEMM16_DT, same as the family.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import fence_async_view_proxy, fence_mbarrier_init
from std.memory import AddressSpace
from max.gpu.sync import barrier, named_barrier
from max.gpu.primitives import (
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

from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG
from wgmma_c_epilogue import _wgmma_store_c_tile

comptime _B5_DT = _GEMM16_DT
comptime _B5_F32 = DType.float32
comptime _B5_PTR = UnsafePointer[Scalar[_B5_DT], MutAnyOrigin]
comptime _B5_F32_PTR = UnsafePointer[Scalar[_B5_F32], MutAnyOrigin]
comptime _B5_BK = 64
comptime _B5_GROUP = 4
comptime _B5_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B


@always_inline
def _b5_regime_tag[bm: Int, bn: Int]() -> StaticString:
    comptime if bm == 128:
        return "m128n256"
    comptime if bm == 64 and bn == 256:
        return "m64n256"
    comptime if bm == 64 and bn == 64:
        return "m64n64"
    return "m64n128"


@always_inline
def _b5_store_tag[tma_store: Bool]() -> StaticString:
    comptime if tma_store:
        return ""
    return "_scalar_c"


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
@__name(
    t"{_GEMM16_TAG}_bmm_nn_batched_persistent_{_b5_regime_tag[bm, bn]()}{_b5_store_tag[tma_store]()}"
)
def _b5_bmm_nn_persistent_ws[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool = True,
](
    a_tma: TMATensorTile[_B5_DT, 3, Index(1, bm, _B5_BK), Index(1, bm, _B5_BK)],
    b_tma: TMATensorTile[_B5_DT, 3, Index(1, _B5_BK, 64), Index(1, _B5_BK, 64)],
    c_tma: TMATensorTile[_B5_DT, 3, Index(1, bm, 64), Index(1, bm, 64)],
    output: _B5_PTR,
    c_bs_arg: Int64,
    batch_arg: Int64,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    a_mul_arg: Int64,  # 0 when A is broadcast across the batch, else 1
    b_mul_arg: Int64,  # 0 when B is broadcast across the batch, else 1
):
    # Int is not device-passable; scalars cross the launch ABI as Int64.
    var batch_count = Int(batch_arg)
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var c_bs = Int(c_bs_arg)
    var a_mul = Int(a_mul_arg)
    var b_mul = Int(b_mul_arg)
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_k_major[
            _B5_DT, bm, _B5_BK, _B5_SWIZZLE
        ]()
        comptime B_LAYOUT = tile_layout_mn_major[
            _B5_DT, bn, _B5_BK, _B5_SWIZZLE
        ]()
        comptime B_CHUNK_LAYOUT = tile_layout_mn_major[
            _B5_DT, 64, _B5_BK, _B5_SWIZZLE
        ]()
        comptime A_PIPE_LAYOUT = Layout.row_major(stages, bm * _B5_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(stages, bn * _B5_BK)
        var a_pipeline = LayoutTensor[
            _B5_DT,
            A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _B5_DT,
            B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        # C staging tile for the TMA-store epilogue (swizzled 128B rows of
        # 64 elements, bn // 64 chunks).  A dummy allocation for the scalar
        # epilogue: staging + a coalesced cooperative flush was tried there
        # and measured SLOWER (41.7 -> 56.7 us on 8x357x789x333) -- the
        # flush sits on the consumer's critical path while the direct
        # stores' 82% sector waste is absorbed by L2 (89% hit, DRAM 7%).
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        var c_smem = LayoutTensor[
            _B5_DT,
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
                empty_barriers[stage].init(Int32(consumers * cluster_m))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
            fence_mbarrier_init()
        cluster_sync_relaxed()

        comptime CFRAG = 64 * bn // 128
        comptime MACRO_BM = bm * cluster_m
        comptime TMA_BYTES = (bm + bn) * _B5_BK * 2
        comptime MCAST_MASK = UInt16((1 << cluster_m) - 1)
        comptime B_CHUNKS = bn // 64
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var rank = Int(block_rank_in_cluster())
        var cluster_id = Int(block_idx.x) // cluster_m
        var num_clusters = Int(grid_dim.x) // cluster_m
        # Per-item tiling; m and n may both be ragged (TMA clips against the
        # item's own extents in the rank-3 descriptor).
        var macro_rows = (m + MACRO_BM - 1) // MACRO_BM
        var blocks_n = (n + bn - 1) // bn
        var works_per_item = macro_rows * blocks_n
        var total_works = batch_count * works_per_item
        # k may be ragged: the last box clips at the descriptor's k extent
        # and TMA zero-fills the remainder (zero contributions).  The
        # mbarrier still receives the full box byte count for clipped loads.
        var num_tiles = (k + _B5_BK - 1) // _B5_BK
        var group_span = _B5_GROUP * blocks_n

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
                    var bidx = w // works_per_item
                    var rem_i = w % works_per_item
                    var group = rem_i // group_span
                    var rem = rem_i % group_span
                    var rows_in_group = min(
                        _B5_GROUP, macro_rows - group * _B5_GROUP
                    )
                    var macro_row = group * _B5_GROUP + rem % rows_in_group
                    var n0 = (rem // rows_in_group) * bn
                    var m0 = macro_row * MACRO_BM + rank * bm
                    var ab = bidx * a_mul
                    var bb = bidx * b_mul
                    var t = 0
                    while t < num_tiles:
                        var stage = gt % stages
                        var phase = UInt32((gt // stages) % 2)
                        empty_barriers[stage].wait(phase)
                        full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                        var a_tile = LayoutTensor[
                            _B5_DT,
                            A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr + stage * bm * _B5_BK)
                        var k0 = t * _B5_BK
                        a_tma.async_copy_3d(
                            a_tile, full_barriers[stage], (k0, m0, ab)
                        )
                        # Cooperative B load: each cluster rank reads its
                        # share of the 64-column chunks once from L2 and
                        # multicasts it to every peer.
                        var cc = rank * B_CHUNKS // cluster_m
                        var cend = (rank + 1) * B_CHUNKS // cluster_m
                        while cc < cend:
                            var b_chunk = LayoutTensor[
                                _B5_DT,
                                B_CHUNK_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr
                                + stage * bn * _B5_BK
                                + cc * 64 * _B5_BK
                            )
                            comptime if cluster_m > 1:
                                b_tma.async_multicast_load_3d(
                                    b_chunk,
                                    full_barriers[stage],
                                    (n0 + cc * 64, k0, bb),
                                    MCAST_MASK,
                                )
                            else:
                                b_tma.async_copy_3d(
                                    b_chunk,
                                    full_barriers[stage],
                                    (n0 + cc * 64, k0, bb),
                                )
                            cc += 1
                        t += 1
                        gt += 1
                    w += num_clusters
        else:
            # Consumer register budget: 232 fits two warp groups per SM; the
            # narrow bn=128 tile (64 accum regs) takes 136 so that several
            # CTAs can be resident per SM in the tiny-work regime.
            comptime if consumers >= 3:
                warpgroup_reg_alloc[160]()
            elif bn <= 128:
                warpgroup_reg_alloc[136]()
            else:
                warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _B5_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            comptime wgmma = TensorCoreAsync[
                _B5_F32,
                _B5_DT,
                _B5_DT,
                Index(64, bn, 16),
                a_swizzle=_B5_SWIZZLE,
                b_swizzle=_B5_SWIZZLE,
                transpose_b=False,
            ]()

            var gt = 0
            var w = cluster_id
            while w < total_works:
                var bidx = w // works_per_item
                var rem_i = w % works_per_item
                var group = rem_i // group_span
                var rem = rem_i % group_span
                var rows_in_group = min(
                    _B5_GROUP, macro_rows - group * _B5_GROUP
                )
                var macro_row = group * _B5_GROUP + rem % rows_in_group
                var n0 = (rem // rows_in_group) * bn
                var m0 = macro_row * MACRO_BM + rank * bm
                _ = accum.fill(0.0)
                var t = 0
                while t < num_tiles:
                    var stage = gt % stages
                    var phase = UInt32((gt // stages) % 2)
                    full_barriers[stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _B5_DT,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * bm * _B5_BK)
                    var b_tile = LayoutTensor[
                        _B5_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * bn * _B5_BK)
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
                comptime NCONS = Int32(consumers * 128)
                comptime if tma_store:
                    # Stage the tile in shared memory and hand it to TMA;
                    # the store drains in the background of the next work's
                    # mainloop.  The rank-3 descriptor clips rows past the
                    # item's m edge and columns past its n edge.
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        c_tma.wait_group[0]()
                    named_barrier[NCONS](1)
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        var pair = SIMD[_B5_DT, 2](
                            accum.ptr[e].cast[_B5_DT](),
                            accum.ptr[e + 1].cast[_B5_DT](),
                        )
                        # 128B-swizzled staging layout: 16B units within each
                        # 64-element row are XORed with (row % 8).
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
                                _B5_DT,
                                Layout.row_major(bm, 64),
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](c_smem.ptr + chunk * bm * 64)
                            c_tma.async_store_3d(
                                c_chunk, (n0 + chunk * 64, m0, bidx)
                            )
                        c_tma.commit_group()
                else:
                    # Scalar epilogue for C layouts TMA cannot describe (row
                    # stride n or batch stride c_bs not a multiple of 8
                    # elements).  Element stores, each bounds-guarded; the
                    # last column of an odd n is stored singly.
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        if m0 + row < m:
                            var off = bidx * c_bs + (m0 + row) * n + n0 + col
                            if n0 + col + 1 < n:
                                output.store[alignment=2](
                                    off,
                                    SIMD[_B5_DT, 2](
                                        accum.ptr[e].cast[_B5_DT](),
                                        accum.ptr[e + 1].cast[_B5_DT](),
                                    ),
                                )
                            elif n0 + col < n:
                                output[off] = accum.ptr[e].cast[_B5_DT]()
                w += num_clusters
            # Outstanding bulk stores must complete before kernel exit.
            comptime if tma_store:
                if warp_group_idx == 1 and warp_group_thread_idx == 0:
                    c_tma.wait_group[0]()

        cluster_sync()


@always_inline
def _b5_p(addr: Int) -> _B5_PTR:
    return UnsafePointer[Scalar[_B5_DT], MutUntrackedOrigin](
        unsafe_from_address=addr
    ).as_unsafe_any_origin()


# Tiny-work kernel: one CTA of a single 128-thread warp group per work item,
# no warp specialization, no persistence.  The batched-BMM unaligned corner
# (short k, many small tiles: e.g. 8x357x789x333 has 6 k-tiles per work) is
# latency-bound, not delivery-bound: the persistent warp-specialized body is
# capped at 2 CTAs/SM by its 256-thread/232-reg layout, and a 6-tile
# pipeline never fills, so warps sit at 25% occupancy waiting on barriers.
# Here the same thread group issues its own 2-stage TMA pipeline and wgmmas;
# per-CTA overlap is worse, but 3-4 CTAs fit per SM and cross-CTA overlap
# covers the latency the way vendor small-shape kernels do.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(128))
)
@__name(
    t"{_GEMM16_TAG}_bmm_nn_batched_tiny_{_b5_regime_tag[64, bn]()}{_b5_store_tag[tma_store]()}"
)
def _b5_bmm_nn_tiny[
    stages: Int,
    bn: Int,
    tma_store: Bool,
](
    a_tma: TMATensorTile[_B5_DT, 3, Index(1, 64, _B5_BK), Index(1, 64, _B5_BK)],
    b_tma: TMATensorTile[_B5_DT, 3, Index(1, _B5_BK, 64), Index(1, _B5_BK, 64)],
    c_tma: TMATensorTile[_B5_DT, 3, Index(1, 64, 64), Index(1, 64, 64)],
    output: _B5_PTR,
    c_bs_arg: Int64,
    batch_arg: Int64,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    a_mul_arg: Int64,
    b_mul_arg: Int64,
):
    comptime bm = 64
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var c_bs = Int(c_bs_arg)
    var a_mul = Int(a_mul_arg)
    var b_mul = Int(b_mul_arg)
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_k_major[
            _B5_DT, bm, _B5_BK, _B5_SWIZZLE
        ]()
        comptime B_LAYOUT = tile_layout_mn_major[
            _B5_DT, bn, _B5_BK, _B5_SWIZZLE
        ]()
        comptime B_CHUNK_LAYOUT = tile_layout_mn_major[
            _B5_DT, 64, _B5_BK, _B5_SWIZZLE
        ]()
        var a_pipeline = LayoutTensor[
            _B5_DT,
            Layout.row_major(stages, bm * _B5_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _B5_DT,
            Layout.row_major(stages, bn * _B5_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        var c_smem = LayoutTensor[
            _B5_DT,
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
        if thread_idx.x == 0:
            comptime for stage in range(stages):
                full_barriers[stage].init()
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
        barrier()

        comptime CFRAG = 64 * bn // 128
        comptime TMA_BYTES = (bm + bn) * _B5_BK * 2
        comptime B_CHUNKS = bn // 64
        var blocks_n = (n + bn - 1) // bn
        var macro_rows = (m + bm - 1) // bm
        var works_per_item = macro_rows * blocks_n
        var num_tiles = (k + _B5_BK - 1) // _B5_BK
        # n-fastest decode: consecutive CTAs share the same A row slab.
        var w = Int(block_idx.x)
        var bidx = w // works_per_item
        var rem = w % works_per_item
        var m0 = (rem // blocks_n) * bm
        var n0 = (rem % blocks_n) * bn
        var ab = bidx * a_mul
        var bb = bidx * b_mul

        @parameter
        @always_inline
        def issue_tile(t: Int):
            var stage = t % stages
            full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
            var k0 = t * _B5_BK
            var a_tile = LayoutTensor[
                _B5_DT,
                A_LAYOUT,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
                alignment=128,
            ](a_pipeline.ptr + stage * bm * _B5_BK)
            a_tma.async_copy_3d(a_tile, full_barriers[stage], (k0, m0, ab))
            comptime for cc in range(B_CHUNKS):
                var b_chunk = LayoutTensor[
                    _B5_DT,
                    B_CHUNK_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * bn * _B5_BK + cc * 64 * _B5_BK)
                b_tma.async_copy_3d(
                    b_chunk, full_barriers[stage], (n0 + cc * 64, k0, bb)
                )

        if thread_idx.x == 0:
            var pre = min(stages, num_tiles)
            for t in range(pre):
                issue_tile(t)

        var accum = LayoutTensor[
            _B5_F32,
            Layout.row_major(1, CFRAG),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        _ = accum.fill(0.0)
        comptime wgmma = TensorCoreAsync[
            _B5_F32,
            _B5_DT,
            _B5_DT,
            Index(64, bn, 16),
            a_swizzle=_B5_SWIZZLE,
            b_swizzle=_B5_SWIZZLE,
            transpose_b=False,
        ]()

        var t = 0
        while t < num_tiles:
            var stage = t % stages
            var phase = UInt32((t // stages) % 2)
            full_barriers[stage].wait(phase)
            var a_tile = LayoutTensor[
                _B5_DT,
                A_LAYOUT,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
                alignment=128,
            ](a_pipeline.ptr + stage * bm * _B5_BK)
            var b_tile = LayoutTensor[
                _B5_DT,
                B_LAYOUT,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
                alignment=128,
            ](b_pipeline.ptr + stage * bn * _B5_BK)
            warpgroup_fence(accum)
            wgmma.arrive()
            wgmma.wgmma[1](a_tile, b_tile, accum, 0)
            wgmma.commit_group()
            warpgroup_fence(accum)
            # Full wait before refilling the just-consumed stage.  A
            # wait_group[1] variant that refills the OTHER stage during the
            # current group (issue tile t+1 after g_{t-1} retires) was tried
            # and measured WORSE: AWK 36.9 -> 37.6-38.2, CONV 77.1 -> 78.8,
            # ATT 128.7 -> 127.3 — the refill is not the critical path and
            # the earlier TMA contends with the running wgmma.
            wgmma.wait_group()
            if thread_idx.x == 0 and t + stages < num_tiles:
                issue_tile(t + stages)
            t += 1

        # Both epilogues (128 B-swizzled TMA store and the scalar 4 B
        # fallback) are pure functions of the wgmma fragment layout, and the
        # implicit-GEMM conv kernel writes exactly the same (batch, m, n)
        # tile, so they live in wgmma_c_epilogue.mojo and are shared instead
        # of copied.  `@always_inline`: verified via
        # scripts/compare_kernel_asm.py (sm_90a) that this is a NEW-FUNCTION-
        # BOUNDARY diff, not a logic change -- the scalar-epilogue kernels'
        # PTX shifts by register renumbering and instruction reordering
        # (param-load order), never a changed opcode/operand/immediate. Not
        # byte-identical, so treat that script's "0 kernels changed" bar as
        # not met here; the correctness net is the 140+ gemm16/bmm/conv unit
        # tests plus benchmarks/test_vision.py's mid/awkward conv shapes
        # (both route through this exact scalar-epilogue kernel), all green.
        _wgmma_store_c_tile[_B5_DT, bm, bn, tma_store](
            c_tma, c_smem.ptr, accum.ptr, output, m, n, c_bs, m0, n0, bidx
        )


def _b5_enqueue_tiny[
    stages: Int,
    bn: Int,
    tma_store: Bool,
](
    output: _B5_PTR,
    a: _B5_PTR,
    b: _B5_PTR,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    a_row_stride: Int,
    b_row_stride: Int,
    ctx: DeviceContext,
) raises:
    comptime bm = 64
    var a_items = 1 if a_bs == 0 else batch_count
    var a_stride = m * a_row_stride if a_bs == 0 else a_bs
    var b_items = 1 if b_bs == 0 else batch_count
    var b_stride = k * b_row_stride if b_bs == 0 else b_bs
    var a_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx, a.address_space_cast[AddressSpace.GENERIC](), 1, owning=False
        ),
        IndexList[3](a_items, m, k),
        IndexList[3](a_stride, a_row_stride, 1),
        IndexList[3](1, bm, _B5_BK),
    )
    var b_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx, b.address_space_cast[AddressSpace.GENERIC](), 1, owning=False
        ),
        IndexList[3](b_items, k, n),
        IndexList[3](b_stride, b_row_stride, 1),
        IndexList[3](1, _B5_BK, 64),
    )
    var c_dim0 = batch_count if tma_store else 1
    var c_dim1 = m if tma_store else bm
    var c_dim2 = n if tma_store else 64
    var c_str0 = c_bs if tma_store else bm * 64
    var c_str1 = n if tma_store else 64
    var c_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx,
            output.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[3](c_dim0, c_dim1, c_dim2),
        IndexList[3](c_str0, c_str1, 1),
        IndexList[3](1, bm, 64),
    )
    var a_tma = TMATensorTile[
        _B5_DT, 3, Index(1, bm, _B5_BK), Index(1, bm, _B5_BK)
    ](a_desc)
    var b_tma = TMATensorTile[
        _B5_DT, 3, Index(1, _B5_BK, 64), Index(1, _B5_BK, 64)
    ](b_desc)
    var c_tma = TMATensorTile[_B5_DT, 3, Index(1, bm, 64), Index(1, bm, 64)](
        c_desc
    )
    var macro_rows = (m + bm - 1) // bm
    var blocks_n = (n + bn - 1) // bn
    var total_works = batch_count * macro_rows * blocks_n
    ctx.enqueue_function[_b5_bmm_nn_tiny[stages, bn, tma_store]](
        a_tma,
        b_tma,
        c_tma,
        output,
        Int64(c_bs),
        Int64(batch_count),
        Int64(m),
        Int64(n),
        Int64(k),
        Int64(0 if a_bs == 0 else 1),
        Int64(0 if b_bs == 0 else 1),
        grid_dim=(total_works,),
        block_dim=(128,),
    )


def _b5_enqueue_batched[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool,
](
    output: _B5_PTR,
    a: _B5_PTR,
    b: _B5_PTR,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    a_row_stride: Int,
    b_row_stride: Int,
    sm_count: Int,
    ctx: DeviceContext,
    occ: Int = 1,
) raises:
    # A broadcast operand (batch stride 0) is described as a single-item
    # tensor and addressed with a zero coordinate multiplier; the dummy
    # stride is never dereferenced because the batch coordinate is always 0.
    var a_items = 1 if a_bs == 0 else batch_count
    var a_stride = m * a_row_stride if a_bs == 0 else a_bs
    var b_items = 1 if b_bs == 0 else batch_count
    var b_stride = k * b_row_stride if b_bs == 0 else b_bs
    var a_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx, a.address_space_cast[AddressSpace.GENERIC](), 1, owning=False
        ),
        IndexList[3](a_items, m, k),
        IndexList[3](a_stride, a_row_stride, 1),
        IndexList[3](1, bm, _B5_BK),
    )
    var b_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx, b.address_space_cast[AddressSpace.GENERIC](), 1, owning=False
        ),
        IndexList[3](b_items, k, n),
        IndexList[3](b_stride, b_row_stride, 1),
        IndexList[3](1, _B5_BK, 64),
    )
    # Without the TMA-store epilogue the C descriptor is a never-used dummy
    # over the output base (the kernel gates every c_tma call off).
    var c_dim0 = batch_count if tma_store else 1
    var c_dim1 = m if tma_store else bm
    var c_dim2 = n if tma_store else 64
    var c_str0 = c_bs if tma_store else bm * 64
    var c_str1 = n if tma_store else 64
    var c_desc = create_tma_descriptor[_B5_DT, 3, _B5_SWIZZLE](
        DeviceBuffer(
            ctx,
            output.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[3](c_dim0, c_dim1, c_dim2),
        IndexList[3](c_str0, c_str1, 1),
        IndexList[3](1, bm, 64),
    )
    var a_tma = TMATensorTile[
        _B5_DT, 3, Index(1, bm, _B5_BK), Index(1, bm, _B5_BK)
    ](a_desc)
    var b_tma = TMATensorTile[
        _B5_DT, 3, Index(1, _B5_BK, 64), Index(1, _B5_BK, 64)
    ](b_desc)
    var c_tma = TMATensorTile[_B5_DT, 3, Index(1, bm, 64), Index(1, bm, 64)](
        c_desc
    )
    var macro_rows = (m + bm * cluster_m - 1) // (bm * cluster_m)
    var blocks_n = (n + bn - 1) // bn
    var total_works = batch_count * macro_rows * blocks_n
    # occ > 1 launches several persistent CTAs per SM (the instantiation must
    # fit the smem/register budget): in the tiny-mainloop regime one CTA's
    # 6-tile pipeline cannot cover its own stalls, and concurrent works per
    # SM overlap them instead.
    var num_clusters = min(occ * (sm_count // cluster_m), total_works)
    var grid_x = num_clusters * cluster_m
    ctx.enqueue_function[
        _b5_bmm_nn_persistent_ws[
            stages, cluster_m, bm, bn, consumers, tma_store
        ]
    ](
        a_tma,
        b_tma,
        c_tma,
        output,
        Int64(c_bs),
        Int64(batch_count),
        Int64(m),
        Int64(n),
        Int64(k),
        Int64(0 if a_bs == 0 else 1),
        Int64(0 if b_bs == 0 else 1),
        grid_dim=(grid_x,),
        block_dim=(128 * (consumers + 1),),
    )


def _b5_dispatch[
    tma_store: Bool
](
    output: _B5_PTR,
    a: _B5_PTR,
    b: _B5_PTR,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    a_row_stride: Int,
    b_row_stride: Int,
    sm_count: Int,
    ctx: DeviceContext,
) raises:
    # Tile-regime dispatch, fitted on H100 PCIe (114 SMs):
    #   - narrow n first: a 256-wide B tile on an n=64 item (attn @ V has
    #     n = head_dim) computes 75% garbage columns.  bn tracks n up to one
    #     ragged step: n <= 96 -> 64-wide, n <= 192 -> 128-wide.
    #   - 128x256 / cluster-2 / 2 consumers (the mm v4 production config)
    #     when the per-256-row-macro grid fills at least one persistent wave
    #     of clusters: works_big >= sm_count means >= 2 works per cluster,
    #     where its B multicast and bigger tile amortize best.
    #   - 64x256 / cluster-1 / 1 consumer otherwise: one warp-group row
    #     tiles, so items with m < 256 (attention/conv-shaped BMM) never
    #     waste a cluster CTA past the item's m edge and small work lists
    #     get 4x the schedulable work items.
    # Tiny-work route: short per-work k (the pipeline never fills) plus a
    # work list big enough that cross-CTA overlap can replace pipeline
    # depth.  The persistent warp-specialized body is capped at 2 CTAs/SM
    # (256 threads x 114-168 static regs); the 128-thread/80-reg tiny body
    # fits 3-4 and measured 41.9 -> 36.7 us on 8x357x789x333 (6 k-tiles),
    # 79.7 -> 77.0 us on the conv im2col 32x64x3136x576 (9 k-tiles) and
    # 143.4 -> 128.4 us on attn@V 96x1024x64x1024 (16 k-tiles).  The
    # 16-tile cut is the deepest k measured to win; the persistent body
    # wins at 128 k-tiles (S4 deep-K, 0.89x) and the band in between is
    # unmeasured.  2*sm_count works is where the extra CTAs/SM matter.
    var num_tiles = (k + _B5_BK - 1) // _B5_BK
    var works_64 = batch_count * ((m + 63) // 64)
    if num_tiles <= 16 and works_64 * ((n + 127) // 128) >= 2 * sm_count:
        if n <= 96:
            _b5_enqueue_tiny[2, 64, tma_store](
                output,
                a,
                b,
                batch_count,
                m,
                n,
                k,
                c_bs,
                a_bs,
                b_bs,
                a_row_stride,
                b_row_stride,
                ctx,
            )
        else:
            _b5_enqueue_tiny[2, 128, tma_store](
                output,
                a,
                b,
                batch_count,
                m,
                n,
                k,
                c_bs,
                a_bs,
                b_bs,
                a_row_stride,
                b_row_stride,
                ctx,
            )
        return
    if n <= 96:
        _b5_enqueue_batched[3, 1, 64, 64, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
        return
    if n <= 192:
        _b5_enqueue_batched[3, 1, 64, 128, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
        return
    var use_big = False
    if m >= 256:
        var works_big = ((m + 255) // 256) * ((n + 255) // 256) * batch_count
        use_big = works_big >= sm_count
    if use_big:
        _b5_enqueue_batched[3, 2, 128, 256, 2, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
    else:
        _b5_enqueue_batched[3, 1, 64, 256, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )


# Strided row repack of up to two operands in ONE launch:
#   dst[(item * rows_per_item + row) * dst_row_stride + c]
#     = src[item * src_item_stride + row * src_row_stride + c]
# Used to give TMA-incompatible operands (row stride or batch stride not a
# multiple of 8 elements) a compatible workspace copy.  The padding tail of
# each dst row is left unwritten: the GEMM descriptor's extent is the REAL
# row length, so TMA never reads the pad.  One warp per row (a 789-element
# bf16 row is ~1.5KB: block-per-row mappings leave most of the block idle),
# rows of operand 0 first, then operand 1; either operand may be empty.
@__name(t"{_GEMM16_TAG}_bmm_row_repack2")
def _b5_repack2_kernel(
    dst0: _B5_PTR,
    src0: _B5_PTR,
    rows0_arg: Int64,
    rpi0_arg: Int64,
    len0_arg: Int64,
    sis0_arg: Int64,
    srs0_arg: Int64,
    drs0_arg: Int64,
    dst1: _B5_PTR,
    src1: _B5_PTR,
    rows1_arg: Int64,
    rpi1_arg: Int64,
    len1_arg: Int64,
    sis1_arg: Int64,
    srs1_arg: Int64,
    drs1_arg: Int64,
):
    var rows0 = Int(rows0_arg)
    var rows1 = Int(rows1_arg)
    var total = rows0 + rows1
    comptime WARPS = 8  # 256-thread blocks, one warp per row (256 threads)
    var lane = Int(thread_idx.x) % 32
    var r = Int(block_idx.x) * WARPS + Int(thread_idx.x) // 32
    var rstride = Int(grid_dim.x) * WARPS
    while r < total:
        var src: _B5_PTR
        var dst: _B5_PTR
        var row: Int
        var rpi: Int
        var row_len: Int
        var sis: Int
        var srs: Int
        var drs: Int
        if r < rows0:
            src = src0
            dst = dst0
            row = r
            rpi = Int(rpi0_arg)
            row_len = Int(len0_arg)
            sis = Int(sis0_arg)
            srs = Int(srs0_arg)
            drs = Int(drs0_arg)
        else:
            src = src1
            dst = dst1
            row = r - rows0
            rpi = Int(rpi1_arg)
            row_len = Int(len1_arg)
            sis = Int(sis1_arg)
            srs = Int(srs1_arg)
            drs = Int(drs1_arg)
        var it = row // rpi
        var src_base = it * sis + (row % rpi) * srs
        var dst_base = row * drs
        # dst rows start 16B-aligned (padded stride, aligned workspace
        # base), so 8-wide chunks store as full 16B sectors; src rows may
        # start anywhere, so the loads declare 2B alignment and the backend
        # legalizes them (same idiom as the v2 kernels' guarded loads).
        var c = lane * 8
        while c + 8 <= row_len:
            var v = src.load[width=8, alignment=2](src_base + c)
            dst.store[alignment=16](dst_base + c, v)
            c += 32 * 8
        if c < row_len:
            comptime for e in range(8):
                if c + e < row_len:
                    dst[dst_base + c + e] = src[src_base + c + e]
        r += rstride


comptime _B5_UPTR = UnsafePointer[Scalar[_B5_DT], MutUntrackedOrigin]


struct _B5RepackOp(ImplicitlyCopyable):
    var dst: _B5_UPTR
    var src: _B5_UPTR
    var rows: Int
    var rows_per_item: Int
    var row_len: Int
    var src_item_stride: Int
    var src_row_stride: Int
    var dst_row_stride: Int

    def __init__(out self, dst: _B5_UPTR, src: _B5_UPTR):
        """An inactive op (rows = 0); the pointers are placeholders that the
        kernel never dereferences."""
        self.dst = dst
        self.src = src
        self.rows = 0
        self.rows_per_item = 1
        self.row_len = 0
        self.src_item_stride = 0
        self.src_row_stride = 0
        self.dst_row_stride = 0


def _b5_repack2(ctx: DeviceContext, op0: _B5RepackOp, op1: _B5RepackOp) raises:
    var total = op0.rows + op1.rows
    var grid = min((total + 7) // 8, 4096)
    if grid < 1:
        return
    ctx.enqueue_function[_b5_repack2_kernel](
        op0.dst.as_unsafe_any_origin(),
        op0.src.as_unsafe_any_origin(),
        Int64(op0.rows),
        Int64(op0.rows_per_item),
        Int64(op0.row_len),
        Int64(op0.src_item_stride),
        Int64(op0.src_row_stride),
        Int64(op0.dst_row_stride),
        op1.dst.as_unsafe_any_origin(),
        op1.src.as_unsafe_any_origin(),
        Int64(op1.rows),
        Int64(op1.rows_per_item),
        Int64(op1.row_len),
        Int64(op1.src_item_stride),
        Int64(op1.src_row_stride),
        Int64(op1.dst_row_stride),
        grid_dim=(grid,),
        block_dim=(32 * 8,),  # must equal 32 * WARPS in the kernel
    )


def _b5_dispatch_variant[
    tma_store: Bool
](
    variant: Int,
    output: _B5_PTR,
    a: _B5_PTR,
    b: _B5_PTR,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    a_row_stride: Int,
    b_row_stride: Int,
    sm_count: Int,
    ctx: DeviceContext,
) raises:
    """Tuning-sweep dispatch: variant 0 is the production rule; the rest are
    fixed instantiations (smem/reg budgets allow the listed CTAs per SM on
    H100: 227 KB smem, 64K regs)."""
    if variant == 1:
        # 2-stage 64x256, 2 CTAs/SM (80KB smem, 232-reg consumer just fits)
        _b5_enqueue_batched[2, 1, 64, 256, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
            occ=2,
        )
    elif variant == 2:
        # 3-stage 64x128, 3 CTAs/SM scalar-C / 2 with TMA store
        _b5_enqueue_batched[3, 1, 64, 128, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
            occ=3 if not tma_store else 2,
        )
    elif variant == 3:
        # 2-stage 64x128, 4 CTAs/SM scalar-C / 3 with TMA store
        _b5_enqueue_batched[2, 1, 64, 128, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
            occ=4 if not tma_store else 3,
        )
    elif variant == 4:
        # 3-stage 64x256 at 1 CTA/SM (the pre-sweep small-tile default)
        _b5_enqueue_batched[3, 1, 64, 256, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
    elif variant == 5:
        # tiny one-CTA-per-work, 2-stage, 64x128
        _b5_enqueue_tiny[2, 128, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            ctx,
        )
    elif variant == 6:
        # tiny one-CTA-per-work, 2-stage, 64x256
        _b5_enqueue_tiny[2, 256, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            ctx,
        )
    elif variant == 7:
        # tiny one-CTA-per-work, 3-stage, 64x128
        _b5_enqueue_tiny[3, 128, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            ctx,
        )
    elif variant == 8:
        # force the big 128x256 cluster-2 instantiation (tests)
        _b5_enqueue_batched[3, 2, 128, 256, 2, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
    elif variant == 9:
        # tiny one-CTA-per-work, 2-stage, 64x64
        _b5_enqueue_tiny[2, 64, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            ctx,
        )
    elif variant == 10:
        # force the persistent 64x64 narrow-n instantiation (tests)
        _b5_enqueue_batched[3, 1, 64, 64, 1, tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )
    else:
        _b5_dispatch[tma_store](
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            a_row_stride,
            b_row_stride,
            sm_count,
            ctx,
        )


def try_enqueue_bmm16_nn_batched(
    output: _B5_PTR,
    a: _B5_PTR,
    b: _B5_PTR,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    ctx: DeviceContext,
    variant: Int = 0,
) raises -> Bool:
    """Route a strided NN BMM to the batched persistent kernel.

    Direct TMA route when every stride is 16B-compatible; otherwise a repack
    route copies the offending operand(s) into stride-padded workspaces and
    C falls back to the scalar epilogue when its own strides are unaligned.
    Returns False when the caller must fall back."""
    comptime if not _has_sm_9x():
        return False
    if ctx.api() != "cuda":
        return False
    var cc_major = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
    var cc_minor = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
    if cc_major != 9 or cc_minor != 0:
        return False
    # Batched TMA regime: boxes must fit inside one item (m, n, k >= 64; k
    # itself may be ragged past the last full tile -- TMA clips and
    # zero-fills).  The ordered bounds make every descriptor and address
    # product machine-width safe.
    if (
        batch_count < 1
        or m < 64
        or n < 64
        or k < _B5_BK
        or a_bs < 0
        or b_bs < 0
        or c_bs < n * (m - 1) + n  # items must not overlap in C
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        or batch_count > 2_147_483_647
        or m > 2_147_483_647
        or n > 2_147_483_647
        or k > 2_147_483_647
        or k > 9_223_372_036_854_775_807 // m
        or k > 9_223_372_036_854_775_807 // n
        or n > 9_223_372_036_854_775_807 // m
        or batch_count > 9_223_372_036_854_775_807 // (m * n)
        or (a_bs > 0 and batch_count > 9_223_372_036_854_775_807 // a_bs)
        or (b_bs > 0 and batch_count > 9_223_372_036_854_775_807 // b_bs)
    ):
        return False
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    if sm_count < 2:
        return False
    var a_ok = k % 8 == 0 and a_bs % 8 == 0
    var b_ok = n % 8 == 0 and b_bs % 8 == 0
    var c_ok = n % 8 == 0 and c_bs % 8 == 0
    if a_ok and b_ok and c_ok:
        _b5_dispatch_variant[True](
            variant,
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a_bs,
            b_bs,
            k,
            n,
            sm_count,
            ctx,
        )
        return True
    # Repack route: stride-pad the TMA-incompatible operand(s) into ONE
    # shared workspace (row strides rounded up to 8 elements; the pad tail
    # is never read because the descriptor extents stay the real lengths).
    # C keeps its caller layout via the scalar epilogue when unaligned.
    # A single allocation serves both operands: the per-call allocator
    # bookkeeping is half of the non-kernel overhead of this route.
    var kp = ((k + 7) // 8) * 8
    var np = ((n + 7) // 8) * 8
    var a_items = 1 if a_bs == 0 else batch_count
    var b_items = 1 if b_bs == 0 else batch_count
    var a_ws_elems = 0 if a_ok else a_items * m * kp
    var b_ws_elems = 0 if b_ok else b_items * k * np
    var ws = ctx.enqueue_create_buffer[_B5_DT](max(1, a_ws_elems + b_ws_elems))
    var ws_base = Int(ws.unsafe_ptr())
    var a2 = a
    var a2_bs = a_bs
    var a2_row = k
    var op_a = _B5RepackOp(
        _B5_UPTR(unsafe_from_address=ws_base),
        _B5_UPTR(unsafe_from_address=Int(a)),
    )
    if not a_ok:
        op_a.rows = a_items * m
        op_a.rows_per_item = m
        op_a.row_len = k
        op_a.src_item_stride = a_bs
        op_a.src_row_stride = k
        op_a.dst_row_stride = kp
        a2 = _b5_p(ws_base)
        a2_bs = 0 if a_bs == 0 else m * kp
        a2_row = kp
    var b2 = b
    var b2_bs = b_bs
    var b2_row = n
    # B's slice starts after A's (kp % 8 == 0 keeps it 16B-aligned).
    var b_ws_base = ws_base + a_ws_elems * 2
    var op_b = _B5RepackOp(
        _B5_UPTR(unsafe_from_address=b_ws_base),
        _B5_UPTR(unsafe_from_address=Int(b)),
    )
    if not b_ok:
        op_b.rows = b_items * k
        op_b.rows_per_item = k
        op_b.row_len = n
        op_b.src_item_stride = b_bs
        op_b.src_row_stride = n
        op_b.dst_row_stride = np
        b2 = _b5_p(b_ws_base)
        b2_bs = 0 if b_bs == 0 else k * np
        b2_row = np
    _b5_repack2(ctx, op_a, op_b)
    if c_ok:
        _b5_dispatch_variant[True](
            variant,
            output,
            a2,
            b2,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a2_bs,
            b2_bs,
            a2_row,
            b2_row,
            sm_count,
            ctx,
        )
    else:
        _b5_dispatch_variant[False](
            variant,
            output,
            a2,
            b2,
            batch_count,
            m,
            n,
            k,
            c_bs,
            a2_bs,
            b2_bs,
            a2_row,
            b2_row,
            sm_count,
            ctx,
        )
    return True
