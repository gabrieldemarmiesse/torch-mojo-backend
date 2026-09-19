"""Fused-bias bf16 NT persistent GEMM candidate; dynamic m, n, k.

Derived from upstream gemm16_nt_v4_kernels.mojo.  Its SM90 clustered TMA
multicast pipeline, persistent scheduler, barriers, TMA stores and dispatch
cost model are preserved.  Only the epilogue and explicit bias launch
argument are new: add the column bias to fp32 WGMMA accumulators before
bf16 conversion and st.matrix staging.  Bias is a contiguous bf16 vector
of length n and is never written.  Bias loads are independently clipped.

Tuning provenance: imported BM=128, BK=64, cluster size 2 and 384 threads,
plus BN/stages/raster 192/4/16 or 256/3/16 and the wave-cost constants below,
are the upstream H100-tuned configuration; no problem dimensions are
compile-time constants.  The gate deliberately accepts bf16 only.

TUNE_NT_ROLLING=True replaces the global K-tile division/modulo with an
explicit ring stage and parity.  Both counters span output-work boundaries;
the default False preserves the original pipeline arithmetic.
TUNE_NT_RASTER selects the host dispatcher's raster height; its default 16
is the upstream H100 choice, and 8 is an independently measured candidate.
"""

from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.mma import st_matrix
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.primitives import block_rank_in_cluster, cluster_sync
from max.gpu.memory import fence_async_view_proxy
from std.memory import AddressSpace
from max.gpu.sync import named_barrier
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.sys import get_defined_bool, get_defined_int
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile
from op_utils import _enqueue_cached

from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG
from gemm16_nn_v4_kernels import (
    _v4_bias_epilogue_quad,
    _v4_dyn_smem_tile,
)

from gemm16_nt_v4_kernels import (
    _V4_DT,
    _V4_F32,
    _V4_PTR,
    _V4_SWIZZLE,
    _V4_BM,
    _V4_BK,
    _V4_THREADS,
    _V4_CONSUMERS,
    _V4_WG_ROWS,
    _V4_C_BOX_N,
    _V4_C_BOX_ELEMS,
    _V4_CLUSTER,
    _V4_CLUSTER_SHAPE,
    _V4_A_LAYOUT,
    _V4_A_TMA,
    _v4_b_layout,
    _v4_b_half_layout,
    _v4c_nt_smem_bytes,
)

comptime _NT_ROLLING = get_defined_bool["TUNE_NT_ROLLING", False]()
# Upstream H100 default; a different height changes scheduling, not shapes.
comptime _NT_RASTER = get_defined_int["TUNE_NT_RASTER", 16]()


@always_inline
def _v4_store_accum_bias_stmatrix[
    bn: Int
](
    wg_half: Pointer[
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
        var data = _v4_bias_epilogue_quad[bn](accum, t, lane, bias, n0, n)
        st_matrix[simd_width=4](wg_half.unsafe_offset(off), data)


@always_inline
def _v4c_nt_defer_tag[defer_release: Bool]() -> StaticString:
    """Name suffix for the deferred-release pipeline variant."""
    comptime if defer_release:
        return "_dr"
    else:
        return ""


@always_inline
def _v4c_nt_rolling_tag[rolling: Bool]() -> StaticString:
    comptime if rolling:
        return "_rolling"
    else:
        return ""


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS)),
    `nvvm.cluster_dim`=_V4_CLUSTER_SHAPE,
)
# Without an explicit name this kernel reaches CUPTI, Nsight and
# torch.profiler as its Mojo mangling -- module path, encoded parameters and
# a 16-hex hash that moves on any refactor -- which is both unreadable and,
# once this family serves float16 too, wrong about the dtype.
@__name(
    t"{_GEMM16_TAG}_gemm_nt_bias_v4_persistent_m{_V4_BM}n{bn}_s{stages}g{raster_h}{_v4c_nt_defer_tag[defer_release]()}{_v4c_nt_rolling_tag[rolling]()}"
)
def _v4c_nt_bias_persistent[
    bn: Int,
    stages: Int,
    raster_h: Int,
    defer_release: Bool = False,
    # The TUNE_NT_ROLLING build identity of the pipeline counters, threaded in
    # rather than read here: the launch cache key is this kernel's linkage
    # name, and a define read in its body would be invisible there.
    rolling: Bool = _NT_ROLLING,
](
    a_tma: _V4_A_TMA,
    b_tma: TMATensorTile[
        _V4_DT,
        2,
        Index(bn // _V4_CLUSTER, _V4_BK),
        Index(bn // _V4_CLUSTER, _V4_BK),
    ],
    c_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_WG_ROWS, bn), Index(_V4_WG_ROWS, _V4_C_BOX_N)
    ],
    bias: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
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
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    comptime B_HALF = bn // _V4_CLUSTER
    comptime B_LAYOUT = _v4_b_layout[bn]()
    comptime B_HALF_LAYOUT = _v4_b_half_layout[bn]()
    comptime CFRAG = 64 * bn // 128
    comptime TMA_BYTES = (_V4_BM + bn) * _V4_BK * 2
    comptime if _is_sm_9x():
        # Three carvings of one extern slab -- see `_v4_dyn_smem_tile`
        # (gemm16_nn_v4_kernels.mojo).
        var a_pipeline = _v4_dyn_smem_tile[
            Layout.row_major(stages, _V4_BM * _V4_BK), 128, 0
        ]()
        var b_pipeline = _v4_dyn_smem_tile[
            Layout.row_major(stages, bn * _V4_BK),
            128,
            stages * _V4_BM * _V4_BK,
        ]()
        # One 64 x bn bf16 staging slice per consumer warp group, arranged
        # as consecutive 64x64 boxes in the canonical 128B-swizzled TMA
        # layout expected by the C descriptor.
        comptime C_STAGING_ELEMS = _V4_CONSUMERS * _V4_WG_ROWS * bn
        comptime C_STAGING_OFFSET = stages * (_V4_BM + bn) * _V4_BK
        var c_staging = _v4_dyn_smem_tile[
            Layout.row_major(_V4_CONSUMERS, _V4_WG_ROWS * bn),
            128,
            C_STAGING_OFFSET,
        ]()
        # The staging tile is the last carving, so its end IS the slab size
        # the launch must ask for; keeping the two in step is not left to a
        # comment.
        comptime assert (
            _v4c_nt_smem_bytes[bn, stages]()
            == (C_STAGING_OFFSET + C_STAGING_ELEMS) * 2
        ), "NT persistent smem carve and launch size disagree"
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
                full_barriers[unsafe_offset=stage].init()
                # Each of the two consumer warp groups in BOTH cluster CTAs
                # signals every slot release (arrive_cluster below).
                empty_barriers[unsafe_offset=stage].init(
                    Int32(_V4_CONSUMERS * _V4_CLUSTER)
                )
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
                empty_barriers[unsafe_offset=stage].arrive_cluster(
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
                var ring_stage = 0
                var ring_phase = UInt32(0)
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
                        var stage = ring_stage
                        var phase = ring_phase
                        comptime if not rolling:
                            stage = gkt % stages
                            phase = UInt32((gkt // stages) % 2)
                        empty_barriers[unsafe_offset=stage].wait(phase)
                        full_barriers[unsafe_offset=stage].expect_bytes(
                            Int32(TMA_BYTES)
                        )
                        var a_tile = LayoutTensor[
                            _V4_DT,
                            _V4_A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr.unsafe_offset(stage * _V4_BM * _V4_BK))
                        var k0 = kt * _V4_BK
                        a_tma.async_copy(
                            a_tile, full_barriers[unsafe_offset=stage], (k0, m0)
                        )
                        if can_multicast:
                            # Load our half of B, multicast to both CTAs.
                            var b_half = LayoutTensor[
                                _V4_DT,
                                B_HALF_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr.unsafe_offset(
                                    stage * bn * _V4_BK + rank * B_HALF * _V4_BK
                                )
                            )
                            b_tma.async_multicast_load(
                                b_half,
                                full_barriers[unsafe_offset=stage],
                                (k0, n0 + rank * B_HALF),
                                UInt16(0b11),
                            )
                        else:
                            # Divergent pair: load the full B tile privately.
                            comptime for half in range(_V4_CLUSTER):
                                var b_half = LayoutTensor[
                                    _V4_DT,
                                    B_HALF_LAYOUT,
                                    MutAnyOrigin,
                                    address_space=AddressSpace.SHARED,
                                    alignment=128,
                                ](
                                    b_pipeline.ptr.unsafe_offset(
                                        stage * bn * _V4_BK
                                        + half * B_HALF * _V4_BK
                                    )
                                )
                                b_tma.async_copy(
                                    b_half,
                                    full_barriers[unsafe_offset=stage],
                                    (k0, n0 + half * B_HALF),
                                )
                        kt += 1
                        comptime if rolling:
                            ring_stage += 1
                            if ring_stage == stages:
                                ring_stage = 0
                                ring_phase = ring_phase ^ UInt32(1)
                        else:
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
                _V4_DT,
                _V4_DT,
                Index(64, bn, 16),
                a_swizzle=_V4_SWIZZLE,
                b_swizzle=_V4_SWIZZLE,
                transpose_b=True,
            ]()

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var wg_half = c_staging.ptr.unsafe_offset(
                (warp_group_idx - 1) * _V4_WG_ROWS * bn
            )

            var gkt = 0
            var ring_stage = 0
            var ring_phase = UInt32(0)
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
                    var stage = ring_stage
                    var phase = ring_phase
                    comptime if not rolling:
                        stage = gkt % stages
                        phase = UInt32((gkt // stages) % 2)
                    full_barriers[unsafe_offset=stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _V4_DT,
                        _V4_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr.unsafe_offset(stage * _V4_BM * _V4_BK))
                    var b_tile = LayoutTensor[
                        _V4_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr.unsafe_offset(stage * bn * _V4_BK))
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
                            empty_barriers[
                                unsafe_offset=prev_stage
                            ].arrive_cluster(UInt32(warp_group_thread_idx))
                        prev_stage = stage
                    else:
                        wgmma.wait_group()
                        if warp_group_thread_idx < _V4_CLUSTER:
                            empty_barriers[unsafe_offset=stage].arrive_cluster(
                                UInt32(warp_group_thread_idx)
                            )
                    kt += 1
                    comptime if rolling:
                        ring_stage += 1
                        if ring_stage == stages:
                            ring_stage = 0
                            ring_phase = ring_phase ^ UInt32(1)
                    else:
                        gkt += 1
                comptime if defer_release:
                    wgmma.wait_group[0]()
                    if prev_stage >= 0 and warp_group_thread_idx < _V4_CLUSTER:
                        empty_barriers[unsafe_offset=prev_stage].arrive_cluster(
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

                _v4_store_accum_bias_stmatrix[bn](
                    wg_half, accum, warp, lane, bias, n0, n
                )

                # Make generic-proxy smem writes visible to the async proxy,
                # then let one thread issue the (clipped) TMA store.  It is
                # only waited on at the next epilogue, so the store drains
                # while the next output tile's mainloop runs.
                fence_async_view_proxy()
                named_barrier[Int32(128)](Int32(warp_group_idx))
                if warp_group_thread_idx == 0:
                    var c_store = LayoutTensor[
                        _V4_DT,
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


def _v4c_enqueue_nt_bias_persistent[
    bn: Int, stages: Int, raster_h: Int, defer_release: Bool = False
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V4_BM, _V4_BK),
    )
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](n, k),
        IndexList[2](k, 1),
        IndexList[2](bn // _V4_CLUSTER, _V4_BK),
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
        IndexList[2](_V4_WG_ROWS, _V4_C_BOX_N),
    )
    var a_tma = _V4_A_TMA(a_desc)
    var b_tma = TMATensorTile[
        _V4_DT,
        2,
        Index(bn // _V4_CLUSTER, _V4_BK),
        Index(bn // _V4_CLUSTER, _V4_BK),
    ](b_desc)
    var c_tma = TMATensorTile[
        _V4_DT, 2, Index(_V4_WG_ROWS, bn), Index(_V4_WG_ROWS, _V4_C_BOX_N)
    ](c_desc)
    comptime DYN_SMEM = _v4c_nt_smem_bytes[bn, stages]()
    # Compiled once per process and context (see gemm16_rolling_kernels.mojo):
    # the key is the kernel's linkage name, so dtype, tile width, stage count,
    # raster height, the release discipline and `rolling` -- everything that
    # selects the code -- are comptime parameters of it, and nothing about
    # this call's pointers or m/n/k. The two-CTA cluster rides on the kernel's
    # own metadata.
    _enqueue_cached[
        _v4c_nt_bias_persistent[bn, stages, raster_h, defer_release],
        dyn_smem=DYN_SMEM,
    ](
        ctx,
        grid_x,
        1,
        1,
        _V4_THREADS,
        a_tma,
        b_tma,
        c_tma,
        bias,
        Int64(m),
        Int64(n),
        Int64(k),
    )


@always_inline
def _v4c_nt_bias_hw_gate(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Runtime hw/alignment/overflow gate shared by every bf16 sm_90a NT+bias
    persistent kernel candidate (this 128-row route and the 192-row rolling
    one in gemm16_rolling_kernels.mojo): exact compute capability 9.0,
    16B-aligned dense operands, a non-null bf16-aligned bias, and TMA/
    product bounds that fit Int32 extents without overflowing Int64 m*n*k
    arithmetic. Each caller still checks its own launch-resource limits
    (grid width, cluster occupancy) separately -- those differ by tile."""
    if ctx.api() != "cuda":
        return False
    if (
        ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) != 9
        or ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR) != 0
    ):
        return False
    return (
        m >= 1
        and n >= _V4_C_BOX_N
        and k >= _V4_BK
        and n % 8 == 0
        and k % _V4_BK == 0
        and Int(output) % 16 == 0
        and Int(a) % 16 == 0
        and Int(b) % 16 == 0
        and Int(bias) != 0
        and Int(bias) % 2 == 0
        and m <= 2_147_483_647
        and n <= 2_147_483_647
        and k <= 2_147_483_647
        and k <= 9_223_372_036_854_775_807 // m
        and k <= 9_223_372_036_854_775_807 // n
        and n <= 9_223_372_036_854_775_807 // m
    )


def maybe_enqueue_gemm16_nt_bias_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Route C=A@B.T+bias to the persistent bf16 NT kernel if allowed.

    Returns True when the work was enqueued.  Callers must fall back to the
    v3 dispatcher when False is returned.  Alignment requirements: 16B base
    pointers (TMA), n % 8 for 16B-aligned gmem rows, k % BK for the
    pipeline, and TMA descriptor dimension limits.  m and n need NOT be
    tile-aligned: TMA clips partial edge tiles on load and store.
    """
    comptime assert _NT_RASTER > 0
    comptime if _GEMM16_DT != DType.bfloat16:
        return False
    comptime if _has_sm_9x():
        if _v4c_nt_bias_hw_gate(output, a, b, bias, m, n, k, ctx):
            var blocks_m = (m + _V4_BM - 1) // _V4_BM
            var sm_count = ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            var max_grid_x = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
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
                var cost_256 = ((pairs_256 + grid_pairs - 1) // grid_pairs) * (
                    256 + 32
                )
                var cost_192 = ((pairs_192 + grid_pairs - 1) // grid_pairs) * (
                    192 + 32
                )
                if cost_192 * 102 < cost_256 * 100:
                    var grid_x = min(pairs_192, grid_pairs) * _V4_CLUSTER
                    _v4c_enqueue_nt_bias_persistent[192, 4, _NT_RASTER](
                        output, a, b, bias, m, n, k, grid_x, ctx
                    )
                    return True
                var grid_x = min(pairs_256, grid_pairs) * _V4_CLUSTER
                _v4c_enqueue_nt_bias_persistent[256, 3, _NT_RASTER](
                    output, a, b, bias, m, n, k, grid_x, ctx
                )
                return True
    return False
