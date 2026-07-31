"""V4 H100 BF16 TN (wgrad) GEMM kernels: split-K and narrow-tile variants.

The nanogpt wgrad family C[m,n] = A[k,m]^T @ B[k,n] has a huge reduction
dimension (K = tokens = 32768) and small outputs (m,n in the hundreds to a
few thousand), so the v3 one-CTA-per-output-tile kernels leave most of the
GPU idle: (768,768) yields only 18 CTAs of 128x256 for 114 SMs.

Two remedies, both dispatched by regime (no model dims hardcoded):

1. Split-K: when output tiles fill less than half the SMs, partition K
   across `splits` CTAs per tile (grid y).  Each CTA accumulates its K-chunk
   into an fp32 workspace slice; a small elementwise kernel reduces the
   slices and casts to bf16.  A workspace + separate reduce is deterministic
   and much faster than atomics on these deep-K shapes.
2. Narrow 128x192 tiles whenever the wave-quantized cost (waves x per-CTA
   work) beats 256-wide tiles.  That covers the one-wave underfilled case
   (72 CTAs of 128x256 on 114 SMs -> 96 fuller CTAs) and the multi-wave
   case with an idle tail (1179 CTAs = 10.3 waves -> 1572 = 13.8 waves but
   25% less work per CTA).  Single-wave grids get a 4-stage pipeline;
   multi-wave sustained grids get 3 stages (measurably less power draw on
   this power-limited card) and 16-row rasterization groups (halves DRAM
   traffic for B, which all row-tiles share).

Both kernels reuse the v3 warp-specialized TMA + WGMMA structure: A is
physical row-major (K, M) and loaded directly into an MN-major shared tile,
which is the column-major A representation accepted by SM90 WGMMA.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    create_tma_descriptor,
)
from std.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    _convert_cfrags_to_simd,
    _convert_cfrags_to_tuple,
    _wgmma_descriptor,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile


comptime _V4_BF16 = DType.bfloat16
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_BF16], MutAnyOrigin]
comptime _V4_F32_PTR = UnsafePointer[Scalar[_V4_F32], MutAnyOrigin]
comptime _V4_BM = 128
comptime _V4_BK = 64
comptime _V4_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime _V4_THREADS = 384
comptime _V4_CONSUMERS = 2
# Split-K sizing: never split below this many BK-tiles per chunk (pipeline
# ramp-up dominates below that), and cap the workspace size.
comptime _V4_MIN_CHUNK_TILES = 16
comptime _V4_MAX_SPLITS = 8
comptime _V4_MAX_WS_BYTES = 256 * 1024 * 1024


# ============================================================================
# Shared warp-specialized TMA + WGMMA body.
#
# SPLITK=True : accumulates BK-tiles [tile_start, tile_start + chunk) and
#               stores the fp32 partial tile into ws at slice block_idx.y.
# SPLITK=False: accumulates the whole K range and stores bf16 into out.
# ============================================================================
@always_inline
def _v4_tn_ws_body[
    BN: Int, STAGES: Int, SPLITK: Bool, GROUP_ROWS: Int
](
    a_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, BN), Index(_V4_BK, 64)],
    output: _V4_PTR,
    ws: _V4_F32_PTR,
    m: Int,
    n: Int,
    k: Int,
    chunk_tiles: Int,
):
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_mn_major[
            _V4_BF16, _V4_BM, _V4_BK, _V4_SWIZZLE
        ]()
        comptime B_LAYOUT = tile_layout_mn_major[
            _V4_BF16, BN, _V4_BK, _V4_SWIZZLE
        ]()
        comptime A_PIPE_LAYOUT = Layout.row_major(STAGES, _V4_BM * _V4_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(STAGES, BN * _V4_BK)

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
        var full_barriers = stack_allocation[
            STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V4_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        # Order barrier initialization before cross-warp-group arrivals.
        barrier()

        comptime CFRAG = 64 * BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V4_BM - 1) // _V4_BM
        var blocks_n = (n + BN - 1) // BN
        var lin = Int(block_idx.x)
        var group_span = GROUP_ROWS * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(GROUP_ROWS, blocks_m - group * GROUP_ROWS)
        var m0 = (group * GROUP_ROWS + rem % rows_in_group) * _V4_BM
        var n0 = (rem // rows_in_group) * BN

        # K range handled by this CTA (whole K unless split-K).
        var tile_start = 0
        var my_tiles = k // _V4_BK
        comptime if SPLITK:
            tile_start = Int(block_idx.y) * chunk_tiles
            my_tiles = min(chunk_tiles, k // _V4_BK - tile_start)
            if my_tiles < 0:
                my_tiles = 0
        comptime TMA_BYTES = (_V4_BM + BN) * _V4_BK * 2

        # Initially release every pipeline slot to the producer.  Thereafter
        # both consumer warp groups arrive only after their WGMMA reads
        # finish.
        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var it = 0
                while it < my_tiles:
                    var stage = it % STAGES
                    var phase = UInt32((it // STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))

                    var a_tile = LayoutTensor[
                        _V4_BF16,
                        A_LAYOUT,
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
                    ](b_pipeline.ptr + stage * BN * _V4_BK)
                    var k0 = (tile_start + it) * _V4_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (m0, k0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (n0, k0))
                    it += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V4_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)

            # MN-major descriptors are required for WGMMA's column-major A
            # and row-major B modes.  The second consumer advances by one
            # 64-row WGMMA tile within the shared A tile.
            comptime a_canonical_layout = tile_to_descriptor[
                _V4_BF16, A_LAYOUT, False
            ]()
            comptime b_canonical_layout = tile_to_descriptor[
                _V4_BF16, B_LAYOUT, False
            ]()
            comptime a_shape00 = a_canonical_layout[0].shape[0].value()
            comptime a_stride01 = a_canonical_layout[0].stride[1].value()
            comptime a_stride11 = a_canonical_layout[1].stride[1].value()
            comptime b_stride11 = b_canonical_layout[1].stride[1].value()
            comptime a_m_stride = a_stride01 * (64 // a_shape00) * 2
            comptime a_k_stride = a_stride11 * 2 * 2
            comptime b_k_stride = b_stride11 * 2 * 2
            comptime NUM_K_MMAS = _V4_BK // 16

            var it = 0
            while it < my_tiles:
                var stage = it % STAGES
                var phase = UInt32((it // STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V4_BF16,
                    A_LAYOUT,
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
                ](b_pipeline.ptr + stage * BN * _V4_BK)
                var a_desc = _wgmma_descriptor[
                    a_canonical_layout, False, _V4_SWIZZLE
                ](a_tile.ptr)
                var b_desc = _wgmma_descriptor[
                    b_canonical_layout, False, _V4_SWIZZLE
                ](b_tile.ptr)
                a_desc += a_m_stride * (warp_group_idx - 1)

                warpgroup_fence(accum)
                wgmma_fence_aligned()
                comptime for k_mma in range(NUM_K_MMAS):
                    var c_tuple = _convert_cfrags_to_tuple[_V4_F32, CFRAG](
                        accum
                    )
                    var c_out = wgmma_async[
                        64,
                        BN,
                        16,
                        a_type=_V4_BF16,
                        b_type=_V4_BF16,
                        layout_a="col",
                        layout_b="row",
                    ](
                        a_desc + k_mma * a_k_stride,
                        b_desc + k_mma * b_k_stride,
                        c_tuple,
                    )
                    _convert_cfrags_to_simd[_V4_F32, CFRAG](c_out, accum)
                wgmma_commit_group_sync()
                warpgroup_fence(accum)
                wgmma_wait_group_sync()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                it += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime if SPLITK:
                var ws_base = ws + Int(block_idx.y) * (m * n)
                comptime for q in range(CFRAG // 2):
                    var e = q * 2
                    var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                    var col = base_col + (q // 2) * 8
                    var pair = SIMD[_V4_F32, 2](accum.ptr[e], accum.ptr[e + 1])
                    if m0 + row < m and n0 + col + 1 < n:
                        ws_base.store[alignment=8](
                            (m0 + row) * n + n0 + col, pair
                        )
            else:
                comptime for q in range(CFRAG // 2):
                    var e = q * 2
                    var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                    var col = base_col + (q // 2) * 8
                    var pair = SIMD[_V4_BF16, 2](
                        accum.ptr[e].cast[_V4_BF16](),
                        accum.ptr[e + 1].cast[_V4_BF16](),
                    )
                    if m0 + row < m and n0 + col + 1 < n:
                        output.store[alignment=4](
                            (m0 + row) * n + n0 + col, pair
                        )


# ============================================================================
# Concrete kernel entry points (thin named wrappers over the shared body).
# ============================================================================
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name("nanogpt_bf16_gemm_tn_v4_splitk_m128n256_s4")
def _v4_tn_splitk_m128n256_s4(
    a_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, 256), Index(_V4_BK, 64)],
    ws: _V4_F32_PTR,
    m: Int,
    n: Int,
    k: Int,
    chunk_tiles: Int,
):
    _v4_tn_ws_body[256, 4, True, 8](
        a_tma, b_tma, ws.bitcast[Scalar[_V4_BF16]](), ws, m, n, k, chunk_tiles
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name("nanogpt_bf16_gemm_tn_v4_direct_m128n192_s4")
def _v4_tn_direct_m128n192_s4(
    a_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    output: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    _v4_tn_ws_body[192, 4, False, 8](
        a_tma, b_tma, output, output.bitcast[Scalar[_V4_F32]](), m, n, k, 0
    )


# Multi-wave variant: 3 stages draw measurably less power than 4 on this
# power-limited card (sustained multi-wave runs throttle), and 16-row
# rasterization groups halve DRAM traffic for B, which all row-tiles share.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name("nanogpt_bf16_gemm_tn_v4_direct_m128n192_s3g16")
def _v4_tn_direct_m128n192_s3g16(
    a_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_BF16, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    output: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    _v4_tn_ws_body[192, 3, False, 16](
        a_tma, b_tma, output, output.bitcast[Scalar[_V4_F32]](), m, n, k, 0
    )


# Elementwise reduction of the split-K fp32 workspace slices into the bf16
# output: out[i] = bf16(sum_s ws[s * count + i]).  Each thread owns
# _V4_RED_GROUPS independent vec4 chains so enough loads are in flight to
# saturate DRAM (a single chain per thread measured only ~53% of peak).
comptime _V4_RED_THREADS = 256
comptime _V4_RED_GROUPS = 4
comptime _V4_RED_SPAN = _V4_RED_THREADS * _V4_RED_GROUPS * 4


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_RED_THREADS))
)
@__name("nanogpt_bf16_gemm_tn_v4_splitk_reduce")
def _v4_tn_splitk_reduce(
    output: _V4_PTR,
    ws: _V4_F32_PTR,
    count: Int,
    splits: Int,
):
    var base = Int(block_idx.x) * _V4_RED_SPAN + Int(thread_idx.x) * 4
    if base + (_V4_RED_GROUPS - 1) * _V4_RED_THREADS * 4 + 4 <= count:
        # Fast path: all four chains fully in range.
        var acc = StaticTuple[SIMD[_V4_F32, 4], _V4_RED_GROUPS]()
        comptime for g in range(_V4_RED_GROUPS):
            acc[g] = ws.load[width=4, alignment=16](
                base + g * _V4_RED_THREADS * 4
            )
        for s in range(1, splits):
            var slice_base = s * count + base
            comptime for g in range(_V4_RED_GROUPS):
                acc[g] += ws.load[width=4, alignment=16](
                    slice_base + g * _V4_RED_THREADS * 4
                )
        comptime for g in range(_V4_RED_GROUPS):
            output.store[alignment=8](
                base + g * _V4_RED_THREADS * 4, acc[g].cast[_V4_BF16]()
            )
    else:
        comptime for g in range(_V4_RED_GROUPS):
            var i = base + g * _V4_RED_THREADS * 4
            if i + 4 <= count:
                var acc4 = ws.load[width=4, alignment=16](i)
                for s in range(1, splits):
                    acc4 += ws.load[width=4, alignment=16](s * count + i)
                output.store[alignment=8](i, acc4.cast[_V4_BF16]())
            else:
                while i < count:
                    var acc1 = ws[i]
                    for s in range(1, splits):
                        acc1 += ws[s * count + i]
                    output[i] = acc1.cast[_V4_BF16]()
                    i += 1


# ============================================================================
# Enqueue helpers
# ============================================================================
def _v4_make_a_tma(
    a: _V4_PTR, m: Int, k: Int, ctx: DeviceContext
) raises -> TMATensorTile[
    _V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)
]:
    var a_desc = create_tma_descriptor[_V4_BF16, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, m),
        IndexList[2](m, 1),
        IndexList[2](_V4_BK, 64),
    )
    return TMATensorTile[_V4_BF16, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)](
        a_desc
    )


def _v4_enqueue_splitk_m128n256(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    splits: Int,
    ctx: DeviceContext,
) raises:
    var a_tma = _v4_make_a_tma(a, m, k, ctx)
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
    var b_tma = TMATensorTile[
        _V4_BF16, 2, Index(_V4_BK, 256), Index(_V4_BK, 64)
    ](b_desc)
    var total_tiles = k // _V4_BK
    var chunk_tiles = (total_tiles + splits - 1) // splits
    var count = m * n
    var ws = ctx.enqueue_create_buffer[DType.float32](splits * count)
    var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
    ctx.enqueue_function[_v4_tn_splitk_m128n256_s4](
        a_tma,
        b_tma,
        ws_ptr,
        m,
        n,
        k,
        chunk_tiles,
        grid_dim=(grid_x, splits),
        block_dim=(_V4_THREADS,),
    )
    ctx.enqueue_function[_v4_tn_splitk_reduce](
        output,
        ws_ptr,
        count,
        splits,
        grid_dim=((count + _V4_RED_SPAN - 1) // _V4_RED_SPAN,),
        block_dim=(_V4_RED_THREADS,),
    )
    # Normal release after both stream-ordered consumers are enqueued.
    _ = ws^


def _v4_enqueue_direct_m128n192(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    multi_wave: Bool,
    ctx: DeviceContext,
) raises:
    var a_tma = _v4_make_a_tma(a, m, k, ctx)
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
    var b_tma = TMATensorTile[
        _V4_BF16, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)
    ](b_desc)
    if multi_wave:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s3g16](
            a_tma,
            b_tma,
            output,
            m,
            n,
            k,
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s4](
            a_tma,
            b_tma,
            output,
            m,
            n,
            k,
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )


# ============================================================================
# Regime dispatch.  Returns True when a v4 kernel handled the call.
# Caller guarantees: TN (transpose_a and not transpose_b), no bias.
# ============================================================================
def try_enqueue_bf16_gemm_tn_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    comptime if not _has_sm_9x():
        return False
    if ctx.api() != "cuda":
        return False
    var cc_major = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
    var cc_minor = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
    if cc_major != 9 or cc_minor != 0:
        return False
    # Aligned full-tile regime with machine-width-safe products (mirrors the
    # v3 gates).
    if (
        m < _V4_BM
        or k < _V4_BK
        or m % _V4_BM != 0
        or k % _V4_BK != 0
        or n <= 0
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        or m > 2_147_483_647
        or n > 2_147_483_647
        or k > 2_147_483_647
        or k > 9_223_372_036_854_775_807 // m
        or k > 9_223_372_036_854_775_807 // n
        or n > 9_223_372_036_854_775_807 // m
    ):
        return False
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var max_grid_x = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
    if sm_count <= 0 or max_grid_x <= 0:
        return False

    # Split-K on 256-wide tiles: only when at least two K-chunks per tile
    # fit within the SM count, each chunk deep enough to amortize pipeline
    # ramp-up, and the fp32 workspace stays modest.
    if n % 256 == 0:
        var tiles = (m // _V4_BM) * (n // 256)
        if tiles > 0 and tiles <= max_grid_x and 2 * tiles <= sm_count:
            var splits = sm_count // tiles
            if splits > _V4_MAX_SPLITS:
                splits = _V4_MAX_SPLITS
            var max_by_depth = (k // _V4_BK) // _V4_MIN_CHUNK_TILES
            if splits > max_by_depth:
                splits = max_by_depth
            if m * n <= _V4_MAX_WS_BYTES // 4 // max(splits, 1):
                if splits >= 2:
                    _v4_enqueue_splitk_m128n256(
                        output, a, b, m, n, k, tiles, splits, ctx
                    )
                    return True

    # Narrow-tile regime: 192-wide tiles trade 33% more CTAs for fuller
    # waves.  Per-CTA time is proportional to BN at fixed BM/BK, so compare
    # wave-quantized cost (waves x BN) and pick 192 when it wins; e.g. 72
    # CTAs on 114 SMs (one third-idle wave) and 1179 CTAs (10.3 waves with
    # an idle tail wave) both improve.
    if n % 192 == 0:
        var tiles192 = (m // _V4_BM) * (n // 192)
        if tiles192 > 0 and tiles192 <= max_grid_x:
            if n % 256 == 0:
                var tiles256 = (m // _V4_BM) * (n // 256)
                var waves256 = (tiles256 + sm_count - 1) // sm_count
                var waves192 = (tiles192 + sm_count - 1) // sm_count
                if waves192 * 192 < waves256 * 256:
                    _v4_enqueue_direct_m128n192(
                        output, a, b, m, n, k, tiles192, waves192 > 1, ctx
                    )
                    return True
            elif tiles192 <= sm_count:
                # No 256-wide alternative; take the single-wave win only.
                _v4_enqueue_direct_m128n192(
                    output, a, b, m, n, k, tiles192, False, ctx
                )
                return True

    return False
