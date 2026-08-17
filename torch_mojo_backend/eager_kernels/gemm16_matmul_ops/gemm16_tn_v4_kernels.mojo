"""V4 H100 16-bit TN (wgrad) GEMM kernels: split-K and narrow-tile variants.

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

The operand dtype is bfloat16 or float16, chosen at compile time by
`_GEMM16_DT` (gemm16_dtype.mojo); every tile size and pipeline constant
here is a function of the 2-byte operand width, not of the exponent
layout, so one source serves both.
"""

from max.gpu.sync import barrier
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from max.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.memory import AddressSpace
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.tma_async import SharedMemBarrier, TMATensorTile

from gemm16_nn_v4_kernels import (
    _v4_mma_tile,
    maybe_enqueue_gemm16_tn_v4_persistent,
)
from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG


comptime _V4_DT = _GEMM16_DT
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_DT], MutAnyOrigin]
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


# Operand-layout parametrization of the shared body.  COL_A means A is
# physically (K, M) -- the wgrad/TN operand read column-major -- and is
# loaded into an MN-major shared tile for WGMMA's "col" A mode; otherwise A
# is physically (M, K) and loaded K-major for the "row" mode.  KMAJ_B means
# B is physically (N, K) (an NT weight reached through .t()) and loaded
# K-major for WGMMA's "col" B mode; otherwise B is physically (K, N) and
# loaded MN-major for the "row" mode.  TN = (True, False), NT = (False,
# True), NN = (False, False), TT = (True, True).  The TT instantiations
# compute C directly into a contiguous row-major (m, n) buffer, so a TT mm
# returns the same strides CUDA torch does (an operand-swapped NN kernel
# would be equally fast but hand back a column-major C).
def _v4_a_smem_layout[COL_A: Bool, BM: Int = _V4_BM]() -> Layout:
    comptime if COL_A:
        return tile_layout_mn_major[_V4_DT, BM, _V4_BK, _V4_SWIZZLE]()
    return tile_layout_k_major[_V4_DT, BM, _V4_BK, _V4_SWIZZLE]()


def _v4_b_smem_layout[BN: Int, KMAJ_B: Bool]() -> Layout:
    comptime if KMAJ_B:
        return tile_layout_k_major[_V4_DT, BN, _V4_BK, _V4_SWIZZLE]()
    return tile_layout_mn_major[_V4_DT, BN, _V4_BK, _V4_SWIZZLE]()


# ============================================================================
# Shared warp-specialized TMA + WGMMA body.
#
# SPLITK=True : accumulates BK-tiles [tile_start, tile_start + chunk) and
#               stores the fp32 partial tile into ws at slice block_idx.y.
# SPLITK=False: accumulates the whole K range and stores bf16 into out.
#
# The TMA tile/descriptor shapes are infer-only: each concrete entry point
# passes descriptors matching its operand majorness (built by the enqueue
# helpers below), and COL_A / KMAJ_B select the matching shared-memory
# layouts, TMA coordinate order and WGMMA modes.
# ============================================================================
@always_inline
def _v4_tn_ws_body[
    a_tile: IndexList[2],
    a_desc: IndexList[2],
    b_tile: IndexList[2],
    b_desc: IndexList[2],
    //,
    BN: Int,
    STAGES: Int,
    SPLITK: Bool,
    GROUP_ROWS: Int,
    COL_A: Bool = True,
    KMAJ_B: Bool = False,
    # BM = 64 with CONSUMERS = 1 is the v3-small-kernel geometry (one
    # 64-row WGMMA warp group, 256 threads); the defaults keep every
    # pre-existing 128-row two-consumer instantiation unchanged.
    BM: Int = _V4_BM,
    CONSUMERS: Int = _V4_CONSUMERS,
](
    a_tma: TMATensorTile[_V4_DT, 2, a_tile, a_desc],
    b_tma: TMATensorTile[_V4_DT, 2, b_tile, b_desc],
    output: _V4_PTR,
    ws: _V4_F32_PTR,
    m: Int,
    n: Int,
    k: Int,
    chunk_tiles: Int,
):
    comptime if _is_sm_9x():
        comptime A_LAYOUT = _v4_a_smem_layout[COL_A, BM]()
        comptime B_LAYOUT = _v4_b_smem_layout[BN, KMAJ_B]()
        comptime A_PIPE_LAYOUT = Layout.row_major(STAGES, BM * _V4_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(STAGES, BN * _V4_BK)

        var a_pipeline = LayoutTensor[
            _V4_DT,
            A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V4_DT,
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
                empty_barriers[stage].init(Int32(CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        # Order barrier initialization before cross-warp-group arrivals.
        barrier()

        comptime CFRAG = 64 * BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + BM - 1) // BM
        var blocks_n = (n + BN - 1) // BN
        var lin = Int(block_idx.x)
        var group_span = GROUP_ROWS * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(GROUP_ROWS, blocks_m - group * GROUP_ROWS)
        var m0 = (group * GROUP_ROWS + rem % rows_in_group) * BM
        var n0 = (rem // rows_in_group) * BN

        # K range handled by this CTA (whole K unless split-K).
        var tile_start = 0
        var my_tiles = k // _V4_BK
        comptime if SPLITK:
            tile_start = Int(block_idx.y) * chunk_tiles
            my_tiles = min(chunk_tiles, k // _V4_BK - tile_start)
            if my_tiles < 0:
                my_tiles = 0
        comptime TMA_BYTES = (BM + BN) * _V4_BK * 2

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
                        _V4_DT,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * BM * _V4_BK)
                    var b_tile = LayoutTensor[
                        _V4_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * BN * _V4_BK)
                    var k0 = (tile_start + it) * _V4_BK
                    # TMA coordinates are (fastest dim, slower dim) of the
                    # global tensor each descriptor was built over.
                    comptime if COL_A:
                        a_tma.async_copy(a_tile, full_barriers[stage], (m0, k0))
                    else:
                        a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                    comptime if KMAJ_B:
                        b_tma.async_copy(b_tile, full_barriers[stage], (k0, n0))
                    else:
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

            var it = 0
            while it < my_tiles:
                var stage = it % STAGES
                var phase = UInt32((it // STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V4_DT,
                    A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * BM * _V4_BK)
                var b_tile = LayoutTensor[
                    _V4_DT,
                    B_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * BN * _V4_BK)
                # Majorness-generic raw WGMMA slab, shared with the
                # persistent body in gemm16_nn_v4_kernels.mojo.
                _v4_mma_tile[BN, COL_A, KMAJ_B, A_LAYOUT, B_LAYOUT](
                    a_tile.ptr, b_tile.ptr, accum, warp_group_idx
                )
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
                    var pair = SIMD[_V4_DT, 2](
                        accum.ptr[e].cast[_V4_DT](),
                        accum.ptr[e + 1].cast[_V4_DT](),
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
@__name(t"{_GEMM16_TAG}_gemm_tn_v4_splitk_m128n256_s4")
def _v4_tn_splitk_m128n256_s4(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 256), Index(_V4_BK, 64)],
    ws: _V4_F32_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    chunk_tiles_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var chunk_tiles = Int(chunk_tiles_arg)
    _v4_tn_ws_body[256, 4, True, 8](
        a_tma, b_tma, ws.bitcast[Scalar[_V4_DT]](), ws, m, n, k, chunk_tiles
    )


# Split-K specializations for the remaining layouts.  Same body, same
# tile/stage/raster configuration as the TN split-K kernel; only the operand
# majorness comptimes differ.  NT reads B through a K-major (N, K) buffer;
# NN reads it MN-major like TN; TT combines TN's col-major A with NT's
# K-major B.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nt_v4_splitk_m128n256_s4")
def _v4_nt_splitk_m128n256_s4(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(256, _V4_BK), Index(256, _V4_BK)],
    ws: _V4_F32_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    chunk_tiles_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var chunk_tiles = Int(chunk_tiles_arg)
    _v4_tn_ws_body[256, 4, True, 8, False, True](
        a_tma, b_tma, ws.bitcast[Scalar[_V4_DT]](), ws, m, n, k, chunk_tiles
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nn_v4_splitk_m128n256_s4")
def _v4_nn_splitk_m128n256_s4(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 256), Index(_V4_BK, 64)],
    ws: _V4_F32_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    chunk_tiles_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var chunk_tiles = Int(chunk_tiles_arg)
    _v4_tn_ws_body[256, 4, True, 8, False, False](
        a_tma, b_tma, ws.bitcast[Scalar[_V4_DT]](), ws, m, n, k, chunk_tiles
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_splitk_m128n256_s4")
def _v4_tt_splitk_m128n256_s4(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(256, _V4_BK), Index(256, _V4_BK)],
    ws: _V4_F32_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    chunk_tiles_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var chunk_tiles = Int(chunk_tiles_arg)
    _v4_tn_ws_body[256, 4, True, 8, True, True](
        a_tma, b_tma, ws.bitcast[Scalar[_V4_DT]](), ws, m, n, k, chunk_tiles
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tn_v4_direct_m128n192_s4")
def _v4_tn_direct_m128n192_s4(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    output: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
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
@__name(t"{_GEMM16_TAG}_gemm_tn_v4_direct_m128n192_s3g16")
def _v4_tn_direct_m128n192_s3g16(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    output: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    _v4_tn_ws_body[192, 3, False, 16](
        a_tma, b_tma, output, output.bitcast[Scalar[_V4_F32]](), m, n, k, 0
    )


# Small-tile direct TT kernel, one CTA per 128x64 output tile.  This is the
# same per-CTA geometry the swap-era TT route reached: the v3 NN m64n128
# small-tile kernel tiling C^T is exactly a 128x64 tiling of C.  It exists
# because the 256-wide kernels above lose 1.5-2.2x to that route on
# single-wave grids (too few CTAs) and cannot serve n % 256 != 0 at all;
# the 64-wide tile needs only n % 64 == 0 and quadruples the CTA count.
# Same warp-group structure as every other v4 entry (BM=128, 2 consumers);
# only BN shrinks.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_direct_m128n64_s4")
def _v4_tt_direct_m128n64_s4(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(64, _V4_BK), Index(64, _V4_BK)],
    output: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    _v4_tn_ws_body[64, 4, False, 8, True, True](
        a_tma, b_tma, output, output.bitcast[Scalar[_V4_F32]](), m, n, k, 0
    )


# 64-row single-consumer TT instantiation of the same shared body (256
# threads, 3 stages): byte-for-byte the v3 NN small kernel's geometry, with
# the TT operand modes.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_direct_m64n128_s3")
def _v4_tt_direct_m64n128_s3(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 64), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(128, _V4_BK), Index(128, _V4_BK)],
    output: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    _v4_tn_ws_body[128, 3, False, 8, True, True, 64, 1](
        a_tma, b_tma, output, output.bitcast[Scalar[_V4_F32]](), m, n, k, 0
    )


# Elementwise reduction of the split-K fp32 workspace slices into the bf16
# output: out[i] = bf16(sum_s ws[s * count + i]).  Each thread owns
# _V4_RED_GROUPS independent vec4 chains so enough loads are in flight to
# saturate DRAM (a single chain per thread measured only ~53% of peak).
comptime _V4_RED_THREADS = 256
comptime _V4_RED_GROUPS = 4
comptime _V4_RED_SPAN = _V4_RED_THREADS * _V4_RED_GROUPS * 4


# `bias` is an (n,)-row vector broadcast over every one of the `m` output
# rows; `output`/`ws` are addressed as the flattened (m, n) row-major
# buffer, so column `i % n` is the bias index for flat offset `i`. Callers
# that engage the bias epilogue always satisfy n % 256 == 0 (the only tile
# width the split-K kernels come in -- see the two `try_enqueue_*` call
# sites that pass has_bias=True), so n is always a multiple of 4 and every
# vec4 group's flat offset is itself a multiple of 4: `base % n` therefore
# never straddles a row boundary, and a 4-element bias read is always a
# contiguous, correctly-ordered slice of one row. has_bias=False (every
# pre-existing caller) skips the extra loads/adds entirely.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_RED_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tn_v4_splitk_reduce")
def _v4_tn_splitk_reduce(
    output: _V4_PTR,
    ws: _V4_F32_PTR,
    bias: _V4_PTR,
    count_arg: Int64,
    splits_arg: Int64,
    n_arg: Int64,
    has_bias_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var count = Int(count_arg)
    var splits = Int(splits_arg)
    var n = Int(n_arg)
    var has_bias = has_bias_arg != 0
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
        if has_bias:
            comptime for g in range(_V4_RED_GROUPS):
                var idx = base + g * _V4_RED_THREADS * 4
                acc[g] += bias.load[width=4, alignment=8](idx % n).cast[
                    _V4_F32
                ]()
        comptime for g in range(_V4_RED_GROUPS):
            output.store[alignment=8](
                base + g * _V4_RED_THREADS * 4, acc[g].cast[_V4_DT]()
            )
    else:
        comptime for g in range(_V4_RED_GROUPS):
            var i = base + g * _V4_RED_THREADS * 4
            if i + 4 <= count:
                var acc4 = ws.load[width=4, alignment=16](i)
                for s in range(1, splits):
                    acc4 += ws.load[width=4, alignment=16](s * count + i)
                if has_bias:
                    acc4 += bias.load[width=4, alignment=8](i % n).cast[
                        _V4_F32
                    ]()
                output.store[alignment=8](i, acc4.cast[_V4_DT]())
            else:
                while i < count:
                    var acc1 = ws[i]
                    for s in range(1, splits):
                        acc1 += ws[s * count + i]
                    if has_bias:
                        acc1 += bias[i % n].cast[_V4_F32]()
                    output[i] = acc1.cast[_V4_DT]()
                    i += 1


# ============================================================================
# Enqueue helpers
# ============================================================================
def _v4_make_a_tma[
    BM: Int = _V4_BM
](a: _V4_PTR, m: Int, k: Int, ctx: DeviceContext) raises -> TMATensorTile[
    _V4_DT, 2, Index(_V4_BK, BM), Index(_V4_BK, 64)
]:
    var a_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
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
    return TMATensorTile[_V4_DT, 2, Index(_V4_BK, BM), Index(_V4_BK, 64)](
        a_desc
    )


# A physically (M, K) row-major, whole (BM, BK) box (the K-major operand of
# the NT / NN split-K kernels).
def _v4_make_a_row_tma(
    a: _V4_PTR, m: Int, k: Int, ctx: DeviceContext
) raises -> TMATensorTile[
    _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
]:
    var a_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
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
    return TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ](a_desc)


# B physically (K, N) row-major, MN-major shared tile (TN and NN).
def _v4_make_b_mn_tma[
    BN: Int
](b: _V4_PTR, n: Int, k: Int, ctx: DeviceContext) raises -> TMATensorTile[
    _V4_DT, 2, Index(_V4_BK, BN), Index(_V4_BK, 64)
]:
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
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
    return TMATensorTile[_V4_DT, 2, Index(_V4_BK, BN), Index(_V4_BK, 64)](
        b_desc
    )


# B physically (N, K) row-major (an NT weight), K-major shared tile.
def _v4_make_b_kmaj_tma[
    BN: Int
](b: _V4_PTR, n: Int, k: Int, ctx: DeviceContext) raises -> TMATensorTile[
    _V4_DT, 2, Index(BN, _V4_BK), Index(BN, _V4_BK)
]:
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](n, k),
        IndexList[2](k, 1),
        IndexList[2](BN, _V4_BK),
    )
    return TMATensorTile[_V4_DT, 2, Index(BN, _V4_BK), Index(BN, _V4_BK)](
        b_desc
    )


def _v4_enqueue_splitk_m128n256[
    COL_A: Bool = True, KMAJ_B: Bool = False
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    splits: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    var total_tiles = k // _V4_BK
    var chunk_tiles = (total_tiles + splits - 1) // splits
    var count = m * n
    var ws = ctx.enqueue_create_buffer[DType.float32](splits * count)
    var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
    comptime if COL_A and not KMAJ_B:
        ctx.enqueue_function[_v4_tn_splitk_m128n256_s4](
            _v4_make_a_tma(a, m, k, ctx),
            _v4_make_b_mn_tma[256](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    elif not COL_A and KMAJ_B:
        ctx.enqueue_function[_v4_nt_splitk_m128n256_s4](
            _v4_make_a_row_tma(a, m, k, ctx),
            _v4_make_b_kmaj_tma[256](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    elif not COL_A and not KMAJ_B:
        ctx.enqueue_function[_v4_nn_splitk_m128n256_s4](
            _v4_make_a_row_tma(a, m, k, ctx),
            _v4_make_b_mn_tma[256](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_tt_splitk_m128n256_s4](
            _v4_make_a_tma(a, m, k, ctx),
            _v4_make_b_kmaj_tma[256](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    ctx.enqueue_function[_v4_tn_splitk_reduce](
        output,
        ws_ptr,
        bias,
        Int64(count),
        Int64(splits),
        Int64(n),
        Int64(1) if has_bias else Int64(0),
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
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
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
    var b_tma = TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)](
        b_desc
    )
    if multi_wave:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s3g16](
            a_tma,
            b_tma,
            output,
            Int64(m),
            Int64(n),
            Int64(k),
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s4](
            a_tma,
            b_tma,
            output,
            Int64(m),
            Int64(n),
            Int64(k),
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )


def _v4_enqueue_tt_direct_m128n64(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_v4_tt_direct_m128n64_s4](
        _v4_make_a_tma(a, m, k, ctx),
        _v4_make_b_kmaj_tma[64](b, n, k, ctx),
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V4_THREADS,),
    )


def _v4_enqueue_tt_direct_m64n128(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_v4_tt_direct_m64n128_s3](
        _v4_make_a_tma[64](a, m, k, ctx),
        _v4_make_b_kmaj_tma[128](b, n, k, ctx),
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(256,),
    )


# ============================================================================
# Regime dispatch.  Returns True when a v4 kernel handled the call.
# Caller guarantees: TN (transpose_a and not transpose_b). `has_bias` may be
# True: only the split-K rung below fuses a bias epilogue (see
# _v4_tn_splitk_reduce), so a bias-carrying call that reaches this function
# but declines split-K (e.g. too few output tiles for a >=2-way split)
# returns False without trying the persistent/narrow-tile rungs, none of
# which support bias -- the caller's own accepted-kernel fallback handles
# it instead, exactly as it did before this function existed.
# ============================================================================
def try_enqueue_gemm16_gemm_tn_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
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
        # has_bias only: the reduce epilogue reads bias 4-wide
        # (_v4_tn_splitk_reduce), so its base must be 16B-aligned too.
        or (has_bias and Int(bias) % 16 != 0)
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
                        output,
                        a,
                        b,
                        bias,
                        m,
                        n,
                        k,
                        tiles,
                        splits,
                        has_bias,
                        ctx,
                    )
                    return True

    if has_bias:
        return False

    # Multi-wave regime: the persistent clustered body (shared with NN)
    # in its col-major-A mode.  Gated inside the helper; it declines
    # single-wave and unaligned shapes, which fall through to the
    # narrow-tile / v3 routes below.
    if maybe_enqueue_gemm16_tn_v4_persistent(output, a, b, m, n, k, ctx):
        return True

    # Same rung for n % 256 != 0 (n % 64 == 0, gated in the helper): the
    # ragged-n instantiation of the same body, the one the TT dispatcher
    # already uses.  Without it every multi-wave half-tile-n wgrad shape
    # fell past the whole v4 ladder: n % 128 == 0 shapes onto the v3
    # 64x128 one-CTA-per-tile grid -- GPT-2's padded vocab gives
    # n = 50304, n % 256 == 128, and 768x50304x49152 lost ~1.9x to stock
    # there (H100 PCIe sweep vs b99e74e: 15.5 ms -> 9.0 with this rung)
    # -- and n % 128 == 64 shapes all the way to the non-TMA wide
    # fallback, a ~6x cliff (1536x4160x1024: 185 us -> 31).  The helper
    # keeps the multi-wave-only engagement, with a ceil-div census so the
    # trailing partial tile column counts; the census edge sits right at
    # the measured crossing (768x4992x1024, 120 census tiles on 114 SMs,
    # engages and wins 28.8 us vs 31.4 on v3; one step below,
    # 768x4224x1024 at 102 tiles, declines to v3's 24.7 us, which the
    # persistent body's per-tile rate would only tie).  Single-wave
    # ragged shapes therefore keep the narrow-tile / v3 routes below.
    if n % 256 != 0:
        if maybe_enqueue_gemm16_tn_v4_persistent[False, False, True](
            output, a, b, m, n, k, ctx
        ):
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


# ============================================================================
# TT regime dispatch.  Returns True when a v4 kernel handled the call.
# Caller guarantees: TT (transpose_a and transpose_b). `has_bias` may be
# True: only the split-K rung fuses a bias epilogue (see
# _v4_tn_splitk_reduce); a bias-carrying call that reaches this function but
# declines split-K returns False immediately rather than trying the
# persistent/small-tile rungs, which do not support bias.
#
# TT is served by the (COL_A, KMAJ_B) = (True, True) instantiations of the
# shared body -- TN's col-major A combined with NT's K-major B -- so C lands
# directly in a contiguous row-major (m, n) buffer with the same strides
# CUDA torch returns.  Ladder: split-K for deep-K underfilled grids, the
# persistent clustered body (ragged-n instantiation) for multi-wave
# m % 256 == 0 grids, and the 128x64 small-tile one-CTA-per-tile kernel for
# everything else -- all single-wave grids, n % 256 != 0 with m % 256 != 0,
# and m == 128.  Everything declined falls back to the v2 all-layout
# dispatcher.
# ============================================================================
def try_enqueue_gemm16_gemm_tt_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
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
    # Aligned regime with machine-width-safe products (mirrors the TN
    # dispatcher above).  n only needs to tile by 64: the small 128x64
    # kernel serves any n % 64 == 0, and the 256-wide split-K / persistent
    # rungs additionally gate n % 256 == 0 themselves below.
    if (
        m < _V4_BM
        or k < _V4_BK
        or m % _V4_BM != 0
        or k % _V4_BK != 0
        or n < 64
        or n % 64 != 0
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        # has_bias only: the reduce epilogue reads bias 4-wide
        # (_v4_tn_splitk_reduce), so its base must be 16B-aligned too.
        or (has_bias and Int(bias) % 16 != 0)
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
    # One-wave coverage of the 128x64 small-tile grid decides the ladder:
    # it is the same tile count as the swap-era v3 NN m64n128 route (a
    # 64x128 tiling of C^T is a 128x64 tiling of C), whose one-CTA-per-tile
    # geometry wins every single-wave shape but loses multi-wave regimes to
    # the persistent body.  Strict <, mirroring the NN dispatcher's
    # small_tile_covers crossing (fitted on H100 PCIe, 114 SMs).
    var small_tiles = (m // _V4_BM) * (n // 64)
    if small_tiles <= 0 or small_tiles > max_grid_x:
        return False
    var small_covers = small_tiles < sm_count

    if n % 256 == 0:
        var tiles = (m // _V4_BM) * (n // 256)
        # Split-K for deep-K underfilled grids.  The trigger reuses TN's
        # H100 PCIe fit (splits = min(sm // tiles, 8, k_tiles // 16) >= 2
        # with at least two K-chunks per SM wave); it was not re-swept for
        # TT.  On the deep-K harness shape it lands where the swap-era NN
        # split-K did (1024x1024x8192: 41.3 us vs 41.0), so the crossing
        # carried over.  In the small-tile-covered regime a shallow split
        # loses to the one-wave small kernel just as it did for NN, so it
        # keeps NN's covered-regime floor of
        # _V4_SPLITK_RM_COVERED_MIN_SPLITS (H100 PCIe fit).
        if tiles > 0 and 2 * tiles <= sm_count:
            var splits = sm_count // tiles
            if splits > _V4_MAX_SPLITS:
                splits = _V4_MAX_SPLITS
            var max_by_depth = (k // _V4_BK) // _V4_MIN_CHUNK_TILES
            if splits > max_by_depth:
                splits = max_by_depth
            var min_splits = _V4_SPLITK_RM_MIN_SPLITS
            if small_covers:
                min_splits = _V4_SPLITK_RM_COVERED_MIN_SPLITS
            if (
                splits >= min_splits
                and m * n <= _V4_MAX_WS_BYTES // 4 // splits
            ):
                _v4_enqueue_splitk_m128n256[True, True](
                    output, a, b, bias, m, n, k, tiles, splits, has_bias, ctx
                )
                return True

    if has_bias:
        return False

    # Multi-wave regimes: persistent clustered body (shared with NN and TN)
    # in its col-major-A + K-major-B + ragged-n mode (any_wave=True because
    # the wave decision was already taken via small_covers above; ragged_n
    # because our gate only guarantees n % 64 == 0, and the body's n-clip
    # path recovers the swap-era route's numbers there -- the swap put the
    # ragged dimension on the persistent body's clippable row side, and
    # this puts it on the column side of the direct instantiation).  The
    # extra m % 256 == 0 gate keeps the 256-row cluster macro-tiles exact:
    # at m % 256 == 128 the trailing half-empty macro row underfills the
    # persistent grid and the small kernel wins (896x1280x704: 9.3 us small
    # vs 12.6 persistent, H100 PCIe) -- the same regime split the swap-era
    # route produced.
    #
    # KNOWN RESIDUALS vs the swap era (H100 PCIe), the price of computing C
    # directly into a contiguous buffer instead of handing back the swap's
    # column-major view:
    #   - The TT-mode instantiation of this shared body (raw _v4_mma_tile
    #     col-A + K-major-B path) runs ~1-2% behind the NN-mode one the
    #     swap reached, worst on sustained multi-wave shapes
    #     (4096x4096x1024: 77.5 us vs 75.9 at the 25ffbbf baseline;
    #     1024x1024x1024 +0.8%, 1024x1024x8192 split-K +0.9%).
    #   - The persistent body prefers tall-M outputs, and the swap used to
    #     flip which orientation pays for that: direct TT is faster on
    #     8192x2048x2048 (146.9 us vs 151.6) and slower on 2048x8192x2048
    #     (152.6 vs 146.7) -- a zero-sum trade across the transposed pair,
    #     and both cells stay within 2% of stock PyTorch.
    if not small_covers and m % 256 == 0:
        if maybe_enqueue_gemm16_tn_v4_persistent[True, True, True](
            output, a, b, m, n, k, ctx
        ):
            return True

    # Everything else: one CTA per small tile.  Single-wave grids (where
    # the 256-wide kernels lose 1.5-2.2x to this geometry, e.g.
    # 512x768x640: 11.8 us persistent vs 6.5 at the swap-era baseline),
    # n % 256 != 0 at any wave count when the persistent rung declines
    # (1024x576x512 was a 22 us v2-fallback cliff vs 6.0 at baseline), and
    # m == 128 wide-n grids the persistent macro-tile cannot serve.
    #
    # Two tile aspects of the same geometry, equal tile counts when both
    # fit.  The 128x64 four-stage two-consumer kernel wins once the
    # mainloop has depth (H100 PCIe: 512x768x640 5.9 us vs 6.3,
    # 512x256x2048 10.6 vs 12.6); at k <= 2 BK-tiles there is no mainloop
    # to pipeline, launch overhead dominates, and the 256-thread 64x128
    # v3-small geometry keeps its edge (256x256x64: 3.6 us vs 3.75, the
    # swap-era route's own number).  Crossing fitted on H100 PCIe;
    # 384x512x192 (3 K-tiles) measured a tie on both sides of it.
    if k <= 2 * _V4_BK and n % 128 == 0:
        var tiles64 = (m // 64) * (n // 128)
        if tiles64 <= max_grid_x:
            _v4_enqueue_tt_direct_m64n128(output, a, b, m, n, k, tiles64, ctx)
            return True
    _v4_enqueue_tt_direct_m128n64(output, a, b, m, n, k, small_tiles, ctx)
    return True


# Split-K engagement for the row-major-A layouts (NT and NN).  Unlike TN, whose
# non-split alternative is a one-CTA-per-output-tile kernel, these layouts
# fall back to persistent v4 kernels that keep every SM busy regardless of
# tile count -- but a persistent CTA still serializes its tiles' full K
# depth, so when the output has far fewer macro-tiles than SMs and K is
# deep, most SMs idle for the whole GEMM.  Split-K restores parallelism by
# partitioning K.
#
# Regime edges, fitted on an H100 PCIe (114 SMs) by sweeping tiles in
# {16, 32, 56, 64, 96} x k in {1024, 2048, 4096, 8192} (device us,
# split-K vs the best pre-existing route on the same shape):
#
#   - Base predicate: splits = min(sm // tiles, 8, k_tiles // 16) >= 2.
#     Engaged cells win 0.42x-0.97x of the persistent kernel's time
#     (e.g. NT tiles=16 k=8192: 29.5 vs 70.3 us; NN tiles=32 k=8192:
#     40.9 vs 91.4 us); cells with sm // tiles < 2 cannot split and the
#     persistent kernels already sit within ~10% of stock PyTorch there.
#   - NT refinement: at sm // tiles == 2 with only the minimum chunk
#     depth (k_tiles < 64, so 16-tile chunks), the split pays workspace
#     + reduce overhead for a wave that was nearly full anyway and
#     LOSES: tiles=56 k=2048 measured 24.9 vs 23.2 us.  One step deeper
#     (k=4096, 32-tile chunks) it wins 38.4 vs 39.5, and at k=8192 it
#     wins 61.5 vs 76.0.  Hence: engage at sm // tiles >= 3, or at 2
#     with k_tiles >= _V4_SPLITK_RM_DEEP_TILES.
#   - NN refinement: when the v3 64x128 small-tile kernel covers the
#     output in a single wave (4 * tiles < sm, the same inequality the
#     v3 dispatcher uses) it beats a shallow split: tiles=16 k=2048
#     measured 13.2 us (v3-small) vs 17.7 us (2-way split).  A deep
#     split still wins: k=4096 (4 splits) 19.8 vs 22.8, k=8192 (7
#     splits) 28.7 vs 43.0.  Hence: in the small-tile-covered regime,
#     engage only with splits >= _V4_SPLITK_RM_COVERED_MIN_SPLITS.
comptime _V4_SPLITK_RM_MIN_SPLITS = 2
comptime _V4_SPLITK_RM_DEEP_TILES = 64  # k BK-tiles; H100 PCIe fit
comptime _V4_SPLITK_RM_COVERED_MIN_SPLITS = 4  # H100 PCIe fit


def try_enqueue_gemm16_gemm_splitk_rm_v4[
    KMAJ_B: Bool
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
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
    # Aligned full-tile regime with machine-width-safe products (same gates
    # as the TN dispatcher above, plus the 256-wide n requirement of the
    # only tile shape the split-K kernels come in).
    if (
        m < _V4_BM
        or k < _V4_BK
        or m % _V4_BM != 0
        or k % _V4_BK != 0
        or n <= 0
        or n % 256 != 0
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        # has_bias only: the reduce epilogue reads bias 4-wide
        # (_v4_tn_splitk_reduce), so its base must be 16B-aligned too.
        or (has_bias and Int(bias) % 16 != 0)
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
    var tiles = (m // _V4_BM) * (n // 256)
    if tiles <= 0 or tiles > max_grid_x:
        return False
    var cap = sm_count // tiles
    if cap < _V4_SPLITK_RM_MIN_SPLITS:
        return False
    var k_tiles = k // _V4_BK
    var splits = cap
    if splits > _V4_MAX_SPLITS:
        splits = _V4_MAX_SPLITS
    var max_by_depth = k_tiles // _V4_MIN_CHUNK_TILES
    if splits > max_by_depth:
        splits = max_by_depth
    if splits < _V4_SPLITK_RM_MIN_SPLITS:
        return False
    comptime if KMAJ_B:
        # NT: a marginal 2-way split of a minimum-depth K loses to the
        # persistent kernel's nearly-full wave (see the fit above).
        if cap == 2 and k_tiles < _V4_SPLITK_RM_DEEP_TILES:
            return False
    else:
        # NN: when the v3 small-tile kernel covers the output in one wave,
        # only a deep split beats it (see the fit above).
        if 4 * tiles < sm_count and splits < _V4_SPLITK_RM_COVERED_MIN_SPLITS:
            return False
    if m * n > _V4_MAX_WS_BYTES // 4 // splits:
        return False
    _v4_enqueue_splitk_m128n256[False, KMAJ_B](
        output, a, b, bias, m, n, k, tiles, splits, has_bias, ctx
    )
    return True
