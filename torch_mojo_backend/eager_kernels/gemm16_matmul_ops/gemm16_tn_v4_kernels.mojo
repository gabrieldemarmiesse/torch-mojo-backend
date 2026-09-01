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

from max.gpu.sync import barrier, named_barrier
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
)
from max.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from max.gpu.memory import fence_async_view_proxy
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
    maybe_enqueue_gemm16_nn_v4,
    maybe_enqueue_gemm16_tn_v4_persistent,
)
from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG


comptime _V4_DT = _GEMM16_DT
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_DT], MutAnyOrigin]
comptime _V4_F32_PTR = UnsafePointer[Scalar[_V4_F32], MutAnyOrigin]
comptime _V4_BM = 128
comptime _V4_BK = 64
# SWIZZLE_128B is not just the default, it is a MEASURED choice, and the
# alternative has been tried: SWIZZLE_64B halves the TMA box of an MN-major B
# from 64 elements to 32, which is the only way to unquantize BN off multiples
# of 64 (it admits BN in {96, 160, 224}, i.e. the CTA counts cuBLAS reaches
# with its 128x80 and 320x128 tiles).  It was built and measured on H100 PCIe
# at 1395 MHz and it LOSES, for a reason that is a property of the request
# path rather than of any tile: a 64B box halves the TMA request size, which
# ncu measures as 2.02x the L2 REQUESTS for the same bytes (3.12M -> 6.29M),
# 1.51x the sector traffic, and in-CTA tensor-pipe occupancy 92.5% -> 80.0%.
# On an IDENTICAL tile and grid that costs +1.4% (1024x1024x8192, 3 splits)
# to +24.4% (1024x1024x16384, 128x192, 2 splits), and the finer tiles it
# unlocks lose on top of that because this mainloop is operand-delivery bound:
# measured wave fill x pipe occupancy is CONSERVED (0.675 at 128x256/96 CTAs,
# 0.649 at 128x160/112 CTAs, 0.615 at 128x224/80 CTAs).  Because the tax is
# not a tile-local constant, no TILE_FACTOR can carry it and no cost model can
# safely dispatch a 64B tile.  The next lever is not the swizzle and not the
# tile: it is CTA-cluster multicast of the operands (which is how
# nvjet_sm90_tst_128x80_64x8_4x1 affords a 128x80 tile, and what the
# persistent kernel in gemm16_nn_v4_kernels.mojo already does).
comptime _V4_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime _V4_THREADS = 384
comptime _V4_CONSUMERS = 2
# Split-K sizing: never split below this many BK-tiles per chunk (pipeline
# ramp-up dominates below that), and cap the workspace size.
comptime _V4_MIN_CHUNK_TILES = 16
comptime _V4_MAX_SPLITS = 8
comptime _V4_MAX_WS_BYTES = 256 * 1024 * 1024
# TMA-store epilogue for the DIRECT (non-split-K) arm of this body only --
# mirrors _V4_PROD_TMA_STORE in gemm16_nn_v4_kernels.mojo, a one-line
# rollback switch.  NOTES.md (deep-K wave-fill engagement), phase 5,
# sections 22-26: the split-K epilogue writes fp32 at exactly one L2
# sector per wavefront
# already (nothing to recover, and it MEASURED a 2.27% regression on
# many-wave split-K grids because the drain holds the staging smem -- which
# here aliases the retired B pipeline -- past the point a scalar store would
# have retired the CTA). The direct epilogue writes bf16 at half a sector per
# wavefront, so TMA halves L2 write traffic there and measured -2% to -8%
# (ABBA) on every direct config tried, all five harness shapes, both
# TN/NN/TT layouts. _v4_tn_ws_body's own comptime assert enforces that this
# can only ever apply when SPLITK is False, so flipping this constant can
# never light up the regressing split-K path.
comptime _V4_DIRECT_TMA_STORE = True


# Register budget per consumer thread, by consumer warp-group count
# (CONSUMERS = BM // 64).  65536 registers/SM total, the producer warp group
# holds 128 x 24 of them, and the accumulator alone costs BN/2 registers per
# thread (CFRAG = 64 * BN // 128) -- see NOTES.md (deep-K wave-fill
# engagement) for the full feasibility table this was measured against.
# CONSUMERS 1 and 2 are pinned to 232: every pre-existing instantiation of
# this body (BM=64/CONSUMERS=1 -- `_v4_tt_direct_m64n128_s3` -- and every
# BM=128/CONSUMERS=2 kernel) already hardcoded `warpgroup_reg_alloc[232]`,
# and this table must keep producing that exact value for both so existing
# kernels stay codegen-identical (verified with
# scripts/compare_kernel_asm.py --accelerator sm_90a).  CONSUMERS >= 3
# (BM >= 192) is not instantiated by any production dispatcher today; 160 is
# carried over from the persistent kernel's own 3-consumer budget
# (gemm16_nn_v4_kernels.mojo) as a plausible value, not one measured for
# THIS body.  CONSUMERS == 4 (BM == 256) is unreachable: the `constrained`
# check in `_v4_tn_ws_body` below rejects it at compile time before this
# value would ever be used -- see that check for why (a measured GPU hang).
@always_inline
def _v4_wave_reg_alloc[CONSUMERS: Int]() -> Int:
    comptime if CONSUMERS <= 2:
        return 232
    comptime if CONSUMERS == 3:
        return 160
    return 120


# Kernel-symbol suffix for a fused-bias direct-store instantiation, so
# profiles and scripts/compare_kernel_asm.py's by-name pairing can tell it
# apart from the (pre-existing, unsuffixed) unbiased kernel.
@always_inline
def _v4_bias_tag[HAS_BIAS: Bool]() -> StaticString:
    comptime if HAS_BIAS:
        return "_bias"
    return ""


# ============================================================================
# Wave-fill cost model -- the dispatch rule between the tile/split
# configurations in the closed menu below, ported from the Python reference
# model of the deep-K wave-fill engagement (model.py; see NOTES.md sections
# 2-3). Not a general tile search: the menu this family actually builds is
# {128x256, 128x192 (direct only), 128x128 (6-stage split only)}, and every
# call site below enumerates it explicitly -- no formula derives a
# candidate that does not have a real kernel behind it.
#
# RAMP is the fixed non-MMA cost of one CTA, in units of BK-tiles of
# mainloop (pipeline fill, WGMMA drain, epilogue store); TILE_FACTOR is a
# per-tile efficiency measured relative to that ramp model (it folds in
# operand traffic per MAC and, for BM=64, the single-consumer-warp-group
# penalty -- this family never dispatches BM=64 through this model, so only
# the BM=128 entries are defined). Both were fitted on an H100 PCIe
# (114 SMs) at a 1395 MHz clock pin; a different card needs its own fit
# (see NOTES.md). The absolute microsecond estimate below is only as
# accurate as that fit; the ARGMIN over the menu is far more robust, since
# every candidate's estimate scales by the same clock/peak-throughput
# constant.
comptime _V4_WAVE_RAMP: Float64 = 17.5
comptime _V4_WAVE_TF_128x256: Float64 = 0.958
comptime _V4_WAVE_TF_128x192: Float64 = 0.939
comptime _V4_WAVE_TF_128x128: Float64 = 0.797
comptime _V4_WAVE_REDUCE_BYTES_PER_US: Float64 = 3.4e6
comptime _V4_WAVE_SPLIT_LAUNCH_US: Float64 = 1.6
comptime _V4_WAVE_PEAK_MAC_PER_SM_PER_US: Float64 = 2048.0 * 1395.0


@always_inline
def _v4_wave_cost_us(
    m: Int,
    n: Int,
    k: Int,
    bn: Int,
    splits: Int,
    tile_factor: Float64,
    sm_count: Int,
) -> Float64:
    """Predicted device microseconds for one (128, bn, splits) config.

    `rate(chunk) = chunk / (chunk + RAMP) * tile_factor`; `us = load /
    peak_mac_per_sm_per_us / rate`, plus the split-K workspace round trip
    when splits > 1 (a fp32 write + read of the whole output per split,
    plus one launch gap) -- the exact formula `model.py::cost_us` computes,
    restricted to BM = 128 (the only row tile this family's non-split-K-64
    menu uses).
    """
    var tiles = ((m + _V4_BM - 1) // _V4_BM) * ((n + bn - 1) // bn)
    var t = tiles * splits
    var rounds = (t + sm_count - 1) // sm_count
    var k_tiles = k // _V4_BK
    var chunk = (k_tiles + splits - 1) // splits
    var load = (
        Float64(rounds)
        * Float64(_V4_BM)
        * Float64(bn)
        * Float64(chunk)
        * Float64(_V4_BK)
    )
    var rate = (Float64(chunk) / (Float64(chunk) + _V4_WAVE_RAMP)) * tile_factor
    var us = load / _V4_WAVE_PEAK_MAC_PER_SM_PER_US / rate
    if splits > 1:
        us += (
            Float64(splits)
            * Float64(m)
            * Float64(n)
            * 4.0
            / _V4_WAVE_REDUCE_BYTES_PER_US
            + _V4_WAVE_SPLIT_LAUNCH_US
        )
    return us


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
    BM: Int,
    BN: Int,
    STAGES: Int,
    SPLITK: Bool,
    GROUP_ROWS: Int,
    COL_A: Bool,
    KMAJ_B: Bool,
    HAS_BIAS: Bool,
    # TMA-store epilogue for the direct (SPLITK=False) arm only -- see
    # _V4_DIRECT_TMA_STORE above and NOTES.md (deep-K wave-fill
    # engagement), phase 5, sections 22-26. Every pre-existing call site
    # omits this parameter, so it defaults False and those instantiations
    # stay codegen-identical (verified with scripts/compare_kernel_asm.py).
    TMA_STORE: Bool = False,
](
    a_tma: TMATensorTile[_V4_DT, 2, a_tile, a_desc],
    b_tma: TMATensorTile[_V4_DT, 2, b_tile, b_desc],
    output: _V4_PTR,
    ws: _V4_F32_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    chunk_tiles: Int,
    # bf16 (m, n) output descriptor, box (BM, 64) -- the one the production
    # persistent body already builds for its own TMA-store epilogue. Only
    # read when TMA_STORE (which the comptime assert below restricts to the
    # SPLITK=False arm); every split-K call site omits it, so this default
    # -- an empty, never-filled descriptor -- is never touched.
    c_tma: TMATensorTile[_V4_DT, 2, Index(BM, 64), Index(BM, 64)] = (
        TMATensorTile[_V4_DT, 2, Index(BM, 64), Index(BM, 64)](TMADescriptor())
    ),
):
    comptime assert not TMA_STORE or not SPLITK, (
        "gemm16 wave body: TMA_STORE is only valid on the direct"
        " (SPLITK=False) arm. NOTES.md (deep-K wave-fill engagement),"
        " phase 5 section 24d measured a 2.27%"
        " REGRESSION on many-wave split-K grids -- the drain holds the"
        " staging smem (which aliases the retired B pipeline) resident"
        " past the point a scalar store would have retired the CTA -- while"
        " the split-K workspace store is already 1.00x sector-efficient"
        " (section 24a), so there is nothing to recover there either."
    )
    # The bf16 staging tile (BM x BN x 2 B) must fit in the retired B
    # pipeline it aliases (STAGES x BN x BK x 2 B) -- cancel BN x 2 B from
    # both sides and the fit condition is BM <= STAGES * BK. Every
    # instantiation that sets TMA_STORE today (BM=128/STAGES in {3, 4} and
    # BM=64/STAGES=3) clears this with room to spare; this guards a future
    # one that doesn't. See NOTES.md (deep-K wave-fill engagement), phase 5,
    # section 23.
    comptime assert not TMA_STORE or BM <= STAGES * _V4_BK, (
        "gemm16 wave body: TMA_STORE's staging tile does not fit in the"
        " retired B pipeline it aliases (BM must be <= STAGES * BK)."
    )
    comptime if _is_sm_9x():
        # BM >= 256 (CONSUMERS >= 4) builds clean -- ptxas accepts the SASS
        # -- but HANGS the GPU at runtime: the body's warpgroup_reg_alloc
        # asks for more registers per thread than the launch-bounds cap
        # leaves it (measured: BM=256/BN=128 needs 120, the cap at 640
        # threads is 102), and it spins forever in the barrier wait.
        # BM=192/BN=256 fails differently but just as unsupported (ptxas:
        # "Insufficient registers (128) ... Try 154 or higher"). Neither
        # failure is caught by any other check in this body, so a caller
        # that instantiates BM >= 256 gets a silently-hanging kernel instead
        # of a build error. Reject it here instead, as loudly as the
        # 192x256 ptxas failure -- see NOTES.md (deep-K wave-fill
        # engagement, probe_bm256) for the measurements.
        comptime assert BM < 256, (
            "gemm16 wave body: BM >= 256 is not supported -- it builds"
            " clean but HANGS the GPU at runtime (warpgroup_reg_alloc"
            " exceeds the sm_90 launch-bound register cap). See"
            " NOTES.md's deep-K wave-fill engagement (probe_bm256) for"
            " the measurement."
        )
        comptime CONSUMERS = BM // 64
        comptime REG_ALLOC = _v4_wave_reg_alloc[CONSUMERS]()
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
            warpgroup_reg_alloc[REG_ALLOC]()
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
            elif TMA_STORE:
                # TMA-store epilogue, direct arm only (the comptime assert
                # above rejects SPLITK=True). Stage the tile in shared
                # memory and hand it to TMA -- L2 write sectors halve versus
                # the scalar path below because a bf16 store is a half-
                # sector (16 B) wavefront but a TMA bulk-tensor store is not.
                # See NOTES.md (deep-K wave-fill engagement), phase 5,
                # sections 22 and 24b.
                #
                # The staging tile ALIASES the retired B pipeline instead of
                # a fresh allocation (section 23): this consumer only
                # reaches here after its own mainloop above has drained
                # every stage of `b_pipeline` (the last `full_barriers[
                # stage].wait` already cleared), and the producer warp group
                # issued its last TMA load before that same wait cleared, so
                # b_pipeline's smem is free to overwrite. The named_barrier
                # orders every consumer's last WGMMA read before the first
                # write into it.
                comptime NCONS = Int32(CONSUMERS * 128)
                var c_smem = b_pipeline.ptr
                named_barrier[NCONS](1)
                comptime for q in range(CFRAG // 2):
                    var e = q * 2
                    var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                    var col = base_col + (q // 2) * 8
                    var v0 = accum.ptr[e]
                    var v1 = accum.ptr[e + 1]
                    comptime if HAS_BIAS:
                        v0 += bias[n0 + col].cast[_V4_F32]()
                        v1 += bias[n0 + col + 1].cast[_V4_F32]()
                    var pair = SIMD[_V4_DT, 2](
                        v0.cast[_V4_DT](), v1.cast[_V4_DT]()
                    )
                    # 128B-swizzled staging layout -- identical formula to
                    # the persistent body's own TMA-store epilogue
                    # (gemm16_nn_v4_kernels.mojo): 16B units within each
                    # 64-element row are XORed with (row % 8). No manual
                    # m/n edge clip here (unlike the scalar path below): the
                    # c_tma descriptor is built over the exact (m, n) output
                    # extent, so the hardware clips a ragged final row/column
                    # tile the same way it already does for a_tma/b_tma
                    # loads.
                    var lcol = col % 64
                    var elem = (
                        (col // 64) * (BM * 64)
                        + row * 64
                        + ((lcol // 8) ^ (row % 8)) * 8
                        + lcol % 8
                    )
                    c_smem.store[alignment=4](elem, pair)
                fence_async_view_proxy()
                named_barrier[NCONS](1)
                if warp_group_idx == 1 and warp_group_thread_idx == 0:
                    comptime for chunk in range(BN // 64):
                        var c_chunk = LayoutTensor[
                            _V4_DT,
                            Layout.row_major(BM, 64),
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](c_smem + chunk * BM * 64)
                        c_tma.async_store(c_chunk, (n0 + chunk * 64, m0))
                    c_tma.commit_group()
                    # Outstanding bulk store must complete before the CTA
                    # (and its aliased smem) tears down.
                    c_tma.wait_group[0]()
            else:
                comptime for q in range(CFRAG // 2):
                    var e = q * 2
                    var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                    var col = base_col + (q // 2) * 8
                    var v0 = accum.ptr[e]
                    var v1 = accum.ptr[e + 1]
                    comptime if HAS_BIAS:
                        # bias is an (n,) row vector broadcast over every
                        # output row; the two accumulated columns are
                        # n0+col and n0+col+1.
                        v0 += bias[n0 + col].cast[_V4_F32]()
                        v1 += bias[n0 + col + 1].cast[_V4_F32]()
                    var pair = SIMD[_V4_DT, 2](
                        v0.cast[_V4_DT](), v1.cast[_V4_DT]()
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
    _v4_tn_ws_body[_V4_BM, 256, 4, True, 8, True, False, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
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
    _v4_tn_ws_body[_V4_BM, 256, 4, True, 8, False, True, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
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
    _v4_tn_ws_body[_V4_BM, 256, 4, True, 8, False, False, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
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
    _v4_tn_ws_body[_V4_BM, 256, 4, True, 8, True, True, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


# 128x128, 6-stage split-K tile: the deep-K wave-fill engagement's fix for
# the short-chunk regime (e.g. 768x768x8192), where the 128x256 tile above
# only reaches 95% of one wave's worth of CTAs and is still ~1.4x off stock
# because each CTA's K-chunk is too shallow to amortize the mainloop's
# fixed per-CTA ramp cost (see NOTES.md section 2 and 3). Same body, same
# split-K store; only BN and STAGES differ. Dispatched by the `_v4_wave_cost_us`
# comparison inline in each caller (try_enqueue_gemm16_gemm_tn_v4,
# try_enqueue_gemm16_gemm_tt_v4, try_enqueue_gemm16_gemm_splitk_rm_v4),
# never unconditionally -- deeper 128x256 splits remain the winner whenever
# the chunk stays reasonably deep (e.g. S4, W3).
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tn_v4_splitk_m128n128_s6")
def _v4_tn_splitk_m128n128_s6(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 128), Index(_V4_BK, 64)],
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
    _v4_tn_ws_body[_V4_BM, 128, 6, True, 8, True, False, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nt_v4_splitk_m128n128_s6")
def _v4_nt_splitk_m128n128_s6(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(128, _V4_BK), Index(128, _V4_BK)],
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
    _v4_tn_ws_body[_V4_BM, 128, 6, True, 8, False, True, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nn_v4_splitk_m128n128_s6")
def _v4_nn_splitk_m128n128_s6(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 128), Index(_V4_BK, 64)],
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
    _v4_tn_ws_body[_V4_BM, 128, 6, True, 8, False, False, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


# NN ragged-N split-K 128x192 tile: the fix for A1-class shapes (n % 256 !=
# 0 but n % 64 == 0, e.g. 1152x1088x7936), where the 256- and 128-wide
# split-K tiles above are both unreachable because their host-side tile
# counts assume an exact factor of n -- even though the shared body
# already ceil-divs blocks_n and predicates its store, exactly like the
# ragged-N *direct* 128x192 kernel a few hundred lines down
# (`_v4_nn_direct_m128n192_s3g16`).  Same body, same workspace + reduce
# store as the 128/256 split-K kernels above; only BN (and the resulting B
# TMA box count) differs.  See NOTES.md, deep-K wave-fill engagement,
# section 20, and `try_enqueue_gemm16_gemm_splitk_rm_v4`'s ragged-N branch
# for the dispatch side.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nn_v4_splitk_m128n192_s4")
def _v4_nn_splitk_m128n192_s4(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
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
    _v4_tn_ws_body[_V4_BM, 192, 4, True, 8, False, False, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_splitk_m128n128_s6")
def _v4_tt_splitk_m128n128_s6(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(128, _V4_BK), Index(128, _V4_BK)],
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
    _v4_tn_ws_body[_V4_BM, 128, 6, True, 8, True, True, False](
        a_tma,
        b_tma,
        ws.bitcast[Scalar[_V4_DT]](),
        ws,
        ws.bitcast[Scalar[_V4_DT]](),
        m,
        n,
        k,
        chunk_tiles,
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(
    t"{_GEMM16_TAG}_gemm_tn_v4_direct_m128n192_s4{_v4_bias_tag[HAS_BIAS]()}"
)
def _v4_tn_direct_m128n192_s4[
    HAS_BIAS: Bool = False
](
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BM, 64), Index(_V4_BM, 64)],
    output: _V4_PTR,
    bias: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    _v4_tn_ws_body[
        _V4_BM, 192, 4, False, 8, True, False, HAS_BIAS, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        bias,
        m,
        n,
        k,
        0,
        c_tma,
    )


# Multi-wave variant: 3 stages draw measurably less power than 4 on this
# power-limited card (sustained multi-wave runs throttle), and 16-row
# rasterization groups halve DRAM traffic for B, which all row-tiles share.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(
    t"{_GEMM16_TAG}_gemm_tn_v4_direct_m128n192_s3g16{_v4_bias_tag[HAS_BIAS]()}"
)
def _v4_tn_direct_m128n192_s3g16[
    HAS_BIAS: Bool = False
](
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BM, 64), Index(_V4_BM, 64)],
    output: _V4_PTR,
    bias: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    _v4_tn_ws_body[
        _V4_BM, 192, 3, False, 16, True, False, HAS_BIAS, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        bias,
        m,
        n,
        k,
        0,
        c_tma,
    )


# NN (dgrad) direct 128x192 tile: the row-major-A twin of the TN kernels
# above, for the same 2-wave underfilled regime (e.g. 2048x2048x8192) --
# NN never had a narrow-tile alternative to the persistent 128x256 kernel
# before this (see NOTES.md, deep-K wave-fill engagement); this is that
# alternative. Same body, same store; only COL_A flips (A is physically
# (M, K), K-major shared tile) so it reads through `_v4_make_a_row_tma`
# instead of the col-major `_v4_make_a_tma`.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nn_v4_direct_m128n192_s4")
def _v4_nn_direct_m128n192_s4(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BM, 64), Index(_V4_BM, 64)],
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
    _v4_tn_ws_body[
        _V4_BM, 192, 4, False, 8, False, False, False, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        output,
        m,
        n,
        k,
        0,
        c_tma,
    )


# Multi-wave variant, matching the TN kernel's rationale: 3 stages draw
# measurably less power on this power-limited card, and 16-row
# rasterization groups halve DRAM traffic for B.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_nn_v4_direct_m128n192_s3g16")
def _v4_nn_direct_m128n192_s3g16(
    a_tma: TMATensorTile[
        _V4_DT, 2, Index(_V4_BM, _V4_BK), Index(_V4_BM, _V4_BK)
    ],
    b_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 192), Index(_V4_BK, 64)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BM, 64), Index(_V4_BM, 64)],
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
    _v4_tn_ws_body[
        _V4_BM, 192, 3, False, 16, False, False, False, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        output,
        m,
        n,
        k,
        0,
        c_tma,
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
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V4_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_direct_m128n64_s4")
def _v4_tt_direct_m128n64_s4(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, _V4_BM), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(64, _V4_BK), Index(64, _V4_BK)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BM, 64), Index(_V4_BM, 64)],
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
    _v4_tn_ws_body[
        _V4_BM, 64, 4, False, 8, True, True, False, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        output,
        m,
        n,
        k,
        0,
        c_tma,
    )


# 64-row single-consumer TT instantiation of the same shared body (256
# threads, 3 stages): byte-for-byte the v3 NN small kernel's geometry, with
# the TT operand modes.
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(t"{_GEMM16_TAG}_gemm_tt_v4_direct_m64n128_s3")
def _v4_tt_direct_m64n128_s3(
    a_tma: TMATensorTile[_V4_DT, 2, Index(_V4_BK, 64), Index(_V4_BK, 64)],
    b_tma: TMATensorTile[_V4_DT, 2, Index(128, _V4_BK), Index(128, _V4_BK)],
    c_tma: TMATensorTile[_V4_DT, 2, Index(64, 64), Index(64, 64)],
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
    _v4_tn_ws_body[
        64, 128, 3, False, 8, True, True, False, _V4_DIRECT_TMA_STORE
    ](
        a_tma,
        b_tma,
        output,
        output.bitcast[Scalar[_V4_F32]](),
        output,
        m,
        n,
        k,
        0,
        c_tma,
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
# that engage the bias epilogue always satisfy n % 64 == 0 -- every split-K
# entry gate requires it (256-, 128- and, since the ragged-N NN candidate,
# 192-wide tiles alike; see the `try_enqueue_*` call sites that pass
# has_bias=True), so n is always a multiple of 4 and every vec4 group's flat
# offset is itself a multiple of 4: `base % n` therefore never straddles a
# row boundary, and a 4-element bias read is always a contiguous,
# correctly-ordered slice of one row. has_bias=False (every pre-existing
# caller) skips the extra loads/adds entirely.
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


# Output as (m, n) row-major, box (BM, 64) -- the descriptor the direct
# arm's TMA-store epilogue drains into (see _v4_tn_ws_body's TMA_STORE
# branch and _V4_DIRECT_TMA_STORE above). Same descriptor shape the
# persistent body in gemm16_nn_v4_kernels.mojo already builds for its own
# TMA-store epilogue.
def _v4_make_c_tma[
    BM: Int
](c: _V4_PTR, m: Int, n: Int, ctx: DeviceContext) raises -> TMATensorTile[
    _V4_DT, 2, Index(BM, 64), Index(BM, 64)
]:
    var c_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            c.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, n),
        IndexList[2](n, 1),
        IndexList[2](BM, 64),
    )
    return TMATensorTile[_V4_DT, 2, Index(BM, 64), Index(BM, 64)](c_desc)


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


# 128x128, 6-stage split-K enqueue: identical structure to
# `_v4_enqueue_splitk_m128n256` above (same workspace + fused-bias reduce),
# only the narrower tile's kernels are selected. See
# `_v4_tn_splitk_m128n128_s6` for why this tile exists.
def _v4_enqueue_splitk_m128n128_s6[
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
        ctx.enqueue_function[_v4_tn_splitk_m128n128_s6](
            _v4_make_a_tma(a, m, k, ctx),
            _v4_make_b_mn_tma[128](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    elif not COL_A and KMAJ_B:
        ctx.enqueue_function[_v4_nt_splitk_m128n128_s6](
            _v4_make_a_row_tma(a, m, k, ctx),
            _v4_make_b_kmaj_tma[128](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    elif not COL_A and not KMAJ_B:
        ctx.enqueue_function[_v4_nn_splitk_m128n128_s6](
            _v4_make_a_row_tma(a, m, k, ctx),
            _v4_make_b_mn_tma[128](b, n, k, ctx),
            ws_ptr,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(chunk_tiles),
            grid_dim=(grid_x, splits),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_tt_splitk_m128n128_s6](
            _v4_make_a_tma(a, m, k, ctx),
            _v4_make_b_kmaj_tma[128](b, n, k, ctx),
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


# 128x192 NN split-K enqueue: same workspace + fused-bias reduce structure
# as the two helpers above, for the ragged-N candidate
# (`_v4_nn_splitk_m128n192_s4`) `try_enqueue_gemm16_gemm_splitk_rm_v4`
# dispatches to.  NN-only (see that dispatcher's ragged-N branch for why).
def _v4_enqueue_nn_splitk_m128n192(
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
    ctx.enqueue_function[_v4_nn_splitk_m128n192_s4](
        _v4_make_a_row_tma(a, m, k, ctx),
        _v4_make_b_mn_tma[192](b, n, k, ctx),
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


def _v4_enqueue_direct_m128n192[
    HAS_BIAS: Bool = False
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
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
    var c_tma = _v4_make_c_tma[_V4_BM](output, m, n, ctx)
    if multi_wave:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s3g16[HAS_BIAS]](
            a_tma,
            b_tma,
            c_tma,
            output,
            bias,
            Int64(m),
            Int64(n),
            Int64(k),
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_tn_direct_m128n192_s4[HAS_BIAS]](
            a_tma,
            b_tma,
            c_tma,
            output,
            bias,
            Int64(m),
            Int64(n),
            Int64(k),
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )


# NN twin of `_v4_enqueue_direct_m128n192` above -- row-major A instead of
# col-major.  No bias epilogue: the NN direct-192 route is only used to
# cover the 2-wave mm regime (see the cost-model comparison inline in
# `try_enqueue_gemm16_gemm_nn_v4` below); the recorded addmm win comes from
# the persistent kernel's own fused-bias epilogue (gemm16_nn_v4_kernels.mojo)
# instead.
def _v4_enqueue_nn_direct_m128n192(
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
    var a_tma = _v4_make_a_row_tma(a, m, k, ctx)
    var b_tma = _v4_make_b_mn_tma[192](b, n, k, ctx)
    var c_tma = _v4_make_c_tma[_V4_BM](output, m, n, ctx)
    if multi_wave:
        ctx.enqueue_function[_v4_nn_direct_m128n192_s3g16](
            a_tma,
            b_tma,
            c_tma,
            output,
            Int64(m),
            Int64(n),
            Int64(k),
            grid_dim=(grid_x,),
            block_dim=(_V4_THREADS,),
        )
    else:
        ctx.enqueue_function[_v4_nn_direct_m128n192_s4](
            a_tma,
            b_tma,
            c_tma,
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
        _v4_make_c_tma[_V4_BM](output, m, n, ctx),
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
        _v4_make_c_tma[64](output, m, n, ctx),
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(256,),
    )


# ============================================================================
# NN (dgrad) regime dispatch: persistent 128x256 vs the new direct 128x192
# tile.  Unlike the TN fix below, NN never had a narrow-tile alternative to
# the persistent kernel before this engagement, so this isn't a dispatch-
# ORDERING bug -- it's a new route, gated by the exact same wave-fill cost
# model and the same "persistent wins outright at >= 3 waves" rule (see
# NOTES.md, deep-K wave-fill engagement, and try_enqueue_gemm16_gemm_tn_v4's
# docstring below for the full rationale). `has_bias` always takes the
# persistent path (it fuses a bias epilogue -- see gemm16_nn_v4_kernels.mojo)
# and skips the 2-wave cost-model comparison entirely: the direct 128x192 NN
# kernel has no bias epilogue, and adding one is out of scope here (the
# recorded addmm win for well-filled NN shapes already comes from the
# persistent kernel).
# ============================================================================
def try_enqueue_gemm16_gemm_nn_v4(
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
    comptime if _has_sm_9x():
        if ctx.api() == "cuda" and not has_bias and n % 256 == 0:
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if (
                cc_major == 9
                and cc_minor == 0
                and m >= _V4_BM
                and k >= _V4_BK
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
                var max_grid_x = ctx.get_attribute(
                    DeviceAttribute.MAX_GRID_DIM_X
                )
                if sm_count > 0 and max_grid_x > 0:
                    var blocks_m256 = (m + _V4_BM - 1) // _V4_BM
                    var tiles256 = blocks_m256 * (n // 256)
                    var waves256 = (tiles256 + sm_count - 1) // sm_count
                    if waves256 == 2:
                        var tiles192 = blocks_m256 * ((n + 191) // 192)
                        if tiles192 > 0 and tiles192 <= max_grid_x:
                            var waves192 = (tiles192 + sm_count - 1) // sm_count
                            var cost256 = _v4_wave_cost_us(
                                m, n, k, 256, 1, _V4_WAVE_TF_128x256, sm_count
                            )
                            var cost192 = _v4_wave_cost_us(
                                m, n, k, 192, 1, _V4_WAVE_TF_128x192, sm_count
                            )
                            if cost192 < cost256:
                                _v4_enqueue_nn_direct_m128n192(
                                    output,
                                    a,
                                    b,
                                    m,
                                    n,
                                    k,
                                    tiles192,
                                    waves192 > 1,
                                    ctx,
                                )
                                return True
    return maybe_enqueue_gemm16_nn_v4(
        output, a, b, bias, m, n, k, False, False, has_bias, ctx
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

    # Split-K regime: only when at least two K-chunks per tile fit within
    # the SM count, each chunk deep enough to amortize pipeline ramp-up,
    # and the fp32 workspace stays modest.  Once engaged, the wave-fill
    # cost model (`_v4_wave_cost_us`, ported from model.py -- see NOTES.md
    # deep-K wave-fill engagement, sections 2-3) picks between the 128x256
    # tile and the narrower 128x128/6-stage one: the latter wins on
    # shallow-chunk shapes where 256-wide splits don't amortize their
    # per-CTA ramp cost (e.g. 768x768x8192: 1.37x -> 1.11x).  n % 256 == 0
    # implies n % 128 == 0, so the 128-wide tile is always in the menu here.
    if n % 256 == 0:
        var tiles256 = (m // _V4_BM) * (n // 256)
        if tiles256 > 0 and tiles256 <= max_grid_x and 2 * tiles256 <= sm_count:
            var splits256 = sm_count // tiles256
            if splits256 > _V4_MAX_SPLITS:
                splits256 = _V4_MAX_SPLITS
            var max_by_depth = (k // _V4_BK) // _V4_MIN_CHUNK_TILES
            if splits256 > max_by_depth:
                splits256 = max_by_depth
            if splits256 >= 2 and m * n <= _V4_MAX_WS_BYTES // 4 // splits256:
                var tiles128 = (m // _V4_BM) * (n // 128)
                var splits128 = sm_count // tiles128
                if splits128 > _V4_MAX_SPLITS:
                    splits128 = _V4_MAX_SPLITS
                if splits128 > max_by_depth:
                    splits128 = max_by_depth
                var use_128 = (
                    splits128 >= 2
                    and tiles128 <= max_grid_x
                    and m * n <= _V4_MAX_WS_BYTES // 4 // splits128
                    and _v4_wave_cost_us(
                        m, n, k, 128, splits128, _V4_WAVE_TF_128x128, sm_count
                    )
                    < _v4_wave_cost_us(
                        m, n, k, 256, splits256, _V4_WAVE_TF_128x256, sm_count
                    )
                )
                if use_128:
                    _v4_enqueue_splitk_m128n128_s6(
                        output,
                        a,
                        b,
                        bias,
                        m,
                        n,
                        k,
                        tiles128,
                        splits128,
                        has_bias,
                        ctx,
                    )
                else:
                    _v4_enqueue_splitk_m128n256(
                        output,
                        a,
                        b,
                        bias,
                        m,
                        n,
                        k,
                        tiles256,
                        splits256,
                        has_bias,
                        ctx,
                    )
                return True

    # Multi-wave regime, exactly 2 waves of 128x256 tiles: compare the
    # persistent clustered kernel against the direct 128x192 tile via the
    # wave-fill cost model and take whichever wins, instead of always
    # taking persistent. This is the fix for the regression the deep-K
    # wave-fill engagement found: the direct 128x192 kernel two paragraphs
    # below already existed, but the unconditional persistent call right
    # after this block made it unreachable for every multi-wave shape (see
    # NOTES.md, deep-K wave-fill engagement, section 8/must-fix 4) --
    # W1-class shapes (2048x2048x8192) measured 1.74x on persistent and
    # 1.32x on 128x192, and the narrow tile was never tried. At 3+ waves
    # the persistent kernel's per-tile amortization (cluster B multicast,
    # background TMA-store epilogue) measures ~6% faster than a plain grid
    # launch of the same tile on this hardware (NOTES.md section 4), an
    # advantage this cost model was not fitted against (its TILE_FACTOR
    # table comes from non-persistent launches) -- so persistent keeps an
    # unconditional win there, exactly as before this change. Both
    # candidates fuse bias into their direct store, so this comparison
    # runs for addmm too.
    if n % 256 == 0 and n % 192 == 0:
        var blocks_m256 = (m + _V4_BM - 1) // _V4_BM
        var tiles256_mw = blocks_m256 * (n // 256)
        var waves256_mw = (tiles256_mw + sm_count - 1) // sm_count
        if waves256_mw == 2:
            var tiles192_mw = blocks_m256 * (n // 192)
            if tiles192_mw > 0 and tiles192_mw <= max_grid_x:
                var waves192_mw = (tiles192_mw + sm_count - 1) // sm_count
                var cost256_mw = _v4_wave_cost_us(
                    m, n, k, 256, 1, _V4_WAVE_TF_128x256, sm_count
                )
                var cost192_mw = _v4_wave_cost_us(
                    m, n, k, 192, 1, _V4_WAVE_TF_128x192, sm_count
                )
                if cost192_mw < cost256_mw:
                    if has_bias:
                        _v4_enqueue_direct_m128n192[True](
                            output,
                            a,
                            b,
                            bias,
                            m,
                            n,
                            k,
                            tiles192_mw,
                            waves192_mw > 1,
                            ctx,
                        )
                    else:
                        _v4_enqueue_direct_m128n192(
                            output,
                            a,
                            b,
                            output,
                            m,
                            n,
                            k,
                            tiles192_mw,
                            waves192_mw > 1,
                            ctx,
                        )
                    return True

    # Multi-wave regime: the persistent clustered body (shared with NN)
    # in its col-major-A mode, with a fused-bias epilogue (see
    # gemm16_nn_v4_kernels.mojo).  Gated inside the helper; it declines
    # single-wave and unaligned shapes, which fall through to the
    # narrow-tile / v3 routes below (bias-incapable, so has_bias returns
    # False from here on if this declines).
    if maybe_enqueue_gemm16_tn_v4_persistent(
        output, a, b, bias, m, n, k, has_bias, ctx
    ):
        return True

    if has_bias:
        return False

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
            output, a, b, output, m, n, k, False, ctx
        ):
            return True

    # Narrow-tile regime: 192-wide tiles trade 33% more CTAs for fuller
    # waves.  Per-CTA time is proportional to BN at fixed BM/BK, so compare
    # wave-quantized cost (waves x BN) and pick 192 when it wins; e.g. 72
    # CTAs on 114 SMs (one third-idle wave) improves. The exactly-2-wave
    # case above already ran the full cost-model comparison and returned;
    # this cheaper wave-quantized proxy covers what's left: single-wave
    # shapes (where persistent already declined outright) and n % 256 != 0
    # ragged shapes (no cost-model candidate above; n % 192 == 0 can still
    # hold there).
    if n % 192 == 0:
        var tiles192 = (m // _V4_BM) * (n // 192)
        if tiles192 > 0 and tiles192 <= max_grid_x:
            if n % 256 == 0:
                var tiles256 = (m // _V4_BM) * (n // 256)
                var waves256 = (tiles256 + sm_count - 1) // sm_count
                var waves192 = (tiles192 + sm_count - 1) // sm_count
                if waves192 * 192 < waves256 * 256:
                    _v4_enqueue_direct_m128n192(
                        output,
                        a,
                        b,
                        output,
                        m,
                        n,
                        k,
                        tiles192,
                        waves192 > 1,
                        ctx,
                    )
                    return True
            elif tiles192 <= sm_count:
                # No 256-wide alternative; take the single-wave win only.
                _v4_enqueue_direct_m128n192(
                    output, a, b, output, m, n, k, tiles192, False, ctx
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
                # Wave-fill cost model: compare against the narrower
                # 128x128/6-stage split tile, same rule as the TN
                # dispatcher above (see NOTES.md, deep-K wave-fill
                # engagement).
                var tiles128 = (m // _V4_BM) * (n // 128)
                var splits128 = sm_count // tiles128
                if splits128 > _V4_MAX_SPLITS:
                    splits128 = _V4_MAX_SPLITS
                if splits128 > max_by_depth:
                    splits128 = max_by_depth
                var use_128 = (
                    splits128 >= min_splits
                    and tiles128 <= max_grid_x
                    and m * n <= _V4_MAX_WS_BYTES // 4 // splits128
                    and _v4_wave_cost_us(
                        m, n, k, 128, splits128, _V4_WAVE_TF_128x128, sm_count
                    )
                    < _v4_wave_cost_us(
                        m, n, k, 256, splits, _V4_WAVE_TF_128x256, sm_count
                    )
                )
                if use_128:
                    _v4_enqueue_splitk_m128n128_s6[True, True](
                        output,
                        a,
                        b,
                        bias,
                        m,
                        n,
                        k,
                        tiles128,
                        splits128,
                        has_bias,
                        ctx,
                    )
                else:
                    _v4_enqueue_splitk_m128n256[True, True](
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
            output, a, b, output, m, n, k, False, ctx
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
    # as the TN dispatcher above).  n only needs to tile by 64: the shared
    # body (`_v4_tn_ws_body`) ceil-divs `blocks_n` and predicates every
    # store, so a ragged n does not need an exact factor of any tile width
    # -- the same reasoning as the phase-3 direct-192 ceil-div fix (git
    # history, "Fix the n%192 gate...").  n % 256 == 0 used to be required
    # here because the 256- and 128-wide candidates below are the only
    # ones that existed; they still gate themselves on their own exact
    # factor (256 and 128 respectively), and the new ragged-N candidate
    # further down (128x192, NN only) is what a non-multiple-of-256 n
    # actually reaches. See NOTES.md, deep-K wave-fill engagement,
    # section 20.
    if (
        m < _V4_BM
        or k < _V4_BK
        or m % _V4_BM != 0
        or k % _V4_BK != 0
        or n <= 0
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
    var k_tiles = k // _V4_BK
    var max_by_depth = k_tiles // _V4_MIN_CHUNK_TILES

    if n % 256 == 0:
        var tiles = (m // _V4_BM) * (n // 256)
        if tiles <= 0 or tiles > max_grid_x:
            return False
        var cap = sm_count // tiles
        if cap < _V4_SPLITK_RM_MIN_SPLITS:
            return False
        var splits = cap
        if splits > _V4_MAX_SPLITS:
            splits = _V4_MAX_SPLITS
        if splits > max_by_depth:
            splits = max_by_depth
        if splits < _V4_SPLITK_RM_MIN_SPLITS:
            return False
        var small_tile_covers = 4 * tiles < sm_count
        comptime if KMAJ_B:
            # NT: a marginal 2-way split of a minimum-depth K loses to the
            # persistent kernel's nearly-full wave (see the fit above).
            if cap == 2 and k_tiles < _V4_SPLITK_RM_DEEP_TILES:
                return False
        else:
            # NN: when the v3 small-tile kernel covers the output in one
            # wave, only a deep split beats it (see the fit above).
            if small_tile_covers and splits < _V4_SPLITK_RM_COVERED_MIN_SPLITS:
                return False
        if m * n > _V4_MAX_WS_BYTES // 4 // splits:
            return False

        # Wave-fill cost model: compare against the narrower 128x128/6-stage
        # split tile, same rule and same closed menu as the TN dispatcher
        # (see NOTES.md, deep-K wave-fill engagement).  The 128-wide
        # candidate is gated through the SAME per-layout floors as the
        # 256-wide one above (re-evaluated at its own split count) so it
        # only engages in a regime already validated for split-K on this
        # layout.
        var tiles128 = (m // _V4_BM) * (n // 128)
        var cap128 = sm_count // tiles128
        var splits128 = cap128
        if splits128 > _V4_MAX_SPLITS:
            splits128 = _V4_MAX_SPLITS
        if splits128 > max_by_depth:
            splits128 = max_by_depth
        var splits128_ok = (
            splits128 >= _V4_SPLITK_RM_MIN_SPLITS and tiles128 <= max_grid_x
        )
        comptime if KMAJ_B:
            if cap128 == 2 and k_tiles < _V4_SPLITK_RM_DEEP_TILES:
                splits128_ok = False
        else:
            if (
                small_tile_covers
                and splits128 < _V4_SPLITK_RM_COVERED_MIN_SPLITS
            ):
                splits128_ok = False
        if splits128_ok and m * n > _V4_MAX_WS_BYTES // 4 // splits128:
            splits128_ok = False
        if splits128_ok and _v4_wave_cost_us(
            m, n, k, 128, splits128, _V4_WAVE_TF_128x128, sm_count
        ) < _v4_wave_cost_us(
            m, n, k, 256, splits, _V4_WAVE_TF_128x256, sm_count
        ):
            _v4_enqueue_splitk_m128n128_s6[False, KMAJ_B](
                output, a, b, bias, m, n, k, tiles128, splits128, has_bias, ctx
            )
            return True
        _v4_enqueue_splitk_m128n256[False, KMAJ_B](
            output, a, b, bias, m, n, k, tiles, splits, has_bias, ctx
        )
        return True

    # Ragged-N regime: n % 256 != 0 but n % 64 == 0 (guaranteed by the
    # entry gate above), e.g. A1 1152x1088x7936 (n % 256 == 64).  Neither
    # exact-factor candidate above is available, but the shared body's
    # ceil-div tiling makes a 128x192 split-K tile just as exact as it is
    # for the already-shipped ragged-N *direct* 128x192 kernel
    # (`_v4_enqueue_nn_direct_m128n192`, phase 3's ceil-div fix). Gated by
    # the SAME split-worthiness predicate as the 128/256 candidates above
    # (min splits, chunk depth, small-tile coverage -- all fitted from the
    # same H100 PCIe sweep the module docstring describes), not by any
    # shape-specific check, so it engages only where the model says a
    # split actually pays for itself.  Wired for NN only (the measured
    # regime, NOTES.md deep-K wave-fill engagement section 20); NT is left
    # for a measured follow-up, same as the TN dispatcher's own n % 192
    # gate (git history, "Fix the n%192 gate...").
    comptime if not KMAJ_B:
        var tiles192 = (m // _V4_BM) * ((n + 191) // 192)
        if tiles192 <= 0 or tiles192 > max_grid_x:
            return False
        var cap192 = sm_count // tiles192
        if cap192 < _V4_SPLITK_RM_MIN_SPLITS:
            return False
        var splits192 = cap192
        if splits192 > _V4_MAX_SPLITS:
            splits192 = _V4_MAX_SPLITS
        if splits192 > max_by_depth:
            splits192 = max_by_depth
        if splits192 < _V4_SPLITK_RM_MIN_SPLITS:
            return False
        if (
            4 * tiles192 < sm_count
            and splits192 < _V4_SPLITK_RM_COVERED_MIN_SPLITS
        ):
            return False
        if m * n > _V4_MAX_WS_BYTES // 4 // splits192:
            return False
        _v4_enqueue_nn_splitk_m128n192(
            output, a, b, bias, m, n, k, tiles192, splits192, has_bias, ctx
        )
        return True
    return False
