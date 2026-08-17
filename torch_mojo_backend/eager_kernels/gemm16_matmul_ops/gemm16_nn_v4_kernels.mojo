"""Persistent clustered H100 16-bit NN GEMM (dgrad) kernels — v4.

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

Dynamic shapes: any problem in the tall-m NN regime with n % 64 == 0 and
k % BK == 0 is handled (m may be ragged: TMA clamps loads and clips
stores; n % BN != 0 selects the ragged_n `_nclip` instantiation, which
clips the trailing partial column of tiles the same way); everything else
must be routed to the existing v3 dispatcher by the caller
(`maybe_enqueue_...` returns False in that case).

The operand dtype is bfloat16 or float16, chosen at compile time by
`_GEMM16_DT` (gemm16_dtype.mojo); every tile size and pipeline constant
here is a function of the 2-byte operand width, not of the exponent
layout, so one source serves both.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import fence_async_view_proxy, fence_mbarrier_init
from std.memory import AddressSpace
from max.gpu.sync import named_barrier
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
    _convert_cfrags_to_simd,
    _convert_cfrags_to_tuple,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from gemm16_kernels import _pick_regime
from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG

comptime _V4_DT = _GEMM16_DT
comptime _V4_F32 = DType.float32
comptime _V4_PTR = UnsafePointer[Scalar[_V4_DT], MutAnyOrigin]
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


# One (64 x BN x BK) slab of WGMMA work per consumer warp group through the
# raw descriptor path, fence to fence.  TensorCoreAsync has no col-major A
# mode, so operand majorness is expressed with COL_A / KMAJ_B exactly like
# the non-persistent shared body in gemm16_tn_v4_kernels.mojo (which
# calls this helper too): canonical descriptor layouts follow each
# operand's majorness and the stride formulas are majorness-generic (they
# mirror TensorCoreAsync.wgmma).  The second consumer warp group advances
# by one 64-row WGMMA tile within the shared A tile.
@always_inline
def _v4_mma_tile[
    BN: Int,
    COL_A: Bool,
    KMAJ_B: Bool,
    A_LAYOUT: Layout,
    B_LAYOUT: Layout,
](
    a_smem: UnsafePointer[
        Scalar[_V4_DT], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    b_smem: UnsafePointer[
        Scalar[_V4_DT], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    accum: LayoutTensor[
        _V4_F32,
        Layout.row_major(1, 64 * BN // 128),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
    warp_group_idx: Int,
):
    comptime CFRAG = 64 * BN // 128
    comptime a_canonical_layout = tile_to_descriptor[
        _V4_DT, A_LAYOUT, not COL_A
    ]()
    comptime b_canonical_layout = tile_to_descriptor[_V4_DT, B_LAYOUT, KMAJ_B]()
    comptime a_shape00 = a_canonical_layout[0].shape[0].value()
    comptime a_stride01 = a_canonical_layout[0].stride[1].value()
    comptime a_stride11 = a_canonical_layout[1].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime a_m_stride = a_stride01 * (64 // a_shape00) * 2
    comptime a_k_stride = a_stride11 * 2 * 2
    comptime b_k_stride = b_stride11 * 2 * 2
    comptime NUM_K_MMAS = _V4_BK // 16
    var a_desc = _wgmma_descriptor[a_canonical_layout, not COL_A, _V4_SWIZZLE](
        a_smem
    )
    var b_desc = _wgmma_descriptor[b_canonical_layout, KMAJ_B, _V4_SWIZZLE](
        b_smem
    )
    a_desc += a_m_stride * (warp_group_idx - 1)

    warpgroup_fence(accum)
    wgmma_fence_aligned()
    comptime for k_mma in range(NUM_K_MMAS):
        var c_tuple = _convert_cfrags_to_tuple[_V4_F32, CFRAG](accum)
        var c_out = wgmma_async[
            64,
            BN,
            16,
            a_type=_V4_DT,
            b_type=_V4_DT,
            layout_a="col" if COL_A else "row",
            layout_b="col" if KMAJ_B else "row",
        ](
            a_desc + k_mma * a_k_stride,
            b_desc + k_mma * b_k_stride,
            c_tuple,
        )
        _convert_cfrags_to_simd[_V4_F32, CFRAG](c_out, accum)
    wgmma_commit_group_sync()
    warpgroup_fence(accum)
    wgmma_wait_group_sync()


# Kernel-symbol layout tag for the persistent body: col_a selects the TN
# (wgrad) instantiation, col_a + kmaj_b the TT one, plain NN (dgrad)
# otherwise.  (kmaj_b alone would be NT, which has its own dedicated
# persistent kernel in gemm16_nt_v4_kernels.mojo and is never
# instantiated here.)
@always_inline
def _v4_persistent_layout_tag[col_a: Bool, kmaj_b: Bool]() -> StaticString:
    comptime if col_a and kmaj_b:
        return "tt"
    comptime if col_a:
        return "tn"
    comptime if kmaj_b:
        return "nt"
    return "nn"


# Kernel-symbol tag for the ragged-n instantiation (the TT route uses it
# for n % 256 != 0): a suffix so profiles and the by-name kernel pairing of
# scripts/compare_kernel_asm.py can tell it apart from the exact-n
# instantiations, which keep their pre-existing bare names.
@always_inline
def _v4_persistent_ragged_tag[ragged_n: Bool]() -> StaticString:
    comptime if ragged_n:
        return "_nclip"
    return ""


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
# every exact-n symbol byte-identical to its pre-existing name.
@__name(
    t"{_GEMM16_TAG}_gemm_{_v4_persistent_layout_tag[col_a, kmaj_b]()}_v4_persistent{_v4_persistent_ragged_tag[ragged_n]()}"
)
def _v4_nn_persistent_ws[
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
    # has_bias fuses an (n,)-row-vector bias add into the epilogue, for
    # both the TMA-store and direct-store paths -- see the epilogue below.
    # Default False keeps every pre-existing instantiation codegen-
    # identical (verified with scripts/compare_kernel_asm.py).
    has_bias: Bool = False,
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
    comptime if _is_sm_9x():
        # Same guard as _v4_tn_ws_body in gemm16_tn_v4_kernels.mojo: a wide
        # consumer count built off bm >= 256 asks warpgroup_reg_alloc for
        # more registers per thread than the sm_90 launch-bound cap leaves
        # it, which builds clean (ptxas accepts the SASS) but HANGS the GPU
        # at runtime instead of failing loudly. Every instantiation here is
        # driven by the hardcoded _V4_PROD_BM = 128 (see above), so this is
        # a consistency guard, not a live constraint -- it exists so a
        # future caller that widens bm fails at compile time instead of at
        # a silent runtime hang.
        comptime assert bm < 256, (
            "gemm16 persistent body: bm >= 256 is not supported -- it"
            " builds clean but HANGS the GPU at runtime"
            " (warpgroup_reg_alloc exceeds the sm_90 launch-bound register"
            " cap). See _v4_tn_ws_body's matching guard in"
            " gemm16_tn_v4_kernels.mojo."
        )
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
        # C staging tile for the TMA-store epilogue (swizzled 128B rows of
        # 64 elements, bn // 64 chunks).  A dummy allocation when disabled.
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        var c_smem = LayoutTensor[
            _V4_DT,
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
        # epilogue stores are row-predicated.  With ragged_n, n may be too
        # (see the parameter comment above).
        var macro_rows = (m + MACRO_BM - 1) // MACRO_BM
        var blocks_n = n // bn
        comptime if ragged_n:
            blocks_n = (n + bn - 1) // bn
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
                            _V4_DT,
                            A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr + stage * bm * _V4_BK)
                        var k0 = t * _V4_BK
                        # TMA coordinates are (fastest dim, slower dim) of
                        # the global tensor the descriptor was built over:
                        # (m, k) for the col-major (K, M) wgrad operand.
                        comptime if col_a:
                            a_tma.async_copy(
                                a_tile, full_barriers[stage], (m0, k0)
                            )
                        else:
                            a_tma.async_copy(
                                a_tile, full_barriers[stage], (k0, m0)
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
                                b_pipeline.ptr
                                + stage * bn * _V4_BK
                                + cc * 64 * _V4_BK
                            )
                            # B TMA coordinates follow the descriptor's
                            # global tensor: (k, n) for the K-major (N, K)
                            # kmaj_b operand, (n, k) for the row-major
                            # (K, N) one.
                            comptime if cluster_m > 1:
                                comptime if kmaj_b:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[stage],
                                        (k0, n0 + cc * 64),
                                        MCAST_MASK,
                                    )
                                else:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[stage],
                                        (n0 + cc * 64, k0),
                                        MCAST_MASK,
                                    )
                            else:
                                comptime if kmaj_b:
                                    b_tma.async_copy(
                                        b_chunk,
                                        full_barriers[stage],
                                        (k0, n0 + cc * 64),
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
                _V4_DT,
                _V4_DT,
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
                        _V4_DT,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * bm * _V4_BK)
                    var b_tile = LayoutTensor[
                        _V4_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * bn * _V4_BK)
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
                        var v0 = accum.ptr[e]
                        var v1 = accum.ptr[e + 1]
                        comptime if has_bias:
                            # bias is an (n,) row vector broadcast over
                            # every output row.
                            v0 += bias[n0 + col].cast[_V4_F32]()
                            v1 += bias[n0 + col + 1].cast[_V4_F32]()
                        var pair = SIMD[_V4_DT, 2](
                            v0.cast[_V4_DT](), v1.cast[_V4_DT]()
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
                                _V4_DT,
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
                        var v0 = accum.ptr[e]
                        var v1 = accum.ptr[e + 1]
                        comptime if has_bias:
                            v0 += bias[n0 + col].cast[_V4_F32]()
                            v1 += bias[n0 + col + 1].cast[_V4_F32]()
                        var pair = SIMD[_V4_DT, 2](
                            v0.cast[_V4_DT](), v1.cast[_V4_DT]()
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
    col_a: Bool = False,
    kmaj_b: Bool = False,
    ragged_n: Bool = False,
    has_bias: Bool = False,
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    sm_count: Int,
    ctx: DeviceContext,
) raises:
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
            a.address_space_cast[AddressSpace.GENERIC](),
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
            b.address_space_cast[AddressSpace.GENERIC](),
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
            output.address_space_cast[AddressSpace.GENERIC](),
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
    var macro_rows = (m + bm * cluster_m - 1) // (bm * cluster_m)
    var blocks_n = n // bn
    comptime if ragged_n:
        blocks_n = (n + bn - 1) // bn
    var total_works = macro_rows * blocks_n
    var num_clusters = min(sm_count // cluster_m, total_works)
    var grid_x = num_clusters * cluster_m
    ctx.enqueue_function[
        _v4_nn_persistent_ws[
            stages,
            cluster_m,
            bm,
            bn,
            consumers,
            tma_store,
            col_a,
            kmaj_b,
            ragged_n,
            has_bias,
        ]
    ](
        a_tma,
        b_tma,
        c_tma,
        output,
        bias,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(128 * (consumers + 1),),
    )


def maybe_enqueue_gemm16_nn_v4(
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    bias: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    """Route an NN GEMM to the persistent clustered v4 kernel if it fits the
    aligned regime and fills the persistent grid.  Returns False when the
    caller must fall back."""
    comptime if _has_sm_9x():
        if ctx.api() == "cuda":
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if cc_major == 9 and cc_minor == 0:
                # Aligned NN regime, any aspect ratio.  m may be ragged (TMA
                # clamps reads, stores are predicated), and so may n down to
                # a multiple of 64: n % 256 == 0 launches the pre-existing
                # exact instantiation, anything else the ragged_n (_nclip)
                # one, whose B TMA reads clamp past the n edge and whose C
                # store clips the partial last column box -- the same rung
                # the TN and TT dispatchers already use.  Without it every
                # NN dgrad shape with n % 256 != 0 fell down the ladder:
                # n % 128 == 0 (GPT-2's padded vocab 50304 has
                # n % 256 == 128) onto the 64x128 one-CTA-per-tile v3 grid,
                # a ~1.9x loss to stock on 768x50304x49152, and
                # n % 128 == 64 all the way to the non-TMA wide fallback, a
                # ~4.6x cliff (1536x4160x1024: 141 us -> 31).  k must still
                # tile exactly.  The ordered bounds make all descriptor and
                # address products machine-width safe.
                if (
                    not transpose_a
                    and not transpose_b
                    and m >= _V4_PROD_BM * _V4_PROD_CLUSTER_M
                    and n >= _V4_PROD_BN
                    and k >= _V4_BK
                    and n % 64 == 0
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
                    # Wave-fill predicate: decline v4 only when the v3
                    # 64x128 one-CTA-per-tile kernel both (a) accepts the
                    # shape -- it additionally needs m % 64 == 0, while v4
                    # tolerates ragged m; without that guard a declined
                    # ragged-m shape falls through to the far slower wide
                    # fallback -- and (b) covers the whole output in a
                    # single wave: each 256x256 macro-tile is eight of its
                    # 64x128 tiles, so that is total_works * 8 < sm_count
                    # (strict, mirroring the v3 dispatcher's own
                    # use_small_tile inequality).  Everything else,
                    # including deep-K underfilled shapes such as
                    # 1024x1024x8192, keeps the persistent v4 kernel.
                    # Empirical device-time crossing, fitted on an H100
                    # PCIe (114 SMs): at total_works = 14 (3584x256x512)
                    # v3-small wins 6.95 us vs v4's 10.49 us; at
                    # total_works = 15 (3840x256x512) the small tile no
                    # longer fits one wave and v4 wins 10.48 us vs the
                    # 128x256 fallback's 12.74 us.
                    var macro_span = _V4_PROD_BM * _V4_PROD_CLUSTER_M
                    var macro_rows = (m + macro_span - 1) // macro_span
                    # Under a ragged n the trailing partial 256-column tile
                    # is a real work item the persistent scheduler will
                    # execute, so the census ceil-divides n; exact-n shapes
                    # keep the pre-existing floor division (equal when
                    # n % 256 == 0), and their dispatch below is unchanged.
                    var blocks_n = n // _V4_PROD_BN
                    if n % _V4_PROD_BN != 0:
                        blocks_n = (n + _V4_PROD_BN - 1) // _V4_PROD_BN
                    var total_works = macro_rows * blocks_n
                    # Exact n keeps the pre-existing *8 single-wave fit (see
                    # above).  The ragged rung declines to two real
                    # alternatives below it on the ladder:
                    #
                    # 1. The 64x64 s64 route at the bottom of the ladder
                    #    (gemm16_kernels.mojo).  Its own dispatcher
                    #    engages it exactly when _pick_regime returns the
                    #    64x64 regime, so the same call is the coverage
                    #    condition here and the two cannot drift apart.  A
                    #    covered shape has blocks_s64 <= sm_count, i.e. at
                    #    most ~sm_count/16 of these 256-wide macro-tiles --
                    #    far below the ~9/16 fill crossing -- and the s64
                    #    grid (plus its split-K arm for deep K) was measured
                    #    ~1.3-2x faster than this body on every covered
                    #    shape, any n % 64 == 0 raggedness, m ragged or not
                    #    (H100 PCIe: 256x320x64 3.3 us vs 5.5; 256x320x1024
                    #    8.1 vs 15.8; 300x640x512 8.9 vs 10.3; k = 64..4096
                    #    swept).  Without this term those small-fill shapes
                    #    were stolen from the s64 route and lost ~2x.
                    #
                    # 2. The 64x128 v3 small tile, which only exists when it
                    #    can serve the shape at all (m % 64 == 0 and
                    #    n % 128 == 0; an n % 128 == 64 shape beyond s64
                    #    coverage has nowhere better to go than the non-TMA
                    #    wide fallback, so it must engage here): the
                    #    persistent grid runs total_works clusters, and when
                    #    that fills under ~9/16 of the machine the idle SMs
                    #    cost more than this body's multicast + TMA-store
                    #    epilogue saves.  Fitted over a 13-shape
                    #    n % 256 == 128 band on an H100 PCIe (114 SMs),
                    #    k = 512..4096 -- the crossing is K-independent: at
                    #    60 CTAs of fill (768x2432x1024) the v3 small tile
                    #    wins 14.1 us vs 16.0, and its deep-K neighbours
                    #    below the cut (640x2176x4096, 54 CTAs, 32.3 vs
                    #    49.2) agree; at 72 CTAs (704x2944x1024) the
                    #    persistent body wins 16.2 us vs 19.8, likewise at
                    #    k = 4096 (49.9 vs 60.1), and its margin only grows
                    #    with fill (768x4224x1024: 17.7 vs 24.6).
                    var small_route_wins = False
                    if n % _V4_PROD_BN == 0:
                        small_route_wins = (
                            m % 64 == 0 and total_works * 8 < sm_count
                        )
                    else:
                        small_route_wins = _pick_regime(
                            m, n, 1, sm_count
                        ) == 3 or (
                            m % 64 == 0
                            and n % 128 == 0
                            and total_works * _V4_PROD_CLUSTER_M * 16
                            < sm_count * 9
                        )
                    if sm_count >= _V4_PROD_CLUSTER_M and not small_route_wins:
                        if n % _V4_PROD_BN == 0:
                            if has_bias:
                                _v4_enqueue_nn_persistent[
                                    _V4_PROD_STAGES,
                                    _V4_PROD_CLUSTER_M,
                                    _V4_PROD_BM,
                                    _V4_PROD_BN,
                                    _V4_PROD_CONSUMERS,
                                    _V4_PROD_TMA_STORE,
                                    False,
                                    False,
                                    False,
                                    True,
                                ](output, a, b, bias, m, n, k, sm_count, ctx)
                            else:
                                _v4_enqueue_nn_persistent[
                                    _V4_PROD_STAGES,
                                    _V4_PROD_CLUSTER_M,
                                    _V4_PROD_BM,
                                    _V4_PROD_BN,
                                    _V4_PROD_CONSUMERS,
                                    _V4_PROD_TMA_STORE,
                                ](output, a, b, bias, m, n, k, sm_count, ctx)
                        else:
                            if has_bias:
                                _v4_enqueue_nn_persistent[
                                    _V4_PROD_STAGES,
                                    _V4_PROD_CLUSTER_M,
                                    _V4_PROD_BM,
                                    _V4_PROD_BN,
                                    _V4_PROD_CONSUMERS,
                                    _V4_PROD_TMA_STORE,
                                    False,
                                    False,
                                    True,
                                    True,
                                ](output, a, b, bias, m, n, k, sm_count, ctx)
                            else:
                                _v4_enqueue_nn_persistent[
                                    _V4_PROD_STAGES,
                                    _V4_PROD_CLUSTER_M,
                                    _V4_PROD_BM,
                                    _V4_PROD_BN,
                                    _V4_PROD_CONSUMERS,
                                    _V4_PROD_TMA_STORE,
                                    False,
                                    False,
                                    True,
                                ](output, a, b, bias, m, n, k, sm_count, ctx)
                        return True
    return False


def maybe_enqueue_gemm16_tn_v4_persistent[
    kmaj_b: Bool = False, any_wave: Bool = False, ragged_n: Bool = False
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
    """Route a multi-wave TN (wgrad) GEMM -- or, with kmaj_b, a TT one --
    to the persistent clustered v4 body in its col-major-A mode.

    Called by the TN and TT dispatchers in gemm16_tn_v4_kernels.mojo
    AFTER their split-K attempt (deep-K underfilled outputs stay on split-K)
    and BEFORE the remaining one-CTA-per-tile routes.  By default it engages
    only when the 128x256 tiling of the output is strictly multi-wave on the
    current GPU: that is the regime where the one-CTA-per-tile kernels pay a
    per-wave pipeline refill plus a serialized scalar epilogue, and where
    this body's persistent scheduler, cluster B multicast and background
    TMA-store epilogue were measured to win (H100 PCIe; same regime split as
    the NN dispatcher above).  Single-wave TN shapes keep the pre-existing
    narrow-tile / v3 routes, which beat the persistent body there.  The TT
    dispatcher passes any_wave=True because it makes its own wave decision
    (its 128x64 small-tile kernel beats this body on every single-wave
    shape measured; see try_enqueue_gemm16_gemm_tt_v4), and ragged_n=True so
    n % 256 != 0 multi-wave shapes (n % 64 == 0, guaranteed by its gate)
    reach the body's n-clip instantiation instead of falling off to the far
    slower one-CTA-per-tile grid.  The TN dispatcher calls twice: once with
    the defaults (exact n, its pre-existing rung) and -- when n % 256 != 0
    -- once more with ragged_n=True, so multi-wave half-tile-n wgrad shapes
    (GPT-2's padded vocab 50304 has n % 256 == 128) stop falling through to
    the one-CTA-per-tile v3 grid, which loses ~2x there.

    Precondition: m % 128 == 0, k % 64 == 0, and n % 256 == 0 unless
    ragged_n (then n % 64 == 0).  Both callers
    (try_enqueue_gemm16_gemm_tn_v4 / _tt_v4) gate m % 128 == 0 before calling,
    so the kernel body's ragged-m clip path (TMA read clamp + store clip) is
    unreachable and untested on these routes.
    Returns False when the caller must fall back."""
    comptime if not _has_sm_9x():
        return False
    if ctx.api() != "cuda":
        return False
    var cc_major = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
    var cc_minor = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
    if cc_major != 9 or cc_minor != 0:
        return False
    # Aligned TN regime: n and k must tile exactly, and m arrives a multiple
    # of 128 (the caller's gate; see the docstring).  The ordered bounds make
    # all descriptor and address products machine-width safe.
    # OPPORTUNITY (not taken here): the body itself could clip a ragged m,
    # and lifting the caller's m % 128 gate is worth ~4x on ragged-m TN
    # (2900x1280x192 measured 48 us on its fallback route vs 11.3 us for the
    # 128-aligned neighbour).  That is new work needing its own correctness
    # and perf pass on the TN clip path, which is untested in tree today.
    comptime N_MOD = 64 if ragged_n else _V4_PROD_BN
    if (
        m < _V4_PROD_BM * _V4_PROD_CLUSTER_M
        or n < _V4_PROD_BN
        or k < _V4_BK
        or n % N_MOD != 0
        or k % _V4_BK != 0
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
    if sm_count < _V4_PROD_CLUSTER_M:
        return False
    comptime if not any_wave:
        # Strictly multi-wave 128x256 grid only (see docstring).  Under
        # ragged_n the trailing partial 256-column tile is a real work item
        # the persistent scheduler will execute, so the wave census
        # ceil-divides n; the exact-n instantiation keeps its pre-existing
        # floor division (equal when n % 256 == 0).
        var blocks_m = (m + _V4_PROD_BM - 1) // _V4_PROD_BM
        var blocks_n = n // _V4_PROD_BN
        comptime if ragged_n:
            blocks_n = (n + _V4_PROD_BN - 1) // _V4_PROD_BN
        if blocks_m * blocks_n <= sm_count:
            return False
    if has_bias:
        _v4_enqueue_nn_persistent[
            _V4_PROD_STAGES,
            _V4_PROD_CLUSTER_M,
            _V4_PROD_BM,
            _V4_PROD_BN,
            _V4_PROD_CONSUMERS,
            _V4_PROD_TMA_STORE,
            True,
            kmaj_b,
            ragged_n,
            True,
        ](output, a, b, bias, m, n, k, sm_count, ctx)
    else:
        _v4_enqueue_nn_persistent[
            _V4_PROD_STAGES,
            _V4_PROD_CLUSTER_M,
            _V4_PROD_BM,
            _V4_PROD_BN,
            _V4_PROD_CONSUMERS,
            _V4_PROD_TMA_STORE,
            True,
            kmaj_b,
            ragged_n,
        ](output, a, b, bias, m, n, k, sm_count, ctx)
    return True
