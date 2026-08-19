"""Dynamically routed H100 16-bit tensor-core GEMM kernels.

The accepted-v2 implementation remains the fallback for every regime not
handled by the optimized NN, NT, and TN routes in this module.

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

from gemm16_kernels import (
    enqueue_gemm16_bmm as _enqueue_accepted_bf16_bmm,
    enqueue_gemm16_gemm as _enqueue_accepted_bf16_gemm,
)
from gemm16_bmm_v5_kernels import try_enqueue_bmm16_nn_batched
from gemm16_nt_v4_kernels import maybe_enqueue_gemm16_nt_v4
from gemm16_tn_v4_kernels import (
    try_enqueue_gemm16_gemm_nn_v4,
    try_enqueue_gemm16_gemm_splitk_rm_v4,
    try_enqueue_gemm16_gemm_tn_v4,
    try_enqueue_gemm16_gemm_tt_v4,
)
from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG


comptime _V3_DT = _GEMM16_DT
comptime _V3_F32 = DType.float32
comptime _V3_PTR = UnsafePointer[Scalar[_V3_DT], MutAnyOrigin]
comptime _V3_BM = 64
comptime _V3_BN = 128
comptime _V3_BK = 64
comptime _V3_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime _V3_NN_BM = 128
comptime _V3_NN_BN = 256
comptime _V3_NN_BK = 64
comptime _V3_NN_STAGES = 3
comptime _V3_NN_THREADS = 384
comptime _V3_NN_CONSUMERS = 2
comptime _V3_NN_WGMMA_SHAPE = Index(64, 256, 16)
comptime _V3_NN_A_LAYOUT = tile_layout_k_major[
    _V3_DT, _V3_NN_BM, _V3_NN_BK, _V3_SWIZZLE
]()
comptime _V3_NN_B_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_NN_BN, _V3_NN_BK, _V3_SWIZZLE
]()
comptime _V3_NN_A_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NN_BM, _V3_NN_BK),
    Index(_V3_NN_BM, _V3_NN_BK),
]
comptime _V3_NN_B_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NN_BK, _V3_NN_BN),
    Index(_V3_NN_BK, 64),
]
comptime _V3_NN_A_PIPE_LAYOUT = Layout.row_major(
    _V3_NN_STAGES, _V3_NN_BM * _V3_NN_BK
)
comptime _V3_NN_B_PIPE_LAYOUT = Layout.row_major(
    _V3_NN_STAGES, _V3_NN_BN * _V3_NN_BK
)
comptime _V3_NN_SMALL_BM = 64
comptime _V3_NN_SMALL_BN = 128
comptime _V3_NN_SMALL_BK = 64
comptime _V3_NN_SMALL_STAGES = 3
comptime _V3_NN_SMALL_THREADS = 256
comptime _V3_NN_SMALL_CONSUMERS = 1
comptime _V3_NN_SMALL_WGMMA_SHAPE = Index(64, 128, 16)
comptime _V3_NN_SMALL_A_LAYOUT = tile_layout_k_major[
    _V3_DT, _V3_NN_SMALL_BM, _V3_NN_SMALL_BK, _V3_SWIZZLE
]()
comptime _V3_NN_SMALL_B_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_NN_SMALL_BN, _V3_NN_SMALL_BK, _V3_SWIZZLE
]()
comptime _V3_NN_SMALL_A_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NN_SMALL_BM, _V3_NN_SMALL_BK),
    Index(_V3_NN_SMALL_BM, _V3_NN_SMALL_BK),
]
comptime _V3_NN_SMALL_B_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NN_SMALL_BK, _V3_NN_SMALL_BN),
    Index(_V3_NN_SMALL_BK, 64),
]
comptime _V3_NN_SMALL_A_PIPE_LAYOUT = Layout.row_major(
    _V3_NN_SMALL_STAGES, _V3_NN_SMALL_BM * _V3_NN_SMALL_BK
)
comptime _V3_NN_SMALL_B_PIPE_LAYOUT = Layout.row_major(
    _V3_NN_SMALL_STAGES, _V3_NN_SMALL_BN * _V3_NN_SMALL_BK
)
comptime _V3_NT_BM = 128
comptime _V3_NT_BN = 256
comptime _V3_NT_BK = 64
comptime _V3_NT_STAGES = 3
comptime _V3_NT_THREADS = 384
comptime _V3_NT_CONSUMERS = 2
comptime _V3_NT_WGMMA_SHAPE = Index(64, 256, 16)
comptime _V3_NT_A_LAYOUT = tile_layout_k_major[
    _V3_DT, _V3_NT_BM, _V3_NT_BK, _V3_SWIZZLE
]()
comptime _V3_B_K_LAYOUT = tile_layout_k_major[
    _V3_DT, _V3_NT_BN, _V3_NT_BK, _V3_SWIZZLE
]()
comptime _V3_NT_A_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NT_BM, _V3_NT_BK),
    Index(_V3_NT_BM, _V3_NT_BK),
]
comptime _V3_B_K_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_NT_BN, _V3_NT_BK),
    Index(_V3_NT_BN, _V3_NT_BK),
]
comptime _V3_NT_A_PIPE_LAYOUT = Layout.row_major(
    _V3_NT_STAGES, _V3_NT_BM * _V3_NT_BK
)
comptime _V3_NT_B_PIPE_LAYOUT = Layout.row_major(
    _V3_NT_STAGES, _V3_NT_BN * _V3_NT_BK
)
comptime _V3_TN_WS_BM = 128
comptime _V3_TN_WS_BN = 256
comptime _V3_TN_WS_BK = 64
comptime _V3_TN_WS_STAGES = 3
comptime _V3_TN_WS_THREADS = 384
comptime _V3_TN_WS_CONSUMERS = 2
comptime _V3_TN_WS_WGMMA_SHAPE = Index(64, 256, 16)
comptime _V3_TN_WS_A_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_TN_WS_BM, _V3_TN_WS_BK, _V3_SWIZZLE
]()
comptime _V3_TN_WS_B_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_TN_WS_BN, _V3_TN_WS_BK, _V3_SWIZZLE
]()
comptime _V3_TN_WS_A_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_TN_WS_BK, _V3_TN_WS_BM),
    Index(_V3_TN_WS_BK, 64),
]
comptime _V3_TN_WS_B_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_TN_WS_BK, _V3_TN_WS_BN),
    Index(_V3_TN_WS_BK, 64),
]
comptime _V3_TN_WS_A_PIPE_LAYOUT = Layout.row_major(
    _V3_TN_WS_STAGES, _V3_TN_WS_BM * _V3_TN_WS_BK
)
comptime _V3_TN_WS_B_PIPE_LAYOUT = Layout.row_major(
    _V3_TN_WS_STAGES, _V3_TN_WS_BN * _V3_TN_WS_BK
)
comptime _V3_TN_SMALL_BM = 64
comptime _V3_TN_SMALL_BN = 128
comptime _V3_TN_SMALL_BK = 64
comptime _V3_TN_SMALL_STAGES = 3
comptime _V3_TN_SMALL_THREADS = 256
comptime _V3_TN_SMALL_CONSUMERS = 1
comptime _V3_TN_SMALL_WGMMA_SHAPE = Index(64, 128, 16)
comptime _V3_TN_SMALL_A_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_TN_SMALL_BM, _V3_TN_SMALL_BK, _V3_SWIZZLE
]()
comptime _V3_TN_SMALL_B_LAYOUT = tile_layout_mn_major[
    _V3_DT, _V3_TN_SMALL_BN, _V3_TN_SMALL_BK, _V3_SWIZZLE
]()
comptime _V3_TN_SMALL_A_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_TN_SMALL_BK, _V3_TN_SMALL_BM),
    Index(_V3_TN_SMALL_BK, 64),
]
comptime _V3_TN_SMALL_B_TMA = TMATensorTile[
    _V3_DT,
    2,
    Index(_V3_TN_SMALL_BK, _V3_TN_SMALL_BN),
    Index(_V3_TN_SMALL_BK, 64),
]
comptime _V3_TN_SMALL_A_PIPE_LAYOUT = Layout.row_major(
    _V3_TN_SMALL_STAGES, _V3_TN_SMALL_BM * _V3_TN_SMALL_BK
)
comptime _V3_TN_SMALL_B_PIPE_LAYOUT = Layout.row_major(
    _V3_TN_SMALL_STAGES, _V3_TN_SMALL_BN * _V3_TN_SMALL_BK
)


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V3_NN_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_v3_nn_ws_m128n256_tma_s3")
def _v3_nn_ws_m128n256_tma_s3(
    a_tma: _V3_NN_A_TMA,
    b_tma: _V3_NN_B_TMA,
    output: _V3_PTR,
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
        var a_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NN_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NN_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_NN_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_NN_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_NN_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_NN_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        # Publish barrier initialization before consumer warp groups mark the
        # initially empty pipeline slots.  The second barrier publishes those
        # arrivals before the producer starts waiting on them.
        barrier()

        comptime CFRAG = 64 * _V3_NN_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_NN_BM - 1) // _V3_NN_BM
        var blocks_n = (n + _V3_NN_BN - 1) // _V3_NN_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_NN_BM
        var n0 = (rem // rows_in_group) * _V3_NN_BN
        var num_tiles = k // _V3_NN_BK
        comptime TMA_BYTES = (_V3_NN_BM + _V3_NN_BN) * _V3_NN_BK * 2

        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_NN_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_NN_STAGES
                    var phase = UInt32((tile // _V3_NN_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                    var a_tile = LayoutTensor[
                        _V3_DT,
                        _V3_NN_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * _V3_NN_BM * _V3_NN_BK)
                    var b_tile = LayoutTensor[
                        _V3_DT,
                        _V3_NN_B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * _V3_NN_BN * _V3_NN_BK)
                    var k0 = tile * _V3_NN_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (n0, k0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)
            comptime wgmma = TensorCoreAsync[
                _V3_F32,
                _V3_DT,
                _V3_DT,
                _V3_NN_WGMMA_SHAPE,
                a_swizzle=_V3_SWIZZLE,
                b_swizzle=_V3_SWIZZLE,
                transpose_b=False,
            ]()

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_NN_STAGES
                var phase = UInt32((tile // _V3_NN_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_DT,
                    _V3_NN_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_NN_BM * _V3_NN_BK)
                var b_tile = LayoutTensor[
                    _V3_DT,
                    _V3_NN_B_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_NN_BN * _V3_NN_BK)
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma[_V3_NN_CONSUMERS](
                    a_tile, b_tile, accum, warp_group_idx - 1
                )
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_DT, 2](
                    accum.ptr[e].cast[_V3_DT](),
                    accum.ptr[e + 1].cast[_V3_DT](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_nn_ws_m128n256_tma_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NN_BM, _V3_NN_BK),
    )
    var b_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_NN_BK, 64),
    )
    var a_tma = _V3_NN_A_TMA(a_desc)
    var b_tma = _V3_NN_B_TMA(b_desc)
    ctx.enqueue_function[_v3_nn_ws_m128n256_tma_s3](
        a_tma,
        b_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V3_NN_THREADS,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_V3_NN_SMALL_THREADS)
    )
)
@__name(t"{_GEMM16_TAG}_gemm_v3_nn_ws_m64n128_tma_s3")
def _v3_nn_ws_m64n128_tma_s3(
    a_tma: _V3_NN_SMALL_A_TMA,
    b_tma: _V3_NN_SMALL_B_TMA,
    output: _V3_PTR,
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
        var a_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NN_SMALL_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NN_SMALL_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_NN_SMALL_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_NN_SMALL_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_NN_SMALL_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_NN_SMALL_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        barrier()

        comptime CFRAG = 64 * _V3_NN_SMALL_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_NN_SMALL_BM - 1) // _V3_NN_SMALL_BM
        var blocks_n = (n + _V3_NN_SMALL_BN - 1) // _V3_NN_SMALL_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_NN_SMALL_BM
        var n0 = (rem // rows_in_group) * _V3_NN_SMALL_BN
        var num_tiles = k // _V3_NN_SMALL_BK
        comptime TMA_BYTES = (
            _V3_NN_SMALL_BM + _V3_NN_SMALL_BN
        ) * _V3_NN_SMALL_BK * 2

        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_NN_SMALL_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_NN_SMALL_STAGES
                    var phase = UInt32((tile // _V3_NN_SMALL_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                    var a_tile = LayoutTensor[
                        _V3_DT,
                        _V3_NN_SMALL_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](
                        a_pipeline.ptr
                        + stage * _V3_NN_SMALL_BM * _V3_NN_SMALL_BK
                    )
                    var b_tile = LayoutTensor[
                        _V3_DT,
                        _V3_NN_SMALL_B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](
                        b_pipeline.ptr
                        + stage * _V3_NN_SMALL_BN * _V3_NN_SMALL_BK
                    )
                    var k0 = tile * _V3_NN_SMALL_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (n0, k0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)
            comptime wgmma = TensorCoreAsync[
                _V3_F32,
                _V3_DT,
                _V3_DT,
                _V3_NN_SMALL_WGMMA_SHAPE,
                a_swizzle=_V3_SWIZZLE,
                b_swizzle=_V3_SWIZZLE,
                transpose_b=False,
            ]()

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_NN_SMALL_STAGES
                var phase = UInt32((tile // _V3_NN_SMALL_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_DT,
                    _V3_NN_SMALL_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_NN_SMALL_BM * _V3_NN_SMALL_BK)
                var b_tile = LayoutTensor[
                    _V3_DT,
                    _V3_NN_SMALL_B_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_NN_SMALL_BN * _V3_NN_SMALL_BK)
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma[_V3_NN_SMALL_CONSUMERS](
                    a_tile, b_tile, accum, warp_group_idx - 1
                )
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_DT, 2](
                    accum.ptr[e].cast[_V3_DT](),
                    accum.ptr[e + 1].cast[_V3_DT](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_nn_ws_m64n128_tma_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NN_SMALL_BM, _V3_NN_SMALL_BK),
    )
    var b_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_NN_SMALL_BK, 64),
    )
    var a_tma = _V3_NN_SMALL_A_TMA(a_desc)
    var b_tma = _V3_NN_SMALL_B_TMA(b_desc)
    ctx.enqueue_function[_v3_nn_ws_m64n128_tma_s3](
        a_tma,
        b_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V3_NN_SMALL_THREADS,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V3_NT_THREADS))
)
@__name(t"{_GEMM16_TAG}_gemm_v3_nt_ws_m128n256_tma_s3")
def _v3_nt_ws_m128n256_tma_s3(
    a_tma: _V3_NT_A_TMA,
    b_tma: _V3_B_K_TMA,
    output: _V3_PTR,
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
        var a_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NT_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_DT,
            _V3_NT_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_NT_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_NT_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_NT_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_NT_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        barrier()

        comptime CFRAG = 64 * _V3_NT_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_NT_BM - 1) // _V3_NT_BM
        var blocks_n = (n + _V3_NT_BN - 1) // _V3_NT_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_NT_BM
        var n0 = (rem // rows_in_group) * _V3_NT_BN
        var num_tiles = k // _V3_NT_BK
        comptime TMA_BYTES = (_V3_NT_BM + _V3_NT_BN) * _V3_NT_BK * 2

        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_NT_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_NT_STAGES
                    var phase = UInt32((tile // _V3_NT_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))

                    var a_tile = LayoutTensor[
                        _V3_DT,
                        _V3_NT_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * _V3_NT_BM * _V3_NT_BK)
                    var b_tile = LayoutTensor[
                        _V3_DT,
                        _V3_B_K_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * _V3_NT_BN * _V3_NT_BK)
                    var k0 = tile * _V3_NT_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (k0, n0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)
            comptime wgmma = TensorCoreAsync[
                _V3_F32,
                _V3_DT,
                _V3_DT,
                _V3_NT_WGMMA_SHAPE,
                a_swizzle=_V3_SWIZZLE,
                b_swizzle=_V3_SWIZZLE,
                transpose_b=True,
            ]()

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_NT_STAGES
                var phase = UInt32((tile // _V3_NT_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_DT,
                    _V3_NT_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_NT_BM * _V3_NT_BK)
                var b_tile = LayoutTensor[
                    _V3_DT,
                    _V3_B_K_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_NT_BN * _V3_NT_BK)
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma[_V3_NT_CONSUMERS](
                    a_tile, b_tile, accum, warp_group_idx - 1
                )
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_DT, 2](
                    accum.ptr[e].cast[_V3_DT](),
                    accum.ptr[e + 1].cast[_V3_DT](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_nt_ws_m128n256_tma_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NT_BM, _V3_NT_BK),
    )
    var b_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](n, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NT_BN, _V3_NT_BK),
    )
    var a_tma = _V3_NT_A_TMA(a_desc)
    var b_tma = _V3_B_K_TMA(b_desc)
    ctx.enqueue_function[_v3_nt_ws_m128n256_tma_s3](
        a_tma,
        b_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V3_NT_THREADS,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_V3_TN_SMALL_THREADS)
    )
)
@__name(t"{_GEMM16_TAG}_gemm_v3_tn_ws_m64n128_tma_col_a_s3")
def _v3_tn_ws_m64n128_tma_col_a_s3(
    a_tma: _V3_TN_SMALL_A_TMA,
    b_tma: _V3_TN_SMALL_B_TMA,
    output: _V3_PTR,
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
        var a_pipeline = LayoutTensor[
            _V3_DT,
            _V3_TN_SMALL_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_DT,
            _V3_TN_SMALL_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_TN_SMALL_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_TN_SMALL_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_TN_SMALL_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_TN_SMALL_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        # Barrier objects must be initialized before the consumer performs
        # the initial empty-slot arrival.  Without this ordering, sufficiently
        # large grids can expose a cross-warp initialization race.
        barrier()

        comptime CFRAG = 64 * _V3_TN_SMALL_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_TN_SMALL_BM - 1) // _V3_TN_SMALL_BM
        var blocks_n = (n + _V3_TN_SMALL_BN - 1) // _V3_TN_SMALL_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_TN_SMALL_BM
        var n0 = (rem // rows_in_group) * _V3_TN_SMALL_BN
        var num_tiles = k // _V3_TN_SMALL_BK
        comptime TMA_BYTES = (
            _V3_TN_SMALL_BM + _V3_TN_SMALL_BN
        ) * _V3_TN_SMALL_BK * 2

        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_TN_SMALL_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_TN_SMALL_STAGES
                    var phase = UInt32((tile // _V3_TN_SMALL_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                    var a_tile = LayoutTensor[
                        _V3_DT,
                        _V3_TN_SMALL_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](
                        a_pipeline.ptr
                        + stage * _V3_TN_SMALL_BM * _V3_TN_SMALL_BK
                    )
                    var b_tile = LayoutTensor[
                        _V3_DT,
                        _V3_TN_SMALL_B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](
                        b_pipeline.ptr
                        + stage * _V3_TN_SMALL_BN * _V3_TN_SMALL_BK
                    )
                    var k0 = tile * _V3_TN_SMALL_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (m0, k0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (n0, k0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)
            comptime a_canonical_layout = tile_to_descriptor[
                _V3_DT, _V3_TN_SMALL_A_LAYOUT, False
            ]()
            comptime b_canonical_layout = tile_to_descriptor[
                _V3_DT, _V3_TN_SMALL_B_LAYOUT, False
            ]()
            comptime a_stride11 = a_canonical_layout[1].stride[1].value()
            comptime b_stride11 = b_canonical_layout[1].stride[1].value()
            comptime a_k_stride = a_stride11 * 2 * 2
            comptime b_k_stride = b_stride11 * 2 * 2
            comptime NUM_K_MMAS = (
                _V3_TN_SMALL_BK // _V3_TN_SMALL_WGMMA_SHAPE[2]
            )

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_TN_SMALL_STAGES
                var phase = UInt32((tile // _V3_TN_SMALL_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_DT,
                    _V3_TN_SMALL_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_TN_SMALL_BM * _V3_TN_SMALL_BK)
                var b_tile = LayoutTensor[
                    _V3_DT,
                    _V3_TN_SMALL_B_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_TN_SMALL_BN * _V3_TN_SMALL_BK)
                var a_desc = _wgmma_descriptor[
                    a_canonical_layout, False, _V3_SWIZZLE
                ](a_tile.ptr)
                var b_desc = _wgmma_descriptor[
                    b_canonical_layout, False, _V3_SWIZZLE
                ](b_tile.ptr)

                warpgroup_fence(accum)
                wgmma_fence_aligned()
                comptime for k_mma in range(NUM_K_MMAS):
                    var c_tuple = _convert_cfrags_to_tuple[_V3_F32, CFRAG](
                        accum
                    )
                    var c_out = wgmma_async[
                        64,
                        _V3_TN_SMALL_BN,
                        16,
                        a_type=_V3_DT,
                        b_type=_V3_DT,
                        layout_a="col",
                        layout_b="row",
                    ](
                        a_desc + k_mma * a_k_stride,
                        b_desc + k_mma * b_k_stride,
                        c_tuple,
                    )
                    _convert_cfrags_to_simd[_V3_F32, CFRAG](c_out, accum)
                wgmma_commit_group_sync()
                warpgroup_fence(accum)
                wgmma_wait_group_sync()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_DT, 2](
                    accum.ptr[e].cast[_V3_DT](),
                    accum.ptr[e + 1].cast[_V3_DT](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_tn_ws_m64n128_tma_col_a_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, m),
        IndexList[2](m, 1),
        IndexList[2](_V3_TN_SMALL_BK, 64),
    )
    var b_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_TN_SMALL_BK, 64),
    )
    var a_tma = _V3_TN_SMALL_A_TMA(a_desc)
    var b_tma = _V3_TN_SMALL_B_TMA(b_desc)
    ctx.enqueue_function[_v3_tn_ws_m64n128_tma_col_a_s3](
        a_tma,
        b_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V3_TN_SMALL_THREADS,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_V3_TN_WS_THREADS)
    )
)
@__name(t"{_GEMM16_TAG}_gemm_v3_tn_ws_m128n256_tma_col_a_s3")
def _v3_tn_ws_m128n256_tma_col_a_s3(
    a_tma: _V3_TN_WS_A_TMA,
    b_tma: _V3_TN_WS_B_TMA,
    output: _V3_PTR,
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
        # A is physical row-major (K, M).  TMA writes each (K, M) box
        # directly into an MN-major shared layout, which is exactly the
        # column-major A representation accepted by SM90 WGMMA.  This avoids
        # the explicit shared-memory transpose used by the fallback TN path.
        var a_pipeline = LayoutTensor[
            _V3_DT,
            _V3_TN_WS_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_DT,
            _V3_TN_WS_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_TN_WS_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_TN_WS_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_TN_WS_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_TN_WS_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
        # Order barrier initialization before cross-warp-group arrivals.
        barrier()

        comptime CFRAG = 64 * _V3_TN_WS_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_TN_WS_BM - 1) // _V3_TN_WS_BM
        var blocks_n = (n + _V3_TN_WS_BN - 1) // _V3_TN_WS_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_TN_WS_BM
        var n0 = (rem // rows_in_group) * _V3_TN_WS_BN
        var num_tiles = k // _V3_TN_WS_BK
        comptime TMA_BYTES = (_V3_TN_WS_BM + _V3_TN_WS_BN) * _V3_TN_WS_BK * 2

        # Initially release every pipeline slot to the producer.  Thereafter
        # both consumer warp groups arrive only after their WGMMA reads finish.
        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_TN_WS_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_TN_WS_STAGES
                    var phase = UInt32((tile // _V3_TN_WS_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))

                    var a_tile = LayoutTensor[
                        _V3_DT,
                        _V3_TN_WS_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * _V3_TN_WS_BM * _V3_TN_WS_BK)
                    var b_tile = LayoutTensor[
                        _V3_DT,
                        _V3_TN_WS_B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * _V3_TN_WS_BN * _V3_TN_WS_BK)
                    var k0 = tile * _V3_TN_WS_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (m0, k0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (n0, k0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)

            # MN-major descriptors are required for WGMMA's column-major A
            # and row-major B modes.  The second consumer advances by one
            # 64-row WGMMA tile within the shared A tile.
            comptime a_canonical_layout = tile_to_descriptor[
                _V3_DT, _V3_TN_WS_A_LAYOUT, False
            ]()
            comptime b_canonical_layout = tile_to_descriptor[
                _V3_DT, _V3_TN_WS_B_LAYOUT, False
            ]()
            comptime a_shape00 = a_canonical_layout[0].shape[0].value()
            comptime a_stride01 = a_canonical_layout[0].stride[1].value()
            comptime a_stride11 = a_canonical_layout[1].stride[1].value()
            comptime b_stride11 = b_canonical_layout[1].stride[1].value()
            comptime a_m_stride = (
                a_stride01 * (_V3_TN_WS_WGMMA_SHAPE[0] // a_shape00) * 2
            )
            comptime a_k_stride = a_stride11 * 2 * 2
            comptime b_k_stride = b_stride11 * 2 * 2
            comptime NUM_K_MMAS = (_V3_TN_WS_BK // _V3_TN_WS_WGMMA_SHAPE[2])

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_TN_WS_STAGES
                var phase = UInt32((tile // _V3_TN_WS_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_DT,
                    _V3_TN_WS_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_TN_WS_BM * _V3_TN_WS_BK)
                var b_tile = LayoutTensor[
                    _V3_DT,
                    _V3_TN_WS_B_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_TN_WS_BN * _V3_TN_WS_BK)
                var a_desc = _wgmma_descriptor[
                    a_canonical_layout, False, _V3_SWIZZLE
                ](a_tile.ptr)
                var b_desc = _wgmma_descriptor[
                    b_canonical_layout, False, _V3_SWIZZLE
                ](b_tile.ptr)
                a_desc += a_m_stride * (warp_group_idx - 1)

                warpgroup_fence(accum)
                wgmma_fence_aligned()
                comptime for k_mma in range(NUM_K_MMAS):
                    var c_tuple = _convert_cfrags_to_tuple[_V3_F32, CFRAG](
                        accum
                    )
                    var c_out = wgmma_async[
                        64,
                        _V3_TN_WS_BN,
                        16,
                        a_type=_V3_DT,
                        b_type=_V3_DT,
                        layout_a="col",
                        layout_b="row",
                    ](
                        a_desc + k_mma * a_k_stride,
                        b_desc + k_mma * b_k_stride,
                        c_tuple,
                    )
                    _convert_cfrags_to_simd[_V3_F32, CFRAG](c_out, accum)
                wgmma_commit_group_sync()
                warpgroup_fence(accum)
                wgmma_wait_group_sync()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_DT, 2](
                    accum.ptr[e].cast[_V3_DT](),
                    accum.ptr[e + 1].cast[_V3_DT](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_tn_ws_m128n256_tma_col_a_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, m),
        IndexList[2](m, 1),
        IndexList[2](_V3_TN_WS_BK, 64),
    )
    var b_desc = create_tma_descriptor[_V3_DT, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_TN_WS_BK, 64),
    )
    var a_tma = _V3_TN_WS_A_TMA(a_desc)
    var b_tma = _V3_TN_WS_B_TMA(b_desc)
    ctx.enqueue_function[_v3_tn_ws_m128n256_tma_col_a_s3](
        a_tma,
        b_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
        grid_dim=(grid_x,),
        block_dim=(_V3_TN_WS_THREADS,),
    )


# ============================================================================
# Single-matrix TN route, factored out of enqueue_gemm16_gemm's ladder below
# so a batched TN BMM (see enqueue_gemm16_bmm) can loop this exact sequence
# once per batch item instead of falling all the way back to the pre-wgmma
# accepted BMM kernel. Caller guarantees TN (transpose_a, not transpose_b);
# every gate and tile choice here is unchanged from the ladder it was pulled
# out of. Returns True when a route enqueued; False declines with no side
# effect (no partial launch), so the caller's own fallback stays correct.
# `has_bias` may be True: only the split-K rung inside
# try_enqueue_gemm16_gemm_tn_v4 fuses a bias epilogue, and that function
# itself returns False without trying its own non-split rungs whenever
# has_bias is set and split-K declined -- so this wrapper's own
# non-split fallbacks below (which never see has_bias) are only reached
# when has_bias is False, exactly preserving their existing contract.
# ============================================================================
def _try_enqueue_gemm16_tn_route(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    bias: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    # V4 TN (wgrad) route: split-K and narrow-tile kernels for the deep-K,
    # underfilled-output regime (huge K, small m*n; see
    # gemm16_tn_v4_kernels.mojo). The dispatcher checks its own
    # alignment/regime gates and returns False whenever it declines, so
    # every route below remains the fallback.
    if try_enqueue_gemm16_gemm_tn_v4(
        output, a, b, bias, m, n, k, has_bias, ctx
    ):
        return True
    if has_bias:
        return False
    # Underfilled aligned TN regime. A smaller 64x128 output tile exposes
    # four times as many CTAs and uses half the consumer warp groups per
    # CTA. Dispatch is based on severe underfill relative to the current
    # GPU's SM count, not model dimensions.
    if (
        m >= _V3_TN_SMALL_BM
        and n >= _V3_TN_SMALL_BN
        and k >= _V3_TN_SMALL_BK
        and m % _V3_TN_SMALL_BM == 0
        and n % _V3_TN_SMALL_BN == 0
        and k % _V3_TN_SMALL_BK == 0
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
        var large_blocks_m = m // _V3_TN_WS_BM
        var large_blocks_n = n // _V3_TN_WS_BN
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        var use_small_tile = (
            m % _V3_TN_WS_BM != 0
            or n % _V3_TN_WS_BN != 0
            or large_blocks_m * large_blocks_n * 4 < sm_count
        )
        if use_small_tile:
            var blocks_m = m // _V3_TN_SMALL_BM
            var blocks_n = n // _V3_TN_SMALL_BN
            var max_grid_x = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
            if (
                blocks_m > 0
                and blocks_n > 0
                and max_grid_x > 0
                and blocks_m <= max_grid_x // blocks_n
            ):
                var grid_x = blocks_m * blocks_n
                if grid_x > 0:
                    _v3_enqueue_tn_ws_m64n128_tma_col_a_s3(
                        output, a, b, m, n, k, grid_x, ctx
                    )
                    return True
    # Large aligned TN regime. Both operands are physical row-major (k, mn);
    # TMA writes directly into MN-major shared tiles and WGMMA consumes A in
    # its column-major mode.
    if (
        m >= _V3_TN_WS_BM
        and n >= _V3_TN_WS_BN
        and k >= _V3_TN_WS_BK
        and m % _V3_TN_WS_BM == 0
        and n % _V3_TN_WS_BN == 0
        and k % _V3_TN_WS_BK == 0
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
        var blocks_m = m // _V3_TN_WS_BM
        var blocks_n = n // _V3_TN_WS_BN
        var max_grid_x = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
        if (
            blocks_m > 0
            and blocks_n > 0
            and max_grid_x > 0
            and blocks_m <= max_grid_x // blocks_n
        ):
            var grid_x = blocks_m * blocks_n
            if grid_x > 0:
                _v3_enqueue_tn_ws_m128n256_tma_col_a_s3(
                    output, a, b, m, n, k, grid_x, ctx
                )
                return True
    return False


def enqueue_gemm16_gemm(
    output: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    a: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    bias: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    # NT (forward linear) route: persistent clustered v4 kernel with TMA
    # multicast and TMA-store epilogue (gemm16_nt_v4_kernels.mojo).  The
    # helper enqueues only for regimes it fully supports (SM90, aligned
    # n/k, TMA-compatible sizes) and returns False otherwise, in which case
    # the pre-existing NT path below remains the fallback.
    if not transpose_a and transpose_b:
        # Deep-K split-K route first: the persistent kernel below keeps all
        # SMs resident but cannot parallelize over K, so an output with few
        # macro-tiles and a deep reduction leaves most of the GPU idle.
        # The helper gates itself on that regime (see
        # gemm16_tn_v4_kernels.mojo) and returns False otherwise. It also
        # fuses a bias epilogue into its reduce kernel (see
        # _v4_tn_splitk_reduce), so it is tried even when has_bias is set --
        # the persistent kernel just below never supports bias.
        if try_enqueue_gemm16_gemm_splitk_rm_v4[True](
            output, a, b, bias, m, n, k, has_bias, ctx
        ):
            return
        if not has_bias:
            if maybe_enqueue_gemm16_nt_v4(output, a, b, m, n, k, ctx):
                return
    comptime if _has_sm_9x():
        if ctx.api() == "cuda":
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if cc_major == 9 and cc_minor == 0:
                # NN dgrad route (v4): persistent warp-specialized kernel
                # with 2-CTA-cluster B multicast and a TMA-store epilogue
                # (gemm16_nn_v4_kernels.mojo).  It gates itself on the
                # aligned NN regime (n % 64 == 0, k % 64 == 0; m may be
                # ragged, and n % 256 != 0 selects the ragged_n _nclip
                # instantiation) and declines only shapes a smaller route
                # below serves better: the 64x128 small tile (m % 64 == 0,
                # n % 128 == 0 and its whole grid fits one wave) or, for
                # ragged n, the bottom-of-ladder 64x64 s64 route whenever
                # _pick_regime would select it; it returns False for
                # those, in which case the v3 NN paths below remain the
                # fallback.
                # Deep-K split-K route for NN: checked before the
                # persistent kernel because a persistent CTA serializes its
                # tiles' whole K depth -- few output macro-tiles plus deep K
                # leaves most SMs idle.  The helper gates itself on that
                # regime, and fuses bias into its reduce epilogue, so it is
                # tried even when has_bias is set (the persistent kernel
                # below still declines outright on bias).
                if not transpose_a and not transpose_b:
                    if try_enqueue_gemm16_gemm_splitk_rm_v4[False](
                        output, a, b, bias, m, n, k, has_bias, ctx
                    ):
                        return
                # NN (dgrad) route: persistent 128x256 kernel, or -- for
                # the 2-wave underfilled regime -- the direct 128x192 tile
                # the wave-fill cost model picks instead (see
                # try_enqueue_gemm16_gemm_nn_v4, gemm16_tn_v4_kernels.mojo).
                if not transpose_a and not transpose_b:
                    if try_enqueue_gemm16_gemm_nn_v4(
                        output, a, b, bias, m, n, k, has_bias, ctx
                    ):
                        return
                # TN (wgrad) route ladder: split-K/narrow-tile v4, then the
                # underfilled small-tile and large-tile v3 wgmma kernels
                # (gemm16_tn_v4_kernels.mojo, this file). Factored into
                # _try_enqueue_gemm16_tn_route so enqueue_gemm16_bmm below can
                # loop the identical single-matrix ladder once per batch item.
                # Bias-carrying calls are tried too: the wrapper's split-K
                # rung fuses bias and declines cleanly (no non-split
                # fallback) when it cannot engage.
                if transpose_a and not transpose_b:
                    if _try_enqueue_gemm16_tn_route(
                        output, a, b, bias, m, n, k, has_bias, ctx
                    ):
                        return
                # V4 TT route: the (COL_A, KMAJ_B) = (True, True)
                # instantiations of the shared warp-specialized body
                # (split-K, persistent, direct; see
                # gemm16_tn_v4_kernels.mojo), so a TT mm writes C
                # directly into the caller's contiguous row-major (m, n)
                # buffer -- the same strides CUDA torch returns.  The
                # dispatcher gates its own aligned regime and returns False
                # otherwise, in which case the all-layout v2 fallback below
                # serves the call. Bias-carrying calls are tried too: the
                # dispatcher's split-K rung fuses bias and declines cleanly
                # (no non-split fallback) when it cannot engage.
                if transpose_a and transpose_b:
                    if try_enqueue_gemm16_gemm_tt_v4(
                        output, a, b, bias, m, n, k, has_bias, ctx
                    ):
                        return
                # A 64x128 tile preserves the prior aligned-NN coverage and
                # increases available CTAs when the 128x256 grid would be
                # severely underfilled on the current GPU.  This predicate is
                # shape-regime based; no model dimension is hardcoded.
                if (
                    not transpose_a
                    and not transpose_b
                    and not has_bias
                    and m >= _V3_NN_SMALL_BM
                    and n >= _V3_NN_SMALL_BN
                    and k >= _V3_NN_SMALL_BK
                    and m % _V3_NN_SMALL_BM == 0
                    and n % _V3_NN_SMALL_BN == 0
                    and k % _V3_NN_SMALL_BK == 0
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
                    var large_blocks_m = m // _V3_NN_BM
                    var large_blocks_n = n // _V3_NN_BN
                    var sm_count = ctx.get_attribute(
                        DeviceAttribute.MULTIPROCESSOR_COUNT
                    )
                    var use_small_tile = (
                        m % _V3_NN_BM != 0
                        or n % _V3_NN_BN != 0
                        or large_blocks_m * large_blocks_n * 4 < sm_count
                    )
                    if use_small_tile:
                        var blocks_m = m // _V3_NN_SMALL_BM
                        var blocks_n = n // _V3_NN_SMALL_BN
                        var max_grid_x = ctx.get_attribute(
                            DeviceAttribute.MAX_GRID_DIM_X
                        )
                        if (
                            blocks_m > 0
                            and blocks_n > 0
                            and max_grid_x > 0
                            and blocks_m <= max_grid_x // blocks_n
                        ):
                            var grid_x = blocks_m * blocks_n
                            if grid_x > 0:
                                _v3_enqueue_nn_ws_m64n128_tma_s3(
                                    output, a, b, m, n, k, grid_x, ctx
                                )
                                return
                # Full-tile NN regime only. The ordered bounds make all
                # descriptor and address products machine-width safe; TMA
                # receives 16B-aligned bases and 128B-compatible rows.
                if (
                    not transpose_a
                    and not transpose_b
                    and not has_bias
                    and m >= _V3_NN_BM
                    and n >= _V3_NN_BN
                    and k >= _V3_NN_BK
                    and m % _V3_NN_BM == 0
                    and n % _V3_NN_BN == 0
                    and k % _V3_NN_BK == 0
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
                    var blocks_m = m // _V3_NN_BM
                    var blocks_n = n // _V3_NN_BN
                    var max_grid_x = ctx.get_attribute(
                        DeviceAttribute.MAX_GRID_DIM_X
                    )
                    if (
                        blocks_m > 0
                        and blocks_n > 0
                        and max_grid_x > 0
                        and blocks_m <= max_grid_x // blocks_n
                    ):
                        var grid_x = blocks_m * blocks_n
                        if grid_x > 0:
                            _v3_enqueue_nn_ws_m128n256_tma_s3(
                                output, a, b, m, n, k, grid_x, ctx
                            )
                            return
                # Full-tile NT regime. B is physical row-major (n, k); both
                # operands use 128B-swizzled k-major shared layouts.
                #
                # Unlike every other route here, this one admits at
                # _V3_BM/_V3_BN (64x128) while the kernel tiles at
                # _V3_NT_BM/_V3_NT_BN (128x256), so shapes like m=192 or
                # n=128 are accepted and produce partial edge tiles. That is
                # why the grid below is a ceil-div rather than an exact
                # division, and why the epilogue bounds check in
                # _v3_nt_ws_m128n256_tma_s3 is load-bearing: it is the only
                # thing keeping an edge block inside the output. The sibling
                # routes admit at their own tile with exact division, so
                # their identical-looking checks are unreachable. Do not
                # unify the five epilogues by dropping the check.
                if (
                    not transpose_a
                    and transpose_b
                    and not has_bias
                    and m >= _V3_BM
                    and n >= _V3_BN
                    and k >= _V3_BK
                    and m % _V3_BM == 0
                    and n % _V3_BN == 0
                    and k % _V3_BK == 0
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
                    var blocks_m = (m + _V3_NT_BM - 1) // _V3_NT_BM
                    var blocks_n = (n + _V3_NT_BN - 1) // _V3_NT_BN
                    var max_grid_x = ctx.get_attribute(
                        DeviceAttribute.MAX_GRID_DIM_X
                    )
                    if (
                        blocks_m > 0
                        and blocks_n > 0
                        and max_grid_x > 0
                        and blocks_m <= max_grid_x // blocks_n
                    ):
                        var grid_x = blocks_m * blocks_n
                        if grid_x > 0:
                            _v3_enqueue_nt_ws_m128n256_tma_s3(
                                output, a, b, m, n, k, grid_x, ctx
                            )
                            return
                # The remaining TN regimes (underfilled small-tile and large
                # aligned) are tried above via _try_enqueue_gemm16_tn_route,
                # right after the v4 split-K/narrow-tile route: same gates,
                # same tile choices, same order, just shared with the batched
                # BMM loop below instead of duplicated here.
    _enqueue_accepted_bf16_gemm(
        output,
        a,
        b,
        bias,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        has_bias,
        ctx,
    )


def enqueue_gemm16_bmm(
    output: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    a: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[_V3_DT], MutAnyOrigin],
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    output_batch_stride: Int,
    a_batch_stride: Int,
    b_batch_stride: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    # TN strided-batch loop over the mm-level wgmma ladder. There is no
    # batched (grid.z) wgmma TN kernel: every BMM layout, including TN,
    # reaches only the pre-wgmma accepted kernel below today, and TN is the
    # worst-served layout there (~1.7x the NN/NT/TT siblings sharing that
    # same kernel, ~6-7x cuBLAS overall) because that kernel predates the
    # tensor-core-async / TMA work the mm-level TN route already has.
    #
    # This loops _try_enqueue_gemm16_tn_route -- the exact single-matrix TN
    # ladder `enqueue_gemm16_gemm` uses -- once per batch item, the same way
    # a caller doing `batch_count` independent `torch.mm` calls would. Every
    # gate that ladder checks (alignment, m/n/k divisibility, SM-count-based
    # tile and split-K choice) depends only on (m, n, k, sm_count) and
    # operand alignment, which are identical across batch items by
    # construction (one m/n/k triple and one batch stride for the whole
    # call) -- so batch item 0's routing decision predicts every other
    # item's, and this only commits to the loop once batch 0 actually fires.
    # Declining (e.g. an odd shape, or a small per-matrix problem the v3/v4
    # routes were never tuned to fill a wave with alone -- attention-shaped
    # BMM, many small matrices, is exactly the case the grid.z-packed
    # kernel below is tuned for) falls through to today's unchanged path.
    # The per-batch retry on a False is a defensive correctness net, not an
    # expected path: same-shape batches only diverge if a batch stride
    # breaks 16B alignment partway through.
    #
    # NN batched route: unlike TN above, this is a single launch over the
    # WHOLE batch (gemm16_bmm_v5_kernels.mojo) rather than a per-item loop --
    # a persistent work list spanning every batch item's tiles, addressed by
    # rank-3 TMA descriptors that carry the batch dimension (and clip ragged
    # m/n/k per item). Looping the mm-level NN ladder the way TN does was
    # tried and measured worse on every shape (small per-item grids -- e.g.
    # attention's batch of narrow-n matrices, or conv's im2col batch --
    # underfill the GPU independently on each loop iteration); see
    # gemm16_bmm_v5_kernels.mojo's module docstring. `a_bs == 0` (a
    # broadcast/expanded A, e.g. conv's shared per-sample weight matrix)
    # is a first-class input here, not a special case: the caller passes it
    # straight through, unlike `_tf32_dense_batched_layout` on the aten_fast
    # side which used to reject it (see that function's history).
    if not transpose_a and not transpose_b:
        if try_enqueue_bmm16_nn_batched(
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            output_batch_stride,
            a_batch_stride,
            b_batch_stride,
            ctx,
        ):
            return
    if transpose_a and not transpose_b:
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
                    and batch_count > 0
                    and _try_enqueue_gemm16_tn_route(
                        output, a, b, output, m, n, k, False, ctx
                    )
                ):
                    for bidx in range(1, batch_count):
                        var oo = output + bidx * output_batch_stride
                        var ao = a + bidx * a_batch_stride
                        var bo = b + bidx * b_batch_stride
                        if not _try_enqueue_gemm16_tn_route(
                            oo, ao, bo, oo, m, n, k, False, ctx
                        ):
                            _enqueue_accepted_bf16_gemm(
                                oo, ao, bo, oo, m, n, k, True, False, False, ctx
                            )
                    return
    _enqueue_accepted_bf16_bmm(
        output,
        a,
        b,
        batch_count,
        m,
        n,
        k,
        output_batch_stride,
        a_batch_stride,
        b_batch_stride,
        transpose_a,
        transpose_b,
        ctx,
    )
