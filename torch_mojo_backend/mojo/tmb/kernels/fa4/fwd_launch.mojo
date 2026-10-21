"""Launch helper for the FA4-target fwd kernel.

Scope (v1): bf16, head_dim=128, non-causal, contiguous (B, L, H, D),
seqlen % BN == 0, Hq == Hk.

PTX dump: `-D MOJO_DUMP_PTX=<path>` (launch_cache.dump_ptx_path).
"""

from max.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from max.gpu.host.device_context import (
    _DeviceContextPtr,
    _DeviceContextCpp,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of
from std.utils.index import IndexList

from layout import UNKNOWN_VALUE
from layout.tma_async import SplitLastDimTMATensorTile, create_split_tma

from tmb.kernels.fa4.tma4 import (
    create_split_tma_3d_strided,
    create_split_tma_4d,
    create_split_tma_4d_strided,
)

from tmb.kernels.fa4.fwd_kernel import fwd_fa4_kernel
from tmb.kernels.fa4.fwd_common import (
    kFa4NThreads,
    kFa4BlockM,
    kFa4BlockN,
    kFa4KVStages,
)
from tmb.kernels.fa4.launch_cache import dump_ptx_path, enqueue_fa4_cached


def launch_fwd_fa4[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
    window: Bool = False,
    window_unaligned: Bool = False,
    softcap_x1000: Int = 0,
    strided_qkv: Bool = False,
    bhsd_qkv: Bool = False,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    o_addr: Int,
    lse_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
    varlen_total_q: Int = 0,
    varlen_total_k: Int = 0,
    varlen_table_addr: Int = 0,
    varlen_num_tiles: Int = 0,
    window_left: Int = 0,
    q_b_stride: Int = 0,
    q_s_stride: Int = 0,
    q_h_stride: Int = 0,
    q_d_stride: Int = 1,
    k_s_stride: Int = 0,
    k_h_stride: Int = 0,
    k_d_stride: Int = 1,
    v_s_stride: Int = 0,
    v_h_stride: Int = 0,
    v_d_stride: Int = 1,
) raises:
    # Strided Q/K/V (zero-copy fused-QKV views) is scoped to the
    # dense causal path in this port (d64 or d128); the caller
    # validates the layout contract (d_stride==1, b_stride==S*s_stride,
    # 16 B-aligned strides, h_stride==head_dim) before this launcher
    # runs.
    comptime assert (not strided_qkv) or (
        (head_dim == 64 or head_dim == 128)
        and causal
        and not varlen
        and not window
        and gqa_ratio == 1
        and softcap_x1000 == 0
    ), "strided_qkv supports only dense d64/d128 (no varlen/window/gqa/softcap)"
    var raw_ctx_ptr = Pointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem: Q (BM x D) + kFa4KVStages ring slots (BN x D) bf16 +
    # mbarriers. head_dim=128: 32 KiB per tile -> 224 KiB (H100
    # opt-in cap is 227 KiB).
    comptime q_bytes: Int = (kFa4BlockM(head_dim) * head_dim * size_of[dtype]())
    comptime kv_slot_bytes: Int = kFa4BlockN * head_dim * size_of[dtype]()
    comptime mbar_bytes: Int = 128
    comptime smem_bytes: Int = (
        q_bytes + kFa4KVStages * kv_slot_bytes + mbar_bytes
    )

    var q_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var lse_ptr = Pointer[Float32, MutAnyOrigin](unsafe_from_address=lse_addr)
    # 3D TMA descriptors over the (B*L, H, D) gmem view.
    comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim)
    comptime q_smem_shape = IndexList[3](kFa4BlockM(head_dim), 1, head_dim)
    comptime kv_smem_shape = IndexList[3](kFa4BlockN, 1, head_dim)

    comptime if bhsd_qkv:
        comptime assert (
            (head_dim == 64 or head_dim == 128)
            and causal
            and not varlen
            and not window
            and softcap_x1000 == 0
            and not strided_qkv
        ), "bhsd_qkv v1 supports only the dense causal d64/d128 path"
        # PUBLIC-layout consumption (the fix for the 30-45% gather-copy
        # overhead): Q/K/V/O are the contiguous (B, H, S, D) tensors the
        # ATen op provides, viewed as (B*H, S, D). Planes are the TMA
        # outer dim and S its OWN dim, so BM=192 tail tiles zero-fill
        # on load and clamp on store at every plane's S edge — no BTHD
        # materialization, no next-plane clobber. This is FA3's scheme
        # (head stride is just another gmem stride).
        comptime q_smem_shape_b = IndexList[3](
            1, kFa4BlockM(head_dim), head_dim
        )
        comptime kv_smem_shape_b = IndexList[3](1, kFa4BlockN, head_dim)
        var planes_q: Int = batch_int * nheads_int
        var planes_kv: Int = batch_int * (nheads_int // gqa_ratio)
        var q_tma_b = create_split_tma[
            q_smem_shape_b, gmem_shape, swizzle_mode=swizzle
        ](ctx, q_ptr, planes_q, seqlen_int)
        var k_tma_b = create_split_tma[
            kv_smem_shape_b, gmem_shape, swizzle_mode=swizzle
        ](ctx, k_ptr, planes_kv, seqlen_int)
        var v_tma_b = create_split_tma[
            kv_smem_shape_b, gmem_shape, swizzle_mode=swizzle
        ](ctx, v_ptr, planes_kv, seqlen_int)
        var o_imm_ptr_b = Pointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=o_addr
        )
        var o_tma_b = create_split_tma[
            q_smem_shape_b, gmem_shape, swizzle_mode=swizzle
        ](ctx, o_imm_ptr_b, planes_q, seqlen_int)
        comptime kernel_inst_b = fwd_fa4_kernel[
            dtype,
            head_dim,
            3,
            type_of(q_tma_b).tile_shape,
            type_of(q_tma_b).desc_shape,
            type_of(k_tma_b).tile_shape,
            type_of(k_tma_b).desc_shape,
            type_of(o_tma_b).tile_shape,
            type_of(o_tma_b).desc_shape,
            causal,
            gqa_ratio,
            False,
            False,
            False,
            0,
            bhsd=True,
        ]
        # Same LPT scheduler as the dense causal rank-4 path.
        var num_m_b: Int = ceildiv(seqlen_int, Int(kFa4BlockM(head_dim)))
        var size_one_kv_head_b: Int = (
            seqlen_int * 2 * head_dim * size_of[dtype]()
        )
        # Work order, wave-dispatched (measured on d64; middle group
        # sizes lose BOTH ways — see the phase-2 harness NOTES.md):
        # - MANY waves (>= ~12): swizzle group 1 — each plane's whole
        #   m-sweep is consecutive, so the CTAs in flight walk a few
        #   planes' K/V phase-aligned (L2-resident, many readers per
        #   tile): P1 -8%, P3 -3%. The 50MB-budget order (128 planes
        #   at the SAME m in flight) shares nothing concurrently:
        #   L2 hit 50%, 1.7x DRAM re-reads, long_scoreboard top stall.
        # - FEW waves: keep the global-LPT order (all planes sweep m
        #   together, heavy tiles first) — the tail is all light
        #   tiles, which dominates short grids (group 1 here cost P0
        #   8% in stragglers).
        var sm_count_b: Int = ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var ctas_per_sm_b: Int = 2 if head_dim == 64 else 1
        var waves_b: Int = (num_m_b * nheads_int * batch_int) // (
            ctas_per_sm_b * sm_count_b
        )
        var l2_ratio_b: Int = (50 * 1024 * 1024) // size_one_kv_head_b
        var sched_swizzle_b: Int
        if waves_b >= 12:
            sched_swizzle_b = 1
        else:
            sched_swizzle_b = 1
            while sched_swizzle_b * 2 <= l2_ratio_b:
                sched_swizzle_b *= 2
        var num_hb_b: Int = nheads_int * batch_int
        var sched_num_hb_q_b: Int = num_hb_b // sched_swizzle_b
        var sched_residual_b: Int = max(num_hb_b % sched_swizzle_b, 1)
        enqueue_fa4_cached[
            kernel_inst_b,
            use_external_stream=use_external_stream,
            dump_asm=dump_ptx_path(),
        ](
            ctx,
            ctx_handle_addr,
            stream_opaque,
            String(t"fwd_bhsd_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"),
            (num_m_b * num_hb_b, 1, 1),
            kFa4NThreads(head_dim),
            smem_bytes,
            FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
            q_tma_b,
            k_tma_b,
            v_tma_b,
            o_tma_b,
            lse_ptr,
            Int64(seqlen_int),
            softmax_scale,
            Int64(nheads_int),
            Int64(sched_swizzle_b),
            Int64(sched_num_hb_q_b),
            Int64(sched_residual_b),
        )
        return

    # Varlen: one flat descriptor over the packed (total_tokens, H, D)
    # arrays; the kernel supplies per-sequence row offsets at copy
    # time (FA4's scheme — no per-seq descriptor rebuilds).
    var rows: Int = batch_int * seqlen_int
    var rows_kv: Int = rows
    comptime if varlen:
        rows = varlen_total_q
        rows_kv = varlen_total_k
    var nheads_kv: Int = nheads_int // gqa_ratio
    var k_tma: SplitLastDimTMATensorTile[dtype, kv_smem_shape, swizzle]
    var v_tma: SplitLastDimTMATensorTile[dtype, kv_smem_shape, swizzle]
    comptime if strided_qkv:
        # Runtime-strided flat (B*S, H, D) rows: valid because the
        # caller checked b_stride == seqlen * s_stride, so row
        # r = b*S + s addresses r * s_stride elements uniformly
        # across batches.
        k_tma = create_split_tma_3d_strided[
            kv_smem_shape, swizzle_mode=swizzle
        ](
            ctx,
            k_ptr,
            rows_kv,
            nheads_kv,
            head_dim,
            k_s_stride,
            k_h_stride,
            k_d_stride,
        )
        v_tma = create_split_tma_3d_strided[
            kv_smem_shape, swizzle_mode=swizzle
        ](
            ctx,
            v_ptr,
            rows_kv,
            nheads_kv,
            head_dim,
            v_s_stride,
            v_h_stride,
            v_d_stride,
        )
    else:
        k_tma = create_split_tma[
            kv_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, k_ptr, rows_kv, nheads_kv)
        v_tma = create_split_tma[
            kv_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, v_ptr, rows_kv, nheads_kv)
    # O store descriptor: SWIZZLE_128B like the loads — the kernel
    # stages O into the dead (swizzled) Q tile and issues ONE
    # whole-tile TMA store. (The previous unswizzled 16B-chunk
    # descriptor cost 16 serialized UTMASTG issues per CTA — ~8% of
    # a short-seq CTA, PC-sampling-verified.)
    var o_imm_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    # Dense hdim64 gives Q/O RANK-4 descriptors (B, S, H, D): with
    # BM=192 not dividing the seqlen envelope, S must be its own TMA
    # dim so tail-tile loads zero-fill and stores CLAMP (a flattened
    # B*S descriptor would clobber the next batch's O rows).
    comptime QO_RANK: Int = 4 if (head_dim == 64 and not varlen) else 3
    comptime gmem_shape4 = IndexList[4](
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim
    )
    comptime q_smem_shape4 = IndexList[4](1, kFa4BlockM(head_dim), 1, head_dim)

    comptime if QO_RANK == 4:
        # Q keeps the rank-4 (B, S, H, D) form in BOTH modes so the
        # BM=192 tail-tile clamp still stops loads at the batch edge;
        # strided mode only swaps row-major strides for the caller's
        # runtime element strides. O stays contiguous rank-4.
        var q_tma: SplitLastDimTMATensorTile[dtype, q_smem_shape4, swizzle]
        comptime if strided_qkv:
            q_tma = create_split_tma_4d_strided[
                q_smem_shape4, swizzle_mode=swizzle
            ](
                ctx,
                q_ptr,
                batch_int,
                seqlen_int,
                nheads_int,
                head_dim,
                q_b_stride,
                q_s_stride,
                q_h_stride,
                q_d_stride,
            )
        else:
            q_tma = create_split_tma_4d[
                q_smem_shape4, gmem_shape4, swizzle_mode=swizzle
            ](ctx, q_ptr, batch_int, seqlen_int, nheads_int)
        var o_tma = create_split_tma_4d[
            q_smem_shape4, gmem_shape4, swizzle_mode=swizzle
        ](ctx, o_imm_ptr, batch_int, seqlen_int, nheads_int)
        comptime kernel_inst = fwd_fa4_kernel[
            dtype,
            head_dim,
            4,
            type_of(q_tma).tile_shape,
            type_of(q_tma).desc_shape,
            type_of(k_tma).tile_shape,
            type_of(k_tma).desc_shape,
            type_of(o_tma).tile_shape,
            type_of(o_tma).desc_shape,
            causal,
            gqa_ratio,
            varlen,
            window,
            window_unaligned,
            softcap_x1000,
        ]
        comptime assert not window, "window v1 is hdim128-only"
        comptime assert softcap_x1000 == 0, "softcap v1 is hdim128-only"
        var num_m: Int = ceildiv(seqlen_int, Int(kFa4BlockM(head_dim)))
        var size_one_kv_head: Int = seqlen_int * 2 * head_dim * size_of[dtype]()
        var l2_ratio: Int = (50 * 1024 * 1024) // size_one_kv_head
        var sched_swizzle: Int = 1
        while sched_swizzle * 2 <= l2_ratio:
            sched_swizzle *= 2
        var num_hb: Int = nheads_int * batch_int
        var sched_num_hb_q: Int = num_hb // sched_swizzle
        var sched_residual: Int = max(num_hb % sched_swizzle, 1)
        var grid: Tuple[Int, Int, Int]
        comptime if causal:
            grid = (num_m * num_hb, 1, 1)
        else:
            grid = (num_m, nheads_int, batch_int)
        enqueue_fa4_cached[
            kernel_inst,
            use_external_stream=use_external_stream,
            dump_asm=dump_ptx_path(),
        ](
            ctx,
            ctx_handle_addr,
            stream_opaque,
            String(
                t"fwd_r4_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"
                t"_vl{varlen}_w{window}_wu{window_unaligned}"
                t"_sc{softcap_x1000}"
            ),
            grid,
            kFa4NThreads(head_dim),
            smem_bytes,
            FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
            q_tma,
            k_tma,
            v_tma,
            o_tma,
            lse_ptr,
            Int64(seqlen_int),
            softmax_scale,
            Int64(nheads_int),
            Int64(sched_swizzle),
            Int64(sched_num_hb_q),
            Int64(sched_residual),
        )
        return

    # QO_RANK == 3 (head_dim != 64, i.e. the d128 dense/strided path):
    # BM == BN == 128 here, so the seqlen % 128 == 0 eligibility gate
    # already makes every m-tile a full tile -- no rank-4 tail
    # descriptor is needed the way d64's BM=192 requires. Q mirrors
    # K/V's existing strided-vs-plain choice above; O always stays
    # the dense contiguous descriptor (strided_qkv only broadens Q/K/V).
    var q_tma: SplitLastDimTMATensorTile[dtype, q_smem_shape, swizzle]
    comptime if strided_qkv:
        q_tma = create_split_tma_3d_strided[q_smem_shape, swizzle_mode=swizzle](
            ctx,
            q_ptr,
            rows,
            nheads_int,
            head_dim,
            q_s_stride,
            q_h_stride,
            q_d_stride,
        )
    else:
        q_tma = create_split_tma[
            q_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, q_ptr, rows, nheads_int)
    var o_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, o_imm_ptr, rows, nheads_int)

    comptime kernel_inst = fwd_fa4_kernel[
        dtype,
        head_dim,
        3,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
        type_of(o_tma).tile_shape,
        type_of(o_tma).desc_shape,
        causal,
        gqa_ratio,
        varlen,
        window,
        window_unaligned,
        softcap_x1000,
    ]

    # Scheduler params (used by the causal LPT decode; harmless
    # otherwise). L2 swizzle: how many (head, batch) pairs share one
    # m_block sweep so their K+V tiles stay L2-resident (FA4's
    # SingleTileLPTScheduler with a 50 MiB L2 budget).
    var num_m: Int = ceildiv(seqlen_int, Int(kFa4BlockM(head_dim)))
    var size_one_kv_head: Int = seqlen_int * 2 * head_dim * size_of[dtype]()
    var l2_ratio: Int = (50 * 1024 * 1024) // size_one_kv_head
    var sched_swizzle: Int = 1
    while sched_swizzle * 2 <= l2_ratio:
        sched_swizzle *= 2
    var num_hb: Int = nheads_int * batch_int
    var sched_num_hb_q: Int = num_hb // sched_swizzle
    var sched_residual: Int = max(num_hb % sched_swizzle, 1)

    var grid: Tuple[Int, Int, Int]
    comptime if varlen:
        grid = (varlen_num_tiles, nheads_int, 1)
    else:
        comptime if causal and not window:
            grid = (num_m * num_hb, 1, 1)
        else:
            # Window: per-m work is uniform — plain grid, no LPT.
            grid = (num_m, nheads_int, batch_int)

    # Varlen reuses three kernel arg slots (signature unchanged vs
    # dense): seq_len carries total_q (packed LSE layout),
    # sched_swizzle carries the work-table address, and
    # sched_num_hb_q carries the raw O pointer (for the row-predicated
    # ragged-tail stores; the LPT scheduler is dense-causal-only).
    var seq_len_arg: Int = seqlen_int
    var sched_swizzle_arg: Int = sched_swizzle
    var sched_num_hb_q_arg: Int = sched_num_hb_q
    comptime if varlen:
        seq_len_arg = varlen_total_q
        sched_swizzle_arg = varlen_table_addr
        sched_num_hb_q_arg = o_addr
    comptime if window and not varlen:
        sched_swizzle_arg = window_left  # the LPT slot is free
        # (varlen keeps the table in this slot; win_left rides the
        # table's col 5 instead.)

    enqueue_fa4_cached[
        kernel_inst,
        use_external_stream=use_external_stream,
        dump_asm=dump_ptx_path(),
    ](
        ctx,
        ctx_handle_addr,
        stream_opaque,
        String(
            t"fwd_r3_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"
            t"_vl{varlen}_w{window}_wu{window_unaligned}"
            t"_sc{softcap_x1000}"
        ),
        grid,
        kFa4NThreads(head_dim),
        smem_bytes,
        FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
        q_tma,
        k_tma,
        v_tma,
        o_tma,
        lse_ptr,
        Int64(seq_len_arg),
        softmax_scale,
        Int64(nheads_int),
        Int64(sched_swizzle_arg),
        Int64(sched_num_hb_q_arg),
        Int64(sched_residual),
    )
