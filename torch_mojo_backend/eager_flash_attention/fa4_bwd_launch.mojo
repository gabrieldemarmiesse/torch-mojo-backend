"""Launch helpers for the FA4-target bwd kernels (preprocess, main,
convert). Scope (v1): bf16, head_dim=128, non-causal, contiguous
(B, L, H, D), seqlen % 128 == 0, Hq == Hk.

Side tensors (allocated by the Python wrapper, padded to
Spad = ceil(S / kBwdBlockM) * kBwdBlockM):
  dpsum, lse_log2: (B, H, Spad) fp32 contiguous (+inf/0 pad rows).
  dq_accum:        B*H*Spad*D f32 fragment dump (zeroed by preprocess).

PTX dump: `-D MOJO_DUMP_PTX=<path>` dumps the *main* kernel's PTX.
"""

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp, _DumpPath
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import get_defined_string, size_of
from std.utils.index import IndexList

from layout import UNKNOWN_VALUE
from layout.tma_async import SplitLastDimTMATensorTile, create_split_tma

from fa4_tma4 import create_split_tma_3d_strided

from fa4_bwd_kernel import bwd_main_kernel, bwd_preprocess_kernel, bwd_convert_kernel
from fa4_bwd_common import (
    kBwdTileM,
    kBwdBlockN,
    kBwdCvtThreads,
    kBwdNMmaWarpgroups,
    kBwdNThreads,
    kBwdQdOStages,
    kBwdPreBlockM,
    kBwdPreThreads,
)
from fa4_launch_cache import enqueue_fa4_cached

comptime MOJO_DUMP_PTX: StaticString = get_defined_string[
    "MOJO_DUMP_PTX", ""
]()


def _dump_ptx_path() -> _DumpPath:
    comptime if MOJO_DUMP_PTX == StaticString(""):
        return _DumpPath(False)
    else:
        return _DumpPath(MOJO_DUMP_PTX)


def _ctx_and_stream(
    ctx_handle_addr: Int,
) -> DeviceContext:
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    return DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))


def launch_bwd_preprocess[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    o_addr: Int,
    do_addr: Int,
    lse_addr: Int,
    dpsum_addr: Int,
    lse_log2_addr: Int,
    dq_accum_addr: Int,
    dk_accum_addr: Int,
    dv_accum_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
    vl_num_q_tiles: Int = 0,
    vl_table_addr: Int = 0,
    vl_total_q: Int = 0,
    vl_total_qpad: Int = 0,
) raises:
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    # Varlen slot-riding (kernel signature unchanged): dk_accum
    # carries the q-tile work table, seq_len carries total_q and
    # nheads carries total_qpad.
    var dk_accum_addr_eff: Int = dk_accum_addr
    var seq_len_arg: Int = seqlen_int
    var nheads_arg: Int = nheads_int
    comptime if varlen:
        dk_accum_addr_eff = vl_table_addr
        seq_len_arg = vl_total_q
        nheads_arg = vl_total_qpad

    var dk_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dk_accum_addr_eff
    )
    var dv_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dv_accum_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var do_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=do_addr
    )
    var lse_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=lse_addr
    )
    var dpsum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dpsum_addr
    )
    var lse_log2_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=lse_log2_addr
    )
    var dq_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )

    comptime kernel_inst = bwd_preprocess_kernel[
        dtype, head_dim, causal, gqa_ratio, varlen
    ]
    # Grid covers Spad rows (side buffers padded to the main-kernel
    # m-block size; pad rows get lse=+inf / dpsum=0). Varlen: one CTA
    # per main-kernel m-block (q-tile table row).
    comptime bm_main: Int = kBwdTileM(head_dim, causal)
    var spad: Int = (
        ceildiv(seqlen_int, Int(bm_main)) * Int(bm_main)
    )
    var grid: Tuple[Int, Int, Int]
    comptime if varlen:
        grid = (vl_num_q_tiles, nheads_int, 1)
    else:
        grid = (
            ceildiv(spad, Int(kBwdPreBlockM)),
            nheads_int,
            batch_int,
        )

    enqueue_fa4_cached[
        kernel_inst,
        use_external_stream=use_external_stream,
    ](
        ctx,
        ctx_handle_addr,
        stream_opaque,
        String(
            t"bwd_pre_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"
            t"_vl{varlen}"
        ),
        grid,
        kBwdPreThreads,
        0,
        None,
        o_ptr,
        do_ptr,
        lse_ptr,
        dpsum_ptr,
        lse_log2_ptr,
        dq_accum_ptr,
        dk_accum_ptr,
        dv_accum_ptr,
        Int64(seq_len_arg),
        Int64(nheads_arg),
    )


def launch_bwd_main[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
    window: Bool = False,
    softcap_x1000: Int = 0,
    strided_qkv: Bool = False,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    do_addr: Int,
    dk_addr: Int,
    dv_addr: Int,
    lse_log2_addr: Int,
    dpsum_addr: Int,
    dq_accum_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
    vl_num_kv_tiles: Int = 0,
    vl_table_addr: Int = 0,
    vl_total_q: Int = 0,
    vl_total_k: Int = 0,
    vl_num_mpad: Int = 0,
    window_left: Int = 0,
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
    # dense d64 path in this port; the caller validates the layout
    # contract (d_stride==1, b_stride==S*s_stride, 16 B-aligned
    # strides) before this launcher runs.
    comptime assert (not strided_qkv) or (
        head_dim == 64
        and causal
        and not varlen
        and not window
        and gqa_ratio == 1
        and softcap_x1000 == 0
    ), "strided_qkv supports only dense d64 MHA"
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem (BM=80): K + V (2 x 32768) + 4-slot Q/dO ring (4 x 20480)
    # + 2 sdS stages (2 x 20480) + lse/dps (1280) + dQ mailbox
    # (2 x 20480) = 230912 B <= 232448 cap.
    comptime bm: Int = kBwdTileM(head_dim, causal)
    comptime kv_bytes: Int = kBwdBlockN * head_dim * size_of[dtype]()
    comptime q_slot_bytes: Int = bm * head_dim * size_of[dtype]()
    comptime sds_bytes: Int = bm * kBwdBlockN * size_of[dtype]()
    comptime mbar_bytes: Int = 256
    # + 2-stage lse_log2/dpsum staging ring (2 x 2 x BM f32).
    comptime lse_dps_bytes: Int = 2 * (kBwdQdOStages // 2) * bm * 4
    # + per-MMA-wg dQ mailbox (64 x DQ_N f32 each, bulk-reduce-
    # drained; DQ_N = bm at D=128, bm/2 at D=64 — the N split).
    comptime dq_n: Int = bm if head_dim == 128 else bm // 2
    comptime dq_mail_bytes: Int = (
        kBwdNMmaWarpgroups * 64 * dq_n * 4
    )
    comptime smem_bytes: Int = (
        2 * kv_bytes
        + kBwdQdOStages * q_slot_bytes
        + 2 * sds_bytes
        + lse_dps_bytes
        + dq_mail_bytes
        + mbar_bytes
    )

    var q_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var do_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=do_addr
    )
    var dk_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=dk_addr
    )
    var dv_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=dv_addr
    )
    var lse_log2_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=lse_log2_addr
    )
    var dpsum_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=dpsum_addr
    )
    var dq_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )

    comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim)
    comptime q_smem_shape = IndexList[3](bm, 1, head_dim)
    comptime kv_smem_shape = IndexList[3](kBwdBlockN, 1, head_dim)

    # Varlen: flat descriptors over the packed (total_tokens, H, D)
    # arrays; per-seq row offsets come from the kv-tile work table,
    # whose address rides the dk_accum_ptr kernel slot (MHA-only
    # v1); seq_len carries num_mpad = total_qpad / BM.
    var rows: Int = batch_int * seqlen_int
    var rows_kv: Int = rows
    var dk_accum_addr_eff: Int = dk_addr
    var seq_len_arg: Int = seqlen_int
    comptime if varlen:
        rows = vl_total_q
        rows_kv = vl_total_k
        dk_accum_addr_eff = vl_table_addr
        seq_len_arg = vl_num_mpad
    comptime if window:
        # win_left rides the high 32 bits of the seq_len kernel arg
        # (signature stays byte-identical to dense; see the kernel's
        # decode comment). Packs ON TOP of the varlen substitution
        # (num_mpad) when both apply.
        seq_len_arg = seq_len_arg | (window_left << 32)
    var nheads_kv: Int = nheads_int // gqa_ratio
    # Under GQA the dk/dv addresses carry the fp32 per-kv-head
    # accumulators (the epilogue bulk-reduce-adds into them; a torch
    # permute-cast converts). The bf16 TMA descriptors below are
    # then unused by the kernel (comptime-dead store path).
    var dk_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dk_accum_addr_eff
    )
    var dv_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dv_addr
    )
    # Strided mode: Q/K/V become runtime-strided flat (B*S, H, D)
    # descriptors (row stride = the caller's s_stride; valid because
    # b_stride == seqlen * s_stride was checked). dO/dK/dV keep the
    # contiguous descriptors below.
    var q_tma: SplitLastDimTMATensorTile[dtype, q_smem_shape, swizzle]
    var k_tma: SplitLastDimTMATensorTile[dtype, kv_smem_shape, swizzle]
    var v_tma: SplitLastDimTMATensorTile[dtype, kv_smem_shape, swizzle]
    comptime if strided_qkv:
        q_tma = create_split_tma_3d_strided[
            q_smem_shape, swizzle_mode=swizzle
        ](
            ctx,
            q_ptr,
            rows,
            nheads_int,
            head_dim,
            q_s_stride,
            q_h_stride,
            q_d_stride,
        )
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
        q_tma = create_split_tma[
            q_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, q_ptr, rows, nheads_int)
        k_tma = create_split_tma[
            kv_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, k_ptr, rows_kv, nheads_kv)
        v_tma = create_split_tma[
            kv_smem_shape, gmem_shape, swizzle_mode=swizzle
        ](ctx, v_ptr, rows_kv, nheads_kv)
    var do_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, do_ptr, rows, nheads_int)
    var dk_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, dk_ptr, rows_kv, nheads_kv)
    var dv_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, dv_ptr, rows_kv, nheads_kv)

    comptime kernel_inst = bwd_main_kernel[
        dtype,
        head_dim,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
        type_of(dk_tma).tile_shape,
        type_of(dk_tma).desc_shape,
        causal,
        gqa_ratio,
        varlen,
        window,
        softcap_x1000,
    ]

    var grid: Tuple[Int, Int, Int]
    comptime if varlen:
        comptime if gqa_ratio > 1:
            # pack-GQA: one CTA per (kv tile, KV head); the kernel
            # walks the whole group's m-sweeps internally.
            grid = (vl_num_kv_tiles, nheads_int // gqa_ratio, 1)
        else:
            grid = (vl_num_kv_tiles, nheads_int, 1)
    else:
        grid = (
            ceildiv(seqlen_int, Int(kBwdBlockN)),
            nheads_int,
            batch_int,
        )

    enqueue_fa4_cached[
        kernel_inst,
        use_external_stream=use_external_stream,
        dump_asm=_dump_ptx_path(),
    ](
        ctx,
        ctx_handle_addr,
        stream_opaque,
        String(
            t"bwd_main_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"
            t"_vl{varlen}_w{window}_sc{softcap_x1000}"
        ),
        grid,
        kBwdNThreads,
        smem_bytes,
        FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
        q_tma,
        do_tma,
        k_tma,
        v_tma,
        dk_tma,
        dv_tma,
        lse_log2_ptr,
        dpsum_ptr,
        dq_accum_ptr,
        dk_accum_ptr,
        dv_accum_ptr,
        Int64(seq_len_arg),
        softmax_scale,
    )


def launch_bwd_convert[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    dq_accum_addr: Int,
    dq_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
    vl_num_q_tiles: Int = 0,
    vl_table_addr: Int = 0,
    vl_num_mpad: Int = 0,
) raises:
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    var dq_accum_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )
    var dq_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=dq_addr
    )

    # Varlen slot-riding (kernel signature unchanged): seq_len
    # carries the q-tile work-table address, nheads carries num_mpad.
    var seq_len_arg: Int = seqlen_int
    var nheads_arg: Int = nheads_int
    comptime if varlen:
        seq_len_arg = vl_table_addr
        nheads_arg = vl_num_mpad

    # (BM q) x (128+4 d) f32 decode tile.
    comptime bm_cvt: Int = kBwdTileM(head_dim, causal)
    comptime cvt_smem_bytes: Int = bm_cvt * (head_dim + 4) * 4

    comptime kernel_inst = bwd_convert_kernel[
        dtype, head_dim, causal, varlen
    ]
    _ = gqa_ratio  # dq convert is ratio-independent
    # One CTA per main-kernel m-block.
    var grid: Tuple[Int, Int, Int]
    comptime if varlen:
        grid = (vl_num_q_tiles, nheads_int, 1)
    else:
        grid = (
            ceildiv(seqlen_int, Int(bm_cvt)),
            nheads_int,
            batch_int,
        )

    enqueue_fa4_cached[
        kernel_inst,
        use_external_stream=use_external_stream,
    ](
        ctx,
        ctx_handle_addr,
        stream_opaque,
        String(
            t"bwd_convert_{dtype}_d{head_dim}_c{causal}_vl{varlen}"
        ),
        grid,
        kBwdCvtThreads,
        cvt_smem_bytes,
        FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(cvt_smem_bytes)
        ),
        dq_accum_ptr,
        dq_ptr,
        Int64(seq_len_arg),
        Int64(nheads_arg),
        softmax_scale,
    )
