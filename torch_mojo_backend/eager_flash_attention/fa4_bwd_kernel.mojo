"""FA4-target flash-attention backward kernels (sm_90a, Hopper).

Three kernels, mirroring FA4's pipeline:

1. `bwd_preprocess_kernel`: dpsum = rowsum(dO * O) (fp32),
   lse_log2 = lse * log2(e), and zero dq_accum.
2. `bwd_main_kernel`: grid over KV tiles. Each block TMA-loads its
   K/V tile once, then loops over Q tiles (BM=80 rows, FA4's
   tile_m), streaming Q/dO through a 4-slot smem ring (2 stages
   each) filled by a producer warpgroup. Per m-tile, the 2 MMA
   warpgroups (each owning 64 of the 128 KV rows) compute, in
   scaled-log2 domain:

       S^T  = K · Q^T            (wgmma SS, m64n80k16, swapAB)
       dP^T = V · dO^T           (wgmma SS, m64n80k16, swapAB)
       P^T  = exp2(S^T*scale_log2 - lse_log2[col])
       dV  += P^T · dO           (wgmma RS, m64n128k16, 5 k-steps)
       dS^T = P^T * (dP^T - dpsum[col])
       dK  += dS^T · Q           (wgmma RS, m64n128k16, 5 k-steps)
       sdS  = dS (stmatrix.x4.trans into a k-major (BM, BN) SW128
                  tile — FA4's staging; wg w writes kv-slab w)
       dQ^T = K^T · dS           (hand-rolled wgmma, m64n80k16,
                                  trans(1,0), split m64 per wg)
       dQ c-frag -> smem mailbox; a producer warp drains it to
       dq_accum via cp.reduce.async.bulk (FA4's design).

   Epilogue: dK *= softmax_scale; dV and dK staged to smem
   (16B-chunk-major) and TMA bulk-stored.
3. `bwd_convert_kernel`: dq = (dq_accum * softmax_scale).bf16.

P^T / dS^T c-frag -> RS a-frag: straight indexwise cast (valid at
num_m_mmas=1, same argument as the fwd kernel).

dq_accum is an OPAQUE fragment dump (FA4's trick): per m-block of
BM rows, a contiguous [wg(2)][chunk(BM/8=10)][tid(128)][4] f32
region — the raw dQ^T wgmma c-frags, bulk-reduce-added linearly.
Element (wg, ch, t, e) is dQ^T row d = wg*64 + (t/32)*16 +
(t%32)/4 + 8*(e/2), col q = ch*8 + 2*(t%4) + e%2. The convert
kernel decodes. dpsum/lse_log2 are (B, H, Spad) fp32 with
Spad = ceil(S/BM)*BM (pad rows: lse=+inf, dpsum=0, so the tail
m-tile's P and dS vanish; the tail's TMA Q/dO reads are zero-fill
or finite next-batch garbage, both annihilated by P=dS=0). All
q/k/v/o/do/dq/dk/dv tensors are contiguous (B, S, H, D) bf16.
"""

from std.math import exp2, tanh
from std.math.constants import log2e
from std.utils.numerics import inf
from std.sys import size_of
from std.utils.index import StaticTuple, IndexList

from std.sys._assembly import inlined_assembly
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
import std.gpu.primitives.warp as warp
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
)
from std.gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    named_barrier,
    named_barrier_arrive,
)
from std.memory import bitcast, stack_allocation

from std.gpu.compute.mma import st_matrix, wgmma_async

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from fa4_wgmma_f16 import wgmma_rs_f16_m64n128, wgmma_rs_f16_m64n64

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

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16
comptime NUM_PRODUCER_REGS: Int = 24
comptime NUM_CONSUMER_REGS: Int = 240

# Temporary perf probes (never commit True): compile out a subsystem
# to attribute pipeline bubbles. Breaks dq correctness only.
comptime PROBE_NO_DQ: Bool = False  # skip sdS/PdS-barrier/dQ/mailbox
comptime PROBE_NO_DQ_GEMM: Bool = False  # keep sdS+barrier, skip GEMM+mailbox
comptime PROBE_NO_MAILBOX: Bool = False  # keep GEMM, skip mailbox+drain
comptime PROBE_NO_EXP2: Bool = False  # skip softmax exp2 chain
comptime PROBE_NO_REDUCE: Bool = False  # full protocol, skip cp.reduce
comptime PROBE_NO_MAIL_FENCE: Bool = False  # skip mailbox proxy fence
comptime SKIP_DQ_GEMM: Bool = PROBE_NO_DQ or PROBE_NO_DQ_GEMM
comptime SKIP_MAILBOX: Bool = SKIP_DQ_GEMM or PROBE_NO_MAILBOX


# ===================================================================
# Main backward kernel
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kBwdNThreads))
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(do_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(dk_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(dv_tma, `nvvm.grid_constant`)
def bwd_main_kernel[
    dtype: DType,
    head_dim: Int,
    q_tile_shape: IndexList[3],
    q_desc_shape: IndexList[3],
    kv_tile_shape: IndexList[3],
    kv_desc_shape: IndexList[3],
    st_tile_shape: IndexList[3],
    st_desc_shape: IndexList[3],
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
    window: Bool = False,
    softcap_x1000: Int = 0,
](
    q_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    do_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    dk_tma: TMATensorTile[dtype, 3, st_tile_shape, st_desc_shape],
    dv_tma: TMATensorTile[dtype, 3, st_tile_shape, st_desc_shape],
    lse_log2_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dpsum_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dq_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dk_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dv_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    softmax_scale: Float32,
):
    # Causal uses FA4's tile_m=64 (_tile_size_bwd_sm90: 64/128 with
    # dQ_swapAB=False; v0 keeps our swapped dQ — valid algebra, the
    # mailbox/convert layouts are BM-parametric).
    comptime BM: Int = kBwdTileM(head_dim, causal)
    comptime BN: Int = kBwdBlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kBwdNMmaWarpgroups
    comptime STAGES: Int = kBwdQdOStages
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # ---- smem layouts.
    comptime kv_smem_layout = tile_layout_k_major[
        dtype, BN, D, swizzle_mode=swizzle
    ]()
    # mn-major (D, BN) view of the same K bytes for the dQ GEMM's B.
    comptime kt_view_layout = tile_layout_mn_major[
        dtype, D, BN, swizzle_mode=swizzle
    ]()
    comptime q_smem_layout = tile_layout_k_major[
        dtype, BM, D, swizzle_mode=swizzle
    ]()
    # mn-major (D, BM) view of a Q/dO slot for the dK/dV GEMMs' B.
    comptime qt_view_layout = tile_layout_mn_major[
        dtype, D, BM, swizzle_mode=swizzle
    ]()
    # dS staged in FA4's orientation: a k-major (BM q-rows, BN
    # kv-cols) SW128 tile — same layout family as the Q slots, two
    # 64-col slabs of BM*64 elems; MMA wg w writes slab w via
    # stmatrix.x4.trans. It is the B operand (k = BN, n = BM) of the
    # hand-rolled dQ^T GEMM below, trans-b = 0.
    comptime sds_layout = tile_layout_k_major[
        dtype, BM, BN, swizzle_mode=swizzle
    ]()
    comptime sds_canonical = tile_to_descriptor[
        dtype, sds_layout, True
    ]()
    comptime sds_slab_bytes: Int = BM * 64 * size_of[dtype]()
    # A operand of dQ^T = K^T (D, BN) = the mn-major view of K's
    # bytes. TensorCoreAsync only supports k-major A, so the dQ^T
    # GEMM is hand-rolled with raw wgmma_async[layout_a="col"].
    comptime kt_canonical = tile_to_descriptor[
        dtype, kt_view_layout, False
    ]()
    comptime kt_shape00: Int = kt_canonical[0].shape[0].value()
    comptime kt_stride01: Int = kt_canonical[0].stride[1].value()
    comptime kt_stride11: Int = kt_canonical[1].stride[1].value()
    comptime a_wg_stride: Int = (
        kt_stride01 * (WGMMA_M // kt_shape00) * size_of[dtype]()
    )
    comptime a_k_stride: Int = kt_stride11 * 2 * size_of[dtype]()

    comptime kv_tile_size: Int = BN * D
    comptime q_slot_size: Int = BM * D
    comptime sds_size: Int = BM * BN

    var smem_base = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]()
    var k_base = smem_base
    var v_base = k_base + kv_tile_size
    var ring_base = v_base + kv_tile_size
    # sdS is double-buffered: stage m%2 is written at iter m and read
    # by the dQ GEMM; the pre-dQ barrier of iter m+1 proves dQ(m)
    # retired before either warpgroup rewrites stage m%2 at iter m+2.
    var sds_base = ring_base + STAGES * q_slot_size
    # lse_log2/dpsum smem ring: 2 stages x BM f32 each, prefetched
    # one m-tile ahead (consumer threads issue the gmem loads; the
    # per-iter named barrier orders store->read across warpgroups).
    # lse_log2/dpsum ride the Q pipeline (FA4's sLSE design): one
    # BM-f32 buffer per Q/dO slot pair, filled by the producer warp
    # and published by the Q slot's full barrier (init(2): TMA
    # expect + producer arrive); recycled with the slot's empties.
    var lse_smem = (sds_base + 2 * sds_size).bitcast[Float32]()
    var dps_smem = lse_smem + (STAGES // 2) * BM
    # dQ mailbox (FA4): per MMA wg, the raw dQ^T c-frag dump
    # [chunk(BM/8)][tid(128)][4] f32 = 20 KiB; the producer's drain
    # warp cp.reduce.async.bulk's it into dq_accum. Named-barrier
    # protocol per wg (count 128 + 32): empty 9+wg (drain arrives,
    # wg syncs), full 6+wg (wg arrives, drain syncs).
    comptime DQ_MAIL_F32: Int = WGMMA_M * (
        BM if D == 128 else BM // 2
    )  # 64 x (q cols per wg): 5120 at hdim128/BM=80, 4096 at hdim64
    comptime DRAIN_BAR: Int = 128 + 32
    var dq_mail = dps_smem + (STAGES // 2) * BM

    var k_smem = LayoutTensor[
        dtype,
        kv_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]((k_base).as_unsafe_any_origin())
    var v_smem = LayoutTensor[
        dtype,
        kv_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]((v_base).as_unsafe_any_origin())
    # ---- mbarriers.
    var mbar_k = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_v = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var full = stack_allocation[
        STAGES,
        SharedMemBarrier,
        address_space=AddressSpace.SHARED,
        alignment=8,
    ]()
    var empty = stack_allocation[
        STAGES,
        SharedMemBarrier,
        address_space=AddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar_k[0].init()
        mbar_v[0].init()
        comptime for s in range(STAGES):
            # One expect-arrive per slot (producer lane 0); the
            # slot's 320-B lse/dps cp.async.bulk rides the same
            # barrier via expect_tx (FA4's design — no stager warp).
            full[s].init(1)
            empty[s].init(Int32(NWG * 128))
    barrier()

    # Sliding window (causal local, v1: dense, left % 128 == 0).
    # win_left rides the HIGH 32 bits of seq_len — the established
    # seq_len slot-riding pattern (varlen already repurposes it as
    # num_mpad) — so the kernel signature stays byte-identical to
    # dense AND the feature survives GQA, where both accum-ptr
    # slots are taken. Non-window variants read seq_len verbatim.
    var win_left: Int = 0
    var slen: Int = seq_len
    comptime if window:
        win_left = seq_len >> 32
        slen = seq_len & 0xFFFFFFFF

    var n_block: Int
    var h_idx: Int
    var b_idx: Int
    var num_m_blocks: Int
    # Causal: KV tile n only receives gradient from q rows >= n*BN,
    # i.e. m-blocks m >= m_start. All warp roles share the offset.
    var m_start: Int = 0
    var kv_row: Int
    # Varlen per-CTA scalars (0/unused when dense). vl_kv_tail < BN
    # only on a sequence's boundary (ragged-tail) kv tile.
    var vl_q_base: Int = 0
    var vl_mpad_base: Int = 0
    var vl_kv_tail: Int = 0
    var vl_mask_base: Int = 0
    comptime if varlen:
        # Host kv-tile work table, one int32[8] row per CTA:
        # (n_block, q_row_base, k_row_base, seqlen_q, seqlen_k,
        # num_m_blocks, m_start, mpad_base). Its address rides the
        # dk_accum_ptr slot (GQA varlen is a follow-up) and seq_len
        # carries num_mpad = total_qpad / BM — the kernel signature
        # stays identical to dense. Every per-CTA scalar is
        # warp.broadcast-laundered (tid-widening hazard class, see
        # HANDOFF.md).
        var tbl = dk_accum_ptr.bitcast[Int32]() + 8 * Int(block_idx.x)
        n_block = Int(warp.broadcast(tbl[0]))
        vl_q_base = Int(warp.broadcast(tbl[1]))
        var vl_k_base: Int = Int(warp.broadcast(tbl[2]))
        var vl_slk: Int = Int(warp.broadcast(tbl[4]))
        num_m_blocks = Int(warp.broadcast(tbl[5]))
        m_start = Int(warp.broadcast(tbl[6]))
        vl_mpad_base = Int(warp.broadcast(tbl[7]))
        h_idx = Int(block_idx.y)
        comptime if gqa_ratio > 1:
            # pack-GQA (varlen): block_idx.y is the KV head; h_idx
            # tracks the group's FIRST q head, so every existing
            # h_idx // gqa_ratio (K/V coords) and stat/dq window
            # base below stays valid. The three loops then walk all
            # gqa_ratio heads' m-sweeps in one CTA: K/V load ONCE
            # per kv head and dK/dV accumulate in registers across
            # the whole group — no f32 accumulators, no cp.reduce
            # epilogue, no permute-cast.
            h_idx = Int(block_idx.y) * gqa_ratio
        b_idx = 0
        kv_row = vl_k_base + n_block * BN
        vl_kv_tail = vl_slk - n_block * BN  # >= BN on full tiles
        comptime if causal:
            # Cross-attention diagonal (bottom-right, FA4): kv row
            # j receives gradient from q cols i >= j - offs (offs =
            # slk - slq; host-asserted slq <= slk). The S^T mask
            # base for trip 0 — self-attn (offs == 0, m_start*BM ==
            # n*BN) reduces to 0, i.e. mask_d = it*BM as in dense.
            var vl_slq: Int = Int(warp.broadcast(tbl[3]))
            vl_mask_base = (
                m_start * BM - n_block * BN + (vl_slk - vl_slq)
            )
    else:
        n_block = Int(block_idx.x)
        h_idx = Int(block_idx.y)
        b_idx = Int(block_idx.z)
        num_m_blocks = (slen + BM - 1) // BM
        comptime if causal:
            m_start = (n_block * BN) // BM
        kv_row = b_idx * slen + n_block * BN
    var m_trips: Int = num_m_blocks - m_start
    # pack-GQA: every loop runs the whole group's m-sweeps.
    comptime pack: Bool = varlen and gqa_ratio > 1
    var total_trips: Int = m_trips
    comptime if pack:
        total_trips = gqa_ratio * m_trips
    comptime if window:
        # Upper trip bound: kv tile n receives gradient only from q
        # tiles with m*BM <= n*BN + BN - 1 + win_left (q rows past
        # the window edge attend none of this tile).
        var m_end: Int = min(
            num_m_blocks,
            (n_block * BN + BN + win_left + BM - 1) // BM,
        )
        m_trips = m_end - m_start

    # The warpgroup index is shfl-broadcast from lane 0: ptxas's
    # tid-uniformity pattern only matches 32-bit shr.u32 on %tid.x,
    # and LLVM re-canonicalizes any 32-bit extract of the widened
    # tid back to shr.u64, which makes every wg-derived value
    # per-thread (per-wg A-descriptor offsets, mailbox base,
    # barrier ids: ~2 R2UR per HGMMA). The convergent shfl is
    # opaque to LLVM and a recognized broadcast to ptxas (probe:
    # tid7w vs tid7c vs shflroot in scripts/ptxas_ur_probe.py).
    var wgid: Int = Int(
        warp.broadcast(Int32(Int(thread_idx.x) >> 7))
    )

    if wgid == 0:
        # ================= producer =================
        warpgroup_reg_dealloc[NUM_PRODUCER_REGS]()
        if thread_idx.x < 32:
            var lane: Int = Int(thread_idx.x)
            if lane == 0:
                mbar_k[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
                k_tma.async_copy_3d(
                    k_smem, mbar_k[0], (0, h_idx // gqa_ratio, kv_row)
                )
                mbar_v[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
                v_tma.async_copy_3d(
                    v_smem, mbar_v[0], (0, h_idx // gqa_ratio, kv_row)
                )

            var slot: Int = 0
            var phase: UInt32 = 0
            var wrap: Int = 0
            var q_row: Int
            var bh_stat: Int
            comptime if varlen:
                # Stats are (H, total_qpad), per-seq window at
                # mpad_base*BM (seq_len carries num_mpad).
                q_row = vl_q_base + m_start * BM
                bh_stat = (
                    (h_idx * slen + vl_mpad_base + m_start) * BM
                ) * 4
            else:
                # lse_log2/dpsum are (B, H, Spad) — padded to a
                # multiple of BM (pad rows +inf / 0; causal: spad == S).
                q_row = b_idx * slen + m_start * BM
                bh_stat = (
                    (b_idx * Int(grid_dim.y) + h_idx)
                    * (num_m_blocks * BM)
                    + m_start * BM
                ) * 4
            var lse_byte: Int = Int(lse_log2_ptr) + bh_stat
            var dps_byte: Int = Int(dpsum_ptr) + bh_stat
            # h_cur: SSA copy of h_idx outside pack-GQA (byte-
            # neutral); walks the group's q heads under pack. The
            # wrap constants are PRECOMPUTED so vl_q_base/m_start/
            # slen don't stay live across the loop (the producer
            # warp runs on the dealloc'd 32-reg budget — extra
            # liveness here is what ptxas spills).
            var h_cur: Int = h_idx
            var pk_m: Int = 0
            var pk_q_row0: Int = 0
            var pk_stat_adv: Int = 0
            comptime if pack:
                pk_q_row0 = vl_q_base + m_start * BM
                pk_stat_adv = (slen - m_trips) * BM * 4
            for _ in range(total_trips):
                # Tight TMA-issue loop. lse (Q slot) and dpsum (dO
                # slot) ride each stage's mbarrier as 320-B 1-D
                # cp.async.bulk copies counted by the same
                # expect_tx (FA4's design — the TMA/DMA engine does
                # the staging; no warp touches gmem).
                empty[slot].wait(phase)
                if lane == 0:
                    var q_st = LayoutTensor[
                        dtype,
                        q_smem_layout,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ]((ring_base + slot * q_slot_size).as_unsafe_any_origin())
                    full[slot].expect_bytes(
                        Int32(BM * D * size_of[dtype]() + BM * 4)
                    )
                    q_tma.async_copy_3d(
                        q_st, full[slot], (0, h_cur, q_row)
                    )
                    inlined_assembly[
                        "cp.async.bulk.shared::cluster.global"
                        + ".mbarrier::complete_tx::bytes"
                        + " [$0], [$1], $2, [$3];",
                        NoneType,
                        constraints="r,l,r,r",
                    ](
                        Int32(Int(lse_smem + (slot // 2) * BM)),
                        Int64(lse_byte),
                        Int32(BM * 4),
                        Int32(Int(full + slot)),
                    )

                empty[slot + 1].wait(phase)
                if lane == 0:
                    var do_st = LayoutTensor[
                        dtype,
                        q_smem_layout,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ]((ring_base + (slot + 1) * q_slot_size).as_unsafe_any_origin())
                    full[slot + 1].expect_bytes(
                        Int32(BM * D * size_of[dtype]() + BM * 4)
                    )
                    do_tma.async_copy_3d(
                        do_st, full[slot + 1], (0, h_cur, q_row)
                    )
                    inlined_assembly[
                        "cp.async.bulk.shared::cluster.global"
                        + ".mbarrier::complete_tx::bytes"
                        + " [$0], [$1], $2, [$3];",
                        NoneType,
                        constraints="r,l,r,r",
                    ](
                        Int32(Int(dps_smem + (slot // 2) * BM)),
                        Int64(dps_byte),
                        Int32(BM * 4),
                        Int32(Int(full + slot + 1)),
                    )

                q_row += BM
                lse_byte += BM * 4
                dps_byte += BM * 4
                comptime if pack:
                    # h-wrap: next q head of the group — q rows
                    # restart at this CTA's m_start; the stat
                    # windows advance one full head stride (slen
                    # carries num_mpad here).
                    pk_m += 1
                    if pk_m == m_trips:
                        pk_m = 0
                        h_cur += 1
                        q_row = pk_q_row0
                        lse_byte += pk_stat_adv
                        dps_byte += pk_stat_adv
                slot += 2
                wrap += 1
                if wrap == STAGES // 2:
                    wrap = 0
                    slot = 0
                    phase ^= 1
        elif thread_idx.x < 64:
            comptime if SKIP_MAILBOX:
                return
            # ---- dQ drain warp (FA4's design). Per m-tile: signal
            # each wg's mailbox empty once its previous bulk reduce
            # finished *reading* smem (wait_group.read), sync the
            # full barrier, then one lane bulk-reduce-adds the 20
            # KiB fragment dump into dq_accum. Two outstanding bulk
            # groups (one per wg) at steady state.
            var lane_d: Int = Int(lane_id())
            var dq_byte_base: Int
            comptime if varlen:
                dq_byte_base = Int(dq_accum_ptr) + (
                    h_idx * slen + vl_mpad_base + m_start
                ) * (2 * DQ_MAIL_F32 * 4)
            else:
                dq_byte_base = Int(dq_accum_ptr) + (
                    (b_idx * Int(grid_dim.y) + h_idx) * num_m_blocks
                    + m_start
                ) * (2 * DQ_MAIL_F32 * 4)
            var pk_md: Int = 0
            var pk_dq_adv: Int = 0
            comptime if pack:
                pk_dq_adv = (slen - m_trips) * (2 * DQ_MAIL_F32 * 4)
            for _ in range(total_trips):
                cp_async_bulk_wait_group[1]()
                named_barrier_arrive[Int32(DRAIN_BAR)](Int32(9))
                cp_async_bulk_wait_group[0]()
                named_barrier_arrive[Int32(DRAIN_BAR)](Int32(10))
                comptime for w in range(2):
                    named_barrier[Int32(DRAIN_BAR)](Int32(6 + w))
                    if lane_d == 0 and not PROBE_NO_REDUCE:
                        inlined_assembly[
                            "cp.reduce.async.bulk.global.shared::cta"
                            + ".bulk_group.add.f32 [$0], [$1], $2;",
                            NoneType,
                            constraints="l,r,r",
                        ](
                            Int64(
                                dq_byte_base + w * DQ_MAIL_F32 * 4
                            ),
                            Int32(Int(dq_mail + w * DQ_MAIL_F32)),
                            Int32(DQ_MAIL_F32 * 4),
                        )
                    cp_async_bulk_commit_group()
                dq_byte_base += 2 * DQ_MAIL_F32 * 4
                comptime if pack:
                    # h-wrap: jump to the next q head's dq_accum
                    # window (one head stride = slen m-rows).
                    pk_md += 1
                    if pk_md == m_trips:
                        pk_md = 0
                        dq_byte_base += pk_dq_adv
            cp_async_bulk_wait_group[0]()
        return

    # ================= MMA warpgroups =================
    warpgroup_reg_alloc[NUM_CONSUMER_REGS]()
    var wg: Int = wgid - 1

    comptime for s in range(STAGES):
        _ = empty[s].arrive()

    # ---- WGMMA operators.
    # S^T / dP^T: (BN x BM) = KV-tile · {Q,dO}-tile^T, M split by wg.
    var wgmma_sdp = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, BM, WGMMA_K),
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        transpose_b=True,
    ]()
    # dV / dK: (BN x D) += {P,dS}^T_regs · {dO,Q} (B = mn-major view).
    var wgmma_dkv = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, D, WGMMA_K),
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle,
        transpose_b=False,
    ]()

    comptime c_frag_sdp: Int = WGMMA_M * BM // 128  # 32
    comptime c_frag_dkv: Int = WGMMA_M * D // 128  # 64
    comptime a_frag: Int = WGMMA_M * WGMMA_K // 128  # 8
    comptime num_k_mmas_rs: Int = BM // WGMMA_K  # 4

    # fp16 RS fork support: the stdlib register-A wgmma is bf16-only;
    # the vendored m64n128k16 f32.f16.f16 emitter (_wgmma_f16.mojo)
    # replicates the TensorCoreAsync RS k-loop at the dV/dK sites.
    # bf16 keeps the stdlib path — byte-identical codegen.
    comptime qt_canonical = tile_to_descriptor[
        dtype, qt_view_layout, False
    ]()
    comptime qt_k_stride: Int = (
        qt_canonical[1].stride[1].value() * 2 * size_of[dtype]()
    )

    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_sdp),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dp_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_sdp),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var p_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_rs, a_frag),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var ds_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_rs, a_frag),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dv_acc = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dkv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dk_acc = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dkv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    _ = dv_acc.fill(0)
    _ = dk_acc.fill(0)
    # Per-WG dQ^T tile: D=128 splits M (each wg owns a 64-row D
    # slab, n = BM); D=64 has a single m64 so the split moves to N
    # (each wg owns 64 of the BM q-columns).
    comptime DQ_N: Int = BM if D == 128 else BM // 2
    comptime c_frag_dq: Int = WGMMA_M * DQ_N // 128
    var dq_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dq),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    var lane: Int = Int(lane_id())
    var warp_in_wg: Int = Int(warp_id()) % 4
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4
    var tid_in_wg: Int = Int(thread_idx.x) & 127

    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()
    # Softcap (Gemma-2), FA4 semantics — see the fwd kernel's note.
    # The COMPTIME cap repoints scale_log2 at cap*log2(e); s_reg
    # holds t = tanh(s*scale/cap) through the masks; the dS pass
    # gains the (1 - t^2) chain factor. dK/dQ keep their existing
    # softmax_scale epilogue multiplies (the chain factor d(qk) =
    # dS_capped * (1 - t^2) * scale preserves the plain scale).
    comptime softcap_on: Bool = softcap_x1000 != 0
    var t_scale: Scalar[accum_type] = Scalar[accum_type](0)
    comptime if softcap_on:
        comptime cap_f32: Float32 = Float32(softcap_x1000) / 1000
        t_scale = (softmax_scale / cap_f32).cast[accum_type]()
        scale_log2 = (
            cap_f32 * Scalar[DType.float32](log2e)
        ).cast[accum_type]()

    # ---- consumer state.
    mbar_k[0].wait(UInt32(0))
    mbar_v[0].wait(UInt32(0))


    var slot: Int = 0
    var phase: UInt32 = 0
    var wrap: Int = 0
    var sds_stage: Int = 0
    # m-position within the current head's sweep (== it outside
    # pack-GQA; wraps at m_trips under pack — the masks key on it).
    var pk_mc: Int = 0

    for it in range(total_trips):
        var q_view = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((ring_base + slot * q_slot_size).as_unsafe_any_origin())
        var qt_view = LayoutTensor[
            dtype,
            qt_view_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((ring_base + slot * q_slot_size).as_unsafe_any_origin())
        var do_view = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((ring_base + (slot + 1) * q_slot_size).as_unsafe_any_origin())
        var dot_view = LayoutTensor[
            dtype,
            qt_view_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((ring_base + (slot + 1) * q_slot_size).as_unsafe_any_origin())

        # Launder the K/V tile pointers through a no-op asm so the
        # A-operand wgmma descriptors are REBUILT here every
        # iteration instead of being hoisted as 24 loop-invariant
        # 64-bit k-step variants (which overflow the 63-UR/warp
        # uniform file and spill to local — the long_scoreboard
        # wall; FA4 rematerializes per iteration, ptxas folds the
        # rebuild into UR immediates).
        # The lane-0 broadcast after the no-op asm is FA4's idiom:
        # it makes the laundered value PROVABLY warp-uniform to
        # ptxas, so the rebuilt descriptor chains live fully on the
        # uniform datapath instead of being R2UR'd per HGMMA.
        var k_lnd: Int32 = inlined_assembly[
            "mov.b32 $0, $1;",
            Int32,
            constraints="=r,r",
            has_side_effect=True,
        ](Int32(Int(k_base)))
        var k_uni: Int = Int(warp.broadcast(k_lnd))
        var k_base_l = k_base + ((k_uni >> 1) - (Int(k_base) >> 1))
        # V root derived from K's (one shfl per iteration, not two:
        # shfl is an MIO op and mio_throttle is a live stall).
        var v_base_l = k_base_l + kv_tile_size
        var k_smem_l = LayoutTensor[
            dtype,
            kv_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((k_base_l).as_unsafe_any_origin())
        var v_smem_l = LayoutTensor[
            dtype,
            kv_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((v_base_l).as_unsafe_any_origin())

        # S^T = K · Q^T
        full[slot].wait(phase)
        # lse_log2/dpsum stay in smem (published with the Q slot) and
        # are loaded pairwise at their use sites below — keeping a
        # staged copy in a stack array put it in LOCAL memory and
        # blew the 168-reg budget (spills + non-uniform descriptors).
        var stat_col: Int = (slot // 2) * BM + 2 * lane_pair
        warpgroup_fence(s_reg)
        wgmma_sdp.arrive()
        wgmma_sdp.wgmma[num_warp_groups=NWG, scale_c=0](
            k_smem_l, q_view, s_reg, wg
        )
        wgmma_sdp.commit_group()

        # dP^T = V · dO^T
        full[slot + 1].wait(phase)
        warpgroup_fence(dp_reg)
        wgmma_sdp.arrive()
        wgmma_sdp.wgmma[num_warp_groups=NWG, scale_c=0](
            v_smem_l, do_view, dp_reg, wg
        )
        wgmma_sdp.commit_group()

        # S^T retired; dP^T still in flight.
        wgmma_sdp.wait_group[1]()
        warpgroup_fence(s_reg)

        comptime if softcap_on:
            # Cap BEFORE the mask arms (FA4 pre-mask score_mod):
            # the masks then write -1e30 into the capped domain, so
            # the exp2 below still yields EXACT zeros for masked
            # entries — the GQA cp.reduce no-op rows and dS
            # exactness both rely on that.
            comptime for c in range(c_frag_sdp):
                s_reg.ptr[c] = tanh(s_reg.ptr[c] * t_scale)

        comptime if causal:
            comptime if varlen:
                # Cross-attention general form: masked iff kv_abs >
                # q_abs + offs, i.e. mrow > ccol + (vl_mask_base +
                # m*BM). Self-attn: vl_mask_base == 0 and the guard
                # degenerates to m < BN//BM as in dense; cross
                # offsets shift which trips the band lands on.
                # (m_it == it outside pack-GQA; under pack it is
                # the m-position within the current head's sweep.)
                var m_it: Int = it
                comptime if pack:
                    m_it = pk_mc
                var mask_dv: Int = vl_mask_base + m_it * BM
                if mask_dv < BN:
                    var vrow_lo: Int = (
                        wg * WGMMA_M + warp_in_wg * 16 + lane_group
                    )
                    comptime for c in range(c_frag_sdp):
                        comptime vcol_base: Int = (c // 4) * 8 + (c & 1)
                        comptime vrow_off: Int = 8 if (c % 4) >= 2 else 0
                        if vrow_lo + vrow_off > (
                            vcol_base + 2 * lane_pair + mask_dv
                        ):
                            s_reg.ptr[c] = Scalar[accum_type](-1.0e30)
            else:
                if it < BN // BM:
                    var mask_d: Int = it * BM
                    var mrow_lo: Int = (
                        wg * WGMMA_M + warp_in_wg * 16 + lane_group
                    )
                    comptime for c in range(c_frag_sdp):
                        comptime ccol_base: Int = (c // 4) * 8 + (c & 1)
                        comptime crow_off: Int = 8 if (c % 4) >= 2 else 0
                        if mrow_lo + crow_off > (
                            ccol_base + 2 * lane_pair + mask_d
                        ):
                            s_reg.ptr[c] = Scalar[accum_type](-1.0e30)

        comptime if window:
            # Trailing window-edge trips: q cols past kv row +
            # win_left fall outside the window (the guard skips
            # trips with no masked pairs; unaligned lefts simply
            # mask one more trailing trip). Masked entries exp2 to
            # 0 in P^T, zeroing their dV/dK/dS/dQ contributions.
            var mask_w: Int
            comptime if varlen:
                # Bottom-right offset folds in via vl_mask_base;
                # within-head m position under pack-GQA.
                var mw_it: Int = it
                comptime if pack:
                    mw_it = pk_mc
                mask_w = win_left - vl_mask_base - mw_it * BM
            else:
                mask_w = (
                    n_block * BN + win_left - (m_start + it) * BM
                )
            if mask_w < BM:
                var wrow_lo: Int = (
                    wg * WGMMA_M + warp_in_wg * 16 + lane_group
                )
                comptime for c in range(c_frag_sdp):
                    comptime wcol_base: Int = (c // 4) * 8 + (c & 1)
                    comptime wrow_off: Int = 8 if (c % 4) >= 2 else 0
                    if wcol_base + 2 * lane_pair > (
                        wrow_lo + wrow_off + mask_w
                    ):
                        s_reg.ptr[c] = Scalar[accum_type](-1.0e30)

        comptime if varlen:
            # Ragged kv tail: garbage S^T ROWS (kv >= seqlen_k) on
            # the boundary CTA, masked EVERY m-trip — unlike the fwd,
            # causality does NOT subsume them (q rows below the
            # diagonal attend everything earlier, including the
            # garbage slots). P^T = exp2(-inf) = 0 zeroes their dV,
            # dS and dQ contributions.
            if vl_kv_tail < BN:
                var trow_lo: Int = (
                    wg * WGMMA_M + warp_in_wg * 16 + lane_group
                )
                comptime for c in range(c_frag_sdp):
                    comptime trow_off: Int = 8 if (c % 4) >= 2 else 0
                    if trow_lo + trow_off >= vl_kv_tail:
                        s_reg.ptr[c] = Scalar[accum_type](-1.0e30)

        comptime if softcap_on:
            # Softcap reorder: the dS chain factor (1 - t^2) needs
            # t (s_reg) BEFORE the exp2 overwrites it with P, so dP
            # retires first and (dP - dpsum) * factor lands in
            # dp_reg ahead of the P pass. max(., 0) clamps masked
            # entries — their (-1e30)^2 overflows to +inf and the
            # factor would otherwise be -inf (NaN against P == 0);
            # clamped, their dS is exactly 0.
            wgmma_sdp.wait_group[0]()
            warpgroup_fence(dp_reg)
            comptime for cc in range(c_frag_sdp // 4):
                var dp2c = (dps_smem + stat_col + cc * 8).load[
                    width=2, alignment=8
                ]()
                comptime for ic in range(4):
                    comptime c: Int = cc * 4 + ic
                    comptime j: Int = c & 1
                    var fac: Scalar[accum_type] = max(
                        s_reg.ptr[c].fma(
                            -s_reg.ptr[c], Scalar[accum_type](1)
                        ),
                        Scalar[accum_type](0),
                    )
                    dp_reg.ptr[c] = (dp_reg.ptr[c] - dp2c[j]) * fac
            comptime for cc in range(c_frag_sdp // 4):
                var lp2c = (lse_smem + stat_col + cc * 8).load[
                    width=2, alignment=8
                ]()
                comptime for ic in range(4):
                    comptime c: Int = cc * 4 + ic
                    comptime j: Int = c & 1
                    s_reg.ptr[c] = exp2(
                        s_reg.ptr[c].fma(scale_log2, -lp2c[j])
                    )
            comptime for c in range(c_frag_sdp):
                dp_reg.ptr[c] = s_reg.ptr[c] * dp_reg.ptr[c]
        else:
            # P^T = exp2(S^T * scale_log2 - lse_log2[q col]), f32 in
            # s_reg. NOT packed yet: FA4's order computs dS first,
            # then packs P and dS together — keeps the f32 P and
            # bf16 P from coexisting at the (former) register peak,
            # and keeps the dS multiply off a bf16->f32 unpack
            # critical path.
            comptime for cc in range(c_frag_sdp // 4):
                var lp2 = (lse_smem + stat_col + cc * 8).load[
                    width=2, alignment=8
                ]()
                comptime for ic in range(4):
                    comptime c: Int = cc * 4 + ic
                    comptime j: Int = c & 1
                    comptime if PROBE_NO_EXP2:
                        s_reg.ptr[c] = s_reg.ptr[c].fma(
                            scale_log2, -lp2[j]
                        )
                    else:
                        s_reg.ptr[c] = exp2(
                            s_reg.ptr[c].fma(scale_log2, -lp2[j])
                        )

            # dP^T retired (wait 0: nothing else in flight — FA4's
            # sequence; dV is committed after the dS store).
            wgmma_sdp.wait_group[0]()
            warpgroup_fence(dp_reg)

            # dS^T = P^T * (dP^T - dpsum[q col]); pack P and dS bf16.
            comptime for cc in range(c_frag_sdp // 4):
                var dp2 = (dps_smem + stat_col + cc * 8).load[
                    width=2, alignment=8
                ]()
                comptime for ic in range(4):
                    comptime c: Int = cc * 4 + ic
                    comptime j: Int = c & 1
                    dp_reg.ptr[c] = s_reg.ptr[c] * (
                        dp_reg.ptr[c] - dp2[j]
                    )
        comptime for c2 in range(c_frag_sdp // 2):
            var pp = SIMD[accum_type, 2](
                s_reg.ptr[2 * c2], s_reg.ptr[2 * c2 + 1]
            ).cast[dtype]()
            p_reg.ptr[2 * c2] = pp[0]
            p_reg.ptr[2 * c2 + 1] = pp[1]
        comptime for c2 in range(c_frag_sdp // 2):
            var dsp = SIMD[accum_type, 2](
                dp_reg.ptr[2 * c2], dp_reg.ptr[2 * c2 + 1]
            ).cast[dtype]()
            ds_reg.ptr[2 * c2] = dsp[0]
            ds_reg.ptr[2 * c2 + 1] = dsp[1]

        # Stage dS in FA4's k-major (q row, kv col) SW128 tile via 5
        # stmatrix.x4.trans per thread (each call: 16 q rows x this
        # warp's 16 kv cols, in this wg's 64-col slab; data = c-frag
        # bf16 pairs 8i..8i+7 in order — FA4's exact scheme, PTX
        # lines 716-781/1606-1611). The 128B swizzle XOR is taken
        # from the ABSOLUTE smem address (addr ^ ((addr>>3)&112)),
        # which folds the dynamic-smem base phase in for free; the
        # XOR term is invariant across i (i steps 2048 B = 16 lines).
        var sds_stage_base = sds_base + sds_stage * sds_size
        comptime if not PROBE_NO_DQ:
            var t_wg: Int = tid_in_wg
            var sds_elems: Int = (
                (t_wg % 8) * 64
                + ((t_wg // 8) % 2) * 8
                + ((t_wg // 16) % 2) * 512
                + ((t_wg // 32) % 4) * 16
                + wg * (BM * 64)
            )
            var sds_ba: Int = Int(sds_stage_base) + 2 * sds_elems
            var sds_sw: Int = sds_ba ^ ((sds_ba >> 3) & 112)
            # shifts, not // 2: Int floor-div emits a 17-op signed
            # rounding-correction chain per address (see HANDOFF).
            var sds_ptr = sds_stage_base + (
                (sds_sw >> 1) - (Int(sds_stage_base) >> 1)
            )
            comptime for i in range(c_frag_sdp // 8):
                var packed = SIMD[DType.float32, 4](0)
                comptime for jm in range(4):
                    packed[jm] = bitcast[DType.float32, 1](
                        SIMD[dtype, 2](
                            ds_reg.ptr[8 * i + 2 * jm],
                            ds_reg.ptr[8 * i + 2 * jm + 1],
                        )
                    )
                # .bitcast[BFloat16]: stdlib st_matrix over-asserts
                # bf16/f32; stmatrix.b16 is dtype-agnostic (no-op
                # for bf16, unblocks fp16).
                st_matrix[simd_width=4, transpose=True](
                    (sds_ptr + i * 1024).bitcast[BFloat16](), packed
                )

        # dV += P^T · dO — committed after the dS store (FA4's
        # order; see the register-pressure note above). The wgmma
        # runs under the PdS barrier wait below.
        warpgroup_fence(dv_acc)
        wgmma_dkv.arrive()
        comptime if dtype == DType.float16:
            var dvb_desc = _wgmma_descriptor[
                qt_canonical, False, swizzle
            ](ring_base + (slot + 1) * q_slot_size)
            var dv_simd = dv_acc.ptr.load[width=c_frag_dkv]()
            comptime for k_mma in range(num_k_mmas_rs):
                comptime if head_dim == 128:
                    dv_simd = rebind[SIMD[accum_type, c_frag_dkv]](
                        wgmma_rs_f16_m64n128(
                            rebind[SIMD[DType.float16, 8]](
                                (p_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (dvb_desc + k_mma * qt_k_stride).desc,
                            rebind[SIMD[DType.float32, 64]](dv_simd),
                        )
                    )
                else:
                    dv_simd = rebind[SIMD[accum_type, c_frag_dkv]](
                        wgmma_rs_f16_m64n64(
                            rebind[SIMD[DType.float16, 8]](
                                (p_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (dvb_desc + k_mma * qt_k_stride).desc,
                            rebind[SIMD[DType.float32, 32]](dv_simd),
                        )
                    )
            dv_acc.ptr.store[width=c_frag_dkv](dv_simd)
        else:
            wgmma_dkv.wgmma(p_reg, dot_view, dv_acc)
        wgmma_dkv.commit_group()

        comptime if not PROBE_NO_DQ:
            fence_async_view_proxy()
            # Single per-iter barrier: proves both warpgroups wrote
            # this stage of sdS *and* (transitively, via last iter's
            # wait_group below) that dQ(iter-1) retired before its
            # stage gets rewritten next iteration.
            named_barrier[Int32(NWG * 128)](Int32(4))

            comptime if not SKIP_DQ_GEMM:
                # dQ^T (D x BM) = K^T · dS, hand-rolled
                # (layout_a="col": A is K's bytes read mn-major;
                # layout_b="col": B is the k-major sdS, trans-b=0 —
                # FA4's imm tail is trans(1,0)). M = D = 128 split
                # m64 per warpgroup -> both warpgroups share the
                # GEMM. B k-steps are two-level: +32 B inside a
                # 64-kv slab (4 steps), +slab_bytes across. c-regs
                # via StaticTuple (SIMD[f32,40] is illegal); the
                # copy back into dq_reg folds to SSA aliases, so
                # the existing fences keep guarding the regs.
                warpgroup_fence(dq_reg)
                wgmma_sdp.arrive()
                var a_desc = _wgmma_descriptor[
                    kt_canonical, False, swizzle
                ](k_base_l)
                comptime if D == 128:
                    a_desc = a_desc + wg * a_wg_stride
                var b_desc = _wgmma_descriptor[
                    sds_canonical, True, swizzle
                ](sds_stage_base)
                comptime if D == 64:
                    # N-split: each wg's B window is 64 sdS q-ROWS.
                    # The canonical (BM, BN) SW128 tile is COLUMN-
                    # SLAB-major (64-col slabs of BM*64 elems; the
                    # k-steps below jump slabs), so 64 rows = 8 cores
                    # x 512 elems = 8 KiB within every slab.
                    b_desc = b_desc + wg * (64 * 64 * size_of[dtype]())
                var dq_tup = StaticTuple[
                    Scalar[accum_type], c_frag_dq
                ]()
                comptime for k_mma in range(BN // WGMMA_K):
                    dq_tup = wgmma_async[
                        WGMMA_M,
                        DQ_N,
                        WGMMA_K,
                        accum_type,
                        c_frag_dq,
                        a_type=dtype,
                        b_type=dtype,
                        layout_a="col",
                        layout_b="col",
                        scale_d = 0 if k_mma == 0 else 1,
                    ](
                        a_desc + k_mma * a_k_stride,
                        b_desc
                        + (
                            (k_mma & 3) * 32
                            + (k_mma >> 2) * sds_slab_bytes
                        ),
                        dq_tup,
                    )
                comptime for c in range(c_frag_dq):
                    dq_reg.ptr[c] = dq_tup[c]
                wgmma_sdp.commit_group()

        # Queue [dV, dQ]: wait ≤1 retires dV -> dO(n) reusable now,
        # one GEMM earlier than waiting on dQ (FA4's release point).
        wgmma_dkv.wait_group[1]()
        warpgroup_fence(dv_acc)
        _ = empty[slot + 1].arrive()

        # dK += dS^T · Q — committed AFTER dQ (FA4's order) so the
        # dQ drain below overlaps the dK GEMM on the tensor core.
        warpgroup_fence(dk_acc)
        wgmma_dkv.arrive()
        comptime if dtype == DType.float16:
            var dkb_desc = _wgmma_descriptor[
                qt_canonical, False, swizzle
            ](ring_base + slot * q_slot_size)
            var dk_simd = dk_acc.ptr.load[width=c_frag_dkv]()
            comptime for k_mma in range(num_k_mmas_rs):
                comptime if head_dim == 128:
                    dk_simd = rebind[SIMD[accum_type, c_frag_dkv]](
                        wgmma_rs_f16_m64n128(
                            rebind[SIMD[DType.float16, 8]](
                                (ds_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (dkb_desc + k_mma * qt_k_stride).desc,
                            rebind[SIMD[DType.float32, 64]](dk_simd),
                        )
                    )
                else:
                    dk_simd = rebind[SIMD[accum_type, c_frag_dkv]](
                        wgmma_rs_f16_m64n64(
                            rebind[SIMD[DType.float16, 8]](
                                (ds_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (dkb_desc + k_mma * qt_k_stride).desc,
                            rebind[SIMD[DType.float32, 32]](dk_simd),
                        )
                    )
            dk_acc.ptr.store[width=c_frag_dkv](dk_simd)
        else:
            wgmma_dkv.wgmma(ds_reg, qt_view, dk_acc)
        wgmma_dkv.commit_group()

        # Queue [dQ, dK]: wait ≤1 retires dQ; dK still runs while we
        # hand dQ off.
        wgmma_dkv.wait_group[1]()
        warpgroup_fence(dq_reg)

        # Hand dQ^T to the drain warp: raw c-frag dump into this
        # wg's mailbox (10 x st.shared.v4, fully coalesced), under
        # the dK GEMM. The drain warp owns the gmem reduce-add.
        comptime if not SKIP_MAILBOX:
            named_barrier[Int32(DRAIN_BAR)](Int32(9 + wg))
            var mail = dq_mail + wg * DQ_MAIL_F32 + tid_in_wg * 4
            comptime for ch in range(c_frag_dq // 4):
                (mail + ch * 512).store[width=4, alignment=16](
                    SIMD[accum_type, 4](
                        dq_reg.ptr[4 * ch],
                        dq_reg.ptr[4 * ch + 1],
                        dq_reg.ptr[4 * ch + 2],
                        dq_reg.ptr[4 * ch + 3],
                    )
                )
            comptime if not PROBE_NO_MAIL_FENCE:
                fence_async_view_proxy()
            named_barrier_arrive[Int32(DRAIN_BAR)](Int32(6 + wg))

        # dK retired -> Q(n) slot reusable.
        wgmma_dkv.wait_group[0]()
        warpgroup_fence(dk_acc)
        _ = empty[slot].arrive()

        sds_stage ^= 1
        slot += 2
        wrap += 1
        if wrap == STAGES // 2:
            wrap = 0
            slot = 0
            phase ^= 1
        comptime if pack:
            pk_mc += 1
            if pk_mc == m_trips:
                pk_mc = 0

    # ---- Epilogue.
    comptime if varlen:
        # Ragged kv tail tile: a full-tile TMA store would overwrite
        # the NEXT sequence's dK/dV rows — store the c-frags straight
        # to gmem, row-predicated (no smem staging, no barrier; both
        # warpgroups take this branch together). The raw dk/dv base
        # addresses live in the work table's aux row (row index
        # grid_dim.x), packed as two int64s. (pack-GQA takes this
        # path too: the group's dK/dV accumulated in registers, so
        # the store is per-KV-head exactly like MHA.)
        if vl_kv_tail < BN:
            var taux = dk_accum_ptr.bitcast[Int64]() + 4 * Int(
                grid_dim.x
            )
            var dk_g = UnsafePointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=Int(taux[0])
            )
            var dv_g = UnsafePointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=Int(taux[1])
            )
            var vscale: Scalar[accum_type] = softmax_scale.cast[
                accum_type
            ]()
            comptime for c in range(c_frag_dkv):
                dk_acc.ptr[c] *= vscale
            var prow_lo: Int = wg * WGMMA_M + warp_in_wg * 16 + lane_group
            comptime for c2 in range(c_frag_dkv // 2):
                comptime p_chunk: Int = c2 // 2
                comptime p_bot: Int = c2 % 2
                var prow: Int = prow_lo + (8 if p_bot == 1 else 0)
                if prow < vl_kv_tail:
                    # dk/dv have nheads_kv heads under pack-GQA;
                    # //gqa_ratio folds away at ratio==1. grid.y is
                    # the dk/dv head count in both modes.
                    var goff: Int = (
                        (kv_row + prow) * Int(grid_dim.y)
                        + h_idx // gqa_ratio
                    ) * D + p_chunk * 8 + 2 * lane_pair
                    (dv_g + goff).store[width=2, alignment=4](
                        SIMD[accum_type, 2](
                            dv_acc.ptr[2 * c2], dv_acc.ptr[2 * c2 + 1]
                        ).cast[dtype]()
                    )
                    (dk_g + goff).store[width=2, alignment=4](
                        SIMD[accum_type, 2](
                            dk_acc.ptr[2 * c2], dk_acc.ptr[2 * c2 + 1]
                        ).cast[dtype]()
                    )
            return

    # The dK/dV staging below overwrites the K/V smem
    # areas; warpgroup 0's final dQ GEMM (retired before its barrier
    # arrival) must not still be reading the transposed K view -> sync first.
    named_barrier[Int32(NWG * 128)](Int32(4))

    # FA4's epilogue: stage each output via 8x stmatrix.x4
    # (non-trans) into the dead K/V SW128 tiles — the layout the
    # tiles were TMA-loaded with — and issue 2 big TMA stores per
    # output (SWIZZLE_128B descriptors), dV's store overlapping
    # dK's scale + staging. Same absolute-address XOR fold as the
    # sdS store; the term is invariant across i (steps 32 B and
    # 16384 B never touch addr bits 7-9).
    var scale_acc: Scalar[accum_type] = softmax_scale.cast[accum_type]()

    comptime if gqa_ratio > 1 and not varlen:
        # Dense GQA epilogue: multiple q-head CTAs accumulate into
        # the same kv-head's dK/dV (FA4's fp32-accum + postprocess
        # design). Varlen GQA packs the group into one CTA instead
        # and exits through the bf16 path below.
        # Stage each tensor as a row-major 128x128 f32 tile in the
        # dead K+V smem (exactly 64 KiB) and bulk-reduce-add it into
        # the per-kv-head accumulator; a torch permute-cast converts.
        comptime for c in range(c_frag_dkv):
            dk_acc.ptr[c] *= scale_acc
        # Staging area for the row-major f32 tile (BN x D x 4 B):
        # D=128 uses the dead K+V smem (exactly 64 KiB); D=64's tile
        # is 32 KiB but dead K+V is also only 32 KiB COMBINED with
        # live mbarriers nearby — use the dead sdS stage 0 instead
        # (32 KiB, dead after the pre-epilogue barrier).
        var acc32 = (
            k_base if D == 128 else sds_base
        ).bitcast[Float32]()
        var kv_acc_base: Int = 0
        comptime if not varlen:
            kv_acc_base = (
                (
                    b_idx * (Int(grid_dim.y) // gqa_ratio)
                    + h_idx // gqa_ratio
                )
                * slen
                + n_block * BN
            ) * D * 4
        comptime for t in range(2):  # 0 = dV, 1 = dK
            comptime for c2 in range(c_frag_dkv // 2):
                comptime g_chunk: Int = c2 // 2
                comptime g_bot: Int = c2 % 2
                var g_row: Int = (
                    wg * WGMMA_M + warp_in_wg * 16 + lane_group
                    + (8 if g_bot == 1 else 0)
                )
                var g_col: Int = g_chunk * 8 + 2 * lane_pair
                var g_pr: SIMD[accum_type, 2]
                comptime if t == 0:
                    g_pr = SIMD[accum_type, 2](
                        dv_acc.ptr[2 * c2], dv_acc.ptr[2 * c2 + 1]
                    )
                else:
                    g_pr = SIMD[accum_type, 2](
                        dk_acc.ptr[2 * c2], dk_acc.ptr[2 * c2 + 1]
                    )
                (acc32 + g_row * D + g_col).store[width=2, alignment=8](
                    g_pr
                )
            fence_async_view_proxy()
            named_barrier[Int32(NWG * 128)](Int32(4))
            if thread_idx.x == 128:
                # The reduce destination is computed HERE (one
                # thread, at issue time) so its addresses are never
                # live across the staging loops (a 16-B spill
                # otherwise). Varlen: packed (Hkv, total_k_alloc, D)
                # f32 accumulators; the raw pointers + total_k_alloc
                # live in the kv table's aux row (dk_accum_ptr
                # carries the table). kv_row is already the global
                # packed row; boundary tiles reduce-add EXACT ZEROS
                # past seqlen_k (masked S^T rows), and the allocator
                # pads the buffer end to a full tile.
                var red_dst: Int
                comptime if varlen:
                    # Everything here is REBUILT from kernel params /
                    # special regs (table reload, block_idx) so no
                    # per-CTA scalar stays live across the main loop
                    # for the epilogue's sake — the combined vl +
                    # GQA liveness otherwise spills ~8 B.
                    var tblr = dk_accum_ptr.bitcast[Int32]() + 8 * Int(
                        block_idx.x
                    )
                    var kv_row_e: Int = Int(tblr[2]) + Int(tblr[0]) * BN
                    var taux64 = dk_accum_ptr.bitcast[Int64]() + 4 * Int(
                        grid_dim.x
                    )
                    var taux32 = dk_accum_ptr.bitcast[Int32]() + 8 * Int(
                        grid_dim.x
                    )
                    red_dst = Int(taux64[0 if t == 1 else 1]) + (
                        (Int(block_idx.y) // gqa_ratio) * Int(taux32[4])
                        + kv_row_e
                    ) * D * 4
                else:
                    red_dst = (
                        Int(dv_accum_ptr if t == 0 else dk_accum_ptr)
                        + kv_acc_base
                    )
                inlined_assembly[
                    "cp.reduce.async.bulk.global.shared::cta"
                    + ".bulk_group.add.f32 [$0], [$1], $2;",
                    NoneType,
                    constraints="l,r,r",
                ](
                    Int64(red_dst),
                    Int32(Int(acc32)),
                    Int32(BN * D * 4),
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()
            # All threads wait for the bulk read before restaging.
            named_barrier[Int32(NWG * 128)](Int32(4))
        return

    var st_lane: Int = lane
    var st_row: Int = (
        wg * WGMMA_M + warp_in_wg * 16 + ((st_lane // 8) % 2) * 8
        + (st_lane % 8)
    )
    var st_off_raw: Int = st_row * 64 + (st_lane // 16) * 8

    comptime if D == 64:
        # D=64: plain paired stores at canonical SW128 addresses
        # (64-elem rows = one 128-B period; the stmatrix scheme
        # below encodes D=128 geometry).
        comptime for c2 in range(c_frag_dkv // 2):
            comptime e_chunk: Int = c2 // 2
            comptime e_bot: Int = c2 % 2
            var e_row: Int = (
                wg * WGMMA_M + warp_in_wg * 16 + lane_group
                + (8 if e_bot == 1 else 0)
            )
            var e_col: Int = e_chunk * 8 + 2 * lane_pair
            var e_pair = SIMD[dtype, 2](
                dv_acc.ptr[2 * c2].cast[dtype](),
                dv_acc.ptr[2 * c2 + 1].cast[dtype](),
            )
            var e_addr: Int = (
                Int(v_base)
                + (e_row >> 3) * 1024
                + (e_row & 7) * 128
                + 2 * e_col
            )
            var e_sw: Int = e_addr ^ ((e_addr >> 3) & 112)
            (
                v_base + ((e_sw >> 1) - (Int(v_base) >> 1))
            ).store[width=2, alignment=4](e_pair)
    else:
        var dv_raw: Int = Int(v_base) + 2 * st_off_raw
        comptime for i in range(c_frag_dkv // 8):
            var packed = SIMD[DType.float32, 4](0)
            comptime for jm in range(4):
                comptime p: Int = 4 * i + jm
                packed[jm] = bitcast[DType.float32, 1](
                    SIMD[accum_type, 2](
                        dv_acc.ptr[2 * p], dv_acc.ptr[2 * p + 1]
                    ).cast[dtype]()
                )
            # XOR per call: the 32-B column steps live in the
            # swizzled bits, so the fold must apply per address.
            var raw_i: Int = (
                dv_raw + (i % 4) * 32 + (i // 4) * (BN * 128)
            )
            var sw_i: Int = raw_i ^ ((raw_i >> 3) & 112)
            st_matrix[simd_width=4](
                (
                    v_base + ((sw_i >> 1) - (Int(v_base) >> 1))
                ).bitcast[BFloat16](),
                packed,
            )
    fence_async_view_proxy()
    named_barrier[Int32(NWG * 128)](Int32(4))
    if thread_idx.x == 128:
        var dv_st = LayoutTensor[
            dtype,
            kv_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((v_base).as_unsafe_any_origin())
        dv_tma.async_store_3d(dv_st, (0, h_idx // gqa_ratio, kv_row))
        dv_tma.commit_group()

    # dK *= scale; staged under dV's in-flight TMA store.
    comptime for c in range(c_frag_dkv):
        dk_acc.ptr[c] *= scale_acc
    comptime if D == 64:
        comptime for c2 in range(c_frag_dkv // 2):
            comptime f_chunk: Int = c2 // 2
            comptime f_bot: Int = c2 % 2
            var f_row: Int = (
                wg * WGMMA_M + warp_in_wg * 16 + lane_group
                + (8 if f_bot == 1 else 0)
            )
            var f_col: Int = f_chunk * 8 + 2 * lane_pair
            var f_pair = SIMD[dtype, 2](
                dk_acc.ptr[2 * c2].cast[dtype](),
                dk_acc.ptr[2 * c2 + 1].cast[dtype](),
            )
            var f_addr: Int = (
                Int(k_base)
                + (f_row >> 3) * 1024
                + (f_row & 7) * 128
                + 2 * f_col
            )
            var f_sw: Int = f_addr ^ ((f_addr >> 3) & 112)
            (
                k_base + ((f_sw >> 1) - (Int(k_base) >> 1))
            ).store[width=2, alignment=4](f_pair)
    else:
        var dk_raw: Int = Int(k_base) + 2 * st_off_raw
        comptime for i in range(c_frag_dkv // 8):
            var packed = SIMD[DType.float32, 4](0)
            comptime for jm in range(4):
                comptime p: Int = 4 * i + jm
                packed[jm] = bitcast[DType.float32, 1](
                    SIMD[accum_type, 2](
                        dk_acc.ptr[2 * p], dk_acc.ptr[2 * p + 1]
                    ).cast[dtype]()
                )
            var raw_i: Int = (
                dk_raw + (i % 4) * 32 + (i // 4) * (BN * 128)
            )
            var sw_i: Int = raw_i ^ ((raw_i >> 3) & 112)
            st_matrix[simd_width=4](
                (
                    k_base + ((sw_i >> 1) - (Int(k_base) >> 1))
                ).bitcast[BFloat16](),
                packed,
            )
    fence_async_view_proxy()
    named_barrier[Int32(NWG * 128)](Int32(4))
    if thread_idx.x == 128:
        var dk_st = LayoutTensor[
            dtype,
            kv_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((k_base).as_unsafe_any_origin())
        dk_tma.async_store_3d(dk_st, (0, h_idx // gqa_ratio, kv_row))
        dk_tma.commit_group()
        dk_tma.wait_group()


# ===================================================================
# Preprocess kernel: dpsum = rowsum(dO * O), lse_log2 = lse*log2(e),
# zero dq_accum. Grid (ceil(Spad/128), H, B), 128 threads.
# Side buffers (dpsum, lse_log2, dq_accum) are padded to
# Spad = ceil(S / kBwdBlockM) * kBwdBlockM; pad rows get lse=+inf /
# dpsum=0 so the main kernel's tail m-block contributes exact zeros.
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kBwdPreThreads)
    )
)
def bwd_preprocess_kernel[
    dtype: DType,
    head_dim: Int,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
](
    o_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    do_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dpsum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    lse_log2_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dq_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dk_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dv_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    nheads: Int,
):
    comptime D: Int = head_dim
    comptime BM: Int = kBwdPreBlockM

    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var tid: Int = Int(thread_idx.x)

    comptime bm_main: Int = kBwdTileM(head_dim, causal)

    comptime if varlen:
        # One CTA per main-kernel m-block (bm_main rows). The q-tile
        # work table — one int32[8] row per CTA: (m_local,
        # q_row_base, seqlen_q, mpad_base, ...) — rides the
        # dk_accum_ptr slot; seq_len carries total_q (packed (H,
        # total_q) LSE reads) and nheads carries total_qpad. stats
        # windows are per-seq at mpad_base*bm_main in (H, total_qpad).
        var tbl = dk_accum_ptr.bitcast[Int32]() + 8 * Int(block_idx.x)
        var m_local: Int = Int(tbl[0])
        var q_base: Int = Int(tbl[1])
        var slq: Int = Int(tbl[2])
        var mpad_base: Int = Int(tbl[3])
        var nh: Int = Int(grid_dim.y)
        var total_q: Int = seq_len
        var total_qpad: Int = nheads

        comptime LANES_PER_ROW: Int = 8
        comptime RVEC: Int = D // LANES_PER_ROW  # 16
        comptime ROWS_PER_PASS: Int = kBwdPreThreads // LANES_PER_ROW
        var sub: Int = tid % LANES_PER_ROW
        var row_in_pass: Int = tid // LANES_PER_ROW
        comptime for p in range(bm_main // ROWS_PER_PASS):
            var s_loc: Int = (
                m_local * bm_main + p * ROWS_PER_PASS + row_in_pass
            )
            var stat_row: Int = (
                h_idx * total_qpad + mpad_base * bm_main + s_loc
            )
            if s_loc < slq:
                var off: Int = (
                    (q_base + s_loc) * nh + h_idx
                ) * D + sub * RVEC
                var o_v = (o_ptr + off).load[width=RVEC]().cast[
                    DType.float32
                ]()
                var do_v = (do_ptr + off).load[width=RVEC]().cast[
                    DType.float32
                ]()
                var part: Float32 = (o_v * do_v).reduce_add()
                var dps = warp.lane_group_sum[num_lanes=LANES_PER_ROW](
                    part
                )
                if sub == 0:
                    (dpsum_ptr + stat_row)[0] = dps
                    (lse_log2_ptr + stat_row)[0] = (
                        lse_ptr + h_idx * total_q + q_base + s_loc
                    )[0] * Float32(log2e)
            else:
                # Per-seq window pad rows: +inf/0 annihilate the main
                # kernel's tail m-rows (same convention as dense —
                # do NOT switch to FA4's lse_log2=0 without also
                # adding the in-loop seqlen_q mask).
                if sub == 0:
                    (dpsum_ptr + stat_row)[0] = Float32(0)
                    (lse_log2_ptr + stat_row)[0] = inf[DType.float32]()

        # Zero this m-block's dq_accum fragment region (bm_main*D
        # f32, exactly divisible by the per-pass footprint).
        comptime ZVEC: Int = 4
        comptime zpasses: Int = (bm_main * D) // (kBwdPreThreads * ZVEC)
        var zbase: Int = (
            h_idx * total_qpad + (mpad_base + m_local) * bm_main
        ) * D
        var zoff: Int = tid * ZVEC
        for _ in range(zpasses):
            (dq_accum_ptr + zbase + zoff).store[width=ZVEC](
                SIMD[DType.float32, ZVEC](0)
            )
            zoff += kBwdPreThreads * ZVEC

        # (Varlen GQA needs no accumulator zeroing: pack-GQA
        # accumulates the group's dK/dV in registers and stores
        # bf16 directly — there are no f32 accumulators.)
        return
    var num_main_blocks: Int = (
        seq_len + bm_main - 1
    ) // bm_main
    var spad: Int = num_main_blocks * bm_main

    # Coalesced: 8 threads per row, 16 rows per pass, 8 passes.
    # Each thread loads a contiguous 16-element (32B) slice.
    comptime LANES_PER_ROW: Int = 8
    comptime RVEC: Int = D // LANES_PER_ROW  # 16
    comptime ROWS_PER_PASS: Int = kBwdPreThreads // LANES_PER_ROW  # 16
    var sub: Int = tid % LANES_PER_ROW
    var row_in_pass: Int = tid // LANES_PER_ROW
    comptime for p in range(BM // ROWS_PER_PASS):
        var s: Int = m_block * BM + p * ROWS_PER_PASS + row_in_pass
        var bh_row: Int = (b_idx * nheads + h_idx) * spad + s
        if s < seq_len:
            var off: Int = (
                (b_idx * seq_len + s) * nheads + h_idx
            ) * D + sub * RVEC
            var o_v = (o_ptr + off).load[width=RVEC]().cast[
                DType.float32
            ]()
            var do_v = (do_ptr + off).load[width=RVEC]().cast[
                DType.float32
            ]()
            var part: Float32 = (o_v * do_v).reduce_add()
            var dps = warp.lane_group_sum[num_lanes=LANES_PER_ROW](part)
            if sub == 0:
                var lse_row: Int = (
                    b_idx * nheads + h_idx
                ) * seq_len + s
                (dpsum_ptr + bh_row)[0] = dps
                (lse_log2_ptr + bh_row)[0] = (lse_ptr + lse_row)[
                    0
                ] * Float32(log2e)
        elif s < spad:
            # Pad rows: exp2(x - inf) = 0 kills P; dpsum = 0 keeps
            # dS = P * (dP - dpsum) = 0 * finite = 0.
            if sub == 0:
                (dpsum_ptr + bh_row)[0] = Float32(0)
                (lse_log2_ptr + bh_row)[0] = inf[DType.float32]()

    # Zero dq_accum (the main kernel bulk-reduce-ADDS into it):
    # Spad*D f32 per (b, h), split as a flat contiguous range across
    # the grid's x dimension.
    comptime ZVEC: Int = 4
    comptime ZCHUNK: Int = kBwdPreThreads * ZVEC  # f32 per pass
    var ztot: Int = spad * D
    var zbase: Int = (b_idx * nheads + h_idx) * ztot
    var nb: Int = Int(grid_dim.x)
    var passes: Int = (ztot + nb * ZCHUNK - 1) // (nb * ZCHUNK)
    var zoff: Int = m_block * passes * ZCHUNK + tid * ZVEC
    for _ in range(passes):
        if zoff < ztot:
            (dq_accum_ptr + zbase + zoff).store[width=ZVEC](
                SIMD[DType.float32, ZVEC](0)
            )
        zoff += ZCHUNK

    # GQA: zero this (b, h_kv) slice of the fp32 dK/dV accumulators
    # (the main kernel bulk-reduce-ADDS into them). Done by the
    # h % ratio == 0 grid columns, split across their m-blocks.
    comptime if gqa_ratio > 1:
        if h_idx % gqa_ratio == 0:
            var kvtot: Int = seq_len * D
            var kvbase: Int = (
                b_idx * (nheads // gqa_ratio) + h_idx // gqa_ratio
            ) * kvtot
            var kpasses: Int = (
                kvtot + nb * ZCHUNK - 1
            ) // (nb * ZCHUNK)
            var koff: Int = m_block * kpasses * ZCHUNK + tid * ZVEC
            for _ in range(kpasses):
                if koff < kvtot:
                    (dk_accum_ptr + kvbase + koff).store[width=ZVEC](
                        SIMD[DType.float32, ZVEC](0)
                    )
                    (dv_accum_ptr + kvbase + koff).store[width=ZVEC](
                        SIMD[DType.float32, ZVEC](0)
                    )
                koff += ZCHUNK


# ===================================================================
# Convert kernel: dq = (dq_accum * softmax_scale) cast to bf16.
# Grid (ceil(S/kBwdBlockM), H, B) — one CTA per main-kernel m-block.
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kBwdCvtThreads)
    )
)
def bwd_convert_kernel[
    dtype: DType,
    head_dim: Int,
    causal: Bool = False,
    varlen: Bool = False,
](
    dq_accum_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dq_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_len: Int,
    nheads: Int,
    softmax_scale: Float32,
):
    """dq[b,s,h,d] = scale * decode(dq_accum) via a (q, d) smem tile.

    256 threads, one CTA per main-kernel m-block (kBwdBlockM q rows).
    Phase 1 reads the fragment dump coalesced (16B per thread per
    cell) and scatters single f32s into tile[q][d] — the scatter is
    bank-conflict-free (banks = lane_group + 8*lane_pair cover all
    32). Phase 2: each thread emits contiguous 16-elem (32B) d-slices
    so every warp store covers full 256B rows (full 32B sectors)."""
    comptime D: Int = head_dim
    comptime BM: Int = kBwdTileM(head_dim, causal)
    comptime PAD: Int = 4  # pad smem rows to dodge bank conflicts
    comptime NT: Int = kBwdCvtThreads  # 256

    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var tid: Int = Int(thread_idx.x)

    # Varlen: one CTA per q-tile-table row (m_local, q_row_base,
    # seqlen_q, mpad_base, ...); the table address rides the seq_len
    # slot and nheads carries num_mpad = total_qpad/BM. H comes from
    # grid_dim.y.
    var vl_q_base: Int = 0
    var vl_slq: Int = 0
    var vl_mpad_base: Int = 0
    comptime if varlen:
        var tbl = (
            UnsafePointer[Int32, ImmutAnyOrigin](
                unsafe_from_address=seq_len
            )
            + 8 * Int(block_idx.x)
        )
        m_block = Int(tbl[0])
        vl_q_base = Int(tbl[1])
        vl_slq = Int(tbl[2])
        vl_mpad_base = Int(tbl[3])
        b_idx = 0

    var num_m_blocks: Int = 0
    comptime if not varlen:
        num_m_blocks = (seq_len + BM - 1) // BM

    var tile = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    # Phase 1: decode the fragment dump (see bwd_main_kernel's
    # docstring): per m-block, layout [wg(2)][chunk(BM/8)][tid(128)]
    # [4] f32. Thread (sub, ft) reads the cells (combo = i*2 + sub,
    # ft) — consecutive ft -> consecutive 16B: fully coalesced.
    # Per-WG dQ^T q-columns: BM at D=128 (M split), BM/2 at D=64
    # (N split) — mirrors the main kernel's DQ_N.
    comptime DQ_COLS_WG: Int = BM if D == 128 else BM // 2
    comptime NCH: Int = DQ_COLS_WG // 8  # chunks per wg
    comptime WG_F32: Int = 64 * DQ_COLS_WG  # == (D//2)*BM at both dims
    comptime NCOMBO: Int = 2 * NCH  # wg x ch
    var sub: Int = tid // 128
    var ft: Int = tid % 128
    var frag_base: Int
    comptime if varlen:
        frag_base = (
            h_idx * nheads + vl_mpad_base + m_block
        ) * (2 * WG_F32)
    else:
        frag_base = (
            (b_idx * nheads + h_idx) * num_m_blocks + m_block
        ) * (2 * WG_F32)
    var d_wl: Int = (ft // 32) * 16 + (ft % 32) // 4
    var q_lp: Int = 2 * (ft % 4)
    comptime for i in range(NCOMBO // 2):
        var c: Int = i * 2 + sub
        var wg: Int = c // NCH
        var ch: Int = c % NCH
        var v = (
            dq_accum_ptr
            + frag_base
            + wg * WG_F32
            + ch * (128 * 4)
            + ft * 4
        ).load[width=4]()
        comptime for e in range(4):
            var d: Int
            var q: Int
            comptime if D == 64:
                # Single m64: d has no wg slab; the wg split is on q.
                d = d_wl + 8 * (e // 2)
                q = wg * 64 + ch * 8 + q_lp + (e % 2)
            else:
                d = wg * 64 + d_wl + 8 * (e // 2)
                q = ch * 8 + q_lp + (e % 2)
            tile[q * (D + PAD) + d] = v[e]
    barrier()

    # Phase 2: 8 lanes per row, 16 d (32B bf16) per lane, so every
    # warp store covers 4 full 256B rows — full 32B sectors, fully
    # coalesced. Tail m-block: rows s >= seq_len are pad, skipped.
    comptime OV: Int = 16
    comptime LPR: Int = D // OV  # lanes per row: 8 at D=128, 4 at 64
    comptime ROWS_PER_PASS: Int = NT // LPR
    var row_in_pass: Int = tid // LPR
    var d_base: Int = (tid % LPR) * OV
    comptime for p in range((BM + ROWS_PER_PASS - 1) // ROWS_PER_PASS):
        var s_local: Int = p * ROWS_PER_PASS + row_in_pass
        var s: Int = m_block * BM + s_local
        comptime if varlen:
            # dq is packed (total_q, H, D); rows >= seqlen_q are pad.
            if s_local < BM and s < vl_slq:
                var dq_off: Int = (
                    (vl_q_base + s) * Int(grid_dim.y) + h_idx
                ) * D + d_base
                var fv = (
                    tile + s_local * (D + PAD) + d_base
                ).load[width=OV, alignment=16]()
                var out = (fv * softmax_scale).cast[dtype]()
                (dq_ptr + dq_off).store[width=OV, alignment=32](out)
        else:
            if s_local < BM and s < seq_len:
                var dq_off: Int = (
                    (b_idx * seq_len + s) * nheads + h_idx
                ) * D + d_base
                var fv = (
                    tile + s_local * (D + PAD) + d_base
                ).load[width=OV, alignment=16]()
                var out = (fv * softmax_scale).cast[dtype]()
                (dq_ptr + dq_off).store[width=OV, alignment=32](out)
