"""FA4-target flash-attention forward kernel (sm_90a, Hopper).

v4: warp-specialized like FA4 — 1 producer warpgroup (WG0, thread 0
issues all TMA loads) + 2 MMA warpgroups, 384 threads. K/V tiles
live in a single 6-slot smem ring (slot(K_n) = 2n%6, slot(V_n) =
(2n+1)%6) guarded by full/empty mbarrier pairs:

    full[i].init(1)     -> flipped by TMA expect_bytes completion
    empty[i].init(256)  -> flipped when every MMA thread arrives

The MMA warpgroups run FA4's intra-warpgroup overlap schedule (from
`flash_attn/cute/flash_fwd_sm90.py::mma_one_n_block_intrawg_overlap`):

    wait full K(n+1); commit QK(n+1) -> s_reg        (no wait)
    wait full V(n);   commit PV(n):  p_reg x V -> o_reg
    wait_group(1)   # QK(n+1) retired -> arrive empty[K(n+1)]
    softmax(n+1)    # overlaps PV(n) on the tensor core
    wait_group(0)   # PV(n) retired   -> arrive empty[V(n)]
    pack P(n+1) bf16; rescale o_reg

No block-wide barriers in the loop (v3's main stall). Single S /
single P register buffer keeps the consumer register count near
FA4's 168/thread; producer deallocates to 24 regs via setmaxnreg.

The exp2 uses the scaled-domain trick: rowmax is kept premultiplied
by softmax_scale*log2(e) so P = exp2(fma(s, scale_log2, -m)).

Grid: (ceildiv(seqlen, BM), nheads, batch). Block: 384 threads.

P c-frag -> a-frag mapping: with num_m_mmas=1 per warpgroup the QK
c-fragment element order (16 col-chunks x [top0 top1 bot0 bot1]) is
identical to the PV a-fragment order (8 k_mmas x 8 halves) — a
straight indexwise cast is correct. (With >1 m_mma it would not be:
the RS wgmma walks fragments k-major, `a_frags[m + k*num_m]`.)
"""

from std.math import exp2, log, tanh
from std.math.constants import log2e
from std.sys import size_of
from std.utils.index import StaticTuple, IndexList

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
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
from std.gpu.sync import named_barrier, named_barrier_arrive
from std.memory import bitcast, stack_allocation

from std.gpu.compute.mma import st_matrix

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)

from fa4_wgmma_f16 import wgmma_rs_f16_m64n128, wgmma_rs_f16_m64n64
from layout.tma_async import SharedMemBarrier, TMATensorTile

from fa4_fwd_common import (
    kFa4NThreads,
    kFa4BlockM,
    kFa4BlockN,
    kFa4NMmaWarpgroups,
    kFa4KVStages,
    kFa4ProducerRegs,
    kFa4ConsumerRegs,
)

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kFa4NThreads(head_dim))
    )
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
def fwd_fa4_kernel[
    dtype: DType,
    head_dim: Int,
    qo_rank: Int,
    q_tile_shape: IndexList[qo_rank],
    q_desc_shape: IndexList[qo_rank],
    kv_tile_shape: IndexList[3],
    kv_desc_shape: IndexList[3],
    o_tile_shape: IndexList[qo_rank],
    o_desc_shape: IndexList[qo_rank],
    causal: Bool = False,
    gqa_ratio: Int = 1,
    varlen: Bool = False,
    window: Bool = False,
    window_unaligned: Bool = False,
    softcap_x1000: Int = 0,
](
    q_tma: TMATensorTile[dtype, qo_rank, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    o_tma: TMATensorTile[dtype, qo_rank, o_tile_shape, o_desc_shape],
    lse_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    softmax_scale: Float32,
    nheads: Int,
    sched_swizzle: Int,
    sched_num_hb_q: Int,
    sched_residual: Int,
):
    comptime BM: Int = kFa4BlockM(head_dim)
    comptime BN: Int = kFa4BlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kFa4NMmaWarpgroups(head_dim)
    comptime STAGES: Int = kFa4KVStages
    comptime NUM_PRODUCER_REGS: Int = kFa4ProducerRegs(head_dim)
    comptime NUM_CONSUMER_REGS: Int = kFa4ConsumerRegs(head_dim)
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    comptime q_smem_layout = tile_layout_k_major[
        dtype, BM, D, swizzle_mode=swizzle
    ]()
    comptime k_smem_layout = tile_layout_k_major[
        dtype, BN, D, swizzle_mode=swizzle
    ]()
    comptime v_smem_layout = tile_layout_mn_major[
        dtype, D, BN, swizzle_mode=swizzle
    ]()

    comptime q_smem_size: Int = q_smem_layout.size()
    comptime kv_slot_size: Int = BN * D

    var smem_base = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]()
    var q_smem = LayoutTensor[
        dtype,
        q_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]((smem_base).as_unsafe_any_origin())
    var kv_smem_base = smem_base + q_smem_size

    var mbar_q = stack_allocation[
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
        mbar_q[0].init()
        comptime for s in range(STAGES):
            full[s].init(1)
            empty[s].init(Int32(NWG * 128))
    barrier()

    var m_block: Int
    var h_idx: Int
    var b_idx: Int
    # Varlen per-CTA scalars (0/unused when dense; from the host work
    # table otherwise).
    var vl_q_base: Int = 0
    var vl_k_base: Int = 0
    var vl_seqlen_q: Int = 0
    var vl_seqlen_k: Int = 0
    var vl_win_left: Int = 0
    comptime if varlen:
        # Host work-item table, one int32[8] row per CTA: (m_block,
        # q_row_base, k_row_base, seqlen_q, seqlen_k, _, _, _). Its
        # address rides the sched_swizzle slot (LPT is dense-causal
        # only) and seq_len carries total_q (for the packed LSE
        # layout) — the kernel signature stays identical to dense.
        # Every per-CTA scalar is warp.broadcast-laundered so ptxas
        # sees it warp-uniform (same hazard class as the tid-widening
        # trap; see HANDOFF.md).
        var tbl = (
            UnsafePointer[Int32, ImmutAnyOrigin](
                unsafe_from_address=sched_swizzle
            )
            + 8 * Int(block_idx.x)
        )
        m_block = Int(warp.broadcast(tbl[0]))
        vl_q_base = Int(warp.broadcast(tbl[1]))
        vl_k_base = Int(warp.broadcast(tbl[2]))
        vl_seqlen_q = Int(warp.broadcast(tbl[3]))
        vl_seqlen_k = Int(warp.broadcast(tbl[4]))
        comptime if window:
            vl_win_left = Int(warp.broadcast(tbl[5]))
        h_idx = Int(block_idx.y)
        b_idx = 0
    else:
        comptime if causal and not window:
            # FA4's SingleTileLPTScheduler (static, non-persistent):
            # flat 1-D grid; heaviest m_blocks launch FIRST (LPT
            # reversal) and sched_swizzle (head,batch) pairs sweep
            # each m together so their K/V tiles stay L2-resident.
            var bx: Int = Int(block_idx.x)
            var num_m: Int = (seq_len + BM - 1) // BM
            var l2_major: Int = num_m * sched_swizzle
            var bidhb: Int = bx // l2_major
            var l2_mod: Int = bx - bidhb * l2_major
            var dvsr: Int = (
                sched_swizzle if bidhb < sched_num_hb_q else sched_residual
            )
            var blk: Int = l2_mod // dvsr
            var res: Int = l2_mod - blk * dvsr
            var bidhb_act: Int = bidhb * sched_swizzle + res
            b_idx = bidhb_act // nheads
            h_idx = bidhb_act - b_idx * nheads
            m_block = num_m - 1 - blk
        else:
            m_block = Int(block_idx.x)
            h_idx = Int(block_idx.y)
            b_idx = Int(block_idx.z)
    # Cross-attention diagonal offset (FA4's bottom-right
    # alignment): row i attends col j iff j <= i + (slk - slq).
    # 0 for self-attention; only the causal arms consume it. v1
    # envelope: slq <= slk per sequence under causal
    # (host-asserted), so every q row attends >= 1 key.
    var vl_offs: Int = 0
    comptime if varlen and causal:
        vl_offs = vl_seqlen_k - vl_seqlen_q
    var num_kv_blocks: Int
    comptime if varlen:
        num_kv_blocks = (vl_seqlen_k + BN - 1) // BN
    else:
        num_kv_blocks = (seq_len + BN - 1) // BN
    comptime if causal:
        comptime if varlen:
            # Cross-attention general form (bottom-right diagonal):
            # row block m attends kv cols < (m+1)*BM + (slk - slq),
            # so the band may straddle two tiles even at BM == BN
            # (offs % BN != 0) — varlen causal always takes the
            # BAND mask arms below. Self-attn (offs == 0) degrades
            # to the dense count with one extra no-op masked tile.
            num_kv_blocks = min(
                num_kv_blocks,
                ((m_block + 1) * BM + vl_offs + BN - 1) // BN,
            )
        elif BM == BN:
            # BM == BN: row block m attends KV tiles 0..m inclusive;
            # tile n == m_block is the (only) masked diagonal tile.
            num_kv_blocks = min(num_kv_blocks, m_block + 1)
        else:
            # BM=192 > BN=128 (hdim64): row block m attends kv cols
            # < (m+1)*BM, i.e. ceil((m+1)*BM/BN) tiles; the diagonal
            # BAND spans the last TWO tiles of the trip range.
            num_kv_blocks = min(
                num_kv_blocks,
                ((m_block + 1) * BM + BN - 1) // BN,
            )

    # Sliding window (causal, any left >= 1): the kv trip range
    # gains a LOWER bound. The leading edges for an m-tile's rows
    # span a BM-wide range, which straddles at most TWO kv tiles
    # (BM == BN): the prologue tile and, when left % BN != 0, the
    # first loop tile — both get the leading mask; the steady loop
    # stays mask-free. (Aligned left makes the second tile's mask a
    # provable no-op.)
    var win_left: Int = 0
    var first_kv: Int = 0
    var kv_trips: Int = num_kv_blocks
    comptime if window:
        comptime assert causal, "window v1 requires causal"
        comptime if varlen:
            # sched_swizzle carries the work table under varlen;
            # win_left rides the table's free col 5 instead. The
            # bottom-right offset shifts the leading edge: attended
            # j ∈ [i + offs - left, i + offs].
            win_left = vl_win_left
        else:
            win_left = sched_swizzle  # rides the (free) LPT slot
        first_kv = max(
            0, (m_block * BM + vl_offs - win_left) // BN
        )
        kv_trips = num_kv_blocks - first_kv

    # Varlen ragged kv tail: garbage columns live only in the
    # sequence's LAST kv tile (kv tiles are seq-local). Non-causal
    # masks them there; causal needs NO extra mask — bottom-right
    # alignment means garbage col j >= slk is attended only by rows
    # i >= j - offs >= slk - (slk - slq) = slq, i.e. never by a
    # STORED row; and the last kv tile, when processed, is always
    # within the band-masked trailing trips, whose col + mask_d >
    # row predicate covers exactly j > i + offs.
    var vl_kv_tail: Int = 0
    comptime if varlen and not causal:
        vl_kv_tail = vl_seqlen_k - (num_kv_blocks - 1) * BN

    # shfl-broadcast warpgroup index (the bwd's tid-widening trap:
    # ptxas's tid-uniformity rule only matches 32-bit shr.u32, and
    # LLVM re-widens any 32-bit extract; the convergent shfl is
    # opaque to LLVM and a recognized broadcast to ptxas — without
    # it every wg-derived descriptor offset costs R2UR per HGMMA).
    var wgid: Int = Int(
        warp.broadcast(Int32(Int(thread_idx.x) >> 7))
    )

    if wgid == 0:
        # ================= producer =================
        warpgroup_reg_dealloc[NUM_PRODUCER_REGS]()
        if thread_idx.x == 0:
            mbar_q[0].expect_bytes(Int32(BM * D * size_of[dtype]()))
            comptime if qo_rank == 4:
                # Dense hdim64: S is its own TMA dim (BM=192 does not
                # divide the seqlen envelope; OOB tail rows zero-fill
                # here and the store-side clamps).
                q_tma.async_copy_4d(
                    q_smem,
                    mbar_q[0],
                    (0, h_idx, m_block * BM, b_idx),
                )
            elif varlen:
                q_tma.async_copy_3d(
                    q_smem, mbar_q[0], (0, h_idx, vl_q_base + m_block * BM)
                )
            else:
                q_tma.async_copy_3d(
                    q_smem,
                    mbar_q[0],
                    (0, h_idx, b_idx * seq_len + m_block * BM),
                )
            # Incremental ring state: K(n) in slot 2n%6, V(n) in
            # (2n+1)%6; the empty-barrier phase flips every 3 tiles.
            # Varlen non-causal walks the kv tiles in REVERSE so the
            # sequence's ragged-tail (boundary) tile is processed in
            # the consumer PROLOGUE — its column mask then never
            # touches the steady loop (FA4's design: the hot loop
            # stays mask-free; online softmax is order-independent).
            var slot: Int = 0
            var phase: UInt32 = 0
            var wrap: Int = 0
            var row: Int
            var row_step: Int = BN
            comptime if window:
                comptime if varlen:
                    row = vl_k_base + first_kv * BN
                else:
                    row = b_idx * seq_len + first_kv * BN
            elif varlen:
                comptime if causal:
                    row = vl_k_base
                else:
                    row = vl_k_base + (num_kv_blocks - 1) * BN
                    row_step = -BN
            else:
                row = b_idx * seq_len
            for _ in range(kv_trips):
                empty[slot].wait(phase)
                var k_st = LayoutTensor[
                    dtype,
                    k_smem_layout,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ]((kv_smem_base + slot * kv_slot_size).as_unsafe_any_origin())
                full[slot].expect_bytes(Int32(BN * D * size_of[dtype]()))
                k_tma.async_copy_3d(
                    k_st, full[slot], (0, h_idx // gqa_ratio, row)
                )

                empty[slot + 1].wait(phase)
                var v_st = LayoutTensor[
                    dtype,
                    v_smem_layout,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ]((kv_smem_base + (slot + 1) * kv_slot_size).as_unsafe_any_origin())
                full[slot + 1].expect_bytes(Int32(BN * D * size_of[dtype]()))
                v_tma.async_copy_3d(
                    v_st, full[slot + 1], (0, h_idx // gqa_ratio, row)
                )

                row += row_step
                slot += 2
                wrap += 1
                if wrap == 3:
                    wrap = 0
                    slot = 0
                    phase ^= 1
        return

    # ================= MMA warpgroups =================
    warpgroup_reg_alloc[NUM_CONSUMER_REGS]()
    var wg: Int = wgid - 1

    # Unblock the producer's first ring cycle.
    comptime for s in range(STAGES):
        _ = empty[s].arrive()

    # Scheduler-pingpong barrier participant count: each barrier id
    # is waited by ONE warpgroup (128) and armed by its predecessor
    # (128) — 256 regardless of NWG (== NWG*128 only at NWG == 2).
    comptime SCHED_BAR_N: Int = 2 * 128

    # Warp-scheduler pingpong (FA4's use_scheduler_barrier): named
    # barrier 1+wg gates each warpgroup's GEMM-issue phase; a
    # warpgroup arrives at the *other* one's barrier after committing
    # its GEMM pair, so issue phases alternate and each warpgroup's
    # softmax overlaps the other's GEMMs. WG0 self-arms its barrier.
    if wg == 0:
        named_barrier_arrive[Int32(SCHED_BAR_N)](Int32(1))

    var wgmma_qk = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, BN, WGMMA_K),
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        transpose_b=True,
    ]()
    var wgmma_pv = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, D, WGMMA_K),
        a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle,
        transpose_b=False,
    ]()

    comptime c_frag_size_qk: Int = WGMMA_M * BN // 128  # 64
    comptime c_frag_size_pv: Int = WGMMA_M * D // 128  # 64
    comptime a_frag_size_pv: Int = WGMMA_M * WGMMA_K // 128  # 8
    comptime num_k_mmas_pv: Int = BN // WGMMA_K  # 8

    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_size_qk),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var o_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_size_pv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    _ = o_reg.fill(0)
    var p_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_pv, a_frag_size_pv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    # Online-softmax state, kept in the scaled (log2) domain:
    # rowmax_s = max(S) * softmax_scale * log2(e). 2 rows per thread.
    comptime rows_per_thread: Int = 2
    var rowmax = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var scale_old = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var neg_inf: Scalar[accum_type] = Scalar[accum_type](-1.0e30)
    comptime for i in range(rows_per_thread):
        rowmax[i] = neg_inf
        rowsum[i] = Scalar[accum_type](0)

    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()
    # Softcap (Gemma-2): S_capped = cap * tanh(S_raw * scale / cap),
    # FA4 semantics (scale applied BEFORE the tanh, score_mod
    # pre-mask). The cap is a COMPTIME constant (one JIT variant per
    # cap value — it is a model-architecture constant), so it costs
    # no kernel arg slot and composes with window/GQA/varlen. The
    # softmax then runs in the capped domain: s_reg holds
    # t = tanh(s*scale/cap) and scale_log2 is repointed at
    # cap * log2(e), so the existing max/exp2 sites fold the cap
    # back in unchanged.
    comptime softcap_on: Bool = softcap_x1000 != 0
    var t_scale: Scalar[accum_type] = Scalar[accum_type](0)
    comptime if softcap_on:
        comptime cap_f32: Float32 = Float32(softcap_x1000) / 1000
        t_scale = (softmax_scale / cap_f32).cast[accum_type]()
        scale_log2 = (
            cap_f32 * Scalar[DType.float32](log2e)
        ).cast[accum_type]()

    @parameter
    @always_inline
    def k_tile(slot: Int) -> LayoutTensor[
        dtype,
        k_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]:
        return {(kv_smem_base + slot * kv_slot_size).as_unsafe_any_origin()}

    @parameter
    @always_inline
    def v_tile(slot: Int) -> LayoutTensor[
        dtype,
        v_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]:
        return {(kv_smem_base + slot * kv_slot_size).as_unsafe_any_origin()}

    # fp16 RS fork: the stdlib's register-A wgmma overload is
    # bf16-only (hardcoded .bf16.bf16 asm); the vendored
    # m64n128k16 f32.f16.f16 emitter (_wgmma_f16.mojo) replicates
    # the TensorCoreAsync RS k-loop here. bf16 keeps the stdlib
    # path — byte-identical codegen.
    comptime v_canonical = tile_to_descriptor[
        dtype, v_smem_layout, False
    ]()
    comptime v_k_stride: Int = (
        v_canonical[1].stride[1].value() * 2 * size_of[dtype]()
    )

    @parameter
    @always_inline
    def pv_gemm(slot_arg: Int):
        comptime if dtype == DType.float16:
            var b_desc = _wgmma_descriptor[v_canonical, False, swizzle](
                kv_smem_base + slot_arg * kv_slot_size
            )
            var o_simd = o_reg.ptr.load[width=c_frag_size_pv]()
            comptime for k_mma in range(num_k_mmas_pv):
                comptime if head_dim == 128:
                    o_simd = rebind[SIMD[accum_type, c_frag_size_pv]](
                        wgmma_rs_f16_m64n128(
                            rebind[SIMD[DType.float16, 8]](
                                (p_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (b_desc + k_mma * v_k_stride).desc,
                            rebind[SIMD[DType.float32, 64]](o_simd),
                        )
                    )
                else:
                    o_simd = rebind[SIMD[accum_type, c_frag_size_pv]](
                        wgmma_rs_f16_m64n64(
                            rebind[SIMD[DType.float16, 8]](
                                (p_reg.ptr + 8 * k_mma).load[width=8]()
                            ),
                            (b_desc + k_mma * v_k_stride).desc,
                            rebind[SIMD[DType.float32, 32]](o_simd),
                        )
                    )
            o_reg.ptr.store[width=c_frag_size_pv](o_simd)
        else:
            wgmma_pv.wgmma(p_reg, v_tile(slot_arg), o_reg)

    # c-frag (row, col) roots for the diagonal mask (within-tile
    # coordinates; global q = m*BM + row, kv = n*BN + col, and on the
    # diagonal tile n == m_block the mask is simply col > row).
    var mask_row_lo: Int = (
        wg * WGMMA_M + (Int(warp_id()) % 4) * 16 + Int(lane_id()) // 4
    )
    var mask_col_lo: Int = 2 * (Int(lane_id()) % 4)
    # BM != BN causal: per-tile global offset (col_g - row_g = col +
    # mask_d - row with mask_d = n*BN - m*BM), set before each
    # masked-tile softmax call. Unused (and DCE'd) when BM == BN.
    var causal_mask_d: Int = 0
    # Window leading-edge offset: mask col + win_mask_d < row on the
    # leading tile(s); per-tile, advanced by BN for the second
    # masked tile (first loop trip).
    var win_mask_d: Int = 0
    comptime if window:
        win_mask_d = first_kv * BN - m_block * BM + win_left
        comptime if varlen:
            # Bottom-right alignment: masked iff j < i + offs - left.
            win_mask_d -= vl_offs

    @parameter
    @always_inline
    def softmax_block(mask_diag: Bool, mask_tail: Bool):
        """Online softmax over s_reg (S just retired): update
        rowmax/rowsum/scale_old, write P (f32) back into s_reg.
        mask_diag (causal only): apply the diagonal-tile mask first.
        mask_tail (varlen non-causal only): this is the sequence's
        last kv tile — mask the ragged-tail garbage columns."""
        comptime if softcap_on:
            # Cap BEFORE the mask arms (FA4 applies score_mod
            # pre-mask; masking after keeps -1e30 * cap_log2 = a
            # true -inf in the exp2). tanh lowers to the sm90 HW
            # tanh.approx.f32 — FA4's "fastmath" tanh emulates via
            # ex2 and pays 3.1x kernel time for it.
            comptime for c in range(c_frag_size_qk):
                s_reg.ptr[c] = tanh(s_reg.ptr[c] * t_scale)
        comptime if causal:
            comptime if BM == BN and not varlen:
                if mask_diag:
                    comptime for c in range(c_frag_size_qk):
                        comptime col_base: Int = (c // 4) * 8 + (c & 1)
                        comptime row_off: Int = 8 if (c % 4) >= 2 else 0
                        if col_base + mask_col_lo > mask_row_lo + row_off:
                            s_reg.ptr[c] = neg_inf
            else:
                # Diagonal-band tile: global mask col + mask_d > row.
                # (Varlen causal always uses this arm — the
                # cross-attention offset shifts the diagonal off the
                # n == m tile and the band may straddle two tiles.)
                if mask_diag:
                    comptime for c in range(c_frag_size_qk):
                        comptime col_base: Int = (c // 4) * 8 + (c & 1)
                        comptime row_off: Int = 8 if (c % 4) >= 2 else 0
                        if (
                            col_base + mask_col_lo + causal_mask_d
                            > mask_row_lo + row_off
                        ):
                            s_reg.ptr[c] = neg_inf
        comptime if window:
            # Leading window edge (prologue tile only): cols before
            # row - left are outside the window.
            if mask_tail:
                comptime for c in range(c_frag_size_qk):
                    comptime col_base: Int = (c // 4) * 8 + (c & 1)
                    comptime row_off: Int = 8 if (c % 4) >= 2 else 0
                    if (
                        col_base + mask_col_lo + win_mask_d
                        < mask_row_lo + row_off
                    ):
                        s_reg.ptr[c] = neg_inf
        comptime if varlen and not causal:
            if mask_tail and vl_kv_tail < BN:
                comptime for c in range(c_frag_size_qk):
                    comptime col_base: Int = (c // 4) * 8 + (c & 1)
                    if col_base + mask_col_lo >= vl_kv_tail:
                        s_reg.ptr[c] = neg_inf
        var local_max = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        comptime for i in range(rows_per_thread):
            local_max[i] = neg_inf
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            local_max[row_idx] = max(local_max[row_idx], s_reg.ptr[c])
        comptime for i in range(rows_per_thread):
            local_max[i] = warp.lane_group_max[num_lanes=4](local_max[i])
            var rmax_new: Scalar[accum_type] = max(
                local_max[i] * scale_log2, rowmax[i]
            )
            scale_old[i] = exp2(rowmax[i] - rmax_new)
            rowmax[i] = rmax_new

        var local_sum = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        comptime for i in range(rows_per_thread):
            local_sum[i] = Scalar[accum_type](0)
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            var p: Scalar[accum_type] = exp2(
                s_reg.ptr[c].fma(scale_log2, -rowmax[row_idx])
            )
            s_reg.ptr[c] = p
            local_sum[row_idx] += p
        comptime for i in range(rows_per_thread):
            rowsum[i] = rowsum[i] * scale_old[i] + local_sum[i]

    @parameter
    @always_inline
    def pack_p():
        comptime for c in range(c_frag_size_qk):
            p_reg.ptr[c] = s_reg.ptr[c].cast[dtype]()

    @parameter
    @always_inline
    def rescale_o():
        comptime for c in range(c_frag_size_pv):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            o_reg.ptr[c] *= scale_old[row_idx]

    # ---- Prologue: S(0) -> P(0).
    mbar_q[0].wait(UInt32(0))
    full[0].wait(UInt32(0))
    warpgroup_fence(s_reg)
    wgmma_qk.arrive()
    wgmma_qk.wgmma[num_warp_groups=NWG, scale_c=0](
        q_smem, k_tile(0), s_reg, wg
    )
    wgmma_qk.commit_group()
    wgmma_qk.wait_group()
    warpgroup_fence(s_reg)
    _ = empty[0].arrive()

    # rowmax starts at -inf -> scale_old==0, rowsum init. For causal,
    # m_block 0's single tile IS the diagonal (BM==BN) or may sit in
    # the 2-tile diagonal band (BM>BN). Varlen non-causal walks kv in
    # reverse, so the FIRST tile here is the sequence's ragged-tail
    # (boundary) tile — the only one that needs the column mask; the
    # steady loop below stays mask-free.
    var prologue_diag: Bool = False
    comptime if causal:
        comptime if BM == BN and not varlen:
            prologue_diag = kv_trips == 1
        else:
            prologue_diag = kv_trips <= 2
            causal_mask_d = -m_block * BM
            comptime if varlen:
                causal_mask_d -= vl_offs
            comptime if window:
                # The prologue tile is first_kv, not 0.
                causal_mask_d += first_kv * BN
    softmax_block(prologue_diag, True)
    pack_p()  # P(0)
    # The window leading edge can straddle into the FIRST loop tile
    # when left % BN != 0: advance the mask offset ONCE here (the
    # prologue consumed the first tile's value). COMPTIME-split:
    # aligned lefts keep the loop's inlined softmax_block free of
    # the window arm entirely (constant-False flag, DCE'd) — having
    # the arm merely PRESENT behind a runtime flag cost a
    # consistent 2-4% at the canonical aligned config.
    comptime if window_unaligned:
        win_mask_d += BN

    # ---- Main loop: QK(n+1) + PV(n) per iteration. Ring slots and
    # empty-barrier phases track incrementally (no div/mod per iter):
    # K(t): slot 2t%6, V(t): (2t+1)%6, phase flips every 3 tiles.
    var k_slot: Int = 2  # K(1)
    var k_phase: UInt32 = 0
    var k_wrap: Int = 1
    var v_slot: Int = 1  # V(0)
    var v_phase: UInt32 = 0
    var v_wrap: Int = 0

    for it in range(kv_trips - 1):
        # Queue QK(n+1) then PV(n) on the tensor core.
        full[k_slot].wait(k_phase)
        named_barrier[Int32(SCHED_BAR_N)](Int32(1 + wg))
        warpgroup_fence(s_reg)
        wgmma_qk.arrive()
        wgmma_qk.wgmma[num_warp_groups=NWG, scale_c=0](
            q_smem, k_tile(k_slot), s_reg, wg
        )
        wgmma_qk.commit_group()

        full[v_slot].wait(v_phase)
        warpgroup_fence(o_reg)
        wgmma_pv.arrive()
        pv_gemm(v_slot)
        wgmma_pv.commit_group()
        # Arrive at the SUCCESSOR's sync barrier: ring W0->W1->...->W0.
        comptime if NWG == 2:
            named_barrier_arrive[Int32(SCHED_BAR_N)](Int32(2 - wg))
        else:
            named_barrier_arrive[Int32(SCHED_BAR_N)](
                Int32(wg + 2 if wg < NWG - 1 else 1)
            )

        # QK(n+1) retired (PV(n) still running on the tensor core).
        wgmma_qk.wait_group[1]()
        warpgroup_fence(s_reg)
        _ = empty[k_slot].arrive()

        # Softmax of S(n+1) overlaps PV(n). For causal the last
        # tile (BM==BN: n+1 == num_kv_blocks-1 == m_block is THE
        # diagonal; BM>BN: the last TWO tiles form the band).
        # (Varlen's tail mask ran in the prologue — reverse order.)
        var loop_diag: Bool = False
        comptime if causal:
            comptime if BM == BN and not varlen:
                loop_diag = it == kv_trips - 2
            else:
                loop_diag = it >= kv_trips - 3
                causal_mask_d = (it + 1) * BN - m_block * BM
                comptime if varlen:
                    causal_mask_d -= vl_offs
                comptime if window:
                    # Loop trip it processes tile first_kv + it + 1.
                    causal_mask_d += first_kv * BN
        # Window, unaligned left only: the first loop tile can hold
        # leading-edge columns (win_mask_d was advanced after the
        # prologue). Aligned variants pass a comptime False so the
        # window arm is DCE'd from the loop's inlined copy.
        var loop_lead: Bool = False
        comptime if window_unaligned:
            loop_lead = it == 0
        softmax_block(loop_diag, loop_lead)

        # PV(n) retired: p_reg and o_reg are safe to touch.
        wgmma_pv.wait_group[0]()
        warpgroup_fence(o_reg)
        _ = empty[v_slot].arrive()
        pack_p()  # P(n+1)
        rescale_o()

        k_slot += 2
        k_wrap += 1
        if k_wrap == 3:
            k_wrap = 0
            k_slot = 0
            k_phase ^= 1
        v_slot += 2
        v_wrap += 1
        if v_wrap == 3:
            v_wrap = 0
            v_slot = 1
            v_phase ^= 1

    # ---- Epilogue: PV(N-1) (v_slot/v_phase left at tile N-1).
    full[v_slot].wait(v_phase)
    warpgroup_fence(o_reg)
    wgmma_pv.arrive()
    pv_gemm(v_slot)
    wgmma_pv.commit_group()
    wgmma_pv.wait_group[0]()
    warpgroup_fence(o_reg)

    # ---- Normalize (reciprocal; one div per row) and store.
    var inv_rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    comptime for i in range(rows_per_thread):
        rowsum[i] = warp.lane_group_sum[num_lanes=4](rowsum[i])
        inv_rowsum[i] = Scalar[accum_type](1) / rowsum[i]

    comptime for c in range(c_frag_size_pv):
        comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
        o_reg.ptr[c] *= inv_rowsum[row_idx]

    # ---- Store: stmatrix-stage O into the dead Q tile (same SW128
    # k-major layout it was loaded with — the bwd dK/dV epilogue's
    # scheme) and issue ONE whole-tile TMA store. 8x stmatrix.x4
    # (non-trans) per thread; the 128B swizzle XOR is taken from the
    # ABSOLUTE address and re-applied per call (32-B column steps
    # live in the swizzled bits 4-6).
    var lane: Int = Int(lane_id())
    var warp_in_wg: Int = Int(warp_id()) % 4
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4
    var row_warp_base: Int = wg * WGMMA_M + warp_in_wg * 16

    # Varlen ragged q tail: the sequence's last m-tile may be partial
    # (vl_rows < BM). A full-tile TMA store would overwrite the NEXT
    # sequence's rows, so the partial tile bypasses the smem staging
    # and stores its c-frags straight to gmem, row-predicated (the O
    # raw pointer rides the sched_num_hb_q slot; boundary-tile-only,
    # at most one per sequence).
    var vl_rows: Int = BM
    comptime if varlen:
        vl_rows = vl_seqlen_q - m_block * BM
    # Dense hdim64 (BM=192): the last m-tile is partial whenever
    # seq_len % BM != 0 — rank-4 TMA clamps the O store, but the LSE
    # writes still need the row predicate (vl_rows doubles as the
    # valid-row count for both paths).
    comptime if qo_rank == 4:
        vl_rows = seq_len - m_block * BM

    var full_tile_store: Bool = True
    comptime if varlen:
        if vl_rows < BM:
            full_tile_store = False
            var o_gptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=sched_num_hb_q
            )
            var o_row0: Int = vl_q_base + m_block * BM
            comptime for c2 in range(c_frag_size_pv // 2):
                comptime col_chunk: Int = c2 // 2
                comptime is_bot: Int = c2 % 2
                var row: Int = (
                    row_warp_base + lane_group + (8 if is_bot == 1 else 0)
                )
                if row < vl_rows:
                    var pair = SIMD[dtype, 2](
                        o_reg.ptr[2 * c2].cast[dtype](),
                        o_reg.ptr[2 * c2 + 1].cast[dtype](),
                    )
                    (
                        o_gptr
                        + ((o_row0 + row) * nheads + h_idx) * D
                        + col_chunk * 8
                        + 2 * lane_pair
                    ).store[width=2, alignment=4](pair)

    if full_tile_store:
        comptime if D == 64:
            # D=64: stage via plain paired stores at the canonical
            # SW128 k-major addresses (64-elem rows = one 128-B
            # swizzle period; XOR re-applied per 16-B store). The
            # stmatrix scheme below encodes D=128 geometry —
            # revisit only if the parity bench demands it.
            comptime for c2 in range(c_frag_size_pv // 2):
                comptime col_chunk: Int = c2 // 2
                comptime is_bot: Int = c2 % 2
                var row: Int = (
                    row_warp_base + lane_group + (8 if is_bot == 1 else 0)
                )
                var col: Int = col_chunk * 8 + 2 * lane_pair
                var pair = SIMD[dtype, 2](
                    o_reg.ptr[2 * c2].cast[dtype](),
                    o_reg.ptr[2 * c2 + 1].cast[dtype](),
                )
                var b_addr: Int = (
                    Int(smem_base)
                    + (row >> 3) * 1024
                    + (row & 7) * 128
                    + 2 * col
                )
                var b_sw: Int = b_addr ^ ((b_addr >> 3) & 112)
                (
                    smem_base + ((b_sw >> 1) - (Int(smem_base) >> 1))
                ).store[width=2, alignment=4](pair)
        else:
            var st_row: Int = (
                row_warp_base + ((lane // 8) % 2) * 8 + (lane % 8)
            )
            var st_off_raw: Int = st_row * 64 + (lane // 16) * 8
            var o_raw: Int = Int(smem_base) + 2 * st_off_raw
            comptime for i in range(c_frag_size_pv // 8):
                var packed = SIMD[DType.float32, 4](0)
                comptime for jm in range(4):
                    comptime p: Int = 4 * i + jm
                    packed[jm] = bitcast[DType.float32, 1](
                        SIMD[accum_type, 2](
                            o_reg.ptr[2 * p], o_reg.ptr[2 * p + 1]
                        ).cast[dtype]()
                    )
                var raw_i: Int = (
                    o_raw + (i % 4) * 32 + (i // 4) * (BM * 128)
                )
                var sw_i: Int = raw_i ^ ((raw_i >> 3) & 112)
                # .bitcast[BFloat16]: the stdlib st_matrix comptime-
                # asserts bf16/f32, but stmatrix.b16 is dtype-agnostic
                # (raw 16-bit stores; the payload is already bit-packed)
                # — the cast unblocks fp16 and is a no-op for bf16.
                st_matrix[simd_width=4](
                    (
                        smem_base
                        + ((sw_i >> 1) - (Int(smem_base) >> 1))
                    ).bitcast[BFloat16](),
                    packed,
                )

    # ---- LSE (natural log), one f32 per row: rowmax is kept in the
    # scaled log2 domain (max*scale*log2e) and rowsum is already
    # row-reduced, so lse = rowmax*ln2 + ln(rowsum). lane_pair 0
    # writes its thread's two rows; (B, H, S) f32, grid.y == nheads.
    if lane_pair == 0:
        var lse_row_base: Int
        comptime if varlen:
            # Packed (H, total_q) layout; seq_len carries total_q.
            lse_row_base = h_idx * seq_len + vl_q_base + m_block * BM
        else:
            lse_row_base = (b_idx * nheads + h_idx) * seq_len + m_block * BM
        comptime LN2: Scalar[accum_type] = 0.6931471805599453
        comptime for i in range(rows_per_thread):
            var r: Int = row_warp_base + lane_group + 8 * i
            # Varlen: rows past the sequence's end belong to the NEXT
            # sequence's packed LSE rows — predicate them off.
            var lse_ok: Bool = True
            comptime if varlen or qo_rank == 4:
                lse_ok = r < vl_rows
            if lse_ok:
                (lse_ptr + lse_row_base + r)[0] = (
                    rowmax[i] * LN2 + log(rowsum[i])
                ).cast[DType.float32]()

    fence_async_view_proxy()
    # Producer warpgroup may have exited -> consumer-only barrier.
    # (id NWG+1: ids 1..NWG are the scheduler pingpong barriers.)
    named_barrier[Int32(NWG * 128)](Int32(NWG + 1))
    if full_tile_store and thread_idx.x == 128:
        var o_st = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((smem_base).as_unsafe_any_origin())
        comptime if qo_rank == 4:
            # S its own dim: the partial tail tile clamps in hardware.
            o_tma.async_store_4d(
                o_st, (0, h_idx, m_block * BM, b_idx)
            )
        elif varlen:
            o_tma.async_store_3d(
                o_st, (0, h_idx, vl_q_base + m_block * BM)
            )
        else:
            o_tma.async_store_3d(
                o_st, (0, h_idx, b_idx * seq_len + m_block * BM)
            )
        o_tma.commit_group()
        o_tma.wait_group()
