# ============================================================================
# PHASE 2c: self-loading single-warpgroup CTA, d64 dense-causal BHSD ONLY.
# A NEW module (not a replacement): fa4_fwd_kernel.mojo (phase 2b) still
# serves head_dim=128 and, below the wave-gate threshold in fa4_ops.mojo,
# the few-wave d64 shapes too. See NOTES.md "Phase 2c" for the merge
# options this module deliberately did not take (see item 1 of agent A's
# handoff there: a shared-body `self_load: Bool` comptime parameter is
# possible but was not done here, to keep every phase-2b instantiation
# byte-identical without re-running the full asm-diff obligation that
# merge would owe).
# ============================================================================
"""FA4-target flash-attention forward kernel (sm_90a, Hopper) -- the
self-loading single-warpgroup CTA (phase 2c).

The CTA is ONE warpgroup of 128 threads that issues its own TMA loads --
no producer warpgroup. That is what buys a THIRD CTA per SM: at the
phase-2b kernel's 256 threads, `nvvm.minctasm`=3 hands ptxas a static
budget of 65536/(3*256) = 80 registers per thread and it refuses to
compile (setmaxnreg moves registers BETWEEN warps, it does not lift the
CTA's static allocation); at 128 threads the budget is 65536/(3*128) =
170 -> 168, and ptxas settles at 154, so registers AND smem (4 ring
stages: 8 KiB Q + 4*16 KiB = 72 KiB, 3*72 <= 227 KiB) both allow 3.

K/V tiles live in a 4-slot smem ring (slot(K_n) = 2n%4, slot(V_n) =
(2n+1)%4) guarded by `full` mbarriers only:

    full[i].init(1)     -> flipped by TMA expect_bytes completion

The empty[] barriers are GONE. The warpgroup that reads a slot is the
one that refills it, so program order plus `wgmma.wait_group` already
orders the refill after the read: a warp's wait_group returns only once
the whole warpgroup's group has retired, i.e. after every warp's share
of that wgmma has finished reading smem. Refills are issued exactly at
the two points where that proof lands -- after `wait_group[1]` (K of the
trip just consumed) and after `wait_group[0]` (V) -- so the prefetch
distance is a constant STAGES/2 = 2 tiles. See the PREFETCH invariant
comment below, next to its definition, before touching STAGES or the
refill call sites.

The MMA warpgroup runs FA4's intra-warpgroup overlap schedule (from
`flash_attn/cute/flash_fwd_sm90.py::mma_one_n_block_intrawg_overlap`):

    wait full K(n+1); commit QK(n+1) -> s_reg        (no wait)
    wait full V(n);   commit PV(n):  p_reg x V -> o_reg
    wait_group(1)   # QK(n+1) retired -> refill K's slot (was: arrive empty[K(n+1)])
    softmax(n+1)    # overlaps PV(n) on the tensor core
    wait_group(0)   # PV(n) retired   -> refill V's slot (was: arrive empty[V(n)])
    pack P(n+1) bf16; rescale o_reg

No block-wide barriers in the loop (v3's main stall). Single S / single P
register buffer keeps the register count near FA4's target; there is no
producer warpgroup left to deallocate registers for via setmaxnreg.

The exp2 uses the scaled-domain trick: rowmax is kept premultiplied
by softmax_scale*log2(e) so P = exp2(fma(s, scale_log2, -m)).

Grid: (ceildiv(seqlen, BM), nheads, batch). Block: 128 threads.

P c-frag -> a-frag mapping: with num_m_mmas=1 per warpgroup the QK
c-fragment element order (16 col-chunks x [top0 top1 bot0 bot1]) is
identical to the PV a-fragment order (8 k_mmas x 8 halves) — a
straight indexwise cast is correct. (With >1 m_mma it would not be:
the RS wgmma walks fragments k-major, `a_frags[m + k*num_m]`.)
"""

from std.math import exp2, log, tanh
from std.math.constants import log2e
from std.sys import inlined_assembly, size_of
from std.utils.index import StaticTuple, IndexList

from max.gpu.sync import barrier
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, lane_id, thread_idx, warp_id
import std.gpu.primitives.warp as warp
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory, fence_async_view_proxy
from std.memory import AddressSpace
from max.gpu.sync import named_barrier, named_barrier_arrive
from std.memory import bitcast, stack_allocation

from max.gpu.compute.mma import st_matrix

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

from fa4_fwd_selfload_common import (
    kFa4NThreads,
    kFa4BlockM,
    kFa4BlockN,
    kFa4NMmaWarpgroups,
    kFa4KVStages,
    kFa4CtasPerSm,
)

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16


@always_inline
def wgmma_wait_group_ordered[group: Int = 0]():
    """`wgmma.wait_group.sync.aligned` that is also a COMPILER ordering point.

    Identical PTX to the stdlib's `wgmma_wait_group_sync` (same one
    instruction, zero extra cost) with one addition: `~{memory}` in the
    constraint string, so LLVM is told the wait may read and write memory
    and may not move memory operations across it.

    Why this kernel needs that and the stdlib call does not suffice.
    This kernel has no empty[] mbarriers: the proof that a ring slot's
    reader has retired before the TMA refills that slot is
    `wgmma.wait_group` plus PROGRAM ORDER (see the PREFETCH invariant
    comment in the kernel body). The PTX ISA backs that proof:

      * "The wgmma.mma_async operations are performed in the asynchronous
        proxy (or async proxy)" (PTX ISA 9.7.16.4), and TMA
        (`cp.async.bulk.tensor`) writes are async-proxy too -- so this is
        an async->async write-after-read, NOT a proxy crossing.
        `fence.proxy.async` (what `fence_async_view_proxy()` emits) is
        defined only "between the async proxy and the generic proxy"
        (9.7.14.4), so it is the wrong instrument here and would order
        nothing: it is deliberately NOT used at the refill sites.
      * "Once the wgmma-group completes, all the wgmma.mma_async
        operations have been performed and completed" (9.7.16) -- the
        reads are done when the wait returns.
      * "These asynchronous operations are ordered after prior
        instructions in the same thread" (8.9.1.1, and cp.async.bulk is
        not among the exceptions) -- the refill cannot start before the
        wait that precedes it.

    Every link in that chain is a statement about the order the two
    instructions appear in, which is exactly the property the compiler is
    free to change: the stdlib's `wgmma_wait_group_sync` is
    `inlined_assembly[..., constraints="n"]` (max/mojo/max/gpu/compute/
    mma.mojo), i.e. `has_side_effect=True` but NO memory clobber, and so
    is the TMA copy asm it must stay ordered against
    (`cp_async_bulk_tensor_shared_cluster_global`, max/mojo/max/gpu/
    memory/memory.mojo). Today both are side-effecting asm and LLVM keeps
    side-effecting asm in relative order, which is why the emitted PTX is
    correct -- but that is a property of the current toolchain, not a
    contract it owes us. The clobber makes it a contract, for free, and
    it is the same idiom the modular repo itself documents as a "hard
    reordering barrier" (has_side_effect=True + `~{memory}`, see
    max/kernels/src/nn/attention/gpu/amd_structured/mha_prefill_v2.mojo).

    ptxas is the other half and cannot be reached from here: it is
    covered by asserting the SASS (not just the PTX) in
    tests/test_fa4_selfload_ptx_ordering.py.

    Parameters:
        group: Number of most recent wgmma-groups allowed to stay pending.
    """
    inlined_assembly[
        "wgmma.wait_group.sync.aligned $0;",
        NoneType,
        constraints="n,~{memory}",
        has_side_effect=True,
    ](group)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kFa4NThreads(head_dim))
    ),
    `nvvm.minctasm`=SIMDLength(kFa4CtasPerSm(head_dim)),
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
def fwd_fa4_selfload_kernel[
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
    bhsd: Bool = False,
](
    q_tma: TMATensorTile[dtype, qo_rank, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    o_tma: TMATensorTile[dtype, qo_rank, o_tile_shape, o_desc_shape],
    lse_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len_arg: Int64,
    softmax_scale: Float32,
    nheads_arg: Int64,
    sched_swizzle_arg: Int64,
    sched_num_hb_q_arg: Int64,
    sched_residual_arg: Int64,
):
    # BHSD-native mode: Q/K/V/O descriptors address the PUBLIC
    # contiguous (B, H, S, D) layout viewed as (B*H, S, D) — planes are
    # the TMA outer dim and S its own dim, so BM=192 tail tiles
    # zero-fill on load and clamp on store at every plane edge. No BTHD
    # materialization (FA3's scheme: the head stride is just another
    # gmem stride). Dense only in v1.
    comptime if bhsd:
        comptime assert qo_rank == 3 and not varlen and not window, (
            "bhsd v1 is the dense (non-varlen, non-window) path"
        )
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var seq_len = Int(seq_len_arg)
    var nheads = Int(nheads_arg)
    var sched_swizzle = Int(sched_swizzle_arg)
    var sched_num_hb_q = Int(sched_num_hb_q_arg)
    var sched_residual = Int(sched_residual_arg)
    # SCOPE GUARD: this body has no producer warpgroup and assumes one
    # 128-thread MMA warpgroup per CTA. head_dim=128 needs BM=BN=128 with
    # TWO warpgroups and 224 KiB of smem (1 CTA/SM) -- it must keep the
    # phase-2b kernel (fa4_fwd_kernel.mojo); instantiating this one at
    # d128 would compute only 64 of its 128 rows.
    comptime assert head_dim == 64, (
        "self-load geometry is d64-only (d128 keeps the phase-2b kernel)"
    )
    comptime BM: Int = kFa4BlockM(head_dim)
    comptime BN: Int = kFa4BlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kFa4NMmaWarpgroups(head_dim)
    comptime STAGES: Int = kFa4KVStages
    # PREFETCH INVARIANT -- READ BEFORE CHANGING STAGES OR THE REFILL
    # CALL SITES. This kernel deleted the empty[] mbarriers that the
    # phase-2b producer/consumer split used to prove a slot's smem read
    # had retired before the next TMA refill into that same slot. The
    # substitute proof is: the warpgroup that CONSUMES a slot (via
    # wgmma) is the same warpgroup that REFILLS it, so program order
    # plus `wgmma.wait_group` orders "read done" before "refill issued"
    # -- wait_group returns only once every warp's share of that wgmma
    # group has retired, which requires the smem read to be complete.
    # That proof holds ONLY as long as the refill targets the SAME slot
    # the just-retired wgmma just read, i.e. PREFETCH (the ring depth in
    # tile-pairs, STAGES/2) must be >= 1: the tile issued after
    # wait_group(k) is trip `it + PREFETCH`, whose slot is
    # `(2*(it+PREFETCH)) % STAGES == (2*it) % STAGES` exactly when
    # `2*PREFETCH % STAGES == 0`, i.e. PREFETCH == STAGES/2. Widening
    # STAGES without moving PREFETCH by the same STAGES/2 relationship
    # (or moving PREFETCH without moving the `issue_k`/`issue_v` call
    # sites' `it + PREFETCH` / `pf < kv_trips` offsets to match) breaks
    # the slot arithmetic silently: the refill would target a slot a
    # DIFFERENT in-flight tile still owns, which is a write-after-read
    # race with no barrier left to catch it. Failure mode if violated:
    # silent corruption (wrong output, no error, no crash) -- see
    # tests/test_fa4_selfload_soak.py, ported from the phase-2c harness's
    # soak_v7.mojo, which stresses exactly this window with back-to-back
    # launches and no inter-launch sync.
    #
    # The other half of the proof is that the refill really is EMITTED
    # after the wait. That is a program-order property, so it is the
    # compiler's to break, and the stdlib's `wgmma.wait_group` asm
    # declares no memory clobber; the refill-gating waits below therefore
    # go through `wgmma_wait_group_ordered` (same single instruction,
    # plus `~{memory}`) instead -- see its docstring for the PTX ISA
    # chain this pins down, and why a `fence.proxy.async` is NOT the
    # instrument (async->async write-after-read: that fence is defined
    # only across the generic/async boundary). The regression guards are
    # tests/test_fa4_selfload_ptx_ordering.py, which checks the emitted
    # PTX *and* the ptxas-scheduled SASS, and the soak above.
    #
    # The "consumer refills its own slot after its own retirement proof"
    # pattern is the same one warp-specialized GEMM mainloops use for
    # their smem-read retirement, not something invented for this
    # kernel: CUTLASS's SM90 TMA+GMMA mainloop retires a pipeline stage
    # with the consumer's own `wait` before the producer may reuse that
    # stage's smem (cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp,
    # `PipelineTmaAsync` release/acquire pair around lines 493-497 of
    # that file); FlashAttention-3's own sm90 warp-specialized mainloop
    # documents the identical rule at the point it advances its KV
    # pipeline (mainloop_fwd_sm90_tma_gmma_ws.hpp:1142); and the
    # modular repo's own SM90 dense GEMM mainloop relies on the same
    # wgmma-wait-then-release proof to retire a shared-memory read
    # before the next refill into that stage
    # (max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo,
    # lines 1554-1577) -- the ISA reference for `wgmma.wait_group.sync`
    # states only that prior wgmma ops complete; it does not by itself
    # say anything about a paired smem consumer/producer schedule being
    # safe, so those three call sites (not the ISA text alone) are what
    # back the claim that this pattern is correct.
    #
    # READ THOSE THREE PRECEDENTS CAREFULLY, because all three ALSO have
    # an mbarrier release/acquire pair where this kernel has nothing, and
    # that difference looks alarming until you see what the pair is for.
    # In every one of them the thread that refills the slot is in a
    # DIFFERENT warp from the threads whose wgmma read it, so the arrive/
    # wait pair is how the refilling thread LEARNS the read retired; the
    # proof that it retired is `wgmma.wait_group` on the consumer side,
    # exactly as here (CUTLASS says so at the call site: "Wait on the
    # GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure
    # smem_pipe_write is consumed"). This CTA is ONE warpgroup: the
    # thread that refills is one of the threads that waited, so there is
    # no second party to inform and the pair would carry no information.
    # What it would carry is ordering, and that comes instead from the
    # PTX ISA chain in `wgmma_wait_group_ordered`'s docstring.
    comptime PREFETCH: Int = STAGES // 2
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B
    # Tree-shaped row reductions (phase 2b), scoped to the geometry they
    # were profiled on: the d64 tile (BM=64, ONE MMA warpgroup, 2/3
    # CTAs/SM), where `wait` (fixed-latency dependency) was the top ncu
    # stall. On the d128 tile (BM=BN=128, two warpgroups, 1 CTA/SM) they
    # measured NEUTRAL, so d128 keeps a4e8462's codegen (and its exact
    # rowsum summation order) rather than taking an unproven change.
    # Ungating is safe if a later measurement wants one code path.
    comptime TREE_REDUCE: Bool = kFa4NMmaWarpgroups(head_dim) == 1

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
    if thread_idx.x == 0:
        mbar_q[0].init()
        comptime for s in range(STAGES):
            full[s].init(1)
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

    # ================= single warpgroup, self-loading =================
    # No producer warpgroup: this warpgroup issues its own TMA loads
    # (thread 0) at the points where `wgmma.wait_group` proves the
    # slot's reader has retired. Program order then replaces the empty[]
    # barriers entirely (see the PREFETCH invariant comment above).
    var wg: Int = 0

    # kv row of trip n is row0 + n * row_step (the same trip numbering
    # the consumer loop uses; window starts at first_kv, varlen
    # non-causal walks backwards).
    var kv_row0: Int
    var kv_row_step: Int = BN
    var kv_plane: Int = 0
    comptime if bhsd:
        kv_plane = b_idx * (nheads // gqa_ratio) + h_idx // gqa_ratio
        kv_row0 = 0
    elif window:
        comptime if varlen:
            kv_row0 = vl_k_base + first_kv * BN
        else:
            kv_row0 = b_idx * seq_len + first_kv * BN
    elif varlen:
        comptime if causal:
            kv_row0 = vl_k_base
        else:
            kv_row0 = vl_k_base + (num_kv_blocks - 1) * BN
            kv_row_step = -BN
    else:
        kv_row0 = b_idx * seq_len

    @parameter
    @always_inline
    def issue_k(n: Int, slot: Int):
        """TMA K(n) into `slot` (thread 0 only, caller-guarded)."""
        var k_st = LayoutTensor[
            dtype,
            k_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((kv_smem_base + slot * kv_slot_size).as_unsafe_any_origin())
        full[slot].expect_bytes(Int32(BN * D * size_of[dtype]()))
        var row: Int = kv_row0 + n * kv_row_step
        comptime if bhsd:
            k_tma.async_copy_3d(k_st, full[slot], (0, row, kv_plane))
        else:
            k_tma.async_copy_3d(
                k_st, full[slot], (0, h_idx // gqa_ratio, row)
            )

    @parameter
    @always_inline
    def issue_v(n: Int, slot: Int):
        """TMA V(n) into `slot` (thread 0 only, caller-guarded)."""
        var v_st = LayoutTensor[
            dtype,
            v_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((kv_smem_base + slot * kv_slot_size).as_unsafe_any_origin())
        full[slot].expect_bytes(Int32(BN * D * size_of[dtype]()))
        var row: Int = kv_row0 + n * kv_row_step
        comptime if bhsd:
            v_tma.async_copy_3d(v_st, full[slot], (0, row, kv_plane))
        else:
            v_tma.async_copy_3d(
                v_st, full[slot], (0, h_idx // gqa_ratio, row)
            )

    # Prologue issues: Q plus PREFETCH whole tile-pairs (the ring holds
    # exactly PREFETCH pairs).
    if thread_idx.x == 0:
        mbar_q[0].expect_bytes(Int32(BM * D * size_of[dtype]()))
        comptime if bhsd:
            q_tma.async_copy_3d(
                q_smem,
                mbar_q[0],
                (0, m_block * BM, b_idx * nheads + h_idx),
            )
        elif qo_rank == 4:
            q_tma.async_copy_4d(
                q_smem, mbar_q[0], (0, h_idx, m_block * BM, b_idx)
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
        comptime for pf in range(PREFETCH):
            if pf < kv_trips:
                issue_k(pf, (2 * pf) % STAGES)
                issue_v(pf, (2 * pf + 1) % STAGES)

    # Scheduler-pingpong barrier participant count: each barrier id
    # is waited by ONE warpgroup (128) and armed by its predecessor
    # (128) — 256 regardless of NWG (== NWG*128 only at NWG == 2).
    comptime SCHED_BAR_N: Int = 2 * 128

    # Warp-scheduler pingpong (FA4's use_scheduler_barrier): named
    # barrier 1+wg gates each warpgroup's GEMM-issue phase; a
    # warpgroup arrives at the *other* one's barrier after committing
    # its GEMM pair, so issue phases alternate and each warpgroup's
    # softmax overlaps the other's GEMMs. WG0 self-arms its barrier.
    # At NWG == 1 the pingpong ring degenerates to a SELF ring: the
    # warpgroup's wait on barrier 1 is satisfied by its own arrive from
    # the previous trip, so it orders nothing (the K/V mbarriers carry
    # every real dependency) while costing two BAR ops and a ~30-cycle
    # barrier latency per kv tile -- `barrier` was 0.51 warps/issue in
    # the ncu profile. Comptime-drop it; NWG >= 2 keeps FA4's schedule.
    comptime if NWG > 1:
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
                    # 32-BIT THRESHOLDS, hoisted out of the 64-element
                    # unrolled body. Mojo's Int is 64-bit, so
                    # `col + mask_col_lo + causal_mask_d > row + off`
                    # reached SASS as a 64-bit compare -- TWO ISETPs
                    # (ISETP.GT.U32.AND + ISETP.GT.AND.EX) per element,
                    # 128 of the masked tile's 192 mask instructions.
                    # Rearranged, the per-element test is
                    # `col_base > row + off - mask_col_lo - mask_d`
                    # whose right-hand side is loop-invariant: two
                    # Int32 registers and one ISETP per element.
                    #
                    # INT32 RANGE ASSUMPTION: mask_row_lo (tile-local,
                    # < BM <= 192) and mask_col_lo (< 8) are always
                    # small, but causal_mask_d carries -vl_offs for
                    # varlen causal -- the cumulative token offset of
                    # this sequence within its packed batch -- and is
                    # otherwise O(seqlen) for dense/window causal. The
                    # Int32() truncation below is exact only while
                    # |mask_row_lo - mask_col_lo - causal_mask_d| stays
                    # under INT32_MAX - 8 (~2.1e9): unreachable at any
                    # physical shape this kernel is built for (it would
                    # need a multi-billion-token packed varlen batch),
                    # but it is new exposure vs a 64-bit compare, which
                    # has no such ceiling. Guarded explicitly rather
                    # than left implicit; off by default like every
                    # other debug_assert (`-D ASSERT=all` or a debug
                    # build turns it on).
                    var mask_thr0_wide: Int = (
                        mask_row_lo - mask_col_lo - causal_mask_d
                    )
                    debug_assert(
                        mask_thr0_wide <= Int(Int32.MAX_FINITE) - 8
                        and mask_thr0_wide >= Int(Int32.MIN_FINITE),
                        (
                            "fa4 fwd (self-load): causal mask threshold"
                            " overflows int32 (seqlen/vl_offs too"
                            " large): "
                        ),
                        mask_thr0_wide,
                    )
                    var mask_thr0: Int32 = Int32(mask_thr0_wide)
                    var mask_thr1: Int32 = mask_thr0 + 8
                    # COLUMN-GROUP SKIP: the c-fragment's column
                    # col_base = (c//4)*8 + (c&1) is monotone in c, so
                    # the mask is a SUFFIX in c and whole quarters of
                    # the fragment can be skipped with one compare. The
                    # causal band is 65 columns wide inside a 128-column
                    # tile, so on the diagonal tile a bit under half the
                    # fragment is below the boundary for every thread.
                    # The test uses mask_thr0 (<= mask_thr1) so it is
                    # conservative for both rows, and the threads of a
                    # warp share a row block, so the branch is nearly
                    # warp-uniform.
                    comptime GROUPS: Int = 4
                    comptime GSZ: Int = c_frag_size_qk // GROUPS
                    comptime for g in range(GROUPS):
                        comptime col_max: Int32 = Int32(
                            ((g * GSZ + GSZ - 1) // 4) * 8 + 1
                        )
                        if col_max > mask_thr0:
                            comptime for cc in range(GSZ):
                                comptime c: Int = g * GSZ + cc
                                comptime col_base: Int32 = Int32(
                                    (c // 4) * 8 + (c & 1)
                                )
                                comptime hi_row: Bool = (c % 4) >= 2
                                if col_base > (
                                    mask_thr1 if hi_row else mask_thr0
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
        # TREE-SHAPED row reductions. The natural
        # `local_max[row] = max(local_max[row], s_reg[c])` sweep is two
        # 32-deep dependent chains (~4-cycle FMNMX latency each = ~128
        # cycles of pure latency for 64 issue slots); `wait` was the
        # top stall in the ncu profile. Four partial accumulators per
        # row turn it into eight 8-deep chains plus a 2-level combine.
        # Same instruction count, ~1/3 the exposed latency, and max is
        # associative/commutative so the result is bit-identical.
        comptime RED_WAYS: Int = 4
        var local_max = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        comptime if TREE_REDUCE:
            var part_max = stack_allocation[
                rows_per_thread * RED_WAYS, Scalar[accum_type]
            ]()
            comptime for i in range(rows_per_thread * RED_WAYS):
                part_max[i] = neg_inf
            comptime for c in range(c_frag_size_qk):
                comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
                comptime part: Int = (c // 4) % RED_WAYS
                part_max[row_idx * RED_WAYS + part] = max(
                    part_max[row_idx * RED_WAYS + part], s_reg.ptr[c]
                )
            comptime for i in range(rows_per_thread):
                local_max[i] = max(
                    max(part_max[i * RED_WAYS], part_max[i * RED_WAYS + 1]),
                    max(part_max[i * RED_WAYS + 2], part_max[i * RED_WAYS + 3]),
                )
        else:
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
        comptime if TREE_REDUCE:
            var part_sum = stack_allocation[
                rows_per_thread * RED_WAYS, Scalar[accum_type]
            ]()
            comptime for i in range(rows_per_thread * RED_WAYS):
                part_sum[i] = Scalar[accum_type](0)
            comptime for c in range(c_frag_size_qk):
                comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
                comptime part: Int = (c // 4) % RED_WAYS
                var p: Scalar[accum_type] = exp2(
                    s_reg.ptr[c].fma(scale_log2, -rowmax[row_idx])
                )
                s_reg.ptr[c] = p
                part_sum[row_idx * RED_WAYS + part] += p
            # Pairwise combine (the ONLY numeric difference vs the
            # a4e8462 kernel: f32 addition is not associative, so rowsum
            # can differ in the last ulp -- tree summation of
            # non-negative exp2 outputs is the better-conditioned order,
            # and the fp64 reference check bounds it).
            comptime for i in range(rows_per_thread):
                local_sum[i] = (
                    part_sum[i * RED_WAYS] + part_sum[i * RED_WAYS + 1]
                ) + (part_sum[i * RED_WAYS + 2] + part_sum[i * RED_WAYS + 3])
        else:
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
    # Refill-gating wait: `wgmma_wait_group_ordered`, not the stdlib's
    # `wgmma_qk.wait_group()`, so the refill below cannot be moved above
    # it (see the helper's docstring).
    wgmma_wait_group_ordered()
    warpgroup_fence(s_reg)
    # K(0)'s slot (0) is free: refill it with K(PREFETCH). 2*PREFETCH
    # == STAGES, so that tile's slot IS slot 0 (see the PREFETCH
    # invariant comment above).
    if thread_idx.x == 0:
        if PREFETCH < kv_trips:
            issue_k(PREFETCH, 0)

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
            causal_mask_d = -m_block * BM
            comptime if varlen:
                causal_mask_d -= vl_offs
            comptime if window:
                # The prologue tile is first_kv, not 0.
                causal_mask_d += first_kv * BN
            comptime if varlen or window:
                prologue_diag = kv_trips <= 2
            else:
                # EXACT diagonal test (dense causal): a tile needs the
                # mask iff its last column can exceed this warpgroup's
                # first row, n*BN + BN-1 > m*BM + wg*WGMMA_M, i.e.
                # causal_mask_d + BN-1 > wg*WGMMA_M. The old "last two
                # tiles of the trip range" rule is a strict superset:
                # at BM=64/BN=128 the 64-row band's diagonal spans 65
                # columns and always lands in ONE tile, so it made 42%
                # of all tiles run the mask arm where 22% need it.
                prologue_diag = causal_mask_d + (BN - 1) > wg * WGMMA_M
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

    # ---- Main loop: QK(n+1) + PV(n) per iteration. Ring slots track
    # incrementally (no div/mod per iter): K(t): slot 2t%STAGES,
    # V(t): (2t+1)%STAGES, phase flips every STAGES//2 tiles.
    var k_slot: Int = 2  # K(1)
    var k_phase: UInt32 = 0
    var k_wrap: Int = 1
    var v_slot: Int = 1  # V(0)
    var v_phase: UInt32 = 0
    var v_wrap: Int = 0

    for it in range(kv_trips - 1):
        # Queue QK(n+1) then PV(n) on the tensor core.
        full[k_slot].wait(k_phase)
        comptime if NWG > 1:
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
        elif NWG > 2:
            named_barrier_arrive[Int32(SCHED_BAR_N)](
                Int32(wg + 2 if wg < NWG - 1 else 1)
            )

        # QK(n+1) retired (PV(n) still running on the tensor core).
        # Refill-gating wait -- memory-clobbering variant (see the helper).
        wgmma_wait_group_ordered[1]()
        warpgroup_fence(s_reg)
        # QK(it+1) has retired for the whole warpgroup -> K(it+1)'s slot
        # is free for the tile PREFETCH trips later (same slot; see the
        # PREFETCH invariant comment above).
        if thread_idx.x == 0:
            var k_next: Int = it + 1 + PREFETCH
            if k_next < kv_trips:
                issue_k(k_next, k_slot)

        # Softmax of S(n+1) overlaps PV(n). For causal the last
        # tile (BM==BN: n+1 == num_kv_blocks-1 == m_block is THE
        # diagonal; BM>BN: the last TWO tiles form the band).
        # (Varlen's tail mask ran in the prologue — reverse order.)
        var loop_diag: Bool = False
        comptime if causal:
            comptime if BM == BN and not varlen:
                loop_diag = it == kv_trips - 2
            else:
                causal_mask_d = (it + 1) * BN - m_block * BM
                comptime if varlen:
                    causal_mask_d -= vl_offs
                comptime if window:
                    # Loop trip it processes tile first_kv + it + 1.
                    causal_mask_d += first_kv * BN
                comptime if varlen or window:
                    loop_diag = it >= kv_trips - 3
                else:
                    loop_diag = causal_mask_d + (BN - 1) > wg * WGMMA_M
        # Window, unaligned left only: the first loop tile can hold
        # leading-edge columns (win_mask_d was advanced after the
        # prologue). Aligned variants pass a comptime False so the
        # window arm is DCE'd from the loop's inlined copy.
        var loop_lead: Bool = False
        comptime if window_unaligned:
            loop_lead = it == 0
        softmax_block(loop_diag, loop_lead)

        # PV(n) retired: p_reg and o_reg are safe to touch.
        # Refill-gating wait -- memory-clobbering variant (see the helper).
        wgmma_wait_group_ordered[0]()
        warpgroup_fence(o_reg)
        if thread_idx.x == 0:
            var v_next: Int = it + PREFETCH
            if v_next < kv_trips:
                issue_v(v_next, v_slot)
        pack_p()  # P(n+1)
        rescale_o()

        k_slot += 2
        k_wrap += 1
        if k_wrap == STAGES // 2:
            k_wrap = 0
            k_slot = 0
            k_phase ^= 1
        v_slot += 2
        v_wrap += 1
        if v_wrap == STAGES // 2:
            v_wrap = 0
            v_slot = 1
            v_phase ^= 1

    # ---- Epilogue: PV(N-1) (v_slot/v_phase left at tile N-1).
    full[v_slot].wait(v_phase)
    warpgroup_fence(o_reg)
    wgmma_pv.arrive()
    pv_gemm(v_slot)
    wgmma_pv.commit_group()
    # Same memory-clobbering wait: what follows is not a TMA refill but the
    # O epilogue staging its c-frags into the DEAD Q tile -- plain
    # (generic-proxy) shared-memory stores into the exact smem the QK
    # wgmmas were reading. Same write-after-read shape, same reliance on
    # "the wait retired the readers, and the stores come after it in
    # program order", and plain stores are freer to move than the refill's
    # side-effecting asm is, so this site wants the clobber at least as
    # much as the three above.
    wgmma_wait_group_ordered[0]()
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
    # valid-row count for both paths). BHSD's rank-3 (planes, S, D)
    # store clamps at the plane's S edge the same way.
    comptime if qo_rank == 4 or bhsd:
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
            comptime if varlen or qo_rank == 4 or bhsd:
                lse_ok = r < vl_rows
            if lse_ok:
                (lse_ptr + lse_row_base + r)[0] = (
                    rowmax[i] * LN2 + log(rowsum[i])
                ).cast[DType.float32]()

    fence_async_view_proxy()
    # No producer warpgroup exists in this kernel -> a plain
    # whole-CTA barrier (id NWG+1: ids 1..NWG are the scheduler
    # pingpong barriers) suffices before the O store reads smem.
    named_barrier[Int32(NWG * 128)](Int32(NWG + 1))
    if full_tile_store and thread_idx.x == 0:
        var o_st = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]((smem_base).as_unsafe_any_origin())
        comptime if bhsd:
            # (B*H, S, D) planes: the partial tail clamps at the
            # plane's S edge in hardware.
            o_tma.async_store_3d(
                o_st, (0, m_block * BM, b_idx * nheads + h_idx)
            )
        elif qo_rank == 4:
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
