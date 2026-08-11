# ===----------------------------------------------------------------------=== #
# Strict-fp32 TN GEMM cores for NVIDIA sm_90 (CUDA-core FFMA only; this path
# IS what torch.set_float32_matmul_precision("highest") means — no TF32, no
# tensor cores).
#
#   C[m,n] = A_phys[k,m]^T @ B[k,n]
#
# Two device kernels, dispatched by `tn_f32_gemm_kernels.mojo`:
#
# `_tn_core_kernel` — the repo `_gemm_pipe3_kernel` (AT=True) restructured
# around the two limiters ncu found while profiling that kernel on 4096^3
# (H100 PCIe, clocks locked at 1500 MHz):
#
#   1. 268M shared-load bank conflicts per launch: the contiguous TN=8
#      fragment (`Bs[kk][tn0 .. tn0+7]`, tn0 = (tid%16)*8) makes every
#      8-lane LDS.128 phase request bytes {0,32,64,...} — lanes 0&4, 1&5, ...
#      collide 2-way on every B-fragment load. Fix: cuBLAS-style QUADRANT
#      warp tiling for B. A warp is an LR x LC lane grid computing a
#      WMxWN tile; each thread owns TM contiguous A rows (broadcast smem
#      loads — conflict-free by nature) x QN float4 B quadrants LC*4 apart,
#      so every 8-lane LDS.128 phase covers one contiguous 128B window (B)
#      or a 16B broadcast (A) — zero conflicts by construction (ncu after:
#      ~5K).
#
#   2. Two barriers per slab: the second barrier (before refilling the just-
#      freed smem slot) is redundant — passing the top barrier of iteration s
#      (which sits after `async_copy_wait_group`) already proves every warp
#      finished computing slab s-1, and the slot refilled at iteration s is
#      exactly the one slab s-1 read ((s+STAGES-1) % STAGES == (s-1) %
#      STAGES). One barrier per slab, and the cp.async for slab s+STAGES-1
#      issues BEFORE the compute so the loads fly behind the FFMAs.
#
# Everything else is inherited from pipe3 verbatim: cp.async ZFILL staging
# (A_phys (k,m) is already the layout the As[kk][mm] slab wants — the TN
# transpose is free), VEC gates for the 16B-alignment rule (VEC_A=4 iff the
# A staging addresses are 16B-aligned, VEC_B=4 likewise for B), BK-rounded
# split-K chunks, per-element edge guards in the epilogue. Accumulation
# order per output element is the same sequential-over-k FFMA chain as
# pipe3/NN, so the NN fp32 comparison tolerances apply unchanged.
#
# `_tn_split_kernel` — block-internal split-K by warp group. Motivation
# (measured on the same card):
#
#   - The 8x8-thread-tile 128x128 core (`_tn_core_kernel`) issues at 83%
#     (4 warps/scheduler from 2 de-phased CTAs) but its FFMA:issue mix caps
#     at ~86% (1024 FFMA vs ~70 LDS + ~95 control per slab).
#   - The cuBLAS-geometry 128x256 / 8x16-thread-tile core reaches ~89-91%
#     mix but only ~75% issue: at 254 regs it is 1 CTA/SM, so each
#     scheduler holds 2 warps of the SAME block, co-phased by the slab
#     barrier -> their stalls correlate and neither can cover the other.
#
# This kernel keeps the 8x16 thread tile (2048 FFMA/slab -> the good mix)
# but restores de-phased dual pipelines WITHIN the single resident block:
# 256 threads = 2 warp GROUPS of 4 warps. Each group covers the WHOLE
# 128x128 tile over HALF of the block's k-chunk (block-internal split-K),
# with its OWN smem stage buffers and its OWN named barrier — the groups
# never synchronize during the main loop, so each scheduler again holds
# two mutually de-phased warps (one per group), like the 2-CTA t128 case,
# while every FFMA belongs to a 128-accumulator thread tile.
#
# At the end, the groups meet at a full-block barrier, then group 1
# publishes its accumulators through shared memory (group 1's stage area,
# reused once both pipelines are done) and group 0 adds and stores C.
# The exchange is fully barrier-serialized: an earlier version let group 1
# store into its own stage area WHILE group 0 was still looping, and that
# overlap produced intermittent slab-sized errors on this card (e.g.
# 256x256x131072, ks=28) — do not reintroduce it. Accumulation order per
# element is two sequential-over-k FFMA chains summed once — the same
# numerics as ordinary two-way split-K.
# ===----------------------------------------------------------------------=== #

from max.gpu.sync import barrier
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from std.memory import AddressSpace
from max.gpu.sync import named_barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // WM) * (BN // WN) * 32)
    ),
    `nvvm.minctasm`=SIMDLength(MINB),
)
@__name(
    t"tn_core_{BM}x{BN}x{BK}_w{WM}x{WN}l{LR}_va{VEC_A}_vb{VEC_B}_s{STAGES}_mb{MINB}_p{PUMP}"
)
def _tn_core_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,  # warp tile rows; warp grid is (BM//WM) x (BN//WN)
    WN: Int,  # warp tile cols
    LR: Int,  # lane-grid rows: a warp's 32 lanes form LR x (32//LR)
    VEC_A: Int,
    VEC_B: Int,
    STAGES: Int,
    MINB: Int,
    PUMP: Int = 1,  # slabs per barrier (1 or 2; 2 needs STAGES >= 6 even)
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    ksplits_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var ksplits = Int(ksplits_arg)

    comptime F32 = DType.float32
    comptime WARPS_M = BM // WM
    comptime WARPS_N = BN // WN
    comptime THREADS = WARPS_M * WARPS_N * 32
    # A warp is an LR x LC lane grid (tr = lane//LC, tc = lane%LC). Each
    # thread's A fragment is TM = WM//LR CONTIGUOUS rows: its smem load is
    # a broadcast across the LC tc-lanes — conflict-free by nature. Each
    # thread's B fragment is QN = WN/(LC*4) float4 QUADRANTS LC*4 apart:
    # every 8-lane LDS.128 phase covers one contiguous window or a
    # broadcast — the conflict-free layout the flat contiguous fragment
    # lacked. The 254-reg cuBLAS sgemm geometry (warpsize 2x2, 128
    # threads, 8x16 thread tile) is LR=8: WM=WN=64 -> TM=8, QN=4.
    comptime LC = 32 // LR
    comptime TM = WM // LR
    comptime QN = WN // (LC * 4)
    comptime NA = (BM * BK) // (THREADS * VEC_A)
    comptime NB = (BK * BN) // (THREADS * VEC_B)
    comptime assert WM % LR == 0 and WN % (LC * 4) == 0
    comptime assert TM % 4 == 0 or TM == 4
    comptime assert BM % WM == 0 and BN % WN == 0
    comptime assert (BM * BK) % (THREADS * VEC_A) == 0
    comptime assert (BK * BN) % (THREADS * VEC_B) == 0

    var ks = block_idx.z
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the cp.async paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [ksplits, m, n] workspace; a reduce kernel
    # sums the slices afterwards. Batch is always 1 here (TN wgrad).
    var c_ptr = c_base + block_idx.z * m * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x
    var warp = tid // 32
    var lane = tid % 32
    var wr = warp // WARPS_N
    var wc = warp % WARPS_N
    var tr = lane // LC
    var tc = lane % LC

    var a_smem = stack_allocation[
        STAGES * BM * BK, F32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        STAGES * BK * BN, F32, address_space=AddressSpace.SHARED
    ]()

    # Fragment base columns within a smem row.
    var am0 = wr * WM + tr * TM
    var bn0 = wc * WN + tc * 4

    var acc = InlineArray[SIMD[F32, 4], TM * QN](fill=SIMD[F32, 4](0))

    var nslabs = ceildiv(k_end - k_start, BK)
    if nslabs == 0:
        return

    # ---- staging invariants, hoisted out of the slab loop --------------
    # Because THREADS is a multiple of the copies-per-tile-row for both
    # operands, EVERY copy of a thread lands in the same tile COLUMN and
    # the copies differ only by a fixed row stride (ROWS_x_STEP). So the
    # column, its edge byte-count and its OOB clamp are per-thread
    # constants, the global source offsets are one per-slab base plus
    # comptime multiples of a hoisted row-stride, and the smem targets are
    # comptime immediates off one base pointer. The row guard (row <
    # k_end) can only fail on the LAST slab of the chunk, so the fast
    # fetch drops it and a GUARDED variant runs once. Copies whose column
    # is fully out of range (bytes == 0, edge blocks) use a clamped
    # column; cp.async with src-size 0 reads nothing and zero-fills.
    comptime COLS_A = BM // VEC_A
    comptime COLS_B = BN // VEC_B
    comptime ROWS_A_STEP = THREADS // COLS_A
    comptime ROWS_B_STEP = THREADS // COLS_B
    comptime assert THREADS % COLS_A == 0 and THREADS % COLS_B == 0
    var a_cm = (tid % COLS_A) * VEC_A
    var a_bkk = tid // COLS_A
    var a_col = bm + a_cm
    var a_bytes = Int32(max(0, min(VEC_A, m - a_col)) * 4)
    var a_colc = a_col if a_bytes > 0 else 0
    var a_rstep = ROWS_A_STEP * m
    var b_cn = (tid % COLS_B) * VEC_B
    var b_bkk = tid // COLS_B
    var b_col = bn + b_cn
    var b_bytes = Int32(max(0, min(VEC_B, n - b_col)) * 4)
    var b_colc = b_col if b_bytes > 0 else 0
    var b_rstep = ROWS_B_STEP * n
    # Rolling per-slab source bases (advanced by one IADD per slab in
    # _fetch_slab; recomputing them with an IMAD per slab measured ~2%
    # slower on the 16B variants).
    var a_srcs = (k_start + a_bkk) * m + a_colc
    var b_srcs = (k_start + b_bkk) * n + b_colc

    @always_inline
    @parameter
    def _fetch[GUARDED: Bool](s: Int, buf: Int):
        # Issues the copies for slab `s` into smem buffer `buf` (passed in
        # explicitly: the PUMP=2 path keeps rolling buffer indices because
        # `s % 6` lowers to a ~20-op magic-number division chain; for the
        # PUMP=1 path keep STAGES a power of two so `% STAGES` is an AND).
        var kt = k_start + s * BK

        # A^T (k, m) row-major: chunks along m into As[kk][mm] (coalesced
        # along m — the physical TN layout IS the slab layout; no
        # transpose, no copy).
        var a_dst = a_smem + (buf * BM * BK + a_bkk * BM + a_cm)
        var a_src = a_srcs
        comptime for t in range(NA):
            var bytes = a_bytes
            comptime if GUARDED:
                if kt + a_bkk + t * ROWS_A_STEP >= k_end:
                    bytes = 0
            async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                (a_base + (a_src + t * a_rstep)).address_space_cast[
                    AddressSpace.GLOBAL
                ](),
                (a_dst + t * (ROWS_A_STEP * BM)).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

        # B (k, n) row-major: chunks along n into Bs[kk][nn].
        var b_dst = b_smem + (buf * BK * BN + b_bkk * BN + b_cn)
        var b_src = b_srcs
        comptime for t in range(NB):
            var bytes = b_bytes
            comptime if GUARDED:
                if kt + b_bkk + t * ROWS_B_STEP >= k_end:
                    bytes = 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_base + (b_src + t * b_rstep)).address_space_cast[
                    AddressSpace.GLOBAL
                ](),
                (b_dst + t * (ROWS_B_STEP * BN)).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

    @always_inline
    @parameter
    def _fetch_slab(s: Int, buf: Int):
        if s == nslabs - 1:
            _fetch[True](s, buf)
        else:
            _fetch[False](s, buf)
        a_srcs += BK * m
        b_srcs += BK * n

    @always_inline
    @parameter
    def _compute(aoff: Int, boff: Int):
        # One slab of FMAs from the smem slab at (aoff, boff).
        # Register-double-buffered fragments: loads for column kk+1 issue
        # before the FMAs of column kk. All smem loads carry alignment=16:
        # without it Mojo emits align-4 vector loads, which reach PTX as
        # scalar ld.shared.b32 and ptxas only partially re-fuses them
        # (that alone cost ~18% FMA pipe on 4096^3).
        var a_base_s = a_smem + aoff
        var b_base_s = b_smem + boff
        var a_cur = a_base_s.load[width=TM, alignment=16](am0)
        var b_cur = InlineArray[SIMD[F32, 4], QN](uninitialized=True)
        comptime for q in range(QN):
            b_cur[q] = b_base_s.load[width=4, alignment=16](bn0 + q * (LC * 4))
        comptime for kk in range(BK - 1):
            var a_nxt = a_base_s.load[width=TM, alignment=16](
                (kk + 1) * BM + am0
            )
            var b_nxt = InlineArray[SIMD[F32, 4], QN](uninitialized=True)
            comptime for q in range(QN):
                b_nxt[q] = b_base_s.load[width=4, alignment=16](
                    (kk + 1) * BN + bn0 + q * (LC * 4)
                )
            comptime for i in range(TM):
                comptime for q in range(QN):
                    acc[i * QN + q] = b_cur[q].fma(
                        SIMD[F32, 4](a_cur[i]), acc[i * QN + q]
                    )
            a_cur = a_nxt
            comptime for q in range(QN):
                b_cur[q] = b_nxt[q]
        comptime for i in range(TM):
            comptime for q in range(QN):
                acc[i * QN + q] = b_cur[q].fma(
                    SIMD[F32, 4](a_cur[i]), acc[i * QN + q]
                )

    comptime if PUMP == 1:
        # Prologue: fill the pipeline (STAGES - 1 slabs in flight).
        for st in range(min(STAGES - 1, nslabs)):
            _fetch_slab(st, st)
            async_copy_commit_group()
        for _ in range(nslabs, STAGES - 1):
            async_copy_commit_group()

        for s in range(nslabs):
            # Wait until the group fetched for slab s has landed.
            async_copy_wait_group(Int32(STAGES - 2))
            barrier()

            # Refill the slot slab s-1 just freed ((s+STAGES-1) % STAGES).
            # Passing the barrier above proves every warp finished slab
            # s-1, so no second barrier is needed; issuing the fetch here
            # lets the cp.async fly behind the FFMAs below.
            if s + STAGES - 1 < nslabs:
                _fetch_slab(s + STAGES - 1, (s + STAGES - 1) % STAGES)
            async_copy_commit_group()

            var buf = s % STAGES
            _compute(buf * (BM * BK), buf * (BK * BN))
    else:
        # Double-pumped: ONE barrier per TWO slabs. At the even slab s the
        # barrier (placed after the wait for slab s) proves every warp
        # finished slab s-1, so refilling the buffers of slabs s-2 and s-1
        # ((s+STAGES-2) % STAGES and (s+STAGES-1) % STAGES) is safe; the
        # odd slab s+1 computes with no barrier at all — its buffer was
        # fetched STAGES-1 slabs ago and no fetch targets a buffer the
        # lagging warps could still be reading (warp skew is bounded by
        # the even-slab barrier). Commit-group accounting stays one group
        # per slab (empty groups keep the wait counts aligned).
        comptime assert STAGES >= 4 and STAGES % 2 == 0
        for st in range(min(STAGES - 2, nslabs)):
            _fetch_slab(st, st)
            async_copy_commit_group()
        for _ in range(nslabs, STAGES - 2):
            async_copy_commit_group()

        # Rolling even-slab buffer index (avoids `% STAGES` for STAGES=6,
        # which lowers to a magic-number division chain).
        var bufe = 0
        var s = 0
        while s < nslabs:
            async_copy_wait_group(Int32(STAGES - 3))
            barrier()

            var buff = bufe + (STAGES - 2) if bufe < 2 else bufe - 2
            if s + STAGES - 2 < nslabs:
                _fetch_slab(s + STAGES - 2, buff)
            async_copy_commit_group()
            if s + STAGES - 1 < nslabs:
                _fetch_slab(s + STAGES - 1, buff + 1)
            async_copy_commit_group()

            _compute(bufe * (BM * BK), bufe * (BK * BN))
            if s + 1 < nslabs:
                async_copy_wait_group(Int32(STAGES - 2))
                _compute((bufe + 1) * (BM * BK), (bufe + 1) * (BK * BN))
            s += 2
            bufe = 0 if bufe + 2 == STAGES else bufe + 2

    # Epilogue: 4-wide stores per (row, col-quadrant); 8-lane groups cover
    # one 128B row window each — coalesced. Edge-guarded per element.
    # When VEC_B == 4 every in-range vector store is 16B aligned (n % 4 == 0
    # and a 16B-aligned C base are part of that gate) — say so, or the
    # store scalarizes like the loads did.
    comptime CALIGN = 16 if VEC_B == 4 else 4
    comptime for i in range(TM):
        var row = bm + wr * WM + tr * TM + i
        if row < m:
            comptime for qn in range(QN):
                var col = bn + wc * WN + qn * (LC * 4) + tc * 4
                var v = acc[i * QN + qn]
                if col + 4 <= n:
                    c_ptr.store[alignment=CALIGN](row * n + col, v)
                else:
                    comptime for j in range(4):
                        if col + j < n:
                            c_ptr[row * n + col + j] = v[j]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)),
    `nvvm.minctasm`=SIMDLength(1),
)
@__name(t"tn_split_{BM}x{BN}x{BK}_va{VEC_A}_vb{VEC_B}_s{STAGES}_l{LR}_z{SERP}")
def _tn_split_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    VEC_A: Int,
    VEC_B: Int,
    STAGES: Int,
    LR: Int = 8,  # lane-grid rows within a warp (8 -> 8x16, 4 -> 16x8)
    SERP: Bool = False,  # serpentine quadrant order (operand-reuse aid)
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    ksplits_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var ksplits = Int(ksplits_arg)

    comptime F32 = DType.float32
    comptime TG = 128  # threads per warp group (4 warps)
    # Warp grid within a group: 2x2 warps of 64x64; lanes LR=8 x LC=4;
    # thread tile TM=8 rows x QN=4 float4 quadrants (8x16 = 128 acc).
    comptime WM = 64
    comptime WN = 64
    comptime LC = 32 // LR
    comptime TM = WM // LR
    comptime QN = WN // (LC * 4)
    comptime NA = (BM * BK) // (TG * VEC_A)
    comptime NB = (BK * BN) // (TG * VEC_B)
    comptime assert BM == 2 * WM and BN == 2 * WN
    comptime assert (BM * BK) % (TG * VEC_A) == 0
    comptime assert (BK * BN) % (TG * VEC_B) == 0
    comptime assert STAGES >= 2

    var ks = block_idx.z
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)
    var ns = ceildiv(k_end - k_start, BK)
    if ns <= 0:
        return

    var c_ptr = c_base + block_idx.z * m * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x
    var gid = tid // TG
    var ltid = tid % TG
    var warp = ltid // 32
    var lane = ltid % 32
    var wr = warp // 2
    var wc = warp % 2
    var tr = lane // LC
    var tc = lane % LC

    # Group-major smem: [g0: A stages | B stages][g1: A stages | B stages].
    # Group 1's contiguous region (GROUP_F floats) doubles as the
    # accumulator-exchange buffer once both main loops have ended.
    comptime GROUP_F = STAGES * BK * (BM + BN)
    # Half the accs per round: a single 32-acc round makes ptxas spill
    # (104 B stack) at the 255-reg ceiling; 16-acc rounds stay spill-free.
    comptime EX_CHUNK = min((TM * QN) // 2, GROUP_F // (TG * 4))
    comptime EX_ROUNDS = (TM * QN + EX_CHUNK - 1) // EX_CHUNK
    comptime assert EX_CHUNK >= 1
    var smem = stack_allocation[
        2 * GROUP_F, F32, address_space=AddressSpace.SHARED
    ]()
    var a_smem = smem + gid * GROUP_F
    var b_smem = a_smem + STAGES * BK * BM
    var ex_smem = smem + GROUP_F

    var am0 = wr * WM + tr * TM
    var bn0 = wc * WN + tc * 4

    var acc = InlineArray[SIMD[F32, 4], TM * QN](fill=SIMD[F32, 4](0))

    # Block-internal split-K: group 0 takes slabs [0, nsl0), group 1 takes
    # [nsl0, ns). Group 0 gets the extra slab and is the finisher.
    var nsl0 = (ns + 1) // 2
    var s_begin = 0 if gid == 0 else nsl0
    var nsg = nsl0 if gid == 0 else ns - nsl0

    # ---- staging invariants (see _tn_core_kernel above) ----------------
    comptime COLS_A = BM // VEC_A
    comptime COLS_B = BN // VEC_B
    comptime ROWS_A_STEP = TG // COLS_A
    comptime ROWS_B_STEP = TG // COLS_B
    comptime assert TG % COLS_A == 0 and TG % COLS_B == 0
    var a_cm = (ltid % COLS_A) * VEC_A
    var a_bkk = ltid // COLS_A
    var a_col = bm + a_cm
    var a_bytes = Int32(max(0, min(VEC_A, m - a_col)) * 4)
    var a_colc = a_col if a_bytes > 0 else 0
    var a_rstep = ROWS_A_STEP * m
    var b_cn = (ltid % COLS_B) * VEC_B
    var b_bkk = ltid // COLS_B
    var b_col = bn + b_cn
    var b_bytes = Int32(max(0, min(VEC_B, n - b_col)) * 4)
    var b_colc = b_col if b_bytes > 0 else 0
    var b_rstep = ROWS_B_STEP * n
    var a_srcs = (k_start + s_begin * BK + a_bkk) * m + a_colc
    var b_srcs = (k_start + s_begin * BK + b_bkk) * n + b_colc

    @always_inline
    @parameter
    def _fetch[GUARDED: Bool](s: Int, buf: Int):
        # Issues the copies for GLOBAL slab `s` into this group's smem
        # buffer `buf`.
        var kt = k_start + s * BK

        var a_dst = a_smem + (buf * BM * BK + a_bkk * BM + a_cm)
        var a_src = a_srcs
        comptime for t in range(NA):
            var bytes = a_bytes
            comptime if GUARDED:
                if kt + a_bkk + t * ROWS_A_STEP >= k_end:
                    bytes = 0
            async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                (a_base + (a_src + t * a_rstep)).address_space_cast[
                    AddressSpace.GLOBAL
                ](),
                (a_dst + t * (ROWS_A_STEP * BM)).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

        var b_dst = b_smem + (buf * BK * BN + b_bkk * BN + b_cn)
        var b_src = b_srcs
        comptime for t in range(NB):
            var bytes = b_bytes
            comptime if GUARDED:
                if kt + b_bkk + t * ROWS_B_STEP >= k_end:
                    bytes = 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_base + (b_src + t * b_rstep)).address_space_cast[
                    AddressSpace.GLOBAL
                ](),
                (b_dst + t * (ROWS_B_STEP * BN)).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

    @always_inline
    @parameter
    def _fetch_slab(s: Int, buf: Int):
        # The row guard can only fail on the LAST slab of the block chunk
        # (the k tail), which lives in exactly one group's range.
        if s == ns - 1:
            _fetch[True](s, buf)
        else:
            _fetch[False](s, buf)
        a_srcs += BK * m
        b_srcs += BK * n

    @always_inline
    @parameter
    def _compute(aoff: Int, boff: Int):
        var a_base_s = a_smem + aoff
        var b_base_s = b_smem + boff
        var a_cur = a_base_s.load[width=TM, alignment=16](am0)
        var b_cur = InlineArray[SIMD[F32, 4], QN](uninitialized=True)
        comptime for q in range(QN):
            b_cur[q] = b_base_s.load[width=4, alignment=16](bn0 + q * (LC * 4))
        comptime for kk in range(BK - 1):
            var a_nxt = a_base_s.load[width=TM, alignment=16](
                (kk + 1) * BM + am0
            )
            var b_nxt = InlineArray[SIMD[F32, 4], QN](uninitialized=True)
            comptime for q in range(QN):
                b_nxt[q] = b_base_s.load[width=4, alignment=16](
                    (kk + 1) * BN + bn0 + q * (LC * 4)
                )
            comptime for i in range(TM):
                comptime for qz in range(QN):
                    comptime q = (qz if not SERP or i % 2 == 0 else QN - 1 - qz)
                    acc[i * QN + q] = b_cur[q].fma(
                        SIMD[F32, 4](a_cur[i]), acc[i * QN + q]
                    )
            a_cur = a_nxt
            comptime for q in range(QN):
                b_cur[q] = b_nxt[q]
        comptime for i in range(TM):
            comptime for qz in range(QN):
                comptime q = qz if not SERP or i % 2 == 0 else QN - 1 - qz
                acc[i * QN + q] = b_cur[q].fma(
                    SIMD[F32, 4](a_cur[i]), acc[i * QN + q]
                )

    # ---- per-group pipelined main loop (never syncs the other group) ---
    for st in range(min(STAGES - 1, nsg)):
        _fetch_slab(s_begin + st, st)
        async_copy_commit_group()
    for _ in range(nsg, STAGES - 1):
        async_copy_commit_group()

    for i in range(nsg):
        async_copy_wait_group(Int32(STAGES - 2))
        named_barrier[Int32(TG)](Int32(1 + gid))

        if i + STAGES - 1 < nsg:
            _fetch_slab(s_begin + i + STAGES - 1, (i + STAGES - 1) % STAGES)
        async_copy_commit_group()

        var buf = i % STAGES
        _compute(buf * (BM * BK), buf * (BK * BN))

    # ---- exchange: group 1 publishes, group 0 adds and stores C --------
    # Fully serialized: both groups first meet at a full barrier (their
    # pipelines are drained — every real cp.async completed before each
    # group's last wait), then rounds of EX_CHUNK accs go through group
    # 1's stage area with a barrier on each side of the store->read edge.
    barrier()
    comptime for r in range(EX_ROUNDS):
        if gid == 1:
            comptime for j in range(EX_CHUNK):
                comptime idx = r * EX_CHUNK + j
                comptime if idx < TM * QN:
                    ex_smem.store[alignment=16](
                        ltid * (EX_CHUNK * 4) + j * 4, acc[idx]
                    )
        barrier()
        if gid == 0:
            comptime for j in range(EX_CHUNK):
                comptime idx = r * EX_CHUNK + j
                comptime if idx < TM * QN:
                    acc[idx] += ex_smem.load[width=4, alignment=16](
                        ltid * (EX_CHUNK * 4) + j * 4
                    )
        comptime if r != EX_ROUNDS - 1:
            barrier()

    if gid != 0:
        return

    # Epilogue (group 0 only): identical to _tn_core_kernel's.
    comptime CALIGN = 16 if VEC_B == 4 else 4
    comptime for i in range(TM):
        var row = bm + wr * WM + tr * TM + i
        if row < m:
            comptime for qn in range(QN):
                var col = bn + wc * WN + qn * (LC * 4) + tc * 4
                var v = acc[i * QN + qn]
                if col + 4 <= n:
                    c_ptr.store[alignment=CALIGN](row * n + col, v)
                else:
                    comptime for j in range(4):
                        if col + j < n:
                            c_ptr[row * n + col + j] = v[j]
