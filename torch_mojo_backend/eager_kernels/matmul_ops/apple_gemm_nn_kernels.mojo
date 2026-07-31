# ===----------------------------------------------------------------------=== #
# Apple-GPU float32 GEMM kernels specialized for the NN layout:
#
#   C[z, m, n] = A[z, m, k] @ B[z, k, n]   (both operands row-major)
#
# Split out of `matmul_ops.mojo` so the NN dispatch can be tuned without
# touching the shared `_apple8_fat_kernel` used by the NT / TN / causal
# callers. Two kernels, both 64x64-block / 4-simdgroup 8x8 simdgroup-matrix
# designs with supertile grid rasterization (SWIZZLE):
#   - `_apple8_nn_smem_kernel`: BK-deep double-buffered threadgroup stages,
#     cooperative float4 fills, one barrier per stage (deep-K shapes, which
#     are DRAM-bound on the ~120 GB/s base M-series parts).
#   - `_apple8_nn_direct_kernel`: DRAM -> register fragment loads with a
#     software-pipelined interior loop (short/mid-K, where staging overhead
#     outweighs once-per-block loads).
# Bigger blocks were measured and lost: 128x64 / 64x128 / 128-thread-2-SG
# staged variants all regress (register pressure), and a thin-K SIMT kernel
# (one thread per 1x4 C tile) ran 3x behind the direct kernel even at k=65
# — Metal SIMT loads cannot compete with the 8x8 mma path here.
#
# The supertile rasterization matters more than any tile-shape choice:
# without it the long-K nanoGPT dgrad shapes are bimodal across processes
# (allocation-address-dependent cache behavior, up to 2x: e.g.
# 1536x384x2048 stock 2.3-2.4 ms vs 1.26-1.55 ms swizzled in the same
# process), and the wide-N 2048x1536x384 shape loses ~2x in multi-tenant
# allocator states.
#
# Hardware note (probed on M4, Mojo 1.0.0b3): the Metal backend only knows
# the `llvm.air.simdgroup_matrix_8x8_multiply_accumulate` (a, b, c) form.
# There is no `simdgroup_matrix_8x8_load` handler (unknown-intrinsic error
# path in libmax), and the transpose-flag mma (`_mma_apple_transposable`,
# "expected expanded operands: A, transpose_a, B, transpose_b, C") lowers to
# the 16x16x16 intrinsics that exist on Apple M5 only. Fragment staging on
# M1-M4 therefore stays software: per-lane vec loads in the
# `_frag8_layout` pattern.
#
# Batched via `block_idx.z`; split-K over the same axis (grid.z =
# batch * ksplits) with partials reduced by `_nn_ksplit_reduce_kernel`.
# Dynamic shapes: any m/n/k works — OOB fill elements are zeroed so the mma
# loop is always unguarded, and stores are edge-masked.
# ===----------------------------------------------------------------------=== #

from std.gpu import barrier, block_idx, lane_id, thread_idx, warp_id
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys import llvm_intrinsic

from op_utils import _enqueue_cached, _make_ptr

comptime NN_MMA8_DIM = 8
comptime NN_FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane

# Blocks-in-flight target for the split-K heuristic (Apple's 8-40 core GPUs
# saturate early; each shard costs an m*n partials round-trip).
comptime NN_TARGET_BLOCKS = 80


@always_inline
def _nn_frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout: lane owns
    (row, col_base) and (row, col_base + 1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _nn_mma8x8(
    a: SIMD[DType.float32, NN_FRAG8],
    b: SIMD[DType.float32, NN_FRAG8],
    c: SIMD[DType.float32, NN_FRAG8],
) -> SIMD[DType.float32, NN_FRAG8]:
    """One 8x8x8 simdgroup-matrix multiply-accumulate: D = A @ B + C."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, NN_FRAG8],
    ](a, b, c)


def _nn_ksplit_reduce_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mn: Int,
    ksplits: Int,
    total: Int,
):
    # 4 outputs per thread: one div/mod chain + vector loads per chunk.
    var i = (Int(block_idx.x) * 256 + Int(thread_idx.x)) * 4
    if i >= total:
        return
    var bz = i // mn
    var off = i % mn
    if off + 4 <= mn and i + 4 <= total:
        var base = bz * ksplits * mn + off
        var acc = SIMD[DType.float32, 4](0)
        for st in range(ksplits):
            acc += ws_ptr.load[width=4](base + st * mn)
        out_ptr.store(i, acc)
    else:
        for u in range(4):
            var iu = i + u
            if iu >= total:
                return
            var bzu = iu // mn
            var offu = iu % mn
            var baseu = bzu * ksplits * mn + offu
            var accu = Scalar[DType.float32](0)
            for st in range(ksplits):
                accu += ws_ptr[baseu + st * mn]
            out_ptr[iu] = accu


# ---------------------------------------------------------------------------
# Threadgroup-staged NN kernel, generalized over block geometry.
#
# BM x BN output block, SGR x SGC simdgroups (THREADS = SGR * SGC * 32),
# each simdgroup owning a (BM/SGR) x (BN/SGC) subtile as a grid of 8x8
# fragments. BK-deep K-stages are double-buffered through threadgroup
# memory: every device element is loaded exactly once per block with a
# float4 access, staged to padded threadgroup arrays, and the
# store -> barrier -> compute order keeps the next stage's device loads in
# flight during the current stage's mmas with a single barrier per stage.
# ---------------------------------------------------------------------------


def _apple8_nn_smem_kernel[
    SPLIT: Bool,
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 16,
    SGR: Int = 2,
    SGC: Int = 2,
    PAD: Int = 2,
    DOUBLE: Bool = True,
    SWIZZLE: Int = 0,
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime THREADS = SGR * SGC * 32
    # Pad the A row stride (and B for symmetry) to stagger threadgroup
    # banks across the 8 fragment rows. PAD = 0 shrinks the footprint to
    # the power-of-two minimum (occupancy experiments).
    comptime LDA = BK + PAD
    comptime LDB = BN + PAD
    comptime NBUF = 2 if DOUBLE else 1
    comptime SG_M = BM // SGR
    comptime SG_N = BN // SGC
    comptime NT_M = SG_M // NN_MMA8_DIM
    comptime NT_N = SG_N // NN_MMA8_DIM
    comptime KV = BK // 4  # float4 chunks per A k-row
    comptime NV = BN // 4  # float4 chunks per B n-row
    comptime AV = (BM * BK) // (THREADS * 4)  # float4 fills per thread
    comptime BV = (BK * BN) // (THREADS * 4)
    comptime assert AV * THREADS * 4 == BM * BK, "A stage must tile evenly"
    comptime assert BV * THREADS * 4 == BK * BN, "B stage must tile evenly"
    comptime assert BK % NN_MMA8_DIM == 0, "BK must be whole 8-slabs"

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = min(k, ks * kchunk)
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice. a_bstride is 0 when A is batch-shared.
    var c_ptr = c_base + Int(block_idx.z) * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    # SWIZZLE > 0: the MN grid is launched flattened into grid.x and
    # rasterized in supertiles of SWIZZLE block-rows — consecutive blocks
    # walk down a column strip, so the ~dozen concurrently-resident blocks
    # share both a narrow band of A rows and one B column slab, instead of
    # re-streaming all of B once per block row (these GEMMs are DRAM-bound
    # on the 120 GB/s parts).
    var bx: Int
    var by: Int
    comptime if SWIZZLE > 0:
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var bid = Int(block_idx.x)
        var span = SWIZZLE * gx
        var group = bid // span
        var rem = bid % span
        var rows = min(SWIZZLE, gy - group * SWIZZLE)  # ragged last group
        by = group * SWIZZLE + rem % rows
        bx = rem // rows
    else:
        bx = Int(block_idx.x)
        by = Int(block_idx.y)
    var bm = by * BM
    var bn = bx * BN
    var tid = Int(thread_idx.x)

    var a_smem = stack_allocation[
        NBUF * BM * LDA, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        NBUF * BK * LDB, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var lane = Int(lane_id())
    var fl = _nn_frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var sgm = (sg // SGC) * SG_M
    var sgn = (sg % SGC) * SG_N

    var accum = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, NN_FRAG8](0)
    )

    # Device -> register fill for one BK stage at k-offset kt (OOB -> 0).
    @always_inline
    @parameter
    def _fill_regs(
        kt: Int,
        mut a_regs: InlineArray[SIMD[DType.float32, 4], AV],
        mut b_regs: InlineArray[SIMD[DType.float32, 4], BV],
    ):
        comptime for q in range(AV):
            var fid = q * THREADS + tid
            var mm = fid // KV
            var kq = (fid % KV) * 4
            var row = bm + mm
            var col = kt + kq
            if row < m and col + 4 <= k_end:
                a_regs[q] = (a_ptr + row * k + col).load[width=4]()
            else:
                var v = SIMD[DType.float32, 4](0)
                if row < m:
                    comptime for j in range(4):
                        if col + j < k_end:
                            v[j] = a_ptr[row * k + col + j]
                a_regs[q] = v
        comptime for q in range(BV):
            var fid = q * THREADS + tid
            var kk = fid // NV
            var nq = (fid % NV) * 4
            var row = kt + kk
            var col = bn + nq
            if row < k_end and col + 4 <= n:
                b_regs[q] = (b_ptr + row * n + col).load[width=4]()
            else:
                var v = SIMD[DType.float32, 4](0)
                if row < k_end:
                    comptime for j in range(4):
                        if col + j < n:
                            v[j] = b_ptr[row * n + col + j]
                b_regs[q] = v

    # Register -> threadgroup store for buffer `buf`.
    @always_inline
    @parameter
    def _store_smem(
        buf: Int,
        a_regs: InlineArray[SIMD[DType.float32, 4], AV],
        b_regs: InlineArray[SIMD[DType.float32, 4], BV],
    ):
        var a_dst = a_smem + buf * BM * LDA
        var b_dst = b_smem + buf * BK * LDB
        comptime for q in range(AV):
            var fid = q * THREADS + tid
            var mm = fid // KV
            var kq = (fid % KV) * 4
            a_dst.store(mm * LDA + kq, a_regs[q])
        comptime for q in range(BV):
            var fid = q * THREADS + tid
            var kk = fid // NV
            var nq = (fid % NV) * 4
            b_dst.store(kk * LDB + nq, b_regs[q])

    # BK/8 8-slab mma sweeps over threadgroup buffer `buf`.
    @always_inline
    @parameter
    def _compute(
        buf: Int,
        mut acc: InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M * NT_N],
    ):
        var a_src = a_smem + buf * BM * LDA + (sgm + frow) * LDA + fcol
        var b_src = b_smem + buf * BK * LDB + frow * LDB + sgn + fcol
        comptime for kc in range(BK // NN_MMA8_DIM):
            var afrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M](
                uninitialized=True
            )
            comptime for mi in range(NT_M):
                afrag[mi] = (
                    a_src + mi * NN_MMA8_DIM * LDA + kc * NN_MMA8_DIM
                ).load[width=NN_FRAG8]()
            var bfrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_N](
                uninitialized=True
            )
            comptime for ni in range(NT_N):
                bfrag[ni] = (
                    b_src + kc * NN_MMA8_DIM * LDB + ni * NN_MMA8_DIM
                ).load[width=NN_FRAG8]()
            comptime for mi in range(NT_M):
                comptime for ni in range(NT_N):
                    acc[mi * NT_N + ni] = _nn_mma8x8(
                        afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                    )

    var stages = ceildiv(k_end - k_start, BK)
    var a_regs = InlineArray[SIMD[DType.float32, 4], AV](uninitialized=True)
    var b_regs = InlineArray[SIMD[DType.float32, 4], BV](uninitialized=True)
    if stages > 0:
        _fill_regs(k_start, a_regs, b_regs)
    comptime if DOUBLE:
        for t in range(stages):
            # Writing buffer t%2 is safe without a leading barrier: its
            # previous readers all passed the barrier below in iteration
            # t-1.
            _store_smem(t & 1, a_regs, b_regs)
            barrier()
            if t + 1 < stages:
                _fill_regs(k_start + (t + 1) * BK, a_regs, b_regs)
            _compute(t & 1, accum)
    else:
        # Single buffer, two barriers per stage: half the threadgroup
        # footprint of the double-buffered loop (more threadgroups can be
        # resident per core), at the cost of a store-fence per stage. The
        # next stage's device loads still issue before the mma sweep.
        for t in range(stages):
            if t > 0:
                barrier()  # previous stage's readers must be done
            _store_smem(0, a_regs, b_regs)
            barrier()
            if t + 1 < stages:
                _fill_regs(k_start + (t + 1) * BK, a_regs, b_regs)
            _compute(0, accum)

    comptime for mi in range(NT_M):
        var grow = bm + sgm + mi * NN_MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = bn + sgn + ni * NN_MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + NN_FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def apple_nn_smem_enqueue[
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 16,
    SGR: Int = 2,
    SGC: Int = 2,
    PAD: Int = 2,
    DOUBLE: Bool = True,
    SWIZZLE: Int = 0,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    """Launch the staged NN kernel, with split-K when the MN grid alone
    cannot keep ~2 threadgroups per core busy."""
    comptime THREADS = SGR * SGC * 32
    var gx = ceildiv(n, BN)
    var gy = ceildiv(m, BM)
    comptime if SWIZZLE > 0:
        # Flattened supertile rasterization: the kernel re-derives the
        # (bx, by) block coordinates from block_idx.x.
        gx = gx * gy
        gy = 1
    var blocks = gx * gy * batch
    var slabs = ceildiv(k, BK)
    var ksplits = 1
    if blocks < NN_TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(NN_TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        _enqueue_cached[
            _apple8_nn_smem_kernel[
                False, BM, BN, BK, SGR, SGC, PAD, DOUBLE, SWIZZLE
            ]
        ](
            ctx,
            String(
                t"apple8_nn_smem_{BM}x{BN}x{BK}_{SGR}x{SGC}_p{PAD}_d{DOUBLE}"
                t"_s{SWIZZLE}"
            ),
            gx,
            gy,
            batch,
            THREADS,
            c,
            a,
            b,
            m,
            n,
            k,
            a_bstride,
            1,
        )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](batch * ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    _enqueue_cached[
        _apple8_nn_smem_kernel[True, BM, BN, BK, SGR, SGC, PAD, DOUBLE, SWIZZLE]
    ](
        ctx,
        String(
            t"apple8_nn_smem_split_{BM}x{BN}x{BK}_{SGR}x{SGC}_p{PAD}"
            t"_d{DOUBLE}_s{SWIZZLE}"
        ),
        gx,
        gy,
        batch * ksplits,
        THREADS,
        ws_mut,
        a,
        b,
        m,
        n,
        k,
        a_bstride,
        ksplits,
    )
    var total = batch * m * n
    var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    _enqueue_cached[_nn_ksplit_reduce_kernel](
        ctx,
        String("nn_ksplit_reduce"),
        ceildiv(total, 1024),
        1,
        1,
        256,
        c_out,
        ws_mut.as_immutable(),
        m * n,
        ksplits,
        total,
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce


# ---------------------------------------------------------------------------
# Direct-load NN kernel: DRAM -> registers with no threadgroup staging (the
# leaner choice for the short/mid-K NN shapes, where the staged kernel's
# store/barrier overhead outweighs its once-per-block loads), NN-specialized
# from `_apple8_fat_kernel` with the same supertile rasterization option as
# the staged kernel above. 64x64 block, 4 simdgroups each owning a 32x32
# subtile; software-pipelined pointer-increment interior loop and a guarded
# slab for ragged edges and the K tail.
# ---------------------------------------------------------------------------


def _apple8_nn_direct_kernel[
    SPLIT: Bool,
    BM: Int = 64,
    BN: Int = 64,
    SGR: Int = 2,
    SGC: Int = 2,
    SWIZZLE: Int = 0,
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime SG_M = BM // SGR
    comptime SG_N = BN // SGC
    comptime NT_M = SG_M // NN_MMA8_DIM
    comptime NT_N = SG_N // NN_MMA8_DIM

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    # Round the k-chunk to whole 8-slabs so only the last split sees a tail.
    var kchunk = ceildiv(ceildiv(k, ksplits), NN_MMA8_DIM) * NN_MMA8_DIM
    var k_start = min(k, ks * kchunk)
    var k_end = min(k, k_start + kchunk)

    var c_ptr = c_base + Int(block_idx.z) * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bx: Int
    var by: Int
    comptime if SWIZZLE > 0:
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var bid = Int(block_idx.x)
        var span = SWIZZLE * gx
        var group = bid // span
        var rem = bid % span
        var rows = min(SWIZZLE, gy - group * SWIZZLE)  # ragged last group
        by = group * SWIZZLE + rem % rows
        bx = rem // rows
    else:
        bx = Int(block_idx.x)
        by = Int(block_idx.y)

    var lane = Int(lane_id())
    var fl = _nn_frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = by * BM + (sg // SGC) * SG_M
    var col_base = bx * BN + (sg % SGC) * SG_N
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    var accum = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, NN_FRAG8](0)
    )

    # One 8-slab with every bound checked: ragged M/N subtiles and the K
    # tail (kk + 8 > k_end). Interior full slabs never come through here.
    @always_inline
    @parameter
    def _slab_guarded(
        kk: Int,
        mut acc: InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M * NT_N],
    ):
        var afrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var grow = row_base + mi * NN_MMA8_DIM + frow
            var af = SIMD[DType.float32, NN_FRAG8](0)
            if grow < m:
                comptime for s in range(NN_FRAG8):
                    if kk + fcol + s < k_end:
                        af[s] = a_ptr[grow * k + kk + fcol + s]
            afrag[mi] = af
        var bfrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            var bf = SIMD[DType.float32, NN_FRAG8](0)
            if kk + frow < k_end:
                comptime for s in range(NN_FRAG8):
                    var gj = col_base + ni * NN_MMA8_DIM + fcol + s
                    if gj < n:
                        bf[s] = b_ptr[(kk + frow) * n + gj]
            bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _nn_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    # Unguarded vector fragment loads for one 8-slab (both operands are
    # k-contiguous in the NN layout).
    @always_inline
    @parameter
    def _load_a_fast(
        ap0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M]:
        var afrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            afrag[mi] = (ap0 + mi * NN_MMA8_DIM * k).load[width=NN_FRAG8]()
        return afrag

    @always_inline
    @parameter
    def _load_b_fast(
        bp0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, NN_FRAG8], NT_N]:
        var bfrag = InlineArray[SIMD[DType.float32, NN_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            bfrag[ni] = (bp0 + ni * NN_MMA8_DIM).load[width=NN_FRAG8]()
        return bfrag

    @always_inline
    @parameter
    def _mma_block(
        afrag: InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M],
        bfrag: InlineArray[SIMD[DType.float32, NN_FRAG8], NT_N],
        mut acc: InlineArray[SIMD[DType.float32, NN_FRAG8], NT_M * NT_N],
    ):
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _nn_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    var full_end = k_start + ((k_end - k_start) // NN_MMA8_DIM) * NN_MMA8_DIM
    if interior:
        # Software-pipelined pointer-increment loop: the next slab's
        # fragments are in flight while the current slab's mmas issue.
        var ap = a_ptr + (row_base + frow) * k + fcol + k_start
        var bp = b_ptr + (k_start + frow) * n + col_base + fcol
        var nslabs = (k_end - k_start) // NN_MMA8_DIM
        if nslabs > 0:
            var cura = _load_a_fast(ap)
            var curb = _load_b_fast(bp)
            for _ in range(nslabs - 1):
                ap += NN_MMA8_DIM
                bp += NN_MMA8_DIM * n
                var nxta = _load_a_fast(ap)
                var nxtb = _load_b_fast(bp)
                _mma_block(cura, curb, accum)
                cura = nxta
                curb = nxtb
            _mma_block(cura, curb, accum)
        if full_end < k_end:
            _slab_guarded(full_end, accum)
    else:
        var kk = k_start
        while kk < k_end:
            _slab_guarded(kk, accum)
            kk += NN_MMA8_DIM

    comptime for mi in range(NT_M):
        var grow = row_base + mi * NN_MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = col_base + ni * NN_MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + NN_FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def apple_nn_direct_enqueue[
    BM: Int = 64,
    BN: Int = 64,
    SGR: Int = 2,
    SGC: Int = 2,
    SWIZZLE: Int = 0,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    """Launch the direct-load NN kernel, with split-K when the MN grid
    alone cannot keep ~2 threadgroups per core busy."""
    comptime THREADS = SGR * SGC * 32
    var gx = ceildiv(n, BN)
    var gy = ceildiv(m, BM)
    comptime if SWIZZLE > 0:
        gx = gx * gy
        gy = 1
    var blocks = gx * gy * batch
    var slabs = ceildiv(k, NN_MMA8_DIM)
    var ksplits = 1
    if blocks < NN_TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(NN_TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        _enqueue_cached[
            _apple8_nn_direct_kernel[False, BM, BN, SGR, SGC, SWIZZLE]
        ](
            ctx,
            String(t"apple8_nn_direct_{BM}x{BN}_{SGR}x{SGC}_s{SWIZZLE}"),
            gx,
            gy,
            batch,
            THREADS,
            c,
            a,
            b,
            m,
            n,
            k,
            a_bstride,
            1,
        )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](batch * ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    _enqueue_cached[_apple8_nn_direct_kernel[True, BM, BN, SGR, SGC, SWIZZLE]](
        ctx,
        String(t"apple8_nn_direct_split_{BM}x{BN}_{SGR}x{SGC}_s{SWIZZLE}"),
        gx,
        gy,
        batch * ksplits,
        THREADS,
        ws_mut,
        a,
        b,
        m,
        n,
        k,
        a_bstride,
        ksplits,
    )
    var total = batch * m * n
    var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    _enqueue_cached[_nn_ksplit_reduce_kernel](
        ctx,
        String("nn_ksplit_reduce"),
        ceildiv(total, 1024),
        1,
        1,
        256,
        c_out,
        ws_mut.as_immutable(),
        m * n,
        ksplits,
        total,
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce
