# ===----------------------------------------------------------------------=== #
# Apple-GPU float32 GEMM kernels specialized for the NT layout:
#
#   C[z, m, n] = A[z, m, k] @ B[z, n, k]^T   (both operands row-major, so
#                                             both are k-contiguous)
#
# This is the most-called layout in a transformer training step:
# `nn.Linear` forward (x @ W^T) and SDPA's Q @ K^T both land here. Split out
# of `matmul_ops.mojo` so the NT dispatch can be tuned without touching the
# shared `_apple8_fat_kernel` (still used by the TN / causal callers). Two
# kernels, both 64x64-block / 4-simdgroup 8x8 simdgroup-matrix designs with
# supertile grid rasterization (SWIZZLE):
#   - `_apple8_nt_smem_kernel`: BK-deep double-buffered threadgroup stages.
#     The NT payoff of staging is bigger than in NN: B's k-contiguous rows
#     are read with float4 accesses and transposed on the threadgroup store,
#     so the mma sweep never issues the stride-k per-lane scalar pairs the
#     direct kernel needs for B fragments.
#   - `_apple8_nt_direct_kernel`: DRAM -> register fragment loads with a
#     software-pipelined interior loop. Wins only where the K loop is too
#     short to amortize the staging barriers; the dispatch cut is k < 256
#     (SDPA's head-dim QK^T lands there). From k = 256 up the staged kernel
#     wins at every MN shape, narrow N included: 2048x65x384 is ~25% faster
#     staged, which is why the NT band has no m/n floor.
#
# SWIZZLE is wired up but the production dispatch leaves it at 0: unlike the
# NN layout, supertile rasterization is a *loss* here. Both operands are
# k-contiguous in NT, so a block's B tile is already 64 dense rows and the
# natural row-major block order keeps the C stores of concurrent blocks in
# one contiguous band; the supertile order scatters those stores across
# SWIZZLE block-rows. Measured on M4, isolated launches, 3 processes each:
# 2048x1152x384 0.94-0.97 ms plain vs 1.25-1.42 ms swizzled, 2048x1536x384
# 1.15 vs 1.62. (Swizzling does help the deep-K narrow-N corner, e.g.
# 2048x384x1536 1.6-2.3 ms plain vs a steady 1.57-1.60 ms, and it helps
# every shape when many GEMMs are in flight at once — but the nanoGPT
# training shapes run one at a time and are dominated by the k = 384 cases.)
#
# Bigger blocks were measured and lost again here, with both per-simdgroup
# geometries: 128x64 / 64x128 / 128x128 with 2x-wide subtiles (register
# pressure) *and* with the proven 4x4-fragment subtile and 8 or 16
# simdgroups (occupancy) all run 20-60% behind the 64x64 / 4-simdgroup tile.
# BK = 8 / 32, PAD = 0 / 4 / 8 / 12, and single-buffered staging are all
# neutral-to-worse.
#
# Hardware note (probed on M4, Mojo 1.0.0b3): the Metal backend only knows
# the `llvm.air.simdgroup_matrix_8x8_multiply_accumulate` (a, b, c) form.
# There is no `simdgroup_matrix_8x8_load` handler (unknown-intrinsic error
# path in libmax), and the transpose-flag mma lowers to 16x16x16 intrinsics
# that exist on Apple M5 only. Fragment staging on M1-M4 therefore stays
# software: per-lane vec loads in the `_frag8_layout` pattern.
#
# Batched via `block_idx.z`; split-K over the same axis (grid.z =
# batch * ksplits) with partials reduced by `_nt_ksplit_reduce_kernel`.
# Dynamic shapes: any m/n/k works — OOB fill elements are zeroed so the mma
# loop is always unguarded, and stores are edge-masked.
# ===----------------------------------------------------------------------=== #

from max.gpu.sync import barrier
from std.gpu import block_idx, lane_id, thread_idx, warp_id
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys import llvm_intrinsic

from op_utils import _enqueue_cached, _make_ptr

comptime NT_MMA8_DIM = 8
comptime NT_FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane

# Blocks-in-flight target for the split-K heuristic (Apple's 8-40 core GPUs
# saturate early; each shard costs an m*n partials round-trip).
comptime NT_TARGET_BLOCKS = 80


@always_inline
def _nt_frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout: lane owns
    (row, col_base) and (row, col_base + 1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _nt_mma8x8(
    a: SIMD[DType.float32, NT_FRAG8],
    b: SIMD[DType.float32, NT_FRAG8],
    c: SIMD[DType.float32, NT_FRAG8],
) -> SIMD[DType.float32, NT_FRAG8]:
    """One 8x8x8 simdgroup-matrix multiply-accumulate: D = A @ B + C."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, NT_FRAG8],
    ](a, b, c)


def _nt_ksplit_reduce_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mn_arg: Int64,
    ksplits_arg: Int64,
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var mn = Int(mn_arg)
    var ksplits = Int(ksplits_arg)
    var total = Int(total_arg)

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
# Threadgroup-staged NT kernel, generalized over block geometry.
#
# BM x BN output block, SGR x SGC simdgroups (THREADS = SGR * SGC * 32),
# each simdgroup owning a (BM/SGR) x (BN/SGC) subtile as a grid of 8x8
# fragments. BK-deep K-stages are double-buffered through threadgroup
# memory: every device element is loaded exactly once per block with a
# float4 access (A row-segments straight, B row-segments transposed on the
# threadgroup store), and the store -> barrier -> compute order keeps the
# next stage's device loads in flight during the current stage's mmas with a
# single barrier per stage.
# ---------------------------------------------------------------------------


def _apple8_nt_smem_kernel[
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
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    a_bstride_arg: Int64,
    ksplits_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var a_bstride = Int(a_bstride_arg)
    var ksplits = Int(ksplits_arg)

    comptime THREADS = SGR * SGC * 32
    # Pad the A row stride (and B for symmetry) to stagger threadgroup
    # banks across the 8 fragment rows.
    comptime LDA = BK + PAD
    comptime LDB = BN + PAD
    comptime NBUF = 2 if DOUBLE else 1
    comptime SG_M = BM // SGR
    comptime SG_N = BN // SGC
    comptime NT_M = SG_M // NT_MMA8_DIM
    comptime NT_N = SG_N // NT_MMA8_DIM
    comptime KV = BK // 4  # float4 chunks per k-row (both operands)
    comptime AV = (BM * BK) // (THREADS * 4)  # float4 fills per thread
    comptime BV = (BK * BN) // (THREADS * 4)
    comptime assert AV * THREADS * 4 == BM * BK, "A stage must tile evenly"
    comptime assert BV * THREADS * 4 == BK * BN, "B stage must tile evenly"
    comptime assert BK % NT_MMA8_DIM == 0, "BK must be whole 8-slabs"

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
    # share both a narrow band of A rows and one B row slab, instead of
    # re-streaming all of B once per block row (these GEMMs move several
    # times their operands' bytes on the 120 GB/s parts).
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
    var fl = _nt_frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var sgm = (sg // SGC) * SG_M
    var sgn = (sg % SGC) * SG_N

    var accum = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, NT_FRAG8](0)
    )

    # Device -> register fill for one BK stage at k-offset kt (OOB -> 0).
    # Both operands are k-contiguous here, so both fills are float4 runs
    # along k; only B's threadgroup store transposes.
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
            var nn = fid // KV
            var kq = (fid % KV) * 4
            var row = bn + nn
            var col = kt + kq
            if row < n and col + 4 <= k_end:
                b_regs[q] = (b_ptr + row * k + col).load[width=4]()
            else:
                var v = SIMD[DType.float32, 4](0)
                if row < n:
                    comptime for j in range(4):
                        if col + j < k_end:
                            v[j] = b_ptr[row * k + col + j]
                b_regs[q] = v

    # Register -> threadgroup store for buffer `buf`. B lands transposed:
    # the mma sweep always reads the logical [BK][BN] tile.
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
            var nn = fid // KV
            var kq = (fid % KV) * 4
            comptime for j in range(4):
                b_dst[(kq + j) * LDB + nn] = b_regs[q][j]

    # BK/8 8-slab mma sweeps over threadgroup buffer `buf`.
    @always_inline
    @parameter
    def _compute(
        buf: Int,
        mut acc: InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M * NT_N],
    ):
        var a_src = a_smem + buf * BM * LDA + (sgm + frow) * LDA + fcol
        var b_src = b_smem + buf * BK * LDB + frow * LDB + sgn + fcol
        comptime for kc in range(BK // NT_MMA8_DIM):
            var afrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M](
                uninitialized=True
            )
            comptime for mi in range(NT_M):
                afrag[mi] = (
                    a_src + mi * NT_MMA8_DIM * LDA + kc * NT_MMA8_DIM
                ).load[width=NT_FRAG8]()
            var bfrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_N](
                uninitialized=True
            )
            comptime for ni in range(NT_N):
                bfrag[ni] = (
                    b_src + kc * NT_MMA8_DIM * LDB + ni * NT_MMA8_DIM
                ).load[width=NT_FRAG8]()
            comptime for mi in range(NT_M):
                comptime for ni in range(NT_N):
                    acc[mi * NT_N + ni] = _nt_mma8x8(
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
        # resident per core), at the cost of a store-fence per stage.
        for t in range(stages):
            if t > 0:
                barrier()  # previous stage's readers must be done
            _store_smem(0, a_regs, b_regs)
            barrier()
            if t + 1 < stages:
                _fill_regs(k_start + (t + 1) * BK, a_regs, b_regs)
            _compute(0, accum)

    comptime for mi in range(NT_M):
        var grow = bm + sgm + mi * NT_MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = bn + sgn + ni * NT_MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + NT_FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def apple_nt_smem_enqueue[
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
    """Launch the staged NT kernel, with split-K when the MN grid alone
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
    if blocks < NT_TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(NT_TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        _enqueue_cached[
            _apple8_nt_smem_kernel[
                False, BM, BN, BK, SGR, SGC, PAD, DOUBLE, SWIZZLE
            ]
        ](
            ctx,
            String(
                t"apple8_nt_smem_{BM}x{BN}x{BK}_{SGR}x{SGC}_p{PAD}_d{DOUBLE}"
                t"_s{SWIZZLE}"
            ),
            gx,
            gy,
            batch,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(a_bstride),
            Int64(1),
        )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](batch * ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    _enqueue_cached[
        _apple8_nt_smem_kernel[True, BM, BN, BK, SGR, SGC, PAD, DOUBLE, SWIZZLE]
    ](
        ctx,
        String(
            t"apple8_nt_smem_split_{BM}x{BN}x{BK}_{SGR}x{SGC}_p{PAD}"
            t"_d{DOUBLE}_s{SWIZZLE}"
        ),
        gx,
        gy,
        batch * ksplits,
        THREADS,
        ws_mut,
        a,
        b,
        Int64(m),
        Int64(n),
        Int64(k),
        Int64(a_bstride),
        Int64(ksplits),
    )
    var total = batch * m * n
    var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    _enqueue_cached[_nt_ksplit_reduce_kernel](
        ctx,
        String("nt_ksplit_reduce"),
        ceildiv(total, 1024),
        1,
        1,
        256,
        c_out,
        ws_mut.as_immutable(),
        Int64(m * n),
        Int64(ksplits),
        Int64(total),
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce


# ---------------------------------------------------------------------------
# Direct-load NT kernel: DRAM -> registers with no threadgroup staging, for
# the short-K / thin-grid shapes where the staged kernel's store/barrier
# overhead outweighs its once-per-block loads. 64x64 block, 4 simdgroups
# each owning a 32x32 subtile; software-pipelined pointer-increment interior
# loop and a guarded slab for ragged edges and the K tail. B fragments are
# per-lane scalar pairs one stored n-row (stride k) apart — the price of
# skipping the staging transpose.
# ---------------------------------------------------------------------------


def _apple8_nt_direct_kernel[
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
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    a_bstride_arg: Int64,
    ksplits_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var a_bstride = Int(a_bstride_arg)
    var ksplits = Int(ksplits_arg)

    comptime SG_M = BM // SGR
    comptime SG_N = BN // SGC
    comptime NT_M = SG_M // NT_MMA8_DIM
    comptime NT_N = SG_N // NT_MMA8_DIM

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    # Round the k-chunk to whole 8-slabs so only the last split sees a tail.
    var kchunk = ceildiv(ceildiv(k, ksplits), NT_MMA8_DIM) * NT_MMA8_DIM
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
    var fl = _nt_frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = by * BM + (sg // SGC) * SG_M
    var col_base = bx * BN + (sg % SGC) * SG_N
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    var accum = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, NT_FRAG8](0)
    )

    # One 8-slab with every bound checked: ragged M/N subtiles and the K
    # tail (kk + 8 > k_end). Interior full slabs never come through here.
    @always_inline
    @parameter
    def _slab_guarded(
        kk: Int,
        mut acc: InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M * NT_N],
    ):
        var afrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var grow = row_base + mi * NT_MMA8_DIM + frow
            var af = SIMD[DType.float32, NT_FRAG8](0)
            if grow < m:
                comptime for s in range(NT_FRAG8):
                    if kk + fcol + s < k_end:
                        af[s] = a_ptr[grow * k + kk + fcol + s]
            afrag[mi] = af
        var bfrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            var bf = SIMD[DType.float32, NT_FRAG8](0)
            if kk + frow < k_end:
                comptime for s in range(NT_FRAG8):
                    var gj = col_base + ni * NT_MMA8_DIM + fcol + s
                    if gj < n:
                        bf[s] = b_ptr[gj * k + kk + frow]
            bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _nt_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    # Unguarded fragment loads for one 8-slab: vector loads for A (k-major
    # rows), per-n-row scalar pairs for B stored (N, K).
    @always_inline
    @parameter
    def _load_a_fast(
        ap0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M]:
        var afrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            afrag[mi] = (ap0 + mi * NT_MMA8_DIM * k).load[width=NT_FRAG8]()
        return afrag^

    @always_inline
    @parameter
    def _load_b_fast(
        bp0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, NT_FRAG8], NT_N]:
        var bfrag = InlineArray[SIMD[DType.float32, NT_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            # B is (N, K): the fragment's two slots differ in n-row.
            var p = bp0 + ni * NT_MMA8_DIM * k
            var bf = SIMD[DType.float32, NT_FRAG8](0)
            bf[0] = p[0]
            bf[1] = p[k]
            bfrag[ni] = bf
        return bfrag^

    @always_inline
    @parameter
    def _mma_block(
        afrag: InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M],
        bfrag: InlineArray[SIMD[DType.float32, NT_FRAG8], NT_N],
        mut acc: InlineArray[SIMD[DType.float32, NT_FRAG8], NT_M * NT_N],
    ):
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _nt_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    var full_end = k_start + ((k_end - k_start) // NT_MMA8_DIM) * NT_MMA8_DIM
    if interior:
        # Software-pipelined pointer-increment loop: the next slab's
        # fragments are in flight while the current slab's mmas issue.
        var ap = a_ptr + (row_base + frow) * k + fcol + k_start
        var bp = b_ptr + (col_base + fcol) * k + frow + k_start
        var nslabs = (k_end - k_start) // NT_MMA8_DIM
        if nslabs > 0:
            var cura = _load_a_fast(ap)
            var curb = _load_b_fast(bp)
            for _ in range(nslabs - 1):
                ap += NT_MMA8_DIM
                bp += NT_MMA8_DIM
                var nxta = _load_a_fast(ap)
                var nxtb = _load_b_fast(bp)
                _mma_block(cura, curb, accum)
                cura = nxta^
                curb = nxtb^
            _mma_block(cura, curb, accum)
        if full_end < k_end:
            _slab_guarded(full_end, accum)
    else:
        var kk = k_start
        while kk < k_end:
            _slab_guarded(kk, accum)
            kk += NT_MMA8_DIM

    comptime for mi in range(NT_M):
        var grow = row_base + mi * NT_MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = col_base + ni * NT_MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + NT_FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def apple_nt_direct_enqueue[
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
    """Launch the direct-load NT kernel, with split-K when the MN grid
    alone cannot keep ~2 threadgroups per core busy."""
    comptime THREADS = SGR * SGC * 32
    var gx = ceildiv(n, BN)
    var gy = ceildiv(m, BM)
    comptime if SWIZZLE > 0:
        gx = gx * gy
        gy = 1
    var blocks = gx * gy * batch
    var slabs = ceildiv(k, NT_MMA8_DIM)
    var ksplits = 1
    if blocks < NT_TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(NT_TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        _enqueue_cached[
            _apple8_nt_direct_kernel[False, BM, BN, SGR, SGC, SWIZZLE]
        ](
            ctx,
            String(t"apple8_nt_direct_{BM}x{BN}_{SGR}x{SGC}_s{SWIZZLE}"),
            gx,
            gy,
            batch,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(a_bstride),
            Int64(1),
        )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](batch * ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    _enqueue_cached[_apple8_nt_direct_kernel[True, BM, BN, SGR, SGC, SWIZZLE]](
        ctx,
        String(t"apple8_nt_direct_split_{BM}x{BN}_{SGR}x{SGC}_s{SWIZZLE}"),
        gx,
        gy,
        batch * ksplits,
        THREADS,
        ws_mut,
        a,
        b,
        Int64(m),
        Int64(n),
        Int64(k),
        Int64(a_bstride),
        Int64(ksplits),
    )
    var total = batch * m * n
    var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    _enqueue_cached[_nt_ksplit_reduce_kernel](
        ctx,
        String("nt_ksplit_reduce"),
        ceildiv(total, 1024),
        1,
        1,
        256,
        c_out,
        ws_mut.as_immutable(),
        Int64(m * n),
        Int64(ksplits),
        Int64(total),
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce
