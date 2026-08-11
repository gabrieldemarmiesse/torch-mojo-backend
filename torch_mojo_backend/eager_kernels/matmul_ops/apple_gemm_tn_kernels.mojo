# ===----------------------------------------------------------------------=== #
# Apple-GPU float32 GEMM kernel specialized for the TN layout:
#
#   C[m, n] = A[k, m]^T @ B[k, n]   (both operands row-major in storage, so
#                                    both are *output*-contiguous and strided
#                                    along the reduction axis k)
#
# This is what `nn.Linear`'s backward hands us for dW = dY^T @ X: dY is
# (rows, n_out), X is (rows, k_in), and the reduction runs over the batch
# rows — 25 of the ~30 GEMMs in a nanoGPT training step. Split out of
# `matmul_ops.mojo` (whose `_apple8_fat_kernel[TRANSPOSE_A=True]` served this
# layout before, and still serves the SDPA/causal callers) so the dW dispatch
# can be tuned on its own.
#
# 64x64 or 32x64 output block, 4 or 2 simdgroups, each owning a 32x32 subtile
# as a 4x4 grid of 8x8 simdgroup-matrix fragments. Operands stream DRAM ->
# registers directly, with a software-pipelined pointer-increment loop for
# fully-interior subtiles and a guarded slab for ragged M/N edges and the K
# tail, so any m/n/k works. Batched via `block_idx.z`; split-K over the same
# axis with partials reduced by `_tn_ksplit_reduce_kernel`.
#
# What is specific to TN (measured on M4, 10 cores, floor-free — see the
# measurement note below):
#
# 1. Threadgroup staging loses, and loses badly. The layout looks made for
#    it: a k-row of A is a dense m-run and a k-row of B a dense n-run, so
#    both stage fills are float4 runs and A can be scattered transposed on
#    the threadgroup store, leaving the mma sweep pure vec2 smem reads. A
#    fully tuned staged kernel (BM/BN/BK/SGR/SGC/PAD sweeps, transposed and
#    natural A tiles, double buffering) still ran 8-35% behind direct loads
#    at *every* shape: 2048x1152x384 0.81 vs 0.60 ms, 2048x384x1536 1.04 vs
#    0.78, 8192x768x768 4.01 vs 3.13. The direct kernel's per-lane scalar A
#    pairs (two elements one stored k-row, i.e. `m` floats, apart) still
#    coalesce into 32-byte segments across the 8 lanes that share a fragment
#    row, so staging buys no DRAM efficiency here — it only adds barriers and
#    a 13-18 KiB threadgroup footprint that costs residency.
#
# 2. The block shape has to follow the reduction depth. Halving BM to 32
#    doubles the number of blocks (better load balance on a 10-core GPU, and
#    it lifts small grids over the split-K threshold) but also doubles how
#    often the B panel is streamed. That trade flips with k: at k <= 2048 the
#    32-row block wins 3-16% (2048x384x384 0.242 vs 0.253 ms, 2048x384x1536
#    0.776 vs 0.814, 2048x65x384 0.101 vs 0.120), while at k >= 4096 it loses
#    just as much (4096x768x768 1.81 vs 1.54, 4096x2048x2048 17.8 vs 10.0).
#    Very large MN grids behave like deep k (2048x2048x2048 5.66 vs 5.02), so
#    `apple_tn_gemm_enqueue` picks 32 rows only for shallow-k, moderate
#    grids — plus whenever m's remainder makes a 64-row tiling compute a
#    third more rows than a 32-row one (m = 65 computes 128 rows per 64-row
#    tiling vs 96, and there the 32-row block wins at any k: 8192x65x384
#    0.307 vs 0.384 ms).
#
# 3. Split-K is worth it only for genuinely tiny grids. `TN_TARGET_BLOCKS`
#    keeps the old "~2 threadgroups per core" heuristic, but with 32-row
#    blocks the shapes that used to shard (2048x384x384: 36 blocks at 64x64)
#    now fill the machine on their own, which is most of that shape's win.
#
# Fragment-load hardware intrinsics do not exist on M1-M4 (Metal only lowers
# `llvm.air.simdgroup_matrix_8x8_multiply_accumulate`; the transpose-flag
# form lowers to M5-only 16x16x16 intrinsics), so fragment staging stays
# software: per-lane vec loads in the `_tn_frag8_layout` pattern.
#
# Measurement note: a per-launch synchronize costs ~0.24-0.30 ms on this
# stack, which is more than several of these GEMMs take. Timings above are
# from a reps loop (N enqueues, one sync, divide); a bench that synchronizes
# per launch measures the sync, not the kernel.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_idx, lane_id, thread_idx, warp_id
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import llvm_intrinsic

from op_utils import _enqueue_cached, _make_ptr

comptime TN_MMA8_DIM = 8
comptime TN_FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane

# Blocks-in-flight target for the split-K heuristic (Apple's 8-40 core GPUs
# saturate early; each shard costs an m*n partials round-trip).
comptime TN_TARGET_BLOCKS = 80

# Reduction depth (and MN-grid size) past which the 32-row block's doubled
# B streaming outweighs its extra parallelism. See note 2 in the header.
comptime TN_SHALLOW_K = 2048
comptime TN_WIDE_GRID = 4 * TN_TARGET_BLOCKS


@always_inline
def _tn_frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout: lane owns
    (row, col_base) and (row, col_base + 1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _tn_mma8x8(
    a: SIMD[DType.float32, TN_FRAG8],
    b: SIMD[DType.float32, TN_FRAG8],
    c: SIMD[DType.float32, TN_FRAG8],
) -> SIMD[DType.float32, TN_FRAG8]:
    """One 8x8x8 simdgroup-matrix multiply-accumulate: D = A @ B + C."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, TN_FRAG8],
    ](a, b, c)


def _tn_ksplit_reduce_kernel(
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


def _apple8_tn_kernel[
    SPLIT: Bool,
    BM: Int = 64,
    BN: Int = 64,
    SGR: Int = 2,
    SGC: Int = 2,
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
    comptime NT_M = SG_M // TN_MMA8_DIM
    comptime NT_N = SG_N // TN_MMA8_DIM

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    # Round the k-chunk to whole 8-slabs so only the last split sees a tail.
    var kchunk = ceildiv(ceildiv(k, ksplits), TN_MMA8_DIM) * TN_MMA8_DIM
    var k_start = min(k, ks * kchunk)
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice. a_bstride is 0 when A is batch-shared.
    var c_ptr = c_base + Int(block_idx.z) * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var lane = Int(lane_id())
    var fl = _tn_frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = Int(block_idx.y) * BM + (sg // SGC) * SG_M
    var col_base = Int(block_idx.x) * BN + (sg % SGC) * SG_N
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    var accum = InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, TN_FRAG8](0)
    )

    # One 8-slab with every bound checked: ragged M/N subtiles and the K
    # tail (kk + 8 > k_end). Interior full slabs never come through here.
    @always_inline
    @parameter
    def _slab_guarded(
        kk: Int,
        mut acc: InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M * NT_N],
    ):
        var afrag = InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var grow = row_base + mi * TN_MMA8_DIM + frow
            var af = SIMD[DType.float32, TN_FRAG8](0)
            if grow < m:
                comptime for s in range(TN_FRAG8):
                    if kk + fcol + s < k_end:
                        # A is (K, M): logical row = stored column.
                        af[s] = a_ptr[(kk + fcol + s) * m + grow]
            afrag[mi] = af
        var bfrag = InlineArray[SIMD[DType.float32, TN_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            var bf = SIMD[DType.float32, TN_FRAG8](0)
            if kk + frow < k_end:
                comptime for s in range(TN_FRAG8):
                    var gj = col_base + ni * TN_MMA8_DIM + fcol + s
                    if gj < n:
                        bf[s] = b_ptr[(kk + frow) * n + gj]
            bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _tn_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    # Unguarded fragment loads for one 8-slab: per-stored-row scalar pairs
    # for A (K, M) — the two lanes' elements are `m` floats apart — and
    # vector loads for B (K, N).
    @always_inline
    @parameter
    def _load_a_fast(
        ap0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M]:
        var afrag = InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var p = ap0 + mi * TN_MMA8_DIM
            var af = SIMD[DType.float32, TN_FRAG8](0)
            af[0] = p[0]
            af[1] = p[m]
            afrag[mi] = af
        return afrag^

    @always_inline
    @parameter
    def _load_b_fast(
        bp0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, TN_FRAG8], NT_N]:
        var bfrag = InlineArray[SIMD[DType.float32, TN_FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            bfrag[ni] = (bp0 + ni * TN_MMA8_DIM).load[width=TN_FRAG8]()
        return bfrag^

    @always_inline
    @parameter
    def _mma_block(
        afrag: InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M],
        bfrag: InlineArray[SIMD[DType.float32, TN_FRAG8], NT_N],
        mut acc: InlineArray[SIMD[DType.float32, TN_FRAG8], NT_M * NT_N],
    ):
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _tn_mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    var full_end = k_start + ((k_end - k_start) // TN_MMA8_DIM) * TN_MMA8_DIM
    if interior:
        # Software-pipelined pointer-increment loop: the next slab's
        # fragments are in flight while the current slab's mmas issue, so a
        # single simdgroup hides most of the device-load latency itself.
        var ap = a_ptr + (k_start + fcol) * m + row_base + frow
        var bp = b_ptr + (k_start + frow) * n + col_base + fcol
        var nslabs = (k_end - k_start) // TN_MMA8_DIM
        if nslabs > 0:
            var cura = _load_a_fast(ap)
            var curb = _load_b_fast(bp)
            for _ in range(nslabs - 1):
                ap += TN_MMA8_DIM * m
                bp += TN_MMA8_DIM * n
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
            kk += TN_MMA8_DIM

    comptime for mi in range(NT_M):
        var grow = row_base + mi * TN_MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = col_base + ni * TN_MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + TN_FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def apple_tn_enqueue[
    BM: Int = 64,
    BN: Int = 64,
    SGR: Int = 2,
    SGC: Int = 2,
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
    """Launch the TN kernel at one block geometry, with split-K when the MN
    grid alone cannot keep ~2 threadgroups per core busy."""
    comptime THREADS = SGR * SGC * 32
    var gx = ceildiv(n, BN)
    var gy = ceildiv(m, BM)
    var blocks = gx * gy * batch
    var slabs = ceildiv(k, TN_MMA8_DIM)
    var ksplits = 1
    # Each shard costs an m*n partials round-trip plus a reduce launch.
    if blocks < TN_TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(TN_TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        _enqueue_cached[_apple8_tn_kernel[False, BM, BN, SGR, SGC]](
            ctx,
            String(t"apple8_tn_{BM}x{BN}_{SGR}x{SGC}"),
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
    _enqueue_cached[_apple8_tn_kernel[True, BM, BN, SGR, SGC]](
        ctx,
        String(t"apple8_tn_split_{BM}x{BN}_{SGR}x{SGC}"),
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
    _enqueue_cached[_tn_ksplit_reduce_kernel](
        ctx,
        String("tn_ksplit_reduce"),
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


@always_inline
def apple_tn_gemm_enqueue(
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
    """C = A^T @ B for A stored (k, m) row-major and B stored (k, n)
    row-major, picking the block height (see note 2 in the file header).

    A 32-row block doubles the block count — worth it when the machine is
    otherwise underfilled or when a 64-row tiling would compute a third more
    rows than it needs, but not when k or the MN grid is large enough that
    the doubled B streaming dominates."""
    var m32 = ceildiv(m, 32) * 32
    var m64 = ceildiv(m, 64) * 64
    var blocks64 = ceildiv(m, 64) * ceildiv(n, 64) * batch
    var ragged = 3 * m64 >= 4 * m32  # 64-row tiling wastes >= a third
    var shallow = k <= TN_SHALLOW_K and blocks64 < TN_WIDE_GRID
    if ragged or shallow:
        apple_tn_enqueue[32, 64, 1, 2](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )
    else:
        apple_tn_enqueue[64, 64, 2, 2](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )
