"""Fable-owned opt-in H100 TF32 GEMM/BMM candidate.

Semantics: ``output = tf32(A_logical) @ tf32(B_logical) (+ bias)`` with FP32
accumulation and FP32 output.  Operands are explicitly rounded to TF32 with
round-to-nearest-even before every tensor-core multiply; on sm_90+ that is a
single ``cvt.rn.tf32.f32`` per fragment element, elsewhere an equivalent
integer RN-even round.  The harness precision probes require exactly this
conversion on both operands.

One tile kernel implements both entry points.  Design:

  * ``mma.m16n8k8`` TF32 fragments through the public ``mma`` primitive.
  * A multi-stage ``cp.async`` pipeline (3-4 stages of dynamic shared
    memory).  Global tiles are copied raw and rounded at fragment-load time.
  * Per-layout shared tiles: each operand is staged so that the contiguous
    global dimension is also contiguous in shared memory, which keeps both
    global loads coalesced and shared stores conflict-free for all four
    transpose combinations.  Padding (+8 words on k-major tiles, +4 on
    k-minor tiles) makes the fragment loads bank-conflict-free.
  * Vectorized 16-byte ``cp.async`` when a runtime proof holds (16-byte base
    alignment, leading dimension % 4 == 0, batch stride % 4 == 0); otherwise
    guarded 4-byte copies.  Out-of-range elements are zero-filled through
    ``src_size`` so MMA tails contribute exact +0 products.
  * Runtime tile regimes 128x128/128x64/64x128/64x64 selected from M, N,
    batch count, and the SM count; no shape is compiled in.
  * BMM batches map to ``grid.z`` (with a stride loop past 65535); each
    batch applies its runtime element stride to the operand base pointers.
  * A group-of-8 block swizzle walks tiles column-major inside each group
    for L2 reuse of the B operand on wide-N shapes (the LM head).

Every logical output element is stored exactly once per call by exactly one
thread with a fixed accumulation order, so replays are bitwise
deterministic.  Bias (2-D only) is added once in FP32 in the epilogue.

Non-NVIDIA targets compile a plain SIMT fallback body behind a compile-time
target predicate; production gating routes those targets to the existing
strict fallback before this candidate is reached.
"""

from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.compute.mma import mma
from std.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from std.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from std.math import ceildiv
from std.memory import AddressSpace, bitcast
from std.sys import inlined_assembly, is_nvidia_gpu
from std.sys.info import _is_sm_9x_or_newer

comptime _BK = 16
comptime _SWIZZLE_GROUP = 8
comptime _MAX_GRID_Z = 65535


@always_inline
def _tf32(value: Float32) -> Float32:
    """Round-to-nearest-even TF32 conversion, kept in an f32 container."""
    comptime if is_nvidia_gpu() and _is_sm_9x_or_newer():
        var out = inlined_assembly[
            "cvt.rn.tf32.f32 $0, $1;",
            UInt32,
            constraints="=r,f",
            has_side_effect=False,
        ](value)
        return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](out))[0]
    else:
        var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](value))[0]
        var magnitude = bits & UInt32(0x7FFFFFFF)
        if magnitude >= UInt32(0x7F800000):
            return value
        var retained_lsb = (bits >> 13) & UInt32(1)
        bits = (bits + UInt32(0x00000FFF) + retained_lsb) & UInt32(0xFFFFE000)
        return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))[0]


@always_inline
def _stage_words[TA: Bool, TB: Bool, BM: Int, BN: Int]() -> Int:
    # Shared tiles mirror the physical operand orientation.  k-major tiles
    # ([BK][major + 8]) hold physically transposed A / untransposed B;
    # k-minor tiles ([major][BK + 4]) hold the other two orientations.
    comptime SA = _BK * (BM + 8) if TA else BM * (_BK + 4)
    comptime SB = BN * (_BK + 4) if TB else _BK * (BN + 8)
    return SA + SB


@always_inline
def _tile_body[
    TA: Bool,
    TB: Bool,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    THREADS: Int,
    STAGES: Int,
](
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    has_bias: Int,
    a_vec: Int,
    b_vec: Int,
    c_vec: Int,
):
    comptime WARPS_M = BM // WM
    comptime MT = WM // 16
    comptime NT = WN // 8
    comptime AR = _BK if TA else BM
    comptime AC = BM if TA else _BK
    comptime APAD = 8 if TA else 4
    comptime BR = BN if TB else _BK
    comptime BC = _BK if TB else BN
    comptime BPAD = 4 if TB else 8
    comptime SA = AR * (AC + APAD)
    comptime SB = BR * (BC + BPAD)
    comptime STAGE_WORDS = SA + SB

    var tid = Int(thread_idx.x)
    var tiles_n = ceildiv(n, BN)
    var tiles_m = ceildiv(m, BM)
    # Group swizzle: consecutive blocks walk M-tiles first inside a group of
    # _SWIZZLE_GROUP rows, so one B tile is L2-resident across the group.
    var linear = Int(block_idx.x)
    var group_blocks = _SWIZZLE_GROUP * tiles_n
    var group = linear // group_blocks
    var first_m = group * _SWIZZLE_GROUP
    var rows_in_group = min(_SWIZZLE_GROUP, tiles_m - first_m)
    var within = linear - group * group_blocks
    var tile_m = first_m + within % rows_in_group
    var tile_n = within // rows_in_group
    var m0 = tile_m * BM
    var n0 = tile_n * BN
    var lda = m if TA else k
    var ldb = k if TB else n
    var k_tiles = ceildiv(k, _BK)

    comptime if is_nvidia_gpu():
        var smem = external_memory[
            Scalar[DType.float32],
            address_space=AddressSpace.SHARED,
            alignment=16,
        ]()
        # NVIDIA MMA fragment ownership is defined in 32-lane warps.
        var warp = tid // 32
        var lane = tid - warp * 32
        var g = lane >> 2
        var t = lane & 3
        var warp_m = warp % WARPS_M
        var warp_n = warp // WARPS_M
        var wm0 = warp_m * WM
        var wn0 = warp_n * WN

        var bz = Int(block_idx.z)
        while bz < batch_count:
            var a_base = a + bz * a_bs
            var b_base = b + bz * b_bs
            var c_base = c + bz * c_bs

            @parameter
            @always_inline
            def copy_a(ktile: Int, stage: Int):
                var sa_ptr = smem + stage * STAGE_WORDS
                var k0 = ktile * _BK
                var tile = a_base + (k0 * lda + m0 if TA else m0 * lda + k0)
                var rows_avail = (k - k0) if TA else (m - m0)
                var cols_avail = (m - m0) if TA else (k - k0)
                if a_vec != 0:
                    comptime AV = AC // 4
                    comptime for i in range(ceildiv(AR * AV, THREADS)):
                        var idx = tid + i * THREADS
                        if idx < AR * AV:
                            var r = idx // AV
                            var col = (idx - r * AV) * 4
                            var dst = sa_ptr + r * (AC + APAD) + col
                            if r < rows_avail and col + 3 < cols_avail:
                                async_copy[16](
                                    (tile + r * lda + col).address_space_cast[
                                        AddressSpace.GLOBAL
                                    ](),
                                    dst,
                                )
                            else:
                                # The pinned async_copy has no zero-fill;
                                # guard tails with plain element copies.
                                comptime for j in range(4):
                                    if r < rows_avail and col + j < cols_avail:
                                        async_copy[4](
                                            (
                                                tile + r * lda + col + j
                                            ).address_space_cast[
                                                AddressSpace.GLOBAL
                                            ](),
                                            dst + j,
                                        )
                                    else:
                                        dst[j] = 0.0
                else:
                    comptime for i in range(ceildiv(AR * AC, THREADS)):
                        var idx = tid + i * THREADS
                        if idx < AR * AC:
                            var r = idx // AC
                            var col = idx - r * AC
                            var dst = sa_ptr + r * (AC + APAD) + col
                            if r < rows_avail and col < cols_avail:
                                async_copy[4](
                                    (tile + r * lda + col).address_space_cast[
                                        AddressSpace.GLOBAL
                                    ](),
                                    dst,
                                )
                            else:
                                dst[0] = 0.0

            @parameter
            @always_inline
            def copy_b(ktile: Int, stage: Int):
                var sb_ptr = smem + stage * STAGE_WORDS + SA
                var k0 = ktile * _BK
                var tile = b_base + (n0 * ldb + k0 if TB else k0 * ldb + n0)
                var rows_avail = (n - n0) if TB else (k - k0)
                var cols_avail = (k - k0) if TB else (n - n0)
                if b_vec != 0:
                    comptime BV = BC // 4
                    comptime for i in range(ceildiv(BR * BV, THREADS)):
                        var idx = tid + i * THREADS
                        if idx < BR * BV:
                            var r = idx // BV
                            var col = (idx - r * BV) * 4
                            var dst = sb_ptr + r * (BC + BPAD) + col
                            if r < rows_avail and col + 3 < cols_avail:
                                async_copy[16](
                                    (tile + r * ldb + col).address_space_cast[
                                        AddressSpace.GLOBAL
                                    ](),
                                    dst,
                                )
                            else:
                                comptime for j in range(4):
                                    if r < rows_avail and col + j < cols_avail:
                                        async_copy[4](
                                            (
                                                tile + r * ldb + col + j
                                            ).address_space_cast[
                                                AddressSpace.GLOBAL
                                            ](),
                                            dst + j,
                                        )
                                    else:
                                        dst[j] = 0.0
                else:
                    comptime for i in range(ceildiv(BR * BC, THREADS)):
                        var idx = tid + i * THREADS
                        if idx < BR * BC:
                            var r = idx // BC
                            var col = idx - r * BC
                            var dst = sb_ptr + r * (BC + BPAD) + col
                            if r < rows_avail and col < cols_avail:
                                async_copy[4](
                                    (tile + r * ldb + col).address_space_cast[
                                        AddressSpace.GLOBAL
                                    ](),
                                    dst,
                                )
                            else:
                                dst[0] = 0.0

            # The batch loop reuses all stages; the previous iteration's
            # consumers must drain before new copies land in stage 0.
            barrier()

            var acc = InlineArray[SIMD[DType.float32, 4], MT * NT](
                uninitialized=True
            )
            comptime for i in range(MT * NT):
                acc[i] = SIMD[DType.float32, 4](0.0)

            comptime for s in range(STAGES - 1):
                if s < k_tiles:
                    copy_a(s, s)
                    copy_b(s, s)
                async_copy_commit_group()

            for kt in range(k_tiles):
                async_copy_wait_group(Int32(STAGES - 2))
                barrier()
                var prefetch = kt + STAGES - 1
                if prefetch < k_tiles:
                    copy_a(prefetch, prefetch % STAGES)
                    copy_b(prefetch, prefetch % STAGES)
                async_copy_commit_group()

                var sa_ptr = smem + (kt % STAGES) * STAGE_WORDS
                var sb_ptr = sa_ptr + SA
                comptime for ks in range(_BK // 8):
                    var a_frag = InlineArray[SIMD[DType.float32, 4], MT](
                        uninitialized=True
                    )
                    comptime for mt in range(MT):
                        comptime if TA:
                            var base = (ks * 8 + t) * (AC + APAD) + (
                                wm0 + mt * 16 + g
                            )
                            a_frag[mt] = SIMD[DType.float32, 4](
                                _tf32(sa_ptr[base]),
                                _tf32(sa_ptr[base + 8]),
                                _tf32(sa_ptr[base + 4 * (AC + APAD)]),
                                _tf32(sa_ptr[base + 4 * (AC + APAD) + 8]),
                            )
                        else:
                            var base = (wm0 + mt * 16 + g) * (AC + APAD) + (
                                ks * 8 + t
                            )
                            a_frag[mt] = SIMD[DType.float32, 4](
                                _tf32(sa_ptr[base]),
                                _tf32(sa_ptr[base + 8 * (AC + APAD)]),
                                _tf32(sa_ptr[base + 4]),
                                _tf32(sa_ptr[base + 8 * (AC + APAD) + 4]),
                            )
                    var b_frag = InlineArray[SIMD[DType.float32, 2], NT](
                        uninitialized=True
                    )
                    comptime for nt in range(NT):
                        comptime if TB:
                            var base = (wn0 + nt * 8 + g) * (BC + BPAD) + (
                                ks * 8 + t
                            )
                            b_frag[nt] = SIMD[DType.float32, 2](
                                _tf32(sb_ptr[base]),
                                _tf32(sb_ptr[base + 4]),
                            )
                        else:
                            var base = (ks * 8 + t) * (BC + BPAD) + (
                                wn0 + nt * 8 + g
                            )
                            b_frag[nt] = SIMD[DType.float32, 2](
                                _tf32(sb_ptr[base]),
                                _tf32(sb_ptr[base + 4 * (BC + BPAD)]),
                            )
                    comptime for mt in range(MT):
                        comptime for nt in range(NT):
                            mma(
                                acc[mt * NT + nt],
                                a_frag[mt],
                                b_frag[nt],
                                acc[mt * NT + nt],
                            )

            comptime for nt in range(NT):
                var col = n0 + wn0 + nt * 8 + t * 2
                var bias0 = Float32(0.0)
                var bias1 = Float32(0.0)
                if has_bias != 0 and col < n:
                    bias0 = bias[col]
                    if col + 1 < n:
                        bias1 = bias[col + 1]
                comptime for mt in range(MT):
                    var v = acc[mt * NT + nt]
                    var row = m0 + wm0 + mt * 16 + g
                    if c_vec != 0:
                        if col < n:
                            if row < m:
                                c_base.store[width=2, alignment=8](
                                    row * n + col,
                                    SIMD[DType.float32, 2](
                                        v[0] + bias0, v[1] + bias1
                                    ),
                                )
                            if row + 8 < m:
                                c_base.store[width=2, alignment=8](
                                    (row + 8) * n + col,
                                    SIMD[DType.float32, 2](
                                        v[2] + bias0, v[3] + bias1
                                    ),
                                )
                    else:
                        if row < m:
                            if col < n:
                                c_base[row * n + col] = v[0] + bias0
                            if col + 1 < n:
                                c_base[row * n + col + 1] = v[1] + bias1
                        if row + 8 < m:
                            if col < n:
                                c_base[(row + 8) * n + col] = v[2] + bias0
                            if col + 1 < n:
                                c_base[(row + 8) * n + col + 1] = v[3] + bias1
            bz += Int(grid_dim.z)
    else:
        # Portable SIMT fallback so importing this module on another
        # accelerator target still compiles; production gating keeps other
        # devices on the existing strict fallback path.
        var bz = Int(block_idx.z)
        while bz < batch_count:
            var a_base = a + bz * a_bs
            var b_base = b + bz * b_bs
            var c_base = c + bz * c_bs
            var idx = tid
            while idx < BM * BN:
                var mm = idx // BN
                var nn = idx - mm * BN
                var row = m0 + mm
                var col = n0 + nn
                if row < m and col < n:
                    var total = Float32(0.0)
                    for r in range(k):
                        var av = a_base[r * lda + row if TA else row * lda + r]
                        var bv = b_base[col * ldb + r if TB else r * ldb + col]
                        total += _tf32(av) * _tf32(bv)
                    if has_bias != 0:
                        total += bias[col]
                    c_base[row * n + col] = total
                idx += THREADS
            bz += Int(grid_dim.z)


@__name("nanogpt_tf32_gemm_tile")
def _gemm_tile[
    TA: Bool,
    TB: Bool,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    THREADS: Int,
    STAGES: Int,
](
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_vec: Int,
    b_vec: Int,
    c_vec: Int,
):
    _tile_body[TA, TB, BM, BN, WM, WN, THREADS, STAGES](
        c, a, b, bias, m, n, k, 1, 0, 0, 0, has_bias, a_vec, b_vec, c_vec
    )


@__name("nanogpt_tf32_bmm_tile")
def _bmm_tile[
    TA: Bool,
    TB: Bool,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    THREADS: Int,
    STAGES: Int,
](
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    a_vec: Int,
    b_vec: Int,
    c_vec: Int,
):
    _tile_body[TA, TB, BM, BN, WM, WN, THREADS, STAGES](
        c,
        a,
        b,
        a,
        m,
        n,
        k,
        batch_count,
        c_bs,
        a_bs,
        b_bs,
        0,
        a_vec,
        b_vec,
        c_vec,
    )


@always_inline
def _launch_tile[
    BATCHED: Bool,
    TA: Bool,
    TB: Bool,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    THREADS: Int,
    STAGES: Int,
](
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    comptime SMEM_BYTES = 4 * STAGES * _stage_words[TA, TB, BM, BN]()
    var lda = m if TA else k
    var ldb = k if TB else n
    # Vector copies only after a runtime proof: 16-byte base alignment plus
    # leading-dimension and batch-stride divisibility keep every 16-byte
    # cp.async source aligned for every tile of every batch.
    var a_vec = 1 if (
        Int(a) % 16 == 0 and lda % 4 == 0 and a_bs % 4 == 0
    ) else 0
    var b_vec = 1 if (
        Int(b) % 16 == 0 and ldb % 4 == 0 and b_bs % 4 == 0
    ) else 0
    var c_vec = 1 if (Int(c) % 8 == 0 and n % 2 == 0 and c_bs % 2 == 0) else 0
    var tiles = ceildiv(m, BM) * ceildiv(n, BN)
    var gz = min(batch_count, _MAX_GRID_Z)
    comptime if BATCHED:
        ctx.enqueue_function[
            _bmm_tile[TA, TB, BM, BN, WM, WN, THREADS, STAGES]
        ](
            c,
            a,
            b,
            m,
            n,
            k,
            batch_count,
            c_bs,
            a_bs,
            b_bs,
            a_vec,
            b_vec,
            c_vec,
            grid_dim=(tiles, 1, gz),
            block_dim=(THREADS,),
            shared_mem_bytes=SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(SMEM_BYTES)
            ),
        )
    else:
        ctx.enqueue_function[
            _gemm_tile[TA, TB, BM, BN, WM, WN, THREADS, STAGES]
        ](
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            1 if has_bias else 0,
            a_vec,
            b_vec,
            c_vec,
            grid_dim=(tiles, 1, 1),
            block_dim=(THREADS,),
            shared_mem_bytes=SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(SMEM_BYTES)
            ),
        )


@always_inline
def _dispatch[
    BATCHED: Bool, TA: Bool, TB: Bool
](
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bs: Int,
    a_bs: Int,
    b_bs: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    var sm_count: Int
    try:
        sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        sm_count = 108
    var target = sm_count
    # Tile regime: avoid tiles that waste more than half a dimension, then
    # shrink 128-wide dimensions while the grid cannot cover the device.
    var bm = 128 if m > 96 else 64
    var bn = 128 if n > 96 else 64
    # Narrow-N shapes (attention P@V / dY@V.T columns) measure faster with
    # square 64 tiles than with a 128-tall tile.
    if n <= 64:
        bm = 64
    if bm == 128 and ceildiv(m, bm) * ceildiv(n, bn) * batch_count < target:
        bm = 64
    if bn == 128 and ceildiv(m, bm) * ceildiv(n, bn) * batch_count < target:
        bn = 64
    if bm == 128 and bn == 128:
        _launch_tile[BATCHED, TA, TB, 128, 128, 64, 32, 256, 4](
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            batch_count,
            c_bs,
            a_bs,
            b_bs,
            has_bias,
            ctx,
        )
    elif bm == 128:
        _launch_tile[BATCHED, TA, TB, 128, 64, 64, 32, 128, 5](
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            batch_count,
            c_bs,
            a_bs,
            b_bs,
            has_bias,
            ctx,
        )
    elif bn == 128:
        _launch_tile[BATCHED, TA, TB, 64, 128, 32, 64, 128, 5](
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            batch_count,
            c_bs,
            a_bs,
            b_bs,
            has_bias,
            ctx,
        )
    else:
        _launch_tile[BATCHED, TA, TB, 64, 64, 32, 32, 128, 5](
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            batch_count,
            c_bs,
            a_bs,
            b_bs,
            has_bias,
            ctx,
        )


def enqueue_tf32_gemm_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    if transpose_a:
        if transpose_b:
            _dispatch[False, True, True](
                output, a, b, bias, m, n, k, 1, 0, 0, 0, has_bias, ctx
            )
        else:
            _dispatch[False, True, False](
                output, a, b, bias, m, n, k, 1, 0, 0, 0, has_bias, ctx
            )
    else:
        if transpose_b:
            _dispatch[False, False, True](
                output, a, b, bias, m, n, k, 1, 0, 0, 0, has_bias, ctx
            )
        else:
            _dispatch[False, False, False](
                output, a, b, bias, m, n, k, 1, 0, 0, 0, has_bias, ctx
            )


def enqueue_tf32_bmm_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    if transpose_a:
        if transpose_b:
            _dispatch[True, True, True](
                output,
                a,
                b,
                a,
                m,
                n,
                k,
                batch_count,
                output_batch_stride,
                a_batch_stride,
                b_batch_stride,
                False,
                ctx,
            )
        else:
            _dispatch[True, True, False](
                output,
                a,
                b,
                a,
                m,
                n,
                k,
                batch_count,
                output_batch_stride,
                a_batch_stride,
                b_batch_stride,
                False,
                ctx,
            )
    else:
        if transpose_b:
            _dispatch[True, False, True](
                output,
                a,
                b,
                a,
                m,
                n,
                k,
                batch_count,
                output_batch_stride,
                a_batch_stride,
                b_batch_stride,
                False,
                ctx,
            )
        else:
            _dispatch[True, False, False](
                output,
                a,
                b,
                a,
                m,
                n,
                k,
                batch_count,
                output_batch_stride,
                a_batch_stride,
                b_batch_stride,
                False,
                ctx,
            )
