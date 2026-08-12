# ===----------------------------------------------------------------------=== #
# Zero-copy strict-fp32 TN GEMM dispatch for NVIDIA sm_90 (H100).
#
#   C[m,n] = A_phys[k,m]^T @ B[k,n]     all float32, strict fp32 (FFMA only)
#
# The generic TensorSpec matmul path's NVIDIA fp32 kernels only support
# `transpose_b`; a transposed-A operand (every `dW = dY^T @ X` weight
# gradient) used to be materialized into a contiguous scratch (a full
# read+write pass over A) before the NN kernel ran on the copy. This route
# reads the (k, m) storage in place instead — TN is structurally *easier*
# than NN for the cp.async pipeline (the physical layout IS the smem slab
# layout), so every suite shape beats NN parity rather than just matching.
#
# One 128x128 core plus a 64x64 small-shape core (tn_f32_gemm_core.mojo):
#
#   - `_tn_core_kernel` 128x128: quadrant-warp-tiled 8x8 thread tile,
#     2 CTAs/SM (228-block waves), STAGES=2. Runs every m,n >= 96 shape
#     regardless of operand alignment (VEC_A/VEC_B are picked per-call
#     from the alignment gates below). A de-phased-pipeline sibling,
#     `_tn_split_kernel` (warp-group split-K-in-block, 8x16 thread tile),
#     used to run instead whenever both operands were 16B-alignable; it
#     is gone because its stage buffers cannot fit the ptxas static
#     shared-memory budget on this toolchain at BM=BN=128,BK=16 for any
#     STAGES it is allowed to run at (see the note in
#     tn_f32_gemm_core.mojo). `_tn_core_kernel` itself only fits that
#     budget at STAGES=2 (0x8000 bytes) rather than the STAGES=4 (0x10000
#     bytes) it ran at alongside the split kernel; both facts came from
#     directly reproducing the ptxas failure (`mojo build --emit
#     shared-lib` with MODULAR_NVPTX_COMPILER_PATH pointed at a real
#     ptxas, on an actual H100) rather than guesswork.
#   - `_tn_core_kernel` 64x64 with the deep-K one-wave split rule for
#     smaller shapes (STAGES=4, unaffected -- its stage buffers are a
#     quarter the 128x128 tile's and were never close to the cap).
#   - `_pick_ksplits_wave` minimizes 1/fill + reduce_penalty over ks with
#     the core's own wave size (228); reduce_penalty charges the
#     split-K workspace traffic (2*ks*m*n*4 B at ~1.5 TB/s) against the
#     ideal GEMM time. Constants are physics; the wave size was
#     measured on H100 PCIe @ 1500 MHz locked clocks (114 SMs).
#   - Split-K partials to a stream-ordered workspace + deterministic
#     reduce: the stock `_ksplit_reduce_kernel` for <= 32 splits, one-pass
#     `_tn_ksplit_reduce_wide_kernel` for deep splits.
#
# Alignment gates (cp.async needs 16B-aligned sources for 16B copies):
#   A tile rows are m*4 bytes apart -> VEC_A=4 requires m % 4 == 0 AND a
#   16B-aligned A base (offset views can misalign it); B likewise with n.
#   Otherwise the 4-byte cp.async variant (VEC=1) runs — fp32 elements are
#   always 4B-aligned. A 16B path for misaligned shapes is impossible with
#   cp.async alone: the 16B variant needs BOTH src and dst 16B-aligned, and
#   any shifted staging scheme misaligns exactly one of them. Measured
#   headroom: the aligned twin 356x788x4571 runs 0.1275 ms vs 357x789's
#   0.1447 ms.
#
# Every tuning constant below was measured on H100 PCIe @ 1500 MHz locked
# clocks (114 SMs); the wave constants are card-fitted in value, portable
# in form (multiples of the SM count). Other NVIDIA parts keep the copy
# path (`try_enqueue_tn_f32_gemm` declines off sm_90).
# ===----------------------------------------------------------------------=== #

from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x

from gemm_splitk_common import TARGET_BLOCKS, _ksplit_reduce_kernel
from op_utils import _enqueue_cached, _make_ptr
from tn_f32_gemm_core import _tn_core_kernel


# Split-K workspace cap. 32 MB never binds for the tiny-MN deep-K regime it
# guards; it also (deliberately) forbids splitting huge-MN grids like
# 4096x4096 where the workspace would be >= 2x the output.
comptime SPLITK_WS_CAP_BYTES = 32 * 1024 * 1024
comptime SPLITK_GENERIC_CAP = 32
# One full wave for the 128x128 quadrant core at its measured 2-blocks/SM
# residency (H100 PCIe @ 1500 MHz, 114 SMs; 122 regs/thread -> 2 CTAs).
# Card-fitted in value, portable in form (2 x SM count). Measured at
# STAGES=4; occupancy here is register-bound, not smem-bound (2 CTAs x
# 64 KiB of stage buffers already fit well inside a 227 KiB/SM Hopper
# budget), so this residency -- and the wave size derived from it -- still
# holds at STAGES=2, which is what actually runs now (see
# tn_f32_gemm_core.mojo for why).
comptime WAVE_T128 = 228
# Minimum k-chunk for a 128x128 split: amortizes the pipeline prologue and
# the per-block epilogue over >= 16 slabs.
comptime KCHUNK_MIN_T128 = 256
# Deep-K constants for the 64x64 branch (measured on H100 PCIe @ 1500 MHz:
# for a tiny MN grid, 228 blocks = one full wave at the t64 kernel's
# 2-blocks/SM residency beat 128 blocks by 42% and 344 blocks by 13% on
# m=n=128; partial waves cost more than shorter k-chunks save).
comptime DEEPK_WAVE_BLOCKS = 228
comptime DEEPK_MAX_BASE = 7


# One-pass wide deep-split reduce: replaces a fold + finisher PAIR with a
# single launch. Block = 256 threads covering 128 elements: 8 j-groups
# (tid//32) each sum slabs j, j+8, j+16, ... of their 32-lane element
# window (coalesced 512 B per warp phase), partials meet in smem, group 0
# writes C. Grid = ceildiv(mn, 128) -> 128 blocks for a 128x128 C and no
# intermediate slab traffic: measured ~7 us faster than a fold+stock pair
# on 128x128x131072 ks=114. The stock reduce stays for <= 32 splits (few
# slabs per thread; not latency-bound there).
@__name("tn_ksplit_reduce_wide")
def _tn_ksplit_reduce_wide_kernel(
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mn_arg: Int64,
    ksplits_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var mn = Int(mn_arg)
    var ksplits = Int(ksplits_arg)

    var tid = Int(thread_idx.x)
    var j = tid // 32
    var lane = tid % 32
    var o = Int(block_idx.x) * 128 + lane * 4
    var smem = stack_allocation[
        1024, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var acc = SIMD[DType.float32, 4](0)
    if o + 4 <= mn:
        var st = j
        while st < ksplits:
            acc += ws_ptr.load[width=4](st * mn + o)
            st += 8
    elif o < mn:
        var st = j
        while st < ksplits:
            comptime for u in range(4):
                if o + u < mn:
                    acc[u] += ws_ptr[st * mn + o + u]
            st += 8
    smem.store[alignment=16](tid * 4, acc)
    barrier()
    if j == 0:
        var total = SIMD[DType.float32, 4](0)
        comptime for g in range(8):
            total += smem.load[width=4, alignment=16]((g * 32 + lane) * 4)
        if o + 4 <= mn:
            c_ptr.store(o, total)
        elif o < mn:
            comptime for u in range(4):
                if o + u < mn:
                    c_ptr[o + u] = total[u]


@always_inline
def _tn_core_launch[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    LR: Int,
    STAGES: Int,
    MINB: Int,
    PUMP: Int = 1,
](
    ctx: DeviceContext,
    gx: Int,
    gy: Int,
    gz: Int,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ksplits: Int,
    va4: Bool,
    vb4: Bool,
) raises:
    """Launch the quadrant core with VEC_A/VEC_B picked by the caller's
    alignment gates (16B cp.async needs both the vector run and the base
    address aligned).

    a_addr points at A_phys (k, m) row-major; the staging reads
    a[(kt+kk)*m + bm+cm] — coalesced along m, no transpose, no copy.
    """
    comptime THREADS = (BM // WM) * (BN // WN) * 32
    var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )

    if va4 and vb4:
        _enqueue_cached[
            _tn_core_kernel[BM, BN, BK, WM, WN, LR, 4, 4, STAGES, MINB, PUMP]
        ](
            ctx,
            String(t"tn_c_{BM}x{BN}x{BK}l{LR}_v44_s{STAGES}_mb{MINB}_p{PUMP}"),
            gx,
            gy,
            gz,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(ksplits),
        )
    elif va4:
        _enqueue_cached[
            _tn_core_kernel[BM, BN, BK, WM, WN, LR, 4, 1, STAGES, MINB, PUMP]
        ](
            ctx,
            String(t"tn_c_{BM}x{BN}x{BK}l{LR}_v41_s{STAGES}_mb{MINB}_p{PUMP}"),
            gx,
            gy,
            gz,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(ksplits),
        )
    elif vb4:
        _enqueue_cached[
            _tn_core_kernel[BM, BN, BK, WM, WN, LR, 1, 4, STAGES, MINB, PUMP]
        ](
            ctx,
            String(t"tn_c_{BM}x{BN}x{BK}l{LR}_v14_s{STAGES}_mb{MINB}_p{PUMP}"),
            gx,
            gy,
            gz,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(ksplits),
        )
    else:
        _enqueue_cached[
            _tn_core_kernel[BM, BN, BK, WM, WN, LR, 1, 1, STAGES, MINB, PUMP]
        ](
            ctx,
            String(t"tn_c_{BM}x{BN}x{BK}l{LR}_v11_s{STAGES}_mb{MINB}_p{PUMP}"),
            gx,
            gy,
            gz,
            THREADS,
            c,
            a,
            b,
            Int64(m),
            Int64(n),
            Int64(k),
            Int64(ksplits),
        )


def _pick_ksplits_wave(base: Int, m: Int, n: Int, k: Int, wave: Int) -> Int:
    """Fill-aware split-K factor for the 128x128 cores.

    Minimizes  1/fill + reduce_penalty  where
      fill = blocks / (ceil(blocks / wave) * wave)   (wave quantization at
             the 2-CTA `_tn_core_kernel`'s measured 228-block/wave
             residency), and
      reduce_penalty = (2 * ks * m * n * 4 B / 1.5e12 B/s) / ideal_gemm_s
             charges the workspace write+read against the ideal GEMM time.
    Candidates capped by kchunk >= KCHUNK_MIN_T128 and the workspace bytes.
    """
    var kcap = max(1, k // KCHUNK_MIN_T128)
    var ws_cap = max(1, SPLITK_WS_CAP_BYTES // (4 * m * n))
    var cap = min(kcap, ws_cap)
    if cap <= 1:
        return 1
    var ideal_gemm_s = 2.0 * Float64(m) * Float64(n) * Float64(k) / 43.8e12
    var best_ks = 1
    var best_score = Float64(1.0e30)
    for ks in range(1, cap + 1):
        var blocks = base * ks
        var rounds = ceildiv(blocks, wave)
        var fill = Float64(blocks) / Float64(rounds * wave)
        var reduce_s = 2.0 * Float64(ks) * Float64(m * n) * 4.0 / 1.5e12
        var score = 1.0 / fill + reduce_s / ideal_gemm_s
        if score < best_score - 1.0e-9:
            best_score = score
            best_ks = ks
    return best_ks


def _stabilize_ksplits(ks0: Int, k: Int, bk: Int) -> Int:
    """Shrink ks until ks == ceildiv(k, kchunk(ks)) so no split is empty.

    The kernel derives kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK from
    the ksplits it is launched with; if the BK rounding makes trailing
    splits empty they would never write their workspace slab and the reduce
    would sum garbage. Iterating ks -> ceildiv(k, kchunk(ks)) converges (ks
    strictly decreases) to a fixed point where every split is non-empty.
    """
    var ks = ks0
    while ks > 1:
        var kchunk = ceildiv(ceildiv(k, ks), bk) * bk
        var ks2 = ceildiv(k, kchunk)
        if ks2 == ks:
            break
        ks = ks2
    return ks


def try_enqueue_tn_f32_gemm(
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Zero-copy fp32 TN GEMM: C[m,n] = A_phys[k,m]^T @ B[k,n].

    Returns False for regimes the dedicated kernels do not want — the
    caller keeps the scratch-copy path:
      - anything but a CUDA device with compute capability 9.0 exactly
        (the tile/wave constants above are fitted to H100's 114 SMs);
      - m == 1 (the copy + GEMV path is bandwidth-optimal there);
      - a C base that is not 16B-aligned (the aligned epilogue stores
        assume it; outputs are fresh contiguous allocations, so this never
        triggers in practice).

    For m, n >= 96: the 128x128 quadrant core (8x8 thread tile, 2 CTAs/SM
    -> 228-block waves) with a fill-aware split-K picker, regardless of
    operand alignment (VEC_A/VEC_B are picked per-call from the alignment
    gates). 64x64 core with the deep-K split rules for smaller shapes.
    Correct for any m, n, k >= 1 (all loads/stores edge-guarded, cp.async
    zero-fills).
    """
    comptime if not _has_sm_9x():
        return False
    if ctx.api() != "cuda":
        return False
    var cc_major = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
    var cc_minor = ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
    if cc_major != 9 or cc_minor != 0:
        return False
    if m == 1:
        return False
    if c_addr % 16 != 0:
        return False

    # 16B staging eligibility: the vector run (m or n) must be a multiple
    # of 4 elements AND the operand base 16B-aligned (offset views can
    # break either).
    var va4 = m % 4 == 0 and a_addr % 16 == 0
    var vb4 = n % 4 == 0 and b_addr % 16 == 0

    var use_t128 = m >= 96 and n >= 96

    var gx: Int
    var gy: Int
    var ksplits: Int
    if use_t128:
        gx = ceildiv(n, 128)
        gy = ceildiv(m, 128)
        ksplits = _stabilize_ksplits(
            _pick_ksplits_wave(gx * gy, m, n, k, WAVE_T128), k, 16
        )
    else:
        gx = ceildiv(n, 64)
        gy = ceildiv(m, 64)
        var base = gx * gy
        var kcap = ceildiv(k, 192)
        ksplits = 1
        if base < TARGET_BLOCKS // 2:
            ksplits = min(
                min(ceildiv(TARGET_BLOCKS, base), kcap), SPLITK_GENERIC_CAP
            )
            # Deep-K, tiny-MN regime: fill exactly one wave.
            if base <= DEEPK_MAX_BASE:
                var ws_cap = max(1, SPLITK_WS_CAP_BYTES // (4 * m * n))
                var deep = min(min(DEEPK_WAVE_BLOCKS // base, kcap), ws_cap)
                if deep > ksplits:
                    ksplits = deep
        ksplits = _stabilize_ksplits(ksplits, k, 16)

    # Split-K partials go to a stream-ordered workspace kept alive until
    # its consuming launches are enqueued (same discipline as
    # `_scratch_copy`); a reduce kernel sums them into C afterwards (plain
    # stores, deterministic — no atomics).
    var ws = Optional[DeviceBuffer[DType.float32]](None)
    var c_target = c_addr
    if ksplits > 1:
        ws = ctx.enqueue_create_buffer[DType.float32](ksplits * m * n)
        c_target = Int(ws.value().unsafe_ptr())

    if use_t128:
        # STAGES=2, not the 4 an earlier revision used: at BM=BN=128,BK=16
        # the stage buffers are STAGES * (BM*BK + BK*BN) floats, 0x10000
        # bytes at STAGES=4 vs 0x8000 at STAGES=2, and ptxas on this
        # toolchain caps static shared memory at 0xc000 bytes on sm_90 --
        # STAGES=4 no longer builds (verified directly against a real
        # ptxas on an H100; see tn_f32_gemm_core.mojo).
        _tn_core_launch[128, 128, 16, 32, 64, 4, 2, 2](
            ctx,
            gx,
            gy,
            ksplits,
            c_target,
            a_addr,
            b_addr,
            m,
            n,
            k,
            ksplits,
            va4,
            vb4,
        )
    else:
        _tn_core_launch[64, 64, 16, 16, 32, 4, 4, 4](
            ctx,
            gx,
            gy,
            ksplits,
            c_target,
            a_addr,
            b_addr,
            m,
            n,
            k,
            ksplits,
            va4,
            vb4,
        )

    if ksplits > 1:
        var total = m * n
        var gxr = ceildiv(total, 1024)
        var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        var ws_ptr = (
            _make_ptr[DType.float32](c_target)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        # Deep splits: one-pass wide reduce (see _tn_ksplit_reduce_wide);
        # shallow splits keep the stock reduce (its 4-elems/thread grid is
        # fine when each thread only chews a handful of slabs).
        if ksplits > SPLITK_GENERIC_CAP:
            _enqueue_cached[_tn_ksplit_reduce_wide_kernel](
                ctx,
                String("tn_ksplit_reduce_wide"),
                ceildiv(total, 128),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                Int64(total),
                Int64(ksplits),
            )
        else:
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                gxr,
                1,
                1,
                256,
                c_out,
                ws_ptr,
                Int64(m * n),
                Int64(ksplits),
                Int64(total),
            )
    _ = ws^
    return True
