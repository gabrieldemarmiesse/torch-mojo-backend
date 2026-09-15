"""Runtime regime gates of the three measured bf16 candidates.

Reached from the top of `enqueue_gemm16_gemm` (the single-matrix GEMM
entry); every helper here returns False WITHOUT launching anything for a
shape it does not serve, and the whole pre-existing ladder is what runs
then. NN and fused-NT launch the kernels in gemm16_rolling_kernels.mojo and
gemm16_nt_bias_kernels.mojo. TN now tries the same shared body's rolling
geometry dispatcher first (gemm16_tn_v4_kernels.mojo::
_try_enqueue_tn_rolling_geom, the same helper try_enqueue_gemm16_gemm_tn_v4
uses further down its own ladder) before falling back to the fixed 128-row
routes below, which launch unchanged upstream device bodies.

Every crossover constant below was fitted on an H100 PCIe (114 SMs) at
1410 MHz. Matrix dimensions and the SM count stay runtime values: the gates
are shape REGIMES (residue-64 widths, bounded aspect ratios, deep K, grids
that fill more than one wave), never particular sizes.
"""
from max.gpu.host import DeviceContext, DeviceAttribute
from std.sys.info import _has_sm_9x
from gemm16_dtype import _GEMM16_DT
from gemm16_nn_v4_kernels import _v4_enqueue_nn_persistent
from gemm16_tn_v4_kernels import (
    _try_enqueue_tn_rolling_geom,
    _try_enqueue_tn_splitk_m128n256,
    _v4_enqueue_direct_m128n192,
)
from gemm16_rolling_kernels import enqueue_rolling_persistent
from gemm16_nt_bias_kernels import (
    _v4c_nt_bias_hw_gate,
    maybe_enqueue_gemm16_nt_bias_v4,
)

comptime PTR = Pointer[Scalar[_GEMM16_DT], MutAnyOrigin]


def try_enqueue_candidate_nt_bias(
    output: PTR,
    a: PTR,
    b: PTR,
    bias: PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    """Conservative large NT bias regime; original routes handle the rest.

    The 192x192 rolling kernel (fused bias epilogue, has_bias=True on
    gemm16_rolling_kernels.mojo's kmaj_b instantiation) runs first --
    worst-case ratio 1.052 against cuBLAS over six H100 SXM shapes
    (8192x4800x1600 down to the ragged 6600x4800x1600), independently
    reproduced across three jobs -- ahead of the 128-row
    maybe_enqueue_gemm16_nt_bias_v4 fallback for whatever it declines (an
    SM count too small for one cluster, or a grid past MAX_GRID_DIM_X;
    dtype/architecture/alignment/overflow are already covered by this
    outer regime plus the shared `_v4c_nt_bias_hw_gate`).

    TUNE_NT_ROLLING=True and TUNE_NT_RASTER=8 reproduce the 128-row
    fallback's H100 measurements; the 192 route takes its raster group (8)
    as an explicit kernel parameter instead, independently measured for
    this regime (see gemm16_rolling_kernels.mojo's module docstring).
    """
    # m % 8 (not the historical m % 128): every A/C TMA descriptor here
    # carries M as the operand's outer extent with K (already % 64 == 0) as
    # the inner one, so no descriptor stride depends on M -- both the
    # 192x192 rolling route and the 128-row fallback already clip a ragged
    # M edge exactly as they clip ragged N (see gemm16_rolling_kernels.mojo
    # and gemm16_nt_bias_kernels.mojo's "m and n need NOT be tile-aligned"
    # note). Measured: the ragged 6600x4800x1600 shape now reaches the
    # fused 192x192 kernel at 148 us vs 547 us on the unfused fallback.
    if (
        not has_bias
        or m < 4096
        or n < 1024
        or k < 1024
        or m % 8 != 0
        or n % 64 != 0
        or k % 64 != 0
    ):
        return False
    if n % 128 != 64 and k % 128 != 64:
        return False
    # Overflow-safe max(N,K)<=8*min(N,K); reject very wide vocabulary work.
    if 1 + (max(n, k) - 1) // 8 > min(n, k):
        return False
    if _try_enqueue_nt_bias_rolling_192(output, a, b, bias, m, n, k, ctx):
        return True
    return maybe_enqueue_gemm16_nt_bias_v4(output, a, b, bias, m, n, k, ctx)


def _try_enqueue_nt_bias_rolling_192(
    output: PTR,
    a: PTR,
    b: PTR,
    bias: PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """192x192 persistent rolling NT+bias kernel, H100 SXM sm:1500 MHz.

    Tuning provenance: BM=BN=192, BK=64, stages=3 (216 KiB of the 227 KiB
    sm_90 dynamic-smem limit), cluster_m=2 with cooperative B multicast,
    3 consumer warp groups (160 registers each), raster group=8 -- measured
    worst-case ratio over six shapes: group 2 -> 1.065, group 4 -> 1.043,
    group 6 -> 1.090, group 8 -> 1.037, group 12 -> 1.100, n-major -> 1.064,
    cluster_m=1 (no B multicast) -> 1.107.

    Shares `_v4c_nt_bias_hw_gate` (gemm16_nt_bias_kernels.mojo) with the
    128-row fallback for the dtype/cc/alignment/overflow checks both routes
    need; the grid-occupancy check below is this tile's own, since a
    192x192 cluster tiles the same (m, n) into a different work census than
    the 128-row kernel's 192/256-wide tiles.
    """
    # Two statements, not one `or`-combined guard: `comptime if A or B: return
    # False` does not stop Mojo from instantiating (and emitting into the
    # binary) the enqueue call below in a build where only A is true -- the
    # call sits lexically AFTER the guard rather than inside a comptime-if
    # branch, so nothing prunes it there. Nesting the rest inside `comptime
    # if _has_sm_9x():`, as maybe_enqueue_gemm16_nt_bias_v4 already does,
    # makes the call site itself conditional on the branch. Verified with
    # scripts/compare_kernel_asm.py --ops Gemm16NTBiasTry --dtypes float16:
    # the combined-guard version emitted the full 192x192 kernel in a
    # float16 build; this form emits zero kernels for it.
    comptime if _GEMM16_DT != DType.bfloat16:
        return False
    comptime if _has_sm_9x():
        if not _v4c_nt_bias_hw_gate(output, a, b, bias, m, n, k, ctx):
            return False
        comptime CLUSTER_M = 2
        comptime BM = 192
        comptime BN = 192
        var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if sms < CLUSTER_M:
            return False
        var macro_rows = (m + BM * CLUSTER_M - 1) // (BM * CLUSTER_M)
        var blocks_n = (n + BN - 1) // BN
        if (
            macro_rows
            > ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X) // blocks_n
        ):
            return False
        # A declined launch (the tile scheduler cannot serve the work
        # census) falls back to the 128-row route, as every other decline
        # above does.
        return enqueue_rolling_persistent[
            3, CLUSTER_M, BM, BN, 3, True, False, True, True, 8, True
        ](output, a, b, bias, m, n, k, sms, ctx)
    return False


def try_enqueue_candidate_nn(
    output: PTR,
    a: PTR,
    b: PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Tall, clipped-N bf16 regime; other products retain their old route.

    Compile the measured epilogue with PAIR_CAST=True. The tile and aspect
    crossovers are fitted on H100 PCIe, using runtime waves. No matrix
    dimension is specialized at compilation.
    """
    # Two statements, not one `or`-combined guard: see
    # _try_enqueue_nt_bias_rolling_192's comment (bc39b78) -- the combined
    # form does not stop Mojo from instantiating the enqueue call below in
    # a non-bfloat16 or non-sm_9x build. This one also failed to
    # cross-compile for gfx942 ("failed to run the pass manager for
    # offload") on both a719286 and this branch before this fix: the AMD
    # backend cannot lower whatever WGMMA/TMA/cluster intrinsics leaked
    # into a compiled unit that should have declined this whole function
    # at comptime.
    comptime if _GEMM16_DT != DType.bfloat16:
        return False
    comptime if _has_sm_9x():
        if ctx.api() != "cuda":
            return False
        if (
            ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) != 9
            or ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR) != 0
        ):
            return False
        if (
            m < 256
            or n < 512
            or k < 1024
            or m % 128 != 0
            or n % 256 != 64
            or k % 64 != 0
            or Int(output) % 16 != 0
            or Int(a) % 16 != 0
            or Int(b) % 16 != 0
            or m > 2_147_483_647
            or n > 2_147_483_647
            or k > 2_147_483_647
            or k > 9_223_372_036_854_775_807 // m
            or k > 9_223_372_036_854_775_807 // n
            or n > 9_223_372_036_854_775_807 // m
        ):
            return False
        if m < 4 * n or m > 32 * n or k < n or k > 8 * n:
            return False
        var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if sms < 2:
            return False
        var clusters = sms // 2
        var macro192 = (m + 383) // 384
        var work256 = ((m + 255) // 256) * ((n + 255) // 256)
        var wave256 = (work256 + clusters - 1) // clusters
        var width = 192
        var work192 = macro192 * ((n + width - 1) // width)
        if 2 * work192 > ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X):
            return False
        var wave192 = (work192 + clusters - 1) // clusters
        # Admit only when the new geometry reduces or preserves
        # rounded-wave arithmetic. Small/short and aligned regimes keep
        # their old kernels.
        if wave192 * 192 * width > wave256 * 128 * 256:
            return False
        # has_bias=False (default): the bias arg is unused, filled with
        # output.
        return enqueue_rolling_persistent[
            3, 2, 192, 192, 3, True, False, False, True
        ](output, a, b, output, m, n, k, sms, ctx)
    return False


def try_enqueue_candidate_tn(
    output: PTR,
    a: PTR,
    b: PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    # Two statements, not one `or`-combined guard: see
    # _try_enqueue_nt_bias_rolling_192's comment (bc39b78) -- the combined
    # form does not stop Mojo from instantiating the enqueue call below in
    # a non-bfloat16 or non-sm_9x build. This one also failed to
    # cross-compile for gfx942 ("failed to run the pass manager for
    # offload") on both a719286 and this branch before this fix: the AMD
    # backend cannot lower whatever WGMMA/TMA/cluster intrinsics leaked
    # into a compiled unit that should have declined this whole function
    # at comptime.
    comptime if _GEMM16_DT != DType.bfloat16:
        return False
    comptime if _has_sm_9x():
        if ctx.api() != "cuda":
            return False
        if (
            ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) != 9
            or ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR) != 0
        ):
            return False
        # Deep-K and wave crossovers are fitted on H100 PCIe. The gate
        # protects underfilled grids, short reductions and very tall
        # vocabulary products.
        if (
            m < 256
            or n < 256
            or k < 4096
            or m % 64 != 0
            or n % 64 != 0
            or k % 64 != 0
            or Int(output) % 16 != 0
            or Int(a) % 16 != 0
            or Int(b) % 16 != 0
            or m > 2_147_483_647
            or n > 2_147_483_647
            or k > 2_147_483_647
            or k > 9_223_372_036_854_775_807 // m
            or k > 9_223_372_036_854_775_807 // n
            or n > 9_223_372_036_854_775_807 // m
        ):
            return False
        var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        var max_grid = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
        if sms < 2:
            return False
        # Split-K first (see _try_enqueue_tn_splitk_m128n256's docstring):
        # a deep-K, underfilled-output shape parallelizes over K here in a
        # way the rolling dispatcher below cannot.  Then the rolling
        # geometry dispatcher: every shape this gate admits also clears
        # its own (looser) gate, and its runtime cost model over three
        # geometries beats what the fixed 128-row routes below produce on
        # every one of the standalone engagement's six measured shapes --
        # including three (c_attn, c_proj, mlp_proj: m % 128 == 64) that
        # this function's OWN "m % 128 == 64" branch below would otherwise
        # have claimed unconditionally, before ever reaching
        # try_enqueue_gemm16_gemm_tn_v4's ladder.  Falls through to the
        # existing routes below for whatever both decline (a GPU too small
        # for its cluster_m=2, a grid past MAX_GRID_DIM_X, or low modeled
        # occupancy -- see _try_enqueue_tn_rolling_geom's own docstring).
        if _try_enqueue_tn_splitk_m128n256(
            output, a, b, m, n, k, sms, max_grid, ctx
        ):
            return True
        if _try_enqueue_tn_rolling_geom(
            output, a, b, m, n, k, sms, max_grid, ctx
        ):
            return True
        var tiles192 = ((m + 127) // 128) * ((n + 191) // 192)
        if tiles192 > max_grid:
            return False
        # A grid that does not reach one full wave still beats the
        # ladder's fallback when it keeps at least three quarters of the
        # SMs busy: the 1600x1600 weight gradient is 117 tiles, more than
        # the 114 SMs of an H100 PCIe but fewer than the 132 of an H100
        # SXM.  (Measured on SXM: the fallback ran that shape at 9x
        # cuBLAS.)
        if tiles192 * 4 < sms * 3:
            return False
        var clusters192 = ((m + 255) // 256) * ((n + 191) // 192)
        var clusters256 = ((m + 255) // 256) * ((n + 255) // 256)
        var waves192 = (clusters192 + sms // 2 - 1) // (sms // 2)
        var waves256 = (clusters256 + sms // 2 - 1) // (sms // 2)
        var direct_waves192 = (tiles192 + sms - 1) // sms
        if m % 128 == 64:
            # A wide output exposes the wasted peer row of an M-tail
            # cluster; independent direct CTAs avoid that work. Ceil
            # launch is essential.
            if n >= 2 * m and direct_waves192 * 192 < waves256 * 256:
                _v4_enqueue_direct_m128n192(
                    output, a, b, m, n, k, tiles192, True, ctx
                )
                return True
            if waves192 * 192 < waves256 * 256:
                return _v4_enqueue_nn_persistent[
                    4, 2, 128, 192, 2, True, True, False, True
                ](output, a, b, m, n, k, sms, ctx)
            return _v4_enqueue_nn_persistent[
                3, 2, 128, 256, 2, True, True, False, True
            ](output, a, b, m, n, k, sms, ctx)
        # Preserve aligned routes except this clipped-column,
        # moderate-aspect regime where the four-stage 192 tile removes
        # excess per-wave work.
        if (
            n % 256 == 64
            and m >= 2 * n
            and m <= 8 * n
            and waves192 * 192 < waves256 * 256
        ):
            return _v4_enqueue_nn_persistent[
                4, 2, 128, 192, 2, True, True, False, True
            ](output, a, b, m, n, k, sms, ctx)
        return False
    return False
