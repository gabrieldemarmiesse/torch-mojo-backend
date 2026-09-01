# ===----------------------------------------------------------------------=== #
# The fused-bias epilogue term, shared by every direct WGMMA route here.
#
# `addmm` and `linear` are a GEMM plus a row-broadcast bias.  Adding that bias
# with a separate elementwise kernel is a SECOND FULL PASS over C, and C is the
# largest tensor in the problem: on an H100 PCIe it measured 67us next to a
# 296us GEMM (S2/S3 4096-class) and 108us next to a 189us GEMM (S6, tall-skinny
# 32768x768) -- i.e. up to +56% device time for an operation that costs nothing
# when it happens where the accumulator already lives, in registers.
#
# So the bias is added to the WGMMA accumulator, in place, immediately before
# whatever the route's store path is.  That placement is deliberate: it is the
# ONE point every direct route has in common (the v3 warp-specialized kernels
# store pairs straight to global, the v4 persistent kernel stages through
# `st.matrix` into a swizzled tile and TMA-stores it), so a single helper serves
# both without either store path knowing a bias exists.  It also composes with
# the TMA-store epilogue by construction rather than by luck: the bias is gone
# from the picture before the first store instruction of either kind.
#
# Numerics: the bias is added to the FP32 accumulator, so the result is
# round(acc + bias) rather than the split path's round(round(acc) + bias).  At
# float32/TF32 that is bit-identical; at bfloat16/float16 it is one rounding
# step FEWER, which is also what cuBLAS's fused epilogue does.
# ===----------------------------------------------------------------------=== #

from std.memory import AddressSpace
from std.sys.defines import get_defined_string

from layout import Layout, LayoutTensor

from gemm16_dtype import _GEMM16_DT, _GEMM16_W


# Whether this specialization carries the fused-bias kernels.
#
# The loader already compiles one .so per (OP, dtype, flags) tuple and the
# `HAS_BIAS` flag is ALREADY part of that tuple -- `_try_gemm16_mm` and
# `_try_tf32_gemm` have always passed it -- so reading it here costs no extra
# build: it only stops the unused specialization from being instantiated into a
# build that can never call it.
#
# The same value ALSO arrives at runtime through the bridge ABI, and
# `enqueue_gemm16_gemm` raises when the two disagree.  That check is the whole
# safety argument for gating a CORRECTNESS property on a define: a define that
# silently reads back as "" would otherwise drop the bias and return a wrong
# answer, which is far worse than the usual define-gating failure of declining
# a call.  A standalone `mojo build` with no `-D HAS_BIAS=1` (the asm-compare
# script, a harness binary) therefore gets exactly the bias-free kernels, which
# is what makes those kernels' assembly comparable across trees at all.
comptime _GEMM16_HAS_BIAS = get_defined_string["HAS_BIAS", ""]() == "1"


@always_inline
def _gemm16_add_bias_to_accum[
    CFRAG: Int
](
    accum: LayoutTensor[
        DType.float32,
        Layout.row_major(1, CFRAG),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
    bias: UnsafePointer[Scalar[_GEMM16_DT], MutAnyOrigin],
    n0: Int,
    n: Int,
    lane: Int,
):
    """Add `bias[n0 + column(i)]` to every accumulator element `i`, in place.

    One warp group of a WGMMA `m64nNk*` instruction holds the N-half of a
    64-row output tile in `CFRAG = N // 2` registers per thread, laid out by
    the hardware as

        row(i)    = warp * 16 + lane // 4 + 8 * ((i % 4) // 2)
        column(i) = 8 * (i // 4) + 2 * (lane % 4) + (i % 2)

    (the same mapping the v3 store loop walks with `q = i // 2`).  The bias is
    a function of the COLUMN only, and the column depends on `i` solely through
    `i // 4` and `i % 2`, so the four elements `4c .. 4c + 3` need exactly one
    two-element bias load: elements `4c` and `4c + 2` share column `8c + 2L`,
    elements `4c + 1` and `4c + 3` share `8c + 2L + 1`.  That is `CFRAG // 4`
    loads per thread instead of `CFRAG`, and it is why the loop is nested.

    Out-of-range columns are CLAMPED, not branched: `min(col, n - 2)` reads a
    valid, in-bounds bias pair whose result is then discarded by the store
    path, which clips the partial edge tile anyway (v3 with its explicit
    `n0 + col + 1 < n` guard, v4 through the TMA store's global extent).  A
    branch per load would cost more than the wasted add.  Both callers gate on
    an even `n` (v3 tf32 requires `n % 2`, v4 requires `n % 8`), so the clamped
    index stays EVEN -- but that only makes the load naturally aligned relative
    to the BASE, so every route that reaches here must also have checked
    `_gemm16_bias_ptr_ok`.  A bias reached through an offset view (`w.bias[1:]`)
    has an odd element offset and would fault the pair load outright.
    """
    var base = n0 + (lane % 4) * 2
    comptime for c in range(CFRAG // 4):
        var pair = bias.load[width=2, alignment=2 * _GEMM16_W](
            min(base + c * 8, n - 2)
        )
        var b0 = pair[0].cast[DType.float32]()
        var b1 = pair[1].cast[DType.float32]()
        comptime for r in range(2):
            accum.ptr[c * 4 + r * 2] += b0
            accum.ptr[c * 4 + r * 2 + 1] += b1


@always_inline
def _gemm16_bias_ptr_ok(
    bias: UnsafePointer[Scalar[_GEMM16_DT], MutAnyOrigin]
) -> Bool:
    """Whether `_gemm16_add_bias_to_accum` may read this bias pointer.

    Its two-element load is naturally aligned, which a bias tensor satisfies
    unless it is an offset view onto a larger buffer -- legal in torch, and a
    hard CUDA_ERROR_MISALIGNED_ADDRESS here.  Every fused-bias route checks
    this and declines, so the fault is unreachable regardless of what the
    host-side routing in aten_fast.py believes; the host has the matching
    check only so such a bias keeps a FAST route instead of falling all the
    way to the accepted kernel.
    """
    return Int(bias) % (2 * _GEMM16_W) == 0


@always_inline
def _gemm16_bias_tag[has_bias: Bool]() -> StaticString:
    """Kernel-name suffix marking the fused-bias specialization.

    Kernel names are what CUPTI, Nsight and torch.profiler print, and the two
    specializations are genuinely different code, so a user profiling `addmm`
    has to be able to tell which one ran.
    """
    comptime if has_bias:
        return "_bias"
    else:
        return ""
