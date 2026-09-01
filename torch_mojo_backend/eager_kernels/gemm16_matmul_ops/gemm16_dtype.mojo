# ===----------------------------------------------------------------------=== #
# Which tensor-core operand dtype this build of the GEMM family carries.
#
# bfloat16 and float16 are the same thing to Hopper's tensor cores: identical
# operand width, identical WGMMA tile shapes, identical FP32 accumulator, and
# only the operand-type token of the instruction differs.  Every tile size,
# pipeline depth, TMA descriptor and wave-aware grid in this directory is a
# function of the operand WIDTH, so one source serves every dtype of the family
# and the choice is made once, here, at compile time.
#
# float32 joins them as a THIRD operand dtype, and it is the same thesis one
# step further out: WGMMA takes float32 operands directly and computes them as
# TF32 -- max/mojo/max/gpu/compute/mma.mojo's `_dtype_to_nvvm_wgmma_type` maps
# `DType.float32` to `#nvvm.wgmma_type<tf32>`, and `tensor_core_async.mojo`'s
# `_supported_mma_shape` accepts the m64/k8 shape that a 4-byte operand needs.
# So the tf32 kernel IS the 16-bit kernel with the width doubled, and the only
# constants that move are the byte-quantity ones below plus the WGMMA
# shared-memory descriptor strides in `_v4_mma_tile`.  Smem per pipeline stage
# (BM * BK * width) and WGMMA steps per stage (BK / wgmma_k) are both invariant
# when the width doubles and BK halves, which is why nothing structural moves.
#
# NOT every route in this directory is float32-verified, and the dispatcher says
# so: `enqueue_gemm16_gemm` serves only the NT split-K and NT 128x256 WGMMA
# routes at float32 and never instantiates the rest, so a widened-but-unmeasured
# kernel cannot reach a user.  Everything else float32 keeps the SM80-class
# `tf32_matmul_ops/` route, chosen on the host in `aten_fast.py`.
#
# The selector is a compile-time define, the same mechanism every other kernel
# family in this package uses: the loader already compiles one .so per
# (OP, dtype, flags) tuple and `DTYPE_ARG_0` already carries the operand dtype
# of the call, so nothing new travels at runtime.
#
# bfloat16 is the default for EVERY other value of the define, including the
# absent one, so a build that never heard of this define -- `mojo build` with no
# `-D` -- emits the bfloat16 kernels.  Note for scripts/compare_kernel_asm.py:
# its float32 pass USED to land on that default and re-emit the bfloat16
# kernels, which is no longer true now that float32 selects its own kernels.
# The bfloat16/float16 byte-invariance check is therefore
# `--dtypes bfloat16 float16`, and the float32 pass is where the new tf32
# kernels appear as additions.
# ===----------------------------------------------------------------------=== #

from std.sys import size_of

from variant_gates import _dtype_arg_on


@always_inline
def _gemm16_dtype() -> DType:
    """The tensor-core operand dtype this specialization was compiled for."""
    comptime if _dtype_arg_on[0, DType.float16]():
        return DType.float16
    elif _dtype_arg_on[0, DType.float32]():
        return DType.float32
    else:
        return DType.bfloat16


@always_inline
def _gemm16_tag() -> StaticString:
    """The dtype token every kernel of this family carries in its name.

    Kernel names are what CUPTI, Nsight and torch.profiler print, so a user
    profiling a float16 model has to read "f16" there and never "bf16", and a
    user profiling a float32 model under a TF32 matmul precision has to read
    "tf32" -- the operand dtype is float32 but the instruction is TF32, and
    "tf32" is the name of the arithmetic they are paying for.
    """
    comptime if _gemm16_dtype() == DType.float16:
        return "f16"
    elif _gemm16_dtype() == DType.float32:
        return "tf32"
    else:
        return "bf16"


comptime _GEMM16_DT = _gemm16_dtype()
comptime _GEMM16_TAG = _gemm16_tag()

# The operand width in bytes, and the two tile quantities derived from it.
# Both are BYTE quantities in the hardware, which is the whole reason one
# source serves 2-byte and 4-byte operands:
#
#   * one SWIZZLE_128B shared-memory row is 128 BYTES, so a k-major tile is
#     `128 // width` elements deep: BK = 64 at bfloat16/float16, 32 at float32.
#   * WGMMA_K_BYTES (layout/tensor_core_async.mojo) is 32, so one WGMMA step
#     covers `32 // width` k-elements: 16 at bfloat16/float16, 8 at float32.
#
# Their ratio -- WGMMA steps per BK tile -- is 4 either way, and smem per stage
# (BM * BK * width) is identical either way.
comptime _GEMM16_W = size_of[Scalar[_GEMM16_DT]]()
comptime _GEMM16_BK = 128 // _GEMM16_W
comptime _GEMM16_WGMMA_K = 32 // _GEMM16_W
# float32 operands, i.e. TF32 tensor-core arithmetic. Selects the reduced
# route ladder in `enqueue_gemm16_gemm` and the partial-k-tile arithmetic that
# the 16-bit routes' stricter admission gates make unnecessary for them.
comptime _GEMM16_TF32 = _GEMM16_DT == DType.float32
