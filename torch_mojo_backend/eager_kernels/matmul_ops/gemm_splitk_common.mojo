# ===----------------------------------------------------------------------=== #
# Split-K pieces shared by the GEMM dispatches in this directory.
#
# `matmul_ops.mojo` (the classic NN/NT tiers) and `tn_f32_gemm_kernels.mojo`
# (the NVIDIA fp32 TN route) both split deep-K GEMMs into per-chunk partial
# results in a workspace and sum them with `_ksplit_reduce_kernel`. The
# kernel and the blocks-in-flight target live here so the TN dispatch can
# import them without a cycle through the extension entry module.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_idx, thread_idx
from std.sys.info import has_apple_gpu_accelerator

# Aim for a few blocks per SM when choosing split-K factors (H100: 114 SMs).
# Blocks-in-flight target for the split-K heuristic. NVIDIA/AMD parts want
# a few hundred blocks to hide latency; Apple's 8-40 core GPUs saturate far
# earlier, and every extra split-K shard costs an m*n float32 workspace
# write plus a read-back in the reduce pass, so oversplitting turns small-M
# GEMMs bandwidth-bound on partials.
comptime TARGET_BLOCKS = 80 if has_apple_gpu_accelerator() else 342


@__name("pure_ksplit_reduce")
def _ksplit_reduce_kernel(
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
    var i = (block_idx.x * 256 + thread_idx.x) * 4
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
