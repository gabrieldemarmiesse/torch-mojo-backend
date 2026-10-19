# ============================================================================
# PHASE 2c: self-loading single-warpgroup CTA, d64 dense-causal BHSD ONLY.
# A NEW module (not a replacement): fwd_launch.mojo (phase 2b) still
# owns every other route -- dense, strided-QKV (zero-copy BTHD), varlen,
# window, and d128 in general. Per NOTES.md "Phase 2c" handoff item 2,
# the self-loading structure was benchmarked and edge-case-tested ONLY
# on dense-causal BHSD at d64, so this launcher deliberately implements
# nothing else: varlen/window/strided_qkv/dense-r4 stay on
# ``launch_fwd_fa4`` (fwd_launch.mojo), unmodified and untouched by
# this change, rather than sharing a comptime-gated body with routes
# this structure was never measured or raced on.
# ============================================================================
"""Launch helper for the self-loading FA4 fwd kernel (dense causal BHSD,
d64 only).

The runtime choice between this launcher and the phase-2b
``launch_fwd_fa4`` bhsd path lives in ``fa4.mojo`` (the wave-count
gate: 3 CTAs/SM buys nothing below roughly 2 waves and costs 5-9% there,
NOTES.md "Phase 2c" section 5) -- this module only implements the
self-loading geometry itself, unconditionally, for whichever caller
already decided to use it.

PTX dump: `-D MOJO_DUMP_PTX=<path>` (launch_cache.dump_ptx_path).
"""

from max.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from max.gpu.host.device_context import (
    _DeviceContextPtr,
    _DeviceContextCpp,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of
from std.utils.index import IndexList

from layout import UNKNOWN_VALUE
from layout.tma_async import create_split_tma

from tmb.kernels.fa4.fwd_selfload_kernel import fwd_fa4_selfload_kernel
from tmb.kernels.fa4.fwd_selfload_common import (
    kFa4NThreads,
    kFa4BlockM,
    kFa4BlockN,
    kFa4KVStages,
    kFa4CtasPerSm,
)
from tmb.kernels.fa4.launch_cache import dump_ptx_path, enqueue_fa4_cached


def launch_fwd_fa4_selfload[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
    causal: Bool = False,
    gqa_ratio: Int = 1,
    softcap_x1000: Int = 0,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    o_addr: Int,
    lse_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    # Dense causal d64 BHSD-native only: the comptime assert in
    # fwd_fa4_selfload_kernel.mojo backstops head_dim == 64 at kernel
    # instantiation; this one gives a clearer message for the rest of
    # the (today, unreachable) envelope this launcher's signature could
    # otherwise be asked to serve.
    comptime assert (head_dim == 64) and causal and softcap_x1000 == 0, (
        "self-load geometry is d64-only, dense-causal-bhsd only"
        " (d128/non-causal keep the phase-2b launcher)"
    )
    var raw_ctx_ptr = Pointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem: Q (BM x D) + kFa4KVStages ring slots (BN x D) bf16/f16 +
    # mbarriers. d64: 8 KiB Q + 4*16 KiB ring = 72 KiB/CTA, so 3 CTAs/SM
    # fit the 227 KiB opt-in cap (see fwd_selfload_common.mojo).
    comptime q_bytes: Int = (kFa4BlockM(head_dim) * head_dim * size_of[dtype]())
    comptime kv_slot_bytes: Int = kFa4BlockN * head_dim * size_of[dtype]()
    comptime mbar_bytes: Int = 128
    comptime smem_bytes: Int = (
        q_bytes + kFa4KVStages * kv_slot_bytes + mbar_bytes
    )

    var q_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var lse_ptr = Pointer[Float32, MutAnyOrigin](unsafe_from_address=lse_addr)
    # PUBLIC-layout consumption (the fix the phase-2 bhsd descriptors
    # bought): Q/K/V/O are the contiguous (B, H, S, D) tensors the ATen
    # op provides, viewed as (B*H, S, D). Planes are the TMA outer dim
    # and S its OWN dim, so BM tail tiles zero-fill on load and clamp on
    # store at every plane's S edge -- no BTHD materialization.
    comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim)
    comptime q_smem_shape = IndexList[3](1, kFa4BlockM(head_dim), head_dim)
    comptime kv_smem_shape = IndexList[3](1, kFa4BlockN, head_dim)

    var planes_q: Int = batch_int * nheads_int
    var planes_kv: Int = batch_int * (nheads_int // gqa_ratio)
    var q_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, q_ptr, planes_q, seqlen_int)
    var k_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, k_ptr, planes_kv, seqlen_int)
    var v_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, v_ptr, planes_kv, seqlen_int)
    var o_imm_ptr = Pointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var o_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, o_imm_ptr, planes_q, seqlen_int)
    comptime kernel_inst = fwd_fa4_selfload_kernel[
        dtype,
        head_dim,
        3,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
        type_of(o_tma).tile_shape,
        type_of(o_tma).desc_shape,
        causal,
        gqa_ratio,
        False,
        False,
        False,
        softcap_x1000,
        bhsd=True,
    ]
    # Same LPT scheduler as the phase-2b bhsd path. This launcher's own
    # wave count (kFa4CtasPerSm(head_dim) == 3 here) picks its L2-swizzle
    # group -- NOT the same computation ``launch_fwd_fa4`` makes with its
    # own ctas_per_sm (2 at d64): that keeps the phase-2b >= 12-wave
    # threshold undisturbed for shapes still dispatched there (see the
    # gate in fa4.mojo and NOTES.md "Phase 2c" handoff item 4).
    var num_m: Int = ceildiv(seqlen_int, Int(kFa4BlockM(head_dim)))
    var size_one_kv_head: Int = seqlen_int * 2 * head_dim * size_of[dtype]()
    var sm_count: Int = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var ctas_per_sm: Int = kFa4CtasPerSm(head_dim)
    var waves: Int = (num_m * nheads_int * batch_int) // (
        ctas_per_sm * sm_count
    )
    var l2_ratio: Int = (50 * 1024 * 1024) // size_one_kv_head
    var sched_swizzle: Int
    if waves >= 12:
        sched_swizzle = 1
    else:
        sched_swizzle = 1
        while sched_swizzle * 2 <= l2_ratio:
            sched_swizzle *= 2
    var num_hb: Int = nheads_int * batch_int
    var sched_num_hb_q: Int = num_hb // sched_swizzle
    var sched_residual: Int = max(num_hb % sched_swizzle, 1)
    enqueue_fa4_cached[
        kernel_inst,
        use_external_stream=use_external_stream,
        dump_asm=dump_ptx_path(),
    ](
        ctx,
        ctx_handle_addr,
        stream_opaque,
        String(t"fwd_selfload_bhsd_{dtype}_d{head_dim}_c{causal}_g{gqa_ratio}"),
        (num_m * num_hb, 1, 1),
        kFa4NThreads(head_dim),
        smem_bytes,
        FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
        q_tma,
        k_tma,
        v_tma,
        o_tma,
        lse_ptr,
        Int64(seqlen_int),
        softmax_scale,
        Int64(nheads_int),
        Int64(sched_swizzle),
        Int64(sched_num_hb_q),
        Int64(sched_residual),
    )
