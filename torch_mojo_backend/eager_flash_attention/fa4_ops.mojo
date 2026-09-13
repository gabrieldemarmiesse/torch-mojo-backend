# ===----------------------------------------------------------------------=== #
# C entry of the vendored dense FA4 kernels (family `fa4_ops`).
#
# Built on demand by the native backend's loader, one `mojo build
# --emit shared-lib` per specialization: `OP` picks the route and the head
# dimension, `DTYPE_ARG_0` the Q/K/V dtype (bfloat16 or float16 -- the same
# width on Hopper's tensor cores, same WGMMA tile shapes, same f32
# accumulator, so one kernel body serves both; f16 only needs its own RS
# wgmma emitter, selected inside the kernel, see `fa4_wgmma_f16.mojo`).
# The package lives beside eager_kernels rather than inside it so a
# FlashAttention change does not rehash every ordinary family; the loader
# finds it through `Loader.family_dir`.
#
# All launches use the caller's MAX DeviceContext (the tensor's current
# stream). They are asynchronous; synchronization belongs at explicit
# consumer/benchmark boundaries, never between forward or backward
# component kernels.
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv

from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.host.device_context import _DeviceContextCpp, _DeviceContextPtr

from fa4_bwd_launch import (
    launch_bwd_convert,
    launch_bwd_main,
    launch_bwd_preprocess,
)
from fa4_fwd_launch import launch_fwd_fa4
from fa4_fwd_selfload_common import kFa4BlockM as kFa4SelfloadBlockM
from fa4_fwd_selfload_common import kFa4CtasPerSm as kFa4SelfloadCtasPerSm
from fa4_fwd_selfload_launch import launch_fwd_fa4_selfload

from op_utils import (
    Arg,
    Argv,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher8,
    _spec_dispatcher9,
    _spec_dispatcher15,
    _spec_dispatcher16,
)
from variant_gates import (
    NO_OP_COMPILED,
    ErrBuf,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

# Q/K/V dtypes with an instantiated FA4 kernel.
comptime FA4_DTYPES = [DType.bfloat16, DType.float16]

# PHASE 2c wave-gate threshold (NOTES.md, /scratch/fa4-fwd-harness-2c,
# "Phase 2c" section 5): the self-loading (3 CTAs/SM) bhsd route beats
# the phase-2b (2 CTAs/SM) geometry once there is enough parallel work
# to fill a third CTA everywhere; below it, the shallower 4-stage ring
# and the missing dedicated producer warpgroup are pure cost with
# nothing to hide behind. Measured waves at this divisor: P0/P1/P2/P3 at
# 4.5/17/3.0/23 all clear the bar; P4/P5 at 0.8/0.2 both regress (5-9%)
# but stay under 1.10x cuDNN either way, so the gate is not delicate.
# FITTED ON H100 PCIe (114 SMs) -- re-derive on other cards; nothing
# here is architecture-specific in principle, but the threshold was
# never measured off Hopper.
comptime _FA4_SELFLOAD_MIN_WAVES: Int = 2


def _fa4_bhsd_selfload_waves(
    batch: Int, seqlen: Int, nheads: Int, ctx_handle_addr: Int
) raises -> Int:
    """Runtime wave count for the self-loading (3 CTAs/SM) bhsd route.

    Deliberately its OWN computation, not shared with
    ``launch_fwd_fa4``'s internal L2-swizzle wave count (which keeps
    dividing by ITS OWN 2-CTAs/SM occupancy): conflating the two would
    move the phase-2b ">= 12 waves" swizzle-group threshold for shapes
    that stay on the phase-2b geometry, an unmeasured combination this
    gate must not create (NOTES.md "Phase 2c" handoff item 4).
    """
    var raw_ctx_ptr = Pointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var num_m = ceildiv(seqlen, kFa4SelfloadBlockM(64))
    return (num_m * nheads * batch) // (kFa4SelfloadCtasPerSm(64) * sm_count)


# ---------------------------------------------------------------------------
# Layout validation (runs BEFORE any descriptor is created or kernel
# enqueued, so a violation never partially launches). The host side in
# ops_attention.mojo gates on the same conditions; these are the defensive
# twins that keep a bypassing caller from slipping an unsupported view past
# descriptor creation.
# ---------------------------------------------------------------------------


def _check_dims(batch: Int, seqlen: Int, nheads: Int) raises:
    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        raise Error(
            "fa4: batch, seqlen and nheads must be positive, got (",
            batch,
            ", ",
            seqlen,
            ", ",
            nheads,
            ")",
        )


def _check_strided_qkv_layout(
    name: StaticString,
    addr: Int,
    b_stride: Int,
    s_stride: Int,
    h_stride: Int,
    d_stride: Int,
    seqlen: Int,
    nheads: Int,
    head_dim: Int,
) raises:
    """Reject any Q/K/V layout outside the strict zero-copy regime.

    Strides are in Q/K/V dtype ELEMENTS (bf16 and f16 are both 2-byte
    types, so the same 16-byte-multiple math applies to either).
    """
    if b_stride <= 0 or s_stride <= 0 or h_stride <= 0 or d_stride <= 0:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " strides must all be positive, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ", ",
            d_stride,
            ")",
        )
    if d_stride != 1:
        raise Error(
            "fa4 strided qkv: ", name, " d_stride must be 1, got ", d_stride
        )
    if h_stride != head_dim:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " h_stride must be ",
            head_dim,
            ", got ",
            h_stride,
        )
    if s_stride < nheads * head_dim:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " s_stride ",
            s_stride,
            " must be >= nheads * head_dim = ",
            nheads * head_dim,
        )
    if b_stride != seqlen * s_stride:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " b_stride ",
            b_stride,
            " must equal seqlen * s_stride = ",
            seqlen * s_stride,
        )
    if addr % 16 != 0:
        raise Error(
            "fa4 strided qkv: ", name, " base address must be 16-byte aligned"
        )
    # TMA global strides are byte strides and every non-innermost one
    # must be a 16-byte multiple (2 bytes per element).
    if (
        (b_stride * 2) % 16 != 0
        or (s_stride * 2) % 16 != 0
        or (h_stride * 2) % 16 != 0
    ):
        raise Error(
            "fa4 strided qkv: ",
            name,
            " non-innermost strides must be multiples of 16 bytes, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ") elements",
        )


def _check_strided_qkv_args(
    batch: Int,
    seqlen: Int,
    nheads: Int,
    head_dim: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    strides: Arg,
) raises:
    """Validate the whole (b, s, h, d) stride triple carried by one tuple
    slot: `[q_b, q_s, q_h, q_d, k_b, k_s, k_h, k_d, v_b, v_s, v_h, v_d]`."""
    _check_dims(batch, seqlen, nheads)
    if seqlen % 128 != 0:
        raise Error(
            "fa4 strided qkv: seqlen must be a multiple of 128, got ", seqlen
        )
    if _raw_tuple_len(strides) != 12:
        raise Error(
            "fa4 strided qkv: expected 12 stride values, got ",
            _raw_tuple_len(strides),
        )
    _check_strided_qkv_layout(
        "q",
        q_addr,
        _raw_tuple_int(strides, 0),
        _raw_tuple_int(strides, 1),
        _raw_tuple_int(strides, 2),
        _raw_tuple_int(strides, 3),
        seqlen,
        nheads,
        head_dim,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        _raw_tuple_int(strides, 4),
        _raw_tuple_int(strides, 5),
        _raw_tuple_int(strides, 6),
        _raw_tuple_int(strides, 7),
        seqlen,
        nheads,
        head_dim,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        _raw_tuple_int(strides, 8),
        _raw_tuple_int(strides, 9),
        _raw_tuple_int(strides, 10),
        _raw_tuple_int(strides, 11),
        seqlen,
        nheads,
        head_dim,
    )


def _check_bhsd_args(
    batch: Int,
    seqlen: Int,
    nheads: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    out_addr: Int,
) raises:
    """Defensive validation for the BHSD-native forward route.

    TMA descriptor creation over the plane-viewed (B*H, S, D) layout
    requires exactly public (B, H, S, D) contiguity and 16-byte-aligned
    base pointers; a sliced/offset view can be fully contiguous yet still
    violate the latter.
    """
    _check_dims(batch, seqlen, nheads)
    if (
        q_addr % 16 != 0
        or k_addr % 16 != 0
        or v_addr % 16 != 0
        or out_addr % 16 != 0
    ):
        raise Error(
            "fa4 bhsd: q/k/v/out base addresses must be 16-byte aligned"
        )


def _dims(dims: Arg) raises -> Tuple[Int, Int, Int]:
    if _raw_tuple_len(dims) != 3:
        raise Error(
            "fa4: expected a (batch, seqlen, nheads) tuple, got ",
            _raw_tuple_len(dims),
            " values",
        )
    return (
        _raw_tuple_int(dims, 0),
        _raw_tuple_int(dims, 1),
        _raw_tuple_int(dims, 2),
    )


# ---------------------------------------------------------------------------
# Forward routes. Slots:
#   dense / bhsd : q, k, v, out, lse, (batch, seqlen, nheads), scale, ctx
#   strided      : q, k, v, out, lse, strides[12], (batch, seqlen, nheads),
#                  scale, ctx
# out and lse are always contiguous (BTHD output for the dense and strided
# routes, BHSD output for the bhsd route; lse is (batch, nheads, seqlen)).
# ---------------------------------------------------------------------------


def _fa4_fwd_go[
    dtype: DType, head_dim: Int
](
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    out_o: Arg,
    lse_o: Arg,
    dims_o: Arg,
    scale_o: Arg,
    ctx_o: Arg,
) raises:
    """Dense causal forward: Q/K/V/O are contiguous (B, S, H, D)."""
    var batch: Int
    var seqlen: Int
    var nheads: Int
    batch, seqlen, nheads = _dims(dims_o)
    _check_dims(batch, seqlen, nheads)
    launch_fwd_fa4[dtype, head_dim, False, True, 1, False, False, False, 0](
        batch,
        seqlen,
        nheads,
        Float32(_raw_f64(scale_o)),
        _raw_int(q_o),
        _raw_int(k_o),
        _raw_int(v_o),
        _raw_int(out_o),
        _raw_int(lse_o),
        0,
        _raw_int(ctx_o),
    )


def _fa4_fwd_bhsd_go[
    dtype: DType, head_dim: Int
](
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    out_o: Arg,
    lse_o: Arg,
    dims_o: Arg,
    scale_o: Arg,
    ctx_o: Arg,
) raises:
    """BHSD-native causal forward: Q/K/V/O TMA descriptors address the
    PUBLIC contiguous (B, H, S, D) layout directly (viewed as (B*H, S, D)
    planes), so no BTHD materialization is needed at all. The tail block is
    zero-filled/clamped here, which is why this is the only route that
    accepts a seqlen that is not a multiple of 128.

    At d64, runtime-gated (phase 2c, `_fa4_bhsd_selfload_waves`) between the
    self-loading single-warpgroup 3-CTAs/SM kernel and the phase-2b
    producer/consumer 2-CTAs/SM one; see `_FA4_SELFLOAD_MIN_WAVES`.
    """
    var batch: Int
    var seqlen: Int
    var nheads: Int
    batch, seqlen, nheads = _dims(dims_o)
    var q = _raw_int(q_o)
    var k = _raw_int(k_o)
    var v = _raw_int(v_o)
    var dst = _raw_int(out_o)
    var ctx = _raw_int(ctx_o)
    var scale = Float32(_raw_f64(scale_o))
    _check_bhsd_args(batch, seqlen, nheads, q, k, v, dst)
    comptime if head_dim == 64:
        if (
            _fa4_bhsd_selfload_waves(batch, seqlen, nheads, ctx)
            >= _FA4_SELFLOAD_MIN_WAVES
        ):
            launch_fwd_fa4_selfload[dtype, head_dim, False, True, 1, 0](
                batch,
                seqlen,
                nheads,
                scale,
                q,
                k,
                v,
                dst,
                _raw_int(lse_o),
                0,
                ctx,
            )
            return
    launch_fwd_fa4[
        dtype,
        head_dim,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        bhsd_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        scale,
        q,
        k,
        v,
        dst,
        _raw_int(lse_o),
        0,
        ctx,
    )


def _fa4_fwd_strided_go[
    dtype: DType, head_dim: Int
](
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    out_o: Arg,
    lse_o: Arg,
    strides_o: Arg,
    dims_o: Arg,
    scale_o: Arg,
    ctx_o: Arg,
) raises:
    """Zero-copy forward: Q/K/V are strided (B, S, H, D) views described by
    per-tensor runtime element strides (b, s, h, d); out/lse keep the
    contiguous layouts of the dense route."""
    var batch: Int
    var seqlen: Int
    var nheads: Int
    batch, seqlen, nheads = _dims(dims_o)
    var q = _raw_int(q_o)
    var k = _raw_int(k_o)
    var v = _raw_int(v_o)
    _check_strided_qkv_args(batch, seqlen, nheads, head_dim, q, k, v, strides_o)
    launch_fwd_fa4[
        dtype,
        head_dim,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        Float32(_raw_f64(scale_o)),
        q,
        k,
        v,
        _raw_int(out_o),
        _raw_int(lse_o),
        0,
        _raw_int(ctx_o),
        q_b_stride=_raw_tuple_int(strides_o, 0),
        q_s_stride=_raw_tuple_int(strides_o, 1),
        q_h_stride=_raw_tuple_int(strides_o, 2),
        q_d_stride=_raw_tuple_int(strides_o, 3),
        k_s_stride=_raw_tuple_int(strides_o, 5),
        k_h_stride=_raw_tuple_int(strides_o, 6),
        k_d_stride=_raw_tuple_int(strides_o, 7),
        v_s_stride=_raw_tuple_int(strides_o, 9),
        v_h_stride=_raw_tuple_int(strides_o, 10),
        v_d_stride=_raw_tuple_int(strides_o, 11),
    )


# ---------------------------------------------------------------------------
# Backward routes (preprocess + main + convert, one enqueue each). Slots:
#   dense   : q, k, v, out, dout, lse, dq, dk, dv, dpsum, lse_log2,
#             dq_accum, (batch, seqlen, nheads), scale, ctx
#   strided : the same with a strides[12] tuple between dq_accum and dims.
# Out/dO, the dq/dk/dv outputs and all scratch are contiguous BTHD; only
# Q/K/V may be strided views.
# ---------------------------------------------------------------------------


def _fa4_bwd_go[
    dtype: DType, head_dim: Int
](
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    out_o: Arg,
    dout_o: Arg,
    lse_o: Arg,
    dq_o: Arg,
    dk_o: Arg,
    dv_o: Arg,
    dpsum_o: Arg,
    lse_log2_o: Arg,
    dq_accum_o: Arg,
    dims_o: Arg,
    scale_o: Arg,
    ctx_o: Arg,
) raises:
    var batch: Int
    var seqlen: Int
    var nheads: Int
    batch, seqlen, nheads = _dims(dims_o)
    _check_dims(batch, seqlen, nheads)
    var scale = Float32(_raw_f64(scale_o))
    var ctx = _raw_int(ctx_o)
    launch_bwd_preprocess[dtype, head_dim, False, True, 1, False](
        batch,
        seqlen,
        nheads,
        _raw_int(out_o),
        _raw_int(dout_o),
        _raw_int(lse_o),
        _raw_int(dpsum_o),
        _raw_int(lse_log2_o),
        _raw_int(dq_accum_o),
        0,
        0,
        0,
        ctx,
    )
    launch_bwd_main[dtype, head_dim, False, True, 1, False, False, 0](
        batch,
        seqlen,
        nheads,
        scale,
        _raw_int(q_o),
        _raw_int(k_o),
        _raw_int(v_o),
        _raw_int(dout_o),
        _raw_int(dk_o),
        _raw_int(dv_o),
        _raw_int(lse_log2_o),
        _raw_int(dpsum_o),
        _raw_int(dq_accum_o),
        0,
        ctx,
    )
    launch_bwd_convert[dtype, head_dim, False, True, 1, False](
        batch,
        seqlen,
        nheads,
        scale,
        _raw_int(dq_accum_o),
        _raw_int(dq_o),
        0,
        ctx,
    )


def _fa4_bwd_strided_go[
    dtype: DType, head_dim: Int
](
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    out_o: Arg,
    dout_o: Arg,
    lse_o: Arg,
    dq_o: Arg,
    dk_o: Arg,
    dv_o: Arg,
    dpsum_o: Arg,
    lse_log2_o: Arg,
    dq_accum_o: Arg,
    strides_o: Arg,
    dims_o: Arg,
    scale_o: Arg,
    ctx_o: Arg,
) raises:
    var batch: Int
    var seqlen: Int
    var nheads: Int
    batch, seqlen, nheads = _dims(dims_o)
    var q = _raw_int(q_o)
    var k = _raw_int(k_o)
    var v = _raw_int(v_o)
    # The whole layout contract is validated up front so preprocess never
    # launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads, head_dim, q, k, v, strides_o)
    var scale = Float32(_raw_f64(scale_o))
    var ctx = _raw_int(ctx_o)
    launch_bwd_preprocess[dtype, head_dim, False, True, 1, False](
        batch,
        seqlen,
        nheads,
        _raw_int(out_o),
        _raw_int(dout_o),
        _raw_int(lse_o),
        _raw_int(dpsum_o),
        _raw_int(lse_log2_o),
        _raw_int(dq_accum_o),
        0,
        0,
        0,
        ctx,
    )
    launch_bwd_main[
        dtype, head_dim, False, True, 1, False, False, 0, strided_qkv=True
    ](
        batch,
        seqlen,
        nheads,
        scale,
        q,
        k,
        v,
        _raw_int(dout_o),
        _raw_int(dk_o),
        _raw_int(dv_o),
        _raw_int(lse_log2_o),
        _raw_int(dpsum_o),
        _raw_int(dq_accum_o),
        0,
        ctx,
        q_s_stride=_raw_tuple_int(strides_o, 1),
        q_h_stride=_raw_tuple_int(strides_o, 2),
        q_d_stride=_raw_tuple_int(strides_o, 3),
        k_s_stride=_raw_tuple_int(strides_o, 5),
        k_h_stride=_raw_tuple_int(strides_o, 6),
        k_d_stride=_raw_tuple_int(strides_o, 7),
        v_s_stride=_raw_tuple_int(strides_o, 9),
        v_h_stride=_raw_tuple_int(strides_o, 10),
        v_d_stride=_raw_tuple_int(strides_o, 11),
    )
    launch_bwd_convert[dtype, head_dim, False, True, 1, False](
        batch,
        seqlen,
        nheads,
        scale,
        _raw_int(dq_accum_o),
        _raw_int(dq_o),
        0,
        ctx,
    )


# ---------------------------------------------------------------------------
# Dtype resolution: `DTYPE_ARG_0` names the Q/K/V dtype of this build, so
# exactly one of the two instantiations below is compiled in.
# ---------------------------------------------------------------------------


def _unsupported_dtype() raises:
    raise Error("fa4: DTYPE_ARG_0 must be bfloat16 or float16 for this build")


@always_inline
def _fwd_dense[head_dim: Int](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in FA4_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _spec_dispatcher8[_fa4_fwd_go[dt, head_dim], "fa4 fwd"](argv, argc)
            handled = True
    if not handled:
        _unsupported_dtype()


@always_inline
def _fwd_bhsd[head_dim: Int](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in FA4_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _spec_dispatcher8[_fa4_fwd_bhsd_go[dt, head_dim], "fa4 fwd bhsd"](
                argv, argc
            )
            handled = True
    if not handled:
        _unsupported_dtype()


@always_inline
def _fwd_strided[head_dim: Int](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in FA4_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _spec_dispatcher9[
                _fa4_fwd_strided_go[dt, head_dim], "fa4 fwd strided"
            ](argv, argc)
            handled = True
    if not handled:
        _unsupported_dtype()


@always_inline
def _bwd_dense[head_dim: Int](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in FA4_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _spec_dispatcher15[_fa4_bwd_go[dt, head_dim], "fa4 bwd"](argv, argc)
            handled = True
    if not handled:
        _unsupported_dtype()


@always_inline
def _bwd_strided[head_dim: Int](argv: Argv, argc: Int) raises:
    var handled = False
    comptime for dt in FA4_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _spec_dispatcher16[
                _fa4_bwd_strided_go[dt, head_dim], "fa4 bwd strided"
            ](argv, argc)
            handled = True
    if not handled:
        _unsupported_dtype()


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one route per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Fa4FwdD64"]():
            _fwd_dense[64](argv, argc)
            return 0
        comptime if _op_on["Fa4FwdD128"]():
            _fwd_dense[128](argv, argc)
            return 0
        comptime if _op_on["Fa4FwdBhsdD64"]():
            _fwd_bhsd[64](argv, argc)
            return 0
        comptime if _op_on["Fa4FwdBhsdD128"]():
            _fwd_bhsd[128](argv, argc)
            return 0
        comptime if _op_on["Fa4FwdStridedD64"]():
            _fwd_strided[64](argv, argc)
            return 0
        comptime if _op_on["Fa4FwdStridedD128"]():
            _fwd_strided[128](argv, argc)
            return 0
        comptime if _op_on["Fa4BwdD64"]():
            _bwd_dense[64](argv, argc)
            return 0
        comptime if _op_on["Fa4BwdD128"]():
            _bwd_dense[128](argv, argc)
            return 0
        comptime if _op_on["Fa4BwdStridedD64"]():
            _bwd_strided[64](argv, argc)
            return 0
        comptime if _op_on["Fa4BwdStridedD128"]():
            _bwd_strided[128](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
