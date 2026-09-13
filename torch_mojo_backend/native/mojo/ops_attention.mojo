"""aten ops: attention group (see docs/native_backend.md).

`F.scaled_dot_product_attention` is CompositeImplicitAutograd: ATen picks a
backend, calls the matching lower op, and autograd differentiates *that* op
through its own formula. So this group registers the lower ops and never the
composite -- a PrivateUse1 kernel on a CompositeImplicitAutograd op takes it
out of reach of the decomposition autograd would otherwise differentiate, and
the autograd fallback then silently produces no gradient at all.

| route | op | kernels |
|---|---|---|
| FA4 (bf16/f16, head_dim 64/128, causal, sm_90a) | `_scaled_dot_product_flash_attention` | fa4_ops |
| fused flash (gfx942) | same | flash_attention_ops |
| decode step (q_len == 1) | `_scaled_dot_product_efficient_attention` | nn_ops AttnDecodeSpec |
| math (bmm + fused causal softmax + bmm) | same | matmul_ops Bmm + nn_ops SoftmaxRows |

Everything else -- an explicit mask, dropout, GQA, a shape no fused route
takes -- is left to ATen's own math decomposition, which composes ordinary
aten ops this backend already implements and differentiates itself.
"""
from std.ffi import external_call
from std.math import sqrt
from std.utils import IndexList

from abi import (
    ST_FLOAT32,
    ST_INT64,
    ST_UINT64,
    Owned,
    T,
    Value,
    Values,
    dtype_code,
    new_tensor,
    own,
    release,
    ret_int,
    ret_owned,
    unsupported,
    v_bool,
    v_f64,
    v_is_none,
    v_opt_tensor,
    v_tensor,
    view_strided,
    TAG_BOOL,
    TAG_INT,
    TAG_NONE,
    TAG_TENSOR,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import cast_to, copy_strided_into, fill_value, release_if_new
from registry import Site, impl, op_address_of

# at::SDPBackend (ATen/SDPBackend.h): what `_fused_sdp_choice` returns.
comptime SDP_MATH = 0
comptime SDP_FLASH = 1
comptime SDP_EFFICIENT = 2

# The fused flash kernels block head_dim in registers, so it is a regime
# rather than a free dimension.
comptime FUSED_FA_MAX_HEAD_DIM = 256


# --- shapes, views, ownership -------------------------------------------------


def _shape(dims: List[Int]) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - len(dims)
    for i in range(len(dims)):
        s[pad + i] = dims[i]
    return s


def _stride_list(vals: List[Int]) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - len(vals)
    for i in range(len(vals)):
        s[pad + i] = vals[i]
    return s


def _alloc(device: Int, stype: Int32, dims: List[Int]) raises -> T:
    return new_tensor(_shape(dims), len(dims), stype, device)


def _view(base: T, dims: List[Int], strides: List[Int]) raises -> T:
    """A zero-copy view over `base`'s storage, at `base`'s storage offset."""
    return view_strided(
        base, _shape(dims), _stride_list(strides), len(dims), base.offset
    )


def _borrowed(t: T) -> Owned:
    """An input handle in the slot an owned one would take: never released,
    so a route can treat "the caller's tensor" and "a copy I made" alike."""
    var o = own(t.copy())
    o.live = False
    return o^


def _materialize(var t: Owned) raises -> Owned:
    """`t` itself when contiguous, else a fresh contiguous copy."""
    if t.t.contig:
        return t^
    var out = own(_alloc(t.t.device, t.t.stype, t.t.logical_shape()))
    copy_strided_into(out.t, t.t)
    return out^


def _contig(t: T) raises -> Owned:
    """The borrowed input when contiguous, else an owned contiguous copy."""
    return _materialize(_borrowed(t))


def _api(device: Int) raises -> String:
    return dev(device)[].api


def _arch(device: Int) raises -> String:
    """ "sm_90a", "gfx942", ... -- the string MAX's Python
    `Device.architecture_name` reports for the same device."""
    return ctx_for(device).arch_name()


def _scale_of(v: Value, head_dim: Int) raises -> Float64:
    """`float? scale`: defaults to 1/sqrt(head_dim) (sdp::calculate_scale)."""
    if v_is_none(v):
        return 1.0 / sqrt(Float64(head_dim))
    var s = v_f64(v)
    # NaN fails the first test, +/-inf the second (inf - inf is NaN).
    if s != s or s - s != 0.0:
        raise Error("scaled_dot_product_attention: scale must be finite")
    return s


def _same_device(q: T, k: T, v: T) -> Bool:
    return (
        q.on_mojo()
        and k.on_mojo()
        and v.on_mojo()
        and q.device == k.device
        and q.device == v.device
    )


def _is_float(t: T) -> Bool:
    return (
        t.dtype == DType.float32
        or t.dtype == DType.bfloat16
        or t.dtype == DType.float16
    )


def _needs_grad(q: T, k: T, v: T) -> Bool:
    """Whether autograd will record this call: grad mode (a C++ TLS bit the
    shim exposes as tmb_grad_enabled; it cannot be inferred through the
    dispatcher from inside a backend kernel) and an input that requires
    grad."""
    if external_call["tmb_grad_enabled", Int32]() == 0:
        return False
    return q.requires_grad() or k.requires_grad() or v.requires_grad()


# ===========================================================================
# FA4: the vendored bf16/f16 causal kernels (Hopper).
# ===========================================================================


@fieldwise_init
struct Fa4Plan(Copyable, ImplicitlyCopyable, Movable):
    """What the FA4 gate decided, reached with no device work done."""

    var ok: Bool
    var batch: Int
    var heads: Int
    var seqlen: Int
    var head_dim: Int
    var bhsd: Bool  # the public (B, H, S, D) layout feeds TMA directly


def _fa4_plan(
    q: T,
    k: T,
    v: T,
    has_mask: Bool,
    dropout_p: Float64,
    is_causal: Bool,
    enable_gqa: Bool,
    allow_any_seqlen: Bool,
) raises -> Fa4Plan:
    """Eligible public (B, H, S, D) inputs, or `ok=False`.

    `head_dim` is a compile-time regime, not a free runtime dimension: only
    64 (GPT-2-class) and 128 (Llama-class) have an instantiated kernel.
    `allow_any_seqlen` lifts `seqlen % 128 == 0` down to `seqlen > 0`, which
    only the BHSD-native route can honour -- it zero-fills and clamps a
    partial last tile, while the strided and dense routes predate that tail
    machinery. The backward never gets it: a partial last tile there would
    be a silently wrong gradient.
    """
    var no = Fa4Plan(False, 0, 0, 0, 0, False)
    if has_mask or enable_gqa or dropout_p != 0.0 or not is_causal:
        return no
    if not _same_device(q, k, v):
        return no
    if _api(q.device) != "cuda" or _arch(q.device) != "sm_90a":
        return no
    if q.dtype != DType.bfloat16 and q.dtype != DType.float16:
        return no
    if k.stype != q.stype or v.stype != q.stype:
        return no
    if q.rank != 4 or not q.same_shape(k) or not q.same_shape(v):
        return no
    var batch = q.dim(0)
    var heads = q.dim(1)
    var seqlen = q.dim(2)
    var head_dim = q.dim(3)
    if batch <= 0 or heads <= 0 or seqlen <= 0:
        return no
    if head_dim != 64 and head_dim != 128:
        return no
    # TMA descriptors over the (B*H, S, D) plane view need exactly this
    # contiguity plus a 16-byte-aligned base; a sliced view can be
    # contiguous and still break the latter.
    var bhsd = (
        q.contig
        and k.contig
        and v.contig
        and q.ptr % 16 == 0
        and k.ptr % 16 == 0
        and v.ptr % 16 == 0
    )
    if seqlen % 128 != 0 and not (allow_any_seqlen and bhsd):
        return no
    return Fa4Plan(True, batch, heads, seqlen, head_dim, bhsd)


def _bthd_view(t: T) raises -> T:
    """The public (B, H, S, D) tensor as FA4-native (B, S, H, D)
    (`transpose(1, 2)`: a pure stride swap, no copy)."""
    return _view(
        t,
        [t.dim(0), t.dim(2), t.dim(1), t.dim(3)],
        [t.stride(0), t.stride(2), t.stride(1), t.stride(3)],
    )


def _fa4_strided_ok(t: T, head_dim: Int) -> Bool:
    """Whether a physical BTHD view is safe for FA4's strided TMA ABI: the
    kernel-side `_check_strided_qkv_layout` restated on the host, so an
    unsupported view is materialized before a route is chosen. bf16 and f16
    are both 2-byte types, so one byte-alignment rule covers either."""
    if t.rank != 4:
        return False
    var seqlen = t.dim(1)
    var heads = t.dim(2)
    if seqlen <= 0 or heads <= 0 or seqlen % 128 != 0:
        return False
    if t.dim(3) != head_dim:
        return False
    var sb = t.stride(0)
    var ss = t.stride(1)
    var sh = t.stride(2)
    if sb <= 0 or ss <= 0 or sh <= 0 or t.stride(3) != 1:
        return False
    if sh != head_dim or ss < heads * head_dim or sb != seqlen * ss:
        return False
    if t.ptr % 16 != 0:
        return False
    # TMA global strides are byte strides; every non-innermost one must be a
    # 16-byte multiple (2 bytes per element).
    return (sb * 2) % 16 == 0 and (ss * 2) % 16 == 0 and (sh * 2) % 16 == 0


def _fa4_native(t: T, head_dim: Int) raises -> Owned:
    """`t`'s BTHD view, materialized unless the strided ABI accepts it."""
    var view = own(_bthd_view(t))
    if _fa4_strided_ok(view.t, head_dim):
        return view^
    return _materialize(view^)


def _push_strides(mut out: List[Int], t: T, n: Int):
    for i in range(n):
        out.append(t.stride(i))


def _fa4_strides(q: T, k: T, v: T) -> List[Int]:
    """`[q_b, q_s, q_h, q_d, k_..., v_...]`, the tuple slot the strided
    routes read."""
    var out = List[Int](capacity=12)
    _push_strides(out, q, 4)
    _push_strides(out, k, 4)
    _push_strides(out, v, 4)
    return out^


def _fa4_forward(
    q: T, k: T, v: T, plan: Fa4Plan, scale: Float64
) raises -> Tuple[T, T]:
    """(output, logsumexp) as owned handles. `output` is (B, H, S, D)
    shaped; the BHSD route stores it contiguously, the BTHD routes store it
    (B, S, H, D) -- what PyTorch's own flash attention returns, so a caller's
    `y.transpose(1, 2).contiguous()` stays a view."""
    var b = plan.batch
    var h = plan.heads
    var s = plan.seqlen
    var d = plan.head_dim
    var ctx = ctx_for(q.device)
    var cp = ctx_ptr(ctx)
    var lse = own(_alloc(q.device, ST_FLOAT32, [b, h, s]))

    if plan.bhsd:
        var out = own(_alloc(q.device, q.stype, [b, h, s, d]))
        var call = KernelCall("fa4_ops", "Fa4FwdBhsdD" + String(d))
        call.arg_dtype(0, q.dtype)
        call.out_dtype(q.dtype)
        call.int(q.ptr)
        call.int(k.ptr)
        call.int(v.ptr)
        call.int(out.t.ptr)
        call.int(lse.t.ptr)
        call.tuple([b, s, h])
        call.f64(scale)
        call.int(cp)
        call.run()
        _ = ctx
        return (out.take(), lse.take())

    var qn = _fa4_native(q, d)
    var kn = _fa4_native(k, d)
    var vn = _fa4_native(v, d)
    # `_fa4_native` materialized everything the strided ABI refuses, so a
    # non-contiguous survivor is exactly one that ABI accepts.
    var strided = not (qn.t.contig and kn.t.contig and vn.t.contig)
    var dense = own(_alloc(q.device, q.stype, [b, s, h, d]))
    var op = String("Fa4FwdStridedD") if strided else String("Fa4FwdD")
    var call = KernelCall("fa4_ops", op + String(d))
    call.arg_dtype(0, q.dtype)
    call.out_dtype(q.dtype)
    call.int(qn.t.ptr)
    call.int(kn.t.ptr)
    call.int(vn.t.ptr)
    call.int(dense.t.ptr)
    call.int(lse.t.ptr)
    if strided:
        call.tuple(_fa4_strides(qn.t, kn.t, vn.t))
    call.tuple([b, s, h])
    call.f64(scale)
    call.int(cp)
    call.run()
    # Mojo destroys a value right after its LAST use, and these owners' last
    # use is the pointer read above: without these, a materialized Q/K/V copy
    # is freed before the kernel is even enqueued.
    _ = qn
    _ = kn
    _ = vn
    _ = ctx
    var out = _view(dense.t, [b, h, s, d], [s * h * d, d, h * d, 1])
    return (out^, lse.take())


def _fa4_backward(
    q: T, k: T, v: T, o: T, lse: T, grad: T, plan: Fa4Plan, scale: Float64
) raises -> Tuple[T, T, T]:
    """(dq, dk, dv) as owned handles, (B, H, S, D) shaped and (B, S, H, D)
    stored. The Q/K/V natives are always re-derived from the public tensors:
    the BHSD forward route consumed the public layout untouched, which is
    the wrong layout for the BTHD-only backward ABI."""
    var b = plan.batch
    var h = plan.heads
    var s = plan.seqlen
    var d = plan.head_dim
    var ctx = ctx_for(q.device)
    var cp = ctx_ptr(ctx)

    var qn = _fa4_native(q, d)
    var kn = _fa4_native(k, d)
    var vn = _fa4_native(v, d)
    var strided = not (qn.t.contig and kn.t.contig and vn.t.contig)
    # The strided ABI broadens only Q/K/V: out and dO keep the contiguous
    # BTHD contract even for a descriptor-safe gapped view.
    var on = _materialize(own(_bthd_view(o)))
    var gn = _materialize(own(_bthd_view(grad)))
    # FA4 consumes LSE as a contiguous (B, H, S), which the zero-copy caller
    # already is; only a genuinely strided one is copied.
    var lsen = _contig(lse)

    var dq = own(_alloc(q.device, q.stype, [b, s, h, d]))
    var dk = own(_alloc(q.device, q.stype, [b, s, h, d]))
    var dv = own(_alloc(q.device, q.stype, [b, s, h, d]))
    var s_pad = ((s + 127) // 128) * 128
    var dpsum = own(_alloc(q.device, ST_FLOAT32, [b, h, s_pad]))
    var lse_log2 = own(_alloc(q.device, ST_FLOAT32, [b, h, s_pad]))
    var dq_accum = own(_alloc(q.device, ST_FLOAT32, [b * h * s_pad * d]))

    var op = String("Fa4BwdStridedD") if strided else String("Fa4BwdD")
    var call = KernelCall("fa4_ops", op + String(d))
    call.arg_dtype(0, q.dtype)
    call.out_dtype(q.dtype)
    call.int(qn.t.ptr)
    call.int(kn.t.ptr)
    call.int(vn.t.ptr)
    call.int(on.t.ptr)
    call.int(gn.t.ptr)
    call.int(lsen.t.ptr)
    call.int(dq.t.ptr)
    call.int(dk.t.ptr)
    call.int(dv.t.ptr)
    call.int(dpsum.t.ptr)
    call.int(lse_log2.t.ptr)
    call.int(dq_accum.t.ptr)
    if strided:
        call.tuple(_fa4_strides(qn.t, kn.t, vn.t))
    call.tuple([b, s, h])
    call.f64(scale)
    call.int(cp)
    call.run()
    # Last-use lifetime extension: every one of these owns memory the kernel
    # reads or writes, and nothing below mentions them again.
    _ = qn
    _ = kn
    _ = vn
    _ = on
    _ = gn
    _ = lsen
    _ = dpsum
    _ = lse_log2
    _ = dq_accum
    _ = ctx

    var gq = _view(dq.t, [b, h, s, d], [s * h * d, d, h * d, 1])
    var gk = _view(dk.t, [b, h, s, d], [s * h * d, d, h * d, 1])
    var gv = _view(dv.t, [b, h, s, d], [s * h * d, d, h * d, 1])
    return (gq^, gk^, gv^)


# ===========================================================================
# Fused flash attention (gfx942 / CDNA3), forward and backward.
# ===========================================================================


@fieldwise_init
struct FusedFaPlan(Copyable, ImplicitlyCopyable, Movable):
    var ok: Bool
    var batch: Int
    var heads: Int
    var seq_q: Int
    var seq_kv: Int
    var head_dim: Int


def _fused_fa_plan(
    q: T, k: T, v: T, has_mask: Bool, dropout_p: Float64, enable_gqa: Bool
) raises -> FusedFaPlan:
    """Eligible BHTD inputs for the fused gfx942 kernels, or `ok=False`.

    Q, K and V are read through their own (batch, head, seq) strides, so the
    transposed view of BTHD storage a transformer produces is taken as it
    stands. The one layout that cannot be expressed is a strided head_dim
    axis -- the axis every vectorized load and every LDS row fill runs along.
    """
    var no = FusedFaPlan(False, 0, 0, 0, 0, 0)
    if has_mask or enable_gqa or dropout_p != 0.0:
        return no
    if not _same_device(q, k, v):
        return no
    if _api(q.device) != "hip" or _arch(q.device) != "gfx942":
        return no
    if not _is_float(q) or k.stype != q.stype or v.stype != q.stype:
        return no
    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        return no
    if q.dim(0) != k.dim(0) or q.dim(1) != k.dim(1):
        return no
    if not k.same_shape(v) or q.dim(3) != k.dim(3):
        return no
    var batch = q.dim(0)
    var heads = q.dim(1)
    var seq_q = q.dim(2)
    var seq_kv = k.dim(2)
    var head_dim = q.dim(3)
    if batch <= 0 or heads <= 0 or seq_q <= 0 or seq_kv <= 0:
        return no
    if head_dim <= 0 or head_dim > FUSED_FA_MAX_HEAD_DIM:
        return no
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        return no
    return FusedFaPlan(True, batch, heads, seq_q, seq_kv, head_dim)


def _bthd_output(
    device: Int, stype: Int32, b: Int, h: Int, s: Int, d: Int
) raises -> T:
    """A `[b, h, s, d]` tensor STORED `[b, s, h, d]` -- the layout PyTorch's
    flash attention returns, so the universal
    `y.transpose(1, 2).contiguous().view(B, T, C)` re-assembly reduces to two
    views. One dense allocation with a view over it: the view spans every
    element exactly once, so it costs no extra memory."""
    var dense = own(_alloc(device, stype, [b, s, h, d]))
    var v = _view(dense.t, [b, h, s, d], [s * h * d, d, h * d, 1])
    _ = dense^  # the view holds the storage from here on
    return v^


def _fused_fa_forward(
    q: T, k: T, v: T, plan: FusedFaPlan, is_causal: Bool, scale: Float64
) raises -> Tuple[T, T]:
    var ctx = ctx_for(q.device)
    var cp = ctx_ptr(ctx)
    var out = own(
        _bthd_output(
            q.device, q.stype, plan.batch, plan.heads, plan.seq_q, plan.head_dim
        )
    )
    var lse = own(
        _alloc(q.device, ST_FLOAT32, [plan.batch, plan.heads, plan.seq_q])
    )
    var strides = List[Int](capacity=12)
    _push_strides(strides, q, 3)
    _push_strides(strides, k, 3)
    _push_strides(strides, v, 3)
    _push_strides(strides, out.t, 3)
    var call = KernelCall("flash_attention_ops", "FlashAttentionForward")
    call.arg_dtype(0, q.dtype)
    call.arg_dtype(1, k.dtype)
    call.arg_dtype(2, v.dtype)
    call.out_dtype_i(0, q.dtype)
    call.out_dtype_i(1, DType.float32)
    call.flag("CAUSAL", 1 if is_causal else 0)
    call.int(out.t.ptr)
    call.int(lse.t.ptr)
    call.int(q.ptr)
    call.int(k.ptr)
    call.int(v.ptr)
    call.tuple([plan.batch, plan.heads, plan.seq_q, plan.seq_kv, plan.head_dim])
    call.tuple(strides)
    call.f64(scale)
    call.int(1 if is_causal else 0)
    call.int(dtype_code(q.dtype))
    call.int(cp)
    call.run()
    _ = ctx
    return (out.take(), lse.take())


def _fused_fa_backward(
    grad: T,
    q: T,
    k: T,
    v: T,
    o: T,
    lse: T,
    plan: FusedFaPlan,
    is_causal: Bool,
    scale: Float64,
) raises -> Tuple[T, T, T]:
    """`q/k/v/out/lse` must be exactly what the fused forward read and wrote:
    the backward recomputes the scores from the same bytes. Only a
    `grad_output` whose head_dim axis is strided has to be materialized --
    that is the axis the vectorized loads run along."""
    if o.stride(3) != 1:
        unsupported("fused flash backward: the output head_dim axis is strided")
    var g = _borrowed(grad)
    if grad.stride(3) != 1:
        g = _contig(grad)
    var ctx = ctx_for(q.device)
    var cp = ctx_ptr(ctx)
    var dq = own(
        _bthd_output(
            q.device, q.stype, plan.batch, plan.heads, plan.seq_q, plan.head_dim
        )
    )
    var dk = own(
        _bthd_output(
            q.device,
            q.stype,
            plan.batch,
            plan.heads,
            plan.seq_kv,
            plan.head_dim,
        )
    )
    var dv = own(
        _bthd_output(
            q.device,
            q.stype,
            plan.batch,
            plan.heads,
            plan.seq_kv,
            plan.head_dim,
        )
    )
    var strides = List[Int](capacity=24)
    _push_strides(strides, g.t, 3)
    _push_strides(strides, q, 3)
    _push_strides(strides, k, 3)
    _push_strides(strides, v, 3)
    _push_strides(strides, o, 3)
    _push_strides(strides, dq.t, 3)
    _push_strides(strides, dk.t, 3)
    _push_strides(strides, dv.t, 3)
    var call = KernelCall("flash_attention_ops", "FlashAttentionBackward")
    call.arg_dtype(0, g.t.dtype)
    call.arg_dtype(1, q.dtype)
    call.arg_dtype(2, k.dtype)
    call.arg_dtype(3, v.dtype)
    call.arg_dtype(4, o.dtype)
    call.arg_dtype(5, lse.dtype)
    call.out_dtype_i(0, q.dtype)
    call.out_dtype_i(1, q.dtype)
    call.out_dtype_i(2, q.dtype)
    call.flag("CAUSAL", 1 if is_causal else 0)
    call.int(dq.t.ptr)
    call.int(dk.t.ptr)
    call.int(dv.t.ptr)
    call.int(g.t.ptr)
    call.int(q.ptr)
    call.int(k.ptr)
    call.int(v.ptr)
    call.int(o.ptr)
    call.int(lse.ptr)
    call.tuple([plan.batch, plan.heads, plan.seq_q, plan.seq_kv, plan.head_dim])
    call.tuple(strides)
    call.f64(scale)
    call.int(1 if is_causal else 0)
    call.int(dtype_code(q.dtype))
    call.int(cp)
    call.run()
    _ = g  # last use above is g.t.ptr; keep the materialized grad alive
    _ = ctx
    return (dq.take(), dk.take(), dv.take())


# ===========================================================================
# The decode kernel and the math decomposition (inference routes).
# ===========================================================================


@fieldwise_init
struct MathPlan(Copyable, ImplicitlyCopyable, Movable):
    var ok: Bool
    var batch: Int
    var heads: Int
    var q_len: Int
    var kv_len: Int
    var head_dim: Int
    var decode: Bool  # the fused single-kernel decode step


def _math_plan(
    q: T,
    k: T,
    v: T,
    has_mask: Bool,
    dropout_p: Float64,
    is_causal: Bool,
    enable_gqa: Bool,
) raises -> MathPlan:
    var no = MathPlan(False, 0, 0, 0, 0, 0, False)
    if has_mask or enable_gqa or dropout_p != 0.0:
        return no
    if not _same_device(q, k, v):
        return no
    if not _is_float(q) or k.stype != q.stype or v.stype != q.stype:
        return no
    if q.rank != 4 or not k.same_shape(v):
        return no
    if q.dim(0) != k.dim(0) or q.dim(1) != k.dim(1) or q.dim(3) != k.dim(3):
        return no
    var batch = q.dim(0)
    var heads = q.dim(1)
    var q_len = q.dim(2)
    var kv_len = k.dim(2)
    var head_dim = q.dim(3)
    if batch <= 0 or heads <= 0 or q_len <= 0 or kv_len <= 0 or head_dim <= 0:
        return no
    # Decode step: one fused kernel instead of bmm + softmax + bmm (single
    # launch, no scratch, coalesced K/V reads). Q is read through its own
    # (batch, head) strides, so the per-head transpose view of a fused qkv
    # projection is never materialized first.
    var decode = (
        _api(q.device) != "cpu"
        and q_len == 1
        and not is_causal
        and head_dim % 4 == 0
        and head_dim <= 256
        and kv_len <= 4096
        and q.stride(3) == 1
        and k.stride(3) == 1
        and v.stride(3) == 1
        and k.stride(2) == head_dim
        and v.stride(2) == head_dim
    )
    return MathPlan(True, batch, heads, q_len, kv_len, head_dim, decode)


def _bmm(
    dst: T, a: T, b: T, batch: Int, m: Int, n: Int, kk: Int, transpose_b: Bool
) raises:
    """`matmul_ops.Bmm` over dense row-major (batch, m, k) x (batch, k, n)
    operands (or (batch, n, k) when transposed)."""
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("matmul_ops", "Bmm")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    call.out_dtype(dst.dtype)
    call.flag("TRANSPOSE_B", 1 if transpose_b else 0)
    call.int(dst.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.tuple([batch, m, n, kk, 1 if transpose_b else 0])
    call.int(dtype_code(a.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _math_forward(
    q_in: T, k_in: T, v_in: T, plan: MathPlan, is_causal: Bool, scale: Float64
) raises -> T:
    """Decomposed SDPA, returning an owned (B, H, L, D) output.

    The score matrix is folded to (B*H, L, S) for the batched GEMMs, and the
    scale and the causal mask are fused into the row softmax rather than
    materialized -- no (L, S) mask tensor and no separate scaling pass.

    KNOWN BUG, in `matmul_ops` rather than here: on the MAX **CPU** device the
    batched GEMM leaves part of C unwritten for some geometries (measured at
    batch 8, m 1, n 130, k 64, float32), so the result depends on what the
    freshly allocated scratch happened to hold. The same shapes are exact on
    GPU, and the old Python eager path called the same kernel the same way.
    """
    var b = plan.batch
    var h = plan.heads
    var lq = plan.q_len
    var lk = plan.kv_len
    var d = plan.head_dim
    var ctx = ctx_for(q_in.device)
    var cp = ctx_ptr(ctx)

    if plan.decode:
        var out = own(_alloc(q_in.device, q_in.stype, [b, h, 1, d]))
        var call = KernelCall("nn_ops", "AttnDecodeSpec")
        call.arg_dtype(0, q_in.dtype)
        call.arg_dtype(1, k_in.dtype)
        call.arg_dtype(2, v_in.dtype)
        call.out_dtype(out.t.dtype)
        call.spec(q_in.spec(cp))
        call.spec(k_in.spec(cp))
        call.spec(v_in.spec(cp))
        call.f64(scale)
        call.spec(out.t.spec(cp))
        call.run()
        _ = ctx
        return out.take()

    var q = _contig(q_in)
    var k = _contig(k_in)
    var v = _contig(v_in)
    # The scores and the softmax run in float32 whatever the inputs are.
    # q @ k^T accumulates head_dim products, which half cannot hold: the
    # review's case (q, k of magnitude 100, head_dim 64) reaches 6.4e5
    # against half's 65504 ceiling and saturates to inf BEFORE the scale and
    # the softmax can bring it back. ATen's math path avoids that by scaling
    # q and k by sqrt(scale) first; computing the scores in float32 is the
    # same fix without an extra pass over q, and the scale stays fused into
    # the softmax. float32 inputs cast to themselves, so they pay nothing.
    var q32 = cast_to(q.t, ST_FLOAT32)
    var k32 = cast_to(k.t, ST_FLOAT32)
    var scores = own(_alloc(q_in.device, ST_FLOAT32, [b * h, lq, lk]))
    _bmm(scores.t, q32, k32, b * h, lq, lk, d, True)
    release_if_new(q32, q.t)
    release_if_new(k32, k.t)
    _ = q
    _ = k
    var probs32 = own(_alloc(q_in.device, ST_FLOAT32, [b * h, lq, lk]))
    var soft = KernelCall("nn_ops", "SoftmaxRows")
    soft.arg_dtype(0, scores.t.dtype)
    soft.out_dtype(probs32.t.dtype)
    soft.flag("CAUSAL", 1 if is_causal else 0)
    soft.int(probs32.t.ptr)
    soft.int(scores.t.ptr)
    soft.int(b * h * lq)
    soft.int(lk)
    soft.f64(scale)
    soft.int(1 if is_causal else 0)
    soft.int(lq)
    soft.int(dtype_code(DType.float32))
    soft.int(cp)
    soft.run()
    _ = scores  # read only as a pointer above: keep the scratch alive
    # The probabilities are in [0, 1]: rounding them back to the input dtype
    # for the second GEMM is exactly what ATen's math path computes.
    var probs = cast_to(probs32.t, q_in.stype)
    var out3 = own(_alloc(q_in.device, q_in.stype, [b * h, lq, d]))
    _bmm(out3.t, probs, v.t, b * h, lq, d, lk, False)
    release_if_new(probs, probs32.t)
    _ = probs32
    _ = v
    _ = ctx
    return _view(out3.t, [b, h, lq, d], [h * lq * d, lq * d, d, 1])


def _flash_forward(
    q: T, k: T, v: T, is_causal: Bool, dropout_p: Float64, scale: Float64
) raises -> Tuple[T, T]:
    """(output, logsumexp) from whichever fused flash kernel takes these
    inputs, or `unsupported`."""
    var fa4 = _fa4_plan(
        q, k, v, False, dropout_p, is_causal, False, allow_any_seqlen=True
    )
    if fa4.ok:
        return _fa4_forward(q, k, v, fa4, scale)
    var fused = _fused_fa_plan(q, k, v, False, dropout_p, False)
    if fused.ok:
        return _fused_fa_forward(q, k, v, fused, is_causal, scale)
    unsupported(
        "no fused flash-attention kernel for these inputs (dtype "
        + String(q.dtype)
        + ", head_dim "
        + String(q.dim(3))
        + " on "
        + _arch(q.device)
        + "): ATen's math decomposition covers them"
    )
    raise Error("unreachable")


def _efficient_forward(
    q: T, k: T, v: T, is_causal: Bool, dropout_p: Float64, scale: Float64
) raises -> T:
    """The fused inference cascade: FA4, the fused gfx942 kernels, the decode
    kernel, then the decomposition. An owned (B, H, L, D) output."""
    var fa4 = _fa4_plan(
        q, k, v, False, dropout_p, is_causal, False, allow_any_seqlen=True
    )
    var fused = _fused_fa_plan(q, k, v, False, dropout_p, False)
    if fa4.ok or fused.ok:
        var pair = _flash_forward(q, k, v, is_causal, dropout_p, scale)
        release(pair[1].h)  # the flash LSE is not part of this schema
        return pair[0].copy()
    var math = _math_plan(q, k, v, False, dropout_p, is_causal, False)
    if math.ok:
        return _math_forward(q, k, v, math, is_causal, scale)
    unsupported(
        "no fused attention route for these inputs: ATen's math decomposition"
        " covers them"
    )
    raise Error("unreachable")


# ===========================================================================
# Registered ops.
# ===========================================================================


# aten::_fused_sdp_choice(Tensor query, Tensor key, Tensor value,
#   Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *,
#   float? scale=None, bool enable_gqa=False) -> int
def op_fused_sdp_choice(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """Which lower op `scaled_dot_product_attention` should call.

    NOTE: ATen's composite reads `_fused_sdp_choice_stub` (a DispatchStub),
    not this op, and nothing registers a PrivateUse1 entry in that stub -- so
    today the composite always picks `math` on this device and this op is
    reached only by a direct call. One line in the C++ shim
    (`REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, ...)` forwarding
    here) is what would route `F.scaled_dot_product_attention` into the fused
    kernels below.

    The user's SDP backend switches are NOT honoured, because there is no
    device-agnostic way to read them from here. They live on the global
    `at::Context` (`userEnabledFlashSDP()` / `userEnabledMemEfficientSDP()` /
    `userEnabledMathSDP()`, what `torch.backends.cuda.enable_flash_sdp()`
    sets), and nothing in the `tmb_*` record ABI exposes them: no aten op
    reports them and no shim entry point reads them. Honouring them needs
    three one-line getters in `native/csrc/shim_runtime.cpp` next to
    `tmb_float32_matmul_precision`, which reads `at::globalContext()` the
    same way; until then a caller who disables flash still gets flash from a
    direct call to this op (the composite, which is what those switches are
    documented to steer, does not reach it at all).
    """
    var q = v_tensor(args[unsafe_offset=0])
    var k = v_tensor(args[unsafe_offset=1])
    var v = v_tensor(args[unsafe_offset=2])
    var has_mask = not v_is_none(args[unsafe_offset=3])
    var dropout_p = v_f64(args[unsafe_offset=4])
    var is_causal = v_bool(args[unsafe_offset=5])
    var enable_gqa = False
    if n_args > 7 and not v_is_none(args[unsafe_offset=7]):
        enable_gqa = v_bool(args[unsafe_offset=7])
    var needs_grad = _needs_grad(q, k, v)
    # The backward tile machinery is unproven on a partial last tile, so a
    # grad-requiring call only takes the flash route at a full seqlen.
    var fa4 = _fa4_plan(
        q,
        k,
        v,
        has_mask,
        dropout_p,
        is_causal,
        enable_gqa,
        allow_any_seqlen=not needs_grad,
    )
    var fused = _fused_fa_plan(q, k, v, has_mask, dropout_p, enable_gqa)
    if fa4.ok or fused.ok:
        ret_int(rets, 0, SDP_FLASH)
        return
    # `_scaled_dot_product_efficient_attention` has an ATen autograd formula
    # whose backward op has no kernel here, so it is offered for inference
    # only; training falls back to the differentiable math decomposition.
    if not needs_grad:
        var math = _math_plan(
            q, k, v, has_mask, dropout_p, is_causal, enable_gqa
        )
        if math.ok:
            ret_int(rets, 0, SDP_EFFICIENT)
            return
    ret_int(rets, 0, SDP_MATH)


# aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key,
#   Tensor value, float dropout_p=0.0, bool is_causal=False,
#   bool return_debug_mask=False, *, float? scale=None)
#   -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k,
#       SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused,
#       Tensor debug_attn_mask)
def op_flash_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var q = v_tensor(args[unsafe_offset=0])
    var k = v_tensor(args[unsafe_offset=1])
    var v = v_tensor(args[unsafe_offset=2])
    var dropout_p = v_f64(args[unsafe_offset=3])
    var is_causal = v_bool(args[unsafe_offset=4])
    if dropout_p != 0.0:
        unsupported("flash attention with dropout")
    if v_bool(args[unsafe_offset=5]):
        unsupported("flash attention with return_debug_mask=True")
    if q.rank != 4:
        unsupported("flash attention expects 4-D query/key/value")
    var scale = _scale_of(args[unsafe_offset=6], q.dim(3))
    var pair = _flash_forward(q, k, v, is_causal, dropout_p, scale)
    var out = own(pair[0].copy())
    var lse = own(pair[1].copy())

    # Dense CUDA returns undefined cumulative-sequence tensors, uint64 RNG
    # state/offset tensors and an empty debug mask when dropout and
    # debugging are off. Empty (rather than undefined) tensors are returned
    # here so nothing downstream reads an undefined TensorImpl; dropout is
    # off, so the RNG payload values are unobserved and only the CUDA-
    # compatible shapes and dtypes are part of this contract.
    var seq_q = own(_alloc(q.device, ST_INT64, [0]))
    var seq_k = own(_alloc(q.device, ST_INT64, [0]))
    var rng_state = own(_alloc(q.device, ST_UINT64, [2]))
    var unused = own(_alloc(q.device, ST_UINT64, List[Int]()))
    var debug = own(_alloc(q.device, q.stype, [0]))
    fill_value(rng_state.t, 0.0)
    fill_value(unused.t, 0.0)
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, lse)
    ret_owned(rets, 2, seq_q)
    ret_owned(rets, 3, seq_k)
    ret_int(rets, 4, q.dim(2))
    ret_int(rets, 5, k.dim(2))
    ret_owned(rets, 6, rng_state)
    ret_owned(rets, 7, unused)
    ret_owned(rets, 8, debug)


# aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out,
#   Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp,
#   Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k,
#   float dropout_p, bool is_causal, Tensor philox_seed,
#   Tensor philox_offset, *, float? scale=None)
#   -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
def op_flash_attention_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """cum_seq_q/cum_seq_k/philox_* are deliberately not read: only this
    backend's own forward can produce a flash result here, it always returns
    them empty (the nested-tensor ragged layout has no kernel), and dropout
    is refused, so the RNG payload is unobserved."""
    var grad = v_tensor(args[unsafe_offset=0])
    var q = v_tensor(args[unsafe_offset=1])
    var k = v_tensor(args[unsafe_offset=2])
    var v = v_tensor(args[unsafe_offset=3])
    var out = v_tensor(args[unsafe_offset=4])
    var lse = v_tensor(args[unsafe_offset=5])
    var dropout_p = v_f64(args[unsafe_offset=10])
    var is_causal = v_bool(args[unsafe_offset=11])
    if dropout_p != 0.0:
        unsupported("flash attention backward with dropout")
    if q.rank != 4:
        unsupported("flash attention backward expects 4-D query/key/value")
    var scale = _scale_of(args[unsafe_offset=14], q.dim(3))
    if grad.stype != q.stype or not grad.same_shape(q):
        unsupported("flash attention backward: grad_out does not match query")

    # No `allow_any_seqlen` here: this IS the backward tile machinery, and a
    # partial last tile would be a silently wrong gradient. A forward that
    # reached an odd seqlen through the BHSD route therefore fails loudly
    # here rather than producing one.
    var fa4 = _fa4_plan(
        q, k, v, False, dropout_p, is_causal, False, allow_any_seqlen=False
    )
    if fa4.ok:
        if out.stype != q.stype or not out.same_shape(q):
            unsupported("flash attention backward: out does not match query")
        if (
            lse.dtype != DType.float32
            or lse.rank != 3
            or lse.dim(0) != fa4.batch
            or lse.dim(1) != fa4.heads
            or lse.dim(2) != fa4.seqlen
        ):
            unsupported("flash attention backward: logsumexp has a bad shape")
        var g = _fa4_backward(q, k, v, out, lse, grad, fa4, scale)
        var dq = own(g[0].copy())
        var dk = own(g[1].copy())
        var dv = own(g[2].copy())
        ret_owned(rets, 0, dq)
        ret_owned(rets, 1, dk)
        ret_owned(rets, 2, dv)
        return
    var fused = _fused_fa_plan(q, k, v, False, dropout_p, False)
    if not fused.ok:
        unsupported("no fused flash-attention backward for these inputs")
    var g = _fused_fa_backward(grad, q, k, v, out, lse, fused, is_causal, scale)
    var dq = own(g[0].copy())
    var dk = own(g[1].copy())
    var dv = own(g[2].copy())
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)


# aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key,
#   Tensor value, Tensor? attn_bias, bool compute_log_sumexp,
#   float dropout_p=0.0, bool is_causal=False, *, float? scale=None)
#   -> (Tensor output, Tensor log_sumexp, Tensor philox_seed,
#       Tensor philox_offset)
def op_efficient_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """The fused inference forward: FA4, the fused gfx942 kernels, the decode
    kernel, or the bmm + fused-causal-softmax + bmm decomposition.

    `compute_log_sumexp=True` is declined: none of the routes below produces
    a log-sum-exp (only the flash kernels do, through
    `aten::_scaled_dot_product_flash_attention`), and the only consumer is
    `_scaled_dot_product_efficient_attention_backward`, which this backend
    has no kernel for -- so returning a zero tensor would hand a silently
    wrong saved value to a backward that cannot run anyway. With
    `compute_log_sumexp=False` the second result is the empty tensor ATen's
    own CUDA path returns. A grad-requiring call is refused here, in the
    forward, where the traceback still names the op.
    """
    var q = v_tensor(args[unsafe_offset=0])
    var k = v_tensor(args[unsafe_offset=1])
    var v = v_tensor(args[unsafe_offset=2])
    var bias = v_opt_tensor(args[unsafe_offset=3])
    var compute_lse = v_bool(args[unsafe_offset=4])
    var dropout_p = v_f64(args[unsafe_offset=5])
    var is_causal = v_bool(args[unsafe_offset=6])
    if _needs_grad(q, k, v):
        unsupported(
            "aten::_scaled_dot_product_efficient_attention would record an"
            " autograd node (aten::_scaled_dot_product_efficient_attention_"
            "backward) that the mojo device does not implement. The forward"
            " itself is supported: run it under torch.no_grad() /"
            " torch.inference_mode(), or let F.scaled_dot_product_attention"
            " pick the differentiable math decomposition."
        )
    if bias:
        unsupported("efficient attention with an attention bias")
    if dropout_p != 0.0:
        unsupported("efficient attention with dropout")
    if compute_lse:
        unsupported(
            "aten::_scaled_dot_product_efficient_attention with"
            " compute_log_sumexp=True: none of the fused routes on this"
            " device produces a log-sum-exp."
            " aten::_scaled_dot_product_flash_attention does, on the inputs"
            " its kernels take."
        )
    if q.rank != 4:
        unsupported("efficient attention expects 4-D query/key/value")
    var scale = _scale_of(args[unsafe_offset=7], q.dim(3))

    var out = own(_efficient_forward(q, k, v, is_causal, dropout_p, scale))
    var lse = own(_alloc(q.device, ST_FLOAT32, [q.dim(0), q.dim(1), 0]))
    var seed = own(_alloc(q.device, ST_INT64, List[Int]()))
    var offset = own(_alloc(q.device, ST_INT64, List[Int]()))
    fill_value(seed.t, 0.0)
    fill_value(offset.t, 0.0)
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, lse)
    ret_owned(rets, 2, seed)
    ret_owned(rets, 3, offset)


# aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key,
#   Tensor value, float dropout_p=0.0, bool is_causal=False, *,
#   Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
def op_flash_attention_for_cpu(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """The composite `scaled_dot_product_attention` calls this overload, not
    the CUDA one, on every device but CUDA/XPU once `_fused_sdp_choice` picked
    flash: re-marshal onto the flash forward above (its autograd formula is
    `_for_cpu_backward`, wrapped the same way below)."""
    if not v_is_none(args[unsafe_offset=5]):
        unsupported("flash attention with an explicit attn_mask")
    var fargs = InlineArray[Value, 7](fill=Value(TAG_NONE, 0, 0, 0))
    for i in range(5):
        fargs[i] = args[unsafe_offset=i].copy()
    fargs[5] = Value(TAG_BOOL, 0, 0, 0)  # return_debug_mask
    fargs[6] = args[unsafe_offset=6].copy()
    var frets = InlineArray[Value, 9](fill=Value(TAG_NONE, 0, 0, 0))
    op_flash_attention(
        Values(unsafe_from_address=Int(fargs.unsafe_ptr())),
        7,
        Values(unsafe_from_address=Int(frets.unsafe_ptr())),
        9,
    )
    rets[unsafe_offset=0] = frets[0].copy()
    rets[unsafe_offset=1] = frets[1].copy()
    for i in range(2, 9):  # the flash-only results nobody asked for
        if frets[i].tag == TAG_TENSOR:
            release(Int(frets[i].a))
    _ = fargs
    _ = frets


# aten::_scaled_dot_product_flash_attention_for_cpu_backward(Tensor grad_out,
#   Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp,
#   float dropout_p, bool is_causal, *, Tensor? attn_mask=None,
#   float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
def op_flash_attention_for_cpu_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if not v_is_none(args[unsafe_offset=8]):
        unsupported("flash attention backward with an explicit attn_mask")
    # the CUDA layout: cum_seq_q/k, max_q/k and the philox pair are never read
    var fargs = InlineArray[Value, 15](fill=Value(TAG_NONE, 0, 0, 0))
    for i in range(6):
        fargs[i] = args[unsafe_offset=i].copy()
    fargs[8] = Value(TAG_INT, 0, 0, 0)
    fargs[9] = Value(TAG_INT, 0, 0, 0)
    fargs[10] = args[unsafe_offset=6].copy()
    fargs[11] = args[unsafe_offset=7].copy()
    fargs[14] = args[unsafe_offset=9].copy()
    op_flash_attention_backward(
        Values(unsafe_from_address=Int(fargs.unsafe_ptr())), 15, rets, n_rets
    )
    _ = fargs


def register_attention(site: Site) raises:
    impl[op_efficient_attention, "_scaled_dot_product_efficient_attention"](
        site
    )
    impl[
        op_flash_attention_for_cpu,
        "_scaled_dot_product_flash_attention_for_cpu",
    ](site)
    impl[
        op_flash_attention_for_cpu_backward,
        "_scaled_dot_product_flash_attention_for_cpu_backward",
    ](site)
    impl[op_flash_attention, "_scaled_dot_product_flash_attention"](site)
    impl[
        op_flash_attention_backward,
        "_scaled_dot_product_flash_attention_backward",
    ](site)
    impl[op_fused_sdp_choice, "_fused_sdp_choice"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_attention]()
