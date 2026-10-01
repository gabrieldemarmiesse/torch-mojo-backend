"""ATen ops: attention group (see agents_docs/native_backend.md).

`F.scaled_dot_product_attention` is CompositeImplicitAutograd: ATen picks a
backend, calls the matching lower op, and autograd differentiates *that* op
through its own formula. So this group registers the lower ops and never the
composite -- a PrivateUse1 kernel on a CompositeImplicitAutograd op takes it
out of reach of the decomposition autograd would otherwise differentiate, and
the autograd fallback then silently produces no gradient at all.

| route | op | kernels |
|---|---|---|
| FA4 (bf16/f16, head_dim 64/128, causal, sm_90a) | `_scaled_dot_product_flash_attention` | fa4 |
| fused flash (gfx942) | same | flash_attention |
| decode step (q_len == 1) | `_scaled_dot_product_efficient_attention` | nn AttnDecodeSpec |
| math (bmm + fused causal softmax + bmm) | same | matmul Bmm + nn SoftmaxRows |

Grouped-query attention (`enable_gqa=True`, K/V with fewer heads than Q) is
served for inference by `_scaled_dot_product_efficient_attention`: K and V
are repeated up to Q's head count with one strided copy each (ATen's own
`repeat_interleave`), then the cascade above runs unchanged -- so a GQA call
still reaches FA4 or the fused gfx942 kernels. `enable_gqa=True` with equal
head counts is plain attention and takes every route above, training included.

Everything else -- an explicit mask, dropout, GQA under autograd, a shape no
fused route takes -- is left to ATen's own math decomposition, which composes
ordinary aten ops this backend already implements and differentiates itself.
"""
from std.ffi import external_call
from std.math import sqrt
from std.utils import IndexList
from std.utils.numerics import inf

from tmb.backend.abi import (
    ST_BOOL,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT64,
    ST_UINT64,
    Owned,
    Results,
    T,
    Value,
    Values,
    dtype_code,
    f64_bits,
    new_tensor,
    own,
    release,
    retain,
    ret_int,
    ret_owned,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_is_none,
    v_opt_tensor,
    v_tensor,
    view_strided,
    TAG_BOOL,
    TAG_DOUBLE,
    TAG_INT,
    TAG_INT_LIST,
    TAG_STRING,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_TENSOR,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    call_op,
    cast_to,
    copy_strided_into,
    fill_value,
    release_if_new,
    shape_str,
)
from tmb.ops.composed import _bool_list
from tmb.backend.registry import Site, impl
from tmb.ops.data_movement import _batched_copy_run, _scalar_type_name
from tmb.ops.foreach import _batch_copy_dtype, _batched_copy_device

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
    _ = t^  # alive past the launch
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


def _gqa_heads_divide(q: T, k: T, v: T) -> Bool:
    """Whether K and V each carry a head count that divides Q's, over
    otherwise matching 4-D (B, H, S, D) shapes -- the grouped-query layout
    `enable_gqa=True` allows (query head `i` reads K/V head `i // n_rep`)."""
    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        return False
    var hq = q.dim(1)
    var hk = k.dim(1)
    var hv = v.dim(1)
    if hk <= 0 or hv <= 0 or hq % hk != 0 or hq % hv != 0:
        return False
    return (
        q.dim(0) == k.dim(0)
        and q.dim(0) == v.dim(0)
        and k.dim(2) == v.dim(2)
        and q.dim(3) == k.dim(3)
    )


def _gqa_expand(t: T, q_heads: Int) raises -> Owned:
    """K or V `(B, Hkv, S, D)` as a dense `(B, q_heads, S, D)`: each KV head
    repeated `q_heads // Hkv` times in place, exactly ATen's
    `repeat_interleave(n_rep, dim=1)`; the borrowed input when nothing
    repeats.

    A KV head whose (S, D) plane is dense is one row of `S * D` elements,
    and the repeat is `n_rep` rectangles over those rows -- one batched
    rectangle copy (CopyBatched, 16-byte accesses) at copy bandwidth. Any
    other layout takes the generic strided copy from a stride-0
    `(B, Hkv, n_rep, S, D)` view, which is several times slower."""
    var kv_heads = t.dim(1)
    if kv_heads == q_heads:
        return _borrowed(t)
    var rep = q_heads // kv_heads
    var b = t.dim(0)
    var s = t.dim(2)
    var d = t.dim(3)
    var out = own(_alloc(t.device, t.stype, [b, q_heads, s, d]))
    var plane = s * d
    # One source pitch walks every (batch, kv head) row.
    var rows = b * kv_heads
    var pitch = -1
    if b == 1 or t.stride(0) == kv_heads * t.stride(1):
        pitch = t.stride(1)
    elif kv_heads == 1:
        pitch = t.stride(0)
    if (
        pitch > 0
        and (t.stride(3) == 1 or d == 1)
        and (t.stride(2) == d or s == 1)
        and pitch < 1 << 31
        and rep * plane < 1 << 31
        and _batch_copy_dtype(t.dtype)
        and _batched_copy_device(t.device)
    ):
        var srcs = List[Int](capacity=rep)
        var dsts = List[Int](capacity=rep)
        var cols = List[Int](capacity=rep)
        for r in range(rep):
            srcs.append(t.ptr)
            dsts.append(out.t.ptr + r * plane * t.itemsize)
            cols.append(plane)
        _batched_copy_run(
            t.device,
            t.dtype,
            t.dtype,
            t.itemsize,
            srcs,
            dsts,
            rows,
            cols,
            pitch,
            rep * plane,
        )
        return out^
    var dst = own(
        _view(
            out.t,
            [b, kv_heads, rep, s, d],
            [q_heads * s * d, rep * s * d, s * d, d, 1],
        )
    )
    var src = own(
        _view(
            t,
            [b, kv_heads, rep, s, d],
            [t.stride(0), t.stride(1), 0, t.stride(2), t.stride(3)],
        )
    )
    copy_strided_into(dst.t, src.t)
    _ = dst
    _ = src
    return out^


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
        var call = KernelCall("fa4", "Fa4FwdBhsdD" + String(d))
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
    var call = KernelCall("fa4", op + String(d))
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
    _ = dense^  # the view holds the storage from here on
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
    var call = KernelCall("fa4", op + String(d))
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
    # The views hold the storage from here on.
    _ = dq^
    _ = dk^
    _ = dv^
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
    var call = KernelCall("flash_attention", "FlashAttentionForward")
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
    var call = KernelCall("flash_attention", "FlashAttentionBackward")
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
    """`matmul.Bmm` over dense row-major (batch, m, k) x (batch, k, n)
    operands (or (batch, n, k) when transposed)."""
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("matmul", "Bmm")
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

    KNOWN BUG, in `matmul` rather than here: on the MAX **CPU** device the
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
        var call = KernelCall("nn", "AttnDecodeSpec")
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
    var soft = KernelCall("nn", "SoftmaxRows")
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
    var out = _view(out3.t, [b, h, lq, d], [h * lq * d, lq * d, d, 1])
    _ = out3^  # the view holds the storage from here on
    return out^


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
# The math route with a log-sum-exp: every fused-attention entry point whose
# inputs no fused kernel takes (an attention bias, a non-causal flash call,
# float32, an odd head_dim, a requested logsumexp, a backward), composed
# from ops this device implements through the dispatcher.
#
# Scores, softmax and both GEMMs run in float32 (float64 for float64
# inputs) and round once at the end, the precision the fused kernels keep.
# The backward recomputes the probabilities from q/k/bias rather than
# reading the forward's logsumexp, so it does not depend on which route,
# or which entry point's logsumexp layout, produced the forward.
# ===========================================================================

comptime _POS_INF = inf[DType.float64]()
comptime _NEG_INF = -inf[DType.float64]()

# How a fully masked query row reports its logsumexp (its output is 0 in
# every backend): what each CUDA backend returns, measured on an H100.
comptime LSE_MASKED_FLASH = 0  # +inf
comptime LSE_MASKED_EFFICIENT = 1  # 0
comptime LSE_MASKED_CUDNN = 2  # -inf

# Causal alignment, `causal_off` below: no mask, or key j visible to query i
# iff j <= i + offset (top-left: 0; bottom-right: S - L).
comptime NO_CAUSAL = -(1 << 62)


struct _Call(Movable):
    """One aten op through the dispatcher, arguments in schema order."""

    var op: String
    var overload: String
    var vals: List[Value]
    var lists: List[List[Int64]]  # int[] arguments, alive across the call

    def __init__(out self, op: StaticString, overload: StaticString):
        self.op = String(op)
        self.overload = String(overload)
        self.vals = List[Value]()
        self.lists = List[List[Int64]]()

    def t(mut self, x: T):
        self.vals.append(Value(TAG_TENSOR, 0, Int64(x.h), 0))

    def i(mut self, x: Int):
        self.vals.append(Value(TAG_INT, 0, Int64(x), 0))

    def b(mut self, x: Bool):
        self.vals.append(Value(TAG_BOOL, 0, Int64(1) if x else Int64(0), 0))

    def s(mut self, x: Float64):
        """A `Scalar` argument."""
        self.vals.append(Value(TAG_SCALAR_DOUBLE, 0, f64_bits(x), 0))

    def f(mut self, x: Float64):
        """A `float` argument."""
        self.vals.append(Value(TAG_DOUBLE, 0, f64_bits(x), 0))

    def str(mut self, x: StaticString):
        """A `str` argument (static storage, so it outlives the call)."""
        self.vals.append(
            Value(
                TAG_STRING,
                Int32(x.byte_length()),
                Int64(Int(x.unsafe_ptr())),
                0,
            )
        )

    def ot(mut self, x: Optional[T]):
        """A `Tensor?` argument."""
        if x:
            self.t(x.value())
        else:
            self.none()

    def none(mut self):
        self.vals.append(Value(TAG_NONE, 0, 0, 0))

    def ints(mut self, xs: List[Int]):
        var l = List[Int64](capacity=max(len(xs), 1))
        for x in xs:
            l.append(Int64(x))
        var addr = Int(l.unsafe_ptr())
        self.lists.append(l^)  # moves the List, not its heap buffer
        self.vals.append(Value(TAG_INT_LIST, Int32(len(xs)), Int64(addr), 0))

    def run(mut self, n_rets: Int) raises -> Results:
        return call_op(self.op, self.overload, self.vals.copy(), n_rets)

    def one(mut self) raises -> Owned:
        var r = self.run(1)
        return own(r.take_tensor(0))


def _d_unary(op: StaticString, x: T) raises -> Owned:
    var c = _Call(op, "")
    c.t(x)
    return c.one()


def _d_binary(op: StaticString, a: T, b: T) raises -> Owned:
    """`add`/`sub`/`mul`/`div` .Tensor (alpha 1 where the schema has one)."""
    var c = _Call(op, "Tensor")
    c.t(a)
    c.t(b)
    if op == "aten::add" or op == "aten::sub":
        c.s(1.0)
    return c.one()


def _d_inplace(op: StaticString, overload: StaticString, a: T, b: T) raises:
    """`add_`/`sub_` .Tensor with alpha 1, or `masked_fill_` style calls
    whose second operand is a tensor and third is absent."""
    var c = _Call(op, overload)
    c.t(a)
    c.t(b)
    c.s(1.0)
    _ = c.run(1)


def _d_mul_scalar_(a: T, s: Float64) raises:
    var c = _Call("aten::mul_", "Scalar")
    c.t(a)
    c.s(s)
    _ = c.run(1)


def _d_matmul(a: T, b: T) raises -> Owned:
    var c = _Call("aten::matmul", "")
    c.t(a)
    c.t(b)
    return c.one()


def _d_sum(x: T, dims: List[Int], keepdim: Bool) raises -> Owned:
    var c = _Call("aten::sum", "dim_IntList")
    c.t(x)
    c.ints(dims)
    c.b(keepdim)
    c.none()
    return c.one()


def _d_amax_last(x: T) raises -> Owned:
    var c = _Call("aten::amax", "")
    c.t(x)
    c.ints([-1])
    c.b(True)
    return c.one()


def _d_eq(x: T, s: Float64) raises -> Owned:
    var c = _Call("aten::eq", "Scalar")
    c.t(x)
    c.s(s)
    return c.one()


def _d_masked_fill(x: T, mask: T, s: Float64) raises -> Owned:
    var c = _Call("aten::masked_fill", "Scalar")
    c.t(x)
    c.t(mask)
    c.s(s)
    return c.one()


def _d_masked_fill_(x: T, mask: T, s: Float64) raises:
    var c = _Call("aten::masked_fill_", "Scalar")
    c.t(x)
    c.t(mask)
    c.s(s)
    _ = c.run(1)


def _d_triu(x: T, diagonal: Int) raises -> Owned:
    var c = _Call("aten::triu", "")
    c.t(x)
    c.i(diagonal)
    return c.one()


def _swap_last(t: T) raises -> Owned:
    """`t.transpose(-2, -1)` as a view (a fresh handle on `t`'s storage)."""
    var dims = t.logical_shape()
    var strides = List[Int](capacity=t.rank)
    for i in range(t.rank):
        strides.append(t.stride(i))
    var r = t.rank
    var d = dims[r - 1]
    dims[r - 1] = dims[r - 2]
    dims[r - 2] = d
    var s = strides[r - 1]
    strides[r - 1] = strides[r - 2]
    strides[r - 2] = s
    return own(_view(t, dims, strides))


def _swap_12(t: T) raises -> Owned:
    """`t.transpose(1, 2)` of a 4-D tensor as a view: (B, S, H, D) <->
    (B, H, S, D)."""
    return own(
        _view(
            t,
            [t.dim(0), t.dim(2), t.dim(1), t.dim(3)],
            [t.stride(0), t.stride(2), t.stride(1), t.stride(3)],
        )
    )


def _as(t: T, stype: Int32) raises -> Owned:
    """`t` in `stype` as an owned handle (a new reference when it already
    is: `cast_to` hands the input itself back then)."""
    if t.stype == stype:
        return own(T(retain(t)))
    return own(cast_to(t, stype))


def _acc_stype(t: T) -> Int32:
    return ST_FLOAT64 if t.dtype == DType.float64 else ST_FLOAT32


def _math_probs(
    q: T,
    k: T,
    bias: Optional[T],
    causal_off: Int,
    scale: Float64,
    acc: Int32,
) raises -> Tuple[T, T]:
    """(P, lse): the softmax probabilities (B, H, L, S) and the row
    logsumexp (B, H, L, 1), both in `acc`. A fully masked row has P = 0 and
    lse = -inf."""
    var qa = _as(q, acc)
    var ka = _as(k, acc)
    var kt = _swap_last(ka.t)
    var s = _d_matmul(qa.t, kt.t)
    _ = kt^
    _ = ka^
    _ = qa^
    _d_mul_scalar_(s.t, scale)
    if bias:
        _d_inplace("aten::add_", "Tensor", s.t, bias.value())
    if causal_off != NO_CAUSAL:
        var ones = own(_alloc(q.device, ST_BOOL, [q.dim(2), k.dim(2)]))
        fill_value(ones.t, 1.0)
        var hidden = _d_triu(ones.t, causal_off + 1)
        _d_masked_fill_(s.t, hidden.t, _NEG_INF)
        _ = hidden^
        _ = ones^
    var m = _d_amax_last(s.t)
    var m_inf = _d_eq(m.t, _NEG_INF)
    var m_safe = _d_masked_fill(m.t, m_inf.t, 0.0)
    _ = m_inf^
    _ = m^
    var e = _d_binary("aten::sub", s.t, m_safe.t)
    _ = s^
    var e2 = _d_unary("aten::exp", e.t)
    _ = e^
    var z = _d_sum(e2.t, [-1], True)
    var z0 = _d_eq(z.t, 0.0)
    var z_safe = _d_masked_fill(z.t, z0.t, 1.0)
    _ = z0^
    var p = _d_binary("aten::div", e2.t, z_safe.t)
    _ = z_safe^
    _ = e2^
    var lse = _d_unary("aten::log", z.t)
    _ = z^
    _d_inplace("aten::add_", "Tensor", lse.t, m_safe.t)
    _ = m_safe^
    return (p.take(), lse.take())


def _math_lse_forward(
    q: T,
    k: T,
    v: T,
    bias: Optional[T],
    causal_off: Int,
    scale: Float64,
    masked_lse: Int,
) raises -> Tuple[T, T]:
    """(out, lse): out (B, H, L, Ev) dense in q's dtype, lse (B, H, L)
    float32, a fully masked row's lse following `masked_lse`."""
    var acc = _acc_stype(q)
    var pr = _math_probs(q, k, bias, causal_off, scale, acc)
    var p = own(pr[0].copy())
    var lse4 = own(pr[1].copy())
    var va = _as(v, acc)
    var o = _d_matmul(p.t, va.t)
    _ = va^
    _ = p^
    var out = _as(o.t, q.stype)
    _ = o^
    var lse32 = _as(lse4.t, ST_FLOAT32)
    _ = lse4^
    if masked_lse != LSE_MASKED_CUDNN:
        var masked = _d_eq(lse32.t, _NEG_INF)
        _d_masked_fill_(
            lse32.t,
            masked.t,
            _POS_INF if masked_lse == LSE_MASKED_FLASH else 0.0,
        )
        _ = masked^
    var lse = own(
        _view(
            lse32.t,
            [q.dim(0), q.dim(1), q.dim(2)],
            [lse32.t.stride(0), lse32.t.stride(1), lse32.t.stride(2)],
        )
    )
    _ = lse32^
    var od = _materialize(out^)
    return (od.take(), lse.take())


def _reduce_to(x: T, shape: List[Int]) raises -> Owned:
    """`x` summed down to `shape`, which broadcasts to `x`'s shape."""
    var cur = own(T(retain(x)))
    var lead = x.rank - len(shape)
    if lead > 0:
        var dims = List[Int]()
        for i in range(lead):
            dims.append(i)
        var summed = _d_sum(cur.t, dims, False)
        _ = cur^  # alive across the call that reads it
        cur = summed^
    var ones = List[Int]()
    for i in range(len(shape)):
        if shape[i] == 1 and cur.t.dim(i) != 1:
            ones.append(i)
    if len(ones) > 0:
        var kept = _d_sum(cur.t, ones, True)
        _ = cur^
        cur = kept^
    return cur^


def _math_lse_backward(
    grad: T,
    q: T,
    k: T,
    v: T,
    bias: Optional[T],
    causal_off: Int,
    scale: Float64,
    want_bias_grad: Bool,
) raises -> Tuple[T, T, T, T]:
    """(dq, dk, dv, dbias): dense, in each input's dtype; dbias has `bias`'s
    shape and dtype, and is q itself (a dummy the caller must not return)
    when not wanted."""
    var acc = _acc_stype(q)
    var pr = _math_probs(q, k, bias, causal_off, scale, acc)
    var p = own(pr[0].copy())
    _ = own(pr[1].copy())
    var ga = _as(grad, acc)
    var va = _as(v, acc)
    var pt = _swap_last(p.t)
    var dv = _d_matmul(pt.t, ga.t)
    _ = pt^
    var vt = _swap_last(va.t)
    var dp = _d_matmul(ga.t, vt.t)
    _ = vt^
    _ = va^
    _ = ga^
    var pdp = _d_binary("aten::mul", p.t, dp.t)
    var delta = _d_sum(pdp.t, [-1], True)
    _ = pdp^
    _d_inplace("aten::sub_", "Tensor", dp.t, delta.t)
    _ = delta^
    var ds = _d_binary("aten::mul", p.t, dp.t)
    _ = dp^
    _ = p^
    var dbias = _borrowed(q)
    if want_bias_grad and bias:
        var bshape = bias.value().logical_shape()
        var red = _reduce_to(ds.t, bshape)
        dbias = _materialize(_as(red.t, bias.value().stype))
        _ = red^
    # Out of place: `dbias` may alias `ds` (same shape and dtype).
    var sc = _Call("aten::mul", "Scalar")
    sc.t(ds.t)
    sc.s(scale)
    var dss = sc.one()
    _ = ds^  # read through its handle by the call above
    var ka = _as(k, acc)
    var dq = _d_matmul(dss.t, ka.t)
    _ = ka^
    var qa = _as(q, acc)
    var dst = _swap_last(dss.t)
    var dk = _d_matmul(dst.t, qa.t)
    _ = dst^
    _ = qa^
    _ = dss^
    var dq_o = _materialize(_as(dq.t, q.stype))
    var dk_o = _materialize(_as(dk.t, k.stype))
    var dv_o = _materialize(_as(dv.t, v.stype))
    _ = dq^
    _ = dk^
    _ = dv^
    return (dq_o.take(), dk_o.take(), dv_o.take(), dbias.take())


def _math_check(q: T, k: T, v: T, what: StaticString) raises:
    """The inputs the math route takes: (B, H, L, E) / (B, H, S, E) /
    (B, H, S, Ev) floating tensors of one dtype on one device."""
    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        unsupported(String(what, " expects 4-D query/key/value"))
    if not _same_device(q, k, v):
        unsupported(String(what, ": query/key/value on different devices"))
    if k.stype != q.stype or v.stype != q.stype:
        raise Error(
            what,
            ": expected query, key and value to have the same dtype",
        )
    if not _is_float(q) and q.dtype != DType.float64:
        raise Error(
            what,
            ": expected a floating point query, got ",
            _scalar_type_name(q.dtype),
        )
    if q.dtype == DType.float64 and _api(q.device) == "metal":
        unsupported(String(what, " in float64: Apple GPUs have no float64"))
    if k.dim(1) != q.dim(1) or v.dim(1) != q.dim(1):
        unsupported(String(what, " with grouped-query (fewer K/V heads)"))
    if (
        k.dim(0) != q.dim(0)
        or v.dim(0) != q.dim(0)
        or k.dim(3) != q.dim(3)
        or v.dim(2) != k.dim(2)
    ):
        raise Error(what, ": query/key/value shapes do not match")


def _causal_off(
    is_causal: Bool, bottom_right: Bool, q_len: Int, kv_len: Int
) -> Int:
    if not is_causal:
        return NO_CAUSAL
    return kv_len - q_len if bottom_right else 0


def _flash_any_forward(
    q: T,
    k: T,
    v: T,
    is_causal: Bool,
    scale: Float64,
    bottom_right: Bool,
) raises -> Tuple[T, T]:
    """(out (B, H, L, D), lse (B, H, L) float32) from a fused flash kernel
    when one takes the inputs, else from the math route."""
    var fa4 = _fa4_plan(
        q, k, v, False, 0.0, is_causal, False, allow_any_seqlen=True
    )
    var fused = _fused_fa_plan(q, k, v, False, 0.0, False)
    if fa4.ok or fused.ok:
        return _flash_forward(q, k, v, is_causal, 0.0, scale)
    _math_check(q, k, v, "flash attention")
    return _math_lse_forward(
        q,
        k,
        v,
        None,
        _causal_off(is_causal, bottom_right, q.dim(2), k.dim(2)),
        scale,
        LSE_MASKED_FLASH,
    )


def _flash_any_backward(
    grad: T,
    q: T,
    k: T,
    v: T,
    o: T,
    lse: T,
    is_causal: Bool,
    scale: Float64,
    bottom_right: Bool,
) raises -> Tuple[T, T, T]:
    """Gradients from the fused flash backward that matches the forward's
    kernel, else from the math route (which recomputes the probabilities,
    so it is right whichever route ran the forward)."""
    # No `allow_any_seqlen` here: a partial last tile in the backward tile
    # machinery would be a silently wrong gradient, so such shapes take the
    # math route instead.
    var fa4 = _fa4_plan(
        q, k, v, False, 0.0, is_causal, False, allow_any_seqlen=False
    )
    if fa4.ok:
        if o.stype != q.stype or not o.same_shape(q):
            unsupported("flash attention backward: out does not match query")
        if (
            lse.dtype != DType.float32
            or lse.rank != 3
            or lse.dim(0) != fa4.batch
            or lse.dim(1) != fa4.heads
            or lse.dim(2) != fa4.seqlen
        ):
            unsupported("flash attention backward: logsumexp has a bad shape")
        return _fa4_backward(q, k, v, o, lse, grad, fa4, scale)
    var fused = _fused_fa_plan(q, k, v, False, 0.0, False)
    if fused.ok:
        return _fused_fa_backward(
            grad, q, k, v, o, lse, fused, is_causal, scale
        )
    _math_check(q, k, v, "flash attention backward")
    var g = _math_lse_backward(
        grad,
        q,
        k,
        v,
        None,
        _causal_off(is_causal, bottom_right, q.dim(2), k.dim(2)),
        scale,
        False,
    )
    return (g[0].copy(), g[1].copy(), g[2].copy())


def _ret_undefined(rets: Values, i: Int):
    """An undefined at::Tensor result (a None record in a Tensor slot)."""
    rets[unsafe_offset=i] = Value(TAG_NONE, 0, 0, 0)


def _philox_pair(device: Int) raises -> Tuple[T, T]:
    """The 0-dim int64 seed/offset pair of a dropout-free call: unobserved,
    zero-filled so nothing reads uninitialized memory."""
    var seed = own(_alloc(device, ST_INT64, List[Int]()))
    var offset = own(_alloc(device, ST_INT64, List[Int]()))
    fill_value(seed.t, 0.0)
    fill_value(offset.t, 0.0)
    return (seed.take(), offset.take())


def _efficient_lse(lse: T, compute: Bool) raises -> Owned:
    """The memory-efficient kernels' logsumexp: (B, H, ceil(L / 32) * 32)
    float32, the padding +inf, or (B, H, 0) when not computed."""
    var b = lse.dim(0)
    var h = lse.dim(1)
    var l = lse.dim(2)
    if not compute:
        return own(_alloc(lse.device, ST_FLOAT32, [b, h, 0]))
    var lp = (l + 31) // 32 * 32
    var out = own(_alloc(lse.device, ST_FLOAT32, [b, h, lp]))
    fill_value(out.t, _POS_INF)
    var head = own(_view(out.t, [b, h, l], [h * lp, lp, 1]))
    copy_strided_into(head.t, lse)
    _ = head^
    return out^


def _bshd_dense(t_bhsd: T) raises -> Owned:
    """A (B, H, S, D) result as the dense (B, S, H, D) tensor the CUDA
    (B, S, H, D)-layout entry points return."""
    var b = t_bhsd.dim(0)
    var h = t_bhsd.dim(1)
    var s = t_bhsd.dim(2)
    var d = t_bhsd.dim(3)
    var out = own(_alloc(t_bhsd.device, t_bhsd.stype, [b, s, h, d]))
    var view = _swap_12(out.t)
    copy_strided_into(view.t, t_bhsd)
    _ = view^
    return out^


def _no_varlen(v: Value, what: StaticString) raises:
    """Decline a defined, non-empty cumulative-sequence (ragged batch)
    argument: the nested-tensor layout has no kernel here."""
    var t = v_opt_tensor(v)
    if t and t.value().numel > 0:
        unsupported(String(what, " with cumulative sequence lengths (varlen)"))


def _no_window(v: Value, what: StaticString) raises:
    if not v_is_none(v) and v_int(v) != -1:
        unsupported(String(what, " with a sliding window"))


def _no_tensor(v: Value, what: String) raises:
    if v_opt_tensor(v):
        unsupported(what)


def _cudnn_dtype(q: T) raises:
    if q.dtype != DType.float16 and q.dtype != DType.bfloat16:
        raise Error(
            "cuDNN attention only supports float16 and bfloat16, got ",
            _scalar_type_name(q.dtype),
        )


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

    ATen's composite reads `_fused_sdp_choice_stub` (a DispatchStub).
    `native/csrc/shim_sdpa.cpp` registers its PrivateUse1 entry and forwards
    to this op, so supported inputs select the fused kernels below.

    The global CUDA SDP backend switches are not exposed through the native
    ABI. This choice currently depends on the tensors and operation arguments.
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
    if enable_gqa and q.rank == 4 and k.rank == 4 and v.rank == 4:
        if k.dim(1) == q.dim(1) and v.dim(1) == q.dim(1):
            # Equal head counts: GQA is a no-op, and ATen hands the chosen
            # op the same K/V either way.
            enable_gqa = False
        elif (
            not needs_grad
            and not has_mask
            and dropout_p == 0.0
            and _same_device(q, k, v)
            and _is_float(q)
            and k.stype == q.stype
            and v.stype == q.stype
            and q.numel > 0
            and k.numel > 0
            and _gqa_heads_divide(q, k, v)
        ):
            # The efficient op repeats K/V up to Q's heads, then runs the
            # fused cascade on dense tensors every route accepts. Training
            # stays on the math decomposition, whose repeat_interleave
            # autograd sums each query group's gradient back onto its KV head.
            ret_int(rets, 0, SDP_EFFICIENT)
            return
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
    # Training stays on the differentiable math decomposition: the efficient
    # op's backward here is that same decomposition run in float32 (the math
    # route below), so routing training through it would buy nothing.
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
def _flash_attention_impl[
    bottom_right: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`bottom_right`: the causal mask's alignment when L != S -- CUDA's
    flash kernels align it bottom-right, the CPU overload top-left."""
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
    var pair = _flash_any_forward(q, k, v, is_causal, scale, bottom_right)
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


def op_flash_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _flash_attention_impl[True](args, n_args, rets, n_rets)


# aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out,
#   Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp,
#   Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k,
#   float dropout_p, bool is_causal, Tensor philox_seed,
#   Tensor philox_offset, *, float? scale=None)
#   -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
def _flash_attention_backward_impl[
    bottom_right: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """`cum_seq_q`/`cum_seq_k`/`philox_*` are deliberately not read: only this
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
    var g = _flash_any_backward(
        grad, q, k, v, out, lse, is_causal, scale, bottom_right
    )
    var dq = own(g[0].copy())
    var dk = own(g[1].copy())
    var dv = own(g[2].copy())
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)


def op_flash_attention_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _flash_attention_backward_impl[True](args, n_args, rets, n_rets)


# aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key,
#   Tensor value, Tensor? attn_bias, bool compute_log_sumexp,
#   float dropout_p=0.0, bool is_causal=False, *, float? scale=None)
#   -> (Tensor output, Tensor log_sumexp, Tensor philox_seed,
#       Tensor philox_offset)
def op_efficient_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """The fused inference cascade -- FA4, the fused gfx942 kernels, the
    decode kernel, or the bmm + fused-causal-softmax + bmm decomposition --
    when there is no bias and no log-sum-exp to return; the math route with
    a log-sum-exp otherwise.

    A grad-requiring call needs no log-sum-exp from here: its backward
    (`_scaled_dot_product_efficient_attention_backward` below) recomputes
    the probabilities.
    """
    var q = v_tensor(args[unsafe_offset=0])
    var k = v_tensor(args[unsafe_offset=1])
    var v = v_tensor(args[unsafe_offset=2])
    var bias = v_opt_tensor(args[unsafe_offset=3])
    var compute_lse = v_bool(args[unsafe_offset=4])
    var dropout_p = v_f64(args[unsafe_offset=5])
    var is_causal = v_bool(args[unsafe_offset=6])
    if dropout_p != 0.0:
        unsupported("efficient attention with dropout")
    if q.rank != 4:
        unsupported("efficient attention expects 4-D query/key/value")
    var scale = _scale_of(args[unsafe_offset=7], q.dim(3))

    if bias or compute_lse:
        _math_check(q, k, v, "efficient attention")
        var pair = _math_lse_forward(
            q,
            k,
            v,
            bias,
            _causal_off(is_causal, False, q.dim(2), k.dim(2)),
            scale,
            LSE_MASKED_EFFICIENT,
        )
        var mout = own(pair[0].copy())
        var mlse = own(pair[1].copy())
        var plse = _efficient_lse(mlse.t, compute_lse)
        _ = mlse^
        var mph = _philox_pair(q.device)
        var mseed = own(mph[0].copy())
        var moff = own(mph[1].copy())
        ret_owned(rets, 0, mout)
        ret_owned(rets, 1, plse)
        ret_owned(rets, 2, mseed)
        ret_owned(rets, 3, moff)
        return

    # Grouped-query K/V (what `_fused_sdp_choice` sends here for
    # `enable_gqa=True`): repeat them up to Q's heads, then run the ordinary
    # equal-head cascade.
    var ke = _borrowed(k)
    var ve = _borrowed(v)
    if (
        k.rank == 4
        and v.rank == 4
        and (k.dim(1) != q.dim(1) or v.dim(1) != q.dim(1))
    ):
        if not _gqa_heads_divide(q, k, v):
            unsupported(
                "efficient attention: the key/value head counts must divide"
                " the query's"
            )
        ke = _gqa_expand(k, q.dim(1))
        ve = _gqa_expand(v, q.dim(1))

    var out = own(
        _efficient_forward(q, ke.t, ve.t, is_causal, dropout_p, scale)
    )
    _ = ke
    _ = ve
    var lse = own(_alloc(q.device, ST_FLOAT32, [q.dim(0), q.dim(1), 0]))
    var ph = _philox_pair(q.device)
    var seed = own(ph[0].copy())
    var offset = own(ph[1].copy())
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, lse)
    ret_owned(rets, 2, seed)
    ret_owned(rets, 3, offset)


# aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_,
#   Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out,
#   Tensor logsumexp, Tensor philox_seed, Tensor philox_offset,
#   float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *,
#   float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)
def op_efficient_attention_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var q = v_tensor(args[unsafe_offset=1])
    var k = v_tensor(args[unsafe_offset=2])
    var v = v_tensor(args[unsafe_offset=3])
    var bias = v_opt_tensor(args[unsafe_offset=4])
    var dropout_p = v_f64(args[unsafe_offset=9])
    var mask = _bool_list(args[unsafe_offset=10])
    var is_causal = v_bool(args[unsafe_offset=11])
    if dropout_p != 0.0:
        unsupported("efficient attention backward with dropout")
    _math_check(q, k, v, "efficient attention backward")
    var scale = _scale_of(args[unsafe_offset=12], q.dim(3))
    var want_bias = Bool(bias) and len(mask) > 3 and mask[3]
    var g = _math_lse_backward(
        grad,
        q,
        k,
        v,
        bias,
        _causal_off(is_causal, False, q.dim(2), k.dim(2)),
        scale,
        want_bias,
    )
    var dq = own(g[0].copy())
    var dk = own(g[1].copy())
    var dv = own(g[2].copy())
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)
    if want_bias:
        var db = own(g[3].copy())
        ret_owned(rets, 3, db)
    else:
        _ret_undefined(rets, 3)


# aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value,
#   Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k,
#   SymInt? max_seqlen_q, SymInt? max_seqlen_k, float dropout_p,
#   int custom_mask_type, bool compute_log_sumexp=False, *,
#   float? scale=None, Tensor? seqlen_k=None, int? window_size=None)
#   -> (Tensor output, Tensor logsumexp, Tensor philox_seed,
#       Tensor philox_offset, SymInt max_seqlen_batch_q,
#       SymInt max_seqlen_batch_k)
def op_efficient_attention_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """The (B, M, H, K)-layout memory-efficient forward, on the math route.
    `custom_mask_type` 1 is causal from the top left, 2 from the bottom
    right."""
    comptime W = "_efficient_attention_forward"
    var q_in = v_tensor(args[unsafe_offset=0])
    var k_in = v_tensor(args[unsafe_offset=1])
    var v_in = v_tensor(args[unsafe_offset=2])
    var bias = v_opt_tensor(args[unsafe_offset=3])
    _no_varlen(args[unsafe_offset=4], W)
    _no_varlen(args[unsafe_offset=5], W)
    var dropout_p = v_f64(args[unsafe_offset=8])
    var mask_type = v_int(args[unsafe_offset=9])
    var compute_lse = v_bool(args[unsafe_offset=10])
    if n_args > 12:
        _no_tensor(args[unsafe_offset=12], W + " with seqlen_k")
    if n_args > 13 and not v_is_none(args[unsafe_offset=13]):
        unsupported(W + " with a sliding window")
    if dropout_p != 0.0:
        unsupported(W + " with dropout")
    if mask_type < 0 or mask_type > 2:
        raise Error(W + ": unsupported custom_mask_type ", mask_type)
    if q_in.rank != 4 or k_in.rank != 4 or v_in.rank != 4:
        unsupported(W + " expects 4-D query/key/value")
    var q = _swap_12(q_in)
    var k = _swap_12(k_in)
    var v = _swap_12(v_in)
    _math_check(q.t, k.t, v.t, W)
    var scale = _scale_of(args[unsafe_offset=11], q.t.dim(3))
    var off = NO_CAUSAL
    if mask_type != 0:
        off = _causal_off(True, mask_type == 2, q.t.dim(2), k.t.dim(2))
    var pair = _math_lse_forward(
        q.t, k.t, v.t, bias, off, scale, LSE_MASKED_EFFICIENT
    )
    _ = q^  # the views are read through their handles above
    _ = k^
    _ = v^
    var out_bhsd = own(pair[0].copy())
    var lse = own(pair[1].copy())
    var out = _bshd_dense(out_bhsd.t)
    _ = out_bhsd^
    var plse = _efficient_lse(lse.t, compute_lse)
    _ = lse^
    var ph = _philox_pair(q_in.device)
    var seed = own(ph[0].copy())
    var offset = own(ph[1].copy())
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, plse)
    ret_owned(rets, 2, seed)
    ret_owned(rets, 3, offset)
    ret_int(rets, 4, q_in.dim(1))
    ret_int(rets, 5, k_in.dim(1))


# aten::_efficient_attention_backward(Tensor grad_out_, Tensor query,
#   Tensor key, Tensor value, Tensor? bias, Tensor out,
#   Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q,
#   SymInt max_seqlen_k, Tensor logsumexp, float dropout_p,
#   Tensor philox_seed, Tensor philox_offset, int custom_mask_type,
#   bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None,
#   int? window_size=None, bool shared_storage_dqdkdv=False)
#   -> (Tensor, Tensor, Tensor, Tensor)
def op_efficient_attention_backward_bmhk(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime W = "_efficient_attention_backward"
    var grad_in = v_tensor(args[unsafe_offset=0])
    var q_in = v_tensor(args[unsafe_offset=1])
    var k_in = v_tensor(args[unsafe_offset=2])
    var v_in = v_tensor(args[unsafe_offset=3])
    var bias = v_opt_tensor(args[unsafe_offset=4])
    _no_varlen(args[unsafe_offset=6], W)
    _no_varlen(args[unsafe_offset=7], W)
    var dropout_p = v_f64(args[unsafe_offset=11])
    var mask_type = v_int(args[unsafe_offset=14])
    var bias_grad = v_bool(args[unsafe_offset=15])
    if n_args > 18 and not v_is_none(args[unsafe_offset=18]):
        unsupported(W + " with a sliding window")
    if dropout_p != 0.0:
        unsupported(W + " with dropout")
    if mask_type < 0 or mask_type > 2:
        raise Error(W + ": unsupported custom_mask_type ", mask_type)
    if q_in.rank != 4 or k_in.rank != 4 or v_in.rank != 4 or grad_in.rank != 4:
        unsupported(W + " expects 4-D query/key/value")
    var grad = _swap_12(grad_in)
    var q = _swap_12(q_in)
    var k = _swap_12(k_in)
    var v = _swap_12(v_in)
    _math_check(q.t, k.t, v.t, W)
    var scale = _scale_of(args[unsafe_offset=16], q.t.dim(3))
    var off = NO_CAUSAL
    if mask_type != 0:
        off = _causal_off(True, mask_type == 2, q.t.dim(2), k.t.dim(2))
    var want_bias = Bool(bias) and bias_grad
    var g = _math_lse_backward(
        grad.t, q.t, k.t, v.t, bias, off, scale, want_bias
    )
    _ = grad^  # the views are read through their handles above
    _ = q^
    _ = k^
    _ = v^
    var dq_b = own(g[0].copy())
    var dk_b = own(g[1].copy())
    var dv_b = own(g[2].copy())
    var dq = _bshd_dense(dq_b.t)
    var dk = _bshd_dense(dk_b.t)
    var dv = _bshd_dense(dv_b.t)
    _ = dq_b^
    _ = dk_b^
    _ = dv_b^
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)
    if want_bias:
        var db = own(g[3].copy())
        ret_owned(rets, 3, db)
    else:
        _ret_undefined(rets, 3)


# aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value,
#   Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k,
#   float dropout_p, bool is_causal, bool return_debug_mask, *,
#   float? scale=None, SymInt? window_size_left=None,
#   SymInt? window_size_right=None, Tensor? seqused_k=None,
#   Tensor? alibi_slopes=None, Tensor? block_table=None,
#   int? num_splits=None)
#   -> (Tensor output, Tensor softmax_logsumexp, Tensor rng_state,
#       Tensor unused, Tensor debug_attn_mask)
def _flash_forward_bshd(
    args: Values, n_args: Int, base: Int, what: StaticString
) raises -> Tuple[T, T]:
    """The (B, S, H, D)-layout flash forward whose schema starts at argument
    `base`: (output (B, L, H, D) dense, lse (B, H, L) float32)."""
    var q_in = v_tensor(args[unsafe_offset=base])
    var k_in = v_tensor(args[unsafe_offset=base + 1])
    var v_in = v_tensor(args[unsafe_offset=base + 2])
    _no_varlen(args[unsafe_offset=base + 3], what)
    _no_varlen(args[unsafe_offset=base + 4], what)
    var dropout_p = v_f64(args[unsafe_offset=base + 7])
    var is_causal = v_bool(args[unsafe_offset=base + 8])
    if v_bool(args[unsafe_offset=base + 9]):
        unsupported(String(what, " with return_debug_mask=True"))
    # The trailing optional arguments vary across torch releases (2.11 has
    # no block_table / num_splits): read only those this schema has.
    if n_args > base + 11:
        _no_window(args[unsafe_offset=base + 11], what)
    if n_args > base + 12:
        _no_window(args[unsafe_offset=base + 12], what)
    if n_args > base + 13:
        _no_tensor(
            args[unsafe_offset=base + 13], String(what, " with seqused_k")
        )
    if n_args > base + 14:
        _no_tensor(args[unsafe_offset=base + 14], String(what, " with ALiBi"))
    if n_args > base + 15:
        _no_tensor(
            args[unsafe_offset=base + 15], String(what, " with a block table")
        )
    if dropout_p != 0.0:
        unsupported(String(what, " with dropout"))
    if q_in.rank != 4 or k_in.rank != 4 or v_in.rank != 4:
        unsupported(String(what, " expects 4-D query/key/value"))
    var q = _swap_12(q_in)
    var k = _swap_12(k_in)
    var v = _swap_12(v_in)
    var scale = _scale_of(args[unsafe_offset=base + 10], q.t.dim(3))
    var pair = _flash_any_forward(q.t, k.t, v.t, is_causal, scale, True)
    _ = q^  # the views are read through their handles above
    _ = k^
    _ = v^
    var out_bhsd = own(pair[0].copy())
    var lse = own(pair[1].copy())
    var out = _bshd_dense(out_bhsd.t)
    _ = out_bhsd^
    return (out.take(), lse.take())


def op_flash_attention_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var q = v_tensor(args[unsafe_offset=0])
    var pair = _flash_forward_bshd(args, n_args, 0, "_flash_attention_forward")
    var out = own(pair[0].copy())
    var lse = own(pair[1].copy())
    var rng_state = own(_alloc(q.device, ST_UINT64, [2]))
    var unused = own(_alloc(q.device, ST_UINT64, List[Int]()))
    var debug = own(_alloc(q.device, q.stype, [0]))
    fill_value(rng_state.t, 0.0)
    fill_value(unused.t, 0.0)
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, lse)
    ret_owned(rets, 2, rng_state)
    ret_owned(rets, 3, unused)
    ret_owned(rets, 4, debug)


# aten::_flash_attention_forward_no_dropout_inplace(Tensor(a!) out,
#   Tensor query, Tensor key, Tensor value, <_flash_attention_forward's
#   remaining arguments>) -> Tensor softmax_logsumexp
def op_flash_attention_forward_no_dropout_inplace(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dst = v_tensor(args[unsafe_offset=0])
    if v_f64(args[unsafe_offset=8]) != 0.0:
        raise Error(
            "_flash_attention_forward_no_dropout_inplace: dropout_p must be 0"
        )
    var pair = _flash_forward_bshd(
        args, n_args, 1, "_flash_attention_forward_no_dropout_inplace"
    )
    var out = own(pair[0].copy())
    var lse = own(pair[1].copy())
    if not dst.same_shape(out.t) or dst.stype != out.t.stype:
        raise Error(
            (
                "_flash_attention_forward_no_dropout_inplace: out must have the"
                " query's shape and dtype, got "
            ),
            shape_str(dst),
        )
    if not dst.on_mojo() or dst.device != out.t.device:
        raise Error(
            "_flash_attention_forward_no_dropout_inplace: out must be on the"
            " query's device"
        )
    copy_strided_into(dst, out.t)
    _ = out^
    dst.bump_version()
    ret_owned(rets, 0, lse)


# aten::_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key,
#   Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q,
#   Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p,
#   bool is_causal, Tensor rng_state, Tensor unused, *, float? scale=None,
#   SymInt? window_size_left=None, SymInt? window_size_right=None)
#   -> (Tensor, Tensor, Tensor)
def op_flash_attention_backward_bshd(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime W = "_flash_attention_backward"
    var grad_in = v_tensor(args[unsafe_offset=0])
    var q_in = v_tensor(args[unsafe_offset=1])
    var k_in = v_tensor(args[unsafe_offset=2])
    var v_in = v_tensor(args[unsafe_offset=3])
    var out_in = v_tensor(args[unsafe_offset=4])
    var lse = v_tensor(args[unsafe_offset=5])
    _no_varlen(args[unsafe_offset=6], W)
    _no_varlen(args[unsafe_offset=7], W)
    var dropout_p = v_f64(args[unsafe_offset=10])
    var is_causal = v_bool(args[unsafe_offset=11])
    if n_args > 15:
        _no_window(args[unsafe_offset=15], W)
    if n_args > 16:
        _no_window(args[unsafe_offset=16], W)
    if dropout_p != 0.0:
        unsupported(W + " with dropout")
    if (
        q_in.rank != 4
        or k_in.rank != 4
        or v_in.rank != 4
        or grad_in.rank != 4
        or out_in.rank != 4
    ):
        unsupported(W + " expects 4-D query/key/value")
    var grad = _swap_12(grad_in)
    var q = _swap_12(q_in)
    var k = _swap_12(k_in)
    var v = _swap_12(v_in)
    var out = _swap_12(out_in)
    var scale = _scale_of(args[unsafe_offset=14], q.t.dim(3))
    var g = _flash_any_backward(
        grad.t, q.t, k.t, v.t, out.t, lse, is_causal, scale, True
    )
    _ = grad^  # the views are read through their handles above
    _ = q^
    _ = k^
    _ = v^
    _ = out^
    var dq_b = own(g[0].copy())
    var dk_b = own(g[1].copy())
    var dv_b = own(g[2].copy())
    var dq = _bshd_dense(dq_b.t)
    var dk = _bshd_dense(dk_b.t)
    var dv = _bshd_dense(dv_b.t)
    _ = dq_b^
    _ = dk_b^
    _ = dv_b^
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)


def _cudnn_forward(
    q: T,
    k: T,
    v: T,
    bias: Optional[T],
    compute_lse: Bool,
    dropout_p: Float64,
    is_causal: Bool,
    scale_v: Value,
    rets: Values,
) raises:
    """cuDNN's forward contract on the math route: causal from the top left,
    the logsumexp (B, H, L, 1) float32 (a fully masked row's -inf), the
    cumulative-sequence tensors and debug mask undefined."""
    if dropout_p != 0.0:
        unsupported("cuDNN attention with dropout")
    _math_check(q, k, v, "cuDNN attention")
    _cudnn_dtype(q)
    var scale = _scale_of(scale_v, q.dim(3))
    var pair = _math_lse_forward(
        q,
        k,
        v,
        bias,
        _causal_off(is_causal, False, q.dim(2), k.dim(2)),
        scale,
        LSE_MASKED_CUDNN,
    )
    var out = own(pair[0].copy())
    var lse = own(pair[1].copy())
    ret_owned(rets, 0, out)
    if compute_lse:
        var lse4 = own(
            _view(
                lse.t,
                [q.dim(0), q.dim(1), q.dim(2), 1],
                [q.dim(1) * q.dim(2), q.dim(2), 1, 1],
            )
        )
        ret_owned(rets, 1, lse4)
    else:
        _ret_undefined(rets, 1)
    _ = lse^
    _ret_undefined(rets, 2)
    _ret_undefined(rets, 3)
    ret_int(rets, 4, q.dim(2))
    ret_int(rets, 5, k.dim(2))
    var ph = _philox_pair(q.device)
    var seed = own(ph[0].copy())
    var offset = own(ph[1].copy())
    ret_owned(rets, 6, seed)
    ret_owned(rets, 7, offset)
    _ret_undefined(rets, 8)


# aten::_scaled_dot_product_cudnn_attention(Tensor query, Tensor key,
#   Tensor value, Tensor? attn_bias, bool compute_log_sumexp,
#   float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False,
#   *, float? scale=None) -> (Tensor output, Tensor logsumexp,
#   Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k,
#   Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
def op_cudnn_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _cudnn_forward(
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
        v_opt_tensor(args[unsafe_offset=3]),
        v_bool(args[unsafe_offset=4]),
        v_f64(args[unsafe_offset=5]),
        v_bool(args[unsafe_offset=6]),
        args[unsafe_offset=8],
        rets,
    )


# aten::_cudnn_attention_forward(Tensor query, Tensor key, Tensor value,
#   Tensor? attn_bias, Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q,
#   SymInt max_k, bool compute_log_sumexp, float dropout_p=0.0,
#   bool is_causal=False, bool return_debug_mask=False, *,
#   float? scale=None, Tensor? seqused_k=None, Tensor? block_table=None)
#   -> (the 9-tuple above)
def op_cudnn_attention_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime W = "_cudnn_attention_forward"
    _no_varlen(args[unsafe_offset=4], W)
    _no_varlen(args[unsafe_offset=5], W)
    if n_args > 13:  # torch >= 2.12
        _no_tensor(args[unsafe_offset=13], W + " with seqused_k")
    if n_args > 14:
        _no_tensor(args[unsafe_offset=14], W + " with a block table")
    _cudnn_forward(
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
        v_opt_tensor(args[unsafe_offset=3]),
        v_bool(args[unsafe_offset=8]),
        v_f64(args[unsafe_offset=9]),
        v_bool(args[unsafe_offset=10]),
        args[unsafe_offset=12],
        rets,
    )


# aten::_scaled_dot_product_cudnn_attention_backward(Tensor grad_out,
#   Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp,
#   Tensor philox_seed, Tensor philox_offset, Tensor attn_bias,
#   Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k,
#   float dropout_p, bool is_causal, *, float? scale=None)
#   -> (Tensor, Tensor, Tensor)
# (`_cudnn_attention_backward` has the same schema.)
def op_cudnn_attention_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime W = "cuDNN attention backward"
    var grad = v_tensor(args[unsafe_offset=0])
    var q = v_tensor(args[unsafe_offset=1])
    var k = v_tensor(args[unsafe_offset=2])
    var v = v_tensor(args[unsafe_offset=3])
    var bias = v_opt_tensor(args[unsafe_offset=8])
    _no_varlen(args[unsafe_offset=9], W)
    _no_varlen(args[unsafe_offset=10], W)
    var dropout_p = v_f64(args[unsafe_offset=13])
    var is_causal = v_bool(args[unsafe_offset=14])
    if dropout_p != 0.0:
        unsupported(W + " with dropout")
    _math_check(q, k, v, W)
    _cudnn_dtype(q)
    var scale = _scale_of(args[unsafe_offset=15], q.dim(3))
    var g = _math_lse_backward(
        grad,
        q,
        k,
        v,
        bias,
        _causal_off(is_causal, False, q.dim(2), k.dim(2)),
        scale,
        False,
    )
    var dq = own(g[0].copy())
    var dk = own(g[1].copy())
    var dv = own(g[2].copy())
    ret_owned(rets, 0, dq)
    ret_owned(rets, 1, dk)
    ret_owned(rets, 2, dv)


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
    var fargs = Array[Value, 7](fill=Value(TAG_NONE, 0, 0, 0))
    for i in range(5):
        fargs[i] = args[unsafe_offset=i].copy()
    fargs[5] = Value(TAG_BOOL, 0, 0, 0)  # return_debug_mask
    fargs[6] = args[unsafe_offset=6].copy()
    var frets = Array[Value, 9](fill=Value(TAG_NONE, 0, 0, 0))
    _flash_attention_impl[False](
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
    var fargs = Array[Value, 15](fill=Value(TAG_NONE, 0, 0, 0))
    for i in range(6):
        fargs[i] = args[unsafe_offset=i].copy()
    fargs[8] = Value(TAG_INT, 0, 0, 0)
    fargs[9] = Value(TAG_INT, 0, 0, 0)
    fargs[10] = args[unsafe_offset=6].copy()
    fargs[11] = args[unsafe_offset=7].copy()
    fargs[14] = args[unsafe_offset=9].copy()
    _flash_attention_backward_impl[False](
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
    impl[
        op_efficient_attention_backward,
        "_scaled_dot_product_efficient_attention_backward",
    ](site)
    impl[op_efficient_attention_forward, "_efficient_attention_forward"](site)
    impl[op_efficient_attention_backward_bmhk, "_efficient_attention_backward"](
        site
    )
    impl[op_flash_attention_forward, "_flash_attention_forward"](site)
    impl[
        op_flash_attention_forward_no_dropout_inplace,
        "_flash_attention_forward_no_dropout_inplace",
    ](site)
    impl[op_flash_attention_backward_bshd, "_flash_attention_backward"](site)
    impl[op_cudnn_attention, "_scaled_dot_product_cudnn_attention"](site)
    impl[
        op_cudnn_attention_backward,
        "_scaled_dot_product_cudnn_attention_backward",
    ](site)
    impl[op_cudnn_attention_forward, "_cudnn_attention_forward"](site)
    impl[op_cudnn_attention_backward, "_cudnn_attention_backward"](site)
