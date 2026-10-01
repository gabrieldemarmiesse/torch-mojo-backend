"""ATen ops: the transformer fast-path entry points (see
agents_docs/native_backend.md).

`nn.TransformerEncoderLayer` and `nn.MultiheadAttention` take a "fast path"
in inference (eval mode, no grad) on CPU, CUDA and PrivateUse1 devices alike:
one call to `_transformer_encoder_layer_fwd` / `_native_multi_head_attention`
instead of the module's Python composition. Both are ports of ATen's own
compositions (aten/src/ATen/native/transformers/{transformer.cpp,
attention.cpp, cuda/attention.cu}, v2.14.0), dispatched op by op onto this
device's kernels: the GEMMs, layer norm, softmax and SDPA all run there.
"""
from std.math import sqrt
from std.utils import IndexList

from tmb.backend.abi import (
    ST_BOOL,
    ST_FLOAT32,
    ST_FLOAT64,
    Owned,
    T,
    TAG_NONE,
    Value,
    Values,
    contiguous_strides,
    own,
    retain,
    ret_owned,
    v_bool,
    v_f64,
    v_int,
    v_is_none,
    v_opt_tensor,
    v_tensor,
    view_strided,
)
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.attention import (
    _Call,
    _NEG_INF,
    _alloc,
    _as,
    _d_binary,
    _d_masked_fill,
    _d_matmul,
    _d_mul_scalar_,
    _d_sum,
    _ret_undefined,
    _swap_last,
)
from tmb.ops.common import copy_strided_into, shape_str
from tmb.ops.data_movement import _scalar_type_name


def _view_at(
    base: T, dims: List[Int], strides: List[Int], offset: Int
) raises -> Owned:
    """A view over `base`'s storage at an absolute storage `offset`."""
    var shape = IndexList[MAX_RANK](1)
    var st = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - len(dims)
    for i in range(len(dims)):
        shape[pad + i] = dims[i]
        st[pad + i] = strides[i]
    return own(view_strided(base, shape, st, len(dims), offset))


def _dense_view(base: T, dims: List[Int]) raises -> Owned:
    """`base` (dense) viewed with another shape of the same numel."""
    var shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - len(dims)
    for i in range(len(dims)):
        shape[pad + i] = dims[i]
    return own(
        view_strided(
            base,
            shape,
            contiguous_strides(shape, len(dims)),
            len(dims),
            base.offset,
        )
    )


def _same(a: T, b: T) -> Bool:
    """`Tensor::is_same`: one TensorImpl behind both handles."""
    return a.impl() == b.impl()


def _float_input(t: T, what: StaticString) raises:
    if (
        t.dtype != DType.float32
        and t.dtype != DType.float16
        and t.dtype != DType.bfloat16
        and t.dtype != DType.float64
    ):
        raise Error(
            '"',
            what,
            "\" not implemented for '",
            _scalar_type_name(t.dtype),
            "'",
        )


def _linear(x: T, w: T, b: T) raises -> Owned:
    var c = _Call("aten::linear", "")
    c.t(x)
    c.t(w)
    c.t(b)
    return c.one()


def _softmax_last(x: T) raises -> Owned:
    var c = _Call("aten::_softmax", "")
    c.t(x)
    c.i(-1)
    c.b(False)
    return c.one()


def _layer_norm(x: T, dim: Int, w: T, b: T, eps: Float64) raises -> Owned:
    var c = _Call("aten::layer_norm", "")
    c.t(x)
    c.ints([dim])
    c.t(w)
    c.t(b)
    c.f(eps)
    c.b(True)
    return c.one()


def _add_(a: T, b: T) raises:
    var c = _Call("aten::add_", "Tensor")
    c.t(a)
    c.t(b)
    c.s(1.0)
    _ = c.run(1)


# --- _transform_bias_rescale_qkv ---------------------------------------------


def _inv_sqrt_dim(dim_per_head: Int, dtype: DType) -> Float64:
    """`1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head))`, rounded to the
    tensor dtype as CUDA passes it to the kernel (a `scalar_t` argument):
    the half types round `dim_per_head`, take a float sqrt, divide in double
    and round the quotient."""
    if dtype == DType.float64:
        return 1.0 / sqrt(Float64(dim_per_head))
    if dtype == DType.float16:
        var d = Float32(Float16(dim_per_head))
        return Float64(Float16(1.0 / Float64(sqrt(d))))
    if dtype == DType.bfloat16:
        var d = Float32(BFloat16(dim_per_head))
        return Float64(BFloat16(1.0 / Float64(sqrt(d))))
    return Float64(Float32(1.0 / Float64(sqrt(Float32(dim_per_head)))))


def _transform_bias_rescale(qkv: T, bias: T, num_head: Int) raises -> Owned:
    """The `(3, B, NH, T, DH)` buffer CUDA's kernel fills: q = (q + b_q) *
    1/sqrt(DH), k = k + b_k, v = v + b_v, each computed in the accumulate
    type (float, or double) and rounded once."""
    comptime W = "transform_bias_rescale_qkv"
    if qkv.rank != 3:
        raise Error("TensorAccessor expected 3 dims but tensor has ", qkv.rank)
    if bias.rank != 1:
        raise Error("TensorAccessor expected 1 dims but tensor has ", bias.rank)
    var b = qkv.dim(0)
    var t = qkv.dim(1)
    var d3 = bias.dim(0)
    var d = d3 // 3
    if num_head == 0 or d % num_head != 0:
        raise Error(
            "Expected D % num_head == 0 to be true, but got false.  (Could"
            " this error message be improved?  If so, please report an"
            " enhancement request to PyTorch.)"
        )
    if qkv.dim(2) != d3 or d3 % 3 != 0:
        raise Error(
            W,
            ": expected qkv of shape (B, T, 3 * D) with 3 * D = ",
            d3,
            ", got ",
            shape_str(qkv),
        )
    if not qkv.on_mojo() or not bias.on_mojo() or qkv.device != bias.device:
        raise Error(W, ": expected qkv and qkv_bias on the same device")
    if bias.stype != qkv.stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(qkv.dtype),
            " but found ",
            _scalar_type_name(bias.dtype),
        )
    _float_input(qkv, W)
    var dh = d // num_head
    var acc = ST_FLOAT64 if qkv.dtype == DType.float64 else ST_FLOAT32
    var out = own(_alloc(qkv.device, qkv.stype, [3, b, num_head, t, dh]))
    if out.t.numel == 0:
        return out^
    var xa = _as(qkv, acc)
    var ba = _as(bias, acc)
    var s = _d_binary("aten::add", xa.t, ba.t)
    _ = xa^
    _ = ba^
    var q_part = _view_at(s.t, [b, t, d], [t * d3, d3, 1], s.t.offset)
    _d_mul_scalar_(q_part.t, _inv_sqrt_dim(dh, qkv.dtype))
    _ = q_part^
    var sh = _as(s.t, qkv.stype)
    _ = s^
    var src = _view_at(
        sh.t, [b, t, 3, num_head, dh], [t * d3, d3, d, dh, 1], sh.t.offset
    )
    var plane = b * num_head * t * dh
    var dst = _view_at(
        out.t,
        [b, t, 3, num_head, dh],
        [num_head * t * dh, dh, plane, t * dh, 1],
        out.t.offset,
    )
    copy_strided_into(dst.t, src.t)
    _ = dst^
    _ = src^
    _ = sh^
    return out^


# aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias,
#   int num_heads) -> (Tensor, Tensor, Tensor)
def op_transform_bias_rescale_qkv(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var qkv = v_tensor(args[unsafe_offset=0])
    var bias = v_tensor(args[unsafe_offset=1])
    var nh = v_int(args[unsafe_offset=2])
    var buf = _transform_bias_rescale(qkv, bias, nh)
    var b = qkv.dim(0)
    var t = qkv.dim(1)
    var dh = (bias.dim(0) // 3) // nh
    var plane = b * nh * t * dh
    for i in range(3):
        var part = _view_at(
            buf.t,
            [b, nh, t, dh],
            [nh * t * dh, t * dh, dh, 1],
            buf.t.offset + i * plane,
        )
        ret_owned(rets, i, part)
    _ = buf^


# --- _native_multi_head_attention -------------------------------------------


def _gemm_nt(x: T, w: T) raises -> Owned:
    """`x @ w.t()` (attention.cpp's gemm_nt: at::native::matmul)."""
    var wt = _swap_last(w)
    var r = _d_matmul(x, wt.t)
    _ = wt^
    return r^


def _rows(w: T, start: Int, n: Int) raises -> Owned:
    """`w[start:start + n]` of a 2-D weight, as a view."""
    return _view_at(
        w,
        [n, w.dim(1)],
        [w.stride(0), w.stride(1)],
        w.offset + start * w.stride(0),
    )


def _qkv_projection(query: T, key: T, value: T, d: Int, w: T) raises -> Owned:
    """attention.cpp's `qkv_projection`: (B, T, 3D), without the bias."""
    if _same(key, value) and _same(query, key):
        return _gemm_nt(query, w)
    var b = query.dim(0)
    var t = query.dim(1)
    var qkv = own(_alloc(query.device, query.stype, [b, t, 3 * d]))
    var inputs = List[T]()
    inputs.append(query.copy())
    inputs.append(key.copy())
    inputs.append(value.copy())
    for i in range(3):
        # key is value: one (2D, D) GEMM against key in CUDA; the same
        # numbers as two (D, D) GEMMs, each output column is one dot.
        var wi = _rows(w, i * d, d)
        var part = _gemm_nt(inputs[i], wi.t)
        _ = wi^
        if part.t.dim(0) != b or part.t.dim(1) != t:
            raise Error("expected `query`/`key`/`value` shapes to match")
        var dst = _view_at(
            qkv.t, [b, t, d], [t * 3 * d, 3 * d, 1], qkv.t.offset + i * d
        )
        copy_strided_into(dst.t, part.t)
        _ = dst^
        _ = part^
    return qkv^


def _masked_softmax(
    scores: T, mask: T, mask_type: Optional[Int]
) raises -> Owned:
    """`_masked_softmax(scores, mask, dim=-1, mask_type)` on (B, H, T, T)
    scores: a True mask entry hides the key."""
    var m = _as(mask, ST_BOOL)
    if not mask_type:
        raise Error("Mask Type should be defined")
    var mt = mask_type.value()
    if mt != 0 and mt != 1 and mt != 2:
        raise Error(
            "Mask Type should be 0 (src_mask), 1 (src_key_padding_mask), or 2"
            " (default_mask)"
        )
    var bsz = scores.dim(0)
    var h = scores.dim(1)
    var tq = scores.dim(2)
    var tk = scores.dim(3)
    var is_bxt = (
        mt == 1
        and m.t.rank == 2
        and m.t.dim(0) == bsz
        and m.t.dim(1) == tq
        and m.t.dim(1) == tk
    )
    var is_txt = (
        mt == 0
        and m.t.rank == 2
        and m.t.dim(1) == tk
        and m.t.dim(0) == tq
        and m.t.dim(0) == m.t.dim(1)
    )
    var me: Owned
    if is_bxt:
        me = _view_at(
            m.t,
            [bsz, h, tq, tk],
            [m.t.stride(0), 0, 0, m.t.stride(1)],
            m.t.offset,
        )
    elif is_txt:
        me = _view_at(
            m.t,
            [bsz, h, tq, tk],
            [0, 0, m.t.stride(0), m.t.stride(1)],
            m.t.offset,
        )
    elif m.t.same_shape(scores):
        me = _as(m.t, ST_BOOL)
    else:
        var ms = shape_str(m.t)
        var ss = shape_str(scores)
        raise Error(
            "Mask shape should match input. mask: [",
            String(ms[byte = 1 : ms.byte_length() - 1]),
            "] input: [",
            String(ss[byte = 1 : ss.byte_length() - 1]),
            "]",
        )
    # A fully hidden row comes out NaN (softmax over -inf), as on CUDA and
    # CPU alike, and the NaN reaches that row of the output too.
    var filled = _d_masked_fill(scores, me.t, _NEG_INF)
    var p = _softmax_last(filled.t)
    _ = filled^
    _ = me^
    _ = m^
    return p^


def _check_mha(
    query: T,
    key: T,
    value: T,
    embed_dim: Int,
    num_head: Int,
    qkv_weight: T,
    qkv_bias: T,
) raises:
    """native_multi_head_attention_cuda's argument checks, verbatim."""
    var d = embed_dim
    if query.rank != 3:
        raise Error("expected 3-D `query`, got ", query.rank, "-D tensor")
    if query.dim(2) != embed_dim:
        raise Error(
            "passed-in embed_dim ",
            embed_dim,
            " didn't match last dim of query ",
            query.dim(2),
        )
    if key.rank != 3:
        raise Error("expected 3-D `key`, got ", key.rank, "-D tensor")
    if value.rank != 3:
        raise Error("expected 3-D `value`, got ", value.rank, "-D tensor")
    if not query.same_shape(key) or not key.same_shape(value):
        raise Error("expected `query`/`key`/`value` shapes to match")
    if qkv_weight.rank != 2:
        raise Error(
            "expected 2-D `qkv_weight`, got ", qkv_weight.rank, "-D tensor"
        )
    if d * 3 != qkv_weight.dim(0):
        raise Error("expected `qkv_weight` first dim to be 3x embed_dim")
    if d != qkv_weight.dim(1):
        raise Error("expected `qkv_weight` second dim to be embed_Dim")
    if qkv_bias.rank != 1:
        raise Error("expected 1-D `qkv_bias`, got ", qkv_bias.rank, "-D tensor")
    if qkv_bias.dim(0) != 3 * d:
        raise Error(
            "expected `qkv_bias` first dim and first dim of query to be equal"
        )
    if num_head <= 0 or d % num_head != 0:
        raise Error("`embed_dim` must divide evenly by `num_heads`")


def _sdpa_is_fused(query: T, num_head: Int, dh: Int) raises -> Bool:
    """CUDA's gate on its SDPA path: `select_sdp_backend` over the
    (B, NH, T, DH) head views of the input must pick a fused backend; a math
    choice keeps the explicit bmm/softmax path below."""
    var b = query.dim(0)
    var t = query.dim(1)
    var view = _view_at(
        query,
        [b, num_head, t, dh],
        [
            query.stride(0),
            dh * query.stride(2),
            query.stride(1),
            query.stride(2),
        ],
        query.offset,
    )
    var c = _Call("aten::_fused_sdp_choice", "")
    c.t(view.t)
    c.t(view.t)
    c.t(view.t)
    c.none()
    c.f(0.0)
    c.b(False)
    c.none()
    c.b(False)
    var r = c.run(1)
    _ = view^
    return Int(r[0].a) != 0  # SDPBackend::math


def _mha_fast(
    query: T,
    embed_dim: Int,
    num_head: Int,
    qkv_weight: T,
    qkv_bias: T,
    proj_weight: T,
    proj_bias: T,
) raises -> Owned:
    """CUDA's SDPA path (self-attention, no mask, no weights): one biased
    in-projection, `scaled_dot_product_attention` over its three head
    views, the out-projection."""
    var x = _linear(query, qkv_weight, qkv_bias)
    var b = x.t.dim(0)
    var t = x.t.dim(1)
    var d = embed_dim
    var dh = d // num_head
    var hq = _view_at(
        x.t, [b, num_head, t, dh], [t * 3 * d, dh, 3 * d, 1], x.t.offset
    )
    var hk = _view_at(
        x.t, [b, num_head, t, dh], [t * 3 * d, dh, 3 * d, 1], x.t.offset + d
    )
    var hv = _view_at(
        x.t,
        [b, num_head, t, dh],
        [t * 3 * d, dh, 3 * d, 1],
        x.t.offset + 2 * d,
    )
    var c = _Call("aten::scaled_dot_product_attention", "")
    c.t(hq.t)
    c.t(hk.t)
    c.t(hv.t)
    c.none()
    c.f(0.0)
    c.b(False)
    c.none()
    c.b(False)
    var y = c.one()
    _ = hq^
    _ = hk^
    _ = hv^
    _ = x^
    # y.transpose(1, 2).reshape({B, -1, embed_dim})
    var yt = _view_at(
        y.t,
        [b, t, num_head, dh],
        [y.t.stride(0), y.t.stride(2), y.t.stride(1), y.t.stride(3)],
        y.t.offset,
    )
    var past = own(_alloc(y.t.device, y.t.stype, [b, t, num_head, dh]))
    copy_strided_into(past.t, yt.t)
    _ = yt^
    _ = y^
    var past3 = _dense_view(past.t, [b, t, d])
    _ = past^
    var proj = _linear(past3.t, proj_weight, proj_bias)
    _ = past3^
    return proj^


def _native_mha(
    query: T,
    key: T,
    value: T,
    embed_dim: Int,
    num_head: Int,
    qkv_weight: T,
    qkv_bias: T,
    proj_weight: T,
    proj_bias: T,
    mask: Optional[T],
    need_weights: Bool,
    average_attn_weights: Bool,
    mask_type: Optional[Int],
    rets: Values,
) raises:
    _check_mha(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias)
    _float_input(query, "native_multi_head_attention")
    var d = embed_dim
    var dh = d // num_head
    if (
        _same(query, key)
        and _same(key, value)
        and not need_weights
        and not mask
        and query.numel > 0
        and _sdpa_is_fused(query, num_head, dh)
    ):
        var proj = _mha_fast(
            query, d, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias
        )
        ret_owned(rets, 0, proj)
        _ret_undefined(rets, 1)
        return

    var qkv = _qkv_projection(query, key, value, d, qkv_weight)
    if qkv.t.numel == 0:
        var empty = own(
            _alloc(query.device, query.stype, query.logical_shape())
        )
        ret_owned(rets, 0, empty)
        _ret_undefined(rets, 1)
        return
    var b = query.dim(0)
    var t = query.dim(1)
    var buf = _transform_bias_rescale(qkv.t, qkv_bias, num_head)
    _ = qkv^
    var plane = b * num_head * t * dh
    # q, k, v as (B * NH, T, DH) planes of the packed buffer
    var q3 = _view_at(
        buf.t, [b * num_head, t, dh], [t * dh, dh, 1], buf.t.offset
    )
    var k3 = _view_at(
        buf.t, [b * num_head, t, dh], [t * dh, dh, 1], buf.t.offset + plane
    )
    var v3 = _view_at(
        buf.t, [b * num_head, t, dh], [t * dh, dh, 1], buf.t.offset + 2 * plane
    )
    var kt = _swap_last(k3.t)
    var qkt3 = _d_matmul(q3.t, kt.t)
    _ = kt^
    var qkt = _dense_view(qkt3.t, [b, num_head, t, t])
    _ = qkt3^
    var p: Owned
    if mask:
        p = _masked_softmax(qkt.t, mask.value(), mask_type)
    else:
        p = _softmax_last(qkt.t)
    _ = qkt^
    var p3 = _dense_view(p.t, [b * num_head, t, t])
    var ctx3 = _d_matmul(p3.t, v3.t)
    _ = p3^
    _ = q3^
    _ = k3^
    _ = v3^
    _ = buf^
    # transform_0213 + linear: (B, NH, T, DH) -> (B, T, NH, DH) -> (B*T, D)
    var ctx_0213 = _view_at(
        ctx3.t,
        [b, t, num_head, dh],
        [num_head * t * dh, dh, t * dh, 1],
        ctx3.t.offset,
    )
    var a = own(_alloc(query.device, ctx3.t.stype, [b, t, num_head, dh]))
    copy_strided_into(a.t, ctx_0213.t)
    _ = ctx_0213^
    _ = ctx3^
    var a2 = _dense_view(a.t, [b * t, d])
    _ = a^
    var r = _linear(a2.t, proj_weight, proj_bias)
    _ = a2^
    var proj = _dense_view(r.t, [b, t, r.t.dim(1)])
    _ = r^
    ret_owned(rets, 0, proj)
    if not need_weights:
        _ret_undefined(rets, 1)
        return
    if average_attn_weights:
        var avg = _d_sum(p.t, [1], False)
        var c = _Call("aten::div_", "Scalar")
        c.t(avg.t)
        c.s(Float64(num_head))
        _ = c.run(1)
        ret_owned(rets, 1, avg)
    else:
        ret_owned(rets, 1, p)


def _opt_int(v: Value) raises -> Optional[Int]:
    if v_is_none(v):
        return None
    return v_int(v)


# aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value,
#   int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias,
#   Tensor proj_weight, Tensor proj_bias, Tensor? mask=None,
#   bool need_weights=True, bool average_attn_weights=True,
#   int? mask_type=None) -> (Tensor, Tensor)
def op_native_multi_head_attention(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _native_mha(
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
        v_int(args[unsafe_offset=3]),
        v_int(args[unsafe_offset=4]),
        v_tensor(args[unsafe_offset=5]),
        v_tensor(args[unsafe_offset=6]),
        v_tensor(args[unsafe_offset=7]),
        v_tensor(args[unsafe_offset=8]),
        v_opt_tensor(args[unsafe_offset=9]),
        v_bool(args[unsafe_offset=10]),
        v_bool(args[unsafe_offset=11]),
        _opt_int(args[unsafe_offset=12]),
        rets,
    )


# --- _transformer_encoder_layer_fwd ----------------------------------------


def _linear_for_ffn(bias: T, mat1: T, mat2: T, activation: Int) raises -> Owned:
    """transformer.cpp's `linear_for_ffn` on a (B, T, K) input: addmm, then
    CUDA's `_addmm_activation` epilogue -- `activation` 0 none, 1 relu,
    2 gelu with the tanh approximation (cuBLASLt's GELU epilogue, and the
    `gelu_(result, "tanh")` CUDA applies when it has no Lt path)."""
    var b = mat1.dim(0)
    var t = mat1.dim(1)
    var kdim = mat1.dim(2)
    var m2: Owned
    if mat1.contig:
        m2 = _dense_view(mat1, [b * t, kdim])
    else:
        var m1 = own(_alloc(mat1.device, mat1.stype, [b, t, kdim]))
        copy_strided_into(m1.t, mat1)
        m2 = _dense_view(m1.t, [b * t, kdim])
        _ = m1^
    var wt = _swap_last(mat2)
    var c = _Call("aten::addmm", "")
    c.t(bias)
    c.t(m2.t)
    c.t(wt.t)
    c.s(1.0)
    c.s(1.0)
    var r = c.one()
    _ = wt^
    _ = m2^
    if activation == 1:
        var a = _Call("aten::relu_", "")
        a.t(r.t)
        _ = a.run(1)
    elif activation == 2:
        var a = _Call("aten::gelu_", "")
        a.t(r.t)
        a.str("tanh")
        _ = a.run(1)
    var out = _dense_view(r.t, [b, t, r.t.dim(1)])
    _ = r^
    return out^


# aten::_transformer_encoder_layer_fwd(Tensor src, int embed_dim,
#   int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight,
#   Tensor proj_bias, bool use_gelu, bool norm_first, float eps,
#   Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2,
#   Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1,
#   Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None,
#   int? mask_type=None) -> Tensor
def op_transformer_encoder_layer_fwd(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var src = v_tensor(args[unsafe_offset=0])
    var e = v_int(args[unsafe_offset=1])
    var nh = v_int(args[unsafe_offset=2])
    var qkv_w = v_tensor(args[unsafe_offset=3])
    var qkv_b = v_tensor(args[unsafe_offset=4])
    var proj_w = v_tensor(args[unsafe_offset=5])
    var proj_b = v_tensor(args[unsafe_offset=6])
    var use_gelu = v_bool(args[unsafe_offset=7])
    var norm_first = v_bool(args[unsafe_offset=8])
    var eps = v_f64(args[unsafe_offset=9])
    var nw1 = v_tensor(args[unsafe_offset=10])
    var nb1 = v_tensor(args[unsafe_offset=11])
    var nw2 = v_tensor(args[unsafe_offset=12])
    var nb2 = v_tensor(args[unsafe_offset=13])
    var w1 = v_tensor(args[unsafe_offset=14])
    var b1 = v_tensor(args[unsafe_offset=15])
    var w2 = v_tensor(args[unsafe_offset=16])
    var b2 = v_tensor(args[unsafe_offset=17])
    var mask = v_opt_tensor(args[unsafe_offset=18])
    var mask_type = _opt_int(args[unsafe_offset=19])
    if src.numel == 0:
        var clone = own(_alloc(src.device, src.stype, src.logical_shape()))
        ret_owned(rets, 0, clone)
        return
    var x = own(T(retain(src)))
    if norm_first:
        var y1 = _layer_norm(x.t, e, nw1, nb1, eps)
        _ = x^  # alive across the call that reads it
        x = y1^
    var mrets = Array[Value, 2](fill=Value(TAG_NONE, 0, 0, 0))
    _native_mha(
        x.t,
        x.t,
        x.t,
        e,
        nh,
        qkv_w,
        qkv_b,
        proj_w,
        proj_b,
        mask,
        False,
        True,
        mask_type,
        Values(unsafe_from_address=Int(mrets.unsafe_ptr())),
    )
    x = own(T(Int(mrets[0].a)))
    _ = mrets
    _add_(x.t, src)
    if not norm_first:
        var y2 = _layer_norm(x.t, e, nw1, nb1, eps)
        _ = x^  # alive across the call that reads it
        x = y2^
    var pre_ffn = own(T(retain(x.t)))
    if norm_first:
        var y3 = _layer_norm(x.t, e, nw2, nb2, eps)
        _ = x^  # alive across the call that reads it
        x = y3^
    if x.t.rank != 3:
        raise Error("batched input size should be 3")
    if w1.rank != 2 or w2.rank != 2:
        raise Error("2d weights expected")
    var h = _linear_for_ffn(b1, x.t, w1, 2 if use_gelu else 1)
    x = _linear_for_ffn(b2, h.t, w2, 0)
    _ = h^
    _add_(x.t, pre_ffn.t)
    _ = pre_ffn^
    if not norm_first:
        var y4 = _layer_norm(x.t, e, nw2, nb2, eps)
        _ = x^  # alive across the call that reads it
        x = y4^
    ret_owned(rets, 0, x)


def register_transformer(site: Site) raises:
    impl[op_transform_bias_rescale_qkv, "_transform_bias_rescale_qkv"](site)
    impl[op_native_multi_head_attention, "_native_multi_head_attention"](site)
    impl[op_transformer_encoder_layer_fwd, "_transformer_encoder_layer_fwd"](
        site
    )
