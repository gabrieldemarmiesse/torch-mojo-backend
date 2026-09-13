"""Ops with no kernel of their own composed from registered ops through the
dispatcher (`call_op`): backward formulas ATen only ships as per-backend
kernels (threshold / sigmoid / tanh / batch norm / softmax backward), the
signed-infinity tests, and out= overloads of ops whose functional form
exists. Each costs a few extra launches; a fused kernel can replace any of
them later without changing the registration."""
from std.utils import IndexList

from abi import (
    Owned,
    ST_BOOL,
    ST_FLOAT32,
    T,
    TAG_BOOL,
    TAG_BOOL_LIST,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    Value,
    Values,
    contiguous_strides,
    f64_bits,
    new_like,
    new_tensor,
    own,
    release,
    retain,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_int,
    v_is_none,
    v_tensor,
    view_strided,
)
from op_utils import MAX_RANK
from ops_common import (
    call_op,
    cast_to,
    copy_strided_into,
    fill_value,
    resize_out,
)
from registry import Site, impl, op_address_of


comptime NEG_INF_BITS: Int64 = -4503599627370496  # 0xFFF0000000000000
comptime POS_INF_BITS: Int64 = 9218868437227405312  # 0x7FF0000000000000


def _tensor_value(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _dispatch(
    name: StaticString, overload: StaticString, var args: List[Value]
) raises -> T:
    """One aten op through the dispatcher, one Tensor result (owned)."""
    var rets = call_op(String(name), String(overload), args^, 1)
    return rets.take_tensor(0)


def _dispatch_into(
    name: StaticString, overload: StaticString, var args: List[Value], target: T
) raises:
    """An out= overload through the dispatcher; its result handle is a fresh
    reference to `out`, which `Results` releases."""
    _ = call_op(String(name), String(overload), args^, 1)
    _ = target


def _one_minus(x: T) raises -> T:
    """1 - x as neg(x - 1): sub.Scalar then neg, both registered."""
    var xm1 = _dispatch(
        "aten::sub",
        "Scalar",
        [
            _tensor_value(x),
            Value(TAG_SCALAR_INT, 0, 1, 0),
            Value(TAG_SCALAR_INT, 0, 1, 0),
        ],
    )
    var r = _dispatch("aten::neg", "", [_tensor_value(xm1)])
    release(xm1.h)
    return r^


# aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
def _threshold_backward_mask(args: Values) raises -> T:
    return _dispatch(
        "aten::gt",
        "Scalar",
        [
            _tensor_value(v_tensor(args[unsafe_offset=1])),
            args[unsafe_offset=2].copy(),
        ],
    )


def op_threshold_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var mask = _threshold_backward_mask(args)
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(mask)]
        )
    )
    release(mask.h)
    ret_owned(rets, 0, g)


# aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input)
def op_threshold_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    var mask = _threshold_backward_mask(args)
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(mask), _tensor_value(out)],
        out,
    )
    release(mask.h)
    ret_ref(rets, 0, out)


# aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor: grad * out * (1 - out)
def _sigmoid_backward_factor(output: T) raises -> T:
    var om = _one_minus(output)
    var f = _dispatch(
        "aten::mul", "Tensor", [_tensor_value(output), _tensor_value(om)]
    )
    release(om.h)
    return f^


def op_sigmoid_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var f = _sigmoid_backward_factor(v_tensor(args[unsafe_offset=1]))
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(f)]
        )
    )
    release(f.h)
    ret_owned(rets, 0, g)


# aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input)
def op_sigmoid_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    var f = _sigmoid_backward_factor(v_tensor(args[unsafe_offset=1]))
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(f), _tensor_value(out)],
        out,
    )
    release(f.h)
    ret_ref(rets, 0, out)


# aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor: grad * (1 - out^2)
def _tanh_backward_factor(output: T) raises -> T:
    var sq = _dispatch(
        "aten::mul", "Tensor", [_tensor_value(output), _tensor_value(output)]
    )
    var f = _one_minus(sq)
    release(sq.h)
    return f^


def op_tanh_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var f = _tanh_backward_factor(v_tensor(args[unsafe_offset=1]))
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(f)]
        )
    )
    release(f.h)
    ret_owned(rets, 0, g)


# aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input)
def op_tanh_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    var f = _tanh_backward_factor(v_tensor(args[unsafe_offset=1]))
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(f), _tensor_value(out)],
        out,
    )
    release(f.h)
    ret_ref(rets, 0, out)


# aten::isneginf(Tensor self) -> Tensor / aten::isposinf(Tensor self) -> Tensor (+ .out):
# eq.Scalar against the signed infinity (an integer tensor is never infinite:
# eq against a double that no integer equals is all-false, like ATen).
def _is_inf(args: Values, rets: Values, bits: Int64, with_out: Bool) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var scalar = Value(TAG_SCALAR_DOUBLE, 0, bits, 0)
    if with_out:
        var out = v_tensor(args[unsafe_offset=1])
        if not x.dtype.is_floating_point():
            resize_out(out, x.shape, x.rank)
            fill_value(out, 0.0)
        else:
            _dispatch_into(
                "aten::eq",
                "Scalar_out",
                [_tensor_value(x), scalar.copy(), _tensor_value(out)],
                out,
            )
        ret_ref(rets, 0, out)
        return
    if not x.dtype.is_floating_point():
        var zeros = own(new_tensor(x.shape, x.rank, ST_BOOL, x.device))
        fill_value(zeros.t, 0.0)
        ret_owned(rets, 0, zeros)
        return
    var r = own(
        _dispatch("aten::eq", "Scalar", [_tensor_value(x), scalar.copy()])
    )
    ret_owned(rets, 0, r)


def op_isneginf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _is_inf(args, rets, NEG_INF_BITS, False)


def op_isneginf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _is_inf(args, rets, NEG_INF_BITS, True)


def op_isposinf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _is_inf(args, rets, POS_INF_BITS, False)


def op_isposinf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _is_inf(args, rets, POS_INF_BITS, True)


# ---------------------------------------------------------------------------
# Shared plumbing for the composed formulas below.
#
# Every intermediate is an `Owned`, and every helper takes its operands as
# `Owned` *parameters* rather than as the `T` behind one. That is not a
# stylistic choice: Mojo destroys a local right after its last use, and
# reading `x.t` copies the view out, ending the borrow -- so `f(x.t)` can
# release the tensor before `f` runs and hand the dispatcher a freed
# `at::Tensor*` (measured: a garbage key set, reported as "could not run
# aten::rsqrt with arguments from the TESTING_ONLY_GenericWrapper backend").
# Borrowing the `Owned` keeps it alive for the whole call, and the wrapper
# still releases on every raising path.
# ---------------------------------------------------------------------------


def _hold(t: T) raises -> Owned:
    """A second owned handle to `t` (the caller's stays untouched), so a
    value that is sometimes a borrowed input and sometimes a fresh
    allocation can be treated uniformly."""
    return own(T(retain(t)))


def _as_dtype(t: T, stype: Int32) raises -> Owned:
    """`t` in `stype`, always as a handle the caller owns: `cast_to` hands
    back the input itself when the dtype already matches, and that one must
    not be released as if it were a fresh allocation."""
    if t.stype == stype:
        return _hold(t)
    return own(cast_to(t, stype))


def _cast(t: Owned, stype: Int32) raises -> Owned:
    return _as_dtype(t.t, stype)


def _dense(t: Owned) raises -> Owned:
    if t.t.contig:
        return _hold(t.t)
    var out = own(new_like(t.t))
    copy_strided_into(out.t, t.t)
    return out^


def _copy_into(dst: T, src: Owned) raises:
    copy_strided_into(dst, src.t)


def _view_as(
    base: Owned, shape: IndexList[MAX_RANK], rank: Int
) raises -> Owned:
    return own(
        view_strided(
            base.t,
            shape,
            contiguous_strides(shape, rank),
            rank,
            base.t.offset,
        )
    )


def _mul(a: Owned, b: Owned) raises -> Owned:
    return own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(a.t), _tensor_value(b.t)]
        )
    )


def _sub(a: Owned, b: Owned) raises -> Owned:
    return own(
        _dispatch(
            "aten::sub",
            "Tensor",
            [
                _tensor_value(a.t),
                _tensor_value(b.t),
                Value(TAG_SCALAR_INT, 0, 1, 0),
            ],
        )
    )


def _add_scalar(a: Owned, v: Float64) raises -> Owned:
    return own(
        _dispatch(
            "aten::add",
            "Scalar",
            [
                _tensor_value(a.t),
                Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v), 0),
                Value(TAG_SCALAR_INT, 0, 1, 0),
            ],
        )
    )


def _div_scalar(a: Owned, v: Float64) raises -> Owned:
    return own(
        _dispatch(
            "aten::div",
            "Scalar",
            [
                _tensor_value(a.t),
                Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v), 0),
            ],
        )
    )


def _rsqrt(a: Owned) raises -> Owned:
    return own(_dispatch("aten::rsqrt", "", [_tensor_value(a.t)]))


def _sum_dims(x: Owned, dims: List[Int64], keepdim: Bool) raises -> Owned:
    return own(
        _dispatch(
            "aten::sum",
            "dim_IntList",
            [
                _tensor_value(x.t),
                Value(
                    TAG_INT_LIST,
                    Int32(len(dims)),
                    Int64(Int(dims.unsafe_ptr())),
                    0,
                ),
                Value(TAG_BOOL, 0, Int64(1) if keepdim else Int64(0), 0),
                Value(TAG_NONE, 0, 0, 0),
            ],
        )
    )


def _bool_list(v: Value) raises -> List[Bool]:
    """A borrowed `bool[]` argument (uint8 per element in the call arena)."""
    var out = List[Bool]()
    if v.tag == TAG_NONE:
        return out^
    if v.tag != TAG_BOOL_LIST:
        raise Error("expected a bool[] argument, got record tag ", v.tag)
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def _empty_result(stype: Int32, device: Int) raises -> Owned:
    """The 0-element stand-in this ABI returns for a gradient autograd did
    not ask for; the same convention `native_layer_norm_backward` uses (there
    is no way to build an undefined `at::Tensor` from Mojo, and the engine
    never reads a masked-off result)."""
    return own(new_tensor(IndexList[MAX_RANK](0), 1, stype, device))


# ---------------------------------------------------------------------------
# where.self_out
# ---------------------------------------------------------------------------


# aten::where.self_out(Tensor condition, Tensor self, Tensor other, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_where_self_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """`where.self` into the caller's tensor. The promoted dtype is read off
    the functional result rather than recomputed, so the two overloads can
    never disagree about it; ATen requires `out` to be exactly that dtype."""
    var out = v_tensor(args[unsafe_offset=3])
    var r = own(
        _dispatch(
            "aten::where",
            "self",
            [
                args[unsafe_offset=0].copy(),
                args[unsafe_offset=1].copy(),
                args[unsafe_offset=2].copy(),
            ],
        )
    )
    if r.t.stype != out.stype:
        raise Error(
            "Expected out type to be ",
            String(r.t.dtype),
            " but got ",
            String(out.dtype),
        )
    if not out.same_shape(r.t):
        resize_out(out, r.t.shape, r.t.rank)
    _copy_into(out, r)
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# native_batch_norm_backward
#
# ATen's formula (Normalization.cpp `batch_norm_backward_cpu_template`, and
# the CUDA kernels it mirrors), over reduce dims = every dim but 1, with
# N = numel / C:
#
#   mean, invstd = save_mean, save_invstd                     (training)
#                = running_mean, rsqrt(running_var + eps)     (evaluation)
#   xhat         = (input - mean) * invstd
#   grad_bias    = sum(grad_out)
#   grad_weight  = sum(grad_out * xhat)
#   grad_input   = invstd * w * (grad_out - grad_bias/N - xhat * grad_weight/N)
#                = invstd * w * grad_out                      (evaluation)
#
# with w = weight or 1. Half / bfloat16 inputs run the whole formula in
# float32 (ATen's opmath_t) and cast back; the two affine gradients come out
# in the weight's dtype, like `at::empty_like(weight)` in the CUDA kernel.
# ---------------------------------------------------------------------------


def _channel_view(vec: Owned, rank: Int) raises -> Owned:
    """A rank-`rank` `[1, C, 1, ...]` view of a per-channel vector, which is
    what lines it up with dim 1 of the input for the broadcast binary
    kernels. Zero-copy, so a non-contiguous vector needs no materializing:
    only the C axis carries a real stride."""
    if vec.t.rank != 1:
        unsupported(
            "native_batch_norm_backward: a per-channel parameter must be 1-D"
        )
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](1)
    shape[MAX_RANK - rank + 1] = vec.t.dim(0)
    strides[MAX_RANK - rank + 1] = vec.t.stride(0)
    return own(view_strided(vec.t, shape, strides, rank, vec.t.offset))


def _planes_view(t: T) raises -> Owned:
    """`t` as a contiguous rank-3 `[N, C, HxW]` tensor, the shape ATen itself
    reduces batch norm to. Only used above rank 4, where the broadcast binary
    kernels stop; the copy is free for an already contiguous input."""
    var c = _dense(_hold(t))
    var inner = 1
    for i in range(2, c.t.rank):
        inner *= c.t.dim(i)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 3] = c.t.dim(0)
    shape[MAX_RANK - 2] = c.t.dim(1)
    shape[MAX_RANK - 1] = inner
    return _view_as(c, shape, 3)


def _bn_operand(t: T, rank: Int) raises -> Owned:
    """The operand the formula runs on: `t` itself up to rank 4, its
    `[N, C, HxW]` view above -- batch norm only ever looks at dim 1, and the
    broadcast binary kernels stop at rank 4."""
    if rank <= 4:
        return _hold(t)
    return _planes_view(t)


def _shaped_like(gi: Owned, out_like: T) raises -> Owned:
    """`gi` in the dtype and shape of `out_like`. The shape only ever differs
    when rank > 4 sent the formula through the `[N, C, HxW]` view."""
    var res = _cast(gi, out_like.stype)
    if res.t.rank == out_like.rank:
        return res^
    var dense = _dense(res)
    return _view_as(dense, out_like.shape, out_like.rank)


def _bn_scale(args: Values, invstd: Owned, cst: Int32) raises -> Owned:
    """`invstd * weight` per channel (`weight = 1` when there is none): the
    common factor of every grad_input term."""
    if v_is_none(args[unsafe_offset=2]):
        return _hold(invstd.t)
    var w = _as_dtype(v_tensor(args[unsafe_offset=2]), cst)
    return _mul(invstd, w)


def _bn_grad_input(
    args: Values,
    gc: Owned,
    xhat: Owned,
    grad_bias: Owned,
    grad_weight: Owned,
    invstd: Owned,
    out_like: T,
    train: Bool,
    wanted: Bool,
    n: Int,
    cst: Int32,
) raises -> Owned:
    """grad_input, in the input's dtype and shape (`out_like`'s)."""
    if not wanted:
        return _empty_result(out_like.stype, out_like.device)
    var rank = gc.t.rank
    var factor = _bn_scale(args, invstd, cst)
    var factor_b = _channel_view(factor, rank)
    if not train:
        var scaled = _mul(gc, factor_b)
        return _shaped_like(scaled, out_like)
    var gb = _div_scalar(grad_bias, Float64(n))
    var gb_b = _channel_view(gb, rank)
    var gw = _div_scalar(grad_weight, Float64(n))
    var gw_b = _channel_view(gw, rank)
    var debiased = _sub(gc, gb_b)
    var projected = _mul(xhat, gw_b)
    var inner = _sub(debiased, projected)
    var gi = _mul(inner, factor_b)
    return _shaped_like(gi, out_like)


def _bn_affine_grad(v: Owned, stype: Int32, wanted: Bool) raises -> Owned:
    if not wanted:
        return _empty_result(stype, v.t.device)
    return _cast(v, stype)


def _bn_backward(
    args: Values,
    rets: Values,
    grad: Owned,
    a: Owned,
    out_like: T,
    mean: Owned,
    invstd: Owned,
    train: Bool,
    cst: Int32,
    pst: Int32,
    mask: List[Bool],
) raises:
    """The formula itself, over operands already reduced to a rank the
    broadcast binary kernels handle. `mean` and `invstd` are per-channel
    vectors in the compute dtype `cst`; the affine gradients come back in
    `pst`, grad_input in `out_like`'s dtype and shape."""
    var rank = a.t.rank
    var n = a.t.numel // a.t.dim(1)
    var dims = List[Int64](capacity=rank - 1)
    dims.append(0)
    for i in range(2, rank):
        dims.append(Int64(i))

    var gc = _cast(grad, cst)
    var ac = _cast(a, cst)
    var mean_b = _channel_view(mean, rank)
    var invstd_b = _channel_view(invstd, rank)
    var centered = _sub(ac, mean_b)
    var xhat = _mul(centered, invstd_b)
    var grad_bias = _sum_dims(gc, dims, False)
    var gxhat = _mul(gc, xhat)
    var grad_weight = _sum_dims(gxhat, dims, False)

    var gi = _bn_grad_input(
        args,
        gc,
        xhat,
        grad_bias,
        grad_weight,
        invstd,
        out_like,
        train,
        mask[0],
        n,
        cst,
    )
    var gw = _bn_affine_grad(grad_weight, pst, mask[1])
    var gb = _bn_affine_grad(grad_bias, pst, mask[2])
    # an output autograd did not ask for is an undefined Tensor (None record)
    if mask[0]:
        ret_owned(rets, 0, gi)
    else:
        rets[unsafe_offset=0] = Value(TAG_NONE, 0, 0, 0)
    if mask[1]:
        ret_owned(rets, 1, gw)
    else:
        rets[unsafe_offset=1] = Value(TAG_NONE, 0, 0, 0)
    if mask[2]:
        ret_owned(rets, 2, gb)
    else:
        rets[unsafe_offset=2] = Value(TAG_NONE, 0, 0, 0)


def _bn_stats_then_backward(
    args: Values,
    rets: Values,
    grad: Owned,
    a: Owned,
    out_like: T,
    train: Bool,
    eps: Float64,
    cst: Int32,
    pst: Int32,
    mask: List[Bool],
) raises:
    if train:
        if v_is_none(args[unsafe_offset=5]) or v_is_none(args[unsafe_offset=6]):
            unsupported(
                "training native_batch_norm_backward needs both saved"
                " statistics"
            )
        var mean = _as_dtype(v_tensor(args[unsafe_offset=5]), cst)
        var invstd = _as_dtype(v_tensor(args[unsafe_offset=6]), cst)
        _bn_backward(
            args, rets, grad, a, out_like, mean, invstd, True, cst, pst, mask
        )
        return
    if v_is_none(args[unsafe_offset=3]) or v_is_none(args[unsafe_offset=4]):
        unsupported(
            "evaluation native_batch_norm_backward needs both running"
            " statistics"
        )
    var mean = _as_dtype(v_tensor(args[unsafe_offset=3]), cst)
    var var_t = _as_dtype(v_tensor(args[unsafe_offset=4]), cst)
    var shifted = _add_scalar(var_t, eps)
    var invstd = _rsqrt(shifted)
    _bn_backward(
        args, rets, grad, a, out_like, mean, invstd, False, cst, pst, mask
    )


def _bn_zero_grad(
    shape: IndexList[MAX_RANK], stype: Int32, device: Int, wanted: Bool
) raises -> Owned:
    if not wanted:
        return _empty_result(stype, device)
    var t = own(new_tensor(shape, 1, stype, device))
    fill_value(t.t, 0.0)
    return t^


# aten::native_batch_norm_backward(Tensor grad_out, Tensor input,
#   Tensor? weight, Tensor? running_mean, Tensor? running_var,
#   Tensor? save_mean, Tensor? save_invstd, bool train, float eps,
#   bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def op_native_batch_norm_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var train = v_bool(args[unsafe_offset=7])
    var eps = v_f64(args[unsafe_offset=8])
    var mask = _bool_list(args[unsafe_offset=9])
    if len(mask) != 3:
        raise Error(
            "native_batch_norm_backward: output_mask must have 3 entries"
        )
    if not a.on_mojo() or not grad.on_mojo() or grad.device != a.device:
        raise Error(
            "native_batch_norm_backward: every operand must be on one mojo"
            " device"
        )
    if not a.dtype.is_floating_point():
        unsupported("native_batch_norm_backward of dtype " + String(a.dtype))
    if a.rank < 2:
        unsupported("native_batch_norm_backward needs a tensor of rank >= 2")
    if not grad.same_shape(a):
        unsupported(
            "native_batch_norm_backward: grad_out and input must share a shape"
        )
    # ATen's opmath_t: the whole formula runs in float32 for a half input.
    var cst = a.stype
    if a.dtype == DType.float16 or a.dtype == DType.bfloat16:
        cst = ST_FLOAT32
    var pst = cst
    if not v_is_none(args[unsafe_offset=2]):
        pst = v_tensor(args[unsafe_offset=2]).stype
    if a.numel == 0:
        # Nothing to reduce: ATen's two affine gradients are zero over an
        # empty extent and grad_input is empty like the input.
        var cshape = IndexList[MAX_RANK](1)
        cshape[MAX_RANK - 1] = a.dim(1)
        var gi = own(new_like(a))
        var gw = _bn_zero_grad(cshape, pst, a.device, mask[1])
        var gb = _bn_zero_grad(cshape, pst, a.device, mask[2])
        ret_owned(rets, 0, gi)
        ret_owned(rets, 1, gw)
        ret_owned(rets, 2, gb)
        return
    var grad_w = _bn_operand(grad, a.rank)
    var a_w = _bn_operand(a, a.rank)
    _bn_stats_then_backward(
        args, rets, grad_w, a_w, a, train, eps, cst, pst, mask
    )


# ---------------------------------------------------------------------------
# _softmax_backward_data
# ---------------------------------------------------------------------------


# aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim,
#   ScalarType input_dtype) -> Tensor
def op_softmax_backward_data(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """`out * (grad - sum(grad * out, dim, keepdim=True))`, ATen's formula
    (SoftMax.cpp `softmax_backward_cpu_out`)."""
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=1])
    var dim = v_int(args[unsafe_offset=2])
    var target = v_dtype_or(args[unsafe_offset=3], grad.stype)
    if not grad.on_mojo() or not out.on_mojo() or grad.device != out.device:
        raise Error(
            "_softmax_backward_data: both operands must be on one mojo device"
        )
    if not out.dtype.is_floating_point():
        unsupported("_softmax_backward_data of dtype " + String(out.dtype))
    if not grad.same_shape(out) or grad.stype != out.stype:
        unsupported(
            "_softmax_backward_data: grad_output and output must match in"
            " shape and dtype"
        )
    if target != grad.stype:
        # `half_to_float`: the forward's input was half and its output float,
        # so the gradient would have to come back narrowed. `_softmax` itself
        # declines that route, so nothing here can produce it.
        unsupported(
            "_softmax_backward_data with an input_dtype different from the"
            " gradient's"
        )
    var rank = out.rank
    if rank == 0:
        unsupported("_softmax_backward_data of a 0-d tensor")
    if dim < -rank or dim >= rank:
        unsupported("_softmax_backward_data dim out of range")
    var dims = List[Int64](capacity=1)
    dims.append(Int64(dim + rank if dim < 0 else dim))
    var g = _hold(grad)
    var o = _hold(out)
    var prod = _mul(g, o)
    var total = _sum_dims(prod, dims, True)
    var diff = _sub(g, total)
    var gi = _mul(o, diff)
    ret_owned(rets, 0, gi)


def register_composed(site: Site) raises:
    impl[op_threshold_backward, "threshold_backward"](site)
    impl[op_threshold_backward_grad_input, "threshold_backward.grad_input"](
        site
    )
    impl[op_sigmoid_backward, "sigmoid_backward"](site)
    impl[op_sigmoid_backward_grad_input, "sigmoid_backward.grad_input"](site)
    impl[op_tanh_backward, "tanh_backward"](site)
    impl[op_tanh_backward_grad_input, "tanh_backward.grad_input"](site)
    impl[op_isneginf, "isneginf"](site)
    impl[op_isneginf_out, "isneginf.out"](site)
    impl[op_isposinf, "isposinf"](site)
    impl[op_isposinf_out, "isposinf.out"](site)
    impl[op_where_self_out, "where.self_out"](site)
    impl[op_native_batch_norm_backward, "native_batch_norm_backward"](site)
    impl[op_softmax_backward_data, "_softmax_backward_data"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_composed]()
