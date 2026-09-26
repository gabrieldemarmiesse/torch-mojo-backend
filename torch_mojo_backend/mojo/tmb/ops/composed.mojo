"""Ops composed from registered ops through the dispatcher (`call_op`), with
fused routes for supported regimes: backward formulas ATen ships as per-backend
kernels (threshold / sigmoid / tanh / batch norm / group norm / softmax
backward) and out= overloads of ops whose
functional form exists. Contiguous FP32 tanh backward has a fused Hopper
route; other regimes retain the composed implementation without changing the
registration."""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    ST_FLOAT32,
    T,
    TAG_BOOL,
    TAG_BOOL_LIST,
    TAG_DOUBLE,
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
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.ops.matmul import _sm90_cuda
from tmb.ops.common import (
    call_op,
    cast_to,
    copy_strided_into,
    fill_value,
    resize_out,
)
from tmb.backend.registry import Site, impl


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


def _tanh_fused_inputs(grad: T, output: T) raises -> Bool:
    return (
        grad.on_mojo()
        and output.on_mojo()
        and grad.device == output.device
        and grad.dtype == DType.float32
        and output.dtype == DType.float32
        and grad.contig
        and output.contig
        and grad.same_shape(output)
        and _sm90_cuda(grad.device)
    )


def _tanh_safe_output(dst: T, input: T) -> Bool:
    # Called only for equal-shape contiguous FP32 tensors. Exact aliases are
    # elementwise-safe; partial overlap keeps the existing temporary route.
    return (
        dst.numel == 0
        or dst.ptr == input.ptr
        or dst.ptr + dst.numel * dst.itemsize <= input.ptr
        or input.ptr + input.numel * input.itemsize <= dst.ptr
    )


def _tanh_fused_into(dst: T, grad: T, output: T) raises:
    if dst.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var call = KernelCall("activation_backward", "TanhBackwardF32")
    call.int(dst.ptr)
    call.int(grad.ptr)
    call.int(output.ptr)
    call.int(dst.numel)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


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
    var output = v_tensor(args[unsafe_offset=1])
    if _tanh_fused_inputs(grad, output):
        var result = own(new_like(grad))
        _tanh_fused_into(result.t, grad, output)
        ret_owned(rets, 0, result)
        return
    var f = _tanh_backward_factor(output)
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
    var output = v_tensor(args[unsafe_offset=1])
    if (
        _tanh_fused_inputs(grad, output)
        and out.on_mojo()
        and out.device == grad.device
        and out.dtype == DType.float32
        and out.contig
        and out.same_shape(grad)
        and _tanh_safe_output(out, grad)
        and _tanh_safe_output(out, output)
    ):
        _tanh_fused_into(out, grad, output)
        ret_ref(rets, 0, out)
        return
    var f = _tanh_backward_factor(output)
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(f), _tensor_value(out)],
        out,
    )
    release(f.h)
    ret_ref(rets, 0, out)


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
# with w = weight or 1, all of it on the `[N, C, HxW]` view. Every
# per-channel broadcast is `(x - a[c]) * b[c]`, which is one pass of the
# eval-mode batch norm kernel (`_channel_affine`); every per-channel sum is a
# spatial then a batch reduction (`_channel_sum`). Half / bfloat16 inputs run
# the whole formula in float32 (ATen's opmath_t) and cast back; the two
# affine gradients come out in the weight's dtype, like
# `at::empty_like(weight)` in the CUDA kernel.
# ---------------------------------------------------------------------------


def _planes_view(t: T) raises -> Owned:
    """`t` as a contiguous rank-3 `[N, C, HxW]` tensor, the shape ATen itself
    reduces batch norm to (batch norm only ever looks at dim 1). The copy is
    free for an already contiguous input."""
    var c = _dense(_hold(t))
    var inner = 1
    for i in range(2, c.t.rank):
        inner *= c.t.dim(i)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 3] = c.t.dim(0)
    shape[MAX_RANK - 2] = c.t.dim(1)
    shape[MAX_RANK - 1] = inner
    return _view_as(c, shape, 3)


def _shaped_like(gi: Owned, out_like: T) raises -> Owned:
    """`gi` (an `[N, C, HxW]` result) in the dtype and shape of `out_like`."""
    var res = _cast(gi, out_like.stype)
    if res.t.rank == out_like.rank:
        return res^
    var dense = _dense(res)
    return _view_as(dense, out_like.shape, out_like.rank)


def _filled_channels(c: Int, value: Float64, device: Int) raises -> Owned:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = c
    var t = own(new_tensor(shape, 1, ST_FLOAT32, device))
    fill_value(t.t, value)
    return t^


def _channel_sum(x: Owned) raises -> Owned:
    """Per-channel sum of an `[N, C, HxW]` operand, as `[C]`: the spatial axis
    first, then the batch axis. Each is a plain reduction shape, where one sum
    over both axes has to reorder the whole tensor first."""
    var spatial = List[Int64](capacity=1)
    spatial.append(2)
    var batch = List[Int64](capacity=1)
    batch.append(0)
    var partial = _sum_dims(x, spatial, False)
    return _sum_dims(partial, batch, False)


def _channel_affine(
    x: Owned, shift: Owned, scale: Owned, zeros: Owned, ones: Owned
) raises -> Owned:
    """`(x - shift[c]) * scale[c]` over dim 1 of a float32 `[N, C, HxW]`
    operand, in one elementwise pass: that is eval-mode batch norm with unit
    variance, zero bias and `eps = 0` (`1 / sqrt(1 + 0)` is exactly 1), whose
    kernel loads the per-channel coefficients once per plane instead of
    through a broadcast-strided binary. Every vector is a contiguous float32
    `[C]`."""
    var rets = call_op(
        "aten::_native_batch_norm_legit_no_training",
        "",
        [
            _tensor_value(x.t),
            _tensor_value(scale.t),
            _tensor_value(zeros.t),
            _tensor_value(shift.t),
            _tensor_value(ones.t),
            Value(TAG_DOUBLE, 0, f64_bits(0.0), 0),
            Value(TAG_DOUBLE, 0, f64_bits(0.0), 0),
        ],
        3,
    )
    return own(rets.take_tensor(0))


def _bn_scale(args: Values, invstd: Owned, cst: Int32) raises -> Owned:
    """`invstd * weight` per channel (`weight = 1` when there is none): the
    common factor of every grad_input term."""
    if v_is_none(args[unsafe_offset=2]):
        return _hold(invstd.t)
    var w = _as_dtype(v_tensor(args[unsafe_offset=2]), cst)
    return _dense(_mul(invstd, w))


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
    """The formula itself, over `[N, C, HxW]` operands already in the compute
    dtype `cst` (float32). `mean` and `invstd` are contiguous `[C]` vectors in
    `cst`; the affine gradients come back in `pst`, grad_input in
    `out_like`'s dtype and shape. Only what the mask needs is computed: an
    evaluation grad_input or a bias-only call never forms `xhat`."""
    var c = a.t.dim(1)
    var n = a.t.numel // c
    var device = a.t.device
    var zeros = _filled_channels(c, 0.0, device)
    var ones = _filled_channels(c, 1.0, device)
    var need_bias = mask[2] or (mask[0] and train)
    var need_weight = mask[1] or (mask[0] and train)
    var grad_bias = _empty_result(cst, device)
    if need_bias:
        grad_bias = _channel_sum(grad)
    var xhat = _empty_result(cst, device)
    var grad_weight = _empty_result(cst, device)
    if need_weight:
        xhat = _channel_affine(a, mean, invstd, zeros, ones)
        var gxhat = _mul(grad, xhat)
        grad_weight = _channel_sum(gxhat)

    # an output autograd did not ask for is an undefined Tensor (None record)
    if mask[0]:
        var factor = _bn_scale(args, invstd, cst)
        if train:
            # invstd * w * (grad - grad_bias/N - xhat * grad_weight/N)
            var gb_n = _div_scalar(grad_bias, Float64(n))
            var debiased = _channel_affine(grad, gb_n, factor, zeros, ones)
            var gw_n = _div_scalar(grad_weight, Float64(n))
            var coef = _mul(gw_n, factor)
            var projected = _channel_affine(xhat, zeros, coef, zeros, ones)
            var gi = _sub(debiased, projected)
            var shaped = _shaped_like(gi, out_like)
            ret_owned(rets, 0, shaped)
        else:
            var gi = _channel_affine(grad, zeros, factor, zeros, ones)
            var shaped = _shaped_like(gi, out_like)
            ret_owned(rets, 0, shaped)
    else:
        rets[unsafe_offset=0] = Value(TAG_NONE, 0, 0, 0)
    if mask[1]:
        var gw = _bn_affine_grad(grad_weight, pst, True)
        ret_owned(rets, 1, gw)
    else:
        rets[unsafe_offset=1] = Value(TAG_NONE, 0, 0, 0)
    if mask[2]:
        var gb = _bn_affine_grad(grad_bias, pst, True)
        ret_owned(rets, 2, gb)
    else:
        rets[unsafe_offset=2] = Value(TAG_NONE, 0, 0, 0)


def _bn_channel_vec(v: Value, c: Int, cst: Int32) raises -> Owned:
    """A per-channel argument as the contiguous `[C]` vector in `cst` that
    `_channel_affine` reads."""
    var t = v_tensor(v)
    if t.rank != 1 or t.dim(0) != c:
        unsupported(
            "native_batch_norm_backward: a per-channel parameter must be 1-D"
            " of size C"
        )
    return _dense(_as_dtype(t, cst))


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
    var c = a.t.dim(1)
    if train:
        if v_is_none(args[unsafe_offset=5]) or v_is_none(args[unsafe_offset=6]):
            unsupported(
                "training native_batch_norm_backward needs both saved"
                " statistics"
            )
        var mean = _bn_channel_vec(args[unsafe_offset=5], c, cst)
        var invstd = _bn_channel_vec(args[unsafe_offset=6], c, cst)
        _bn_backward(
            args, rets, grad, a, out_like, mean, invstd, True, cst, pst, mask
        )
        return
    if v_is_none(args[unsafe_offset=3]) or v_is_none(args[unsafe_offset=4]):
        unsupported(
            "evaluation native_batch_norm_backward needs both running"
            " statistics"
        )
    var mean = _bn_channel_vec(args[unsafe_offset=3], c, cst)
    var var_t = _bn_channel_vec(args[unsafe_offset=4], c, cst)
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
    if (
        a.dtype != DType.float32
        and a.dtype != DType.bfloat16
        and a.dtype != DType.float16
    ):
        unsupported("native_batch_norm_backward of dtype " + String(a.dtype))
    if a.rank < 2:
        unsupported("native_batch_norm_backward needs a tensor of rank >= 2")
    if not grad.same_shape(a):
        unsupported(
            "native_batch_norm_backward: grad_out and input must share a shape"
        )
    # ATen's opmath_t: the whole formula runs in float32 for a half input.
    var cst = ST_FLOAT32
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
    if not v_is_none(args[unsafe_offset=2]):
        var w = v_tensor(args[unsafe_offset=2])
        if w.rank != 1 or w.dim(0) != a.dim(1):
            unsupported("native_batch_norm_backward: weight must be 1-D of C")
    var grad_w = _cast(_planes_view(grad), cst)
    var a_w = _cast(_planes_view(a), cst)
    _bn_stats_then_backward(
        args, rets, grad_w, a_w, a, train, eps, cst, pst, mask
    )


# ---------------------------------------------------------------------------
# native_group_norm_backward
#
# Group norm normalizes the trailing K = (C / group) * HxW elements of the
# [N, group, K] view, so its grad_input is the layer-norm formula on that view
# once the per-channel gamma is folded into grad_out:
#
#   q  = grad_out * w                                  (w = weight or 1)
#   dx = rstd * (q - mean_K(q) - xhat * mean_K(q * xhat))
#
# which is exactly what the LayerNorm backward dx kernel computes over
# rows = N * group, cols = K. The affine gradients reduce over the batch and
# the spatial extent per channel instead, through the per-(sample, channel)
# partial sums ATen's own kernel uses (group_norm_kernel.cu):
#
#   S1[n, c]  = sum_hw grad_out        S2[n, c] = sum_hw grad_out * input
#   dgamma[c] = sum_n rstd[n, g] * (S2[n, c] - mean[n, g] * S1[n, c])
#   dbeta[c]  = sum_n S1[n, c]
#
# Half / bfloat16 run all of it in float32 (ATen's opmath_t) and cast back.
# ---------------------------------------------------------------------------


def _gn_planes(t: T, shape: IndexList[MAX_RANK], rank: Int) raises -> Owned:
    """`t` in float32, contiguous, viewed as `shape`."""
    var f = _as_dtype(t, ST_FLOAT32)
    var d = _dense(f)
    return _view_as(d, shape, rank)


def _gn_dims3(a: Int, b: Int, c: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 3] = a
    shape[MAX_RANK - 2] = b
    shape[MAX_RANK - 1] = c
    return shape


def _gn_vector(n: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    return shape


def _gn_grad_input(
    args: Values,
    gc: Owned,
    ac: Owned,
    mean: Owned,
    rstd: Owned,
    out_like: T,
    rows: Int,
    cols: Int,
) raises -> Owned:
    """grad_input through the LayerNorm backward dx kernel on the
    [N * group, K] view, in the input's dtype and shape."""
    var q = _hold(gc.t)
    if not v_is_none(args[unsafe_offset=4]):
        var c = gc.t.dim(1)
        var w = _dense(_as_dtype(v_tensor(args[unsafe_offset=4]), ST_FLOAT32))
        var zeros = _filled_channels(c, 0.0, gc.t.device)
        var ones = _filled_channels(c, 1.0, gc.t.device)
        q = _dense(_channel_affine(gc, zeros, w, zeros, ones))
    var gi = own(
        new_tensor(out_like.shape, out_like.rank, ST_FLOAT32, out_like.device)
    )
    var ctx = ctx_for(out_like.device)
    var call = KernelCall("normalization_backward", "LayerNormBackwardF32")
    call.arg_dtype(0, DType.float32)
    call.arg_dtype(1, DType.float32)
    call.arg_dtype(2, DType.float32)
    call.arg_dtype(3, DType.float32)
    call.arg_dtype(4, DType.float32)
    call.out_dtype_i(0, DType.float32)
    call.out_dtype_i(1, DType.float32)
    call.out_dtype_i(2, DType.float32)
    call.flag("OUTPUT_MASK", 1)
    call.int(gi.t.ptr)
    call.int(0)
    call.int(0)
    call.int(q.t.ptr)
    call.int(ac.t.ptr)
    call.int(mean.t.ptr)
    call.int(rstd.t.ptr)
    call.int(0)
    call.int(rows)
    call.int(cols)
    call.int(1)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = q.t.ptr
    _ = ac.t.ptr
    _ = mean.t.ptr
    _ = rstd.t.ptr
    _ = ctx
    return _as_dtype(gi.t, out_like.stype)


# aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean,
#   Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group,
#   bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def op_native_group_norm_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var mean_t = v_tensor(args[unsafe_offset=2])
    var rstd_t = v_tensor(args[unsafe_offset=3])
    var has_w = not v_is_none(args[unsafe_offset=4])
    var n = v_int(args[unsafe_offset=5])
    var c = v_int(args[unsafe_offset=6])
    var hxw = v_int(args[unsafe_offset=7])
    var group = v_int(args[unsafe_offset=8])
    var mask = _bool_list(args[unsafe_offset=9])
    if len(mask) != 3:
        raise Error(
            "native_group_norm_backward: output_mask must have 3 entries"
        )
    if (
        not a.on_mojo()
        or grad.device != a.device
        or mean_t.device != a.device
        or rstd_t.device != a.device
    ):
        unsupported(
            "native_group_norm_backward: every operand must be on one mojo"
            " device"
        )
    if (
        a.dtype != DType.float32
        and a.dtype != DType.bfloat16
        and a.dtype != DType.float16
    ):
        unsupported("native_group_norm_backward of dtype " + String(a.dtype))
    if not grad.same_shape(a) or grad.stype != a.stype:
        unsupported(
            "native_group_norm_backward: grad_out and input must share a"
            " shape and dtype"
        )
    if (
        group <= 0
        or c <= 0
        or c % group != 0
        or a.numel != n * c * hxw
        or mean_t.numel != n * group
        or rstd_t.numel != n * group
    ):
        unsupported("native_group_norm_backward: bad group geometry")
    var pst = a.stype
    if has_w:
        var w = v_tensor(args[unsafe_offset=4])
        if w.device != a.device or w.rank != 1 or w.dim(0) != c:
            unsupported("native_group_norm_backward: unsupported weight")
        pst = w.stype
    if a.numel == 0:
        # Nothing to reduce: both affine gradients are zero over an empty
        # extent (ATen fills them with zeros too) and grad_input is empty.
        var gi = own(new_like(a))
        var gw = _bn_zero_grad(_gn_vector(c), pst, a.device, mask[1])
        var gb = _bn_zero_grad(_gn_vector(c), pst, a.device, mask[2])
        ret_owned(rets, 0, gi)
        ret_owned(rets, 1, gw)
        ret_owned(rets, 2, gb)
        return
    var cpg = c // group
    var planes = _gn_dims3(n, c, hxw)
    var gc = _gn_planes(grad, planes, 3)
    var ac = _gn_planes(a, planes, 3)
    var stats = _gn_dims3(n, group, 1)
    var mean = _gn_planes(mean_t, stats, 3)
    var rstd = _gn_planes(rstd_t, stats, 3)

    # an output autograd did not ask for is an undefined Tensor (None record)
    if mask[0]:
        var gi = _gn_grad_input(
            args, gc, ac, mean, rstd, a, n * group, cpg * hxw
        )
        ret_owned(rets, 0, gi)
    else:
        rets[unsafe_offset=0] = Value(TAG_NONE, 0, 0, 0)
    if not mask[1] and not mask[2]:
        rets[unsafe_offset=1] = Value(TAG_NONE, 0, 0, 0)
        rets[unsafe_offset=2] = Value(TAG_NONE, 0, 0, 0)
        return
    var spatial = List[Int64](capacity=1)
    spatial.append(2)
    var batch = List[Int64](capacity=1)
    batch.append(0)
    var per_group = _gn_dims3(n, group, cpg)
    var s1 = _sum_dims(gc, spatial, False)
    if mask[1]:
        var gx = _mul(gc, ac)
        var s2 = _dense(_sum_dims(gx, spatial, False))
        var s1g = _view_as(_dense(s1), per_group, 3)
        var s2g = _view_as(s2, per_group, 3)
        var shifted = _mul(s1g, mean)
        var centered = _sub(s2g, shifted)
        var scaled = _mul(centered, rstd)
        var total = _dense(_sum_dims(scaled, batch, False))
        var flat = _view_as(total, _gn_vector(c), 1)
        var gw = _cast(flat, pst)
        ret_owned(rets, 1, gw)
    else:
        rets[unsafe_offset=1] = Value(TAG_NONE, 0, 0, 0)
    if mask[2]:
        var bias_sum = _sum_dims(s1, batch, False)
        var gb = _cast(bias_sum, pst)
        ret_owned(rets, 2, gb)
    else:
        rets[unsafe_offset=2] = Value(TAG_NONE, 0, 0, 0)


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
    impl[op_where_self_out, "where.self_out"](site)
    impl[op_native_batch_norm_backward, "native_batch_norm_backward"](site)
    impl[op_native_group_norm_backward, "native_group_norm_backward"](site)
    impl[op_softmax_backward_data, "_softmax_backward_data"](site)
