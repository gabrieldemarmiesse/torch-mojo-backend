"""ATen ops: loss group (see agents_docs/native_backend.md).

NLL loss (`nll_loss_*`, `nll_loss2d_*`), the multi-class and multi-label
margin losses and CTC, every overload torch's CUDA backend registers, for
float32 / float16 / bfloat16 / float64 (no float64 on Metal). The kernels
are the `loss` family (tmb/kernels/loss/), ported from torch's CUDA ones
with their accumulation dtypes and rounding points; each op below names
the CUDA function whose checks and results it reproduces.

`out=` overloads compute their functional overload into fresh tensors, then
resize and copy into the caller's (`store_out`), so an `out` that aliases an
input is only written after every input was read.
"""
from std.utils import IndexList
from std.utils.numerics import nan

from tmb.backend.abi import (
    Owned,
    ST_INT64,
    ST_UINT8,
    T,
    Value,
    Values,
    new_tensor,
    none_arg,
    own,
    release,
    tensor_arg,
    view_strided,
    ret_owned,
    ret_ref,
    unsupported,
    v_int,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    call_op,
    check_out_as,
    contiguous,
    device_str,
    fill_value,
    forward_args,
    scalar_to_float,
    scalar_to_int,
    store_out,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.nn import sizes_str
from tmb.backend.registry import Site, impl


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


struct Dense(Movable):
    """`t` as a contiguous tensor: the input itself when it already is, a
    copy this op owns (released with it) otherwise."""

    var t: T
    var mine: Bool

    def __init__(out self, t: T) raises:
        var c = contiguous(t)
        self.mine = c.h != t.h
        self.t = c^

    def __deinit__(deinit self):
        if self.mine:
            release(self.t.h)


def _loss_float(t: T) raises:
    """AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16) of a mojo operand,
    float64 excepted on Apple GPUs (no double arithmetic there)."""
    if not t.on_mojo():
        raise Error("expected a tensor on the mojo device, got ", device_str(t))
    if t.dtype == DType.float64:
        if dev(t.device)[].api == "metal":
            unsupported("float64 losses are unavailable on Apple GPUs")
        return
    if (
        t.dtype != DType.float32
        and t.dtype != DType.float16
        and t.dtype != DType.bfloat16
    ):
        raise Error(
            '"loss" not implemented for \'', _scalar_type_name(t.dtype), "'"
        )


def _same_device(a: T, b: T) raises:
    if b.device_type != a.device_type or b.device != a.device:
        raise Error(
            (
                "Expected all tensors to be on the same device, but found at"
                " least two devices, "
            ),
            device_str(a),
            " and ",
            device_str(b),
            "!",
        )


def _expect_dtype(t: T, like: T) raises:
    """`Tensor::data_ptr<scalar_t>()`'s check on a second operand."""
    if t.stype != like.stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(like.dtype),
            " but found ",
            _scalar_type_name(t.dtype),
        )


def _shape1(n: Int) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - 1] = n
    return s


def _scalar(stype: Int32, device: Int) raises -> Owned:
    return own(new_tensor(IndexList[MAX_RANK](1), 0, stype, device))


struct OwnedPair(Movable):
    """Two results of a two-output op, each released unless taken."""

    var first: Owned
    var second: Owned

    def __init__(out self, var first: Owned, var second: Owned):
        self.first = first^
        self.second = second^


# ---------------------------------------------------------------------------
# NLL loss
# ---------------------------------------------------------------------------


struct NllArgs(Movable):
    """The validated operands of one nll call: dense input / target / weight
    and the `[batch, classes, map]` geometry the kernels address."""

    var input: Dense
    var target: Dense
    var weight: Optional[Dense]
    var batch: Int
    var classes: Int
    var map: Int
    var one_d: Bool
    var spatial: Bool

    def __init__(
        out self,
        var input: Dense,
        var target: Dense,
        var weight: Optional[Dense],
        batch: Int,
        classes: Int,
        map: Int,
        one_d: Bool,
        spatial: Bool,
    ):
        self.input = input^
        self.target = target^
        self.weight = weight^
        self.batch = batch
        self.classes = classes
        self.map = map
        self.one_d = one_d
        self.spatial = spatial

    def weight_ptr(self) -> Int:
        if self.weight:
            return self.weight.value().t.ptr
        return 0


def _nll_weight(args: Values, i: Int, input: T) raises -> Optional[Dense]:
    if v_is_none(args[unsafe_offset=i]):
        return None
    var w = v_tensor(args[unsafe_offset=i])
    _same_device(input, w)
    _expect_dtype(w, input)
    return Dense(w)


def _nll_checks(args: Values, self_i: Int, spatial: Bool) raises -> NllArgs:
    """LossNLL.cpp's `nll_loss_forward` / `nll_loss_backward` meta checks
    (torch 2.11 messages), or NLLLoss2d.cu's `check_inputs_nll_loss2d`."""
    var input = v_tensor(args[unsafe_offset=self_i])
    var target = v_tensor(args[unsafe_offset=self_i + 1])
    var weight_i = self_i + 2
    if spatial:
        if target.rank != 3:
            raise Error(
                (
                    "only batches of spatial targets supported (3D tensors) but"
                    " got targets of size: : "
                ),
                sizes_str(target),
            )
        if input.rank != 4:
            raise Error(
                (
                    "only batches of spatial inputs supported (4D tensors), but"
                    " got input of size: "
                ),
                sizes_str(input),
            )
        if not v_is_none(args[unsafe_offset=weight_i]):
            if v_tensor(args[unsafe_offset=weight_i]).numel != input.dim(1):
                raise Error(
                    "weight tensor should be defined either for all or no"
                    " classes"
                )
        if (
            input.dim(0) != target.dim(0)
            or input.dim(2) != target.dim(1)
            or input.dim(3) != target.dim(2)
        ):
            raise Error(
                "input and target batch or spatial sizes don't match: target ",
                sizes_str(target),
                ", input ",
                sizes_str(input),
            )
        if target.stype != ST_INT64:
            raise Error(
                "expected scalar type Long but found ",
                _scalar_type_name(target.dtype),
            )
    else:
        if input.rank < 1 or input.rank > 2:
            raise Error("input tensor should be 1D or 2D")
        if target.rank > 1:
            raise Error(
                "0D or 1D target tensor expected, multi-target not supported"
            )
        if target.stype != ST_INT64 and target.stype != ST_UINT8:
            raise Error(
                "expected target dtype to be Long or Byte, but got ",
                _scalar_type_name(target.dtype),
            )
    _loss_float(input)
    _same_device(input, target)
    var weight = _nll_weight(args, weight_i, input)
    var batch: Int
    var classes: Int
    var map = 1
    if spatial:
        batch = input.dim(0)
        classes = input.dim(1)
        map = input.dim(2) * input.dim(3)
    else:
        classes = input.dim(-1)
        batch = 1 if input.rank == 1 else input.dim(0)
    return NllArgs(
        Dense(input),
        Dense(target),
        weight^,
        batch,
        classes,
        map,
        input.rank == 1,
        spatial,
    )


def _nll_forward_shape_checks(args: Values) raises:
    """The forward-only part of the `nll_loss_forward` meta."""
    var input = v_tensor(args[unsafe_offset=0])
    var target = v_tensor(args[unsafe_offset=1])
    if input.rank == 1 and target.rank == 1 and target.dim(0) != 1:
        raise Error(
            "For 1D input, 1D target must have size 1, but got target size: ",
            target.dim(0),
        )
    if input.rank != 1 and input.dim(0) != target.dim(0):
        raise Error(
            "size mismatch (got input: ",
            sizes_str(input),
            ", target: ",
            sizes_str(target),
            ")",
        )
    var n_classes = input.dim(-1)
    if not v_is_none(args[unsafe_offset=2]):
        var w = v_tensor(args[unsafe_offset=2])
        if w.rank != 1 or w.numel != n_classes:
            raise Error(
                "weight tensor should be defined either for all ",
                n_classes,
                " classes or no classes but got weight tensor of shape: ",
                sizes_str(w),
            )


def _reduction(v: Int) -> Int:
    """CUDA's branches: None (0), Mean (1), and anything else sums."""
    if v == 0 or v == 1:
        return v
    return 2


def _nll_params(a: NllArgs, reduction: Int, ignore_index: Int) -> List[Int]:
    var p = List[Int](capacity=7)
    p.append(a.batch)
    p.append(a.classes)
    p.append(a.map)
    p.append(reduction)
    p.append(ignore_index)
    p.append(1 if a.one_d else 0)
    p.append(1 if a.spatial else 0)
    return p^


def _nll_fast_f32(a: NllArgs, reduction: Int, weighted: Bool) -> Bool:
    """The f32 / int64 / unweighted 2-D regime of the hand-tuned
    `NllLossForwardF32` / `NllLossBackwardF32` kernels (entry.mojo)."""
    return (
        not a.spatial
        and not a.one_d
        and not weighted
        and a.input.t.dtype == DType.float32
        and a.target.t.stype == ST_INT64
        and a.batch > 0
        and a.classes > 0
    )


def _nll_forward(args: Values, spatial: Bool) raises -> OwnedPair:
    """`nll_loss_forward` (Loss.cu nll_loss_forward_out_cuda_template) or
    `nll_loss2d_forward` (NLLLoss2d.cu): (output, total_weight)."""
    if not spatial:
        _nll_forward_shape_checks(args)
    var a = _nll_checks(args, 0, spatial)
    var reduction = _reduction(v_int(args[unsafe_offset=3]))
    var ignore_index = v_int(args[unsafe_offset=4])
    var input = a.input.t.copy()
    var tw = _scalar(input.stype, input.device)
    var none_mode = reduction == 0 and not a.one_d
    var out: Owned
    if none_mode:
        if spatial:
            var shape = IndexList[MAX_RANK](1)
            shape[MAX_RANK - 3] = input.dim(0)
            shape[MAX_RANK - 2] = input.dim(2)
            shape[MAX_RANK - 1] = input.dim(3)
            out = own(new_tensor(shape, 3, input.stype, input.device))
        else:
            out = own(
                new_tensor(_shape1(a.batch), 1, input.stype, input.device)
            )
        fill_value(tw.t, 0.0)
        if a.batch * a.map == 0:
            return OwnedPair(out^, tw^)
    else:
        out = _scalar(input.stype, input.device)
        if a.target.t.numel == 0:
            fill_value(out.t, nan[DType.float64]() if reduction == 1 else 0.0)
            fill_value(tw.t, 0.0)
            return OwnedPair(out^, tw^)
    var ctx = ctx_for(input.device)
    if _nll_fast_f32(a, reduction, a.weight_ptr() != 0):
        var call = KernelCall("loss", "NllLossForwardF32")
        call.arg_dtype(0, input.dtype)
        call.arg_dtype(1, a.target.t.dtype)
        call.out_dtype_i(0, DType.float32)
        call.out_dtype_i(1, DType.float32)
        call.flag("REDUCTION", reduction)
        call.int(out.t.ptr)
        call.int(tw.t.ptr)
        call.int(input.ptr)
        call.int(a.target.t.ptr)
        call.int(a.batch)
        call.int(a.classes)
        call.int(reduction)
        call.int(ignore_index)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = a^
        return OwnedPair(out^, tw^)
    var scratch = own(new_tensor(_shape1(1), 1, input.stype, input.device))
    if spatial and not none_mode:
        # NLLLoss2d.cu: GET_BLOCKS(map) / 128 blocks per sample, at least 1
        # (nll2d_blocks_per_sample in kernels/loss/nll_kernels.mojo).
        var bps = max(1, ((a.map + 127) // 128) // 128)
        scratch = own(
            new_tensor(_shape1(2 * bps * a.batch), 1, input.stype, input.device)
        )
    var call = KernelCall("loss", "Nll")
    call.arg_dtype(0, input.dtype)
    call.arg_dtype(1, a.target.t.dtype)
    call.int(out.t.ptr)
    call.int(tw.t.ptr)
    call.int(input.ptr)
    call.int(a.target.t.ptr)
    call.int(a.weight_ptr())
    call.int(scratch.t.ptr)
    call.tuple(_nll_params(a, reduction, ignore_index))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = scratch^
    _ = a^
    return OwnedPair(out^, tw^)


def _nll_backward_checks(args: Values, a: NllArgs, spatial: Bool) raises:
    """The grad_output / total_weight checks of `nll_loss_backward`'s meta
    (or NLLLoss2d.cu's backward template)."""
    var grad = v_tensor(args[unsafe_offset=0])
    var input = v_tensor(args[unsafe_offset=1])
    var target = v_tensor(args[unsafe_offset=2])
    var tw = v_tensor(args[unsafe_offset=6])
    var reduction = _reduction(v_int(args[unsafe_offset=4]))
    if spatial:
        if tw.numel != 1:
            raise Error(
                "expected total_weight to be a single element tensor, got: ",
                sizes_str(tw),
                " (",
                tw.numel,
                " elements)",
            )
        if reduction == 0:
            if grad.rank != 3:
                raise Error(
                    (
                        "grad_output must have same dimension as target (3) but"
                        " got dimension: "
                    ),
                    sizes_str(grad),
                )
            if (
                grad.dim(0) != target.dim(0)
                or grad.dim(1) != target.dim(1)
                or grad.dim(2) != target.dim(2)
            ):
                raise Error(
                    "grad_output sizes don't match target sizes: target ",
                    sizes_str(target),
                    ", grad_output ",
                    sizes_str(grad),
                )
    else:
        var no_batch = input.rank == 1 and target.rank == 0
        if not no_batch and input.dim(0) != target.dim(0):
            raise Error(
                "size mismatch (got input: ",
                sizes_str(input),
                ", target: ",
                sizes_str(target),
                ")",
            )
        if tw.numel != 1:
            raise Error(
                "expected total_weight to be a  single element tensor, got: ",
                sizes_str(tw),
                " (",
                tw.numel,
                " elements)",
            )
        if not v_is_none(args[unsafe_offset=3]):
            if v_tensor(args[unsafe_offset=3]).numel != input.dim(-1):
                raise Error(
                    "weight tensor should be defined either for all or no"
                    " classes"
                )
        if reduction == 0 and input.rank == 2:
            if grad.rank != 1 or grad.dim(0) != input.dim(0):
                raise Error(
                    "Expected a tensor of dimension 1 and tensor.size[0] == ",
                    input.dim(0),
                    " but got: dimension ",
                    grad.rank,
                    " and tensor.size[0] = ",
                    grad.dim(0) if grad.rank > 0 else 1,
                )
        elif grad.rank > 1 or grad.numel != 1:
            raise Error(
                "Expected a single element grad_output tensor, but got: ",
                sizes_str(grad),
            )
    _same_device(input, grad)
    _same_device(input, tw)
    _expect_dtype(grad, input)
    _expect_dtype(tw, input)


def _nll_backward(args: Values, spatial: Bool) raises -> Owned:
    """`nll_loss_backward` (Loss.cu nll_loss_backward_out_cuda) or
    `nll_loss2d_backward` (NLLLoss2d.cu): a zeroed grad_input with the
    target slots written."""
    var a = _nll_checks(args, 1, spatial)
    _nll_backward_checks(args, a, spatial)
    var reduction = _reduction(v_int(args[unsafe_offset=4]))
    var ignore_index = v_int(args[unsafe_offset=5])
    var input = a.input.t.copy()
    var gi = own(new_tensor(input.shape, input.rank, input.stype, input.device))
    fill_value(gi.t, 0.0)
    if a.batch * a.map == 0 or input.numel == 0:
        return gi^
    var grad = Dense(v_tensor(args[unsafe_offset=0]))
    var tw = Dense(v_tensor(args[unsafe_offset=6]))
    var ctx = ctx_for(input.device)
    if _nll_fast_f32(a, reduction, a.weight_ptr() != 0):
        var call = KernelCall("loss", "NllLossBackwardF32")
        call.arg_dtype(0, grad.t.dtype)
        call.arg_dtype(1, a.target.t.dtype)
        call.arg_dtype(2, tw.t.dtype)
        call.out_dtype(DType.float32)
        call.flag("REDUCTION", reduction)
        call.int(gi.t.ptr)
        call.int(grad.t.ptr)
        call.int(a.target.t.ptr)
        call.int(tw.t.ptr)
        call.int(a.batch)
        call.int(a.classes)
        call.int(reduction)
        call.int(ignore_index)
        call.int(ctx_ptr(ctx))
        call.run()
    else:
        var call = KernelCall("loss", "NllBackward")
        call.arg_dtype(0, input.dtype)
        call.arg_dtype(1, a.target.t.dtype)
        call.int(gi.t.ptr)
        call.int(grad.t.ptr)
        call.int(a.target.t.ptr)
        call.int(a.weight_ptr())
        call.int(tw.t.ptr)
        call.tuple(_nll_params(a, reduction, ignore_index))
        call.int(ctx_ptr(ctx))
        call.run()
    _ = ctx
    _ = grad^
    _ = tw^
    _ = a^
    return gi^


def _nll_forward_ret(args: Values, rets: Values, spatial: Bool) raises:
    var r = _nll_forward(args, spatial)
    ret_owned(rets, 0, r.first)
    ret_owned(rets, 1, r.second)


def _nll_forward_out(args: Values, rets: Values, spatial: Bool) raises:
    var like = v_tensor(args[unsafe_offset=0])
    var output = v_tensor(args[unsafe_offset=5])
    var total_weight = v_tensor(args[unsafe_offset=6])
    check_out_as(output, like.stype, like)
    check_out_as(total_weight, like.stype, like)
    var r = _nll_forward(args, spatial)
    store_out(output, r.first.take())
    store_out(total_weight, r.second.take())
    ret_ref(rets, 0, output)
    ret_ref(rets, 1, total_weight)


def _nll_backward_out(args: Values, rets: Values, spatial: Bool) raises:
    var like = v_tensor(args[unsafe_offset=1])
    var grad_input = v_tensor(args[unsafe_offset=7])
    check_out_as(grad_input, like.stype, like)
    var gi = _nll_backward(args, spatial)
    store_out(grad_input, gi.take())
    ret_ref(rets, 0, grad_input)


# aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight,
#   int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)
def op_nll_loss_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_forward_ret(args, rets, False)


# aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight,
#   int reduction, SymInt ignore_index, *, Tensor(a!) output,
#   Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
def op_nll_loss_forward_output(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_forward_out(args, rets, False)


# aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target,
#   Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight)
#   -> Tensor
def op_nll_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var gi = _nll_backward(args, False)
    ret_owned(rets, 0, gi)


# aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self,
#   Tensor target, Tensor? weight, int reduction, SymInt ignore_index,
#   Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_nll_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_backward_out(args, rets, False)


# aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight,
#   int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)
def op_nll_loss2d_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_forward_ret(args, rets, True)


# aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight,
#   int reduction, SymInt ignore_index, *, Tensor(a!) output,
#   Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
def op_nll_loss2d_forward_output(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_forward_out(args, rets, True)


# aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target,
#   Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight)
#   -> Tensor
def op_nll_loss2d_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var gi = _nll_backward(args, True)
    ret_owned(rets, 0, gi)


# aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self,
#   Tensor target, Tensor? weight, int reduction, SymInt ignore_index,
#   Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_nll_loss2d_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _nll_backward_out(args, rets, True)


# ---------------------------------------------------------------------------
# Multi-class margin loss (MultiMarginLoss.cu)
# ---------------------------------------------------------------------------


struct MarginGeom(Copyable, Movable):
    var nframe: Int
    var dim: Int

    def __init__(out self, nframe: Int, dim: Int):
        self.nframe = nframe
        self.dim = dim


def _margin_input_check(input: T) raises:
    """The first check of LossMulti.h's two shape checks."""
    var ok = (
        (input.rank == 2 and input.dim(1) != 0)
        or (input.rank == 1 and input.dim(0) != 0)
        or input.rank == 0
    )
    if not ok:
        raise Error(
            (
                "Expected non-empty vector or matrix with optional 0-dim batch"
                " size, but got: "
            ),
            sizes_str(input),
        )


def _margin_geom(input: T) -> MarginGeom:
    if input.rank <= 1:
        return MarginGeom(1, 1 if input.rank == 0 else input.dim(0))
    return MarginGeom(input.dim(0), input.dim(1))


def _multi_margin_checks(
    input: T, target: T, args: Values, weight_i: Int
) raises -> MarginGeom:
    """LossMulti.h `multi_margin_loss_shape_check` (torch 2.11 messages)."""
    _margin_input_check(input)
    var g = _margin_geom(input)
    if target.rank > 1 or target.numel != g.nframe:
        raise Error(
            "inconsistent target size, expected ",
            g.nframe,
            " but got ",
            sizes_str(target),
        )
    if not v_is_none(args[unsafe_offset=weight_i]):
        var w = v_tensor(args[unsafe_offset=weight_i])
        if w.rank > 1 or w.numel != g.dim:
            raise Error(
                "inconsistent weight size, expected ",
                g.dim,
                " but got ",
                sizes_str(w),
            )
    return g^


def _margin_p(v: Value, what: StaticString) raises -> Int:
    var p = scalar_to_int(v, ST_INT64)
    if p != 1 and p != 2:
        raise Error(what, ": Invalid p, expected 1 or 2 but got ", p)
    return p


def _optional_dense(args: Values, i: Int, like: T) raises -> Optional[Dense]:
    if v_is_none(args[unsafe_offset=i]):
        return None
    var t = v_tensor(args[unsafe_offset=i])
    _same_device(like, t)
    _expect_dtype(t, like)
    return Dense(t)


def _ptr_or_zero(t: Optional[Dense]) -> Int:
    if t:
        return t.value().t.ptr
    return 0


def _multi_margin_forward(args: Values) raises -> Owned:
    """multi_margin_loss_cuda_out's result, into a fresh tensor."""
    var input = v_tensor(args[unsafe_offset=0])
    var target = v_tensor(args[unsafe_offset=1])
    var p = _margin_p(args[unsafe_offset=2], "multi_margin_loss")
    var g = _multi_margin_checks(input, target, args, 4)
    var reduction = _reduction(v_int(args[unsafe_offset=5]))
    var out: Owned
    if reduction == 0 and target.rank > 0:
        out = own(new_tensor(_shape1(g.nframe), 1, input.stype, input.device))
    else:
        out = _scalar(input.stype, input.device)
    if input.numel == 0:
        # CUDA hands back the resized, uninitialized result.
        fill_value(out.t, 0.0)
        return out^
    _loss_float(input)
    _same_device(input, target)
    if target.stype != ST_INT64:
        raise Error(
            "expected scalar type Long but found ",
            _scalar_type_name(target.dtype),
        )
    var margin = scalar_to_float(args[unsafe_offset=3], input.stype)
    var weight = _optional_dense(args, 4, input)
    var x = Dense(input)
    var tg = Dense(target)
    # 2-D with a reduction: per-sample values (already divided by
    # nframe * dim for the mean), then their sum, as CUDA's `at::sum_out`.
    var per_sample = input.rank == 2 and reduction != 0
    var dst = own(new_tensor(_shape1(g.nframe), 1, input.stype, input.device))
    var size_average = reduction == 1
    if input.rank == 2 and reduction == 0:
        size_average = False
    var ctx = ctx_for(input.device)
    var call = KernelCall("loss", "MultiMargin")
    call.arg_dtype(0, input.dtype)
    call.int(dst.t.ptr)
    call.int(x.t.ptr)
    call.int(tg.t.ptr)
    call.int(_ptr_or_zero(weight))
    var params = List[Int]()
    params.append(g.nframe)
    params.append(g.dim)
    params.append(p)
    params.append(1 if size_average else 0)
    call.tuple(params)
    call.f64(margin)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = x^
    _ = tg^
    _ = weight^
    if per_sample:
        var s = call_op(
            String("aten::sum"),
            String(""),
            [tensor_arg(dst.t), none_arg()],
            1,
        )
        _ = dst^
        return own(s.take_tensor(0))
    if out.t.rank == dst.t.rank:
        return dst^
    var v = own(
        view_strided(
            dst.t, out.t.shape, out.t.strides, out.t.rank, dst.t.offset
        )
    )
    return v^


# aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1,
#   Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
def op_multi_margin_loss(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _multi_margin_forward(args)
    ret_owned(rets, 0, r)


# aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1,
#   Scalar margin=1, Tensor? weight=None, int reduction=Mean, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_multi_margin_loss_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var like = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=6])
    check_out_as(dst, like.stype, like)
    var r = _multi_margin_forward(args)
    store_out(dst, r.take())
    ret_ref(rets, 0, dst)


def _multi_margin_backward(args: Values) raises -> Owned:
    """multi_margin_loss_cuda_backward_out's result, into a fresh tensor."""
    var grad = v_tensor(args[unsafe_offset=0])
    var input = v_tensor(args[unsafe_offset=1])
    var target = v_tensor(args[unsafe_offset=2])
    var p = _margin_p(args[unsafe_offset=3], "multi_margin_loss_backward")
    var g = _multi_margin_checks(input, target, args, 5)
    var reduction = _reduction(v_int(args[unsafe_offset=6]))
    var gi = own(new_tensor(input.shape, input.rank, input.stype, input.device))
    if input.numel == 0:
        return gi^
    _loss_float(input)
    _same_device(input, target)
    _same_device(input, grad)
    if target.stype != ST_INT64:
        raise Error(
            "expected scalar type Long but found ",
            _scalar_type_name(target.dtype),
        )
    _expect_dtype(grad, input)
    var margin = scalar_to_float(args[unsafe_offset=4], input.stype)
    var weight = _optional_dense(args, 5, input)
    var x = Dense(input)
    var tg = Dense(target)
    var go = Dense(grad)
    var ctx = ctx_for(input.device)
    var call = KernelCall("loss", "MultiMarginBackward")
    call.arg_dtype(0, input.dtype)
    call.int(gi.t.ptr)
    call.int(go.t.ptr)
    call.int(x.t.ptr)
    call.int(tg.t.ptr)
    call.int(_ptr_or_zero(weight))
    var params = List[Int]()
    params.append(g.nframe)
    params.append(g.dim)
    params.append(p)
    params.append(1 if reduction == 1 else 0)
    params.append(1 if reduction != 0 else 0)
    call.tuple(params)
    call.f64(margin)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = x^
    _ = tg^
    _ = go^
    _ = weight^
    return gi^


# aten::multi_margin_loss_backward(Tensor grad_output, Tensor self,
#   Tensor target, Scalar p, Scalar margin, Tensor? weight=None,
#   int reduction=Mean) -> Tensor
def op_multi_margin_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _multi_margin_backward(args)
    ret_owned(rets, 0, r)


# aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self,
#   Tensor target, Scalar p, Scalar margin, Tensor? weight=None,
#   int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_multi_margin_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var like = v_tensor(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=7])
    check_out_as(dst, like.stype, like)
    var r = _multi_margin_backward(args)
    store_out(dst, r.take())
    ret_ref(rets, 0, dst)


# ---------------------------------------------------------------------------
# Multi-label margin loss (MultiLabelMarginCriterion.cu)
# ---------------------------------------------------------------------------


def _multilabel_checks(input: T, target: T) raises -> MarginGeom:
    """LossMulti.h `multilabel_margin_loss_shape_check`."""
    _margin_input_check(input)
    var g = _margin_geom(input)
    var ok: Bool
    if input.rank <= 1:
        ok = target.rank <= 1 and target.numel == g.dim
    else:
        ok = (
            target.rank == 2
            and target.dim(0) == g.nframe
            and target.dim(1) == g.dim
        )
    if not ok:
        raise Error(
            "inconsistent target size: ",
            sizes_str(target),
            " for input of size: ",
            sizes_str(input),
        )
    return g^


def _multilabel_forward(args: Values) raises -> OwnedPair:
    """multilabel_margin_loss_forward_out_cuda_template, into fresh
    tensors: (output, is_target)."""
    var input = v_tensor(args[unsafe_offset=0])
    var target = v_tensor(args[unsafe_offset=1])
    var reduction = _reduction(v_int(args[unsafe_offset=2]))
    var g = _multilabel_checks(input, target)
    if input.numel == 0:
        # CUDA returns before resizing either `at::empty({0})` result.
        return OwnedPair(
            own(new_tensor(_shape1(0), 1, input.stype, input.device)),
            own(new_tensor(_shape1(0), 1, input.stype, input.device)),
        )
    _loss_float(input)
    _same_device(input, target)
    if target.stype != ST_INT64:
        raise Error(
            "expected scalar type Long but found ",
            _scalar_type_name(target.dtype),
        )
    var x = Dense(input)
    var tg = Dense(target)
    var is_target = own(
        new_tensor(target.shape, target.rank, input.stype, input.device)
    )
    var per = own(new_tensor(_shape1(g.nframe), 1, input.stype, input.device))
    var ctx = ctx_for(input.device)
    var call = KernelCall("loss", "MultilabelMargin")
    call.arg_dtype(0, input.dtype)
    call.int(per.t.ptr)
    call.int(x.t.ptr)
    call.int(tg.t.ptr)
    call.int(is_target.t.ptr)
    var params = List[Int]()
    params.append(g.nframe)
    params.append(g.dim)
    params.append(1 if reduction == 1 else 0)
    call.tuple(params)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = x^
    _ = tg^
    if input.rank <= 1:
        var v = own(
            view_strided(
                per.t,
                IndexList[MAX_RANK](1),
                IndexList[MAX_RANK](0),
                0,
                per.t.offset,
            )
        )
        _ = per^
        return OwnedPair(v^, is_target^)
    if reduction == 0:
        return OwnedPair(per^, is_target^)
    var s = call_op(
        String("aten::sum"), String(""), [tensor_arg(per.t), none_arg()], 1
    )
    _ = per^
    return OwnedPair(own(s.take_tensor(0)), is_target^)


# aten::multilabel_margin_loss_forward(Tensor self, Tensor target,
#   int reduction) -> (Tensor output, Tensor is_target)
def op_multilabel_margin_loss_forward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _multilabel_forward(args)
    ret_owned(rets, 0, r.first)
    ret_owned(rets, 1, r.second)


# aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target,
#   int reduction, *, Tensor(a!) output, Tensor(b!) is_target)
#   -> (Tensor(a!), Tensor(b!))
def op_multilabel_margin_loss_forward_output(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var like = v_tensor(args[unsafe_offset=0])
    var output = v_tensor(args[unsafe_offset=3])
    var is_target = v_tensor(args[unsafe_offset=4])
    check_out_as(output, like.stype, like)
    check_out_as(is_target, like.stype, like)
    var r = _multilabel_forward(args)
    if like.numel == 0:
        # CUDA's template returns before touching either out.
        ret_ref(rets, 0, output)
        ret_ref(rets, 1, is_target)
        return
    store_out(output, r.first.take())
    store_out(is_target, r.second.take())
    ret_ref(rets, 0, output)
    ret_ref(rets, 1, is_target)


def _multilabel_backward(args: Values) raises -> Owned:
    """multilabel_margin_loss_backward_cuda_out_template's result."""
    var grad = v_tensor(args[unsafe_offset=0])
    var input = v_tensor(args[unsafe_offset=1])
    var target = v_tensor(args[unsafe_offset=2])
    var reduction = _reduction(v_int(args[unsafe_offset=3]))
    var is_target = v_tensor(args[unsafe_offset=4])
    var g = _multilabel_checks(input, target)
    var gi = own(new_tensor(input.shape, input.rank, input.stype, input.device))
    if input.numel == 0:
        return gi^
    if input.rank <= 1:
        var target_size = 1 if target.rank == 0 else target.dim(0)
        if target.numel == 0 or target.rank > 1 or target_size != g.dim:
            raise Error("inconsistent target size")
    elif (
        input.dim(1) == 0
        or target.rank != 2
        or target.dim(0) != g.nframe
        or target.dim(1) != g.dim
    ):
        raise Error("inconsistent target size")
    if not target.same_shape(is_target):
        raise Error("inconsistent is_target size")
    _loss_float(input)
    _same_device(input, target)
    _same_device(input, grad)
    _same_device(input, is_target)
    if target.stype != ST_INT64:
        raise Error(
            "expected scalar type Long but found ",
            _scalar_type_name(target.dtype),
        )
    _expect_dtype(grad, input)
    _expect_dtype(is_target, input)
    var reduce = reduction != 0
    var denom = g.nframe * g.dim if reduction == 1 else g.dim
    var x = Dense(input)
    var tg = Dense(target)
    var it = Dense(is_target)
    var go = Dense(grad)
    var ctx = ctx_for(input.device)
    var call = KernelCall("loss", "MultilabelMarginBackward")
    call.arg_dtype(0, input.dtype)
    call.int(gi.t.ptr)
    call.int(go.t.ptr)
    call.int(x.t.ptr)
    call.int(tg.t.ptr)
    call.int(it.t.ptr)
    var params = List[Int]()
    params.append(g.nframe)
    params.append(g.dim)
    params.append(1 if reduce else 0)
    call.tuple(params)
    # `1. / static_cast<accscalar_t>(n)`: the count rounded to the
    # accumulation dtype, the quotient in double.
    var n = Float64(denom)
    if input.dtype != DType.float64:
        n = Float64(Float32(denom))
    call.f64(1.0 / n)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = x^
    _ = tg^
    _ = it^
    _ = go^
    return gi^


# aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self,
#   Tensor target, int reduction, Tensor is_target) -> Tensor
def op_multilabel_margin_loss_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _multilabel_backward(args)
    ret_owned(rets, 0, r)


# aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output,
#   Tensor self, Tensor target, int reduction, Tensor is_target, *,
#   Tensor(a!) grad_input) -> Tensor(a!)
def op_multilabel_margin_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var like = v_tensor(args[unsafe_offset=1])
    var dst = v_tensor(args[unsafe_offset=5])
    check_out_as(dst, like.stype, like)
    var r = _multilabel_backward(args)
    store_out(dst, r.take())
    ret_ref(rets, 0, dst)


def register_loss(site: Site) raises:
    impl[op_nll_loss_forward, "nll_loss_forward"](site)
    impl[op_nll_loss_forward_output, "nll_loss_forward.output"](site)
    impl[op_nll_loss_backward, "nll_loss_backward"](site)
    impl[op_nll_loss_backward_grad_input, "nll_loss_backward.grad_input"](site)
    impl[op_nll_loss2d_forward, "nll_loss2d_forward"](site)
    impl[op_nll_loss2d_forward_output, "nll_loss2d_forward.output"](site)
    impl[op_nll_loss2d_backward, "nll_loss2d_backward"](site)
    impl[op_nll_loss2d_backward_grad_input, "nll_loss2d_backward.grad_input"](
        site
    )
    impl[op_multi_margin_loss, "multi_margin_loss"](site)
    impl[op_multi_margin_loss_out, "multi_margin_loss.out"](site)
    impl[op_multi_margin_loss_backward, "multi_margin_loss_backward"](site)
    impl[
        op_multi_margin_loss_backward_grad_input,
        "multi_margin_loss_backward.grad_input",
    ](site)
    impl[op_multilabel_margin_loss_forward, "multilabel_margin_loss_forward"](
        site
    )
    impl[
        op_multilabel_margin_loss_forward_output,
        "multilabel_margin_loss_forward.output",
    ](site)
    impl[op_multilabel_margin_loss_backward, "multilabel_margin_loss_backward"](
        site
    )
    impl[
        op_multilabel_margin_loss_backward_grad_input,
        "multilabel_margin_loss_backward.grad_input",
    ](site)
