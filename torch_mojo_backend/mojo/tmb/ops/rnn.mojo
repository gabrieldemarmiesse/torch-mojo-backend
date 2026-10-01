"""ATen ops: fused recurrent cells (see agents_docs/native_backend.md).

`nn.LSTM` / `nn.GRU` / `nn.LSTMCell` / `nn.GRUCell` reach these through
ATen's `LSTMCell` / `GRUCell` (aten/src/ATen/native/RNN.cpp), which on a
PrivateUse1 device does the two gate GEMMs with ordinary `linear` calls and
hands the pointwise remainder to `_thnn_fused_lstm_cell` /
`_thnn_fused_gru_cell`. Their autograd formulas call the backward ops below
(`_thnn_fused_lstm_cell_backward` is a composite over
`_thnn_fused_lstm_cell_backward_impl`). The kernels port
aten/src/ATen/native/cuda/RNN.cu: tmb/kernels/rnn.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    T,
    TAG_BOOL,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_TENSOR,
    Value,
    Values,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    unsupported,
    v_bool,
    v_opt_tensor,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import call_op, contiguous, shape_str
from tmb.ops.data_movement import _scalar_type_name


# --- argument checks (ATen's TensorUtils, with its messages) ----------------


def _arg(name: StaticString, pos: Int) -> String:
    return String("argument #", pos, " '", name, "'")


def _check_dim(
    c: StaticString, t: T, name: StaticString, pos: Int, dim: Int
) raises:
    if t.rank != dim:
        raise Error(
            "Expected ",
            dim,
            "-dimensional tensor, but got ",
            t.rank,
            "-dimensional tensor for ",
            _arg(name, pos),
            " (while checking arguments for ",
            c,
            ")",
        )


def _size_str(t: T) raises -> String:
    var s = shape_str(t)
    return "[" + String(s[byte = 1 : s.byte_length() - 1]) + "]"


def _check_same_size(
    c: StaticString,
    a: T,
    an: StaticString,
    ap: Int,
    b: T,
    bn: StaticString,
    bp: Int,
) raises:
    if not a.same_shape(b):
        raise Error(
            "Expected tensor for ",
            _arg(an, ap),
            " to have same size as tensor for ",
            _arg(bn, bp),
            "; but ",
            _size_str(a),
            " does not equal ",
            _size_str(b),
            " (while checking arguments for ",
            c,
            ")",
        )


def _check_numel(
    c: StaticString, t: T, name: StaticString, pos: Int, numel: Int
) raises:
    if t.numel != numel:
        raise Error(
            "Expected tensor for ",
            _arg(name, pos),
            " to have ",
            numel,
            " elements; but it actually has ",
            t.numel,
            " elements (while checking arguments for ",
            c,
            ")",
        )


def _check_size2(
    c: StaticString, t: T, name: StaticString, pos: Int, d0: Int, d1: Int
) raises:
    if t.rank != 2 or t.dim(0) != d0 or t.dim(1) != d1:
        raise Error(
            "Expected tensor of size [",
            d0,
            ", ",
            d1,
            "], but got tensor of size ",
            _size_str(t),
            " for ",
            _arg(name, pos),
            " (while checking arguments for ",
            c,
            ")",
        )


def _check_device(c: StaticString, first: T, t: T) raises:
    if not t.on_mojo() or t.device != first.device:
        raise Error(
            (
                "Expected all tensors to be on the same device as the first one"
                " (while checking arguments for "
            ),
            c,
            ")",
        )


def _check_dtype(first: T, t: T) raises:
    """`data_ptr<scalar_t>()` on an operand of another dtype."""
    if t.stype != first.stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(first.dtype),
            " but found ",
            _scalar_type_name(t.dtype),
        )


def _float_dtype(t: T, what: StaticString) raises:
    if (
        t.dtype != DType.float32
        and t.dtype != DType.float16
        and t.dtype != DType.bfloat16
    ):
        if t.dtype == DType.float64:
            unsupported(String(what, " in float64"))
        raise Error(
            '"',
            what,
            "\" not implemented for '",
            _scalar_type_name(t.dtype),
            "'",
        )


def _check_cell_sizes(
    c: StaticString,
    input_gates: T,
    hidden_gates: T,
    input_bias: Optional[T],
    hidden_bias: Optional[T],
    factor: Int,
    prev_hidden: T,
) raises:
    """RNN.cu's `checkSizes`."""
    _check_dim(c, input_gates, "input_gates", 1, 2)
    _check_same_size(
        c, input_gates, "input_gates", 1, hidden_gates, "hidden_gates", 2
    )
    var gates_size = input_gates.dim(1)
    if input_bias:
        _check_dim(c, input_bias.value(), "input_bias", 3, 1)
        _check_numel(c, input_bias.value(), "input_bias", 3, gates_size)
        if not hidden_bias:
            raise Error(
                (
                    "Expected tensor for argument #4 'hidden_bias' to be"
                    " defined (while checking arguments for "
                ),
                c,
                ")",
            )
        _check_same_size(
            c,
            input_bias.value(),
            "input_bias",
            3,
            hidden_bias.value(),
            "hidden_bias",
            4,
        )
    _check_dim(c, prev_hidden, "prev_hidden", 5, 2)
    _check_numel(
        c,
        prev_hidden,
        "prev_hidden",
        5,
        input_gates.dim(0) * gates_size // factor,
    )
    _check_device(c, input_gates, input_gates)
    _check_device(c, input_gates, hidden_gates)
    _check_device(c, input_gates, prev_hidden)
    _check_dtype(input_gates, hidden_gates)
    _check_dtype(input_gates, prev_hidden)
    if input_bias:
        _check_device(c, input_gates, input_bias.value())
        _check_device(c, input_gates, hidden_bias.value())
        _check_dtype(input_gates, input_bias.value())
        _check_dtype(input_gates, hidden_bias.value())


# --- helpers -----------------------------------------------------------------


def _shape2(a: Int, b: Int) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - 2] = a
    s[MAX_RANK - 1] = b
    return s


def _dense(t: T) raises -> Owned:
    """`t` itself (borrowed) when contiguous, else an owned dense copy."""
    return own_if_new(contiguous(t), t)


def _sum0(t: T) raises -> Owned:
    """`t.sum(0)` through the dispatcher (half sums accumulate in float)."""
    var dims = List[Int64]()
    dims.append(0)
    var rets = call_op(
        "aten::sum",
        "dim_IntList",
        [
            Value(TAG_TENSOR, 0, Int64(t.h), 0),
            Value(TAG_INT_LIST, 1, Int64(Int(dims.unsafe_ptr())), 0),
            Value(TAG_BOOL, 0, 0, 0),
            Value(TAG_NONE, 0, 0, 0),
        ],
        1,
    )
    _ = dims^
    return own(rets.take_tensor(0))


def _ret_undefined(rets: Values, i: Int):
    """An undefined at::Tensor result (a None record in a Tensor slot), as
    CUDA returns for the gradients it does not compute."""
    rets[unsafe_offset=i] = Value(TAG_NONE, 0, 0, 0)


# aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates,
#   Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None)
#   -> (Tensor, Tensor, Tensor)
def op_thnn_fused_lstm_cell(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime C = "_thnn_fused_lstm_cell_cuda"
    var ig = v_tensor(args[unsafe_offset=0])
    var hg = v_tensor(args[unsafe_offset=1])
    var cx = v_tensor(args[unsafe_offset=2])
    var b1 = v_opt_tensor(args[unsafe_offset=3])
    var b2 = v_opt_tensor(args[unsafe_offset=4])
    _check_cell_sizes(C, ig, hg, b1, b2, 4, cx)
    _float_dtype(ig, C)
    var has_bias = Bool(b1)
    var hsz = cx.dim(1)
    var total = cx.numel
    var workspace = own(new_tensor(ig.shape, 2, ig.stype, ig.device))
    var hy = own(new_tensor(_shape2(cx.dim(0), hsz), 2, cx.stype, cx.device))
    var cy = own(new_tensor(_shape2(cx.dim(0), hsz), 2, cx.stype, cx.device))
    if total > 0:
        var igd = _dense(ig)
        var hgd = _dense(hg)
        var cxd = _dense(cx)
        var b1d = _dense(b1.value()) if has_bias else _dense(ig)
        var b2d = _dense(b2.value()) if has_bias else _dense(ig)
        var ctx = ctx_for(ig.device)
        var call = KernelCall("rnn", "LstmCellFwd")
        call.arg_dtype(0, ig.dtype)
        call.int(igd.t.ptr)
        call.int(hgd.t.ptr)
        call.int(b1d.t.ptr)
        call.int(b2d.t.ptr)
        call.int(cxd.t.ptr)
        call.int(hy.t.ptr)
        call.int(cy.t.ptr)
        call.int(workspace.t.ptr)
        call.int(hsz)
        call.int(total)
        call.int(1 if has_bias else 0)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = igd^
        _ = hgd^
        _ = cxd^
        _ = b1d^
        _ = b2d^
    ret_owned(rets, 0, hy)
    ret_owned(rets, 1, cy)
    ret_owned(rets, 2, workspace)


# aten::_thnn_fused_lstm_cell_backward_impl(Tensor? grad_hy, Tensor? grad_cy,
#   Tensor cx, Tensor cy, Tensor workspace, bool has_bias)
#   -> (Tensor, Tensor, Tensor)
def op_thnn_fused_lstm_cell_backward_impl(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime C = "fused_lstm_cell_backward"
    var ghy = v_opt_tensor(args[unsafe_offset=0])
    var gcy = v_opt_tensor(args[unsafe_offset=1])
    var cx = v_tensor(args[unsafe_offset=2])
    var cy = v_tensor(args[unsafe_offset=3])
    var ws = v_tensor(args[unsafe_offset=4])
    var has_bias = v_bool(args[unsafe_offset=5])
    if not ghy and not gcy:
        # CUDA returns three undefined tensors.
        for i in range(3):
            _ret_undefined(rets, i)
        return
    var defined = ghy.value().copy() if ghy else gcy.value().copy()
    if ghy:
        _check_dim(C, defined, "grad_hy", 1, 2)
    else:
        _check_dim(C, defined, "grad_cy", 2, 2)
    var n = defined.dim(0)
    var hsz = defined.dim(1)
    if ghy:
        _check_size2(C, ghy.value(), "grad_hy", 1, n, hsz)
    if gcy:
        _check_size2(C, gcy.value(), "grad_cy", 2, n, hsz)
    _check_size2(C, cx, "cx", 3, n, hsz)
    _check_size2(C, cy, "cy", 4, n, hsz)
    _check_dim(C, ws, "workspace", 5, 2)
    _check_numel(C, ws, "workspace", 5, n * hsz * 4)
    _float_dtype(ws, "_thnn_fused_lstm_cell_cuda_backward")
    _check_device(C, ws, ws)
    _check_device(C, ws, cx)
    _check_device(C, ws, cy)
    _check_device(C, ws, defined)
    _check_dtype(ws, cx)
    _check_dtype(ws, cy)
    _check_dtype(ws, defined)
    if Bool(ghy) and Bool(gcy):
        _check_device(C, ws, gcy.value())
        _check_dtype(ws, gcy.value())

    var grad_gates = own(new_tensor(ws.shape, 2, ws.stype, ws.device))
    var grad_cx = own(new_tensor(_shape2(n, hsz), 2, cx.stype, cx.device))
    var total = n * hsz
    if total > 0:
        var wsd = _dense(ws)
        var cxd = _dense(cx)
        var cyd = _dense(cy)
        var ghyd = _dense(ghy.value()) if ghy else _dense(cx)
        var gcyd = _dense(gcy.value()) if gcy else _dense(cx)
        var ctx = ctx_for(ws.device)
        var call = KernelCall("rnn", "LstmCellBwd")
        call.arg_dtype(0, ws.dtype)
        call.int(wsd.t.ptr)
        call.int(grad_gates.t.ptr)
        call.int(cxd.t.ptr)
        call.int(cyd.t.ptr)
        call.int(ghyd.t.ptr)
        call.int(gcyd.t.ptr)
        call.int(grad_cx.t.ptr)
        call.int(hsz)
        call.int(total)
        call.int(1 if ghy else 0)
        call.int(1 if gcy else 0)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = wsd^
        _ = cxd^
        _ = cyd^
        _ = ghyd^
        _ = gcyd^
    if has_bias:
        var grad_bias = _sum0(grad_gates.t)
        ret_owned(rets, 2, grad_bias)
    else:
        _ret_undefined(rets, 2)
    ret_owned(rets, 0, grad_gates)
    ret_owned(rets, 1, grad_cx)


comptime GRU_WORKSPACE_MULTIPLIER = 5


# aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates,
#   Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None)
#   -> (Tensor, Tensor)
def op_thnn_fused_gru_cell(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime C = "_thnn_fused_gru_cell_cuda"
    var ig = v_tensor(args[unsafe_offset=0])
    var hg = v_tensor(args[unsafe_offset=1])
    var hx = v_tensor(args[unsafe_offset=2])
    var b1 = v_opt_tensor(args[unsafe_offset=3])
    var b2 = v_opt_tensor(args[unsafe_offset=4])
    _check_cell_sizes(C, ig, hg, b1, b2, 3, hx)
    _float_dtype(ig, C)
    var has_bias = Bool(b1)
    var hsz = hx.dim(1)
    var total = hx.numel
    var workspace = own(
        new_tensor(
            _shape2(hx.dim(0), hsz * GRU_WORKSPACE_MULTIPLIER),
            2,
            hx.stype,
            hx.device,
        )
    )
    var hy = own(new_tensor(_shape2(hx.dim(0), hsz), 2, hx.stype, hx.device))
    if total > 0:
        var igd = _dense(ig)
        var hgd = _dense(hg)
        var hxd = _dense(hx)
        var b1d = _dense(b1.value()) if has_bias else _dense(ig)
        var b2d = _dense(b2.value()) if has_bias else _dense(ig)
        var ctx = ctx_for(ig.device)
        var call = KernelCall("rnn", "GruCellFwd")
        call.arg_dtype(0, ig.dtype)
        call.int(igd.t.ptr)
        call.int(hgd.t.ptr)
        call.int(b1d.t.ptr)
        call.int(b2d.t.ptr)
        call.int(hxd.t.ptr)
        call.int(hy.t.ptr)
        call.int(workspace.t.ptr)
        call.int(hsz)
        call.int(total)
        call.int(1 if has_bias else 0)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = igd^
        _ = hgd^
        _ = hxd^
        _ = b1d^
        _ = b2d^
    ret_owned(rets, 0, hy)
    ret_owned(rets, 1, workspace)


# aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace,
#   bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
def op_thnn_fused_gru_cell_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    comptime C = "fused_gru_cell_backward"
    var ghy = v_tensor(args[unsafe_offset=0])
    var ws = v_tensor(args[unsafe_offset=1])
    var has_bias = v_bool(args[unsafe_offset=2])
    _check_dim(C, ghy, "grad_hy", 1, 2)
    _check_size2(
        C,
        ws,
        "workspace",
        2,
        ghy.dim(0),
        ghy.dim(1) * GRU_WORKSPACE_MULTIPLIER,
    )
    _float_dtype(ghy, "_thnn_fused_gru_cell_cuda_backward")
    _check_device(C, ghy, ghy)
    _check_device(C, ghy, ws)
    _check_dtype(ghy, ws)
    var n = ws.dim(0)
    var hsz = ws.dim(1) // GRU_WORKSPACE_MULTIPLIER
    var grad_ig = own(new_tensor(_shape2(n, hsz * 3), 2, ws.stype, ws.device))
    var grad_hg = own(new_tensor(_shape2(n, hsz * 3), 2, ws.stype, ws.device))
    var grad_hx = own(new_tensor(_shape2(n, hsz), 2, ghy.stype, ghy.device))
    var total = ghy.numel
    if total > 0:
        var ghyd = _dense(ghy)
        var wsd = _dense(ws)
        var ctx = ctx_for(ws.device)
        var call = KernelCall("rnn", "GruCellBwd")
        call.arg_dtype(0, ws.dtype)
        call.int(grad_ig.t.ptr)
        call.int(grad_hg.t.ptr)
        call.int(ghyd.t.ptr)
        call.int(grad_hx.t.ptr)
        call.int(wsd.t.ptr)
        call.int(hsz)
        call.int(total)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = ghyd^
        _ = wsd^
    if has_bias:
        var gib = _sum0(grad_ig.t)
        var ghb = _sum0(grad_hg.t)
        ret_owned(rets, 3, gib)
        ret_owned(rets, 4, ghb)
    else:
        _ret_undefined(rets, 3)
        _ret_undefined(rets, 4)
    ret_owned(rets, 0, grad_ig)
    ret_owned(rets, 1, grad_hg)
    ret_owned(rets, 2, grad_hx)


def register_rnn(site: Site) raises:
    impl[op_thnn_fused_lstm_cell, "_thnn_fused_lstm_cell"](site)
    impl[
        op_thnn_fused_lstm_cell_backward_impl,
        "_thnn_fused_lstm_cell_backward_impl",
    ](site)
    impl[op_thnn_fused_gru_cell, "_thnn_fused_gru_cell"](site)
    impl[op_thnn_fused_gru_cell_backward, "_thnn_fused_gru_cell_backward"](site)
