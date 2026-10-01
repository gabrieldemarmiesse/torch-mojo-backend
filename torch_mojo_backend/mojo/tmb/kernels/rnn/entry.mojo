# C entry of the fused recurrent-cell kernels (kernels.mojo): the LSTM and
# GRU cell forward/backward gate math of aten::_thnn_fused_*. Slots are
# unpacked here; nothing is read from the host or synchronized.

from tmb.kernels.rnn.kernels import (
    enqueue_gru_cell_bwd,
    enqueue_gru_cell_fwd,
    enqueue_lstm_cell_bwd,
    enqueue_lstm_cell_fwd,
)
from tmb.kernels.common.op_utils import (
    FLOAT_DTYPES,
    Arg,
    Argv,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher8,
    _spec_dispatcher11,
    _spec_dispatcher12,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)


def _lstm_fwd_go(
    input_gates: Arg,
    hidden_gates: Arg,
    bias1: Arg,
    bias2: Arg,
    cx: Arg,
    hy: Arg,
    cy: Arg,
    workspace: Arg,
    hsz: Arg,
    total: Arg,
    has_bias: Arg,
    ctx: Arg,
) raises:
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_lstm_cell_fwd[dt](
                _raw_ctx(ctx),
                _raw_int(input_gates),
                _raw_int(hidden_gates),
                _raw_int(bias1),
                _raw_int(bias2),
                _raw_int(cx),
                _raw_int(hy),
                _raw_int(cy),
                _raw_int(workspace),
                _raw_int(hsz),
                _raw_int(total),
                _raw_int(has_bias),
            )
            return
    raise Error("lstm cell: no dtype compiled into this module")


def _lstm_bwd_go(
    workspace: Arg,
    grad_gates: Arg,
    cx: Arg,
    cy: Arg,
    grad_hy: Arg,
    grad_cy: Arg,
    grad_cx: Arg,
    hsz: Arg,
    total: Arg,
    has_grad_hy: Arg,
    has_grad_cy: Arg,
    ctx: Arg,
) raises:
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_lstm_cell_bwd[dt](
                _raw_ctx(ctx),
                _raw_int(workspace),
                _raw_int(grad_gates),
                _raw_int(cx),
                _raw_int(cy),
                _raw_int(grad_hy),
                _raw_int(grad_cy),
                _raw_int(grad_cx),
                _raw_int(hsz),
                _raw_int(total),
                _raw_int(has_grad_hy),
                _raw_int(has_grad_cy),
            )
            return
    raise Error("lstm cell backward: no dtype compiled into this module")


def _gru_fwd_go(
    input_gates: Arg,
    hidden_gates: Arg,
    bias1: Arg,
    bias2: Arg,
    hx: Arg,
    hy: Arg,
    workspace: Arg,
    hsz: Arg,
    total: Arg,
    has_bias: Arg,
    ctx: Arg,
) raises:
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_gru_cell_fwd[dt](
                _raw_ctx(ctx),
                _raw_int(input_gates),
                _raw_int(hidden_gates),
                _raw_int(bias1),
                _raw_int(bias2),
                _raw_int(hx),
                _raw_int(hy),
                _raw_int(workspace),
                _raw_int(hsz),
                _raw_int(total),
                _raw_int(has_bias),
            )
            return
    raise Error("gru cell: no dtype compiled into this module")


def _gru_bwd_go(
    grad_input_gates: Arg,
    grad_hidden_gates: Arg,
    grad_hy: Arg,
    grad_hx: Arg,
    workspace: Arg,
    hsz: Arg,
    total: Arg,
    ctx: Arg,
) raises:
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_gru_cell_bwd[dt](
                _raw_ctx(ctx),
                _raw_int(grad_input_gates),
                _raw_int(grad_hidden_gates),
                _raw_int(grad_hy),
                _raw_int(grad_hx),
                _raw_int(workspace),
                _raw_int(hsz),
                _raw_int(total),
            )
            return
    raise Error("gru cell backward: no dtype compiled into this module")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["LstmCellFwd"]():
            _spec_dispatcher12[_lstm_fwd_go, "LstmCellFwd"](argv, argc)
            return 0
        comptime if _op_on["LstmCellBwd"]():
            _spec_dispatcher12[_lstm_bwd_go, "LstmCellBwd"](argv, argc)
            return 0
        comptime if _op_on["GruCellFwd"]():
            _spec_dispatcher11[_gru_fwd_go, "GruCellFwd"](argv, argc)
            return 0
        comptime if _op_on["GruCellBwd"]():
            _spec_dispatcher8[_gru_bwd_go, "GruCellBwd"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
